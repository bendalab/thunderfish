"""
# thunderfish

Automatically detect and analyze all fish present in an EOD recording
and generate a summary plot and data tables.

Run it from the thunderfish development directory as:
python3 -m thunderfish.thunderfish audiofile.wav   or
python -m thunderfish.thunderfish audiofile.wav
"""

import sys
import os
import glob
import argparse
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool, freeze_support, cpu_count
from audioio import play, fade
from .version import __version__, __year__
from .configfile import ConfigFile
from .dataloader import load_data
from .bestwindow import add_clip_config, add_best_window_config, clip_args, best_window_args
from .bestwindow import find_best_window, plot_best_data
from .harmonicgroups import add_psd_peak_detection_config, add_harmonic_groups_config
from .checkpulse import check_pulse_width, add_check_pulse_width_config, check_pulse_width_args
from .powerspectrum import decibel, plot_decibel_psd, multi_resolution_psd
from .harmonicgroups import harmonic_groups, harmonic_groups_args, psd_peak_detection_args
from .harmonicgroups import fundamental_freqs, fundamental_freqs_and_power
from .harmonicgroups import colors_markers, plot_harmonic_groups
from .consistentfishes import consistent_fishes
from .eodanalysis import eod_waveform, analyze_wave, analyze_pulse
from .eodanalysis import eod_recording_plot, eod_waveform_plot
from .eodanalysis import pulse_spectrum_plot, wave_spectrum_plot
from .eodanalysis import add_eod_analysis_config, eod_waveform_args
from .eodanalysis import analyze_wave_args, analyze_pulse_args
from .eodanalysis import wave_quality, wave_quality_args, add_eod_quality_config
from .eodanalysis import pulse_quality, pulse_quality_args
from .eodanalysis import save_eod_waveform, save_wave_eods, save_pulse_eods
from .eodanalysis import save_wave_spectrum, save_pulse_spectrum, save_pulse_peaks
from .tabledata import TableData, add_write_table_config, write_table_args


def configuration(config_file, save_config=False, file_name='', verbose=0):
    """
    Assemble, save, and load configuration parameter for thunderfish.

    Parameters
    ----------
    config_file: string
        Name of the configuration file to be loaded.
    save_config: boolean
        If True write the configuration file to the current working directory
        after loading existing configuration files.
    file_name: string
        Data file to be analyzed. Config files will be loaded from this path
        and up to three levels up.
    verbose: int
        Print out information about loaded configuration files if greater than zero.

    Returns
    -------
    cfg: ConfigFile
        Configuration parameters.
    """
    cfg = ConfigFile()
    cfg.add_section('Power spectrum estimation:')
    cfg.add('frequencyResolution', 0.5, 'Hz', 'Frequency resolution of the power spectrum.')
    cfg.add('numberPSDWindows', 1, '', 'Number of windows on which power spectra are computed.')
    cfg.add('numberPSDResolutions', 1, '', 'Number of power spectra computed within each window with decreasing resolution.')
    cfg.add('frequencyThreshold', 1.0, 'Hz', 'The fundamental frequency of each fish needs to be detected in each power spectrum within this threshold.')
    # TODO: make this threshold dependent on frequency resolution!
    add_psd_peak_detection_config(cfg)
    add_harmonic_groups_config(cfg)
    add_clip_config(cfg)
    cfg.add('unwrapData', False, '', 'Unwrap scrambled wav-file data.')
    add_best_window_config(cfg, win_size=8.0, w_cv_ampl=10.0)
    add_eod_analysis_config(cfg, min_pulse_win=0.004)
    del cfg['eodSnippetFac']
    del cfg['eodMinSnippet']
    add_eod_quality_config(cfg)
    add_write_table_config(cfg, table_format='csv', unitstyle='row', format_width=True,
                           shrink_width=False)
    
    # load configuration from working directory and data directories:
    cfg.load_files(config_file, file_name, 3, verbose)

    # save configuration:
    if save_config:
        print('write configuration to %s ...' % config_file)
        del cfg['fileColumnNumbers']
        del cfg['fileShrinkColumnWidth']
        del cfg['fileMissing']
        cfg.dump(config_file)
            
    return cfg


def output_plot(base_name, raw_data, samplerate, idx0, idx1,
                clipped, fishlist, mean_eods, eod_props, peak_data,
                spec_data, unit, psd_data, power_n_harmonics, label_power, max_freq=3000.0,
                output_folder='', save_plot=False, show_plot=False):
    """
    Creates an output plot for the Thunderfish program.

    This output contains the raw trace where the analysis window is marked, the power-spectrum of this analysis window
    where the detected fish are marked, a mean EOD plot, a histogram of the inter EOD interval and further information
    about the recording that is analysed.


    Parameters
    ----------
    base_name: string
        Basename of audio_file.
    raw_data: array
        Dataset.
    samplerate: float
        Sampling-rate of the dataset.
    idx0: float
        Index of the beginning of the analysis window in the dataset.
    idx1: float
        Index of the end of the analysis window in the dataset.
    clipped: float
        Fraction of clipped amplitudes.
    fishlist: array
        Frequency and power of fundamental frequency/harmonics of several fish.
    mean_eods: list of 2-D arrays with time, mean and std.
        Mean trace for the mean EOD plot.
    eod_props: list of dict
        Properties for each waveform in mean_eods.
    peak_data: list of 2_D arrays
        For each pulsefish a list of peak properties (index, time, and amplitude).
    spec_data: list of 2_D arrays
        For each pulsefish a power spectrum of the single pulse and for each wavefish
        the relative amplitudes and phases of the harmonics.
    unit: string
        Unit of the trace and the mean EOD.
    psd_data: array
        Power spectrum of the analysed data for different frequency resolutions.
    power_n_harmonics: int
        Maximum number of harmonics over which the total power of the signal is computed.
    label_power: boolean
        If `True` put the power in decibel in addition to the frequency into the legend.
    output_folder: string
        Path indicating where output-files will be saved.
    save_plot: bool
        If True, the plot will be saved in output_folder.
    show_plot: bool
        If True (and saveplot=False) it will show the plot without saving.
    """

    def keypress(event):
        if event.key in 'pP':
            if idx1 > idx0:
                playdata = 1.0 * raw_data[idx0:idx1]
            else:
                playdata = 1.0 * raw_data[:]
            fade(playdata, samplerate, 0.1)
            play(playdata, samplerate, blocking=False)

    fig = plt.figure(facecolor='white', figsize=(14., 10.))
    ax1 = fig.add_axes([0.02, 0.9, 0.96, 0.1])   # title
    ax2 = fig.add_axes([0.075, 0.06, 0.9, 0.09]) # whole trace
    ax3 = fig.add_axes([0.075, 0.6, 0.7, 0.3])   # psd
    ax4 = fig.add_axes([0.075, 0.2, 0.4, 0.3])   # mean eod
    ax5 = fig.add_axes([0.575, 0.2, 0.4, 0.3])   # pulse spectrum
    ax6 = fig.add_axes([0.575, 0.36, 0.4, 0.14]) # amplitude spectrum
    ax7 = fig.add_axes([0.575, 0.2, 0.4, 0.14])  # phase spectrum
    ax8 = fig.add_axes([0.075, 0.6, 0.4, 0.3])   # recording xoom-in

    if show_plot:
        fig.canvas.mpl_connect('key_press_event', keypress)

    # count number of detected fish types:
    nwave = 0
    npulse = 0
    for props in eod_props:
        if props['type'] == 'pulse':
            npulse += 1
        elif props['type'] == 'wave':
            nwave += 1
                    
    # plot title
    wavetitle = ""
    if nwave > 0:
        wavetitle = "%d wave-type fish" % nwave
    pulsetitle = ""
    if npulse > 0:
        pulsetitle = "%d pulse-type fish" % npulse
    if npulse==0 and nwave==0:
        ax1.text(0.0, .72, '%s     - no fish detected -' % base_name, fontsize=22)
    elif npulse>0 and nwave>0:
        ax1.text(0.0, .72, '%s     %s and %s' % (base_name, pulsetitle, wavetitle),
                 fontsize=22)
    else:
        ax1.text(0.0, .72, '%s     %s' % (base_name, pulsetitle+wavetitle),
                 fontsize=22)
        
    ax1.text(1.0, .77, 'thunderfish by Benda-Lab', fontsize=16, ha='right')
    ax1.text(1.0, .5, 'Version %s' % __version__, fontsize=16, ha='right')
    ax1.set_frame_on(False)
    ax1.get_xaxis().set_visible(False)
    ax1.get_yaxis().set_visible(False)

    ############

    # plot trace
    plot_best_data(raw_data, samplerate, unit, idx0, idx1, clipped, ax2)
    
    ############

    # plot psd
    if len(spec_data) > 0 and len(spec_data[0]) > 0 and \
       len(eod_props) > 0 and 'EODf' in eod_props[0]:
        ax3.plot(spec_data[0][:,0], decibel(5.0*eod_props[0]['EODf']**2.0*spec_data[0][:,1]), '#CCCCCC', lw=1)
    if len(fishlist) > 0:
        if len(fishlist) == 1:
            title = None
            bbox = (1.0, 1.0)
            loc = 'upper right'
        else:
            title = '%d EOD frequencies' % len(fishlist)
            bbox = (1.0, 1.1)
            loc = 'upper left'
        colors, markers = colors_markers()
        plot_harmonic_groups(ax3, fishlist, max_freq=max_freq, max_groups=12, sort_by_freq=True,
                             power_n_harmonics=power_n_harmonics, label_power=label_power,
                             colors=colors, markers=markers, legend_rows=12,
                             frameon=False, bbox_to_anchor=bbox, loc=loc,
                             title=title)
    plot_decibel_psd(ax3, psd_data[0][1], psd_data[0][0], max_freq=max_freq, color='blue')
    ax3.set_title('Powerspectrum', y=1.05, fontsize=14)
    
    ############

    # plot recording
    if len(eod_props) == 1:
        ax3.set_position([0.575, 0.6, 0.4, 0.3])
        width = 0.1
        if eod_props[0]['type'] == 'wave':
            width = 5.0/eod_props[0]['EODf']
        else:
            width = 3.0/eod_props[0]['EODf']
        width = (1+width//0.005)*0.005
        eod_recording_plot(raw_data[idx0:idx1], samplerate, ax8, width, unit, idx0/samplerate)
        ax8.set_title('Recording', fontsize=14, y=1.05)
    else:
        ax8.set_visible(False)        

    ##########

    # plot mean EOD
    usedax4 = False
    usedax5 = False
    eodaxes = [ax4, ax5]
    for axeod, mean_eod, props, peaks in zip(eodaxes[:2], mean_eods[:2], eod_props[0:2], peak_data[0:2]):
        if axeod is ax4:
            usedax4 = True
        if axeod is ax5:
            usedax5 = True
        axeod.text(-0.1, 1.08, '{EODf:.1f} Hz {type}-type fish'.format(**props), transform = axeod.transAxes, fontsize=14)
        axeod.text(0.5, 1.08, 'Averaged EOD', transform = axeod.transAxes, fontsize=14, ha='center')
        if len(unit) == 0 or unit == 'a.u.':
            unit = ''
        tau = props['tau'] if 'tau' in props else None
        eod_waveform_plot(mean_eod, peaks, axeod, unit, tau=tau)
        props['unit'] = unit
        props['eods'] = 'EODs' if props['n'] > 1 else 'EOD'
        label = 'p-p amplitude = {p-p-amplitude:.3g} {unit}\nn = {n} {eods}\n'.format(**props)
        if props['flipped']:
            label += 'flipped\n'
        if -mean_eod[0,0] < 0.6*mean_eod[-1,0]:
            axeod.text(0.97, 0.97, label, transform = axeod.transAxes, va='top', ha='right')
        else:
            axeod.text(0.03, 0.97, label, transform = axeod.transAxes, va='top')
        if props['type'] == 'wave':
            lim = 750.0/props['EODf']
            axeod.set_xlim([-lim, +lim])
        else:
            break

    ################

    # plot spectra:
    ax5.set_visible(True)
    ax6.set_visible(False)
    ax7.set_visible(False)
    if not usedax5 and len(eod_props) > 0:
        usedax5 = True
        if  eod_props[0]['type'] == 'pulse':
            pulse_spectrum_plot(spec_data[0], eod_props[0], ax5)
            ax5.set_title('Single pulse spectrum', fontsize=14, y=1.05)
        else:
            ax5.set_visible(False)
            ax6.set_visible(True)
            ax7.set_visible(True)
            wave_spectrum_plot(spec_data[0], eod_props[0], ax6, ax7, unit)
            ax6.set_title('Amplitude and phase spectrum', fontsize=14, y=1.05)
            ax6.set_xticklabels([])

    ################

    # plot data trace in case no fish was found:
    if not usedax4:
        if len(fishlist) < 2:
            ax3.set_position([0.075, 0.6, 0.9, 0.3])   # enlarge psd
        ax4.set_position([0.075, 0.2, 0.9, 0.3])
        rdata = raw_data[idx0:idx1] if idx1 > idx0 else raw_data
        eod_recording_plot(rdata, samplerate, ax4, 0.1, unit, idx0/samplerate)
        ax4.set_title('Recording', fontsize=14, y=1.05)
        usedax4 = True
            
    # cosmetics
    for ax in [ax2, ax3, ax4, ax5, ax6, ax7, ax8]:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.get_xaxis().tick_bottom()
        ax.get_yaxis().tick_left()
    if not usedax4:
        ax4.set_visible(False)
    if not usedax5:
        ax5.set_visible(False)

    # save figure as pdf
    if save_plot:
        plt.savefig(os.path.join(output_folder, base_name + '.pdf'))
    elif show_plot:
        plt.show()
    plt.close()


def thunderfish(filename, cfg, channel=0, save_data=False, save_plot=False,
                output_folder='.', keep_path=False, show_bestwindow=False, verbose=0):
    # check data file:
    if len(filename) == 0:
        return 'you need to specify a file containing some data'

    # file names:
    fn = filename if keep_path else os.path.basename(filename)
    outfilename = os.path.splitext(fn)[0]

    # check channel:
    if channel < 0:
        return '%s: invalid channel %d' % (filename, channel)

    # load data:
    try:
        raw_data, samplerate, unit = load_data(filename, channel, verbose=verbose)
    except IOError as e:
        return '%s: failed to open file: %s' % (filename, str(e))
    if len(raw_data) <= 1:
        return '%s: empty data file' % filename

    # best_window:
    data, idx0, idx1, clipped = find_best_window(raw_data, samplerate, cfg, show_bestwindow)
    if show_bestwindow:
        return None
    found_bestwindow = idx1 > 0
    if not found_bestwindow:
        print(filename + ': in best_window(): ' + str(e) + '! You may want to adjust the bestWindowSize parameter in the configuration file.')

    # pulse-type fish?
    pulse_fish, pta_value, pulse_period = \
        check_pulse_width(data, samplerate, verbose=verbose,
                          **check_pulse_width_args(cfg))

    # calculate powerspectra within sequential windows and different frequency resolutions:
    numpsdwindows = cfg.value('numberPSDWindows')
    numpsdresolutions = cfg.value('numberPSDResolutions')
    minfres = cfg.value('frequencyResolution')
    fresolution=[minfres]
    for i in range(1, numpsdresolutions):
        fresolution.append(2*fresolution[-1])
    n_incr = len(data)//(numpsdwindows+1) # half overlapping
    psd_data = []
    deltaf = minfres
    for k in range(numpsdwindows):
        mr_psd_data = multi_resolution_psd(data[k*n_incr:(k+2)*n_incr], samplerate,
                                           fresolution=fresolution)
        deltaf = mr_psd_data[0][1][1] - mr_psd_data[0][1][0]
        psd_data.extend(mr_psd_data)
    
    # find the fishes in the different powerspectra:
    h_kwargs = psd_peak_detection_args(cfg)
    h_kwargs.update(harmonic_groups_args(cfg))
    fishlists = []
    for i, psd in enumerate(psd_data):
        fishlist = harmonic_groups(psd[1], psd[0], verbose-1, **h_kwargs)[0]
        if verbose > 0:
            print('fundamental frequencies detected in power spectrum of window %d at resolution %d:'
                  % (i//numpsdresolutions, i%numpsdresolutions))
            if len(fishlist) > 0:
                print('  ' + ' '.join(['%.1f' % freq[0, 0] for freq in fishlist]))
            else:
                print('  none')
        fishlists.append(fishlist)
    # filter the different fishlists to get a fishlist with consistent fishes:
    fishlist = consistent_fishes(fishlists, df_th=cfg.value('frequencyThreshold'))
    if verbose > 0:
        if len(fishlist) > 0:
            print('fundamental frequencies consistent in all power spectra:')
            print('  ' + ' '.join(['%.1f' % freq[0, 0] for freq in fishlist]))
        else:
            print('no fundamental frequencies are consistent in all power spectra')

    # analysis results:
    eod_props = []
    wave_props = []
    pulse_props = []
    mean_eods = []
    spec_data = []
    peak_data = []
    power_thresh = []
    skip_reason = []
    
    # analyse eod waveform of pulse-fish:
    max_eods = cfg.value('eodMaxEODs')
    if pulse_fish:
        mean_eod, eod_times = \
            eod_waveform(data, samplerate,
                         win_fac=0.8, min_win=cfg.value('eodMinPulseSnippet'),
                         **eod_waveform_args(cfg))
        mean_eod, props, peaks, power = analyze_pulse(mean_eod, eod_times,
                                                      fresolution=minfres,
                                                      **analyze_pulse_args(cfg))
        props['n'] = len(eod_times) if len(eod_times) < max_eods or max_eods == 0 else max_eods
        props['index'] = len(eod_props)
        power_thresh = np.zeros(power.shape)
        power_thresh[:,0] = power[:,0]
        power_thresh[:,1] = 5.0*props['EODf']**2.0 * power[:,1]
        # add good waveforms only:
        skips, msg = pulse_quality(0, clipped, props['rmvariance'],
                                   **pulse_quality_args(cfg))
        if len(skips) == 0:
            eod_props.append(props)
            pulse_props.append(props)
            mean_eods.append(mean_eod)
            spec_data.append(power)
            peak_data.append(peaks)
            if verbose > 0:
                print('take %6.1fHz pulse-type fish: %s' % (props['EODf'], msg))
        else:
            skip_reason += ['%d %.1fHz pulse-type fish %s' % (0, props['EODf'], skips)]
            if verbose > 0:
                print('skip %6.1fHz pulse-type fish: %s (%s)' %
                      (props['EODf'], skips, msg))

    if len(power_thresh) > 0:
        n = len(fishlist)
        for k, fish in enumerate(reversed(fishlist)):
            df = power_thresh[1,0] - power_thresh[0,0]
            hfrac = float(np.sum(fish[:,1] < power_thresh[np.array(fish[:,0]//df, dtype=int),1]))/float(len(fish[:,1]))
            if hfrac >= 0.5:
                fishlist.pop(n-1-k)
                if verbose > 0:
                    print('removed frequency %.1f Hz, because %.0f%% of the harmonics where below pulsefish threshold' % (fish[0,0], 100.0*hfrac))        

    # analyse EOD waveform of all wavefish:
    powers = np.array([np.sum(fish[:cfg.value('powerNHarmonics'), 1])
                       for fish in fishlist])
    for k, idx in enumerate(np.argsort(-powers)):
        fish = fishlist[idx]
        mean_eod, eod_times = \
            eod_waveform(data, samplerate,
                         win_fac=3.0, min_win=0.0, period=1.0/fish[0,0],
                         **eod_waveform_args(cfg))
        mean_eod, props, sdata, error_str = \
            analyze_wave(mean_eod, fish, **analyze_wave_args(cfg))
        if error_str:
            print(filename + ': ' + error_str)
        props['n'] = len(eod_times) if len(eod_times) < max_eods or max_eods == 0 else max_eods
        props['index'] = len(eod_props)
        # add good waveforms only:
        skips, msg = wave_quality(k, clipped, props['rmvariance'], props['rmserror'], sdata,
                                  **wave_quality_args(cfg))
        if len(skips) == 0:
            eod_props.append(props)
            wave_props.append(props)
            mean_eods.append(mean_eod)
            spec_data.append(sdata)
            peak_data.append([])
            if verbose > 0:
                print('%d take %6.1fHz wave-type fish: %s' % (idx, props['EODf'], msg))
        else:
            skip_reason += ['%d %.1fHz wave-type fish %s' % (idx, props['EODf'], skips)]
            if verbose > 0:
                print('%d skip waveform of %6.1fHz fish: %s (%s)' %
                      (idx, props['EODf'], skips, msg))
        
    if not found_bestwindow:
        pulsefish = False
        fishlist = []
        eod_props = []
        wave_props = []
        pulse_props = []
        mean_eods = []

    # warning message in case no fish has been found:
    if found_bestwindow and not eod_props :
        msg = ', '.join(skip_reason)
        if msg:
            print(filename + ': no fish found: %s' % msg)
        else:
            print(filename + ': no fish found.')

    # write results to files:
    if save_data:
        fext = TableData.extensions[cfg.value('fileFormat')]
        # remove all files from previous runs of thunderfish:
        for fn in glob.glob(os.path.join(output_folder, '%s-*.%s' % (outfilename, fext))):
            os.remove(fn)
            if verbose > 0:
                print('removed file %s' % fn)
        if found_bestwindow:
            output_basename = os.path.join(output_folder, outfilename)
            if keep_path:
                outpath = os.path.dirname(output_basename)
                if not os.path.exists(outpath):
                    if verbose > 0:
                        print('mkdir %s' % outpath)
                    os.makedirs(outpath)
            # for each fish:
            for i, (mean_eod, sdata, pdata) in enumerate(zip(mean_eods, spec_data, peak_data)):
                fp = save_eod_waveform(mean_eod, unit, i, output_basename,
                                       **write_table_args(cfg))
                if verbose > 0:
                    print('wrote file %s' % fp)
                # power spectrum:
                if len(sdata)>0:
                    if sdata.shape[1] == 2:
                        fp = save_pulse_spectrum(sdata, unit, i, output_basename,
                                                 **write_table_args(cfg))
                    else:
                        fp = save_wave_spectrum(sdata, unit, i, output_basename,
                                                **write_table_args(cfg))
                    if verbose > 0:
                        print('wrote file %s' % fp)
                # peaks:
                fp = save_pulse_peaks(pdata, unit, i, output_basename,
                                      **write_table_args(cfg))
                if verbose > 0 and not fp is None:
                    print('wrote file %s' % fp)
            # fish properties:
            if wave_props:
                fp = save_wave_eods(wave_props, unit, output_basename,
                                    **write_table_args(cfg))
                if verbose > 0:
                    print('wrote file %s' % fp)
            if pulse_props:
                fp = save_pulse_eods(pulse_props, unit, output_basename,
                                     **write_table_args(cfg))
                if verbose > 0:
                    print('wrote file %s' % fp)

    if save_plot or not save_data:
        output_plot(outfilename, raw_data, samplerate, idx0, idx1, clipped, fishlist,
                    mean_eods, eod_props, peak_data, spec_data, unit,
                    psd_data, cfg.value('powerNHarmonics'), True, 3000.0, output_folder,
                    save_plot=save_plot, show_plot=not save_data)


pool_args = None

def run_thunderfish(file):
    """
    Helper function for mutlithreading Pool().map().
    """
    verbose = pool_args[-1]+1
    if verbose > 0:
        if verbose > 1:
            print('='*60)
        print('analyze recording %s ...' % file)
    msg = thunderfish(file, *pool_args)
    if msg:
        print(msg)


def main():
    # config file name:
    cfgfile = __package__ + '.cfg'

    # command line arguments:
    parser = argparse.ArgumentParser(add_help=False,
        description='Analyze EOD waveforms of weakly electric fish.',
        epilog='version %s by Benda-Lab (2015-%s)' % (__version__, __year__))
    parser.add_argument('-h', '--help', action='store_true',
                        help='show this help message and exit')
    parser.add_argument('--version', action='version', version=__version__)
    parser.add_argument('-v', action='count', dest='verbose',
                        help='verbosity level. Increase by specifying -v multiple times, or like -vvv.')
    parser.add_argument('-c', dest='save_config', action='store_true',
                        help='save configuration to file {0} after reading all configuration files'.format(cfgfile))
    parser.add_argument('--channel', default=0, type=int,
                        help='channel to be analyzed. Default is to use first channel.')
    parser.add_argument('-j', dest='jobs', nargs='?', type=int, default=None, const=0,
                        help='number of jobs run in parallel. Without argument use all CPU cores.')
    parser.add_argument('-s', dest='save_data', action='store_true',
                        help='save analysis results to files')
    parser.add_argument('-f', dest='format', default='auto', type=str,
                        choices=TableData.formats,
                        help='file format used for saving analysis results, defaults to the format specified in the configuration file or "dat")')
    parser.add_argument('-p', dest='save_plot', action='store_true',
                        help='save output plot as pdf file')
    parser.add_argument('-k', dest='keep_path', action='store_true',
                        help='keep path of input file when saving analysis files')
    parser.add_argument('-o', dest='outpath', default='.', type=str,
                        help='path where to store results and figures')
    parser.add_argument('-b', dest='show_bestwindow', action='store_true',
                        help='show the cost function of the best window algorithm')
    parser.add_argument('file', nargs='*', default='', type=str,
                        help='name of the file with the time series data')
    args = parser.parse_args()

    # help:
    if args.help:
        parser.print_help()
        print('')
        print('examples:')
        print('- analyze the single file data.wav interactively:')
        print('  > thunderfish data.wav')
        print('- automatically analyze all wav files in the current working directory and save analysis results and plot to files:')
        print('  > thunderfish -s -p *.wav')
        print('- analyze all wav files in the river1/ directory, use all CPUs, and write files directly to "results/":')
        print('  > thunderfish -j -s -p -o results/ *.wav')
        print('- analyze all wav files in the river1/ directory and write files to "results/river1/":')
        print('  > thunderfish -s -p -o results/ -k river1/*.wav')
        print('- write configuration file:')
        print('  > thunderfish -c')
        parser.exit()

    # set verbosity level from command line:
    verbose = 0
    if args.verbose != None:
        verbose = args.verbose

    # interactive plot:
    plt.rcParams['keymap.quit'] = 'ctrl+w, alt+q, q'

    if args.save_config:
        # configuration:
        file_name = args.file[0] if len(args.file) else ''
        configuration(cfgfile, args.save_config, file_name, verbose)
    elif len(args.file) == 0:
        parser.error('you need to specify at least one file for the analysis')
    else:
        # analyze data files:
        cfg = configuration(cfgfile, False, args.file[0], verbose-1)
        if args.format != 'auto':
            cfg.set('fileFormat', args.format)
        # create output folder:
        if args.save_data or args.save_plot:
            if not os.path.exists(args.outpath):
                if verbose > 1:
                    print('mkdir %s' % args.outpath)
                os.makedirs(args.outpath)
        # run on pool:
        global pool_args
        pool_args = (cfg, args.channel, args.save_data,
                     args.save_plot, args.outpath, args.keep_path,
                     args.show_bestwindow, verbose-1)
        if args.jobs is not None and (args.save_data or args.save_plot) and len(args.file) > 1:
            cpus = cpu_count() if args.jobs == 0 else args.jobs
            if verbose > 1:
                print('run on %d cpus' % cpus)
            p = Pool(cpus)
            p.map(run_thunderfish, args.file)
        else:
            list(map(run_thunderfish, args.file))


if __name__ == '__main__':
    freeze_support()  # needed by multiprocessing for some weired windows stuff
    main()
