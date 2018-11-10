"""
thunderfish

Run it from the thunderfish development directory as:
python3 -m thunderfish.thunderfish audiofile.wav   or
python -m thunderfish.thunderfish audiofile.wav
"""

import sys
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from .version import __version__
from .configfile import ConfigFile
from .dataloader import load_data
from .bestwindow import add_clip_config, add_best_window_config, clip_args, best_window_args
from .bestwindow import clip_amplitudes, best_window_indices, plot_best_window
from .harmonicgroups import add_psd_peak_detection_config, add_harmonic_groups_config
from .checkpulse import check_pulse_width, add_check_pulse_width_config, check_pulse_width_args
from .powerspectrum import decibel, plot_decibel_psd, multi_resolution_psd
from .harmonicgroups import harmonic_groups, harmonic_groups_args, psd_peak_detection_args, fundamental_freqs, fundamental_freqs_and_power, colors_markers, plot_harmonic_groups
from .consistentfishes import consistent_fishes
from .eodanalysis import eod_waveform, analyze_wave, analyze_pulse
from .eodanalysis import eod_waveform_plot, pulse_spectrum_plot
from .csvmaker import write_csv


def output_plot(base_name, pulse_fish, inter_eod_intervals,
                raw_data, samplerate, idx0, idx1, fishlist, mean_eods, eod_props, peak_data,
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
    pulse_fish: bool
        True if a pulse-fish has been detected by analysis of the EODs.
    inter_eod_intervals: array
        Time difference from one to another detected EOD.
    raw_data: array
        Dataset.
    samplerate: float
        Sampling-rate of the dataset.
    idx0: float
        Index of the beginning of the analysis window in the dataset.
    idx1: float
        Index of the end of the analysis window in the dataset.
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

    fig = plt.figure(facecolor='white', figsize=(14., 10.))
    ax1 = fig.add_axes([0.02, 0.9, 0.96, 0.1])   # title
    ax2 = fig.add_axes([0.075, 0.06, 0.9, 0.09]) # whole trace
    ax3 = fig.add_axes([0.075, 0.6, 0.7, 0.3])   # psd
    ax4 = fig.add_axes([0.075, 0.2, 0.4, 0.3])   # mean eod
    ax5 = fig.add_axes([0.575, 0.2, 0.4, 0.3])   # pusle spectrum
    
    # plot title
    wavetitle = ""
    if len(fishlist) == 1:
        wavetitle = "a wavefish"
    elif len(fishlist) > 1:
        wavetitle = "%d wavefish" % len(fishlist)
    pulsetitle = ""
    if pulse_fish:
        pulsetitle = "a pulsefish"
    if len(wavetitle)==0 and len(pulsetitle)==0:
        ax1.text(0.0, .72, '%s     Recording - no fish.' % base_name, fontsize=22)
    elif len(wavetitle)>0 and len(pulsetitle)>0:
        ax1.text(0.0, .72, '%s     Recording of %s and %s.' % (base_name, pulsetitle, wavetitle),
                 fontsize=22)
    else:
        ax1.text(0.0, .72, '%s     Recording of %s.' % (base_name, pulsetitle+wavetitle),
                 fontsize=22)
        
    ax1.text(1.0, .77, 'thunderfish by Benda-Lab', fontsize=16, ha='right')
    ax1.text(1.0, .5, 'Version %s' % __version__, fontsize=16, ha='right')
    ax1.set_frame_on(False)
    ax1.get_xaxis().set_visible(False)
    ax1.get_yaxis().set_visible(False)
    ############

    # plot trace
    time = np.arange(len(raw_data)) / samplerate
    ax2.plot(time[:idx0], raw_data[:idx0], color='blue')
    ax2.plot(time[idx1:], raw_data[idx1:], color='blue')
    ax2.plot(time[idx0:idx1], raw_data[idx0:idx1], color='red', label='analysis\nwindow')
    ax2.text(time[(idx0+idx1)//2], 0.0, 'analysis\nwindow', ha='center', va='center')
    ax2.set_xlim(time[0], time[-1])
    ax2.set_xlabel('Time [sec]')
    ax2.set_ylabel('Amplitude [a.u.]')
    ############

    # plot psd
    if len(spec_data) > 0 and len(spec_data[0]) > 0.0:
        ax3.plot(spec_data[0][:,0], decibel(5.0*eod_props[0]['EODf']**2.0*spec_data[0][:,1]), '#CCCCCC', lw=1)
    if len(fishlist) > 0:
        colors, markers = colors_markers()
        plot_harmonic_groups(ax3, fishlist, max_freq=max_freq, max_groups=12, sort_by_freq=True,
                             power_n_harmonics=power_n_harmonics, label_power=label_power,
                             colors=colors, markers=markers, legend_rows=12,
                             frameon=False, bbox_to_anchor=(1.0, 1.1), loc='upper left',
                             title='EOD frequencies')
    plot_decibel_psd(ax3, psd_data[0][1], psd_data[0][0], max_freq=max_freq, color='blue')
    ax3.set_title('Powerspectrum (%d detected wave-fish)' % len(fishlist), y=1.05)

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
        axeod.set_title('Average EOD of {EODf:.1f} Hz {type}-type fish'.format(**props),
                        fontsize=14, y=1.05)
        if unit == 'a.u.':
            unit = ''
        eod_waveform_plot(mean_eod, peaks, axeod, unit)
        props['unit'] = unit
        label = 'p-p amplitude = {p-p-amplitude:.3g} {unit}\nn = {n} EODs\n'.format(**props)
        if props['flipped']:
            label += 'flipped\n'
        axeod.text(0.03, 0.97, label, transform = axeod.transAxes, va='top')
        if props['type'] == 'wave':
            lim = 750.0/props['EODf']
            axeod.set_xlim([-lim, +lim])
        else:
            break

    ################

    if not usedax5 and len(eod_props) > 0 and eod_props[0]['type'] == 'pulse':
        usedax5 = True
        pulse_spectrum_plot(spec_data[0], eod_props[0], ax5)

    ## # plot inter EOD interval histogram
    ## if len(inter_eod_intervals)>2:
    ##     usedax5 = True
    ##     tmp_period = 1000. * np.mean(inter_eod_intervals)
    ##     tmp_period = tmp_period - tmp_period % 0.05
    ##     inter_eod_intervals *= 1000.  # transform sec in msec
    ##     median_IPI = 1000. * eod_props[0]['medianinterval']
    ##     mean_IPI = 1000. * eod_props[0]['meaninterval']
    ##     std_IPI = 1000. * eod_props[0]['stdinterval']
    ##     n, edges = np.histogram(inter_eod_intervals, bins=np.arange(tmp_period - 5., tmp_period + 5., 0.05))

    ##     ax5.bar(edges[:-1], n, edges[1] - edges[0] - 0.001)
    ##     median_line, = ax5.plot([median_IPI, median_IPI], [0, np.max(n)], '--', color='red', lw=2, label='median %.2f ms' % median_IPI)
    ##     ax5.set_xlabel('inter EOD interval [ms]')
    ##     ax5.set_ylabel('n')
    ##     ax5.set_title('Inter-EOD-interval histogram', fontsize=14, y=1.05)

    ##     max_IPI = np.ceil(np.max(inter_eod_intervals)+0.5)
    ##     if max_IPI/median_IPI < 1.2:
    ##         max_IPI = np.ceil(1.2*median_IPI)
    ##     min_IPI = np.floor(np.min(inter_eod_intervals)-0.5)
    ##     if min_IPI/median_IPI > 0.8:
    ##         min_IPI = np.floor(0.8*median_IPI)
    ##     ax5.set_xlim(min_IPI, max_IPI)
    ##     nopatch = mpatches.Patch(color='white', alpha=0.0)
    ##     ax5.legend([median_line, nopatch],
    ##                ['median %.2f ms' % median_IPI,
    ##                 '(%.2f +/- %.2f ms)' % (mean_IPI, std_IPI)],
    ##                 loc='upper right', frameon=False)

    # cosmetics
    for ax in [ax2, ax3, ax4, ax5]:
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


def thunderfish(filename, channel=0, save_csvs=False, save_plot=False,
                output_folder='.', show_bestwindow=False, cfgfile='',
                save_config='', verbose=0):
    # configuration options:
    cfg = ConfigFile()
    cfg.add_section('Power spectrum estimation:')
    cfg.add('frequencyResolution', 0.2, 'Hz', 'Frequency resolution of the power spectrum.')
    cfg.add('numberPSDWindows', 2, '', 'Number of windows on which power spectra are computed.')
    cfg.add('numberPSDResolutions', 1, '', 'Number of power spectra computed within each window with decreasing resolution.')
    cfg.add('frequencyThreshold', 1.0, 'Hz', 'The fundamental frequency of each fish needs to be detected in each power spectrum within this threshold.')
    # TODO: make this threshold dependent on frequency resolution!
    add_psd_peak_detection_config(cfg)
    add_harmonic_groups_config(cfg)
    add_clip_config(cfg)
    add_best_window_config(cfg, win_size=8.0, w_cv_ampl=10.0)
    add_check_pulse_width_config(cfg)
    cfg.add_section('Waveform selection:')
    cfg.add('maximumClippedFraction', 0.01, '', 'Take waveform of the fish with the highest power only if the fraction of clipped signals is below this value.')
    cfg.add('maximumFirstHarmonicAmplitude', 2.0, '', 'Skip waveform of wave-type fish if the ampltude of the first harmonic is higher than this factor times the amplitude of the fundamental.')
    cfg.add('maximumSecondHarmonicAmplitude', 0.8, '', 'Skip waveform of wave-type fish if the ampltude of the second harmonic is higher than this factor times the amplitude of the fundamental. That is, the waveform appears to have twice the frequency than the fundamental.')
    cfg.add('maximumRMSError', 0.05, '', 'Skip waveform of wave-type fish if the root-mean-squared error relative to the peak-to-peak amplitude is larger than this number.')
    
    # load configuration from working directory and data directories:
    cfg.load_files(cfgfile, filename, 3, verbose)

    # save configuration:
    if len(save_config) > 0:
        ext = os.path.splitext(save_config)[1]
        if ext != os.extsep + 'cfg':
            print('configuration file name must have .cfg as extension!')
        else:
            print('write configuration to %s ...' % save_config)
            cfg.dump(save_config)
        return None

    # check data file:
    if len(filename) == 0:
        return 'you need to specify a file containing some data'

    # create output folder
    if save_csvs or save_plot:
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
    outfilename = os.path.splitext(os.path.basename(filename))[0]

    # check channel:
    if channel < 0:
        return 'invalid channel %d' % channel

    # load data:
    try:
        raw_data, samplerate, unit = load_data(filename, channel, verbose=verbose)
    except IOError as e:
        return 'failed to open file %s: %s' % (filename, str(e))
    if len(raw_data) == 0:
        return 'empty data file %s' % filename

    # calculate best_window:
    found_bestwindow = True
    min_clip = cfg.value('minClipAmplitude')
    max_clip = cfg.value('maxClipAmplitude')
    if min_clip == 0.0 or max_clip == 0.0:
        min_clip, max_clip = clip_amplitudes(raw_data, **clip_args(cfg, samplerate))
    if show_bestwindow:
        fig, ax = plt.subplots(5, sharex=True, figsize=(14., 10.))
        try:
            best_window_indices(raw_data, samplerate,
                                min_clip=min_clip, max_clip=max_clip,
                                plot_data_func=plot_best_window, ax=ax,
                                **best_window_args(cfg))
            plt.show()
        except UserWarning as e:
            print('best_window: ' + str(e))
        return
    try:
        idx0, idx1, clipped = best_window_indices(raw_data, samplerate,
                                                  min_clip=min_clip, max_clip=max_clip,
                                                  **best_window_args(cfg))
    except UserWarning as e:
        print('best_window: ' + str(e))
        found_bestwindow = False
        idx0 = 0
        idx1 = len(raw_data)
    data = raw_data[idx0:idx1]

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
    mean_eods = []
    spec_data = []
    peak_data = []
    intervals = []
    power_thresh = []
    
    # analyse eod waveform of pulse-fish:
    if pulse_fish:
        mean_eod_window = 0.002
        mean_eod, eod_times = \
            eod_waveform(data, samplerate,
                         percentile=cfg.value('pulseWidthPercentile'),
                         th_factor=cfg.value('pulseWidthThresholdFactor'),
                         start=-mean_eod_window, stop=mean_eod_window)
        mean_eod, props, peaks, power, intervals = analyze_pulse(mean_eod, eod_times,
                                                                 fresolution=minfres)
        # TODO: add config parameter to analyze_pulse
        mean_eods.append(mean_eod)
        spec_data.append(power)
        peak_data.append(peaks)
        eod_props.append(props)
        power_thresh = np.zeros(power.shape)
        power_thresh[:,0] = power[:,0]
        power_thresh[:,1] = 5.0*props['EODf']**2.0 * power[:,1]

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
                         percentile=cfg.value('pulseWidthPercentile'),
                         th_factor=cfg.value('pulseWidthThresholdFactor'),
                         period=1.0/fish[0,0])
        mean_eod, props, sdata = analyze_wave(mean_eod, fish)
        props['n'] = len(eod_times)
        # add good waveforms only:
        if (k > 0 or clipped < cfg.value('maximumClippedFraction')) and \
            sdata[1,2] < cfg.value('maximumFirstHarmonicAmplitude') and \
            sdata[2,2] < cfg.value('maximumSecondHarmonicAmplitude') and \
            props['rmserror'] < cfg.value('maximumRMSError'):
            eod_props.append(props)
            mean_eods.append(mean_eod)
            spec_data.append(sdata)
            peak_data.append([])
            if verbose > 0:
                print('%d take waveform of %6.1fHz fish: clipped=%4.2f ampl1=%4.2f ampl2=%4.2f ampl3=%4.2f rmserror=%4.3f'
                      % (idx, fish[0,0], clipped, sdata[1,2], sdata[2,2], sdata[3,2], props['rmserror']))
        else:
            if verbose > 0:
                print('%d skip waveform of %.1fHz fish: clipped=%4.2f ampl1=%.2f ampl2=%.2f rmserror=%.3f'
                      % (idx, fish[0,0], clipped, sdata[1,2], sdata[2,2], props['rmserror']))
        
    if not found_bestwindow:
        pulsefish = False
        fishlist = []
        eod_props = []
        mean_eods = []
        intervals = []

    # write results to files:
    if save_csvs and found_bestwindow:
        # eod waveforms:
        for i, (mean_eod, sdata, pdata) in enumerate(zip(mean_eods, spec_data, peak_data)):
            header = ['time (s)', 'mean', 'std']
            if mean_eod.shape[1] > 3:
                header.append('fit')
            write_csv(os.path.join(output_folder, outfilename + '-waveform-%d.csv' % i),
                      header, mean_eod)
            if len(sdata)>0:
                if sdata.shape[1] == 2:
                    header = ['frequency (Hz)', 'power']
                else:
                    header = ['harmonics', 'frequency (Hz)', 'amplitude', 'phase']
                    if sdata.shape[1] > 4:
                        header.append('dataampl')
                write_csv(os.path.join(output_folder, outfilename + '-spectrum-%d.csv' % i),
                          header, sdata)
            if len(pdata)>0:
                header = ['P', 'time (ms)', 'amplitude', 'relaampl']
                write_csv(os.path.join(output_folder, outfilename + '-peaks-%d.csv' % i),
                          header, pdata)
        # wavefish frequencies and amplitudes:
        if len(fishlist) > 0:
            # write csv file with main EODf and corresponding power in dB of detected fishes:
            csv_matrix = fundamental_freqs_and_power(fishlist, cfg.value('powerNHarmonics'))
            csv_name = os.path.join(output_folder, outfilename + '-wavefish-eodfs.csv')
            header = ['fundamental frequency (Hz)', 'power (dB)']
            write_csv(csv_name, header, csv_matrix)

    if save_plot or not save_csvs:
        output_plot(outfilename, pulse_fish,
                    intervals, raw_data, samplerate, idx0, idx1, fishlist,
                    mean_eods, eod_props, peak_data, spec_data, unit,
                    psd_data, cfg.value('powerNHarmonics'), True, 3000.0, output_folder,
                    save_plot=save_plot, show_plot=not save_csvs)


def main():
    # config file name:
    cfgfile = __package__ + '.cfg'

    # command line arguments:
    parser = argparse.ArgumentParser(
        description='Analyze EOD waveforms of weakly electric fish.',
        epilog='by Benda-Lab (2015-2018)')
    parser.add_argument('--version', action='version', version=__version__)
    parser.add_argument('-v', action='count', dest='verbose', help='verbosity level')
    parser.add_argument('-c', '--save-config', nargs='?', default='', const=cfgfile,
                        type=str, metavar='cfgfile',
                        help='save configuration to file cfgfile (defaults to {0})'.format(cfgfile))
    parser.add_argument('file', nargs='?', default='', type=str, help='name of the file with the time series data')
    parser.add_argument('channel', nargs='?', default=0, type=int, help='channel to be analyzed')
    parser.add_argument('-p', dest='save_plot', action='store_true', help='save output plot as pdf file')
    parser.add_argument('-s', dest='save_csvs', action='store_true',
                        help='save analysis results as csv-files')
    parser.add_argument('-o', dest='output_folder', default=".", type=str,
                        help="path where to store results and figures")
    parser.add_argument('-b', dest='show_bestwindow', action='store_true', help='show the cost function of the best window algorith,')
    args = parser.parse_args()

    # set verbosity level from command line:
    verbose = 0
    if args.verbose != None:
        verbose = args.verbose

    msg = thunderfish(args.file, args.channel, args.save_csvs, args.save_plot,
                      args.output_folder, args.show_bestwindow, cfgfile,
                      args.save_config, verbose=verbose)
    if msg is not None:
        parser.error(msg)


if __name__ == '__main__':
    main()
