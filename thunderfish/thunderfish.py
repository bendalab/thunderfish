"""
# thunderfish

Automatically detect and analyze all EOD waveforms in a short recording
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
import matplotlib.ticker as ticker
from matplotlib.transforms import Bbox
from multiprocessing import Pool, freeze_support, cpu_count
from audioio import play, fade
from .version import __version__, __year__
from .configfile import ConfigFile
from .dataloader import load_data
from .bestwindow import add_clip_config, add_best_window_config
from .bestwindow import clip_args, best_window_args
from .bestwindow import find_best_window, plot_best_data
from .checkpulse import check_pulse_width, check_pulse_width_args
from .lizpulses import extract_pulsefish
from .powerspectrum import decibel, plot_decibel_psd, multi_psd
from .powerspectrum import add_multi_psd_config, multi_psd_args
from .harmonics import add_psd_peak_detection_config, add_harmonic_groups_config
from .harmonics import harmonic_groups, harmonic_groups_args, psd_peak_detection_args
from .harmonics import colors_markers, plot_harmonic_groups
from .consistentfishes import consistent_fishes
from .eodanalysis import eod_waveform, analyze_wave, analyze_pulse
from .eodanalysis import plot_eod_recording, plot_pulse_eods, plot_eod_waveform
from .eodanalysis import plot_pulse_spectrum, plot_wave_spectrum
from .eodanalysis import add_eod_analysis_config, eod_waveform_args
from .eodanalysis import analyze_wave_args, analyze_pulse_args
from .eodanalysis import wave_quality, wave_quality_args, add_eod_quality_config
from .eodanalysis import pulse_quality, pulse_quality_args
from .eodanalysis import save_eod_waveform, save_wave_eodfs, save_wave_fish, save_pulse_fish
from .eodanalysis import save_wave_spectrum, save_pulse_spectrum, save_pulse_peaks
from .fakefish import normalize_wavefish, export_wavefish
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
        Print out information about loaded configuration files
        if greater than zero.

    Returns
    -------
    cfg: ConfigFile
        Configuration parameters.
    """
    cfg = ConfigFile()
    add_multi_psd_config(cfg)
    cfg.add('frequencyThreshold', 1.0, 'Hz',
            'The fundamental frequency of each fish needs to be detected in each power spectrum within this threshold.')
    # TODO: make this threshold dependent on frequency resolution!
    cfg.add('minPSDAverages', 3, '', 'Minimum number of fft averages for estimating the power spectrum.')  # needed by fishfinder
    add_psd_peak_detection_config(cfg)
    add_harmonic_groups_config(cfg)
    add_clip_config(cfg)
    cfg.add('unwrapData', False, '', 'Unwrap scrambled wav-file data.')
    add_best_window_config(cfg, win_size=8.0, w_cv_ampl=10.0)
    add_eod_analysis_config(cfg, min_pulse_win=0.004)
    del cfg['eodSnippetFac']
    del cfg['eodMinSnippet']
    add_eod_quality_config(cfg)
    add_write_table_config(cfg, table_format='csv', unit_style='row',
                           align_columns=True, shrink_width=False)
    
    # load configuration from working directory and data directories:
    cfg.load_files(config_file, file_name, 3, verbose)

    # save configuration:
    if save_config:
        ext = os.path.splitext(config_file)[1]
        if ext != os.extsep + 'cfg':
            print('configuration file name must have .cfg as extension!')
        else:
            print('write configuration to %s ...' % config_file)
            del cfg['fileColumnNumbers']
            del cfg['fileShrinkColumnWidth']
            del cfg['fileMissing']
            del cfg['fileLaTeXLabelCommand']
            del cfg['fileLaTeXMergeStd']
            cfg.dump(config_file)
    return cfg


def detect_eods(data, samplerate, clipped, name, verbose, cfg):
    """ Detect EODs of all fish present in the data.

    Parameters
    ----------
    data: array of floats
        The recording in which to detect EODs.
    samplerate: float
        Sampling rate of the dataset.
    clipped: float
        Fraction of clipped amplitudes.
    name: string
        Name of the recording (e.g. its filename).
    verbose: int
        Print out information about EOD detection if greater than zero.
    cfg: ConfigFile
        Configuration parameters.

    Returns
    -------
    psd_data: list of 2D array
        List of power spectra (frequencies and power) of the analysed data
        for different frequency resolutions.
    wave_eodfs: array
        Frequency and power of fundamental frequency/harmonics of all wave fish.
    wave_indices: array of int
        Indices of wave fish mapping from wave_eodfs to eod_props.
        If negative, then that EOD frequency has no waveform described in eod_props.
    eod_props: list of dict
        Lists of EOD properties as returned by analyze_pulse() and analyze_wave()
        for each waveform in mean_eods.
    mean_eods: list of 2-D arrays with time, mean, sem, and fit.
        Averaged EOD waveforms of pulse and wave fish.
    spec_data: list of 2_D arrays
        For each pulsefish a power spectrum of the single pulse and for
        each wavefish the relative amplitudes and phases of the harmonics.
    peak_data: list of 2_D arrays
        For each pulse fish a list of peak properties
        (index, time, and amplitude), empty array for wave fish.
    power_thresh:  2 D array or None
        Frequency (first column) and power (second column) of threshold
        derived from single pulse spectra to discard false wave fish.
        None if no pulse fish was detected.
    skip_reason: list of string
        Reasons, why an EOD was discarded.
    """
    # detect pulse fish:
    """
    pulse_fish, _, eod_times = check_pulse_width(data, samplerate, verbose=verbose,
                                                 **check_pulse_width_args(cfg))
    eod_times = [eod_times] if pulse_fish else []
    pulse_unreliabilities = [0.0]
    """
    _, eod_times, pulse_unreliabilities = extract_pulsefish(data, samplerate)
    
    # calculate power spectra:
    psd_data = multi_psd(data, samplerate, **multi_psd_args(cfg))
            
    # find the fishes in the different powerspectra:
    h_kwargs = psd_peak_detection_args(cfg)
    h_kwargs.update(harmonic_groups_args(cfg))
    wave_eodfs_list = []
    for i, psd in enumerate(psd_data):
        wave_eodfs = harmonic_groups(psd[:,0], psd[:,1], verbose-1, **h_kwargs)[0]
        if verbose > 0:
            numpsdresolutions = cfg.value('numberPSDResolutions')
            print('fundamental frequencies detected in power spectrum of window %d at resolution %d:'
                  % (i//numpsdresolutions, i%numpsdresolutions))
            if len(wave_eodfs) > 0:
                print('  ' + ' '.join(['%.1f' % freq[0, 0] for freq in wave_eodfs]))
            else:
                print('  none')
        wave_eodfs_list.append(wave_eodfs)
    # filter the different wave_eodfs_list to get a wave_eodfs with consistent fishes:
    wave_eodfs = consistent_fishes(wave_eodfs_list,
                                   df_th=cfg.value('frequencyThreshold'))
    if verbose > 0:
        if len(wave_eodfs) > 0:
            print('fundamental frequencies consistent in all power spectra:')
            print('  ' + ' '.join(['%.1f' % freq[0, 0] for freq in wave_eodfs]))
        else:
            print('no fundamental frequencies are consistent in all power spectra')

    # analysis results:
    eod_props = []
    mean_eods = []
    spec_data = []
    peak_data = []
    power_thresh = None
    skip_reason = []
    
    # analyse eod waveform of pulse-fish:
    max_eods = cfg.value('eodMaxEODs')
    minfres = cfg.value('frequencyResolution')
    for eod_ts, unreliability in zip(eod_times, pulse_unreliabilities):
        if unreliability > 0.5:
            continue
        mean_eod, eod_times0 = \
            eod_waveform(data, samplerate, eod_ts,
                         win_fac=0.8, min_win=cfg.value('eodMinPulseSnippet'),
                         **eod_waveform_args(cfg))
        mean_eod, props, peaks, power = analyze_pulse(mean_eod, eod_times0,
                                                      freq_resolution=minfres,
                                                      **analyze_pulse_args(cfg))
        props['index'] = len(eod_props)
        props['clipped'] = clipped
        p_thresh = 5.0*props['EODf']**2.0 * power[:,1]
        if power_thresh is None:
            power_thresh = np.zeros(power.shape)
            power_thresh[:,0] = power[:,0]
            power_thresh[:,1] = p_thresh
        else:
            p_thresh = np.interp(power_thresh[:,0], power[:,0], p_thresh)
            power_thresh[:,1] = np.max(np.vstack(power_thresh[:,1], p_thresh), axis=1)
        # add good waveforms only:
        skips, msg = pulse_quality(0, clipped, props['rmssem'],
                                   **pulse_quality_args(cfg))
        if len(skips) == 0:
            eod_props.append(props)
            mean_eods.append(mean_eod)
            spec_data.append(power)
            peak_data.append(peaks)
            if verbose > 0:
                print('take %6.1fHz pulse fish: %s' % (props['EODf'], msg))
        else:
            skip_reason += ['%d %.1fHz pulse fish %s' % (0, props['EODf'], skips)]
            if verbose > 0:
                print('skip %6.1fHz pulse fish: %s (%s)' %
                      (props['EODf'], skips, msg))

    # remove wavefish below pulse fish power:
    if power_thresh is not None:
        n = len(wave_eodfs)
        for k, fish in enumerate(reversed(wave_eodfs)):
            df = power_thresh[1,0] - power_thresh[0,0]
            hfrac = float(np.sum(fish[:,1] < power_thresh[np.array(fish[:,0]//df, dtype=int),1]))/float(len(fish[:,1]))
            if hfrac >= 0.5:
                wave_eodfs.pop(n-1-k)
                if verbose > 0:
                    print('removed frequency %.1f Hz, because %.0f%% of the harmonics where below pulsefish threshold' % (fish[0,0], 100.0*hfrac))        

    # analyse EOD waveform of all wavefish:
    powers = np.array([np.sum(fish[:, 1]) for fish in wave_eodfs])
    wave_indices = np.zeros(len(wave_eodfs), dtype=np.int)
    for k, idx in enumerate(np.argsort(-powers)):
        fish = wave_eodfs[idx]
        eod_times = np.arange(0.0, len(data)/samplerate, 1.0/fish[0,0])
        mean_eod, eod_times = \
            eod_waveform(data, samplerate, eod_times, win_fac=3.0, min_win=0.0,
                         **eod_waveform_args(cfg))
        mean_eod, props, sdata, error_str = \
            analyze_wave(mean_eod, fish, **analyze_wave_args(cfg))
        if error_str:
            print(name + ': ' + error_str)
        props['n'] = len(eod_times)
        props['index'] = len(eod_props)
        props['clipped'] = clipped if k == 0 else 0.0
        # add good waveforms only:
        skips, msg = wave_quality(k, clipped, props['rmssem'],
                                  props['rmserror'], props['power'],
                                  sdata[1:,3], **wave_quality_args(cfg))
        if len(skips) == 0:
            wave_indices[idx] = props['index']
            eod_props.append(props)
            mean_eods.append(mean_eod)
            spec_data.append(sdata)
            peak_data.append([])
            if verbose > 0:
                print('%d take %6.1fHz wave fish: %s' % (idx, props['EODf'], msg))
        else:
            wave_indices[idx] = -1
            skip_reason += ['%d %.1fHz wave fish %s' % (idx, props['EODf'], skips)]
            if verbose > 0:
                print('%d skip waveform of %6.1fHz fish: %s (%s)' %
                      (idx, props['EODf'], skips, msg))
    return (psd_data, wave_eodfs, wave_indices, eod_props, mean_eods,
            spec_data, peak_data, power_thresh, skip_reason)


def remove_eod_files(output_basename, verbose, cfg):
    """ Remove all files from previous runs of thunderfish
    """
    ff = cfg.value('fileFormat')
    if ff == 'py':
        fext = 'py'
    else:
        fext = TableData.extensions[cfg.value('fileFormat')]
    # remove all files from previous runs of thunderfish:
    for fn in glob.glob('%s*.%s' % (output_basename, fext)):
        os.remove(fn)
        if verbose > 0:
            print('removed file %s' % fn)

            
def save_eods(output_basename, eod_props, mean_eods, spec_data, peak_data,
              wave_eodfs, wave_indices, unit, verbose, cfg):
    """ Save analysis results of all EODs to files.
    """
    if write_table_args(cfg)['table_format'] == 'py':
        with open(output_basename+'.py', 'w') as f:
            name = os.path.basename(output_basename)
            for k, sdata in enumerate(spec_data):
                # save wave fish only:
                if len(sdata)>0 and sdata.shape[1] > 2:
                    fish = dict(amplitudes=sdata[:,3], phases=sdata[:,5])
                    fish = normalize_wavefish(fish)
                    export_wavefish(fish, name+'-%d_harmonics' % k, f)
    else:
        # all wave fish in wave_eodfs:
        if len(wave_eodfs) > 0:
            fp = save_wave_eodfs(wave_eodfs, wave_indices, output_basename,
                                 **write_table_args(cfg))
            if verbose > 0:
                print('wrote file %s' % fp)
        # all wave and pulse fish:
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
            if verbose > 0 and fp is not None:
                print('wrote file %s' % fp)
        # wave fish properties:
        fp = save_wave_fish(eod_props, unit, output_basename,
                            **write_table_args(cfg))
        if verbose > 0 and fp:
            print('wrote file %s' % fp)
        # pulse fish properties:
        fp = save_pulse_fish(eod_props, unit, output_basename,
                             **write_table_args(cfg))
        if verbose > 0 and fp:
            print('wrote file %s' % fp)

                            
def plot_eods(base_name, raw_data, samplerate, idx0, idx1,
              clipped, wave_eodfs, mean_eods, eod_props, peak_data, spec_data,
              indices, unit, psd_data, power_thresh, label_power,
              max_freq=3000.0, interactive=True):
    """
    Creates an output plot for the Thunderfish program.

    This output contains the raw trace where the analysis window is
    marked, the power-spectrum of this analysis window where the
    detected fish are marked, a mean EOD plot, a histogram of the inter
    EOD interval and further information about the recording that is
    analysed.

    Parameters
    ----------
    base_name: string
        Basename of audio_file.
    raw_data: array
        Dataset.
    samplerate: float
        Sampling rate of the dataset.
    idx0: float
        Index of the beginning of the analysis window in the dataset.
    idx1: float
        Index of the end of the analysis window in the dataset.
    clipped: float
        Fraction of clipped amplitudes.
    wave_eodfs: array
        Frequency and power of fundamental frequency/harmonics of several fish.
    mean_eods: list of 2-D arrays with time, mean and std.
        Mean trace for the mean EOD plot.
    eod_props: list of dict
        Properties for each waveform in mean_eods.
    peak_data: list of 2_D arrays
        For each pulsefish a list of peak properties
        (index, time, and amplitude).
    spec_data: list of 2_D arrays
        For each pulsefish a power spectrum of the single pulse and for
        each wavefish the relative amplitudes and phases of the harmonics.
    indices: list of int or None
        Indices of the fish in eod_props to be plotted.
        If None try to plot all.
    unit: string
        Unit of the trace and the mean EOD.
    psd_data: array
        Power spectrum of the analysed data for different frequency resolutions.
    power_thresh:  2 D array or None
        Frequency (first column) and power (second column) of threshold
        derived from single pulse spectra to discard false wave fish.
    label_power: boolean
        If `True` put the power in decibel in addition to the frequency
        into the legend.
    interactive: bool
        If True install some keyboard interaction.

    Returns
    -------
    fig: plt.figure
        Figure with the plots.
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
    if interactive:
        fig.canvas.mpl_connect('key_press_event', keypress)
    ax1 = fig.add_axes([0.02, 0.9, 0.96, 0.1])   # title
    ax2 = fig.add_axes([0.075, 0.06, 0.9, 0.09]) # whole trace
    ax3 = fig.add_axes([0.075, 0.6, 0.7, 0.3])   # psd
    ax4 = fig.add_axes([0.075, 0.2, 0.4, 0.3])   # mean eod 1
    ax5 = fig.add_axes([0.575, 0.2, 0.4, 0.3])   # pulse/amplitude spectrum or mean eod 2
    ax6 = fig.add_axes([0.075, 0.2, 0.4, 0.14])  # mean eod 3
    ax7 = fig.add_axes([0.575, 0.2, 0.4, 0.14])  # phase spectrum or mean eod 4
    ax8 = fig.add_axes([0.075, 0.6, 0.4, 0.3])   # recording zoom-in
    
    # plot title:
    ax1.text(0.0, .64, base_name, fontsize=22)
        
    ax1.text(1.0, .72, 'thunderfish by Benda-Lab', fontsize=16, ha='right')
    ax1.text(1.0, .45, 'Version %s' % __version__, fontsize=16, ha='right')
    ax1.set_frame_on(False)
    ax1.get_xaxis().set_visible(False)
    ax1.get_yaxis().set_visible(False)

    # count number of fish types to be plotted:
    if indices is None:
        indices = list(range(len(eod_props)))
    indices = np.array(indices, dtype=np.int)
    nwave = 0
    npulse = 0
    for idx in indices:
        if eod_props[idx]['type'] == 'pulse':
            npulse += 1
        elif eod_props[idx]['type'] == 'wave':
            nwave += 1
    
    ############

    # plot trace
    plot_best_data(ax2, raw_data, samplerate, unit, idx0, idx1, clipped)
    
    ############

    colors, markers = colors_markers()
    
    # plot psd
    if len(wave_eodfs) > 0 or len(eod_props) == 0:
        if power_thresh is not None:
            ax3.plot(power_thresh[:,0], decibel(power_thresh[:,1]), '#CCCCCC', lw=1)
        if len(wave_eodfs) > 0:
            if len(wave_eodfs) == 1:
                title = None
                bbox = (1.0, 1.0)
                loc = 'upper right'
                legend_rows = 2
            else:
                title = '%d EOD frequencies' % len(wave_eodfs)
                bbox = (1.0, 1.1)
                loc = 'upper left'
                legend_rows = 12
            plot_harmonic_groups(ax3, wave_eodfs, max_freq=max_freq, max_groups=0,
                                 sort_by_freq=True, label_power=label_power,
                                 colors=colors, markers=markers,
                                 legend_rows=legend_rows, frameon=False,
                                 bbox_to_anchor=bbox, loc=loc, title=title)
        plot_decibel_psd(ax3, psd_data[0][:,0], psd_data[0][:,1], max_freq=max_freq,
                         color='blue')
        ax3.yaxis.set_major_locator(ticker.MaxNLocator(6))
        if len(wave_eodfs) == 1:
            ax3.get_legend().set_visible(False)
            label = '%6.1f Hz' % wave_eodfs[0][0, 0]
            ax3.set_title('Powerspectrum: %s' % label, y=1.05, fontsize=14)
        else:
            ax3.set_title('Powerspectrum', y=1.05, fontsize=14)
    else:
        ax3.set_visible(False)        
        ax8.set_position([0.075, 0.6, 0.9, 0.3])
    
    ############

    # plot recording
    if len(wave_eodfs) <= 2 and len(eod_props) > 0:
        ax3.set_position([0.575, 0.6, 0.4, 0.3])
        width = 0.1
        if len(indices) > 0:
            if eod_props[indices[0]]['type'] == 'wave':
                width = 5.0/eod_props[indices[0]]['EODf']
            else:
                if len(wave_eodfs) > 0:
                    width = 3.0/eod_props[indices[0]]['EODf']
                else:
                    width = 10.0/eod_props[indices[0]]['EODf']
        width = (1+width//0.005)*0.005
        plot_eod_recording(ax8, raw_data[idx0:idx1], samplerate, width, unit,
                           idx0/samplerate)
        plot_pulse_eods(ax8, raw_data[idx0:idx1], samplerate, eod_props, idx0/samplerate,
                        colors=colors[3:], markers=markers[3:])
        ax8.set_title('Recording', fontsize=14, y=1.05)
    else:
        ax8.set_visible(False)        

    ##########

    # sort indices by p-p amplitude:
    pp_ampls = [eod_props[idx]['p-p-amplitude'] for idx in indices]
    pp_indices = np.argsort(pp_ampls)[::-1]
    usedax = [False, False, False, False]
    eodaxes = [ax4, ax5, ax6, ax7]
    if len(indices) > 2:
        ax4.set_position([0.075, 0.39, 0.4, 0.13])
        ax5.set_position([0.575, 0.39, 0.4, 0.13])
        ax6.set_position([0.075, 0.2, 0.4, 0.13])
        ax7.set_position([0.575, 0.2, 0.4, 0.13])
        ty = 1.1
        ny = 4
    else:
        ty = 1.08
        ny = 6

    # plot mean EOD
    for k, (axeod, idx) in enumerate(zip(eodaxes, indices[pp_indices])):
        mean_eod = mean_eods[idx]
        props = eod_props[idx]
        peaks = peak_data[idx]
        usedax[k] = True
        axeod.text(-0.1, ty, '{EODf:.1f} Hz {type} fish'.format(**props),
                   transform = axeod.transAxes, fontsize=14)
        axeod.text(0.5, ty, 'Averaged EOD',
                   transform = axeod.transAxes, fontsize=14, ha='center')
        if len(unit) == 0 or unit == 'a.u.':
            unit = ''
        tau = props['tau'] if 'tau' in props else None
        plot_eod_waveform(axeod, mean_eod, peaks, unit, tau=tau)
        props['unit'] = unit
        props['eods'] = 'EODs' if props['n'] > 1 else 'EOD'
        label = 'p-p amplitude = {p-p-amplitude:.3g} {unit}\nn = {n} {eods}\n'.format(**props)
        if props['flipped']:
            label += 'flipped\n'
        if -mean_eod[0,0] < 0.6*mean_eod[-1,0]:
            axeod.text(0.97, 0.97, label, transform = axeod.transAxes,
                       va='top', ha='right')
        else:
            axeod.text(0.03, 0.97, label, transform = axeod.transAxes, va='top')
        axeod.yaxis.set_major_locator(ticker.MaxNLocator(ny))
        if props['type'] == 'wave':
            lim = 750.0/props['EODf']
            axeod.set_xlim([-lim, +lim])
        if len(indices) > 2 and k < 2:
            axeod.set_xlabel('')
            

    ################

    # plot spectra:
    if len(indices) == 1:
        usedax[1] = True
        if  eod_props[indices[0]]['type'] == 'pulse':
            plot_pulse_spectrum(ax5, spec_data[indices[0]], eod_props[indices[0]])
            ax5.set_title('Single pulse spectrum', fontsize=14, y=1.05)
        else:
            usedax[3] = True
            ax5.set_position([0.575, 0.36, 0.4, 0.14])
            plot_wave_spectrum(ax5, ax7, spec_data[indices[0]], eod_props[indices[0]],
                               unit)
            ax5.set_title('Amplitude and phase spectrum', fontsize=14, y=1.05)
            ax5.set_xticklabels([])
            ax5.yaxis.set_major_locator(ticker.MaxNLocator(4))

    ################

    # plot data trace in case no fish was found:
    if not usedax[0]:
        ax3.set_position([0.075, 0.6, 0.9, 0.3])   # enlarge psd
        ax4.set_position([0.075, 0.2, 0.9, 0.3])
        rdata = raw_data[idx0:idx1] if idx1 > idx0 else raw_data
        plot_eod_recording(ax4, rdata, samplerate, 0.1, unit, idx0/samplerate)
        ax4.set_title('Recording', fontsize=14, y=1.05)
        usedax[0] = True

    # hide unused axes:
    for used, ax in zip(usedax, eodaxes):
        if not used:
            ax.set_visible(False)
            
    # cosmetics
    for ax in [ax2, ax3, ax4, ax5, ax6, ax7, ax8]:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.get_xaxis().tick_bottom()
        ax.get_yaxis().tick_left()
        
    return fig


def thunderfish(filename, cfg, channel=0, save_data=False, save_plot=False,
                save_subplots=False, output_folder='.', keep_path=False,
                show_bestwindow=False, verbose=0):
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
        raw_data, samplerate, unit = load_data(filename, channel,
                                               verbose=verbose)
    except IOError as e:
        return '%s: failed to open file: %s' % (filename, str(e))
    if len(raw_data) <= 1:
        return '%s: empty data file' % filename

    # best_window:
    data, idx0, idx1, clipped = find_best_window(raw_data, samplerate, cfg,
                                                 show_bestwindow)
    if show_bestwindow:
        return None
    found_bestwindow = idx1 > 0
    if not found_bestwindow:
        print(filename + ': not enough data for requested best window length. You may want to adjust the bestWindowSize parameter in the configuration file.')

    # detect EODs in the data:
    psd_data, wave_eodfs, wave_indices, eod_props, \
    mean_eods, spec_data, peak_data, power_thresh, skip_reason = \
      detect_eods(data, samplerate, clipped, filename, verbose, cfg)
    if not found_bestwindow:
        wave_eodfs = []
        wave_indices = []
        eod_props = []
        mean_eods = []

    # warning message in case no fish has been found:
    if found_bestwindow and not eod_props :
        msg = ', '.join(skip_reason)
        if msg:
            print(filename + ': no fish found: %s' % msg)
        else:
            print(filename + ': no fish found.')

    # save results to files:
    output_basename = os.path.join(output_folder, outfilename)
    if save_data:
        remove_eod_files(output_basename, verbose, cfg)
        if found_bestwindow:
            if keep_path:
                outpath = os.path.dirname(output_basename)
                if not os.path.exists(outpath):
                    if verbose > 0:
                        print('mkdir %s' % outpath)
                    os.makedirs(outpath)
            save_eods(output_basename, eod_props, mean_eods, spec_data, peak_data,
                      wave_eodfs, wave_indices, unit, verbose, cfg)

    if save_plot or not save_data:
        fig = plot_eods(outfilename, raw_data, samplerate, idx0, idx1, clipped,
                        wave_eodfs, mean_eods, eod_props, peak_data, spec_data,
                        None, unit, psd_data, power_thresh, True, 3000.0,
                        interactive=not save_data)
        if save_plot:
            # save figure as pdf:
            fig.savefig(output_basename + '.pdf')
            if save_subplots:
                # make figures and call plot functions on them individually
                print('sorry, saving subplots separately is not implemented yet!')
            plt.close()
        elif not save_data:
            fig.canvas.set_window_title('thunderfish')
            plt.show()


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
                        help='verbosity level. Increase by specifying -v multiple times, or like -vvv')
    parser.add_argument('-c', dest='save_config', action='store_true',
                        help='save configuration to file {0} after reading all configuration files'.format(cfgfile))
    parser.add_argument('--channel', default=0, type=int,
                        help='channel to be analyzed (defaults to first channel)')
    parser.add_argument('-j', dest='jobs', nargs='?', type=int, default=None, const=0,
                        help='number of jobs run in parallel. Without argument use all CPU cores.')
    parser.add_argument('-s', dest='save_data', action='store_true',
                        help='save analysis results to files')
    parser.add_argument('-f', dest='format', default='auto', type=str,
                        choices=TableData.formats + ['py'],
                        help='file format used for saving analysis results, defaults to the format specified in the configuration file or "dat"')
    parser.add_argument('-p', dest='save_plot', action='store_true',
                        help='save output plot as pdf file')
    parser.add_argument('-P', dest='save_subplots', action='store_true',
                        help='save subplots as separate pdf files')
    parser.add_argument('-o', dest='outpath', default='.', type=str,
                        help='path where to store results and figures (defaults to current working directory)')
    parser.add_argument('-k', dest='keep_path', action='store_true',
                        help='keep path of input file when saving analysis files, i.e. append path of input file to OUTPATH')
    parser.add_argument('-b', dest='show_bestwindow', action='store_true',
                        help='show the cost function of the best window algorithm')
    parser.add_argument('file', nargs='*', default='', type=str,
                        help='name of a file with time series data of an EOD recording')
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
        print('  > thunderfish -j -s -p -o results/ river1/*.wav')
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
        # save configuration:
        file_name = args.file[0] if len(args.file) else ''
        configuration(cfgfile, args.save_config, file_name, verbose)
        exit()
    elif len(args.file) == 0:
        parser.error('you need to specify at least one file for the analysis')

    # analyze data files:
    cfg = configuration(cfgfile, False, args.file[0], verbose-1)
    if args.format != 'auto':
        cfg.set('fileFormat', args.format)
    # save plots:
    if args.save_subplots:
        args.save_plot = True
    # create output folder:
    if args.save_data or args.save_plot:
        if not os.path.exists(args.outpath):
            if verbose > 1:
                print('mkdir %s' % args.outpath)
            os.makedirs(args.outpath)
    # run on pool:
    global pool_args
    pool_args = (cfg, args.channel, args.save_data,
                 args.save_plot, args.save_subplots, args.outpath, args.keep_path,
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
