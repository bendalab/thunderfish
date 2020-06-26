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
import matplotlib.gridspec as gridspec
import matplotlib.ticker as ticker
import matplotlib.lines as ml
from matplotlib.transforms import Bbox
from matplotlib.backends.backend_pdf import PdfPages
from multiprocessing import Pool, freeze_support, cpu_count
from audioio import play, fade
from .version import __version__, __year__
from .configfile import ConfigFile
from .dataloader import load_data
from .bestwindow import add_clip_config, add_best_window_config
from .bestwindow import clip_args, best_window_args
from .bestwindow import find_best_window, plot_best_data
from .checkpulse import check_pulse, add_check_pulse_config, check_pulse_args
from .lizpulses import extract_pulsefish
from .powerspectrum import decibel, plot_decibel_psd, multi_psd
from .powerspectrum import add_multi_psd_config, multi_psd_args
from .harmonics import add_psd_peak_detection_config, add_harmonic_groups_config
from .harmonics import harmonic_groups, harmonic_groups_args, psd_peak_detection_args
from .harmonics import colors_markers, plot_harmonic_groups
from .consistentfishes import consistent_fishes
from .eodanalysis import eod_waveform, analyze_wave, analyze_pulse, clipped_fraction
from .eodanalysis import plot_eod_recording, plot_pulse_eods
from .eodanalysis import plot_eod_waveform, plot_eod_snippets
from .eodanalysis import plot_pulse_spectrum, plot_wave_spectrum
from .eodanalysis import add_eod_analysis_config, eod_waveform_args
from .eodanalysis import analyze_wave_args, analyze_pulse_args
from .eodanalysis import wave_quality, wave_quality_args, add_eod_quality_config
from .eodanalysis import pulse_quality, pulse_quality_args
from .eodanalysis import save_eod_waveform, save_wave_eodfs, save_wave_fish, save_pulse_fish
from .eodanalysis import save_wave_spectrum, save_pulse_spectrum, save_pulse_peaks
from .fakefish import normalize_wavefish, export_wavefish
from .tabledata import TableData, add_write_table_config, write_table_args

import matplotlib
matplotlib.use('TkAgg')

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
    add_check_pulse_config(cfg)
    add_eod_analysis_config(cfg, min_pulse_win=0.004)
    del cfg['eodSnippetFac']
    del cfg['eodMinSnippet']
    del cfg['eodMinSem']
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


def detect_eods(data, samplerate, clipped, min_clip, max_clip, name, verbose, cfg):
    """ Detect EODs of all fish present in the data.

    Parameters
    ----------
    data: array of floats
        The recording in which to detect EODs.
    samplerate: float
        Sampling rate of the dataset.
    clipped: float
        Fraction of clipped amplitudes.
    min_clip: float
        Minimum amplitude that is not clipped.
    max_clip: float
        Maximum amplitude that is not clipped.
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
    # detect wave fish:
    psd_data = multi_psd(data, samplerate, **multi_psd_args(cfg))
    h_kwargs = psd_peak_detection_args(cfg)
    h_kwargs.update(harmonic_groups_args(cfg))
    wave_eodfs_list = []
    for i, psd in enumerate(psd_data):
        wave_eodfs = harmonic_groups(psd[:,0], psd[:,1], verbose-1, **h_kwargs)[0]
        if verbose > 0 and len(psd_data) > 1:
            numpsdresolutions = cfg.value('numberPSDResolutions')
            print('fundamental frequencies detected in power spectrum of window %d at resolution %d:'
                  % (i//numpsdresolutions, i%numpsdresolutions))
            if len(wave_eodfs) > 0:
                print('  ' + ' '.join(['%.1f' % freq[0, 0] for freq in wave_eodfs]))
            else:
                print('  none')
        wave_eodfs_list.append(wave_eodfs)
    wave_eodfs = consistent_fishes(wave_eodfs_list,
                                   df_th=cfg.value('frequencyThreshold'))
    if verbose > 0:
        if len(wave_eodfs) > 0:
            print('fundamental frequencies consistent in all power spectra:')
            print('  ' + ' '.join(['%.1f' % freq[0, 0] for freq in wave_eodfs]))
        else:
            print('no fundamental frequencies are consistent in all power spectra')

    # detect pulse fish:
    _, eod_times, eod_peaktimes, zoom_window = extract_pulsefish(data, samplerate, verbose=verbose)
            
    # analysis results:
    eod_props = []
    mean_eods = []
    spec_data = []
    peak_data = []
    power_thresh = None
    skip_reason = []
    
    # analyse eod waveform of pulse-fish:
    min_freq_res = cfg.value('frequencyResolution')

    for k, (eod_ts, eod_pts) in enumerate(zip(eod_times, eod_peaktimes)):
        mean_eod, eod_times0 = \
            eod_waveform(data, samplerate, eod_ts, win_fac=0.8,
                         min_win=cfg.value('eodMinPulseSnippet'),
                         min_sem=False, **eod_waveform_args(cfg))
        mean_eod, props, peaks, power = analyze_pulse(mean_eod, eod_times0,
                                                      freq_resolution=min_freq_res,
                                                      **analyze_pulse_args(cfg))
        if len(peaks) == 0:
            print('error: no peaks in pulse EOD detected')
            continue
        clipped_frac = clipped_fraction(data, samplerate, eod_times0, mean_eod,
                                        min_clip, max_clip)

        props['peaktimes'] = eod_pts      # XXX that should go into analyze pulse
        props['index'] = len(eod_props)
        props['clipped'] = clipped_frac

        # add good waveforms only:
        skips, msg, skipped_clipped = pulse_quality(clipped_frac, props['rmssem'], peaks,
                                                    **pulse_quality_args(cfg))

        if len(skips) == 0:
            eod_props.append(props)
            mean_eods.append(mean_eod)
            spec_data.append(power)
            peak_data.append(peaks)
            if verbose > 0:
                print('take %6.1fHz pulse fish: %s' % (props['EODf'], msg))
        else:
            skip_reason += ['%.1fHz pulse fish %s' % (props['EODf'], skips)]
            if verbose > 0:
                print('skip %6.1fHz pulse fish: %s (%s)' %
                      (props['EODf'], skips, msg))

        # threshold for wave fish peaks based on single pulse spectra:
        if len(skips) == 0 or skipped_clipped:
            i0 = np.argmin(np.abs(mean_eod[:,0]))
            i1 = len(mean_eod) - i0
            pulse_data = np.zeros(len(data))
            for t in props['peaktimes']:
                idx = int(t*samplerate)
                ii0 = i0 if idx-i0 >= 0 else idx
                ii1 = i1 if idx+i1 < len(pulse_data) else len(pulse_data)-1-idx
                pulse_data[idx-ii0:idx+ii1] = mean_eod[i0-ii0:i0+ii1,1]
            pulse_psd = multi_psd(pulse_data, samplerate, **multi_psd_args(cfg))
            p_thresh = pulse_psd[0]
            p_thresh[:,1] *= len(data)/samplerate/props['period']/len(props['peaktimes'])
            p_thresh[:,1] *= 3.0
            if power_thresh is None:
                power_thresh = p_thresh
            else:
                power_thresh[:,1] = np.max(np.vstack((power_thresh[:,1].T, p_thresh[:,1])),
                                           axis=0)
                
    # remove wavefish below pulse fish power:
    if power_thresh is not None:
        n = len(wave_eodfs)
        maxh = 3
        df = power_thresh[1,0] - power_thresh[0,0]
        for k, fish in enumerate(reversed(wave_eodfs)):
            idx = np.array(fish[:maxh,0]//df, dtype=int)
            for offs in range(-2, 3):
                nbelow = np.sum(fish[:maxh,1] < power_thresh[idx+offs,1])
                if nbelow > 0:
                    wave_eodfs.pop(n-1-k)
                    if verbose > 0:
                        print('skip %6.1fHz wave  fish: %2d harmonics are below pulsefish threshold' % (fish[0,0], nbelow))
                    break

    # analyse EOD waveform of all wavefish:
    powers = np.array([np.sum(fish[:, 1]) for fish in wave_eodfs])
    wave_indices = np.zeros(len(wave_eodfs), dtype=np.int)
    for k, idx in enumerate(np.argsort(-powers)):
        fish = wave_eodfs[idx]
        eod_times = np.arange(0.0, len(data)/samplerate, 1.0/fish[0,0])
        mean_eod, eod_times = \
            eod_waveform(data, samplerate, eod_times, win_fac=3.0, min_win=0.0,
                         min_sem=(k==0), **eod_waveform_args(cfg))
        mean_eod, props, sdata, error_str = \
            analyze_wave(mean_eod, fish, **analyze_wave_args(cfg))
        if error_str:
            print(name + ': ' + error_str)
        clipped_frac = clipped_fraction(data, samplerate, eod_times, mean_eod,
                                        min_clip, max_clip)
        props['n'] = len(eod_times)
        props['index'] = len(eod_props)
        props['clipped'] = clipped_frac
        # add good waveforms only:
        skips, msg = wave_quality(clipped_frac, props['ncrossings'],
                                  props['rmssem'], props['rmserror'], props['power'],
                                  **wave_quality_args(cfg))
        if len(skips) == 0:
            wave_indices[idx] = props['index']
            eod_props.append(props)
            mean_eods.append(mean_eod)
            spec_data.append(sdata)
            peak_data.append([])
            if verbose > 0:
                print('take %6.1fHz wave  fish: %s' % (props['EODf'], msg))
        else:
            wave_indices[idx] = -1
            skip_reason += ['%.1fHz wave fish %s' % (props['EODf'], skips)]
            if verbose > 0:
                print('skip %6.1fHz wave  fish: %s (%s)' %
                      (props['EODf'], skips, msg))
    return (psd_data, wave_eodfs, wave_indices, eod_props, mean_eods,
            spec_data, peak_data, power_thresh, skip_reason, zoom_window)


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


def plot_style():
    """ Set style of plots.
    """
    plt.rcParams['axes.facecolor'] = 'none'
    plt.rcParams['xtick.direction'] = 'out'
    plt.rcParams['ytick.direction'] = 'out'


def axes_style(ax):
    """ Fix style of axes.

    Parameters
    ----------
    ax: matplotlib axes
    """
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()

                                
def plot_eods(base_name, raw_data, samplerate, idx0, idx1, clipped,
              psd_data, wave_eodfs, wave_indices, mean_eods, eod_props, peak_data, spec_data,
              indices, unit, zoom_window, n_snippets=10, power_thresh=None, label_power=True,
              log_freq=False, min_freq=0.0, max_freq=3000.0, interactive=True, verbose=0):
    """
    Creates an output plot for the thunderfish program.

    This output contains the raw trace where the analysis window is
    marked, the power-spectrum of this analysis window where the
    detected fish are marked, plots of averaged EOD plots, and
    spectra of the EOD waveforms.

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
    psd_data: 2D array
        Power spectrum (frequencies and power) of the analysed data.
    wave_eodfs: array
        Frequency and power of fundamental frequency/harmonics of several fish.
    wave_indices: array of int
        Indices of wave fish mapping from wave_eodfs to eod_props.
        If negative, then that EOD frequency has no waveform described in eod_props.
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
    n_snippets: int
        Number of EOD waveform snippets to be plotted. If zero do not plot any.
    power_thresh:  2 D array or None
        Frequency (first column) and power (second column) of threshold
        derived from single pulse spectra to discard false wave fish.
    label_power: boolean
        If `True` put the power in decibel in addition to the frequency
        into the legend.
    log_freq: boolean
        Logarithmic (True) or linear (False) frequency axis of power spectrum of recording.
    min_freq: float
        Limits of frequency axis of power spectrum of recording
        are set to `(min_freq, max_freq)` if `max_freq` is greater than zero
    max_freq: float
        Limits of frequency axis of power spectrum of recording
        are set to `(min_freq, max_freq)` and limits of power axis are computed
        from powers below max_freq if `max_freq` is greater than zero
    interactive: bool
        If True install some keyboard interaction.
    verbose: int
        Print out information about data to be plotted if greater than zero.

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

    def recording_format_coord(x, y):
        return 'full recording: x=%.3f s, y=%.3f' % (x, y)

    def recordingzoom_format_coord(x, y):
        return 'recording zoom-in: x=%.3f s, y=%.3f' % (x, y)
            
    def psd_format_coord(x, y):
        return 'power spectrum: x=%.1f Hz, y=%.1f dB' % (x, y)

    def meaneod_format_coord(x, y):
        return 'mean EOD waveform: x=%.2f ms, y=%.3f' % (x, y)

    def ampl_format_coord(x, y):
        return u'amplitude spectrum: x=%.0f, y=%.2f' % (x, y)

    def phase_format_coord(x, y):
        return u'phase spectrum: x=%.0f, y=%.2f \u03c0' % (x, y/np.pi)
            
    def pulsepsd_format_coord(x, y):
        return 'single pulse power spectrum: x=%.1f Hz, y=%.1f dB' % (x, y)

    plot_style()
    fig = plt.figure(facecolor='white', figsize=(14., 10.))
    if interactive:
        fig.canvas.mpl_connect('key_press_event', keypress)
    ax1 = fig.add_axes([0.02, 0.9, 0.96, 0.1])   # title
    ax2 = fig.add_axes([0.075, 0.6, 0.4, 0.3])   # recording zoom-in
    ax3 = fig.add_axes([0.075, 0.6, 0.7, 0.3])   # psd
    ax4 = fig.add_axes([0.075, 0.2, 0.4, 0.3])   # mean eod 1
    ax5 = fig.add_axes([0.575, 0.2, 0.4, 0.3])   # pulse/amplitude spectrum or mean eod 2
    ax6 = fig.add_axes([0.075, 0.2, 0.4, 0.14])  # mean eod 3
    ax7 = fig.add_axes([0.575, 0.2, 0.4, 0.14])  # phase spectrum or mean eod 4
    ax8 = fig.add_axes([0.075, 0.06, 0.9, 0.09]) # whole trace
    
    # plot title:
    ax1.text(0.0, .64, base_name, fontsize=22)
        
    ax1.text(1.0, .72, 'thunderfish by Benda-Lab', fontsize=16, ha='right')
    ax1.text(1.0, .45, 'Version %s' % __version__, fontsize=16, ha='right')
    ax1.set_frame_on(False)
    ax1.set_axis_off()
    ax1.set_navigate(False)

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

    if verbose > 0:
        print('plot: %2d waveforms: %2d wave fish, %2d pulse fish and %2d EOD frequencies.'
              % (len(indices), nwave, npulse, len(wave_eodfs)))
    
    ############

    force_both = True   # XXX set to True for debugging pulse and wave detection
    #force_both = False

    # best window data:
    data = raw_data[idx0:idx1] if idx1 > idx0 else raw_data
    
    # plot psd
    wave_colors, wave_markers = colors_markers()
    if len(indices) == 0 or len(wave_eodfs) > 0 or npulse == 0 or force_both:
        legend_inside = True
        if len(indices) == 0:
            ax3.set_position([0.075, 0.2, 0.9, 0.3]) # bottom, wide
        elif len(wave_eodfs) <= 2 or npulse > 0 or force_both:
            ax3.set_position([0.575, 0.6, 0.4, 0.3]) # top, right
        else:
            ax3.set_position([0.075, 0.6, 0.7, 0.3]) # top, wide
            legend_inside = False
        if power_thresh is not None:
            ax3.plot(power_thresh[:,0], decibel(power_thresh[:,1]), '#CCCCCC', lw=1)
        if len(wave_eodfs) > 0:
            kwargs = {}
            if len(wave_eodfs) > 1:
                title = '%d EOD frequencies' % len(wave_eodfs)
                kwargs = {'title': title if len(wave_eodfs) > 2 else None }
                if legend_inside:
                    kwargs.update({'bbox_to_anchor': (1.05, 1.1),
                                   'loc': 'upper right', 'legend_rows': 10})
                else:
                    kwargs.update({'bbox_to_anchor': (1.0, 1.1),
                                   'loc': 'upper left', 'legend_rows': 12})
            plot_harmonic_groups(ax3, wave_eodfs, wave_indices, max_groups=0,
                                 sort_by_freq=True, label_power=label_power,
                                 colors=wave_colors, markers=wave_markers,
                                 frameon=False, **kwargs)
        plot_decibel_psd(ax3, psd_data[:,0], psd_data[:,1], log_freq=log_freq,
                         min_freq=min_freq, max_freq=max_freq, ymarg=5.0, color='blue')
        ax3.yaxis.set_major_locator(ticker.MaxNLocator(6))
        if len(wave_eodfs) == 1:
            ax3.get_legend().set_visible(False)
            label = '%6.1f Hz' % wave_eodfs[0][0, 0]
            ax3.set_title('Powerspectrum: %s' % label, y=1.05, fontsize=14)
        else:
            ax3.set_title('Powerspectrum', y=1.05, fontsize=14)
        ax3.format_coord = psd_format_coord
    else:
        ax3.set_visible(False)
    
    ############

    # plot recording
    pulse_colors, pulse_markers = colors_markers()
    pulse_colors = pulse_colors[3:]
    pulse_markers = pulse_markers[3:]
    if len(indices) == 0 or npulse > 0 or len(wave_eodfs) <= 2 or force_both:
        if len(indices) > 0 and ((len(wave_eodfs) > 0 and len(wave_eodfs) <= 2) or force_both):
            ax2.set_position([0.075, 0.6, 0.4, 0.3]) # top, left
        else:
            ax2.set_position([0.075, 0.6, 0.9, 0.3]) # top, wide
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
        plot_eod_recording(ax2, data, samplerate, width, unit, idx0/samplerate)
        plot_pulse_eods(ax2, data, samplerate, zoom_window, width, eod_props, idx0/samplerate,
                        colors=pulse_colors, markers=pulse_markers)

        ax2.set_title('Recording', fontsize=14, y=1.05)
        ax2.format_coord = recordingzoom_format_coord
    else:
        ax2.set_visible(False)        

    ##########

    # sort indices by p-p amplitude:
    pp_ampls = [eod_props[idx]['p-p-amplitude'] for idx in indices]
    pp_indices = np.argsort(pp_ampls)[::-1]
    eodaxes = [ax4, ax5, ax6, ax7]
    for ax in eodaxes:
        ax.set_visible(False)
    if len(indices) > 2:
        ax4.set_position([0.075, 0.39, 0.4, 0.13])
        ax5.set_position([0.575, 0.39, 0.4, 0.13])
        ax6.set_position([0.075, 0.2, 0.4, 0.13])
        ax7.set_position([0.575, 0.2, 0.4, 0.13])
        ty = 1.10
        my = 1.16
        ny = 4
    else:
        ty = 1.08
        my = 1.10
        ny = 6

    # plot mean EOD
    w, _ = ax3.get_legend_handles_labels()
    eodf_labels = [wi.get_label().split()[0] for wi in w]
    legend_wave_eodfs = np.array([float(f) if f[0] != '(' else np.nan for f in eodf_labels])
    p, _ = ax2.get_legend_handles_labels()
    eodf_labels = [pi.get_label().split()[0] for pi in p]
    legend_pulse_eodfs = np.array([float(f) if f[0] != '(' else np.nan for f in eodf_labels])
    for k, (axeod, idx) in enumerate(zip(eodaxes, indices[pp_indices])):
        mean_eod = mean_eods[idx]
        props = eod_props[idx]
        peaks = peak_data[idx]
        axeod.set_visible(True)
        axeod.yaxis.set_major_locator(ticker.MaxNLocator(ny))
        if len(indices) > 1:
            axeod.text(0.3, ty, '{EODf:.1f} Hz {type} fish'.format(**props),
                       transform=axeod.transAxes, fontsize=14)
            mx = 0.25
        else:
            axeod.text(-0.1, ty, '{EODf:.1f} Hz {type} fish'.format(**props),
                       transform=axeod.transAxes, fontsize=14)
            axeod.text(0.5, ty, 'Averaged EOD',
                       transform=axeod.transAxes, fontsize=14, ha='center')
            mx = -0.14
        eodf = props['EODf']
        if props['type'] == 'wave':
            wk = np.nanargmin(np.abs(legend_wave_eodfs - eodf))
            ma = ml.Line2D([mx], [my], color=w[wk].get_color(), marker=w[wk].get_marker(),
                           markersize=w[wk].get_markersize(), mec='none', clip_on=False,
                           label=w[wk].get_label(), transform=axeod.transAxes)
            axeod.add_line(ma)
        else:
            pk = np.argmin(np.abs(legend_pulse_eodfs - eodf))
            ma = ml.Line2D([mx], [my], color=p[pk].get_color(), marker=p[pk].get_marker(),
                           markersize=p[pk].get_markersize(), mec='none', clip_on=False,
                           label=p[pk].get_label(), transform=axeod.transAxes)
            axeod.add_line(ma)
        plot_eod_waveform(axeod, mean_eod, props, peaks, unit)
        if props['type'] == 'pulse':
            plot_eod_snippets(axeod, data, samplerate, mean_eod[0,0], mean_eod[-1,0],
                              props['peaktimes'], n_snippets)
        if len(indices) > 2 and k < 2:
            axeod.set_xlabel('')
        axeod.format_coord = meaneod_format_coord
            

    ################

    # plot spectra:
    if len(indices) == 1:
        ax5.set_visible(True)
        if  eod_props[indices[0]]['type'] == 'pulse':
            plot_pulse_spectrum(ax5, spec_data[indices[0]], eod_props[indices[0]])
            ax5.set_title('Single pulse spectrum', fontsize=14, y=1.05)
            ax5.format_coord = pulsepsd_format_coord
        else:
            ax7.set_visible(True)
            ax5.set_position([0.575, 0.36, 0.4, 0.14])
            plot_wave_spectrum(ax5, ax7, spec_data[indices[0]], eod_props[indices[0]],
                               unit)
            ax5.set_title('Amplitude and phase spectrum', fontsize=14, y=1.05)
            ax5.set_xticklabels([])
            ax5.yaxis.set_major_locator(ticker.MaxNLocator(4))
            ax5.format_coord = ampl_format_coord
            ax7.format_coord = phase_format_coord
    
    ############

    # plot trace
    plot_best_data(ax8, raw_data, samplerate, unit, idx0, idx1, clipped)
    ax8.format_coord = recording_format_coord
    
    ############
            
    # cosmetics
    for ax in [ax2, ax3, ax4, ax5, ax6, ax7, ax8]:
        axes_style(ax)

    # remove invisible axes:
    for ax in [ax2, ax3, ax4, ax5, ax6, ax7]:
        if not ax.get_visible():
            ax.set_position([0.99, 0.99, 0.01, 0.01])
            ax.set_navigate(False)
        
    return fig

                            
def plot_eod_subplots(base_name, subplots, raw_data, samplerate, idx0, idx1, clipped,
                      psd_data, wave_eodfs, wave_indices, mean_eods, eod_props, peak_data,
                      spec_data, unit, zoom_window, n_snippets=10, power_thresh=None,
                      label_power=True, log_freq=False, min_freq=0.0, max_freq=3000.0):
    """
    Plot time traces and spectra into separate files.

    Parameters
    ----------
    base_name: string
        Basename of audio_file.
    subplots: string
        Specifies which subplots to plot:
        r) recording with best window, t) data trace with detected pulse fish,
        p) power spectrum with detected wave fish, w/W) mean EOD waveform,
        s/S) EOD spectrum, e/E) EOD waveform and spectra. With capital letters
        all fish are saved into a single pdf filem with small letters each fish
        is saved into a separate file.
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
    psd_data: 2D array
        Power spectrum (frequencies and power) of the analysed data.
    wave_eodfs: array
        Frequency and power of fundamental frequency/harmonics of several fish.
    wave_indices: array of int
        Indices of wave fish mapping from wave_eodfs to eod_props.
        If negative, then that EOD frequency has no waveform described in eod_props.
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
    unit: string
        Unit of the trace and the mean EOD.
    n_snippets: int
        Number of EOD waveform snippets to be plotted. If zero do not plot any.
    power_thresh:  2 D array or None
        Frequency (first column) and power (second column) of threshold
        derived from single pulse spectra to discard false wave fish.
    label_power: boolean
        If `True` put the power in decibel in addition to the frequency
        into the legend.
    log_freq: boolean
        Logarithmic (True) or linear (False) frequency axis of power spectrum of recording.
    min_freq: float
        Limits of frequency axis of power spectrum of recording
        are set to `(min_freq, max_freq)` if `max_freq` is greater than zero
    max_freq: float
        Limits of frequency axis of power spectrum of recording
        are set to `(min_freq, max_freq)` and limits of power axis are computed
        from powers below max_freq if `max_freq` is greater than zero
    """
    plot_style()
    if 'r' in subplots:
        fig, ax = plt.subplots(figsize=(10, 2))
        fig.subplots_adjust(left=0.07, right=0.99, bottom=0.22, top=0.95)
        plot_best_data(ax, raw_data, samplerate, unit, idx0, idx1, clipped)
        ax.yaxis.set_major_locator(ticker.MaxNLocator(5))
        axes_style(ax)
        fig.savefig(base_name + '-recording.pdf')
    if 't' in subplots:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, 'not implemented yet',
                transform=axeod.transAxes, ha='center', va='center')
        axes_style(ax)
        fig.savefig(base_name + '-trace.pdf')
    if 'p' in subplots:
        fig, ax = plt.subplots(figsize=(10, 5))
        fig.subplots_adjust(left=0.08, right=0.975, bottom=0.11, top=0.9)
        axes_style(ax)
        if power_thresh is not None:
            ax.plot(power_thresh[:,0], decibel(power_thresh[:,1]), '#CCCCCC', lw=1)
        if len(wave_eodfs) > 0:
            kwargs = {}
            if len(wave_eodfs) > 1:
                title = '%d EOD frequencies' % len(wave_eodfs)
                kwargs = {'title': title if len(wave_eodfs) > 2 else None }
                if len(wave_eodfs) > 2:
                    fig.subplots_adjust(left=0.08, right=0.72, bottom=0.11, top=0.9)
                    kwargs.update({'bbox_to_anchor': (1.0, 1.1),
                                   'loc': 'upper left', 'legend_rows': 12})
                else:
                    kwargs.update({'bbox_to_anchor': (1.05, 1.1),
                                   'loc': 'upper right', 'legend_rows': 10})
            wave_colors, wave_markers = colors_markers()
            plot_harmonic_groups(ax, wave_eodfs, wave_indices, max_groups=0,
                                 sort_by_freq=True, label_power=label_power,
                                 colors=wave_colors, markers=wave_markers,
                                 frameon=False, **kwargs)
        plot_decibel_psd(ax, psd_data[:,0], psd_data[:,1], log_freq=log_freq,
                         min_freq=min_freq, max_freq=max_freq, ymarg=5.0, color='blue')
        ax.yaxis.set_major_locator(ticker.MaxNLocator(6))
        if len(wave_eodfs) == 1:
            ax.get_legend().set_visible(False)
            label = '%6.1f Hz' % wave_eodfs[0][0, 0]
            ax.set_title('Powerspectrum: %s' % label, y=1.05)
        else:
            ax.set_title('Powerspectrum', y=1.05)
        fig.savefig(base_name + '-psd.pdf')
    if 'w' in subplots or 'W' in subplots:
        mpdf = None
        if 'W' in subplots:
            mpdf = PdfPages(base_name + '-waveforms.pdf')
        for meod, props, peaks in zip(mean_eods, eod_props, peak_data):
            fig, ax = plt.subplots(figsize=(5, 3))
            fig.subplots_adjust(left=0.18, right=0.98, bottom=0.15, top=0.9)
            ax.set_title('{index:d}: {EODf:.1f} Hz {type} fish'.format(**props))
            plot_eod_waveform(ax, meod, props, peaks, unit)
            data = raw_data[idx0:idx1] if idx1 > idx0 else raw_data
            if props['type'] == 'pulse':
                plot_eod_snippets(ax, data, samplerate, meod[0,0], meod[-1,0],
                                  props['peaktimes'], n_snippets)
            ax.yaxis.set_major_locator(ticker.MaxNLocator(6))
            axes_style(ax)
            if mpdf is None:
                fig.savefig(base_name + '-waveform-%d.pdf' % props['index'])
            else:
                mpdf.savefig(fig)
        if mpdf is not None:
            mpdf.close()
    if 's' in subplots or 'S' in subplots:
        mpdf = None
        if 'S' in subplots:
            mpdf = PdfPages(base_name + '-spectrum.pdf')
        for props, peaks, spec in zip(eod_props, peak_data, spec_data):
            if props['type'] == 'pulse':
                fig, ax = plt.subplots(figsize=(5, 3.5))
                fig.subplots_adjust(left=0.15, right=0.967, bottom=0.16, top=0.88)
                axes_style(ax)
                ax.set_title('{index:d}: {EODf:.1f} Hz {type} fish'.format(**props), y=1.07)
                plot_pulse_spectrum(ax, spec, props)
            else:
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(5, 3.5))
                fig.subplots_adjust(left=0.15, right=0.97, bottom=0.16, top=0.88, hspace=0.4)
                axes_style(ax1)
                axes_style(ax2)
                ax1.set_title('{index:d}: {EODf:.1f} Hz {type} fish'.format(**props), y=1.15)
                plot_wave_spectrum(ax1, ax2, spec, props, unit)
                ax1.set_xticklabels([])
                ax1.yaxis.set_major_locator(ticker.MaxNLocator(4))
            if mpdf is None:
                fig.savefig(base_name + '-spectrum-%d.pdf' % props['index'])
            else:
                mpdf.savefig(fig)
        if mpdf is not None:
            mpdf.close()
    if 'e' in subplots or 'E' in subplots:
        mpdf = None
        if 'E' in subplots:
            mpdf = PdfPages(base_name + '-eods.pdf')
        for meod, props, peaks, spec in zip(mean_eods, eod_props, peak_data, spec_data):
            fig = plt.figure(figsize=(10, 3.5))
            gs = gridspec.GridSpec(nrows=2, ncols=2, left=0.09, right=0.98,
                                   bottom=0.16, top=0.88, wspace=0.4, hspace=0.4)
            ax1 = fig.add_subplot(gs[:,0])
            ax1.set_title('{index:d}: {EODf:.1f} Hz {type} fish'.format(**props), y=1.07)
            plot_eod_waveform(ax1, meod, props, peaks, unit)
            data = raw_data[idx0:idx1] if idx1 > idx0 else raw_data
            if props['type'] == 'pulse':
                plot_eod_snippets(ax1, data, samplerate, meod[0,0], meod[-1,0],
                                  props['peaktimes'], n_snippets)
            ax1.yaxis.set_major_locator(ticker.MaxNLocator(6))
            axes_style(ax1)
            if props['type'] == 'pulse':
                ax2 = fig.add_subplot(gs[:,1])
                axes_style(ax2)
                plot_pulse_spectrum(ax2, spec, props)
                ax2.set_title('Single pulse spectrum', y=1.07)
            else:
                ax2 = fig.add_subplot(gs[0,1])
                ax3 = fig.add_subplot(gs[1,1])
                axes_style(ax2)
                axes_style(ax3)
                plot_wave_spectrum(ax2, ax3, spec, props, unit)
                ax2.set_title('Amplitude and phase spectrum', y=1.15)
                ax2.set_xticklabels([])
                ax2.yaxis.set_major_locator(ticker.MaxNLocator(4))
            if mpdf is None:
                fig.savefig(base_name + '-eod-%d.pdf' % props['index'])
            else:
                mpdf.savefig(fig)
        if mpdf is not None:
            mpdf.close()
    plt.close()


def thunderfish(filename, cfg, channel=0, log_freq=0.0, save_data=False,
                save_plot=False, multi_pdf=None, save_subplots='',
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
        raw_data, samplerate, unit = load_data(filename, channel,
                                               verbose=verbose)
    except IOError as e:
        return '%s: failed to open file: %s' % (filename, str(e))
    if len(raw_data) <= 1:
        return '%s: empty data file' % filename

    # best_window:
    data, idx0, idx1, clipped, min_clip, max_clip = find_best_window(raw_data, samplerate,
                                                                     cfg, show_bestwindow)
    if show_bestwindow:
        return None
    found_bestwindow = idx1 > 0
    if not found_bestwindow:
        print(filename + ': not enough data for requested best window length. You may want to adjust the bestWindowSize parameter in the configuration file.')

    # detect EODs in the data:
    psd_data, wave_eodfs, wave_indices, eod_props, \
    mean_eods, spec_data, peak_data, power_thresh, skip_reason, zoom_window = \
      detect_eods(data, samplerate, clipped, min_clip, max_clip, filename, verbose, cfg)
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
        min_freq = 0.0
        max_freq = 3000.0
        if log_freq > 0.0:
            min_freq = log_freq
            max_freq = min_freq*20
            if max_freq < 2000:
                max_freq = 2000
            log_freq = True
        else:
            log_freq = False
        n_snippets = 10
        fig = plot_eods(outfilename, raw_data, samplerate, idx0, idx1, clipped,
                        psd_data[0], wave_eodfs, wave_indices, mean_eods, eod_props,
                        peak_data, spec_data, None, unit, zoom_window, n_snippets,
                        power_thresh, True, log_freq, min_freq, max_freq,
                        interactive=not save_data, verbose=verbose)
        if save_plot:
            if multi_pdf is not None:
                multi_pdf.savefig(fig)
            else:
                # save figure as pdf:
                fig.savefig(output_basename + '.pdf')
                plt.close('all')
            if len(save_subplots) > 0:
                plot_eod_subplots(output_basename, save_subplots,
                                  raw_data, samplerate, idx0, idx1, clipped, psd_data[0],
                                  wave_eodfs, wave_indices, mean_eods, eod_props,
                                  peak_data, spec_data, unit, zoom_window, n_snippets,
                                  power_thresh, True, log_freq, min_freq, max_freq)
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
    try:
        msg = thunderfish(file, *pool_args)
        if msg:
            print(msg)
    except:
        print('THUNDEFISH ERROR!!!!!!!!')


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
                        help='save output plot of each recording as pdf file')
    parser.add_argument('-P', dest='save_subplots', default='', type=str, metavar='rtpwse',
                        help='save subplots as separate pdf files: r) recording with best window, t) data trace with detected pulse fish, p) power spectrum with detected wave fish, w/W) mean EOD waveform, s/S) EOD spectrum, e/E) EOD waveform and spectra. Capital letters produce a single multipage pdf containing plots of all detected fish')
    parser.add_argument('-m', dest='multi_pdf', default='', type=str, metavar='PDFFILE',
                        help='save all plots of all recordings in a multi pages pdf file. Disables parallel jobs.')
    parser.add_argument('-l', dest='log_freq', type=float, metavar='MINFREQ',
                        nargs='?', const=100.0, default=0.0,
                        help='logarithmic frequency axis in  power spectrum with optional minimum frequency (defaults to 100 Hz)')
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
    if len(args.save_subplots) > 0:
        args.save_plot = True
    multi_pdf = None
    if len(args.multi_pdf) > 0:
        args.save_plot = True
        args.jobs = None
        ext = os.path.splitext(args.multi_pdf)[1]
        if ext != os.extsep + 'pdf':
            args.multi_pdf += os.extsep + 'pdf'
        multi_pdf = PdfPages(args.multi_pdf)
    # create output folder:
    if args.save_data or args.save_plot:
        if not os.path.exists(args.outpath):
            if verbose > 1:
                print('mkdir %s' % args.outpath)
            os.makedirs(args.outpath)
    # run on pool:
    global pool_args
    pool_args = (cfg, args.channel, args.log_freq, args.save_data,
                 args.save_plot, multi_pdf, args.save_subplots,
                 args.outpath, args.keep_path, args.show_bestwindow, verbose-1)
    if args.jobs is not None and (args.save_data or args.save_plot) and len(args.file) > 1:
        cpus = cpu_count() if args.jobs == 0 else args.jobs
        if verbose > 1:
            print('run on %d cpus' % cpus)
        p = Pool(cpus)
        p.map(run_thunderfish, args.file)
    else:
        list(map(run_thunderfish, args.file))
    if multi_pdf is not None:
        multi_pdf.close()


if __name__ == '__main__':
    freeze_support()  # needed by multiprocessing for some weired windows stuff
    main()
