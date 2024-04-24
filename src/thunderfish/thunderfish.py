"""# thunderfish

Automatically detect and analyze all EOD waveforms in short recordings
and generated summary plots and data tables.

Run it from the thunderfish development directory as:
```sh
> python3 -m thunderfish.thunderfish audiofile.wav
```
Or install thunderfish
```sh
> sudo pip3 install .
```
Then you can run it directly from every directory:
```sh
> thunderfish audiofile.wav
```
"""

import sys
import os
import glob
import io
import zipfile
import argparse
import traceback
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as ticker
import matplotlib.lines as ml
from matplotlib.transforms import Bbox
from matplotlib.backends.backend_pdf import PdfPages
from multiprocessing import Pool, freeze_support, cpu_count
from audioio import play, fade, load_audio
from thunderlab.configfile import ConfigFile
from thunderlab.dataloader import load_data
from thunderlab.powerspectrum import decibel, plot_decibel_psd, multi_psd
from thunderlab.powerspectrum import add_multi_psd_config, multi_psd_args
from thunderlab.tabledata import TableData, add_write_table_config, write_table_args
from .version import __version__, __year__
from .bestwindow import add_clip_config, add_best_window_config
from .bestwindow import clip_args, best_window_args
from .bestwindow import analysis_window, plot_data_window
from .checkpulse import check_pulse, add_check_pulse_config, check_pulse_args
from .pulses import extract_pulsefish
from .harmonics import add_psd_peak_detection_config, add_harmonic_groups_config
from .harmonics import harmonic_groups, harmonic_groups_args, psd_peak_detection_args
from .harmonics import colors_markers, plot_harmonic_groups
from .consistentfishes import consistent_fishes
from .eodanalysis import eod_waveform, analyze_wave, analyze_pulse
from .eodanalysis import clipped_fraction
from .eodanalysis import plot_eod_recording, plot_pulse_eods
from .eodanalysis import plot_eod_waveform, plot_eod_snippets
from .eodanalysis import plot_pulse_spectrum, plot_wave_spectrum
from .eodanalysis import add_eod_analysis_config, eod_waveform_args
from .eodanalysis import analyze_wave_args, analyze_pulse_args
from .eodanalysis import add_species_config
from .eodanalysis import wave_quality, wave_quality_args, add_eod_quality_config
from .eodanalysis import pulse_quality, pulse_quality_args
from .eodanalysis import save_eod_waveform, save_wave_eodfs, save_wave_fish, save_pulse_fish
from .eodanalysis import save_wave_spectrum, save_pulse_spectrum, save_pulse_peaks, save_pulse_times
from .eodanalysis import load_eod_waveform, load_wave_eodfs, load_wave_fish, load_pulse_fish
from .eodanalysis import load_wave_spectrum, load_pulse_spectrum, load_pulse_peaks
from .eodanalysis import save_analysis, load_analysis, load_recording
from .eodanalysis import parse_filename, file_types
from .fakefish import normalize_wavefish, export_wavefish


def configuration():
    """Assemble configuration parameter for thunderfish.

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
    cfg.add('unwrapData', False, '', 'Unwrap clipped voltage traces.')
    add_best_window_config(cfg, win_size=8.0, w_cv_ampl=10.0)
    add_check_pulse_config(cfg)
    add_eod_analysis_config(cfg, min_pulse_win=0.004)
    del cfg['eodSnippetFac']
    del cfg['eodMinSnippet']
    del cfg['eodMinSem']
    add_eod_quality_config(cfg)
    add_species_config(cfg)
    add_write_table_config(cfg, table_format='csv', unit_style='row',
                           align_columns=True, shrink_width=False)
    return cfg


def save_configuration(cfg, config_file):
    """Save configuration parameter for thunderfish to a file.

    Parameters
    ----------
    cfg: ConfigFile
        Configuration parameters and their values.
    config_file: string
        Name of the configuration file to be loaded.
    """
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


def detect_eods(data, samplerate, min_clip, max_clip, name, mode,
                verbose, plot_level, cfg):
    """Detect EODs of all fish present in the data.

    Parameters
    ----------
    data: array of floats
        The recording in which to detect EODs.
    samplerate: float
        Sampling rate of the dataset.
    min_clip: float
        Minimum amplitude that is not clipped.
    max_clip: float
        Maximum amplitude that is not clipped.
    name: string
        Name of the recording (e.g. its filename).
    mode: string
        Characters in the string indicate what and how to analyze:
        - 'w': analyze wavefish
        - 'p': analyze pulsefish
        - 'P': analyze only the pulsefish with the largest amplitude (not implemented yet) 
    verbose: int
        Print out information about EOD detection if greater than zero.
    plot_level : int
        Similar to verbosity levels, but with plots. 
    cfg: ConfigFile
        Configuration parameters.

    Returns
    -------
    psd_data: list of 2D arrays
        List of power spectra (frequencies and power) of the analysed data
        for different frequency resolutions.
    wave_eodfs: list of 2D arrays
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
    dfreq = np.nan
    nfft = 0
    psd_data = [[]]
    wave_eodfs = []
    wave_indices = []
    if 'w' in mode:
        # detect wave fish:
        psd_data = multi_psd(data, samplerate, **multi_psd_args(cfg))
        dfreq = np.mean(np.diff(psd_data[0][:,0]))
        nfft = int(samplerate/dfreq)
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
                print('found %2d EOD frequencies consistent in all power spectra:' % len(wave_eodfs))
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
    max_pulse_amplitude = 0.0
    zoom_window = []

    if 'p' in mode:
        # detect pulse fish:
        _, eod_times, eod_peaktimes, zoom_window, _ = extract_pulsefish(data, samplerate, verbose=verbose-1, plot_level=plot_level, save_path=os.path.splitext(os.path.basename(name))[0])

        #eod_times = []
        #eod_peaktimes = []
        if verbose > 0:
            if len(eod_times) > 0:
                print('found %2d pulsefish EODs' % len(eod_times))
            else:
                print('no pulsefish EODs found')

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
            clipped_frac = clipped_fraction(data, samplerate, eod_times0,
                                            mean_eod, min_clip, max_clip)
            props['peaktimes'] = eod_pts  # XXX that should go into analyze pulse
            props['index'] = len(eod_props)
            props['clipped'] = clipped_frac
            props['samplerate'] = samplerate
            props['nfft'] = nfft
            props['dfreq'] = dfreq

            # add good waveforms only:
            skips, msg, skipped_clipped = pulse_quality(props, **pulse_quality_args(cfg))

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
                if max_pulse_amplitude < props['p-p-amplitude']:
                    max_pulse_amplitude = props['p-p-amplitude']
                i0 = np.argmin(np.abs(mean_eod[:,0]))
                i1 = len(mean_eod) - i0
                pulse_data = np.zeros(len(data))
                for t in props['peaktimes']:
                    idx = int(t*samplerate)
                    ii0 = i0 if idx-i0 >= 0 else idx
                    ii1 = i1 if idx+i1 < len(pulse_data) else len(pulse_data)-1-idx
                    pulse_data[idx-ii0:idx+ii1] = mean_eod[i0-ii0:i0+ii1,1]
                pulse_psd = multi_psd(pulse_data, samplerate, **multi_psd_args(cfg))
                pulse_power = pulse_psd[0][:,1]
                pulse_power *= len(data)/samplerate/props['period']/len(props['peaktimes'])
                pulse_power *= 5.0
                if power_thresh is None:
                    power_thresh = pulse_psd[0]
                    power_thresh[:,1] = pulse_power
                else:
                    power_thresh[:,1] += pulse_power

        # remove wavefish below pulse fish power:
        if 'w' in mode and power_thresh is not None:
            n = len(wave_eodfs)
            maxh = 3  # XXX make parameter
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

    if 'w' in mode:
        # analyse EOD waveform of all wavefish:
        powers = np.array([np.sum(fish[:,1]) for fish in wave_eodfs])
        power_indices = np.argsort(-powers)
        wave_indices = np.zeros(len(wave_eodfs), dtype=int) - 3
        for k, idx in enumerate(power_indices):
            fish = wave_eodfs[idx]
            eod_times = np.arange(0.0, len(data)/samplerate, 1.0/fish[0,0])
            mean_eod, eod_times = \
                eod_waveform(data, samplerate, eod_times, win_fac=3.0, min_win=0.0,
                             min_sem=(k==0), **eod_waveform_args(cfg))
            mean_eod, props, sdata, error_str = \
                analyze_wave(mean_eod, fish, **analyze_wave_args(cfg))
            if error_str:
                print(name + ': ' + error_str)
            clipped_frac = clipped_fraction(data, samplerate, eod_times,
                                            mean_eod, min_clip, max_clip)
            props['n'] = len(eod_times)
            props['index'] = len(eod_props)
            props['clipped'] = clipped_frac
            props['samplerate'] = samplerate
            props['nfft'] = nfft
            props['dfreq'] = dfreq
            # remove wave fish that are smaller than the largest pulse fish:
            if props['p-p-amplitude'] < 0.01*max_pulse_amplitude:
                rm_indices = power_indices[k:]
                if verbose > 0:
                    print('skip %6.1fHz wave  fish: power=%5.1fdB, p-p amplitude=%5.1fdB smaller than pulse fish=%5.1dB - 20dB' %
                          (props['EODf'], decibel(powers[idx]),
                           decibel(props['p-p-amplitude']), decibel(max_pulse_amplitude)))
                    for idx in rm_indices[1:]:
                        print('skip %6.1fHz wave  fish: power=%5.1fdB even smaller' %
                              (wave_eodfs[idx][0,0], decibel(powers[idx])))
                wave_eodfs = [eodfs for idx, eodfs in enumerate(wave_eodfs)
                              if idx not in rm_indices]
                wave_indices = np.array([idcs for idx, idcs in enumerate(wave_indices)
                                        if idx not in rm_indices], dtype=int)
                break
            # add good waveforms only:
            remove, skips, msg = wave_quality(props, sdata[1:,3], **wave_quality_args(cfg))
            if len(skips) == 0:
                wave_indices[idx] = props['index']
                eod_props.append(props)
                mean_eods.append(mean_eod)
                spec_data.append(sdata)
                peak_data.append([])
                if verbose > 0:
                    print('take   %6.1fHz wave  fish: %s' % (props['EODf'], msg))
            else:
                wave_indices[idx] = -2 if remove else -1
                skip_reason += ['%.1fHz wave fish %s' % (props['EODf'], skips)]
                if verbose > 0:
                    print('%-6s %6.1fHz wave  fish: %s (%s)' %
                          ('remove' if remove else 'skip', props['EODf'], skips, msg))
        wave_eodfs = [eodfs for idx, eodfs in zip(wave_indices, wave_eodfs) if idx > -2]
        wave_indices = np.array([idx for idx in wave_indices if idx > -2], dtype=int)
    return (psd_data, wave_eodfs, wave_indices, eod_props, mean_eods,
            spec_data, peak_data, power_thresh, skip_reason, zoom_window)


def remove_eod_files(output_basename, verbose, cfg):
    """Remove all files from previous runs of thunderfish
    """
    ff = cfg.value('fileFormat')
    if ff == 'py':
        fext = 'py'
    else:
        fext = TableData.extensions[ff]
    # remove all files from previous runs of thunderfish:
    for fn in glob.glob('%s*.%s' % (output_basename, fext)):
        os.remove(fn)
        if verbose > 0:
            print('removed file %s' % fn)


def plot_style():
    """Set style of plots.
    """
    plt.rcParams['figure.facecolor'] = 'white'
    plt.rcParams['axes.facecolor'] = 'none'
    plt.rcParams['xtick.direction'] = 'out'
    plt.rcParams['ytick.direction'] = 'out'


def axes_style(ax):
    """Fix style of axes.

    Parameters
    ----------
    ax: matplotlib axes
    """
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()

                                
def plot_eods(base_name, message_filename,
              raw_data, samplerate, channel, idx0, idx1, clipped,
              psd_data, wave_eodfs, wave_indices, mean_eods, eod_props,
              peak_data, spec_data, indices, unit, zoom_window,
              n_snippets=10, power_thresh=None, label_power=True,
              all_eods=False, spec_plots='auto', skip_bad=True,
              log_freq=False, min_freq=0.0, max_freq=3000.0,
              interactive=True, verbose=0):
    """Creates an output plot for the thunderfish program.

    This output contains the raw trace where the analysis window is
    marked, the power-spectrum of this analysis window where the
    detected fish are marked, plots of averaged EOD plots, and
    spectra of the EOD waveforms.

    Parameters
    ----------
    base_name: string
        Basename of audio_file.
    message_filename: string or None
        Path to meta-data message.
    raw_data: array
        Dataset.
    samplerate: float
        Sampling rate of the dataset.
    channel: int or None
        Channel of the recording to be put into the plot title.
        If None, do not write the channel into the title.
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
    all_eods: bool
        Plot all EOD waveforms.
    spec_plots: bool or 'auto'
        Plot amplitude spectra of EOD waveforms.
        If 'auto', plot them if there is a singel waveform only.
    skip_bad: bool
        Skip harmonic groups without index (entry in indices is negative).
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
        if event.key in 'mM' and message_filename:
            # play voice message:
            msg, msg_rate = load_audio(message_filename)
            play(msg, msg_rate, blocking=False)

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

    # count number of fish types to be plotted:
    if indices is None:
        indices = np.arange(len(eod_props))
    else:
        indices = np.array(indices, dtype=int)
    nwave = 0
    npulse = 0
    for idx in indices:
        if eod_props[idx]['type'] == 'pulse':
            npulse += 1
        elif eod_props[idx]['type'] == 'wave':
            nwave += 1
    neods = nwave + npulse

    if verbose > 0:
        print('plot: %2d waveforms: %2d wave fish, %2d pulse fish and %2d EOD frequencies.'
              % (len(indices), nwave, npulse, len(wave_eodfs)))

    # size and positions:
    if spec_plots == 'auto':
        spec_plots = len(indices) == 1
    large_plots = spec_plots or len(indices) <= 2
    width = 14.0
    height = 10.0
    if all_eods and len(indices) > 0:
        nrows = len(indices) if spec_plots else (len(indices)+1)//2
        if large_plots:
            height = 6.0 + 4.0*nrows
        else:
            height = 6.4 + 1.9*nrows
    leftx = 1.0/width
    midx = 0.5 + leftx
    fullwidth = 1.0-1.4/width
    halfwidth = 0.5-1.4/width
    pheight = 3.0/height
    
    # figure:
    plot_style()
    fig = plt.figure(figsize=(width, height))
    if interactive:
        fig.canvas.mpl_connect('key_press_event', keypress)
    
    # plot title:
    title = base_name
    if channel is not None:
        title += ' c%d' % channel
    ax = fig.add_axes([0.2/width, 1.0-0.6/height, 1.0-0.4/width, 0.55/height])
    ax.text(0.0, 1.0, title, fontsize=22, va='top')
    ax.text(1.0, 1.0, 'thunderfish by Benda-Lab', fontsize=16, ha='right', va='top')
    ax.text(1.0, 0.0, 'version %s' % __version__, fontsize=16, ha='right', va='bottom')
    ax.set_frame_on(False)
    ax.set_axis_off()
    ax.set_navigate(False)

    # layout of recording and psd plots:
    #force_both = True  # set to True for debugging pulse and wave detection
    force_both = False
    posy = 1.0 - 4.0/height
    axr = None
    axp = None
    legend_inside = True
    legendwidth = 2.2/width if label_power else 1.7/width
    if neods == 0:
        axr = fig.add_axes([leftx, posy, fullwidth, pheight])                    # top, wide
        if len(psd_data) > 0:
            axp = fig.add_axes([leftx, 2.0/height, fullwidth, pheight])              # bottom, wide
    else:
        if npulse == 0 and nwave > 2 and psd_data is not None and \
           len(psd_data) > 0 and not force_both:
            axp = fig.add_axes([leftx, posy, fullwidth-legendwidth, pheight])    # top, wide
            legend_inside = False
        elif (npulse > 0 or psd_data is None or len(psd_data) == 0) \
             and len(wave_eodfs) == 0 and not force_both:
            axr = fig.add_axes([leftx, posy, fullwidth, pheight])                # top, wide
        else:
            axr = fig.add_axes([leftx, posy, halfwidth, pheight])                # top left
            label_power = False
            legendwidth = 2.2/width
            axp = fig.add_axes([midx, posy, halfwidth, pheight])                 # top, right
        
    # best window data:
    data = raw_data[idx0:idx1] if idx1 > idx0 else raw_data

    # plot recording
    pulse_colors, pulse_markers = colors_markers()
    pulse_colors = pulse_colors[3:]
    pulse_markers = pulse_markers[3:]
    if axr is not None:
        axes_style(axr)
        twidth = 0.1
        if len(indices) > 0:
            if eod_props[indices[0]]['type'] == 'wave':
                twidth = 5.0/eod_props[indices[0]]['EODf']
            else:
                if len(wave_eodfs) > 0:
                    twidth = 3.0/eod_props[indices[0]]['EODf']
                else:
                    twidth = 10.0/eod_props[indices[0]]['EODf']
            twidth = (1+twidth//0.005)*0.005
        if data is not None and len(data) > 0:
            plot_eod_recording(axr, data, samplerate, unit, twidth,
                               idx0/samplerate)
            plot_pulse_eods(axr, data, samplerate,
                            zoom_window, twidth, eod_props,
                            idx0/samplerate, colors=pulse_colors,
                            markers=pulse_markers, frameon=True,
                            loc='upper right')
        if axr.get_legend() is not None:
            axr.get_legend().get_frame().set_color('white')
        axr.set_title('Recording', fontsize=14, y=1.05)
        axr.format_coord = recordingzoom_format_coord
    
    # plot psd
    wave_colors, wave_markers = colors_markers()
    if axp is not None:
        axes_style(axp)
        if power_thresh is not None:
            axp.plot(power_thresh[:,0], decibel(power_thresh[:,1]), '#CCCCCC', lw=1)
        if len(wave_eodfs) > 0:
            kwargs = {}
            if len(wave_eodfs) > 1:
                title = '%d EOD frequencies' % len(wave_eodfs)
                kwargs = {'title': title if len(wave_eodfs) > 2 else None }
                if legend_inside:
                    kwargs.update({'bbox_to_anchor': (1.05, 1.1),
                                   'loc': 'upper right', 'legend_rows': 10,
                                   'frameon': True})
                else:
                    kwargs.update({'bbox_to_anchor': (1.02, 1.1),
                                   'loc': 'upper left', 'legend_rows': 14,
                                   'labelspacing': 0.6, 'frameon': False})
            plot_harmonic_groups(axp, wave_eodfs, wave_indices, max_groups=0,
                                 skip_bad=skip_bad,
                                 sort_by_freq=True, label_power=label_power,
                                 colors=wave_colors, markers=wave_markers,
                                 **kwargs)
            if legend_inside:
                axp.get_legend().get_frame().set_color('white')
        if psd_data is not None and len(psd_data) > 0:
            plot_decibel_psd(axp, psd_data[:,0], psd_data[:,1], log_freq=log_freq,
                             min_freq=min_freq, max_freq=max_freq, ymarg=5.0, color='blue')
        axp.yaxis.set_major_locator(ticker.MaxNLocator(6))
        if len(wave_eodfs) == 1:
            axp.get_legend().set_visible(False)
            label = '%6.1f Hz' % wave_eodfs[0][0, 0]
            axp.set_title('Powerspectrum: %s' % label, y=1.05, fontsize=14)
        else:
            axp.set_title('Powerspectrum', y=1.05, fontsize=14)
        axp.format_coord = psd_format_coord

    # get fish labels from legends:
    if axp is not None:
        w, _ = axp.get_legend_handles_labels()
        eodf_labels = [wi.get_label().split()[0] for wi in w]
        legend_wave_eodfs = np.array([float(f) if f[0] != '(' else np.nan for f in eodf_labels])
    if axr is not None:
        p, _ = axr.get_legend_handles_labels()
        eodf_labels = [pi.get_label().split()[0] for pi in p]
        legend_pulse_eodfs = np.array([float(f) if f[0] != '(' else np.nan for f in eodf_labels])

    # layout:
    sheight = 1.4/height
    sstep = 1.6/height
    max_plots = len(indices)
    if not all_eods:
        if large_plots:
            max_plots = 1 if spec_plots else 2
        else:
            max_plots = 4
    if large_plots:
        pstep = pheight + 1.0/height
        ty = 1.08
        my = 1.10
        ny = 6
    else:
        posy -= 0.2/height
        pheight = 1.3/height
        pstep = 1.9/height
        ty = 1.10
        my = 1.16
        ny = 4
    posy -= pstep
            
    # sort indices by p-p amplitude:
    pp_ampls = [eod_props[idx]['p-p-amplitude'] for idx in indices]
    pp_indices = np.argsort(pp_ampls)[::-1]
        
    # plot EOD waveform and spectra:
    for k, idx in enumerate(indices[pp_indices]):
        if k >= max_plots:
            break
        # plot EOD waveform:
        mean_eod = mean_eods[idx]
        props = eod_props[idx]
        peaks = peak_data[idx]
        lx = leftx if spec_plots or k%2 == 0 else midx
        ax = fig.add_axes([lx, posy, halfwidth, pheight])
        axes_style(ax)
        ax.yaxis.set_major_locator(ticker.MaxNLocator(ny))
        if len(indices) > 1:
            ax.text(0.3, ty, '{EODf:.1f} Hz {type} fish'.format(**props),
                       transform=ax.transAxes, fontsize=14, zorder=20)
            mx = 0.25
        else:
            ax.text(-0.1, ty, '{EODf:.1f} Hz {type} fish'.format(**props),
                       transform=ax.transAxes, fontsize=14, zorder=20)
            ax.text(0.5, ty, 'Averaged EOD', ha='center',
                       transform=ax.transAxes, fontsize=14, zorder=20)
            mx = -0.14
        eodf = props['EODf']
        if props['type'] == 'wave':
            if axp is not None:
                wk = np.nanargmin(np.abs(legend_wave_eodfs - eodf))
                ma = ml.Line2D([mx], [my], color=w[wk].get_color(), marker=w[wk].get_marker(),
                               markersize=w[wk].get_markersize(), mec='none', clip_on=False,
                               label=w[wk].get_label(), transform=ax.transAxes)
                ax.add_line(ma)
        else:
            if axr is not None and len(legend_pulse_eodfs) > 0:
                pk = np.argmin(np.abs(legend_pulse_eodfs - eodf))
                ma = ml.Line2D([mx], [my], color=p[pk].get_color(), marker=p[pk].get_marker(),
                               markersize=p[pk].get_markersize(), mec='none', clip_on=False,
                               label=p[pk].get_label(), transform=ax.transAxes)
                ax.add_line(ma)
        plot_eod_waveform(ax, mean_eod, props, peaks, unit)
        if props['type'] == 'pulse' and 'times' in props:
            plot_eod_snippets(ax, data, samplerate, mean_eod[0,0], mean_eod[-1,0],
                              props['times'], n_snippets, props['flipped'])
        if not large_plots and k < max_plots-2:
            ax.set_xlabel('')
        ax.format_coord = meaneod_format_coord

        # plot spectra:
        if spec_plots:
            spec = spec_data[idx]
            if  props['type'] == 'pulse':
                ax = fig.add_axes([midx, posy, halfwidth, pheight])
                axes_style(ax)
                plot_pulse_spectrum(ax, spec, props)
                ax.set_title('Single pulse spectrum', fontsize=14, y=1.05)
                ax.format_coord = pulsepsd_format_coord
            else:
                axa = fig.add_axes([midx, posy+sstep, halfwidth, sheight])
                axes_style(axa)
                axp = fig.add_axes([midx, posy, halfwidth, sheight])
                axes_style(axp)
                plot_wave_spectrum(axa, axp, spec, props, unit)
                axa.set_title('Amplitude and phase spectrum', fontsize=14, y=1.05)
                axa.set_xticklabels([])
                axa.yaxis.set_major_locator(ticker.MaxNLocator(4))
                axa.format_coord = ampl_format_coord
                axp.format_coord = phase_format_coord

        if spec_plots or k%2 == 1:
            posy -= pstep

    # whole trace:
    ax = fig.add_axes([leftx, 0.6/height, fullwidth, 0.9/height])
    axes_style(ax)
    if raw_data is not None and len(raw_data) > 0:
        plot_data_window(ax, raw_data, samplerate, unit, idx0, idx1, clipped)
    ax.format_coord = recording_format_coord

    return fig

                            
def plot_eod_subplots(base_name, subplots, raw_data, samplerate, idx0, idx1,
                      clipped, psd_data, wave_eodfs, wave_indices, mean_eods,
                      eod_props, peak_data, spec_data, unit, zoom_window,
                      n_snippets=10, power_thresh=None, label_power=True,
                      skip_bad=True, log_freq=False,
                      min_freq=0.0, max_freq=3000.0, save=True):
    """Plot time traces and spectra into separate windows or files.

    Parameters
    ----------
    base_name: string
        Basename of audio_file.
    subplots: string
        Specifies which subplots to plot:
        r) recording with best window, t) data trace with detected pulse fish,
        p) power spectrum with detected wave fish, w/W) mean EOD waveform,
        s/S) EOD spectrum, e/E) EOD waveform and spectra. With capital letters
        all fish are saved into a single pdf file, with small letters each fish
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
    skip_bad: bool
        Skip harmonic groups without index (entry in indices is negative).
    log_freq: boolean
        Logarithmic (True) or linear (False) frequency axis of power spectrum of recording.
    min_freq: float
        Limits of frequency axis of power spectrum of recording
        are set to `(min_freq, max_freq)` if `max_freq` is greater than zero
    max_freq: float
        Limits of frequency axis of power spectrum of recording
        are set to `(min_freq, max_freq)` and limits of power axis are computed
        from powers below max_freq if `max_freq` is greater than zero
    save: bool
        If True save plots to files instead of showing them.
    """
    plot_style()
    if 'r' in subplots:
        fig, ax = plt.subplots(figsize=(10, 2))
        fig.subplots_adjust(left=0.07, right=0.99, bottom=0.22, top=0.95)
        plot_data_window(ax, raw_data, samplerate, unit, idx0, idx1, clipped)
        ax.yaxis.set_major_locator(ticker.MaxNLocator(5))
        axes_style(ax)
        if save:
            fig.savefig(base_name + '-recording.pdf')
    if 't' in subplots:
        fig, ax = plt.subplots(figsize=(10, 6))
        twidth = 0.1
        if len(eod_props) > 0:
            if eod_props[0]['type'] == 'wave':
                twidth = 5.0/eod_props[0]['EODf']
            else:
                if len(wave_eodfs) > 0:
                    twidth = 3.0/eod_props[0]['EODf']
                else:
                    twidth = 10.0/eod_props[0]['EODf']
        twidth = (1+twidth//0.005)*0.005
        pulse_colors, pulse_markers = colors_markers()
        pulse_colors = pulse_colors[3:]
        pulse_markers = pulse_markers[3:]
        plot_eod_recording(ax, raw_data[idx0:idx1], samplerate, unit,
                           twidth, idx0/samplerate)
        plot_pulse_eods(ax, raw_data[idx0:idx1], samplerate,
                        zoom_window, twidth, eod_props,
                        idx0/samplerate, colors=pulse_colors,
                        markers=pulse_markers, frameon=True,
                        loc='upper right')
        if ax.get_legend() is not None:
            ax.get_legend().get_frame().set_color('white')
        axes_style(ax)
        if save:
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
                    fig.subplots_adjust(left=0.08, right=0.78, bottom=0.11, top=0.9)
                    kwargs.update({'bbox_to_anchor': (1.01, 1.1),
                                   'loc': 'upper left', 'legend_rows': 14,
                                   'labelspacing': 0.6})
                else:
                    kwargs.update({'bbox_to_anchor': (1.05, 1.1),
                                   'loc': 'upper right', 'legend_rows': 10})
            wave_colors, wave_markers = colors_markers()
            plot_harmonic_groups(ax, wave_eodfs, wave_indices, max_groups=0,
                                 skip_bad=skip_bad,
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
        if save:
            fig.savefig(base_name + '-psd.pdf')
    if 'w' in subplots or 'W' in subplots:
        mpdf = None
        if 'W' in subplots:
            mpdf = PdfPages(base_name + '-waveforms.pdf')
        for meod, props, peaks in zip(mean_eods, eod_props, peak_data):
            if meod is None:
                continue
            fig, ax = plt.subplots(figsize=(5, 3))
            fig.subplots_adjust(left=0.18, right=0.98, bottom=0.15, top=0.9)
            if not props is None:
                ax.set_title('{index:d}: {EODf:.1f} Hz {type} fish'.format(**props))
            plot_eod_waveform(ax, meod, props, peaks, unit)
            data = raw_data[idx0:idx1] if idx1 > idx0 else raw_data
            if not props is None and props['type'] == 'pulse' and \
               'times' in props:
                plot_eod_snippets(ax, data, samplerate, meod[0,0], meod[-1,0],
                                  props['times'], n_snippets)
            ax.yaxis.set_major_locator(ticker.MaxNLocator(6))
            axes_style(ax)
            if mpdf is None:
                if save:
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
            if spec is None:
                continue
            if props is not None and props['type'] == 'pulse':
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
                if not props is None:
                    ax1.set_title('{index:d}: {EODf:.1f} Hz {type} fish'.format(**props), y=1.15)
                plot_wave_spectrum(ax1, ax2, spec, props, unit)
                ax1.set_xticklabels([])
                ax1.yaxis.set_major_locator(ticker.MaxNLocator(4))
            if mpdf is None:
                if save:
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
            if meod is None or spec is None:
                continue
            fig = plt.figure(figsize=(10, 3.5))
            gs = gridspec.GridSpec(nrows=2, ncols=2, left=0.09, right=0.98,
                                   bottom=0.16, top=0.88, wspace=0.4, hspace=0.4)
            ax1 = fig.add_subplot(gs[:,0])
            if not props is None:
                ax1.set_title('{index:d}: {EODf:.1f} Hz {type} fish'.format(**props), y=1.07)
            plot_eod_waveform(ax1, meod, props, peaks, unit)
            data = raw_data[idx0:idx1] if idx1 > idx0 else raw_data
            if not props is None and props['type'] == 'pulse' and 'times' in props:
                plot_eod_snippets(ax1, data, samplerate, meod[0,0], meod[-1,0],
                                  props['times'], n_snippets)
            ax1.yaxis.set_major_locator(ticker.MaxNLocator(6))
            axes_style(ax1)
            if not props is None and props['type'] == 'pulse':
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
                if save:
                    fig.savefig(base_name + '-eod-%d.pdf' % props['index'])
            else:
                mpdf.savefig(fig)
        if mpdf is not None:
            mpdf.close()
    if not save:
        plt.show()
    plt.close('all')


def thunderfish_plot(files, data_path=None, load_kwargs={},
                     all_eods=False, spec_plots='auto', skip_bad=True,
                     save_plot=False, multi_pdf=None,
                     save_subplots='', log_freq=False, min_freq=0.0,
                     max_freq=3000.0, output_folder='.',
                     keep_path=False, verbose=0):
    """Generate plots from saved analysis results.

    Parameters
    ----------
    files: list of str
        Analysis files from a single recording.
    data_path: str
        Path where to find the raw data.
    load_kwargs: dict
        Key-word arguments for the `load_data()` function.
    all_eods: bool
        If True, plot all EOD waveforms.
    spec_plots: bool or 'auto'
        Plot amplitude spectra of EOD waveforms.
        If 'auto', plot them if there is a singel waveform only.
    skip_bad: bool
        Skip harmonic groups without index in the spectrum plot.
    save_plot: bool
        If True, save plots as pdf file.
    multi_pdf: matplotlib.PdfPages or None
        PdfPages instance in which to save plots.
    save_subplots: string
        If not empty, specifies subplots to be saved as separate pdf
        files: r) recording with best window, t) data trace with
        detected pulse fish, p) power spectrum with detected wave
        fish, w/W) mean EOD waveform, s/S) EOD spectrum, e/E) EOD
        waveform and spectra. Capital letters produce a single
        multipage pdf containing plots of all detected fish.
    log_freq: boolean
        Logarithmic (True) or linear (False) frequency axis of
        power spectrum of recording.
    min_freq: float
        Limits of frequency axis of power spectrum of recording
        are set to `(min_freq, max_freq)`, if `max_freq` is greater than zero
    max_freq: float
        Limits of frequency axis of power spectrum of recording
        are set to `(min_freq, max_freq)` and limits of power axis are computed
        from powers below max_freq, if `max_freq` is greater than zero
    output_folder: string
        Folder where to save results.
    keep_path: bool
        Add relative path of data files to output path.
    verbose: int
       Verbosity level (for debugging).
    """
    if len(save_subplots) == 0:
        save_subplots = 'rtpwsed'  # plot everything
    # load analysis results:
    mean_eods, wave_eodfs, wave_indices, eod_props, spec_data, \
        peak_data, base_name, channel, unit = load_analysis(files)
    if len(mean_eods) == 0 or all(me is None for me in mean_eods):
        save_subplots = save_subplots.replace('w', '')
        save_subplots = save_subplots.replace('W', '')
        save_subplots = save_subplots.replace('e', '')
        save_subplots = save_subplots.replace('E', '')
        save_subplots = save_subplots.replace('d', '')
    if len(spec_data) == 0 or all(sd is None for sd in spec_data):
        save_subplots = save_subplots.replace('s', '')
        save_subplots = save_subplots.replace('S', '')
        save_subplots = save_subplots.replace('e', '')
        save_subplots = save_subplots.replace('E', '')
        save_subplots = save_subplots.replace('d', '')
    clipped = 0.0
    if len(eod_props) > 0 and not eod_props[0] is None and \
       'winclipped' in eod_props[0]:
        clipped = eod_props[0]['winclipped']
    zoom_window = [1.2, 1.3]
    # load recording:
    psd_data = None
    if base_name:
        name = os.path.basename(base_name) if data_path and data_path != '.' else base_name
        data_path = os.path.join(data_path, name)
    data, samplerate, idx0, idx1, data_path = \
        load_recording(data_path, channel, load_kwargs,
                       eod_props, verbose-1)
    if data is None:
        save_subplots = save_subplots.replace('r', '')
        save_subplots = save_subplots.replace('t', '')
        save_subplots = save_subplots.replace('d', '')
    if verbose > 0:
        print('loaded', data_path)
    if len(eod_props) > 0 and not eod_props[0] is None and \
       'dfreq' in eod_props[0] and data is not None and len(data) > 0:
        psd_data = multi_psd(data[idx0:idx1],
                             samplerate,
                             1.1*eod_props[0]['dfreq'])[0]
    if psd_data is not None and len(psd_data) > 0:
        for idx, fish in zip(wave_indices, wave_eodfs):
            if idx < 0:
                for k in range(len(fish)):
                    fish[k,1] = psd_data[np.argmin(np.abs(psd_data[:,0] - fish[k,0])),1]
    if psd_data is None:
        save_subplots = save_subplots.replace('p', '')
        save_subplots = save_subplots.replace('d', '')
    # file name for output files:
    fn = base_name if keep_path else os.path.basename(base_name)
    output_basename = os.path.join(output_folder, fn)
    if channel >= 0:
        output_basename += f'-c{channel}'
    # make directory if necessary:
    if keep_path:
        outpath = os.path.dirname(output_basename)
        if not os.path.exists(outpath):
            if verbose > 0:
                print('mkdir %s' % outpath)
            os.makedirs(outpath)
    # plot:
    if len(save_subplots) == 0 or 'd' in save_subplots:
        fig = plot_eods(os.path.basename(base_name), None,
                        data, samplerate,
                        channel, idx0, idx1, clipped, psd_data,
                        wave_eodfs, wave_indices, mean_eods,
                        eod_props, peak_data, spec_data, None, unit,
                        zoom_window, 10, None, True, all_eods,
                        spec_plots, skip_bad, log_freq, min_freq,
                        max_freq, interactive=not save_plot,
                        verbose=verbose-1)
        if save_plot:
            if multi_pdf is not None:
                multi_pdf.savefig(fig)
            else:
                fig.savefig(output_basename + '.pdf')
        else:
            fig.canvas.manager.set_window_title('thunderfish')
            plt.show()
        plt.close()
        save_subplots = save_subplots.replace('d', '')
    if len(save_subplots) > 0:
        plot_eod_subplots(output_basename, save_subplots, data,
                          samplerate, idx0, idx1, clipped,
                          psd_data, wave_eodfs, wave_indices,
                          mean_eods, eod_props, peak_data,
                          spec_data, unit, zoom_window, 10, None,
                          True, skip_bad, log_freq, min_freq,
                          max_freq, save_plot)
    return None

                
def thunderfish(filename, load_kwargs, cfg, channel=0,
                time=None, time_file=False,
                mode='wp', log_freq=False, min_freq=0.0, max_freq=3000,
                save_data=False, zip_file=False,
                all_eods=False, spec_plots='auto', skip_bad=True,
                save_plot=False, multi_pdf=None, save_subplots='',
                output_folder='.', keep_path=False,
                verbose=0, plot_level=0):
    """Automatically detect and analyze all EOD waveforms in a short recording.

    Parameters
    ----------
    filename: string
        Path of the data file to be analyzed.
    load_kwargs: dict
        Key-word arguments for the `load_data()` function.
    cfg: dict
    channel: int
        Channel to be analyzed.
    time: string, float, or None
        Start time of analysis window: "beginning", "center", "end",
        "best", or time in seconds (as float or string). If not None
        overwrites "windowPosition" in cofiguration file.
    time_file: bool
        If `True` add time of analysis window to output file names.
    mode: 'w', 'p', 'P', 'wp', or 'wP'
        Analyze wavefish ('w'), all pulse fish ('p'), or largest pulse
        fish only ('P').
    log_freq: boolean
        Logarithmic (True) or linear (False) frequency axis of
        power spectrum of recording.
    min_freq: float
        Limits of frequency axis of power spectrum of recording
        are set to `(min_freq, max_freq)`, if `max_freq` is greater than zero
    max_freq: float
        Limits of frequency axis of power spectrum of recording
        are set to `(min_freq, max_freq)` and limits of power axis are computed
        from powers below max_freq, if `max_freq` is greater than zero
    save_data: bool
        If True save analysis results in files. If False, just plot the data.
    zip_data: bool
        If True, store all analysis results in a single zip file.
    all_eods: bool
        If True, plot all EOD waveforms.
    spec_plots: bool or 'auto'
        Plot amplitude spectra of EOD waveforms.
        If 'auto', plot them if there is a singel waveform only.
    skip_bad: bool
        Skip harmonic groups without index in the spectrum plot.
    save_plot: bool
        If True, save plots as pdf file.
    multi_pdf: matplotlib.PdfPages or None
        PdfPages instance in which to save plots.
    save_subplots: string
        If not empty, specifies subplots to be saved as separate pdf
        files: r) recording with best window, t) data trace with
        detected pulse fish, p) power spectrum with detected wave
        fish, w/W) mean EOD waveform, s/S) EOD spectrum, e/E) EOD
        waveform and spectra. Capital letters produce a single
        multipage pdf containing plots of all detected fish.
    output_folder: string
        Folder where to save results.
    keep_path: bool
        Add relative path of data files to output path.
    verbose: int
       Verbosity level (for debugging).
    plot_level: int
       Plot intermediate results.

    Returns
    -------
    msg: string or None
        In case of errors, an error message.
    """
    # check data file:
    if len(filename) == 0:
        return 'you need to specify a file containing some data'

    # file names:
    fn = filename if keep_path else os.path.basename(filename)
    outfilename = os.path.splitext(fn)[0]
    messagefilename = os.path.splitext(fn)[0] + '-message.wav'
    if not os.path.isfile(messagefilename):
        messagefilename = None

    # load data:
    try:
        all_data, samplerate, unit, ampl_max = load_data(filename,
                                                         verbose=verbose,
                                                         **load_kwargs)
    except IOError as e:
        return '%s: failed to open file: %s' % (filename, str(e))
    # select channel:
    channels = all_data.shape[1]
    chan_list = [channel]
    if channel < 0:
        chan_list = range(channels)
    elif channel >= channels:
        return '%s: invalid channel %d (%d channels)' % (filename, channel, channels)
    # process all channels:
    for chan in chan_list:
        raw_data = all_data[:,chan]
        if len(raw_data) <= 1:
            return '%s: empty data file' % filename
        if verbose >= 0 and len(chan_list) > 1:
            print('  channel %d' % chan)

        # analysis window:
        win_pos = cfg.value('windowPosition')
        if time is not None:
            win_pos = time
        data, idx0, idx1, clipped, min_clip, max_clip = \
            analysis_window(raw_data, samplerate, ampl_max, win_pos,
                            cfg, plot_level>0)
        found_bestwindow = idx1 > 0
        if not found_bestwindow:
            return '%s: not enough data for requested window length. You may want to adjust the windowSize parameter in the configuration file.' % filename

        # detect EODs in the data:
        psd_data, wave_eodfs, wave_indices, eod_props, \
        mean_eods, spec_data, peak_data, power_thresh, skip_reason, zoom_window = \
          detect_eods(data, samplerate, min_clip, max_clip, filename,
                      mode, verbose, plot_level, cfg)
        if not found_bestwindow:
            wave_eodfs = []
            wave_indices = []
            eod_props = []
            mean_eods = []

        # add analysis window to EOD properties:
        for props in eod_props:
            props['twin'] = idx0/samplerate
            props['window'] = (idx1 - idx0)/samplerate
            props['winclipped'] = clipped

        # warning message in case no fish has been found:
        if found_bestwindow and not eod_props :
            msg = ', '.join(skip_reason)
            if msg:
                print(filename + ': no fish found: %s' % msg)
            else:
                print(filename + ': no fish found.')

        # file name for output files:
        output_basename = os.path.join(output_folder, outfilename)
        if channels > 1:
            if channels > 100:
                output_basename += '-c%03d' % chan
            elif channels > 10:
                output_basename += '-c%02d' % chan
            else:
                output_basename += '-c%d' % chan
        if time_file:
            output_basename += '-t%.0fs' % (idx0/samplerate)
        # make directory if necessary:
        if keep_path and found_bestwindow:
            outpath = os.path.dirname(output_basename)
            if not os.path.exists(outpath):
                if verbose > 0:
                    print('mkdir %s' % outpath)
                os.makedirs(outpath)
        # save results to files:
        if save_data:
            remove_eod_files(output_basename, verbose, cfg)
            if found_bestwindow:
                save_analysis(output_basename, zip_file, eod_props,
                              mean_eods, spec_data, peak_data,
                              wave_eodfs, wave_indices, unit, verbose,
                              **write_table_args(cfg))
        # summary plots:
        if save_plot or not save_data:
            n_snippets = 10
            if len(save_subplots) == 0 or 'd' in save_subplots:
                chl = chan if channels > 1 else None
                fig = plot_eods(outfilename, messagefilename,
                                raw_data, samplerate,
                                chl, idx0, idx1, clipped, psd_data[0],
                                wave_eodfs, wave_indices, mean_eods,
                                eod_props, peak_data, spec_data, None,
                                unit, zoom_window, n_snippets,
                                power_thresh, True, all_eods,
                                spec_plots, skip_bad, log_freq,
                                min_freq, max_freq, interactive=not
                                save_plot, verbose=verbose)
                if save_plot:
                    if multi_pdf is not None:
                        multi_pdf.savefig(fig)
                    else:
                        fig.savefig(output_basename + '.pdf')
                else:
                    fig.canvas.manager.set_window_title('thunderfish')
                    plt.show()
                plt.close()
                save_subplots = save_subplots.replace('d', '')
            if len(save_subplots) > 0:
                plot_eod_subplots(output_basename, save_subplots,
                                  raw_data, samplerate, idx0, idx1,
                                  clipped, psd_data[0], wave_eodfs,
                                  wave_indices, mean_eods, eod_props,
                                  peak_data, spec_data, unit,
                                  zoom_window, n_snippets,
                                  power_thresh, True, skip_bad,
                                  log_freq, min_freq, max_freq,
                                  save_plot)
    return None


def run_thunderfish(file_args):
    """Helper function for mutlithreading Pool().map().
    """
    results = file_args[1][0]
    verbose = file_args[1][-1] if results else file_args[1][-2]+1
    if verbose > 1:
        print('='*70)
    try:
        if results:
            thunderfish_plot(file_args[0], *file_args[1][1:])
        else:
            if verbose > 0:
                print('analyze recording %s ...' % file_args[0])
            msg = thunderfish(file_args[0], *file_args[1][1:])
            if msg:
                print(msg)
    except (KeyboardInterrupt, SystemExit):
        print('\nthunderfish interrupted by user... exit now.')
        sys.exit(0)
    except:
        print(traceback.format_exc())


def main(cargs=None):
    # config file name:
    cfgfile = __package__ + '.cfg'

    # command line arguments:
    if cargs is None:
        cargs = sys.argv[1:]
    parser = argparse.ArgumentParser(add_help=False,
        description='Analyze EOD waveforms of weakly electric fish.',
        epilog='version %s by Benda-Lab (2015-%s)' % (__version__, __year__))
    parser.add_argument('-h', '--help', action='store_true',
                        help='show this help message and exit')
    parser.add_argument('--version', action='version', version=__version__)
    parser.add_argument('-v', action='count', dest='verbose', default=0,
                        help='verbosity level. Increase by specifying -v multiple times, or like -vvv')
    parser.add_argument('-V', action='count', dest='plot_level', default=0,
                        help='level for debugging plots. Increase by specifying -V multiple times, or like -VVV')
    parser.add_argument('-c', dest='save_config', action='store_true',
                        help='save configuration to file {0} after reading all configuration files'.format(cfgfile))
    parser.add_argument('--channel', default=0, type=int,
                        help='channel to be analyzed (defaults to first channel, negative channel selects all channels)')
    parser.add_argument('-t', dest='time', default=None, type=str, metavar='TIME',
                        help='start time of analysis window in recording: "beginning", "center", "end", "best", or time in seconds (overwrites "windowPosition" in cofiguration file)')
    parser.add_argument('-u', dest='unwrap', action='store_true',
                        help='unwrap clipped files, toggles unwrap setting of config file.')
    parser.add_argument('-T', dest='time_file', action='store_true',
                        help='add start time of analysis file to output file names')
    parser.add_argument('-m', dest='mode', default='wp', type=str,
                        choices=['w', 'p', 'wp'],
                        help='extract wave "w" and/or pulse "p" fish EODs')
    parser.add_argument('-a', dest='all_eods', action='store_true',
                        help='show all EOD waveforms in the summary plot')
    parser.add_argument('-S', dest='spec_plots', action='store_true',
                        help='plot spectra for all EOD waveforms in the summary plot')
    parser.add_argument('-b', dest='skip_bad', action='store_false',
                        help='indicate bad EODs in legend of power spectrum')
    parser.add_argument('-l', dest='log_freq', type=float, metavar='MINFREQ',
                        nargs='?', const=100.0, default=0.0,
                        help='logarithmic frequency axis in  power spectrum with optional minimum frequency (defaults to 100 Hz)')
    parser.add_argument('-p', dest='save_plot', action='store_true',
                        help='save output plots as pdf files')
    parser.add_argument('-M', dest='multi_pdf', default='', type=str, metavar='PDFFILE',
                        help='save all summary plots of all recordings in a multi page pdf file. Disables parallel jobs.')
    parser.add_argument('-P', dest='save_subplots', default='', type=str, metavar='rtpwsed',
                        help='save subplots as separate pdf files: r) recording with analysis window, t) data trace with detected pulse fish, p) power spectrum with detected wave fish, w/W) mean EOD waveform, s/S) EOD spectrum, e/E) EOD waveform and spectra, d) the default summary plot. Capital letters produce a single multipage pdf containing plots of all detected fish')
    parser.add_argument('-d', dest='rawdata_path', default='.', type=str, metavar='PATH',
                        help='path to raw EOD recordings needed for plotting based on analysis results')
    parser.add_argument('-j', dest='jobs', nargs='?', type=int, default=None, const=0,
                        help='number of jobs run in parallel. Without argument use all CPU cores.')
    parser.add_argument('-s', dest='save_data', action='store_true',
                        help='save analysis results to files')
    parser.add_argument('-z', dest='zip_file', action='store_true',
                        help='save analysis results in a single zip file')
    parser.add_argument('-f', dest='format', default='auto', type=str,
                        choices=TableData.formats + ['py'],
                        help='file format used for saving analysis results, defaults to the format specified in the configuration file or "csv"')
    parser.add_argument('-o', dest='outpath', default='.', type=str,
                        help='path where to store results and figures (defaults to current working directory)')
    parser.add_argument('-k', dest='keep_path', action='store_true',
                        help='keep path of input file when saving analysis files, i.e. append path of input file to OUTPATH')
    parser.add_argument('-i', dest='load_kwargs', default=[],
                        action='append', metavar='KWARGS',
                        help='key-word arguments for the data loader function')
    parser.add_argument('file', nargs='*', default='', type=str,
                        help='name of a file with time series data of an EOD recording, may include wildcards')
    args = parser.parse_args(cargs)

    # help:
    if args.help:
        parser.print_help()
        print('')
        print('examples:')
        print('- analyze the single file data.wav interactively:')
        print('  > thunderfish data.wav')
        print('- extract wavefish only:')
        print('  > thunderfish -m w data.wav')
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
    verbose = args.verbose
    plot_level = args.plot_level
    if verbose < plot_level:
        verbose = plot_level

    # interactive plot:
    plt.rcParams['keymap.quit'] = 'ctrl+w, alt+q, q'
    plt.ioff()

    # expand wildcard patterns:
    files = []
    if os.name == 'nt':
        for fn in args.file:
            files.extend(glob.glob(fn))
    else:
        files = [f for f in args.file if '-message' not in f]

    # save configuration:
    if args.save_config:
        file_name = files[0] if len(files) else ''
        cfg = configuration()
        cfg.load_files(cfgfile, file_name, 4, verbose)
        save_configuration(cfg, cfgfile)
        exit()
    elif len(files) == 0:
        parser.error('you need to specify at least one file for the analysis')

    # configure:
    cfg = configuration()
    cfg.load_files(cfgfile, files[0], 4, verbose)
    if args.format != 'auto':
        cfg.set('fileFormat', args.format)
    if args.unwrap:
        cfg.set('unwrapData', not cfg.value('unwrapData'))
        
    # plot parameter:
    spec_plots = 'auto'
    if args.spec_plots:
        spec_plots = True

    # multi-page pdfs:
    multi_pdf = None
    if len(args.multi_pdf) > 0:
        args.save_plot = True
        args.jobs = None  # PdfPages does not work yet with mutliprocessing
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

    # kwargs for data loader:
    load_kwargs = {}
    for s in args.load_kwargs:
        for kw in s.split(','):
            kws = kw.split(':')
            if len(kws) == 2:
                load_kwargs[kws[0].strip()] = kws[1].strip()

    # frequency limits for power spectrum:
    min_freq = 0.0
    max_freq = 3000.0
    log_freq = args.log_freq
    if log_freq > 0.0:
        min_freq = log_freq
        max_freq = min_freq*20
        if max_freq < 2000:
            max_freq = 2000
        log_freq = True
    else:
        log_freq = False

    # check if all input files are results:
    exts = TableData.ext_formats.values()
    results = True
    # check and group by recording:
    result_files = []
    for f in sorted(files):
        _, base_name, _, _, ftype, _, ext = parse_filename(f)
        if ext == 'zip' or (ext in exts and ftype in file_types):
            if len(result_files) == 0 or \
               not result_files[-1][-1].startswith(base_name):
                result_files.append([f])
            else:
                result_files[-1].append(f)
        else:
            results = False
            break
    if results:
        files = result_files

    # adjust verbosity:
    v = verbose
    if len(files) > 1:
        v += 1
    
    # run on pool:
    pool_args = (results, load_kwargs, cfg, args.channel, args.time,
                 args.time_file, args.mode, log_freq, min_freq,
                 max_freq, args.save_data, args.zip_file,
                 args.all_eods, spec_plots, args.skip_bad,
                 args.save_plot, multi_pdf, args.save_subplots,
                 args.outpath, args.keep_path, v-1, plot_level)
    if results:
        pool_args = (results, args.rawdata_path, load_kwargs,
                     args.all_eods, spec_plots, args.skip_bad,
                     args.save_plot, multi_pdf, args.save_subplots,
                     log_freq, min_freq, max_freq, args.outpath,
                     args.keep_path, v)
    if args.jobs is not None and (args.save_data or args.save_plot) and len(files) > 1:
        cpus = cpu_count() if args.jobs == 0 else args.jobs
        if verbose > 1:
            print('run on %d cpus' % cpus)
        p = Pool(cpus)
        p.map(run_thunderfish, zip(files, [pool_args]*len(files)))
    else:
        list(map(run_thunderfish, zip(files, [pool_args]*len(files))))
    if multi_pdf is not None:
        multi_pdf.close()


if __name__ == '__main__':
    freeze_support()  # needed by multiprocessing for some weired windows stuff
    main()
