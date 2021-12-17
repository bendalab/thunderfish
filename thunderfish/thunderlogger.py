"""# thunderlogger

Detect segments of interest in large data files and extract EOD waveforms.
"""

import sys
import os
import glob
import argparse
import traceback
import datetime as dt
import numpy as np
import pandas as pd
from scipy.signal import butter, lfilter
from types import SimpleNamespace
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from .version import __version__, __year__
from .configfile import ConfigFile
from .dataloader import DataLoader
from .tabledata import TableData, write_table_args
from .eventdetection import hist_threshold
from .eodanalysis import save_eod_waveform, save_wave_fish, save_pulse_fish
from .eodanalysis import save_wave_spectrum, save_pulse_spectrum, save_pulse_peaks
from .eodanalysis import load_eod_waveform, load_wave_fish, load_pulse_fish
from .eodanalysis import load_wave_spectrum, load_pulse_spectrum, load_pulse_peaks
from .eodanalysis import wave_similarity, pulse_similarity
from .thunderfish import configuration, save_configuration
from .thunderfish import detect_eods, remove_eod_files


def add_thunderlogger_config(cfg, detection_thresh='auto',
                             default_thresh=0.002, thresh_fac=3.0,
                             thresh_nbins=500):
    """Add parameters needed for by thunderlogger.

    Parameters
    ----------
    cfg: ConfigFile
        The configuration.
    detection_thresh: float or 'auto'
        Only data segments with standard deviation larger than this value
        are analyzed for EODs. If set to 'auto' a threshold is computed
        from all the data segments of a recording channel.
    default_thresh: float
        Threshold that is used if "detection_thresh" is set to "auto" and
        no data are available.
    thresh_fac: float
        The threshold for analysing data segments is set to the mean of the
        most-likely standard deviations plus this factor times the corresponding
        standard deviation.
    thresh_nbins: int
        The number of bins used to compute a histogram of the standard
        deviations of the data segments, from which the mean and standard
        deviation are estimated for automatically computing a threshold.
    """
    cfg.add_section('Thunderlogger:')
    cfg.add('detectionThreshold', detection_thresh, '', 'Only analyse data segements with a standard deviation that is larger than this threshold. If set to "auto" compute threshold from all the standard deviations of a recording channel.')
    cfg.add('detectionThresholdDefault', default_thresh, '', 'Threshold that is used if "detectionThreshold" is set to "auto" and no data are available.')
    cfg.add('detectionThresholdStdFac', thresh_fac, '', 'An automatically computed threshold for analysing data segments is set to the mean of the most-likely standard deviations plus this factor times the corresponding standard deviation.')
    cfg.add('detectionThresholdNBins', thresh_nbins, '', 'The number of bins used to compute a histogram of the standard deviations of the data segments, from which the mean and standard deviation are estimated for automatically computing a threshold.')
    cfg.add('startTime', 'none', '', 'Provide a start time for the recordings overwriting the meta data of the data files (YYYY-mm-ddTHH:MM:SS format).')


def extract_eods(files, thresholds, stds_only, cfg, verbose, plot_level,
                 thresh=0.002, max_deltaf=1.0, max_dist=0.00005,
                 deltat_max=dt.timedelta(minutes=5), start_time=None):
    t0s = []
    stds = None
    supra_thresh = None
    wave_fishes = None
    pulse_fishes = None
    if start_time is None:
        # XXX we should read this from the meta data:
        filename = os.path.splitext(os.path.basename(files[0]))[0]
        times = filename.split('-')[1]
        start_time = dt.datetime.strptime(times, '%Y%m%dT%H%M%S')
    toffs = start_time
    t1 = start_time
    unit = None
    for file in files:
        try:
            with DataLoader(file) as sf:
                # analyze:
                sys.stdout.write(file + ': ')
                unit = sf.unit
                if max_dist < 1.1/sf.samplerate:
                    max_dist = 1.1/sf.samplerate
                best_window_size = cfg.value('bestWindowSize')
                ndata = int(best_window_size * sf.samplerate)
                step = ndata//2
                b, a = butter(1, 10.0, 'hp', fs=sf.samplerate, output='ba')
                if stds is None:
                    stds = [[] for c in range(sf.channels)]
                    supra_thresh = [[] for c in range(sf.channels)]
                if wave_fishes is None:
                    wave_fishes = [[] for c in range(sf.channels)]
                if pulse_fishes is None:
                    pulse_fishes = [[] for c in range(sf.channels)]
                for k, data in enumerate(sf.blocks(ndata, step)):
                    sys.stdout.write('.')
                    sys.stdout.flush()
                    t0 = toffs + dt.timedelta(seconds=k*step/sf.samplerate)
                    t1 = t0 + dt.timedelta(seconds=ndata/sf.samplerate)
                    t0s.append(t0)
                    for channel in range(sf.channels):
                        if thresholds:
                            thresh = thresholds[channel]
                        fdata = lfilter(b, a, data[:,channel] - np.mean(data[:ndata//20,channel]))
                        sd = np.std(fdata)
                        stds[channel].append(sd)
                        supra_thresh[channel].append(1 if sd > thresh else 0)
                        if stds_only:
                            continue
                        if sd > thresh:
                            # clipping:
                            min_clip = cfg.value('minClipAmplitude')
                            if min_clip == 0.0:
                                min_clip = cfg.value('minDataAmplitude')
                            max_clip = cfg.value('maxClipAmplitude')
                            if max_clip == 0.0:
                                max_clip = cfg.value('maxDataAmplitude')
                            name = file
                            # detect EODs in the data:
                            _, _, _, eod_props, mean_eods, spec_data, peak_data, _, _, _ = \
                              detect_eods(data[:,channel], sf.samplerate,
                                          min_clip, max_clip,
                                          name, verbose, plot_level, cfg)
                            first_fish = True
                            for props, eod, spec, peaks in zip(eod_props, mean_eods,
                                                               spec_data, peak_data):
                                fish = None
                                fish_deltaf = 100000.0
                                if props['type'] == 'wave':
                                    for wfish in wave_fishes[channel]:
                                        deltaf = np.abs(wfish.props['EODf'] - props['EODf'])
                                        if deltaf < fish_deltaf:
                                            fish_deltaf = deltaf
                                            fish = wfish
                                    if fish_deltaf > max_deltaf:
                                        fish = None
                                    peaks = None
                                else:
                                    fish_dist = 10000.0
                                    for pfish in pulse_fishes[channel]:
                                        ddist = np.abs(pfish.props['dist'] -
                                                       props['dist'])
                                        if ddist < fish_dist:
                                            fish_dist = ddist
                                            fish_deltaf = np.abs(pfish.props['EODf'] -
                                                                 props['EODf'])
                                            fish = pfish
                                    if fish_dist > max_dist or \
                                       fish_deltaf > max_deltaf:
                                        fish = None
                                    spec = None
                                if fish is not None and \
                                   t0 - fish.times[-1][1] < deltat_max:
                                    if fish.times[-1][1] >= t0 and \
                                       np.abs(fish.times[-1][2] - props['EODf']) < 0.5 and \
                                       fish.times[-1][3] == channel and \
                                       fish.times[-1][4] == file:
                                        fish.times[-1][1] = t1
                                    else:
                                        fish.times.append([t0, t1, props['EODf'], channel, file])
                                    if props['p-p-amplitude'] > fish.props['p-p-amplitude']:
                                        fish.props = props
                                        fish.waveform = eod
                                        fish.spec = spec
                                        fish.peaks = peaks
                                else:
                                    new_fish = SimpleNamespace(props=props,
                                                               waveform=eod,
                                                               spec=spec,
                                                               peaks=peaks,
                                                               times=[[t0, t1, props['EODf'], channel, file]])
                                    if props['type'] == 'pulse':
                                        pulse_fishes[channel].append(new_fish)
                                    else:
                                        wave_fishes[channel].append(new_fish)
                                    if first_fish:
                                        sys.stdout.write('\n  ')
                                        first_fish = False
                                    sys.stdout.write('%6.1fHz %5s-fish @ %s\n  ' %
                                                     (props['EODf'], props['type'],
                                                      t0.strftime('%Y-%m-%dT%H:%M:%S')))
                toffs += dt.timedelta(seconds=len(sf)/sf.samplerate)
                sys.stdout.write('\n')
                sys.stdout.flush()
        except EOFError as error:
            # XXX we need to update toffs by means of the metadata of the next file!
            sys.stdout.write(file + ': ' + str(error) + '\n')
    if pulse_fishes is not None and len(pulse_fishes) > 0:
        pulse_fishes = [[pulse_fishes[c][i] for i in
                         np.argsort([fish.props['EODf'] for fish in pulse_fishes[c]])]
                        for c in range(len(pulse_fishes))]
    if wave_fishes is not None and len(wave_fishes) > 0:
        wave_fishes = [[wave_fishes[c][i] for i in
                        np.argsort([fish.props['EODf'] for fish in wave_fishes[c]])]
                       for c in range(len(wave_fishes))]
    return pulse_fishes, wave_fishes, start_time, toffs, t0s, stds, supra_thresh, unit


def save_times(times, idx, output_basename, name, **kwargs):
    td = TableData()
    td.append('index', '', '%d', [idx] * len(times))
    td.append('tstart', '', '%s',
              [t[0].strftime('%Y-%m-%dT%H:%M:%S') for t in times])
    td.append('tend', '', '%s',
              [t[1].strftime('%Y-%m-%dT%H:%M:%S') for t in times])
    if len(times[0]) > 2:
        td.append('EODf', 'Hz', '%.1f', [t[2] for t in times])
    td.append('device', '', '%s',
              [name for t in times])
    if len(times[0]) > 2:
        td.append('channel', '', '%d', [t[3] for t in times])
        td.append('file', '', '%s', [t[4] for t in times])
    fp = output_basename + '-times'
    if idx is not None:
        fp += '-%d' % idx
    td.write(fp, **kwargs)
    

def load_times(file_path):
    data = TableData(file_path).data_frame()
    data['index'] = data['index'].astype(np.int)
    data['tstart'] = pd.to_datetime(data['tstart'])
    data['tstart'] = pd.Series(data['tstart'].dt.to_pydatetime(), dtype=object)
    data['tend'] = pd.to_datetime(data['tend'])
    data['tend'] = pd.Series(data['tend'].dt.to_pydatetime(), dtype=object)
    if 'channel' in data:
        data['channel'] = data['channel'].astype(np.int)
    return data


def save_power(times, stds, supra_thresh, unit, output_basename, **kwargs):
    td = TableData()
    td.append('index', '', '%d', list(range(len(times))))
    td.append('time', '', '%s',
              [t.strftime('%Y-%m-%dT%H:%M:%S') for t in times])
    for c, (std, thresh) in enumerate(zip(stds, supra_thresh)):
        td.append('channel%d'%c, unit, '%g', std)
        td.append('thresh%d'%c, '', '%d', thresh)
    fp = output_basename + '-stdevs'
    td.write(fp, **kwargs)


def load_power(file_path, start_time=None):
    base = os.path.basename(file_path)
    device = base[0:base.find('-stdevs')]
    data = TableData(file_path)
    times = []
    for row in range(data.rows()):
        times.append(dt.datetime.strptime(data[row,'time'],
                                          '%Y-%m-%dT%H:%M:%S'))
    if start_time is not None:
        deltat = start_time - times[0]
        for k in range(len(times)):
            times[k] += deltat
    channels = (data.columns()-2)//2
    stds = np.zeros((len(times), channels))
    supra_thresh = np.zeros((len(times), channels), dtype=np.int)
    for c in range(channels):
        stds[:,c] = data[:,'channel%d'%c]
        supra_thresh[:,c] = data[:,'thresh%d'%c]
    return np.array(times), stds, supra_thresh, device


def save_data(output_folder, name, pulse_fishes, wave_fishes,
              tstart, tend, t0s, stds, supra_thresh, unit, cfg):
    output_basename = os.path.join(output_folder, name)
    if pulse_fishes is not None:
        for c in range(len(pulse_fishes)):
            out_path = output_basename + '-c%d' % c
            idx = 0
            # pulse fish:
            pulse_props = []
            for fish in pulse_fishes[c]:
                save_eod_waveform(fish.waveform, unit, idx, out_path,
                                  **write_table_args(cfg))
                if fish.peaks is not None:
                    save_pulse_peaks(fish.peaks, unit, idx, out_path,
                                     **write_table_args(cfg))
                save_times(fish.times, idx, out_path, name,
                           **write_table_args(cfg))
                pulse_props.append(fish.props)
                pulse_props[-1]['index'] = idx
                idx += 1
            save_pulse_fish(pulse_props, unit, out_path,
                            **write_table_args(cfg))
        # wave fish:
        wave_props = []
        if wave_fishes is not None:
            for fish in wave_fishes[c]:
                save_eod_waveform(fish.waveform, unit, idx, out_path,
                                  **write_table_args(cfg))
                if fish.spec is not None:
                    save_wave_spectrum(fish.spec, unit, idx, out_path,
                                       **write_table_args(cfg))
                save_times(fish.times, idx, out_path, name,
                           **write_table_args(cfg))
                wave_props.append(fish.props)
                wave_props[-1]['index'] = idx
                idx += 1
            save_wave_fish(wave_props, unit, out_path,
                           **write_table_args(cfg))
    # recording time window:
    save_times([(tstart, tend)], None, output_basename, name,
               **write_table_args(cfg))
    # signal power:
    if stds is not None and len(stds) > 0:
        save_power(t0s, stds, supra_thresh, unit, output_basename,
                   **write_table_args(cfg))


def load_data(files, start_time=None):
    all_files = []
    for file in files:
        if os.path.isdir(file):
            all_files.extend(glob.glob(os.path.join(file, '*fish*')))
        else:
            all_files.append(file)
    pulse_fishes = []
    wave_fishes = []
    channels = set()
    for file in all_files:
        if 'pulse' in os.path.basename(file):
            pulse_props = load_pulse_fish(file)
            base_file, ext = os.path.splitext(file)
            base_file = base_file[:base_file.rfind('pulse')]
            count = 0
            for props in pulse_props:
                idx = props['index']
                waveform, unit = \
                    load_eod_waveform(base_file + 'eodwaveform-%d'%idx + ext)
                times = load_times(base_file + 'times-%d'%idx + ext)
                times['index'] = idx
                fish = SimpleNamespace(props=props,
                                       waveform=waveform,
                                       unit=unit,
                                       times=times)
                for i, t in times.iterrows():
                    channels.add((t['device'], t['channel']))
                try:
                    peaks, unit = \
                        load_pulse_peaks(base_file + 'pulsepeaks-%d'%idx + ext)
                    fish.peaks = peaks
                except FileNotFoundError:
                    fish.peaks = None
                pulse_fishes.append(fish)
                count += 1
                #if count > 300: # XXX REMOVE
                #    break
        elif 'wave' in os.path.basename(file):
            wave_props = load_wave_fish(file)
            base_file, ext = os.path.splitext(file)
            base_file = base_file[:base_file.rfind('wave')]
            count = 0
            for props in wave_props:
                idx = props['index']
                waveform, unit = \
                    load_eod_waveform(base_file + 'eodwaveform-%d'%idx + ext)
                times = load_times(base_file + 'times-%d'%idx + ext)
                times['index'] = idx
                fish = SimpleNamespace(props=props,
                                       waveform=waveform,
                                       unit=unit,
                                       times=times)
                for i, t in times.iterrows():
                    channels.add((t['device'], t['channel']))
                try:
                    spec, unit = \
                        load_wave_spectrum(base_file + 'wavespectrum-%d'%idx + ext)
                    fish.spec = spec
                except FileNotFoundError:
                    fish.spec = None
                wave_fishes.append(fish)
                count += 1
                #if count > 300: # XXX REMOVE
                #    break
    base_file = base_file[:base_file.rfind('-c')+1]
    times = load_times(base_file + 'times' + ext)
    tstart = times.tstart[0]
    tend = times.tend[0]
    if start_time is not None:
        deltat = start_time - tstart
        tstart += deltat
        tend += deltat
        for fish in pulse_fishes + wave_fishes:
            fish.times.tstart += deltat
            fish.times.tend += deltat
    return pulse_fishes, wave_fishes, tstart, tend, sorted(channels)


def plot_signal_power(times, stds, supra_threshs, devices, thresholds,
                      title, output_folder):
    plt.rcParams['axes.xmargin'] = 0
    plt.rcParams['axes.ymargin'] = 0
    n = 0
    for s in stds:
        n += s.shape[1]
    h = n*4.0
    t = 0.8 if title else 0.1
    fig, axs = plt.subplots(n, 1, figsize=(16/2.54, h/2.54),
                            sharex=True, sharey=True)
    fig.subplots_adjust(left=0.1, right=0.99, top=1-t/h, bottom=1.6/h,
                        hspace=0)
    i = 0
    for time, cstds, threshs, device in zip(times, stds, supra_threshs, devices):
        for c, (std, thresh) in enumerate(zip(cstds.T, threshs.T)):
            ax = axs[i]
            ax.plot(time, std)
            if thresholds:
                ax.axhline(thresholds[i], color='k', lw=0.5)
            elif len(std[thresh<1]) > 0:
                thresh = np.max(std[thresh<1])
                ax.axhline(thresh, color='k', lw=0.5)
            #stdm = np.ma.masked_where(thresh < 1, std)
            #ax.plot(time, stdm)
            ax.set_yscale('log')
            #ax.set_ylim(bottom=0)
            ax.set_ylabel('%s-c%d' % (device, c))
            i += 1
    if title:
        axs[0].set_title(title)
    axs[-1].xaxis.set_major_formatter(mdates.DateFormatter('%b %d %Hh'))
    plt.setp(axs[-1].get_xticklabels(), ha='right',
             rotation=30, rotation_mode='anchor')
    fig.savefig(os.path.join(output_folder, 'signalpowers.pdf'))
    plt.show()
    

def merge_fish(pulse_fishes, wave_fishes,
               max_noise=0.1, max_deltaf=10.0, max_dist=0.0002, max_rms=0.3):
    pulse_eods = []
    for i in np.argsort([fish.props['P2-P1-dist'] for fish in pulse_fishes]):
        if pulse_fishes[i].props['noise'] > max_noise:
            continue
        if pulse_fishes[i].props['P2-P1-dist'] <= 0.0:
            continue
        if len(pulse_eods) > 0 and \
           np.abs(pulse_fishes[i].props['P2-P1-dist'] - pulse_eods[-1].props['P2-P1-dist']) <= max_dist and \
           pulse_similarity(pulse_fishes[i].waveform, pulse_eods[-1].waveform, 4) < max_rms:
            pulse_eods[-1].times.append(pulse_fishes[i].times)
            if not hasattr(pulse_eods[-1], 'othereods'):
                pulse_eods[-1].othereods = [pulse_eods[-1].waveform]
            pulse_eods[-1].othereods.append(pulse_fishes[i].waveform)
            if pulse_fishes[i].props['p-p-amplitude'] > pulse_eods[-1].props['p-p-amplitude']:
                pulse_eods[-1].waveform = pulse_fishes[i].waveform
                pulse_eods[-1].props = pulse_fishes[i].props
            continue
        pulse_eods.append(pulse_fishes[i])
    pulse_eods = [pulse_eods[i] for i in np.argsort([fish.props['EODf'] for fish in pulse_eods])]
    
    wave_eods = []
    for i in np.argsort([fish.props['EODf'] for fish in wave_fishes]):
        if wave_fishes[i].props['noise'] > max_noise:
            continue
        if len(wave_eods) > 0 and \
           np.abs(wave_fishes[i].props['EODf'] - wave_eods[-1].props['EODf']) < max_deltaf and \
           wave_similarity(wave_fishes[i].waveform, wave_eods[-1].waveform,
                           wave_fishes[i].props['EODf'], wave_eods[-1].props['EODf']) < max_rms:
            wave_eods[-1].times.append(wave_fishes[i].times)
            if not hasattr(wave_eods[-1], 'othereods'):
                wave_eods[-1].othereods = []
            wave_eods[-1].othereods.append(wave_fishes[i].waveform)
            if wave_fishes[i].props['p-p-amplitude'] > wave_eods[-1].props['p-p-amplitude']:
                wave_eods[-1].waveform = wave_fishes[i].waveform
                wave_eods[-1].props = wave_fishes[i].props
            continue
        wave_eods.append(wave_fishes[i])
    return pulse_eods, wave_eods


def plot_eod_occurances(pulse_fishes, wave_fishes, tstart, tend,
                        channels, output_folder):
    channel_colors = ['#2060A7', '#40A787', '#478010', '#F0D730',
                      '#C02717', '#873770', '#008797', '#007030',
                      '#AAB71B', '#F78017', '#D03050', '#53379B']
    plt.rcParams['axes.facecolor'] = 'none'
    plt.rcParams['axes.xmargin'] = 0
    plt.rcParams['axes.ymargin'] = 0.05
    plt.rcParams['font.family'] = 'sans-serif'
    n = len(pulse_fishes) + len(wave_fishes)
    h = n*2.5 + 2.5 + 0.3
    fig, axs = plt.subplots(n, 2, squeeze=False, figsize=(16/2.54, h/2.54),
                            gridspec_kw=dict(width_ratios=(1,2)))
    fig.subplots_adjust(left=0.02, right=0.97, top=1-0.3/h, bottom=2.2/h,
                        hspace=0.2)
    pi = 0
    prev_xscale = 0.0
    for ax, fish in zip(axs, pulse_fishes + wave_fishes):
        # EOD waveform:
        ax[0].spines['left'].set_visible(False)
        ax[0].spines['right'].set_visible(False)
        ax[0].spines['top'].set_visible(False)
        ax[0].spines['bottom'].set_visible(False)
        ax[0].xaxis.set_visible(False)
        ax[0].yaxis.set_visible(False)
        time = 1000.0 * fish.waveform[:,0]
        #ax[0].plot([time[0], time[-1]], [0.0, 0.0],
        #           zorder=-10, lw=1, color='#AAAAAA')
        if hasattr(fish, 'othereods'):
            for eod in fish.othereods:
                ax[0].plot(1000.0*eod[:,0], eod[:,1],
                           zorder=5, lw=1, color='#AAAAAA')
        ax[0].plot(time, fish.waveform[:,1],
                   zorder=10, lw=2, color='#C02717')
        ax[0].text(0.0, 1.0, '%.0f\u2009Hz' % fish.props['EODf'],
                   transform=ax[0].transAxes, va='baseline', zorder=20)
        if fish.props['type'] == 'wave':
            lim = 750.0/fish.props['EODf']
            ax[0].set_xlim([-lim, +lim])
            tmax = lim
        else:
            ax[0].set_xlim(time[0], time[-1])
            tmax = time[-1]
        xscale = 1.0
        if tmax < 1.0:
            xscale = 0.5
        elif tmax > 10.0:
            xscale = 5.0
        ymin = np.min(fish.waveform[:,1])
        ymax = np.max(fish.waveform[:,1])
        ax[0].plot((tmax-xscale, tmax), (ymin - 0.04*(ymax-ymin),)*2,
                   'k', lw=3, clip_on=False, zorder=0)
        if ax[0] is axs[-1,0] or xscale != prev_xscale:
            if xscale < 1.0:
                ax[0].text(tmax-0.5*xscale, ymin - 0.1*(ymax-ymin),
                           '%.0f\u2009\u00b5s' % (1000.0*xscale),
                           ha='center', va='top', zorder=0)
            else:
                ax[0].text(tmax-0.5*xscale, ymin - 0.1*(ymax-ymin),
                           '%.0f\u2009ms' % xscale,
                           ha='center', va='top', zorder=0)
        prev_xscale = xscale
        # time bar:
        ax[1].xaxis.set_major_formatter(mdates.DateFormatter('%b %d %Hh'))
        min_eodf = 10000
        max_eodf = 0
        for index, time in fish.times.iterrows():
            if time['EODf'] < min_eodf:
                min_eodf = time['EODf']
            if time['EODf'] > max_eodf:
                max_eodf = time['EODf']
            ax[1].plot([time['tstart'], time['tend']],
                       [time['EODf'], time['EODf']],
                       lw=5, color=channel_colors[channels.index((time['device'], time['channel']))%len(channel_colors)])
        if max_eodf > min_eodf + 10.0:
            ax[1].text(0.0, 1.0, '%.0f \u2013 %.0f\u2009Hz' % (min_eodf, max_eodf),
                       transform=ax[1].transAxes, va='baseline', zorder=20)
        ax[1].set_xlim(tstart, tend)
        ax[1].spines['left'].set_visible(False)
        ax[1].spines['right'].set_visible(False)
        ax[1].yaxis.set_visible(False)
        ax[1].spines['top'].set_visible(False)
        if ax[1] is not axs[-1,1]:
            ax[1].spines['bottom'].set_visible(False)
            ax[1].xaxis.set_visible(False)
        else:
            plt.setp(ax[1].get_xticklabels(), ha='right',
                     rotation=30, rotation_mode='anchor')
    fig.savefig(os.path.join(output_folder, 'eodwaveforms.pdf'))

        
def main():
    # config file name:
    cfgfile = __package__ + '.cfg'

    # command line arguments:
    parser = argparse.ArgumentParser(add_help=False,
        description='Extract EOD waveforms of weakly electric fish from logger data.',
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
    parser.add_argument('-f', dest='format', default='auto', type=str,
                        choices=TableData.formats + ['py'],
                        help='file format used for saving analysis results, defaults to the format specified in the configuration file or "csv"')
    parser.add_argument('-p', dest='save_plot', action='store_true',
                        help='plot analyzed data')
    parser.add_argument('-m', dest='merge', action='store_true',
                        help='merge similar EODs before plotting')
    parser.add_argument('-s', dest='stds_only', action='store_true',
                        help='analyze or plot standard deviation of data only')
    parser.add_argument('-o', dest='outpath', default='.', type=str,
                        help='path where to store results and figures (defaults to current working directory)')
    parser.add_argument('-k', dest='keep_path', action='store_true',
                        help='keep path of input file when saving analysis files, i.e. append path of input file to OUTPATH')
    parser.add_argument('-n', dest='name', default='', type=str,
                        help='base name of all output files or title of plots')
    parser.add_argument('file', nargs='*', default='', type=str,
                        help='name of a file with time series data of an EOD recording, may include wildcards')
    args = parser.parse_args()

    # help:
    if args.help:
        parser.print_help()
        print('')
        print('examples:')
        print('- write configuration file:')
        print('  > thunderlogger -c')
        print('- compute standard deviations of data segments:')
        print('  > thunderlogger -o results -k -n logger1 -s river1/logger1-*.wav')
        print('- plot the standard deviations and the computed threshold:')
        print('  > thunderlogger -o plots -k -s -p -n river1 results/river1/logger1-stdevs.*')
        print('  you may adapt the settings in the configureation file "thunderfish.cfg"')
        print('- extract EODs from the data:')
        print('  > thunderlogger -o results -k river1/logger1-*.wav')
        print('- merge and plot extracted EODs:')
        print('  > thunderlogger -o plots -k -p -m results/river*/*fish.*')
        parser.exit()

    # set verbosity level from command line:
    verbose = args.verbose
    plot_level = args.plot_level
    if verbose < plot_level:
        verbose = plot_level

    # expand wildcard patterns:
    files = []
    if os.name == 'nt':
        for fn in args.file:
            files.extend(glob.glob(fn))
    else:
        files = args.file

    if args.save_config:
        # save configuration:
        file_name = files[0] if len(files) else ''
        cfg = configuration()
        add_thunderlogger_config(cfg)
        cfg.load_files(cfgfile, file_name, 4, verbose)
        save_configuration(cfg, cfgfile)
        exit()
    elif len(files) == 0:
        parser.error('you need to specify at least one file for the analysis')

    # configure:
    cfg = configuration()
    add_thunderlogger_config(cfg)
    cfg.load_files(cfgfile, files[0], 4, verbose-1)
    if args.format != 'auto':
        cfg.set('fileFormat', args.format)

    # create output folder for data and plots:
    output_folder = args.outpath
    if args.keep_path:
        output_folder = os.path.join(output_folder,
                                     os.path.split(files[0])[0])
    if not os.path.exists(output_folder):
        if verbose > 1:
            print('mkdir %s' % output_folder)
        os.makedirs(output_folder)
        
    # start time:
    start_time = None
    if cfg.value('startTime') != 'none':
        cfg.value('startTime')
        start_time = dt.datetime.strptime(cfg.value('startTime'), '%Y-%m-%dT%H:%M:%S')
        
    # analyze and save data:
    plt.ioff()
    if not args.save_plot:
        # assemble device name and output file:
        if len(args.name) > 0:
            device_name = args.name
        else:
            device_name = os.path.basename(files[0])
            device_name = device_name[:device_name.find('-')]
        output_basename = os.path.join(output_folder, device_name)
        # compute thresholds:
        thresholds = []
        power_file = output_basename + '-stdevs.csv'
        thresh = cfg.value('detectionThreshold')
        if thresh == 'auto':
            thresh = cfg.value('detectionThresholdDefault')
            if os.path.isfile(power_file):
                _, powers, _, _ = load_power(power_file)
                for std in powers.T:
                    ss, cc = hist_threshold(std,
                        thresh_fac=cfg.value('detectionThresholdStdFac'),
                        nbins=cfg.value('detectionThresholdNBins'))
                    thresholds.append(cc + ss)
        else:
            thresh = float(thresh)
        pulse_fishes, wave_fishes, tstart, tend, t0s, \
            stds, supra_thresh, unit = \
            extract_eods(files, thresholds,
                         args.stds_only, cfg, verbose, plot_level,
                         thresh=thresh, start_time=start_time)
        remove_eod_files(output_basename, verbose, cfg)
        save_data(output_folder, device_name, pulse_fishes, wave_fishes,
                  tstart, tend, t0s, stds, supra_thresh, unit, cfg)
        sys.stdout.write('DONE!\n')
        if args.stds_only:
            sys.stdout.write('Signal powers saved in %s\n' % (output_folder+'-stdevs.csv'))
            sys.stdout.write('To generate plots run thunderlogger with the -p and -s flags on the generated file:\n')
            sys.stdout.write('> thunderlogger -p -s %s\n' % (output_folder+'-stdevs.csv'))
        else:
            sys.stdout.write('Extracted EOD waveforms saved in %s\n' % output_folder)
            sys.stdout.write('To generate plots run thunderlogger with the -p flag on the generated files:\n')
            sys.stdout.write('> thunderlogger -p -o %s%s %s\n' %
                         (args.outpath, ' -k' if args.keep_path else '',
                          output_basename))
    else:
        if args.stds_only:
            times = []
            stds = []
            supra_threshs = []
            devices = []
            thresholds = []
            for file in files:
                t, p, s, d = load_power(file, start_time)
                times.append(t)
                stds.append(p)
                supra_threshs.append(s)
                devices.append(d)
                # compute detection thresholds:
                for std in p.T:
                    if cfg.value('detectionThreshold') == 'auto':
                        ss, cc = hist_threshold(std,
                            thresh_fac=cfg.value('detectionThresholdStdFac'),
                            nbins=cfg.value('detectionThresholdNBins'))
                        thresholds.append(cc + ss)
                    else:
                        thresholds.append(float(cfg.value('detectionThreshold')))
            plot_signal_power(times, stds, supra_threshs, devices, thresholds,
                              args.name, output_folder)
        else:
            pulse_fishes, wave_fishes, tstart, tend, channels = \
                load_data(files, start_time)
            if args.merge:
                pulse_fishes, wave_fishes = merge_fish(pulse_fishes, wave_fishes)
            plot_eod_occurances(pulse_fishes, wave_fishes, tstart, tend,
                                channels, output_folder)


if __name__ == '__main__':
    main()
