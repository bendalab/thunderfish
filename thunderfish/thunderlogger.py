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
from scipy.signal import butter, lfilter
from types import SimpleNamespace
#import matplotlib
#matplotlib.use('Agg')  # for headless system
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
import matplotlib.dates as mdates
#from matplotlib.backends.backend_pdf import PdfPages
from .version import __version__, __year__
from .configfile import ConfigFile
from .dataloader import DataLoader
from .tabledata import TableData, write_table_args
from .eodanalysis import save_eod_waveform, save_wave_fish, save_pulse_fish
from .eodanalysis import save_wave_spectrum, save_pulse_spectrum, save_pulse_peaks
from .eodanalysis import load_eod_waveform, load_wave_fish, load_pulse_fish
from .eodanalysis import load_wave_spectrum, load_pulse_spectrum, load_pulse_peaks
from .thunderfish import configuration, detect_eods, remove_eod_files


def extract_eods(files, cfg, verbose, plot_level, sd_thresh=0.0015,
                 max_deltaf=1.0, max_dist=0.00005,
                 deltat_max=dt.timedelta(minutes=5)):
    t0s = []
    stds = None
    wave_fishes = None
    pulse_fishes = None
    # XXX we should read this from the meta data:
    filename = os.path.splitext(os.path.basename(files[0]))[0]
    times = filename.split('-')[1]
    tstart = dt.datetime.strptime(times, '%Y%m%dT%H%M%S')
    toffs = tstart
    t1 = tstart
    unit = None
    for file in files:
        with DataLoader(file) as sf:
            # common prefix:
            fn = os.path.splitext(os.path.basename(file))[0]
            for i, c in enumerate(filename):
                if c != fn[i]:
                    filename = filename[:i]
                    break
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
            if wave_fishes is None:
                wave_fishes = [[] for c in range(sf.channels)]
            if pulse_fishes is None:
                pulse_fishes = [[] for c in range(sf.channels)]
            for k, data in enumerate(sf.blocks(ndata, step)):
                if k<80:
                    continue
                sys.stdout.write('.')
                sys.stdout.flush()
                t0 = toffs + dt.timedelta(seconds=k*step/sf.samplerate)
                t1 = t0 + dt.timedelta(seconds=ndata/sf.samplerate)
                t0s.append(t0)
                for channel in range(sf.channels):
                    fdata = lfilter(b, a, data[:,channel] - np.mean(data[:ndata//20,channel]))
                    sd = np.std(fdata)
                    stds[channel].append(sd)
                    if sd > sd_thresh:
                        # clipping:
                        # XXX TODO:
                        clipped = False
                        min_clip = -1.0
                        max_clip = 1.0
                        name = file
                        # detect EODs in the data:
                        _, _, _, eod_props, mean_eods, spec_data, peak_data, _, _, _ = \
                          detect_eods(data[:,channel], sf.samplerate,
                                      clipped, min_clip, max_clip,
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
    pulse_fishes = [[pulse_fishes[c][i] for i in
                     np.argsort([fish.props['EODf'] for fish in pulse_fishes[c]])]
                    for c in range(len(pulse_fishes))]
    wave_fishes = [[wave_fishes[c][i] for i in
                    np.argsort([fish.props['EODf'] for fish in wave_fishes[c]])]
                   for c in range(len(wave_fishes))]
    return pulse_fishes, wave_fishes, tstart, toffs, t0s, stds, unit, filename


def save_times(times, idx, output_basename, name, **kwargs):
    td = TableData()
    td.append('index', '', '%d', list(range(len(times))))
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
    data = TableData(file_path)
    times = []
    for row in range(data.rows()):
        tstart = dt.datetime.strptime(data[row,'tstart'], '%Y-%m-%dT%H:%M:%S')
        tend = dt.datetime.strptime(data[row,'tend'], '%Y-%m-%dT%H:%M:%S')
        t = [tstart, tend]
        if 'EODf' in data:
            t.append(data[row, 'EODf'])
        if 'device' in data:
            t.append(data[row, 'device'])
        if 'channel' in data:
            t.append(data[row,'channel'])
        if 'file' in data:
            t.append(data[row,'file'])
        times.append(t)
    return times
    

def save_data(output_folder, name, pulse_fishes, wave_fishes,
              tstart, tend, t0s, stds, unit, cfg):
    output_basename = os.path.join(output_folder, name)
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
    td = TableData()
    td.append('index', '', '%d', list(range(len(t0s))))
    td.append('time', '', '%s',
              [t.strftime('%Y-%m-%dT%H:%M:%S') for t in t0s])
    for c, std in enumerate(stds):
        td.append('channel%d'%c, unit, '%g', std)
    fp = output_basename + '-stdevs'
    td.write(fp, **write_table_args(cfg))


def load_data(files):
    all_files = []
    for file in files:
        if os.path.isdir(file):
            all_files.extend(glob.glob(os.path.join(file, '*fish*')))
        else:
            all_files.append(file)
    pulse_fishes = []
    wave_fishes = []
    for file in all_files:
        if 'pulse' in os.path.basename(file):
            pulse_props = load_pulse_fish(file)
            base_file, ext = os.path.splitext(file)
            base_file = base_file[:base_file.rfind('pulse')]
            for props in pulse_props:
                idx = props['index']
                waveform, unit = \
                    load_eod_waveform(base_file + 'eodwaveform-%d'%idx + ext)
                times = load_times(base_file + 'times-%d'%idx + ext)
                fish = SimpleNamespace(props=props,
                                       waveform=waveform,
                                       unit=unit,
                                       times=times)
                try:
                    peaks, unit = \
                        load_pulse_peaks(base_file + 'pulsepeaks-%d'%idx + ext)
                    fish.peaks = peaks
                except FileNotFoundError:
                    fish.peaks = None
                pulse_fishes.append(fish)
        elif 'wave' in os.path.basename(file):
            wave_props = load_wave_fish(file)
            base_file, ext = os.path.splitext(file)
            base_file = base_file[:base_file.rfind('wave')]
            for props in wave_props:
                idx = props['index']
                waveform, unit = \
                    load_eod_waveform(base_file + 'eodwaveform-%d'%idx + ext)
                times = load_times(base_file + 'times-%d'%idx + ext)
                fish = SimpleNamespace(props=props,
                                       waveform=waveform,
                                       unit=unit,
                                       times=times)
                try:
                    spec, unit = \
                        load_wave_spectrum(base_file + 'wavespectrum-%d'%idx + ext)
                    fish.spec = spec
                except FileNotFoundError:
                    fish.spec = None
                wave_fishes.append(fish)
    base_file = base_file[:base_file.rfind('-c')+1]
    times = load_times(base_file + 'times' + ext)
    tstart = times[0][0]
    tend = times[0][1]
    return pulse_fishes, wave_fishes, tstart, tend


def compress_fish(pulse_fishes, wave_fishes,
                  max_noise=0.1, max_deltaf=1.0, max_dist=0.00002):
    pulse_eods = []
    for i in np.argsort([fish.props['P2-P1-dist'] for fish in pulse_fishes]):
        if pulse_fishes[i].props['noise'] > max_noise:
            continue
        if pulse_fishes[i].props['P2-P1-dist'] <= 0.0:
            continue
        if len(pulse_eods) > 0 and \
           np.abs(pulse_fishes[i].props['P2-P1-dist'] - pulse_eods[-1].props['P2-P1-dist']) <= max_dist:
            pulse_eods[-1].times.extend(pulse_fishes[i].times)
            continue
        pulse_eods.append(pulse_fishes[i])
    pulse_eods = [pulse_eods[i] for i in np.argsort([fish.props['EODf'] for fish in pulse_eods])]
    
    wave_eods = []
    for i in np.argsort([fish.props['EODf'] for fish in wave_fishes]):
        if wave_fishes[i].props['noise'] > max_noise:
            continue
        if len(wave_eods) > 0 and \
           np.abs(wave_fishes[i].props['EODf'] - wave_eods[-1].props['EODf']) < max_deltaf:
            wave_eods[-1].times.extend(wave_fishes[i].times)
            continue
        wave_eods.append(wave_fishes[i])
    return pulse_eods, wave_eods


def plot_eod_occurances(pulse_fishes, wave_fishes, tstart, tend,
                        save_plot, output_folder):
    n = len(pulse_fishes) + len(wave_fishes)
    h = n*2.5
    fig, axs = plt.subplots(n, 2, figsize=(16/2.54, h/2.54),
                            gridspec_kw=dict(width_ratios=(1,2)))
    fig.subplots_adjust(left=0.02, right=0.97, top=1-0.2/h, bottom=2.5/h)
    pi = 0
    for ax, fish in zip(axs, pulse_fishes + wave_fishes):
        # EOD waveform:
        time = 1000.0 * fish.waveform[:,0]
        #ax[0].plot([time[0], time[-1]], [0.0, 0.0],
        #           zorder=-10, lw=1, color='#AAAAAA')
        ax[0].plot(time, fish.waveform[:,1],
                   zorder=10, lw=2, color='#C02717')
        ax[0].text(0.0, 1.0, '%.1f\u2009Hz' % fish.props['EODf'],
                   transform=ax[0].transAxes, va='top')
        if fish.props['type'] == 'wave':
            lim = 750.0/fish.props['EODf']
            ax[0].set_xlim([-lim, +lim])
            tmax = lim
        else:
            ax[0].set_xlim(time[0], time[-1])
            tmax = time[-1]
        trans = transforms.blended_transform_factory(ax[0].transData,
                                                     ax[0].transAxes)
        ax[0].plot((tmax-1.0, tmax), (-0.05, -0.05),
                   'k', lw=3, transform=trans, clip_on=False)
        if ax[0] is axs[-1,0]:
            ax[0].text(tmax-0.5, -0.13, '1\u2009ms', transform=trans, ha='center', va='top')
        ax[0].spines['left'].set_visible(False)
        ax[0].spines['right'].set_visible(False)
        ax[0].spines['top'].set_visible(False)
        ax[0].spines['bottom'].set_visible(False)
        ax[0].xaxis.set_visible(False)
        ax[0].yaxis.set_visible(False)
        # time bar:
        ax[1].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%dT%H:%M'))
        for time in fish.times:
            ax[1].plot(time[:2], [time[3], time[3]], lw=5, color='#2060A7')
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
    if save_plot:
        fig.savefig(os.path.join(output_folder, 'plot.pdf'))
    else:
        plt.show()

        
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
    parser.add_argument('-v', action='count', dest='verbose', default=0,
                        help='verbosity level. Increase by specifying -v multiple times, or like -vvv')
    parser.add_argument('-V', action='count', dest='plot_level', default=0,
                        help='level for debugging plots. Increase by specifying -V multiple times, or like -VVV')
    parser.add_argument('-c', dest='save_config', action='store_true',
                        help='save configuration to file {0} after reading all configuration files'.format(cfgfile))
    parser.add_argument('--channel', default=0, type=int,
                        help='channel to be analyzed (defaults to first channel)')
    parser.add_argument('-f', dest='format', default='auto', type=str,
                        choices=TableData.formats + ['py'],
                        help='file format used for saving analysis results, defaults to the format specified in the configuration file or "dat"')
    parser.add_argument('-p', dest='save_plot', action='store_true',
                        help='plot analyzed data')
    parser.add_argument('-o', dest='outpath', default='.', type=str,
                        help='path where to store results and figures (defaults to current working directory)')
    parser.add_argument('-k', dest='keep_path', action='store_true',
                        help='keep path of input file when saving analysis files, i.e. append path of input file to OUTPATH')
    parser.add_argument('-n', dest='name', default='', type=str,
                        help='base name of all output files')
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
    verbose = args.verbose
    plot_level = args.plot_level
    if verbose < plot_level:
        verbose = plot_level

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

    # create output folder for data and plots:
    output_folder = args.outpath
    if args.keep_path:
        output_folder = os.path.join(output_folder,
                                     os.path.split(args.file[0])[0])
    if not os.path.exists(output_folder):
        if verbose > 1:
            print('mkdir %s' % output_folder)
        os.makedirs(output_folder)
    # analyze and save data:
    if not args.save_plot:
        pulse_fishes, wave_fishes, tstart, tend, t0s, stds, unit, filename = \
            extract_eods(args.file, cfg, verbose, plot_level)
        if len(args.name) > 0:
            filename = args.name
        if len(filename) == 0:
            filename = 'thunder'
        elif filename[-1] == '-':
            filename = filename[:,-1]
        output_basename = os.path.join(output_folder, filename)
        remove_eod_files(output_basename, verbose, cfg)
        save_data(output_folder, filename, pulse_fishes, wave_fishes,
                  tstart, tend, t0s, stds, unit, cfg)
        sys.stdout.write('DONE!\n')
        sys.stdout.write('Extracted EOD waveforms saved in %s\n' % output_folder)
        sys.stdout.write('To generate plots run thunderlogger with the -p flag on the generated files:\n')
        sys.stdout.write('> thunderlogger -p -o %s%s %s\n' %
                         (args.outpath, ' -k' if args.keep_path else '',
                          output_basename))
    else:
        pulse_fishes, wave_fishes, tstart, tend = load_data(args.file)
        pulse_fishes, wave_fishes = compress_fish(pulse_fishes, wave_fishes)
        plot_eod_occurances(pulse_fishes, wave_fishes, tstart, tend,
                            True, output_folder)


if __name__ == '__main__':
    main()
