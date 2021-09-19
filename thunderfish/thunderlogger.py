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
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
import matplotlib.dates as mdates
from matplotlib.backends.backend_pdf import PdfPages
from .version import __version__, __year__
from .configfile import ConfigFile
from .dataloader import DataLoader
from .tabledata import TableData
from .thunderfish import configuration, detect_eods, remove_eod_files, save_eods


def extract_eods(files, cfg, verbose, plot_level):
    wave_fishes = []
    pulse_fishes = []
    # XXX we should read this from the meta data:
    times = os.path.splitext(os.path.basename(files[0]))[0].split('-')[1]
    tstart = dt.datetime.strptime(times, '%Y%m%dT%H%M%S')
    toffs = tstart
    t1 = tstart
    for file in files:
        with DataLoader(file) as sf:
            sys.stdout.write(file + ': ')
            best_window_size = cfg.value('bestWindowSize')
            ndata = int(best_window_size * sf.samplerate)
            step = ndata//2
            b, a = butter(1, 10.0, 'hp', fs=sf.samplerate, output='ba')
            thresh = 0.0015  # XXX parameter
            stds = []
            channel = 0
            for k, data in enumerate(sf.blocks(ndata, step)):
                sys.stdout.write('.')
                sys.stdout.flush()
                t0 = toffs + dt.timedelta(seconds=k*step/sf.samplerate)
                t1 = t0 + dt.timedelta(seconds=ndata/sf.samplerate)
                fdata = lfilter(b, a, data[:,channel] - np.mean(data[:ndata//20,channel]))
                sd = np.std(fdata)
                stds.append(sd)
                if sd > thresh:
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
                        pulse_fish = None
                        wave_fish = None
                        fish_deltaf = 100000.0
                        if props['type'] == 'wave':
                            for k, fish in enumerate(wave_fishes):
                                deltaf = np.abs(fish.EODf - props['EODf'])
                                if deltaf < fish_deltaf:
                                    fish_deltaf = deltaf
                                    wave_fish = fish
                            if fish_deltaf > 2.0: # XXX Parameter
                                wave_fish = None
                            fish = wave_fish
                        else:
                            fish_dist = 10000.0
                            for k, fish in enumerate(pulse_fishes):
                                ddist = np.abs(fish.props['dist'] -
                                               props['dist'])
                                if ddist < fish_dist:
                                    fish_dist = ddist
                                    fish_deltaf = np.abs(fish.EODf -
                                                         props['EODf'])
                                    pulse_fish = fish
                            if fish_dist > 0.00005 or fish_deltaf > 10.0: # XXX Parameter
                                pulse_fish = None
                            fish = pulse_fish
                        if fish is not None:
                            # XXX we should have a maximum temporal distance
                            # XXX we need to know where the largest amplitude was!
                            fish.EODf = props['EODf']
                            fish.t1 = t1
                            fish.times.append((t0, t1))
                            if props['p-p-amplitude'] > fish.props['p-p-amplitude']:
                                fish.props = props
                                fish.waveform = eod
                                fish.spec = spec
                                fish.peaks = peaks
                        else:
                            new_fish = SimpleNamespace(props=props,
                                                       waveform=eod,
                                                       spec=spec, peaks=peaks,
                                                       EODf=props['EODf'],
                                                       t0=t0, t1=t1,
                                                       times=[(t0, t1)])
                            if props['type'] == 'pulse':
                                pulse_fishes.append(new_fish)
                            else:
                                wave_fishes.append(new_fish)
                            if first_fish:
                                sys.stdout.write('\n  ')
                                first_fish = False
                            sys.stdout.write('%6.1fHz %5s-fish @ %s\n  ' %
                                             (props['EODf'], props['type'],
                                              t0.strftime('%Y-%m-%dT%H:%M:%S')))
            toffs += dt.timedelta(seconds=len(sf)/sf.samplerate)
            sys.stdout.write('\n')
            sys.stdout.flush()
    return pulse_fishes, wave_fishes, tstart, toffs


def plot_eod_occurances(pulse_fishes, wave_fishes, tstart, toffs):
    n = len(pulse_fishes) + len(wave_fishes)
    h = n*2.5
    fig, axs = plt.subplots(n, 2, figsize=(16/2.54, h/2.54),
                            gridspec_kw=dict(width_ratios=(1,2)))
    fig.subplots_adjust(left=0.02, right=0.97, top=1-0.2/h, bottom=2.5/h)
    pi = 0
    pulse_fishes = [pulse_fishes[i] for i in
                    np.argsort([fish.props['EODf'] for fish in pulse_fishes])]
    wave_fishes = [wave_fishes[i] for i in
                   np.argsort([fish.props['EODf'] for fish in wave_fishes])]
    for ax, fish in zip(axs, pulse_fishes + wave_fishes):
        # EOD waveform:
        time = 1000.0 * fish.waveform[:,0]
        #ax[0].plot([time[0], time[-1]], [0.0, 0.0],
        #           zorder=-10, lw=1, color='#AAAAAA')
        ax[0].plot(time, fish.waveform[:,1],
                   zorder=10, lw=2, color='#C02717')
        ax[0].text(0.0, 1.0, '%.1fHz' % fish.props['EODf'],
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
            ax[0].text(tmax-0.5, -0.13, '1 ms', transform=trans, ha='center', va='top')
        ax[0].spines['left'].set_visible(False)
        ax[0].spines['right'].set_visible(False)
        ax[0].spines['top'].set_visible(False)
        ax[0].spines['bottom'].set_visible(False)
        ax[0].xaxis.set_visible(False)
        ax[0].yaxis.set_visible(False)
        # time bar:
        ax[1].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%dT%H:%M'))
        for time in fish.times:
            ax[1].plot(time, [1, 1], lw=5, color='#2060A7')
        ax[1].set_xlim(tstart, toffs)
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
    fig.savefig('plot.pdf')
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
    parser.add_argument('-s', dest='save_data', action='store_true',
                        help='save analysis results to files')
    parser.add_argument('-f', dest='format', default='auto', type=str,
                        choices=TableData.formats + ['py'],
                        help='file format used for saving analysis results, defaults to the format specified in the configuration file or "dat"')
    parser.add_argument('-p', dest='save_plot', action='store_true',
                        help='plot previously analyzed data')
    parser.add_argument('-o', dest='outpath', default='.', type=str,
                        help='path where to store results and figures (defaults to current working directory)')
    parser.add_argument('-k', dest='keep_path', action='store_true',
                        help='keep path of input file when saving analysis files, i.e. append path of input file to OUTPATH')
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
    if not os.path.exists(args.outpath):
        if verbose > 1:
            print('mkdir %s' % args.outpath)
        os.makedirs(args.outpath)
    # analyze data:
    if args.save_data and args.save_plot:
        pulse_fishes, wave_fishes, tstart, toffs = \
            extract_eods(args.file, cfg, verbose, plot_level)
        plot_eod_occurances(pulse_fishes, wave_fishes, tstart, toffs)
    else:
        if args.save_data:
            extract_eods(args.file, cfg, verbose, plot_level)
        """
        if args.save_plot:
            pulse_fishes, wave_fishes, tstart, toffs = load_data()
            plot_eod_occurances(pulse_fishes, wave_fishes, tstart, toffs)
        """

if __name__ == '__main__':
    main()
