"""
# thunderlogger

Detect segments of interest in large data files and analyze them for
EOD waveforms.

"""

import sys
import os
import glob
import argparse
import traceback
import numpy as np
from scipy.signal import butter, lfilter
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from multiprocessing import Pool, freeze_support, cpu_count
from .version import __version__, __year__
from .configfile import ConfigFile
from .dataloader import DataLoader
from .tabledata import TableData
from .thunderfish import configuration, detect_eods, remove_eod_files, save_eods, plot_eods


def thunderlogger(data, samplerate, unit, filename, cfg, log_freq=0.0,
                  save_data=False, all_eods=False, spec_plots='auto',
                  save_plot=False, multi_pdf=None, save_subplots='',
                  output_folder='.', keep_path=False,
                  verbose=0, plot_level=0):
    # file names:
    fn = filename if keep_path else os.path.basename(filename)
    outfilename = os.path.splitext(fn)[0]
    # clipping:
    # XXX TODO:
    clipped = False
    min_clip = -1.0
    max_clip = 1.0
    # detect EODs in the data:
    psd_data, wave_eodfs, wave_indices, eod_props, \
    mean_eods, spec_data, peak_data, power_thresh, skip_reason, zoom_window = \
      detect_eods(data, samplerate, clipped, min_clip, max_clip, filename,
                  verbose, plot_level, cfg)

    # warning message in case no fish has been found:
    if not eod_props :
        msg = ', '.join(skip_reason)
        if msg:
            print(filename + ': no fish found: %s' % msg)
        else:
            print(filename + ': no fish found.')

    # save results to files:
    output_basename = os.path.join(output_folder, outfilename)
    if save_data:
        remove_eod_files(output_basename, verbose, cfg)
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
        fig = plot_eods(outfilename, data, samplerate, 0, len(data), clipped,
                        psd_data[0], wave_eodfs, wave_indices, mean_eods, eod_props,
                        peak_data, spec_data, None, unit, zoom_window, n_snippets,
                        power_thresh, True, all_eods, spec_plots, log_freq, min_freq, max_freq,
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
                                  data, samplerate, 0, len(data), clipped, psd_data[0],
                                  wave_eodfs, wave_indices, mean_eods, eod_props,
                                  peak_data, spec_data, unit, zoom_window, n_snippets,
                                  power_thresh, True, log_freq, min_freq, max_freq)
        elif not save_data:
            fig.canvas.set_window_title('thunderlogger')
            plt.show()


pool_args = None

def run_thunderlogger(data):
    """
    Helper function for mutlithreading Pool().map().
    """
    verbose = pool_args[-2]
    try:
        msg = thunderlogger(data, *pool_args)
        if msg:
            print(msg)
    except (KeyboardInterrupt, SystemExit):
        print('\nthunderlogger interrupted by user... exit now.')
        sys.exit(0)
    except:
        print(traceback.format_exc())


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
    parser.add_argument('-a', dest='all_eods', action='store_true',
                        help='plot all EOD waveforms')
    parser.add_argument('-S', dest='spec_plots', action='store_true',
                        help='plot spectra for all EOD waveforms')
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
    verbose = args.verbose
    plot_level = args.plot_level
    if verbose < plot_level+1:
        verbose = plot_level+1

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
    spec_plots = 'auto'
    if args.spec_plots:
        spec_plots = True
    if len(args.save_subplots) > 0:
        args.save_plot = True
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
    # find segments of interest:
    with DataLoader(args.file[0]) as sf:
        global pool_args
        pool_args = (sf.samplerate, sf.unit, args.file[0], cfg, args.log_freq, args.save_data,
                     args.all_eods, spec_plots, args.save_plot, multi_pdf, args.save_subplots,
                     args.outpath, args.keep_path, verbose-1, plot_level)
        best_window_size = cfg.value('bestWindowSize')
        ndata = int(best_window_size * sf.samplerate)
        b, a = butter(1, 10.0, 'hp', fs=sf.samplerate, output='ba')
        thresh = 0.0015
        stds = []
        channel = 1
        for data in sf.blocks(ndata, ndata//2):
            fdata = lfilter(b, a, data[:,channel] - np.mean(data[:ndata//20,channel]))
            sd = np.std(fdata)
            stds.append(sd)
            if sd > thresh:
                list(map(run_thunderlogger, [data[:,channel]]))
        #plt.plot(stds)
        #plt.show()
        """
        # run on pool:
        if args.jobs is not None and (args.save_data or args.save_plot) and len(args.file) > 1:
            cpus = cpu_count() if args.jobs == 0 else args.jobs
            if verbose > 1:
                print('run on %d cpus' % cpus)
            p = Pool(cpus)
            p.map(run_thunderlogger, args.file)
        else:
            list(map(run_thunderlogger, args.file))

    if multi_pdf is not None:
        multi_pdf.close()
        """

if __name__ == '__main__':
    freeze_support()  # needed by multiprocessing for some weired windows stuff
    main()
