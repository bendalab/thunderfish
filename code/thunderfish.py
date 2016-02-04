import numpy as np
import sys
import os
import argparse
import FishTracker as FT
import FishRecording as FR
import load_tools as lt
import Auxiliary as aux
import sorting_tools as st
import config_tools as ct
from IPython import embed


def main():
    # config file name:
    progs = sys.argv[0].split('/')
    cfgfile = progs[-1].split('.')[0] + '.cfg'

    # command line arguments:
    parser = argparse.ArgumentParser(
        description='Display waveform, spectrogram, and power spectrum of time series data.',
        epilog='by Jan Benda (2015)')
    parser.add_argument('--version', action='version', version='1.0')
    parser.add_argument('-v', action='count', dest='verbose')
    parser.add_argument('file', nargs='?', default='', type=str, help='name of the file wih the time series data')
    parser.add_argument('npy_savefile_wave', nargs='?', default='', type=str)
    parser.add_argument('npy_savefile_puls', nargs='?', default='', type=str)
    parser.add_argument('channel', nargs='?', default=0, type=int, help='channel to be displayed')
    args = parser.parse_args()

    # get config dictionary
    cfg = ct.get_config_dict()

    # load configuration from the current directory:
    if os.path.isfile(cfgfile):
        print('load configuration ' + cfgfile)
        lt.load_config(cfgfile, cfg)

    # load configuration files from higher directories:
    filepath = args.file
    absfilepath = os.path.abspath(filepath)
    dirs = os.path.dirname(absfilepath).split('/')
    dirs.append('')
    maxlevel = len(dirs) - 1
    if maxlevel > 3:
        maxlevel = 3
    for k in xrange(maxlevel, 0, -1):
        path = '/'.join(dirs[:-k]) + '/' + cfgfile
        if os.path.isfile(path):
            print 'load configuration', path
            lt.load_config(path, cfg)

    # set configuration from command line:
    if args.verbose != None:
        cfg['verboseLevel'][0] = args.verbose

    channel = args.channel

    try:
        import audioread
    except ImportError:
        print 'python module "audioread" is not installed.'
        exit(2)

    with audioread.audio_open(filepath) as af:
        tracen = af.channels
        if channel >= tracen:
            print 'number of traces in file is', tracen
            quit()
        ft = FT.FishTracker(af.samplerate)
        index = 0

        data = ft.get_data()

        for buffer in af:
            fulldata = np.fromstring(buffer, dtype='<i2').reshape(-1, af.channels)
            n = fulldata.shape[0]
            if index + n > len(data):
                if index == 0:
                    print "panic!!!! I need a larger buffer!"
                # ft.processdata(data[:index] / 2.0 ** 15)
                index = 0
            if n > 0:
                data[index:index + n] = fulldata[:n, channel]
                index += n
            else:
                break

        # long file analysis
        good_file = ft.exclude_short_files(data, index)
        if good_file == False:
            print "file too short !!!"
            exit()

        # best window algorithm
        mod_file = aux.conv_to_single_ch_audio(filepath)

        Fish = FR.FishRecording(mod_file)
        bwin, win_width = Fish.detect_best_window()

        print '\nbest window is between: %.2f' % bwin, '& %.2f' % (bwin + win_width), 'seconds.\n'

        os.remove(mod_file)

        # fish_type algorithm
        fish_type = Fish.type_detector()
        print('current fish is a ' + fish_type + '-fish')

        # data process: creation of fish-lists containing frequencies, power of fundamentals and harmonics of all fish

        if index > 0:
            power_fres1, freqs_fres1, psd_type, fish_type,\
            fishlist = ft.processdata(data[:index] / 2.0 ** 15, fish_type, bwin, win_width, config_dict=cfg)

        # Pulse analysis
        pulse_data = []
        pulse_freq = []
        if psd_type == 'pulse' or fish_type == 'pulse':
            print ''
            print 'try to create MEAN PULSE-EOD'
            print ''
            pulse_data, pulse_freq = ft.pulse_sorting(bwin, win_width, data[:index] / 2.0 ** 15)

        # create EOD plots
        out_folder = aux.create_outp_folder(filepath)
        ft.bw_psd_and_eod_plot(power_fres1, freqs_fres1, bwin, win_width, data[:index] / 2.0 ** 15, psd_type, fish_type,
                               fishlist, pulse_data, pulse_freq, out_folder)

        # saves fundamentals of all wave fish !!!
        st.save_fundamentals(fishlist, out_folder)

        print('\nAnalysis completed! .npy arrays located in %s\n' %out_folder)


if __name__ == '__main__':
    main()