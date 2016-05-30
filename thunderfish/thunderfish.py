import numpy as np
import os
import argparse
import FishTracker as FT
import FishRecording as FR
import Auxiliary as aux
import sorting_tools as st
import config_tools as ct
import audioread
from IPython import embed
import matplotlib.pyplot as plt
#import seaborn as sns
import Auxiliary as aux


def main(audio_file, channel=0, output_folder='.' + os.path.sep + 'analysis_output', verbose=None, beat_plot= False):
    # get config dictionary
    cfg = ct.get_config_dict()

    if verbose is not None:
        cfg['verboseLevel'][0] = verbose
    channel = channel

    with audioread.audio_open(audio_file) as af:
        tracen = af.channels
        if channel >= tracen:
            print('number of traces in file is', tracen)
            quit()
        ft = FT.FishTracker(audio_file.split(os.path.sep)[-1], af.samplerate)
        index = 0

        data = ft.get_data()

        for buffer in af:
            fulldata = np.fromstring(buffer, dtype='<i2').reshape(-1, af.channels)
            n = fulldata.shape[0]
            if index + n > len(data):
                if index == 0:
                    print("panic!!!! I need a larger buffer!")
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
            print("file too short !!!")
            exit()

        # best window algorithm
        mod_file = aux.conv_to_single_ch_audio(audio_file)
        Fish = FR.FishRecording(mod_file)
        bwin, win_width = Fish.detect_best_window()

        print ('\nbest window is between: %.2f' % bwin, '& %.2f' % (bwin + win_width), 'seconds.\n')

        os.remove(mod_file)

        # fish_type algorithm
        fish_type, r_value = Fish.type_detector()
        print('current fish is a ' + fish_type + '-fish')

        # data process: creation of fish-lists containing frequencies, power of fundamentals and harmonics of all fish

        if index > 0:
            power_fres1, freqs_fres1, psd_type, fish_type,\
            fishlist, mean_proportions = ft.processdata(data[:index] / 2.0 ** 15, fish_type, bwin, win_width, config_dict=cfg)

        #####################################################
        # collect data for mat meth figure:
        # bw_data = data[(bwin * af.samplerate):(bwin * af.samplerate + win_width * af.samplerate)]
        # np.save('pulse_trace_data.npy', bw_data)

        if beat_plot:
            # designed for parama_data/20140519_Rioanita/EN099.wav
            beat_data = data[(1.2 * af.samplerate): (1.9 * af.samplerate)] / 2.0 ** 15
            beat_time = np.arange(len(beat_data)) * 1.0 / af.samplerate
            aux.beat_plot(beat_data, beat_time)
            embed()
            quit()

        #####################################################

        # Pulse analysis
        pulse_data = []
        pulse_freq = []
        if psd_type == 'pulse' or fish_type == 'pulse':
            print('')
            print('try to create MEAN PULSE-EOD')
            print('')
            pulse_data, pulse_freq = ft.pulse_sorting(bwin, win_width, data[:index] / 2.0 ** 15, fish_type, psd_type)

        wave_data = []
        if fish_type == 'wave' and len(fishlist) == 1:
            print('')
            print('try to create MEAN WAVE-EOD')
            print('')
            wave_data = ft.wave_sorting(bwin, win_width, data[:index] / 2.0 ** 15)

        # create EOD plots
        out_folder = aux.create_outp_folder(audio_file, output_folder)
        ft.bw_psd_and_eod_plot(power_fres1, freqs_fres1, bwin, win_width, data[:index] / 2.0 ** 15, psd_type, fish_type,
                               fishlist, pulse_data, pulse_freq, out_folder, mean_proportions, r_value, wave_data)

        # saves fundamentals of all wave fish !!!
        st.save_fundamentals(fishlist, out_folder)

        print('\nAnalysis completed! .npy arrays located in %s\n' %out_folder)


if __name__ == '__main__':
    # command line arguments:
    parser = argparse.ArgumentParser(
        description='Display waveform, spectrogram, and power spectrum of time series data.',
        epilog='by bendalab (2015/2016)')
    parser.add_argument('--version', action='version', version='1.0')
    parser.add_argument('-v', action='count', dest='verbose')
    parser.add_argument('file', nargs='?', default='', type=str, help='name of the file wih the time series data')
    parser.add_argument('channel', nargs='?', default=0, type=int, help='channel to be displayed')
    parser.add_argument('output_folder', nargs='?', default=".", type=str, help="location to store results, figures")
    args = parser.parse_args()

    main(args.file, args.channel, args.output_folder, args.verbose)
