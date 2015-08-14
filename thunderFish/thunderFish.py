#!/usr/bin/python
"""
Created on Fri March 21 14:40:04 2014

@author: Juan Felipe Sehuanes
"""

import getopt
import re
import os
import sys
import glob
import wave
from scipy.signal import butter
import numpy as np
import numpy.ma as ma
from matplotlib import mlab
import matplotlib.pyplot as plt
import odml
from FishRecording import *
from IPython import embed

if __name__ == '__main__':

    ##### Create Parent Folders and Sub-folders #####

         # -t , --threshold      : tau for the kernel (default: 25)

    helptxt = """
        Welcome to thunderFISH!

        thunderFish.py [optional arguments] recording_location

        -h              : display help
        -e , --experimenter      : (Optional) Type the name of the person who recorded the files
        """
    if len(sys.argv) < 2:
        print helptxt
        print """
        Please type the name of the location your recordings come from
        (e.g. River_name) as the last or only argument.
        """

        sys.exit(2)
    try:
        opts, args = getopt.getopt(sys.argv[1:-1], "he:", ["experimenter="])
    except:
        print helptxt
        sys.exit(2)

    # threshold = 50.
    new_dir = '../thunderFISH_analysis_from_' + str(sys.argv[-1])

    experimenter = None

    for opt, arg in opts:
        if opt == '-h':
            print helptxt
            sys.exit()
        elif opt in ("-e", "--experimenter"):
            experimenter = arg

    if not os.path.exists(new_dir):
        os.makedirs(new_dir)
    ##-------------------------------------------------##
    ##-------------------------------------------------##

    s = odml.Section(str(sys.argv[-1]), "location")
    if experimenter is not None:
        s.append(odml.Property("Experimenter", experimenter))

    ##-------------------------------------------------##
    ##-------------------------------------------------##

    types = ('*.wav', '*.WAV', '*.MP3', '*.mp3')
    files = reduce(lambda a, b: a+b, [glob.glob(pattern) for pattern in types])

    for file_no, curr_file in enumerate(sorted(files)):

        try:  # ACHTUNG!! This try is VERY dangerous and only serves to print the error message at the end!
            base, ext = os.path.splitext(curr_file)
            new_mod_filename = base + '_mod.wav'
            os.system('avconv -i {0:s} -ac 1 -y -acodec pcm_s16le {1:s}'.format(curr_file, new_dir + '/' + new_mod_filename))

            # pcm_s16le means: pcm format, signed integer, 16-bit, "little endian"

        ##-------------------------------------------------##
        ##-------------------------------------------------##

        ##### Loop through modified files and calculate spectogram #####

            file_no += 1
            fish = FishRecording(new_dir + '/' + new_mod_filename)
            fish_file_number = re.findall(r'\d+\_', new_mod_filename)[0][:-1]
            print 5 * '============'
            if fish._time[-1] > 5.:  # This if clause skips recordings of lesser length than 10 seconds.
                pass
            else:
                print '%%% File No. ' + fish_file_number\
                      + ' was not analyzed, because the length of the recording was less than 10 seconds. %%%'
                continue

            print '\n'
            print 'Analyzing Fish No. ' + fish_file_number + ' ...'
            current_fish = "fish" + fish_file_number
            fund_freq = fish.fund_freq
            s.append(odml.Section(current_fish, "subject"))
            s[current_fish].append(odml.Property("Audio_file", new_mod_filename))
            s[current_fish].append(odml.Property("EOD_frequency", odml.Value(fund_freq, unit="Hz")))
            print 'Fundamental Frequency of Fish No.' + fish_file_number + ' is %.1f Hz...' % fund_freq
            print '\n'

            w_start, w_width = fish.detect_best_window()

            pt, _, _, _ = fish.w_pt  # pt stands for peak-troughs...
            if len(pt) <= 5:
                print '%%%%% Warning! No or few peaks detected! %%%%%'
                continue

            b_win_inxs = [i for i, elem in enumerate(fish._time == w_start, 1) if elem]
            inx = b_win_inxs[0]  # Get inx where the analyze window starts.
            max_inx = inx + int(fish._sample_rate / 2.)  # Get inx where the analyze window ends.
            window_edges = (fish._time[inx], fish._time[max_inx])
            bwin_time = fish._time[inx:max_inx]
            bwin_eod = fish._eod[inx:max_inx]

            fish_type = fish.type_detector()
            print 'Fish No. ' + fish_file_number + ' is ' + fish_type + ' type!'
            s[current_fish].append(odml.Property("Fish_type", fish_type))

            fig = plt.figure(num='Fish # ' + fish_file_number, figsize=(11, 7))
            ax1 = fig.add_subplot(2, 2, 1)
            ax2 = fig.add_subplot(2, 2, 2)
            ax3 = fig.add_subplot(2, 1, 2)

            lowp_filter = 4  # This number multiplied by the fundamental frequency of the fish
                             # sets the low pass filter (not valid for pulse fishes!).

            fish.plot_eodform(ax=ax1, filtr=lowp_filter)
            fish.plot_spectogram(ax=ax2)
            fish.plot_wavenvelope(ax=ax3, win_edges=window_edges)

            plt.savefig(new_dir + '/' + new_mod_filename[:-7] + '.pdf')
            s[current_fish].append(odml.Property("Figure_file", new_mod_filename[:-7] + '.pdf'))

        except:
            print """

            Sorry, that shouldn't have happened! There seems to be a problem with the recorded file.
            Make sure to get a more or less clean recording of 30 seconds!
            Proceeding with the next recording ...

            """
    doc = odml.Document()
    doc.append(s)
    # doc.author("Juan Sehuanes")
    writer = odml.tools.xmlparser.XMLWriter(doc)

    writer.write_file(new_dir + '/' + 'odml_file_from_' + str(sys.argv[-1]) + '.xml')
    print '\n Your analyze is done! Thank you for using thunderFish!'
