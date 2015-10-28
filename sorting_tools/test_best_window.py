__author__ = 'juan'

# imports

from FishRecording import *
from Auxiliary import *
import sys, os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

if __name__ == '__main__':

    if not 3 >= len(sys.argv) >= 2:  # only if len(sys.argv) == 2 or 3, the codes continues running.
        print("\nI need an audiofile as first argument. Please give me the path + filename as argument! "
              "Then you can tell me the folder name you wish me to save the figures in, as a second argument "
              "(don't use any '/'). Only of you like though.\n")
        sys.exit(2)
    # Load, convert and process the audiofile
    audio_file = sys.argv[1]
    modfile = conv_to_single_ch_audio(audio_file)
    fish = FishRecording(modfile)

    # detect best window
    w_start, w_width = fish.detect_best_window()

    # define save path and create the plots with the best window
    if len(sys.argv) == 3:
        folder_name = sys.argv[2]
        savepath = './control_figures/best_window_detection/' + folder_name + '/'
    else:
        savepath = './control_figures/best_window_detection/'

    if not os.path.exists(savepath):
        os.mkdir(savepath)

    recording_name = '_'.join(modfile.split('.')[0].split('_')[:2]) + '.pdf'

    fig, ax = plt.subplots(figsize=(12, 8))
    fish.plot_wavenvelope(ax, w_start, w_start+w_width)
    ax.set_title(recording_name, fontsize=18)
    fig.tight_layout()
    fig.savefig(savepath + recording_name)
    plt.close()
    os.remove(modfile)
    sys.exit(2)
