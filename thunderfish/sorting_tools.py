import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as sps
import os
from IPython import embed


def filter_fishes(fishlists):
    ###################################################################
    # extract fishes which are consistent for different resolutions
    fundamentals = [[] for i in fishlists]
    for i in np.arange(len(fishlists)):
        for j in np.arange(len(fishlists[i])):
            fundamentals[i].append(fishlists[i][j][0][0])
    consistent_fish_help = [1 for k in fundamentals[0]]

    for i in np.arange(len(fundamentals[0])):
        for j in np.arange(len(fundamentals) - 1):
            for k in np.arange(len(fundamentals[j + 1])):
                if fundamentals[0][i] < fundamentals[j + 1][k] + 2 and fundamentals[0][i] > fundamentals[j + 1][k] - 2:
                    consistent_fish_help[i] += 1
    index_of_valid_fish = []
    for i in np.arange(len(consistent_fish_help)):
        if consistent_fish_help[i] == len(fishlists):
            index_of_valid_fish.append(i)

    fishlist = []
    for i in index_of_valid_fish:
        fishlist.append(fishlists[0][i])

    return fishlist


def wave_or_pulse_psd(power, freqs, data, rate, fresolution, create_dataset=False, category='wave'):
    freq_steps = int(125 / fresolution)

    proportions = []
    mean_powers = []
    # embed()
    # power_db = 10.0 * np.log10(power)
    for i in np.arange((1500 / fresolution) // freq_steps):  # does all of this till the frequency of 3k Hz
        power_db = 10.0 * np.log10(power[i * freq_steps:i * freq_steps + freq_steps])
        power_db_p25_75 = []
        power_db_p99 = np.percentile(power_db, 99)
        power_db_p1 = np.percentile(power_db, 1)
        power_db_p25 = np.percentile(power_db, 25)
        power_db_p75 = np.percentile(power_db, 75)

        ### proportion psd (idea by Jan)
        proportions.append((power_db_p75 - power_db_p25) / (
            power_db_p99 - power_db_p1))  # value between 0 and 1; pulse psds have much bigger values than wave psds
        ###
    mean_proportions = np.mean(proportions)
    if np.mean(proportions) < 0.27:
        psd_type = 'wave'
    else:
        psd_type = 'pulse'

    if create_dataset:
        if category is 'wave':
            if not os.path.exists('wave_PSD_algor.npy'):
                np.save('wave_PSD_algor.npy', np.array([]))
            wave_prop = np.load('wave_PSD_algor.npy')
            wave_prop = wave_prop.tolist()
            wave_prop.append(np.mean(proportions))
            wave_prop = np.asarray(wave_prop)
            np.save('wave_PSD_algor.npy', wave_prop)

        elif category is 'pulse':
            if not os.path.exists('pulse_PSD_algor.npy'):
                np.save('pulse_PSD_algor.npy', np.array([]))
            pulse_prop = np.load('pulse_PSD_algor.npy')
            pulse_prop = pulse_prop.tolist()
            pulse_prop.append(np.mean(proportions))
            pulse_prop = np.asarray(pulse_prop)
            np.save('pulse_PSD_algor.npy', pulse_prop)

        else:
            print("unknown fish category: %s" % category)
            quit()

    return psd_type, mean_proportions


def save_fundamentals(fishlist, output_folder):

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    file_name = output_folder + 'fish_wave.npy'
    if not os.path.exists(file_name):
        np.save(file_name, np.array([]))
    fundamentals = np.load(file_name)
    fundamentals = fundamentals.tolist()

    for fish in np.arange(len(fishlist)):
        fundamentals.append(fishlist[fish][0][0])

    fundamentals = np.asarray(fundamentals)
    np.save(file_name, fundamentals)

    print('current fundamental frequencies are: ', fundamentals)