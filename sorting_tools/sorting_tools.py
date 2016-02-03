import numpy as np
import matplotlib.pyplot as plt
import os


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


def wave_or_pulse_psd(power, freqs, data, rate, fresolution, create_dataset=False):
    if create_dataset is True:
        fig, ax = plt.subplots()
        plt.axis([0, 3000, -110, -30])
        ax.plot(freqs, 10.0 * np.log10(power))
    freq_steps = int(125 / fresolution)

    proportions = []
    mean_powers = []
    # embed()
    for i in np.arange((3000 / fresolution) // freq_steps):  # does all of this till the frequency of 3k Hz
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

        ### difference psd (idea by Till)
        for j in np.arange(len(power_db)):
            if power_db[j] > power_db_p25 and power_db[j] < power_db_p75:
                power_db_p25_75.append(power_db[j])
        mean_powers.append(np.mean(power_db_p25_75))

        if create_dataset is True:
            ax.plot(freqs[i * freq_steps + freq_steps / 2], np.mean(power_db_p25_75), 'o', color='r')
    mean_powers_p10 = np.percentile(mean_powers, 10)
    mean_powers_p90 = np.percentile(mean_powers, 90)
    diff = abs(mean_powers_p90 - mean_powers_p10)
    ###

    ### proportion sound trace
    rate = float(rate)
    time = np.arange(len(data)) / rate
    trace_proportions = []
    for i in np.arange(len(data) // (4.0 * rate)):
        temp_trace_data = data[i * rate / 4.0:i * rate / 4.0 + rate / 4.0]
        temp_trace_data_p1 = np.percentile(temp_trace_data, 1)
        temp_trace_data_p25 = np.percentile(temp_trace_data, 25)
        temp_trace_data_p75 = np.percentile(temp_trace_data, 75)
        temp_trace_data_p99 = np.percentile(temp_trace_data, 99)
        trace_proportions.append(
            (temp_trace_data_p75 - temp_trace_data_p25) / (temp_trace_data_p99 - temp_trace_data_p1))
    if np.mean(proportions) < 0.25:
        psd_type = 'wave'
    else:
        psd_type = 'pulse'

    # if diff < 15 and np.mean(proportions) < 0.25:
    #     psd_type = 'wave'
    # elif diff > 15 and np.mean(proportions) > 0.25:
    #     psd_type = 'pulse'
    # else:
    #     psd_type = 'unknown'

    #####################################################################################
    # Creating dataset #

    if create_dataset is True:
        plt.draw()
        plt.pause(1)

        response = raw_input('wave- or pulse psd ? [w/p]')
        plt.close()
        if response == "w":
            file = "wave_diffs.npy"
            file2 = "wave_proportions.npy"
            file3 = "wave_trace_proportions.npy"
        elif response == "p":
            file = "pulse_diffs.npy"
            file2 = "pulse_proportions.npy"
            file3 = "pulse_trace_proportions.npy"
        else:
            quit()

        if not os.path.exists(file):
            np.save(file, np.array([]))
        all_diffs = np.load(file)
        all_diffs = all_diffs.tolist()
        all_diffs.append(diff)
        all_diffs = np.asarray(all_diffs)
        np.save(file, all_diffs)

        if not os.path.exists(file2):
            np.save(file2, np.array([]))
        all_propertions = np.load(file2)
        all_propertions = all_propertions.tolist()
        all_propertions.append(np.mean(proportions))
        all_propertions = np.asarray(all_propertions)
        np.save(file2, all_propertions)

        if not os.path.exists(file3):
            np.save(file3, np.array([]))
        all_trace_proportions = np.load(file3)
        all_trace_proportions = all_trace_proportions.tolist()
        all_trace_proportions.append(np.mean(trace_proportions))
        all_trace_proportions = np.asarray(all_trace_proportions)
        np.save(file3, all_trace_proportions)
    return psd_type


def save_fundamentals(fishlist):
    mean_path = 'fish_wave.npy'
    if not os.path.exists(mean_path):
        np.save(mean_path, np.array([]))
    fundamentals = np.load(mean_path)
    fundamentals = fundamentals.tolist()

    for fish in np.arange(len(fishlist)):
        fundamentals.append(fishlist[fish][0][0])

    fundamentals = np.asarray(fundamentals)
    np.save(mean_path, fundamentals)

    print 'current fundamental frequencies are: ', fundamentals