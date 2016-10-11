"""
Functions to track wave-type electric fish frequencies over longer periods of time.

fish_tracker(): main function which performs all steps including loading data, fish tracking and -sorting and more.
"""
import sys
import numpy as np
import matplotlib.pyplot as plt
import thunderfish.dataloader as dl
import thunderfish.powerspectrum as ps
import thunderfish.harmonicgroups as hg
import thunderfish.config_tools as ct


def first_level_fish_sorting(all_fundamentals, audio_file, all_times, max_time_tolerance=5., freq_tolerance = 2., save_original_fishes=False):
    """
    Sorts fundamental frequencies of wave-type electric fish detected at certain timestamps to fishes.

    There is an array of fundamental frequencies for every timestamp (all_fundamentals). Each of these frequencies is
    compared to the last frequency of already detected fishes (last_fish_fundamentals). If the frequency difference
    between the new frequency and one or multiple already detected fishes the frequency is appended to the array
    containing all frequencies of the fish (fishes) that has been absent for the shortest period of time. If the
    frequency doesn't fit to one fish, a new fish array is created. If a fish has not been detected at one time-step
    a NaN is added to this fish array.

    The result is for each fish a array containing frequencies or nans with the same length than the time array
    (all_times). These fish arrays can be saved as .npy file to access the code after the time demanding step.

    :param all_fundamentals: (list) containing arrays with the fundamentals frequencies of fishes detected at a certain time.
    :param audio_file: (string) filepath of the analysed audiofile.
    :param all_times: (array) containing time stamps of frequency detection. (  len(all_times) == len(fishes[xy])  )
    :param max_time_tolerance: (int) time in minutes from when a certain fish is no longer tracked.
    :param freq_tolerance: (float) maximum frequency difference to assign a frequency to a certain fish.
    :param save_original_fishes: (boolean) if True saves the sorted fishes after the first level of fish sorting.
    :return fishes: (list) containing arrays of sorted fish frequencies. Each array represents one fish.
    """
    # ToDo: insert clean up function that deletes fishes with less than 5 detections after every hour...
    detection_time_diff = np.median(np.diff(all_times))
    dpm = 60. / detection_time_diff  # detections per minutes

    fishes = [[ 0. ]]
    last_fish_fundamentals = [ 0. ]
    end_nans = [0]

    # for every list of fundamentals ...
    print len(all_fundamentals)
    for t_list in range(len(all_fundamentals)):
        # ... first add a nan to all fishes. replace later !!!
        for fish in fishes:
            fish.append(np.nan)

        for idx in range(len(all_fundamentals[t_list])):
            diff = abs(np.asarray(last_fish_fundamentals) - all_fundamentals[t_list][idx])
            sorted_diff_idx = np.argsort(diff)
            tollerated_diff_idx = sorted_diff_idx[diff[sorted_diff_idx] < freq_tolerance]

            last_detect_of_tollerated = np.array(end_nans)[tollerated_diff_idx]

            if len(tollerated_diff_idx) == 0:
                fishes.append([np.nan for i in range(len(fishes[0]) - 1)])
                fishes[-1].append(all_fundamentals[t_list][idx])
                last_fish_fundamentals.append(all_fundamentals[t_list][idx])
                end_nans.append(0)
            else:
                for i in tollerated_diff_idx[np.argsort(last_detect_of_tollerated)]:
                # for i in tollerated_diff_idx:
                    if np.isnan(fishes[i][-1]):
                        fishes[i][-1] = all_fundamentals[t_list][idx]
                        last_fish_fundamentals[i] = all_fundamentals[t_list][idx]
                        end_nans[i] = 0
                        break
                    if i == tollerated_diff_idx[-1]:
                        fishes.append([np.nan for i in range(len(fishes[0]) - 1)])
                        fishes[-1].append(all_fundamentals[t_list][idx])
                        last_fish_fundamentals.append(all_fundamentals[t_list][idx])
                        end_nans.append(0)

        for fish in range(len(end_nans)):
            if end_nans[fish] >= max_time_tolerance * dpm:
                last_fish_fundamentals[fish] = 0.

        for fish in range(len(fishes)):
            if np.isnan(fishes[fish][-1]):
                end_nans[fish] += 1


    # reshape everything to arrays
    for fish in range(len(fishes)):
        fishes[fish].pop(0)
        fishes[fish] = np.asarray(fishes[fish])
    # remove first fish because it has been used for the first comparison !
    fishes.pop(0)

    if save_original_fishes:
        filename = audio_file.split('/')[-1].split('.')[-2]
        np.save('fishes_'+ filename + '.npy', np.asarray(fishes))

    return fishes


def combine_fishes(fishes, all_times, max_time_tolerance = 5., max_freq_tolerance= 10.):
    """
    Combine fishes when frequency and time of occurrence don't differ above the threshold.

    Extracts for every fish array the time of appearance and disappearance. When two appear shortly after each other and
    the mean frequency of end period of the one fish and the start period of the other fish differ below the threshold
    they will be combined (This case often occurs when a fish performs a rise: the two fishes are separated at the
    rise).

    :param fishes: (array) containing arrays of sorted fish frequencies. Each array represents one fish.
    :param all_times: (array) containing time stamps of frequency detection. (  len(all_times) == len(fishes[xy])  )
    :param max_time_tolerance: (int) maximum time between one fish disappears and another fish appears to try to combine them.
    :param max_freq_tolerance: (int) maximal frequency difference of two fishes to combine these.
    :return fishes: (array) containing arrays of sorted fish frequencies. Each array represents one fish.
    """
    detection_time_diff = np.median(np.diff(all_times))
    dpm = 60. / detection_time_diff  # detections per minutes

    occure_idx = []
    delete_idx = []
    for fish in range(len(fishes)):
        non_nan_idx = np.arange(len(fishes[fish]))[~np.isnan(fishes[fish])]
        first_and_last_idx = np.array([non_nan_idx[0], non_nan_idx[-1]])
        occure_idx.append(first_and_last_idx)

    fish_occure_order = np.arange(len(fishes))[np.argsort(np.asarray(occure_idx)[:,0])]

    for fish in reversed(fish_occure_order):
        help_idx = np.where(fish_occure_order==fish)[0][0]
        for comp_fish in reversed(fish_occure_order[:help_idx]):
            if occure_idx[fish][0] > occure_idx[comp_fish][1] and occure_idx[fish][0] - occure_idx[comp_fish][1] <= max_time_tolerance * dpm:

                # ToDO: replace index with time ... 5 min ?!
                if abs(np.mean(fishes[fish][~np.isnan(fishes[fish])][:200]) - np.mean(fishes[comp_fish][~np.isnan(fishes[comp_fish])][-200:])) <= max_freq_tolerance:

                    fishes[comp_fish][np.isnan(fishes[comp_fish])] = fishes[fish][np.isnan(fishes[comp_fish])]
                    delete_idx.append(fish)

                    occure_idx[comp_fish][1] = occure_idx[fish][1]
                    occure_idx.pop(fish)
                    break

    return_idx = np.setdiff1d(np.arange(len(fishes)), np.array(delete_idx))

    return fishes[return_idx]


def exclude_fishes(fishes, all_times, min_occure_time = 1.):
    """
    Delete fishes that are present for a to short period of time.

    :param fishes: (list) containing arrays of sorted fish frequencies. Each array represents one fish.
    :param all_times: (array) containing time stamps of frequency detection. (  len(all_times) == len(fishes[xy])  )
    :param min_occure_time (int) minimum duration a fish has to be available to not get excluded.
    :return fishes: (list) containing arrays of sorted fish frequencies. Each array represents one fish.
    """
    # ToDo: maybe delete only fishes that have less than 10 datapoints in the beginning. otherwise parts of rises may be lost.
    keep_idx = []
    detection_time_diff = np.median(np.diff(all_times))
    dpm = 60. / detection_time_diff # detections per minute

    for fish in range(len(fishes)):
        if len(fishes[fish][~np.isnan(fishes[fish])]) >= min_occure_time * dpm:
            keep_idx.append(fish)

    return np.asarray(fishes)[keep_idx]


def regress_combine(fishes, all_times, max_time_tolerance= 45., max_freq_tolerance = 2.):
    """
    Combine fishes when the linear regression of on fish fits to the first data-point of another fish.

    For every fish a regression is processed. If the first detection of another fish that appears later on first the
    regression and the time difference is below threshold the two fishes are combined.

    :param fishes: (list) containing arrays of sorted fish frequencies. Each array represents one fish.
    :param all_times: (array) containing time stamps of frequency detection. (  len(all_times) == len(fishes[xy])  )
    :param max_time_tolerance: (int) maximum time between one fish disappears and another fish appears to try to combine them.
    :param max_freq_tolerance: (float) maximum frequency difference between the first detection of one fish compared with a regression of another fish.
    :return: fishes: (list) containing arrays of sorted fish frequencies. Each array represents one fish.
    """
    detection_time_diff = np.median(np.diff(all_times))
    dpm = 60. / detection_time_diff  # detections per minutes

    delete_idx = []
    occure_idx = []
    for fish in range(len(fishes)):
        non_nan_idx = np.arange(len(fishes[fish]))[~np.isnan(fishes[fish])]
        first_and_last_idx = np.array([non_nan_idx[0], non_nan_idx[-1]])
        occure_idx.append(first_and_last_idx)

    for fish in reversed(range(len(fishes))):
        for comp_fish in reversed(range(fish)):

            if occure_idx[fish][0] > occure_idx[comp_fish][1]:
                if (occure_idx[fish][0] - occure_idx[comp_fish][1]) <= max_time_tolerance * dpm:

                    comp_fish_snippet = fishes[comp_fish][occure_idx[comp_fish][1]-int(30*dpm): occure_idx[comp_fish][1]]
                    comp_fish_snippet_time = all_times[occure_idx[comp_fish][1] - int(30 * dpm): occure_idx[comp_fish][1]]

                    comp_regress = np.polyfit(comp_fish_snippet_time[~np.isnan(comp_fish_snippet)],
                                              comp_fish_snippet[~np.isnan(comp_fish_snippet)], 1)

                    if abs((all_times[occure_idx[fish][0]] * comp_regress[0] + comp_regress[1]) - (fishes[fish][occure_idx[fish][0]])) <= max_freq_tolerance:
                        fishes[comp_fish][np.isnan(fishes[comp_fish])] = fishes[fish][np.isnan(fishes[comp_fish])]
                        delete_idx.append(fish)

                        occure_idx[comp_fish][1] = occure_idx[fish][1]
                        occure_idx.pop(fish)
                        break

    return_idx = np.setdiff1d(np.arange(len(fishes)), np.array(delete_idx))

    return fishes[return_idx]


def plot_fishes(fishes, all_times):
    """
    Plot shows the detected fish fundamental frequencies plotted against the time in hours.

    :param fishes: (list) containing arrays of sorted fish frequencies. Each array represents one fish.
    :param all_times: (array) containing time stamps of frequency detection. (  len(all_times) == len(fishes[xy])  )
    """
    fig, ax = plt.subplots(facecolor='white')
    for fish in fishes:
        ax.plot(all_times[~np.isnan(fish)] / 3600., fish[~np.isnan(fish)], color=np.random.rand(3, 1), marker='.')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [h]')
    plt.show()


def fish_tracker(audio_file, data_snippet_secs = 60., nffts_per_psd = 4, start_time= 0, end_time = None, plot_data_func=None,
                 save_original_fishes=False):
    """
    Performs the steps to analyse long-term recordings of wave-type weakly electric fish including frequency analysis,
    fish tracking and more.

    In small data snippets spectrograms and power-spectra are calculated. With the help of the power-spectra harmonic
    groups and therefore electric fish fundamental frequencies can be detected. These fundamental frequencies are
    detected for every time-step throughout the whole file. Afterwards the fundamental frequencies get assigned to
    different fishes.

    :param audio_file: (string) filepath of the analysed audiofile.
    :param data_snippet_secs: (float) duration of data snipped processed at once in seconds. Necessary because of memory issues.
    :param nffts_per_psd: (int) amount of nffts used to calculate one psd.
    :param start_time: (int) time in seconds when the analysis shall begin.
    :param end_time: (int) time in seconds when the analysis shall end.
    :param plot_data_func: (function) if plot_data_func = plot_fishes creates a plot of the sorted fishes.
    :param save_original_fishes: (boolean) if True saves the sorted fishes after the first level of fish sorting.
    """
    all_fundamentals = []
    all_times = np.array([])

    # load data
    cfg = ct.get_config_dict()
    data = dl.open_data(audio_file, 0, 60.0, 10.0)
    samplrate = data.samplerate

    while start_time < int((len(data)-data_snippet_secs*samplrate) / samplrate):
        tmp_data = data[start_time*samplrate : (start_time+data_snippet_secs)*samplrate] # gaps between snippets !!!!

        # spectrogram
        spectrum, freqs, time = ps.spectrogram(tmp_data, samplrate, fresolution=0.5, overlap_frac=.9)  # nfft window = 2 sec

        all_times = np.concatenate((all_times, time + start_time))

        # psd and fish fundamentals frequency detection
        for t in range(len(time)-(nffts_per_psd)):
            power = np.mean(spectrum[:, t:t+nffts_per_psd], axis=1)

            fishlist = hg.harmonic_groups(freqs, power, cfg)[0]

            if not fishlist == []:
                fundamentals = hg.extract_fundamental_freqs(fishlist)
                all_fundamentals.append(fundamentals)
            else:
                all_fundamentals.append(np.array([]))

        if (int(start_time) % int(data_snippet_secs * 30)) > -1 and (int(start_time) % int(data_snippet_secs * 30)) < 1:
            print('Minute')
            print start_time / 60

        start_time += data_snippet_secs
        if end_time:
            if start_time >= end_time:
                print('End time reached!')
                break

    print('sorting fishes')
    fishes = first_level_fish_sorting(all_fundamentals, audio_file, all_times, save_original_fishes=save_original_fishes)

    print('exclude fishes')
    fishes = exclude_fishes(fishes, all_times)

    print('combining fishes')
    fishes = combine_fishes(fishes, all_times)
    #
    print('combining fishes based on regression')
    fishes = regress_combine(fishes, all_times)

    print('%.0f fishes left' % len(fishes))

    if plot_data_func:
        plot_data_func(fishes, all_times)
    print('Whole file processed.')

if __name__ == '__main__':
    if len(sys.argv) == 1:
        print('Tracks fundamental freuqnecies of wave-type weakly electric fish.')
        print('')
        print('Usage:')
        print('  python tracker.py <audio_file> [start_time] [end_time]')
        print('  -> start- and endtime (in minutes) can be used to only analyse parts of a audio file.')
        print('')
        print('or:')
        print('  python tracker.py <npy_file>')
        print('  -> loads the numpy file containing the fishes after the first, time demanding, sorting step.')
        quit()

    if sys.argv[1].split('.')[-1] == 'npy':
        a = np.load(sys.argv[1], mmap_mode='r+')

        fishes = a.copy()

        all_times = np.arange(len(fishes[0])) * 0.2

        print('excluding fishes')
        fishes = exclude_fishes(fishes, all_times)

        print('combining fishes')
        fishes = combine_fishes(fishes, all_times)

        print('combining fishes based on regression')
        fishes = regress_combine(fishes, all_times)

        print('%.0f fishes left' % len(fishes))

        print('plotting')
        plot_fishes(fishes, all_times)

    else:
        audio_file = sys.argv[1]
        if len(sys.argv) == 4:
            start_time = float(sys.argv[2]) * 60
            end_time = float(sys.argv[3]) * 60
            fish_tracker(audio_file, start_time=start_time, end_time=end_time, plot_data_func= plot_fishes)
        else:
            fish_tracker(audio_file, plot_data_func= plot_fishes, save_original_fishes=True) # whole file
