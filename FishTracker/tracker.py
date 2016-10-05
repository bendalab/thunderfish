"""
Functions to track wave-type electric fish over longer periods of time.

fish_tracker(): main function which performs all steps including loading data, fish tracking and -sorting, chirp detection and more.
"""
import sys
import numpy as np
import matplotlib.pyplot as plt
from IPython import embed
import thunderfish.dataloader as dl
import thunderfish.powerspectrum as ps
import thunderfish.harmonicgroups as hg
import thunderfish.config_tools as ct
import thunderfish.chirp as cp


def frequencies_for_chirp_detection(all_fundamentals, spectrum, freqs):
    """

    :param all_fundamentals: (list) containing arrays with the fundamentals frequencies of fishes detected at a certain time.
    :param spectrum: (array) spectrum created by the matplotlib.mlab.spectogram function.
    :param freqs: (array) frequency array created by the matplotlib.mlab.spectogram function.
    :return test_fundamentals: (array) most common and powerful frequencies in the spectrogram.
    """
    all_fundamentals_array = np.concatenate((all_fundamentals))
    hist_data = np.histogram(all_fundamentals_array, bins= np.arange(min(all_fundamentals_array), max(all_fundamentals_array)+2., 2.))

    # find most common fundamentals
    th_n = sum(hist_data[0]) * 0.01
    test_fundamentals = hist_data[1][hist_data[0] > th_n]
    test_fundamentals = test_fundamentals[ np.concatenate((np.array([True]), np.diff(test_fundamentals) > 5.)) ]

    # check if most common fundamentals have enough power for chirp- (and rises-) detection
    # if not --> exclude !
    for i in reversed(range(len(test_fundamentals))):
        if np.median(np.max(spectrum[(freqs > test_fundamentals[i] - 5.) & (freqs < test_fundamentals[i] + 5.)],
                            axis=0)) < 0.005:
            test_fundamentals = np.delete(test_fundamentals, i)

    return test_fundamentals


def first_level_fish_sorting(all_fundamentals, audio_file, all_times, save_original_fishes=False):
    """

    :param all_fundamentals: (list) containing arrays with the fundamentals frequencies of fishes detected at a certain time.
    :param audio_file: (string) filepath of the analysed audiofile.
    :param save_original_fishes: (boolean) if True saves the sorted fishes after the first level of fish sorting.
    :return fishes: (list) containing arrays of sorted fish frequencies. Each array represents one fish.
    """

    detection_time_diff = np.median(np.diff(all_times))
    dpfm = 5 * 60 / detection_time_diff  # detections per five minutes

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
            tollerated_diff_idx = sorted_diff_idx[diff[sorted_diff_idx] < 2.]

            last_detect_of_tollerated = np.array(end_nans)[tollerated_diff_idx]
            # last_detect_of_tollerated = np.array([(len(fishes[fish]) - np.arange(len(fishes[fish]))[~np.isnan(fishes[fish])][-1]) for fish in tollerated_diff_idx])

            if len(tollerated_diff_idx) == 0:
                # print 'new fish'
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
                    # if i == tollerated_diff_idx[np.argsort(last_detect_of_tollerated)][-1]:
                    if i == tollerated_diff_idx[-1]:
                        fishes.append([np.nan for i in range(len(fishes[0]) - 1)])
                        fishes[-1].append(all_fundamentals[t_list][idx])
                        last_fish_fundamentals.append(all_fundamentals[t_list][idx])
                        end_nans.append(0)

        for fish in range(len(end_nans)):
            if end_nans[fish] >= dpfm:
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
        # del a, filename

    return fishes


def combine_fishes(fishes, all_times):
    """

    :param fishes: (list) containing arrays of sorted fish frequencies. Each array represents one fish.
    :return fishes: (list) containing arrays of sorted fish frequencies. Each array represents one fish.
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
            if occure_idx[fish][0] > occure_idx[comp_fish][1] and occure_idx[fish][0] - occure_idx[comp_fish][1] <= dpm / 2.:

                fish_slice = fishes[fish][occure_idx[fish][0] : occure_idx[fish][0] + int(np.floor(dpm))]

                comp_fish_slice = fishes[comp_fish][occure_idx[comp_fish][0] : occure_idx[comp_fish][0] + int(np.floor(dpm))]

                if abs(np.mean(fishes[fish][~np.isnan(fishes[fish])][-200:]) - np.mean(fishes[comp_fish][~np.isnan(fishes[comp_fish])][-200:])) <= 20:
                    fishes[comp_fish][np.isnan(fishes[comp_fish])] = fishes[fish][np.isnan(fishes[comp_fish])]
                    delete_idx.append(fish)

                    occure_idx[comp_fish][1] = occure_idx[fish][1]
                    occure_idx.pop(fish)
                    break

    return_idx = np.setdiff1d(np.arange(len(fishes)), np.array(delete_idx))

    return fishes[return_idx]
    # means = np.array([np.mean(fishes[i][~np.isnan(fishes[i])]) for i in range(len(fishes))])
    #
    # for i in reversed(range(len(fishes))):
    #     mean_diff = abs(means - means[i])
    #     compare_idx = np.arange(len(fishes))[np.argsort(mean_diff)]
    #
    #     for j in compare_idx:
    #         if abs(np.mean(fishes[i][~np.isnan(fishes[i])]) - np.mean(fishes[j][~np.isnan(fishes[j])])) < 5.:
    #             if len(fishes[i]) != len(fishes[j]):
    #                 print("ERROR !!! fishes don't have same length")
    #                 break
    #             test = fishes[i] + fishes[j]
    #             if len(test[~np.isnan(test)]) <= 5:
    #                 fishes[j][np.isnan(fishes[j])] = fishes[i][np.isnan(fishes[j])]
    #                 means[j] = np.mean(fishes[j][~np.isnan(fishes[j])])
    #                 # fishes.pop(i)
    #                 # fishes = np.delete(fishes, i)
    #                 means = np.delete(means, i)
    #                 break
    # return fishes

def delete_single_detections(fishes):
    """

    :param fishes: (list) containing arrays of sorted fish frequencies. Each array represents one fish.
    :return fishes: (list) containing arrays of sorted fish frequencies. Each array represents one fish.
    """
    # for fish in fishes:
    #     detect_idx = np.arange(len(fish))[~np.isnan(fish)]
    #
    #     for idx in detect_idx:
    #         if len(fish[idx-25:idx+25][~np.isnan(fish[idx-25:idx+25])]) < len(fish[idx-25:idx+25]) / 4:
    #             fish[idx] = np.nan
    # return fishes
    for fish in reversed(range(len(fishes))):
        detect_idx = np.arange(len(fishes[fish]))[~np.isnan(fishes[fish])]

        for idx in detect_idx:
            if len(fishes[fish][idx-25:idx+25][~np.isnan(fishes[fish][idx-25:idx+25])]) < len(fishes[fish][idx-25:idx+25]) / 10:
                fishes[fish][idx] = np.nan
        if len(fishes[fish][~np.isnan(fishes[fish])]) <= 2:
            fishes.pop(fish)
    return fishes


def exclude_fishes(fishes, all_times):
    """

    :param fishes: (list) containing arrays of sorted fish frequencies. Each array represents one fish.
    :return fishes: (list) containing arrays of sorted fish frequencies. Each array represents one fish.
    """
    keep_idx = []
    detection_time_diff = np.median(np.diff(all_times))
    dpm = 60 / detection_time_diff # detections per minute

    for fish in range(len(fishes)):
        if len(fishes[fish][~np.isnan(fishes[fish])]) >= dpm:
            keep_idx.append(fish)

    return fishes[keep_idx]


def cut_fishes(fishes, all_times):
    """

    :param fishes: (list) containing arrays of sorted fish frequencies. Each array represents one fish.
    :param all_times: (array) containing time stamps of frequency detection. (  len(all_times) == len(fishes[xy])  )
    :return fishes: (list) containing arrays of sorted fish frequencies. Each array represents one fish.
    """
    detection_time_diff = np.median(np.diff(all_times))
    dptm = 10 * 60 / detection_time_diff   # detections per ten minutes

    for fish in reversed(range(len(fishes))):
        non_nan_idx = np.arange(len(fishes[fish]))[~np.isnan(fishes[fish])]
        nans_between_non_nans= np.diff(non_nan_idx)-1

        split_idx = non_nan_idx[nans_between_non_nans >= dptm] + 1
        split_fish = np.split(fishes[fish], split_idx)

        help_idx = 0
        for new_fish in split_fish:
            tmp_fish = np.array([np.nan for i in range(len(fishes[fish]))])
            tmp_fish[help_idx:help_idx+len(new_fish)] = new_fish
            fishes.append(tmp_fish)
            help_idx += len(new_fish)
        fishes.pop(fish)

    return fishes


def recombine_fishes(fishes, all_times):
    """

    :param fishes:(list) containing arrays of sorted fish frequencies. Each array represents one fish.
    :return fishes:(list) containing arrays of sorted fish frequencies. Each array represents one fish.
    """
    occur_idx = []
    detection_time_diff = np.median(np.diff(all_times))
    dpfm = 5 * 60 / detection_time_diff   # detections per five minutes

    for fish in range(len(fishes)):
        non_nan_idx = np.arange(len(fishes[fish]))[~np.isnan(fishes[fish])]
        first_and_last_idx = [non_nan_idx[0], non_nan_idx[-1]]
        occur_idx.append(first_and_last_idx)

    for fish in reversed(range(len(fishes))):
        for compare_fish in reversed(range(fish)):
            combinable = False
            if occur_idx[fish][0] > occur_idx[compare_fish][0] and occur_idx[fish][1] < occur_idx[compare_fish][1]:
                combinable = True
                if occur_idx[fish][1] - occur_idx[fish][0] >= dpfm:
                    if len(fishes[compare_fish][occur_idx[fish][0] : occur_idx[fish][1]][~np.isnan(fishes[compare_fish][occur_idx[fish][0] : occur_idx[fish][1]])]) == 0:
                        combinable = False
            elif occur_idx[compare_fish][0] > occur_idx[fish][0] and occur_idx[compare_fish][1] < occur_idx[fish][1]:
                combinable = True
                if occur_idx[compare_fish][1] - occur_idx[compare_fish][0] >= dpfm:
                    if len(fishes[fish][occur_idx[compare_fish][0] : occur_idx[compare_fish][1]][~np.isnan(fishes[fish][occur_idx[compare_fish][0] : occur_idx[compare_fish][1]])]) == 0:
                        combinable = False
            elif occur_idx[fish][0] < occur_idx[compare_fish][1] and occur_idx[fish][1] > occur_idx[compare_fish][1]:
                combinable = True
            elif occur_idx[compare_fish][0] < occur_idx[fish][1] and occur_idx[compare_fish][1] > occur_idx[fish][1]:
                combinable = True

            if combinable:
                if abs(np.mean(fishes[fish][~np.isnan(fishes[fish])]) - np.mean(fishes[compare_fish][~np.isnan(fishes[compare_fish])])) <= 20:
                    test = fishes[fish] + fishes[compare_fish]
                    if len(test[~np.isnan(test)]) <= 5:
                        # print (fish, compare_fish)
                        fishes[compare_fish][np.isnan(fishes[compare_fish])] = fishes[fish][np.isnan(fishes[compare_fish])]
                        occur_idx[compare_fish][0] = min([occur_idx[fish][0], occur_idx[compare_fish][0]])
                        occur_idx[compare_fish][1] = max([occur_idx[fish][1], occur_idx[compare_fish][1]])

                        fishes.pop(fish)
                        occur_idx.pop(fish)
                        break

    return fishes


# def regress_combine(fishes, all_times):


def plot_fishes(fishes, all_times):
    """

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
    all_chirp_time = np.array([])
    all_chirp_freq = np.array([])

    # load data
    cfg = ct.get_config_dict()
    data = dl.open_data(audio_file, 0, 60.0, 10.0)
    samplrate = data.samplerate

    while start_time < int((len(data)-data_snippet_secs*samplrate) / samplrate):
        tmp_data = data[start_time*samplrate : (start_time+data_snippet_secs)*samplrate] # gaps between snippets !!!!

        # spectrogram
        spectrum, freqs, time = ps.spectrogram(tmp_data, samplrate, fresolution=0.5, overlap_frac=.9)  # nfft window = 2 sec
        # spectrum, freqs, time = ps.spectrogram(tmp_data, samplrate, fresolution=2., overlap_frac=.95)
        # fish_detect_time_diff = 0.5  # time steps in sec for psd ...
        # nfft_steps = int(np.floor(fish_detect_time_diff / np.median(np.diff(time))))  # ... transformed into index
        #
        # all_times = np.concatenate((all_times, time[::nfft_steps]+start_time))
        all_times = np.concatenate((all_times, time + start_time))

        # psd and fish fundamentals frequency detection
        # for t in range(len(time)-(nffts_per_psd-1))[::nfft_steps]:
        for t in range(len(time)-(nffts_per_psd)):
            power = np.mean(spectrum[:, t:t+nffts_per_psd], axis=1)

            fishlist = hg.harmonic_groups(freqs, power, cfg)[0]

            if not fishlist == []:
                fundamentals = hg.extract_fundamental_freqs(fishlist)
                all_fundamentals.append(fundamentals)
            else:
                all_fundamentals.append(np.array([]))

        # chirp detection
        # test_fundamentals = frequencies_for_chirp_detection(all_fundamentals, spectrum, freqs)
        #
        # chirp_time, chirp_freq = cp.chirp_detection(spectrum, freqs, time, fundamentals=test_fundamentals,
        #                                             freq_tolerance=4.)
        #
        ## Compiling output
        # if len(chirp_time) > 0:
        #     print('--- CHIRP ---', '%.1f sec' % (chirp_time + start_time), '%.1f Hz')
        #     all_chirp_time = np.concatenate((all_chirp_time, chirp_time + start_time))
        #     all_chirp_freq = np.concatenate((all_chirp_freq, chirp_freq))

        if (int(start_time) % int(data_snippet_secs * 30)) > -1 and (int(start_time) % int(data_snippet_secs * 30)) < 1:
            print('Minute')
            print start_time / 60

        start_time += data_snippet_secs
        if end_time:
            if start_time >= end_time:
                print('End time reached!')
                break

    print('sorting fishes')
    fishes = first_level_fish_sorting(all_fundamentals, audio_file, all_times, save_original_fishes)

    print('exclude fishes')
    fishes = exclude_fishes(fishes, all_times)

    print('combining fishes')
    fishes = combine_fishes(fishes, all_times)
    #
    # print('detele single detections')
    # fishes = delete_single_detections(fishes)

    # print('cutting fishes')
    # fishes = cut_fishes(fishes, all_times)
    #
    # print('recombining fishes')
    # fishes = recombine_fishes(fishes, all_times)

    # print('exclude fishes')
    # fishes = exclude_fishes(fishes, all_times)

    if plot_data_func:
        plot_data_func(fishes, all_times)
    print('Whole file processed.')

if __name__ == '__main__':

    if sys.argv[1].split('.')[-1] == 'npy':
        a = np.load(sys.argv[1], mmap_mode='r+')

        fishes = a.copy()

        all_times = np.arange(len(fishes[0])) * 0.2
        print('excluding fishes')
        fishes = exclude_fishes(fishes, all_times)

        print('combining fishes')
        fishes = combine_fishes(fishes, all_times)

        # print('delete single detections')
        # fishes = delete_single_detections(fishes)

        # print('cutting fishes')
        # fishes = cut_fishes(fishes, all_times)
        #
        # print('recombining fishes')
        # fishes = recombine_fishes(fishes, all_times)

        # print('excluding fishes')
        # fishes = exclude_fishes(fishes, all_times)

        # ToDO: maybe create linear regression for each fish using np.polyfit(x, y, 1) and combine !!!

        # fishes = regress_combine(fishes, all_times)

        print('plotting')
        plot_fishes(fishes, all_times)

    else:
        audio_file = sys.argv[1]
        if len(sys.argv) == 4:
            start_time = float(sys.argv[2]) * 60
            end_time = float(sys.argv[3]) * 60
            fish_tracker(audio_file, start_time=start_time, end_time=end_time, plot_data_func= plot_fishes) # whole file
        else:
            fish_tracker(audio_file, plot_data_func= plot_fishes, save_original_fishes=True) # whole file
            # fish_tracker(audio_file, start_time=18000, end_time=36000, plot_data_func= plot_fishes) # 5h: 5-10
            # fish_tracker(audio_file, start_time=18000, end_time=28800, plot_data_func= plot_fishes) # 2h: 5-8
            # fish_tracker(audio_file, start_time=25200, end_time=28800, plot_data_func= plot_fishes) # 2h: 7-8
            # fish_tracker(audio_file, start_time=18000, end_time=21600, plot_data_func= plot_fishes) # 1h: 5-6
