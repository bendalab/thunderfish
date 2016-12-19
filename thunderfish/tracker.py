"""
Functions to track wave-type electric fish frequencies over longer periods of time.

fish_tracker(): main function which performs all steps including loading data, fish tracking and -sorting and more.
"""
import sys
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from .dataloader import open_data
from .powerspectrum import spectrogram, next_power_of_two
from .harmonicgroups import harmonic_groups, fundamental_freqs
 

def long_term_recording_fundamental_extraction(data, samplerate, start_time=0.0, end_time=-1.0, data_snippet_secs=60.0,
                                               nffts_per_psd=4, fresolution=0.5, overlap_frac=.9, verbose=0, **kwargs):
    """
    For a long data array calculates spectograms of small data snippets, computes PSDs, extracts harmonic groups and
    extracts fundamental frequncies.

    :param data: (array) raw data.
    :param samplerate: (int) samplerate of data.
    :param start_time: (int) analyze data from this time on (in seconds).  XXX this should be a float!!!! Internally I would use indices.
    :param end_time: (int) stop analysis at this time (in seconds). If -1 then analyse to the end of the data. XXX this should be a float!!!! Internally I would use indices.
    :param data_snippet_secs: (float) duration of data snipped processed at once in seconds. Necessary because of memory issues.
    :param nffts_per_psd: (int) number of nffts used for calculating one psd.
    :param fresolution: (float) frequency resolution for the spectrogram.
    :param overlap_frac: (float) overlap of the nffts (0 = no overlap; 1 = total overlap).
    :param verbose: (int) with increasing value provides more output on console.
    :param kwargs: further arguments are passed on to harmonic_groups().
    :return all_fundamentals: (list) containing arrays with the fundamentals frequencies of fishes detected at a certain time.
    :return all_times: (array) containing time stamps of frequency detection. (  len(all_times) == len(fishes[xy])  )
    """
    all_fundamentals = []
    all_times = np.array([])

    if end_time < 0.0:
        end_time = len(data)/samplerate

    nfft = next_power_of_two(samplerate / fresolution)
    if len(data.shape) > 1:
        channels = range(data.shape[1])
    else:
        channels = range(1)

    while start_time < int((len(data)- data_snippet_secs*samplerate) / samplerate):
        if verbose > 2:
            print('Minute %.2f' % (start_time/60))

        for channel in channels:
            # print(channel)
            if len(channels) > 1:
                tmp_data = data[int(start_time*samplerate) : int((start_time+data_snippet_secs)*samplerate), channel]
            else:
                tmp_data = data[int(start_time*samplerate) : int((start_time+data_snippet_secs)*samplerate)]

            # spectrogram
            spectrum, freqs, time = spectrogram(tmp_data, samplerate, fresolution=fresolution, overlap_frac=overlap_frac)  # nfft window = 2 sec

            # psd and fish fundamentals frequency detection
            tmp_power = [np.array([]) for i in range(len(time)-(nffts_per_psd-1))]
            for t in range(len(time)-(nffts_per_psd-1)):
                # power = np.mean(spectrum[:, t:t+nffts_per_psd], axis=1)
                tmp_power[t] = np.mean(spectrum[:, t:t+nffts_per_psd], axis=1)
            if channel == 0:
                power = tmp_power
            else:
                for t in range(len(power)):
                    power[t] += tmp_power[t]

        all_times = np.concatenate((all_times, time[:-(nffts_per_psd-1)] + start_time))

        for p in range(len(power)):
            fishlist = harmonic_groups(freqs, power[p], **kwargs)[0]
            fundamentals = fundamental_freqs(fishlist)
            all_fundamentals.append(fundamentals)

        if (len(all_times) % ((len(time) - (nffts_per_psd-1)) * 30)) > -1 and (
                    len(all_times) % ((len(time) - (nffts_per_psd-1)) * 30)) < 1:
            if verbose >= 2:
                print('Minute %.0f' % (start_time/60))

        start_time += time[-nffts_per_psd] - (0.5 -(1-overlap_frac)) * nfft / samplerate


        if end_time > 0:
            if start_time >= end_time:
                if verbose >= 2:
                    print('End time reached!')
                break

    return all_fundamentals, all_times


def first_level_fish_sorting(all_fundamentals, base_name, all_times, max_time_tolerance=5., freq_tolerance = 1.,
                             save_original_fishes=False, verbose=0):
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
    :param base_name: (string) filename.
    :param all_times: (array) containing time stamps of frequency detection. (  len(all_times) == len(fishes[xy])  )
    :param max_time_tolerance: (int) time in minutes from when a certain fish is no longer tracked.
    :param freq_tolerance: (float) maximum frequency difference to assign a frequency to a certain fish.
    :param save_original_fishes: (boolean) if True saves the sorted fishes after the first level of fish sorting.
    :param verbose: (int) with increasing value provides more shell output.
    :return fishes: (list) containing arrays of sorted fish frequencies. Each array represents one fish.
    """
    def clean_up(fishes, last_fish_fundamentals, end_nans):
        """
        Delete fish arrays with too little data points to reduce memory usage.

        :param fishes: (list) containing arrays of sorted fish frequencies. Each array represents one fish.
        :param last_fish_fundamentals: (list) contains for every fish in fishes the last detected fundamental frequency.
        :param end_nans: (list) for every fish contains the counts of nans since the last fundamental detection.
        :return: fishes: (list) cleaned up input list.
        :return: last_fish_fundamentals: (list) cleaned up input list.
        :return: end_nans: (list) cleaned up input list.
        """
        for fish in reversed(range(len(fishes))):
            if len(np.array(fishes[fish])[~np.isnan(fishes[fish])]) <= 10:
                fishes.pop(fish)
                last_fish_fundamentals.pop(fish)
                end_nans.pop(fish)

        return fishes, last_fish_fundamentals, end_nans

    detection_time_diff = all_times[1] - all_times[0]
    dpm = 60. / detection_time_diff  # detections per minutes

    fishes = [np.full(len(all_fundamentals)+1, np.nan)]
    fishes[0][0] = 0.
    last_fish_fundamentals = [ 0. ]
    end_nans = [0]

    # for every list of fundamentals ...
    if verbose >=2:
        print('len of all_fundamentals: ', len(all_fundamentals))

    clean_up_idx = int(30 * dpm)

    # ToDo: enumerate and fundamentals in all_fundamentals
    for enu, fundamentals in enumerate(all_fundamentals):
        if enu == clean_up_idx:
            if verbose >= 2:
                print('cleaning up ...')
            fishes, last_fish_fundamentals, end_nans = clean_up(fishes, last_fish_fundamentals, end_nans)
            clean_up_idx += int(30 * dpm)

        for idx in range(len(fundamentals)):
            diff = np.abs(np.asarray(last_fish_fundamentals) - fundamentals[idx])
            sorted_diff_idx = np.argsort(diff)
            tolerated_diff_idx = sorted_diff_idx[diff[sorted_diff_idx] < freq_tolerance]

            last_detect_of_tolerated = np.array(end_nans)[tolerated_diff_idx]

            if len(tolerated_diff_idx) == 0:
                fishes.append(np.full(len(all_fundamentals)+1, np.nan))
                fishes[-1][enu+1] = fundamentals[idx]
                last_fish_fundamentals.append(fundamentals[idx])
                end_nans.append(0)
            else:
                found = False
                for i in tolerated_diff_idx[np.argsort(last_detect_of_tolerated)]:
                    if np.isnan(fishes[i][enu+1]):
                        fishes[i][enu+1] = fundamentals[idx]
                        last_fish_fundamentals[i] = fundamentals[idx]
                        end_nans[i] = 0
                        found = True
                        break
                if not found:
                    fishes.append(np.full(len(all_fundamentals)+1, np.nan))
                    fishes[-1][enu+1] = fundamentals[idx]
                    last_fish_fundamentals.append(fundamentals[idx])
                    end_nans.append(0)

        for fish in range(len(fishes)):
            if end_nans[fish] >= max_time_tolerance * dpm:
                last_fish_fundamentals[fish] = 0.

            if np.isnan(fishes[fish][enu+1]):
                end_nans[fish] += 1

    fishes, last_fish_fundamentals, end_nans = clean_up(fishes, last_fish_fundamentals, end_nans)
    # reshape everything to arrays
    for fish in range(len(fishes)):
        fishes[fish] = fishes[fish][1:]

    # if not removed be clean_up(): remove first fish because it has been used for the first comparison !
    if fishes[0][0] == 0.:
        fishes.pop(0)

    if save_original_fishes:
        print('saving')
        np.save(base_name + '-fishes.npy', np.asarray(fishes))
        np.save(base_name + '-times.npy', all_times)
    return np.asarray(fishes)


def combine_fishes(fishes, all_times, max_time_tolerance = 5., max_freq_tolerance= 5.):
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
    detection_time_diff = all_times[1] - all_times[0]
    dpm = 60. / detection_time_diff  # detections per minutes

    occure_idx = []
    delete_idx = []
    for fish in range(len(fishes)):
        non_nan_idx = np.arange(len(fishes[fish]))[~np.isnan(fishes[fish])]
        first_and_last_idx = np.array([non_nan_idx[0], non_nan_idx[-1]])
        occure_idx.append(first_and_last_idx)

    fish_occure_order = np.arange(len(fishes))[np.argsort(np.asarray(occure_idx)[:,0])]

    # for fish in reversed(fish_occure_order):
    #     help_idx = np.where(fish_occure_order==fish)[0][0]
    #     for comp_fish in reversed(fish_occure_order[:help_idx+1]):
    for fish in reversed(range(len(fishes))):
        for comp_fish in reversed(range(fish)):
            #####################################################
            combinable = False
            try:
                if occure_idx[fish][0] > occure_idx[comp_fish][1] and occure_idx[fish][0] - occure_idx[comp_fish][1] <= max_time_tolerance * dpm:
                    combinable=True
                elif occure_idx[fish][0] < occure_idx[comp_fish][1] and occure_idx[comp_fish][0] < occure_idx[fish][0] and occure_idx[comp_fish][1] < occure_idx[fish][1]:
                    combinable=True
                elif occure_idx[fish][0] > occure_idx[comp_fish][0] and occure_idx[fish][1] < occure_idx[comp_fish][1]:
                    combinable=True
            except IndexError:
                from IPython import embed
                embed()
                quit()
            if combinable:
                nantest = fishes[fish] + fishes[comp_fish]
                if len(nantest[~np.isnan(nantest)]) >= 10:
                    combinable = False

            if combinable:
                # ToDO: replace index with time ... 5 min ?! AND!!!! ... other data snippets for different cases...
                if np.abs(np.mean(fishes[fish][~np.isnan(fishes[fish])][:200]) - np.mean(fishes[comp_fish][~np.isnan(fishes[comp_fish])][-200:])) <= max_freq_tolerance:

                    fishes[comp_fish][np.isnan(fishes[comp_fish])] = fishes[fish][np.isnan(fishes[comp_fish])]
                    delete_idx.append(fish)

                    occure_idx[comp_fish][1] = occure_idx[fish][1]
                    occure_idx.pop(fish)
                    break

    return_idx = np.setdiff1d(np.arange(len(fishes)), np.array(delete_idx))

    return fishes[return_idx]


def check_frequency_consistency(fishes, all_times, f_th = 3.):
    # ToDo: Do it in bins !!!
    # for fish in reversed(range(len(fishes))):
    # print len(fishes)
    detection_time_diff = all_times[1] - all_times[0]
    dpm = 60. / detection_time_diff

    new_fishes = []
    for fish in reversed(range(len(fishes))):
        non_nan_idx = np.arange(len(fishes[fish]))[~np.isnan(fishes[fish])]

        all_t0 = np.arange((non_nan_idx[-1] - non_nan_idx[0]) // np.floor((dpm * 5) / 2) + 1) * np.floor((dpm * 5) / 2) + non_nan_idx[0]

        # mean_time = all_times[fishes[fish] > np.mean(fishes[fish][~np.isnan(fishes[fish])])][0]
        # ToDo: ab hier in for loop mit t0 ... data[t0: t1]; t1 = t0+ np.floor(dpm * 30)
        for t0 in reversed(all_t0):
            t1 = t0 + np.floor(dpm * 5)

            tmp_fish = fishes[fish][t0:t1]
            tmp_time = all_times[t0:t1]
            tmp_nnan_idx = np.arange(len(tmp_fish))[~np.isnan(tmp_fish)]

            if len(tmp_nnan_idx) > 0:
                p1 = np.percentile(tmp_fish[tmp_nnan_idx], 1)
                p99 = np.percentile(tmp_fish[tmp_nnan_idx], 99)

                cut=False
                if p99 - p1 > f_th:
                    p90 = p99 - ((p99 - p1) / 10.)
                    p10 = p1 + ((p99 - p1) / 10.)

                    time_start_rise = tmp_time[tmp_nnan_idx][tmp_fish[tmp_nnan_idx] > p90][-1]
                    # freq_start_rise = fishes[fish][fishes[fish] >= p90][-1]

                    time_end_rise = tmp_time[tmp_nnan_idx][tmp_fish[tmp_nnan_idx] < p10][0]
                    # freq_end_rise = fishes[fish][fishes[fish] <= p10][0]

                    if time_end_rise - time_start_rise <= 600 and time_end_rise - time_start_rise > 0:
                        cut = True
                    if time_start_rise - all_times[non_nan_idx[0]] < 120:
                        cut = False

                    fig, ax = plt.subplots()
                    ax.plot(tmp_time[tmp_nnan_idx], tmp_fish[tmp_nnan_idx])
                    ax.plot([tmp_time[tmp_nnan_idx[0]], tmp_time[tmp_nnan_idx[-1]]], [p99, p99], '-', color='green')
                    ax.plot([tmp_time[tmp_nnan_idx[0]], tmp_time[tmp_nnan_idx[-1]]], [p1, p1], '-', color='green')

                    if cut:
                        mid_freq = np.min(tmp_fish[tmp_nnan_idx]) + (np.max(tmp_fish[tmp_nnan_idx]) - np.min(tmp_fish[tmp_nnan_idx])) / 2.
                        tmp_cut_idx = np.arange(len(tmp_fish))[tmp_nnan_idx][tmp_fish[tmp_nnan_idx] >= mid_freq]
                        ax.plot(tmp_time[tmp_cut_idx][-1], tmp_fish[tmp_cut_idx][-1], 'o', color='red', markersize=10)
                        # ToDo: cut_idx finden vor dem der value abfaellt und t diff nicht zu hoch.

                        # cut_idx = tmp_cut_idx(np.where((np.diff(tmp_fish[tmp_cut_idx])) > 0 & (np.diff(tmp_cut_idx) < 10) )[0][-1] + 1)

                        # cut_idx = tmp_cut_idx[np.diff(tmp_fish[tmp_cut_idx]) > 0]

                        # from IPython import embed
                        # embed()
                        # quit()
                        #########################################

                        cut_idx = tmp_cut_idx[-1]

                        new_fishes.append(np.full(len(all_times), np.nan))
                        new_fishes[-1][cut_idx:] = fishes[fish][cut_idx:]

                        fishes[fish][cut_idx:] = np.full(len(fishes[fish][cut_idx:]), np.nan)
                    plt.show()

    if new_fishes == []:
        return fishes
    else:
        return np.append(fishes, new_fishes, axis=0)


def exclude_fishes(fishes, all_times, min_occure_time = 1):
    """
    Delete fishes that are present for a to short period of time.

    :param fishes: (list) containing arrays of sorted fish frequencies. Each array represents one fish.
    :param all_times: (array) containing time stamps of frequency detection. (  len(all_times) == len(fishes[xy])  )
    :param min_occure_time (int) minimum duration a fish has to be available to not get excluded.
    :return fishes: (array) containing arrays of sorted fish frequencies. Each array represents one fish.
    """
    keep_idx = []
    detection_time_diff = all_times[1] - all_times[0]
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

    :param fishes: (array) containing arrays of sorted fish frequencies. Each array represents one fish.
    :param all_times: (array) containing time stamps of frequency detection. (  len(all_times) == len(fishes[xy])  )
    :param max_time_tolerance: (int) maximum time between one fish disappears and another fish appears to try to combine them.
    :param max_freq_tolerance: (float) maximum frequency difference between the first detection of one fish compared with a regression of another fish.
    :return: fishes: (array) containing arrays of sorted fish frequencies. Each array represents one fish.
    """
    # ToDo: wenn steigung zu krass oder zu wenig datenpunkte NICHT kombinieren
    detection_time_diff = all_times[1] - all_times[0]
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
                    snippet_start_idx = occure_idx[comp_fish][1]-int(30*dpm)
                    if snippet_start_idx < 0:
                        # break
                        snippet_start_idx = 0
                    comp_fish_snippet = fishes[comp_fish][snippet_start_idx: occure_idx[comp_fish][1]]
                    comp_fish_snippet_time = all_times[snippet_start_idx: occure_idx[comp_fish][1]]

                    comp_regress = np.polyfit(comp_fish_snippet_time[~np.isnan(comp_fish_snippet)],
                                              comp_fish_snippet[~np.isnan(comp_fish_snippet)], 1)

                    if np.abs((all_times[occure_idx[fish][0]] * comp_regress[0] + comp_regress[1]) - (fishes[fish][occure_idx[fish][0]])) <= max_freq_tolerance:
                        fishes[comp_fish][np.isnan(fishes[comp_fish])] = fishes[fish][np.isnan(fishes[comp_fish])]
                        delete_idx.append(fish)

                        occure_idx[comp_fish][1] = occure_idx[fish][1]
                        occure_idx.pop(fish)
                        break

    return_idx = np.setdiff1d(np.arange(len(fishes)), np.array(delete_idx))

    return fishes[return_idx]


def plot_fishes(fishes, all_times, base_name, save_plot):
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

    if save_plot:
        # ToDo: save as pdf... import PdfPages or something....
        plt.savefig(base_name)
        plt.close(fig)
    else:
        plt.show()


def fish_tracker(data_file, start_time=0.0, end_time=-1.0, gridfile=False, save_plot=False,
                 save_original_fishes=False, data_snippet_secs = 60., nffts_per_psd = 4, verbose=0, **kwargs):
    """
    Performs the steps to analyse long-term recordings of wave-type weakly electric fish including frequency analysis,
    fish tracking and more.

    In small data snippets spectrograms and power-spectra are calculated. With the help of the power-spectra harmonic
    groups and therefore electric fish fundamental frequencies can be detected. These fundamental frequencies are
    detected for every time-step throughout the whole file. Afterwards the fundamental frequencies get assigned to
    different fishes.

    :param data_file: (string) filepath of the analysed data file.
    :param data_snippet_secs: (float) duration of data snipped processed at once in seconds. Necessary because of memory issues.
    :param nffts_per_psd: (int) amount of nffts used to calculate one psd.
    :param start_time: (int) analyze data from this time on (in seconds).  XXX this should be a float!!!!
    :param end_time: (int) stop analysis at this time (in seconds).  XXX this should be a float!!!!
    :param plot_data_func: (function) if plot_data_func = plot_fishes creates a plot of the sorted fishes.
    :param save_original_fishes: (boolean) if True saves the sorted fishes after the first level of fish sorting.
    :param kwargs: further arguments are passed on to harmonic_groups().
    """
    # ToDo: how to recognize grid file? load all channels -1 in grid; else 0
    if gridfile:
        data = open_data(data_file, -1, 60.0, 10.0)
        print('--- GRID FILE ANALYSIS ---')
        print('ALL traces are analysed')
        print('--------------------------')
    else:
        data = open_data(data_file, 0, 60.0, 10.0)
        print('--- ONE TRACE ANALYSIS ---')
        print('ONLY 1 trace is analysed')
        print('--------------------------')

    # with open_data(data_file, 0, 60.0, 10.0) as data:
    samplerate = data.samplerate
    base_name = os.path.splitext(os.path.basename(data_file))[0]

    if verbose >= 1:
        print('extract fundamentals')
    all_fundamentals, all_times = long_term_recording_fundamental_extraction(data, samplerate, start_time, end_time,
                                                                             data_snippet_secs, nffts_per_psd,
                                                                             fresolution=0.5, overlap_frac=.9,
                                                                             verbose=verbose, **kwargs)

    if verbose >= 1:
        print('sorting fishes')
    fishes = first_level_fish_sorting(all_fundamentals, base_name, all_times, save_original_fishes=save_original_fishes,
                                      verbose=verbose)

    if verbose >= 1:
        print('exclude fishes')
    fishes = exclude_fishes(fishes, all_times)
    if len(fishes) == 0:
        print('excluded all fishes. Change parameters.')
        quit()

    if verbose >= 1:
        print('combining fishes')
    fishes = combine_fishes(fishes, all_times)

    # if verbose >= 1:
    #     print('combining fishes based on regression')
    # fishes = regress_combine(fishes, all_times)

    if verbose >= 1:
        print('%.0f fishes left' % len(fishes))

    plot_fishes(fishes, all_times, base_name, save_plot)
    if verbose >= 1:
        print('Whole file processed.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Analyse long single- or multi electrode EOD recordings of weakly electric fish.',
        epilog='by bendalab (2015-2016)')
    parser.add_argument('file', nargs='?', default='', type=str, help='name of the file wih the time series data')
    parser.add_argument('start_time', nargs='?', default=0, type=int, help='start time of analysis in min.')
    parser.add_argument('end_time', nargs='?', default=-1, type=int, help='end time of analysis in min.')
    parser.add_argument('-g', dest='grid', action='store_true', help='use this argument to analysis 64 electrode grid data.')
    parser.add_argument('-p', dest='save_plot', action='store_true', help='use this argument to save output plot')
    parser.add_argument('-s', dest='save_fish', action='store_true',
                        help='use this argument to save fish EODs after first stage of sorting.')
    args = parser.parse_args()

    if len(sys.argv) == 1:
        print('Tracks fundamental freuqnecies of wave-type weakly electric fish.')
        print('')
        print('Usage:')
        print('  python[3] -m thunderfish.tracker <data_file> [start_time] [end_time] [-g] [-p] [-s]')
        print('  -> start- and endtime: (in minutes) can be used to only analyse parts of a data file.')
        print('  -> -g: the powerspectra of all channels at a time are summed up.')
        print('         can be used when there are multiple channels to analyse.')
        print('  -> -p: saves the final plot as png')
        print('  -> -s: saves the of array of array, one for every detected fish, containing their frequency at all')
        print('         time of the recording. Usefull when you analyse data for the first time to reduce compilation')
        print('         time in further runs.')
        print('')
        print('or:')
        print('  python[3] -m thunderfish.tracker <npy_file>')
        print('  -> loads the numpy file containing the fishes after the first, time demanding, sorting step.')
        print('     base_name + "-fishes.npy"')
        quit()

    if sys.argv[1].split('.')[-1] == 'npy':
        a = np.load(sys.argv[1], mmap_mode='r+')
        fishes = a.copy()

        all_times = np.load(sys.argv[1].replace('-fishes', '-times'))

        print('excluding fishes')
        fishes = exclude_fishes(fishes, all_times)
        #############################################################################################
        # NEW FUNCTION TO CHECK FREQUENCY CONSISTANCY

        # print ('')
        # print len(fishes)
        # print ('')
        # fishes = check_frequency_consistency(fishes, all_times)

        # print('excluding fishes')
        # fishes = exclude_fishes(fishes, all_times)
        # if len(fishes) == 0:
        #     print('excluded all fishes. Change parameters.')
        #     quit()

        print('combining fishes')
        fishes = combine_fishes(fishes, all_times)
        #
        # print('combining fishes based on regression')
        # fishes = regress_combine(fishes, all_times)

        print('%.0f fishes left' % len(fishes))

        base_name = os.path.splitext(os.path.basename(sys.argv[1]))[0]
        print('plotting')
        plot_fishes(fishes, all_times, base_name, args.save_plot)

    else:
        fish_tracker(args.file, args.start_time*60, args.end_time*60, args.grid, args.save_plot, args.save_fish, verbose=3)
