"""
Detection of chirps in weakly electric fish recordings.

- `chirp_analysis()`: calculates spectrogram, detects fishes and extracts chirp times(combined for all fishes).
                  !!! recommended for short recordings (up to 5 min) where only the chirp times shall be extracted !!!
- `chirp_detection()`: extracts chirp times with help of given spectrogram and fishlist.
"""

import numpy as np
from .harmonics import harmonic_groups
from .powerspectrum import spectrogram
from .eventdetection import std_threshold, detect_peaks, trim_to_peak
import matplotlib.pyplot as plt


def true_chirp_power_drop(chirp_time_idx, power, power_window=100):
    """
    Chirp is only accepted as such if the power of the frequency drops down as expected.

    :param chirp_time_idx: (array) indices of chirps.
    :param power: (array) power array containing for each timestamp the max value in power of a certain frequency range.
    :param power_window (int) datapoints arroung a detected chirp used to verify that there is a chirp.
    :return: chirp_time_idx: (array) indices of chirps that have been confirmed to be chirps.
    """

    true_chirp_time_idx = []

    for i in range(len(chirp_time_idx)):
        idx0 = int(chirp_time_idx[i] - power_window/2)
        if idx0 < 0:
            idx0 = 0
        idx1 = int(chirp_time_idx[i] + power_window/2)
        if idx1 > len(power):
            idx1 = len(power)

        tmp_median = np.median(power[idx0:idx1])
        tmp_std = np.std(power[idx0:idx1], ddof=1)

        if np.min(power[idx0:idx1]) < tmp_median - 3*tmp_std:
            true_chirp_time_idx.append(chirp_time_idx[i])

    return np.array(true_chirp_time_idx)


def true_chirp_power_rise_above(chirp_time_idx, power_above):

    median_power_above = np.median(power_above)
    std_power_above = np.std(power_above, ddof=1)

    if median_power_above > 0.001:
        print('another fish disturbs the chirp approval! Have to rely on other algorithms.')
        return chirp_time_idx
    else:
        true_chirp_time_idx = []

        for i in range(len(chirp_time_idx)):
            if power_above[int(chirp_time_idx[i])] > median_power_above + 3*std_power_above:
                true_chirp_time_idx.append(chirp_time_idx[i])

        return true_chirp_time_idx


def chirp_detection(spectrum, freqs, time, fishlist=None, fundamentals=None, min_power= 0.005, freq_tolerance=1., chirp_th=1.,
                    plot_data_func=None):
    """
    Detects chirps on the basis of a spectrogram.

    :param spectrum: (2d-array) spectrum calulated with the numpy.spectrogram function.
    :param freqs: (array) frequencies of the spectrum.
    :param time: (array) time of the nffts used in the spectrum.
    :param fishlist: (array) power und frequncy for each fundamental/harmonic of a detected fish.
                     fishlist[fish][harmonic][frequency, power]
    :param min_power: (float) minimum power of the fundamental frequency for each fish to participate in chirp detection.
    :param freq_tolerance: (float) frequency tolerance in the spectrum to detect the power of a certain frequency.
    :param chirp_th: (float) minimum chirp duration to be accepted as a chirp.
    :param plot_data: (bool) If True: plots the process of chirp detection.
    :return:chirp_time: (array) array of times (in sec) where chirps have been detected.
    """
    if not hasattr(fundamentals, '__len__'):
        if not hasattr(fishlist, '__len__'):
            print('fishlist or fundamentals missing as argument !!!')
            quit()
        else:
            fundamentals = []
            for fish in fishlist:
                if fish[0][1] > min_power:
                    fundamentals.append(fish[0][0])

    chirp_time = np.array([])
    chirp_freq = np.array([])

    for enu, fundamental in enumerate(fundamentals):
        # extract power of only the part of the spectrum that has to be analysied for each fundamental and get the peak
        # power of every piont in time.
        power = np.max(spectrum[(freqs >= fundamental - freq_tolerance) & (freqs <= fundamental + freq_tolerance)], axis=0)
        power_above = np.max(spectrum[(freqs >= fundamental + 50.0 -freq_tolerance) & (freqs <= fundamental + 50.0 + freq_tolerance)], axis=0)
        #power = np.mean(spectrum[(freqs >= fundamental - freq_tolerance) & (freqs <= fundamental + freq_tolerance)], axis=0)
        # calculate the slope by calculating the difference in the power
        power_diff = np.diff(power)

        # peak detection in the power_diff to detect drops in power indicating chrips
        threshold = std_threshold(power_diff)
        peaks, troughs = detect_peaks(power_diff, threshold)
        troughs, peaks = trim_to_peak(troughs, peaks) # reversed troughs and peaks in output and input to get trim_to_troughs

        # exclude peaks and troughs with to much time diff to be a chirp
        # ToDO: not nice !!!
        peaks = peaks[(troughs - peaks) < chirp_th]
        troughs = troughs[(troughs - peaks) < chirp_th]

        if len(troughs) > 0:
            # chirps times defined as the mean time between the troughs and peaks
            chirp_time_idx = np.mean([troughs, peaks], axis=0)

            # exclude detected chirps if the powervalue doesn't drop far enought
            chirp_time_idx = true_chirp_power_drop(chirp_time_idx, power)

            # chirp_time_idx = true_chirp_power_rise_above(chirp_time_idx, power_above)
            # add times of detected chirps to the list.
            chirp_time = np.concatenate((chirp_time, np.array([time[int(i)] for i in chirp_time_idx])))
            chirp_freq = np.concatenate((chirp_freq, np.array(fundamental* np.ones(len(chirp_time_idx)))))

        else:
            chirp_time = np.array([])
            chirp_freq = np.array([])

        if plot_data_func:
            plot_data_func(enu, chirp_time, time, power, power_above, power_diff, fundamental)

    return chirp_time, chirp_freq


def chirp_detection_plot(enu, chirp_time, time, power, power2, power_diff, fundamental):
    """
    plots the process of chirp detection.

    :param enu: (int) indication which fish in list is processed.
    :param chirp_time: (array) timestamps when chirps have been detected.
    :param time: (array) time array.
    :param power: (array) power of a certain frequency band.
    :param power_diff: (array) slope of the power array.
    :param fundamental: (float) fundamental frequency around which the algorithm looked for chirps.
    """
    try:
        ax
    except NameError:
        fig, ax = plt.subplots()
        colors = ['r', 'g', 'k', 'blue', 'r', 'g', 'k', 'blue', 'r', 'g', 'k', 'blue', 'r', 'g', 'k', 'blue']
    # if enu == 0:
    #     fig, ax = plt.subplots()
    #     colors = ['r', 'g', 'k', 'blue', 'r', 'g', 'k', 'blue', 'r', 'g', 'k', 'blue', 'r', 'g', 'k', 'blue']
    ax.plot(chirp_time, np.zeros(len(chirp_time)), 'o', markersize=10, color=colors[enu], alpha=0.8, label='chirps')
    ax.set_xlabel('time in sec')
    ax.set_ylabel('power')

    ax.plot(time, power, colors[enu], marker='.', label='%.1f Hz' % fundamental)
    ax.plot(time, power2, colors[enu+1], label='%.1f Hz' % (fundamental+50.0))
    ax.plot(time[:len(power_diff)], power_diff, colors[enu], label='slope')
    ax.legend(loc='upper right', bbox_to_anchor=(1, 1), frameon=False)


def chirp_analysis(data, samplerate):
    """
    Performs all steps to detect chirps in a given dataset. This includes spectrogram calculation, fish detection and
    analysing of specific frequency bands.
    For further documentation see functions chirp_spectrogram() and chirp_detection().
    !!! recommended for short recordings (up to 5 min) where only the chirp times shall be extracted !!!

    :param data: (array) data.
    :param samplerate: (float) smaplerate of the data.
    :param min_power: (float) minimal power of the fish fundamental to include this fish in chirp detection.
    """
    spectrum, freqs, time = spectrogram(data, samplerate, freq_resolution=2., overlap_frac=0.95)

    power = np.mean(spectrum, axis=1) # spectrum[:, t0:t1] to only let spectrum of certain time....

    fishlist = harmonic_groups(freqs, power)[0]

    chirp_time, chirp_freq = chirp_detection(spectrum, freqs, time, fishlist, plot_data_func=chirp_detection_plot)

    plt.show()

    return chirp_time, chirp_freq


if __name__ == '__main__':
    ###
    # If you want to test the code I propose to use the file '60427L05.WAV' of the transect
    # '2016_04_27__downstream_stonewall_at_pool' made in colombia, 2016.
    ###
    import sys
    import matplotlib.pyplot as plt
    from .dataloader import load_data

    data_file = sys.argv[1]
    raw_data, samplerate, unit = load_data(data_file, 0)

    chirp_time, chirp_freq = chirp_analysis(raw_data, samplerate)

    # power = np.mean(spectrum[:, t:t + nffts_per_psd], axis=1)
