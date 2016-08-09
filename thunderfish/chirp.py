"""
Detection of chirps in weakly electric fish recordings.

chirp_analysis(): calculates spectrogram, detects fishes and extracts chirp times(combined for all fishes).
                  !!! recommended for short recordings (up to 5 min) where only the chirp times shall be extracted !!!
chirp_detection(): extracts chirp times with help of given spectrogram and fishlist.
chirp_spectrogram(): performs spectrogram suitable for chip detection.
"""

import sys
import numpy as np
import matplotlib.mlab as mlab
import dataloader as dl
import harmonicgroups as hg
import config_tools as ct
import powerspectrum as ps
import matplotlib.pyplot as plt
import peakdetection as pkd

def chirp_spectrogram(data, samplerate, fresolution=0.5, detrend=mlab.detrend_none, window=mlab.window_hanning, overlap_frac=0.5,
                      pad_to=None, sides='default', scale_by_freq=None):
    """
    Spectrogram of a given frequency resolution.

    :param data: (array) data for the spectrogram.
    :param samplerate: (float) samplerate of data in Hertz.
    :param fresolution: (float) frequency resolution for the spectrogram.
    :param overlap_frac: (float) overlap of the nffts (0 = no overlap; 1 = total overlap).
    :return spectrum: (2d array) contains for every timestamp the power of the frequencies listed in the array "freqs".
    :return freqs: (array) frequencies of the spectrogram.
    :return time: (array) time of the nffts.
    """

    def nfft_noverlap(freq_resolution, samplerate, overlap_frac, min_nfft=0, nfft_multiplikation = 1.):
        """The required number of points for an FFT to achieve a minimum frequency resolution
        and the number of overlapping data points.

        :param freq_resolution: (float) the minimum required frequency resolution in Hertz.
        :param samplerate: (float) the sampling rate of the data in Hertz.
        :param overlap_frac: (float) the fraction the FFT windows should overlap.
        :param min_nfft: (int) the smallest value of nfft to be used.
        :param nfft_multiplikation (float) value the nfft is multiplicated with to increase/decrease nfft size.
        :return nfft: (int) the number of FFT points.
        :return noverlap: (int) the number of overlapping FFT points.
        """
        nfft = ps.next_power_of_two(samplerate / freq_resolution)
        nfft *= nfft_multiplikation
        nfft = int(nfft)
        if nfft < min_nfft:
            nfft = min_nfft
        noverlap = int(nfft * overlap_frac)
        return nfft, noverlap

    nfft, noverlap = nfft_noverlap(fresolution, samplerate, overlap_frac, min_nfft=16, nfft_multiplikation= 0.25)

    spectrum, freqs, time = mlab.specgram(data, NFFT=nfft, Fs=samplerate, detrend=detrend, window=window,
                                          noverlap=noverlap, pad_to=pad_to, sides=sides, scale_by_freq=scale_by_freq)
    return spectrum, freqs, time

def clean_chirps(chirp_time_idx, power, power_window=100):
    """
    Chirp is only accepted as such if the power of the frequency drops down as expected.

    :param chirp_time_idx: (array) indices of chirps.
    :param power: (array) power array containing for each timestamp the max value in power of a certain frequency range.
    :param power_window (int) datapoints arroung a detected chirp used to verify that there is a chirp.
    :return: chirp_time_idx: (array) indices of chirps that have been confirmed to be chirps.
    """
    true_chirps = np.array([], dtype=bool)
    for i in range(len(chirp_time_idx)):
        idx0 = int(chirp_time_idx[i] - power_window/2)
        idx1 = int(chirp_time_idx[i] + power_window/2)

        tmp_median = np.median(power[idx0:idx1])
        tmp_std = np.std(power[idx0:idx1], ddof=1)

        if np.min(power[idx0:idx1]) < tmp_median - 3*tmp_std:
            true_chirps = np.append(true_chirps, True)
        else:
            true_chirps = np.append(true_chirps, False)
    return chirp_time_idx[true_chirps]

def chirp_detection(spectrum, freqs, time, fishlist, min_power= 0.005, freq_tolerance=5., chirp_th=1., plot_data=False):
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
    fundamentals = []
    for fish in fishlist:
        if fish[0][1] > min_power:
            fundamentals.append(fish[0][0])

    chirp_time = np.array([])

    for enu, fundamental in enumerate(fundamentals):
        # extract power of only the part of the spectrum that has to be analysied for each fundamental
        spectrum1 = spectrum[freqs >= fundamental - freq_tolerance]
        freqs1 = freqs[freqs >= fundamental - freq_tolerance]
        spectrum2 = spectrum1[freqs1 <= fundamental + freq_tolerance]
        # freqs2 = freqs1[freqs1 <= fundamental + freq_tolerance]

        # get the peak power of every piont in time
        power = np.max(spectrum2[:], axis=0)

        # calculate the slope by calculating the difference in the power
        power_diff = np.diff(power)

        # peakdetection in the power_diff to detect drops in power indicating chrips
        threshold = pkd.std_threshold(power_diff)
        peaks, troughs = pkd.detect_peaks(power_diff, threshold)
        troughs, peaks = pkd.trim_to_peak(troughs, peaks) # reversed troughs and peaks in output and input to get trim_to_troughs

        # exclude peaks and troughs with to much time diff to be a chirp
        for i in np.arange(len(troughs))[::-1]:
            if abs(time[troughs[i]] - time[peaks[i]]) > chirp_th:
                peaks = np.delete(peaks, i)
                troughs = np.delete(troughs, i)

        if len(troughs) > 0:
            # chirps times defined as the mean time between the troughs and peaks
            chirp_time_idx = np.mean([troughs, peaks], axis=0)

            # exclude detected chirps if the powervalue doesn't drop far enought
            chirp_time_idx = clean_chirps(chirp_time_idx, power)

            # add times of detected chirps to the list.
            chirp_time = np.concatenate((chirp_time, np.array([time[int(i)] for i in chirp_time_idx])))

        else:
            chirp_time = np.array([])

        if plot_data:
            if enu == 0:
                fig, ax = plt.subplots()
                colors = ['r', 'g', 'k', 'blue', 'r', 'g', 'k', 'blue', 'r', 'g', 'k', 'blue', 'r', 'g', 'k', 'blue']
            ax.plot(chirp_time, [0 for i in chirp_time], 'o', markersize=10, color=colors[enu], alpha=0.8, label='chirps')
            ax.set_xlabel('time in sec')
            ax.set_ylabel('power')

            ax.plot(time, power, colors[enu], marker= '.', label='%.1f Hz' % fundamental)
            ax.plot(time[:len(power_diff)], power_diff, colors[enu], label='slope')
            plt.legend(loc='upper right', bbox_to_anchor=(1, 1),frameon=False)

    return chirp_time

def chirp_analysis(data, samplerate, cfg, min_power=0.005):
    """
    Performs the steps to detect chirps in a given dataset.
    For further documentation see functions chirp_spectrogram() and chirp_detection().

    :param data: (array) data.
    :param samplerate: (float) smaplerate of the data.
    :param cfg:(dict) HAS TO BE REMOVED !!!!
    :param min_power: (float) minimal power of the fish fundamental to include this fish in chirp detection.
    """
    spectrum, freqs, time = chirp_spectrogram(data, samplerate, overlap_frac=0.95)

    power = np.mean(spectrum, axis=1) # spectrum[:, t0:t1] to only let spectrum of certain time....

    fishlist = hg.harmonic_groups(freqs, power, cfg)[0]

    fundamentals = []
    for fish in fishlist:
        if fish[0][1] > min_power:
            fundamentals.append(fish[0][0])

    chirp_time = chirp_detection(spectrum, freqs, time, fishlist, plot_data=True)

    plt.show()

    return chirp_time

if __name__ == '__main__':
    ###
    # If you want to test the code I propose to use the file '60427L05.WAV' of the transect
    # '2016_04_27__downstream_stonewall_at_pool' made in colombia, 2016.
    ###

    cfg = ct.get_config_dict()

    audio_file = sys.argv[1]
    raw_data, samplerate, unit = dl.load_data(audio_file, channel=0)

    chirp_time = chirp_analysis(raw_data, samplerate, cfg)

    # power = np.mean(spectrum[:, t:t + nffts_per_psd], axis=1)