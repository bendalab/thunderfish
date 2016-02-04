"""
Created on Fri March 21 14:40:04 2014

@author: Juan Felipe Sehuanes
"""

from Auxiliary import *
import glob
import sys
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from matplotlib import mlab
from IPython import embed


def needs_roi(func):
    def new_func(*args, **kwargs):
        if args[0].roi is None:
            args[0].detect_best_window()
        return func(*args, **kwargs)
    return new_func


class FishRecording:

    def __init__(self, wavfile):
        self._wavfile = wavfile
        self._time, self._eod, self._sample_rate = load_trace(wavfile)
        self._eod_peak_idx = None
        self._eod_trough_idx = None
        self.roi = None

    def filtr_data(self, filter_fac=5.5):
        fund_freq = self.fund_freq()
        filtered_data = butter_lowpass_filter(self._eod, fund_freq * filter_fac, self._sample_rate)
        return filtered_data

    def detect_peak_and_trough_indices(self, peak_threshold=None, norm_window=.1):
        """This function finds the indices of peaks and troughs of each EOD-cycle in the recording

        :param peak_threshold: This is the threshold to be used for the peakdet function (Translated Matlab-Code...).
        :param norm_window:
        :return: two arrays. The first contains the peak indices and the second contains the trough indices.
        """
        if self._eod_peak_idx is None or self._eod_trough_idx is None:

            w = np.ones(self._sample_rate*norm_window)
            w[:] /= len(w)
            eod2 = np.sqrt(np.correlate(self._eod**2., w, mode='same') - np.correlate(self._eod, w, mode='same')**2.)
            eod2 = self._eod / eod2
            if peak_threshold is None:
                peak_threshold = np.percentile(np.abs(eod2), 99.9)-np.percentile(np.abs(eod2), 70)
              # The Threshold is 1.5 times the standard deviation of the eod

            _, self._eod_peak_idx, _, self._eod_trough_idx = peakdet(eod2, peak_threshold)
            # refine by matching troughs and peaks
            everything = list(self.peak_trough_iterator())
            _, _, self._eod_peak_idx, _, _, self._eod_trough_idx = map(lambda x: np.asarray(x), zip(*everything))

        return self._eod_peak_idx, self._eod_trough_idx

    def forget_peaks(self):
        self._eod_peak_idx = None
        self._eod_trough_idx = None
        pass

    def peak_trough_iterator(self):
        pidx, tidx = self.detect_peak_and_trough_indices()

        t = np.hstack((self._time[pidx], self._time[tidx]))
        y = np.hstack((self._time[pidx]*0+1, self._time[tidx]*0-1))
        all_idx = np.hstack((pidx, tidx))
        a = np.hstack((self._eod[pidx], self._eod[tidx]))
        idx = np.argsort(t)
        pt = y[idx]
        t = t[idx]
        a = a[idx]
        all_idx = all_idx[idx]

        total_length = len(t)
        hit_upper_limit = False
        hit_lower_limit = False
        for i in np.where(pt == 1)[0]:
            t_peak = t[i]
            a_peak = a[i]

            k_next = i
            k_prev = i
            while pt[k_next] > 0:
                k_next += 1
                if k_next == total_length:
                    hit_upper_limit = True
                    break
            if hit_upper_limit:
                break

            t_next_trough = t[k_next]

            while pt[k_prev] > 0:
                k_prev -= 1
                if k_prev < 0:
                    hit_lower_limit = True
                    break
            if hit_lower_limit:
                hit_lower_limit = False
                continue

            t_prev_trough = t[k_prev]
            trough_idx = None
            if np.abs(t_next_trough-t_peak) < np.abs(t_prev_trough-t_peak):
                t_trough = t_next_trough
                a_trough = a[k_next]
                trough_idx = k_next
            else:
                t_trough = t_prev_trough
                a_trough = a[k_prev]
                trough_idx = k_prev

            yield t_peak, a_peak, all_idx[i], t_trough, a_trough, all_idx[trough_idx]

    def detect_best_window(self, win_size=8., plot_debug=False, ax=False):
        """ This function detects the best window of the file to be analyzed. The core mechanism is in the
        best_window_algorithm function. For plot debug, call this function in "main" with argument plot_debug=True

        :param win_size: float. size of the best window in seconds
        :param plot_debug: boolean. use True to plot filter parameters (and thresholds) for detecting best window
        :param ax: axes of the debugging plots.
        :return: two floats. The first float marks the start of the best window and the second the defined window-size.
        """
        filename = self._wavfile
        p_idx, t_idx = self.detect_peak_and_trough_indices()
        peak_time, peak_ampl, trough_time, trough_ampl = self._time[p_idx], self._eod[p_idx],\
                                                         self._time[t_idx], self._eod[t_idx]
        # peaks and troughs here refer to those found in each eod-cycle. For each cycle there should be one peak and
        # one trough if the detect_peak_indices function worked fine.
        my_times = peak_time[peak_time <= peak_time[-1] - win_size]  # Upper window-boundaries solution
        cvs = np.empty(len(my_times))
        no_of_peaks = np.empty(len(my_times))
        mean_ampl = np.empty(len(my_times))

        for i, curr_t in enumerate(my_times):
            # This for-loop goes through each eod-cycle. Isn't this too much? It considerably makes the code slow.
            window_idx = (peak_time >= curr_t) & (peak_time <= curr_t + win_size)
            # the last line makes a window from curr_t and adds 8. seconds to it. Lower window-boundaries solution.
            p2t_ampl = peak_ampl[window_idx] - trough_ampl[window_idx]
            cvs[i] = np.std(p2t_ampl, ddof=1) / np.mean(p2t_ampl)
            mean_ampl[i] = np.mean(p2t_ampl)
            no_of_peaks[i] = len(p2t_ampl)

        if plot_debug:
            fig = plt.figure(figsize=(14, 15), num='Fish No. '+filename[-10:-8])
            ax1 = fig.add_subplot(3, 1, 1)
            ax2 = fig.add_subplot(3, 1, 2)
            ax3 = fig.add_subplot(3, 1, 3)

            a = np.arange(len(no_of_peaks))
            ax1.scatter(a, no_of_peaks, s=50, c='blue', label='Fish # ' + filename[-10:-8])
            ax1.set_ylabel('# of peaks (# of detected EOD-cycles)')
            ax1.legend(frameon=False, loc='best')

            ax2.scatter(a, cvs, s=50, c='red')
            ax2.set_ylabel('cvs (std of p2t_amplitude / mean p2t_amplitude)')

            ax3.scatter(a, mean_ampl, s=50, c='green')
            ax3.set_ylabel('mean amplitude')
            ax3.set_xlabel('# of Peak Time Window')

            ax = np.array([ax1, ax2, ax3])

        bwin_bool_array = self.best_window_algorithm(no_of_peaks, mean_ampl, cvs, plot_debug=plot_debug, axs=ax)
        bwin_bool_inx = np.where(bwin_bool_array)[0][0]  # Gets the index of the best window out of all windows.
        entire_time_idx = self._eod_peak_idx[bwin_bool_inx]
        bwin = my_times[bwin_bool_inx]

        self.roi = (entire_time_idx, entire_time_idx + int(self._sample_rate/2.))
        return bwin, win_size

    def best_window_algorithm(self, peak_no, mean_amplitudes, cvs, pks_th=0.15, ampls_percentile_th=85.,
                              cvs_percentile_th=15., plot_debug=False, axs=None):

        """This is the algorithm that chooses the best window. It first filters out the windows that have a siginificant
        different amount of peaks compared to the stats.mode peak number of all windows. Secondly, it filters out
        windows that have a higher coefficient of variation than a certain percentile of the distribution of cvs.
        From those windows that get through both filters, the one with the highest peak-to-trough-amplitude
        (that is not clipped!) is chosen as the best window. We assume clipping as amplitudes above 85% percentile of
        the distribution of peak to trough amplitude.

        :param cvs_percentile_th: threshold of how much amplitude-variance (covariance coefficient) of the signal
        is allowed. Default is 10%.
        :param peak_no: array with number of peaks
        :param mean_amplitudes: array with mean peak-to-trough-amplitudes
        :param cvs: array with covariance coefficients
        :param pks_th: threshold for number-of-peaks-filter
        :param ampls_percentile_th: choose a percentile threshold to avoid clipping. Default is 85
        :param plot_debug: boolean for showing plot-debugging.
        :param axs: axis of plot debugging.
        :return: boolean array with a single True element. This is the Index of the best window out of all windows.
        """
        # First filter: Stable # of detected peaks
        pk_mode = stats.mode(peak_no)
        tot_pks = max(peak_no)-min(peak_no)
        lower = peak_no >= pk_mode[0][0] - tot_pks*pks_th
        upper = peak_no <= pk_mode[0][0] + tot_pks*pks_th
        valid_pks = lower * upper

        # Second filter: Low variance in the amplitude
        cvs[np.isnan(cvs)] = np.median(cvs)  # Set a huge number where NaN to avoid NaN!!
        cov_th = np.percentile(cvs, cvs_percentile_th)
        valid_cv = cvs < cov_th

        # Third filter: From the remaining windows, choose the one with the highest p2t_amplitude that's not clipped.

        # replace the median ampl where 0's
        ampl_means = np.where(mean_amplitudes == 0., np.median(mean_amplitudes), mean_amplitudes)
        tot_ampls = max(ampl_means)-min(ampl_means)
        ampls_th = np.percentile(ampl_means, ampls_percentile_th)
        valid_ampls = ampl_means <= ampls_th

        valid_windows = valid_pks * valid_cv * valid_ampls

        # If there is no best window, run the algorithm again with more flexible threshodlds.
        if not True in valid_windows:
            print('\nNo best window found. Rerunning best_window_algorithm with more flexible arguments.\n')
            return self.best_window_algorithm(peak_no, mean_amplitudes, cvs,
                                              ampls_percentile_th=ampls_percentile_th-5.,
                                              cvs_percentile_th=cvs_percentile_th+5., plot_debug=plot_debug)
            # This return is a Recursion! Need to return the value in the embeded function, otherwise the root_function
            # will not return anything!

        else:
            max_ampl_window = ampl_means == np.max(ampl_means[valid_windows])  # Boolean array with a single True element

            best_window = valid_windows * max_ampl_window
            bwin_found = True

            if plot_debug:

                ax1 = axs[0]
                ax2 = axs[1]
                ax3 = axs[2]

                windows = np.arange(len(peak_no))
                ax1.plot([windows[0], windows[-1]], [pk_mode[0][0] - tot_pks*pks_th, pk_mode[0][0] - tot_pks*pks_th], '--k')
                ax1.plot([windows[0], windows[-1]], [pk_mode[0][0] + tot_pks*pks_th, pk_mode[0][0] + tot_pks*pks_th], '--k')

                ax2.plot([windows[0], windows[-1]], [cov_th, cov_th], '--k')

                ax3.plot([windows[0], windows[-1]], [ampls_th, ampls_th], '--k')
                ax3.plot(windows[best_window], ampl_means[best_window], 'o', ms=20,
                         color='purple', alpha=0.8, label='Best Window')
                ax3.legend(frameon=False, loc='best')

                plt.show()

            return best_window

    @property
    @needs_roi
    def w_time(self):
        assert self.roi is not None, "Must detect window first"
        return self._time[self.roi[0]:self.roi[1]]

    @property
    @needs_roi
    def w_eod(self):
        assert self.roi is not None, "Must detect window first"
        return self._eod[self.roi[0]:self.roi[1]]

    @property
    @needs_roi
    def w_pt(self):
        assert self.roi is not None, "Must detect window first"
        trough_t = self._time[self._eod_trough_idx]
        trough_eod = self._eod[self._eod_trough_idx]
        peak_t = self._time[self._eod_peak_idx]
        peak_eod = self._eod[self._eod_peak_idx]

        start, end = self._time[self.roi[0]], self._time[self.roi[1]]

        idx_t = (trough_t >= start) & (trough_t <= end)
        idx_p = (peak_t >= start) & (peak_t <= end)
        idx = idx_t & idx_p
        return peak_t[idx], trough_t[idx], peak_eod[idx], trough_eod[idx]

    @property
    @needs_roi
    def fund_freq(self):

        wpeak_t, _, wpeak_eod, _ = self.w_pt

        inter_peak_interv = np.diff(wpeak_t)

        hp = np.histogram(inter_peak_interv, bins=max(int(len(inter_peak_interv)/10), 10))
                          #bins=np.arange(min(inter_peak_interv), max(inter_peak_interv), 0.00001))

        fund_freq = 1. / hp[1][np.argmax(hp[0])]
        return fund_freq

    def plot_peaks_troughs(self, mod_filename):
        plot_t = self.w_time
        plot_eod = self.w_eod
        wpeak_t, wtrough_t, wpeak_eod, wtrough_eod = self.w_pt

        # filtr_eod = self.filtr_data
        fig = plt.figure(figsize=(7, 8), num='Fish No. ' + mod_filename[-10:-8])
        ax = fig.add_subplot(1, 1, 1)

        # ax.plot(self._time, filtr_eod, color='blue', alpha=0.8, lw=2.)
        ax.plot(plot_t, plot_eod, color='blue', alpha=0.8, lw=2.)
        ax.plot(wpeak_t, wpeak_eod, 'ok', mfc='crimson')
        ax.plot(wtrough_t, wtrough_eod, 'ok', mfc='lime')

        pass

    def type_detector(self, thres=.5):

        pk_t, tr_t, _, _ = self.w_pt
        pk_2_pk = pk_t[1:] - pk_t[:-1]
        pk_2_tr = np.abs(pk_t - tr_t)
        med = np.median(2*pk_2_tr)

        prop_in_2med = sum((pk_2_pk < 2*med) & (pk_2_pk > med))/float(len(pk_2_pk))
        # in order to detect the type, we check the proportion of pk2pk time differences within 2* the median of pk2tr
        # There should be a large proportion (~1.) for a wave type and a small proportion (~0.) for a pulse type.

        return 'pulse' if prop_in_2med < thres else 'wave'

    def plot_spectogram(self, ax):

        fu_freq = self.fund_freq
        nwindowpoints = 4096
        noverlap = nwindowpoints / 2

        Pxx, freqs, t = mlab.specgram(self.w_eod, NFFT=nwindowpoints, Fs=self._sample_rate,
                                      window=np.hamming(nwindowpoints),
                                      noverlap=noverlap)
        ax.plot(freqs, np.mean(Pxx, axis=1), label='Fundamental Frequency: %.1f Hz' % fu_freq)
        # ax.set_xlim(0, fund_freq*filtr)  # Power Spec Xlim is 4 times the fundamental freq of the fish
        ax.set_xlim(0, 4000)
        ax.set_ylabel('Power Spectrum [dB]')
        ax.set_xlabel('Frequency [Hz]')
        ax.set_yscale('log')
        ax.legend(frameon=False, loc='best', prop={'size': 11})

        pass

    def plot_eodform(self, ax, filtr):

        fu_freq = self.fund_freq
        low_pass = fu_freq*filtr
        if self.type_detector() == 'pulse':
            low_pass = 3000.

        f_eod = butter_lowpass_filter(self.w_eod, low_pass, self._sample_rate)
        pt, tt, pe, te = self.w_pt
        plot_inxs = (self.w_time >= tt[1]) & (self.w_time <= tt[3])

        ax.plot(1000. * (self.w_time[plot_inxs] - self.w_time[0]), f_eod[plot_inxs], color='green', lw=2)
        ax.set_ylabel('Amplitude [au]')  # "au" stands for arbitrary unit
        ax.set_xlabel('Time [ms]')

        pass

    def plot_wavenvelope(self, ax, w_start, w_end):

        """ This function plots the envelope of the recording.

        :param ax: The axis in which you wish to plot.
        :param w_start: Start of the best window.
        :param w_end: End of the best window.
        """
        window_size = int(0.05 * self._sample_rate)  # 0.050 are 50 milliseconds for the envelope window!
        w = 1.0 * np.ones(window_size) / window_size
        envelope = (np.sqrt((np.correlate(self._eod ** 2, w, mode='same') -
                    np.correlate(self._eod, w, mode='same') ** 2)).ravel()) * np.sqrt(2.)
        upper_bound = np.max(envelope) + np.percentile(envelope, 1)
        ax.fill_between(self._time[::500], y1=-envelope[::500], y2=envelope[::500], color='purple', alpha=0.5)
        ax.plot((w_start, w_start), (-upper_bound, upper_bound), 'k--', linewidth=2)
        ax.plot((w_end, w_end), (-upper_bound, upper_bound), 'k--', linewidth=2)
        ax.text((w_start + w_end) / 2., upper_bound - np.percentile(envelope, 10), 'Analysis Window',
                rotation='horizontal', horizontalalignment='center', verticalalignment='center', fontsize=14)

        ax.set_ylim(-upper_bound, upper_bound)
        ax.set_xlabel('Time [s]', fontsize=16)
        ax.set_ylabel('Signal Amplitude [au]', fontsize=16)
        ax.tick_params(axis='both', which='major', labelsize=14)

        pass

    def __str__(self):
        return """Fish trace from %s with %i sample points""" % (self._wavfile, self._eod.size)