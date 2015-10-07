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

    def detect_peak_indices(self, peak_threshold=None, norm_window=.1):

        # time = self._time if window == false
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
        pidx, tidx = self.detect_peak_indices()

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

    def detect_best_window(self, win_width=8., plot_debug=False, ax=False):
    # for plot debug, call this function in "main" with plot_debug=True
        filename = self._wavfile
        p_idx, t_idx = self.detect_peak_indices()
        #everything = list(self.peak_trough_iterator())
        #peak_time, peak_ampl, _, trough_time, trough_ampl, _ = map(lambda x: np.asarray(x), zip(*everything))
        peak_time, peak_ampl, trough_time, trough_ampl = self._time[p_idx], self._eod[p_idx], self._time[t_idx], self._eod[t_idx]
        my_times = peak_time[peak_time <= peak_time[-1] - win_width]
        cvs = np.empty(len(my_times))
        no_of_peaks = np.empty(len(my_times))
        mean_ampl = np.empty(len(my_times))

        for i, curr_t in enumerate(my_times):
            window_idx = (peak_time >= curr_t) & (peak_time <= curr_t + win_width)
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
            ax1.set_ylabel('# of peaks')
            ax1.legend(frameon=False, loc='best')
            ax2.scatter(a, mean_ampl, s=50, c='green')
            ax2.set_ylabel('mean amplitude')
            ax3.scatter(a, cvs, s=50, c='red', label=filename[-10:-8])
            ax3.set_ylabel('cvs')
            ax3.set_xlabel('# of Peak Time Window')
            ax3.legend(frameon=False, loc='best')
            ax = np.array([ax1, ax2, ax3])

        bwin_bool_inx = self.best_window_algorithm(no_of_peaks, mean_ampl, cvs, plot_debug=plot_debug, axs=ax)
        bwin_bool_inx = np.where(bwin_bool_inx)[0][0]
        entire_time_idx = self._eod_peak_idx[bwin_bool_inx]
        bwin = my_times[bwin_bool_inx]

        self.roi = (entire_time_idx, entire_time_idx + int(self._sample_rate/2.))
        return bwin, win_width

    def best_window_algorithm(self, peak_no, mean_amplitudes, cov_coeffs, pks_th=0.15, ampls_th=(0.85, 0.3),
                              plot_debug=False, axs=None):

        pk_mode = stats.mode(peak_no)
        tot_pks = max(peak_no)-min(peak_no)
        lower = peak_no >= pk_mode[0][0] - tot_pks*pks_th
        upper = peak_no <= pk_mode[0][0] + tot_pks*pks_th
        valid_pks = lower * upper

        ampl_means = valid_pks * mean_amplitudes
        ampl_means = np.where(ampl_means == 0., np.median(ampl_means), ampl_means)  # replace the median ampl where 0's
        tot_ampls = max(ampl_means)-min(ampl_means)
        amplitude_th_up = ampl_means <= min(ampl_means) + tot_ampls * ampls_th[0]
        amplitude_th_dn = ampl_means >= min(ampl_means) + tot_ampls * ampls_th[1]
        valid_ampls = valid_pks * amplitude_th_up * amplitude_th_dn

        cov_coeffs[np.isnan(cov_coeffs)] = 1000.0  # Set a huge number where NaN to avoid NaN!!
        valid_cv = min(cov_coeffs[valid_ampls])
        best_window = valid_cv == cov_coeffs

        if plot_debug:

            ax1 = axs[0]
            ax2 = axs[1]
            ax3 = axs[2]

            a = np.arange(len(peak_no))
            ax1.plot([a[0], a[-1]], [pk_mode[0][0] - tot_pks*pks_th, pk_mode[0][0] - tot_pks*pks_th], '--k')
            ax1.plot([a[0], a[-1]], [pk_mode[0][0] + tot_pks*pks_th, pk_mode[0][0] + tot_pks*pks_th], '--k')

            ax2.plot([a[0], a[-1]], [min(ampl_means) + tot_ampls * ampls_th[0], min(ampl_means) + tot_ampls * ampls_th[0]], '--k')
            ax2.plot([a[0], a[-1]], [min(ampl_means) + tot_ampls * ampls_th[1], min(ampl_means) + tot_ampls * ampls_th[1]], '--k')

            ax3.plot([a[best_window][0], a[best_window][0]], [min(cov_coeffs), max(cov_coeffs)], '--k')
            plt.show()

        # cond3 = cond1 & cond2 (initially cond3 is false)
        # while not np.any(cond3): increase threshold do stuff compute new cond3

        # check if cond1 & cond2 has only false np.any(cond1 & cond2)
        # if any is False return self.best_window_algorithm(.... mit mehr threshold)

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

    def plot_wavenvelope(self, ax, win_edges):

        window_size = int(0.05 * self._sample_rate)  # 0.050 are 50 milliseconds for the envelope window!
        w = 1.0 * np.ones(window_size) / window_size
        envelope = (np.sqrt((np.correlate(self._eod ** 2, w, mode='same') -
                    np.correlate(self._eod, w, mode='same') ** 2)).ravel()) * np.sqrt(2.)
        ax.fill_between(self._time[::500], y1=-envelope[::500], y2=envelope[::500], color='purple', alpha=0.5)
        ax.plot((win_edges[0], win_edges[0]), (-22000, 16000), 'k--', linewidth=2)
        ax.plot((win_edges[1], win_edges[1]), (-22000, 16000), 'k--', linewidth=2)
        ax.text(win_edges[0]+(win_edges[1] - win_edges[0]), 19000, 'Analysis Window', rotation='horizontal',
                horizontalalignment='center', verticalalignment='center', fontsize=10)

        ax.set_ylim(-22000, 22000)
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('Signal Amplitude [au]')

        pass

    def __str__(self):
        return """Fish trace from %s with %i sample points""" % (self._wavfile, self._eod.size)