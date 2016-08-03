import sys
import numpy as np
import matplotlib.mlab as mlab
import dataloader as dl
import bestwindow as bw
import harmonicgroups as hg
import config_tools as ct
from scipy.signal import butter, lfilter
import powerspectrum as ps
import matplotlib.pyplot as plt
from IPython import embed
import peakdetection as pkd
import time as tm

def spectogram(data, samplerate, fresolution=0.5, detrend=mlab.detrend_none, window=mlab.window_hanning, overlap_frac=0.5,
               pad_to=None, sides='default', scale_by_freq=None):

    def nfft_noverlap(freq_resolution, samplerate, overlap_frac, min_nfft=0):
        """The required number of points for an FFT to achieve a minimum frequency resolution
        and the number of overlapping data points.

        :param freq_resolution: (float) the minimum required frequency resolution in Hertz.
        :param samplerate: (float) the sampling rate of the data in Hertz.
        :param overlap_frac: (float) the fraction the FFT windows should overlap.
        :param min_nfft: (int) the smallest value of nfft to be used.
        :return nfft: (int) the number of FFT points.
        :return noverlap: (int) the number of overlapping FFT points.
        """
        nfft = ps.next_power_of_two(samplerate / freq_resolution)
        if nfft < min_nfft:
            nfft = min_nfft
        noverlap = int(nfft * overlap_frac)
        return nfft, noverlap

    nfft, noverlap = nfft_noverlap(fresolution, samplerate, overlap_frac, min_nfft=16)

    spectrum, freqs, time = mlab.specgram(data, NFFT=nfft, Fs=samplerate, detrend=detrend, window=window,
                                          noverlap=noverlap, pad_to=pad_to, sides=sides, scale_by_freq=scale_by_freq)
    return spectrum, freqs, time

def chirp_detection(spectrum, freqs, time, fundamentals):
    fig, ax = plt.subplots()
    colors = ['r', 'g', 'k', 'blue']
    chirp_time = np.array([])
    for enu, fundamental in enumerate(fundamentals):
        spectrum1 = spectrum[freqs >= fundamental-5.]
        freqs1 = freqs[freqs >= fundamental-5.]

        spectrum2 = spectrum1[freqs1 <= fundamental+5.]
        freqs2 = freqs1[freqs1 <= fundamental+5.]

        power = np.max(spectrum2[:], axis=0)

        power_diff = np.diff(power)

        threshold = pkd.percentile_threshold(power_diff, th_factor=0.4)
        peaks, troughs = pkd.detect_peaks(power_diff, threshold)
        troughs, peaks = pkd.trim_to_peak(troughs, peaks) # reversed troughs and peaks in output and input to get trim_to_troughs

        chirp_time_idx = np.mean([troughs, peaks], axis=0)
        chirp_time = np.concatenate((chirp_time, np.array([time[int(i)] for i in chirp_time_idx])))

        ax.plot(time, power, colors[enu], marker= '.', label='%.1f Hz' % fundamental)
        ax.plot(time[:len(power_diff)], power_diff, colors[enu], label='slope')

        ax.plot(chirp_time, [0 for i in chirp_time], 'o', markersize=10, color=colors[enu], alpha=0.8, label='chirps')

        ax.set_xlabel('time in sec')
        ax.set_ylabel('power')

    plt.legend(loc='upper right', bbox_to_anchor=(1, 1),frameon=False)

    return chirp_time

def chirp_data_snippets(chirp_times):
    snippets = []
    chirp_times = np.array(sorted(chirp_times))

    while len(chirp_times) > 0:
        snippets.append([chirp_times[0]-1, chirp_times[0]+9])
        chirp_times = chirp_times[chirp_times > chirp_times[0] + 9 ]

    for s_idx in np.arange(1, len(snippets))[::-1]:
        if snippets[s_idx][0] < snippets[s_idx-1][1]:
            snippets[s_idx][0] = snippets[s_idx-1][0]
            snippets.pop(s_idx-1)

    return snippets

def chirp_analysis(data, samplerate, fundamentals):

    spectrum, freqs, time = spectogram(data, samplerate, overlap_frac=0.95)

    chirp_time = chirp_detection(spectrum, freqs, time, fundamentals)

    chirp_snippets = chirp_data_snippets(chirp_time)

    embed()
    plt.show()

if __name__ == '__main__':
    cfg = ct.get_config_dict()

    audio_file = sys.argv[1]
    raw_data, samplerate, unit = dl.load_data(audio_file, channel=0)

    clip_win_size = 0.5
    min_clip, max_clip = bw.clip_amplitudes(raw_data, int(clip_win_size * samplerate))
    idx0, idx1, clipped = bw.best_window_indices(raw_data, samplerate, single=True, win_size=8.0, min_clip=min_clip,
                                                 max_clip=max_clip, w_cv_ampl=10.0, th_factor=0.8)
    data = raw_data[idx0:idx1]

    psd_data = ps.multi_resolution_psd(data, samplerate)

    fishlist = hg.harmonic_groups(psd_data[1], psd_data[0], cfg)[0]

    # find fishes powerful enough to detect chirps
    fundamentals = []
    for fish in fishlist:
        if fish[0][1] > 0.01:
            fundamentals.append(fish[0][0])

    chirp_analysis(raw_data, samplerate, fundamentals)