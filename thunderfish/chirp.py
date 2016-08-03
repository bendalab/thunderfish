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

    for enu, fundamental in enumerate(fundamentals):
        spectrum1 = spectrum[freqs >= fundamental-5.]
        freqs1 = freqs[freqs >= fundamental-5.]

        spectrum2 = spectrum1[freqs1 <= fundamental+5.]
        freqs2 = freqs1[freqs1 <= fundamental+5.]

        power = np.max(spectrum2[:], axis=0)

        power_diff = np.diff(power)
        diff_std = np.std(power_diff, ddof=1)
        diff_mean = np.mean(power_diff)

        ax.plot(time, power, colors[enu], marker= '.')
        ax.plot(time[:len(power_diff)], power_diff, colors[enu])
        ax.plot([0, time[len(power_diff)]], [diff_mean + 3*diff_std, diff_mean + 3*diff_std], colors[enu])
        ax.plot([0, time[len(power_diff)]], [diff_mean - 3*diff_std, diff_mean - 3*diff_std], colors[enu])


def chirp_analysis(data, samplerate, fundamentals):

    spectrum, freqs, time = spectogram(data, samplerate, overlap_frac=0.95)

    chirp_detection(spectrum, freqs, time, fundamentals)

    plt.show()
    embed()
    quit()

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
        # fundamentals.append(fish[0][0])

    chirp_analysis(raw_data, samplerate, fundamentals)