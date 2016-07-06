"""
Functions to calculate a powerspectrum o n the basis of a given dataset, a given samplingrate and a given
frequencyresolution for the psd.

multi_resolution_psd(): Performs the steps to calculate a powerspectrum.
"""

import numpy as np
import matplotlib.mlab as mlab


def next_power_of_two(n):
    """The next integer power of two for an arbitray number.
    
    :param n: (int or float) a positive number
    :return: (int) the next integer power of two
    """
    return int(2 ** np.floor(np.log(n) / np.log(2.0) + 1.0-1e-8))


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
    nfft = next_power_of_two(samplerate / freq_resolution)
    if nfft < min_nfft:
        nfft = min_nfft
    noverlap = int(nfft * overlap_frac)
    return nfft, noverlap


def psd(data, samplerate, fresolution, detrend=mlab.detrend_none,
        window=mlab.window_hanning, overlap_frac=0.5, pad_to=None,
        sides='default', scale_by_freq=None):
    """Power spectrum density of a given frequency resolution.

    From the requested frequency resolution and the samplerate nfft is computed.
    
    :param data:                (1-D array) data array you want to calculate a psd of.
    :param samplerate:          (float) sampling rate of the data in Hertz.
    :param fresolution:         (float) frequency resolution of the psd in Hertz.
    :param overlap_frac:             (float) fraction of overlap for the fft windows.
    See numpy.psd for the remaining parameter.

    :return:                    (2-D array) power and frequency.
    """

    nfft, noverlap = nfft_noverlap(fresolution, samplerate, overlap_frac, min_nfft=16)
    power, freqs = mlab.psd(data, NFFT=nfft, noverlap=noverlap, Fs=samplerate, detrend=detrend, window=window,
                            pad_to=pad_to, sides=sides, scale_by_freq=scale_by_freq)
    return np.asarray([np.squeeze(power), freqs])   # squeeze is necessary when nfft is to large with respect to the data


def plot_decibel_psd(power, freqs, ax, max_freq=3000, fs=12, color='blue', alpha=1.):
    """
    Plots the powerspectum in decibel.

    :param power:               (1-D array) power array of a psd.
    :param freqs:               (1-D array) frequency array of a psd.
    :param ax:                  (axis for plot) empty axis that is filled with content in the function.
    :param max_freq:            (float) maximum frequency that shall appear in the plot.
    :param fs:                  (int) fontsize for the plot.
    :param color:               (string) color that shall be used for the plot.
    :param alpha:               (float) transparency of the plot.
    """
    decibel_psd = power.copy()
    decibel_psd[decibel_psd < 1e-20] = np.nan
    decibel_psd[decibel_psd >= 1e-20] = 10.0 * np.log10(decibel_psd[decibel_psd >= 1e-20])
    ax.plot(freqs[freqs < max_freq], decibel_psd[freqs < max_freq], color=color, alpha=alpha)
    ax.set_ylabel('Power [dB]', fontsize=fs)
    ax.set_xlabel('Frequency [Hz]', fontsize=fs)


def multi_resolution_psd(data, samplerate, fresolution=0.5,
                         detrend=mlab.detrend_none, window=mlab.window_hanning,
                         overlap=0.5, pad_to=None, sides='default',
                         scale_by_freq=None):
    """Performs the steps to calculate a powerspectrum on the basis of a given dataset, a given samplingrate and a given
    frequencyresolution for the psd.

    Two other functions are called to first calculate the nfft value and second calculate the powerspectrum. The given
    frequencyresolution can be a float or a list/array of floats.

    (for further argument information see numpy.psd documentation)
    :param data:                (1-D array) data array you want to calculate a psd of.
    :param samplerate:          (float) sampling rate of the data in Hertz.
    :param fresolution:         (float or 1-D array) frequency resolutions for one or multiple psds in Hertz.
    :param overlap:             (float) fraction of overlap for the fft windows.
    :param plot_data_func:      (function) function (powerspectrum_plot()) that is used to create a axis for later
                                plotting containing the calculated powerspectrum.
    :param **kwargs:            additional arguments that are passed to the plot_data_func().
    :return multi_psd_data:     (3-D or 2-D array) if the psd shall only be calculated for one frequency resolution
                                this Outupt is a 2-D array ( psd_data[power, freq] )
                                If the psd shall be calculated for multiple frequency resolutions its a 3-D array
                                (psd_data[frequency_resolution][power, freq])
    :return ax:                 (axis for plot) axis that is ready for plotting containing a figure that shows what the
                                modul did.
    """
    return_list = True
    if not hasattr(fresolution, '__len__'):
        return_list = False
        fresolution = [fresolution]

    multi_psd_data = []
    for fres in fresolution:
        psd_data = psd(data, samplerate, fres, detrend, window, overlap, pad_to, sides, scale_by_freq)
        multi_psd_data.append(psd_data)

    if not return_list:
        multi_psd_data = multi_psd_data[0]

    return multi_psd_data


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    print('Computes powerspectrum of a created signal of two wavefish (300 and 450 Hz)')
    print('')
    print('Usage:')
    print('  python powerspectrum.py')
    print('')

    fundamental = [300, 450]  # Hz
    samplerate = 100000
    time = np.linspace(0, 8 - 1 / samplerate, 8 * samplerate)
    data = np.sin(time * 2 * np.pi * fundamental[0]) + np.sin(time * 2 * np.pi * fundamental[1])

    psd_data = multi_resolution_psd(data, samplerate, fresolution=[0.5, 1])

    fig, ax = plt.subplots()
    plot_decibel_psd(psd_data[0][0], psd_data[0][1], ax=ax, fs=12, color='firebrick', alpha=0.9)
    plt.show()
