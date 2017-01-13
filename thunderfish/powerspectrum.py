"""
Computing and plot powerspectra and spectrograms.

next_power_of_two(): rounds an integer up to the next power of two.
nfff_overlap():      computes nfft and overlap based on a requested minimum frequency resolution
                     and overlap fraction.
                
psd():                  Compute power spectrum with a given frequency resolution.
decibel():              Transforms power to decibel.
plot_decibel_psd():     Plot power spectrum in decibel.
multi_resolution_psd(): Performs the steps to calculate a powerspectrum.
spectrogram():          Spectrogram of a given frequency resolution and overlap fraction.
"""

import numpy as np
import matplotlib.mlab as mlab


def next_power_of_two(n):
    """The next integer power of two for an arbitray number.
    
    :param n: (int or float) a positive number
    :return: (int) the next integer power of two
    """
    return int(2 ** np.floor(np.log(n) / np.log(2.0) + 1.0-1e-8))


def nfft_noverlap(freq_resolution, samplerate, overlap_frac, min_nfft=16):
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


def psd(data, samplerate, fresolution, min_nfft=16, detrend=mlab.detrend_none,
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

    nfft, noverlap = nfft_noverlap(fresolution, samplerate, overlap_frac, min_nfft=min_nfft)
    power, freqs = mlab.psd(data, NFFT=nfft, noverlap=noverlap, Fs=samplerate, detrend=detrend, window=window,
                            pad_to=pad_to, sides=sides, scale_by_freq=scale_by_freq)
    return np.asarray([np.squeeze(power), freqs])   # squeeze is necessary when nfft is to large with respect to the data


def decibel(power, ref_power=1.0, min_power=1e-20):
    """
    Transforms power to decibel relative to ref_power.

    decibel_psd = 10 * log10(power/ref_power)

    Parameters
    ----------
    power: array
        the power values of the power spectrum or spectrogram.
    ref_power: float
        the reference power for computing decibel. If set to None the maximum power is used.
    min_power: float
        power values smaller than min_power are set to np.nan.

    Returns
    -------
    decibel_psd: array
        the power values in decibel
    """
    if ref_power is None:
        ref_power = np.max(power)
    decibel_psd = power.copy()
    decibel_psd[power < min_power] = np.nan
    decibel_psd[power >= min_power] = 10.0 * np.log10(decibel_psd[power >= min_power]/ref_power)
    return decibel_psd


def plot_decibel_psd(ax, freqs, power, ref_power=1.0, min_power=1e-20, max_freq=2000.0, **kwargs):
    """
    Plot the powerspectum in decibel relative to ref_power.

    Parameters
    ----------
    ax:
        axis for plot
    freqs: 1-D array
        frequency array of a psd.
    power: 1-D array
        power array of a psd.
    ref_power: float
        the reference power for computing decibel. If set to None the maximum power is used.
    min_power: float
        power values smaller than min_power are set to np.nan.
    max_freq: float
        limits of frequency axis are set to (0, max_freq) if max_freq is greater than zero
    kwargs:
        plot parameter that are passed on to the plot() function.
    """
     
    decibel_psd = decibel(power, ref_power=ref_power, min_power=min_power)
    ax.plot(freqs, decibel_psd, **kwargs)
    ax.set_xlabel('Frequency [Hz]')
    if max_freq > 0.0:
        ax.set_xlim(0, max_freq)
    else:
        max_freq = freqs[-1]
    pmin = np.nanmin(decibel_psd[freqs < max_freq])
    pmin = np.floor(pmin / 10.0) * 10.0
    pmax = np.nanmax(decibel_psd[freqs < max_freq])
    pmax = np.ceil(pmax / 10.0) * 10.0
    ax.set_ylim(pmin, pmax)
    ax.set_ylabel('Power [dB]')


def multi_resolution_psd(data, samplerate, fresolution=0.5,
                         detrend=mlab.detrend_none, window=mlab.window_hanning,
                         overlap=0.5, pad_to=None, sides='default',
                         scale_by_freq=None, min_nfft=16):
    """Compute powerspectrum with a given frequency resolution.

    Two other functions are called to first calculate the nfft value and second calculate the powerspectrum. The given
    frequencyresolution can be a float or a list/array of floats.

    (for information on further arguments see numpy.psd documentation)
    :param data:                (1-D array) data array you want to calculate a psd of.
    :param samplerate:          (float) sampling rate of the data in Hertz.
    :param fresolution:         (float or 1-D array) frequency resolutions for one or multiple psds in Hertz.
    :param overlap:             (float) fraction of overlap for the fft windows.
    :return multi_psd_data:     (3-D or 2-D array) if the psd is calculated for one frequency resolution
                                a 2-D array with the single power spectrum is returned (psd_data[power, freq]).
                                If the psd is calculated for multiple frequency resolutions
                                a list of 2-D array is returned (psd_data[frequency_resolution][power, freq]).
    """
    return_list = True
    if not hasattr(fresolution, '__len__'):
        return_list = False
        fresolution = [fresolution]

    multi_psd_data = []
    for fres in fresolution:
        psd_data = psd(data, samplerate, fres, min_nfft, detrend, window, overlap, pad_to, sides, scale_by_freq)
        multi_psd_data.append(psd_data)

    if not return_list:
        multi_psd_data = multi_psd_data[0]

    return multi_psd_data


def spectrogram(data, samplerate, fresolution=0.5, detrend=mlab.detrend_none, window=mlab.window_hanning,
                overlap_frac=0.5, pad_to=None, sides='default', scale_by_freq=None, min_nfft=16):
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

    nfft, noverlap = nfft_noverlap(fresolution, samplerate, overlap_frac, min_nfft=min_nfft)

    spectrum, freqs, time = mlab.specgram(data, NFFT=nfft, Fs=samplerate, detrend=detrend, window=window,
                                          noverlap=noverlap, pad_to=pad_to, sides=sides, scale_by_freq=scale_by_freq)
    return spectrum, freqs, time


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    print('Computes powerspectrum of a created signal of two wavefish (300 and 450 Hz)')
    print('')
    print('Usage:')
    print('  python powerspectrum.py')
    print('')

    fundamentals = [300, 450]  # Hz
    samplerate = 100000.0      # Hz
    time = np.arange(0.0, 8.0, 1.0/samplerate)
    data = np.sin(2*np.pi*fundamentals[0]*time) + 0.5*np.sin(2*np.pi*fundamentals[1]*time)

    psd_data = multi_resolution_psd(data, samplerate, fresolution=[0.5, 1])

    fig, ax = plt.subplots()
    plot_decibel_psd(ax, psd_data[0][1], psd_data[0][0], lw=2)
    plot_decibel_psd(ax, psd_data[1][1], psd_data[1][0], lw=2)
    plt.show()
