"""
# Powerspectra and spectrograms for a given frequency resolution

## Computation of nfft
- `next_power_of_two()`: round an integer up to the next power of two.
- `nfff_overlap()`: compute nfft and overlap based on a given frequency resolution.

## Decibel
- `decibel()`: transform power to decibel.
- `power()`: transform decibel to power.

## Power spectra                
- `psd()`: power spectrum for a given frequency resolution.
- `multi_resolution_psd()`: power spectra for a range of frequency resolutions.
- `spectrogram()`: spectrogram of a given frequency resolution and overlap fraction.

## Power spectrum analysis
- `peak_freqs()`: peak frequencies computed for each of the data snippets.

## Visualization
- `plot_decibel_psd()`: plot power spectrum in decibel.
"""

import numpy as np
import scipy.signal as sig
try:
    import matplotlib.mlab as mlab
except ImportError:
    pass
from .eventdetection import detect_peaks


def next_power_of_two(n):
    """The next integer power of two.
    
    Parameters
    ----------
    n: int
        A positive number.

    Returns
    -------
    m: int
        The next integer power of two equal or larger than `n`.
    """
    return int(2 ** np.floor(np.log(n) / np.log(2.0) + 1.0-1e-8))


def nfft_noverlap(freq_resolution, samplerate, overlap_frac=0.5, min_nfft=16):
    """Required number of samples for an FFT of a given frequency resolution
    and number of overlapping data points.

    The returned number of FFT samples results in frequency intervals that are
    smaller or equal to `freq_resolution`.

    Parameters
    ----------
    freq_resolution: float
        Minimum frequency resolution in Hertz.
    samplerate: float
        Sampling rate of the data in Hertz.
    overlap_frac: float
        Fraction the FFT windows should overlap.
    min_nfft: int
        Smallest value of nfft to be used.

    Returns
    -------
    nfft: int
        Number of FFT points.
    noverlap: int
        Number of overlapping FFT points.
    """
    nfft = next_power_of_two(samplerate / freq_resolution)
    if nfft < min_nfft:
        nfft = min_nfft
    noverlap = int(nfft * overlap_frac)
    return nfft, noverlap


def decibel(power, ref_power=1.0, min_power=1e-20):
    """
    Transform power to decibel relative to ref_power.
    ```
    decibel = 10 * log10(power/ref_power)
    ```
    Power values smaller than `min_power` are set to `-np.inf`.

    Parameters
    ----------
    power: float or array
        Power values, for example from a power spectrum or spectrogram.
    ref_power: float
        Reference power for computing decibel. If set to `None` the maximum power is used.
    min_power: float
        Power values smaller than `min_power` are set to `-inf`.

    Returns
    -------
    decibel_psd: array
        Power values in decibel relative to `ref_power`.
    """
    if hasattr(power, '__len__'):
        tmp_power = power
        decibel_psd = power.copy()
    else:
        tmp_power = np.array([power])
        decibel_psd = np.array([power])
    if ref_power is None:
        ref_power = np.max(decibel_psd)
    decibel_psd[tmp_power <= min_power] = float('-inf')
    decibel_psd[tmp_power > min_power] = 10.0 * np.log10(decibel_psd[tmp_power > min_power]/ref_power)
    if hasattr(power, '__len__'):
        return decibel_psd
    else:
        return decibel_psd[0]


def power(decibel, ref_power=1.0):
    """
    Transform decibel back to power relative to ref_power.
    ```
    power = ref_power * 10**(0.1 * decibel)
    ```
    
    Parameters
    ----------
    decibel: array
        Decibel values of the power spectrum or spectrogram.
    ref_power: float
        Reference power for computing power.

    Returns
    -------
    power: array
        Power values of the power spectrum or spectrogram.
    """
    return ref_power * 10.0 ** (0.1 * decibel)


def psd(data, samplerate, freq_resolution, min_nfft=16, overlap_frac=0.5, **kwargs):
    """Power spectrum density of a given frequency resolution.

    NFFT is computed from the requested frequency resolution and the samplerate.
    Check the returned frequency array for the actually used freqeuncy resolution.
    The frequency intervals are smaller or equal to `freq_resolution`.
    NFFT is `samplerate` divided by the actual frequency resolution.
    
    data: 1-D array
        Data array you want to calculate a psd of.
    samplerate: float
        Sampling rate of the data in Hertz.
    freq_resolution: float
        Frequency resolution of the psd in Hertz.
    min_nfft: int
        Smallest value of nfft to be used.
    overlap_frac: float
        Fraction of overlap for the fft windows.
    kwargs: dict
        Further arguments for mlab.psd().

    Returns
    -------
    psd: 2-D array
        Power (first row) and frequency (second row).
    """
    nfft, noverlap = nfft_noverlap(freq_resolution, samplerate, overlap_frac, min_nfft=min_nfft)
    if 'NFFT' in kwargs:
        del kwargs['NFFT']
    if 'noverlap' in kwargs:
        del kwargs['noverlap']
    power, freqs = mlab.psd(data, NFFT=nfft, noverlap=noverlap, Fs=samplerate, **kwargs)
    # squeeze is necessary when nfft is to large with respect to the data:
    return np.asarray([np.squeeze(power), freqs])


def multi_resolution_psd(data, samplerate, freq_resolution=0.5, min_nfft=16,
                         overlap_frac=0.5, **kwargs):
    """Compute power spectra for a range of frequency resolutions.

    Check the returned frequency arrays for the actually used freqeuncy resolutions.
    The frequency intervals are smaller or equal to `freq_resolution`.
    NFFT is `samplerate` divided by the actual frequency resolution.
                         
    Parameters
    ----------
    data: 1-D array
        Data array you want to calculate a psd of.
    samplerate: float
        Sampling rate of the data in Hertz.
    freq_resolution: float or 1-D array
        Frequency resolutions for one or multiple psds in Hertz.
    min_nfft: int
        Smallest value of nfft to be used.
    overlap_frac: float
        Fraction of overlap for the fft windows.
    kwargs: dict
        Further arguments for mlab.psd().

    Returns
    -------
    multi_psd_data: 3-D or 2-D array
        If the power spectrum is calculated for one frequency resolution
        a 2-D array with the single power spectrum is returned (`psd_data[power, freq]`).
        If the power sepctrum is calculated for multiple frequency resolutions
        a list of 2-D array is returned (`psd_data[frequency_resolution][power, freq]`).
    """
    return_list = True
    if not hasattr(freq_resolution, '__len__'):
        return_list = False
        freq_resolution = [freq_resolution]
    multi_psd_data = []
    for fres in freq_resolution:
        psd_data = psd(data, samplerate, fres, min_nfft, overlap_frac, **kwargs)
        multi_psd_data.append(psd_data)
    if not return_list:
        multi_psd_data = multi_psd_data[0]
    return multi_psd_data


def spectrogram(data, samplerate, freq_resolution=0.5, min_nfft=16, overlap_frac=0.5, **kwargs):
    """
    Spectrogram of a given frequency resolution.

    Check the returned frequency array for the actually used freqeuncy resolution.
    The frequency intervals are smaller or equal to `freq_resolution`.
    NFFT is `samplerate` divided by the actual frequency resolution.
    
    Parameters
    ----------
    data: array
        Data for the spectrogram.
    samplerate: float
        Samplerate of data in Hertz.
    freq_resolution: float
        Frequency resolution for the spectrogram.
    min_nfft: int
        Smallest value of nfft to be used.
    overlap_frac: float
        Overlap of the nffts (0 = no overlap; 1 = total overlap).
    kwargs: dict
        Further arguments for mlab.specgram().

    Returns
    -------
    spectrum: 2d array
        Contains for every timestamp the power of the frequencies listed in the array `freqs`.
    freqs: array
        Frequencies of the spectrogram.
    time: array
        Time of the nfft windows.
    """
    nfft, noverlap = nfft_noverlap(freq_resolution, samplerate, overlap_frac, min_nfft=min_nfft)
    spectrum, freqs, time = mlab.specgram(data, NFFT=nfft, Fs=samplerate,
                                          noverlap=noverlap, **kwargs)
    return spectrum, freqs, time


def plot_decibel_psd(ax, freqs, power, ref_power=1.0, min_power=1e-20,
                     max_freq=2000.0, **kwargs):
    """
    Plot the powerspectum in decibel relative to ref_power.

    Parameters
    ----------
    ax:
        Axis for plot.
    freqs: 1-D array
        Frequency array of the power spectrum.
    power: 1-D array
        Power values of the power spectrum.
    ref_power: float
        Reference power for computing decibel. If set to `None` the maximum power is used.
    min_power: float
        Power values smaller than `min_power` are set to `np.nan`.
    max_freq: float
        Limits of frequency axis are set to `(0, max_freq)` if `max_freq` is greater than zero
    kwargs:
        Plot parameter that are passed on to the `plot()` function.
    """
     
    decibel_psd = decibel(power, ref_power=ref_power, min_power=min_power)
    ax.plot(freqs, decibel_psd, **kwargs)
    ax.set_xlabel('Frequency [Hz]')
    if max_freq > 0.0:
        ax.set_xlim(0, max_freq)
    else:
        max_freq = freqs[-1]
    pmin = np.min(decibel_psd[np.isfinite(decibel_psd[freqs < max_freq])])
    pmin = np.floor(pmin / 10.0) * 10.0
    pmax = np.max(decibel_psd[np.isfinite(decibel_psd[freqs < max_freq])])
    pmax = np.ceil(pmax / 10.0) * 10.0
    ax.set_ylim(pmin, pmax)
    ax.set_ylabel('Power [dB]')


def peak_freqs(onsets, offsets, data, rate, freq_resolution=1.0, min_nfft=16, thresh=None):
    """Peak frequencies computed for each of the data snippets.

    Parameters
    ----------
    onsets: array of ints
        Indices indicating the onsets of the snippets in `data`.
    offsets: array of ints
        Indices indicating the offsets of the snippets in `data`.
    data: 1-D array
        Data array that contains the data snippets defined by `onsets` and `offsets`.
    rate: float
        Samplerate of data in Hertz.
    freq_resolution: float
        Desired frequency resolution of the computed power spectra in Hertz.
    min_nfft: int
        The smallest value of nfft to be used.
    thresh: None or float
        If not None than this is the threshold required for the minimum hight of the peak
        in the power spectrum. If the peak is too small than the peak frequency of
        that snippet is set to NaN.

    Returns
    -------
    freqs: array of floats
        For each data snippet the frequency of the maximum power.
    """
    freqs = []
    # for all events:
    for i0, i1 in zip(onsets, offsets):
        nfft, _ = nfft_noverlap(freq_resolution, rate, 0.5, min_nfft)
        if nfft > i1 - i0 :
            nfft = next_power_of_two((i1 - i0)/2)
        f, Pxx = sig.welch(data[i0:i1], fs=rate, nperseg=nfft, noverlap=nfft//2, nfft=None)
        if thresh is None:
            fpeak = f[np.argmax(Pxx)]
        else:
            p, _ = detect_peaks(decibel(Pxx, None), thresh)
            if len(p) > 0:
                ipeak = np.argmax(Pxx[p])
                fpeak = f[p[ipeak]]
            else:
                fpeak = float('NaN')
        freqs.append(fpeak)
    return np.array(freqs)


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    print('Compute powerspectra of two sine waves (300 and 450 Hz)')

    # generate data:
    fundamentals = [300, 450]  # Hz
    samplerate = 100000.0      # Hz
    time = np.arange(0.0, 8.0, 1.0/samplerate)
    data = np.sin(2*np.pi*fundamentals[0]*time) + 0.5*np.sin(2*np.pi*fundamentals[1]*time)

    # compute power spectra:
    fr = [0.5, 1]
    psd_data = multi_resolution_psd(data, samplerate, freq_resolution=fr)

    # plot power spectra:
    fig, ax = plt.subplots()
    for k in range(len(psd_data)):
        plot_decibel_psd(ax, psd_data[k][1], psd_data[k][0], lw=2,
                         label='$\\Delta f = %.1f$ Hz' % (np.mean(np.diff(psd_data[k][1]))))
    ax.legend(loc='upper right')
    plt.show()
