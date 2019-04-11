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
- `multi_psd()`: power spectra for consecutive data windows and mutiple frequency resolutions.
- `spectrogram()`: spectrogram of a given frequency resolution and overlap fraction.

## Power spectrum analysis
- `peak_freqs()`: peak frequencies computed for each of the data snippets.

## Visualization
- `plot_decibel_psd()`: plot power spectrum in decibel.

## Configuration parameter
- `add_multi_psd_config()': add parameters for multi_psd() to configuration.
- `multi_psd_args()`: retrieve parameters for mulit_psd() from configuration.
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


def nfft_noverlap(samplerate, freq_resolution,
                  min_nfft=16, max_nfft=None, overlap_frac=0.5):
    """Required number of samples for an FFT of a given frequency resolution
    and number of overlapping data points.

    Note that the returned number of FFT samples results
    in frequency intervals that are smaller or equal to `freq_resolution`.

    Parameters
    ----------
    samplerate: float
        Sampling rate of the data in Hertz.
    freq_resolution: float
        Minimum frequency resolution in Hertz.
    min_nfft: int
        Smallest value of nfft to be used.
    max_nfft: int or None
        If not None, largest value of nfft to be used.
    overlap_frac: float
        Fraction the FFT windows should overlap.

    Returns
    -------
    nfft: int
        Number of FFT points.
    noverlap: int
        Number of overlapping FFT points.
    """
    nfft = next_power_of_two(samplerate / freq_resolution)
    if not max_nfft is None:
        if nfft > max_nfft:
            nfft = next_power_of_two(max_nfft//2 + 1)
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
    ref_power: float or None
        Reference power for computing decibel.
        If set to `None` the maximum power is used.
    min_power: float
        Power values smaller than `min_power` are set to `-inf`.

    Returns
    -------
    decibel_psd: array
        Power values in decibel relative to `ref_power`.
    """
    if isinstance(power, (list, tuple, np.ndarray)):
        tmp_power = power
        decibel_psd = power.copy()
    else:
        tmp_power = np.array([power])
        decibel_psd = np.array([power])
    if ref_power is None:
        ref_power = np.max(decibel_psd)
    decibel_psd[tmp_power <= min_power] = float('-inf')
    decibel_psd[tmp_power > min_power] = 10.0 * np.log10(decibel_psd[tmp_power > min_power]/ref_power)
    if isinstance(power, (list, tuple, np.ndarray)):
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


def psd(data, samplerate, freq_resolution, min_nfft=16, max_nfft=None,
        overlap_frac=0.5, **kwargs):
    """Power spectrum density of a given frequency resolution.

    NFFT is computed from the requested frequency resolution and the samplerate.
    Check the returned frequency array for the actually used freqeuncy resolution.
    The frequency intervals are smaller or equal to `freq_resolution`.
    NFFT is `samplerate` divided by the actual frequency resolution.
    
    data: 1-D array 
        Data from which power spectra are computed.
    samplerate: float
        Sampling rate of the data in Hertz.
    freq_resolution: float
        Frequency resolution of the psd in Hertz.
    min_nfft: int
        Smallest value of nfft to be used.
    max_nfft: int or None
        If not None, largest value of nfft to be used.
    overlap_frac: float
        Fraction of overlap for the fft windows.
    kwargs: dict
        Further arguments for mlab.psd().

    Returns
    -------
    power: 1-D array
        Power.
    freq: 1-D array
        Frequencies corresponding to power array.
    """
    nfft, noverlap = nfft_noverlap(samplerate, freq_resolution,
                                   min_nfft, max_nfft, overlap_frac)
    for k in ['Fs', 'NFFT', 'noverlap']:
        if k in kwargs:
            del kwargs[k]
    power, freqs = mlab.psd(data, Fs=samplerate, NFFT=nfft,
                            noverlap=noverlap, **kwargs)
    # f, Pxx = sig.welch(data, fs=samplerate, nperseg=nfft, nfft=None,
    #                    noverlap=noverlap, **kwargs)
    # squeeze is necessary when nfft is to large with respect to the data:
    return np.squeeze(power), freqs


def multi_psd(data, samplerate, freq_resolution=0.5,
              num_resolutions=1, num_windows=1,
              min_nfft=16, overlap_frac=0.5, **kwargs):
    """Power spectra computed for consecutive data windows and
    mutiple frequency resolutions.

    Parameters
    ----------
    data: 1-D array
        Data from which power spectra are computed.
    samplerate: float
        Sampling rate of the data in Hertz.
    freq_resolution: float or 1-D array
        Frequency resolutions for one or multiple psds in Hertz.
    num_resolutions: int
        If freq_resolution is a single number,
        then generate `num_resolutions` frequency resolutions
        starting with `freq_resolution` und subsequently multiplied by two.
    num_windows: int
        Data are chopped into `num_windows` segments that overlap by half
        for which power spectra are computed.
    min_nfft: int
        Smallest value of nfft to be used.
    overlap_frac: float
        Fraction of overlap for the fft windows within a single power spectrum.
    kwargs: dict
        Further arguments for mlab.psd().

    Returns
    -------
    multi_psd_data: list of 2-D arrays
        List of the power spectra for each window and frequency resolution
        (`psd_data[i][power, freq]`).
    """
    if not isinstance(freq_resolution, (list, tuple, np.ndarray)):
        freq_resolution = [freq_resolution]
        for i in range(1, num_resolutions):
            freq_resolution.append(2*freq_resolution[-1])
    n_incr = len(data)//(num_windows+1)  # overlap by half a window
    multi_psd_data = []
    for k in range(num_windows):
        for fres in freq_resolution:
            power, freq = psd(data[k*n_incr:(k+2)*n_incr], samplerate, fres,
                              min_nfft, 2*n_incr, overlap_frac, **kwargs)
            multi_psd_data.append(np.asarray([power, freq]))
    return multi_psd_data


def spectrogram(data, samplerate, freq_resolution=0.5, min_nfft=16,
                max_nfft=None, overlap_frac=0.5, **kwargs):
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
    max_nfft: int or None
        If not None, largest value of nfft to be used.
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
    nfft, noverlap = nfft_noverlap(samplerate, freq_resolution,
                                   min_nfft, max_nfft, overlap_frac)
    for k in ['Fs', 'NFFT', 'noverlap']:
        if k in kwargs:
            del kwargs[k]
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


def peak_freqs(onsets, offsets, data, samplerate, freq_resolution=1.0,
               thresh=None, **kwargs):
    """Peak frequencies computed for each of the data snippets.

    Parameters
    ----------
    onsets: array of ints
        Indices indicating the onsets of the snippets in `data`.
    offsets: array of ints
        Indices indicating the offsets of the snippets in `data`.
    data: 1-D array
        Data array that contains the data snippets defined by `onsets` and `offsets`.
    samplerate: float
        Samplerate of data in Hertz.
    freq_resolution: float
        Desired frequency resolution of the computed power spectra in Hertz.
    thresh: None or float
        If not None than this is the threshold required for the minimum hight of the peak
        in the power spectrum. If the peak is too small than the peak frequency of
        that snippet is set to NaN.
    kwargs: dict
        Further arguments passed on to psd().

    Returns
    -------
    freqs: array of floats
        For each data snippet the frequency of the maximum power.
    """
    freqs = []
    for i0, i1 in zip(onsets, offsets):
        power, f = psd(data[i0:i1], samplerate, freq_resolution,
                       max_nfft=i1-i0, **kwargs)
        if thresh is None:
            fpeak = f[np.argmax(power)]
        else:
            p, _ = detect_peaks(decibel(power, None), thresh)
            if len(p) > 0:
                ipeak = np.argmax(power[p])
                fpeak = f[p[ipeak]]
            else:
                fpeak = float('NaN')
        freqs.append(fpeak)
    return np.array(freqs)


def add_multi_psd_config(cfg, freq_resolution=0.5,
                         num_resolutions=1, num_windows=1):
    """ Add all parameters needed for the multi_psd() function as
    a new section to a configuration.

    Parameters
    ----------
    cfg: ConfigFile
        The configuration.
        
    See multi_psd() for details on the remaining arguments.
    """
    cfg.add_section('Power spectrum estimation:')
    cfg.add('frequencyResolution', freq_resolution, 'Hz', 'Frequency resolution of the power spectrum.')
    cfg.add('numberPSDWindows', num_resolutions, '', 'Number of windows on which power spectra are computed.')
    cfg.add('numberPSDResolutions', num_windows, '', 'Number of power spectra computed within each window with decreasing resolution.')


def multi_psd_args(cfg):
    """ Translates a configuration to the
    respective parameter names of the multi_psd() function.
    
    The return value can then be passed as key-word arguments to
    this functions.

    Parameters
    ----------
    cfg: ConfigFile
        The configuration.

    Returns
    -------
    a: dict
        Dictionary with names of arguments of the multi_psd() function
        and their values as supplied by `cfg`.
    """
    a = cfg.map({'freq_resolution': 'frequencyResolution',
                 'num_resolutions': 'numberPSDWindows',
                 'num_windows': 'numberPSDResolutions'})
    return a


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
    psd_data = multi_psd(data, samplerate, freq_resolution=fr)

    # plot power spectra:
    fig, ax = plt.subplots()
    for k in range(len(psd_data)):
        plot_decibel_psd(ax, psd_data[k][1], psd_data[k][0], lw=2,
                         label='$\\Delta f = %.1f$ Hz' % (np.mean(np.diff(psd_data[k][1]))))
    ax.legend(loc='upper right')
    plt.show()
