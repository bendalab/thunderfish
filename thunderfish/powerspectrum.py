"""
Powerspectra and spectrograms for a given frequency resolution

## Computation of nfft

- `next_power_of_two()`: round an integer up to the next power of two.
- `nfft()`: compute nfft based on a given frequency resolution.

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

- `add_multi_psd_config()`: add parameters for multi_psd() to configuration.
- `multi_psd_args()`: retrieve parameters for mulit_psd() from configuration.
"""

import numpy as np
from scipy.signal import get_window
try:
    from scipy.signal import welch
    psdscipy  = True
except ImportError:
    from matplotlib.mlab import psd as mpsd
    from matplotlib.mlab import detrend_linear, detrend_mean, detrend_none
    psdscipy  = False
try:
    from matplotlib.mlab import specgram as mspecgram
    specgrammlab = True
except ImportError:
    specgrammlab = False
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


def nfft(samplerate, freq_resolution, min_nfft=16, max_nfft=None):
    """Required number of samples for an FFT of a given frequency resolution.

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

    Returns
    -------
    nfft: int
        Number of FFT points.
    """
    nfft = next_power_of_two(samplerate / freq_resolution)
    if not max_nfft is None:
        if nfft > max_nfft:
            nfft = next_power_of_two(max_nfft//2 + 1)
    if nfft < min_nfft:
        nfft = min_nfft
    return nfft


def decibel(power, ref_power=1.0, min_power=1e-20):
    """
    Transform power to decibel relative to ref_power.
    \\[ decibel = 10 \\cdot \\log_{10}(power/ref\\_power) \\]
    Power values smaller than `min_power` are set to `-np.inf`.

    Parameters
    ----------
    power: float or array
        Power values, for example from a power spectrum or spectrogram.
    ref_power: float or None or 'peak'
        Reference power for computing decibel.
        If set to `None` or 'peak', the maximum power is used.
    min_power: float
        Power values smaller than `min_power` are set to `-np.inf`.

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
    if ref_power is None or ref_power == 'peak':
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
    \\[ power = ref\\_power \\cdot 10^{decibel/10} \\]
    
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


def psd(data, ratetime, freq_resolution, min_nfft=16, max_nfft=None,
        overlap_frac=0.5, detrend='constant', window='hanning'):
    """Power spectrum density of a given frequency resolution.

    NFFT is computed from the requested frequency resolution and the
    samplerate.  Check the returned frequency array for the actually
    used frequency resolution.  The frequency intervals are smaller or
    equal to `freq_resolution`.  NFFT can be retrieved by dividing
    `samplerate` by the actual frequency resolution:
    ```
    freq, power = psd(data, samplerate, 0.1)
    df = np.mean(np.diff(freq))  # the actual frequency resolution
    nfft = int(samplerate/df)
    ```

    Uses `scipy signal.welch()` if available, otherwise
    `matplotlib.mlab.psd()`.

    Parameters
    ----------
    data: 1-D array 
        Data from which power spectra are computed.
    ratetime: float or array
        If float, sampling rate of the data in Hertz.
        If array, assume `ratetime` to be the time vector corresponding to the data.
        Compute sampling rate as `1/(ratetime[1]-ratetime[0])`.
    freq_resolution: float
        Frequency resolution of the psd in Hertz.
    min_nfft: int
        Smallest value of nfft to be used.
    max_nfft: int or None
        If not None, largest value of nfft to be used.
    overlap_frac: float
        Fraction of overlap for the fft windows.
    detrend: string
        If 'constant' subtract mean of data.
        If 'linear' subtract line fitted to the data.
        If 'none' do not deternd the data.
    window: string
        Function used for windowing data segements.
        One of hanning, blackman, hamming, bartlett, boxcar, triang, parzen,
        bohman, blackmanharris, nuttall, fattop, barthann
        (see scipy.signal window functions).

    Returns
    -------
    freq: 1-D array
        Frequencies corresponding to power array.
    power: 1-D array
        Power spectral density in [data]^2/Hz.
    """
    samplerate = ratetime if np.isscalar(ratetime) else 1.0/(ratetime[1]-ratetime[0])
    n_fft = nfft(samplerate, freq_resolution, min_nfft, max_nfft)
    noverlap = int(n_fft * overlap_frac)
    if psdscipy:
        if detrend == 'none':
            detrend = lambda x: x
        freqs, power = welch(data, fs=samplerate, nperseg=n_fft, nfft=None,
                             noverlap=noverlap, detrend=detrend,
                             window=window, scaling='density')
    else:
        if detrend == 'linear':
            detrend_func = detrend_linear
        elif detrend == 'none':
            detrend_func = detrend_none
        else:
            detrend_func = detrend_mean
        power, freqs = mpsd(data, Fs=samplerate, NFFT=n_fft,
                                noverlap=noverlap, detrend=detrend_func,
                                window=get_window(window, n_fft),
                                scale_by_freq=True)
    # squeeze is necessary when n_fft is to large with respect to the data:
    return freqs, np.squeeze(power)


def multi_psd(data, ratetime, freq_resolution=0.5,
              num_resolutions=1, num_windows=1,
              min_nfft=16, overlap_frac=0.5,
              detrend='constant', window='hanning'):
    """Power spectra computed for consecutive data windows and
    mutiple frequency resolutions.

    See also psd() for more information on power spectra with given
    frequency resolution.

    Parameters
    ----------
    data: 1-D array
        Data from which power spectra are computed.
    ratetime: float or array
        If float, sampling rate of the data in Hertz.
        If array, assume `ratetime` to be the time vector corresponding to the data.
        Compute sampling rate as `1/(ratetime[1]-ratetime[0])`.
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
    detrend: string
        If 'constant' subtract mean of data.
        If 'linear' subtract line fitted to the data.
        If 'none' do not deternd the data.
    window: string
        Function used for windowing data segements.
        One of hanning, blackman, hamming, bartlett, boxcar, triang, parzen,
        bohman, blackmanharris, nuttall, fattop, barthann
        (see scipy.signal window functions).

    Returns
    -------
    multi_psd_data: list of 2-D arrays
        List of the power spectra for each window and frequency resolution
        (`psd_data[i][freq, power]`).
    """
    samplerate = ratetime if np.isscalar(ratetime) else 1.0/(ratetime[1]-ratetime[0])
    if not isinstance(freq_resolution, (list, tuple, np.ndarray)):
        freq_resolution = [freq_resolution]
        for i in range(1, num_resolutions):
            freq_resolution.append(2*freq_resolution[-1])
    n_incr = len(data)//(num_windows+1)  # overlap by half a window
    multi_psd_data = []
    for k in range(num_windows):
        for fres in freq_resolution:
            freq, power = psd(data[k*n_incr:(k+2)*n_incr], samplerate, fres,
                              min_nfft, 2*n_incr, overlap_frac, detrend, window)
            multi_psd_data.append(np.column_stack((freq, power)))
    return multi_psd_data


def spectrogram(data, ratetime, freq_resolution=0.5, min_nfft=16,
                max_nfft=None, overlap_frac=0.5,
                detrend='constant', window='hanning'):
    """
    Spectrogram of a given frequency resolution.

    Check the returned frequency array for the actually used frequency resolution.
    The frequency intervals are smaller or equal to `freq_resolution`.
    NFFT is `samplerate` divided by the actual frequency resolution.
    ```
    spec, freq, time = spectrum(data, samplerate, 0.1)
    df = np.mean(np.diff(freq))  # the actual frequency resolution
    nfft = int(samplerate/df)
    ```
    
    Parameters
    ----------
    data: array
        Data for the spectrogram.
    ratetime: float or array
        If float, sampling rate of the data in Hertz.
        If array, assume `ratetime` to be the time vector corresponding to the data.
        Compute sampling rate as `1/(ratetime[1]-ratetime[0])`.
    freq_resolution: float
        Frequency resolution for the spectrogram.
    min_nfft: int
        Smallest value of nfft to be used.
    max_nfft: int or None
        If not None, largest value of nfft to be used.
    overlap_frac: float
        Overlap of the nffts (0 = no overlap; 1 = total overlap).
    detrend: string
        If 'constant' subtract mean of data.
        If 'linear' subtract line fitted to the data.
    window: string
        Function used for windowing data segements.
        One of hanning, blackman, hamming, bartlett, boxcar, triang, parzen,
        bohman, blackmanharris, nuttall, fattop, barthann
        (see scipy.signal window functions).

    Returns
    -------
    spectrum: 2D array
        Power spectral density for each time and frequency.
    freqs: array
        Frequencies of the spectrogram.
    time: array
        Time of the nfft windows.
    """
    samplerate = ratetime if np.isscalar(ratetime) else 1.0/(ratetime[1]-ratetime[0])
    n_fft = nfft(samplerate, freq_resolution, min_nfft, max_nfft)
    noverlap = int(n_fft * overlap_frac)
    if specgrammlab:
        try:
            spec, freqs, time = mspecgram(data, NFFT=n_fft, Fs=samplerate,
                                          noverlap=noverlap, detrend=detrend,
                                          scale_by_freq=True, scale='linear',
                                          mode='psd',
                                          window=get_window(window, n_fft))
        except TypeError:
            spec, freqs, time = mspecgram(data, NFFT=n_fft, Fs=samplerate,
                                          noverlap=noverlap, detrend=detrend,
                                          scale_by_freq=True,
                                          window=get_window(window, n_fft))
        return spec, freqs, time
    else:
        # ... some alternative implementation ...
        return None, None, None


def plot_decibel_psd(ax, freqs, power, ref_power=1.0, min_power=1e-20,
                     log_freq=False, min_freq=0.0, max_freq=2000.0, ymarg=0.0, **kwargs):
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
    log_freq: boolean
        Logarithmic (True) or linear (False) frequency axis.
    min_freq: float
        Limits of frequency axis are set to `(min_freq, max_freq)`
        if `max_freq` is greater than zero
    max_freq: float
        Limits of frequency axis are set to `(min_freq, max_freq)`
        and limits of power axis are computed from powers below max_freq
        if `max_freq` is greater than zero
    ymarg: float
        Add this to the maximum decibel power for setting the ylim.
    kwargs: dict
        Plot parameter that are passed on to the `plot()` function.
    """
    decibel_psd = decibel(power, ref_power=ref_power, min_power=min_power)
    ax.plot(freqs, decibel_psd, **kwargs)
    ax.set_xlabel('Frequency [Hz]')
    if max_freq > 0.0:
        if log_freq and min_freq < 1.0:
            min_freq = 1.0
        ax.set_xlim(min_freq, max_freq)
    else:
        max_freq = freqs[-1]
    if log_freq:
        ax.set_xscale('log')
    dpmf = decibel_psd[freqs < max_freq]
    pmin = np.min(dpmf[np.isfinite(dpmf)])
    pmin = np.floor(pmin / 10.0) * 10.0
    pmax = np.max(dpmf[np.isfinite(dpmf)])
    pmax = np.ceil((pmax + ymarg) / 10.0) * 10.0
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
        if 'max_nfft' in kwargs:
            del kwargs['max_nfft']
        f, power = psd(data[i0:i1], samplerate, freq_resolution,
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
    psd_data = multi_psd(data, samplerate, freq_resolution=fr, detrend='none', window='hanning')

    # plot power spectra:
    fig, ax = plt.subplots()
    for k in range(len(psd_data)):
        df = np.mean(np.diff(psd_data[k][:,0]))
        nfft = int(samplerate/df)
        plot_decibel_psd(ax, psd_data[k][:,0], psd_data[k][:,1], lw=2,
                         label='$\\Delta f = %.1f$ Hz, nnft=%d' % (df, nfft))
    ax.legend(loc='upper right')
    plt.show()
