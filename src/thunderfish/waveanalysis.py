"""
Analysis of wave-type EOD waveforms.

## Analysis of wave-type EODs

- `extract_wave()`: retrieve average EOD waveform via Fourier transform.
- `condition_wave()`: subtract offset and flip wave-type EOD waveform.
- `analyze_wave_properties()`: characterize basic properties of a wave-type EOD.
- `analyse_wave_spectrum()`: analyze the spectrum of a wave-type EOD.
- `analyze_wave()`: analyze the EOD waveform of a wave fish.

## Visualization

- `plot_wave_spectrum()`: plot and annotate spectrum of wave EODs.

## Storage

- `save_wave_eodfs()`: save frequencies of wave EODs to file.
- `load_wave_eodfs()`: load frequencies of wave EODs from file.
- `save_wave_fish()`: save properties of wave EODs to file.
- `load_wave_fish()`: load properties of wave EODs from file.
- `save_wave_spectrum()`: save amplitude and phase spectrum of wave EOD to file.
- `load_wave_spectrum()`: load amplitude and phase spectrum of wave EOD from file.
"""

import numpy as np

try:
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MultipleLocator
    from matplotlib.artist import setp
except ImportError:
    pass

from pathlib import Path
from scipy.optimize import curve_fit
from numba import jit
from thunderlab.eventdetection import threshold_crossings, threshold_crossing_times, merge_events
from thunderlab.fourier import fourier_coeffs, normalize_fourier_coeffs
from thunderlab.fourier import fourier_synthesis
from thunderlab.powerspectrum import decibel
from thunderlab.tabledata import TableData

from .harmonics import fundamental_freqs_and_power


def extract_wave(data, rate, freq, freq_resolution, periods=5,
                 frate=1e6, n_harmonics=21,
                 min_corr=0.99, min_ampl_frac=0.5,
                 verbose=0, plot_level=0):
    """Retrieve average EOD waveform via Fourier transform.

    Fourier series are extracted for frequencies from `freq` \\(\\pm\\)
    `freq_resolution`.

    Parameters
    ----------
    data: 1-D array of float
        The data to be analysed.
    rate: float
        Sampling rate of the data in Hertz.
    freq: float
        Estimated EOD frequency.
    freq_resolution: float
        Frequency resolution in Hertz. Ususally the resolution of the power spectrum
        from which `freq` was deduced.
    periods: float
        The snippet size is the EOD period (`1/freq`)  times `periods`.
    frate: float
        Sampling rate used for the waveform estimates.
    n_harmonics: int
        Number of harmonics used for the Fourier decomposition.
    min_corr: float
        Minimum required correlation between two waveform estimates.
    min_ampl_fac: float
        Minimum required standard deviation of waveform estimate relative
        to the largest one.
    verbose: int
        Verbosity level.
    plot_level: int
        When larger than zero, plot some debug information.
    
    Returns
    -------
    mean_coeffs: 1-D array of complex
        The averaged Fourier coefficients of the extracted EOD waveform.
        They correspond to the returnd `mean_eod`.
    mean_eod: 2-D array of float
        Average of the EOD snippets. First column is time in seconds,
        second column the mean eod, third column the standard error.
    eod_freq: float
        Refined EOD frequency.
    times: 1-D array
        Start times of data segments in which Fourier coefficients
        have been extracted.
    n_eods: int
        Number of EODs that went into the estimate of the mean waveform.
    skip_reason: str
        An empty string if the waveform is good, otherwise a string
        indicating the failure.

    """

    def fourier_freq_range(data, rate, frange, nh, n):
        """ Extract Fourier coefficients for some frequencies and
        return waveform with largest p-p amplitude.
        """
        wave = np.zeros(1)
        freq = 0.0
        for f in frange:
            twave = np.arange(n)/rate
            w = np.zeros(len(twave))
            t = np.arange(len(data))/rate
            for k in range(nh):
                Xk = np.trapz(data*np.exp(-2j*np.pi*k*f*t), t)*2/t[-1]
                w += np.real(Xk*np.exp(2j*np.pi*k*f*twave))
            if np.max(w) - np.min(w) > np.max(wave) - np.min(wave):
                wave = w
                freq = f
        return wave, freq

    # reduce frequency resolution and time window for high frequency fish:
    ffac = max(1, int(np.round(freq/400)))
    freq_resolution *= ffac
    t_segment = min(len(data)/rate, 2/freq_resolution) # twice as long window is essential!
    nfreqs = 1 + 2*2*ffac         # twice the frequency resolution is necessary and sufficient!
    step = max(16, int(t_segment*rate))
    # extract Fourier series from data segements:
    n = int(periods/freq*rate)
    freqs = []
    indices = np.arange(0, max(1, len(data) - step + 1), max(1, step//8))
    if len(indices) <= 1:
        t_segment /= 2
        step = max(8, int(t_segment*rate))
        indices = np.arange(0, max(1, len(data) - step + 1), max(1, step//8))
    times = indices/rate
    frange = np.linspace(freq - freq_resolution, freq + freq_resolution, nfreqs)
    for i in indices:
        w, f = fourier_freq_range(data[i:i + step], rate, frange, 6, n)
        freqs.append(f)
    freqs = np.array(freqs)
    mean_coeffs = np.zeros(0, dtype=complex)
    mean_eod = np.zeros((0, 3))
    if len(freqs) == 0:
        # TODO: Why??? How can indices be empty?
        return mean_coeffs, mean_eod, freq, np.array([]), 0, f'no frequencies detected ({len(indicies)} indices, freqs={freqs})'
    """
    # just take the frequencies from the spectrum and keep the segement size:
    # this does not perform well in mutli-fish settings!
    # the improved frequency resolution seems to be essential!
    t_segment = min(len(data)/rate, 1/freq_resolution)
    step = max(16, int(t_segment*rate))
    indices = np.arange(0, max(1, len(data) - step + 1), max(1, step//8))
    times = indices/rate
    freqs = np.ones(len(indices))*freq
    """
    # refined Fourier series and waveforms:
    n = int(periods/np.mean(freqs)*frate)
    coeffs = np.zeros((len(indices), n_harmonics), dtype=complex)
    waves = np.zeros((len(indices), n))
    for k in range(len(indices)):
        i = indices[k]
        c = fourier_coeffs(data[i:i + step], rate, freqs[k],
                           n_harmonics)
        c = normalize_fourier_coeffs(c)
        w = fourier_synthesis(freqs[k], c, frate, n)
        coeffs[k] = c
        waves[k] = w
    if plot_level > 0:
        fig, axs = plt.subplots(2, 3, width_ratios=[8, 8, 1],
                                layout='constrained')
        axs[1, 2].set_visible(False)
        axs[0, 0].set_title(f'EODf={np.mean(freqs):.2f}Hz')
        t = np.arange(waves.shape[1])*1000/frate
        for w in waves:
            axs[0, 0].plot(t, w)
        axs[0, 0].set_xlabel('time [ms]')
    # only snippets that are most similar:
    if len(waves) <= 1:
        eodf = np.mean(freqs) if len(freqs) > 0 else freq
        print(f'extract {freq:7.2f}Hz wave  fish: {len(waves)} segments, EODf={eodf:.2f}Hz')
        # TODO: what to do with single segment?
    else:
        corr = np.corrcoef(waves)
        np.fill_diagonal(corr, 0.0)
        corr_vals = np.sort(corr[corr > min_corr])
        if len(corr_vals) == 0:
            if plot_level > 0:
                plt.show()
            return mean_coeffs, mean_eod, freq, times, 0, f'waveforms not stable (max_corr={np.max(corr):.4f} SMALLER than {min_corr:.4f})'
        # higher correlation threshold:
        min_c = corr_vals[len(corr_vals)//2]
        # number of high correlations for each segment:
        num_c = np.sum(corr > min_c, axis=1)
        num_cmax = np.max(num_c)
        num_cthresh = max(1, num_cmax//2)
        # collect pairs with high correlations,
        # to stay within a single cluster in the correlation matrix.
        mask = []
        sub_corr = np.array(corr)
        # pair with maximum correlation:
        while not np.all(np.isnan(sub_corr)):
            i_max_corr = np.nanargmax(sub_corr)
            max_corr_a = i_max_corr//len(corr)
            max_corr_b = i_max_corr%len(corr)
            if corr[max_corr_a, max_corr_b] < max(min_c, 1- 0.25*(1 - min_corr)):
                break
            sub_corr[:, max_corr_a] = np.nan
            sub_corr[max_corr_a, :] = np.nan
            sub_corr[:, max_corr_b] = np.nan
            sub_corr[max_corr_b, :] = np.nan
            if num_c[max_corr_a] > num_cthresh:
                mask.append(max_corr_a)
            if num_c[max_corr_b] > num_cthresh:
                mask.append(max_corr_b)
            for k in range(len(corr)):
                if k >= len(mask):
                    break
                corr_col = corr[:, mask[k]]
                idx = np.argsort(corr_col)[::-1]
                for i in idx:
                    if corr_col[i] > min_c and num_c[i] > num_cthresh and \
                       i not in mask:
                        mask.append(i)
                        break
            # if only one was selected, add pair:
            if len(mask) == 1:
                mask.append(np.argmax(corr[:, mask[0]]))
            if len(mask) > 0:
                break
        
        waves = waves[mask]
        coeffs = coeffs[mask]
        freqs = freqs[mask]
        times = times[mask]
        if plot_level > 0:
            axs[0, 1].set_title('Correlations')
            m = axs[0, 1].pcolormesh(corr, vmin=min_corr, vmax=1)
            axs[0, 1].plot(i_max_corr%len(corr) + 0.5,
                           i_max_corr//len(corr) + 0.5, 'ok')
            axs[0, 1].plot(i_max_corr//len(corr) + 0.5,
                           i_max_corr%len(corr) + 0.5, 'ok')
            axs[0, 1].set_aspect('equal')
            axs[0, 1].set_xlabel('segment')
            axs[0, 1].set_ylabel('segment')
            plt.colorbar(m, axs[0, 2], label='correlation coefficient')
            selected = np.zeros(len(corr))
            selected[mask] = 1
            axs[1, 1].axhline(num_cthresh + 0.5, color='k',
                              label='num_cthresh')
            axs[1, 1].plot(np.arange(len(num_c)), num_c, '-o',
                           label='num_c')
            axs[1, 1].plot(np.arange(len(num_c))[mask], num_c[mask], 'o',
                           label='selected')
            axs[1, 1].plot(np.arange(len(selected)), selected, '-o',
                           label='mask')
            axs[1, 1].set_ylim(bottom=0)
            axs[1, 1].set_xlabel('segment')
            axs[1, 1].set_ylabel('num_c')
            axs[1, 1].xaxis.set_major_locator(plt.MultipleLocator(1))
            axs[1, 1].yaxis.set_major_locator(plt.MultipleLocator(1))
            axs[1, 1].legend()
        if verbose > 0:
            eodf = np.mean(freqs) if len(freqs) > 0 else np.nan
            with np.printoptions(formatter={'float': lambda x: f'{x:.2f}'},
                                 linewidth=10000):
                print(f'extract {freq:7.2f}Hz wave  fish: min_corr={min_c:.4f}, max_corr={corr_vals[-1]:.4f}, num_cmax={num_cmax}, segments={len(corr)}, num_selected={len(mask)}, selected={mask}, EODfs={freqs}, EODf={eodf:.2f}Hz')
        if len(waves) == 0:
            if plot_level > 0:
                plt.show()
            return mean_coeffs, mean_eod, freq, times, 0, f'waveforms not stable (min_corr={min_c:.4f}, max_corr={corr_vals[-1]:.4f}, num_cmax={num_cmax})'
    # only the largest snippets:
    ampls = np.std(waves, axis=1)
    mask = ampls >= min_ampl_frac*np.max(ampls)
    if verbose > 0 and np.sum(mask) < len(ampls):
        print(f'                              removed {len(ampls) - np.sum(mask)} small amplitude segments')
    waves = waves[mask]
    coeffs = coeffs[mask]
    freqs = freqs[mask]
    times = times[mask]
    if len(waves) == 0:
        if plot_level > 0:
            plt.show()
        return mean_coeffs, mean_eod, freq, times, 0, 'no large waveform'
    if plot_level > 0:
        axs[1, 0].set_title(f'selected EODs: EODf={np.mean(freqs):.2f}Hz')
        t = np.arange(waves.shape[1])*1000/frate
        for w in waves:
            axs[1, 0].plot(t, w)
        axs[1, 0].set_xlabel('time [ms]')
        plt.show()

    eod_freq = np.mean(freqs)
    mean_coeffs = np.mean(coeffs, 0)
    mean_coeffs[0] = 0  # no offset
    mean_eod = np.zeros((n, 3))
    mean_eod[:, 0] = np.arange(len(mean_eod))/frate
    mean_eod[:, 1] = np.mean(waves, axis=0)
    mean_eod[:, 1] = fourier_synthesis(eod_freq, mean_coeffs, frate, n)
    mean_eod[:, 2] = np.std(waves, axis=0)
    t_covered = 0
    t_end = -1
    for t in times:
        if t_end < t:
            t_end = t
            t_covered += t + t_segment - t_end
        t_end = t + t_segment
    n_eods = int(t_covered*eod_freq)
    return mean_coeffs, mean_eod, eod_freq, times, n_eods, ''


def condition_wave(eod, ratetime, freq, coeffs=None, flip='none'):
    """Subtract offset and flip wave-type EOD waveform.
    
    Parameters
    ----------
    eod: 1-D or 2-D array
        The eod waveform to be analyzed.  If an 1-D array, then this
        is the waveform and you need to also pass a time array or
        sampling rate in `ratetime`.  If a 2-D array, then first
        column is time in seconds, second column the EOD
        waveform, and the last column - if present - is the waveform
        obtained from Fourier decomposition. Allother columns are optional
        and are not used.
    ratetime: None or float or array of float
        If a 1-D array is passed on to `eod` then either the sampling
        rate in Hertz or the time array corresponding to `eod`.
    freq: float
        The frequency of the EOD.
    coeffs: None or 1-D array of complex
        The Fourier coefficients of an EOD waveform.
    flip: 'auto', 'none', 'flip'
        - 'auto' flip waveform such that the larger extremum is positive.
        - 'flip' flip waveform.
        - 'none' do not flip waveform.

    Returns
    -------
    eod: 1-D or 2-D array
        Same shape as the input `eod`.
        If no `coeffs` are provided, the mean over integer multiples of
        the period of the waveform was subtracted.
        If waveform was flipped, it was flipped in the second and last column.
        If a last column is present, the waveform in the last column
        is recomputed from the flipped FOurier coefficients.
    coeffs: None or 1-D array of complex
        The Fourier coefficients of an EOD waveform with \\(\\pi\\)
        added to the phases if waveform has been flipped.
        The Fourier coefficients are then normalized such
        that the phase of the fundamental is zero.
        This is only returned, if `coeffs` is not None.
    flipped: bool
        True if waveform was flipped.
    """
    if eod.ndim == 2:
        rate = 1.0/(eod[1, 0] - eod[0, 0])
        eodw = eod[:, -1] if eod.shape[1] > 2 else eod[:, 1]
    else:
        eodw = eod
        if isinstance(ratetime, (list, tuple, np.ndarray)):
            rate = 1.0/(ratetime[1] - ratetime[0])
        else:
            rate = ratetime
        
    # subtract mean:
    if coeffs is None:
        pinx = int(np.ceil(rate/freq)) # one period
        maxn = (len(eodw)//pinx)*pinx   # integer multiple of period
        if maxn < pinx:
            maxn = len(eodw)
        offs = (len(eodw) - maxn)//2    # center
        if eod.ndim == 2:
            eod -= np.mean(eod[offs:offs + pinx])
        else:
            eod[:, 1] -= np.mean(eod[offs:offs + pinx, 1])

    # flip:
    flipped = False
    if flip.lower() in ['flip', 'true', 'yes']:
        flipped = True
    elif flip.lower() == 'auto':
        if -np.min(eodw) > np.max(eodw):
            flipped = True
    if flipped:
        if eod.ndim == 2:
            eod[:, 1] *= -1
            if eod.shape[1] > 2:
                eod[:, -1] *= -1
        else:
            eod *= -1
        if coeffs is not None:
            coeffs *= np.exp(1j*np.pi)
            coeffs = normalize_fourier_coeffs(coeffs)
            if eod.ndim == 2 and eod.shape[1] > 2:
                eod[:, -1] = fourier_synthesis(freq, coeffs, eod[:, 0])

    # return:
    if coeffs is not None:
        return eod, coeffs, flipped
    else:
        return eod, flipped


def analyze_wave_properties(eod, ratetime, freq):
    """Characterize basic properties of a wave-type EOD.
    
    Parameters
    ----------
    eod: 1-D or 2-D array
        The eod waveform to be analyzed.  If an 1-D array, then this
        is the waveform and you need to also pass a time array or
        sampling rate in `ratetime`.  If a 2-D array, then first
        column is time in seconds, second column the EOD
        waveform, and the last column - if present - is the waveform
        obtained from Fourier decomposition. Allother columns are optional
        and are not used.
    ratetime: None or float or array of float
        If a 1-D array is passed on to `eod` then either the sampling
        rate in Hertz or the time array corresponding to `eod`.
    freq: float
        The frequency of the EOD.

    Returns
    -------
    pos_ampl: float
        Amplitude of largest positive peak.
    neg_ampl: float
        Amplitude of largest negative trough (absolute value).
    distance: float
        Temporal distance between largest negative trough and positive peak.
    min_distance: float
        Temporal distance between largest negative trough and positive peak
        normalized to first half of the EOD cycle.
        That is, if `distance` is larger than half an EOD cycle, then the
        duration of an EOD cycle minus `distnace`.
    """
    if eod.ndim == 2:
        time = eod[:, 0]
        eod = eod[:, -1] if eod.shape[1] > 2 else eod[:, 1]
    else:
        eod = eod
        if isinstance(ratetime, (list, tuple, np.ndarray)):
            time = ratetime
        else:
            time = np.arange(len(eod))/ratetime
    deltat = np.mean(np.diff(time))
    
    # indices of exactly one period:
    period = 1/freq
    pinx = int(np.ceil(period/deltat)) # one period
    if len(eod) < 2*pinx:
        raise IndexError('data need to contain at least two EOD periods')
    i0 = (len(eod) - pinx)//2
    i1 = i0 + pinx
    eodp = eod[i0:i1]
    timep = time[i0:i1]

    # amplitudes:
    pos_idx = np.argmax(eodp)
    pos_ampl = abs(eodp[pos_idx])
    neg_idx = np.argmin(eodp)
    neg_ampl = abs(eodp[neg_idx])
    distance = abs(timep[neg_idx] - timep[pos_idx])
    min_distance = distance
    if distance > period/2:
        min_distance = period - distance

    """
    # zero crossings:
    ui, di = threshold_crossings(meod[:, 1], 0.0)
    ut, dt = threshold_crossing_times(meod[:, 0], meod[:, 1], 0.0, ui, di)
    ut, dt = merge_events(ut, dt, 0.02/freq1)
    ncrossings = int(np.round((len(ut) + len(dt))/(meod[-1,0]-meod[0,0])/freq1))
    if np.any(ut<0.0):    
        up_time = ut[ut<0.0][-1]
    else:
        up_time = 0.0 
        error_str += '%.1f Hz wave fish: no upward zero crossing. ' % freq1
    if np.any(dt>0.0):
        down_time = dt[dt>0.0][0]
    else:
        down_time = 0.0
        error_str += '%.1f Hz wave fish: no downward zero crossing. ' % freq1
    peak_width = down_time - up_time
    trough_width = period - peak_width
    peak_time = 0.0
    trough_time = meod[maxinx + np.argmin(meod[maxinx:maxinx + pinx, 1]), 0] - meod[maxinx, 0]
    phase1 = peak_time - up_time
    phase2 = down_time - peak_time
    phase3 = trough_time - down_time
    phase4 = up_time + period - trough_time

    # peak-to-peak and trough amplitudes:
    ppampl = np.max(eodw[i0:i1]) - np.min(eodw[i0:i1])
    relpeakampl = max(np.max(eodw[i0:i1]), np.abs(np.min(eodw[i0:i1])))/ppampl
    """
    
    return pos_ampl, neg_ampl, distance, min_distance
    

    
def analyse_wave_spectrum(freq, coeffs, power_add_harmonics=3):
    """Analyze the spectrum of a wave-type EOD.
    
    Parameters
    ----------
    freq: float or 2-D array
        The frequency of the EOD or the list of harmonics (rows) with
        frequency and peak height (columns) as returned from
        `harmonics.harmonic_groups()`.
    coeffs: None or 1-D array of complex
        The Fourier coefficients of an EOD waveform.
        If provided, they are taken for the spectrum and the waveform
        is updated from them.
    power_add_harmonics: int
        The maximum power of higher harmonics is computed from
        harmonics higher than the maximum harmonics within the first
        three harmonics plus `power_add_harmonics`.
    
    Returns
    -------
    spec: 2-D array of floats
        First six columns are from the spectrum of the extracted
        waveform.  First column is the harmonics (fundamental is one),
        second column its frequency, third column its amplitude, fourth
        column its amplitude relative to the fundamental, fifth column
        is power of harmonics relative to fundamental in decibel, and
        sixth column the phase shift relative to the fundamental.
        If `freq` is a list of harmonics, a seventh column is added to
        `spec_data` that contains the powers of the harmonics from the
        original power spectrum of the raw data.  Rows are the
        harmonics, first row is the fundamental frequency with index
        0, relative amplitude of one, relative power of 0dB, and phase
        shift of zero.

    power: float
        Total power, i.e. sum of squared Fourier amplitudes.
    data_power: float or None
        Total power (sum of data powers) in the data, if available.
        Only sum over as many harmonics as we have Fourier coefficients.
    thd: float
        Total harmonic distortion. Square root of the sum of the squared
        amplitudes of all harmonics relativ to the amplitude of
        the fundamental. 
    db_diff: float
        Standard deviation of the differences of the decibel powers.
        As a measure of smoothness of the spectrum.
    max_harmonics_power: float
        Maximum power in decibel ofhigher harmonics.
    """
    if hasattr(freq, 'shape'):
        freq1 = freq[0][0]
        n = len(coeffs) - 1
        n += np.sum(freq[:, 0] > (len(coeffs) - 0.5)*freq1)
        spec = np.zeros((n, 7))
        spec[:, :] = np.nan
        k = 0
        for i in range(len(coeffs) - 1):
            while k < len(freq) and freq[k, 0] < (i - 0.5)*freq1:
                k += 1
            if k >= len(freq):
                break
            if freq[k, 0] < (i + 0.5)*freq1:
                spec[i, 6] = freq[k, 1]
                k += 1
        for i in range(len(coeffs) - 1, n):
            if k >= len(freq):
                break
            spec[i, :2] = [np.round(freq[k, 0]/freq1), freq[k, 0]]
            spec[i, 6] = freq[k, 1]
            k += 1
    else:
        freq1 = freq
        spec = np.zeros((len(coeffs) - 1, 6))
    ampl1 = np.abs(coeffs[1])
    for i in range(1, min(len(coeffs), len(spec) + 1)):
        ampl = np.abs(coeffs[i])
        phase = np.angle(coeffs[i])
        spec[i - 1, :6] = [i, i*freq1, ampl, ampl/ampl1,
                           decibel((ampl/ampl1)**2.0), phase]

    # harmonics, frequency, amplitude, relative amplitude, power, phase, data power

    # total power:
    pnh = len(coeffs) - 1
    power = decibel(np.sum(spec[:pnh, 2]**2))
    data_power = None
    if spec.shape[1] > 6:
        data_power = decibel(np.nansum(spec[:pnh, 6]))
        
    # smoothness of power spectrum:
    db_powers = decibel(spec[:len(coeffs) - 1, 2]**2)
    db_diff = np.std(np.diff(db_powers))
    # maximum relative power of higher harmonics:
    p_max = np.argmax(db_powers[:3])
    db_powers -= db_powers[p_max]
    if len(db_powers[p_max + power_add_harmonics:]) == 0:
        max_harmonics_power = -100.0
    else:
        max_harmonics_power = np.max(db_powers[p_max + power_add_harmonics:])
    # total harmonic distortion:
    thd = np.sqrt(np.nansum(spec[1:, 3])**2)

    return spec, power, data_power, thd, db_diff, max_harmonics_power

    
def analyze_wave(eod, ratetime, freq, coeffs=None, n_harmonics=21,
                 power_add_harmonics=3, flip='none'):
    """Analyze the EOD waveform of a wave fish.
    
    Parameters
    ----------
    eod: 1-D or 2-D array
        The eod waveform to be analyzed.
        If an 1-D array, then this is the waveform and you
        need to also pass a time array or sampling rate in `ratetime`.
        If a 2-D array, then first column is time in seconds, second
        column the EOD waveform, third column, if present, is the
        standard error of the EOD waveform. Further columns are
        optional and are not used.
    ratetime: None or float or array of float
        If a 1-D array is passed on to `eod` then either the sampling
        rate in Hertz or the time array corresponding to `eod`.
    freq: float or 2-D array
        The frequency of the EOD or the list of harmonics (rows) with
        frequency and peak height (columns) as returned from
        `harmonics.harmonic_groups()`.
    coeffs: None or 1-D array of complex
        The Fourier coefficients of an EOD waveform.
        If provided, they are taken for the spectrum and the waveform
        is updated from them.
    n_harmonics: int
        Number of harmonics used for the Fourier decomposition.
    power_add_harmonics: int
        The maximum power of higher harmonics is computed from
        harmonics higher than the maximum harmonics within the first
        three harmonics plus `power_add_harmonics`.
    flip: 'auto', 'none', 'flip'
        - 'auto' flip waveform such that the larger extremum is positive.
        - 'flip' flip waveform.
        - 'none' do not flip waveform.
    
    Returns
    -------
    meod: 2-D array of floats
        The eod waveform. First column is time in seconds, second
        column the eod waveform.  Further columns are kept from the
        input `eod`. And a column is added with the fit of the fourier
        series to the waveform.
    props: dict
        A dictionary with properties of the analyzed EOD waveform.

        - type: set to 'wave'.
        - EODf: is set to the EOD fundamental frequency.
        - p-p-amplitude: peak-to-peak amplitude of the Fourier fit.
        - flipped: True if the waveform was flipped.
        - amplitude: amplitude factor of the Fourier fit.
        - noise: root-mean squared standard error mean of the averaged
          EOD waveform relative to the p-p amplitude.
        - rmserror: root-mean-square error between Fourier-fit and
          EOD waveform relative to the p-p amplitude. If larger than
          about 0.05 the data are bad.
        - ncrossings: number of zero crossings per period
        - peakwidth: width of the peak at the averaged amplitude relative
          to EOD period.
        - troughwidth: width of the trough at the averaged amplitude
          relative to EOD period.
        - minwidth: peakwidth or troughwidth, whichever is smaller.
        - leftpeak: time from positive zero crossing to peak relative
          to EOD period.
        - rightpeak: time from peak to negative zero crossing relative to
          EOD period.
        - lefttrough: time from negative zero crossing to trough relative to
          EOD period.
        - righttrough: time from trough to positive zero crossing relative to
          EOD period.
        - p-p-distance: time between peak and trough relative to EOD period.
        - min-p-p-distance: p-p-distance or EOD period minus p-p-distance,
          whichever is smaller, relative to EOD period.
        - relpeakampl: amplitude of peak or trough, whichever is larger,
          relative to p-p amplitude.
        - power: summed power of all harmonics of the extracted EOD waveform
          in decibel relative to one.
        - datapower: summed power of all harmonics of the original data in
          decibel relative to one. Only if `freq` is a list of harmonics.
        - thd: total harmonic distortion, i.e. square root of sum of
          amplitudes squared of harmonics relative to amplitude
          of fundamental.  
        - dbdiff: smoothness of power spectrum as standard deviation of
          differences in decibel power.
        - maxdb: maximum power of higher harmonics relative to peak power
          in decibel.

    spec: 2-D array of floats
        First six columns are from the spectrum of the extracted
        waveform.  First column is the harmonics (fundamental is one),
        second column its frequency, third column its amplitude, fourth
        column its amplitude relative to the fundamental, fifth column
        is power of harmonics relative to fundamental in decibel, and
        sixth column the phase shift relative to the fundamental.
        If `freq` is a list of harmonics, a seventh column is added to
        `spec` that contains the powers of the harmonics from the
        original power spectrum of the raw data.  Rows are the
        harmonics, first row is the fundamental frequency with index
        0, relative amplitude of one, relative power of 0dB, and phase
        shift of zero.
    error_str: string
        If fitting of the fourier series failed,
        this is reported in this string.

    Raises
    ------
    IndexError:
        EOD data is less than one period long.
    """
    if eod.ndim == 2:
        if eod.shape[1] > 2:
            eeod = eod
        else:
            eeod = np.column_stack((eod, np.zeros(len(eod))))
        rate = 1/np.mean(np.diff(eeod[:, 0]))
    else:
        if isinstance(ratetime, (list, tuple, np.ndarray)):
            time = ratetime
            rate = 1/np.mean(np.diff(time))
        else:
            rate = ratetime
            time = np.arange(len(eod))/rate
        eeod = np.zeros((len(eod), 3))
        eeod[:, 0] = time
        eeod[:, 1] = eod
    # storage:
    meod = np.zeros((eeod.shape[0], eeod.shape[1] + 1))
    meod[:, :eeod.shape[1]] = eeod
    meod[:, -1] = np.nan
    
    freq1 = freq
    if hasattr(freq, 'shape'):
        freq1 = freq[0][0]

    error_str = ''

    # spectrum:
    has_spec = coeffs is not None
    if not has_spec:
        coeffs = fourier_coeffs(meod[:, 1], meod[:, 0], freq1, n_harmonics)
        phase1 = np.angle(coeffs[1])
        deltat = phase1/(2*np.pi*freq1)
        meod[:, 0] -= deltat   # TODO: test direction of shift

    # update waveform:
    coeffs = normalize_fourier_coeffs(coeffs)
    eodw = fourier_synthesis(freq1, coeffs, meod[:, 0])
    if has_spec:
        meod[:, 1] = eodw
    meod[:, -1] = eodw

    # subtract mean and flip:
    meod, coeffs, flipped = condition_wave(meod, ratetime, freq1, coeffs, flip)
    if has_spec:
        meod[:, 1] = meod[:, -1]

    # waveform properties:
    pos_ampl, neg_ampl, distance, min_distance = \
        analyze_wave_properties(meod, None, freq1)
    pp_ampl = pos_ampl + neg_ampl
    max_ampl = max(pos_ampl, neg_ampl)
    rel_max_ampl = max_ampl/pp_ampl
    
    # indices of exactly one period:
    pinx = int(np.ceil(rate/freq1)) # one period
    if len(meod) < 2*pinx:
        raise IndexError('data need to contain at least two EOD periods')
    i0 = (len(meod) - pinx)//2
    i1 = i0 + pinx

    # maximum:
    maxinx = i0 + np.argmax(eodw[i0:i1])
    meod[:, 0] -= meod[maxinx, 0]

    # zero crossings:
    ui, di = threshold_crossings(meod[:, 1], 0.0)
    ut, dt = threshold_crossing_times(meod[:, 0], meod[:, 1], 0.0, ui, di)
    ut, dt = merge_events(ut, dt, 0.02/freq1)
    ncrossings = int(np.round((len(ut) + len(dt))/(meod[-1,0]-meod[0,0])/freq1))
    if np.any(ut<0.0):    
        up_time = ut[ut<0.0][-1]
    else:
        up_time = 0.0 
        error_str += '%.1f Hz wave fish: no upward zero crossing. ' % freq1
    if np.any(dt>0.0):
        down_time = dt[dt>0.0][0]
    else:
        down_time = 0.0
        error_str += '%.1f Hz wave fish: no downward zero crossing. ' % freq1
    period = 1/freq1
    peak_width = down_time - up_time
    trough_width = period - peak_width
    peak_time = 0.0
    trough_time = meod[maxinx + np.argmin(meod[maxinx:maxinx + pinx, 1]), 0] - meod[maxinx, 0]
    phase1 = peak_time - up_time
    phase2 = down_time - peak_time
    phase3 = trough_time - down_time
    phase4 = up_time + period - trough_time
    
    # variance and fit error:
    rmssem = np.sqrt(np.mean(meod[i0:i1, 2]**2.0))/pp_ampl if meod.shape[1] > 2 else None
    rmserror = np.sqrt(np.mean((meod[i0:i1, 1] - meod[i0:i1, -1])**2.0))/pp_ampl

    # spectral analysis:
    spec, power, data_power, thd, db_diff, max_harmonics_power = \
        analyse_wave_spectrum(freq, coeffs, power_add_harmonics)

    # store results:
    props = {}
    props['type'] = 'wave'
    props['flipped'] = flipped
    props['EODf'] = freq1
    props['period'] = 1/freq1
    props['pos-ampl'] = pos_ampl
    props['neg-ampl'] = neg_ampl
    props['max-ampl'] = max_ampl
    props['p-p-amplitude'] = pp_ampl
    props['rel-max-amplitude'] = rel_max_ampl
    props['p-p-distance'] = distance/period
    props['min-p-p-distance'] = min_distance/period
    if rmssem:
        props['noise'] = rmssem
    props['rmserror'] = rmserror
    props['ncrossings'] = ncrossings
    props['peakwidth'] = peak_width/period
    props['troughwidth'] = trough_width/period
    props['minwidth'] = min(peak_width, trough_width)/period
    props['leftpeak'] = phase1/period
    props['rightpeak'] = phase2/period
    props['lefttrough'] = phase3/period
    props['righttrough'] = phase4/period
    props['power'] = power
    if data_power is not None:
        props['datapower'] = data_power
    props['thd'] = thd
    props['dbdiff'] = db_diff
    props['maxdb'] = max_harmonics_power
    
    return meod, props, spec, error_str


def plot_wave_spectrum(axa, axp, spec, props, unit=None,
                       ampl_style=dict(ls='', marker='o', color='tab:blue', markersize=6),
                       ampl_stem_style=dict(color='tab:blue', alpha=0.5, lw=2),
                       phase_style=dict(ls='', marker='p', color='tab:blue', markersize=6),
                       phase_stem_style=dict(color='tab:blue', alpha=0.5, lw=2)):
    """Plot and annotate spectrum of wave EOD.

    Parameters
    ----------
    axa: matplotlib axes
        Axes for amplitude plot.
    axp: matplotlib axes
        Axes for phase plot.
    spec: 2-D array
        The spectrum of the wave-type EOD as returned by
        `analyze_wave()`.  First column is the harmonics,
        second column its frequency, third column its amplitude,
        fourth column its amplitude relative to the fundamental, fifth
        column is power of harmonics relative to fundamental in
        decibel, and sixth column the phase shift relative to the
        fundamental.
    props: dict
        A dictionary with properties of the analyzed EOD waveform as
        returned by `analyze_wave()`.
    unit: string
        Optional unit of the data used for y-label.
    ampl_style: dict
        Properties of the markers of the amplitude plot.
    ampl_stem_style: dict
        Properties of the stems of the amplitude plot.
    phase_style: dict
        Properties of the markers of the phase plot.
    phase_stem_style: dict
        Properties of the stems of the phase plot.
    """
    n = min(9, np.sum(np.isfinite(spec[:, 2])))
    # amplitudes:
    markers, stemlines, _ = axa.stem(spec[:n, 0], spec[:n, 2],
                                     basefmt='none')
    setp(markers, clip_on=False, **ampl_style)
    setp(stemlines, **ampl_stem_style)
    axa.set_xlim(0.5, n + 0.5)
    axa.set_ylim(bottom=0)
    axa.xaxis.set_major_locator(MultipleLocator(1))
    axa.tick_params('x', direction='out')
    if unit:
        axa.set_ylabel(f'Amplitude [{unit}]')
    else:
        axa.set_ylabel('Amplitude')
    # phases:
    phases = spec[:n, 5]
    phases[phases<0.0] = phases[phases<0.0] + 2*np.pi
    markers, stemlines, _ = axp.stem(spec[:n, 0], phases[:n],
                                     basefmt='none')
    setp(markers, clip_on=False, **phase_style)
    setp(stemlines, **phase_stem_style)
    axp.set_xlim(0.5, n + 0.5)
    axp.xaxis.set_major_locator(MultipleLocator(1))
    axp.tick_params('x', direction='out')
    axp.set_ylim(0, 2*np.pi)
    axp.set_yticks([0, np.pi, 2*np.pi])
    axp.set_yticklabels(['0', '\u03c0', '2\u03c0'])
    axp.set_xlabel('Harmonics')
    axp.set_ylabel('Phase')


def save_wave_eodfs(wave_eodfs, wave_indices, basename, **kwargs):
    """Save frequencies of wave EODs to file.

    Parameters
    ----------
    wave_eodfs: list of 2D arrays
        Each item is a matrix with the frequencies and powers
        (columns) of the fundamental and harmonics (rows) as returned
        by `harmonics.harmonic_groups()`.
    wave_indices: array
        Indices identifying each fish or NaN.
        If None no index column is inserted.
    basename: string or stream
        If string, path and basename of file.
        If `basename` does not have an extension,
        '-waveeodfs' and a file extension according to `kwargs` are appended.
        If stream, write EOD frequencies data into this stream.
    kwargs:
        Arguments passed on to `TableData.write()`.

    Returns
    -------
    filename: Path
        Path and full name of the written file in case of `basename`
        being a string. Otherwise, the file name and extension that
        would have been appended to a basename.

    See Also
    --------
    load_wave_eodfs()

    """
    eodfs = fundamental_freqs_and_power(wave_eodfs)
    td = TableData()
    if wave_indices is not None:
        td.append('index', '', '%d',
                  value=[wi if wi >= 0 else np.nan
                         for wi in wave_indices])
    td.append('EODf', 'Hz', '%7.2f', value=eodfs[:, 0])
    td.append('datapower', 'dB', '%7.2f', value=eodfs[:, 1])
    ext = Path(basename).suffix if not hasattr(basename, 'write') else ''
    fp = '-waveeodfs' if not ext else ''
    return td.write_file_stream(basename, fp, **kwargs)


def load_wave_eodfs(file_path):
    """Load frequencies of wave EODs from file.

    Parameters
    ----------
    file_path: string
        Path of the file to be loaded.

    Returns
    -------
    eodfs: 2D array of floats
        EODfs and power of wave type fish.
    indices: array of ints
        Corresponding indices of fish, can contain negative numbers to
        indicate frequencies without fish.

    Raises
    ------
    FileNotFoundError:
        If `file_path` does not exist.

    See Also
    --------
    save_wave_eodfs()
    """
    data = TableData(file_path)
    eodfs = data.array()
    if 'index' in data:
        indices = data[:, 'index']
        indices[~np.isfinite(indices)] = -1
        indices = np.array(indices, dtype=int)
        eodfs = eodfs[:, 1:]
    else:
        indices = np.zeros(data.rows(), dtype=int) - 1
    return eodfs, indices

    
def save_wave_fish(eod_props, unit, basename, **kwargs):
    """Save properties of wave EODs to file.

    Parameters
    ----------
    eod_props: list of dict
        Properties of EODs as returned by `analyze_wave()` and
        `analyze_pulse()`.  Only properties of wave fish are saved.
    unit: string
        Unit of the waveform data.
    basename: string or stream
        If string, path and basename of file.
        If `basename` does not have an extension,
        '-wavefish' and a file extension are appended.
        If stream, write wave fish properties into this stream.
    kwargs:
        Arguments passed on to `TableData.write()`.

    Returns
    -------
    filename: Path or None
        Path and full name of the written file in case of `basename`
        being a string. Otherwise, the file name and extension that
        would have been appended to a basename.
        None if no wave fish are contained in eod_props and
        consequently no file was written.

    See Also
    --------
    load_wave_fish()
    """
    wave_props = [p for p in eod_props if p['type'] == 'wave']
    if len(wave_props) == 0:
        return None
    td = TableData()
    if 'twin' in wave_props[0] or 'rate' in wave_props[0] or \
       'nfft' in wave_props[0]:
        td.append_section('recording')
    if 'twin' in wave_props[0]:
        td.append('twin', 's', '%7.2f', value=wave_props)
        td.append('window', 's', '%7.2f', value=wave_props)
        td.append('winclipped', '%', '%.2f',
                  value=wave_props, fac=100.0)
    if 'samplerate' in wave_props[0]:
        td.append('samplerate', 'kHz', '%.3f',
                  value=wave_props, fac=0.001)
    if 'nfft' in wave_props[0]:
        td.append('nfft', '', '%d', value=wave_props)
        td.append('dfreq', 'Hz', '%.2f', value=wave_props)
    td.append_section('waveform')
    td.append('index', '', '%d', value=wave_props)
    td.append('n', '', '%5d', value=wave_props)
    td.append('flipped', '', '%d', value=wave_props)
    td.append('EODf', 'Hz', '%7.2f', value=wave_props)
    td.append('period', 'ms', '%7.3f', value=wave_props, fac=1000)
    td.append('pos-ampl', unit, '%.5g', value=wave_props)
    td.append('neg-ampl', unit, '%.5g', value=wave_props)
    td.append('max-ampl', unit, '%.5g', value=wave_props)
    td.append('p-p-amplitude', unit, '%.5g', value=wave_props)
    td.append('rel-max-amplitude', '%', '%.2f', value=wave_props, fac=100)
    td.append('p-p-distance', '%', '%.2f', value=wave_props, fac=100)
    td.append('min-p-p-distance', '%', '%.2f',
              value=wave_props, fac=100)
    td.append('power', 'dB', '%7.2f', value=wave_props)
    if 'datapower' in wave_props[0]:
        td.append('datapower', 'dB', '%7.2f', value=wave_props)
    td.append('thd', '%', '%.2f', value=wave_props, fac=100)
    td.append('dbdiff', 'dB', '%7.2f', value=wave_props)
    td.append('maxdb', 'dB', '%7.2f', value=wave_props)
    if 'noise' in wave_props[0]:
        td.append('noise', '%', '%.1f', value=wave_props, fac=100)
    td.append('rmserror', '%', '%.2f', value=wave_props, fac=100)
    if 'clipped' in wave_props[0]:
        td.append('clipped', '%', '%.1f', value=wave_props, fac=100)
    td.append_section('timing')
    td.append('ncrossings', '', '%d', value=wave_props)
    td.append('peakwidth', '%', '%.2f', value=wave_props, fac=100)
    td.append('troughwidth', '%', '%.2f', value=wave_props, fac=100)
    td.append('minwidth', '%', '%.2f', value=wave_props, fac=100)
    td.append('leftpeak', '%', '%.2f', value=wave_props, fac=100)
    td.append('rightpeak', '%', '%.2f', value=wave_props, fac=100)
    td.append('lefttrough', '%', '%.2f', value=wave_props, fac=100)
    td.append('righttrough', '%', '%.2f', value=wave_props, fac=100)
    ext = Path(basename).suffix if not hasattr(basename, 'write') else ''
    fp = '-wavefish' if not ext else ''
    return td.write_file_stream(basename, fp, **kwargs)


def load_wave_fish(file_path):
    """Load properties of wave EODs from file.

    All times are scaled to seconds, all frequencies to Hertz and all
    percentages to fractions.

    Parameters
    ----------
    file_path: string
        Path of the file to be loaded.

    Returns
    -------
    eod_props: list of dict
        Properties of EODs.

    Raises
    ------
    FileNotFoundError:
        If `file_path` does not exist.

    See Also
    --------
    save_wave_fish()

    """
    data = TableData(file_path)
    eod_props = data.dicts()
    for props in eod_props:
        if 'winclipped' in props:
            props['winclipped'] /= 100
        if 'samplerate' in props:
            props['samplerate'] *= 1000
        if 'nfft' in props:
            props['nfft'] = int(props['nfft'])
        props['type'] = 'wave'
        props['index'] = int(props['index'])
        props['n'] = int(props['n'])
        props['period'] /= 1000
        props['rel-max-amplitude'] /= 100
        props['p-p-distance'] /= 100
        props['min-p-p-distance'] /= 100
        props['thd'] /= 100
        props['noise'] /= 100
        props['rmserror'] /= 100
        if 'clipped' in props:
            props['clipped'] /= 100
        props['ncrossings'] = int(props['ncrossings'])
        props['peakwidth'] /= 100
        props['troughwidth'] /= 100
        props['minwidth'] /= 100
        props['leftpeak'] /= 100
        props['rightpeak'] /= 100
        props['lefttrough'] /= 100
        props['righttrough'] /= 100
    return eod_props


def save_wave_spectrum(spec_data, unit, idx, basename, **kwargs):
    """Save amplitude and phase spectrum of wave EOD to file.

    Parameters
    ----------
    spec_data: 2D array of floats
        Amplitude and phase spectrum of wave EOD as returned by
        `analyze_wave()`.
    unit: string
        Unit of the waveform data.
    idx: int or None
        Index of fish.
    basename: string or stream
        If string, path and basename of file.
        If `basename` does not have an extension,
        '-wavespectrum', the fish index, and a file extension are appended.
        If stream, write wave spectrum into this stream.
    kwargs:
        Arguments passed on to `TableData.write()`.

    Returns
    -------
    filename: Path
        Path and full name of the written file in case of `basename`
        being a string. Otherwise, the file name and extension that
        would have been appended to a basename.

    See Also
    --------
    load_wave_spectrum()

    """
    td = TableData(spec_data[:, :6]*[1.0, 1.0, 1.0, 100.0, 1.0, 1.0],
                   ['harmonics', 'frequency', 'amplitude',
                    'relampl', 'relpower', 'phase'],
                   ['', 'Hz', unit, '%', 'dB', 'rad'],
                   ['%.0f', '%.2f', '%.6f', '%10.2f', '%6.2f', '%8.4f'])
    if spec_data.shape[1] > 6:
        td.append('datapower', '%s^2/Hz' % unit, '%11.4e',
                  value=spec_data[:, 6])
    fp = ''
    ext = Path(basename).suffix if not hasattr(basename, 'write') else ''
    if not ext:
        fp = '-wavespectrum'
        if idx is not None:
            fp += f'-{idx}'
    return td.write_file_stream(basename, fp, **kwargs)


def load_wave_spectrum(file_path):
    """Load amplitude and phase spectrum of wave EOD from file.

    Parameters
    ----------
    file_path: string
        Path of the file to be loaded.

    Returns
    -------
    spec: 2D array of floats
        Amplitude and phase spectrum of wave EOD:
        harmonics, frequency, amplitude, relative amplitude in dB,
        relative power in dB, phase, data power in unit squared.
        Can contain NaNs.
    unit: string
        Unit of amplitudes.

    Raises
    ------
    FileNotFoundError:
        If `file_path` does not exist.

    See Also
    --------
    save_wave_spectrum()
    """
    data = TableData(file_path)
    spec = data.array()
    spec[:, 3] *= 0.01
    return spec, data.unit('amplitude')


def main():
    import matplotlib.pyplot as plt
    from thunderlab.eventdetection import snippets
    from .fakefish import wavefish_eods, export_wavefish
    from .eodanalysis import plot_eod_waveform

    print('Analysis of wave-type EOD waveforms.')

    # data:
    eodf = 456 # Hz
    rate = 96_000
    tmax = 5 # s
    data = wavefish_eods('Eigenmannia', eodf, rate, tmax, noise_std=0.02)
    unit = 'mV'
    eod_times = np.arange(0.5/eodf, tmax - 1/eodf, 1/eodf)
    eod_idx = (eod_times*rate).astype(int)

    # mean EOD:
    snips = snippets(data, eod_idx, start=-300, stop=300)
    mean_eod = np.mean(snips, 0)

    # analyse EOD:
    mean_eod, props, power, error_str = \
        analyze_wave(mean_eod, rate, eodf)

    # write out as python code:
    export_wavefish(power, '', 'Eigenmannia spec')

    # plot:
    fig, axs = plt.subplot_mosaic('wa\nwp', layout='constrained')
    plot_eod_waveform(axs['w'], mean_eod, props, unit=unit)
    axs['w'].set_title(f'wave fish: EODf = {props["EODf"]:.1f} Hz')
    plot_wave_spectrum(axs['a'], axs['p'], power, props)
    plt.show()


if __name__ == '__main__':
    main()
