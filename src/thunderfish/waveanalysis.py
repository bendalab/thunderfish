"""
Analysis of wave-type EODs.

## Analysis

- `extract_wave()`: retrieve average EOD waveform via Fourier transform.
- `condition_wave()`: subtract offset, flip, and shift wave-type EOD waveform.
- `analyze_wave_properties()`: characterize basic properties of a wave-type EOD.
- `analyze_wave_phases()`: characterize all phases of a wave-type EOD.
- `analyse_wave_spectrum()`: analyze the spectrum of a wave-type EOD.

### Complete analysis

Calls all the functions listed above:

- `analyze_wave()`: full analysis the EOD waveform of a wave fish.

## Quality assessment

- `wave_quality()`: asses quality of EOD waveform of a wave fish.

## Visualization

- `plot_wave_eod()`: plot and annotate a wave-type EOD waveform.
- `plot_wave_spectrum()`: plot and annotate spectrum of wave-type EODs.

## Storage

- `save_wave_eodfs()`: save frequencies of wave EODs to file.
- `load_wave_eodfs()`: load frequencies of wave EODs from file.
- `save_wave_fish()`: save properties of wave EODs to file.
- `load_wave_fish()`: load properties of wave EODs from file.
- `save_wave_phases()`: save phase properties of wave-type EOD to file.
- `load_wave_phases()`: load phase properties of wave-type EOD from file.
- `save_wave_spectrum()`: save amplitude and phase spectrum of wave EOD to file.
- `load_wave_spectrum()`: load amplitude and phase spectrum of wave EOD from file.

## Configuration

- `add_extract_wave_config()`: add parameters for `extract_wave()` to configuration.
- `extract_wave_args()`: retrieve parameters for `extract_wave()` from configuration.
- `add_analyze_wave_config()`: add parameters for `analyze_wave()` to configuration.
- `analyze_wave_args()`: retrieve parameters for `analyze_wave()` from configuration.
- `wave_quality_args()`: retrieve parameters for `wave_quality()` from configuration.

"""

import numpy as np

try:
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MultipleLocator, FuncFormatter
    from matplotlib.artist import setp
except ImportError:
    pass

from pathlib import Path
from scipy.stats import linregress
from numba import jit
from thunderlab.eventdetection import detect_peaks
from thunderlab.eventdetection import threshold_crossings, threshold_crossing_times, merge_events
from thunderlab.fourier import fourier_coeffs, normalize_fourier_coeffs
from thunderlab.fourier import fourier_synthesis
from thunderlab.powerspectrum import decibel
from thunderlab.tabledata import TableData

from .harmonics import fundamental_freqs_and_power


def extract_wave(data, rate, freq, deltaf,
                 win_fac='auto', min_segments=6, periods=5,
                 frate=1e6, max_harmonics=20,
                 min_corr=0.99, min_ampl_frac=0.5,
                 verbose=0, plot_level=0):
    """Retrieve average EOD waveform via Fourier decomposition.

    Fourier series are extracted for \\(n\\) frequencies
    from `freq` \\(\\pm\\) `deltaf` with `n = 1 + 4*win_fac`
    in multiple data segments of `1/(deltaf*win_fac)` duration.

    For each data segment, the frequency for which the corresponding
    Fourier decomposition results in the largest peak-peak amplitude
    is taken.

    Then a correlation matrix between the waveform estimates of each
    data segments is computed and the segments that are most similar
    to each other are selected. From these the ones whose standard
    deviation is smaller than `min_ampl_fac` of the maximum one are
    also discarded. The returned waveform and Fourier series is the
    average of the remaining data segments.

    Parameters
    ----------
    data: 1-D array of float
        The data to be analysed.
    rate: float
        Sampling rate of the data in Hertz.
    freq: float
        Estimated EOD frequency.
    deltaf: float
        Frequency resolution in Hertz. Ususally the resolution of the power spectrum
        from which `freq` was deduced.
    win_fac: int or 'auto'
        Integer factor by which the time window and
        thus the frequency resolution is reduced.
        Reducing the time window increases the chance for
        stable waveforms, but might miss the right frequency.
        If 'auto' then it is set to `max(1, int(np.round(freq/400)))`,
        i.e. for higher EOD frequencies thisfactor gets larger.
    min_segments: int
        Minimum required data segments that fit into the data with
        an overlap of three quarters. If necessary, the segment sizes
        and thus also the frequency resolution is reduced by
        powers of two.
    periods: float
        The duration of the returned waveform estimate is the
        EOD period (`1/freq`) times `periods`.
    frate: float
        Sampling rate used for the returned waveform estimates.
    max_harmonics: int
        Highest harmonics used for the Fourier decomposition.
    min_corr: float
        Minimum required correlation between two waveform estimates.
    min_ampl_fac: float
        Minimum required standard deviation of waveform estimate
        relative to the largest one.
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
    if win_fac == 'auto':
        win_fac = max(1, int(np.round(freq/400)))
    deltaf *= win_fac
    t_segment = min(len(data)/rate, 1/deltaf)
    nfreqs = 1 + 2*2*win_fac # twice the frequency resolution is necessary and sufficient!
    step = max(16, int(t_segment*rate))
    # extract Fourier series from data segments:
    n = int(periods/freq*rate)
    freqs = []
    indices = np.arange(0, max(1, len(data) - step + 1), max(1, step//4))
    while len(indices) < min_segments or t_segment > 0.1:
        t_segment /= 2
        step = max(8, int(t_segment*rate))
        indices = np.arange(0, max(1, len(data) - step + 1), max(1, step//4))
    times = indices/rate
    frange = np.linspace(freq - deltaf, freq + deltaf, nfreqs)
    for i in indices:
        w, f = fourier_freq_range(data[i:i + step], rate, frange, 6, n)
        freqs.append(f)
    freqs = np.array(freqs)
    mean_coeffs = np.zeros(0, dtype=complex)
    mean_eod = np.zeros((0, 3))
    if len(freqs) == 0:
        # TODO: Why??? How can indices be empty?
        return mean_coeffs, mean_eod, freq, np.array([]), 0, f'no frequencies detected ({len(indicies)} indices, freqs={freqs})'
    # refined Fourier series and waveforms:
    n = int(periods/np.mean(freqs)*frate)
    coeffs = np.zeros((len(indices), max_harmonics), dtype=complex)
    waves = np.zeros((len(indices), n))
    for k in range(len(indices)):
        i = indices[k]
        c = fourier_coeffs(data[i:i + step], rate, freqs[k], max_harmonics)
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
            if corr[max_corr_a, max_corr_b] < max(min_c, 1 - 0.25*(1 - min_corr)):
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
            if len(num_c) < 10:
                axs[1, 1].xaxis.set_major_locator(MultipleLocator(1))
            if np.max(num_c) < 10:
                axs[1, 1].yaxis.set_major_locator(MultipleLocator(1))
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


def condition_wave(eod, ratetime, freq, coeffs=None, flip_wave='none'):
    """Subtract offset, flip, and shift wave-type EOD waveform.
    
    Parameters
    ----------
    eod: 1-D or 2-D array
        The eod waveform to be analyzed.  If an 1-D array, then this
        is the waveform and you need to also pass a time array or
        sampling rate in `ratetime`.  If a 2-D array, then first
        column is time in seconds, second column the EOD
        waveform, and the last column - if present - is the waveform
        obtained from Fourier decomposition. All other columns are optional
        and are not used.
    ratetime: None or float or array of float
        If a 1-D array is passed on to `eod` then either the sampling
        rate in Hertz or the time array corresponding to `eod`.
    freq: float
        The frequency of the EOD.
    coeffs: None or 1-D array of complex
        The Fourier coefficients of an EOD waveform.
    flip_wave: 'auto', 'none', 'flip'
        - 'auto' flip waveform such that the larger extremum is positive.
        - 'flip' flip waveform.
        - 'none' do not flip waveform.

    Returns
    -------
    eod: 1-D or 2-D array
        Same shape as the input `eod`.
        If no `coeffs` are provided, the mean over integer multiples of
        the period of the waveform is subtracted.
        If waveform is flipped, it is flipped in the second and last column.
        If a last column is present, the waveform in the last column
        is recomputed from the flipped Fourier coefficients.
        In the first time column, time zero is set to the maximum
        of the first Fourier component.
    time: 1-D array
        If `eod` is an 1-D array, then a time array is returned,
        with zero set to the maximum of the first Fourier component.
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
        time = eod[:, 0]
        eodw = eod[:, -1] if eod.shape[1] > 2 else eod[:, 1]
    else:
        eodw = eod
        if isinstance(ratetime, (list, tuple, np.ndarray)):
            time = ratetime
        else:
            time = np.arange(len(eod))/ratetime
        
    # subtract mean:
    if coeffs is None:
        period = 1/freq
        p_time = np.floor((time[-1] - time[0])/period)*periods
        mask = (time >= time[-1] - p_time)
        if eod.ndim == 2:
            eod[:, 1] -= np.mean(eod[mask, 1])
        else:
            eod -= np.mean(eod[mask])

    # flip:
    flipped = False
    if flip_wave.lower() in ['flip', 'true', 'yes']:
        flipped = True
    elif flip_wave.lower() == 'auto':
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

    # shift time:
    if coeffs is None:
        c = fourier_coeffs(eodw, time, freq, 1)
    else:
        c = coeffs
    period = 1/freq
    phi1 = np.angle(c[1])
    t_zero = -phi1/(2*np.pi*freq)
    min_t = period
    if time[-1] - time[0] < 3*period:
        min_t = 0.5*period
    while t_zero < min_t:
        t_zero += period
    time -= time[0] + t_zero
    if coeffs is not None:
        for k in range(1, len(coeffs)):
            coeffs[k] *= np.exp(-1j*k*phi1)
    if eod.ndim == 2:
        eod[:, 0] = time
    
    # return:
    r = [eod]
    if eod.ndim == 1:
        r.append(time)
    if coeffs is not None:
        r.append(coeffs)
    r.append(flipped)
    return r


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
    pp_ampl: float
        Peak-to-peak amplitude.
    rel_max_ampl: float
        Amplitude of minimum or maximum, whichever is larger, relative to p-p amplitude.
    distance: float
        Temporal distance between largest negative trough and positive peak.
    min_distance: float
        Temporal distance between largest negative trough and positive peak
        normalized to first half of the EOD cycle.
        That is, if `distance` is larger than half an EOD cycle, then the
        duration of an EOD cycle minus `distance`.
    rms_sem: None or float
        Root-mean squared standard deviation of the extracted
        EOD waveform relative to the p-p amplitude.
    rms_error: float
        Root-mean-square difference between Fourier decomposition and
        EOD waveform relative to the p-p amplitude.
    """
    if eod.ndim == 2:
        time = eod[:, 0]
        eodw = eod[:, -1] if eod.shape[1] > 2 else eod[:, 1]
    else:
        eodw = eod
        if isinstance(ratetime, (list, tuple, np.ndarray)):
            time = ratetime
        else:
            time = np.arange(len(eod))/ratetime
    
    # cut out one period:
    period = 1/freq
    mask = (time >= 0) & (time <= period)
    eodp = eodw[mask]
    timep = time[mask]

    # amplitudes:
    pos_idx = np.argmax(eodp)
    pos_ampl = abs(eodp[pos_idx])
    neg_idx = np.argmin(eodp)
    neg_ampl = abs(eodp[neg_idx])
    pp_ampl = pos_ampl + neg_ampl
    max_ampl = max(pos_ampl, neg_ampl)
    rel_max_ampl = max_ampl/pp_ampl

    # timing:
    distance = abs(timep[neg_idx] - timep[pos_idx])
    min_distance = distance
    if distance > period/2:
        min_distance = period - distance
    
    # variance and fit error:
    rms_sem = None
    if eod.ndim == 2 and eod.shape[1] > 2:
        rms_sem = np.sqrt(np.mean(eod[mask, 2]**2.0))/pp_ampl
    rms_error = np.sqrt(np.mean((eod[mask, 1] - eod[mask, -1])**2.0))/pp_ampl
    
    return pp_ampl, rel_max_ampl, distance, min_distance, rms_sem, rms_error

    
def analyze_wave_phases(eod, ratetime, freq, thresh_frac=0.05):
    """Characterize all phases of a wave-type EOD.
    
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
    thresh_frac: float
        Threshold for detecting peaks and troughs as a fraction of
        the p-p amplitude.
    
    Returns
    -------
    phases: dict
        Dictionary with
    
        - "indices": indices of each phase
          (1 is P1, i.e. the largest positive peak)
        - "times": times of each phase within an EOD cycle
        - "amplitudes": amplitudes of each phase
        - "relamplitudes": amplitudes normalized to amplitude of P1 phase
        - "widths": widths of each phase computed from zeros
        - "zeros": time point where amplitude between this and the next phase is half their difference.

    """
    if eod.ndim == 2:
        time = eod[:, 0]
        eod = eod[:, -1] if eod.shape[1] > 2 else eod[:, 1]
    elif isinstance(ratetime, (list, tuple, np.ndarray)):
        time = ratetime
    else:
        time = np.arange(len(eod))/ratetime
    dt = np.mean(np.diff(time))
    period = 1/freq

    # threshold:
    mask = (time >= 0) & (time <= period)
    pp_ampl = np.max(eod[mask]) - np.min(eod[mask])
    thresh = thresh_frac*pp_ampl

    # find peaks and troughs:
    peak_idx, trough_idx = detect_peaks(eod, thresh)
    pt_idx = np.sort(np.concatenate((peak_idx, trough_idx)))

    # maximum peak in first period after zero:
    pt_pidx = pt_idx[(time[pt_idx] >= -0.25*period) & (time[pt_idx] < 0.75*period)]
    pi = np.argmax(eod[pt_pidx])
    max_inx = np.nonzero(pt_idx == pt_pidx[pi])[0][0]
    max_time = time[pt_idx[max_inx]]

    # analyse phases:
    times = []
    amplitudes = []
    widths = []
    zero_times = []
    p_time = 0
    sign_fac = 1
    i = 0
    for k in range(max_inx - 1, len(pt_idx)):
        idx = pt_idx[k]
        if time[idx] - max_time >= period - dt:
            break
        n_idx = pt_idx[k + 1]
        th = 0.5*(eod[idx] + eod[n_idx])
        snippet = eod[idx:n_idx] - th
        stimes = time[idx:n_idx]
        zidx = np.nonzero(snippet[:-1]*snippet[1:] < 0)[0]
        if len(zidx) == 0:
            zero_times,append(np.nan)
        else:
            zidx = zidx[len(zidx)//2]  # reduce to single zero crossing
            snippet = snippet[zidx:zidx + 2]
            stimes = stimes[zidx:zidx + 2]
            if sign_fac > 0:
                z_time = np.interp(0, snippet[::-1], stimes[::-1])
            else:
                z_time = np.interp(0, snippet, stimes)
        if i > 0:                
            times.append(time[idx])
            amplitudes.append(eod[idx])
            widths.append(z_time - p_time)
            zero_times.append(z_time)
        p_time = z_time
        i += 1
        sign_fac *= -1
    amplitudes = np.array(amplitudes)
    
    # store phase properties:
    phases = dict(indices=np.arange(len(times)) + 1,
                  times=np.array(times),
                  amplitudes=amplitudes,
                  relamplitudes=amplitudes/amplitudes[0],
                  widths=np.array(widths),
                  zeros=np.array(zero_times))
    return phases
    
    
def analyse_wave_spectrum(freq, coeffs, n_phase_harmonics=8):
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
    n_phase_harmonics: int
        Number of harmonics over which to compute the slope of the phases.
    
    Returns
    -------
    spec: 2-D array of float
        Amplitudes and phases for each harmonic.
        Rows are the harmonics, first row is the fundamental frequency
        with multiplier 1, relative amplitude of one, relative power of 0dB,
        and phase shift of zero.
        First six columns are from the spectrum of the extracted
        waveform, optional column 6 is from the spectrum of the recording:

        - column 0: multiplier of the harmonics (fundamental is one)
        - column 1: frequency in Hertz
        - column 2: amplitude
        - column 3: amplitude relative to the one of the fundamental
        - column 4: power of harmonics relative to fundamental in decibel
        - column 5: phase shift relative to the fundamental
        - column 6: if `freq` is a list of harmonics, the powers of
          the harmonics from the power spectrum of the raw data.

    power: float
        Total power, i.e. sum of squared Fourier amplitudes.
    data_power: float or None
        Total power (sum of data powers) in the data, if available.
        Only sum over as many harmonics as there are Fourier coefficients.
    thd: float
        Total harmonic distortion of the amplitudes \\(a_i\\) of the harmonics
        is power in the higher harmonics relative to the fundamental:
        \\[ \\text{thd} = \\sqrt{\\sum_{i=2}^n \\left(\\frac{a_i}{a_1}\\right)^2} \\]
    max_harmonics: int
        Harmonics with the largest power. Fundamental is one.
    db_diff: float
        Standard deviation of the differences of the decibel powers.
        A measure of smoothness of the spectrum.
    phase_slope: float
        Slope of a linear regression between phases and multipliers of
        the first `n_phase_harmonics` harmonics.
    """
    if hasattr(freq, 'shape'):
        freq1 = freq[0][0]
        n = len(coeffs) - 1
        n += np.sum(freq[:, 0] > (len(coeffs) - 0.5)*freq1)
        spec = np.zeros((n, 7))
        spec[:, :] = np.nan
        k = 0
        for i in range(len(coeffs) - 1):
            while k < len(freq) and freq[k, 0] < (i + 0.5)*freq1:
                k += 1
            if k >= len(freq):
                break
            if freq[k, 0] < (i + 1.5)*freq1:
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
        
    # total harmonic distortion:
    thd = np.sqrt(np.nansum(spec[1:, 3])**2)

    # harmonic with maximum power:
    db_powers = spec[:len(coeffs) - 1, 4]
    max_harmonics = np.argmax(db_powers) + 1
        
    # smoothness of power spectrum:
    db_diff = np.std(np.diff(db_powers))

    # slope of unwraped phases:
    phases = spec[:n_phase_harmonics, 5]
    phases = np.unwrap(phases)
    r = linregress(np.arange(len(phases)), phases)
    phase_slope = r.slope

    return spec, power, data_power, thd, max_harmonics, db_diff, phase_slope

    
def analyze_wave(eod, ratetime, freq, coeffs=None,
                 max_harmonics=20, flip_wave='none', thresh_frac=0.05,
                 n_phase_harmonics=8):
    """Full analysis of the EOD waveform of a wave fish.
    
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
    max_harmonics: int
        Highest harmonics used for the Fourier decomposition.
    flip_wave: 'auto', 'none', 'flip'
        - 'auto' flip waveform such that the larger extremum is positive.
        - 'flip' flip waveform.
        - 'none' do not flip waveform.
    thresh_frac: float
        Threshold for detecting peaks and troughs as a fraction of
        the p-p amplitude.
    n_phase_harmonics: int
        Number of harmonics over which to compute the slope of the phases.
    
    Returns
    -------
    meod: 2-D array of floats
        The eod waveform. First column is time in seconds, second
        column the eod waveform.  Further columns are kept from the
        input `eod`. And a column is added with the Fourier series.
    props: dict
        A dictionary with properties of the analyzed EOD waveform.

        - type: set to 'wave'.
        - flipped: true if the waveform was flipped.
        - EODf: EOD fundamental frequency.
        - period: period of the EOD, i.e. 1/EODf.
        - ppampl: peak-to-peak amplitude of the Fourier decomposed EOD waveform.
        - relmaxampl: amplitude of peak or trough, whichever is larger,
          relative to p-p amplitude.
        - power: summed power of all harmonics of the extracted
          EOD waveform in decibel relative to one.
        - datapower: summed power of all harmonics of the original
          data in decibel relative to one.
          Only if `freq` is a list of harmonics.
        - noise: root-mean squared standard deviation of the extracted
          EOD waveform relative to the p-p amplitude.
        - rmserror: root-mean-square difference between
          Fourier-decomposition and EOD waveform relative
          to the p-p amplitude.
        - ppdist: time between peak and trough relative to EOD period.
        - minppdist: p-p-distance or EOD period minus p-p-distance,
          whichever is smaller, relative to EOD period.
        - peakwidth: width of the maximum peak phase relative
          to EOD period.
        - troughwidth: width of minimum trough phase relative
          to EOD period.
        - minwidth: peakwidth or troughwidth, whichever is smaller.
        - phasethresh: threshold used for detecting phases as a
          factor of the p-p-amplitude.
        - nphases: number of phases of one EOD cycle.
        - thd: total harmonic distortion, i.e. square root of sum
          of squared relative amplitudes of higher harmonics.
        - maxharmonics: harmonics with maximum amplitude.
        - power2: power of the second harmonics.
        - dbdiff: smoothness of power spectrum as standard deviation of
          differences in decibel power.
        - phaseslope: slope of a linear regression between phases and
          multipliers of the harmonics.

    phases: dict
        Dictionary with
    
        - "indices": indices of each phase
          (1 is P1, i.e. the largest positive peak)
        - "times": times of each phase within an EOD cycle
        - "amplitudes": amplitudes of each phase
        - "relamplitudes": amplitudes normalized to amplitude of P1 phase
        - "widths": widths of each phase computed from zeros
        - "zeros": time point where amplitude between this and the next phase
          is half their difference.
    
    spec: 2-D array of floats
    spec: 2-D array of float
        Amplitudes and phases for each harmonic.
        Rows are the harmonics, first row is the fundamental frequency
        with multiplier 1, relative amplitude of one, relative power of 0dB,
        and phase shift of zero.
        First six columns are from the spectrum of the extracted
        waveform, optional column 6 is from the spectrum of the recording:

        - column 0: multiplier of the harmonics (fundamental is one)
        - column 1: frequency in Hertz
        - column 2: amplitude
        - column 3: amplitude relative to the one of the fundamental
        - column 4: power of harmonics relative to fundamental in decibel
        - column 5: phase shift relative to the fundamental
        - column 6: if `freq` is a list of harmonics, the powers of
          the harmonics from the power spectrum of the raw data.
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
    period = 1/freq1

    # spectrum:
    has_spec = coeffs is not None
    if not has_spec:
        coeffs = fourier_coeffs(meod[:, 1], meod[:, 0], freq1, max_harmonics)
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
    meod, coeffs, flipped = condition_wave(meod, None, freq1, coeffs, flip_wave)
    if has_spec:
        meod[:, 1] = meod[:, -1]

    # waveform properties:
    pp_ampl, rel_max_ampl, distance, min_distance, rms_sem, rms_error = \
        analyze_wave_properties(meod, None, freq1)

    # phases:
    phases = analyze_wave_phases(meod, None, freq1, thresh_frac=thresh_frac)
    p_inx = np.argmax(phases['amplitudes'])
    peak_width = phases['widths'][p_inx]
    t_inx = np.argmin(phases['amplitudes'])
    trough_width = phases['widths'][t_inx]
    
    # spectral analysis:
    spec, power, data_power, thd, max_harmonics, db_diff, phase_slope = \
        analyse_wave_spectrum(freq, coeffs,
                              n_phase_harmonics=n_phase_harmonics)

    # store results:
    props = {}
    props['type'] = 'wave'
    props['flipped'] = flipped
    props['EODf'] = freq1
    props['period'] = period
    props['ppampl'] = pp_ampl
    props['relmaxampl'] = rel_max_ampl
    props['power'] = power
    if data_power is not None:
        props['datapower'] = data_power
    if rms_sem:
        props['noise'] = rms_sem
    props['rmserror'] = rms_error
    props['ppdist'] = distance/period
    props['minppdist'] = min_distance/period
    props['peakwidth'] = peak_width/period
    props['troughwidth'] = trough_width/period
    props['minwidth'] = min(peak_width, trough_width)/period
    props['phasethresh'] = thresh_frac
    props['nphases'] = len(phases['times'])
    props['thd'] = thd
    props['maxharmonics'] = max_harmonics
    props['power2'] = spec[1, 4]
    props['dbdiff'] = db_diff
    props['phaseslope'] = phase_slope
    
    return meod, props, phases, spec


def wave_quality(props, harm_relampl=None, min_freq=0.0,
                 max_freq=2000.0, max_clipped_frac=0.1,
                 max_phases=4, max_rms_sem=0.0, max_rms_error=0.05,
                 min_power=-100.0, max_thd=0.0, max_db_diff=20.0,
                 max_relampl_harm2=0.0, max_relampl_harm3=0.0,
                 max_relampl_harm4=0.0):
    """Assess the quality of an EOD waveform of a wave fish.
    
    Parameters
    ----------
    props: dict
        A dictionary with properties of the analyzed EOD waveform
        as returned by `analyze_wave()`.
    harm_relampl: 1-D array of floats or None
        Relative amplitude of at least the first 3 harmonics without
        the fundamental.
    min_freq: float
        Minimum EOD frequency (`props['EODf']`).
    max_freq: float
        Maximum EOD frequency (`props['EODf']`).
    max_clipped_frac: float
        If larger than zero, maximum allowed fraction of clipped data
        (`props['clipped']`).
    max_phases: int
        If larger than zero, maximum number of phases per EOD period
        (`props['nphases']`).
    max_rms_sem: float
        If larger than zero, maximum allowed standard deviation of the
        data relative to p-p amplitude (`props['noise']`).
    max_rms_error: float
        If larger than zero, maximum allowed root-mean-square difference
        between EOD waveform and Fourier series relative to p-p amplitude
        (`props['rmserror']`).
    min_power: float
        Minimum power of the EOD in dB (`props['power']`).
    max_thd: float
        If larger than zero, then maximum total harmonic distortion
        (`props['thd']`).
    max_db_diff: float
        If larger than zero, maximum standard deviation of differences between
        logarithmic powers of harmonics in decibel (`props['dbdiff']`).
        Low values enforce smoother power spectra.
    max_relampl_harm2: float
        If larger than zero, maximum allowed amplitude of second harmonic
        relative to fundamental (=first harmonics).
    max_relampl_harm3: float
        If larger than zero, maximum allowed amplitude of third harmonic
        relative to fundamental (=first harmonics).
    max_relampl_harm4: float
        If larger than zero, maximum allowed amplitude of fourth harmonic
        relative to fundamental (=first harmonics).
                                       
    Returns
    -------
    remove: bool
        If True then this is most likely not an electric fish. Remove
        it from both the waveform properties and the list of EOD
        frequencies.  If False, keep it in the list of EOD
        frequencies, but remove it from the waveform properties if
        `skip_reason` is not empty.
    skip_reason: str
        An empty string if the waveform is good, otherwise a string
        indicating the failure.
    msg: str
        A textual representation of the values tested.
    """
    remove = False
    msg = []
    skip_reason = []
    # EOD frequency:
    if 'EODf' in props:
        eodf = props['EODf']
        msg += [f'EODf={eodf:6.1f}Hz']
        if eodf < min_freq or eodf > max_freq:
            remove = True
            skip_reason += [f'invalid EODf={eodf:6.1f}Hz (minimumFrequency={min_freq:6.1f}Hz, maximumFrequency={max_freq:6.1f}Hz)']
    # clipped fraction:
    if 'clipped' in props:
        clipped_frac = props['clipped']
        msg += [f'clipped={100*clipped_frac:3.0f}%']
        if max_clipped_frac > 0 and clipped_frac >= max_clipped_frac:
            skip_reason += [f'clipped={100*clipped_frac:3.0f}% (maximumClippedFraction={100*max_clipped_frac:3.0f}%)']
    # too many zero crossings:
    if 'nphases' in props:
        nphases = props['nphases']
        msg += [f'phases={nphases}']
        if max_phases > 0 and nphases > max_phases:
            skip_reason += [f'too many phases={nphases} (maximumPhases={max_phases})']
    # noise:
    rms_sem = None
    if 'rmssem' in props:
        rms_sem = props['rmssem']
    if 'noise' in props:
        rms_sem = props['noise']
    if rms_sem is not None:
        msg += [f'rms sem waveform={100*rms_sem:6.2f}%']
        if max_rms_sem > 0.0 and rms_sem >= max_rms_sem:
            skip_reason += [f'noisy waveform s.e.m.={100*rms_sem:6.2f}% (max {100*max_rms_sem:6.2f}%)']
    # fit error:
    if 'rmserror' in props:
        rms_error = props['rmserror']
        msg += [f'rmserror={100*rms_error:6.2f}%']
        if max_rms_error > 0.0 and rms_error >= max_rms_error:
            skip_reason += [f'noisy rmserror={100*rms_error:6.2f}% (maximumVariance={100*max_rms_error:6.2f}%)']
    # wave power:
    if 'power' in props:
        power = props['power']
        msg += [f'power={power:6.1f}dB']
        if power < min_power:
            skip_reason += [f'small power={power:6.1f}dB (minimumPower={min_power:6.1f}dB)']
    # total harmonic distortion:
    if 'thd' in props:
        thd = props['thd']
        msg += [f'thd={100*thd:5.1f}%']
        if max_thd > 0.0 and thd > max_thd:
            skip_reason += [f'large THD={100*thd:5.1f}% (maxximumTotalHarmonicDistortion={100*max_thd:5.1f}%)']
    # smoothness of spectrum:
    if 'dbdiff' in props:
        db_diff = props['dbdiff']
        msg += [f'dBdiff={db_diff:5.1f}dB']
        if max_db_diff > 0.0 and db_diff > max_db_diff:
            remove = True
            skip_reason += [f'not smooth s.d. diff={db_diff:5.1f}dB (maxximumPowerDifference={max_db_diff:5.1f}dB)']
    # relative amplitude of harmonics:
    if harm_relampl is not None:
        for k, max_relampl in enumerate([max_relampl_harm2, max_relampl_harm3, max_relampl_harm4]):
            if k >= len(harm_relampl):
                break
            msg += [f'ampl{k + 2}={100*harm_relampl[k]:5.1f}%']
            if max_relampl > 0.0 and k < len(harm_relampl) and harm_relampl[k] >= max_relampl:
                num_str = ['Second', 'Third', 'Fourth']
                skip_reason += [f'distorted ampl{k + 2}={100*harm_relampl[k]:5.1f}% (maximum{num_str[k]}HarmonicAmplitude={100*max_relampl:5.1f}%)']
    return remove, ', '.join(skip_reason), ', '.join(msg)


def plot_wave_eod(ax, eod_waveform, props, phases=None,
                  unit=None, wave_periods=2, rel_width=True,
                  wave_style=dict(lw=1.5, color='tab:red'),
                  sem_style=dict(color='0.8'),
                  phase_style=dict(zorder=0, ls='', marker='o', color='tab:red',
                                   markersize=5, mec='white', mew=1),
                  zerox_style=dict(zorder=50, ls='', marker='o', color='black',
                                   markersize=5, mec='white', mew=1),
                  zero_style=dict(lw=0.5, color='0.7'),
                  fontsize='medium'):
    """Plot and annotate a wave-type EOD waveform.

    Parameters
    ----------
    ax: matplotlib axes
        Axes used for plotting.
    eod_waveform: 2-D array
        EOD waveform. First column is time in seconds, second column
        the (mean) eod waveform. The optional third column is the
        standard error. Further columns are ignored.
    props: dict
        A dictionary with properties of the analyzed EOD waveform as
        returned by `analyze_wave()`.
    phases: dict
        Dictionary with phase properties as returned by
        `analyze_wave_phases()`, `analyze_wave()`, and
        `load_wave_phases()`.
    unit: str
        Optional unit of the data used for y-label.
    wave_periods: float
        How many periods of a wave EOD are shown.
    rel_width: bool
        If True annotate width in percent relative to the EOD period,
        otherwise report them in ms.
    wave_style: dict
        Arguments passed on to the plot command for the EOD waveform.
    sem_style: dict
        Arguments passed on to the fill_between command for the
        standard error of the EOD.
    phase_style: dict
        Arguments passed on to the plot command for marking EOD phases.
    zerox_style: dict
        Arguments passed on to the plot command for marking zero crossings.
    zero_style: dict
        Arguments passed on to the plot command for the zero line.
    fontsize: str or float or int
        Fontsize for annotation text.

    """
    time = 1000*eod_waveform[:, 0]
    eod = eod_waveform[:, 1]
    # time axis:                
    period = 1000/props['EODf']
    xlim_l = -0.5*wave_periods*period
    xlim_r = +0.5*wave_periods*period + period
    xlim = xlim_r - xlim_l
    ax.set_xlim(xlim_l, xlim_r)
    if xlim < 2:
        ax.xaxis.set_major_locator(MultipleLocator(0.5))
    elif xlim < 4:
        ax.xaxis.set_major_locator(MultipleLocator(1))
    elif xlim < 8:
        ax.xaxis.set_major_locator(MultipleLocator(2))
    ax.set_xlabel('Time [msec]')
    # amplitude axis:                
    ylim = np.max(np.abs(eod[(time >= xlim_l) & (time <= xlim_r)])) 
    ax.set_ylim(-1.2*ylim, +1.2*ylim)
    if unit:
        ax.set_ylabel(f'Amplitude [{unit}]')
    else:
        ax.set_ylabel('Amplitude')
    # ax height dimensions:
    t = ax.text(0, 0, 'test', fontsize=fontsize)
    fs = t.get_fontsize()
    t.remove()
    pixelx = np.abs(np.diff(ax.get_window_extent().get_points()[:, 0]))[0]
    dxu = 2*xlim/pixelx
    xfs = fs*dxu
    pixely = np.abs(np.diff(ax.get_window_extent().get_points()[:, 1]))[0]
    dyu = 2*ylim/pixely
    yfs = fs*dyu
    # plot zero line:
    ax.axhline(0.0, zorder=10, **zero_style)
    # plot waveform:
    ax.plot(time, eod, zorder=45, **wave_style)
    # plot standard error:
    if eod_waveform.shape[1] > 2:
        std_eod = eod_waveform[:, 2]
        ax.fill_between(time, eod + std_eod, eod - std_eod,
                        zorder=20, **sem_style)
    # plot and annotate phases:
    if phases is not None and len(phases) > 0:
        # mark zero crossings:
        zeros = 1000*phases['zeros']
        zeros_ampls = np.interp(zeros, time, eod)
        ax.plot(zeros, zeros_ampls, **zerox_style)
        ax.plot(zeros[-1] - period, zeros_ampls[-1], **zerox_style)
        # mark phase peak/trough:
        times = 1000*phases['times']
        amplitudes = phases['amplitudes']
        ax.plot(times, amplitudes, **phase_style)
        # annotate period:
        ax.plot(times[0] + period, amplitudes[0], **phase_style)
        ax.text(times[0] + period + xfs, amplitudes[0], f'T={period:.3g}ms',
                va='center', zorder=100, fontsize=fontsize)
        # phase peaks and troughs:
        widths = phases['widths']
        for i in range(len(times)):
            index = phases['indices'][i]
            ptime = times[i]
            if ptime < xlim_l or ptime > xlim_r:
                continue
            pi = np.argmin(np.abs(time - ptime))
            pampl = amplitudes[i]
            relampl = phases['relamplitudes'][i]
            # text for phase label:
            label = f'P{index:.0f}'
            if np.abs(ptime) < 1:
                ts = f'{1000*ptime:.0f}\u00b5s'
            elif np.abs(ptime) < 10:
                ts = f'{ptime:.2f}ms'
            else:
                ts = f'{ptime:.3g}ms'
            if index == 1:
                label += f'(@ {ts})'
            else:
                if np.abs(relampl) < 0.05:
                    ps = f'{100*relampl:.1f}%'
                else:
                    ps = f'{100*relampl:.0f}%'
                label += f'({ps} @ {ts})'
            # position of phase label:
            nampl = amplitudes[i - 1 if i > 0 else i + 1]
            local_min = pampl < nampl
            if i == 0:
                ax.text(ptime + xfs, pampl, label,
                        ha='left', va='center', zorder=100, fontsize=fontsize)
            else:
                dy = 1.5*yfs
                valign = 'bottom'
                if local_min:
                    dy = -dy
                    valign = 'top'
                ax.text(ptime, pampl + dy, label,
                        ha='center', va=valign, zorder=100, fontsize=fontsize)
            # text for width label:
            pwidth = 1000*widths[i]
            if rel_width:
                ws = f'{100*pwidth/period:.2g}%'
            elif pwidth < 1:
                ws = f'{1000*pwidth:.0f}\u00b5s'
            elif pwidth < 10:
                ws = f'{pwidth:.2f}ms'
            else:
                ws = f'{pwidth:.3g}ms'
            if i > 0:
                x = 0.5*(zeros[i - 1] + zeros[i])
            else:
                x = 0.5*(zeros[-1] - period + zeros[i])
            y = 0.5*(zeros_ampls[i - 1] + zeros_ampls[i])
            if local_min:
                if y > amplitudes[i - 1] - yfs:
                    y = amplitudes[i - 1] - yfs
                if y > amplitudes[(i + 1)%len(amplitudes)] - yfs:
                    y = amplitudes[(i + 1)%len(amplitudes)] - yfs
            else:
                if y < amplitudes[i - 1] + yfs:
                    y = amplitudes[i - 1] + yfs
                if y < amplitudes[(i + 1)%len(amplitudes)] + yfs:
                    y = amplitudes[(i + 1)%len(amplitudes)] + yfs
            if pwidth < 0.25*period:
                ax.text(x, y, ws, ha='center', va='center', rotation='vertical', zorder=100, fontsize=fontsize)
            else:
                ax.text(x, y, ws, ha='center', va='center', zorder=100, fontsize=fontsize)
    # annotate plot:
    if props is not None:
        label = ''
        if 'n' in props:
            eods = 'EODs' if props['n'] > 1 else 'EOD'
            segs = ''
            if 'nsegments' in props:
                n_segs = props['nsegments']
                segs = f' (in {n_segs} segment{"s" if n_segs > 1 else ""})'
            label += f'n = {props["n"]} {eods}{segs}'
        if 'flipped' in props and props['flipped']:
            if len(label) > 0:
                label += '\n'
            label += 'flipped'
        ax.text(0.03, 1.03, label, transform=ax.transAxes,
                va='top', ha='left', zorder=100)


def plot_wave_spectrum(axa, axp, spec, props, unit=None,
                       ampl_style=dict(ls='', marker='o', color='tab:blue', markersize=6),
                       ampl_stem_style=dict(color='tab:blue', alpha=0.5, lw=2),
                       phase_style=dict(ls='', marker='p', color='tab:blue', markersize=6),
                       phase_stem_style=dict(color='tab:blue', alpha=0.5, lw=2),
                       phase_twopi_style=dict(color='k', ls='--', lw=0.5)):
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
    phase_twopi_style: dict
        Properties of the horizontal lines marking \\(2\\pi\\)
        in the phase plot.
    """
    def pi_fmt(x, pos):
        i = int(np.round(x/np.pi))
        if i == 1:
            return '\u03c0'
        elif i == 0:
            return '0'
        elif i == -1:
            return '-\u03c0'
        else:
            return f'{i}\u03c0'
        
    n = min(9, np.sum(np.isfinite(spec[:, 2])))
    # amplitudes:
    markers, stemlines, _ = axa.stem(spec[:n, 0], spec[:n, 2],
                                     basefmt='none')
    setp(markers, clip_on=False, **ampl_style)
    setp(stemlines, **ampl_stem_style)
    if props and 'thd' in props:
        axa.text(1, 1, f'thd={100*props["thd"]:.0f}%',
                 ha='right', va='top', transform=axa.transAxes)
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
    phases = np.unwrap(phases)
    min_p = np.floor(np.min(phases)/np.pi)*np.pi
    max_p = np.ceil(np.max(phases)/np.pi)*np.pi
    if max_p - min_p < 1.5*np.pi:
        max_p += np.pi
    for p in np.arange(min_p + np.pi, max_p, np.pi):
        if np.abs(p - np.round(p/2/np.pi)*2*np.pi) < 1e-8:
            axp.axhline(p, **phase_twopi_style)
    markers, stemlines, _ = axp.stem(spec[:n, 0], phases[:n],
                                     basefmt='none')
    setp(markers, clip_on=False, **phase_style)
    setp(stemlines, **phase_stem_style)
    axp.set_xlim(0.5, n + 0.5)
    axp.xaxis.set_major_locator(MultipleLocator(1))
    axp.tick_params('x', direction='out')
    if props and 'phaseslope' in props:
        axp.text(0.03, 1, f'slope={props["phaseslope"]:.2g}',
                 va='top', transform=axp.transAxes)
    axp.set_ylim(min_p, max_p)
    if max_p - min_p < 3.5*np.pi:
        axp.yaxis.set_major_locator(MultipleLocator(1*np.pi))
    else:
        axp.yaxis.set_major_locator(MultipleLocator(2*np.pi))
    axp.yaxis.set_major_formatter(FuncFormatter(pi_fmt))
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
    if 'twin' in wave_props[0] or 'samplerate' in wave_props[0] or \
       'nfft' in wave_props[0]:
        td.append_section('recording')
    if 'twin' in wave_props[0]:
        td.append('twin', 's', '%7.2f', value=wave_props)
        td.append('window', 's', '%7.2f', value=wave_props)
        td.append('channel', '', '%d', value=wave_props)
        td.append('winclipped', '%', '%.2f',
                  value=wave_props, fac=100.0)
    if 'nsegments' in wave_props[0]:
        td.append('nsegments', '', '%d', value=wave_props)
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
    td.append('ppampl', unit, '%.5g', value=wave_props)
    td.append('relmaxampl', '%', '%.2f', value=wave_props, fac=100)
    td.append('power', 'dB', '%7.2f', value=wave_props)
    if 'datapower' in wave_props[0]:
        td.append('datapower', 'dB', '%7.2f', value=wave_props)
    if 'noise' in wave_props[0]:
        td.append('noise', '%', '%.1f', value=wave_props, fac=100)
    td.append('rmserror', '%', '%.2f', value=wave_props, fac=100)
    if 'clipped' in wave_props[0]:
        td.append('clipped', '%', '%.1f', value=wave_props, fac=100)
    td.append_section('timing')
    td.append('ppdist', '%', '%.2f', value=wave_props, fac=100)
    td.append('minppdist', '%', '%.2f', value=wave_props, fac=100)
    td.append('peakwidth', '%', '%.2f', value=wave_props, fac=100)
    td.append('troughwidth', '%', '%.2f', value=wave_props, fac=100)
    td.append('minwidth', '%', '%.2f', value=wave_props, fac=100)
    td.append_section('phases')
    td.append('phasethresh', '%', '%.1f', value=wave_props, fac=100)
    td.append('nphases', '', '%d', value=wave_props)
    td.append_section('spectrum')
    td.append('thd', '%', '%.2f', value=wave_props, fac=100)
    td.append('maxharmonics', '', '%d', value=wave_props)
    td.append('power2', 'dB', '%7.2f', value=wave_props)
    td.append('dbdiff', 'dB', '%7.2f', value=wave_props)
    td.append('phaseslope', '', '%.3f', value=wave_props)
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
        if 'nsegments' in props:
            props['nsegments'] = int(props['nsegments'])
        if 'samplerate' in props:
            props['samplerate'] *= 1000
        if 'nfft' in props:
            props['nfft'] = int(props['nfft'])
        props['type'] = 'wave'
        props['channel'] = int(props['channel'])
        props['index'] = int(props['index'])
        props['n'] = int(props['n'])
        props['period'] /= 1000
        props['relmaxampl'] /= 100
        props['ppdist'] /= 100
        props['minppdist'] /= 100
        props['phasethresh'] /= 100
        props['nphases'] = int(props['nphases'])
        props['thd'] /= 100
        props['noise'] /= 100
        props['rmserror'] /= 100
        if 'clipped' in props:
            props['clipped'] /= 100
        props['peakwidth'] /= 100
        props['troughwidth'] /= 100
        props['minwidth'] /= 100
    return eod_props


def save_wave_phases(phases, unit, idx, basename, **kwargs):
    """Save phase properties of wave-type EOD to file.

    Parameters
    ----------
    phases: dict
        Dictionary with
    
        - "indices": indices of each phase
          (1 is P1, i.e. the largest positive peak)
        - "times": times of each phase within an EOD cycle
        - "amplitudes": amplitudes of each phase
        - "relamplitudes": amplitudes normalized to amplitude of P1 phase
        - "widths": widths of each phase computed from zeros
        - "zeros": time point where amplitude between this and the next phase is half the difference.
    
        as returned by `analyze_wave_phases()` and  `analyze_wave()`.
    unit: string
        Unit of the waveform data.
    idx: int or None
        Index of fish.
    basename: string or stream
        If string, path and basename of file.
        If `basename` does not have an extension,
        '-wavephases', the fish index, and a file extension are appended.
        If stream, write wave phases into this stream.
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
    load_wave_phases()
    """
    if not phases or len(phases['times']) == 0:
        return None
    td = TableData()
    td.append('index', '', '%.0f', value=phases['indices'])
    td.append('time', 'ms', '%.4f', value=phases['times'], fac=1000)
    td.append('amplitude', unit, '%.5f', value=phases['amplitudes'])
    td.append('relampl', '%', '%.2f', value=phases['relamplitudes'], fac=100)
    td.append('width', 'ms', '%.4f', value=phases['widths'], fac=1000)
    td.append('zeros', 'ms', '%.4f', value=phases['zeros'], fac=1000)
    fp = ''
    ext = Path(basename).suffix if not hasattr(basename, 'write') else ''
    if not ext:
        fp = '-wavephases'
        if idx is not None:
            fp += f'-{idx}'
    return td.write_file_stream(basename, fp, **kwargs)


def load_wave_phases(file_path):
    """Load phase properties of wave-type EOD from file.

    Parameters
    ----------
    file_path: string
        Path of the file to be loaded.

    Returns
    -------
    phases: dict
        Dictionary with
    
        - "indices": indices of each phase
          (1 is P1, i.e. the largest positive peak)
        - "times": times of each phase within an EOD cycle
        - "amplitudes": amplitudes of each phase
        - "relamplitudes": amplitudes normalized to amplitude of P1 phase
        - "widths": widths of each phase computed from zeros
        - "zeros": time point where amplitude between this and the next phase is half the difference.
    
    unit: string
        Unit of phase amplitudes.

    Raises
    ------
    FileNotFoundError:
        If `file_path` does not exist.

    See Also
    --------
    save_wave_phases()
    """
    data = TableData(file_path)
    phases = dict(indices=data['index'].astype(int),
                  times=data['time']*0.001,
                  amplitudes=data['amplitude'],
                  relamplitudes=data['relampl']*0.01,
                  widths=data['width']*0.001,
                  zeros=data['zeros']*0.001)
    return phases, data.unit('amplitude')


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
    td = TableData(spec_data[:, :6]*[1, 1, 1, 100, 1, 1],
                   ['harmonics', 'frequency', 'amplitude',
                    'relampl', 'relpower', 'phase'],
                   ['', 'Hz', unit, '%', 'dB', 'rad'],
                   ['%.0f', '%.2f', '%.6f', '%10.2f', '%6.2f', '%8.4f'])
    if spec_data.shape[1] > 6:
        td.append('datapower', f'{unit}^2/Hz', '%11.4e',
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


def add_extract_wave_config(cfg, win_fac='auto', min_segments=6,
                            periods=5, frate=1e6, max_harmonics=20,
                            min_corr=0.99, min_ampl_frac=0.5):
    """Add all parameters needed for `extract_wave()` as a new
    section to a configuration.

    Parameters
    ----------
    cfg: ConfigFile
        The configuration.
        
    See `extract_wave()` for details on the remaining arguments.
    """
    cfg.add_section('Wave-type EOD extraction:')
    cfg.add('dataSegmentDivider', win_fac, '', 'Reduce duration of data segments for EOD waveform estimation by this factor.')
    cfg.add('minimumDataSegments', min_segments, '', 'Minimum number of data segments required for EOD waveform estimation.')
    cfg.add('periodsWaveEOD', periods, '', 'Number of periods computed for an EOD estimate.')
    cfg.add('samplingRateWaveEOD', 0.001*frate, 'kHz', 'Sampling rate of EOD waveform estimates.')
    cfg.add('eodHarmonics', max_harmonics, '', 'Highest harmonics of the Fourier decomposition.')
    cfg.add('minimumEODCorrelations', min_corr, '', 'Minimum correlation between good EOD waveform estimates.')
    cfg.add('minimumEODAmplitude', 100*min_ampl_frac, '%', 'Minimum amplitude of an EOD estimate relative to the largest one.')


def extract_wave_args(cfg):
    """Retrieve parameters for `extract_wave()` from configuration.
    
    The return value can then be passed as key-word arguments to this
    function.

    Parameters
    ----------
    cfg: ConfigFile
        The configuration.

    Returns
    -------
    a: dict
        Dictionary with names of arguments of the `extract_wave()` function
        and their values as supplied by `cfg`.
    """
    a = cfg.map(win_fac='dataSegmentDivider',
                min_segments='minimumDataSegments',
                periods='periodsWaveEOD',
                frate='samplingRateWaveEOD',
                max_harmonics='eodHarmonics',
                min_corr='minimumEODCorrelations',
                min_ampl_frac='minimumEODAmplitude')
    if a['win_fac'] != 'auto':
        a['win_fac'] = int(a['win_fac'])
    a['frate'] *= 1000
    a['min_ampl_frac'] *= 0.01
    return a

        
def add_analyze_wave_config(cfg, max_harmonics=20, flip_wave='none',
                            thresh_frac=0.05, n_phase_harmonics=8):
    """Add all parameters needed for `analyse_wave()` as a new
    section to a configuration.

    Parameters
    ----------
    cfg: ConfigFile
        The configuration.
        
    See `analyze_wave()` for details on the remaining arguments.
    """
    cfg.add_section('Wave-type EOD analysis:')
    cfg.add('eodHarmonics', max_harmonics, '', 'Highest harmonics of the Fourier decomposition.')
    cfg.add('flipWaveEOD', flip_wave, '', 'Flip EOD of wave fish to make largest extremum positive (flip, none, or auto).')
    cfg.add('waveEODThresholdFraction', 100*thresh_frac, '%', 'Threshold for detecting peaks and troughs in wave-type EODs as a fraction of the p-p amplitude.')
    cfg.add('waveEODPhaseHarmonics', n_phase_harmonics, '', 'Number of harmonics over which to compute the slope of the phases..')


def analyze_wave_args(cfg):
    """Retrieve parameters for `analyze_wave()` from configuration.
    
    The return value can then be passed as key-word arguments to this
    function.

    Parameters
    ----------
    cfg: ConfigFile
        The configuration.

    Returns
    -------
    a: dict
        Dictionary with names of arguments of the `analyze_wave()` function
        and their values as supplied by `cfg`.
    """
    a = cfg.map(max_harmonics='eodHarmonics',
                flip_wave='flipWaveEOD',
                thresh_frac='waveEODThresholdFraction',
                n_phase_harmonics='waveEODPhaseHarmonics')
    a['thresh_frac'] *= 0.01
    return a


def wave_quality_args(cfg):
    """Retrieve parameters for `wave_quality()` from configuration.
    
    The return value can then be passed as key-word arguments to this
    function.

    Parameters
    ----------
    cfg: ConfigFile
        The configuration.

    Returns
    -------
    a: dict
        Dictionary with names of arguments of the `wave_quality()` function
        and their values as supplied by `cfg`.

    See Also
    --------
    eodanalysis.add_eod_quality_config()
    
    """
    a = cfg.map(min_freq='minimumFrequency',
                max_freq='maximumFrequency',
                max_clipped_frac='maximumClippedFraction',
                max_phases='maximumPhases',
                max_rms_sem='maximumVariance',
                max_rms_error='maximumRMSError',
                min_power='minimumPower',
                max_thd='maximumTotalHarmonicDistortion',
                max_db_diff='maximumPowerDifferences',
                max_relampl_harm2='maximumSecondHarmonicAmplitude',
                max_relampl_harm3='maximumThirdHarmonicAmplitude',
                max_relampl_harm4='maximumFourthHarmonicAmplitude')
    a['max_clipped_frac'] *= 0.01
    return a


def main():
    import matplotlib.pyplot as plt
    from thunderlab.eventdetection import snippets
    from .fakefish import wavefish_eods, export_wavefish

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
    plot_wave_eod(axs['w'], mean_eod, props, unit=unit)
    axs['w'].set_title(f'wave fish: EODf = {props["EODf"]:.1f} Hz')
    plot_wave_spectrum(axs['a'], axs['p'], power, props)
    plt.show()


if __name__ == '__main__':
    main()
