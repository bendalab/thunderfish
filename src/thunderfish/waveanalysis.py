"""
Analysis of wave-type EOD waveforms.

## Analysis of wave-type EODs

- `extract_wave()`: retrieve average EOD waveform via Fourier transform.
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

## Fit functions

- `fourier_series()`: Fourier series of sine waves with amplitudes and phases.
- `exp_decay()`: exponential decay.
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
from thunderlab.powerspectrum import decibel
from thunderlab.tabledata import TableData

from .harmonics import fundamental_freqs_and_power


def extract_wave(data, rate, freq, freq_resolution, periods=5,
                 frate=1e6, max_harmonics=21,
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
    max_hamonics: int
        Compute Fourier series up to this order.
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
    mean_eod: 2-D array
        Average of the EOD snippets. First column is time in seconds,
        second column the mean eod, third column the standard error.
    eod_freq: float
        Refined EOD frequency.
    times: 1-D array
        Start times of windows in which Fourier series have been extracted.
    skip_reason: str
        An empty string if the waveform is good, otherwise a string
        indicating the failure.

    """

    #@jit(nopython=True)
    def fourier_wave(data, rate, freq, nh, frate, n):
        """
        Extract wave via fourier coefficients
        """
        t = np.arange(len(data))/rate
        coeffs = np.zeros(nh, dtype=complex)
        for k in range(nh):
            coeffs[k] = np.trapz(data*np.exp(-1j*2*np.pi*k*freq*t), t)*2/t[-1]
        # set phase of first harmonics to zero:
        phi0 = np.angle(coeffs[1])
        for k in range(1, nh):
            coeffs[k] *= np.exp(-1j*k*phi0)
        twave = np.arange(n)/frate
        wave = np.zeros(len(twave))
        for k in range(nh):
            wave += np.real(coeffs[k]*np.exp(1j*2*np.pi*k*freq*twave))
        return wave

    #@jit(nopython=True)   with jit it takes longer???
    def fourier_freq_range(data, rate, frange, nh, n):
        wave = np.zeros(1)
        freq = 0.0
        for f in frange:
            twave = np.arange(n)/rate
            w = np.zeros(len(twave))
            t = np.arange(len(data))/rate
            for k in range(nh):
                Xk = np.trapz(data*np.exp(-1j*2*np.pi*k*f*t), t)*2/t[-1]
                w += np.real(Xk*np.exp(1j*2*np.pi*k*f*twave))
            if np.max(w) - np.min(w) > np.max(wave) - np.min(wave):
                wave = w
                freq = f
        return wave, freq

    # reduce frequency resolution and time window for high frequency fish:
    ffac = max(1, int(np.round(freq/400)))
    freq_resolution *= ffac
    tsnippet = 2/freq_resolution   # twice as long window is essential!
    nfreqs = 1 + 2*2*ffac          # twice the frequency resolution is necessary and sufficient!
    step = int(tsnippet*rate)
    # extract Fourier series from data segements:
    n = int(periods/freq*rate)
    freqs = []
    indices = np.arange(0, max(1, len(data) - step + 1), step//8)
    if len(indices) <= 1:
        step //= 2
        indices = np.arange(0, max(1, len(data) - step + 1), step//8)
    frange = np.linspace(freq - freq_resolution, freq + freq_resolution, nfreqs)
    for i in indices:
        w, f = fourier_freq_range(data[i:i + step], rate, frange, 6, n)
        freqs.append(f)
    freqs = np.array(freqs)
    mean_eod = np.zeros((0, 3))
    if len(freqs) == 0:
        # TODO: Why??? How can indices be empty?
        return mean_eod, freq, np.array([]), f'no frequencies detected ({len(indicies)} indices, freqs={freqs})'
    n = int(periods/np.mean(freqs)*frate)
    waves = np.zeros((len(indices), n))
    for k in range(len(indices)):
        i = indices[k]
        w = fourier_wave(data[i:i + step], rate, freqs[k],
                         max_harmonics, frate, n)
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
        print(f'extract {freq:7.2f}Hz wave  fish: {len(indices)} segments, EODf={eodf:.2f}Hz')
        # TODO: what to do with single segment?
    else:
        corr = np.corrcoef(waves)
        np.fill_diagonal(corr, 0.0)
        corr_vals = np.sort(corr[corr > min_corr])
        if len(corr_vals) == 0:
            if plot_level > 0:
                plt.show()
            return mean_eod, freq, indices/rate, f'waveforms not stable (max_corr={np.max(corr):.4f} SMALLER than {min_corr:.4f})'
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
        freqs = freqs[mask]
        indices = indices[mask]
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
            np.set_printoptions(formatter={'float': lambda x: f'{x:.2f}'})
            eodf = np.mean(freqs) if len(freqs) > 0 else np.nan
            print(f'extract {freq:7.2f}Hz wave  fish: min_corr={min_c:.4f}, max_corr={corr_vals[-1]:.4f}, num_cmax={num_cmax}, segments={len(corr)}, num_selected={np.sum(mask)}, selected={np.nonzero(mask)[0]}, EODfs={freqs}, EODf={eodf:.2f}Hz')
        if len(waves) == 0:
            if plot_level > 0:
                plt.show()
            return mean_eod, freq, indices/rate, f'waveforms not stable (min_corr={min_c:.4f}, max_corr={corr_vals[-1]:.4f}, num_cmax={num_cmax})'
    # only the largest snippets:
    ampls = np.std(waves, axis=1)
    mask = ampls >= min_ampl_frac*np.max(ampls)
    if verbose > 0 and np.sum(mask) < len(ampls):
        print(f'                              removed {len(ampls) - np.sum(mask)} small amplitude segments')
    waves = waves[mask]
    freqs = freqs[mask]
    indices = indices[mask]
    if len(waves) == 0:
        if plot_level > 0:
            plt.show()
        return mean_eod, freq, indices/rate, 'ERROR: no large waveform'
    if plot_level > 0:
        axs[1, 0].set_title(f'selected EODs: EODf={np.mean(freqs):.2f}Hz')
        t = np.arange(waves.shape[1])*1000/frate
        for w in waves:
            axs[1, 0].plot(t, w)
        axs[1, 0].set_xlabel('time [ms]')
        plt.show()

    mean_eod = np.zeros((n, 3))
    mean_eod[:, 0] = np.arange(len(mean_eod))/frate
    mean_eod[:, 1] = np.mean(waves, axis=0)
    mean_eod[:, 2] = np.std(waves, axis=0)
    eod_freq = np.mean(freqs)
    return mean_eod, eod_freq, indices/rate, ''


def fourier_series(t, freq, *ap):
    """Fourier series of sine waves with amplitudes and phases.

    x(t) = sum_{i=0}^n ap[2*i]*sin(2 pi (i+1) freq t + ap[2*i+1])
    
    Parameters
    ----------
    t: float or array
        Time.
    freq: float
        Fundamental frequency.
    *ap: list of floats
        The amplitudes and phases (in rad) of the fundamental and harmonics.
        
    Returns
    -------
    x: float or array
        The Fourier series evaluated at times `t`.
    """
    omega = 2.0*np.pi*freq
    x = 0.0
    for i, (a, p) in enumerate(zip(ap[0:-1:2], ap[1::2])):
        x += a*np.sin((i+1)*omega*t+p)
    return x


def analyze_wave(eod, ratetime, freq, n_harm=10, power_n_harmonics=0,
                 n_harmonics=3, flip_wave='none'):
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
    n_harm: int
        Maximum number of harmonics used for the Fourier decomposition.
    power_n_harmonics: int
        Sum over the first `power_n_harmonics` harmonics for computing
        the total power.  If 0 sum over all harmonics.
    n_harmonics: int
        The maximum power of higher harmonics is computed from
        harmonics higher than the maximum harmonics within the first
        three harmonics plus `n_harmonics`.
    flip_wave: 'auto', 'none', 'flip'
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

    spec_data: 2-D array of floats
        First six columns are from the spectrum of the extracted
        waveform.  First column is the index of the harmonics, second
        column its frequency, third column its amplitude, fourth
        column its amplitude relative to the fundamental, fifth column
        is power of harmonics relative to fundamental in decibel, and
        sixth column the phase shift relative to the fundamental.
        If `freq` is a list of harmonics, a seventh column is added to
        `spec_data` that contains the powers of the harmonics from the
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
    else:
        if isinstance(ratetime, (list, tuple, np.ndarray)):
            time = ratetime
        else:
            time = np.arange(len(eod))/ratetime
        eeod = np.zeros((len(eod), 3))
        eeod[:, 0] = time
        eeod[:, 1] = eod
    # storage:
    meod = np.zeros((eeod.shape[0], eeod.shape[1] + 1))
    meod[:, :eeod.shape[1]] = eeod
    meod[:, -1] = np.nan
    
    freq0 = freq
    if hasattr(freq, 'shape'):
        freq0 = freq[0][0]

    error_str = ''
        
    # subtract mean and flip:
    period = 1.0/freq0
    pinx = int(np.ceil(period/(meod[1,0]-meod[0,0])))
    maxn = (len(meod)//pinx)*pinx
    if maxn < pinx: maxn = len(meod)
    offs = (len(meod) - maxn)//2
    meod[:, 1] -= np.mean(meod[offs:offs+pinx,1])
    flipped = False
    if 'flip' in flip_wave or ('auto' in flip_wave and -np.min(meod[:, 1]) > np.max(meod[:, 1])):
        meod[:, 1] = -meod[:, 1]
        flipped = True
    
    # move peak of waveform to zero:
    offs = len(meod)//4
    maxinx = offs+np.argmax(meod[offs:3*offs,1])
    meod[:, 0] -= meod[maxinx,0]
    
    # indices of exactly one or two periods around peak:
    if len(meod) < pinx:
        raise IndexError('data need to contain at least one EOD period')
    if len(meod) >= 2*pinx:
        i0 = maxinx - pinx if maxinx >= pinx else 0
        i1 = i0 + 2*pinx
        if i1 > len(meod):
            i1 = len(meod)
            i0 = i1 - 2*pinx
    else:
        i0 = maxinx - pinx//2 if maxinx >= pinx//2 else 0
        i1 = i0 + pinx

    # subtract mean:
    meod[:, 1] -= np.mean(meod[i0:i1,1])

    # zero crossings:
    ui, di = threshold_crossings(meod[:, 1], 0.0)
    ut, dt = threshold_crossing_times(meod[:, 0], meod[:, 1], 0.0, ui, di)
    ut, dt = merge_events(ut, dt, 0.02/freq0)
    ncrossings = int(np.round((len(ut) + len(dt))/(meod[-1,0]-meod[0,0])/freq0))
    if np.any(ut<0.0):    
        up_time = ut[ut<0.0][-1]
    else:
        up_time = 0.0 
        error_str += '%.1f Hz wave fish: no upward zero crossing. ' % freq0
    if np.any(dt>0.0):
        down_time = dt[dt>0.0][0]
    else:
        down_time = 0.0
        error_str += '%.1f Hz wave fish: no downward zero crossing. ' % freq0
    peak_width = down_time - up_time
    trough_width = period - peak_width
    peak_time = 0.0
    trough_time = meod[maxinx+np.argmin(meod[maxinx:maxinx+pinx,1]),0]
    phase1 = peak_time - up_time
    phase2 = down_time - peak_time
    phase3 = trough_time - down_time
    phase4 = up_time + period - trough_time
    distance = trough_time - peak_time
    min_distance = distance
    if distance > period/2:
        min_distance = period - distance
    
    # fit fourier series:
    ampl = 0.5*(np.max(meod[:, 1])-np.min(meod[:, 1]))
    while n_harm > 1:
        params = [freq0]
        for i in range(1, n_harm+1):
            params.extend([ampl/i, 0.0])
        try:
            popt, pcov = curve_fit(fourier_series, meod[i0:i1,0],
                                   meod[i0:i1,1], params, maxfev=2000)
            break
        except (RuntimeError, TypeError):
            error_str += '%.1f Hz wave fish: fit of fourier series failed for %d harmonics. ' % (freq0, n_harm)
            n_harm //= 2
    # store fourier fit:
    meod[:, -1] = fourier_series(meod[:, 0], *popt)
    # make all amplitudes positive:
    for i in range(n_harm):
        if popt[i*2+1] < 0.0:
            popt[i*2+1] *= -1.0
            popt[i*2+2] += np.pi
    # phases relative to fundamental:
    # phi0 = 2*pi*f0*dt <=> dt = phi0/(2*pi*f0)
    # phik = 2*pi*i*f0*dt = i*phi0
    phi0 = popt[2]
    for i in range(n_harm):
        popt[i*2+2] -= (i + 1)*phi0
        # all phases in the range -pi to pi:
        popt[i*2+2] %= 2*np.pi
        if popt[i*2+2] > np.pi:
            popt[i*2+2] -= 2*np.pi
    # store fourier spectrum:
    if hasattr(freq, 'shape'):
        n = n_harm
        n += np.sum(freq[:, 0] > (n_harm+0.5)*freq[0,0])
        spec_data = np.zeros((n, 7))
        spec_data[:, :] = np.nan
        k = 0
        for i in range(n_harm):
            while k < len(freq) and freq[k,0] < (i+0.5)*freq0:
                k += 1
            if k >= len(freq):
                break
            if freq[k,0] < (i+1.5)*freq0:
                spec_data[i,6] = freq[k,1]
                k += 1
        for i in range(n_harm, n):
            if k >= len(freq):
                break
            spec_data[i,:2] = [np.round(freq[k,0]/freq0)-1, freq[k,0]]
            spec_data[i,6] = freq[k,1]
            k += 1
    else:
        spec_data = np.zeros((n_harm, 6))
    for i in range(n_harm):
        spec_data[i,:6] = [i, (i+1)*freq0, popt[i*2+1], popt[i*2+1]/popt[1],
                           decibel((popt[i*2+1]/popt[1])**2.0), popt[i*2+2]]
    # smoothness of power spectrum:
    db_powers = decibel(spec_data[:n_harm,2]**2)
    db_diff = np.std(np.diff(db_powers))
    # maximum relative power of higher harmonics:
    p_max = np.argmax(db_powers[:3])
    db_powers -= db_powers[p_max]
    if len(db_powers[p_max+n_harmonics:]) == 0:
        max_harmonics_power = -100.0
    else:
        max_harmonics_power = np.max(db_powers[p_max+n_harmonics:])
    # total harmonic distortion:
    thd = np.sqrt(np.nansum(spec_data[1:, 3]))

    # peak-to-peak and trough amplitudes:
    ppampl = np.max(meod[i0:i1,1]) - np.min(meod[i0:i1,1])
    relpeakampl = max(np.max(meod[i0:i1,1]), np.abs(np.min(meod[i0:i1,1])))/ppampl
    
    # variance and fit error:
    rmssem = np.sqrt(np.mean(meod[i0:i1,2]**2.0))/ppampl if meod.shape[1] > 2 else None
    rmserror = np.sqrt(np.mean((meod[i0:i1,1] - meod[i0:i1,-1])**2.0))/ppampl

    # store results:
    props = {}
    props['type'] = 'wave'
    props['EODf'] = freq0
    props['p-p-amplitude'] = ppampl
    props['flipped'] = flipped
    props['amplitude'] = 0.5*ppampl  # remove it
    props['rmserror'] = rmserror
    if rmssem:
        props['noise'] = rmssem
    props['ncrossings'] = ncrossings
    props['peakwidth'] = peak_width/period
    props['troughwidth'] = trough_width/period
    props['minwidth'] = min(peak_width, trough_width)/period
    props['leftpeak'] = phase1/period
    props['rightpeak'] = phase2/period
    props['lefttrough'] = phase3/period
    props['righttrough'] = phase4/period
    props['p-p-distance'] = distance/period
    props['min-p-p-distance'] = min_distance/period
    props['relpeakampl'] = relpeakampl
    pnh = power_n_harmonics if power_n_harmonics > 0 else n_harm
    pnh = min(n_harm, pnh)
    props['power'] = decibel(np.sum(spec_data[:pnh,2]**2.0))
    if hasattr(freq, 'shape'):
        props['datapower'] = decibel(np.sum(freq[:pnh,1]))
    props['thd'] = thd
    props['dbdiff'] = db_diff
    props['maxdb'] = max_harmonics_power
    
    return meod, props, spec_data, error_str


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
        The amplitude spectrum of a single pulse as returned by
        `analyze_wave()`.  First column is the index of the harmonics,
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
    markers, stemlines, _ = axa.stem(spec[:n, 0] + 1, spec[:n, 2],
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
    markers, stemlines, _ = axp.stem(spec[:n, 0] + 1, phases[:n],
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
    td.append('EODf', 'Hz', '%7.2f', value=wave_props)
    td.append('p-p-amplitude', unit, '%.5f', value=wave_props)
    td.append('power', 'dB', '%7.2f', value=wave_props)
    if 'datapower' in wave_props[0]:
        td.append('datapower', 'dB', '%7.2f', value=wave_props)
    td.append('thd', '%', '%.2f', value=wave_props, fac=100.0)
    td.append('dbdiff', 'dB', '%7.2f', value=wave_props)
    td.append('maxdb', 'dB', '%7.2f', value=wave_props)
    if 'noise' in wave_props[0]:
        td.append('noise', '%', '%.1f', value=wave_props, fac=100.0)
    td.append('rmserror', '%', '%.2f', value=wave_props, fac=100.0)
    if 'clipped' in wave_props[0]:
        td.append('clipped', '%', '%.1f', value=wave_props, fac=100.0)
    td.append('flipped', '', '%d', value=wave_props)
    td.append('n', '', '%5d', value=wave_props)
    td.append_section('timing')
    td.append('ncrossings', '', '%d', value=wave_props)
    td.append('peakwidth', '%', '%.2f', value=wave_props, fac=100.0)
    td.append('troughwidth', '%', '%.2f', value=wave_props, fac=100.0)
    td.append('minwidth', '%', '%.2f', value=wave_props, fac=100.0)
    td.append('leftpeak', '%', '%.2f', value=wave_props, fac=100.0)
    td.append('rightpeak', '%', '%.2f', value=wave_props, fac=100.0)
    td.append('lefttrough', '%', '%.2f', value=wave_props, fac=100.0)
    td.append('righttrough', '%', '%.2f', value=wave_props, fac=100.0)
    td.append('p-p-distance', '%', '%.2f', value=wave_props, fac=100.0)
    td.append('min-p-p-distance', '%', '%.2f',
              value=wave_props, fac=100.0)
    td.append('relpeakampl', '%', '%.2f', value=wave_props, fac=100.0)
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
        props['index'] = int(props['index'])
        props['n'] = int(props['n'])
        props['type'] = 'wave'
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
        props['p-p-distance'] /= 100
        props['min-p-p-distance'] /= 100
        props['relpeakampl'] /= 100
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
