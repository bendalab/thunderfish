"""
Analysis of wave-type EOD waveforms.

## Analysis of wave-type EODs

- `waveeod_waveform()`: retrieve average EOD waveform via Fourier transform.
- `analyze_wave()`: analyze the EOD waveform of a wave fish.

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
except ImportError:
    pass

from pathlib import Path
from scipy.optimize import curve_fit
from numba import jit
from thunderlab.eventdetection import threshold_crossings, threshold_crossing_times, merge_events
from thunderlab.powerspectrum import decibel
from thunderlab.tabledata import TableData

from .harmonics import fundamental_freqs_and_power


def waveeod_waveform(data, rate, freq, win_fac=2.0):
    """Retrieve average EOD waveform via Fourier transform.

    TODO: use power spectra to check minimum data segment needed and
    check for changes in frequency over several segments!

    TODO: return waveform with higher samplign rate? (important for
    2kHz waves on 24kHz sampling). But this seems to render some EODs
    inacceptable in the further thunderfish processing pipeline.

    Parameters
    ----------
    data: 1-D array of float
        The data to be analysed.
    rate: float
        Sampling rate of the data in Hertz.
    freq: float
        EOD frequency.
    win_fac: float
        The snippet size is the EOD period times `win_fac`. The EOD period
        is determined as the minimum interval between EOD times.
    
    Returns
    -------
    mean_eod: 2-D array
        Average of the EOD snippets. First column is time in seconds,
        second column the mean eod, third column the standard error.
    eod_times: 1-D array
        Times of EOD peaks in seconds that have been actually used to
        calculate the averaged EOD waveform.

    """

    @jit(nopython=True)
    def fourier_wave(data, rate, freq):
        """
        extracting wave via fourier coefficients
        """
        twave = np.arange(0, (1+win_fac)/freq, 1/rate)
        wave = np.zeros(len(twave))
        t = np.arange(len(data))/rate
        for k in range(0, 31):
            Xk = np.trapz(data*np.exp(-1j*2*np.pi*k*freq*t), t)*2/t[-1]
            wave += np.real(Xk*np.exp(1j*2*np.pi*k*freq*twave))
        return wave

    @jit(nopython=True)
    def fourier_range(data, rate, f0, f1, df):
        wave = np.zeros(1)
        freq = f0
        for f in np.arange(f0, f1, df):
            w = fourier_wave(data, rate, f)
            if np.max(w) - np.min(w) > np.max(wave) - np.min(wave):
                wave = w
                freq = f
        return wave, freq

    # TODO: parameterize!
    tsnippet = 2
    min_corr = 0.98
    min_ampl_frac = 0.5
    frange = 0.1
    fstep = 0.1
    waves = []
    freqs = []
    times = []
    step = int(tsnippet*rate)
    for i in range(0, len(data) - step//2, step//2):
        w, f = fourier_range(data[i:i + step], rate, freq - frange,
                             freq + frange + fstep/2, fstep)
        waves.append(w)
        freqs.append(f)
        """
        waves.append(np.zeros(1))
        freqs.append(freq)
        for f in np.arange(freq - frange, freq + frange + fstep/2, fstep):
            w = fourier_wave(data[i:i + step], rate, f)
            if np.max(w) - np.min(w) > np.max(waves[-1]) - np.min(waves[-1]):
                waves[-1] = w
                freqs[-1] = f
        """
        times.append(np.arange(i/rate, (i + step)/rate, 1/freqs[-1]))
    eod_freq = np.mean(freqs)
    mean_eod = np.zeros((0, 3))
    eod_times = np.zeros((0))
    if len(waves) == 0:
        return mean_eod, eod_times
    for k in range(len(waves)):
        period = int(np.ceil(rate/freqs[k]))
        i = np.argmax(waves[k][:period])
        waves[k] = waves[k][i:]
    n = np.min([len(w) for w in waves])
    waves = np.array([w[:n] for w in waves])
    # only snippets that are similar:
    if len(waves) > 1:
        corr = np.corrcoef(waves)
        nmax = np.argmax(np.sum(corr > min_corr, axis=1))
        if nmax <= 1:
            nmax = 2
        select = np.sum(corr > min_corr, axis=1) >= nmax
        waves = waves[select]
        times = [times[k] for k in range(len(times)) if select[k]]
        if len(waves) == 0:
            return mean_eod, eod_times
    # only the largest snippets:
    ampls = np.std(waves, axis=1)
    select = ampls >= min_ampl_frac*np.max(ampls)
    waves = waves[select]
    times = [times[k] for k in range(len(times)) if select[k]]
    if len(waves) == 0:
        return mean_eod, eod_times
    """
    #plt.plot(freqs)
    plt.plot(waves.T)
    plt.show()
    """
    mean_eod = np.zeros((n, 3))
    mean_eod[:, 0] = np.arange(len(mean_eod))/rate
    mean_eod[:, 1] = np.mean(waves, axis=0)
    mean_eod[:, 2] = np.std(waves, axis=0)
    eod_times = np.concatenate(times)
    return mean_eod, eod_times


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


def analyze_wave(eod, freq, n_harm=10, power_n_harmonics=0,
                 n_harmonics=3, flip_wave='none'):
    """Analyze the EOD waveform of a wave fish.
    
    Parameters
    ----------
    eod: 2-D array
        The eod waveform. First column is time in seconds, second
        column the EOD waveform, third column, if present, is the
        standard error of the EOD waveform, Further columns are
        optional but not used.
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
    error_str = ''
    
    freq0 = freq
    if hasattr(freq, 'shape'):
        freq0 = freq[0][0]
        
    # storage:
    meod = np.zeros((eod.shape[0], eod.shape[1]+1))
    meod[:, :-1] = eod

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
    rmssem = np.sqrt(np.mean(meod[i0:i1,2]**2.0))/ppampl if eod.shape[1] > 2 else None
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
                   ['harmonics', 'frequency', 'amplitude', 'relampl', 'relpower', 'phase'],
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

