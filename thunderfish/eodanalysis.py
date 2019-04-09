"""
# Analyse EOD waveforms.

## EOD analysis
- `eod_waveform()`: compute an averaged EOD waveform.
- `unfilter()`: apply inverse low-pass filter on data.
- `analyze_wave()`: analyze the EOD waveform of a wave-type fish.
- `analyze_pulse()`: analyze the EOD waveform of a pulse-type fish.

## Visualization
- `eod_recording_plot()`: plot a zoomed in range of the recorded trace.
- `eod_waveform_plot()`: plot and annotate the averaged EOD-waveform with standard deviation.
- `wave_spectrum_plot()`: plot and annotate spectrum of wave-type EODs.
- `pulse_spectrum_plot()`: plot and annotate spectrum of single pulse-type EOD.

## Fit functions
- `fourier_series()`: Fourier series of sine waves with amplitudes and phases.
- `exp_decay()`: expontenial decay.

## Configuration parameter
- `add_eod_analysis_config()': add parameters for EOD analysis functions to configuration.
- `eod_waveform_args()`: retrieve parameters for `eod_waveform()` from configuration.
- `analyze_wave_args()`: retrieve parameters for `analyze_wave()` from configuration.
- `analyze_pulse_args()`: retrieve parameters for `analyze_pulse()` from configuration.
- `add_eod_quality_config()`: add parameters needed for assesing the quality of an EOD waveform.
"""

import numpy as np
from scipy.optimize import curve_fit
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from .eventdetection import percentile_threshold, detect_peaks, snippets, peak_width
from .eventdetection import threshold_crossings, threshold_crossing_times
from .powerspectrum import psd, nfft_noverlap, decibel


def eod_waveform(data, samplerate, thresh_fac=0.8, percentile=1.0,
                 win_fac=2.0, min_win=0.01, max_eods=None, period=None):
    """Detect EODs in the given data, extract data snippets around each EOD,
    and compute a mean waveform with standard deviation.

    Parameters
    ----------
    data: 1-D array
        The data to be analysed.
    samplerate: float
        Sampling rate of the data in Hertz.
    percentile: float
        Percentile parameter in percent for the eventdetection.percentile_threshold() function
        used to estimate thresholds for detecting EOD peaks in the data.
    thresh_fac: float
        thresh_fac parameter for the eventdetection.percentile_threshold() function used to
        estimate thresholds for detecting EOD peaks in the data.
    win_fac: float
        The snippet size is the period times `win_fac`.
    min_win: float
        The minimum size of the snippets in seconds.
    max_eods: int or None
        Maximum number of EODs to be used for averaging.
    period: float or None
        Average waveforms with this period instead of peak times.
    
    Returns
    -------
    mean_eod: 2-D array
        Average of the EOD snippets. First column is time in seconds,
        second column the mean eod, third column the standard deviation
    eod_times: 1-D array
        Times of EOD peaks in seconds.
    """
    if period is None:
        # threshold for peak detection:
        threshold = percentile_threshold(data, thresh_fac=thresh_fac, percentile=percentile)

        # detect peaks:
        eod_idx, _ = detect_peaks(data, threshold)
        if len(eod_idx) == 0:
            return np.array([]), np.array([]), np.array([]), np.array([])

        # eod indices and times:
        eod_times = eod_idx / samplerate
    else:
        eod_times = np.arange(0.0, len(data)/samplerate, period)
        eod_idx = np.asarray(eod_times * samplerate, dtype=int)

    # window size:
    tmp_period = period
    if tmp_period is None:
        tmp_period = np.mean(np.diff(eod_times))
    win = 0.5*win_fac*tmp_period
    if 2*win < min_win:
        win = 0.5*min_win
    win_inx = int(win * samplerate)

    # extract snippets:
    if max_eods and max_eods > 0 and len(eod_idx) > max_eods:
        dn = (len(eod_idx) - max_eods)//2
        eod_idx = eod_idx[dn:dn+max_eods]
    eod_snippets = snippets(data, eod_idx, -win_inx, win_inx)

    # mean and std of snippets:
    mean_eod = np.zeros((len(eod_snippets[0]), 3))
    mean_eod[:,1] = np.mean(eod_snippets, axis=0)
    if len(eod_snippets) > 1:
        mean_eod[:,2] = np.std(eod_snippets, axis=0, ddof=1)

    # time axis:
    mean_eod[:,0] = (np.arange(len(mean_eod)) - win_inx) / samplerate
    
    return mean_eod, eod_times


def unfilter(data, samplerate, tau=1.0, cutoff=None):
    """
    Apply inverse high-pass filter on data.

    Either the timeconstant `tau` or the cutoff frequency `cutoff` of the
    high-pass filter need to be specified.

    Assumes high-pass filter
    \[ \tau \dot y = -y + \tau \dot x \]
    has been applied on the original data $x$. To recover $x$
    the ODE
    \[ \tau \dot x = y + \tau \dot y \]
    is applied on the filtered data $y$.

    Parameters:
    -----------
    data: ndarray
        High-pass filtered original data.
    samplerate: float
        Sampling rate of `data` in Hertz.
    tau: float
        Time-constant of the high-pass filter in seconds.
    cutoff: float
        Cutoff frequency of the high-passfilter. Overwrites `tau` if specified.

    Returns:
    --------
    data: ndarray
        Recovered original data.
    """
    if cutoff:
        tau = 0.5/np.pi/cutoff
    fac = tau*samplerate
    data -= np.mean(data)
    d0 = data[0]
    x = d0
    for k in range(len(data)):
        d1 = data[k]
        x += (d1 - d0) + d0/fac
        data[k] = x
        d0 = d1
    return data


def fourier_series(t, freq, delay, ampl, *ap):
    """
    Fourier series of sine waves with amplitudes and phases.

    x(t) = ampl sin(2 pi freq (t-delay) + ampl sum_{k=0}^n ap[2*i]*sin(2 pi (k+2) freq (t-delay) + ap[2*i+1])
    
    Parameters
    ----------
    t: float or array
        Time.
    freq: float
        Fundamental frequency.
    delay: float
        Shift of sinewaves in time.
    ampl: float
        Amplitude of the sinewave with the fundamental frequency.
    *ap: list of floats
        The relative amplitudes and phases (in rad) of the harmonics.
        
    Returns
    -------
    x: float or array
        The Fourier series evaluated at times `t`.
    """
    tt = t - delay
    omega = 2.0*np.pi*freq
    x = np.sin(omega*tt)
    for k, (a, p) in enumerate(zip(ap[0:-1:2], ap[1::2])):
        x += a*np.sin((k+2)*omega*tt+p)
    return ampl*x


def analyze_wave(eod, freq, n_harm=20, power_n_harmonics=1000, flip_wave='none'):
    """
    Analyze the EOD waveform of a wave-type fish.
    
    Parameters
    ----------
    eod: 2-D array
        The eod waveform. First column is time in seconds, second column the EOD waveform,
        third column, if present, is the standarad deviation of the EOD waveform,
        Further columns are optional but not used.
    freq: float or 2-D array
        The frequency of the EOD or the list of harmonics (rows)
        with frequency and peak height (columns) as returned from `harmonic_groups()`.
    n_harm: int
        Maximum number of harmonics used for the fit.
    power_n_harmonics: int
        Sum over the first `power_n_harmonics` harmonics for computing the total power.
        If 0 sum over all harmonics.
    flip_wave: 'auto', 'none', 'flip'
        - 'auto' flip waveform such that the larger extremum is positive.
        - 'flip' flip waveform.
        - 'none' do not flip waveform.
    
    Returns
    -------
    meod: 2-D array of floats
        The eod waveform. First column is time in seconds, second column the eod waveform.
        Further columns are kept from the input `eod`. And a column is added with the
        fit of the fourier series to the waveform.
    props: dict
        A dictionary with properties of the analyzed EOD waveform.
        - type: set to 'wave'.
        - EODf: is set to the EOD fundamental frequency.
        - p-p-amplitude: peak-to-peak amplitude of the Fourier fit.
        - flipped: True if the waveform was flipped.
        - amplitude: amplitude factor of the Fourier fit.
        - rmvariance: root-mean variance of the averaged EOD waveform relative to
          the p-p amplitude (only if a standard deviation is given in `eod`).
        - rmserror: root-mean-square error between Fourier-fit and EOD waveform relative to
          the p-p amplitude. If larger than 0.05 the data are bad.
        - peakwidth: width of the peak at the averaged amplitude relative to EOD period.
        - troughwidth: width of the trough at the averaged amplitude relative to EOD period.
        - leftpeak: time from positive zero crossing to peak relative to EOD period.
        - rightpeak: time from peak to negative zero crossing relative to EOD period.
        - lefttrough: time from negative zero crossing to trough relative to EOD period.
        - righttrough: time from trough to positive zero crossing relative to EOD period.
        - p-p-distance: time between peak and trough relative to EOD period.
        - power: summed power of all harmonics in decibel relative to one.
    spec_data: 2-D array of floats
        First column is the index of the harmonics, second column its frequency,
        third column its amplitude, fourth column its amplitude relative to the fundamental,
        fifth column is power of harmonics relative to fundamental in decibel,
        and sixth column the phase shift relative to the fundamental.
        If `freq` is a list of harmonics, a seventh column is added to `spec_data`
        that contains the powers of the harmonics from the original power spectrum of the
        raw data.
        Rows are the harmonics, first row is the fundamental frequency with index 0,
        relative amplitude of one, relative power of 0dB, and phase shift of zero.
        If the relative amplitude of the first harmonic (spec-data[1,3]) is larger than 2,
        or the relative amplitude of the second harmonic (spec-data[2,3]) is larger than 0.2,
        then this probably is not a proper EOD waveform and
        should not be used for further analysis.
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
    meod[:,:-1] = eod

    # subtract mean and flip:
    period = 1.0/freq0
    pinx = int(np.ceil(period/(meod[1,0]-meod[0,0])))
    maxn = (len(meod)//pinx)*pinx
    if maxn < pinx: maxn = len(meod)
    offs = (len(meod) - maxn)//2
    meod[:,1] -= np.mean(meod[offs:offs+pinx,1])
    flipped = False
    if 'flip' in flip_wave or ('auto' in flip_wave and -np.min(meod[:,1]) > np.max(meod[:,1])):
        meod[:,1] = -meod[:,1]
        flipped = True
    
    # move peak of waveform to zero:
    offs = len(meod)//4
    maxinx = offs+np.argmax(meod[offs:3*offs,1])
    meod[:,0] -= meod[maxinx,0]
    
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
    meod[:,1] -= np.mean(meod[i0:i1,1])

    # zero crossings:
    ui, di = threshold_crossings(meod[:,1], 0.0)
    ut, dt = threshold_crossing_times(meod[:,0], meod[:,1], 0.0, ui, di)
    uidx = np.argmax(ui>maxinx-pinx//2)
    didx = np.argmax(di>ui[uidx])
    up_time = ut[uidx]
    down_time = dt[didx]
    peak_width = down_time - up_time
    trough_width = period - peak_width
    peak_time = 0.0
    trough_time = meod[maxinx+np.argmin(meod[maxinx:maxinx+pinx,1]),0]
    phase1 = peak_time - up_time
    phase2 = down_time - peak_time
    phase3 = trough_time - down_time
    phase4 = up_time + period - trough_time
    distance = trough_time - peak_time
    
    # fit fourier series:
    ampl = 0.5*(np.max(meod[:,1])-np.min(meod[:,1]))
    while n_harm > 1:
        params = [freq0, -0.25/freq0, ampl]
        for i in range(1, n_harm):
            params.extend([1.0/(i+1), 0.0])
        try:
            popt, pcov = curve_fit(fourier_series, meod[i0:i1,0], meod[i0:i1,1],
                                   params, maxfev=2000)
            break
        except (RuntimeError, TypeError):
            error_str = '%.1f Hz wave-type fish: fit of fourier series failed for %d harmonics.' % (freq0, n_harm)
            n_harm //= 2
    ampl = popt[2]
    for i in range(1, n_harm):
        # make all amplitudes positive:
        if popt[1+i*2] < 0.0:
            popt[1+i*2] *= -1.0
            popt[2+i*2] += np.pi
        # all phases in the range -pi to pi:
        popt[2+i*2] %= 2.0*np.pi
        if popt[2+i*2] > np.pi:
            popt[2+i*2] -= 2.0*np.pi
    meod[:,-1] = fourier_series(meod[:,0], *popt)

    # variance and fit error:
    ppampl = np.max(meod[i0:i1,1]) - np.min(meod[i0:i1,1])
    rmvariance = np.sqrt(np.mean(meod[i0:i1,2]**2.0))/ppampl if eod.shape[1] > 2 else None
    rmserror = np.sqrt(np.mean((meod[i0:i1,1] - meod[i0:i1,-1])**2.0))/ppampl

    # store results:
    props = {}
    props['type'] = 'wave'
    props['EODf'] = freq0
    props['p-p-amplitude'] = ppampl
    props['flipped'] = flipped
    props['amplitude'] = ampl
    props['rmserror'] = rmserror
    if rmvariance:
        props['rmvariance'] = rmvariance
    props['peakwidth'] = peak_width/period
    props['troughwidth'] = trough_width/period
    props['leftpeak'] = phase1/period
    props['rightpeak'] = phase2/period
    props['lefttrough'] = phase3/period
    props['righttrough'] = phase4/period
    props['p-p-distance'] = distance/period
    if hasattr(freq, 'shape'):
        spec_data = np.zeros((n_harm, 7))
        powers = freq[:n_harm, 1]
        spec_data[:len(powers), 6] = powers
    else:
        spec_data = np.zeros((n_harm, 6))
    spec_data[0,:6] = [0.0, freq0, ampl, 1.0, 0.0, 0.0]
    for i in range(1, n_harm):
        spec_data[i,:6] = [i, (i+1)*freq0, ampl*popt[1+i*2], popt[1+i*2],
                           decibel(popt[1+i*2]**2.0), popt[2+i*2]]
    pnh = power_n_harmonics if power_n_harmonics > 0 else n_harm
    props['power'] = decibel(np.sum(spec_data[:pnh,2]**2.0))
    
    return meod, props, spec_data, error_str


def exp_decay(t, tau, ampl, offs):
    """
    Expontenial decay.

    x(t) = ampl*exp(-t/tau) + offs

    Parameters
    ----------
    t: float or array
        Time.
    tau: float
        Time constant of exponential decay.
    ampl: float
        Amplitude of exponential decay, i.e. initial value minus steady-state value.
    offs: float
        Steady-state value.
    
    Returns
    -------
    x: float or array
        The exponential decay evaluated at times `t`.
    
    """
    return offs + ampl*np.exp(-t/tau)


def analyze_pulse(eod, eod_times, min_pulse_win=0.001,
                  peak_thresh_fac=0.01, min_dist=50.0e-6,
                  width_frac = 0.5, fit_frac = 0.5,
                  fresolution=1.0, flip_pulse='none'):
    """
    Analyze the EOD waveform of a pulse-type fish.
    
    Parameters
    ----------
    eod: 2-D array
        The eod waveform. First column is time in seconds,
        second column the eod waveform.
        Further columns are optional but not used.
    eod_times: 1-D array
        List of times of detected EOD peaks.
    min_pulse_win: float
        The minimum size of cut-out EOD waveform.
    peak_thresh_fac: float
        Set the threshold for peak detection to the maximum pulse amplitude times this factor.
    min_dist: float
        Minimum distance between peak and troughs of the pulse.
    width_frac: float
        The width of a peak is measured at this fraction of a peak's height (0-1).
    fit_frac: float or None
        An exponential is fitted to the tail of the last peak/trough starting where the
        waveform falls below this fraction of the peak's height (0-1).
    fresolution: float
        The frequency resolution of the power spectrum of the single pulse.
    flip_pulse: 'auto', 'none', 'flip'
        - 'auto' flip waveform such that the first large extremum is positive.
        - 'flip' flip waveform.
        - 'none' do not flip waveform.
    
    Returns
    -------
    meod: 2-D array of floats
        The eod waveform. First column is time in seconds,
        second column the eod waveform.
        Further columns are kept from the input `eod`.
        As a last column the fit to the tail of the last peak is appended.
    props: dict
        A dictionary with properties of the analyzed EOD waveform.
        - type: set to 'pulse'.
        - EODf: the inverse of the mean interval between `eod_times`.
        - period: the mean interval between `eod_times`.
        - max-amplitude: the amplitude of the largest positive peak (P1).
        - min-amplitude: the amplitude of the largest negative peak (P2).
        - p-p-amplitude: peak-to-peak amplitude of the EOD waveform.
        - rmvariance: root-mean variance of the averaged EOD waveform relative to
          the p-p amplitude (only if a standard deviation is given in `eod`).
        - tstart: time in seconds where the pulse starts,
          i.e. crosses the threshold for the first time.
        - tend: time in seconds where the pulse ends,
          i.e. crosses the threshold for the last time.
        - width: total width of the pulse in seconds (tend-tstart).
        - tau: time constant of exponential decay of pulse tail in seconds.
        - firstpeak: index of the first peak in the pulse (i.e. -1 for P-1)
        - lastpeak: index of the last peak in the pulse (i.e. 3 for P3)
        - peakfrequency: frequency at peak power of the single pulse spectrum in Hertz.
        - peakpower: peak power of the single pulse spectrum in decibel.
        - lowfreqattenuation5: how much the average power below 5 Hz is attenuated
          relative to the peak power in decibel.
        - lowfreqattenuation50: how much the average power below 5 Hz is attenuated
          relative to the peak power in decibel.
        - powerlowcutoff: frequency at which the power reached half of the peak power
          relative to the initial power in Hertz.
        - flipped: True if the waveform was flipped.
        - n: number of pulses analyzed.
    peaks: 2-D array
        For each peak and trough (rows) of the EOD waveform
        5 columns: the peak index (1 is P1, i.e. the largest positive peak),
        time relative to largest positive peak, amplitude,
        amplitude normalized to largest postive peak,
        and width of peak/trough at half height.
    power: 2-D array
        The power spectrum of a single pulse. First column are the frequencies,
        second column the power.
    """
        
    # storage:
    meod = np.zeros((eod.shape[0], eod.shape[1]+1))
    meod[:,:eod.shape[1]] = eod
    meod[:,-1] = float('nan')
    
    # subtract mean at the ends of the snippet:
    n = len(meod)//20
    meod[:,1] -= 0.5*(np.mean(meod[:n,1]) + np.mean(meod[-n:,1]))

    # largest positive and negative peak:
    flipped = False
    max_idx = np.argmax(meod[:,1])
    max_ampl = np.abs(meod[max_idx,1])
    min_idx = np.argmin(meod[:,1])
    min_ampl = np.abs(meod[min_idx,1])
    amplitude = np.max((max_ampl, min_ampl))
    if max_ampl > 0.2*amplitude and min_ampl > 0.2*amplitude:
        # two major peaks:
        if 'flip' in flip_pulse or ('auto' in flip_pulse and min_idx < max_idx):
            # flip:
            meod[:,1] = -meod[:,1]
            peak_idx = min_idx
            min_idx = max_idx
            max_idx = peak_idx
            flipped = True
    elif 'flip' in flip_pulse or ('auto' in flip_pulse and min_ampl > 0.2*amplitude):
        # flip:
        meod[:,1] = -meod[:,1]
        peak_idx = min_idx
        min_idx = max_idx
        max_idx = peak_idx
        flipped = True
    max_ampl = np.abs(meod[max_idx,1])
    min_ampl = np.abs(meod[min_idx,1])
                
    # move peak of waveform to zero:
    meod[:,0] -= meod[max_idx,0]

    # threshold for peak detection:
    n = len(meod[:,1])//10
    thl_max = np.max(meod[:n,1])
    thl_min = np.min(meod[:n,1])
    thr_max = np.max(meod[-n:,1])
    thr_min = np.min(meod[-n:,1])
    min_thresh = 2.0*(np.max([thl_max, thr_max]) - np.min([thl_min, thr_min]))
    threshold = max_ampl*peak_thresh_fac
    if threshold < min_thresh:
        threshold = min_thresh

    # cut out relevant signal:
    lidx = np.argmax(np.abs(meod[:,1])>0.5*threshold)
    ridx = len(meod) - 1 - np.argmax(np.abs(meod[::-1,1])>0.5*threshold)
    t0 = meod[lidx,0]
    t1 = meod[ridx,0]
    width = t1 - t0
    if width < min_pulse_win:
        width = min_pulse_win
    dt = meod[1,0] - meod[0,0]
    width_idx = int(np.round(width/dt))
    # expand width:
    leidx = lidx - width_idx//2
    if leidx < 0:
        leidx = 0
    reidx = ridx + width_idx//2
    if reidx >= len(meod):
        reidx = len(meod)
    meod = meod[leidx:reidx,:]
    lidx -= leidx
    ridx -= leidx
    max_idx -= leidx
    min_idx -= leidx
    tau = None
    peaks = []

    # amplitude and variance:
    ppampl = max_ampl + min_ampl
    rmvariance = np.sqrt(np.mean(meod[:,2]**2.0))/ppampl if eod.shape[1] > 2 else None
    
    # find smaller peaks:
    peak_idx, trough_idx = detect_peaks(meod[:,1], threshold)
    if len(peak_idx) > 0:
        # and their width:
        peak_widths = peak_width(meod[:,0], meod[:,1], peak_idx, trough_idx,
                                 peak_frac=width_frac, base='max')
        trough_widths = peak_width(meod[:,0], -meod[:,1], trough_idx, peak_idx,
                                   peak_frac=width_frac, base='max')
        # combine peaks and troughs:
        pt_idx = np.concatenate((peak_idx, trough_idx))
        pt_widths = np.concatenate((peak_widths, trough_widths))
        pts_idx = np.argsort(pt_idx)
        peak_list = pt_idx[pts_idx]
        width_list = pt_widths[pts_idx]
        # remove multiple peaks that are too close: XXX replace by Dexters function that keeps the maximum peak
        rmidx = [(k, k+1) for k in np.where(np.diff(meod[peak_list,0]) < min_dist)[0]]
        # flatten and keep maximum peak:
        rmidx = np.unique([k for kk in rmidx for k in kk if peak_list[k] != max_idx])
        # delete:
        peak_list = np.delete(peak_list, rmidx)
        width_list = np.delete(width_list, rmidx)
        # find P1:
        p1i = np.argmax(peak_list == max_idx)
        offs = 0 if p1i <= 2 else p1i - 2
        peak_list = peak_list[offs:]
        width_list = width_list[offs:]
        # store peaks:
        peaks = np.zeros((len(peak_list), 5))
        for i, pi in enumerate(peak_list):
            peaks[i,:] = [i+1-p1i+offs, meod[pi,0], meod[pi,1], meod[pi,1]/max_ampl, width_list[i]]

        # fit exponential to last peak/trough:
        if fit_frac:
            pi = peak_list[-1]
            if ridx >= len(meod)-1:
                ridx = len(meod)-1
            sign = 1.0 if meod[pi,1] > meod[ridx,1] else -1.0
            thresh = meod[ridx,1]*(1.0-fit_frac) + meod[pi,1]*fit_frac
            inx = pi + np.argmax(sign*meod[pi:ridx,1] < sign*thresh)
            thresh = meod[ridx,1]*(1.0-np.exp(-1.0)) + meod[inx,1]*np.exp(-1.0)
            tau_inx = np.argmax(sign*meod[inx:ridx,1] < sign*thresh)
            if tau_inx < 2:
                tau_inx = 2
            tau = meod[inx+tau_inx,0]-meod[inx,0]
            rridx = len(meod)-1 if inx + 6*tau_inx >= len(meod) else inx + 6*tau_inx
            params = [tau, meod[inx,1]-meod[rridx,1], meod[rridx,1]]
            popt, pcov = curve_fit(exp_decay, meod[inx:rridx,0]-meod[inx,0], meod[inx:rridx,1], params)
            if popt[0] > 1.2*tau:
                tau_inx = int(np.round(popt[0]/dt))
                rridx = len(meod)-1 if inx + 6*tau_inx >= len(meod) else inx + 6*tau_inx
                popt, pcov = curve_fit(exp_decay, meod[inx:rridx,0]-meod[inx,0], meod[inx:rridx,1], popt)
            tau = popt[0]
            meod[inx:rridx,-1] = exp_decay(meod[inx:rridx,0]-meod[inx,0], *popt)

    # power spectrum of single pulse:
    samplerate = 1.0/(meod[1,0]-meod[0,0])
    nfft, _ = nfft_noverlap(fresolution, samplerate)
    n = len(meod)//4
    nn = np.max([nfft, 2*n])
    data = np.zeros(nn)
    data[nn//2-n:nn//2+n] = meod[max_idx-n:max_idx+n,1]
    power, freqs = psd(data, samplerate, fresolution)
    ppower = np.zeros((len(freqs), 2))
    ppower[:,0] = freqs
    ppower[:,1] = power
    maxpower = np.max(power)
    att5 = decibel(np.mean(power[freqs<5.0])/maxpower)
    att50 = decibel(np.mean(power[freqs<50.0])/maxpower)
    lowcutoff = freqs[decibel(power/maxpower) > 0.5*att5][0]

    # analyze pulse timing:
    inter_pulse_intervals = np.diff(eod_times)
    period = np.mean(inter_pulse_intervals)
    
    # store properties:
    props = {}
    props['type'] = 'pulse'
    props['EODf'] = 1.0/period
    props['period'] = period
    props['max-amplitude'] = max_ampl
    props['min-amplitude'] = min_ampl
    props['p-p-amplitude'] = ppampl
    if rmvariance:
        props['rmvariance'] = rmvariance
    props['tstart'] = t0
    props['tend'] = t1
    props['width'] = t1-t0
    if tau:
        props['tau'] = tau
    props['firstpeak'] = peaks[0, 0] if len(peaks) > 0 else 1
    props['lastpeak'] = peaks[-1, 0] if len(peaks) > 0 else 1
    props['peakfrequency'] = freqs[np.argmax(power)]
    props['peakpower'] = decibel(maxpower)
    props['lowfreqattenuation5'] = att5
    props['lowfreqattenuation50'] = att50
    props['powerlowcutoff'] = lowcutoff
    props['flipped'] = flipped
    props['n'] = len(eod_times)
    
    return meod, props, peaks, ppower


def eod_recording_plot(data, samplerate, ax, width=0.1, unit=None, toffs=0.0,
                       kwargs={'lw': 2, 'color': 'red'}):
    """
    Plot a zoomed in range of the recorded trace.

    Parameters
    ----------
    data: 1D ndarray
        Recorded data.
    samplerate: float
        Sampling rate of the data in Hertz.
    ax:
        Axis for plot.
    width: float
        Width of data segment to be plotted in seconds.
    unit: string
        Optional unit of the data used for y-label.
    toffs: float
        Time of first data value in seconds.
    kwargs: dict
        Arguments passed on to the plot command for the recorded trace.
    """
    widx2 = int(width*samplerate)/2
    i0 = len(data)//2 - widx2
    i0 = (i0//widx2)*widx2
    i1 = i0 + 2*widx2
    if i0 < 0:
        i0 = 0
    if i1 >= len(data):
        i1 = len(data)
    time = np.arange(len(data))/samplerate + toffs
    tunit = 'sec'
    if np.abs(time[i0]) < 1.0 and np.abs(time[i1]) < 1.0:
        time *= 1000.0
        tunit = 'ms'
    ax.plot(time, data, **kwargs)
    ax.set_xlim(time[i0], time[i1])
    ax.set_xlabel('Time [%s]' % tunit)
    ymin = np.min(data[i0:i1])
    ymax = np.max(data[i0:i1])
    dy = ymax - ymin
    ax.set_ylim(ymin-0.05*dy, ymax+0.05*dy)
    if len(unit) == 0 or unit == 'a.u.':
        ax.set_ylabel('Amplitude')
    else:
        ax.set_ylabel('Amplitude [%s]' % unit)


def eod_waveform_plot(eod_waveform, peaks, ax, unit=None, tau=None,
                      mkwargs={'lw': 2, 'color': 'red'},
                      skwargs={'color': '#CCCCCC'},
                      fkwargs={'lw': 6, 'color': 'steelblue'},
                      zkwargs={'lw': 1, 'color': '#AAAAAA'}):
    """
    Plot mean EOD, its standard deviation, and an optional fit to the EOD.

    Parameters
    ----------
    eod_waveform: 2-D array
        EOD waveform. First column is time in seconds,
        second column the (mean) eod waveform. The optional third column is the
        standard deviation, and the optional fourth column is a fit on the waveform.
    peaks: 2_D arrays or None
        List of peak properties (index, time, and amplitude) of a EOD pulse
        as returned by `analyze_pulse()`.
    ax:
        Axis for plot.
    unit: string
        Optional unit of the data used for y-label.
    tau: float
        Optional time constant of a fit.
    mkwargs: dict
        Arguments passed on to the plot command for the mean EOD.
    skwargs: dict
        Arguments passed on to the fill_between command for the standard deviation of the EOD.
    fkwargs: dict
        Arguments passed on to the plot command for the fitted EOD.
    zkwargs: dict
        Arguments passed on to the plot command for the zero line.
    """
    ax.autoscale(True)
    time = 1000.0 * eod_waveform[:,0]
    # plot zero line:
    ax.plot([time[0], time[-1]], [0.0, 0.0], zorder=2, **zkwargs)
    # plot fit:
    if eod_waveform.shape[1] > 3:
        ax.plot(time, eod_waveform[:,3], zorder=3, **fkwargs)
    # plot waveform:
    mean_eod = eod_waveform[:,1]
    ax.plot(time, mean_eod, zorder=5, **mkwargs)
    # plot standard deviation:
    if eod_waveform.shape[1] > 2:
        ax.autoscale(False)
        std_eod = eod_waveform[:,2]
        ax.fill_between(time, mean_eod + std_eod, mean_eod - std_eod,
                        zorder=1, **skwargs)
    # annotate fit:
    if not tau is None and eod_waveform.shape[1] > 3:
        if tau < 0.001:
            label = u'\u03c4=%.0f\u00b5s' % (1.e6*tau)
        else:
            label = u'\u03c4=%.2fms' % (1.e3*tau)
        inx = np.argmin(np.isnan(eod_waveform[:,3]))
        x = eod_waveform[inx,0] + tau
        y = 0.7*eod_waveform[inx,3]
        maxa = np.max(np.abs(mean_eod))
        if np.abs(y) < 0.07*maxa:
            y = -0.07*maxa*np.sign(y)
        va = 'bottom' if y > 0.0 else 'top'
        ax.text(1000.0*x, y, label, ha='left', va=va, zorder=10)
    # annotate peaks:
    if peaks is not None and len(peaks)>0:
        maxa = np.max(peaks[:,2])
        for p in peaks:
            ax.scatter(1000.0*p[1], p[2], s=80, clip_on=False, zorder=4, alpha=0.4,
                       c=mkwargs['color'], edgecolors=mkwargs['color'])
            label = u'P%d' % p[0]
            if p[0] != 1:
                if p[1] < 0.001:
                    label += u'(%.0f%% @ %.0f\u00b5s)' % (100.0*p[3], 1.0e6*p[1])
                else:
                    label += u'(%.0f%% @ %.2gms)' % (100.0*p[3], 1.0e3*p[1])
            va = 'bottom'
            y = 0.02*maxa
            if p[0] % 2 == 0:
                va = 'top'
                y = -y
            if p[0] == 1 or p[0] == 2:
                va = 'bottom'
                y = 0.0
            dx = 0.05*time[-1]
            if p[1] >= 0.0:
                ax.text(1000.0*p[1]+dx, p[2]+y, label, ha='left', va=va,
                        zorder=10)
            else:
                ax.text(1000.0*p[1]-dx, p[2]+y, label, ha='right', va=va,
                        zorder=10)
    ax.set_xlim(time[0], time[-1])
    ax.set_xlabel('Time [msec]')
    if unit:
        ax.set_ylabel('Amplitude [%s]' % unit)
    else:
        ax.set_ylabel('Amplitude')


def wave_spectrum_plot(spec, props, axa, axp, unit=None, color='b', lw=2, markersize=10):
    """Plot and annotate spectrum of wave-type EOD.

    Parameters
    ----------
    spec: 2-D array
        The amplitude spectrum of a single pulse as returned by `analyze_wave()`.
        First column is the index of the harmonics, second column its frequency,
        third column its amplitude, fourth column its amplitude relative to the fundamental,
        fifth column is power of harmonics relative to fundamental in decibel,
        and sixth column the phase shift relative to the fundamental.
    props: dict
        A dictionary with properties of the analyzed EOD waveform as
        returned by `analyze_wave()`.
    axa:
        Axis for amplitude plot.
    axa:
        Axis for phase plot.
    unit: string
        Optional unit of the data used for y-label.
    color:
        Color for line and points of spectrum.
    lw: float
        Linewidth for spectrum.
    markersize: float
        Size of points on spectrum.
    """
    n = 9
    # amplitudes:
    markers, stemlines, baseline = axa.stem(spec[:n,0], spec[:n,2])
    plt.setp(markers, color=color, markersize=markersize, clip_on=False)
    plt.setp(stemlines, color=color, lw=lw)
    axa.set_xlim(-1.0, n-0.5)
    axa.set_xticks(np.arange(0, n, 1))
    axa.tick_params('x', direction='out')
    if unit:
        axa.set_ylabel('Amplitude [%s]' % unit)
    else:
        axa.set_ylabel('Amplitude')
    # phases:
    phases = spec[:,5]
    phases[phases<0.0] = phases[phases<0.0] + 2.0*np.pi
    markers, stemlines, baseline = axp.stem(spec[:n,0], phases[:n])
    plt.setp(markers, color=color, markersize=markersize, clip_on=False)
    plt.setp(stemlines, color=color, lw=lw)
    axp.set_xlim(-1.0, n-0.5)
    axp.set_xticks(np.arange(0, n, 1))
    axp.tick_params('x', direction='out')
    axp.set_ylim(0, 2.0*np.pi)
    axp.set_yticks([0, np.pi, 2.0*np.pi])
    axp.set_yticklabels([u'0', u'\u03c0', u'2\u03c0'])
    axp.set_xlabel('Harmonics')
    axp.set_ylabel('Phase')


def pulse_spectrum_plot(power, props, ax, color='b', lw=3, markersize=80):
    """Plot and annotate spectrum of single pulse-type EOD.

    Parameters
    ----------
    power: 2-D array
        The power spectrum of a single pulse as returned by `analyze_pulse()`.
        First column are the frequencies, second column the power.
    props: dict
        A dictionary with properties of the analyzed EOD waveform as
        returned by `analyze_pulse()`.
    ax:
        Axis for plot.
    color:
        Color for line and points of spectrum.
    lw: float
        Linewidth for spectrum.
    markersize: float
        Size of points on spectrum.
    """
    box = mpatches.Rectangle((1,-60), 49, 60, linewidth=0, facecolor='#DDDDDD',
                             zorder=1)
    ax.add_patch(box)
    att = props['lowfreqattenuation50']
    ax.text(10.0, att+1.0, '%.0f dB' % att, ha='left', va='bottom', zorder=10)
    box = mpatches.Rectangle((1,-60), 4, 60, linewidth=0, facecolor='#CCCCCC',
                             zorder=2)
    ax.add_patch(box)
    att = props['lowfreqattenuation5']
    ax.text(4.0, att+1.0, '%.0f dB' % att, ha='right', va='bottom', zorder=10)
    lowcutoff = props['powerlowcutoff']
    ax.plot([lowcutoff, lowcutoff, 1.0], [-60.0, 0.5*att, 0.5*att], '#BBBBBB',
            zorder=3)
    ax.text(1.2*lowcutoff, 0.5*att-1.0, '%.0f Hz' % lowcutoff, ha='left', va='top', zorder=10)
    db = decibel(power[:,1])
    smax = np.nanmax(db)
    ax.plot(power[:,0], db - smax, color, lw=lw, zorder=4)
    peakfreq = props['peakfrequency']
    ax.scatter([peakfreq], [0.0], c=color, edgecolors=color, s=markersize, alpha=0.4, zorder=5)
    ax.text(peakfreq*1.2, 1.0, '%.0f Hz' % peakfreq, va='bottom', zorder=10)
    ax.set_xlim(1.0, 10000.0)
    ax.set_xscale('log')
    ax.set_ylim(-60.0, 2.0)
    ax.set_xlabel('Frequency [Hz]')
    ax.set_ylabel('Power [dB]')


def add_eod_analysis_config(cfg, thresh_fac=0.8, percentile=1.0,
                            win_fac=2.0, min_win=0.01, max_eods=None,
                            flip_wave='none', flip_pulse='none',
                            n_harm=20, min_pulse_win=0.001, peak_thresh_fac=0.01,
                            min_dist=50.0e-6, width_frac = 0.5, fit_frac = 0.5,
                            pulse_percentile=1.0):
    """ Add all parameters needed for the eod analysis functions as
    a new section to a configuration.

    Parameters
    ----------
    cfg: ConfigFile
        The configuration.
        
    See eod_waveform(), analyze_wave(), and analyze_pulse() for details on
    the remaining arguments.
    """
    cfg.add_section('EOD analysis:')
    if not 'pulseWidthPercentile' in cfg:
        cfg.add('pulseWidthPercentile', percentile, '%', 'The variance of the data is measured as the interpercentile range.')
    if not 'pulseWidthThresholdFactor' in cfg:
        cfg.add('pulseWidthThresholdFactor', thresh_fac, '', 'The threshold for detection of EOD peaks is this factor multiplied with the interpercentile range of the data.')
    cfg.add('eodSnippetFac', win_fac, '', 'The duration of EOD snippets is the EOD period times this factor.')
    cfg.add('eodMinSnippet', min_win, 's', 'Minimum duration of cut out EOD snippets.')
    cfg.add('eodMaxEODs', max_eods or 0, '', 'The maximum number of EODs used to compute the average EOD. If 0 use all EODs.')
    cfg.add('flipWaveEOD', flip_wave, '', 'Flip EOD of wave-type fish to make largest extremum positive.')
    cfg.add('flipPulseEOD', flip_pulse, '', 'Flip EOD of pulse-type fish to make the first large peak positive.')
    cfg.add('eodHarmonics', n_harm, '', 'Number of harmonics fitted to the EOD waveform.')
    cfg.add('eodMinPulseSnippet', min_pulse_win, 's', 'Minimum duration of cut out EOD snippets for a pulse fish.')
    cfg.add('eodPeakThresholdFactor', peak_thresh_fac, '', 'Threshold for detection of peaks in pulse-type EODs as a fraction of the pulse amplitude.')
    cfg.add('eodMinimumDistance', min_dist, 's', 'Minimum distance between peaks and troughs in a EOD pulse.')
    cfg.add('eodPulseWidthFraction', width_frac, '', 'The width of a pulse is measured at this fraction of the pulse height.')
    cfg.add('eodExponentialFitFraction', fit_frac, '', 'An exponential function is fitted on the tail of a pulse starting at this fraction of the height of the last peak.')


def eod_waveform_args(cfg):
    """ Translates a configuration to the
    respective parameter names of the function eod_waveform().
    
    The return value can then be passed as key-word arguments to this function.

    Parameters
    ----------
    cfg: ConfigFile
        The configuration.

    Returns
    -------
    a: dict
        Dictionary with names of arguments of the eod_waveform() function
        and their values as supplied by `cfg`.
    """
    a = cfg.map({'thresh_fac': 'pulseWidthThresholdFactor',
                 'percentile': 'pulseWidthPercentile',
                 'win_fac': 'eodSnippetFac',
                 'min_win': 'eodMinSnippet',
                 'max_eods': 'eodMaxEODs'})
    return a


def analyze_wave_args(cfg):
    """ Translates a configuration to the
    respective parameter names of the function analyze_wave().
    
    The return value can then be passed as key-word arguments to this function.

    Parameters
    ----------
    cfg: ConfigFile
        The configuration.

    Returns
    -------
    a: dict
        Dictionary with names of arguments of the analyze_wave() function
        and their values as supplied by `cfg`.
    """
    a = cfg.map({'n_harm': 'eodHarmonics',
                 'power_n_harmonics': 'powerNHarmonics',
                 'flip_wave': 'flipWaveEOD'})
    return a


def analyze_pulse_args(cfg):
    """ Translates a configuration to the
    respective parameter names of the function analyze_pulse().
    
    The return value can then be passed as key-word arguments to this function.

    Parameters
    ----------
    cfg: ConfigFile
        The configuration.

    Returns
    -------
    a: dict
        Dictionary with names of arguments of the analyze_pulse() function
        and their values as supplied by `cfg`.
    """
    a = cfg.map({'min_pulse_win': 'eodMinPulseSnippet',
                 'peak_thresh_fac': 'eodPeakThresholdFactor',
                 'min_dist': 'eodMinimumDistance',
                 'width_frac': 'eodPulseWidthFraction',
                 'fit_frac': 'eodExponentialFitFraction',
                 'flip_pulse': 'flipPulseEOD'})
    return a


def add_eod_quality_config(cfg, max_clipped_frac=0.01, max_relampl_harm1=2.0,
                           max_relampl_harm2=0.8, max_relampl_harm3=0.5,
                           max_rms_error=0.05):
    """Add parameters needed for assesing the quality of an EOD waveform.

    Parameters
    ----------
    cfg: ConfigFile
        The configuration.
        
    See check_wave_quality( and check_pulse_quality() for details on
    the remaining arguments.
    """
    cfg.add_section('Waveform selection:')
    cfg.add('maximumClippedFraction', max_clipped_frac, '', 'Take waveform of the fish with the highest power only if the fraction of clipped signals is below this value.')
    cfg.add('maximumFirstHarmonicAmplitude', max_relampl_harm1, '', 'Skip waveform of wave-type fish if the amplitude of the first harmonic is higher than this factor times the amplitude of the fundamental.')
    cfg.add('maximumSecondHarmonicAmplitude', max_relampl_harm2, '', 'Skip waveform of wave-type fish if the ampltude of the second harmonic is higher than this factor times the amplitude of the fundamental. That is, the waveform appears to have twice the frequency than the fundamental.')
    cfg.add('maximumThirdHarmonicAmplitude', max_relampl_harm3, '', 'Skip waveform of wave-type fish if the ampltude of the third harmonic is higher than this factor times the amplitude of the fundamental.')
    cfg.add('maximumRMSError', max_rms_error, '', 'Skip waveform of wave-type fish if the root-mean-squared error relative to the peak-to-peak amplitude is larger than this number.')


if __name__ == '__main__':
    import sys
    import matplotlib.pyplot as plt
    from .fakefish import generate_biphasic_pulses
    from .dataloader import load_data
    from .bestwindow import best_window

    print('Analysis of EOD waveforms.')
    print('')
    print('Usage:')
    print('  python eodanalysis.py [<audiofile>]')
    print('')

    # data:
    if len(sys.argv) <= 1:
        samplerate = 44100.0
        data = generate_biphasic_pulses(200.0, samplerate, 5.0, noise_std=0.02)
        unit = 'mV'
    else:
        rawdata, samplerate, unit = load_data(sys.argv[1], 0)
        data, _ = best_window(rawdata, samplerate)

    # analyse EOD:
    mean_eod, eod_times = eod_waveform(data, samplerate)
    mean_eod, props, peaks, power, intervals = analyze_pulse(mean_eod, eod_times)

    # plot:
    fig = plt.figure()
    ax = fig.add_subplot(1, 2, 1)
    eod_waveform_plot(mean_eod, peaks, ax, unit=unit)
    props['unit'] = unit
    label = '{type}-type fish\nEODf = {EODf:.1f} Hz\np-p amplitude = {p-p-amplitude:.3g} {unit}\nn = {n} EODs\n'.format(**props)
    if props['flipped']:
        label += 'flipped\n'
    ax.text(0.03, 0.97, label, transform = ax.transAxes, va='top')
    ax = fig.add_subplot(1, 2, 2)
    pulse_spectrum_plot(power, props, ax)
    plt.show()
