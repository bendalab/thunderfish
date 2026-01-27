"""
Analysis of pulse-type EOD waveforms.

## Analysis of pulse-type EODs

- `condition_pulse()`: subtract offset, flip, shift, and cut out pulse EOD waveform.
- `analyze_pulse_properties()`: characterize basic properties of a pulse-type EOD.
- `analyze_pulse_phases()`: characterize all phases of a pulse-type EOD.
- `decompose_pulse()`: decompose single pulse waveform into sum of Gaussians.
- `analyze_pulse_tail()`: fit exponential to last peak/trough of pulse EOD.
- `pulse_spectrum()`: compute the spectrum of a single pulse-type EOD.
- `analyze_pulse_spectrum()`: analyze the spectrum of a pulse-type EOD.
- `analyze_pulse_intervals()`: basic statistics of interpulse intervals.
- `analyze_pulse()`: analyze the EOD waveform of a pulse fish.

## Storage

- `save_pulse_fish()`: save properties of pulse EODs to file.
- `load_pulse_fish()`: load properties of pulse EODs from file.
- `save_pulse_spectrum()`: save spectrum of pulse EOD to file.
- `load_pulse_spectrum()`: load spectrum of pulse EOD from file.
- `save_pulse_phases()`: save phase properties of pulse EOD to file.
- `load_pulse_phases()`: load phase properties of pulse EOD from file.
- `save_pulse_gaussians()`: save Gaussian phase properties of pulse EOD to file.
- `load_pulse_gaussians()`: load Gaussian phase properties of pulse EOD from file.
- `save_pulse_times()`: save times of pulse EOD to file.
- `load_pulse_times()`: load times of pulse EOD from file.

## Fit functions

- `gaussian_sum()`: sum of Gaussians.
- `gaussian_sum_spectrum()`: energy spectrum of sum of Gaussians.
- `gaussian_sum_costs()`: cost function for fitting sum of Gaussians.
"""

import numpy as np
try:
    import matplotlib.pyplot as plt
except ImportError:
    pass

from pathlib import Path
from scipy.optimize import curve_fit, minimize
from thunderlab.eventdetection import detect_peaks, peak_width
from thunderlab.powerspectrum import next_power_of_two, nfft, decibel
from thunderlab.tabledata import TableData

from .fakefish import pulsefish_waveform


def condition_pulse(eod, ratetime=None, sem=None, flip='none',
                    baseline_frac=0.05, large_phase_frac=0.2,
                    min_pulse_win=0.001):
    """Subtract offset, flip, shift, and cut out pulse EOD waveform.
    
    Parameters
    ----------
    eod: 1-D or 2-D array of float
        The eod waveform of which the spectrum is computed.
        If an 1-D array, then this is the waveform and you
        need to also pass a sampling rate in `rate`.
        If a 2-D array, then first column is time in seconds and second
        column is the eod waveform. If a third column is present,
        it is interpreted as the standard error of the mean
        corresponding to the averaged waveform.
        Further columns are ignored.
    ratetime: None or float or array of float
        If a 1-D array is passed on to `eod` then either the sampling
        rate in Hertz or the time array corresponding to `eod`.
    sem: None or float or array of float
        If a 1-D array is passed on to `eod`, then the optional
        standard error of the averaged waveform. Either as a single value
        or as an 1-D array for the whole waveform.
    flip: 'auto', 'none', 'flip'
        - 'auto' flip waveform such that the first large extremum is positive.
        - 'flip' flip waveform.
        - 'none' do not flip waveform.
    baseline_frac: float
        Fraction of data points from which the amplitude offset is computed.
    large_phase_frac: float
        Minimum amplitude of a large phase as a fraction of the largest one.
    min_pulse_win: float
        The minimum size of the cut-out EOD waveform.
    
    Returns
    -------
    eod: 1-D or 2-D array of float
        The input `eod` in the format of the input `eod`
        with the conditioned waveform and optional time.
    time: 1-D array of float
        In case `eod` is a 1-D array, then the time array is also returned.
    toffs: float
        Time that was subtracted from the time axis,
        such that the maximum of the EOD waveform was shifted to zero.
    aoffs: float
        Offset that was subtracted from the EOD waveform.
        This is the average over the first `baseline_frac` data points
        of the EOD waveform.
    flipped: bool
        True if waveform was flipped.
    noise_thresh: float
        Minimum threshold that is just higher than the noise level
        within the first `baseline_frac` data points of the EOD waveform.
        
    """
    if eod.ndim == 2:
        time = eod[:, 0]
        meod = eod[:, 1]
        if eod.shape[1] > 2:
            sem = eod[:, 2]
    else:
        meod = eod
        if isinstance(ratetime, (list, tuple, np.ndarray)):
            time = ratetime
        else:
            time = np.arange(len(meod))/rate
        if np.isscalar(sem):
            sem = np.ones(len(meod))*sem
            
    # subtract mean computed from the left end:
    n_base = int(baseline_frac*len(meod))
    if n_base < 5:
        n_base = 5
    aoffs = np.mean(meod[:n_base])  # baseline
    meod -= aoffs
    
    # flip waveform:
    max_idx = np.argmax(meod)
    min_idx = np.argmin(meod)
    flipped = 'flip' in flip.lower()
    if 'auto' in flip.lower():
        max_ampl = abs(meod[max_idx])
        min_ampl = abs(meod[min_idx])
        amplitude = max(max_ampl, min_ampl)
        if max_ampl > large_phase_frac*amplitude and \
           min_ampl > large_phase_frac*amplitude:
            # two major peaks:
            if min_idx < max_idx:
                flipped = True
        elif min_ampl > large_phase_frac*amplitude:
            flipped = True
    if flipped:
        meod = -meod
        idx = min_idx
        min_idx = max_idx
        max_idx = idx
    max_ampl = abs(meod[max_idx])
    min_ampl = abs(meod[min_idx])
    
    # shift maximum to zero:
    toffs = time[max_idx]
    time -= toffs
    
    # threshold from baseline maximum and minimum:
    th_max = np.max(meod[:n_base])
    th_min = np.min(meod[:n_base])
    range_thresh = 2*(th_max - th_min)
    # threshold from standard error:
    if sem is not None:
        msem = np.mean(sem[:n_base])
        sem_thresh = 8*msem
    # noise threshold:
    noise_thresh = max(range_thresh, sem_thresh)

    # generous left edge of waveform:
    l1_idx = np.argmax(np.abs(meod) > noise_thresh)
    l2_idx = np.argmax(np.abs(meod) > 2*noise_thresh)
    w = 2*(l2_idx - l1_idx)
    if w < n_base:
        w = n_base
    l_idx = l1_idx - w
    if l_idx < 0:
        l_idx = 0
    # generous right edge of waveform:
    r1_idx = len(meod) - 1 - np.argmax(np.abs(meod[::-1]) > noise_thresh)
    r2_idx = len(meod) - 1 - np.argmax(np.abs(meod[::-1]) > 2*noise_thresh)
    w = 2*(r1_idx - r2_idx)
    if w < n_base:
        w = n_base
    r_idx = r1_idx + w
    if r_idx >= len(meod):
        r_idx = len(meod)
    # cut out relevant signal:
    if time[r_idx - 1] - time[l_idx] < min_pulse_win:
        ct = time[(l_idx + r_idx)//2]
        mask = (time >= ct - min_pulse_win/2) & (time <= ct + min_pulse_win/2)
        meod = meod[mask]
        time = time[mask]
        if eod.ndim == 2:
            eod = eod[mask, :]
    else:
        meod = meod[l_idx:r_idx]
        time = time[l_idx:r_idx]
        if eod.ndim == 2:
            eod = eod[l_idx:r_idx, :]
    
    # return offset, flipped and, shifted waveform:
    if eod.ndim == 2:
        eod[:, 0] = time
        eod[:, 1] = meod
        return eod, toffs, aoffs, flipped, noise_thresh
    else:
        return meod, time, toffs, aoffs, flipped, noise_thresh


def analyze_pulse_properties(noise_thresh, eod, ratetime=None):
    """Characterize basic properties of a pulse-type EOD.
    
    Parameters
    ----------
    noise_thresh: float
        Minimum threshold that is just higher than the baseline noise level.
        As returned by `condition_pulse()`.
    eod: 1-D or 2-D array
        The eod waveform to be analyzed.
        If an 1-D array, then this is the waveform and you
        need to also pass a sampling rate in `rate`.
        If a 2-D array, then first column is time in seconds and second
        column is the eod waveform. Further columns are ignored.
    ratetime: None or float or array of float
        If a 1-D array is passed on to `eod` then either the sampling
        rate in Hertz or the time array corresponding to `eod`.

    Returns
    -------
    pos_ampl: float
        Amplitude of largest positive peak.
    neg_ampl: float
        Amplitude of largest negative trough (absolute value).
    dist: float
        Temporal distance between largest negative trough and positive peak.
    pos_area: float
        Integral under all positive values of EOD waveform.
    neg_area: float
        Integral under all negative values of EOD waveform (absolute value).
    polarity_balance: float
        Contrast between positive and negative areas of EOD waveform, i.e.
        (pos_area - neg_area)/(pos_area + neg_area).
    center: float
        Center of mass (first moment when treating the absolute value of
        the waveform as a distribution).
    stdev: float
        Standard deviation of mass (square root of second central moment
        when treating the absolute value of the waveform as a distribution).
    median: float
        Median of the distribution of the absolute EOD waveform.
    quartile1: float
        First quartile of the distribution of the absolute EOD waveform.
    quartile3: float
        Third quartile of the distribution of the absolute EOD waveform.
    """
    if eod.ndim == 2:
        time = eod[:, 0]
        eod = eod[:, 1]
    elif isinstance(ratetime, (list, tuple, np.ndarray)):
        time = ratetime
    else:
        time = np.arange(len(eod))/rate
    dt = time[1] - time[0]

    # amplitudes:
    pos_idx = np.argmax(eod)
    pos_ampl = abs(eod[pos_idx])
    if pos_ampl < noise_thresh:
        pos_ampl = 0
    neg_idx = np.argmin(eod)
    neg_ampl = abs(eod[neg_idx])
    if neg_ampl < noise_thresh:
        neg_ampl = 0
    dist = time[neg_idx] - time[pos_idx]
    if pos_ampl < noise_thresh or neg_ampl < noise_thresh:
        dist = np.inf

    # integrals (areas) and polarity balance:
    pos_area = abs(np.sum(eod[eod >= 0]))*dt
    neg_area = abs(np.sum(eod[eod < 0]))*dt
    total_area = np.sum(np.abs(eod))*dt
    integral_area = np.sum(eod)*dt   # = pos_area - neg_area
    polarity_balance = integral_area/total_area

    # moments (EOD waveforms are not Gaussian!):
    #center = np.sum(time*np.abs(eod))/np.sum(np.abs(eod))
    #var = np.sum((time - center)**2*np.abs(eod))/np.sum(np.abs(eod))
    #stdev = np.sqrt(var)

    # cumulative:
    cumul = np.cumsum(np.abs(eod))/np.sum(np.abs(eod))
    median = time[np.argmax(cumul > 0.5)]
    quartile1 = time[np.argmax(cumul > 0.25)]
    quartile3 = time[np.argmax(cumul > 0.75)]
    
    return pos_ampl, neg_ampl, dist, \
        pos_area, neg_area, polarity_balance, \
        median, quartile1, quartile3

    
def analyze_pulse_phases(peak_thresh, startend_thresh,
                         eod, ratetime=None,
                         min_dist=50.0e-6, width_frac=0.5):
    """Characterize all phases of a pulse-type EOD.
    
    Parameters
    ----------
    peak_thresh: float
        Threshold for detecting peaks and troughs.
    startend_thresh: float or None
        Threshold for detecting start and end time of EOD.
        If None, use `peak_thresh`.
    eod: 1-D or 2-D array
        The eod waveform to be analyzed.
        If an 1-D array, then this is the waveform and you
        need to also pass a sampling rate in `rate`.
        If a 2-D array, then first column is time in seconds and second
        column is the eod waveform. Further columns are ignored.
    ratetime: None or float or array of float
        If a 1-D array is passed on to `eod` then either the sampling
        rate in Hertz or the time array corresponding to `eod`.
    min_dist: float
        Minimum distance between peak and troughs of the pulse.
    width_frac: float
        The width of an EOD phase is measured at this fraction of a peak's
        height (0-1).
    
    Returns
    -------
    tstart: float
        Start time of EOD waveform, i.e. the first time it crosses `threshold`.
    tend: float
        End time of EOD waveform, i.e. the last time it falls under `threshold`.
    phases: dict
        Dictionary with
    
        - "indices": indices of each phase
          (1 is P1, i.e. the largest positive peak)
        - "times": times of each phase relative to P1 in seconds
        - "amplitudes": amplitudes of each phase
        - "relamplitudes": amplitudes normalized to amplitude of P1 phase
        - "widths": widths of each phase at `width_frac` height
        - "areas": areas of each phase
        - "relareas": areas of the phases relative to total area
        - "zeros": time of zero crossing towards next phase in seconds
    
        Empty dictionary if waveform is not a pulse EOD.

    """
    if eod.ndim == 2:
        time = eod[:, 0]
        eod = eod[:, 1]
    elif isinstance(ratetime, (list, tuple, np.ndarray)):
        time = ratetime
    else:
        time = np.arange(len(eod))/rate
    dt = time[1] - time[0]
    
    # start and end time:
    if startend_thresh is None:
        startend_thresh = peak_thresh
    l_idx = np.argmax(np.abs(eod) > startend_thresh)
    r_idx = len(eod) - 1 - np.argmax(np.abs(eod[::-1]) > startend_thresh)
    tstart = time[l_idx]
    tend = time[r_idx]
    
    # find peaks and troughs:
    peak_idx, trough_idx = detect_peaks(eod, peak_thresh)
    if len(peak_idx) == 0:
        return tstart, tend, {}
    # and their width:
    peak_widths = peak_width(time, eod, peak_idx, trough_idx,
                             peak_frac=width_frac, base='max')
    trough_widths = peak_width(time, -eod, trough_idx, peak_idx,
                               peak_frac=width_frac, base='max')
    # combine peaks and troughs:
    pt_idx = np.concatenate((peak_idx, trough_idx))
    pt_widths = np.concatenate((peak_widths, trough_widths))
    pts_idx = np.argsort(pt_idx)
    phase_list = pt_idx[pts_idx]
    width_list = pt_widths[pts_idx]
    # remove phases at the start and end of the signal:
    n = len(eod)//20
    if n < 5:
        n = 5
    mask = (phase_list > n) & (phase_list < len(eod) - 20)
    phase_list = phase_list[mask]
    width_list = width_list[mask]
    if len(phase_list) == 0:
        return tstart, tend, {}
    # remove multiple peaks that are too close:
    # TODO: XXX replace by Dexters function that keeps the maximum peak
    rmidx = [(k, k+1) for k in np.where(np.diff(time[phase_list]) < min_dist)[0]]
    # flatten and keep maximum and minimum phase:
    max_idx = np.argmax(eod)
    min_idx = np.argmin(eod)
    rmidx = np.unique([k for kk in rmidx for k in kk
                       if phase_list[k] != max_idx and phase_list[k] != min_idx])
    # delete:
    if len(rmidx) > 0:
        phase_list = np.delete(phase_list, rmidx)
        width_list = np.delete(width_list, rmidx)
    if len(phase_list) == 0:
        return tstart, tend, {}
    # find P1:
    p1i = np.argmax(phase_list == max_idx)
    # amplitudes:
    amplitudes = eod[phase_list]
    max_ampl = np.abs(amplitudes[p1i])
    # phase areas and zeros:
    phase_areas = np.zeros(len(phase_list))
    zero_times = np.zeros(len(phase_list))
    for i in range(len(phase_list)):
        sign_fac = np.sign(eod[phase_list[i]])
        i0 = phase_list[i - 1] if i > 0 else 0
        i1 = phase_list[i + 1] if i + 1 < len(phase_list) else len(eod)
        if i0 > 0 and sign_fac*eod[i0] > 0 and \
           i1 < len(eod) and sign_fac*eod[i1] > 0:
            phase_areas[i] = 0
        else:
            snippet = sign_fac*eod[i0:i1]
            phase_areas[i] = np.sum(snippet[snippet > 0])*dt
            phase_areas[i] *= sign_fac
        if i < len(phase_list) - 1:
            i0 = phase_list[i]
            snippet = eod[i0:i1]
            stimes = time[i0:i1]
            zidx = np.nonzero(snippet[:-1]*snippet[1:] < 0)[0]
            if len(zidx) == 0:
                zero_times[i] = np.nan
            else:
                zidx = zidx[len(zidx)//2]  # reduce to single zero crossing
                snippet = snippet[zidx:zidx + 2]
                stimes = stimes[zidx:zidx + 2]
                if sign_fac > 0:
                    zero_times[i] = np.interp(0, snippet[::-1], stimes[::-1])
                else:
                    zero_times[i] = np.interp(0, snippet, stimes)
        else:
            zero_times[i] = np.nan
    total_area = np.sum(np.abs(phase_areas))
    # store phase properties:
    phases = dict(indices=np.arange(len(phase_list)) + 1 - p1i,
                  times=time[phase_list],
                  amplitudes=amplitudes,
                  relamplitudes=amplitudes/max_ampl,
                  widths=width_list,
                  areas=phase_areas,
                  relareas=phase_areas/total_area,
                  zeros=zero_times)
    return tstart, tend, phases


def gaussian_sum(x, *pas):
    """Sum of Gaussians.
    
    Parameters
    ----------
    x: array of float
        The x array over which the sum of Gaussians is evaluated.
    *pas: list of floats
        The parameters of the Gaussians in a flat list.
        Position, amplitude, and standard deviation of first Gaussian,
        position, amplitude, and standard deviation of second Gaussian,
        and so on.
    
    Returns
    -------
    sg: array of float
        The sum of Gaussians for the times given in `t`.
    """
    sg = np.zeros(len(x))
    for pos, ampl, std in zip(pas[0:-2:3], pas[1:-1:3], pas[2::3]):
        sg += ampl*np.exp(-0.5*((x - pos)/std)**2)
    return sg


def gaussian_sum_spectrum(f, *pas):
    """Energy spectrum of sum of Gaussians.

    Parameters
    ----------
    f: 1-D array of float
        The frequencies at which to evaluate the spectrum.
    *pas: list of floats
        The parameters of the Gaussians in a flat list.
        Position, amplitude, and standard deviation of first Gaussian,
        position, amplitude, and standard deviation of second Gaussian,
        and so on.
    
    Returns
    -------
    energy: 1-D array of float
        The one-sided energy spectrum of the sum of Gaussians.
    """
    spec = np.zeros(len(f), dtype=complex)
    for dt, a, s in zip(pas[0:-2:3], pas[1:-1:3], pas[2::3]):
        gauss = a*np.sqrt(2*np.pi)*s*np.exp(-0.5*(2*np.pi*s*f)**2)
        shift = np.exp(-2j*np.pi*f*dt)
        spec += gauss*shift
    spec *= np.sqrt(2)    # because of one-sided spectrum
    return np.abs(spec)**2


def gaussian_sum_costs(pas, time, eod, freqs, energy):
    """ Cost function for fitting sum of Gaussian to pulse EOD.
    
    Parameters
    ----------
    pas: list of floats
        The pulse parameters in a flat list.
        Position, amplitude, and standard deviation of first phase,
        position, amplitude, and standard deviation of second phase,
        and so on.
    time: 1-D array of float
        Time points of the EOD waveform.
    eod: 1-D array of float
        The real EOD waveform.
    freqs: 1-D array of float
        The frequency components of the spectrum.
    energy: 1-D array of float
        The energy spectrum of the real pulse.
    
    Returns
    -------
    costs: float
        Weighted sum of rms waveform difference and rms spectral difference.
    """
    eod_fit = gaussian_sum(time, *pas)
    eod_rms = np.sqrt(np.mean((eod_fit - eod)**2))/np.max(np.abs(eod))
    level = decibel(energy)
    level_range = 30
    n = np.argmax(level)
    energy_fit = gaussian_sum_spectrum(freqs, *pas)
    level_fit = decibel(energy_fit)
    weight = np.ones(n)
    weight[freqs[:n] < 10] = 100
    weight /= np.sum(weight)
    spec_rms = np.sqrt(np.mean(weight*(level_fit[:n] - level[:n])**2))/level_range
    costs = eod_rms + 5*spec_rms
    #print(f'{costs:.4f}, {eod_rms:.4f}, {spec_rms:.4f}')
    return costs


def decompose_pulse(eod, freqs, energy, phases, width_frac=0.5, verbose=0):
    """Decompose single pulse waveform into sum of Gaussians.

    Use the output to simulate pulse-type EODs using the functions
    provided in the thunderfish.fakefish module.
    
    Parameters
    ----------
    eod: 2-D array of float
        The eod waveform. First column is time in seconds and second
        column is the eod waveform. Further columns are ignored.
    freqs: 1-D array of float
        The frequency components of the spectrum.
    energy: 1-D array of float
        The energy spectrum of the real pulse.
    phases: dict
        Properties of the EOD phases as returned by analyze_pulse_phases(). 
    width_frac: float
        The width of a peak is measured at this fraction of a peak's
        height (0-1).
    verbose: int
        Verbosity level passed for error and info messages.

    Returns
    -------
    pulse: dict
        Dictionary with
    
        - "times": phase times in seconds,
        - "amplitudes": amplitudes, and
        - "stdevs": standard deviations in seconds
    
        of Gaussians fitted to the pulse waveform.  Use the functions
        provided in thunderfish.fakefish to simulate pulse fish EODs
        from this data.

    """
    pulse = {}
    if len(phases) == 0:
        return pulse
    if eod.ndim == 2:
        time = eod[:, 0]
        eod = eod[:, 1]
    elif isinstance(ratetime, (list, tuple, np.ndarray)):
        time = ratetime
    else:
        time = np.arange(len(eod))/rate
    # convert half width to standard deviation:
    fac = 0.5/np.sqrt(-2*np.log(width_frac))
    # fit parameter as single list:
    tas = []
    for t, a, s in zip(phases['times'], phases['amplitudes'],
                       phases['widths']*fac):
        tas.extend((t, a, s))
    tas = np.asarray(tas)
    # fit EOD waveform:
    try:
        tas, _ = curve_fit(gaussian_sum, time, eod, tas)
    except RuntimeError as e:
        if verbose > 0:
            print('Fit gaussian_sum failed in decompose_pulse():', e)
        return pulse
    # fit EOD waveform and spectrum:
    bnds = [(1e-5, None) if k%3 == 2 else (None, None)
            for k in range(len(tas))]
    res = minimize(gaussian_sum_costs, tas,
                   args=(time, eod, freqs, energy), bounds=bnds)
    if not res.success and verbose > 0:
        print('warning: optimization gaussian_sum_costs failed in decompose_pulse():',
              res.message)
    else:
        tas = res.x
        # add another Gaussian:
        rms_norm = np.max(np.abs(eod))
        rms_old = np.sqrt(np.mean((gaussian_sum(time, *tas) - eod)**2))/rms_norm
        eod_diff = np.abs(gaussian_sum(time, *tas) - eod)/rms_norm
        if np.max(eod_diff) > 0.1:
            if verbose > 1:
                print(f'decompose_pulse(): added Gaussian because maximum rms error was {100*np.max(eod_diff):.0f}%')
            ntas = np.concatenate((tas, (time[np.argmax(eod_diff)], np.max(eod_diff),
                                         np.mean(tas[2::3]))))
            bnds = [(1e-5, None) if k%3 == 2 else (None, None)
                    for k in range(len(ntas))]
            res = minimize(gaussian_sum_costs, ntas,
                           args=(time, eod, freqs, energy), bounds=bnds)
            if res.success:
                rms_new = np.sqrt(np.mean((gaussian_sum(time, *res.x) - eod)**2))/rms_norm
                if rms_new < 0.8*rms_old:
                    tas = res.x
                elif verbose > 1:
                    print('decompose_pulse(): removed added Gaussian because it did not improve the fit')
            elif verbose > 0:
                print('warnong: optimization gaussian_sum_costs for additional Gaussian failed in decompose_pulse():',
                      res.message)
    times = np.asarray(tas[0::3])
    ampls = np.asarray(tas[1::3])
    stdevs = np.asarray(tas[2::3])
    pulse = dict(times=times, amplitudes=ampls, stdevs=stdevs)
    return pulse

            
def analyze_pulse_tail(peak_index, eod, ratetime=None,
                       threshold=0.0, fit_frac=0.5, verbose=0):
    """ Fit exponential to last peak/trough of pulse EOD.
    
    Parameters
    ----------
    peak_index: int
        Index of last peak in `eod`.
    eod: 1-D or 2-D array
        The eod waveform to be analyzed.
        If an 1-D array, then this is the waveform and you
        need to also pass a sampling rate in `rate`.
        If a 2-D array, then first column is time in seconds and second
        column is the eod waveform. Further columns are ignored.
    ratetime: None or float or array of float
        If a 1-D array is passed on to `eod` then either the sampling
        rate in Hertz or the time array corresponding to `eod`.
    threshold: float
        Maximum noise level of the pulse waveform.
    fit_frac: float or None
        An exponential is fitted to the tail of the last peak/trough
        starting where the waveform falls below this fraction of the
        peak's height (0-1).
        If None, do not attempt to fit.
    verbose: int
        Verbosity level passed for error and info messages.
    
    Returns
    -------
    tau: float or np.nan
        Time constant of pulse tail in seconds.
    tstart: float or np.nan
        Time where fit started in seconds.
    fit: 1-D array of float or np.nan
        Time trace of the fit corresponding to `eod`.
    """
    if fit_frac is None:
        return np.nan, np.nan, None
    if eod.ndim == 2:
        time = eod[:, 0]
        eod = eod[:, 1]
    elif isinstance(ratetime, (list, tuple, np.ndarray)):
        time = ratetime
    else:
        time = np.arange(len(eod))/rate
    dt = np.mean(np.diff(time))
    pi = peak_index
    # positive or negative decay:
    sign = 1
    eodpp = eod[pi:] - 0.5*threshold
    eodpn = -eod[pi:] - 0.5*threshold
    if np.sum(eodpn[eodpn > 0]) > np.sum(eodpp[eodpp > 0]):
        sign = -1
    if sign*eod[pi] < 0:
        pi += np.argmax(sign*eod[pi:])
    pi_ampl = np.abs(eod[pi])
    # no sufficiently large initial value:
    sampl = sign*eod[pi]*fit_frac
    if sampl <= threshold:
        if verbose > 0:
            print(f'exponential fit to tail of pulse failed: initial amplitude {sampl:.5f} smaller than threshold {threshold:.5f}')
        return np.nan, np.nan, None
    # no sufficiently long decay:
    sinx = pi + np.argmax(sign*eod[pi:] < sampl)
    n = 2*len(eod[sinx:])//3
    if n < 10:
        if verbose > 0:
            print(f'exponential fit to tail of pulse failed: less than 10 samples {n}')
        return np.nan, np.nan, None
    # not decaying towards zero:
    max_line = sampl - (sampl - threshold)*np.arange(n)/n + 1e-8
    above_frac = np.sum(sign*eod[sinx:sinx + n] > max_line)/n
    if above_frac > 0.05:
        if verbose > 0:
            print(f'exponential fit to tail of pulse failed: not decaying towards zero {100*above_frac:.1f}% > 5%')
        return np.nan, np.nan, None
    # estimate tau:
    thresh = eod[sinx]*np.exp(-1.0)
    tau_inx = np.argmax(sign*eod[sinx:] < sign*thresh)
    if tau_inx < 2:
        tau_inx = 2
    rinx = sinx + 6*tau_inx
    if rinx > len(eod) - 1:
        rinx = len(eod) - 1
        if rinx - sinx < 2*tau_inx:
            if verbose > 0:
                print(f'exponential fit to tail of pulse failed: less samples {rinx - sinx} than two time constants 3*{tau_inx}')
            return np.nan, np.nan, None
    tau = time[sinx + tau_inx] - time[sinx]
    params = [tau, eod[sinx] - eod[rinx], eod[rinx]]
    try:
        popt, pcov = curve_fit(exp_decay, time[sinx:rinx] - time[sinx],
                               eod[sinx:rinx], params,
                               bounds=([0.0, -np.inf, -np.inf], np.inf))
    except TypeError:
        popt, pcov = curve_fit(exp_decay, time[sinx:rinx] - time[sinx],
                               eod[sinx:rinx], params)
    if popt[0] > 1.2*tau:
        tau_inx = int(np.round(popt[0]/dt))
        rinx = sinx + 6*tau_inx
        if rinx > len(eod) - 1:
            rinx = len(eod) - 1
        try:
            popt, pcov = curve_fit(exp_decay, time[sinx:rinx] - time[sinx],
                                   eod[sinx:rinx], popt,
                                   bounds=([0.0, -np.inf, -np.inf], np.inf))
        except TypeError:
            popt, pcov = curve_fit(exp_decay, time[sinx:rinx] - time[sinx],
                                   eod[sinx:rinx], popt)
    tau = popt[0]
    tstart = time[sinx]
    fit = np.zeros(len(eod))
    fit[:] = np.nan
    fit[sinx:rinx] = exp_decay(time[sinx:rinx] - time[sinx], *popt)
    if verbose > 0:
        print(f'exponential fit to tail of pulse: got time constant {1000*tau:.3f}ms')
    return tau, tstart, fit

    
def pulse_spectrum(eod, ratetime=None, freq_resolution=1.0, fade_frac=0.0):
    """Compute the spectrum of a single pulse-type EOD.
    
    Parameters
    ----------
    eod: 1-D or 2-D array
        The eod waveform of which the spectrum is computed.
        If an 1-D array, then this is the waveform and you
        need to also pass a sampling rate in `rate`.
        If a 2-D array, then first column is time in seconds and second
        column is the eod waveform. Further columns are ignored.
    ratetime: None or float or array of float
        If a 1-D array is passed on to `eod` then either the sampling
        rate in Hertz or the time array corresponding to `eod`.
    freq_resolution: float
        The frequency resolution of the spectrum.
    fade_frac: float
        Fraction of time of the EOD waveform that is used to fade in
        and out to zero baseline.
    
    Returns
    -------
    freqs: 1-D array of float
        The frequency components of the energy spectrum.
    energy: 1-D array of float
        The energy spectrum of the single pulse EOD
        with unit (x s)^2 = x^2 s/Hz.
        The integral over the energy spectrum `np.sum(energy)*freqs[1]`
        equals the integral over the squared eod, `np.sum(eod**2)/rate`.
        That is, by making the energy spectrum a power sepctrum
        (dividing the energy by the FFT window duration), the integral
        over the power spectrum equals the mean-squared signal
        (variance). But the single-pulse spectrum is not a power-spectrum.
        because in the limit to infinitely long window, the power vanishes!

    See Also
    --------
    thunderfish.fakefish.pulsefish_spectrum(): analytically computed spectra for simulated pulse EODs.

    """
    if eod.ndim == 2:
        rate = 1.0/(eod[1, 0] - eod[0, 0])
        eod = eod[:, 1]
    elif isinstance(ratetime, (list, tuple, np.ndarray)):
        rate = 1.0/(ratetime[1] - ratetime[0])
    else:
        rate = ratetime
    n_fft = nfft(rate, freq_resolution)
    # subtract mean computed from the ends of the EOD snippet:
    n = len(eod)//20 if len(eod) >= 20 else 1
    eod = eod - 0.5*(np.mean(eod[:n]) + np.mean(eod[-n:]))
    # zero padding:
    max_idx = np.argmax(eod)
    n0 = max_idx
    n1 = len(eod) - max_idx
    n = 2*max(n0, n1)
    if n_fft < n:
        n_fft = next_power_of_two(n)
    data = np.zeros(n_fft)
    data[n_fft//2 - n0:n_fft//2 + n1] = eod
    # fade in and out:
    if fade_frac > 0:
        fn = int(fade_frac*len(eod))
        data[n_fft//2 - n0:n_fft//2 - n0 + fn] *= np.arange(fn)/fn
        data[n_fft//2 + n1 - fn:n_fft//2 + n1] *= np.arange(fn)[::-1]/fn
    # spectrum:
    dt = 1/rate
    freqs = np.fft.rfftfreq(n_fft, dt)
    fourier = np.fft.rfft(data)*dt
    energy = 2*np.abs(fourier)**2     # one-sided spectrum!
    return freqs, energy


def analyze_pulse_spectrum(freqs, energy):
    """Analyze the spectrum of a pulse-type EOD.
    
    Parameters
    ----------
    freqs: 1-D array of float
        The frequency components of the energy spectrum.
    energy: 1-D array of float
        The energy spectrum of the single pulse-type EOD.
    
    Returns
    -------
    peak_freq: float
        Frequency at peak energy of the spectrum in Hertz.
    peak_energy: float
        Peak energy of the pulse spectrum in x^2 s/Hz.
    trough_freq: float
        Frequency at trough before peak in Hertz.
    trough_energy: float
        Energy of trough before peak in x^2 s/Hz.
    att5: float
        Attenuation of average energy below 5 Hz relative to
        peak energy in decibel.
    att50: float
        Attenuation of average energy below 50 Hz relative to
        peak energy in decibel.
    low_cutoff: float
        Frequency at which the energy reached half of the
        peak energy relative to the DC energy in Hertz.
    high_cutoff: float
        3dB roll-off frequency in Hertz.
    """
    ip = np.argmax(energy)
    peak_freq = freqs[ip]
    peak_energy = energy[ip]
    it = np.argmin(energy[:ip]) if ip > 0 else 0
    trough_freq = freqs[it]
    trough_energy = energy[it]
    att5 = decibel(np.mean(energy[freqs<5.0]), peak_energy)
    att50 = decibel(np.mean(energy[freqs<50.0]), peak_energy)
    low_cutoff = freqs[decibel(energy, peak_energy) > 0.5*att5][0]
    high_cutoff = freqs[decibel(energy, peak_energy) > -3.0][-1]
    return peak_freq, peak_energy, trough_freq, trough_energy, \
        att5, att50, low_cutoff, high_cutoff


def analyze_pulse_intervals(eod_times, ipi_cv_thresh=0.5,
                            ipi_percentile=30.0):
    """ Basic statistics of interpulse intervals.
    
    Parameters
    ----------
    eod_times: 1-D array or None
        List of times of detected EODs.
    ipi_cv_thresh: float
        If the coefficient of variation of the interpulse intervals
        (IPIs) is smaller than this threshold, then the statistics of
        IPIs is estimated from all IPIs. Otherwise only intervals
        smaller than a certain percentile are used.
    ipi_percentile: float
        When computing the statistics of IPIs from a subset of the
        IPIs, only intervals smaller than this percentile (between 0
        and 100) are used.

    Returns
    -------
    ipi_median: float
        Median inter-pulse interval.
    ipi_mean: float
        Mean inter-pulse interval.
    ipi_std: float
        Standard deviation of inter-pulse intervals.

    """
    if eod_times is None:
        return None, None, None
    inter_pulse_intervals = np.diff(eod_times)
    ipi_cv = np.std(inter_pulse_intervals)/np.mean(inter_pulse_intervals)
    if ipi_cv < ipi_cv_thresh:
        ipi_median = np.median(inter_pulse_intervals)
        ipi_mean = np.mean(inter_pulse_intervals)
        ipi_std = np.std(inter_pulse_intervals)
    else:
        intervals = inter_pulse_intervals[inter_pulse_intervals <
                                np.percentile(inter_pulse_intervals, ipi_percentile)]
        ipi_median = np.median(intervals)
        ipi_mean = np.mean(intervals)
        ipi_std = np.std(intervals)
    return ipi_median, ipi_mean, ipi_std

            
def analyze_pulse(eod, ratetime=None, eod_times=None,
                  min_pulse_win=0.001,
                  start_end_thresh_fac=0.01, peak_thresh_fac=0.002,
                  min_dist=50.0e-6, width_frac=0.5, fit_frac=0.5,
                  freq_resolution=1.0, fade_frac=0.0,
                  flip_pulse='none', ipi_cv_thresh=0.5,
                  ipi_percentile=30.0, verbose=0):
    """Analyze the EOD waveform of a pulse fish.
    
    Parameters
    ----------
    eod: 1-D or 2-D array
        The eod waveform to be analyzed.
        If an 1-D array, then this is the waveform and you
        need to also pass a time array or sampling rate in `ratetime`.
        If a 2-D array, then first column is time in seconds, second
        column the EOD waveform, third column, if present, is the
        standard error of the EOD waveform. Further columns are
        optional but not used.
    ratetime: None or float or array of float
        If a 1-D array is passed on to `eod` then either the sampling
        rate in Hertz or the time array corresponding to `eod`.
    eod_times: 1-D array or None
        List of times of detected EOD peaks.
    min_pulse_win: float
        The minimum size of cut-out EOD waveform.
    start_end_thresh_fac: float
        Set the threshold for the start and end time to the p-p amplitude
        times this factor.
    peak_thresh_fac: float
        Set the threshold for peak and trough  detection to the p-p amplitude
        times this factor.
    min_dist: float
        Minimum distance between peak and troughs of the pulse.
    width_frac: float
        The width of a peak is measured at this fraction of a peak's
        height (0-1).
    fit_frac: float or None
        An exponential is fitted to the tail of the last peak/trough
        starting where the waveform falls below this fraction of the
        peak's height (0-1).
    freq_resolution: float
        The frequency resolution of the spectrum of the single pulse.
    fade_frac: float
        Fraction of time of the EOD waveform that is fade in and out
        to zero baseline.
    flip_pulse: 'auto', 'none', 'flip'
        - 'auto' flip waveform such that the first large extremum is positive.
        - 'flip' flip waveform.
        - 'none' do not flip waveform.
    ipi_cv_thresh: float
        If the coefficient of variation of the interpulse intervals
        (IPIs) is smaller than this threshold, then the statistics of
        IPIs is estimated from all IPIs. Otherwise only intervals
        smaller than a certain percentile are used.
    ipi_percentile: float
        When computing the statistics of IPIs from a subset of the
        IPIs, only intervals smaller than this percentile (between 0
        and 100) are used.
    verbose: int
        Verbosity level passed for error and info messages.
    
    Returns
    -------
    meod: 2-D array of floats
        The eod waveform. First column is time in seconds,
        second column the eod waveform.
        Further columns are kept from the input `eod`.
        As the two last columns the waveform resulting from the
        decomposition into Gaussians and the fit to the tail of the
        last peak are appended.
    props: dict
        A dictionary with properties of the analyzed EOD waveform.

        - type: set to 'pulse'.
        - flipped: True if the waveform was flipped.
        - n: number of pulses analyzed  (i.e. `len(eod_times)`), if provided.
        - times: the times of the detected EOD pulses (i.e. `eod_times`),
          if provided.
        - EODf: the inverse of the median interval between `eod_times`,
          if provided.
        - period: median interval between `eod_times`, if provided.
        - IPI-mean: mean interval between `eod_times`, if provided.
        - IPI-std: standard deviation of the intervals between
          `eod_times`, if provided.
        - IPI-CV: coefficient of variation of the intervals between
          `eod_times`, if provided.
        - aoffs: Offset that was subtracted from the average EOD waveform.
        - pos-ampl: amplitude of the largest positive peak.
        - neg-ampl: amplitude of the largest negative trough.
        - max-ampl: maximum of largest peak or trough amplitude in the units of the input data.
        - p-p-amplitude: peak-to-peak amplitude of the EOD waveform.
        - p-p-dist: distance between minimum and maximum phase in seconds.
        - noise: average standard error mean of the averaged
          EOD waveform relative to the p-p amplitude.
        - noise: average standard error of the averaged EOD waveform relative to the peak-to_peak amplitude in percent.
        - rmserror: root-mean-square error between fit with sum of Gaussians and
          EOD waveform relative to the p-p amplitude. Infinity if fit failed.
        - peakthresh: Threshold for detecting peaks and troughs is at this factor times p-p-ampl.
        - startendthresh: Threshold for start and end time is at this factor times p-p-ampl.
        - tstart: time in seconds where the pulse starts,
          i.e. crosses the threshold for the first time.
        - tend: time in seconds where the pulse ends,
          i.e. crosses the threshold for the last time.
        - width: total width of the pulse in seconds (tend-tstart).
        - totalarea: sum of areas under positive and negative peaks.
        - pos-area: area under positive phases relative to total area.
        - neg-area: area under negative phases relative to total area.
        - polaritybalance: contrast between areas under positive and
          negative phases.
        - median: median of the distribution of the absolute EOD waveform.
        - quartile1: first quartile of the distribution of the absolute EOD waveform.
        - quartile3: third quartile of the distribution of the absolute EOD waveform.
        - iq-range: inter-quartile range of the distribution of the absolute EOD waveform.
        - tau: time constant of exponential decay of pulse tail in seconds.
        - firstphase: index of the first phase in the pulse (i.e. -1 for P-1)
        - lastphase: index of the last phase in the pulse (i.e. 3 for P3)
        - peakfreq: frequency at peak energy of the single pulse spectrum
          in Hertz.
        - peakenergy: peak energy of the single pulse spectrum.
        - troughfreq: frequency at trough before peak in Hertz.
        - troughenergy: energy of trough before peak in x^2 s/Hz.
        - energyatt5: attenuation of average energy below 5 Hz relative to
          peak energy in decibel.
        - energyatt50: attenuation of average energy below 50 Hz relative to
          peak energy in decibel.
        - lowcutoff: frequency at which the energy reached half of the
          peak energy relative to the DC energy in Hertz.
        - highcutoff: 3dB roll-off frequency of spectrum in Hertz.

        Empty if waveform is not a pulse EOD.
    phases: dict
        Dictionary with
    
        - "indices": indices of each phase
          (1 is P1, i.e. the largest positive peak)
        - "times": times of each phase relative to P1 in seconds
        - "amplitudes": amplitudes of each phase
        - "relamplitudes": amplitudes normalized to amplitude of P1 phase
        - "widths": widths of each phase at `width_frac` height
        - "areas": areas of each phase
        - "relareas": areas of the phases relative to total area
        - "zeros": time of zero crossing towards next phase in seconds
    
        Empty dictionary if waveform is not a pulse EOD.
    pulse: dict
        Dictionary with
    
        - "times": phase times in seconds,
        - "amplitudes": amplitudes, and
        - "stdevs": standard deviations in seconds
    
        of Gaussians fitted to the pulse waveform.  Use the functions
        provided in thunderfish.fakefish to simulate pulse fish EODs
        from this data.
    energy: 2-D array
        The energy spectrum of a single pulse. First column are the
        frequencies, second column the energy in x^2 s/Hz.
        Empty if waveform is not a pulse EOD.

    """
    if eod.ndim == 2:
        eeod = eod
    else:
        if isinstance(ratetime, (list, tuple, np.ndarray)):
            time = ratetime
        else:
            time = np.arange(len(eod))/rate
        eeod = np.zeros((len(eod), 2))
        eeod[:, 0] = time
        eeod[:, 1] = eod
    # storage:
    meod = np.zeros((eeod.shape[0], eeod.shape[1] + 2))
    meod[:, :eeod.shape[1]] = eeod
    meod[:, -2] = np.nan
    meod[:, -1] = np.nan

    # conditioning of the waveform:
    meod, toffs, aoffs, flipped, noise_thresh = \
        condition_pulse(meod, flip=flip_pulse,
                        baseline_frac=0.05, large_phase_frac=0.2,
                        min_pulse_win=min_pulse_win)

    # analysis of pulse waveform:
    pos_ampl, neg_ampl, dist, pos_area, neg_area, \
        polarity_balance, median, quartile1, quartile3 = \
        analyze_pulse_properties(noise_thresh, meod)
    pp_ampl = pos_ampl + neg_ampl
    max_ampl = max(pos_ampl, neg_ampl)
    total_area = pos_area + neg_area

    # threshold for start and end time:
    start_end_thresh = pp_ampl*start_end_thresh_fac
    if start_end_thresh < 2*noise_thresh:
        start_end_thresh = 2*noise_thresh
        start_end_thresh_fac = start_end_thresh/pp_ampl if pp_ampl > 0 else 1

    # threshold for peak detection:
    peak_thresh = pp_ampl*peak_thresh_fac
    if peak_thresh < noise_thresh:
        peak_thresh = noise_thresh
        peak_thresh_fac = peak_thresh/pp_ampl if pp_ampl > 0 else 1
            
    # characterize EOD phases:
    tstart, tend, phases = analyze_pulse_phases(peak_thresh,
                                                start_end_thresh, meod,
                                                min_dist=min_dist,
                                                width_frac=width_frac)
        
    # fit exponential to last phase:
    tau = np.nan
    taustart = np.nan
    if len(phases) > 0 and len(phases['times']) > 1:
        pi = np.argmin(np.abs(meod[:, 0] - phases['times'][-1]))
        tau, taustart, fit = analyze_pulse_tail(pi, meod, None,
                                                threshold=noise_thresh,
                                                fit_frac=fit_frac,
                                                verbose=verbose)
        if fit is not None:
            meod[:, -1] = fit

    # energy spectrum of single EOD pulse:
    freqs, energy = pulse_spectrum(meod, None, freq_resolution, fade_frac)
    # store spectrum:
    eenergy = np.zeros((len(energy), 2))
    eenergy[:, 0] = freqs
    eenergy[:, 1] = energy
    # analyse spectrum:
    peakfreq, peakenergy, troughfreq, troughenergy, \
        att5, att50, lowcutoff, highcutoff = \
        analyze_pulse_spectrum(freqs, energy)

    # decompose EOD waveform:
    rmserror = np.inf
    pulse = decompose_pulse(meod, freqs, energy, phases,
                            width_frac, verbose=verbose)
    if len(pulse) > 0:
        eod_fit = pulsefish_waveform(pulse, meod[:, 0])
        meod[:, -2] = eod_fit
        rmserror = np.sqrt(np.mean((meod[:, 1] - meod[:, -2])**2))/pp_ampl

    # analyze pulse intervals:
    ipi_median, ipi_mean, ipi_std = \
        analyze_pulse_intervals(eod_times,  ipi_cv_thresh, ipi_percentile)
    
    # store properties:
    props = {}
    props['type'] = 'pulse'
    props['flipped'] = flipped
    if eod_times is not None:
        props['n'] = len(eod_times)
        props['times'] = eod_times + toffs
        props['EODf'] = 1.0/ipi_median
        props['period'] = ipi_median
        props['IPI-mean'] = ipi_mean
        props['IPI-std'] = ipi_std
        props['IPI-CV'] = ipi_std/ipi_mean
    props['aoffs'] = aoffs
    props['pos-ampl'] = pos_ampl
    props['neg-ampl'] = neg_ampl
    props['max-ampl'] = max_ampl
    props['p-p-amplitude'] = pp_ampl
    props['p-p-dist'] = dist
    if eod.shape[1] > 2:
        props['noise'] = np.mean(meod[:, 2])/pp_ampl if pp_ampl > 0 else 1
    props['rmserror'] = rmserror
    props['peakthresh'] = peak_thresh_fac
    props['startendthresh'] = start_end_thresh_fac
    props['tstart'] = tstart
    props['tend'] = tend
    props['width'] = tstart - tend
    props['totalarea'] = total_area
    props['pos-area'] = pos_area/total_area
    props['neg-area'] = neg_area/total_area
    props['polaritybalance'] = polarity_balance
    props['median'] = median
    props['quartile1'] = quartile1
    props['quartile3'] = quartile3
    props['iq-range'] = quartile3 - quartile1
    props['tau'] = tau
    props['taustart'] = taustart
    props['firstphase'] = phases['indices'][0] if len(phases) > 0 else 1
    props['lastphase'] = phases['indices'][-1] if len(phases) > 0 else 1
    props['peakfreq'] = peakfreq
    props['peakenergy'] = peakenergy
    props['troughfreq'] = troughfreq
    props['troughenergy'] = troughenergy
    props['energyatt5'] = att5
    props['energyatt50'] = att50
    props['lowcutoff'] = lowcutoff
    props['highcutoff'] = highcutoff
    
    return meod, props, phases, pulse, eenergy


def save_pulse_fish(eod_props, unit, basename, **kwargs):
    """Save properties of pulse EODs to file.

    Parameters
    ----------
    eod_props: list of dict
        Properties of EODs as returned by `analyze_wave()` and
        `analyze_pulse()`.  Only properties of pulse fish are saved.
    unit: string
        Unit of the waveform data.
    basename: string or stream
        If string, path and basename of file.
        If `basename` does not have an extension,
        '-pulsefish' and a file extension are appended.
        If stream, write pulse fish properties into this stream.
    kwargs:
        Arguments passed on to `TableData.write()`.

    Returns
    -------
    filename: Path or None
        Path and full name of the written file in case of `basename`
        being a string. Otherwise, the file name and extension that
        would have been appended to a basename.
        None if no pulse fish are contained in eod_props and
        consequently no file was written.

    See Also
    --------
    load_pulse_fish()
    """
    pulse_props = [p for p in eod_props if p['type'] == 'pulse']
    if len(pulse_props) == 0:
        return None
    td = TableData()
    if 'twin' in pulse_props[0] or 'samplerate' in pulse_props[0] or \
       'nfft' in pulse_props[0]:
        td.append_section('recording')
    if 'twin' in pulse_props[0]:
        td.append('twin', 's', '%7.2f', value=pulse_props)
        td.append('window', 's', '%7.2f', value=pulse_props)
        td.append('winclipped', '%', '%.2f',
                  value=pulse_props, fac=100)
    if 'samplerate' in pulse_props[0]:
        td.append('samplerate', 'kHz', '%.3f', value=pulse_props,
                  fac=0.001)
    if 'nfft' in pulse_props[0]:
        td.append('nfft', '', '%d', value=pulse_props)
        td.append('dfreq', 'Hz', '%.2f', value=pulse_props)
    td.append_section('waveform')
    td.append('index', '', '%d', value=pulse_props)
    td.append('n', '', '%d', value=pulse_props)
    td.append('EODf', 'Hz', '%7.2f', value=pulse_props)
    td.append('period', 'ms', '%7.2f', value=pulse_props, fac=1000)
    td.append('aoffs', unit, '%.5f', value=pulse_props)
    td.append('pos-ampl', unit, '%.5f', value=pulse_props)
    td.append('neg-ampl', unit, '%.5f', value=pulse_props)
    td.append('max-ampl', unit, '%.5f', value=pulse_props)
    td.append('p-p-amplitude', unit, '%.5f', value=pulse_props)
    td.append('p-p-dist', 'ms', '%.3f', value=pulse_props, fac=1000)
    if 'noise' in pulse_props[0]:
        td.append('noise', '%', '%.2f', value=pulse_props, fac=100)
    td.append('rmserror', '%', '%.2f', value=pulse_props, fac=100)
    if 'clipped' in pulse_props[0]:
        td.append('clipped', '%', '%.2f', value=pulse_props, fac=100)
    td.append('flipped', '', '%d', value=pulse_props)
    td.append('startendthresh', '%', '%.2f', value=pulse_props, fac=100)
    td.append('peakthresh', '%', '%.2f', value=pulse_props, fac=100)
    td.append('tstart', 'ms', '%.3f', value=pulse_props, fac=1000)
    td.append('tend', 'ms', '%.3f', value=pulse_props, fac=1000)
    td.append('width', 'ms', '%.3f', value=pulse_props, fac=1000)
    td.append('totalarea', f'{unit}*ms', '%.4f', value=pulse_props, fac=1000)
    td.append('pos-area', '%', '%.2f', value=pulse_props, fac=100)
    td.append('neg-area', '%', '%.2f', value=pulse_props, fac=100)
    td.append('polaritybalance', '%', '%.2f', value=pulse_props, fac=100)
    td.append('median', 'ms', '%.3f', value=pulse_props, fac=1000)
    td.append('quartile1', 'ms', '%.3f', value=pulse_props, fac=1000)
    td.append('quartile3', 'ms', '%.3f', value=pulse_props, fac=1000)
    td.append('iq-range', 'ms', '%.3f', value=pulse_props, fac=1000)
    td.append('tau', 'ms', '%.3f', value=pulse_props, fac=1000)
    td.append('taustart', 'ms', '%.3f', value=pulse_props, fac=1000)
    td.append('firstphase', '', '%d', value=pulse_props)
    td.append('lastphase', '', '%d', value=pulse_props)
    td.append_section('spectrum')
    td.append('peakfreq', 'Hz', '%.2f', value=pulse_props)
    td.append('peakenergy', f'{unit}^2s/Hz', '%.3g', value=pulse_props)
    td.append('troughfreq', 'Hz', '%.2f', value=pulse_props)
    td.append('troughenergy', f'{unit}^2s/Hz', '%.3g', value=pulse_props)
    td.append('energyatt5', 'dB', '%.2f', value=pulse_props)
    td.append('energyatt50', 'dB', '%.2f', value=pulse_props)
    td.append('lowcutoff', 'Hz', '%.2f', value=pulse_props)
    td.append('highcutoff', 'Hz', '%.2f', value=pulse_props)
    ext = Path(basename).suffix if not hasattr(basename, 'write') else ''
    fp = '-pulsefish' if not ext else ''
    return td.write_file_stream(basename, fp, **kwargs)


def load_pulse_fish(file_path):
    """Load properties of pulse EODs from file.

    All times are scaled to seconds, all frequencies to Hertz, and all
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
    save_pulse_fish()

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
        if 'clipped' in props:
            props['clipped'] /= 100
        props['type'] = 'pulse'
        props['index'] = int(props['index'])
        props['n'] = int(props['n'])
        props['totalarea'] /= 1000
        props['pos-area'] /= 100
        props['neg-area'] /= 100
        props['polaritybalance'] /= 100
        props['median'] /= 1000
        props['quartile1'] /= 1000
        props['quartile3'] /= 1000
        props['iq-range'] /= 1000
        props['firstphase'] = int(props['firstphase'])
        props['lastphase'] = int(props['lastphase'])
        props['period'] /= 1000
        props['noise'] /= 100
        props['startendthresh'] /= 100
        props['peakthresh'] /= 100
        props['tstart'] /= 1000
        props['tend'] /= 1000
        props['p-p-dist'] /= 1000
        props['width'] /= 1000
        props['tau'] /= 1000
        props['taustart'] /= 1000
        props['rmserror'] /= 100
    return eod_props

                        
def save_pulse_spectrum(spec_data, unit, idx, basename, **kwargs):
    """Save energy spectrum of pulse EOD to file.

    Parameters
    ----------
    spec_data: 2D array of floats
        Energy spectrum of single pulse as returned by `analyze_pulse()`.
    unit: string
        Unit of the waveform data.
    idx: int or None
        Index of fish.
    basename: string or stream
        If string, path and basename of file.
        If `basename` does not have an extension,
        '-pulsespectrum', the fish index, and a file extension are appended.
        If stream, write pulse spectrum into this stream.
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
    load_pulse_spectrum()
    """
    td = TableData(spec_data[:, :2], ['frequency', 'energy'],
                   ['Hz', '%s^2s/Hz' % unit], ['%.2f', '%.4e'])
    fp = ''
    ext = Path(basename).suffix if not hasattr(basename, 'write') else ''
    if not ext:
        fp = '-pulsespectrum'
        if idx is not None:
            fp += f'-{idx}'
    return td.write_file_stream(basename, fp, **kwargs)


def load_pulse_spectrum(file_path):
    """Load energy spectrum of pulse EOD from file.

    Parameters
    ----------
    file_path: string
        Path of the file to be loaded.

    Returns
    -------
    spec: 2D array of floats
        Energy spectrum of single pulse: frequency, energy

    Raises
    ------
    FileNotFoundError:
        If `file_path` does not exist.

    See Also
    --------
    save_pulse_spectrum()
    """
    data = TableData(file_path)
    spec = data.array()
    return spec


def save_pulse_phases(phases, unit, idx, basename, **kwargs):
    """Save phase properties of pulse EOD to file.

    Parameters
    ----------
    phases: dict
        Dictionary with
    
        - "indices": indices of each phase
          (1 is P1, i.e. the largest positive peak)
        - "times": times of each phase relative to P1 in seconds
        - "amplitudes": amplitudes of each phase
        - "relamplitudes": amplitudes normalized to amplitude of P1 phase
        - "widths": widths of each phase at `width_frac` height
        - "areas": areas of each phase
        - "relareas": areas of the phases relative to total area
        - "zeros": time of zero crossing towards next phase in seconds
    
        as returned by `analyze_pulse_phases()` and  `analyze_pulse()`.
    unit: string
        Unit of the waveform data.
    idx: int or None
        Index of fish.
    basename: string or stream
        If string, path and basename of file.
        If `basename` does not have an extension,
        '-pulsephases', the fish index, and a file extension are appended.
        If stream, write pulse phases into this stream.
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
    load_pulse_phases()
    """
    if len(phases) == 0:
        return None
    td = TableData()
    td.append('type', '', '%s', value=['P']*len(phases['indices']))
    td.append('index', '', '%.0f', value=phases['indices'])
    td.append('time', 'ms', '%.4f', value=phases['times'], fac=1000)
    td.append('amplitude', unit, '%.5f', value=phases['amplitudes'])
    td.append('relampl', '%', '%.2f', value=phases['relamplitudes'], fac=100)
    td.append('width', 'ms', '%.4f', value=phases['widths'], fac=1000)
    td.append('area', f'{unit}*ms', '%.4f', value=phases['areas'], fac=1000)
    td.append('relarea', '%', '%.2f', value=phases['relareas'], fac=100)
    td.append('zeros', 'ms', '%.4f', value=phases['zeros'], fac=1000)
    fp = ''
    ext = Path(basename).suffix if not hasattr(basename, 'write') else ''
    if not ext:
        fp = '-pulsephases'
        if idx is not None:
            fp += f'-{idx}'
    return td.write_file_stream(basename, fp, **kwargs)


def load_pulse_phases(file_path):
    """Load phase properties of pulse EOD from file.

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
        - "times": times of each phase relative to P1 in seconds
        - "amplitudes": amplitudes of each phase
        - "relamplitudes": amplitudes normalized to amplitude of P1 phase
        - "widths": widths of each phase at `width_frac` height
        - "areas": areas of each phase
        - "relareas": areas of the phases relative to total area
        - "zeros": time of zero crossing towards next phase in seconds
    
    unit: string
        Unit of phase amplitudes.

    Raises
    ------
    FileNotFoundError:
        If `file_path` does not exist.

    See Also
    --------
    save_pulse_phases()
    """
    data = TableData(file_path)
    phases = dict(indices=data['index'].astype(int),
                  times=data['time']*0.001,
                  amplitudes=data['amplitude'],
                  relamplitudes=data['relampl']*0.01,
                  widths=data['width']*0.001,
                  areas=data['area']*0.001,
                  relareas=data['relarea']*0.01,
                  zeros=data['zeros']*0.001)
    return phases, data.unit('amplitude')


def save_pulse_gaussians(pulse, unit, idx, basename, **kwargs):
    """Save Gaussian phase properties of pulse EOD to file.

    Parameters
    ----------
    pulse: dict
        Dictionary with
    
        - "times": phase times in seconds,
        - "amplitudes": amplitudes, and
        - "stdevs": standard deviations in seconds
    
        of Gaussians fitted to the pulse waveform as returned by
        `decompose_pulse()` and `analyze_pulse()`.
    unit: string
        Unit of the waveform data.
    idx: int or None
        Index of fish.
    basename: string or stream
        If string, path and basename of file.
        If `basename` does not have an extension,
        '-pulsegaussians', the fish index, and a file extension are appended.
        If stream, write pulse phases into this stream.
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
    load_pulse_gaussians()

    """
    if len(pulse) == 0:
        return None
    td = TableData(pulse,
                   units=['ms', unit, 'ms'],
                   formats=['%.3f', '%.5f', '%.3f'])
    td['times'] *= 1000
    td['stdevs'] *= 1000
    fp = ''
    ext = Path(basename).suffix if not hasattr(basename, 'write') else ''
    if not ext:
        fp = '-pulsegaussians'
        if idx is not None:
            fp += f'-{idx}'
    return td.write_file_stream(basename, fp, **kwargs)


def load_pulse_gaussians(file_path):
    """Load Gaussian phase properties of pulse EOD from file.

    Parameters
    ----------
    file_path: string
        Path of the file to be loaded.

    Returns
    -------
    pulse: dict
        Dictionary with
    
        - "times": phase times in seconds,
        - "amplitudes": amplitudes, and
        - "stdevs": standard deviations in seconds
    
        of Gaussians fitted to the pulse waveform.
        Use the functions provided in thunderfish.fakefish to simulate
        pulse fish EODs from this data.
    unit: string
        Unit of Gaussian amplitudes.

    Raises
    ------
    FileNotFoundError:
        If `file_path` does not exist.

    See Also
    --------
    save_pulse_gaussians()

    """
    data = TableData(file_path)
    pulses = data.dict()
    pulses['times'] = 0.001*np.array(data['times'])
    pulses['amplitudes'] = np.array(data['amplitudes'])
    pulses['stdevs'] = 0.001*np.array(data['stdevs'])
    return pulses, data.unit('amplitudes')


def save_pulse_times(pulse_times, idx, basename, **kwargs):
    """Save times of pulse EOD to file.

    Parameters
    ----------
    pulse_times: dict or array of floats
        Times of EOD pulses. Either as array of times or
        `props['peaktimes']` or `props['times']` as returned by
        `analyze_pulse()`.
    idx: int or None
        Index of fish.
    basename: string or stream
        If string, path and basename of file.
        If `basename` does not have an extension,
        '-pulsetimes', the fish index, and a file extension are appended.
        If stream, write pulse times into this stream.
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
    load_pulse_times()
    """
    if isinstance(pulse_times, dict):
        props = pulse_times
        pulse_times = props.get('times', [])
        pulse_times = props.get('peaktimes', pulse_times)
    if len(pulse_times) == 0:
        return None
    td = TableData()
    td.append('time', 's', '%.4f', value=pulse_times)
    fp = ''
    ext = Path(basename).suffix if not hasattr(basename, 'write') else ''
    if not ext:
        fp = '-pulsetimes'
        if idx is not None:
            fp += f'-{idx}'
    return td.write_file_stream(basename, fp, **kwargs)


def load_pulse_times(file_path):
    """Load times of pulse EOD from file.

    Parameters
    ----------
    file_path: string
        Path of the file to be loaded.

    Returns
    -------
    pulse_times: array of floats
        Times of pulse EODs in seconds.

    Raises
    ------
    FileNotFoundError:
        If `file_path` does not exist.

    See Also
    --------
    save_pulse_times()
    """
    data = TableData(file_path)
    pulse_times = data.array()[:, 0]
    return pulse_times

