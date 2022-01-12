"""
Analysis of EOD waveforms.

## EOD waveform analysis

- `eod_waveform()`: compute an averaged EOD waveform.
- `analyze_wave()`: analyze the EOD waveform of a wave fish.
- `analyze_pulse()`: analyze the EOD waveform of a pulse fish.
- `adjust_eodf()`: adjust EOD frequencies to a standard temperature.

## Similarity of EOD waveforms

- `wave_similarity()`: root-mean squared difference between two wave fish EODs.
- `pulse_similarity()`: root-mean squared difference between two pulse fish EODs.
- `load_species_waveforms()`: load template EOD waveforms for species matching.

## Quality assessment

- `clipped_fraction()`: compute fraction of clipped EOD waveform snippets.
- `wave_quality()`: asses quality of EOD waveform of a wave fish.
- `pulse_quality()`: asses quality of EOD waveform of a pulse fish.

## Visualization

- `plot_eod_recording()`: plot a zoomed in range of the recorded trace.
- `plot_pulse_eods()`: mark pulse EODs in a plot of an EOD recording.
- `plot_eod_snippets()`: plot a few EOD waveform snippets.
- `plot_eod_waveform()`: plot and annotate the averaged EOD-waveform with standard error.
- `plot_wave_spectrum()`: plot and annotate spectrum of wave EODs.
- `plot_pulse_spectrum()`: plot and annotate spectrum of single pulse EOD.

## Storage

- `save_eod_waveform()`: save mean EOD waveform to file.
- `load_eod_waveform()`: load EOD waveform from file.
- `save_wave_eodfs()`: save frequencies of wave EODs to file.
- `load_wave_eodfs()`: load frequencies of wave EODs from file.
- `save_wave_fish()`: save properties of wave EODs to file.
- `load_wave_fish()`: load properties of wave EODs from file.
- `save_pulse_fish()`: save properties of pulse EODs to file.
- `load_pulse_fish()`: load properties of pulse EODs from file.
- `save_wave_spectrum()`: save amplitude and phase spectrum of wave EOD to file.
- `load_wave_spectrum()`: load amplitude and phase spectrum of wave EOD from file.
- `save_pulse_spectrum()`: save power spectrum of pulse EOD to file.
- `load_pulse_spectrum()`: load power spectrum of pulse EOD from file.
- `save_pulse_peaks()`: save peak properties of pulse EOD to file.
- `load_pulse_peaks()`: load peak properties of pulse EOD from file.

## Fit functions

- `fourier_series()`: Fourier series of sine waves with amplitudes and phases.
- `exp_decay()`: exponential decay.

## Filter functions

- `unfilter()`: apply inverse low-pass filter on data.

## Configuration

- `add_eod_analysis_config()`: add parameters for EOD analysis functions to configuration.
- `eod_waveform_args()`: retrieve parameters for `eod_waveform()` from configuration.
- `analyze_wave_args()`: retrieve parameters for `analyze_wave()` from configuration.
- `analyze_pulse_args()`: retrieve parameters for `analyze_pulse()` from configuration.
- `add_species_config()`: add parameters needed for assigning EOD waveforms to species.
- `add_eod_quality_config()`: add parameters needed for assesing the quality of an EOD waveform.
- `wave_quality_args()`: retrieve parameters for `wave_quality()` from configuration.
- `pulse_quality_args()`: retrieve parameters for `pulse_quality()` from configuration.
"""

import os
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from .eventdetection import percentile_threshold, detect_peaks, snippets, peak_width
from .eventdetection import threshold_crossings, threshold_crossing_times, merge_events
from .powerspectrum import next_power_of_two, nfft, decibel
from .harmonics import fundamental_freqs_and_power
from .tabledata import TableData


def eod_waveform(data, samplerate, eod_times, win_fac=2.0, min_win=0.01,
                 min_sem=False, max_eods=None, unfilter_cutoff=0.0):
    """Detect EODs in the given data, extract data snippets around each EOD,
    and compute a mean waveform with standard error.

    Retrieving the EOD waveform of a wave fish works under the following
    conditions: (i) at a signal-to-noise ratio \\(SNR = P_s/P_n\\),
    i.e. the power \\(P_s\\) of the EOD of interest relative to the
    largest other EOD \\(P_n\\), we need to average over at least \\(n >
    (SNR \\cdot c_s^2)^{-1}\\) snippets to bring the standard error of the
    averaged EOD waveform down to \\(c_s\\) relative to its
    amplitude. For a s.e.m. less than 5% ( \\(c_s=0.05\\) ) and an SNR of
    -10dB (the signal is 10 times smaller than the noise, \\(SNR=0.1\\) ) we
    get \\(n > 0.00025^{-1} = 4000\\) data snippets - a recording a
    couple of seconds long.  (ii) Very important for wave fish is that
    they keep their frequency constant.  Slight changes in the EOD
    frequency will corrupt the average waveform.  If the period of the
    waveform changes by \\(c_f=\\Delta T/T\\), then after \\(n =
    1/c_f\\) periods moved the modified waveform through a whole period.
    This is in the range of hundreds or thousands waveforms.

    NOTE: we need to take into account a possible error in the estimate
    of the EOD period. This will limit the maximum number of snippets to
    be averaged.

    If `min_sem` is set, the algorithm checks for a global minimum of
    the s.e.m.  as a function of snippet number. If there is one then
    the average is computed for this number of snippets, otherwise all
    snippets are taken from the provided data segment. Note that this
    check only works for the strongest EOD in a recording.  For weaker
    EOD the s.e.m. always decays with snippet number (empirical
    observation).

    TODO: use power spectra to check for changes in EOD frequency!

    Parameters
    ----------
    data: 1-D array of float
        The data to be analysed.
    samplerate: float
        Sampling rate of the data in Hertz.
    eod_times: 1-D array of float
        Array of EOD times in seconds over which the waveform should be averaged.
        WARNING: The first data point must be at time zero!
    win_fac: float
        The snippet size is the EOD period times `win_fac`. The EOD period is determined
        as the minimum interval between EOD times.
    min_win: float
        The minimum size of the snippets in seconds.
    min_sem: bool
        If set, check for minimum in s.e.m. to set the maximum numbers of EODs to be used
        for computing the average waveform.
    max_eods: int or None
        Maximum number of EODs to be used for averaging.
    unfilter_cutoff: float
        If not zero, the cutoff frequency for an inverse high-pass filter
        applied to the mean EOD waveform.
    
    Returns
    -------
    mean_eod: 2-D array
        Average of the EOD snippets. First column is time in seconds,
        second column the mean eod, third column the standard error.
    eod_times: 1-D array
        Times of EOD peaks in seconds that have been actually used to calculate the
        averaged EOD waveform.
    """
    # indices of EOD times:
    eod_idx = np.round(eod_times * samplerate).astype(np.int)
        
    # window size:
    period = np.min(np.diff(eod_times))
    win = 0.5*win_fac*period
    if 2*win < min_win:
        win = 0.5*min_win
    win_inx = int(win * samplerate)

    # extract snippets:
    eod_times = eod_times[(eod_idx >= win_inx) & (eod_idx < len(data)-win_inx)]
    eod_idx = eod_idx[(eod_idx >= win_inx) & (eod_idx < len(data)-win_inx)]
    if max_eods and max_eods > 0 and len(eod_idx) > max_eods:
        dn = (len(eod_idx) - max_eods)//2
        eod_times = eod_times[dn:dn+max_eods]
        eod_idx = eod_idx[dn:dn+max_eods]
    eod_snippets = snippets(data, eod_idx, -win_inx, win_inx)
    if len(eod_snippets) == 0:
        return np.zeros((0, 3)), eod_times

    # optimal number of snippets:
    step = 10
    if min_sem and len(eod_snippets) > step:
        sems = [np.mean(np.std(eod_snippets[:k], axis=0, ddof=1)/np.sqrt(k))
                for k in range(step, len(eod_snippets), step)]
        idx = np.argmin(sems)
        # there is a local minimum:
        if idx > 0 and idx < len(sems)-1:
            maxn = step*(idx+1)
            eod_snippets = eod_snippets[:maxn]
            eod_times = eod_times[:maxn]
    
    # mean and std of snippets:
    mean_eod = np.zeros((len(eod_snippets[0]), 3))
    mean_eod[:,1] = np.mean(eod_snippets, axis=0)
    if len(eod_snippets) > 1:
        mean_eod[:,2] = np.std(eod_snippets, axis=0, ddof=1)/np.sqrt(len(eod_snippets))
        
    # apply inverse filter:
    if unfilter_cutoff and unfilter_cutoff > 0.0:
        unfilter(mean_eod[:,1], samplerate, unfilter_cutoff)
        
    # time axis:
    mean_eod[:,0] = (np.arange(len(mean_eod)) - win_inx) / samplerate
    
    return mean_eod, eod_times


def unfilter(data, samplerate, cutoff):
    """
    Apply inverse high-pass filter on data.

    Assumes high-pass filter \\[ \\tau \\dot y = -y + \\tau \\dot x \\] has
    been applied on the original data \\(x\\), where \\(\\tau=(2\\pi
    f_{cutoff})^{-1}\\) is the time constant of the filter. To recover \\(x\\)
    the ODE \\[ \\tau \\dot x = y + \\tau \\dot y \\] is applied on the
    filtered data \\(y\\).

    Parameters
    ----------
    data: ndarray
        High-pass filtered original data.
    samplerate: float
        Sampling rate of `data` in Hertz.
    cutoff: float
        Cutoff frequency \\(f_{cutoff}\\) of the high-pass filter in Hertz.

    Returns
    -------
    data: ndarray
        Recovered original data.
    """
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


def fourier_series(t, freq, *ap):
    """
    Fourier series of sine waves with amplitudes and phases.

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


def analyze_wave(eod, freq, n_harm=10, power_n_harmonics=0, n_harmonics=3, flip_wave='none'):
    """
    Analyze the EOD waveform of a wave fish.
    
    Parameters
    ----------
    eod: 2-D array
        The eod waveform. First column is time in seconds, second column the EOD waveform,
        third column, if present, is the standard error of the EOD waveform,
        Further columns are optional but not used.
    freq: float or 2-D array
        The frequency of the EOD or the list of harmonics (rows)
        with frequency and peak height (columns) as returned from `harmonics.harmonic_groups()`.
    n_harm: int
        Maximum number of harmonics used for the Fourier decomposition.
    power_n_harmonics: int
        Sum over the first `power_n_harmonics` harmonics for computing the total power.
        If 0 sum over all harmonics.
    n_harmonics: int
        The maximum power of higher harmonics is computed from harmonics higher than the
        maximum harmonics within the first three harmonics plus `n_harmonics`.
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
        - rmssem: root-mean squared standard error mean of the averaged EOD waveform
          relative to the p-p amplitude.
        - rmserror: root-mean-square error between Fourier-fit and EOD waveform relative to
          the p-p amplitude. If larger than about 0.05 the data are bad.
        - ncrossings: number of zero crossings per period
        - peakwidth: width of the peak at the averaged amplitude relative to EOD period.
        - troughwidth: width of the trough at the averaged amplitude relative to EOD period.
        - leftpeak: time from positive zero crossing to peak relative to EOD period.
        - rightpeak: time from peak to negative zero crossing relative to EOD period.
        - lefttrough: time from negative zero crossing to trough relative to EOD period.
        - righttrough: time from trough to positive zero crossing relative to EOD period.
        - p-p-distance: time between peak and trough relative to EOD period.
        - reltroughampl: amplitude of trough relative to peak amplitude.
        - power: summed power of all harmonics of the extracted EOD waveform in decibel relative to one.
        - datapower: summed power of all harmonics of the original data in decibel relative to one. Only if `freq` is a list of harmonics.
        - thd: total harmonic distortion, i.e. square root of sum of amplitudes squared
          of harmonics relative to amplitude of fundamental.  
        - dbdiff: smoothness of power spectrum as standard deviation of differences in decibel power.
        - maxdb: maximum power of higher harmonics relative to peak power in decibel.

    spec_data: 2-D array of floats
        First size columns are from the spectrum of the extracted waveform.
        First column is the index of the harmonics, second column its frequency,
        third column its amplitude, fourth column its amplitude relative to the fundamental,
        fifth column is power of harmonics relative to fundamental in decibel,
        and sixth column the phase shift relative to the fundamental.
        If `freq` is a list of harmonics, a seventh column is added to `spec_data`
        that contains the powers of the harmonics from the original power spectrum of the
        raw data.
        Rows are the harmonics, first row is the fundamental frequency with index 0,
        relative amplitude of one, relative power of 0dB, and phase shift of zero.
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
    
    # fit fourier series:
    ampl = 0.5*(np.max(meod[:,1])-np.min(meod[:,1]))
    while n_harm > 1:
        params = [freq0]
        for i in range(1, n_harm+1):
            params.extend([ampl/i, 0.0])
        try:
            popt, pcov = curve_fit(fourier_series, meod[i0:i1,0], meod[i0:i1,1],
                                   params, maxfev=2000)
            break
        except (RuntimeError, TypeError):
            error_str += '%.1f Hz wave fish: fit of fourier series failed for %d harmonics. ' % (freq0, n_harm)
            n_harm //= 2
    for i in range(n_harm):
        # make all amplitudes positive:
        if popt[i*2+1] < 0.0:
            popt[i*2+1] *= -1.0
            popt[i*2+2] += np.pi
        # all phases in the range -pi to pi:
        popt[i*2+2] %= 2.0*np.pi
        if popt[i*2+2] > np.pi:
            popt[i*2+2] -= 2.0*np.pi
    # store fourier fit:
    meod[:,-1] = fourier_series(meod[:,0], *popt)
    # store fourier spectrum:
    if hasattr(freq, 'shape'):
        spec_data = np.zeros((n_harm, 7))
        powers = freq[:n_harm, 1]
        spec_data[:len(powers), 6] = powers
    else:
        spec_data = np.zeros((n_harm, 6))
    for i in range(n_harm):
        spec_data[i,:6] = [i, (i+1)*freq0, popt[i*2+1], popt[i*2+1]/popt[1],
                           decibel((popt[i*2+1]/popt[1])**2.0), popt[i*2+2]]
    # smoothness of power spectrum:
    db_powers = decibel(spec_data[:,2]**2)
    db_diff = np.std(np.diff(db_powers))
    # maximum relative power of higher harmonics:
    p_max = np.argmax(db_powers[:3])
    db_powers -= db_powers[p_max]
    if len(db_powers[p_max+n_harmonics:]) == 0:
        max_harmonics_power = -100.0
    else:
        max_harmonics_power = np.max(db_powers[p_max+n_harmonics:])
    # total harmonic distortion:
    thd = np.sqrt(np.sum(spec_data[1:,3]))

    # peak and trough amplitudes:
    ppampl = np.max(meod[i0:i1,1]) - np.min(meod[i0:i1,1])
    relptampl = np.min(meod[i0:i1,1])/np.max(meod[i0:i1,1])
    
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
        props['rmssem'] = rmssem
    props['ncrossings'] = ncrossings
    props['peakwidth'] = peak_width/period
    props['troughwidth'] = trough_width/period
    props['leftpeak'] = phase1/period
    props['rightpeak'] = phase2/period
    props['lefttrough'] = phase3/period
    props['righttrough'] = phase4/period
    props['p-p-distance'] = distance/period
    props['reltroughampl'] = np.abs(relptampl)
    pnh = power_n_harmonics if power_n_harmonics > 0 else n_harm
    props['power'] = decibel(np.sum(spec_data[:pnh,2]**2.0))
    if hasattr(freq, 'shape'):
        props['datapower'] = decibel(np.sum(freq[:pnh,1]))
    props['thd'] = thd
    props['dbdiff'] = db_diff
    props['maxdb'] = max_harmonics_power
    
    return meod, props, spec_data, error_str


def exp_decay(t, tau, ampl, offs):
    """
    Exponential decay function.

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


def analyze_pulse(eod, eod_times=None, min_pulse_win=0.001, peak_thresh_fac=0.01,
                  min_dist=50.0e-6, width_frac = 0.5, fit_frac = 0.5,
                  freq_resolution=1.0, flip_pulse='none',
                  ipi_cv_thresh=0.5, ipi_percentile=30.0):
    """
    Analyze the EOD waveform of a pulse fish.
    
    Parameters
    ----------
    eod: 2-D array
        The eod waveform. First column is time in seconds, second column the EOD waveform,
        third column, if present, is the standard error of the EOD waveform,
        Further columns are optional but not used.
    eod_times: 1-D array or None
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
    freq_resolution: float
        The frequency resolution of the power spectrum of the single pulse.
    flip_pulse: 'auto', 'none', 'flip'
        - 'auto' flip waveform such that the first large extremum is positive.
        - 'flip' flip waveform.
        - 'none' do not flip waveform.
    ipi_cv_thresh: float
        If the coefficient of variation of the interpulse intervals are smaller than this
        threshold, then the EOD frequency is computed as the inverse of the mean of
        all interpulse intervals. Otherwise only intervals smaller than a certain quantile
        are used.
    ipi_percentile: float
        When computing the EOD frequency, period, mean and standard deviation of
        interpulse intervals from a subset of the interpulse intervals,
        only intervals smaller than this percentile (between 0 and 100) are used.
    
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
        - EODf: the inverse of the median interval between `eod_times`, if provided.
        - period: the median interval between `eod_times`, if provided.
        - IPI-mean: the mean interval between `eod_times`, if provided.
        - IPI-std: the standard deviation of the intervals between `eod_times`, if provided.
        - max-amplitude: the amplitude of the largest positive peak (P1).
        - min-amplitude: the amplitude of the largest negative peak (P2).
        - p-p-amplitude: peak-to-peak amplitude of the EOD waveform.
        - rmssem: root-mean squared standard error mean of the averaged EOD waveform
          relative to the p-p amplitude.
        - tstart: time in seconds where the pulse starts,
          i.e. crosses the threshold for the first time.
        - tend: time in seconds where the pulse ends,
          i.e. crosses the threshold for the last time.
        - width: total width of the pulse in seconds (tend-tstart).
        - dist: distance between P2 and P1 in seconds.
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
        - n: number of pulses analyzed  (i.e. `len(eod_times)`), if provided.
        - times: the times of the detected EOD pulses (i.e. `eod_times`), if provided.

        Empty if waveform is not a pulse EOD.
    peaks: 2-D array
        For each peak and trough (rows) of the EOD waveform
        5 columns: the peak index (1 is P1, i.e. the largest positive peak),
        time relative to largest positive peak, amplitude,
        amplitude normalized to largest postive peak,
        and width of peak/trough at half height.
        Empty if waveform is not a pulse EOD.
    power: 2-D array
        The power spectrum of a single pulse. First column are the frequencies,
        second column the power in x^2/Hz such that the integral equals the variance.
        Empty if waveform is not a pulse EOD.
    """
    # storage:
    meod = np.zeros((eod.shape[0], eod.shape[1]+1))
    meod[:,:eod.shape[1]] = eod
    meod[:,-1] = np.nan
    toffs = 0
    
    # cut out stable estimate if standard deviation:
    if eod.shape[1] > 2 and np.max(meod[:,2]) > 3*np.min(meod[:,2]):
        n = len(meod)
        idx0 = np.argmax(np.abs(meod[n//10:9*n//10,1])) + n//10
        toffs += meod[idx0,0]
        meod[:,0] -= meod[idx0,0]
        # minimum in standard deviation:
        lstd_idx = np.argmin(meod[:idx0-5,2])
        rstd_idx = np.argmin(meod[idx0+5:,2]) + idx0
        # central, left, and right maximum of standard deviation:
        max_std = np.max(meod[lstd_idx:rstd_idx,2])
        l_std = np.max(meod[:len(meod)//4,2])
        r_std = np.max(meod[len(meod)*3//4:,2])
        lidx = 0
        ridx = len(meod)
        if l_std > max_std and lstd_idx > lidx:
            lidx = lstd_idx - np.argmax(meod[lstd_idx:0:-1,2] >= 0.25*l_std + 0.75*meod[lstd_idx,2])
        if r_std > max_std and rstd_idx < ridx:
            ridx = rstd_idx + np.argmax(meod[rstd_idx:,2] >= 0.25*r_std + 0.75*meod[rstd_idx,2])
        #plt.plot(meod[:,0], meod[:,1])
        #plt.plot(meod[:,0], meod[:,2], '-r')
        #plt.plot([meod[lidx,0], meod[lidx,0]], [-0.1, 0.1], '-k')
        #plt.plot([meod[ridx-1,0], meod[ridx-1,0]], [-0.1, 0.1], '-b')
        #plt.show()
        meod = meod[lidx:ridx,:]
    
    # subtract mean computed from the ends of the snippet:
    n = len(meod)//20 if len(meod) >= 20 else 1
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
    toffs += meod[max_idx,0]
    meod[:,0] -= meod[max_idx,0]

    # minimum threshold for peak detection:
    n = len(meod[:,1])//10 if len(meod) >= 20 else 2
    thl_max = np.max(meod[:n,1])
    thl_min = np.min(meod[:n,1])
    thr_max = np.max(meod[-n:,1])
    thr_min = np.min(meod[-n:,1])
    min_thresh = 2*np.max([thl_max, thr_max]) - np.min([thl_min, thr_min])
    if min_thresh > 0.5*(max_ampl + min_ampl):
        min_thresh = 0.5*(max_ampl + min_ampl)
        fit_frac = None
    # threshold for peak detection:
    threshold = max_ampl*peak_thresh_fac
    if threshold < min_thresh:
        threshold = min_thresh
    if threshold <= 0.0:
        return meod, {}, [], []
        
    # cut out relevant signal:
    lidx = np.argmax(np.abs(meod[:,1]) > threshold)
    ridx = len(meod) - 1 - np.argmax(np.abs(meod[::-1,1]) > threshold)
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
    dist = 0.0
    peaks = []

    # amplitude and variance:
    ppampl = max_ampl + min_ampl
    rmssem = np.sqrt(np.mean(meod[:,2]**2.0))/ppampl if eod.shape[1] > 2 else None
    
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
        if len(rmidx) > 0:
            peak_list = np.delete(peak_list, rmidx)
            width_list = np.delete(width_list, rmidx)
        if len(peak_list) == 0:
            return meod, {}, [], []
        # find P1:
        p1i = np.argmax(peak_list == max_idx)
        # truncate peaks to the left: XXX REALLY? WHY?
        offs = 0 if p1i <= 2 else p1i - 2
        peak_list = peak_list[offs:]
        width_list = width_list[offs:]
        p1i -= offs
        # store peaks:
        peaks = np.zeros((len(peak_list), 5))
        for i, pi in enumerate(peak_list):
            peaks[i,:] = [i+1-p1i, meod[pi,0], meod[pi,1], meod[pi,1]/max_ampl, width_list[i]]
        # P2 - P1 distance:
        dist = peaks[p1i+1,1] - peaks[p1i,1] if p1i+1 < len(peaks) else 0.0
        # fit exponential to last peak/trough:
        pi = peak_list[-1]
        # positive or negative decay:
        sign = 1.0
        if np.sum(meod[pi:,1] < -0.5*threshold) > np.sum(meod[pi:,1] > 0.5*threshold):
            sign = -1.0
        if sign*meod[pi,1] < 0.0:
            pi += np.argmax(sign*meod[pi:,1])
        pi_ampl = np.abs(meod[pi,1])
        n = len(meod[pi:])
        # no sufficiently large initial value:
        if fit_frac and pi_ampl*fit_frac <= 0.5*threshold:
            fit_frac = False
        # no sufficiently long decay:
        if n < 10:
            fit_frac = False
        # not decaying towards zero:
        max_line = pi_ampl - (pi_ampl-threshold)*np.arange(n)/n + 1e-8
        if np.any(np.abs(meod[pi+2:,1]) > max_line[2:]):
            fit_frac = False
        if fit_frac:
            thresh = meod[pi,1]*fit_frac
            inx = pi + np.argmax(sign*meod[pi:,1] < sign*thresh)
            thresh = meod[inx,1]*np.exp(-1.0)
            tau_inx = np.argmax(sign*meod[inx:,1] < sign*thresh)
            if tau_inx < 2:
                tau_inx = 2
            rridx = inx + 6*tau_inx
            if rridx > len(meod)-1:
                tau = None
            else:
                tau = meod[inx+tau_inx,0]-meod[inx,0]
                params = [tau, meod[inx,1]-meod[rridx,1], meod[rridx,1]]
                try:
                    popt, pcov = curve_fit(exp_decay, meod[inx:rridx,0]-meod[inx,0],
                                           meod[inx:rridx,1], params,
                                           bounds=([0.0, -np.inf, -np.inf], np.inf))
                except TypeError:
                    popt, pcov = curve_fit(exp_decay, meod[inx:rridx,0]-meod[inx,0],
                                           meod[inx:rridx,1], params)
                if popt[0] > 1.2*tau:
                    tau_inx = int(np.round(popt[0]/dt))
                    rridx = inx + 6*tau_inx
                    if rridx > len(meod)-1:
                        rridx = len(meod)-1
                    try:
                        popt, pcov = curve_fit(exp_decay, meod[inx:rridx,0]-meod[inx,0],
                                               meod[inx:rridx,1], popt,
                                               bounds=([0.0, -np.inf, -np.inf], np.inf))
                    except TypeError:
                        popt, pcov = curve_fit(exp_decay, meod[inx:rridx,0]-meod[inx,0],
                                               meod[inx:rridx,1], popt)
                tau = popt[0]
                meod[inx:rridx,-1] = exp_decay(meod[inx:rridx,0]-meod[inx,0], *popt)

    # power spectrum of single pulse:
    samplerate = 1.0/(meod[1,0]-meod[0,0])
    n_fft = nfft(samplerate, freq_resolution)

    n0 = max_idx
    n1 = len(meod)-max_idx
    n = 2*max(n0, n1)
    if n_fft < n:
        n_fft = next_power_of_two(n)
    data = np.zeros(n_fft)
    data[n_fft//2-n0:n_fft//2+n1] = meod[:,1]
    nr = n//4
    data[n_fft//2-n0:n_fft//2-n0+nr] *= np.arange(nr)/nr
    data[n_fft//2+n1-nr:n_fft//2+n1] *= np.arange(nr)[::-1]/nr
    freqs = np.fft.rfftfreq(n_fft, 1.0/samplerate)
    fourier = np.fft.rfft(data)/n_fft/freqs[1]
    power = np.abs(fourier)**2.0
    ppower = np.zeros((len(power), 2))

    ppower[:,0] = freqs
    ppower[:,1] = power
    maxpower = np.max(power)
    att5 = decibel(np.mean(power[freqs<5.0]), maxpower)
    att50 = decibel(np.mean(power[freqs<50.0]), maxpower)
    lowcutoff = freqs[decibel(power, maxpower) > 0.5*att5][0]

    # analyze pulse timing:
    if eod_times is not None:
        inter_pulse_intervals = np.diff(eod_times)
        ipi_cv = np.std(inter_pulse_intervals)/np.mean(inter_pulse_intervals)
        if ipi_cv < ipi_cv_thresh:
            period = np.median(inter_pulse_intervals)
            ipi_mean = np.mean(inter_pulse_intervals)
            ipi_std = np.std(inter_pulse_intervals)
        else:
            intervals = inter_pulse_intervals[inter_pulse_intervals <
                                    np.percentile(inter_pulse_intervals, ipi_percentile)]
            period = np.median(intervals)
            ipi_mean = np.mean(intervals)
            ipi_std = np.std(intervals)
    
    # store properties:
    props = {}
    props['type'] = 'pulse'
    if eod_times is not None:
        props['EODf'] = 1.0/period
        props['period'] = period
        props['IPI-mean'] = ipi_mean
        props['IPI-std'] = ipi_std
    props['max-amplitude'] = max_ampl
    props['min-amplitude'] = min_ampl
    props['p-p-amplitude'] = ppampl
    if rmssem:
        props['rmssem'] = rmssem
    props['tstart'] = t0
    props['tend'] = t1
    props['width'] = t1-t0
    props['dist'] = dist
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
    if eod_times is not None:
        props['n'] = len(eod_times)
        props['times'] = eod_times + toffs
    
    return meod, props, peaks, ppower


def adjust_eodf(eodf, temp, temp_adjust=25.0, q10=1.62):
    """ Adjust EOD frequencies to a standard temperature using Q10.

    Parameters
    ----------
    eodf: float or ndarray
        EOD frequencies.
    temp: float
        Temperature in degree celsisus at which EOD frequencies in `eodf` were measured.
    temp_adjust: float
        Standard temperature in degree celsisus  to which EOD frequencies are adjusted.
    q10: float
        Q10 value describing temperature dependence of EOD frequencies.
        The default of 1.62 is from Dunlap, Smith, Yetka (2000) Brain Behav Evol,
        measured for Apteronotus lepthorhynchus in the lab.

    Returns
    -------
    eodf_corrected: float or array
        EOD frequencies adjusted to `temp_adjust` using `q10`.
    """
    return eodf * q10 ** ((temp_adjust - temp) / 10.0)


def load_species_waveforms(species_file='none'):
    """ Load template EOD waveforms for species matching.
    
    Parameters
    ----------
    species_file: string
        Name of file containing species definitions. The content of this file is as follows:
        
        - Empty lines and line starting with a hash ('#') are skipped.
        - A line with the key-word 'wavefish' marks the beginning of the table for wave fish.
        - A line with the key-word 'pulsefish' marks the beginning of the table for wave fish.
        - Each line in a species table has three fields, separated by colons (':'):
        
          1. First field is an abbreviation of the species name.
          2. Second field is the filename of the recording containing the EOD waveform.
          3. The optional third field is the EOD frequency of the EOD waveform.

          The EOD frequency is used to normalize the time axis of a wave
          fish EOD to one EOD period. If it is not specified in the third field,
          it is taken from the corresponding *-wavespectrum-* file, if present.
          Otherwise the species is excluded and a warning is issued.

        Example file content:
        ``` plain
        Wavefish
        Aptero : F_91009L20-eodwaveform-0.csv : 823Hz
        Eigen  : C_91008L01-eodwaveform-0.csv

        Pulsefish
        Gymnotus : pulsefish/gymnotus.csv
        Brachy   : H_91009L11-eodwaveform-0.csv
        ```
    
    Returns
    -------
    wave_names: list of strings
        List of species names of wave-type fish.
    wave_eods: list of 2-D arrays
        List of EOD waveforms of wave-type fish corresponding to `wave_names`.
        First column of a waveform is time in seconds, second column is the EOD waveform.
        The waveforms contain exactly one EOD period.
    pulse_names: list of strings
        List of species names of pulse-type fish.
    pulse_eods: list of 2-D arrays
        List of EOD waveforms of pulse-type fish corresponding to `pulse_names`.
        First column of a waveform is time in seconds, second column is the EOD waveform.
    """
    if len(species_file) == 0 or species_file == 'none' or not os.path.isfile(species_file):
        return [], [], [], []
    wave_names = []
    wave_eods = []
    pulse_names = []
    pulse_eods = []
    fish_type = 'wave'
    with open(species_file, 'r') as sf:
        for line in sf:
            line = line.strip()
            if len(line) == 0 or line[0] == '#':
                continue
            if line.lower() == 'wavefish':
                fish_type = 'wave'
            elif line.lower() == 'pulsefish':
                fish_type = 'pulse'
            else:
                ls = [s.strip() for s in line.split(':')]
                if len(ls) <  2:
                    continue
                name = ls[0]
                waveform_file = ls[1]
                eod = TableData(waveform_file).array()
                eod[:,0] *= 0.001
                if fish_type == 'wave':
                    eodf = None
                    if len(ls) >  2:
                        eodf = float(ls[2].replace('Hz', '').strip())
                    else:
                        spectrum_file = waveform_file.replace('eodwaveform', 'wavespectrum')
                        try:
                            wave_spec = TableData(spectrum_file)
                            eodf = wave_spec[0, 1]
                        except FileNotFoundError:
                            pass
                    if eodf is None:
                        print('warning: unknown EOD frequency of %s. Skip.' % name)
                        continue
                    eod[:,0] *= eodf
                    wave_names.append(name)
                    wave_eods.append(eod[:,:2])
                elif fish_type == 'pulse':
                    pulse_names.append(name)
                    pulse_eods.append(eod[:,:2])
    return wave_names, wave_eods, pulse_names, pulse_eods


def wave_similarity(eod1, eod2, eod1f=1.0, eod2f=1.0):
    """Root-mean squared difference between two wave fish EODs.

    Compute the root-mean squared difference between two wave fish
    EODs over one period. The better sampled signal is down-sampled to
    the worse sampled one. Amplitude are normalized to peak-to-peak
    amplitude before computing rms difference.  Also compute the rms
    difference between the one EOD and the other one inverted in
    amplitude. The smaller of the two rms values is returned.

    Parameters
    ----------
    eod1: 2-D array
        Time and amplitude of reference EOD.
    eod2: 2-D array
        Time and amplitude of EOD that is to be compared to `eod1`.
    eod1f: float
        EOD frequency of `eod1` used to transform the time axis of `eod1`
        to multiples of the EOD period. If already normalized to EOD period,
        as for example by the `load_species_waveforms()` function, then
        set the EOD frequency to one (default).
    eod2f: float
        EOD frequency of `eod2` used to transform the time axis of `eod2`
        to multiples of the EOD period. If already normalized to EOD period,
        as for example by the `load_species_waveforms()` function, then
        set the EOD frequency to one (default).

    Returns
    -------
    rmse: float
        Root-mean-squared difference between the two EOD waveforms relative to
        their standard deviation over one period.

    """
    # copy:
    eod1 = np.array(eod1[:,:2])
    eod2 = np.array(eod2[:,:2])
    # scale to multiples of EOD period:
    eod1[:,0] *= eod1f
    eod2[:,0] *= eod2f
    # make eod1 the waveform with less samples per period:
    n1 = int(1.0/(eod1[1,0]-eod1[0,0]))
    n2 = int(1.0/(eod2[1,0]-eod2[0,0]))
    if n1 > n2:
        eod1, eod2 = eod2, eod1
        n1, n2 = n2, n1
    # one period around time zero:
    i0 = np.argmin(np.abs(eod1[:,0]))
    k0 = i0-n1//2
    if k0 < 0:
        k0 = 0
    k1 = k0 + n1 + 1
    if k1 >= len(eod1):
        k1 = len(eod1)
    # cut out one period from the reference EOD around maximum:
    i = k0 + np.argmax(eod1[k0:k1,1])
    k0 = i-n1//2
    if k0 < 0:
        k0 = 0
    k1 = k0 + n1 + 1
    if k1 >= len(eod1):
        k1 = len(eod1)
    eod1 = eod1[k0:k1,:]
    # normalize amplitudes of first EOD:
    eod1[:,1] -= np.min(eod1[:,1])
    eod1[:,1] /= np.max(eod1[:,1])
    sigma = np.std(eod1[:,1])
    # set time zero to maximum of second EOD:
    i0 = np.argmin(np.abs(eod2[:,0]))
    k0 = i0-n2//2
    if k0 < 0:
        k0 = 0
    k1 = k0 + n2 + 1
    if k1 >= len(eod2):
        k1 = len(eod2)
    i = k0 + np.argmax(eod2[k0:k1,1])
    eod2[:,0] -= eod2[i,0]
    # interpolate eod2 to the time base of eod1:
    eod2w = np.interp(eod1[:,0], eod2[:,0], eod2[:,1])
    # normalize amplitudes of second EOD:
    eod2w -= np.min(eod2w)
    eod2w /= np.max(eod2w)
    # root-mean-square difference:
    rmse1 = np.sqrt(np.mean((eod1[:,1] - eod2w)**2))/sigma
    # root-mean-square difference of the flipped signal:
    i = k0 + np.argmin(eod2[k0:k1,1])
    eod2[:,0] -= eod2[i,0]
    eod2w = np.interp(eod1[:,0], eod2[:,0], -eod2[:,1])
    eod2w -= np.min(eod2w)
    eod2w /= np.max(eod2w)
    rmse2 = np.sqrt(np.mean((eod1[:,1] - eod2w)**2))/sigma
    # take the smaller value:
    rmse = min(rmse1, rmse2)
    return rmse


def pulse_similarity(eod1, eod2, wfac=10.0):
    """Root-mean squared difference between two pulse fish EODs.

    Compute the root-mean squared difference between two pulse fish
    EODs over `wfac` times the distance between minimum and maximum of
    the wider EOD. The waveforms are normalized to their maxima prior
    to computing the rms difference.  Also compute the rms difference
    between the one EOD and the other one inverted in amplitude. The
    smaller of the two rms values is returned.

    Parameters
    ----------
    eod1: 2-D array
        Time and amplitude of reference EOD.
    eod2: 2-D array
        Time and amplitude of EOD that is to be compared to `eod1`.
    wfac: float
        Multiply the distance between minimum and maximum by this factor
        to get the window size over which to compute the rms difference.

    Returns
    -------
    rmse: float
        Root-mean-squared difference between the two EOD waveforms relative to
        their standard deviation over the analysis window.

    """
    # copy:
    eod1 = np.array(eod1[:,:2])
    eod2 = np.array(eod2[:,:2])
    # width of the pulses:
    imin1 = np.argmin(eod1[:,1])
    imax1 = np.argmax(eod1[:,1])
    w1 = np.abs(eod1[imax1,0]-eod1[imin1,0])
    imin2 = np.argmin(eod2[:,1])
    imax2 = np.argmax(eod2[:,1])
    w2 = np.abs(eod2[imax2,0]-eod2[imin2,0])
    w = wfac*max(w1, w2)
    # cut out signal from the reference EOD:
    n = int(w/(eod1[1,0]-eod1[0,0]))
    i0 = imax1-n//2
    if i0 < 0:
        i0 = 0
    i1 = imax1+n//2+1
    if i1 >= len(eod1):
        i1 = len(eod1)
    eod1[:,0] -= eod1[imax1,0]
    eod1 = eod1[i0:i1,:]
    # normalize amplitude of first EOD:
    eod1[:,1] /= np.max(eod1[:,1])
    sigma = np.std(eod1[:,1])
    # interpolate eod2 to the time base of eod1:
    eod2[:,0] -= eod2[imax2,0]
    eod2w = np.interp(eod1[:,0], eod2[:,0], eod2[:,1])
    # normalize amplitude of second EOD:
    eod2w /= np.max(eod2w)
    # root-mean-square difference:
    rmse1 = np.sqrt(np.mean((eod1[:,1] - eod2w)**2))/sigma
    # root-mean-square difference of the flipped signal:
    eod2[:,0] -= eod2[imin2,0]
    eod2w = np.interp(eod1[:,0], eod2[:,0], -eod2[:,1])
    eod2w /= np.max(eod2w)
    rmse2 = np.sqrt(np.mean((eod1[:,1] - eod2w)**2))/sigma
    # take the smaller value:
    rmse = min(rmse1, rmse2)
    return rmse


def clipped_fraction(data, samplerate, eod_times, mean_eod,
                     min_clip=-np.inf, max_clip=np.inf):
    """Compute fraction of clipped EOD waveform snippets.

    Cut out snippets at each `eod_times` based on time axis of
    `mean_eod`.  Check which fraction of snippets exceeds clipping
    amplitude `min_clip` and `max_clip`.

    Parameters
    ----------
    data: 1-D array of float
        The data to be analysed.
    samplerate: float
        Sampling rate of the data in Hertz.
    eod_times: 1-D array of float
        Array of EOD times in seconds.
    mean_eod: 2-D array with time, mean, sem, and fit.
        Averaged EOD waveform of wave fish. Only the time axis is used
        to set width of snippets.
    min_clip: float
        Minimum amplitude that is not clipped.
    max_clip: float
        Maximum amplitude that is not clipped.
    
    Returns
    -------
    clipped_frac: float
        Fraction of snippets that are clipped.

    """
    # snippets:
    idx0 = np.argmin(np.abs(mean_eod[:,0])) # index of time zero
    w0 = -idx0
    w1 = len(mean_eod[:,0]) - idx0
    eod_idx = np.round(eod_times * samplerate).astype(np.int)
    eod_snippets = snippets(data, eod_idx, w0, w1)
    # fraction of clipped snippets:
    clipped_frac = np.sum(np.any((eod_snippets > max_clip) |
                                 (eod_snippets < min_clip), axis=1))\
                   / len(eod_snippets)
    return clipped_frac


def wave_quality(props, harm_relampl=None, min_freq=0.0, max_freq=2000.0, max_clipped_frac=0.1,
                 max_crossings=4, max_rms_sem=0.0, max_rms_error=0.05,
                 min_power=-100.0, max_thd=0.0, max_db_diff=20.0, max_harmonics_db=-5.0,
                 max_relampl_harm1=0.0, max_relampl_harm2=0.0, max_relampl_harm3=0.0):
    """
    Assess the quality of an EOD waveform of a wave fish.
    
    Parameters
    ----------
    props: dict
        A dictionary with properties of the analyzed EOD waveform
        as returned by `analyze_wave()`.
    harm_relampl: 1-D array of floats or None
        Relative amplitude of at least the first 3 harmonics without the fundamental.
    min_freq: float
        Minimum EOD frequency (`props['EODf']`).
    max_freq: float
        Maximum EOD frequency (`props['EODf']`).
    max_clipped_frac: float
        If larger than zero, maximum allowed fraction of clipped data (`props['clipped']`).
    max_crossings: int
        If larger than zero, maximum number of zero crossings per EOD period
        (`props['ncrossings']`).
    max_rms_sem: float
        If larger than zero, maximum allowed standard error of the data relative
        to p-p amplitude (`props['rmssem']`).
    max_rms_error: float
        If larger than zero, maximum allowed root-mean-square error between EOD waveform and
        Fourier fit relative to p-p amplitude (`props['rmserror']`).
    min_power: float
        Minimum power of the EOD in dB (`props['power']`).
    max_thd: float
        If larger than zero, then maximum total harmonic distortion (`props['thd']`).
    max_db_diff: float
        If larger than zero, maximum standard deviation of differences between
        logarithmic powers of harmonics in decibel (`props['dbdiff']`).
        Low values enforce smoother power spectra.
    max_harmonics_db:
        Maximum power of higher harmonics relative to peak power in decibel (`props['maxdb']`).
    max_relampl_harm1: float
        If larger than zero, maximum allowed amplitude of first harmonic
        relative to fundamental.
    max_relampl_harm2: float
        If larger than zero, maximum allowed amplitude of second harmonic
        relative to fundamental.
    max_relampl_harm3: float
        If larger than zero, maximum allowed amplitude of third harmonic
        relative to fundamental.
                                       
    Returns
    -------
    remove: bool
        If True then this is most likely not an electric fish. Remove it from
        both the waveform properties and the list of EOD frequencies.
        If False, keep it in the list of EOD frequencies, but remove it from the
        waveform properties if `skip_reason` is not empty.
    skip_reason: string
        An empty string if the waveform is good, otherwise a string indicating the failure.
    msg: string
        A textual representation of the values tested.
    """
    remove = False
    msg = []
    skip_reason = []
    # EOD frequency:
    if 'EODf' in props:
        eodf = props['EODf']
        msg += ['EODf=%5.1fHz' % eodf]
        if eodf < min_freq or eodf > max_freq:
            remove = True
            skip_reason += ['invalid EODf=%5.1fHz (minimumFrequency=%5.1fHz, maximumFrequency=%5.1f))' %
                            (eodf, min_freq, max_freq)]
    # clipped fraction:
    if 'clipped' in props:
        clipped_frac = props['clipped']
        msg += ['clipped=%3.0f%%' % (100.0*clipped_frac)]
        if max_clipped_frac > 0 and clipped_frac >= max_clipped_frac:
            skip_reason += ['clipped=%3.0f%% (maximumClippedFraction=%3.0f%%)' %
                            (100.0*clipped_frac, 100.0*max_clipped_frac)]
    # too many zero crossings:
    if 'ncrossings' in props:
        ncrossings = props['ncrossings']
        msg += ['zero crossings=%d' % ncrossings]
        if max_crossings > 0 and ncrossings > max_crossings:
            skip_reason += ['too many zero crossings=%d (maximumCrossings=%d)' %
                            (ncrossings, max_crossings)]
    # noise:
    rms_sem = None
    if 'rmssem' in props:
        rms_sem = props['rmssem']
    if 'noise' in props:
        rms_sem = props['noise']
    if rms_sem is not None:
        msg += ['rms sem waveform=%6.2f%%' % (100.0*rms_sem)]
        if max_rms_sem > 0.0 and rms_sem >= max_rms_sem:
            skip_reason += ['noisy waveform s.e.m.=%6.2f%% (max %6.2f%%)' %
                            (100.0*rms_sem, 100.0*max_rms_sem)]
    # fit error:
    if 'rmserror' in props:
        rms_error = props['rmserror']
        msg += ['rmserror=%6.2f%%' % (100.0*rms_error)]
        if max_rms_error > 0.0 and rms_error >= max_rms_error:
            skip_reason += ['noisy rmserror=%6.2f%% (maximumVariance=%6.2f%%)' %
                            (100.0*rms_error, 100.0*max_rms_error)]
    # wave power:
    if 'power' in props:
        power = props['power']
        msg += ['power=%6.1fdB' % power]
        if power < min_power:
            skip_reason += ['small power=%6.1fdB (minimumPower=%6.1fdB)' %
                            (power, min_power)]
    # total harmonic distortion:
    if 'thd' in props:
        thd = props['thd']
        msg += ['thd=%5.1f%%' % (100.0*thd)]
        if max_thd > 0.0 and thd > max_thd:
            skip_reason += ['large THD=%5.1f%% (maxximumTotalHarmonicDistortion=%5.1f%%)' %
                            (100.0*thd, 100.0*max_thd)]
    # smoothness of spectrum:
    if 'dbdiff' in props:
        db_diff = props['dbdiff']
        msg += ['dBdiff=%5.1fdB' % db_diff]
        if max_db_diff > 0.0 and db_diff > max_db_diff:
            remove = True
            skip_reason += ['not smooth s.d. diff=%5.1fdB (maxximumPowerDifference=%5.1fdB)' %
                            (db_diff, max_db_diff)]
    # maximum power of higher harmonics:
    if 'maxdb' in props:
        max_harmonics = props['maxdb']
        msg += ['max harmonics=%5.1fdB' % max_harmonics]
        if max_harmonics > max_harmonics_db:
            remove = True
            skip_reason += ['maximum harmonics=%5.1fdB too strong (maximumHarmonicsPower=%5.1fdB)' %
                            (max_harmonics, max_harmonics_db)]
    # relative amplitude of harmonics:
    if harm_relampl is not None:
        for k, max_relampl in enumerate([max_relampl_harm1, max_relampl_harm2, max_relampl_harm3]):
            if k >= len(harm_relampl):
                break
            msg += ['ampl%d=%5.1f%%' % (k+1, 100.0*harm_relampl[k])]
            if max_relampl > 0.0 and k < len(harm_relampl) and harm_relampl[k] >= max_relampl:
                num_str = ['First', 'Second', 'Third']
                skip_reason += ['distorted ampl%d=%5.1f%% (maximum%sHarmonicAmplitude=%5.1f%%)' %
                                (k+1, 100.0*harm_relampl[k], num_str[k], 100.0*max_relampl)]
    return remove, ', '.join(skip_reason), ', '.join(msg)


def pulse_quality(props, max_clipped_frac=0.1, max_rms_sem=0.0):
    """
    Assess the quality of an EOD waveform of a pulse fish.
    
    Parameters
    ----------
    props: dict
        A dictionary with properties of the analyzed EOD waveform
        as returned by `analyze_pulse()`.
    max_clipped_frac: float
        Maximum allowed fraction of clipped data.
    max_rms_sem: float
        If not zero, maximum allowed standard error of the data relative to p-p amplitude.

    Returns
    -------
    skip_reason: string
        An empty string if the waveform is good, otherwise a string indicating the failure.
    msg: string
        A textual representation of the values tested.
    skipped_clipped: bool
        True if waveform was skipped because of clipping.
    """
    msg = []
    skip_reason = []
    skipped_clipped = False
    # clipped fraction:
    if 'clipped' in props:
        clipped_frac = props['clipped']
        msg += ['clipped=%3.0f%%' % (100.0*clipped_frac)]
        if clipped_frac >= max_clipped_frac:
            skip_reason += ['clipped=%3.0f%% (maximumClippedFraction=%3.0f%%)' %
                            (100.0*clipped_frac, 100.0*max_clipped_frac)]
            skipped_clipped = True
    # noise:
    rms_sem = None
    if 'rmssem' in props:
        rms_sem = props['rmssem']
    if 'noise' in props:
        rms_sem = props['noise']
    if rms_sem is not None:
        msg += ['rms sem waveform=%6.2f%%' % (100.0*rms_sem)]
        if max_rms_sem > 0.0 and rms_sem >= max_rms_sem:
            skip_reason += ['noisy waveform s.e.m.=%6.2f%% (maximumRMSNoise=%6.2f%%)' %
                            (100.0*rms_sem, 100.0*max_rms_sem)]
    return ', '.join(skip_reason), ', '.join(msg), skipped_clipped


def plot_eod_recording(ax, data, samplerate, width=0.1, unit=None, toffs=0.0,
                       kwargs={'lw': 2, 'color': 'red'}):
    """
    Plot a zoomed in range of the recorded trace.

    Parameters
    ----------
    ax: matplotlib axes
        Axes used for plotting.
    data: 1D ndarray
        Recorded data.
    samplerate: float
        Sampling rate of the data in Hertz.
    width: float
        Width of data segment to be plotted in seconds.
    unit: string
        Optional unit of the data used for y-label.
    toffs: float
        Time of first data value in seconds.
    kwargs: dict
        Arguments passed on to the plot command for the recorded trace.
    """
    widx2 = int(width*samplerate)//2
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


def plot_pulse_eods(ax, data, samplerate, zoom_window, width, eod_props, toffs=0.0,
                    colors=None, markers=None, marker_size=10,
                    legend_rows=8, **kwargs):
    """
    Mark pulse EODs in a plot of an EOD recording.

    Parameters
    ----------
    ax: matplotlib axes
        Axes used for plotting.
    data: 1D ndarray
        Recorded data (these are not plotted!).
    samplerate: float
        Sampling rate of the data in Hertz.
    eod_props: list of dictionaries
            Lists of EOD properties as returned by `analyze_pulse()` and `analyze_wave()`.
            From the entries with 'type' == 'pulse' the properties 'EODf' and 'times'
            are used. 'EODf' is the averaged EOD frequency, and 'times' is a list of
            detected EOD pulse times.
    toffs: float
        Time of first data value in seconds that will be added
        to the pulse times in `eod_props`.
    colors: list of colors or None
            If not None list of colors for plotting each cluster
    markers: list of markers or None
            If not None list of markers for plotting each cluster
    marker_size: float
            Size of markers used to mark the pulses.
    legend_rows: int
            Maximum number of rows to be used for the legend.
    kwargs: 
            Key word arguments for the legend of the plot.
    """
    k = 0
    for eod in eod_props:
        if eod['type'] != 'pulse':
            continue
        if 'times' not in eod:
            continue

        width = np.min([width, np.diff(zoom_window)])
        while len(eod['peaktimes'][(eod['peaktimes']>(zoom_window[1]-width)) & (eod['peaktimes']<(zoom_window[1]))]) == 0:
            width = width*2
            if zoom_window[1] - width < 0:
                width = width/2
                break  

        x = eod['peaktimes'] + toffs
        y = data[np.round(eod['peaktimes']*samplerate).astype(np.int)]
        color_kwargs = {}
        #if colors is not None:
        #    color_kwargs['color'] = colors[k%len(colors)]
        if marker_size is not None:
            color_kwargs['ms'] = marker_size
        label = '%6.1f Hz' % eod['EODf']
        if legend_rows > 5 and k >= legend_rows:
            label = None
        if markers is None:
            ax.plot(x, y, 'o', label=label, zorder=-1, **color_kwargs)
        else:
            ax.plot(x, y, linestyle='none', label=label,
                    marker=markers[k%len(markers)], mec=None, mew=0.0, zorder=-1, **color_kwargs)
        k += 1

    # legend:
    if k > 1:
        if legend_rows > 0:
            if legend_rows > 5:
                ncol = 1
            else:
                ncol = (len(idx)-1) // legend_rows + 1
            ax.legend(numpoints=1, ncol=ncol, **kwargs)
        else:
            ax.legend(numpoints=1, **kwargs)

    # reset window so at least one EOD of each cluster is visible    
    if len(zoom_window)>0:
        ax.set_xlim(max(toffs,toffs+zoom_window[1]-width), toffs+zoom_window[1])

        i0 = max(0,int((zoom_window[1]-width)*samplerate))
        i1 = int(zoom_window[1]*samplerate)

        ymin = np.min(data[i0:i1])
        ymax = np.max(data[i0:i1])
        dy = ymax - ymin
        ax.set_ylim(ymin-0.05*dy, ymax+0.05*dy)

        
def plot_eod_snippets(ax, data, samplerate, tmin, tmax, eod_times, n_snippets=10, flip=False,
                      kwargs={'zorder': -5, 'scaley': False, 'lw': 0.5, 'color': '#CCCCCC'}):
    """
    Plot a few EOD waveform snippets.

    Parameters
    ----------
    ax: matplotlib axes
        Axes used for plotting.
    data: 1D ndarray
        Recorded data from which the snippets are taken.
    samplerate: float
        Sampling rate of the data in Hertz.
    tmin: float
        Start time of each snippet.
    tmax: float
        End time of each snippet.
    eod_times: 1-D array
        EOD peak times from which a few are selected to be plotted.
    n_snippets: int
        Number of snippets to be plotted. If zero do not plot anything.
    flip: bool
        If True flip the snippets upside down.
    kwargs: dict
        Arguments passed on to the plot command for plotting the snippets.
    """
    if n_snippets <= 0:
        return
    i0 = int(tmin*samplerate)
    i1 = int(tmax*samplerate)
    time = 1000.0*np.arange(i0, i1)/samplerate
    step = len(eod_times)//n_snippets
    if step < 1:
        step = 1
    for t in eod_times[n_snippets//2::step]:
        idx = int(np.round(t*samplerate))
        if idx+i0 < 0 or idx+i1 >= len(data):
            continue
        snippet = data[idx+i0:idx+i1]
        if flip:
            snippet *= -1
        ax.plot(time, snippet - np.mean(snippet[:len(snippet)//4]), **kwargs)

        
def plot_eod_waveform(ax, eod_waveform, props, peaks=None, unit=None,
                      mkwargs={'zorder': 10, 'lw': 2, 'color': 'red'},
                      skwargs={'zorder': 5, 'color': '#CCCCCC'},
                      fkwargs={'zorder': 0, 'lw': 6, 'color': 'steelblue'},
                      zkwargs={'zorder': -10, 'lw': 1, 'color': '#AAAAAA'}):
    """
    Plot mean EOD, its standard error, and an optional fit to the EOD.

    Parameters
    ----------
    ax: matplotlib axes
        Axes used for plotting.
    eod_waveform: 2-D array
        EOD waveform. First column is time in seconds,
        second column the (mean) eod waveform. The optional third column is the
        standard error, and the optional fourth column is a fit on the waveform.
    props: dict
        A dictionary with properties of the analyzed EOD waveform as
        returned by `analyze_wave()` and `analyze_pulse()`.
    peaks: 2_D arrays or None
        List of peak properties (index, time, and amplitude) of a EOD pulse
        as returned by `analyze_pulse()`.
    unit: string
        Optional unit of the data used for y-label.
    mkwargs: dict
        Arguments passed on to the plot command for the mean EOD.
    skwargs: dict
        Arguments passed on to the fill_between command for the standard error of the EOD.
    fkwargs: dict
        Arguments passed on to the plot command for the fitted EOD.
    zkwargs: dict
        Arguments passed on to the plot command for the zero line.
    """
    ax.autoscale(True)
    time = 1000.0 * eod_waveform[:,0]
    # plot zero line:
    ax.plot([time[0], time[-1]], [0.0, 0.0], **zkwargs)
    # plot fit:
    if eod_waveform.shape[1] > 3:
        ax.plot(time, eod_waveform[:,3], **fkwargs)
    # plot waveform:
    mean_eod = eod_waveform[:,1]
    ax.plot(time, mean_eod, **mkwargs)
    # plot standard error:
    if eod_waveform.shape[1] > 2:
        std_eod = eod_waveform[:,2]
        if np.mean(std_eod)/(np.max(mean_eod) - np.min(mean_eod)) > 0.1:
            ax.autoscale(False)
        ax.fill_between(time, mean_eod + std_eod, mean_eod - std_eod, **skwargs)
    # ax height dimensions:
    pixely = np.abs(np.diff(ax.get_window_extent().get_points()[:,1]))[0]
    ymin, ymax = ax.get_ylim()
    unity = ymax - ymin
    dyu = np.abs(unity)/pixely
    font_size = plt.rcParams['font.size']*dyu
    # annotate fit:
    tau = props['tau'] if 'tau' in props else None
    ty = 0.0
    if tau is not None and eod_waveform.shape[1] > 3:
        if tau < 0.001:
            label = u'\u03c4=%.0f\u00b5s' % (1.e6*tau)
        else:
            label = u'\u03c4=%.2fms' % (1.e3*tau)
        inx = np.argmin(np.isnan(eod_waveform[:,3]))
        x = eod_waveform[inx,0] + 1.5*tau
        ty = 0.7*eod_waveform[inx,3]
        if np.abs(ty) < 0.5*font_size:
            ty = 0.5*font_size*np.sign(ty)
        va = 'bottom' if ty > 0.0 else 'top'
        ax.text(1000.0*x, ty, label, ha='left', va=va, zorder=20)
    # annotate peaks:
    if peaks is not None and len(peaks)>0:
        for p in peaks:
            ax.scatter(1000.0*p[1], p[2], s=80, clip_on=False, zorder=0, alpha=0.4,
                       c=mkwargs['color'], edgecolors=mkwargs['color'])
            label = u'P%d' % p[0]
            if p[0] != 1:
                if np.abs(p[1]) < 0.001:
                    ts = u'%.0f\u00b5s' % (1.0e6*p[1])
                elif np.abs(p[1]) < 0.01:
                    ts = u'%.2fms' % (1.0e3*p[1])
                else:
                    ts = u'%.3gms' % (1.0e3*p[1])
                if np.abs(p[3]) < 0.05:
                    ps = u'%.1f%%' % (100*p[3])
                else:
                    ps = u'%.0f%%' % (100*p[3])
                label += u'(%s @ %s)' % (ps, ts)
            va = 'bottom'
            dy = 0.4*font_size
            if p[0] % 2 == 0:
                va = 'top'
                dy = -dy
            if p[0] == 1:
                dy = 0.0
            """
            if p[2] <= np.min(peaks[:,2]):
                dy = -0.8*font_size
                va = 'bottom'
            """
            if p[2] + dy < ymin + 1.3*font_size:
                dy = ymin + 1.3*font_size - p[2]
            sign = np.sign(p[2])
            if p[0] == np.max(peaks[:,0]) and ty*p[2] > 0.0 and \
               sign*p[2]+dy < sign*ty+1.2*font_size:
                dy = ty + sign*1.2*font_size - p[2]
            dx = 0.05*time[-1]
            if p[1] >= 0.0:
                ax.text(1000.0*p[1]+dx, p[2]+dy, label, ha='left', va=va, zorder=20)
            else:
                ax.text(1000.0*p[1]-dx, p[2]+dy, label, ha='right', va=va, zorder=20)
    # annotate plot:
    if unit is None or len(unit) == 0 or unit == 'a.u.':
        unit = ''
    props['unit'] = unit
    label = 'p-p amplitude = {p-p-amplitude:.3g} {unit}\n'.format(**props)
    if 'n' in props:
        props['eods'] = 'EODs' if props['n'] > 1 else 'EOD'
        label += 'n = {n} {eods}\n'.format(**props)
    if props['flipped']:
        label += 'flipped\n'
    if -eod_waveform[0,0] < 0.6*eod_waveform[-1,0]:
        ax.text(0.97, 0.97, label, transform = ax.transAxes, va='top', ha='right')
    else:
        ax.text(0.03, 0.97, label, transform = ax.transAxes, va='top')
    # axis:                
    if props['type'] == 'wave':
        lim = 750.0/props['EODf']
        ax.set_xlim([-lim, +lim])
    else:
        ax.set_xlim(time[0], time[-1])
    ax.set_xlabel('Time [msec]')
    if unit:
        ax.set_ylabel('Amplitude [%s]' % unit)
    else:
        ax.set_ylabel('Amplitude')


def plot_wave_spectrum(axa, axp, spec, props, unit=None, color='b', lw=2, markersize=10):
    """Plot and annotate spectrum of wave EOD.

    Parameters
    ----------
    axa: matplotlib axes
        Axes for amplitude plot.
    axp: matplotlib axes
        Axes for phase plot.
    spec: 2-D array
        The amplitude spectrum of a single pulse as returned by `analyze_wave()`.
        First column is the index of the harmonics, second column its frequency,
        third column its amplitude, fourth column its amplitude relative to the fundamental,
        fifth column is power of harmonics relative to fundamental in decibel,
        and sixth column the phase shift relative to the fundamental.
    props: dict
        A dictionary with properties of the analyzed EOD waveform as
        returned by `analyze_wave()`.
    unit: string
        Optional unit of the data used for y-label.
    color:
        Color for line and points of spectrum.
    lw: float
        Linewidth for spectrum.
    markersize: float
        Size of points on spectrum.
    """
    n = 9 if len(spec) > 9 else len(spec)
    # amplitudes:
    markers, stemlines, _ = axa.stem(spec[:n,0]+1, spec[:n,2], basefmt='none')
    plt.setp(markers, color=color, markersize=markersize, clip_on=False)
    plt.setp(stemlines, color=color, lw=lw)
    axa.set_xlim(0.5, n+0.5)
    axa.set_ylim(bottom=0)
    axa.xaxis.set_major_locator(plt.MultipleLocator(1))
    axa.tick_params('x', direction='out')
    if unit:
        axa.set_ylabel('Amplitude [%s]' % unit)
    else:
        axa.set_ylabel('Amplitude')
    # phases:
    phases = spec[:,5]
    phases[phases<0.0] = phases[phases<0.0] + 2.0*np.pi
    markers, stemlines, _ = axp.stem(spec[:n,0]+1, phases[:n], basefmt='none')
    plt.setp(markers, color=color, markersize=markersize, clip_on=False)
    plt.setp(stemlines, color=color, lw=lw)
    axp.set_xlim(0.5, n+0.5)
    axp.xaxis.set_major_locator(plt.MultipleLocator(1))
    axp.tick_params('x', direction='out')
    axp.set_ylim(0, 2.0*np.pi)
    axp.set_yticks([0, np.pi, 2.0*np.pi])
    axp.set_yticklabels([u'0', u'\u03c0', u'2\u03c0'])
    axp.set_xlabel('Harmonics')
    axp.set_ylabel('Phase')


def plot_pulse_spectrum(ax, power, props, min_freq=1.0, max_freq=10000.0,
                        color='b', lw=3, markersize=80):
    """Plot and annotate spectrum of single pulse EOD.

    Parameters
    ----------
    ax: matplotlib axes
        Axes used for plotting.
    power: 2-D array
        The power spectrum of a single pulse as returned by `analyze_pulse()`.
        First column are the frequencies, second column the power.
    props: dict
        A dictionary with properties of the analyzed EOD waveform as
        returned by `analyze_pulse()`.
    min_freq: float
        Minimun frequency of the spectrum to be plotted (logscale!).
    max_freq: float
        Maximun frequency of the spectrum to be plotted (logscale!).
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
    if att < -5.0:
        ax.text(10.0, att+1.0, '%.0f dB' % att, ha='left', va='bottom', zorder=10)
    else:
        ax.text(10.0, att-1.0, '%.0f dB' % att, ha='left', va='top', zorder=10)
    box = mpatches.Rectangle((1,-60), 4, 60, linewidth=0, facecolor='#CCCCCC',
                             zorder=2)
    ax.add_patch(box)
    att = props['lowfreqattenuation5']
    if att < -5.0:
        ax.text(4.0, att+1.0, '%.0f dB' % att, ha='right', va='bottom', zorder=10)
    else:
        ax.text(4.0, att-1.0, '%.0f dB' % att, ha='right', va='top', zorder=10)
    lowcutoff = props['powerlowcutoff']
    if lowcutoff >= min_freq:
        ax.plot([lowcutoff, lowcutoff, 1.0], [-60.0, 0.5*att, 0.5*att], '#BBBBBB',
                zorder=3)
        ax.text(1.2*lowcutoff, 0.5*att-1.0, '%.0f Hz' % lowcutoff, ha='left', va='top', zorder=10)
    db = decibel(power[:,1])
    smax = np.nanmax(db)
    ax.plot(power[:,0], db - smax, color, lw=lw, zorder=4)
    peakfreq = props['peakfrequency']
    if peakfreq >= min_freq:
        ax.scatter([peakfreq], [0.0], c=color, edgecolors=color, s=markersize, alpha=0.4, zorder=5)
        ax.text(peakfreq*1.2, 1.0, '%.0f Hz' % peakfreq, va='bottom', zorder=10)
    ax.set_xlim(min_freq, max_freq)
    ax.set_xscale('log')
    ax.set_ylim(-60.0, 2.0)
    ax.set_xlabel('Frequency [Hz]')
    ax.set_ylabel('Power [dB]')


def save_eod_waveform(mean_eod, unit, idx, basename, **kwargs):
    """ Save mean EOD waveform to file.

    Parameters
    ----------
    mean_eod: 2D array of floats
        Averaged EOD waveform as returned by `eod_waveform()`, `analyze_wave()`,
        and `analyze_pulse()`.
    unit: string
        Unit of the waveform data.
    idx: int or None
        Index of fish.
    basename: string
        Path and basename of file.
        '-eodwaveform', the fish index, and a file extension are appended.
    kwargs:
        Arguments passed on to `TableData.write()`.

    Returns
    -------
    filename: string
        The path and full name of the written file.

    See Also
    --------
    load_eod_waveform()
    """
    td = TableData(mean_eod[:,:3]*[1000.0, 1.0, 1.0], ['time', 'mean', 'sem'],
                   ['ms', unit, unit], ['%.3f', '%.5f', '%.5f'])
    if mean_eod.shape[1] > 3:
        td.append('fit', unit, '%.5f', mean_eod[:,3])
    fp = basename + '-eodwaveform'
    if idx is not None:
        fp += '-%d' % idx
    file_name = td.write(fp, **kwargs)
    return file_name


def load_eod_waveform(file_path):
    """ Load EOD waveform from file.

    Parameters
    ----------
    file_path: string
        Path of the file to be loaded.

    Returns
    -------
    mean_eod: 2D array of floats
        Averaged EOD waveform: time in seconds, mean, standard deviation, fit.
    unit: string
        Unit of EOD waveform.

    Raises
    ------
    FileNotFoundError:
        If `file_path` does not exist.

    See Also
    --------
    save_eod_waveform()
    """
    data = TableData(file_path)
    mean_eod = data.array()
    mean_eod[:,0] *= 0.001
    return mean_eod, data.unit('mean')


def save_wave_eodfs(wave_eodfs, wave_indices, basename, **kwargs):
    """ Save frequencies of wave EODs to file.

    Parameters
    ----------
    wave_eodfs: list of 2D arrays
        Each item is a matrix with the frequencies and powers (columns) of the
        fundamental and harmonics (rows) as returned by `harmonics.harmonic_groups()`.
    wave_indices: array
        Indices identifying each fish or NaN.
        If None no index column is inserted.
    basename: string
        Path and basename of file.
        '-waveeodfs' and a file extension are appended.
    kwargs:
        Arguments passed on to `TableData.write()`.

    Returns
    -------
    filename: string
        The path and full name of the written file.

    See Also
    --------
    load_wave_eodfs()
    """
    eodfs = fundamental_freqs_and_power(wave_eodfs)
    td = TableData()
    if wave_indices is not None:
        td.append('index', '', '%d', [wi if wi >= 0 else np.nan for wi in wave_indices])
    td.append('EODf', 'Hz', '%7.2f', eodfs[:,0])
    td.append('datapower', 'dB', '%7.2f', eodfs[:,1])
    fp = basename + '-waveeodfs'
    file_name = td.write(fp, **kwargs)
    return file_name


def load_wave_eodfs(file_path):
    """ Load frequencies of wave EODs from file.

    Parameters
    ----------
    file_path: string
        Path of the file to be loaded.

    Returns
    -------
    eodfs: 2D array of floats
        Indices, EODfs, and power of wave type fish.
        Indices can contain NaNs.

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
    if not 'index' in data:
        indices = np.empty(data.rows())
        indices[:] = np.nan
        eodfs = np.column_stack((indices, eodfs))
    return eodfs

    
def save_wave_fish(eod_props, unit, basename, **kwargs):
    """ Save properties of wave EODs to file.

    Parameters
    ----------
    eod_props: list of dict
        Properties of EODs as returned by `analyze_wave()` and `analyze_pulse()`.
        Only properties of wave fish are saved.
    unit: string
        Unit of the waveform data.
    basename: string
        Path and basename of file.
        '-wavefish' and a file extension are appended.
    kwargs:
        Arguments passed on to `TableData.write()`.

    Returns
    -------
    filename: string or None
        The path and full name of the written file.
        None if no pulse fish are contained in eod_props and no file was written.

    See Also
    --------
    load_wave_fish()
    """
    wave_props = [p for p in eod_props if p['type'] == 'wave']
    if len(wave_props) == 0:
        return None
    td = TableData()
    td.append_section('waveform')
    td.append('index', '', '%d', wave_props, 'index')
    td.append('EODf', 'Hz', '%7.2f', wave_props, 'EODf')
    td.append('p-p-amplitude', unit, '%.5f', wave_props, 'p-p-amplitude')
    td.append('power', 'dB', '%7.2f', wave_props, 'power')
    if 'datapower' in wave_props[0]:
        td.append('datapower', 'dB', '%7.2f', wave_props, 'datapower')
    td.append('thd', '%', '%.2f', wave_props, 'thd', 100.0)
    td.append('dbdiff', 'dB', '%7.2f', wave_props, 'dbdiff')
    td.append('maxdb', 'dB', '%7.2f', wave_props, 'maxdb')
    if 'rmssem' in wave_props[0]:
        td.append('noise', '%', '%.1f', wave_props, 'rmssem', 100.0)
    td.append('rmserror', '%', '%.2f', wave_props, 'rmserror', 100.0)
    if 'clipped' in wave_props[0]:
        td.append('clipped', '%', '%.1f', wave_props, 'clipped', 100.0)
    td.append('flipped', '', '%d', wave_props, 'flipped')
    td.append('n', '', '%5d', wave_props, 'n')
    td.append_section('timing')
    td.append('ncrossings', '', '%d', wave_props, 'ncrossings')
    td.append('peakwidth', '%', '%.2f', wave_props, 'peakwidth', 100.0)
    td.append('troughwidth', '%', '%.2f', wave_props, 'troughwidth', 100.0)
    td.append('leftpeak', '%', '%.2f', wave_props, 'leftpeak', 100.0)
    td.append('rightpeak', '%', '%.2f', wave_props, 'rightpeak', 100.0)
    td.append('lefttrough', '%', '%.2f', wave_props, 'lefttrough', 100.0)
    td.append('righttrough', '%', '%.2f', wave_props, 'righttrough', 100.0)
    td.append('p-p-distance', '%', '%.2f', wave_props, 'p-p-distance', 100.0)
    td.append('reltroughampl', '%', '%.2f', wave_props, 'reltroughampl', 100.0)
    fp = basename + '-wavefish'
    file_name = td.write(fp, **kwargs)
    return file_name


def load_wave_fish(file_path):
    """ Load properties of wave EODs from file.

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
        props['type'] = 'wave'
        props['thd'] /= 100
        props['noise'] /= 100
        if 'rmssem' in props:
            props['rmssem'] /= 100
        props['rmserror'] /= 100
        if 'clipped' in props:
            props['clipped'] /= 100
        props['peakwidth'] /= 100
        props['troughwidth'] /= 100
        props['leftpeak'] /= 100
        props['rightpeak'] /= 100
        props['lefttrough'] /= 100
        props['righttrough'] /= 100
        props['p-p-distance'] /= 100
        props['reltroughampl'] /= 100
    return eod_props


def save_pulse_fish(eod_props, unit, basename, **kwargs):
    """ Save properties of pulse EODs to file.

    Parameters
    ----------
    eod_props: list of dict
        Properties of EODs as returned by `analyze_wave()` and `analyze_pulse()`.
        Only properties of pulse fish are saved.
    unit: string
        Unit of the waveform data.
    basename: string
        Path and basename of file.
        '-pulsefish' and a file extension are appended.
    kwargs:
        Arguments passed on to `TableData.write()`.

    Returns
    -------
    filename: string or None
        The path and full name of the written file.
        None if no pulse fish are contained in eod_props and no file was written.

    See Also
    --------
    load_pulse_fish()
    """
    pulse_props = [p for p in eod_props if p['type'] == 'pulse']
    if len(pulse_props) == 0:
        return None
    td = TableData()
    td.append_section('waveform')
    td.append('index', '', '%d', pulse_props, 'index')
    td.append('EODf', 'Hz', '%7.2f', pulse_props, 'EODf')
    td.append('period', 'ms', '%7.2f', pulse_props, 'period', 1000.0)
    td.append('max-ampl', unit, '%.5f', pulse_props, 'max-amplitude')
    td.append('min-ampl', unit, '%.5f', pulse_props, 'min-amplitude')
    td.append('p-p-amplitude', unit, '%.5f', pulse_props, 'p-p-amplitude')
    if 'rmssem' in pulse_props[0]:
        td.append('noise', '%', '%.1f', pulse_props, 'rmssem', 100.0)
    if 'clipped' in pulse_props[0]:
        td.append('clipped', '%', '%.1f', pulse_props, 'clipped', 100.0)
    td.append('flipped', '', '%d', pulse_props, 'flipped')
    td.append('tstart', 'ms', '%.3f', pulse_props, 'tstart', 1000.0)
    td.append('tend', 'ms', '%.3f', pulse_props, 'tend', 1000.0)
    td.append('width', 'ms', '%.3f', pulse_props, 'width', 1000.0)
    td.append('P2-P1-dist', 'ms', '%.3f', pulse_props, 'dist', 1000.0)
    td.append('tau', 'ms', '%.3f', pulse_props, 'tau', 1000.0)
    td.append('firstpeak', '', '%d', pulse_props, 'firstpeak')
    td.append('lastpeak', '', '%d', pulse_props, 'lastpeak')
    td.append('n', '', '%d', pulse_props, 'n')
    td.append_section('power spectrum')
    td.append('peakfreq', 'Hz', '%.2f', pulse_props, 'peakfrequency')
    td.append('peakpower', 'dB', '%.2f', pulse_props, 'peakpower')
    td.append('poweratt5', 'dB', '%.2f', pulse_props, 'lowfreqattenuation5')
    td.append('poweratt50', 'dB', '%.2f', pulse_props, 'lowfreqattenuation50')
    td.append('lowcutoff', 'Hz', '%.2f', pulse_props, 'powerlowcutoff')
    fp = basename + '-pulsefish'
    file_name = td.write(fp, **kwargs)
    return file_name


def load_pulse_fish(file_path):
    """ Load properties of pulse EODs from file.

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
        props['type'] = 'pulse'
        if 'rmssem' in props:
            props['rmssem'] /= 100
        if 'clipped' in props:
            props['clipped'] /= 100
        props['noise'] /= 100
        props['period'] /= 1000
        props['tstart'] /= 1000
        props['tend'] /= 1000
        props['width'] /= 1000
        props['P2-P1-dist'] /= 1000
        props['tau'] /= 1000
    return eod_props


def save_wave_spectrum(spec_data, unit, idx, basename, **kwargs):
    """ Save amplitude and phase spectrum of wave EOD to file.

    Parameters
    ----------
    spec_data: 2D array of floats
        Amplitude and phase spectrum of wave EOD as returned by `analyze_wave()`.
    unit: string
        Unit of the waveform data.
    idx: int or None
        Index of fish.
    basename: string
        Path and basename of file.
        '-wavespectrum', the fish index, and a file extension are appended.
    kwargs:
        Arguments passed on to `TableData.write()`.

    Returns
    -------
    filename: string
        The path and full name of the written file.

    See Also
    --------
    load_wave_spectrum()
    """
    td = TableData(spec_data[:,:6]*[1.0, 1.0, 1.0, 100.0, 1.0, 1.0],
                   ['harmonics', 'frequency', 'amplitude', 'relampl', 'relpower', 'phase'],
                   ['', 'Hz', unit, '%', 'dB', 'rad'],
                   ['%.0f', '%.2f', '%.5f', '%10.2f', '%6.2f', '%8.4f'])
    if spec_data.shape[1] > 6:
        td.append('datapower', '%s^2/Hz' % unit, '%11.4e', spec_data[:,6])
    fp = basename + '-wavespectrum'
    if idx is not None:
        fp += '-%d' % idx
    file_name = td.write(fp, **kwargs)
    return file_name


def load_wave_spectrum(file_path):
    """ Load amplitude and phase spectrum of wave EOD from file.

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
    spec[:,3] *= 0.01
    return spec, data.unit('amplitude')

                        
def save_pulse_spectrum(spec_data, unit, idx, basename, **kwargs):
    """ Save power spectrum of pulse EOD to file.

    Parameters
    ----------
    spec_data: 2D array of floats
        Power spectrum of single pulse as returned by `analyze_pulse()`.
    unit: string
        Unit of the waveform data.
    idx: int or None
        Index of fish.
    basename: string
        Path and basename of file.
        '-pulsespectrum', the fish index, and a file extension are appended.
    kwargs:
        Arguments passed on to `TableData.write()`.

    Returns
    -------
    filename: string
        The path and full name of the written file.

    See Also
    --------
    load_pulse_spectrum()
    """
    td = TableData(spec_data[:,:2], ['frequency', 'power'],
                   ['Hz', '%s^2/Hz' % unit], ['%.2f', '%.4e'])
    fp = basename + '-pulsespectrum'
    if idx is not None:
        fp += '-%d' % idx
    file_name = td.write(fp, **kwargs)
    return file_name


def load_pulse_spectrum(file_path):
    """ Load power spectrum of pulse EOD from file.

    Parameters
    ----------
    file_path: string
        Path of the file to be loaded.

    Returns
    -------
    spec: 2D array of floats
        Power spectrum of single pulse: frequency, power

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

                        
def save_pulse_peaks(peak_data, unit, idx, basename, **kwargs):
    """ Save peak properties of pulse EOD to file.

    Parameters
    ----------
    peak_data: 2D array of floats
        Properties of peaks and troughs of pulse EOD as returned by `analyze_pulse()`.
    unit: string
        Unit of the waveform data.
    idx: int or None
        Index of fish.
    basename: string
        Path and basename of file.
        '-pulsepeaks', the fish index, and a file extension are appended.
    kwargs:
        Arguments passed on to `TableData.write()`.

    Returns
    -------
    filename: string
        The path and full name of the written file.

    See Also
    --------
    load_pulse_peaks()
    """
    if len(peak_data) == 0:
        return None
    td = TableData(peak_data[:,:5]*[1.0, 1000.0, 1.0, 100.0, 1000.0],
                   ['P', 'time', 'amplitude', 'relampl', 'width'],
                   ['', 'ms', unit, '%', 'ms'],
                   ['%.0f', '%.3f', '%.5f', '%.2f', '%.3f'])
    fp = basename + '-pulsepeaks'
    if idx is not None:
        fp += '-%d' % idx
    file_name = td.write(fp, **kwargs)
    return file_name


def load_pulse_peaks(file_path):
    """ Load peak properties of pulse EOD from file.

    Parameters
    ----------
    file_path: string
        Path of the file to be loaded.

    Returns
    -------
    peak_data: 2D array of floats
        Properties of peaks and troughs of pulse EOD:
        P, time, amplitude, relampl, width
    unit: string
        Unit of peak amplitudes.

    Raises
    ------
    FileNotFoundError:
        If `file_path` does not exist.

    See Also
    --------
    save_pulse_peaks()
    """
    data = TableData(file_path)
    peaks = data.array()
    peaks[:,1] *= 0.001
    peaks[:,3] *= 0.01
    peaks[:,4] *= 0.001
    return peaks, data.unit('amplitude')

        
def add_eod_analysis_config(cfg, thresh_fac=0.8, percentile=0.1,
                            win_fac=2.0, min_win=0.01, max_eods=None,
                            min_sem=False, unfilter_cutoff=0.0,
                            flip_wave='none', flip_pulse='none',
                            n_harm=10, min_pulse_win=0.001,
                            peak_thresh_fac=0.01, min_dist=50.0e-6,
                            width_frac = 0.5, fit_frac = 0.5,
                            ipi_cv_thresh=0.5, ipi_percentile=30.0):
    """ Add all parameters needed for the eod analysis functions as
    a new section to a configuration.

    Parameters
    ----------
    cfg: ConfigFile
        The configuration.
        
    See `eod_waveform()`, `analyze_wave()`, and `analyze_pulse()` for details on
    the remaining arguments.
    """
    cfg.add_section('EOD analysis:')
    cfg.add('eodSnippetFac', win_fac, '', 'The duration of EOD snippets is the EOD period times this factor.')
    cfg.add('eodMinSnippet', min_win, 's', 'Minimum duration of cut out EOD snippets.')
    cfg.add('eodMaxEODs', max_eods or 0, '', 'The maximum number of EODs used to compute the average EOD. If 0 use all EODs.')
    cfg.add('eodMinSem', min_sem, '', 'Use minimum of s.e.m. to set maximum number of EODs used to compute the average EOD.')
    cfg.add('unfilterCutoff', unfilter_cutoff, 'Hz', 'If non-zero remove effect of high-pass filter with this cut-off frequency.')
    cfg.add('flipWaveEOD', flip_wave, '', 'Flip EOD of wave fish to make largest extremum positive (flip, none, or auto).')
    cfg.add('flipPulseEOD', flip_pulse, '', 'Flip EOD of pulse fish to make the first large peak positive (flip, none, or auto).')
    cfg.add('eodHarmonics', n_harm, '', 'Number of harmonics fitted to the EOD waveform.')
    cfg.add('eodMinPulseSnippet', min_pulse_win, 's', 'Minimum duration of cut out EOD snippets for a pulse fish.')
    cfg.add('eodPeakThresholdFactor', peak_thresh_fac, '', 'Threshold for detection of peaks in pulse EODs as a fraction of the pulse amplitude.')
    cfg.add('eodMinimumDistance', min_dist, 's', 'Minimum distance between peaks and troughs in a EOD pulse.')
    cfg.add('eodPulseWidthFraction', width_frac, '', 'The width of a pulse is measured at this fraction of the pulse height.')
    cfg.add('eodExponentialFitFraction', fit_frac, '', 'An exponential function is fitted on the tail of a pulse starting at this fraction of the height of the last peak.')
    cfg.add('ipiCVThresh', ipi_cv_thresh, '', 'If coefficient of variation of interpulse intervals is smaller than this threshold, then use all intervals for computing EOD frequency.')
    cfg.add('ipiPercentile', ipi_percentile, '%', 'Use only interpulse intervals shorter than this percentile to compute EOD frequency.')


def eod_waveform_args(cfg):
    """ Translates a configuration to the
    respective parameter names of the function `eod_waveform()`.
    
    The return value can then be passed as key-word arguments to this function.

    Parameters
    ----------
    cfg: ConfigFile
        The configuration.

    Returns
    -------
    a: dict
        Dictionary with names of arguments of the `eod_waveform()` function
        and their values as supplied by `cfg`.
    """
    a = cfg.map({'win_fac': 'eodSnippetFac',
                 'min_win': 'eodMinSnippet',
                 'max_eods': 'eodMaxEODs',
                 'min_sem': 'eodMinSem', 
                 'unfilter_cutoff': 'unfilterCutoff'})
    return a


def analyze_wave_args(cfg):
    """ Translates a configuration to the
    respective parameter names of the function `analyze_wave()`.
    
    The return value can then be passed as key-word arguments to this function.

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
    a = cfg.map({'n_harm': 'eodHarmonics',
                 'power_n_harmonics': 'powerNHarmonics',
                 'flip_wave': 'flipWaveEOD'})
    return a


def analyze_pulse_args(cfg):
    """ Translates a configuration to the
    respective parameter names of the function `analyze_pulse()`.
    
    The return value can then be passed as key-word arguments to this function.

    Parameters
    ----------
    cfg: ConfigFile
        The configuration.

    Returns
    -------
    a: dict
        Dictionary with names of arguments of the `analyze_pulse()` function
        and their values as supplied by `cfg`.
    """
    a = cfg.map({'min_pulse_win': 'eodMinPulseSnippet',
                 'peak_thresh_fac': 'eodPeakThresholdFactor',
                 'min_dist': 'eodMinimumDistance',
                 'width_frac': 'eodPulseWidthFraction',
                 'fit_frac': 'eodExponentialFitFraction',
                 'flip_pulse': 'flipPulseEOD',
                 'ipi_cv_thresh': 'ipiCVThresh',
                 'ipi_percentile': 'ipiPercentile'})
    return a


def add_species_config(cfg, species_file='none', wave_max_rms=0.2, pulse_max_rms=0.2):
    """Add parameters needed for assigning EOD waveforms to species.

    Parameters
    ----------
    cfg: ConfigFile
        The configuration.
    species_file: string
        File path to a file containing species names and corresponding file names
        of EOD waveform templates. If 'none', no species assignemnt is performed.
    wave_max_rms: float
        Maximum allowed rms difference (relative to standard deviation of EOD waveform)
        to an EOD waveform template for assignment to a wave fish species.
    pulse_max_rms: float
        Maximum allowed rms difference (relative to standard deviation of EOD waveform)
        to an EOD waveform template for assignment to a pulse fish species.
    """
    cfg.add_section('Species assignment:')
    cfg.add('speciesFile', species_file, '', 'File path to a file containing species names and corresponding file names of EOD waveform templates.')
    cfg.add('maximumWaveSpeciesRMS', wave_max_rms, '', 'Maximum allowed rms difference (relative to standard deviation of EOD waveform) to an EOD waveform template for assignment to a wave fish species.')
    cfg.add('maximumPulseSpeciesRMS', pulse_max_rms, '', 'Maximum allowed rms difference (relative to standard deviation of EOD waveform) to an EOD waveform template for assignment to a pulse fish species.')


def add_eod_quality_config(cfg, max_clipped_frac=0.1, max_variance=0.0,
                           max_rms_error=0.05, min_power=-100.0, max_thd=0.0,
                           max_crossings=4, max_relampl_harm1=0.0,
                           max_relampl_harm2=0.0, max_relampl_harm3=0.0):
    """Add parameters needed for assesing the quality of an EOD waveform.

    Parameters
    ----------
    cfg: ConfigFile
        The configuration.
        
    See `wave_quality()` and `pulse_quality()` for details on
    the remaining arguments.
    """
    cfg.add_section('Waveform selection:')
    cfg.add('maximumClippedFraction', max_clipped_frac, '', 'Take waveform of the fish with the highest power only if the fraction of clipped signals is below this value.')
    cfg.add('maximumVariance', max_variance, '', 'Skip waveform of fish if the standard error of the EOD waveform relative to the peak-to-peak amplitude is larger than this number. A value of zero allows any variance.')
    cfg.add('maximumRMSError', max_rms_error, '', 'Skip waveform of wave fish if the root-mean-squared error of the fit relative to the peak-to-peak amplitude is larger than this number.')
    cfg.add('minimumPower', min_power, 'dB', 'Skip waveform of wave fish if its power is smaller than this value.')
    cfg.add('maximumTotalHarmonicDistortion', max_thd, '', 'Skip waveform of wave fish if its total harmonic distortion is larger than this value. If set to zero do not check.')
    cfg.add('maximumCrossings', max_crossings, '', 'Maximum number of zero crossings per EOD period.')
    cfg.add('maximumFirstHarmonicAmplitude', max_relampl_harm1, '', 'Skip waveform of wave fish if the amplitude of the first harmonic is higher than this factor times the amplitude of the fundamental. If set to zero do not check.')
    cfg.add('maximumSecondHarmonicAmplitude', max_relampl_harm2, '', 'Skip waveform of wave fish if the ampltude of the second harmonic is higher than this factor times the amplitude of the fundamental. That is, the waveform appears to have twice the frequency than the fundamental. If set to zero do not check.')
    cfg.add('maximumThirdHarmonicAmplitude', max_relampl_harm3, '', 'Skip waveform of wave fish if the ampltude of the third harmonic is higher than this factor times the amplitude of the fundamental. If set to zero do not check.')


def wave_quality_args(cfg):
    """ Translates a configuration to the
    respective parameter names of the function `wave_quality()`.
    
    The return value can then be passed as key-word arguments to this function.

    Parameters
    ----------
    cfg: ConfigFile
        The configuration.

    Returns
    -------
    a: dict
        Dictionary with names of arguments of the `wave_quality()` function
        and their values as supplied by `cfg`.
    """
    a = cfg.map({'max_clipped_frac': 'maximumClippedFraction',
                 'max_rms_sem': 'maximumVariance',
                 'max_rms_error': 'maximumRMSError',
                 'min_power': 'minimumPower',
                 'max_crossings': 'maximumCrossings',
                 'min_freq': 'minimumFrequency',
                 'max_freq': 'maximumFrequency',
                 'max_thd': 'maximumTotalHarmonicDistortion',
                 'max_db_diff': 'maximumPowerDifference',
                 'max_harmonics_db': 'maximumHarmonicsPower',
                 'max_relampl_harm1': 'maximumFirstHarmonicAmplitude',
                 'max_relampl_harm2': 'maximumSecondHarmonicAmplitude',
                 'max_relampl_harm3': 'maximumThirdHarmonicAmplitude'})
    return a


def pulse_quality_args(cfg):
    """ Translates a configuration to the
    respective parameter names of the function `pulse_quality()`.
    
    The return value can then be passed as key-word arguments to this function.

    Parameters
    ----------
    cfg: ConfigFile
        The configuration.

    Returns
    -------
    a: dict
        Dictionary with names of arguments of the `pulse_quality()` function
        and their values as supplied by `cfg`.
    """
    a = cfg.map({'max_clipped_frac': 'maximumClippedFraction',
                 'max_rms_sem': 'maximumRMSNoise'})
    return a


if __name__ == '__main__':
    import sys
    import matplotlib.pyplot as plt
    from .fakefish import pulsefish_eod
    from .eventdetection import detect_peaks

    print('Analysis of EOD waveforms.')

    # data:
    samplerate = 44100.0
    data = pulsefish_eod('biphasic', 83.0, samplerate, 5.0, noise_std=0.05)
    unit = 'mV'
    eod_idx, _ = detect_peaks(data, 1.0)
    eod_times = eod_idx/samplerate

    # analyse EOD:
    mean_eod, eod_times = eod_waveform(data, samplerate, eod_times)
    mean_eod, props, peaks, power = analyze_pulse(mean_eod, eod_times)

    # plot:
    fig = plt.figure()
    ax = fig.add_subplot(1, 2, 1)
    plot_eod_waveform(ax, mean_eod, peaks, unit=unit)
    props['unit'] = unit
    label = '{type} fish\nEODf = {EODf:.1f} Hz\np-p amplitude = {p-p-amplitude:.3g} {unit}\nn = {n} EODs\n'.format(**props)
    if props['flipped']:
        label += 'flipped\n'
    ax.text(0.03, 0.97, label, transform = ax.transAxes, va='top')
    ax = fig.add_subplot(1, 2, 2)
    plot_pulse_spectrum(ax, power, props)
    plt.show()
