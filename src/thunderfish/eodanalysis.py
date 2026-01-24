"""
Analysis of EOD waveforms.

## EOD waveform analysis

- `eod_waveform()`: compute an averaged EOD waveform.
- `adjust_eodf()`: adjust EOD frequencies to a standard temperature.

## Analysis of wave-type EODs

- `waveeod_waveform()`: retrieve average EOD waveform via Fourier transform.
- `analyze_wave()`: analyze the EOD waveform of a wave fish.

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
- `save_pulse_spectrum()`: save spectrum of pulse EOD to file.
- `load_pulse_spectrum()`: load spectrum of pulse EOD from file.
- `save_pulse_phases()`: save phase properties of pulse EOD to file.
- `load_pulse_phases()`: load phase properties of pulse EOD from file.
- `save_pulse_gaussians()`: save Gaussian phase properties of pulse EOD to file.
- `load_pulse_gaussians()`: load Gaussian phase properties of pulse EOD from file.
- `save_pulse_times()`: save times of pulse EOD to file.
- `load_pulse_times()`: load times of pulse EOD from file.
- `parse_filename()`: parse components of an EOD analysis file name.
- `save_analysis(): save EOD analysis results to files.
- `load_analysis()`: load EOD analysis files.
- `load_recording()`: load recording.

## Fit functions

- `fourier_series()`: Fourier series of sine waves with amplitudes and phases.
- `exp_decay()`: exponential decay.
- `gaussian_sum()`: sum of Gaussians.
- `gaussian_sum_spectrum()`: energy spectrum of sum of Gaussians.
- `gaussian_sum_costs()`: cost function for fitting sum of Gaussians.

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
import io
import glob
import zipfile
import numpy as np
try:
    import matplotlib.pyplot as plt
except ImportError:
    pass

from pathlib import Path
from scipy.optimize import curve_fit, minimize
from numba import jit
from audioio import get_str
from thunderlab.eventdetection import percentile_threshold, detect_peaks, snippets, peak_width
from thunderlab.eventdetection import threshold_crossings, threshold_crossing_times, merge_events
from thunderlab.powerspectrum import next_power_of_two, nfft, decibel
from thunderlab.tabledata import TableData
from thunderlab.dataloader import DataLoader

from .harmonics import fundamental_freqs_and_power
from .fakefish import pulsefish_waveform
from .fakefish import normalize_pulsefish, export_pulsefish
from .fakefish import normalize_wavefish, export_wavefish


def eod_waveform(data, rate, eod_times, win_fac=2.0, min_win=0.01,
                 min_sem=False, max_eods=None, unfilter_cutoff=0.0):
    """Extract data snippets around each EOD, and compute a mean waveform with standard error.

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
    rate: float
        Sampling rate of the data in Hertz.
    eod_times: 1-D array of float
        Array of EOD times in seconds over which the waveform should be
        averaged.
        WARNING: The first data point must be at time zero!
    win_fac: float
        The snippet size is the EOD period times `win_fac`. The EOD period
        is determined as the minimum interval between EOD times.
    min_win: float
        The minimum size of the snippets in seconds.
    min_sem: bool
        If set, check for minimum in s.e.m. to set the maximum numbers
        of EODs to be used for computing the average waveform.
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
    eod_idx = np.round(eod_times*rate).astype(int)
        
    # window size:
    period = np.min(np.diff(eod_times))
    win = 0.5*win_fac*period
    if 2*win < min_win:
        win = 0.5*min_win
    win_inx = int(win*rate)

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
    mean_eod[:, 1] = np.mean(eod_snippets, axis=0)
    if len(eod_snippets) > 1:
        mean_eod[:, 2] = np.std(eod_snippets, axis=0, ddof=1)/np.sqrt(len(eod_snippets))
        
    # apply inverse filter:
    if unfilter_cutoff and unfilter_cutoff > 0.0:
        unfilter(mean_eod[:, 1], rate, unfilter_cutoff)
        
    # time axis:
    mean_eod[:, 0] = (np.arange(len(mean_eod)) - win_inx) / rate
    
    return mean_eod, eod_times


def adjust_eodf(eodf, temp, temp_adjust=25.0, q10=1.62):
    """Adjust EOD frequencies to a standard temperature using Q10.

    Parameters
    ----------
    eodf: float or ndarray
        EOD frequencies.
    temp: float
        Temperature in degree celsisus at which EOD frequencies in
        `eodf` were measured.
    temp_adjust: float
        Standard temperature in degree celsisus to which EOD
        frequencies are adjusted.
    q10: float
        Q10 value describing temperature dependence of EOD
        frequencies.  The default of 1.62 is from Dunlap, Smith, Yetka
        (2000) Brain Behav Evol, measured for Apteronotus
        lepthorhynchus in the lab.

    Returns
    -------
    eodf_corrected: float or array
        EOD frequencies adjusted to `temp_adjust` using `q10`.
    """
    return eodf*q10**((temp_adjust - temp) / 10.0)


def waveeod_waveform(data, rate, freq, win_fac=2.0, unfilter_cutoff=0.0):
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
    unfilter_cutoff: float
        If not zero, the cutoff frequency for an inverse high-pass filter
        applied to the mean EOD waveform.
    
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

    # apply inverse filter:
    if unfilter_cutoff and unfilter_cutoff > 0.0:
        unfilter(mean_eod[:, 1], rate, unfilter_cutoff)
    
    return mean_eod, eod_times


def unfilter(data, rate, cutoff):
    """Apply inverse high-pass filter on data.

    Assumes high-pass filter
    \\[ \\tau \\dot y = -y + \\tau \\dot x \\]
    has been applied on the original data \\(x\\), where
    \\(\\tau=(2\\pi f_{cutoff})^{-1}\\) is the time constant of the
    filter. To recover \\(x\\) the ODE
    \\[ \\tau \\dot x = y + \\tau \\dot y \\]
    is applied on the filtered data \\(y\\).

    Parameters
    ----------
    data: ndarray
        High-pass filtered original data.
    rate: float
        Sampling rate of `data` in Hertz.
    cutoff: float
        Cutoff frequency \\(f_{cutoff}\\) of the high-pass filter in Hertz.

    Returns
    -------
    data: ndarray
        Recovered original data.
    """
    tau = 0.5/np.pi/cutoff
    fac = tau*rate
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


def exp_decay(t, tau, ampl, offs):
    """Exponential decay function.

    x(t) = ampl*exp(-t/tau) + offs

    Parameters
    ----------
    t: float or array
        Time.
    tau: float
        Time constant of exponential decay.
    ampl: float
        Amplitude of exponential decay, i.e. initial value minus
        steady-state value.
    offs: float
        Steady-state value.
    
    Returns
    -------
    x: float or array
        The exponential decay evaluated at times `t`.

    """
    return offs + ampl*np.exp(-t/tau)


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

    
def analyze_pulse_phases(threshold, eod, ratetime=None,
                         min_dist=50.0e-6, width_frac=0.5):
    """Characterize all phases of a pulse-type EOD.
    
    Parameters
    ----------
    threshold: float
        Threshold for detecting peaks and troughs.
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
    l_idx = np.argmax(np.abs(eod) > threshold)
    r_idx = len(eod) - 1 - np.argmax(np.abs(eod[::-1]) > threshold)
    tstart = time[l_idx]
    tend = time[r_idx]
    # find peaks and troughs:
    peak_idx, trough_idx = detect_peaks(eod, threshold)
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
                if sign_fac > 0:
                    zero_times[i] = np.interp(0, snippet[zidx + 1:zidx - 1:-1],
                                              stimes[zidx + 1:zidx - 1:-1])
                else:
                    zero_times[i] = np.interp(0, snippet[zidx:zidx + 2],
                                              stimes[zidx:zidx + 2])
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
                       threshold=0.0, fit_frac=0.5):
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
        Minimum size of peaks of the pulse waveform.
    fit_frac: float or None
        An exponential is fitted to the tail of the last peak/trough
        starting where the waveform falls below this fraction of the
        peak's height (0-1).
        If None, do not attempt to fit.
    
    Returns
    -------
    tau: float
        Time constant of pulse tail in seconds.
    fit: 1-D array of float
        Time trace of the fit corresponding to `eod`.
    """
    if fit_frac is None:
        return None, None
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
    n = len(eod[pi:])
    # no sufficiently large initial value:
    if pi_ampl*fit_frac <= 0.5*threshold:
        fit_frac = False
    # no sufficiently long decay:
    if n < 10:
        fit_frac = False
    # not decaying towards zero:
    max_line = pi_ampl - (pi_ampl - threshold)*np.arange(n)/n + 1e-8
    if np.sum(np.abs(eod[pi + 2:]) > max_line[2:])/n > 0.05:
        fit_frac = False
    if not fit_frac:
        return None, None
    thresh = eod[pi]*fit_frac
    inx = pi + np.argmax(sign*eod[pi:] < sign*thresh)
    thresh = eod[inx]*np.exp(-1.0)
    tau_inx = np.argmax(sign*eod[inx:] < sign*thresh)
    if tau_inx < 2:
        tau_inx = 2
    rridx = inx + 6*tau_inx
    if rridx > len(eod) - 1:
        rridx = len(eod) - 1
        if rridx - inx < 3*tau_inx:
            return None, None
    tau = time[inx + tau_inx] - time[inx]
    params = [tau, eod[inx] - eod[rridx], eod[rridx]]
    try:
        popt, pcov = curve_fit(exp_decay, time[inx:rridx] - time[inx],
                               eod[inx:rridx], params,
                               bounds=([0.0, -np.inf, -np.inf], np.inf))
    except TypeError:
        popt, pcov = curve_fit(exp_decay, time[inx:rridx] - time[inx],
                               eod[inx:rridx], params)
    if popt[0] > 1.2*tau:
        tau_inx = int(np.round(popt[0]/dt))
        rridx = inx + 6*tau_inx
        if rridx > len(eod) - 1:
            rridx = len(eod) - 1
        try:
            popt, pcov = curve_fit(exp_decay, time[inx:rridx] - time[inx],
                                   eod[inx:rridx], popt,
                                   bounds=([0.0, -np.inf, -np.inf], np.inf))
        except TypeError:
            popt, pcov = curve_fit(exp_decay, time[inx:rridx] - time[inx],
                                   eod[inx:rridx], popt)
    tau = popt[0]
    fit = np.zeros(len(eod))
    fit[:] = np.nan
    fit[inx:rridx] = exp_decay(time[inx:rridx] - time[inx], *popt)
    return tau, fit

    
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
                  thresh_fac=0.01, min_dist=50.0e-6,
                  width_frac=0.5, fit_frac=0.5,
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
    thresh_fac: float
        Set the threshold for peak detection to the maximum pulse
        amplitude times this factor.
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
        - threshfac: Threshold for peak detection is at this factor times
          maximum EOD amplitude.
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

    # threshold for peak detection:
    threshold = pp_ampl*thresh_fac
    if threshold < 2*noise_thresh:
        threshold = 2*noise_thresh
        thresh_fac = threshold/pp_ampl
            
    # characterize EOD phases:
    tstart, tend, phases = analyze_pulse_phases(threshold, meod,
                                                min_dist=min_dist,
                                                width_frac=width_frac)
        
    # fit exponential to last phase:
    tau = None
    if len(phases) > 0 and len(phases['times']) > 1:
        if noise_thresh > fit_frac*max_ampl:
            fit_frac = None
        pi = np.argmin(np.abs(meod[:, 0] - phases['times'][-1]))
        tau, fit = analyze_pulse_tail(pi, meod, None,
                                      threshold, fit_frac)
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
        props['noise'] = np.mean(meod[:, 2])/pp_ampl
    props['rmserror'] = rmserror
    props['threshfac'] = thresh_fac
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
    if tau:
        props['tau'] = tau
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


def load_species_waveforms(species_file='none'):
    """Load template EOD waveforms for species matching.
    
    Parameters
    ----------
    species_file: string
        Name of file containing species definitions. The content of
        this file is as follows:
        
        - Empty lines and line starting with a hash ('#') are skipped.
        - A line with the key-word 'wavefish' marks the beginning of the
          table for wave fish.
        - A line with the key-word 'pulsefish' marks the beginning of the
          table for pulse fish.
        - Each line in a species table has three fields,
          separated by colons (':'):
        
          1. First field is an abbreviation of the species name.
          2. Second field is the filename of the recording containing the
             EOD waveform.
          3. The optional third field is the EOD frequency of the EOD waveform.

          The EOD frequency is used to normalize the time axis of a
          wave fish EOD to one EOD period. If it is not specified in
          the third field, it is taken from the corresponding
          *-wavespectrum-* file, if present.  Otherwise the species is
          excluded and a warning is issued.

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
        List of EOD waveforms of wave-type fish corresponding to
        `wave_names`.  First column of a waveform is time in seconds,
        second column is the EOD waveform.  The waveforms contain
        exactly one EOD period.
    pulse_names: list of strings
        List of species names of pulse-type fish.
    pulse_eods: list of 2-D arrays
        List of EOD waveforms of pulse-type fish corresponding to
        `pulse_names`.  First column of a waveform is time in seconds,
        second column is the EOD waveform.
    """
    if not Path(species_file).is_file():
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
                eod[:, 0] *= 0.001
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
                    eod[:, 0] *= eodf
                    wave_names.append(name)
                    wave_eods.append(eod[:, :2])
                elif fish_type == 'pulse':
                    pulse_names.append(name)
                    pulse_eods.append(eod[:, :2])
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
    eod1 = np.array(eod1[:, :2])
    eod2 = np.array(eod2[:, :2])
    # scale to multiples of EOD period:
    eod1[:, 0] *= eod1f
    eod2[:, 0] *= eod2f
    # make eod1 the waveform with less samples per period:
    n1 = int(1.0/(eod1[1,0]-eod1[0,0]))
    n2 = int(1.0/(eod2[1,0]-eod2[0,0]))
    if n1 > n2:
        eod1, eod2 = eod2, eod1
        n1, n2 = n2, n1
    # one period around time zero:
    i0 = np.argmin(np.abs(eod1[:, 0]))
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
    eod1[:, 1] -= np.min(eod1[:, 1])
    eod1[:, 1] /= np.max(eod1[:, 1])
    sigma = np.std(eod1[:, 1])
    # set time zero to maximum of second EOD:
    i0 = np.argmin(np.abs(eod2[:, 0]))
    k0 = i0-n2//2
    if k0 < 0:
        k0 = 0
    k1 = k0 + n2 + 1
    if k1 >= len(eod2):
        k1 = len(eod2)
    i = k0 + np.argmax(eod2[k0:k1,1])
    eod2[:, 0] -= eod2[i,0]
    # interpolate eod2 to the time base of eod1:
    eod2w = np.interp(eod1[:, 0], eod2[:, 0], eod2[:, 1])
    # normalize amplitudes of second EOD:
    eod2w -= np.min(eod2w)
    eod2w /= np.max(eod2w)
    # root-mean-square difference:
    rmse1 = np.sqrt(np.mean((eod1[:, 1] - eod2w)**2))/sigma
    # root-mean-square difference of the flipped signal:
    i = k0 + np.argmin(eod2[k0:k1,1])
    eod2[:, 0] -= eod2[i,0]
    eod2w = np.interp(eod1[:, 0], eod2[:, 0], -eod2[:, 1])
    eod2w -= np.min(eod2w)
    eod2w /= np.max(eod2w)
    rmse2 = np.sqrt(np.mean((eod1[:, 1] - eod2w)**2))/sigma
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
        Root-mean-squared difference between the two EOD waveforms
        relative to their standard deviation over the analysis window.
    """
    # copy:
    eod1 = np.array(eod1[:, :2])
    eod2 = np.array(eod2[:, :2])
    # width of the pulses:
    imin1 = np.argmin(eod1[:, 1])
    imax1 = np.argmax(eod1[:, 1])
    w1 = np.abs(eod1[imax1,0]-eod1[imin1,0])
    imin2 = np.argmin(eod2[:, 1])
    imax2 = np.argmax(eod2[:, 1])
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
    eod1[:, 0] -= eod1[imax1,0]
    eod1 = eod1[i0:i1,:]
    # normalize amplitude of first EOD:
    eod1[:, 1] /= np.max(eod1[:, 1])
    sigma = np.std(eod1[:, 1])
    # interpolate eod2 to the time base of eod1:
    eod2[:, 0] -= eod2[imax2,0]
    eod2w = np.interp(eod1[:, 0], eod2[:, 0], eod2[:, 1])
    # normalize amplitude of second EOD:
    eod2w /= np.max(eod2w)
    # root-mean-square difference:
    rmse1 = np.sqrt(np.mean((eod1[:, 1] - eod2w)**2))/sigma
    # root-mean-square difference of the flipped signal:
    eod2[:, 0] -= eod2[imin2,0]
    eod2w = np.interp(eod1[:, 0], eod2[:, 0], -eod2[:, 1])
    eod2w /= np.max(eod2w)
    rmse2 = np.sqrt(np.mean((eod1[:, 1] - eod2w)**2))/sigma
    # take the smaller value:
    rmse = min(rmse1, rmse2)
    return rmse


def clipped_fraction(data, rate, eod_times, mean_eod,
                     min_clip=-np.inf, max_clip=np.inf):
    """Compute fraction of clipped EOD waveform snippets.

    Cut out snippets at each `eod_times` based on time axis of
    `mean_eod`.  Check which fraction of snippets exceeds clipping
    amplitude `min_clip` and `max_clip`.

    Parameters
    ----------
    data: 1-D array of float
        The data to be analysed.
    rate: float
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
    idx0 = np.argmin(np.abs(mean_eod[:, 0])) # index of time zero
    w0 = -idx0
    w1 = len(mean_eod[:, 0]) - idx0
    eod_idx = np.round(eod_times*rate).astype(int)
    eod_snippets = snippets(data, eod_idx, w0, w1)
    # fraction of clipped snippets:
    clipped_frac = np.sum(np.any((eod_snippets > max_clip) |
                                 (eod_snippets < min_clip), axis=1))\
                   / len(eod_snippets)
    return clipped_frac


def wave_quality(props, harm_relampl=None, min_freq=0.0,
                 max_freq=2000.0, max_clipped_frac=0.1,
                 max_crossings=4, max_rms_sem=0.0, max_rms_error=0.05,
                 min_power=-100.0, max_thd=0.0, max_db_diff=20.0,
                 max_harmonics_db=-5.0, max_relampl_harm1=0.0,
                 max_relampl_harm2=0.0, max_relampl_harm3=0.0):
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
    max_crossings: int
        If larger than zero, maximum number of zero crossings per EOD period
        (`props['ncrossings']`).
    max_rms_sem: float
        If larger than zero, maximum allowed standard error of the
        data relative to p-p amplitude (`props['noise']`).
    max_rms_error: float
        If larger than zero, maximum allowed root-mean-square error
        between EOD waveform and Fourier fit relative to p-p amplitude
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
    max_harmonics_db:
        Maximum power of higher harmonics relative to peak power in
        decibel (`props['maxdb']`).
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
        If True then this is most likely not an electric fish. Remove
        it from both the waveform properties and the list of EOD
        frequencies.  If False, keep it in the list of EOD
        frequencies, but remove it from the waveform properties if
        `skip_reason` is not empty.
    skip_reason: string
        An empty string if the waveform is good, otherwise a string
        indicating the failure.
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
    """Assess the quality of an EOD waveform of a pulse fish.
    
    Parameters
    ----------
    props: dict
        A dictionary with properties of the analyzed EOD waveform
        as returned by `analyze_pulse()`.
    max_clipped_frac: float
        Maximum allowed fraction of clipped data.
    max_rms_sem: float
        If not zero, maximum allowed standard error of the data
        relative to p-p amplitude.

    Returns
    -------
    skip_reason: string
        An empty string if the waveform is good, otherwise a string
        indicating the failure.
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


def plot_eod_recording(ax, data, rate, unit=None, width=0.1,
                       toffs=0.0, rec_style=dict(lw=2, color='tab:red')):
    """Plot a zoomed in range of the recorded trace.

    Parameters
    ----------
    ax: matplotlib axes
        Axes used for plotting.
    data: 1D ndarray
        Recorded data to be plotted.
    rate: float
        Sampling rate of the data in Hertz.
    unit: string
        Optional unit of the data used for y-label.
    width: float
        Width of data segment to be plotted in seconds.
    toffs: float
        Time of first data value in seconds.
    rec_style: dict
        Arguments passed on to the plot command for the recorded trace.
    """
    widx2 = int(width*rate)//2
    i0 = len(data)//2 - widx2
    i0 = (i0//widx2)*widx2
    i1 = i0 + 2*widx2
    if i0 < 0:
        i0 = 0
    if i1 >= len(data):
        i1 = len(data)
    time = np.arange(len(data))/rate + toffs
    tunit = 'sec'
    if np.abs(time[i0]) < 1.0 and np.abs(time[i1]) < 1.0:
        time *= 1000.0
        tunit = 'ms'
    ax.plot(time, data, **rec_style)
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


def plot_pulse_eods(ax, data, rate, zoom_window, width, eod_props,
                    toffs=0.0, colors=None, markers=None, marker_size=10,
                    legend_rows=8, **kwargs):
    """Mark pulse EODs in a plot of an EOD recording.

    Parameters
    ----------
    ax: matplotlib axes
        Axes used for plotting.
    data: 1D ndarray
        Recorded data (these are not plotted!).
    rate: float
        Sampling rate of the data in Hertz.
    zoom_window: tuple of floats
       Start and end time of the data to be plotted in seconds.
    width: float
       Minimum width of the data to be plotted in seconds.
    eod_props: list of dictionaries
            Lists of EOD properties as returned by `analyze_pulse()`
            and `analyze_wave()`.  From the entries with 'type' ==
            'pulse' the properties 'EODf' and 'times' are used. 'EODf'
            is the averaged EOD frequency, and 'times' is a list of
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

        width = np.min([width, np.diff(zoom_window)[0]])
        while len(eod['peaktimes'][(eod['peaktimes']>(zoom_window[1]-width)) & (eod['peaktimes']<(zoom_window[1]))]) == 0:
            width *= 2
            if zoom_window[1] - width < 0:
                width = width/2
                break  

        x = eod['peaktimes'] + toffs
        pidx = np.round(eod['peaktimes']*rate).astype(int)
        y = data[pidx[(pidx>0)&(pidx<len(data))]]
        x = x[(pidx>0)&(pidx<len(data))]
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
                    marker=markers[k%len(markers)], mec=None, mew=0.0,
                    zorder=-1, **color_kwargs)
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

        i0 = max(0,int((zoom_window[1]-width)*rate))
        i1 = int(zoom_window[1]*rate)

        ymin = np.min(data[i0:i1])
        ymax = np.max(data[i0:i1])
        dy = ymax - ymin
        ax.set_ylim(ymin-0.05*dy, ymax+0.05*dy)

        
def plot_eod_snippets(ax, data, rate, tmin, tmax, eod_times,
                      n_snippets=10, flip=False, aoffs=0,
                      snippet_style=dict(scaley=False,
                                         lw=0.5, color='0.6')):
    """Plot a few EOD waveform snippets.

    Parameters
    ----------
    ax: matplotlib axes
        Axes used for plotting.
    data: 1D ndarray
        Recorded data from which the snippets are taken.
    rate: float
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
    aoffs: float
        Offset that was subtracted from the average EOD waveform.
    snippet_style: dict
        Arguments passed on to the plot command for plotting the snippets.
    """
    if data is None or n_snippets <= 0:
        return
    i0 = int(tmin*rate)
    i1 = int(tmax*rate)
    time = 1000.0*np.arange(i0, i1)/rate
    step = len(eod_times)//n_snippets
    if step < 1:
        step = 1
    for t in eod_times[n_snippets//2::step]:
        idx = int(np.round(t*rate))
        if idx + i0 < 0 or idx + i1 >= len(data):
            continue
        snippet = data[idx + i0:idx + i1] - aoffs
        if flip:
            snippet *= -1
        ax.plot(time, snippet - np.mean(snippet[:len(snippet)//4]),
                zorder=-5, **snippet_style)

        
def plot_eod_waveform(ax, eod_waveform, props, phases=None,
                      unit=None, wave_periods=2,
                      magnification_factor=20,
                      wave_style=dict(lw=1.5, color='tab:red'),
                      magnified_style=dict(lw=0.8, color='tab:red'),
                      positive_style=dict(facecolor='tab:green', alpha=0.2,
                                          edgecolor='none'),
                      negative_style=dict(facecolor='tab:blue', alpha=0.2,
                                          edgecolor='none'),
                      sem_style=dict(color='0.8'),
                      fit_style=dict(lw=1.5, color='tab:blue'),
                      phase_style=dict(zorder=0, ls='', marker='o', color='tab:red',
                                       markersize=6, mec='none', mew=0,
                                       alpha=0.4),
                      zerox_style=dict(zorder=50, ls='', marker='o', color='tab:red',
                                       markersize=5, mec='white', mew=1),
                      zero_style=dict(lw=0.5, color='0.7'),
                      fontsize='medium'):
    """Plot mean EOD, its standard error, and an optional fit to the EOD.

    Parameters
    ----------
    ax: matplotlib axes
        Axes used for plotting.
    eod_waveform: 2-D array
        EOD waveform. First column is time in seconds, second column
        the (mean) eod waveform. The optional third column is the
        standard error, the optional fourth column is a fit of the
        whole waveform, and the optional fourth column is a fit of 
        the tails of a pulse waveform.
    props: dict
        A dictionary with properties of the analyzed EOD waveform as
        returned by `analyze_wave()` and `analyze_pulse()`.
    phases: dict
        Dictionary with phase properties as returned by
        `analyze_pulse_phases()`, `analyze_pulse()`, and
        `load_pulse_phases()`.
    unit: string
        Optional unit of the data used for y-label.
    wave_periods: float
        How many periods of a wave EOD are shown.
    magnification_factor: float
        If larger than one, plot a magnified version of the EOD
        waveform magnified by this factor.
    wave_style: dict
        Arguments passed on to the plot command for the EOD waveform.
    magnified_style: dict
        Arguments passed on to the plot command for the magnified EOD waveform.
    positive_style: dict
        Arguments passed on to the fill_between command for coloring
        positive phases.
    negative_style: dict
        Arguments passed on to the fill_between command for coloring
        negative phases.
    sem_style: dict
        Arguments passed on to the fill_between command for the
        standard error of the EOD.
    fit_style: dict
        Arguments passed on to the plot command for the fitted EOD.
    phase_style: dict
        Arguments passed on to the plot command for marking EOD phases.
    zerox_style: dict
        Arguments passed on to the plot command for marking zero crossings.
    zero_style: dict
        Arguments passed on to the plot command for the zero line.
    fontsize: str or float or int
        Fontsize for annotation text.

    """
    ax.autoscale(True)
    time = 1000*eod_waveform[:, 0]
    eod = eod_waveform[:, 1]
    # time axis:                
    if props is not None and props['type'] == 'wave':
        period = 1000.0/props['EODf']
        xlim = 0.5*wave_periods*period
        xlim_l = -xlim
        xlim_r = +xlim
    elif props is not None and props['type'] == 'pulse':
        # width of maximum peak:
        meod = np.abs(eod)
        ip = np.argmax(meod)
        thresh = 0.5*meod[ip]
        i0 = ip - np.argmax(meod[ip::-1] < thresh)
        i1 = ip + np.argmax(meod[ip:] < thresh)
        w = 4*(time[i1] - time[i0])
        w = np.ceil(w/0.5)*0.5
        # make sure tstart and tend are included:
        if props is not None:
            if 'tstart' in props and 1000*props['tstart'] < -w:
                w = np.ceil(abs(1000*props['tstart'])/0.5)*0.5
            if 'tend' in props and 1000*props['tend'] > 2*w:
                w = np.ceil(0.5*abs(1000*props['tend'])/0.5)*0.5
        # make sure center of mass is included:
        cumul = np.cumsum(meod)/np.sum(meod)
        m = time[np.argmax(cumul > 0.5)]
        q1 = time[np.argmax(cumul > 0.25)]
        q3 = time[np.argmax(cumul > 0.75)]
        if m - 2*(m - q1) < -w:
            w = np.ceil(abs(m - 2*(m - q1))/0.5)*0.5
        if m + 2*(q3 - m) > 2*w:
            w = np.ceil(0.5*abs(m + 2*(q3 - m))/0.5)*0.5
        # set xaxis limits:
        xlim_l = -w
        xlim_r = 2*w
        xlim = (xlim_r - xlim_l)/2
    else:
        w = (time[-1] - time[0])/2
        w = np.floor(w/0.5)*0.5
        xlim_l = -w
        xlim_r = +w
        xlim = w
    ax.set_xlim(xlim_l, xlim_r)
    if xlim < 2:
        ax.xaxis.set_major_locator(plt.MultipleLocator(0.5))
    elif xlim < 4:
        ax.xaxis.set_major_locator(plt.MultipleLocator(1))
    elif xlim < 8:
        ax.xaxis.set_major_locator(plt.MultipleLocator(2))
    ax.set_xlabel('Time [msec]')
    # amplitude axis:                
    ylim = np.max(np.abs(eod[(time >= xlim_l) & (time <= xlim_r)])) 
    ax.set_ylim(-1.15*ylim, +1.15*ylim)
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
    texts = []
    # plot zero line:
    ax.axhline(0.0, zorder=-5, **zero_style)
    # plot areas:
    if phases is not None and len(phases) > 0:
        if positive_style is not None and len(positive_style) > 0:
            ax.fill_between(time, eod, 0, eod >= 0, zorder=4,
                            **positive_style)
        if negative_style is not None and len(negative_style) > 0:
                ax.fill_between(time, eod, 0, eod <= 0, zorder=4,
                                **negative_style)
    # plot fits:
    if eod_waveform.shape[1] > 3 and np.all(np.isfinite(eod_waveform[:, 3])):
        ax.plot(time, eod_waveform[:, 3], zorder=4, **fit_style)
    if eod_waveform.shape[1] > 4:
        fs = dict(**fit_style)
        if 'lw' in fs:
            fs['lw'] *= 2
        ax.plot(time, eod_waveform[:, 4], zorder=5, **fs)
    # plot waveform:
    ax.plot(time, eod, zorder=10, **wave_style)
    # plot standard error:
    if eod_waveform.shape[1] > 2:
        std_eod = eod_waveform[:, 2]
        if np.mean(std_eod)/(np.max(eod) - np.min(eod)) > 0.1:
            ax.autoscale_view(False)
            ax.autoscale(False)
        ax.fill_between(time, eod + std_eod, eod - std_eod,
                        zorder=-10, **sem_style)
    # plot magnified pulse waveform:
    magnification_mask = np.zeros(len(time), dtype=bool)
    if magnification_factor > 1 and phases is not None and len(phases) > 0:
        ax.autoscale_view(False)
        ax.autoscale(False)
        mag_thresh = 0.95*np.max(np.abs(eod))/magnification_factor
        i0 = np.argmax(np.abs(eod) > mag_thresh)
        if i0 > 0:
            left_eod = magnification_factor*eod[:i0]
            magnification_mask[:i0] = True
            ax.plot(time[:i0], left_eod, zorder=9, **magnified_style)
            if left_eod[-1] > 0:
                it = np.argmax(left_eod > 0.95*np.max(eod))
                if it < len(left_eod)//2:
                    it = len(left_eod) - 1
                ty = left_eod[it] if left_eod[it] < np.max(eod) else np.max(eod)
                ta = ax.text(time[it], ty, f'x{magnification_factor:.0f} ',
                             ha='right', va='top', fontsize=fontsize)
            else:
                it = np.argmax(left_eod < 0.95*np.min(eod))
                if it < len(left_eod)//2:
                    it = len(left_eod) - 1
                ty = left_eod[it] if left_eod[it] > np.min(eod) else np.min(eod)
                ta = ax.text(time[it], ty, f'x{magnification_factor:.0f} ',
                             ha='right', va='bottom', fontsize=fontsize)
            texts.append(ta)
        i1 = len(eod) - np.argmax(np.abs(eod[::-1]) > mag_thresh)
        right_eod = magnification_factor*eod[i1:]
        magnification_mask[i1:] = True
        ax.plot(time[i1:], right_eod, zorder=9, **magnified_style)
    # annotate fit:
    tau = None if props is None else props.get('tau', None)
    ty = 0.0
    if tau is not None and eod_waveform.shape[1] > 4:
        if tau < 0.001:
            label = f'\u03c4={1.e6*tau:.0f}\u00b5s'
        else:
            label = f'\u03c4={1.e3*tau:.2f}ms'
        inx = np.argmin(np.isnan(eod_waveform[:, 4]))
        x = eod_waveform[inx, 0] + 1.5*tau
        ty = 0.7*eod_waveform[inx, 4]
        if np.abs(ty) < 0.5*yfs:
            ty = 0.5*yfs*np.sign(ty)
        va = 'bottom' if ty > 0.0 else 'top'
        ta = ax.text(1000*x, ty, label, ha='left', va=va, zorder=20, fontsize=fontsize)
        texts.append(ta)
    if props is not None:
        # mark start and end:
        if 'tstart' in props:
            ax.axvline(1000*props['tstart'], 0.45, 0.55,
                       color='k', lw=0.5, zorder=80)
        if 'tend' in props:
            ax.axvline(1000*props['tend'], 0.45, 0.55,
                       color='k', lw=0.5, zorder=80)
        # mark cumulative:
        if 'median' in props:
            y = -1.07*ylim
            m = 1000*props['median']
            q1 = 1000*props['quartile1']
            q3 = 1000*props['quartile3']
            w = q3 - q1
            ax.plot([q1, q3], [y, y], 'gray', lw=4, zorder=80)
            ax.plot(m, y, 'o', color='white', ms=3, zorder=81)
            label = f'{w:.2f}ms' if w >= 1 else f'{1000*w:.0f}\u00b5s'
            ax.text(q3 + xfs, y, label,
                    va='center', zorder=100, fontsize=fontsize)
    # plot and annotate phases:
    if phases is not None and len(phases) > 0:
        upper_area_text = False
        lower_area_text = False
        # mark zero crossings:
        zeros = 1000*phases['zeros']
        ax.plot(zeros, np.zeros(len(zeros)), **zerox_style)
        # phase peaks and troughs:
        max_peak_idx = np.argmax(phases['amplitudes'])
        min_trough_idx = np.argmin(phases['amplitudes'])
        for i in range(len(phases['times'])):
            index = phases['indices'][i]
            ptime = 1000*phases['times'][i]
            pi = np.argmin(np.abs(time - ptime))
            mfac = magnification_factor if magnification_mask[pi] else 1
            pampl = mfac*phases['amplitudes'][i]
            relampl = phases['relamplitudes'][i]
            relarea = phases['relareas'][i]
            # classify phase:
            ampl_phase = phases['amplitudes'][i]
            ampl_left = phases['amplitudes'][i - 1] if i > 0 else 0
            ampl_right = phases['amplitudes'][i + 1] if i + 1 < len(phases['amplitudes']) else 0
            local_maximum = ampl_phase > ampl_left and ampl_phase > ampl_right
            if local_maximum:
                right_phase = (i >= max_peak_idx)
                min_max_phase = (i == max_peak_idx)
                local_phase = (ampl_phase < 0)
            else:
                right_phase = i >= min_trough_idx 
                min_max_phase = (i == min_trough_idx)
                local_phase = (ampl_phase > 0)
            sign = np.sign(pampl)
            # mark phase peak/trough:
            ax.plot(ptime, pampl, **phase_style)
            # text for phase label:
            label = f'P{index:.0f}'
            if index != 1 and not local_phase:
                if np.abs(ptime) < 1:
                    ts = f'{1000*ptime:.0f}\u00b5s'
                elif np.abs(ptime) < 10:
                    ts = f'{ptime:.2f}ms'
                else:
                    ts = f'{ptime:.3g}ms'
                if np.abs(relampl) < 0.05:
                    ps = f'{100*relampl:.1f}%'
                else:
                    ps = f'{100*relampl:.0f}%'
                label += f'({ps} @ {ts})'
            # position of phase label:
            ltime = ptime
            lampl = pampl
            valign = 'top' if sign < 0 else 'baseline'
            if local_phase or (min_max_phase and abs(pampl)/ylim < 0.8):
                halign = 'center'
                dx = 0
                dy = 0.6*yfs
            elif min_max_phase:
                halign = 'left' if right_phase else 'right'
                dx = xfs if right_phase else -xfs
                dy = 0
                if abs(relampl) > 0.85:
                    dx *= 2
                    dy = -1.5*yfs
            else:
                dx = 0
                dy = 0.8*yfs
                if right_phase:
                    halign = 'left'
                    if i > 0 and np.isfinite(phases['zeros'][i - 1]):
                        ltime = 1000*phases['zeros'][i - 1]
                    else:
                        dx = -2*xfs
                    #np.sum(phases['amplitudes'][i + 1:]*pampl > 0)
                else:
                    halign = 'right'
                    if np.isfinite(phases['zeros'][i]):
                        ltime = 1000*phases['zeros'][i]
                    else:
                        dx = 2*xfs
            if sign < 0:
                dy = -dy
            ta = ax.text(ltime + dx, lampl + dy, label,
                         ha=halign, va=valign, zorder=100, fontsize=fontsize)
            texts.append(ta)
            # area:
            if np.abs(relarea) < 0.01:
                continue
            elif np.abs(relarea) < 0.05:
                label = f'{100*relarea:.1f}%'
            else:
                label = f'{100*relarea:.0f}%'
            x = ptime
            if i > 0 and i < len(phases['times']) - 1:
                xl = 1000*phases['times'][i - 1]
                xr = 1000*phases['times'][i + 1]
                tsnippet = time[(time > xl) & (time < xr)]
                snippet = eod[(time > xl) & (time < xr)]
                tsnippet = tsnippet[np.sign(pampl)*snippet > 0]
                snippet = snippet[np.sign(pampl)*snippet > 0]
                x = np.sum(tsnippet*snippet)/np.sum(snippet)
            if abs(relampl) > 0.5:
                ax.text(x, sign*0.6*yfs, label,
                        rotation='vertical',
                        va='top' if sign < 0 else 'bottom',
                        ha='center', zorder=20, fontsize=fontsize)
            elif abs(relampl) > 0.25 and abs(relarea) > 0.19:
                ax.text(x, sign*0.6*yfs, label,
                        va='top' if sign < 0 else 'baseline',
                        ha='center', zorder=20, fontsize=fontsize)
            else:
                dx = 0.5*xfs if right_phase else -0.5*xfs
                ta = ax.text(ltime + dx, -sign*0.4*yfs, label,
                             va='baseline' if sign < 0 else 'top',
                             ha=halign, zorder=100, fontsize=fontsize)
                if -sign > 0 and not upper_area_text:
                    texts.append(ta)
                    upper_area_text = True
                if -sign < 0 and not lower_area_text:
                    texts.append(ta)
                    lower_area_text = True
        # arrange text vertically to avoid overlaps:
        ul_texts = []
        ur_texts = []
        ll_texts = []
        lr_texts = []
        for t in texts:
            x, y = t.get_position()
            if y > 0:
                if x >= phases['times'][max_peak_idx]:
                    ur_texts.append(t)
                else:
                    ul_texts.append(t)
            else:
                if x >= phases['times'][min_trough_idx]:
                    lr_texts.append(t)
                else:
                    ll_texts.append(t)
        for ts in [ul_texts, ur_texts, ll_texts, lr_texts]:
            if len(ts) > 1:
                ys = []
                for t in ts:
                    # alternative:
                    #renderer = ax.get_fig().canvas.renderer
                    #bbox = t.get_window_extent(renderer).transformed(ax.transData.inverted())
                    x, y = t.get_position()
                    ys.append(abs(y))
                idx = np.argsort(ys)
                x, y = ts[idx[0]].get_position()
                yp = abs(y)
                for i in idx[1:]:
                    t = ts[i]
                    x, y = t.get_position()
                    if abs(y) < abs(yp) + 2*yfs:
                        y = np.sign(y)*(abs(yp) + 2*yfs)
                        t.set_y(y)
                        #print('moved', t.get_text())
                    yp = y
    # annotate plot:
    if unit is None or len(unit) == 0 or unit == 'a.u.':
        unit = ''
    if props is not None:
        label = '' # f'p-p amplitude = {props["p-p-amplitude"]:.3g} {unit}\n'
        if 'n' in props:
            eods = 'EODs' if props['n'] > 1 else 'EOD'
            label += f'n = {props["n"]} {eods}\n'
        if 'flipped' in props and props['flipped']:
            label += 'flipped\n'
        if 'polaritybalance' in props:
            label += f'PB={100*props["polaritybalance"]:.0f} %\n'
        if -eod_waveform[0, 0] < 0.6*eod_waveform[-1, 0]:
            ax.text(0.97, 1, label, transform=ax.transAxes,
                    va='top', ha='right', zorder=20)
        else:
            ax.text(0.03, 1, label, transform=ax.transAxes,
                    va='top', zorder=20)


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
    plt.setp(markers, clip_on=False, **ampl_style)
    plt.setp(stemlines, **ampl_stem_style)
    axa.set_xlim(0.5, n + 0.5)
    axa.set_ylim(bottom=0)
    axa.xaxis.set_major_locator(plt.MultipleLocator(1))
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
    plt.setp(markers, clip_on=False, **phase_style)
    plt.setp(stemlines, **phase_stem_style)
    axp.set_xlim(0.5, n + 0.5)
    axp.xaxis.set_major_locator(plt.MultipleLocator(1))
    axp.tick_params('x', direction='out')
    axp.set_ylim(0, 2*np.pi)
    axp.set_yticks([0, np.pi, 2*np.pi])
    axp.set_yticklabels(['0', '\u03c0', '2\u03c0'])
    axp.set_xlabel('Harmonics')
    axp.set_ylabel('Phase')


def plot_pulse_spectrum(ax, energy, props, min_freq=1.0, max_freq=10000.0,
                        spec_style=dict(lw=3, color='tab:blue'),
                        analytic_style=dict(lw=4, color='tab:cyan'),
                        peak_style=dict(ls='', marker='o', markersize=6,
                                        color='tab:blue', mec='none', mew=0,
                                        alpha=0.4),
                        cutoff_style=dict(lw=1, ls='-', color='0.5'),
                        att5_color='0.8', att50_color='0.9',
                        fontsize='medium'):
    """Plot and annotate spectrum of single pulse EOD.

    Parameters
    ----------
    ax: matplotlib axes
        Axes used for plotting.
    energy: 2-D array
        The energy spectrum of a single pulse as returned by `analyze_pulse()`.
        First column are the frequencies, second column the energy.
        An optional third column is an analytically computed spectrum.
    props: dict
        A dictionary with properties of the analyzed EOD waveform as
        returned by `analyze_pulse()`.
    min_freq: float
        Minimun frequency of the spectrum to be plotted (logscale!).
    max_freq: float
        Maximun frequency of the spectrum to be plotted (logscale!).
    spec_style: dict
        Arguments passed on to the plot command for the energy spectrum
        computed from the data.
    analytic_style: dict
        Arguments passed on to the plot command for the energy spectrum
        that was analytically computed from the Gaussian fits
        (optional third column in `energy`).
    peak_style: dict
        Arguments passed on to the plot commands for marking the peak
        and trough frequency.
    cutoff_style: dict
        Arguments passed on to the plot command for the lines marking
        cutoff frequencies.
    att5_color: matplotlib color specification
        Color for the rectangular patch marking the first 5 Hz.
    att50_color: matplotlib color specification
        Color for the rectangular patch marking the first 50 Hz.
    fontsize: str or float or int
        Fontsize for annotation text.
    """
    ax.axvspan(1, 50, color=att50_color, zorder=10)
    att = props['energyatt50']
    if att < -10:
        ax.text(10, att + 1, f'{att:.0f}dB',
                ha='left', va='bottom', zorder=100, fontsize=fontsize)
    else:
        ax.text(10, att - 1, f'{att:.0f}dB',
                ha='left', va='top', zorder=100, fontsize=fontsize)
    ax.axvspan(1, 5, color=att5_color, zorder=20)
    att = props['energyatt5']
    if att < -10:
        ax.text(4, att + 1, f'{att:.0f}dB',
                ha='right', va='bottom', zorder=100, fontsize=fontsize)
    else:
        ax.text(4, att - 1, f'{att:.0f}dB',
                ha='right', va='top', zorder=100, fontsize=fontsize)
    lowcutoff = props['lowcutoff']
    if lowcutoff >= min_freq:
        ax.plot([lowcutoff, lowcutoff, 1], [-60, 0.5*att, 0.5*att],
                zorder=30, **cutoff_style)
        ax.text(1.2*lowcutoff, 0.5*att - 1, f'{lowcutoff:.0f}Hz',
                ha='left', va='top', zorder=100, fontsize=fontsize)
    highcutoff = props['highcutoff']
    ax.plot([highcutoff, highcutoff], [-60, -3], zorder=30, **cutoff_style)
    ax.text(1.2*highcutoff, -3, f'{highcutoff:.0f}Hz',
            ha='left', va='center', zorder=100, fontsize=fontsize)
    ref_energy = np.max(energy[:, 1])
    if energy.shape[1] > 2 and np.all(np.isfinite(energy[:, 2])) and len(analytic_style) > 0:
        db = decibel(energy[:, 2], ref_energy)
        ax.plot(energy[:, 0], db, zorder=45, **analytic_style)
    db = decibel(energy[:, 1], ref_energy)
    ax.plot(energy[:, 0], db, zorder=50, **spec_style)
    peakfreq = props['peakfreq']
    if peakfreq >= min_freq:
        ax.plot(peakfreq, 0, zorder=60, clip_on=False, **peak_style)
        ax.text(peakfreq*1.2, 1, f'{peakfreq:.0f}Hz',
                va='bottom', zorder=100, fontsize=fontsize)
    troughfreq = props['troughfreq']
    if troughfreq >= min_freq:
        troughenergy = decibel(props['troughenergy'], ref_energy)
        ax.plot(troughfreq, troughenergy, zorder=60, **peak_style)
        ax.text(troughfreq, troughenergy - 3,
                f'{troughenergy:.0f}dB @ {troughfreq:.0f}Hz',
                ha='center', va='top', zorder=100, fontsize=fontsize)
    ax.set_xlim(min_freq, max_freq)
    ax.set_xscale('log')
    ax.set_ylim(-60, 2)
    ax.set_xlabel('Frequency [Hz]')
    ax.set_ylabel('Energy [dB]')

    
def save_eod_waveform(mean_eod, unit, idx, basename, **kwargs):
    """Save mean EOD waveform to file.

    Parameters
    ----------
    mean_eod: 2D array of floats
        Averaged EOD waveform as returned by `eod_waveform()`,
        `analyze_wave()`, and `analyze_pulse()`.
    unit: string
        Unit of the waveform data.
    idx: int or None
        Index of fish.
    basename: string or stream
        If string, path and basename of file.
        If `basename` does not have an extension,
        '-eodwaveform', the fish index, and a file extension are appended.
        If stream, write EOD waveform data into this stream.
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
    load_eod_waveform()
    """
    td = TableData(mean_eod[:, :3]*[1000.0, 1.0, 1.0],
                   ['time', 'mean', 'sem'],
                   ['ms', unit, unit],
                   ['%.3f', '%.6g', '%.6g'])
    if mean_eod.shape[1] > 3:
        td.append('fit', unit, '%.5f', value=mean_eod[:, 3])
    if mean_eod.shape[1] > 4:
        td.append('tailfit', unit, '%.5f', value=mean_eod[:, 4])
    fp = ''
    ext = Path(basename).suffix if not hasattr(basename, 'write') else ''
    if not ext:
        fp = '-eodwaveform'
        if idx is not None:
            fp += f'-{idx}'
    return td.write_file_stream(basename, fp, **kwargs)


def load_eod_waveform(file_path):
    """Load EOD waveform from file.

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
    mean_eod[:, 0] *= 0.001
    return mean_eod, data.unit('mean')


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
    td.append('threshfac', '%', '%.2f', value=pulse_props, fac=100)
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
        if 'clipped' in props:
            props['clipped'] /= 100
        props['period'] /= 1000
        props['noise'] /= 100
        props['threshfac'] /= 100
        props['tstart'] /= 1000
        props['tend'] /= 1000
        props['p-p-dist'] /= 1000
        props['width'] /= 1000
        props['tau'] /= 1000
        props['rmserror'] /= 100
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


file_types = ['waveeodfs', 'wavefish', 'pulsefish', 'eodwaveform',
              'wavespectrum', 'pulsephases', 'pulsegaussians', 'pulsespectrum', 'pulsetimes']
"""List of all file types generated and supported by the `save_*` and `load_*` functions."""


def parse_filename(file_path):
    """Parse components of an EOD analysis file name.

    Analysis files generated by the `eodanalysis` module are named
    according to
    ```plain
    PATH/RECORDING-CHANNEL-TIME-FTYPE-N.EXT
    ```

    Parameters
    ----------
    file_path: string
        Path of the file to be parsed.

    Returns
    -------
    recording: string
        Path and basename of the recording, i.e. 'PATH/RECORDING'.
        A leading './' is removed.
    base_path: string
        Path and basename of the analysis results,
        i.e. 'PATH/RECORDING-CHANNEL-TIME'. A leading './' is removed.
    channel: int
        Channel of the recording
        ('CHANNEL' component of the file name if present).
        -1 if not present in `file_path`.
    time: float
        Start time of analysis window in seconds
        ('TIME' component of the file name if present).
        `None` if not present in `file_path`.
    ftype: string
        Type of analysis file (e.g. 'wavespectrum', 'pulsephases', etc.),
        ('FTYPE' component of the file name if present).
        See `file_types` for a list of all supported file types.
        Empty string if not present in `file_path`.
    index: int
        Index of the EOD.
        ('N' component of the file name if present).
        -1 if not present in `file_path`.
    ext: string
        File extension *without* leading period
        ('EXT' component of the file name).

    """
    file_path = Path(file_path)
    ext = file_path.suffix
    ext = ext[1:]
    parts = file_path.stem.split('-')
    index = -1
    if len(parts) > 0 and parts[-1].isdigit():
        index = int(parts[-1])
        parts = parts[:-1]
    ftype = ''
    if len(parts) > 0:
        ftype = parts[-1]
        parts = parts[:-1]
    base_path = file_path.parent / '-'.join(parts)
    time = None
    if len(parts) > 0 and len(parts[-1]) > 0 and \
       parts[-1][0] == 't' and parts[-1][-1] == 's' and \
       parts[-1][1:-1].isdigit():
        time = float(parts[-1][1:-1])
        parts = parts[:-1]
    channel = -1
    if len(parts) > 0 and len(parts[-1]) > 0 and \
       parts[-1][0] == 'c' and parts[-1][1:].isdigit():
        channel = int(parts[-1][1:])
        parts = parts[:-1]
    recording = '-'.join(parts)
    return recording, base_path, channel, time, ftype, index, ext

            
def save_analysis(output_basename, zip_file, eod_props, mean_eods, spec_data,
                  phase_data, pulse_data, wave_eodfs, wave_indices, unit,
                  verbose, **kwargs):
    """Save EOD analysis results to files.

    Parameters
    ----------
    output_basename: string
        Path and basename of files to be saved.
    zip_file: bool
        If `True`, write all analysis results into a zip archive.
    eod_props: list of dict
        Properties of EODs as returned by `analyze_wave()` and
        `analyze_pulse()`.
    mean_eods: list of 2D array of floats
        Averaged EOD waveforms as returned by `eod_waveform()`,
        `analyze_wave()`, and `analyze_pulse()`.
    spec_data: list of 2D array of floats
        Energy spectra of single pulses as returned by
        `analyze_pulse()`.
    phase_data: list of dict
        Properties of phases of pulse EODs as returned by
        `analyze_pulse()` and `analyze_pulse_phases()`.
    pulse_data: list of dict
        For each pulse fish a dictionary with phase times, amplitudes and standard
        deviations of Gaussians fitted to the pulse waveform.
    wave_eodfs: list of 2D array of float
        Each item is a matrix with the frequencies and powers
        (columns) of the fundamental and harmonics (rows) as returned
        by `harmonics.harmonic_groups()`.
    wave_indices: array of int
        Indices identifying each fish in `wave_eodfs` or NaN.
    unit: string
        Unit of the waveform data.
    verbose: int
        Verbosity level.
    kwargs:
        Arguments passed on to `TableData.write()`.
    """
    def write_file_zip(zf, save_func, output, *args, **kwargs):
        if zf is None:
            fp = save_func(*args, basename=output, **kwargs)
            if verbose > 0 and fp is not None:
                print('wrote file', fp)
        else:
            with io.StringIO() as df:
                fp = save_func(*args, basename=df, **kwargs)
                if fp is not None:
                    fp = Path(output + str(fp))
                    zf.writestr(fp.name, df.getvalue())
                    if verbose > 0:
                        print('zipped file', fp.name)

    
    if 'table_format' in kwargs and kwargs['table_format'] == 'py':
        with open(output_basename + '.py', 'w') as f:
            name = Path(output_basename).stem
            for k in range(len((spec_data))):
                if len(pulse_data[k]) > 0:
                    fish = normalize_pulsefish(pulse_data[k])
                    export_pulsefish(fish, f'{name}-{k}_phases', f)
                else:
                    sdata = spec_data[k]
                    if len(sdata) > 0 and sdata.shape[1] > 2:
                        fish = dict(amplitudes=sdata[:, 3], phases=sdata[:, 5])
                        fish = normalize_wavefish(fish)
                        export_wavefish(fish, f'{name}-{k}_harmonics', f)
    else:
        zf = None
        if zip_file:
            zf = zipfile.ZipFile(output_basename + '.zip', 'w')
        # all wave fish in wave_eodfs:
        if len(wave_eodfs) > 0:
            write_file_zip(zf, save_wave_eodfs, output_basename,
                           wave_eodfs, wave_indices, **kwargs)
        # all wave and pulse fish:
        for i, (mean_eod, sdata, pdata, pulse, props) in enumerate(zip(mean_eods, spec_data, phase_data,
                                                                       pulse_data, eod_props)):
            write_file_zip(zf, save_eod_waveform, output_basename,
                           mean_eod, unit, i, **kwargs)
            # spectrum:
            if len(sdata)>0:
                if sdata.shape[1] == 2:
                    write_file_zip(zf, save_pulse_spectrum, output_basename,
                                   sdata, unit, i, **kwargs)
                else:
                    write_file_zip(zf, save_wave_spectrum, output_basename,
                                   sdata, unit, i, **kwargs)
            # phases:
            write_file_zip(zf, save_pulse_phases, output_basename,
                           pdata, unit, i, **kwargs)
            # pulses:
            write_file_zip(zf, save_pulse_gaussians, output_basename,
                           pulse, unit, i, **kwargs)
            # times:
            write_file_zip(zf, save_pulse_times, output_basename,
                           props, i, **kwargs)
        # wave fish properties:
        write_file_zip(zf, save_wave_fish, output_basename,
                       eod_props, unit, **kwargs)
        # pulse fish properties:
        write_file_zip(zf, save_pulse_fish, output_basename,
                       eod_props, unit, **kwargs)


def load_analysis(file_pathes):
    """Load all EOD analysis files.

    Parameters
    ----------
    file_pathes: list of string
        Pathes of the analysis files of a single recording to be loaded.

    Returns
    -------
    mean_eods: list of 2D array of floats
        Averaged EOD waveforms: time in seconds, mean, standard deviation, fit.
    wave_eodfs: 2D array of floats
        EODfs and power of wave type fish.
    wave_indices: array of ints
        Corresponding indices of fish, can contain negative numbers to
        indicate frequencies without fish.
    eod_props: list of dict
        Properties of EODs. The 'index' property is an index into the
        reurned lists.
    spec_data: list of 2D array of floats
        Amplitude and phase spectrum of wave-type EODs with columns
        harmonics, frequency, amplitude, relative amplitude in dB,
        relative power in dB, phase, data power in unit squared.
        Energy spectrum of single pulse-type EODs with columns
        frequency and energy.
    phase_data: list of dict
        Properties of phases of pulse-type EODs with keys
        indices, times, amplitudes, relamplitudes, widths, areas, relareas, zeros
    pulse_data: list of dict
        For each pulse fish a dictionary with phase times, amplitudes and standard
        deviations of Gaussians fitted to the pulse waveform.  Use the
        functions provided in thunderfish.fakefish to simulate pulse
        fish EODs from this data.
    recording: string
        Path and base name of the recording file.
    channel: int
        Analysed channel of the recording.
    unit: string
        Unit of EOD waveform.
    """
    recording = None
    channel = -1
    eod_props = []
    zf = None
    if len(file_pathes) == 1 and Path(file_pathes[0]).suffix[1:] == 'zip':
        zf = zipfile.ZipFile(file_pathes[0])
        file_pathes = sorted(zf.namelist())
    # read wave- and pulse-fish summaries:
    pulse_fish = False
    wave_fish = False
    for f in file_pathes:
        recording, _, channel, _, ftype, _, _ = parse_filename(f)
        if zf is not None:
            f = io.TextIOWrapper(zf.open(f, 'r'))
        if ftype == 'wavefish':
            eod_props.extend(load_wave_fish(f))
            wave_fish = True
        elif ftype == 'pulsefish':
            eod_props.extend(load_pulse_fish(f))
            pulse_fish = True
    idx_offs = 0
    if wave_fish and not pulse_fish:
        idx_offs = sorted([ep['index'] for ep in eod_props])[0]
    # load all other files:
    neods = len(eod_props)
    if neods < 1:
        neods = 1
        eod_props = [None]
    wave_eodfs = np.array([])
    wave_indices = np.array([])
    mean_eods = [None]*neods
    spec_data = [None]*neods
    phase_data = [None]*neods
    pulse_data = [None]*neods
    unit = None
    for f in file_pathes:
        recording, _, channel, _, ftype, idx, _ = parse_filename(f)
        if neods == 1 and idx > 0:
            idx = 0
        idx -= idx_offs
        if zf is not None:
            f = io.TextIOWrapper(zf.open(f, 'r'))
        if ftype == 'waveeodfs':
            wave_eodfs, wave_indices = load_wave_eodfs(f)
        elif ftype == 'eodwaveform':
            mean_eods[idx], unit = load_eod_waveform(f)
        elif ftype == 'wavespectrum':
            spec_data[idx], unit = load_wave_spectrum(f)
        elif ftype == 'pulsephases':
            phase_data[idx], unit = load_pulse_phases(f)
        elif ftype == 'pulsegaussians':
            pulse_data[idx], unit = load_pulse_gaussians(f)
        elif ftype == 'pulsetimes':
            pulse_times = load_pulse_times(f)
            eod_props[idx]['times'] = pulse_times
            eod_props[idx]['peaktimes'] = pulse_times
        elif ftype == 'pulsespectrum':
            spec_data[idx] = load_pulse_spectrum(f)
    # fix wave spectra:
    wave_eodfs = [fish.reshape(1, 2) if len(fish)>0 else fish
                  for fish in wave_eodfs]
    if len(wave_eodfs) > 0 and len(spec_data) > 0:
        eodfs = []
        for idx, fish in zip(wave_indices, wave_eodfs):
            if idx >= 0:
                spec = spec_data[idx]
                specd = np.zeros((np.sum(np.isfinite(spec[:, -1])),
                                  2))
                specd[:, 0] = spec[np.isfinite(spec[:, -1]),1]
                specd[:, 1] = spec[np.isfinite(spec[:, -1]),-1]
                eodfs.append(specd)
            else:
                specd = np.zeros((10, 2))
                specd[:, 0] = np.arange(len(specd))*fish[0,0]
                specd[:, 1] = np.nan
                eodfs.append(specd)
        wave_eodfs = eodfs
    return mean_eods, wave_eodfs, wave_indices, eod_props, spec_data, \
        phase_data, pulse_data, recording, channel, unit


def load_recording(file_path, channel=0, load_kwargs={},
                   eod_props=None, verbose=0):
    """Load recording.

    Parameters
    ----------
    file_path: string or Path
        Full path of the file with the recorded data.
        Extension is optional. If absent, look for the first file
        with a reasonable extension.
    channel: int
        Channel of the recording to be returned.
    load_kwargs: dict
        Keyword arguments that are passed on to the 
        format specific loading functions.
    eod_props: list of dict or None
        List of EOD properties from which start and end times of
        analysis window are extracted.
    verbose: int
        Verbosity level passed on to load function.

    Returns
    -------
    data: array of float
        Data of the requested `channel`.
    rate: float
        Sampling rate in Hertz.
    idx0: int
        Start index of the analysis window.
    idx1: int
        End index of the analysis window.
    info_dict: dict
        Dictionary with path, name, species, channel, chanstr, time.
    """
    data = None
    rate = 0.0
    idx0 = 0
    idx1 = 0
    info_dict = dict(path='',
                     name='',
                     species='',
                     channel=0,
                     chanstr='',
                     time='')
    for k in range(1, 10):
        info_dict[f'part{k}'] = ''
    data_file = Path()
    file_path = Path(file_path)
    if len(file_path.suffix) > 1:
        data_file = file_path
    else:
        data_files = file_path.parent.glob(file_path.stem + '*')
        for dfile in data_files:
            if not dfile.suffix[1:] in ['zip'] + list(TableData.ext_formats.values()):
                data_file = dfile
                break
    if data_file.is_file():
        all_data = DataLoader(data_file, verbose=verbose, **load_kwargs)
        rate = all_data.rate
        unit = all_data.unit
        ampl_max = all_data.ampl_max
        data = all_data[:, channel]
        species = get_str(all_data.metadata(), ['species'], default='')
        if len(species) > 0:
            species += ' '
        info_dict.update(path=os.fsdecode(all_data.filepath),
                         name=all_data.basename(),
                         species=species,
                         channel=channel)
        offs = 1
        for k, part in enumerate(all_data.filepath.parts):
            if k == 0 and part == all_data.filepath.anchor:
                offs = 0
                continue
            if part == all_data.filepath.name:
                break
            info_dict[f'part{k + offs}'] = part
        if all_data.channels > 1:
            if all_data.channels > 100:
                info_dict['chanstr'] = f'-c{channel:03d}'
            elif all_data.channels > 10:
                info_dict['chanstr'] = f'-c{channel:02d}'
            else:
                info_dict['chanstr'] = f'-c{channel:d}'
        else:
            info_dict['chanstr'] = ''
        idx0 = 0
        idx1 = len(data)
        if eod_props is not None and len(eod_props) > 0 and 'twin' in eod_props[0]:
            idx0 = int(eod_props[0]['twin']*rate)
        if len(eod_props) > 0 and 'window' in eod_props[0]:
            idx1 = idx0 + int(eod_props[0]['window']*rate)
        info_dict['time'] = f'-t{idx0/rate:.0f}s'
        all_data.close()
            
    return data, rate, idx0, idx1, info_dict

        
def add_eod_analysis_config(cfg, win_fac=2.0, min_win=0.01, max_eods=None,
                            min_sem=False, unfilter_cutoff=0.0,
                            flip_wave='none', flip_pulse='none',
                            n_harm=10, min_pulse_win=0.001,
                            thresh_fac=0.01, min_dist=50.0e-6,
                            width_frac = 0.5, fit_frac = 0.5,
                            freq_resolution=1.0, fade_frac=0.0,
                            ipi_cv_thresh=0.5, ipi_percentile=30.0):
    """Add all parameters needed for the eod analysis functions as a new
    section to a configuration.

    Parameters
    ----------
    cfg: ConfigFile
        The configuration.
        
    See `eod_waveform()`, `analyze_wave()`, and `analyze_pulse()` for
    details on the remaining arguments.
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
    cfg.add('eodPeakThresholdFactor', thresh_fac, '', 'Threshold for detection of peaks and troughs in pulse EODs as a fraction of the pulse amplitude.')
    cfg.add('eodMinimumDistance', min_dist, 's', 'Minimum distance between peaks and troughs in a EOD pulse.')
    cfg.add('eodPulseWidthFraction', 100*width_frac, '%', 'The width of a pulse is measured at this fraction of the pulse height.')
    cfg.add('eodExponentialFitFraction', 100*fit_frac, '%', 'An exponential function is fitted on the tail of a pulse starting at this fraction of the height of the last peak.')
    cfg.add('eodPulseFrequencyResolution', freq_resolution, 'Hz', 'Frequency resolution of single pulse spectrum.')
    cfg.add('eodPulseFadeFraction', 100*fade_frac, '%', 'Fraction of time of the EOD waveform snippet that is used to fade in and out to zero baseline.')
    cfg.add('ipiCVThresh', ipi_cv_thresh, '', 'If coefficient of variation of interpulse intervals is smaller than this threshold, then use all intervals for computing EOD frequency.')
    cfg.add('ipiPercentile', ipi_percentile, '%', 'Use only interpulse intervals shorter than this percentile to compute EOD frequency.')


def eod_waveform_args(cfg):
    """Translates a configuration to the respective parameter names of
    the function `eod_waveform()`.
    
    The return value can then be passed as key-word arguments to this
    function.

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
    """Translates a configuration to the respective parameter names of
    the function `analyze_wave()`.
    
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
    a = cfg.map({'n_harm': 'eodHarmonics',
                 'power_n_harmonics': 'powerNHarmonics',
                 'flip_wave': 'flipWaveEOD'})
    return a


def analyze_pulse_args(cfg):
    """Translates a configuration to the respective parameter names of
    the function `analyze_pulse()`.
    
    The return value can then be passed as key-word arguments to this
    function.

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
                 'thresh_fac': 'eodPeakThresholdFactor',
                 'min_dist': 'eodMinimumDistance',
                 'width_frac': 'eodPulseWidthFraction',
                 'fit_frac': 'eodExponentialFitFraction',
                 'flip_pulse': 'flipPulseEOD',
                 'freq_resolution': 'eodPulseFrequencyResolution',
                 'fade_frac': 'eodPulseFadeFraction',
                 'ipi_cv_thresh': 'ipiCVThresh',
                 'ipi_percentile': 'ipiPercentile'})
    a['width_frac'] *= 0.01
    a['fit_frac'] *= 0.01
    a['fade_frac'] *= 0.01
    return a


def add_species_config(cfg, species_file='none', wave_max_rms=0.2,
                       pulse_max_rms=0.2):
    """Add parameters needed for assigning EOD waveforms to species.

    Parameters
    ----------
    cfg: ConfigFile
        The configuration.
    species_file: string
        File path to a file containing species names and corresponding
        file names of EOD waveform templates. If 'none', no species
        assignemnt is performed.
    wave_max_rms: float
        Maximum allowed rms difference (relative to standard deviation
        of EOD waveform) to an EOD waveform template for assignment to
        a wave fish species.
    pulse_max_rms: float
        Maximum allowed rms difference (relative to standard deviation
        of EOD waveform) to an EOD waveform template for assignment to
        a pulse fish species.
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
    cfg.add('maximumClippedFraction', 100*max_clipped_frac, '%', 'Take waveform of the fish with the highest power only if the fraction of clipped signals is below this value.')
    cfg.add('maximumVariance', max_variance, '', 'Skip waveform of fish if the standard error of the EOD waveform relative to the peak-to-peak amplitude is larger than this number. A value of zero allows any variance.')
    cfg.add('maximumRMSError', max_rms_error, '', 'Skip waveform of wave fish if the root-mean-squared error of the fit relative to the peak-to-peak amplitude is larger than this number.')
    cfg.add('minimumPower', min_power, 'dB', 'Skip waveform of wave fish if its power is smaller than this value.')
    cfg.add('maximumTotalHarmonicDistortion', max_thd, '', 'Skip waveform of wave fish if its total harmonic distortion is larger than this value. If set to zero do not check.')
    cfg.add('maximumCrossings', max_crossings, '', 'Maximum number of zero crossings per EOD period.')
    cfg.add('maximumFirstHarmonicAmplitude', max_relampl_harm1, '', 'Skip waveform of wave fish if the amplitude of the first harmonic is higher than this factor times the amplitude of the fundamental. If set to zero do not check.')
    cfg.add('maximumSecondHarmonicAmplitude', max_relampl_harm2, '', 'Skip waveform of wave fish if the ampltude of the second harmonic is higher than this factor times the amplitude of the fundamental. That is, the waveform appears to have twice the frequency than the fundamental. If set to zero do not check.')
    cfg.add('maximumThirdHarmonicAmplitude', max_relampl_harm3, '', 'Skip waveform of wave fish if the ampltude of the third harmonic is higher than this factor times the amplitude of the fundamental. If set to zero do not check.')


def wave_quality_args(cfg):
    """Translates a configuration to the respective parameter names of
    the function `wave_quality()`.
    
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
    a['max_clipped_frac'] *= 0.01
    return a


def pulse_quality_args(cfg):
    """Translates a configuration to the respective parameter names of
    the function `pulse_quality()`.
    
    The return value can then be passed as key-word arguments to this
    function.

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
    a['max_clipped_frac'] *= 0.01
    return a


def main():
    import matplotlib.pyplot as plt
    from .fakefish import pulsefish_eods

    print('Analysis of EOD waveforms.')

    # data:
    rate = 96_000
    data = pulsefish_eods('Triphasic', 83.0, rate, 5.0, noise_std=0.02)
    unit = 'mV'
    eod_idx, _ = detect_peaks(data, 1.0)
    eod_times = eod_idx/rate

    # analyse EOD:
    mean_eod, eod_times = eod_waveform(data, rate, eod_times)
    mean_eod, props, peaks, pulses, energy = analyze_pulse(mean_eod, eod_times)

    # plot:
    fig, axs = plt.subplots(1, 2)
    plot_eod_waveform(axs[0], mean_eod, props, peaks, unit=unit)
    axs[0].set_title(f'{props["type"]} fish: EODf = {props["EODf"]:.1f} Hz')
    plot_pulse_spectrum(axs[1], energy, props)
    plt.show()


if __name__ == '__main__':
    main()
