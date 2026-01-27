"""
Analysis of EOD waveforms.

## EOD waveform analysis

- `eod_waveform()`: compute an averaged EOD waveform.
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
- `plot_eod_snippets()`: plot a few EOD waveform snippets.
- `plot_eod_waveform()`: plot and annotate the averaged EOD-waveform with standard error.

## Storage

- `save_eod_waveform()`: save mean EOD waveform to file.
- `load_eod_waveform()`: load EOD waveform from file.
- `parse_filename()`: parse components of an EOD analysis file name.
- `save_analysis(): save EOD analysis results to files.
- `load_analysis()`: load EOD analysis files.
- `load_recording()`: load recording.

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
import zipfile
import numpy as np
try:
    import matplotlib.pyplot as plt
except ImportError:
    pass

from pathlib import Path
from audioio import get_str
from thunderlab.eventdetection import detect_peaks, snippets
from thunderlab.powerspectrum import decibel
from thunderlab.tabledata import TableData
from thunderlab.dataloader import DataLoader

from .fakefish import normalize_pulsefish, export_pulsefish
from .fakefish import normalize_wavefish, export_wavefish
from .waveanalysis import waveeod_waveform, analyze_wave
from .waveanalysis import plot_wave_spectrum
from .waveanalysis import save_wave_eodfs, load_wave_eodfs
from .waveanalysis import save_wave_fish, load_wave_fish
from .waveanalysis import save_wave_spectrum, load_wave_spectrum
from .pulseanalysis import analyze_pulse
from .pulseanalysis import plot_pulse_eods, plot_pulse_spectrum
from .pulseanalysis import save_pulse_fish, load_pulse_fish
from .pulseanalysis import save_pulse_spectrum, load_pulse_spectrum
from .pulseanalysis import save_pulse_phases, load_pulse_phases
from .pulseanalysis import save_pulse_gaussians, load_pulse_gaussians
from .pulseanalysis import save_pulse_times, load_pulse_times


def eod_waveform(data, rate, eod_times, win_fac=2.0, min_win=0.01,
                 min_sem=False, max_eods=None):
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
        # make sure tstart, tend, and time constant are included:
        if props is not None:
            if 'tstart' in props and 1000*props['tstart'] < -w:
                w = np.ceil(abs(1000*props['tstart'])/0.5)*0.5
            if 'tend' in props and 1000*props['tend'] > 2*w:
                w = np.ceil(0.5*abs(1000*props['tend'])/0.5)*0.5
            if 'taustart' in props and props['taustart'] is not None and \
               1100*props['taustart'] > 2*w:
                w = np.ceil(0.5*abs(1100*props['taustart'])/0.5)*0.5
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
    quadrants = np.zeros((2, 2), dtype=int)
    # magnification threshold:
    if magnification_factor > 1:
        mag_thresh = 0.95*np.max(np.abs(eod))/magnification_factor
    else:
        mag_thresh = 0
    # plot zero line:
    ax.axhline(0.0, zorder=10, **zero_style)
    # plot areas:
    if phases is not None and len(phases) > 0:
        if positive_style is not None and len(positive_style) > 0:
            ax.fill_between(time, eod, 0, eod >= 0, zorder=0,
                            **positive_style)
        if negative_style is not None and len(negative_style) > 0:
                ax.fill_between(time, eod, 0, eod <= 0, zorder=0,
                                **negative_style)
    # plot Fourier/Gaussian fit:
    if eod_waveform.shape[1] > 3 and np.all(np.isfinite(eod_waveform[:, 3])):
        ax.plot(time, eod_waveform[:, 3], zorder=30, **fit_style)
    # plot time constant fit:
    tau_magnified = False
    if eod_waveform.shape[1] > 4:
        if np.nanmax(np.abs(eod_waveform[:, 4])) < mag_thresh:
            tau_magnified = True
            ax.plot(time, magnification_factor*eod_waveform[:, 4],
                    zorder=35, **fit_style)
        else:
            fs = dict(**fit_style)
            if 'lw' in fs:
                fs['lw'] *= 2
            ax.plot(time, eod_waveform[:, 4], zorder=35, **fs)
    # plot waveform:
    ax.plot(time, eod, zorder=45, **wave_style)
    # plot standard error:
    if eod_waveform.shape[1] > 2:
        std_eod = eod_waveform[:, 2]
        ax.fill_between(time, eod + std_eod, eod - std_eod,
                        zorder=20, **sem_style)
    # plot magnified pulse waveform:
    magnification_mask = np.zeros(len(time), dtype=bool)
    if magnification_factor > 1 and phases is not None and len(phases) > 0:
        i0 = np.argmax(np.abs(eod) > mag_thresh)
        if i0 > 0:
            left_eod = magnification_factor*eod[:i0]
            magnification_mask[:i0] = True
            ax.plot(time[:i0], left_eod, zorder=40, **magnified_style)
            if left_eod[-1] > 0:
                it = np.argmax(left_eod > 0.95*np.max(eod))
                if it < len(left_eod)//2:
                    it = len(left_eod) - 1
                ty = left_eod[it] if left_eod[it] < np.max(eod) else np.max(eod)
                ta = ax.text(time[it], ty, f'x{magnification_factor:.0f} ',
                             ha='right', va='top', zorder=100,
                             fontsize=fontsize)
                if ty > 0.5*ylim:
                    quadrants[0, 0] += 1
            else:
                it = np.argmax(left_eod < 0.95*np.min(eod))
                if it < len(left_eod)//2:
                    it = len(left_eod) - 1
                ty = left_eod[it] if left_eod[it] > np.min(eod) else np.min(eod)
                ta = ax.text(time[it], ty, f'x{magnification_factor:.0f} ',
                             ha='right', va='bottom', zorder=100,
                             fontsize=fontsize)
                if ty < -0.5*ylim:
                    quadrants[1, 0] += 1
            texts.append(ta)
            if quadrants[0, 0] == 0:
                quadrants[0, 0] += np.max(left_eod[time[:i0] < 0.1*xlim_l]) > 0.5*ylim
            if quadrants[1, 0] == 0:
                quadrants[1, 0] += np.min(left_eod[time[:i0] < 0.1*xlim_l]) < -0.5*ylim
        i1 = len(eod) - np.argmax(np.abs(eod[::-1]) > mag_thresh)
        right_eod = magnification_factor*eod[i1:]
        magnification_mask[i1:] = True
        ax.plot(time[i1:], right_eod, zorder=40, **magnified_style)
        quadrants[0, 1] += np.max(right_eod[time[i1:] > 0.4*xlim_r]) > 0.5*ylim
        quadrants[1, 1] += np.min(right_eod[time[i1:] > 0.4*xlim_r]) < -0.5*ylim
    # annotate time constant fit:
    tau = None if props is None else props.get('tau', None)
    if tau is not None and eod_waveform.shape[1] > 4:
        if tau < 0.001:
            label = f'\u03c4={1.e6*tau:.0f}\u00b5s'
        else:
            label = f'\u03c4={1.e3*tau:.2f}ms'
        inx = np.argmin(np.isnan(eod_waveform[:, 4]))
        x0 = time[inx]
        x = x0 + 1000*np.log(2)*tau
        if x + 4*xfs >= xlim_r:
            if xlim_r - x0 >= 4*xfs:
                x = xlim_r - 8*xfs
            else:
                x = x0
        elif x + 8*xfs > xlim_r:
            x = xlim_r - 8*xfs
        if x < x0:
            x = x0
        y = eod_waveform[np.argmin(np.abs(time - x)), 4]
        if tau_magnified:
            y *= magnification_factor
        va = 'bottom' if eod[inx] > 0 else 'top'
        if eod[inx] < 0:
            y -= 0.5*yfs
        ta = ax.text(x + xfs, y, label, ha='left', va=va,
                     zorder=100, fontsize=fontsize)
        texts.append(ta)
        if x + xfs > 0.4*xlim_r:
            if y > 0.5*ylim:
                quadrants[0, 1] += 1
            elif y < -0.5*ylim:
                quadrants[1, 1] += 1
    if props is not None:
        # mark start and end:
        if 'tstart' in props:
            ax.axvline(1000*props['tstart'], 0.45, 0.55,
                       color='k', lw=0.5, zorder=25)
        if 'tend' in props:
            ax.axvline(1000*props['tend'], 0.45, 0.55,
                       color='k', lw=0.5, zorder=25)
        # mark cumulative:
        if 'median' in props:
            y = -1.07*ylim
            m = 1000*props['median']
            q1 = 1000*props['quartile1']
            q3 = 1000*props['quartile3']
            w = q3 - q1
            ax.plot([q1, q3], [y, y], 'gray', lw=4, zorder=25)
            ax.plot(m, y, 'o', color='white', ms=3, zorder=26)
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
            if ptime < xlim_l or ptime > xlim_r:
                continue
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
            add = True
            if local_phase or (min_max_phase and abs(pampl)/ylim < 0.8):
                halign = 'center'
                dx = 0
                dy = 0.6*yfs
                if local_phase:
                    add = False
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
            if add:
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
                        ha='center', zorder=35, fontsize=fontsize)
            elif abs(relampl) > 0.25 and abs(relarea) > 0.19:
                ax.text(x, sign*0.6*yfs, label,
                        va='top' if sign < 0 else 'baseline',
                        ha='center', zorder=35, fontsize=fontsize)
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
        for ts, (j, k) in zip([ul_texts, ur_texts, ll_texts, lr_texts],
                              [(0, 0), (0, 1), (1, 0), (1, 1)]):
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
                    s = t.get_text()
                    if abs(y) < abs(yp) + 2*yfs and \
                       len(s) > 4 and s[:2] != '\u03c4=':
                        y = np.sign(y)*(abs(yp) + 2*yfs)
                        t.set_y(y)
                    if len(s) >= 4 and abs(y) > 0.5*ylim:
                        quadrants[j, k] += 1
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
        # weigh left quadrants less:
        quadrants *= 2
        quadrants[quadrants[:, 1] > 0, 1] -= 1 
        # find free quadrant:
        q_row, q_col = np.unravel_index(np.argmin(quadrants), quadrants.shape)
        # place text in quadrant:
        y = 1 if q_row == 0 else 0
        va = 'top' if q_row == 0 else 'bottom'
        x = 0.03 if q_col == 0 else 0.97
        ha = 'left' if q_col == 0 else 'right'
        ax.text(x, y, label, transform=ax.transAxes,
                va=va, ha=ha, zorder=100)

    
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
                species = eod_props[k].get('species', '')
                if len(pulse_data[k]) > 0:
                    fish = normalize_pulsefish(pulse_data[k])
                    export_pulsefish(fish, f'{name}-{k}_phases',
                                     species, f)
                    f.write('\n')
                else:
                    sdata = spec_data[k]
                    if len(sdata) > 0 and sdata.shape[1] > 2:
                        fish = dict(amplitudes=sdata[:, 3], phases=sdata[:, 5])
                        fish = normalize_wavefish(fish)
                        export_wavefish(fish, f'{name}-{k}_harmonics',
                                        species, f)
                        f.write('\n')
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
                            start_end_thresh_fac=0.01, peak_thresh_fac=0.002,
                            min_dist=50.0e-6, width_frac = 0.5, fit_frac = 0.5,
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
    cfg.add('eodPeakThresholdFactor', peak_thresh_fac, '', 'Threshold for detection of peaks and troughs in pulse EODs as a fraction of the p-p amplitude.')
    cfg.add('eodStartEndThresholdFactor', start_end_thresh_fac, '', 'Threshold for for start and end time of pulse EODs as a fraction of the p-p amplitude.')
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
                 'min_sem': 'eodMinSem'})
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
                 'start_end_thresh_fac': 'eodStartEndThresholdFactor',
                 'peak_thresh_fac': 'eodPeakThresholdFactor',
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
    mean_eod, props, peaks, pulses, energy = \
        analyze_pulse(mean_eod, None, eod_times)

    # plot:
    fig, axs = plt.subplots(1, 2)
    plot_eod_waveform(axs[0], mean_eod, props, peaks, unit=unit)
    axs[0].set_title(f'{props["type"]} fish: EODf = {props["EODf"]:.1f} Hz')
    plot_pulse_spectrum(axs[1], energy, props)
    plt.show()


if __name__ == '__main__':
    main()
