"""
Analyse EOD waveforms.

## EOD waveform analysis

- `eod_waveform()`: compute an averaged EOD waveform.
- `adjust_eodf()`: adjust EOD frequencies to a standard temperature.

## Undo high-pass filter

- `unfilter()`: apply inverse high-pass filter on time series.
- `unfilter_coeff()`: apply inverse high-pass filter on Fourier coefficients.

## Similarity of EOD waveforms

- `wave_similarity()`: root-mean squared difference between two wave fish EODs.
- `pulse_similarity()`: root-mean squared difference between two pulse fish EODs.
- `load_species_waveforms()`: load template EOD waveforms for species matching.

## Quality assessment

- `clipped_fraction()`: compute fraction of clipped EOD waveform snippets.

## Visualization

- `plot_eod_recording()`: plot a zoomed in range of the recorded trace.
- `plot_eod_snippets()`: plot a few EOD waveform snippets.

## Storage

- `save_eod_waveform()`: save mean EOD waveform to file.
- `load_eod_waveform()`: load EOD waveform from file.
- `parse_filename()`: parse components of an EOD analysis file name.
- `save_analysis(): save EOD analysis results to files.
- `load_analysis()`: load EOD analysis files.
- `load_recording()`: load recording.

## Configuration

- `add_eod_analysis_config()`: add parameters for EOD analysis functions to configuration.
- `eod_waveform_args()`: retrieve parameters for `eod_waveform()` from configuration.
- `add_species_config()`: add parameters needed for assigning EOD waveforms to species.
- `add_eod_quality_config()`: add parameters for `wave_quality()` and `pulse_quality()` to configuration.
"""

import os
import io
import numpy as np

from pathlib import Path
from zipfile import ZipFile
from audioio import get_str
from thunderlab.eventdetection import detect_peaks, snippets
from thunderlab.fourier import normalize_fourier_coeffs
from thunderlab.powerspectrum import decibel
from thunderlab.tabledata import TableData
from thunderlab.dataloader import DataLoader

from .fakefish import normalize_pulsefish, export_pulsefish
from .fakefish import normalize_wavefish, export_wavefish
from .waveanalysis import save_wave_eodfs, load_wave_eodfs
from .waveanalysis import save_wave_fish, load_wave_fish
from .waveanalysis import save_wave_phases, load_wave_phases
from .waveanalysis import save_wave_spectrum, load_wave_spectrum
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


def unfilter(data, rate, fcutoff):
    """Apply inverse high-pass filter on time series.

    Assumes high-pass filter
    \\[ \\tau \\dot y = -y + \\tau \\dot x \\]
    has been applied on the original data \\(x\\), where
    \\(\\tau=(2\\pi f_{cutoff})^{-1}\\) is the time constant of the
    filter. To recover \\(x\\) the ODE
    \\[ \\tau \\dot x = y + \\tau \\dot y \\]
    is applied on the filtered data \\(y\\).

    Parameters
    ----------
    data: 1D ndarray of float
        High-pass filtered original data.
    rate: float
        Sampling rate of `data` in Hertz.
    fcutoff: float
        Cutoff frequency \\(f_{cutoff}\\) of the high-pass filter in Hertz.

    Returns
    -------
    data: ndarray
        Recovered original data.
    """
    tau = 0.5/np.pi/fcutoff
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


def unfilter_coeff(freq, coeffs, fcutoff):
    """Apply inverse high-pass filter on Fourier coefficients.

    Assumes first-order high-pass filter
    \\[ \\tau \\dot y = -y + \\tau \\dot x \\]
    has been applied on the original data \\(x\\), where
    \\(\\tau=(2\\pi f_{cutoff})^{-1}\\) is the time constant of the
    filter. The transfer function of this high-pass filter is
    \\[ H(\\omega) = \\frac{i\\omega\\tau}{1 + i\\omega\\tau} \\]

    To undo the effect of the high-pass filter, the complex-valued
    Fourier coefficients \\(c_k\\) of the waveform for frequencies \\(k f_0\\)
    are multiplied with the inverse transfer function
    \\[ IH(\\omega) = \\frac{1}{H(\\omega)} = 1 - i\\frac{1}{\\omega\\tau} \\]
    which simplifies for the harmonics to
    \\[ IH(kf_0) = 1 - i\\frac{f_{cutoff}}{k f_0} \\]
    

    Parameters
    ----------
    freq: float
        Fundamental frequency \\(f_0\\).
    coeffs: 1D array of complex
        For each harmonics the complex valued Fourier coefficient
        as, for example, returned by `fourier_coeffs()`.
        The first one is the offset.
    fcutoff: float
        Cutoff frequency \\(f_{cutoff}\\) of the high-pass filter in Hertz.

    Returns
    -------
    coeffs: 1D array of complex
        Unfiltered Fourier coefficients. Normalized to zero phase shift
        of the fundamental and no offset.
    """
    ihp_coeffs = np.zeros(len(coeffs), dtype=complex)
    ihp_coeffs[1:] = 1 - 1j/(np.arange(1, len(coeffs))*freq/fcutoff)
    coeffs *= ihp_coeffs
    coeffs = normalize_fourier_coeffs(coeffs)
    return coeffs


def load_species_waveforms(species_file='none'):
    """Load template EOD waveforms for species matching.
    
    Parameters
    ----------
    species_file: str
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
    wave_names: list of str
        List of species names of wave-type fish.
    wave_eods: list of 2-D arrays
        List of EOD waveforms of wave-type fish corresponding to
        `wave_names`.  First column of a waveform is time in seconds,
        second column is the EOD waveform.  The waveforms contain
        exactly one EOD period.
    pulse_names: list of str
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
    if len(eod_snippets) == 0:
        return 0
    else:
        clipped_frac = np.sum(np.any((eod_snippets > max_clip) |
                                     (eod_snippets < min_clip), axis=1)) \
                             / len(eod_snippets)
        return clipped_frac


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
    unit: str
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
        i1 = len(data) - 1
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
    ax.set_ylim(ymin - 0.05*dy, ymax + 0.05*dy)
    if len(unit) == 0 or unit == 'a.u.':
        ax.set_ylabel('Amplitude')
    else:
        ax.set_ylabel(f'Amplitude [{unit}]')

        
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

    
def save_eod_waveform(mean_eod, unit, idx, basename, **kwargs):
    """Save mean EOD waveform to file.

    Parameters
    ----------
    mean_eod: 2D array of floats
        Averaged EOD waveform as returned by `eod_waveform()`,
        `analyze_wave()`, and `analyze_pulse()`.
    unit: str
        Unit of the waveform data.
    idx: int or None
        Index of fish.
    basename: str or Path or stream
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
    file_path: str or Path
        Path of the file to be loaded.

    Returns
    -------
    mean_eod: 2D array of floats
        Averaged EOD waveform: time in seconds, mean, standard deviation, fit.
    unit: str
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


file_types = ['waveeodfs', 'wavefish', 'pulsefish', 'eodwaveform', 'wavephases',
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
    file_path: str or Path
        Path of the file to be parsed.

    Returns
    -------
    recording: str
        Path and basename of the recording, i.e. 'PATH/RECORDING'.
        A leading './' is removed.
    base_path: Path
        Path and basename of the analysis results,
        i.e. 'PATH/RECORDING-CHANNEL-TIME'.
    channel: int
        Channel of the recording
        ('CHANNEL' component of the file name if present).
        -1 if not present in `file_path`.
    time: float
        Start time of analysis window in seconds
        ('TIME' component of the file name if present).
        `None` if not present in `file_path`.
    ftype: str
        Type of analysis file (e.g. 'wavespectrum', 'pulsephases', etc.),
        ('FTYPE' component of the file name if present).
        See `file_types` for a list of all supported file types.
        Empty string if not present in `file_path`.
    index: int
        Index of the EOD.
        ('N' component of the file name if present).
        -1 if not present in `file_path`.
    ext: str
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
    output_basename: str or Path
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
    unit: str
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
                    fp = Path(output.stem + str(fp))
                    zf.writestr(fp.name, df.getvalue())
                    if verbose > 0:
                        print('zipped file', fp.name)


    output_basename = Path(output_basename)                        
    if 'table_format' in kwargs and kwargs['table_format'] == 'py':
        with open(output_basename.with_suffix('.py'), 'w') as f:
            name = output_basename.stem
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
            zf = ZipFile(output_basename.with_suffix('.zip'), 'w')
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
                if sdata.shape[1] <= 3:
                    write_file_zip(zf, save_pulse_spectrum, output_basename,
                                   sdata, unit, i, **kwargs)
                else:
                    write_file_zip(zf, save_wave_spectrum, output_basename,
                                   sdata, unit, i, **kwargs)
            # phases:
            if 'areas' in pdata:
                write_file_zip(zf, save_pulse_phases, output_basename,
                               pdata, unit, i, **kwargs)
            else:
                write_file_zip(zf, save_wave_phases, output_basename,
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
        if zf is not None:
            zf.close()


def load_analysis(file_pathes):
    """Load all EOD analysis files.

    Parameters
    ----------
    file_pathes: list of str or Path
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
    recording: str
        Path and base name of the recording file.
    channel: int
        Analysed channel of the recording.
    unit: str
        Unit of EOD waveform.
    """
    recording = None
    channel = -1
    eod_props = []
    zf = None
    if len(file_pathes) == 1 and Path(file_pathes[0]).suffix[1:] == 'zip':
        zf = ZipFile(file_pathes[0], 'r')
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
        elif ftype == 'wavephases':
            phase_data[idx], unit = load_wave_phases(f)
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
    if zf is not None:
        zf.close()
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
    file_path: str or Path
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
                            min_sem=False, unfilter_cutoff=0.0):
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
    cfg.add('unfilterCutoff', unfilter_cutoff, 'Hz', 'If non-zero remove effect of high-pass filter of recording device using the specified cutoff frequency of the filter.')


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
    a = cfg.map(win_fac='eodSnippetFac',
                min_win='eodMinSnippet',
                max_eods='eodMaxEODs',
                min_sem='eodMinSem')
    return a


def add_species_config(cfg, species_file='none', wave_max_rms=0.2,
                       pulse_max_rms=0.2):
    """Add parameters needed for assigning EOD waveforms to species.

    Parameters
    ----------
    cfg: ConfigFile
        The configuration.
    species_file: str
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

    
def add_eod_quality_config(cfg, max_clipped_frac=0.1, max_phases=4,
                           max_rms_sem=0.0, max_rms_error=0.05,
                           min_power=-100.0, max_thd=0.0, max_db_diff=20.0,
                           max_relampl_harm2=0.0, max_relampl_harm3=0.0,
                           max_relampl_harm4=0.0):
    """Add parameters for `wave_quality()` and  `pulse_quality()` to configuration.

    Parameters
    ----------
    cfg: ConfigFile
        The configuration.
        
    See `wave_quality()` and `pulse_quality()` for details on the remaining arguments.
    """
    cfg.add_section('EOD selection:')
    cfg.add('maximumClippedFraction', 100*max_clipped_frac, '%', 'Take waveform of the fish with the highest power only if the fraction of clipped EODs is below this value.')
    cfg.add('maximumVariance', max_rms_sem, '', 'Skip waveform if the root-mean-squared standard deviation of the extracted waveform relative to the peak-to-peak amplitude is larger than this number. If set to zero do not check.')
    cfg.add('maximumRMSError', max_rms_error, '', 'Skip waveform of wave fish if the root-mean-squared difference between waveform and Fourier decomposition relative to the peak-to-peak amplitude is larger than this number. If set to zero do not check.')
    cfg.add('maximumPhases', max_phases, '', 'Maximum number of phases per EOD period of a wave fish.')
    cfg.add('minimumPower', min_power, 'dB', 'Skip waveform of wave fish if its power is smaller than this value.')
    cfg.add('maximumTotalHarmonicDistortion', max_thd, '', 'Skip waveform of wave fish if its total harmonic distortion is larger than this value. If set to zero do not check.')
    cfg.add('maximumPowerDifferences', max_db_diff, '', 'Skip waveform of wave fish if the standard deviation of the differences between powers of harmonics is larger than this value. If set to zero do not check.')
    cfg.add('maximumSecondHarmonicAmplitude', max_relampl_harm2, '', 'Skip waveform of wave fish if the amplitude of the second harmonic is higher than this factor times the amplitude of the fundamental (=first harmonics). That is, the waveform appears to have twice the frequency than the fundamental. If set to zero do not check.')
    cfg.add('maximumThirdHarmonicAmplitude', max_relampl_harm3, '', 'Skip waveform of wave fish if the ampltude of the third harmonic is higher than this factor times the amplitude of the fundamental (=first harmonics). If set to zero do not check.')
    cfg.add('maximumFourthHarmonicAmplitude', max_relampl_harm4, '', 'Skip waveform of wave fish if the ampltude of the fourth harmonic is higher than this factor times the amplitude of the fundamental (=first harmonics). If set to zero do not check.')


def main():
    import matplotlib.pyplot as plt
    from .fakefish import pulsefish_eods
    from .pulseanalysis import analyze_pulse, plot_pulse_eod, plot_pulse_spectrum
    
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
    plot_pulse_eod(axs[0], mean_eod, props, peaks, unit=unit)
    axs[0].set_title(f'{props["type"]} fish: EODf = {props["EODf"]:.1f} Hz')
    plot_pulse_spectrum(axs[1], energy, props)
    plt.show()


if __name__ == '__main__':
    main()
