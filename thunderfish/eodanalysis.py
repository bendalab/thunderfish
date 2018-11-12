"""
# Analysis of EOD waveforms of weakly-electric fish.

## Main functions
- `eod_waveform()`: compute an averaged EOD waveform.
- `analyze_wave()`: analyze the EOD waveform of a wave-type fish.

## Visualization
- `eod_waveform_plot()`: plot and annotate the averaged EOD-waveform with standard deviation.
- `pulse_spectrum_plot()`: plot and annotate spectrum of single pulse.
"""

import numpy as np
from scipy.optimize import curve_fit
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from .eventdetection import percentile_threshold, detect_peaks, snippets
from .powerspectrum import psd, nfft_noverlap, decibel


def eod_waveform(data, samplerate, th_factor=0.8, percentile=1.0,
                 win_fac=2.0, min_win=0.01, period=None):
    """Detect EODs in the given data, extract data snippets around each EOD,
    and compute a mean waveform with standard deviation.

    Parameters
    ----------
    data: 1-D array
        The data to be analysed.
    samplerate: float
        Sampling rate of the data in Hertz.
    percentile: int
        Percentile parameter for the eventdetection.percentile_threshold() function used to
        estimate thresholds for detecting EOD peaks in the data.
    th_factor: float
        th_factor parameter for the eventdetection.percentile_threshold() function used to
        estimate thresholds for detecting EOD peaks in the data.
    win_fac: float
        The snippet size is the period times `win_fac`.
    min_win: float
        The minimum size of the snippets.
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
        threshold = percentile_threshold(data, th_factor=th_factor, percentile=percentile)

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
    eod_snippets = snippets(data, eod_idx, -win_inx, win_inx)

    # mean and std of snippets:
    mean_eod = np.zeros((len(eod_snippets[0]), 3))
    mean_eod[:,1] = np.mean(eod_snippets, axis=0)
    mean_eod[:,2] = np.std(eod_snippets, axis=0, ddof=1)

    # time axis:
    mean_eod[:,0] = (np.arange(len(mean_eod)) - win_inx) / samplerate
    
    return mean_eod, eod_times


def sinewaves(t, freq, delay, ampl, *ap):
    """
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
        The relative amplitudes and phases of the harmonics.
        
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


def analyze_wave(eod, freq, n_harm=6):
    """
    Analyze the EOD waveform of a wave-type fish.
    
    Parameters
    ----------
    eod: 2-D array
        The eod waveform. First column is time in seconds, second column the eod waveform.
        Further columns are optional but not used.
    freq: float or 2-D array
        The frequency of the EOD or the list of harmonics (rows)
        with frequency and peak height (columns) as returned from `harmonic_groups()`.
    n_harm: int
        Maximum number of harmonics used for the fit.
    
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
        - rmserror: root-mean-square error between Fourier-fit and EOD waveform relative to
          the p-p amplitude. If larger than 0.05 the data are bad.
        - power: if `freq` is list of harmonics then `power` is set to the summed power
          of all harmonics and transformed to decibel.
    spec_data: 2-D array of floats
        First column is the index of the harmonics, second column its frequency,
        third column its relative amplitude, and fourth column the phase shift
        relative to the fundamental.
        If `freq` is a list of harmonics, a fith column is added to `spec_data` that
        contains the sqaure rooted powers of the harmonics
        normalized to the one ofthe fundamental.
        Rows are the harmonics, first row is the fundamental frequency with index 0,
        amplitude of one, and phase shift of zero.
        If the amplitude of the first harmonic (spec-data[1,3]) is larger than 2,
        or the amplitude if the second harmonic (spec-data[2,3]) is larger than 0.2,
        then the EOD waveform has the wrong waveform and
        should not be used for further analysis.
    """
    freq0 = freq
    if hasattr(freq, 'shape'):
        freq0 = freq[0][0]
        
    # storage:
    meod = np.zeros((eod.shape[0], eod.shape[1]+1))
    meod[:,:-1] = eod
    
    # subtract mean:
    meod[:,1] -= np.mean(meod[:,1])

    # flip:
    flipped = False
    if -np.min(meod[:,1]) > np.max(meod[:,1]):
        meod[:,1] = -meod[:,1]
        flipped = True
    
    # move peak of waveform to zero:
    offs = len(meod[:,1])//4
    meod[:,0] -= meod[offs+np.argmax(meod[offs:3*offs,1]),0]

    # fit sine wave:
    ampl = 0.5*(np.max(meod[:,1])-np.min(meod[:,1]))
    params = [freq0, -0.25/freq0, ampl]
    for i in range(1, n_harm):
        params.extend([1.0/(i+1), 0.0])
    popt, pcov = curve_fit(sinewaves, meod[:,0], meod[:,1], params)
    for i in range(1, n_harm):
        # make all amplitudes positive:
        if popt[1+i*2] < 0.0:
            popt[1+i*2] *= -1.0
            popt[2+i*2] += np.pi
        # all phases in the range 0 to 2 pi:
        popt[2+i*2] %= 2.0*np.pi
        # all phases except of 2nd harmonic in the range -pi to pi:
        if popt[2+i*2] > np.pi and i != 2:
            popt[2+i*2] -= 2.0*np.pi
    meod[:,-1] = sinewaves(meod[:,0], *popt)

    # fit error:
    ppampl = np.max(meod[:,3]) - np.min(meod[:,3])
    rmserror = np.sqrt(np.mean((meod[:,1] - meod[:,3])**2.0))/ppampl

    # store results:
    props = {}
    props['type'] = 'wave'
    props['EODf'] = freq0
    props['p-p-amplitude'] = ppampl
    props['flipped'] = flipped
    props['amplitude'] = popt[2]
    props['rmserror'] = rmserror
    ncols = 4
    if hasattr(freq, 'shape'):
        spec_data = np.zeros((n_harm, 5))
        ampls = np.sqrt(freq[:n_harm, 1])
        ampls /= ampls[0]
        spec_data[:len(ampls),4] = ampls
        props['power'] = decibel(np.sum(freq[:,1]))
    else:
        spec_data = np.zeros((n_harm, 4))
    spec_data[0,:4] = [0.0, freq0, 1.0, 0.0]
    for i in range(1, n_harm):
        spec_data[i,:4] = [i, (i+1)*freq0, popt[1+i*2], popt[2+i*2]]
    
    return meod, props, spec_data


def analyze_pulse(eod, eod_times, thresh_fac=0.01, min_dist=50.0e-6,
                  fresolution=1.0, percentile=1.0):
    """
    Analyze the EOD waveform of a pulse-type fish.
    
    Parameters
    ----------
    eod: 2-D array
        The eod waveform. First column is time in seconds, second column the eod waveform.
        Further columns are optional but not used.
    eod_times: 1-D array
        List of times of detected EOD peaks.
    thresh_fac: float
        Set the threshold for peak detection to the maximum pulse amplitude times this factor.
    min_dist: float
        Minimum distance between peak and troughs of the pulse.
    fresolution: float
        The frequency resolution of the power spectrum of the single pulse.
    percentile: float
        Remove extreme values of the inter-pulse intervals when computing interval statistics.
        All intervals below the `percentile` and above the `100-percentile` percentile
        are ignored. `percentile` is given in percent.
    
    Returns
    -------
    meod: 2-D array of floats
        The eod waveform. First column is time in seconds, second column the eod waveform.
        Further columns are kept from the input `eod`.
    props: dict
        A dictionary with properties of the analyzed EOD waveform.
        - type: set to 'pulse'.
        - EODf: the inverse of the mean interval between `eod_times`.
        - period: the mean interval between `eod_times`.
        - max-amplitude: the amplitude of the largest positive peak (P1).
        - min-amplitude: the amplitude of the largest negative peak (P1).
        - p-p-amplitude: peak-to-peak amplitude of the EOD waveform.
        - flipped: True if the waveform was flipped.
        - n: number of pulses analyzed.
        - medianinterval: the median interval between pulses after removal
          of extrem interval values.
        - meaninterval: the mean interval between pulses after removal
          of extrem interval values.
        - stdinterval: the standard deviation of the intervals between pulses
          after removal of extrem interval values.
    peaks: 2-D array
        For each peak and trough of the EOD waveform (first index) the
        peak index (1 is P1, i.e. the largest positive peak),
        time relative to largest positive peak, amplitude,
        and amplitude normalized to largest postive peak.
    power: 2-D array
        The power spectrum of a single pulse. First column are the frequencies,
        second column the power.
    intervals: 1-D array
        List of inter-EOD intervals with extreme values removed.
    """
        
    # storage:
    meod = np.zeros(eod.shape)
    meod[:,:] = eod
    
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
        if min_idx < max_idx:
            # flip:
            meod[:,1] = -meod[:,1]
            peak_idx = min_idx
            min_idx = max_idx
            max_idx = peak_idx
            flipped = True
    elif min_ampl > 0.2*amplitude:
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

    # amplitude:
    ppampl = max_ampl + min_ampl

    # threshold for peak detection:
    n = len(meod[:,1])*3//8
    thl_max = 0.0
    thl_min = 0.0
    if max_idx - n > 10:
        thl_max = np.max(meod[:max_idx-n,1])
        thl_min = np.min(meod[:max_idx-n,1])
    thr_max = 0.0
    thr_min = 0.0
    if len(meod[:,1]) - (max_idx + n) > 10:
        thr_max = np.max(meod[max_idx + n:,1])
        thr_min = np.min(meod[max_idx + n:,1])
    min_thresh = 1.2*(np.max([thl_max, thr_max]) - np.min([thl_min, thr_min]))
    threshold = max_ampl*thresh_fac
    if threshold < min_thresh:
        threshold = min_thresh
    
    # find smaller peaks:
    peak_idx, trough_idx = detect_peaks(meod[:,1], threshold)
    peak_l = np.sort(np.concatenate((peak_idx, trough_idx)))
    # remove mutliple peaks that do not oszillate around zero:
    peak_list = np.array([prange[np.argmax(np.abs(meod[prange,1]))]
                 for prange in np.split(peak_l, np.where(np.diff(meod[peak_l,1]>0.0)!=0)[0]+1)])
    # remove multiple peaks that are too close and too small:
    ridx = [(k, k+1) for k in np.where(np.diff(meod[peak_list,0]) < min_dist)]
    #  & (meod[peak_list[:-1],1]<0.1)
    peak_list = np.delete(peak_list, ridx)
    # find P1:
    p1i = np.where(peak_list == max_idx)[0][0]
    offs = 0 if p1i <= 2 else p1i - 2

    # store:
    peaks = np.zeros((len(peak_list)-offs,4))
    for i, pi in enumerate(peak_list[offs:]):
        peaks[i,:] = [i+1-p1i+offs, meod[pi,0], meod[pi,1], meod[pi,1]/max_ampl]

    # analyze pulse timing:
    inter_pulse_intervals = np.diff(eod_times)
    period = np.mean(inter_pulse_intervals)

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
    
    # store properties:
    props = {}
    props['type'] = 'pulse'
    props['EODf'] = 1.0/period
    props['period'] = period
    props['max-amplitude'] = max_ampl
    props['min-amplitude'] = min_ampl
    props['p-p-amplitude'] = ppampl
    props['peakfrequency'] = freqs[np.argmax(power)]
    props['lowfreqattenuation5'] = att5
    props['lowfreqattenuation50'] = att50
    props['powerlowcutoff'] = lowcutoff
    props['flipped'] = flipped
    props['n'] = len(eod_times)

    # analyze central intervals:    
    lower, upper = np.percentile(inter_pulse_intervals, [percentile, 100.0-percentile])
    intervals = inter_pulse_intervals[(inter_pulse_intervals > lower) &
                                      (inter_pulse_intervals < upper)]
    if len(intervals) > 2:
        props['medianinterval'] = np.median(intervals)
        props['meaninterval'] = np.mean(intervals)
        props['stdinterval'] = np.std(intervals, ddof=1)
    
    return meod, props, peaks, ppower, intervals


def eod_waveform_plot(eod_waveform, peaks, ax, unit=None,
                      mkwargs={'lw': 2, 'color': 'red'},
                      skwargs={'color': '#CCCCCC'},
                      fkwargs={'lw': 6, 'color': 'steelblue'}):
    """Plot mean eod and its standard deviation.

    Parameters
    ----------
    eod_waveform: 2-D array
        EOD waveform. First column is time in seconds,
        second column the (mean) eod waveform. The optional third column is the
        standard deviation and the optional fourth column is a fit on the waveform.
    peaks: 2_D arrays or None
        List of peak properties (index, time, and amplitude) of a EOD pulse
        as returned by `analyze_pulse()`.
    ax:
        Axis for plot.
    unit: string
        Optional unit of the data used for y-label.
    mkwargs: dict
        Arguments passed on to the plot command for the mean eod.
    skwargs: dict
        Arguments passed on to the fill_between command for the standard deviation of the eod.
    fkwargs: dict
        Arguments passed on to the plot command for the fitted eod.
    """
    ax.autoscale(True)
    time = 1000.0 * eod_waveform[:,0]
    # plot fit:
    if eod_waveform.shape[1] > 3:
        ax.plot(time, eod_waveform[:,3], zorder=2, **fkwargs)
    # plot waveform:
    mean_eod = eod_waveform[:,1]
    ax.plot(time, mean_eod, zorder=3, **mkwargs)
    # plot standard deviation:
    if eod_waveform.shape[1] > 2:
        ax.autoscale(False)
        std_eod = eod_waveform[:,2]
        ax.fill_between(time, mean_eod + std_eod, mean_eod - std_eod,
                        zorder=1, **skwargs)
    # annotate peaks:
    if peaks is not None and len(peaks)>0:
        maxa = np.max(peaks[:,2])
        for p in peaks:
            ax.scatter(1000.0*p[1], p[2], s=80, clip_on=False,
                       c=mkwargs['color'], edgecolors=mkwargs['color'])
            label = u'P%d' % p[0]
            if p[0] != 1:
                if p[1] < 0.001:
                    label += u'(%.0f%% @ %.0f\u00b5s)' % (100.0*p[3], 1.0e6*p[1])
                else:
                    label += u'(%.0f%% @ %.3gms)' % (100.0*p[3], 1.0e3*p[1])
            va = 'bottom' if p[2] > 0.0 else 'top'
            y = np.sign(p[2])*0.02*maxa
            if p[0] == 1 or p[0] == 2:
                va = 'bottom'
                y = 0.0
            dx = 0.05*time[-1]
            if p[1] >= 0.0:
                ax.text(1000.0*p[1]+dx, p[2]+y, label, ha='left', va=va)
            else:
                ax.text(1000.0*p[1]-dx, p[2]+y, label, ha='right', va=va)
    ax.set_xlabel('Time [msec]')
    if unit is not None and len(unit)>0:
        ax.set_ylabel('Amplitude [%s]' % unit)
    else:
        ax.set_ylabel('Amplitude')



def pulse_spectrum_plot(power, props, ax, color='b', lw=3, markersize=80):
    """Plot and annotate spectrum of single pulse.

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
    box = mpatches.Rectangle((1,-60), 49, 60, linewidth=0, facecolor='#DDDDDD')
    ax.add_patch(box)
    att = props['lowfreqattenuation50']
    ax.text(10.0, att+1.0, '%.0f dB' % att, ha='left', va='bottom')
    box = mpatches.Rectangle((1,-60), 4, 60, linewidth=0, facecolor='#CCCCCC')
    ax.add_patch(box)
    att = props['lowfreqattenuation5']
    ax.text(4.0, att+1.0, '%.0f dB' % att, ha='right', va='bottom')
    lowcutoff = props['powerlowcutoff']
    ax.plot([lowcutoff, lowcutoff, 1.0], [-60.0, 0.5*att, 0.5*att], '#BBBBBB')
    ax.text(1.2*lowcutoff, 0.5*att-1.0, '%.0f Hz' % lowcutoff, ha='left', va='top')
    db = decibel(power[:,1])
    smax = np.nanmax(db)
    ax.plot(power[:,0], db - smax, color, lw=lw)
    peakfreq = props['peakfrequency']
    ax.scatter([peakfreq], [0.0], c=color, edgecolors=color, s=markersize)
    ax.text(peakfreq*1.2, 1.0, '%.0f Hz' % peakfreq, va='bottom')
    ax.set_xlim(1.0, 10000.0)
    ax.set_xscale('log')
    ax.set_ylim(-60.0, 2.0)
    ax.set_xlabel('Frequency [Hz]')
    ax.set_ylabel('Power [dB]')

        
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
