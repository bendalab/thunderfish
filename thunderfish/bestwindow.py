"""
Select the best region within a recording with the most stable signal of largest amplitude that is not clipped.

## Main functions
- `clip_amplitudes()`: estimated clipping amplitudes from the data.
- `best_window_indices()`: select start- and end-indices of the best window
- `best_window_times()`: select start end end-time of the best window
- `best_window()`: return data of the best window
- `find_best_window()`: set clipping amplitudes and find best window.

## Configuration
- `add_clip_config()`: add parameters for `clip_amplitudes()` to configuration.
- `clip_args()`: retrieve parameters for `clip_amplitudes()` from configuration.
- `add_best_window_config()`: add parameters for `best_window()` to configuration.
- `best_window_args()`: retrieve parameters for `best_window*()` from configuration.

## Visualization
- `plot_clipping()`: visualization of the algorithm for detecting clipped amplitudes in `clip_amplitudes()`.
- `plot_best_window()`: visualization of the algorithm used in `best_window_indices()`.
- `plot_best_data()`: plot the data and the selected best window.
"""

import numpy as np
from .eventdetection import percentile_threshold, detect_peaks, trim_to_peak
from audioio.audioloader import unwrap
try:
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
except ImportError:
    pass


def clip_amplitudes(data, win_indices, min_fac=2.0, nbins=20,
                    min_ampl=-1.0, max_ampl=1.0,
                    plot_hist_func=None, **kwargs):
    """Find the amplitudes where the signal clips by looking at
    the histograms in data segements of win_indices length.
    If the bins at the edges are more than min_fac times as large as
    the neighboring bins, clipping at the bin's amplitude is assumed.

    Parameters
    ----------
    data: 1-D array
        The data.
    win_indices: int
        Size of the analysis window in indices.
    min_fac: float
        If the first or the second bin is at least `min_fac` times
        as large as the third bin, their upper bin edge is set as min_clip.
        Likewise for the last and next-to last bin.
    nbins: int
        Number of bins used for computing a histogram within `min_ampl` and `max_ampl`.
    min_ampl: float
        Minimum to be expected amplitude of the data.
    max_ampl: float
        Maximum to be expected amplitude of the data
    plot_hist_func: function
        Function for visualizing the histograms, is called for every window.
        `plot_clipping()` is a simple function that can be passed as `plot_hist_func`
        to quickly visualize what is going on in `clip_amplitudes()`.
        
        Signature:

        `plot_hist_func(data, winx0, winx1, bins, h, min_clip, max_clip,
        min_ampl, max_ampl, kwargs)`

        with the arguments:
        
        - `data` (array): the full data array.
        - `winx0` (int): the start index of the current window.
        - `winx1` (int): the end index of the current window.
        - `bins` (array): the bin edges of the histogram.
        - `h` (array): the histogram, plot it with
           ```
           plt.bar(bins[:-1], h, width=np.mean(np.diff(bins)))
           ```
        - `min_clip` (float): the current value of the minimum clip amplitude.
        - `max_clip` (float): the current value of the minimum clip amplitude.
        - `min_ampl` (float): the minimum amplitude of the data.
        - `max_ampl` (float): the maximum amplitude of the data.
        - `kwargs` (dict): further user supplied key-word arguments.

    Returns
    -------
      min_clip: float
          Minimum amplitude that is not clipped.
      max_clip: float
          Maximum amplitude that is not clipped.
    """

    min_clipa = min_ampl
    max_clipa = max_ampl
    bins = np.linspace(min_ampl, max_ampl, nbins, endpoint=True)
    win_tinxs = np.arange(0, len(data) - win_indices, win_indices)
    for wtinx in win_tinxs:
        h, b = np.histogram(data[wtinx:wtinx + win_indices], bins)
        if h[0] > min_fac * h[2] and b[0] < 0.4*min_ampl:
            if h[1] > min_fac * h[2] and b[2] > min_clipa:
                min_clipa = b[2]
            elif b[1] > min_clipa:
                min_clipa = b[1]
        if h[-1] > min_fac * h[-3] and b[-1] > 0.4*max_ampl:
            if h[-2] > min_fac * h[-3] and b[-3] < max_clipa:
                max_clipa = b[-3]
            elif b[-2] < max_clipa:
                max_clipa = b[-2]
        if plot_hist_func:
            plot_hist_func(data, wtinx, wtinx + win_indices,
                           b, h, min_clipa, max_clipa,
                           min_ampl, max_ampl, **kwargs)
    return min_clipa, max_clipa


def plot_clipping(data, winx0, winx1, bins,
                  h, min_clip, max_clip, min_ampl, max_ampl):
    """Visualize the data histograms and the detected clipping amplitudes.
    Pass this function as the `plot_hist_func` argument to `clip_amplitudes()`.
    """
    plt.subplot(2, 1, 1)
    plt.plot(data[winx0:winx1], 'b')
    plt.axhline(min_clip, color='r')
    plt.axhline(max_clip, color='r')
    plt.ylim(-1.0, 1.0)
    plt.subplot(2, 1, 2)
    plt.bar(bins[:-1], h, width=np.mean(np.diff(bins)))
    plt.axvline(min_clip, color='r')
    plt.axvline(max_clip, color='r')
    plt.xlim(-1.0, 1.0)
    plt.show()


def add_clip_config(cfg, min_clip=0.0, max_clip=0.0,
                    window=1.0, min_fac=2.0, nbins=20,
                    min_ampl=-1.0, max_ampl=1.0):
    """ Add parameter needed for `clip_amplitudes()` as
    a new section to a configuration.

    Parameters
    ----------
    cfg: ConfigFile
        The configuration.
    min_clip: float
        Default minimum clip amplitude.
    max_clip: float
        Default maximum clip amplitude.
        
    See `clip_amplitudes()` for details on the remaining arguments.
    """

    cfg.add_section('Clipping amplitudes:')
    cfg.add('minClipAmplitude', min_clip, '', 'Minimum amplitude that is not clipped. If zero estimate from data.')
    cfg.add('maxClipAmplitude', max_clip, '', 'Maximum amplitude that is not clipped. If zero estimate from data.')
    cfg.add('clipWindow', window, 's', 'Window size for estimating clip amplitudes.')
    cfg.add('clipBins', nbins, '', 'Number of bins used for constructing histograms of signal amplitudes.')
    cfg.add('minClipFactor', min_fac, '',
            'Edge bins of the histogram of clipped signals have to be larger then their neighbors by this factor.')
    cfg.add('minDataAmplitude', min_ampl, '', 'Minimum amplitude that is to be expected  in the data.')
    cfg.add('maxDataAmplitude', max_ampl, '', 'Maximum amplitude that is to be expected  in the data.')


def clip_args(cfg, rate):
    """ Translates a configuration to the
    respective parameter names of the function `clip_amplitudes()`.
    The return value can then be passed as key-word arguments to this function.

    Parameters
    ----------
    cfg: ConfigFile
        The configuration.
    rate: float
        The sampling rate of the data.

    Returns
    -------
    a: dict
        Dictionary with names of arguments of the `clip_amplitudes()` function
        and their values as supplied by `cfg`.
    """
    a = cfg.map({'min_fac': 'minClipFactor',
                 'nbins': 'clipBins',
                 'min_ampl': 'minDataAmplitude',
                 'max_ampl': 'maxDataAmplitude'})
    a['win_indices'] = int(cfg.value('clipWindow') * rate)
    return a


def best_window_indices(data, samplerate, expand=False, win_size=1., win_shift=0.5,
                        thresh_fac=0.8, percentile=0.1, min_clip=-np.inf, max_clip=np.inf,
                        w_cv_interv=1.0, w_ampl=1.0, w_cv_ampl=1.0, tolerance=0.2,
                        plot_data_func=None, **kwargs):
    """Find the window within data most suitable for subsequent analysis.
    
    First, large peaks and troughs of the data are detected.  Peaks and
    troughs have to be separated in amplitude by at least the value of a
    dynamic threshold.  The threshold is computed in `win_shift` wide
    windows as `thresh_fac` times the interpercentile range at
    the `percentile`-th and 100.0-`percentile`-th percentile of the data
    using the `eventdetection.percentile_threshold()` function.

    Second, criteria for selecting the best window are computed for each
    window of width `win_size` shifted by `win_shift` trough the data. The
    three criteria are:

    - the mean peak-to-trough amplitude multiplied with the fraction of
      non clipped peak and trough amplitudes.
    - the coefficient of variation of the peak-to-trough amplitude.
    - the coefficient of variation of the inter-peak and inter-trough
      intervals.

    Third, a cost function is computed as a weighted sum of the three
    criteria (the mean amplitude is taken negatively). The respective
    weights are given by `w_ampl`, `w_cv_ampl`, and `w_cv_interv`.

    Finally, a threshold is set to the minimum value of the cost
    function plus tolerance.  Then the largest region with the cost
    function below this threshold is selected as the best window.  If
    `expand` is `False`, then only the single window with smallest cost
    within the selected largest region is returned.

    The data used by best window algorithm can be visualized by supplying the
    function `plot_data_func`.  Additional arguments for this function can
    be supplied via key-word arguments `kwargs`.

    Parameters
    ----------
    data: 1-D array
        The data to be analyzed.
    samplerate: float
        Sampling rate of the data in Hertz.
    expand: boolean
        If `False` return only the single window with the smallest cost.
        If `True` return the largest window with the cost below the minimum cost
        plus tolerance.
    win_size: float
        Minimum size of the desired best window in seconds.
        Choose it large enough for the subsequent analysis.
    win_shift: float
        Time shift in seconds between windows. Should be smaller or equal to `win_size`.
    percentile: float
        `percentile` parameter for the `eventdetection.percentile_threshold()` function
        used to estimate thresholds for detecting peaks in the data.
    thresh_fac: float
        `thresh_fac` parameter for the `eventdetection.percentile_threshold()` function
        used to estimate thresholds for detecting peaks in the data.
    min_clip: float
        Minimum amplitude below which data are clipped.
    max_clip: float
        Maximum amplitude above which data are clipped.
    w_cv_interv: float
        Weight for the coefficient of variation of the intervals between detected
        peaks and throughs.
    w_ampl: float
        Weight for the mean peak-to-trough amplitude.
    w_cv_ampl: float
        Weight for the coefficient of variation of the amplitudes.
    tolerance: float
        Added to the minimum cost for expanding the region of the best window.
    plot_data_func: function
        Function for plotting the raw data, detected peaks and troughs, the criteria,
        the cost function and the selected best window.
        `plot_best_window()` is a simple function that can be passed as the `plot_data_func`
        parameter to quickly visualize what is going on in selecting the best window.
        
        Signature:
        
        ````
        plot_data_func(data, rate, peak_thresh, peak_idx, trough_idx, idx0, idx1,
                       win_start_times, cv_interv, mean_ampl, cv_ampl, clipped_frac, cost_thresh,
                       thresh, valid_wins, **kwargs)
        ```

        with the arguments:
        
        - `data` (array): raw data.
        - `rate` (float): sampling rate of the data.
        - `peak_thresh` (array): thresholds used for detecting peaks and troughs in each data window.
        - `peak_idx` (array): indices into raw data indicating detected peaks.
        - `trough_idx` (array): indices into raw data indicating detected troughs.
        - `idx0` (int): index of the start of the best window.
        - `idx1` (int): index of the end of the best window.
        - `win_start_times` (array): times of the analysis windows.
        - `cv_interv` (array): coefficients of variation of the inter-peak and -trough
           intervals.
        - `mean_ampl` (array): mean peak-to-trough amplitudes.
        - `cv_ampl` (array): coefficients of variation of the peak-to-trough amplitudes.
        - `clipped_frac` (array): fraction of clipped peaks or troughs.
        - `cost` (array): cost function.
        - `cost_thresh` (float): threshold for the cost function.
        - `valid_wins` (array): boolean array indicating the windows which fulfill
          all three criteria.
        - `**kwargs` (dict): further user supplied key-word arguments.
    kwargs: dict
        Keyword arguments passed to `plot_data_func`. 
    
    Returns
    -------
    start_index: int
        Index of the start of the best window.
    end_index: int
        Index of the end of the best window.
    clipped: float.
        The fraction of clipped peaks or troughs.
    """

    # too little data:
    if len(data) / samplerate < win_size:
        raise UserWarning('not enough data (data=%gs, win=%gs)' %
                          (len(data) / samplerate, win_size))

    # threshold for peak detection:
    threshold = percentile_threshold(data, samplerate, win_shift,
                                     thresh_fac=thresh_fac,
                                     percentile=percentile)

    # detect large peaks and troughs:
    peak_idx, trough_idx = detect_peaks(data, threshold)
    if len(peak_idx) == 0 or len(trough_idx) == 0:
        raise UserWarning('no peaks or troughs detected')

    # compute cv of intervals, mean peak amplitude and its cv:
    invalid_cv = 1000.0
    win_size_indices = int(win_size * samplerate)
    win_start_inxs = np.arange(0, len(data) - win_size_indices,
                               int(0.5*win_shift*samplerate))
    if len(win_start_inxs) == 0:
        win_start_inxs = [0]
    cv_interv = np.zeros(len(win_start_inxs))
    mean_ampl = np.zeros(len(win_start_inxs))
    cv_ampl = np.zeros(len(win_start_inxs))
    clipped_frac = np.zeros(len(win_start_inxs))
    for i, wtinx in enumerate(win_start_inxs):
        # indices of peaks and troughs inside analysis window:
        pinx = (peak_idx >= wtinx) & (peak_idx <= wtinx + win_size_indices)
        tinx = (trough_idx >= wtinx) & (trough_idx <= wtinx + win_size_indices)
        p_idx, t_idx = trim_to_peak(peak_idx[pinx], trough_idx[tinx])
        # interval statistics:
        ipis = np.diff(p_idx)
        itis = np.diff(t_idx)
        if len(ipis) > 2:
            cv_interv[i] = 0.5 * (np.std(ipis) / np.mean(ipis) + np.std(itis) / np.mean(itis))
            # penalize regions without detected peaks:
            mean_interv = np.mean(ipis)
            if p_idx[0] - wtinx > mean_interv:
                cv_interv[i] *= (p_idx[0] - wtinx) / mean_interv
            if wtinx + win_size_indices - p_idx[-1] > mean_interv:
                cv_interv[i] *= (wtinx + win_size_indices - p_idx[-1]) / mean_interv
        else:
            cv_interv[i] = invalid_cv
        # statistics of peak-to-trough amplitude:
        p2t_ampl = data[p_idx] - data[t_idx]
        if len(p2t_ampl) > 2:
            mean_ampl[i] = np.mean(p2t_ampl)
            cv_ampl[i] = np.std(p2t_ampl) / mean_ampl[i]
            # penalize for clipped peaks:
            clipped_frac[i] = float(np.sum(data[p_idx] > max_clip) +
                                    np.sum(data[t_idx] < min_clip)) / 2.0 / len(p2t_ampl)
            mean_ampl[i] *= (1.0 - clipped_frac[i]) ** 2.0
        else:
            mean_ampl[i] = 0.0
            cv_ampl[i] = invalid_cv

    # check:
    if len(mean_ampl[mean_ampl >= 0.0]) < 0:
        raise UserWarning('no finite amplitudes detected')
    if len(cv_interv[cv_interv < invalid_cv]) <= 0:
        raise UserWarning('no valid interval cv detected')
    if len(cv_ampl[cv_ampl < invalid_cv]) <= 0:
        raise UserWarning('no valid amplitude cv detected')

    # cost function:
    cost = w_cv_interv * cv_interv + w_cv_ampl * cv_ampl - w_ampl * mean_ampl
    thresh = np.min(cost) + tolerance

    # find largest region with low costs:
    valid_win_idx = np.nonzero(cost <= thresh)[0]
    cidx0 = valid_win_idx[0]  # start of current window
    cidx1 = cidx0 + 1  # end of current window
    win_idx0 = cidx0   # start of largest window
    win_idx1 = cidx1   # end of largest window
    i = 1
    while i < len(valid_win_idx):  # loop through all valid window positions
        if valid_win_idx[i] == valid_win_idx[i - 1] + 1:
            cidx1 = valid_win_idx[i] + 1
        else:
            cidx0 = valid_win_idx[i]
        if cidx1 - cidx0 > win_idx1 - win_idx0:  # current window is largest
            win_idx0 = cidx0
            win_idx1 = cidx1
        i += 1

    # find single best window within the largest region:
    if not expand:
        win_idx0 += np.argmin(cost[win_idx0:win_idx1])
        win_idx1 = win_idx0 + 1

    # retrive indices of best window for data:
    idx0 = win_start_inxs[win_idx0]
    idx1 = win_start_inxs[win_idx1 - 1] + win_size_indices

    # clipped data?
    clipped = np.mean(clipped_frac[win_idx0:win_idx1])

    if plot_data_func:
        plot_data_func(data, samplerate, threshold, peak_idx, trough_idx, idx0, idx1,
                       win_start_inxs / samplerate, cv_interv, mean_ampl, cv_ampl, clipped_frac,
                       cost, thresh, win_idx0, win_idx1, **kwargs)

    return idx0, idx1, clipped


def best_window_times(data, samplerate, expand=False, win_size=1., win_shift=0.5,
                      thresh_fac=0.8, percentile=0.1, min_clip=-np.inf, max_clip=np.inf,
                      w_cv_interv=1.0, w_ampl=1.0, w_cv_ampl=1.0, tolerance=0.2,
                      plot_data_func=None, **kwargs):
    """Find the window within data most suitable for subsequent analysis.

    See `best_window_indices()` for details.

    Returns
    -------
    start_time: float
        Time of the start of the best window.
    end_time: float
        Time of the end of the best window.
    clipped: float
        The fraction of clipped peaks or troughs.
    """
    start_inx, end_inx, clipped = best_window_indices(data, samplerate, expand,
                                                      win_size, win_shift,
                                                      thresh_fac, percentile,
                                                      min_clip, max_clip,
                                                      w_cv_interv, w_ampl, w_cv_ampl, tolerance,
                                                      plot_data_func, **kwargs)
    return start_inx / samplerate, end_inx / samplerate, clipped


def best_window(data, samplerate, expand=False, win_size=1., win_shift=0.5,
                thresh_fac=0.8, percentile=0.1, min_clip=-np.inf, max_clip=np.inf,
                w_cv_interv=1.0, w_ampl=1.0, w_cv_ampl=1.0, tolerance=0.2,
                plot_data_func=None, **kwargs):
    """Find the window within data most suitable for subsequent analysis.

    See `best_window_indices()` for details.

    Returns
    -------
    data: array
        The data of the best window.
    clipped: float
        The fraction of clipped peaks or troughs.
    """
    start_inx, end_inx, clipped = best_window_indices(data, samplerate, expand,
                                                      win_size, win_shift,
                                                      thresh_fac, percentile,
                                                      min_clip, max_clip,
                                                      w_cv_interv, w_ampl, w_cv_ampl,
                                                      tolerance, plot_data_func, **kwargs)
    return data[start_inx:end_inx], clipped


def plot_best_window(data, rate, threshold, peak_idx, trough_idx, idx0, idx1,
                     win_times, cv_interv, mean_ampl, cv_ampl, clipped_frac,
                     cost, thresh, win_idx0, win_idx1, ax):
    """Visualize the cost function of used for finding the best window for analysis.

    Pass this function as the `plot_data_func` to the `best_window_*` functions.

    Parameters
    ----------
    See documentation of the `best_window_indices()` functions.
    """
    # raw data:
    time = np.arange(0.0, len(data)) / rate
    ax[0].plot(time, data, 'b', lw=3)
    if np.mean(clipped_frac[win_idx0:win_idx1]) > 0.01:
        ax[0].plot(time[idx0:idx1], data[idx0:idx1], color='magenta', lw=3)
    else:
        ax[0].plot(time[idx0:idx1], data[idx0:idx1], color='grey', lw=3)
    ax[0].plot(time[peak_idx], data[peak_idx], 'o', mfc='red', ms=6)
    ax[0].plot(time[trough_idx], data[trough_idx], 'o', mfc='green', ms=6)
    ax[0].plot(time, threshold, '#CCCCCC', lw=2)
    up_lim = np.max(data) * 1.05
    down_lim = np.min(data) * .95
    ax[0].set_ylim((down_lim, up_lim))
    ax[0].set_ylabel('Amplitude [a.u]')

    # cv of inter-peak intervals:
    ax[1].plot(win_times[cv_interv < 1000.0], cv_interv[cv_interv < 1000.0], 'o', ms=10, color='grey', mew=2.,
               mec='black', alpha=0.6)
    ax[1].plot(win_times[win_idx0:win_idx1], cv_interv[win_idx0:win_idx1], 'o', ms=10, color='red', mew=2., mec='black',
               alpha=0.6)
    ax[1].set_ylabel('CV intervals')
    ax[1].set_ylim(bottom=0.0)

    # mean amplitude:
    ax[2].plot(win_times[mean_ampl > 0.0], mean_ampl[mean_ampl > 0.0], 'o', ms=10, color='grey', mew=2., mec='black',
               alpha=0.6)
    ax[2].plot(win_times[win_idx0:win_idx1], mean_ampl[win_idx0:win_idx1], 'o', ms=10, color='red', mew=2., mec='black',
               alpha=0.6)
    ax[2].set_ylabel('Mean amplitude [a.u]')
    ax[2].set_ylim(bottom=0.0)

    # cv:
    ax[3].plot(win_times[cv_ampl < 1000.0], cv_ampl[cv_ampl < 1000.0], 'o', ms=10, color='grey', mew=2., mec='black',
               alpha=0.6)
    ax[3].plot(win_times[win_idx0:win_idx1], cv_ampl[win_idx0:win_idx1], 'o', ms=10, color='red', mew=2., mec='black',
               alpha=0.6)
    ax[3].set_ylabel('CV amplitude')
    ax[3].set_ylim(bottom=0.0)

    # cost:
    ax[4].plot(win_times[cost < thresh + 10], cost[cost < thresh + 10], 'o', ms=10, color='grey', mew=2., mec='black',
               alpha=0.6)
    ax[4].plot(win_times[win_idx0:win_idx1], cost[win_idx0:win_idx1], 'o', ms=10, color='red', mew=2., mec='black',
               alpha=0.6)
    ax[4].axhline(thresh, color='k')
    ax[4].set_ylabel('Cost')
    ax[4].set_xlabel('Time [sec]')


def plot_best_data(ax, data, samplerate, unit, idx0, idx1, clipped,
                   data_color='blue', window_color='red'):
    """Plot the data and mark the best window.

    Parameters
    ----------
    ax: matplotlib axes
        Axes used for plotting.
    data: 1-D array
        The full data trace.
    samplerate: float
        Sampling rate of the data in Hertz.
    unit: string
        The unit of the data.
    idx0: int
        Start index of the best window.
    idx1: int
        Stop index of the best window.
    clipped: float
        Fraction of clipped peaks.
    data_color:
        Color used for plotting the data trace.
    window_color:
        Color used for plotting the selected best window.
    """
    time = np.arange(len(data)) / samplerate
    ax.plot(time[:idx0], data[:idx0], color=data_color)
    ax.plot(time[idx1:], data[idx1:], color=data_color)
    if idx1 > idx0:
        ax.plot(time[idx0:idx1], data[idx0:idx1], color=window_color)
        label = 'analysis\nwindow'
        if clipped > 0.0:
            label += '\n%.0f%% clipped' % (100.0*clipped)
        ax.text(time[(idx0+idx1)//2], 0.0, label, ha='center', va='center')
    ax.set_xlim(time[0], time[-1])
    ax.set_xlabel('Time [sec]')
    if len(unit) == 0 or unit == 'a.u.':
        ax.set_ylabel('Amplitude')
    else:
        ax.set_ylabel('Amplitude [%s]' % unit)
    ax.yaxis.set_major_locator(ticker.MaxNLocator(3))

        
def add_best_window_config(cfg, expand=False, win_size=1., win_shift=0.5,
                           thresh_fac=0.8, percentile=0.1,
                           min_clip=-np.inf, max_clip=np.inf,
                           w_cv_interv=1.0, w_ampl=1.0, w_cv_ampl=1.0,
                           tolerance=0.2):
    """ Add parameter needed for the `best_window()` functions as
    a new section to a configuration dictionary.

    Parameters
    ----------
    cfg: ConfigFile
        The configuration.
        
    See `best_window_indices()` for details on the remaining arguments.
    """

    cfg.add_section('Best window detection:')
    cfg.add('bestWindowSize', win_size, 's', 'Size of the best window. This should be much larger than the expected period of the signal. If 0 select the whole time series.')
    cfg.add('bestWindowShift', win_shift, 's',
            'Increment for shifting the analysis windows trough the data. Should be larger than the expected period of the signal.')
    cfg.add('bestWindowThresholdPercentile', percentile, '%',
            'Percentile for estimating interpercentile range. Should be smaller than the duty cycle of the periodic signal.')
    cfg.add('bestWindowThresholdFactor', thresh_fac, '',
            'Threshold for detecting peaks is interperecntile range of the data times this factor.')
    cfg.add('weightCVInterval', w_cv_interv, '',
            'Weight factor for the coefficient of variation of the inter-peak and inter-trough intervals.')
    cfg.add('weightAmplitude', w_ampl, '',
            'Weight factor for the mean peak-to-trough amplitudes.')
    cfg.add('weightCVAmplitude', w_cv_ampl, '',
            'Weight factor for the coefficient of variation of the peak-to-trough amplitude.')
    cfg.add('bestWindowTolerance', tolerance, '',
            'Add this to the minimum value of the cost function to get a threshold for selecting the largest best window.')
    cfg.add('expandBestWindow', expand, '',
            'Return the largest valid best window. If False return sole best window. ')


def best_window_args(cfg):
    """ Translates a configuration to the
    respective parameter names of the functions `best_window*()`.
    The return value can then be passed as key-word arguments to these functions.

    Parameters
    ----------
    cfg: ConfigFile
        The configuration.

    Returns
    -------
    a: dict
        Dictionary with names of arguments of the `best_window*()` functions
        and their values as supplied by `cfg`.
    """
    return cfg.map({'win_size': 'bestWindowSize',
                    'win_shift': 'bestWindowShift',
                    'percentile': 'bestWindowThresholdPercentile',
                    'thresh_fac': 'bestWindowThresholdFactor',
                    'w_cv_interv': 'weightCVInterval',
                    'w_ampl': 'weightAmplitude',
                    'w_cv_ampl': 'weightCVAmplitude',
                    'tolerance': 'bestWindowTolerance',
                    'expand': 'expandBestWindow'})

        
def find_best_window(raw_data, samplerate, cfg, show_bestwindow=False):
    """
    Set clipping amplitudes and find best window.

    Parameters
    ----------
    data: 1-D array
        The data to be analyzed.
    samplerate: float
        Sampling rate of the data in Hertz.
    cfg: ConfigFile
        Configuration for clipping and best window.
    show_bestwindow: boolean
        If true show a plot with the best window cost functions.

    Returns
    -------
    data: 1-D array
        The data array of the best window
    idx0: int
        The start index of the best window in the original data.
    idx1: int
        The end index of the best window in the original data.
    clipped: float
        Fraction of clipped amplitudes.
    min_clip: float
        Minimum amplitude that is not clipped.
    max_clip: float
        Maximum amplitude that is not clipped.
    """
    found_bestwindow = True
    min_clip = cfg.value('minClipAmplitude')
    max_clip = cfg.value('maxClipAmplitude')
    clipped = 0
    if min_clip == 0.0 or max_clip == 0.0:
        min_clip, max_clip = clip_amplitudes(raw_data, **clip_args(cfg, samplerate))
    if cfg.value('unwrapData'):
        raw_data = unwrap(raw_data)
        min_clip *= 2
        max_clip *= 2
    # best window size parameter:
    bwa = best_window_args(cfg)
    if 'win_size' in bwa:
        del bwa['win_size']
    best_window_size = cfg.value('bestWindowSize')
    if best_window_size <= 0.0:
        best_window_size = (len(raw_data)-1)/samplerate
    # show cost function:
    if show_bestwindow:
        fig, ax = plt.subplots(5, sharex=True, figsize=(14., 10.))
        try:
            idx0, idx1, clipped = best_window_indices(raw_data, samplerate,
                                                      min_clip=min_clip, max_clip=max_clip,
                                                      win_size=best_window_size,
                                                      plot_data_func=plot_best_window, ax=ax,
                                                      **bwa)
            plt.show()
        except UserWarning as e:
            found_bestwindow = False
    else:
        try:
            idx0, idx1, clipped = best_window_indices(raw_data, samplerate,
                                                      min_clip=min_clip,
                                                      max_clip=max_clip,
                                                      win_size=best_window_size,
                                                      **bwa)
        except UserWarning as e:
            found_bestwindow = False
    if found_bestwindow:
        return raw_data[idx0:idx1], idx0, idx1, clipped, min_clip, max_clip
    else:
        return raw_data, 0, 0, False, min_clip, max_clip


if __name__ == "__main__":
    print("Checking bestwindow module ...")
    import sys

    title = "bestwindow"
    if len(sys.argv) < 2:
        # generate data:
        print("generate waveform...")
        rate = 100000.0
        time = np.arange(0.0, 1.0, 1.0 / rate)
        f = 600.0
        snippets = []
        amf = 20.0
        for ampl in [0.2, 0.5, 0.8]:
            for am_ampl in [0.0, 0.3, 0.9]:
                data = ampl * np.sin(2.0 * np.pi * f * time) * (1.0 + am_ampl * np.sin(2.0 * np.pi * amf * time))
                data[data > 1.3] = 1.3
                data[data < -1.3] = -1.3
                snippets.extend(data)
        data = np.asarray(snippets)
        title = "test sines"
        data += 0.01 * np.random.randn(len(data))
    else:
        from .dataloader import load_data

        print("load %s ..." % sys.argv[1])
        data, rate, unit = load_data(sys.argv[1], 0)
        title = sys.argv[1]

    # determine clipping amplitudes:
    clip_win_size = 0.5
    min_clip_fac = 2.0
    min_clip, max_clip = clip_amplitudes(data, int(clip_win_size * rate),
                                         min_fac=min_clip_fac)
    # min_clip, max_clip = clip_amplitudes(data, int(clip_win_size*rate),
    #                                      min_fac=min_clip_fac,
    #                                      plot_hist_func=plot_clipping)

    # setup plots:
    fig, ax = plt.subplots(5, sharex=True, figsize=(20, 12))
    fig.canvas.set_window_title(title)

    # compute best window:
    print("call bestwindow() function...")
    best_window_indices(data, rate, expand=False,
                        win_size=4.0, win_shift=0.5, thresh_fac=0.8, percentile=0.1,
                        min_clip=min_clip, max_clip=max_clip,
                        w_cv_ampl=10.0, tolerance=0.5,
                        plot_data_func=plot_best_window, ax=ax)

    plt.tight_layout()
    plt.show()
