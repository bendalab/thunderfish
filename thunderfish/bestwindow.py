# the best window detector as functions...

def long_enough(rate, data, minduration=10.0):
    """
    Returns:
      true if the data are longer than minduration
    """
    return = len(data) / rate <= minduration


def peakdet(v, delta, x=None):
    """
    Converted from MATLAB script at http://billauer.co.il/peakdet.html
    Returns two arrays
    function [maxtab, mintab]=peakdet(v, delta, x)
    %PEAKDET Detect peaks in a vector
    % [MAXTAB, MINTAB] = PEAKDET(V, DELTA) finds the local
    % maxima and minima ("peaks") in the vector V.
    % MAXTAB and MINTAB consists of two columns. Column 1
    % contains indices in V, and column 2 the found values.
    %
    % With [MAXTAB, MINTAB] = PEAKDET(V, DELTA, X) the indices
    % in MAXTAB and MINTAB are replaced with the corresponding
    % X-values.
    %
    % A point is considered a maximum peak if it has the maximal
    % value, and was preceded (to the left) by a value lower by
    % DELTA.
    % Eli Billauer, 3.4.05 (Explicitly not copyrighted).
    % This function is released to the public domain; Any use is allowed.
    """
    maxtab = []
    maxidx = []

    mintab = []
    minidx = []

    if x is None:
        x = np.arange(len(v), dtype=int)

    v = np.asarray(v)

    if len(v) != len(x):
        sys.exit('Input vectors v and x must have same length')

    if not np.isscalar(delta):
        sys.exit('Input argument delta must be a scalar')

    if delta <= 0:
        sys.exit('Input argument delta must be positive')

    mn, mx = np.Inf, -np.Inf
    mnpos, mxpos = np.NaN, np.NaN

    lookformax = True

    for i in np.arange(len(v)):
        this = v[i]
        if this > mx:
            mx = this
            mxpos = x[i]

        if this < mn:
            mn = this
            mnpos = x[i]


        if lookformax:
            if this < mx-delta:
                maxtab.append(mx)
                maxidx.append(mxpos)
                mn = this
                mnpos = x[i]
                lookformax = False
        else:
            if this > mn+delta:
                mintab.append(mn)
                minidx.append(mnpos)
                mx = this
                mxpos = x[i]
                lookformax = True

    return np.array(maxtab), np.array(maxidx), np.array(mintab), np.array(minidx)


def detect_peak_and_trough_indices(self, peak_threshold=None, norm_window=.1):
    """This function finds the indices of peaks and troughs of each EOD-cycle in the recording

    :param peak_threshold: This is the threshold to be used for the peakdet function (Translated Matlab-Code...).
    :param norm_window:
    :return: two arrays. The first contains the peak indices and the second contains the trough indices.
    """
    if self._eod_peak_idx is None or self._eod_trough_idx is None:

        w = np.ones(self._sample_rate*norm_window)
        w[:] /= len(w)
        eod2 = np.sqrt(np.correlate(self._eod**2., w, mode='same') - np.correlate(self._eod, w, mode='same')**2.)
        eod2 = self._eod / eod2
        if peak_threshold is None:
            peak_threshold = np.percentile(np.abs(eod2), 99.9)-np.percentile(np.abs(eod2), 70)
          # The Threshold is 1.5 times the standard deviation of the eod

        _, self._eod_peak_idx, _, self._eod_trough_idx = aux.peakdet(eod2, peak_threshold)
        # refine by matching troughs and peaks
        everything = list(self.peak_trough_iterator())
        _, _, self._eod_peak_idx, _, _, self._eod_trough_idx = map(lambda x: np.asarray(x), zip(*everything))

    return self._eod_peak_idx, self._eod_trough_idx


def best_window_algorithm(self, peak_no, mean_amplitudes, cvs, pks_th=0.15, ampls_percentile_th=85.,
                          cvs_percentile_th=15., plot_debug=False, axs=None, win_shift = 0.2):

    """This is the algorithm that chooses the best window. It first filters out the windows that have a siginificant
    different amount of peaks compared to the stats.mode peak number of all windows. Secondly, it filters out
    windows that have a higher coefficient of variation than a certain percentile of the distribution of cvs.
    From those windows that get through both filters, the one with the highest peak-to-trough-amplitude
    (that is not clipped!) is chosen as the best window. We assume clipping as amplitudes above 85% percentile of
    the distribution of peak to trough amplitude.

    :param cvs_percentile_th: threshold of how much amplitude-variance (covariance coefficient) of the signal
    is allowed. Default is 10%.
    :param peak_no: array with number of peaks
    :param mean_amplitudes: array with mean peak-to-trough-amplitudes
    :param cvs: array with covariance coefficients
    :param pks_th: threshold for number-of-peaks-filter
    :param ampls_percentile_th: choose a percentile threshold to avoid clipping. Default is 85
    :param plot_debug: boolean for showing plot-debugging.
    :param axs: axis of plot debugging.
    :param win_shift: float. Size in seconds between windows. Default is 0.2 seconds.
    :return: boolean array with a single True element. This is the Index of the best window out of all windows.
    """
    # First filter: Stable # of detected peaks
    pk_mode = stats.mode(peak_no)
    tot_pks = max(peak_no)-min(peak_no)
    lower = peak_no >= pk_mode[0][0] - tot_pks*pks_th
    upper = peak_no <= pk_mode[0][0] + tot_pks*pks_th
    valid_pks = lower * upper

    # Second filter: Low variance in the amplitude
    cvs[np.isnan(cvs)] = np.median(cvs)  # Set a huge number where NaN to avoid NaN!!
    cov_th = np.percentile(cvs, cvs_percentile_th)
    valid_cv = cvs < cov_th

    # Third filter: From the remaining windows, choose the one with the highest p2t_amplitude that's not clipped.

    # replace the median ampl where 0's
    ampl_means = np.where(mean_amplitudes == 0., np.median(mean_amplitudes), mean_amplitudes)
    tot_ampls = max(ampl_means)-min(ampl_means)
    ampls_th = np.percentile(ampl_means, ampls_percentile_th)
    valid_ampls = ampl_means <= ampls_th

    valid_windows = valid_pks * valid_cv * valid_ampls

    # If there is no best window, run the algorithm again with more flexible threshodlds.
    if not True in valid_windows and cvs_percentile_th == 100. and ampls_percentile_th == 0.:
        print('\nWARNING. The recording %s seems to be of very bad quality for analysis. '
              'Skipping recording...\n' % self._wavfile)
        os.remove(self._wavfile)
        quit()

    elif not True in valid_windows:
        print('\nNo best window found. Rerunning best_window_algorithm with more flexible arguments.\n')
        if cvs_percentile_th <= 95.:
            cvs_percentile_th += 5.
        if ampls_percentile_th >= 5.:
            ampls_percentile_th -= 5.
        return self.best_window_algorithm(peak_no, mean_amplitudes, cvs,
                                          ampls_percentile_th=ampls_percentile_th,
                                          cvs_percentile_th=cvs_percentile_th, plot_debug=plot_debug,
                                          win_shift=win_shift)
        # This return is a Recursion! Need to return the value in the embeded function, otherwise the root_function
        # will not return anything!

    else:
        max_ampl_window = ampl_means == np.max(ampl_means[valid_windows])  # Boolean array with a single True element

        best_window = valid_windows * max_ampl_window

        if plot_debug:

            windows = np.arange(len(peak_no)) * win_shift
            up_th = np.ones(len(windows)) * pk_mode[0][0] + tot_pks*pks_th
            down_th = np.ones(len(windows)) * pk_mode[0][0] - tot_pks*pks_th
            axs[1].fill_between(windows, y1=down_th, y2=up_th, color='forestgreen', alpha=0.4, edgecolor='k', lw=1)

            cvs_th_array = np.ones(len(windows)) * cov_th
            axs[2].fill_between(windows, y1=np.zeros(len(windows)), y2=cvs_th_array, color='forestgreen',
                                alpha=0.4, edgecolor='k', lw=1)

            clipping_lim = np.ones(len(windows)) * axs[3].get_ylim()[-1]
            clipping_th = np.ones(len(windows))*ampls_th
            axs[3].fill_between(windows, y1=clipping_th, y2=clipping_lim,
                                color='tomato', alpha=0.6, edgecolor='k', lw=1)
            axs[3].plot(windows[best_window], ampl_means[best_window], 'o', ms=25, mec='black', mew=3,
                        color='purple', alpha=0.8)

        return best_window

def detect_best_window(self, win_size=8., win_shift=0.2, plot_debug=False, ax=False, savefig=False):
    """ This function detects the best window of the file to be analyzed. The core mechanism is in the
    best_window_algorithm function. For plot debug, call this function in "main" with argument plot_debug=True

    :param win_size: float. Size in seconds of the best window in seconds.
    :param win_shift: float. Size in seconds between windows. Default is 0.2 seconds.
    :param plot_debug: boolean. use True to plot filter parameters (and thresholds) for detecting best window
    :param ax: axes of the debugging plots.
    :return: two floats. The first float marks the start of the best window and the second the defined window-size.
    """
    filename = self._wavfile
    p_idx, t_idx = self.detect_peak_and_trough_indices()
    peak_time, peak_ampl, trough_time, trough_ampl = self._time[p_idx], self._eod[p_idx],\
                                                     self._time[t_idx], self._eod[t_idx]
    # peaks and troughs here refer to those found in each eod-cycle. For each cycle there should be one peak and
    # one trough if the detect_peak_indices function worked fine.
    my_times = peak_time[peak_time <= peak_time[-1] - win_size]  # Upper window-boundaries solution
    my_times = np.arange(0.0, peak_time[-1] - win_size, win_shift)
    cvs = np.empty(len(my_times))
    no_of_peaks = np.empty(len(my_times))
    mean_ampl = np.empty(len(my_times))

    for i, curr_t in enumerate(my_times):
        # This for-loop goes through each eod-cycle. Isn't this too much? It considerably makes the code slow.
        window_idx = (peak_time >= curr_t) & (peak_time <= curr_t + win_size)
        # the last line makes a window from curr_t and adds 8. seconds to it. Lower window-boundaries solution.
        p2t_ampl = peak_ampl[window_idx] - trough_ampl[window_idx]
        cvs[i] = np.std(p2t_ampl, ddof=1) / np.mean(p2t_ampl)
        mean_ampl[i] = np.mean(p2t_ampl)
        no_of_peaks[i] = len(p2t_ampl)

    if plot_debug:

        ax = aux.draw_bwin_analysis_plot(filename, self._time, self._eod, no_of_peaks, cvs, mean_ampl)

    # ToDo: Need to find a way to plot ax5!! It's not as easy as it seems...
    # ToDo: WOULD BE GREAT TO DO ALL THE PLOTTING IN EXTRA FUNCTION IN AUXILIARY!!!

    bwin_bool_array = self.best_window_algorithm(no_of_peaks, mean_ampl, cvs, plot_debug=plot_debug, axs=ax,
                                                 win_shift=win_shift)
    bwin_bool_inx = np.where(bwin_bool_array)[0][0]  # Gets the index of the best window out of all windows.
    entire_time_idx = self._eod_peak_idx[bwin_bool_inx]
    bwin = my_times[bwin_bool_inx]

    self.roi = (entire_time_idx, entire_time_idx + int(self._sample_rate/2.))

    # plotting the best window in ax[5]
    if plot_debug and len(ax) > 0:
        aux.draw_bwin_in_plot(ax, filename, self._time, self._eod, bwin, win_size, p_idx, t_idx,
                              savefig=savefig)

    return bwin, win_size
