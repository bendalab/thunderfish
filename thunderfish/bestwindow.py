# the best window detector as functions...

import numpy as np
import peakdetection as pd


def long_enough(rate, data, minduration=10.0):
    """
    Returns:
      true if the data are longer than minduration
    """
    return len(data) / rate <= minduration

        
def normalized_signal(data, rate, win_duration=.1, min_std=0.1) :
    """
    Removes mean and normalizes data by dividing by the standard deviation.
    Mean and standard deviation are computed in win_duration long windows.

    Args:
      data (array): the data as a 1-D array
      rate (float): the sampling rate of the data
      win_duration (float): the length of the analysis window given in inverse unit of rate
      min_std (float): minimum standard deviation to be used for scaling

    Returns:
      scaled_data (array): the de-meaned and normalized data as an 1-D numpy array
    """
    w = np.ones(rate*win_duration)
    w /= len(w)
    mean = np.convolve(data, w, mode='same')
    std = np.sqrt(np.convolve(data**2., w, mode='same') - mean**2.)
    if min_std > 0.0 :
        std[std<min_std] = min_std
    return (data - mean) / std


def best_window_algorithm(peak_rate, mean_ampl, cv_ampl, rate_th=0.15, ampls_percentile_th=85.,
                          cv_percentile_th=15., plot_debug=False, axs=None, win_shift = 0.2,
                          verbose=1):

    """This is the algorithm that chooses the best window. It first filters out the windows that have a siginificant
    different amount of peaks compared to the stats.mode peak number of all windows. Secondly, it filters out
    windows that have a higher coefficient of variation than a certain percentile of the distribution of cv_ampl.
    From those windows that get through both filters, the one with the highest peak-to-trough-amplitude
    (that is not clipped!) is chosen as the best window. We assume clipping as amplitudes above 85% percentile of
    the distribution of peak to trough amplitude.

    :param cv_percentile_th: threshold of how much amplitude-variance (covariance coefficient) of the signal
    is allowed. Default is 10%.
    :param peak_rate: array with number of peaks
    :param mean_ampl: array with mean peak-to-trough-amplitudes
    :param cv_ampl: array with cv of amplitudes
    :param rate_th: threshold for peak-rate filter
    :param ampls_percentile_th: choose a percentile threshold to avoid clipping. Default is 85
    :param cv_percentile_th: Threshold for cv. Default is 85
    :param plot_debug: boolean for showing plot-debugging.
    :param axs: axis of plot debugging.
    :param win_shift: float. Size in seconds between windows. Default is 0.2 seconds.
    :return: boolean array with a single True element. This is the Index of the best window out of all windows.
    """
    # First filter: stable peak rate
    ## pk_mode = stats.mode(peak_rate)[0][0]
    ## tot_pks = max(peak_rate)-min(peak_rate)
    ## lower = peak_rate >= pk_mode - tot_pks*rate_th
    ## upper = peak_rate <= pk_mode + tot_pks*rate_th
    mean_rate = np.mean(peak_rate)
    std_rate = np.std(peak_rate)
    lower = peak_rate >= mean_rate - rate_th*std_rate
    upper = peak_rate >= mean_rate + rate_th*std_rate
    finite = peak_rate > 0.0
    valid_rate = lower * upper * finite

    # Second filter: low variance in the amplitude
    # TODO: Would be better to use absolute CVs as threshold?
    cv_th = np.nanpercentile(cv_ampl, cv_percentile_th)
    valid_cv = cv_ampl < cv_th

    # Third filter: choose the one with the highest amplitude that is not clipped
    ampls_th = np.percentile(ampl_means[mean_ampl>0.0], ampls_percentile_th)
    valid_ampls = (ampl_means > 0.0) * (ampl_means <= ampls_th)

    # All three conditions must be fulfilled:
    valid_windows = valid_rate * valid_cv * valid_ampls

    # If there is no best window, run the algorithm again with more flexible thresholds:
    if not True in valid_windows :
        if cv_percentile_th >= 100. and ampls_percentile_th <= 0.:
            if verbose > 0 :
                print('WARNING. Did not find an appropriate window for analysis.')
            return -1

        else :
            # TODO: increase only threshold of the criterion with the smallest True range
            if cv_percentile_th <= 95.:
                cv_percentile_th += 5.
            if ampls_percentile_th >= 5.:
                ampls_percentile_th -= 5.
            if verbose > 0 :
                print('Rerunning best_window_algorithm with more relaxed threshold values: cv=%g, ampls=%g' % (cv_percentile_th, ampls_percentile_th))
            return best_window_algorithm(peak_rate, mean_ampl, cv_ampl,
                                         ampls_percentile_th=ampls_percentile_th,
                                         cv_percentile_th=cv_percentile_th,
                                         plot_debug=plot_debug,
                                         win_shift=win_shift, verbose=verbose)
    else:
        # max_ampl_window = np.max(ampl_means[valid_windows])  # Boolean array with a single True element ## NO! It is only a float with the largest amplitude
        # best_window = valid_windows * max_ampl_window

        # choose the window with the largest amplitude:
        max_ampl_idx = np.argmax(ampl_means[valid_windows])
        best_window = valid_windows[max_ampl_idx] # this is also wrong

        ## if plot_debug:

        ##     windows = np.arange(len(peak_rate)) * win_shift
        ##     up_th = np.ones(len(windows)) * pk_mode[0][0] + tot_pks*rate_th
        ##     down_th = np.ones(len(windows)) * pk_mode[0][0] - tot_pks*rate_th
        ##     axs[1].fill_between(windows, y1=down_th, y2=up_th, color='forestgreen', alpha=0.4, edgecolor='k', lw=1)

        ##     cvs_th_array = np.ones(len(windows)) * cv_th
        ##     axs[2].fill_between(windows, y1=np.zeros(len(windows)), y2=cvs_th_array, color='forestgreen',
        ##                         alpha=0.4, edgecolor='k', lw=1)

        ##     clipping_lim = np.ones(len(windows)) * axs[3].get_ylim()[-1]
        ##     clipping_th = np.ones(len(windows))*ampls_th
        ##     axs[3].fill_between(windows, y1=clipping_th, y2=clipping_lim,
        ##                         color='tomato', alpha=0.6, edgecolor='k', lw=1)
        ##     axs[3].plot(windows[best_window], max_ampl_window[best_window], 'o', ms=25, mec='black', mew=3,
        ##                 color='purple', alpha=0.8)

        return best_window

    
def best_window(data, rate, win_size=8., win_shift=0.1,
                min_std=0.1, threshold = 1.5,
                plot_debug=False, ax=False, savefig=False, title=""):
    """ Detect the best window of the data to be analyzed. The core mechanism is in the
    best_window_algorithm function. For plot debug, call this function with argument plot_debug=True

    :param win_size: float. Size in seconds of the best window in seconds.
    :param win_shift: float. Size in seconds between windows.
    :param min_std: float. Minimum standard deviation to be used for scaling
    :param threshold: float. Threshold for peak detection as multiples of local standard deviation
    :param plot_debug: boolean. use True to plot filter parameters (and thresholds) for detecting best window
    :param ax: axes of the debugging plots.
    :return: two floats. The first float marks the start of the best window and the second the defined window-size.
    """

    # scale the data to their local standard deviation:
    scaled_data = normalized_signal(data, rate, win_shift, min_std)

    # detect peaks and troughs in the normalized data:
    peak_idx, trough_idx = pd.detect_peaks_troughs(scaled_data, threshold)
    # we might want to make sure that peaks and troughs of the same index
    # are temporally close to each other.
    peak_time = peak_idx/rate
    peak_ampl = data[peak_idx]
    trough_time = trough_idx/rate
    trough_ampl = data[trough_idx]

    # compute peak rate, mean peak amplitude and its cv:
    win_times = np.arange(0.0, peak_time[-1] - win_size, win_shift)
    peak_rate = np.zeros(len(win_times))
    mean_ampl = np.zeros(len(win_times))
    cv_ampl = np.zeros(len(win_times))
    for i, t in enumerate(win_times):
        window_idx = (peak_time >= t) & (peak_time <= t + win_size)
        # TODO: align peak and troughs (same len and peaks first)
        p2t_ampl = peak_ampl[window_idx] - trough_ampl[window_idx]
        peak_rate[i] = len(p2t_ampl)/win_size
        mean_ampl[i] = np.mean(p2t_ampl)
        cv_ampl[i] = np.std(p2t_ampl, ddof=1) / mean_ampl[i]

    #if plot_debug:
    #    ax = aux.draw_bwin_analysis_plot(title, self._time, self._eod, peak_rate, cv_ampl, mean_ampl)

    # ToDo: Need to find a way to plot ax5!! It's not as easy as it seems...
    # ToDo: WOULD BE GREAT TO DO ALL THE PLOTTING IN EXTRA FUNCTION IN AUXILIARY!!!

    bwin_bool_array = best_window_algorithm(peak_rate, mean_ampl, cv_ampl, plot_debug=plot_debug, axs=ax, win_shift=win_shift)
    bwin_bool_inx = np.where(bwin_bool_array)[0][0]  # Gets the index of the best window out of all windows.
    bwin = win_times[bwin_bool_inx]

    # plotting the best window in ax[5]
    #if plot_debug and len(ax) > 0:
    #    aux.draw_bwin_in_plot(ax, title, self._time, self._eod, bwin, win_size, p_idx, t_idx,
    #                          savefig=savefig)

    return bwin, win_size


def accept_peak_std_threshold(time, data, event_inx, index, min_inx, threshold, check_conditions) :
    """
    Accept each detected peak/trough and return its index (or time) and its data value.

    Args:
        freqs (array): frequencies of the power spectrum
        data (array): the power spectrum
        event_inx: index of the current peak/trough
        index: current index
        min_inx: index of the previous trough/peak
        threshold: threshold value
        check_conditions: not used
    
    Returns: 
        index (int): index of the peak/trough if time is None
        time (float): time of the peak/trough if time is not None
    """

    std = np.std(data[event_inx:event_inx+4000])
    min_std = 0.1
    if min_std > 0.0 and std < min_std :
        std = min_std

    threshold = 1.5*std
    
    if time is None :
        return event_inx, threshold
    else :
        return time[event_inx], threshold


if __name__ == "__main__":
    print("Checking bestwindow module ...")
    import matplotlib.pyplot as plt

    print
    # generate data:
    rate = 40000.0
    time = np.arange(0.0, 2.0, 1./rate)
    f = 600.0
    data = (0.5*np.sin(2.0*np.pi*f*time)+0.5)**4.0
    data -= 0.5
    amf = 1.
    data *= 1.0-np.cos(2.0*np.pi*amf*time)
    data += 0.2
    data += 0.01*np.random.randn(len(data))
    print("generated waveform")
    plt.plot(time, data)

    # normalize data:
    #scaled_data = normalized_signal(data, rate)
    # this is computational expensive! We put this into the peak detection via an adaptive threshold
    #plt.hist(scaled_data, 100)
    #plt.plot(time, scaled_data)

    # detect peaks:
    #peak_idx, trough_idx = pd.detect_peaks_troughs(scaled_data, 1.5)
    peak_idx, trough_idx = pd.detect_peaks_troughs(data, 0.1, None, accept_peak_std_threshold)

    plt.plot(time[peak_idx], data[peak_idx], '.r', ms=10)
    plt.plot(time[trough_idx], data[trough_idx], '.g', ms=10)
    #plt.plot(time[peak_idx], scaled_data[peak_idx], '.r', ms=10)
    #plt.plot(time[trough_idx], scaled_data[trough_idx], '.g', ms=10)
    plt.show()
