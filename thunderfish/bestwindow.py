# the best window detector as functions...

import numpy as np
import scipy.stats as stats
import peakdetection as pd

        
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


def accept_peak_size_threshold(time, data, event_inx, index, min_inx, threshold,
                               min_thresh, tau, thresh_fac=0.75, thresh_frac=0.02) :
    """
    Accept each detected peak/trough and return its index.
    Adjust the threshold to the size of the detected peak.

    Args:
        time (array): time values, can be None
        data (array): the data in wich peaks and troughs are detected
        event_inx: index of the current peak/trough
        index: current index
        min_inx: index of the previous trough/peak
        threshold: threshold value
        min_thresh (float): the minimum value the threshold is allowed to assume.
        tau (float): the time constant of the the decay of the threshold value
                     given in indices (time is None) or time units (time is not None)
        thresh_fac (float): the new threshold is thresh_fac times the size of the current peak
        thresh_frac (float): new threshold is weighed against current threshold with thresh_frac

    Returns: 
        index (int): index of the peak/trough
        threshold (float): the new threshold to be used
    """
    size = data[event_inx] - data[min_inx]
    threshold += thresh_frac*(thresh_fac*size - threshold)
    return event_inx, threshold


def best_window_algorithm(peak_rate, rate_mode, rate_spread, mean_ampl, ampls, cv_ampl, cvs,
                          rate_th=0.1, ampl_percentile_th=0.85,
                          cv_percentile_th=0.15,
                          verbose=1, plot_window_func=None, **kwargs):

    """This is the algorithm that chooses the best window. It first filters out the windows that have a siginificant
    different amount of peaks compared to the stats.mode peak number of all windows. Secondly, it filters out
    windows that have a higher coefficient of variation than a certain percentile of the distribution of cv_ampl.
    From those windows that get through both filters, the one with the highest peak-to-trough-amplitude
    (that is not clipped!) is chosen as the best window. We assume clipping as amplitudes above 85% percentile of
    the distribution of peak to trough amplitude.

    :param cv_percentile_th: threshold of how much amplitude-variance (coefficient of variation) of the signal is allowed. Between 0 and 1.
    :param peak_rate: array with number of peaks
    :param mean_ampl: array with mean peak-to-trough-amplitudes
    :param cv_ampl: array with cv of amplitudes
    :param rate_th: threshold for peak-rate filter
    :param ampl_percentile_th: choose a percentile threshold to avoid clipping. Between 0 and 1.
    :param cv_percentile_th: Threshold for cv.
    :param plot_debug: boolean for showing plot-debugging.
    :param axs: axis of plot debugging.
    :param win_shift: float. Size in seconds between windows. Default is 0.2 seconds.
    :return: boolean array with a single True element. This is the Index of the best window out of all windows.
    """

    # First filter: stable peak rate
    # TODO: It would be better to take from the good amplitudes that section with the most stable rate! Check 40517L28.WAV
    # TODO: Or skip this filter entirely?
    # TODO: Better make sure for most regular peak intervals WITHIN a window! 40517L12.WAV 40517L17.WAV
    lower_rate_th = rate_mode - rate_th*rate_spread
    upper_rate_th = rate_mode + rate_th*rate_spread
    lower = peak_rate >= lower_rate_th
    upper = peak_rate <= upper_rate_th
    finite = peak_rate > 0.0
    valid_rate = lower * upper * finite

    # Second filter: low variance in the amplitude
    cvs_inx = int(np.floor(cv_percentile_th*len(cvs)))
    if cvs_inx >= len(cvs) :
        cvs_inx = -1
    cv_th = cvs[cvs_inx]
    valid_cv = cv_ampl <= cv_th

    # Third filter: choose the one with the highest amplitude that is not clipped
    # TODO: Clipping not yet implemented!!!! See 40517L1[456].WAV L26
    ampl_th = np.percentile(mean_ampl[mean_ampl>0.0], ampl_percentile_th)
    ampl_th = ampls[np.floor(ampl_percentile_th*len(ampls))]
    valid_ampls = mean_ampl >= ampl_th

    # All three conditions must be fulfilled:
    #valid_windows = valid_rate * valid_cv * valid_ampls
    valid_windows = valid_cv * valid_ampls

    # If there is no best window, run the algorithm again with more flexible thresholds:
    if not True in valid_windows :
        if cv_percentile_th >= 1. and ampl_percentile_th <= 0.:
            print('did not find best window')
            if plot_window_func :
                plot_window_func(lower_rate_th, upper_rate_th, ampl_th, cv_th, **kwargs)
            if verbose > 0 :
                print('WARNING. Did not find an appropriate window for analysis.')
            return []

        else :
            # TODO: increase only threshold of the criterion with the smallest/zero True range?
            rate_th += 0.05
            if rate_th > 1.0 :
                rate_th = 1.0
            cv_percentile_th += 0.05
            if cv_percentile_th > 1. :
                cv_percentile_th = 1.
            ampl_percentile_th -= 0.05
            if ampl_percentile_th < 0. :
                ampl_percentile_th = 0.

            if verbose > 1 :
                print('  rerunning best_window_algorithm() with rate_th=%.2f, cv_percentile_th=%.2f, ampl_percentile_th=%.2f' % (rate_th, cv_percentile_th, ampl_percentile_th))
            return best_window_algorithm(peak_rate, rate_mode, rate_spread, mean_ampl, ampls,
                                         cv_ampl, cvs, rate_th=rate_th,
                                         ampl_percentile_th=ampl_percentile_th,
                                         cv_percentile_th=cv_percentile_th, verbose=verbose,
                                         plot_window_func=plot_window_func, **kwargs)
    else:
        if plot_window_func :
            plot_window_func(lower_rate_th, upper_rate_th, ampl_th, cv_th, **kwargs)
        if verbose > 1 :
            print('found best window')
        return valid_windows

    
def best_window(data, rate, mode='first',
                min_thresh=0.1, thresh_fac=0.75, thresh_frac=0.02, thresh_tau=1.0, win_size=8., win_shift=0.1, cv_th=0.05,
                verbose=0, plot_data_func=None, plot_window_func=None, **kwargs):
    """ Detect the best window of the data to be analyzed. The core mechanism is in the
    best_window_algorithm function. For plot debug, call this function with argument plot_debug=True

    :param data: 1-D array. The data to be analyzed
    :param rate: float. Sampling rate of the data in Hz
    :param mode: string. 'first' returns first matching window, 'expand' expands the first matching window as far as possible, 'largest' returns the largest matching range.
    :param min_thresh: float. Minimum allowed value for the threshold
    :param thresh_fac: float. New threshold is thresh_fac times the size of the current peak
    :param thresh_frac: float. New threshold is weighed against current threshold with thresh_frac
    :param thresh_tau: float. Time constant of the decay of the threshold towards min_thresh in seconds
    :param win_size: float. Size of the best window in seconds.
    :param win_shift: float. Size in seconds between windows.
    :param verbose: int. Verbosity level >= 0.
    :param plot_data_func: Function for plotting the raw data and detected peaks and troughs. 
    :param plot_window_func: Function for plotting the window selection criteria.
    :param kwargs: Keyword arguments passed to plot_data_func and plot_window_func. 
    
    :return: two floats. The first float marks the start of the best window and the second the defined window-size.
    """

    # too little data:
    if len(data) / rate <= win_size :
        if verbose > 0 :
            print 'no best window found: not enough data'
        return

    # detect large peaks and troughs:
    thresh = 1.5*np.std(data[0:win_shift*rate])
    tauidx = thresh_tau*rate
    peak_idx, trough_idx = pd.detect_dynamic_peaks_troughs(data, thresh, min_thresh,
                                                           tauidx, None,
                                                           accept_peak_size_threshold, None,
                                                           thresh_fac=thresh_fac,
                                                           thresh_frac=thresh_frac)

    # compute peak rate, mean peak amplitude and its cv:
    win_sinx = win_size*rate
    win_tinxs = np.arange(0.0, len(data) - win_sinx, win_shift*rate)
    peak_rate = np.zeros(len(win_tinxs))
    mean_ampl = np.zeros(len(win_tinxs))
    cv_ampl = np.zeros(len(win_tinxs))
    for i, tinx in enumerate(win_tinxs):
        # indices of peaks and troughs inside analysis window:
        pinx = (peak_idx >= tinx) & (peak_idx <= tinx + win_sinx)
        tinx = (trough_idx >= tinx) & (trough_idx <= tinx + win_sinx)
        p_idx, t_idx = pd.trim_to_peak(peak_idx[pinx], trough_idx[tinx])
        # statistics of peak-to-trough amplitude:
        p2t_ampl = data[p_idx] - data[t_idx]
        peak_rate[i] = len(p2t_ampl)/win_size
        if len(p2t_ampl) > 0 :
            mean_ampl[i] = np.mean(p2t_ampl)
            cv_ampl[i] = np.std(p2t_ampl) / mean_ampl[i]
        else :
            mean_ampl[i] = 0.0
            cv_ampl[i] = 1000.0
    # mode of rate:
    p_rate = peak_rate[peak_rate>0.0]
    if len(p_rate) < 50 :
        rate_mode = stats.mode(p_rate)[0][0]
    else :
        n = np.ceil(len(p_rate)/10.)
        h, b = np.histogram(p_rate, n)
        inx = np.argmax(h)
        rate_mode = 0.5*(b[inx]+b[inx+1])
    rate_spread = np.max(p_rate)-np.min(p_rate)
    # cumulative functions:
    ampls = np.sort(mean_ampl)
    ampls = ampls[ampls>0.0]
    cvs = np.sort(cv_ampl)
    cvs = cvs[cvs<1000.0]
    cv_percentile_th = float(len(cvs[cvs<cv_th])/float(len(cvs)))
    print cv_percentile_th

    # find best window:
    valid_wins = best_window_algorithm(peak_rate, rate_mode, rate_spread,
                                        mean_ampl, ampls, cv_ampl, cvs,
                                        cv_percentile_th=cv_percentile_th, verbose=verbose,
                                        plot_window_func=plot_window_func, **kwargs)

    # extract best window:
    idx0 = 0
    idx1 = 0
    if len(valid_wins) > 0 :
        valid_idx = np.nonzero(valid_wins)[0]
        if mode == 'expand' :
            idx0 = valid_idx[0]
            idx1 = idx0
            while idx1<len(valid_wins) and valid_wins[idx1] :
                idx1 += 1
            valid_wins[idx1:] = False
        elif mode == 'largest' :
            for lidx0 in valid_idx :
                lidx1 = lidx0
                while lidx1<len(valid_wins) and valid_wins[lidx1] :
                    lidx1 += 1
                if lidx1 - lidx0 > idx1 - idx0 :
                    idx0 = lidx0
                    idx1 = lidx1
                valid_wins[:idx0] = False
                valid_wins[idx1:] = False
        else : # first only:
            valid_wins[valid_idx[0]+1:] = False
        valid_idx = np.nonzero(valid_wins)[0]
        idx0 = win_tinxs[valid_idx[0]]
        idx1 = win_tinxs[valid_idx[-1]]+win_sinx
        
    if plot_data_func :
        plot_data_func(data, rate, peak_idx, trough_idx, idx0, idx1,
                       win_tinxs/rate, peak_rate, mean_ampl, cv_ampl, valid_wins, **kwargs)

    return idx0, idx1


if __name__ == "__main__":
    print("Checking bestwindow module ...")

    print
    # generate data:
    rate = 40000.0
    time = np.arange(0.0, 2.0, 1./rate)
    f1 = 100.0
    data1 = (0.5*np.sin(2.0*np.pi*f1*time)+0.5)**20.0
    amf1 = 1.
    data1 *= 1.0-np.cos(2.0*np.pi*amf1*time)
    data1 += 0.2
    f2 = f1*2.0*np.pi
    data2 = 0.1*np.sin(2.0*np.pi*f2*time)
    amf2 = 0.5
    data2 *= 1.0-np.cos(2.0*np.pi*amf2*time)
    data = data1+data2
    data += 0.01*np.random.randn(len(data))
    print("generated waveform")

    best_window(data, rate, mode='first',
                min_thresh=0.1, thresh_fac=0.8, thresh_frac=0.02, thresh_tau=0.25,
                win_size=0.2, win_shift=0.1)
