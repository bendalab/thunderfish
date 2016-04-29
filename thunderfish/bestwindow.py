# the best window detector as functions...

import numpy as np
import scipy.stats as stats
import peakdetection as pd

# TODO: use warnings.warn

        
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


def clip_amplitudes(data, win_indices, min_fac=2.0, nbins=20) :
    """
    Find the amplitudes where the signals clips by looking at
    the histograms in data segements of win_indices length.
    If the bins at the edges are more than min_fac times as large as
    the neighboring bins, clipping at the bin's amplitude is assumed.

    Args:
      data (array): 1-D array with the data.
      win_indices: size of the analysis window in indices.
      min_fac: if the first or the second bin is at least min_fac times
        as large as the third bin, their upper bin edge is set as min_clip.
        Likewise for the last and next-to last bin.
      nbins: number of bins used for computing a histogram

    Returns:
      min_clip : minimum amplitude that is not clipped.
      max_clip : maximum amplitude that is not clipped.
    """
    
    #import matplotlib.pyplot as plt
    min_ampl = np.min(data)
    max_ampl = np.max(data)
    min_clipa = min_ampl
    max_clipa = max_ampl
    bins = np.linspace(min_ampl, max_ampl, nbins, endpoint=True)
    win_tinxs = np.arange(0, len(data) - win_indices, win_indices)
    for wtinx in win_tinxs:
        h, b = np.histogram(data[wtinx:wtinx+win_indices], bins)
        if h[0] > min_fac*h[2] and b[0] < -0.4 :
            if h[1] > min_fac*h[2] and b[2] > min_clipa:
                min_clipa = b[2]
            elif b[1] > min_clipa :
                min_clipa = b[1]
        if h[-1] > min_fac*h[-3] and b[-1] > 0.4 :
            if h[-2] > min_fac*h[-3] and b[-3] < max_clipa:
                max_clipa = b[-3]
            elif b[-2] < max_clipa :
                max_clipa = b[-2]
        #plt.bar(b[:-1], h, width=np.mean(np.diff(b)))
        #plt.axvline(min_clipa, color='r')
        #plt.axvline(max_clipa, color='r')
        #plt.show()
    #plt.hist(data, 20)
    #plt.axvline(min_clipa, color='r')
    #plt.axvline(max_clipa, color='r')
    #plt.show()
    return min_clipa, max_clipa


def accept_peak_size_threshold(time, data, event_inx, index, min_inx, threshold,
                               min_thresh, tau, thresh_ampl_fac=0.75, thresh_weight=0.02) :
    """
    To be passed to the detect_dynamic_peak_trough() function.
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
        thresh_ampl_fac (float): the new threshold is thresh_ampl_fac times the size of the current peak
        thresh_weight (float): new threshold is weighted against current threshold with thresh_weight

    Returns: 
        index (int): index of the peak/trough
        threshold (float): the new threshold to be used
    """
    size = data[event_inx] - data[min_inx]
    threshold += thresh_weight*(thresh_ampl_fac*size - threshold)
    return event_inx, threshold

    
def best_window_indices(data, rate, mode='first',
                        min_thresh=0.1, thresh_ampl_fac=0.8, thresh_weight=0.02, thresh_tau=1.0,
                        win_size=8., win_shift=0.1, min_clip=-np.inf, max_clip=np.inf,
                        percentile=0.15, cvi_th=0.05, cva_th=0.05, tolerance=1.1,
                        verbose=0, plot_data_func=None, plot_window_func=None, **kwargs):
    """
    Detect the best window of the data to be analyzed. The data have been sampled with rate Hz.
    
    First, large peaks and troughs of the data are detected.
    Peaks and troughs have to be separated in amplitude by at least the value of a dynamic threshold.
    The threshold is never smaller than min_thresh. Upon detection of a peak a new threshold value is set to
    thresh_ampl_fac times the amplitude of the peak minus the th eone of the previous trough.
    The current threshold is updated towards the new threshold value weighted by thresh_weight.
    Between peaks, the current threshold decays towards min_thresh with a time constant thresh_tau.

    Second, criteria for selecting the best window are computed for each window of width win_size
    shifted by win_shift trough the data. The three criteria are:
    - the coefficient of variation of the inter-peak and inter-trough intervals.
    - the mean peak-to-trough amplitude multiplied with the fraction of non clipped peak and trough amplitudes.
    - the coefficient of variation of the peak-to-trough amplitude.

    Third, the best window is defined as the window where the cv of the intervals and the amplitudes are below
    a threshold and the mean amplitude is above a threshold.
    Threshold values are set based on percentiles of the criteria.
    Initial thresholds are set such that the fraction of data given by percentile
    is selected by the threshold. The threshold percentile for the cv of intervals and amplitudes is increased
    such that the corresponding threshold is at least cvi_th and cva_th.
    All threshold values are then multiplied by tolerance.

    Fourth, if there is no window where all three criteria fall below their thresholds, the percentiles
    from which the thresholds are determined are increased in steps of 5%, until a window is found.

    Finally, the indices to the start and the end of the best window are
    returned. If mode='first' then the first window with all criteria
    below their threshold is returned.  If mode='expand', then the first
    best window is expanded as far as all three criteria are still below
    threshold. If mode='largest', then the largest contiguous region
    with all criteria below their threshold is returned.
    If no best window was found, then 0, 0 is returned.

    Output of warning and info messages to console can be controlled by setting verbose. No output is produced
    if verbose = 1. higher values produce more output.

    The algorithm can be visualized by supplying the functions plot_data_func and plot_window_func.
    Additional arguments for these function are supplied vie kwargs.

    :param data: 1-D array. The data to be analyzed
    :param rate: float. Sampling rate of the data in Hz
    :param mode: string. 'first': return the first best window.
    'expand': return the first best window enlarged as far as possible.
    'largest' return the largest region with all three criteria below their threshold.
    :param mode: string. 'first' returns first matching window, 'expand' expands the first matching window as far as possible, 'largest' returns the largest matching range.
    :param min_thresh: float. Minimum allowed value for the threshold. Set this above the noise level of the data.
    :param thresh_ampl_fac: float. New threshold is thresh_ampl_fac times the size of the current peak, between 0 and 1. Set this close to 1. The smaller the more small amplitude peaks are detected.
    :param thresh_weight: float. New threshold is weighed against current threshold with thresh_weight. The inverse of thresh_weight is approximately the number of peaks need for the threshold to approach the new threshold value.
    :param thresh_tau: float. Time constant of the decay of the threshold towards min_thresh in seconds.
    This should approximately match the fastest changes in signal amplitude.
    :param win_size: float. Size of the best window in seconds. Choose it large enough for a minimum analysis.
    :param win_shift: float. Time shift in seconds between windows. Should be smaller or equal to win_size and not smaller than about one thenth of win_shift.
    :param min_clip: float. Minimum amplitude below which data are clipped.
    :param max_clip: float. Maximum amplitude above which data are clipped.
    :param percentile: float. Fraction of the windows that is required to be below the initial correspoding thresholds.
    :param cvi_th: float. Coefficients of variation of the intervals smaller than this are selected initially.
    :param cva_th: float. Coefficients of variation of the amplitudes smaller than this are selected initially.
    :param tolerance: float. Multiply thresholds obtained from percentiles by this factor.
    :param verbose: int. Verbosity level >= 0.
    :param plot_data_func: Function for plotting the raw data, detected peaks and troughs and the criteria.
        plot_data_func(data, rate, peak_idx, trough_idx, idx0, idx1,
                       win_start_times, cv_interv, mean_ampl, cv_ampl, valid_wins, **kwargs)
        :param data (array): the raw data.
        :param rate (float): the sampling rate of the data.
        :param peak_idx (array): indices into raw data indicating detected peaks.
        :param trough_idx (array): indices into raw data indicating detected troughs.
        :param idx0 (int): index of the start of the best window.
        :param idx1 (int): index of the end of the best window.
        :param win_start_times (array): the times of the analysis windows.
        :param cv_interv (array): the coefficient of variation of the inter-peak and -trough intervals.
        :param mean_ampl (array): the mean peak-to-trough amplitude.
        :param cv_ampl (array): the coefficient of variation of the peak-to-trough amplitudes.
        :param valid_wins (array): boolean array indicating the windows which fulfill all three criteria.
        :param **kwargs: further user supplied arguments.
    :param plot_window_func: Function for plotting the window selection criteria.
        plot_window_func(cvi_th, ampl_th, cva_th, **kwargs)
        :param cvi_th (float): the final threshold value of the cv of the intervals.
        :param ampl_th (float): the final threshold value of the amplitudes.
        :param cva_th (float): the final threshold value for the cv of the amplitudes.
        :param **kwargs: further user supplied arguments.
    :param kwargs: Keyword arguments passed to plot_data_func and plot_window_func. 
    
    :return start_index: int. Index of the start of the best window.
    :return end_index: int. Index of the end of the best window.
    """

    def find_best_window(cvi_percentile, cva_percentile, ampl_percentile):
        """
        Based on the percentiles, thresholds are determined. The windows are selected where the data
        fall below the thresholds. If not a single window exists, where all three criteria are fulfilled
        the percentiles are increased by 5% and the the function is called again.

        :param cvi_percentile: The percentile from which the threshold for the cv of peak intervals is determined.
                Between 0 and 1.
        :param cva_percentile: The percentile from which the threshold for amplitude intervals is determined.
                Between 0 and 1.
        :param ampl_percentile: The percentile from which the threshold for mean amplitudes is determined.
                Between 0 and 1.
        :return: boolean array of valid windows.
        """

        if verbose > 1 :
            print('  run find_best_window() with cvi_percentile=%.2f, cva_percentile=%.2f, ampl_percentile=%.2f' % (cvi_percentile, cva_percentile, ampl_percentile))

        # First filter: least variable interpeak intervals
        cv_interval_sorted_inx = int(np.floor(cvi_percentile*len(cv_interval_sorted)))
        if cv_interval_sorted_inx >= len(cv_interval_sorted) :
            cv_interval_sorted_inx = -1
        cvi_th = cv_interval_sorted[cv_interval_sorted_inx]
        cvi_th *= tolerance
        valid_intervals = cv_interv <= cvi_th

        # Second filter: low variance in the amplitude
        cv_ampl_sorted_inx = int(np.floor(cva_percentile*len(cv_ampl_sorted)))
        if cv_ampl_sorted_inx >= len(cv_ampl_sorted) :
            cv_ampl_sorted_inx = -1
        cva_th = cv_ampl_sorted[cv_ampl_sorted_inx]
        cva_th *= tolerance
        valid_cv = cv_ampl <= cva_th

        # Third filter: choose the one with the highest amplitude
        ampl_th = ampl_sorted[np.floor(ampl_percentile*len(ampl_sorted))]
        ampl_th /= tolerance
        valid_ampls = mean_ampl >= ampl_th

        # All three conditions must be fulfilled:
        valid_windows = valid_intervals * valid_cv * valid_ampls

        # If there is no best window, run the algorithm again with more flexible thresholds:
        if not True in valid_windows :
            if cvi_percentile >= 1. and cva_percentile >= 1. and ampl_percentile <= 0.:
                print('did not find best window')
                if plot_window_func :
                    plot_window_func(cvi_th, ampl_th, cva_th, **kwargs)
                if verbose > 0 :
                    print('WARNING. Did not find an appropriate window for analysis.')
                # we failed:
                return []
            else :
                # round to 0.05 precision :
                cvi_percentile = np.round(cvi_percentile/0.05)*0.05
                cva_percentile = np.round(cva_percentile/0.05)*0.05
                ampl_percentile = np.round(ampl_percentile/0.05)*0.05
                # increase the single smallest percentiles first:
                if cvi_percentile < cva_percentile-0.01 and cvi_percentile < 1.0-ampl_percentile-0.01 :
                    cvi_percentile += 0.05
                elif cva_percentile < cvi_percentile-0.01 and cva_percentile < 1.0-ampl_percentile-0.01 :
                    cva_percentile += 0.05
                elif 1.0-ampl_percentile < cvi_percentile-0.01 and 1.0-ampl_percentile < cva_percentile-0.01 :
                    ampl_percentile -= 0.05
                else :
                    # increase the two smalles ones:
                    if cvi_percentile < cva_percentile-0.01 or cvi_percentile < 1.0-ampl_percentile-0.01 :
                        cvi_percentile += 0.05
                    elif cva_percentile < cvi_percentile-0.01 or cva_percentile < 1.0-ampl_percentile-0.01 :
                        cva_percentile += 0.05
                    elif 1.0-ampl_percentile < cvi_percentile-0.01 or 1.0-ampl_percentile < cva_percentile-0.01 :
                        ampl_percentile -= 0.05
                    else :
                        # if all percentiles are the same, increase all of them:
                        cvi_percentile += 0.05
                        cva_percentile += 0.05
                        ampl_percentile -= 0.05
                # check for overflow:
                if cvi_percentile > 1.0 :
                    cvi_percentile = 1.0
                if cva_percentile > 1.0 :
                    cva_percentile = 1.0
                if ampl_percentile < 0.0 :
                    ampl_percentile = 0.0
                # run again with relaxed threshold values:
                return find_best_window(cvi_percentile, cva_percentile, ampl_percentile)
        else:
            if plot_window_func :
                plot_window_func(cvi_th, ampl_th, cva_th, **kwargs)
            if verbose > 1 :
                print('found best window')
            # we are done:
            return valid_windows


    # too little data:
    if len(data) / rate <= win_size :
        if verbose > 0 :
            print 'no best window found: not enough data'
        return 0, 0

    # detect large peaks and troughs:
    thresh = 1.5*np.std(data[0:win_shift*rate])
    tauidx = thresh_tau*rate
    peak_idx, trough_idx = pd.detect_dynamic_peaks_troughs(data, thresh, min_thresh,
                                                           tauidx, None,
                                                           accept_peak_size_threshold, None,
                                                           thresh_ampl_fac=thresh_ampl_fac,
                                                           thresh_weight=thresh_weight)
    if len(peak_idx) == 0 or len(trough_idx) == 0 :
        if verbose > 0 :
            print 'best_window(): no peaks and troughs detected'
        return 0, 0
    
    # compute cv of intervals, mean peak amplitude and its cv:
    win_size_indices = int(win_size*rate)
    win_start_inxs = np.arange(0, len(data) - win_size_indices, int(win_shift*rate))
    cv_interv = np.zeros(len(win_start_inxs))
    mean_ampl = np.zeros(len(win_start_inxs))
    cv_ampl = np.zeros(len(win_start_inxs))
    for i, wtinx in enumerate(win_start_inxs):
        # indices of peaks and troughs inside analysis window:
        pinx = (peak_idx >= wtinx) & (peak_idx <= wtinx + win_size_indices)
        tinx = (trough_idx >= wtinx) & (trough_idx <= wtinx + win_size_indices)
        p_idx, t_idx = pd.trim_to_peak(peak_idx[pinx], trough_idx[tinx])
        # interval statistics:
        ipis = np.diff(p_idx)
        itis = np.diff(t_idx)
        if len(ipis) > 10 :
            cv_interv[i] = 0.5*(np.std(ipis)/np.mean(ipis) + np.std(itis)/np.mean(itis))
            # penalize regions without detected peaks:
            mean_interv = np.mean(ipis)
            if p_idx[0] - wtinx > mean_interv :
                cv_interv[i] *= (p_idx[0] - wtinx)/mean_interv
            if wtinx + win_size_indices - p_idx[-1] > mean_interv :
                cv_interv[i] *= (wtinx + win_size_indices - p_idx[-1])/mean_interv
        else :
            cv_interv[i] = 1000.0
        # statistics of peak-to-trough amplitude:
        p2t_ampl = data[p_idx] - data[t_idx]
        if len(p2t_ampl) > 0 :
            mean_ampl[i] = np.mean(p2t_ampl)
            cv_ampl[i] = np.std(p2t_ampl) / mean_ampl[i]
            # penalize for clipped peaks:
            clipped_frac = float(np.sum(data[p_idx]>max_clip) +
                                 np.sum(data[t_idx]<min_clip))/2.0/len(p2t_ampl)
            mean_ampl[i] *= (1.0-clipped_frac)**2.0
        else :
            mean_ampl[i] = 0.0
            cv_ampl[i] = 1000.0
    # cumulative function for mean amplitudes and percentile:
    ampl_sorted = np.sort(mean_ampl)
    ampl_sorted = ampl_sorted[ampl_sorted>0.0]
    if len(ampl_sorted) <= 0 :
        if verbose > 0 :
            print 'best_window(): no finite amplitudes detected'
        return 0, 0
    ampl_percentile = 1.0 - percentile
    # cumulative function for interval cvs and percentile of threshold:
    cv_interval_sorted = np.sort(cv_interv)
    cv_interval_sorted = cv_interval_sorted[cv_interval_sorted<1000.0]
    if len(cv_interval_sorted) <= 0 :
        if verbose > 0 :
            print 'best_window(): no valid interval cvs detected'
        return 0, 0
    cvi_percentile = float(len(cv_interval_sorted[cv_interval_sorted<cvi_th])/float(len(cv_interval_sorted)))
    if cvi_percentile < percentile :
        cvi_percentile = percentile
    # cumulative function for amplitude cvs and percentile of threshold:
    cv_ampl_sorted = np.sort(cv_ampl)
    cv_ampl_sorted = cv_ampl_sorted[cv_ampl_sorted<1000.0]
    if len(cv_ampl_sorted) <= 0 :
        if verbose > 0 :
            print 'best_window(): no valid amplitude cvs detected'
        return 0, 0
    cva_percentile = float(len(cv_ampl_sorted[cv_ampl_sorted<cva_th])/float(len(cv_ampl_sorted)))
    if cva_percentile < percentile :
        cva_percentile = percentile
    
    # find best window:
    valid_wins = find_best_window(cvi_percentile, cva_percentile, ampl_percentile)

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
        idx0 = win_start_inxs[valid_idx[0]]
        idx1 = win_start_inxs[valid_idx[-1]]+win_size_indices
        
    if plot_data_func :
        plot_data_func(data, rate, peak_idx, trough_idx, idx0, idx1,
                       win_start_inxs/rate, cv_interv, mean_ampl, cv_ampl, valid_wins, **kwargs)

    return idx0, idx1


# TODO: make sure the arguments are still right!
def best_window_times(data, rate, mode='first',
                        min_thresh=0.1, thresh_ampl_fac=0.75, thresh_weight=0.02, thresh_tau=1.0,
                        win_size=8., win_shift=0.1, min_clip=-np.inf, max_clip=np.inf,
                        percentile=0.15, cvi_th=0.05, cva_th=0.05, tolerance=1.1,
                        verbose=0, plot_data_func=None, plot_window_func=None, **kwargs):
    """
    Finds the window within data with the best data. See best_window_indices() for details.

    Returns:
      start_time (float): Time of the start of the best window.
      end_time (float): Time of the end of the best window.
    """
    start_inx, end_inx = best_window_indices(data, rate, mode,
                        min_thresh, thresh_ampl_fac, thresh_weight, thresh_tau,
                        win_size, win_shift, min_clip, max_clip,
                        percentile, cvi_th, cva_th, tolerance,
                        verbose, plot_data_func, plot_window_func, **kwargs)
    return start_inx/rate, end_inx/rate


# TODO: make sure the arguments are still right!
def best_window(data, rate, mode='first',
                min_thresh=0.1, thresh_ampl_fac=0.75, thresh_weight=0.02, thresh_tau=1.0,
                win_size=8., win_shift=0.1, min_clip=-np.inf, max_clip=np.inf,
                percentile=0.15, cvi_th=0.05, cva_th=0.05, tolerance=1.1,
                verbose=0, plot_data_func=None, plot_window_func=None, **kwargs):
    """
    Finds the window within data with the best data. See best_window_indices() for details.

    Returns:
      data (array): the data of the best window.
    """
    start_inx, end_inx = best_window_indices(data, rate, mode,
                        min_thresh, thresh_ampl_fac, thresh_weight, thresh_tau,
                        win_size, win_shift, min_clip, max_clip,
                        percentile, cvi_th, cva_th, tolerance,
                        verbose, plot_data_func, plot_window_func, **kwargs)
    return data[start_inx:end_inx]


if __name__ == "__main__":
    print("Checking bestwindow module ...")
    import sys

    if len(sys.argv) < 2 :
        # generate data:
        print("generate waveform...")
        rate = 40000.0
        time = np.arange(0.0, 10.0, 1./rate)
        f1 = 100.0
        data0 = (0.5*np.sin(2.0*np.pi*f1*time)+0.5)**20.0
        amf1 = 0.3
        data1 = data0*(1.0-np.cos(2.0*np.pi*amf1*time))
        data1 += 0.2
        f2 = f1*2.0*np.pi
        data2 = 0.1*np.sin(2.0*np.pi*f2*time)
        amf3 = 0.15
        data3 = data2*(1.0-np.cos(2.0*np.pi*amf3*time))
        #data = data1+data3
        data = data2
        data += 0.01*np.random.randn(len(data))
    else :
        import dataloader as dl
        print("load %s ..." % sys.argv[1])
        data, rate, unit = dl.load_data(sys.argv[1], 0)

    
    # determine clipping amplitudes:
    clip_win_size = 0.5
    min_clip_fac = 2.0
    min_clip, max_clip = clip_amplitudes(data, int(clip_win_size*rate), min_fac=min_clip_fac)
    
    # find best window:
    best_window_indices(data, rate, mode='first',
                        min_thresh=0.1, thresh_ampl_fac=0.8, thresh_weight=0.02, thresh_tau=0.25,
                        win_size=1.0, win_shift=0.5, min_clip=min_clip, max_clip=max_clip)
