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

    
def best_window_indices(data, rate, mode='first',
                        min_thresh=0.1, thresh_fac=0.75, thresh_frac=0.02, thresh_tau=1.0,
                        win_size=8., win_shift=0.1, cvi_th=0.05, cva_th=0.05,
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
    
    :return start_index: int. Index of the start of the best window.
    :return end_index: int. Index of the end of the best window.
    """

    def best_window_algorithm(cvi_percentile=0.15, cva_percentile=0.15, ampl_percentile=0.85):

        """This is the algorithm that chooses the best window. It first filters out the windows that have a siginificant
        different amount of peaks compared to the stats.mode peak number of all windows. Secondly, it filters out
        windows that have a higher coefficient of variation than a certain percentile of the distribution of cv_ampl.
        From those windows that get through both filters, the one with the highest peak-to-trough-amplitude
        (that is not clipped!) is chosen as the best window. We assume clipping as amplitudes above 85% percentile of
        the distribution of peak to trough amplitude.

        :param cvi_percentile: threshold for peak interval filter between 0 and 1
        :param cva_percentile: threshold for cv of amplitudes between 0 and 1.
        :param ampl_percentile: choose a percentile threshold to avoid clipping. Between 0 and 1.
        :return: boolean array of valid windows.
        """

        # TODO: this needs to be an argument and needs to be explained
        tolerance = 1.1

        # First filter: least variable interpeak intervals
        cv_interval_sorted_inx = int(np.floor(cvi_percentile*len(cv_interval_sorted)))
        if cv_interval_sorted_inx >= len(cv_interval_sorted) :
            cv_interval_sorted_inx = -1
        cvi_th = cv_interval_sorted[cv_interval_sorted_inx]
        cvi_th *= tolerance
        valid_times = cv_interv <= cvi_th

        # Second filter: low variance in the amplitude
        cv_ampl_sorted_inx = int(np.floor(cva_percentile*len(cv_ampl_sorted)))
        if cv_ampl_sorted_inx >= len(cv_ampl_sorted) :
            cv_ampl_sorted_inx = -1
        cva_th = cv_ampl_sorted[cv_ampl_sorted_inx]
        cva_th *= tolerance
        valid_cv = cv_ampl <= cva_th

        # Third filter: choose the one with the highest amplitude that is not clipped
        # TODO: Clipping not yet implemented!!!! See 40517L1[456].WAV L26
        ampl_th = ampl_sorted[np.floor(ampl_percentile*len(ampl_sorted))]
        ampl_th /= tolerance
        valid_ampls = mean_ampl >= ampl_th

        # All three conditions must be fulfilled:
        valid_windows = valid_times * valid_cv * valid_ampls

        # If there is no best window, run the algorithm again with more flexible thresholds:
        if not True in valid_windows :
            if cvi_percentile >= 1. and cva_percentile >= 1. and ampl_percentile <= 0.:
                print('did not find best window')
                if plot_window_func :
                    plot_window_func(cvi_th, ampl_th, cva_th, **kwargs)
                if verbose > 0 :
                    print('WARNING. Did not find an appropriate window for analysis.')
                return []

            else :
                # TODO: increase only threshold of the criterion with the smallest/zero True range?
                cvi_percentile += 0.05
                if cvi_percentile > 1.0 :
                    cvi_percentile = 1.0
                cva_percentile += 0.05
                if cva_percentile > 1. :
                    cva_percentile = 1.
                ampl_percentile -= 0.05
                if ampl_percentile < 0. :
                    ampl_percentile = 0.

                if verbose > 1 :
                    print('  rerunning best_window_algorithm() with cvi_th=%.2f, cva_percentile=%.2f, ampl_percentile=%.2f' % (cvi_percentile, cva_percentile, ampl_percentile))
                return best_window_algorithm(cvi_percentile, cva_percentile, ampl_percentile)
        else:
            if plot_window_func :
                plot_window_func(cvi_th, ampl_th, cva_th, **kwargs)
            if verbose > 1 :
                print('found best window')
            return valid_windows


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

    # clipping amplitudes:
    min_ampl = np.min(data)
    max_ampl = np.max(data)
    # TODO: do this per window, see 0517L16.WAV
    h, b = np.histogram(data, np.linspace(min_ampl, max_ampl, 20, endpoint=True))
    if h[0] > 2.0*h[2] and b[0] < -0.4 :
        if h[1] > 2.0*h[2] :
            min_ampl = b[2]
        else :
            min_ampl = b[1]
    if h[-1] > 2.0*h[-3] and b[-1] > 0.4 :
        if h[-2] > 2.0*h[-3] :
            max_ampl = b[-3]
        else :
            max_ampl = b[-2]
    #import matplotlib.pyplot as plt
    #plt.plot(data)
    #plt.bar(b[:-1], h, width=np.mean(np.diff(b)))
    #plt.axvline(min_ampl, color='r')
    #plt.axvline(max_ampl, color='r')
    #plt.show()

    # compute cv of intervals, mean peak amplitude and its cv:
    win_sinx = win_size*rate
    win_tinxs = np.arange(0.0, len(data) - win_sinx, win_shift*rate)
    cv_interv = np.zeros(len(win_tinxs))
    mean_ampl = np.zeros(len(win_tinxs))
    cv_ampl = np.zeros(len(win_tinxs))
    for i, wtinx in enumerate(win_tinxs):
        # indices of peaks and troughs inside analysis window:
        pinx = (peak_idx >= wtinx) & (peak_idx <= wtinx + win_sinx)
        tinx = (trough_idx >= wtinx) & (trough_idx <= wtinx + win_sinx)
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
            if wtinx + win_sinx - p_idx[-1] > mean_interv :
                cv_interv[i] *= (wtinx + win_sinx - p_idx[-1])/mean_interv
        else :
            cv_interv[i] = 1000.0
        # statistics of peak-to-trough amplitude:
        p2t_ampl = data[p_idx] - data[t_idx]
        if len(p2t_ampl) > 0 :
            mean_ampl[i] = np.mean(p2t_ampl)
            cv_ampl[i] = np.std(p2t_ampl) / mean_ampl[i]
            # penalize for clipped peaks:
            clipped_frac = float(np.sum(data[p_idx]>max_ampl)+np.sum(data[t_idx]<min_ampl))/2.0/len(p2t_ampl)
            #if clipped_frac > 0.01 :
            cv_ampl[i] += 10.0*clipped_frac
        else :
            mean_ampl[i] = 0.0
            cv_ampl[i] = 1000.0
    # TODO: check for empty data here and exit!
    # cumulative functions:
    ampl_sorted = np.sort(mean_ampl)
    ampl_sorted = ampl_sorted[ampl_sorted>0.0]
    cv_interval_sorted = np.sort(cv_interv)
    cv_interval_sorted = cv_interval_sorted[cv_interval_sorted<1000.0]
    cvi_percentile = float(len(cv_interval_sorted[cv_interval_sorted<cvi_th])/float(len(cv_interval_sorted)))
    cv_ampl_sorted = np.sort(cv_ampl)
    cv_ampl_sorted = cv_ampl_sorted[cv_ampl_sorted<1000.0]
    cva_percentile = float(len(cv_ampl_sorted[cv_ampl_sorted<cva_th])/float(len(cv_ampl_sorted)))
    # TODO: this needs to be an argument???
    ampl_percentile = 0.85
    
    # find best window:
    valid_wins = best_window_algorithm(cvi_percentile, cva_percentile, ampl_percentile)

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
                       win_tinxs/rate, cv_interv, mean_ampl, cv_ampl, valid_wins, **kwargs)

    return idx0, idx1


# TODO: make sure the arguments are still right!
def best_window_times(data, rate, mode='first',
                        min_thresh=0.1, thresh_fac=0.75, thresh_frac=0.02, thresh_tau=1.0,
                        win_size=8., win_shift=0.1, cvi_th=0.05, cva_th=0.05,
                        verbose=0, plot_data_func=None, plot_window_func=None, **kwargs):
    """
    Finds the window within data with the best data. See best_window_indices() for details.

    Returns:
      start_time (float): Time of the start of the best window.
      end_time (float): Time of the end of the best window.
    """
    start_inx, end_inx = best_window_indices(data, rate, mode,
                        min_thresh, thresh_fac, thresh_frac, thresh_tau,
                        win_size, win_shift, cvi_th, cva_th,
                        verbose, plot_data_func, plot_window_func, **kwargs)
    return start_inx/rate, end_inx/rate


# TODO: make sure the arguments are still right!
def best_window(data, rate, mode='first',
                min_thresh=0.1, thresh_fac=0.75, thresh_frac=0.02, thresh_tau=1.0,
                win_size=8., win_shift=0.1, cvi_th=0.05, cva_th=0.05,
                verbose=0, plot_data_func=None, plot_window_func=None, **kwargs):
    """
    Finds the window within data with the best data. See best_window_indices() for details.

    Returns:
      data (array): the data of the best window.
    """
    start_inx, end_inx = best_window_indices(data, rate, mode,
                        min_thresh, thresh_fac, thresh_frac, thresh_tau,
                        win_size, win_shift, cvi_th, cva_th,
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

    best_window_indices(data, rate, mode='first',
                        min_thresh=0.1, thresh_fac=0.8, thresh_frac=0.02, thresh_tau=0.25,
                        win_size=0.2, win_shift=0.1)
