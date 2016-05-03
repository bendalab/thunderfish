"""Functions needed for selecting the region within a recording with the
most stable signal of largest amplitude that is not clipped.

Main functions:
clip_amplitudes(): estimated clipping amplitudes from the data.
best_window_indices(): select start- and end-indices of the best window
best_window_times(): select start end end-time of the best window
best_window(): return data of the best window

See bestwindowplots module for visualizing the best_window algorithm
and for usage.
"""

import warnings
import numpy as np
import scipy.stats as stats
import peakdetection as pd


def clip_amplitudes(data, win_indices, min_fac=2.0, nbins=20) :
    """Find the amplitudes where the signals clips by looking at
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

    
def best_window_indices(data, rate, expand=False,
                        min_thresh=0.1, thresh_ampl_fac=0.8, thresh_weight=0.02, thresh_tau=1.0,
                        win_size=8., win_shift=0.1, min_clip=-np.inf, max_clip=np.inf,
                        w_cv_interv=1.0, w_ampl=1.0, w_cv_ampl=1.0, tolerance=0.5,
                        verbose=0, plot_data_func=None, plot_window_func=None, **kwargs):
    """Detect the best window of the data to be analyzed. The data have been sampled with rate Hz.
    
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

    Third, a cost function is computed as a weightted sum of the three criteria
    (mean-amplitude is taken negatively). The weights are given by w_cv_interv, w_ampl, and w_cv_ampl.

    Finally, a threshold is set to the minimum value of the cost function plus tolerance.
    Then the largest region with the cost function below this threshold is selected as the best window.
    If expand is False, then only the single window with smalles cost
    within the selected largest region is returned.

    Output of warning and info messages to console can be controlled by setting verbose. No output is produced
    if verbose = 1. higher values produce more output.

    The algorithm can be visualized by supplying the functions plot_data_func and plot_window_func.
    Additional arguments for these function are supplied vie kwargs.

    :param data: 1-D array. The data to be analyzed
    :param rate: float. Sampling rate of the data in Hz
    :param expand: boolean. If true return the largest region with small costs.
    :param min_thresh: float. Minimum allowed value for the threshold. Set this above the noise level of the data.
    :param thresh_ampl_fac: float. New threshold is thresh_ampl_fac times the size of the current peak, between 0 and 1. Set this close to 1. The smaller the more small amplitude peaks are detected.
    :param thresh_weight: float. New threshold is weighed against current threshold with thresh_weight. The inverse of thresh_weight is approximately the number of peaks need for the threshold to approach the new threshold value.
    :param thresh_tau: float. Time constant of the decay of the threshold towards min_thresh in seconds.
    This should approximately match the fastest changes in signal amplitude.
    :param win_size: float. Size of the best window in seconds. Choose it large enough for a minimum analysis.
    :param win_shift: float. Time shift in seconds between windows. Should be smaller or equal to win_size and not smaller than about one thenth of win_shift.
    :param min_clip: float. Minimum amplitude below which data are clipped.
    :param max_clip: float. Maximum amplitude above which data are clipped.
    :param w_cv_interv: float. Weight for the coefficient of variation of the intervals.
    :param w_ampl: float. Weight for the mean peak-to-trough amplitude.
    :param w_cv_ampl: float. Weight for the coefficient of variation of the amplitudes.
    :param tolerance: float. Added to the minimum cost for selecting the region of best windows.
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

    # too little data:
    if len(data) / rate <= win_size :
        warnings.warn('no best window found: not enough data')
        return 0, 0

    # detect large peaks and troughs:
    thresh = 2.0*np.std(data[0:win_shift*rate])
    tauidx = thresh_tau*rate
    peak_idx, trough_idx = pd.detect_dynamic_peaks(data, thresh, min_thresh, tauidx, None,
                                                   pd.accept_peak_size_threshold, None,
                                                   thresh_ampl_fac=thresh_ampl_fac,
                                                   thresh_weight=thresh_weight)
    if len(peak_idx) == 0 or len(trough_idx) == 0 :
        if verbose > 0 :
            print('best_window(): no peaks and troughs detected')
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
        if len(ipis) > 2 :
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
        if len(p2t_ampl) > 2 :
            mean_ampl[i] = np.mean(p2t_ampl)
            cv_ampl[i] = np.std(p2t_ampl) / mean_ampl[i]
            # penalize for clipped peaks:
            clipped_frac = float(np.sum(data[p_idx]>max_clip) +
                                 np.sum(data[t_idx]<min_clip))/2.0/len(p2t_ampl)
            mean_ampl[i] *= (1.0-clipped_frac)**2.0
        else :
            mean_ampl[i] = 0.0
            cv_ampl[i] = 1000.0

    # cost function:
    cost = w_cv_interv*cv_interv + w_cv_ampl*cv_ampl - w_ampl*mean_ampl
    thresh = np.min(cost) + tolerance
    valid_wins = cost <= thresh

    # find largest region with low costs:
    valid_win_idx = np.nonzero(valid_wins)[0]
    win_idx0 = 0
    win_idx1 = 0
    for lidx0 in valid_win_idx :
        lidx1 = lidx0
        while lidx1<len(valid_wins) and valid_wins[lidx1] :
            lidx1 += 1
        if lidx1 - lidx0 > win_idx1 - win_idx0 :
            win_idx0 = lidx0
            win_idx1 = lidx1

    # find single best window within the largest region:
    if not expand :
        win_idx0 += np.argmin(cost[win_idx0:win_idx1])
        win_idx1 = win_idx0 + 1

    # retrive indices of best window for data:
    idx0 = win_start_inxs[win_idx0]
    idx1 = win_start_inxs[win_idx1-1]+win_size_indices
        
    if plot_data_func :
        valid_wins[:win_idx0] = False
        valid_wins[win_idx1:] = False
        plot_data_func(data, rate, peak_idx, trough_idx, idx0, idx1,
                       win_start_inxs/rate, cv_interv, mean_ampl, cv_ampl, valid_wins, **kwargs)

    return idx0, idx1


def best_window_times(data, rate, expand=False,
                        min_thresh=0.1, thresh_ampl_fac=0.8, thresh_weight=0.02, thresh_tau=1.0,
                        win_size=8., win_shift=0.1, min_clip=-np.inf, max_clip=np.inf,
                        w_cv_interv=1.0, w_ampl=1.0, w_cv_ampl=1.0, tolerance=0.5,
                        verbose=0, plot_data_func=None, plot_window_func=None, **kwargs):
    """Finds the window within data with the best data. See best_window_indices() for details.

    Returns:
      start_time (float): Time of the start of the best window.
      end_time (float): Time of the end of the best window.
    """
    start_inx, end_inx = best_window_indices(data, rate, expand,
                                             min_thresh, thresh_ampl_fac, thresh_weight, thresh_tau,
                                             win_size, win_shift, min_clip, max_clip,
                                             w_cv_interv, w_ampl, w_cv_ampl, tolerance,
                                             verbose, plot_data_func, plot_window_func, **kwargs)
    return start_inx/rate, end_inx/rate


def best_window(data, rate, expand=False,
                min_thresh=0.1, thresh_ampl_fac=0.8, thresh_weight=0.02, thresh_tau=1.0,
                win_size=8., win_shift=0.1, min_clip=-np.inf, max_clip=np.inf,
                w_cv_interv=1.0, w_ampl=1.0, w_cv_ampl=1.0, tolerance=0.5,
                verbose=0, plot_data_func=None, plot_window_func=None, **kwargs):
    """Finds the window within data with the best data. See best_window_indices() for details.

    Returns:
      data (array): the data of the best window.
    """
    start_inx, end_inx = best_window_indices(data, rate, expand,
                                             min_thresh, thresh_ampl_fac, thresh_weight, thresh_tau,
                                             win_size, win_shift, min_clip, max_clip,
                                             w_cv_interv, w_ampl, w_cv_ampl, tolerance,
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
                        win_size=1.0, win_shift=0.5, min_clip=min_clip, max_clip=max_clip,
                        w_cv_ampl=10.0)
