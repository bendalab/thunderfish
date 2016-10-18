"""
Detecting peaks in arrays.

detect_peaks(): peak and trough detection with a relative threshold.
detect_dynamic_peaks(): peak and trough detection with a dynamically adapted threshold.

accept_peak(): make detect_peaks() return index/time and size of peaks.
accept_peak_size_threshold(): adapt the dection threshold to the size of the detected peaks.
accept_peaks_size_width(): make detect_peaks() return time, height, size, and width of peaks.

std_threshold(): estimate detection threshold based on the standard deviation.
minmax_threshold(): estimate detection threshold based on maximum minus minimum value.
percentile_threshold(): estimate detection threshold based on interpercentile range.

snippets(): cut out data snippets around a list of indices.

trim(): make the list of peaks and troughs returned by detect_peaks() the same length.
trim_to_peak(): ensure that the peak is first.
trim_closest(): ensure that peaks minus troughs is smallest.
"""

import numpy as np


def detect_peaks(data, threshold, time=None,
                 check_peak_func=None, check_trough_func=None, **kwargs):
    """
    Detect peaks and troughs using a fixed, relative threshold according to
    Bryan S. Todd and David C. Andrews (1999): The identification of peaks in physiological signals.
    Computers and Biomedical Research 32, 322-335.

    Args:
        data (array): an 1-D array of input data where peaks are detected
        threshold (float or array): a positive number setting the minimum distance between peaks and troughs
        time (array): the (optional) 1-D array with the time corresponding to the data values
        check_peak_func (function): an optional function to be used for further evaluating and analysing a peak
          The signature of the function is
          r, th = check_peak_func(time, data, peak_inx, index, min_inx, threshold, **kwargs)
          with
            time (array): the full time array that might be None
            data (array): the full data array
            peak_inx (int): the index of the  detected peak
            index (int): the current index
            min_inx (int): the index of the trough preceeding the peak (might be 0)
            threshold (float): the threshold value
            **kwargs: further arguments
            r (scalar or np.array): a single number or an array with properties of the peak or None to skip the peak
            th (float): a new value for the threshold or None (to keep the original value)
        check_trough_func (function): an optional function to be used for further evaluating and analysing a trough
          The signature of the function is
          r, th = check_trough_func(time, data, trough_inx, index, max_inx, threshold, **kwargs)
          with
            time (array): the full time array that might be None
            data (array): the full data array
            trough_inx (int): the index of the  detected trough
            index (int): the current index
            max_inx (int): the index of the peak preceeding the trough (might be 0)
            threshold (float): the threshold value
            **kwargs: further arguments
            r (scalar or np.array): a single number or an array with properties of the trough
                                    or None to skip the trough
            th (float): a new value for the threshold (is overwritten by an threshold array)
                        or None (to keep the original value)            
        kwargs: arguments passed on to check_peak_func and check_trough_func
    
    Returns: 
        peak_list (np.array): a list of peaks
        trough_list (np.array): a list of troughs
          if time is None and no check_peak_func is given, then these are lists of the indices where the peaks/troughs occur.
          if time is given and no check_peak_func/check_trough_func is given, then these are lists of the times where the peaks/troughs occur.
          if check_peak_func or check_trough_func is given, then these are lists of whatever check_peak_func/check_trough_func return.
    """

    thresh_array = True
    thresh = 0.0
    if np.isscalar(threshold):
        thresh_array = False
        thresh = threshold
        if threshold <= 0:
            raise ValueError('input argument threshold must be positive!')
    elif len(data) != len(threshold):
        raise ValueError('input arrays data and threshold must have same length!')
    if time is not None and len(data) != len(time):
        raise ValueError('input arrays time and data must have same length!')

    peaks_list = list()
    troughs_list = list()

    # initialize:
    dir = 0
    min_inx = 0
    max_inx = 0
    min_value = data[0]
    max_value = min_value

    # loop through the data:
    for index, value in enumerate(data):

        if thresh_array:
            thresh = threshold[index]

        # rising?
        if dir > 0:
            # if the new value is bigger than the old maximum: set it as new maximum:
            if value > max_value:
                max_inx = index  # maximum element
                max_value = value

            # otherwise, if the new value is falling below the maximum value minus the threshold:
            # the maximum is a peak!
            elif max_value >= value + thresh:
                # check and update peak with the check_peak_func function:
                if check_peak_func:
                    r, th = check_peak_func(time, data, max_inx, index,
                                            min_inx, thresh, **kwargs)
                    if r is not None:
                        # this really is a peak:
                        peaks_list.append(r)
                    if th is not None:
                        thresh = th
                else:
                    # this is a peak:
                    if time is None:
                        peaks_list.append(max_inx)
                    else:
                        peaks_list.append(time[max_inx])

                # change direction:
                min_inx = index  # minimum element
                min_value = value
                dir = -1

        # falling?
        elif dir < 0:
            if value < min_value:
                min_inx = index  # minimum element
                min_value = value

            elif value >= min_value + thresh:
                # there was a trough:

                # check and update trough with the check_trough function:
                if check_trough_func:
                    r, th = check_trough_func(time, data, min_inx, index,
                                              max_inx, thresh, **kwargs)
                    if r is not None:
                        # this really is a trough:
                        troughs_list.append(r)
                    if th is not None:
                        thresh = th
                else:
                    # this is a trough:
                    if time is None:
                        troughs_list.append(min_inx)
                    else:
                        troughs_list.append(time[min_inx])

                # change direction:
                max_inx = index  # maximum element
                max_value = value
                dir = 1

        # don't know direction yet:
        else:
            if max_value >= value + thresh:
                dir = -1  # falling
            elif value >= min_value + thresh:
                dir = 1  # rising

            if max_value < value:
                max_inx = index  # maximum element
                max_value = value

            elif value < min_value:
                min_inx = index  # minimum element
                min_value = value

    return np.asarray(peaks_list), np.asarray(troughs_list)


def detect_dynamic_peaks(data, threshold, min_thresh, tau, time=None,
                         check_peak_func=None, check_trough_func=None, **kwargs):
    """
    Detect peaks and troughs using a relative threshold according to
    Bryan S. Todd and David C. Andrews (1999): The identification of peaks in physiological signals.
    Computers and Biomedical Research 32, 322-335.
    The threshold decays dynamically towards min_thresh with time constant tau.
    Use check_peak_func or check_trough_func to reset the threshold to an appropriate size.

    Args:
        data (array): an 1-D array of input data where peaks are detected
        threshold (float): a positive number setting the minimum distance between peaks and troughs
        min_thresh (float): the minimum value the threshold is allowed to assume.
        tau (float): the time constant of the the decay of the threshold value
                     given in indices (time is None) or time units (time is not None)
        time (array): the (optional) 1-D array with the time corresponding to the data values
        check_peak_func (function): an optional function to be used for further evaluating and analysing a peak
          The signature of the function is
          r, th = check_peak_func(time, data, peak_inx, index, min_inx, threshold, **kwargs)
          with
            time (array): the full time array that might be None
            data (array): the full data array
            peak_inx (int): the index of the  detected peak
            index (int): the current index
            min_inx (int): the index of the trough preceeding the peak (might be 0)
            threshold (float): the threshold value
            min_thresh (float): the minimum value the threshold is allowed to assume.
            tau (float): the time constant of the the decay of the threshold value
                         given in indices (time is None) or time units (time is not None)
            **kwargs: further keyword arguments provided by the user
            r (scalar or np.array): a single number or an array with properties of the peak or None to skip the peak
            th (float): a new value for the threshold or None (to keep the original value)
        check_trough_func (function): an optional function to be used for further evaluating and analysing a trough
          The signature of the function is
          r, th = check_trough_func(time, data, trough_inx, index, max_inx, threshold, **kwargs)
          with
            time (array): the full time array that might be None
            data (array): the full data array
            trough_inx (int): the index of the  detected trough
            index (int): the current index
            max_inx (int): the index of the peak preceeding the trough (might be 0)
            threshold (float): the threshold value
            min_thresh (float): the minimum value the threshold is allowed to assume.
            tau (float): the time constant of the the decay of the threshold value
                         given in indices (time is None) or time units (time is not None)
            **kwargs: further keyword arguments provided by the user
            r (scalar or np.array): a single number or an array with properties of the trough or None to skip the trough
            th (float): a new value for the threshold or None (to keep the original value)            
        kwargs: arguments passed on to check_peak_func and check_trough_func
    
    Returns: 
        peak_list (np.array): a list of peaks
        trough_list (np.array): a list of troughs
          if time is None and no check_peak_func is given, then these are lists of the indices where the peaks/troughs occur.
          if time is given and no check_peak_func/check_trough_func is given, then these are lists of the times where the peaks/troughs occur.
          if check_peak_func or check_trough_func is given, then these are lists of whatever check_peak_func/check_trough_func return.
    """

    if threshold <= 0:
        raise ValueError('input argument threshold must be positive!')
    if min_thresh <= 0:
        raise ValueError('input argument min_thresh must be positive!')
    if tau <= 0:
        raise ValueError('input argument tau must be positive!')
    if time is not None and len(data) != len(time):
        raise ValueError('input arrays time and data must have same length!')

    peaks_list = list()
    troughs_list = list()

    # initialize:
    dir = 0
    min_inx = 0
    max_inx = 0
    min_value = data[0]
    max_value = min_value

    # loop through the data:
    for index, value in enumerate(data):

        # decaying threshold (1. order low pass filter):
        if time is None:
            threshold += (min_thresh - threshold) / tau
        else:
            idx = index
            if idx + 1 >= len(data):
                idx = len(data) - 2
            threshold += (min_thresh - threshold) * (time[idx + 1] - time[idx]) / tau

        # rising?
        if dir > 0:
            # if the new value is bigger than the old maximum: set it as new maximum:
            if value > max_value:
                max_inx = index  # maximum element
                max_value = value

            # otherwise, if the new value is falling below the maximum value minus the threshold:
            # the maximum is a peak!
            elif max_value >= value + threshold:
                # check and update peak with the check_peak_func function:
                if check_peak_func:
                    r, th = check_peak_func(time, data, max_inx, index,
                                            min_inx, threshold,
                                            min_thresh=min_thresh, tau=tau, **kwargs)
                    if r is not None:
                        # this really is a peak:
                        peaks_list.append(r)
                    if th is not None:
                        threshold = th
                        if threshold < min_thresh:
                            threshold = min_thresh
                else:
                    # this is a peak:
                    if time is None:
                        peaks_list.append(max_inx)
                    else:
                        peaks_list.append(time[max_inx])

                # change direction:
                min_inx = index  # minimum element
                min_value = value
                dir = -1

        # falling?
        elif dir < 0:
            if value < min_value:
                min_inx = index  # minimum element
                min_value = value

            elif value >= min_value + threshold:
                # there was a trough:

                # check and update trough with the check_trough function:
                if check_trough_func:
                    r, th = check_trough_func(time, data, min_inx, index,
                                              max_inx, threshold,
                                              min_thresh=min_thresh, tau=tau, **kwargs)
                    if r is not None:
                        # this really is a trough:
                        troughs_list.append(r)
                    if th is not None:
                        threshold = th
                        if threshold < min_thresh:
                            threshold = min_thresh
                else:
                    # this is a trough:
                    if time is None:
                        troughs_list.append(min_inx)
                    else:
                        troughs_list.append(time[min_inx])

                # change direction:
                max_inx = index  # maximum element
                max_value = value
                dir = 1

        # don't know direction yet:
        else:
            if max_value >= value + threshold:
                dir = -1  # falling
            elif value >= min_value + threshold:
                dir = 1  # rising

            if max_value < value:
                max_inx = index  # maximum element
                max_value = value

            elif value < min_value:
                min_inx = index  # minimum element
                min_value = value

    return np.asarray(peaks_list), np.asarray(troughs_list)


def accept_peak(time, data, event_inx, index, min_inx, threshold):
    """
    Accept each detected peak/trough and return its index (or time) and its data value.

    Args:
        freqs (array): frequencies of the power spectrum
        data (array): the power spectrum
        event_inx: index of the current peak/trough
        index: current index
        min_inx: index of the previous trough/peak
        threshold: threshold value
    
    Returns: 
        index (int): index of the peak/trough
        time (float): time of the peak/trough if time is not None
        value (float): value of data at the peak
    """
    size = data[event_inx]
    if time is None:
        return [event_inx, size], None
    else:
        return [event_inx, time[event_inx], size], None


def accept_peak_size_threshold(time, data, event_inx, index, min_inx, threshold,
                               min_thresh, tau, thresh_ampl_fac=0.75, thresh_weight=0.02):
    """Accept each detected peak/trough and return its index or time.
    Adjust the threshold to the size of the detected peak.
    To be passed to the detect_dynamic_peaks() function.

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
        index (int): index of the peak/trough if time is None
        time (int): time of the peak/trough if time is not None
        threshold (float): the new threshold to be used
    """
    size = data[event_inx] - data[min_inx]
    threshold += thresh_weight * (thresh_ampl_fac * size - threshold)
    if time is None:
        return event_inx, threshold
    else:
        return time[event_inx], threshold


def accept_peaks_size_width(time, data, peak_inx, index, min_inx, threshold, pfac=0.75):
    """
    Accept each detected peak and compute its size and width.

    Args:
        time (array): time, must not be None
        data (array): the data with teh peaks
        peak_inx: index of the current peak
        index: current index
        min_inx: index of the previous trough
        threshold: threshold value
        pfac: fraction of peak height where its width is measured
    
    Returns: 
        time (float): time of the peak
        height (float): height of the peak (value of data at the peak)
        size (float): size of the peak (peak minus previous trough)
        width (float): width of the peak at 0.75*size
        count (float): zero
    """

    size = data[peak_inx] - data[min_inx]
    wthresh = data[min_inx] + pfac * size
    width = 0.0
    for k in range(peak_inx, min_inx, -1):
        if data[k] < wthresh:
            width = time[peak_inx] - time[k]
            break
    for k in range(peak_inx, index):
        if data[k] < wthresh:
            width += time[k] - time[peak_inx]
            break
    return [time[peak_inx], data[peak_inx], size, width, 0.0], None


def std_threshold(data, samplerate=None, win_size=None, th_factor=5.):
    """Esimates a threshold for detect_peaks() based on the standard deviation of the data.

    The threshold is computed as the standard deviation of the data multiplied with th_factor.

    If samplerate and win_size is given, then the threshold is computed for
    each non-overlapping window of duration win_size separately.
    In this case the returned threshold is an array of the same size as data.
    Without a samplerate and win_size a single threshold value determined from
    the whole data array is returned.

    :param data: (1-D array). The data to be analyzed.
    :param samplerate: (float or None). Sampling rate of the data in Hz.
    :param win_size: (float or None). Size of window in which a threshold value is computed.
    :param th_factor: (float). Factor by which the standard deviation is multiplied to set the threshold.
    :return: threshold: (float or 1-D array). The computed threshold.
    """

    if samplerate and win_size:
        threshold = np.zeros(len(data))
        win_size_indices = int(win_size * samplerate)

        for inx0 in range(0, len(data), win_size_indices):
            inx1 = inx0 + win_size_indices
            std = np.std(data[inx0:inx1], ddof=1)
            threshold[inx0:inx1] = std * th_factor
        return threshold
    else:
        return np.std(data, ddof=1) * th_factor

    
def hist_threshold(data, samplerate=None, win_size=None, th_factor=5.,
                   nbins=100, hist_height=1.0/ np.sqrt(np.e)):
    """Esimates a threshold for detect_peaks() based on a histogram of the data.

    The standard deviation of the data is estimated from the
    width of the histogram of the data at hist_height relative height.

    If samplerate and win_size is given, then the threshold is computed for
    each non-overlapping window of duration win_size separately.
    In this case the returned threshold is an array of the same size as data.
    Without a samplerate and win_size a single threshold value determined from
    the whole data array is returned.

    Args:
        data: the data from which to estimate the threshold.
        noise_factor (float): multiplies the estimate of the standard deviation
                              of the noise to result in the low_threshold
                              
        data (1-D array): the data to be analyzed.
        samplerate (float or None): sampling rate of the data in Hz.
        win_size (float or None): Size of window in which a threshold value is computed in sec.
        th_factor (float): factor by which the width of the histogram is multiplied to set the threshold.
        nbins (int or list of floats): number of bins or the bins for computing the histogram.
        hist_height (float): height between 0 and 1 at which the width of the histogram is computed.

    Returns:
        threshold (float or 1-D array): the computed threshold.
        center (float or 1-D array): the center (mean) of the width of the histogram.
    """

    if samplerate and win_size:
        threshold = np.zeros(len(data))
        centers = np.zeros(len(data))
        win_size_indices = int(win_size * samplerate)

        for inx0 in range(0, len(data), win_size_indices):
            inx1 = inx0 + win_size_indices
            hist, bins = np.histogram(data[inx0:inx1], nbins, density=True)
            inx = hist > np.max(hist) * hist_height
            lower = bins[0:-1][inx][0]
            upper = bins[1:][inx][-1]  # needs to return the next bin
            center = 0.5 * (lower + upper)
            std = 0.5 * (upper - lower)
            threshold[inx0:inx1] = std * th_factor
            centers[inx0:inx1] = center
        return threshold, centers
    else:
        hist, bins = np.histogram(data, nbins, density=True)
        inx = hist > np.max(hist) * hist_height
        lower = bins[0:-1][inx][0]
        upper = bins[1:][inx][-1]  # needs to return the next bin
        center = 0.5 * (lower + upper)
        std = 0.5 * (upper - lower)
        return std * th_factor, center

    
def minmax_threshold(data, samplerate=None, win_size=None, th_factor=0.8):
    """Esimate a threshold for detect_peaks() based on minimum and maximum values of the data.

    The threshold is computed as the difference between maximum and
    minimum value of the data multiplied with th_factor.

    If samplerate and win_size is given, then the threshold is computed for
    each non-overlapping window of duration win_size separately.
    In this case the returned threshold is an array of the same size as data.
    Without a samplerate and win_size a single threshold value determined from
    the whole data array is returned.

    :param data: (1-D array). The data to be analyzed.
    :param samplerate: (float or None). Sampling rate of the data in Hz.
    :param win_size: (float or None). Size of window in which a threshold value is computed.
    :param th_factor: (float). Factor by which the difference between minimum and maximum data value is multiplied to set the threshold.

    :return: threshold: (float or 1-D array). The computed threshold.
    """
    if samplerate and win_size:
        threshold = np.zeros(len(data))
        win_size_indices = int(win_size * samplerate)

        for inx0 in range(0, len(data), win_size_indices):
            inx1 = inx0 + win_size_indices

            window_min = np.min(data[inx0:inx1])
            window_max = np.max(data[inx0:inx1])

            threshold[inx0:inx1] = (window_max - window_min) * th_factor
        return threshold

    else:
        return (np.max(data) - np.min(data)) * th_factor


def percentile_threshold(data, samplerate=None, win_size=None, th_factor=0.8, percentile=0.1):
    """Esimate a threshold for detect_peaks() based on an inter-percentile range of the data.

    The threshold is computed as the range between the percentile and
    100.0-percentile percentiles of the data multiplied with
    th_factor.

    For very small values of percentile the estimated threshold
    approaches the one returned by minmax_threshold() (for same values
    of th_factor). For percentile=16.0 and Gaussian distributed data,
    the returned theshold is twice the one returned by std_threshold()
    (two times the standard deviation).

    If samplerate and win_size is given, then the threshold is computed for
    each non-overlapping window of duration win_size separately.
    In this case the returned threshold is an array of the same size as data.
    Without a samplerate and win_size a single threshold value determined from
    the whole data array is returned.

    :param data: (1-D array). The data to be analyzed.
    :param samplerate: (float or None). Sampling rate of the data in Hz.
    :param win_size: (float or None). Size of window in which a threshold value is computed.
    :param percentile: (int). The interpercentile range is computed at percentile and 100.0-percentile.
    :param th_factor: (float). Factor by which the inter-percentile range of the data is multiplied to set the threshold.

    :return: threshold: (float or 1-D array). The computed threshold.
    """
    if samplerate and win_size:
        threshold = np.zeros(len(data))
        win_size_indices = int(win_size * samplerate)

        for inx0 in range(0, len(data), win_size_indices):
            inx1 = inx0 + win_size_indices
            threshold[inx0:inx1] = np.squeeze(np.abs(np.diff(
                np.percentile(data[inx0:inx1], [100.0 - percentile, percentile])))) * th_factor
        return threshold
    else:
        return np.squeeze(np.abs(np.diff(
            np.percentile(data, [100.0 - percentile, percentile])))) * th_factor


def snippets(data, indices, start=-10, stop=10):
    """
    Cut out data arround each position given in indices.

    :param data: (1-D array) Data array from which snippets are extracted.
    :param indices: (list of int) Indices around which snippets are cut out.
    :param start: (int) Each snippet starts at index + start.
    :param stop: (int) Each snippet ends at index + stop.
    :return snippet_data: (2-D array) The snippets: first index number of snippet, second index time.
    """
    idxs = indices[(indices>=-start) & (indices<len(data)-stop)]
    snippet_data = np.empty((len(idxs), stop-start))
    for k, idx in enumerate(idxs):
        snippet_data[k] = data[idx+start:idx+stop]
    return snippet_data


def trim(peaks, troughs):
    """
    Trims the peaks and troughs arrays such that they have the same length.
    
    Args:
        peaks (numpy array): list of peak indices or times
        troughs (numpy array): list of trough indices or times

    Returns:
        peaks (numpy array): list of peak indices or times
        troughs (numpy array): list of trough indices or times
    """
    # common len:
    n = min(len(peaks), len(troughs))
    # align arrays:
    return peaks[:n], troughs[:n]


def trim_to_peak(peaks, troughs):
    """
    Trims the peaks and troughs arrays such that they have the same length
    and the first peak comes first.
    
    Args:
        peaks (numpy array): list of peak indices or times
        troughs (numpy array): list of trough indices or times

    Returns:
        peaks (numpy array): list of peak indices or times
        troughs (numpy array): list of trough indices or times
    """
    # start index for troughs:
    tidx = 0
    if len(peaks) > 0 and len(troughs) > 0 and troughs[0] < peaks[0]:
        tidx = 1
    # common len:
    n = min(len(peaks), len(troughs[tidx:]))
    # align arrays:
    return peaks[:n], troughs[tidx:tidx + n]


def trim_closest(peaks, troughs):
    """
    Trims the peaks and troughs arrays such that they have the same length
    and that peaks-troughs is on average as small as possible.
    
    Args:
        peaks (numpy array): list of peak indices or times
        troughs (numpy array): list of trough indices or times

    Returns:
        peaks (numpy array): list of peak indices or times
        troughs (numpy array): list of trough indices or times
    """
    pidx = 0
    tidx = 0
    nn = min(len(peaks), len(troughs))
    if nn == 0:
        return np.array([]), np.array([])
    dist = np.abs(np.mean(peaks[:nn] - troughs[:nn]))
    if len(peaks) == 0 or len(troughs) == 0:
        nn = 0
    else:
        if peaks[0] < troughs[0]:
            nnp = min(len(peaks[1:]), len(troughs))
            distp = np.abs(np.mean(peaks[1:nnp] - troughs[:nnp - 1]))
            if distp < dist:
                pidx = 1
                nn = nnp
        else:
            nnt = min(len(peaks), len(troughs[1:]))
            distt = np.abs(np.mean(peaks[:nnt - 1] - troughs[1:nnt]))
            if distt < dist:
                tidx = 1
                nn = nnt
    # align arrays:
    return peaks[pidx:pidx + nn], troughs[tidx:tidx + nn]


if __name__ == "__main__":
    print("Checking peakdetection module ...")
    import matplotlib.pyplot as plt

    print('')
    # generate data:
    time = np.arange(0.0, 10.0, 0.01)
    f = 2.0
    data = (0.5 * np.sin(2.0 * np.pi * f * time) + 0.5) ** 4.0
    data += -0.1 * time * (time - 10.0)
    data += 0.1 * np.random.randn(len(data))

    print("generated waveform with %d peaks" % int(np.round(time[-1] * f)))
    plt.plot(time, data)

    print('')
    print('check detect_peaks(data, 0.5, time)...')
    peaks, troughs = detect_peaks(data, 0.5, time)
    # print peaks
    print('detected %d peaks with period %g that differs from the real frequency by %g' % (
        len(peaks), np.mean(np.diff(peaks)), f - 1.0 / np.mean(np.diff(peaks))))
    # print troughs
    print('detected %d troughs with period %g that differs from the real frequency by %g' % (
        len(troughs), np.mean(np.diff(troughs)), f - 1.0 / np.mean(np.diff(troughs))))

    print('')
    print('check detect_peaks(data, 0.5)...')
    peaks, troughs = detect_peaks(data, 0.5)
    # print peaks
    print('detected %d peaks with period %g that differs from the real frequency by %g' % (
        len(peaks), np.mean(np.diff(peaks)), f - 1.0 / np.mean(np.diff(peaks)) / np.mean(np.diff(time))))
    # print troughs
    print('detected %d troughs with period %g that differs from the real frequency by %g' % (
        len(troughs), np.mean(np.diff(troughs)), f - 1.0 / np.mean(np.diff(troughs)) / np.mean(np.diff(time))))

    print('')
    print('check detect_peaks(data, 0.5, time, accept_peak, accept_peak)...')
    peaks, troughs = detect_peaks(data, 0.5, time, accept_peak, accept_peak)
    # print peaks
    print('detected %d peaks with period %g that differs from the real frequency by %g' % (
        len(peaks), np.mean(np.diff(peaks[:, 1])), f - 1.0 / np.mean(np.diff(peaks[:, 1]))))
    # print troughs
    print('detected %d troughs with period %g that differs from the real frequency by %g' % (
        len(troughs), np.mean(np.diff(troughs[:, 1])), f - 1.0 / np.mean(np.diff(troughs[:, 1]))))
    plt.plot(peaks[:, 1], peaks[:, 2], '.r', ms=20)
    plt.plot(troughs[:, 1], troughs[:, 2], '.g', ms=20)

    plt.ylim(-0.5, 4.0)
    plt.show()
