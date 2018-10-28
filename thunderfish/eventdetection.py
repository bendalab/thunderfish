"""
# Event detection
Detecting and handling peaks and troughs and threshold crossings in data arrays.

## Peak detection
- `detect_peaks()`: peak and trough detection with a relative threshold.
- `peak_size_width()`: compute for each peak its size and width.

## Threshold crossings
- `threshold_crossings()`: detect crossings of an absolute threshold.

## Event manipulation
- `trim()`: make the list of peaks and troughs the same length.
- `trim_to_peak()`: ensure that the peak is first.
- `trim_closest()`: ensure that peaks minus troughs is smallest.

- `merge_events()`: Merge events if they are closer than a minimum distance.
- `remove_events()`: Remove events that are too short or too long.
- `widen_events()`: Enlarge events on both sides without overlap.

## Threshold estimation
- `std_threshold()`: estimate detection threshold based on the standard deviation.
- `hist_threshold()`: esimate detection threshold based on a histogram of the data.
- `minmax_threshold()`: estimate detection threshold based on maximum minus minimum value.
- `percentile_threshold()`: estimate detection threshold based on interpercentile range.

## Snippets
- `snippets(): cut out data snippets around a list of indices.

## Peak detection with dynamic threshold:
- `detect_dynamic_peaks()`: peak and trough detection with a dynamically adapted threshold.
- `accept_peak_size_threshold()`: adapt the dection threshold to the size of the detected peaks.
"""

import numpy as np


def detect_peaks(data, threshold):
    """
    Detect peaks and troughs using a fixed, relative threshold according to
    Bryan S. Todd and David C. Andrews (1999): The identification of peaks in physiological signals.
    Computers and Biomedical Research 32, 322-335.

    Parameters
    ----------
    data: array
        An 1-D array of input data where peaks are detected.
    threshold: float or array
        A positive number or array of numbers setting the detection threshold,
        i.e. the minimum distance between peaks and troughs.
    
    Returns
    -------
    peak_array: array of ints
        A list of indices of detected peaks.
    trough_array: array of ints
        A list of indices of detected troughs.

    Raises
    ------
    ValueError: If `threshold <= 0`.
    IndexError: If `data` and `threshold` arrays differ in length.
    """

    thresh_array = True
    thresh = 0.0
    if np.isscalar(threshold):
        thresh_array = False
        thresh = threshold
        if threshold <= 0:
            raise ValueError('input argument threshold must be positive!')
    elif len(data) != len(threshold):
        raise IndexError('input arrays data and threshold must have same length!')

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
                # this is a peak:
                peaks_list.append(max_inx)
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
                troughs_list.append(min_inx)
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


def peak_size_width(time, data, peak_indices, trough_indices, pfac=0.75):
    """
    Compute for each peak its size and width.

    Parameters
    ----------
    time: array
        Time, must not be `None`.
    data: array
        The data with the peaks.
    peak_indices: array
        Indices of the peaks.
    trough_indices: array
        Indices of the troughs.
    pfac: float
        Fraction of peak height where its width is measured.
    
    Returns 
    -------
    peaks: 2-D array
        First dimension is the peak index. Second dimension is
        time, height (value of data at the peak),
        size (peak height minus height of closest trough),
        width (at `pfac` size), 0.0 (count) of the peak.
    """
    peaks = np.zeros((len(peak_indices), 5))
    if len(peak_indices) == 0:
        return peaks
    # time point of peaks:
    peaks[:, 0] = time[peak_indices]
    # height of peaks:
    peaks[:, 1] = data[peak_indices]
    # we need a trough before and after each peak:
    peak_inx = np.asarray(peak_indices, dtype=int)
    trough_inx = np.asarray(trough_indices, dtype=int)
    if len(trough_inx) == 0 or peak_inx[0] < trough_inx[0]:
         trough_inx = np.hstack((0, trough_inx))
    if peak_inx[-1] > trough_inx[-1]:
         trough_inx = np.hstack((trough_inx, len(data)-1))
    # size of peaks:
    offs = 0
    if np.mean(peak_inx - trough_inx[:-1]) < np.mean(peak_inx - trough_inx[1:]):
        peaks[:, 2] = data[peak_inx] - data[trough_inx[:-1]]
    else:
        peaks[:, 2] = data[peak_inx] - data[trough_inx[1:]]
        offs = 1
    # width of peaks:
    for j in range(len(peak_inx)):
        wthresh = data[trough_inx[j+offs]] + pfac * peaks[j, 2]
        width = 0.0
        for k in range(peak_inx[j], trough_inx[j], -1):
            if data[k] < wthresh:
                break
            width += time[k+1] - time[k]
        for k in range(peak_inx[j], trough_inx[j+1]):
            if data[k] < wthresh:
                break
            width += time[k] - time[k-1]
        peaks[j, 3] = width
    return peaks
    

def threshold_crossings(data, threshold):
    """
    Detect crossings of a threshold with positive and negative slope.

    Parameters
    ----------
    data: array
        An 1-D array of input data where threshold crossings are detected.
    threshold: float or array
        A number or array of numbers setting the threshold
        that needs to be crossed.
    
    Returns
    -------
    up_array: array of ints
        A list of indices where the threshold is crossed with positive slope.
    down_array: array of ints
        A list of indices where the threshold is crossed with negative slope.

    Raises
    ------
    IndexError: If `data` and `threshold` arrays differ in length.
    """

    if np.isscalar(threshold):
        up_array = np.nonzero((data[1:]>threshold) & (data[:-1]<=threshold))[0]
        down_array = np.nonzero((data[1:]<=threshold) & (data[:-1]>threshold))[0]
    else:
        if len(data) != len(threshold):
            raise IndexError('input arrays data and threshold must have same length!')
        up_array = np.nonzero((data[1:]>threshold[1:]) & (data[:-1]<=threshold[:-1]))[0]
        down_array = np.nonzero((data[1:]<=threshold[1:]) & (data[:-1]>threshold[:-1]))[0]
    return up_array, down_array


def trim(peaks, troughs):
    """
    Trims the peaks and troughs arrays such that they have the same length.
    
    Parameters
    ----------
    peaks: array
        List of peak indices or times.
    troughs: array
        List of trough indices or times.

    Returns
    -------
    peaks: array
        List of peak indices or times.
    troughs: array
        List of trough indices or times.
    """
    # common len:
    n = min(len(peaks), len(troughs))
    # align arrays:
    return peaks[:n], troughs[:n]


def trim_to_peak(peaks, troughs):
    """
    Trims the peaks and troughs arrays such that they have the same length
    and the first peak comes first.
    
    Parameters
    ----------
    peaks: array
        List of peak indices or times.
    troughs: array
        List of trough indices or times.

    Returns
    -------
    peaks: array
        List of peak indices or times.
    troughs: array
        List of trough indices or times.
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
    
    Parameters
    ----------
    peaks: array
        List of peak indices or times.
    troughs: array
        List of trough indices or times.

    Returns
    -------
    peaks: array
        List of peak indices or times.
    troughs: array
        List of trough indices or times.
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


def merge_events(onsets, offsets, min_distance):
    """Merge events if they are closer than a minimum distance.

    If the beginning of an event (onset, peak, or positive threshold crossing,
    is too close to the end of the previous event (offset, trough, or negative
    threshold crossing) the two events are merged into a single one that begins
    with the first one and ends with the second one.
    
    Parameters
    ----------
    onsets: 1-D array
        The onsets (peaks, or positive threshold crossings) of the events
        as indices or times.
    offsets: 1-D array
        The offsets (troughs, or negative threshold crossings) of the events
        as indices or times.
    min_distance: int or float
        The minimum distance between events. If the beginning of an event is separated
        from the end of the previous event by less than this distance then the two events
        are merged into one. If the event onsets and offsets are given in indices than
        min_distance is also in indices. 

    Returns
    -------
    merged_onsets: 1-D array
        The onsets (peaks, or positive threshold crossings) of the merged events
        as indices or times according to onsets.
    merged_offsets: 1-D array
        The offsets (troughs, or negative threshold crossings) of the merged events
        as indices or times according to offsets.
    """
    onsets, offsets = trim_to_peak(onsets, offsets)
    if len(onsets) == 0 or len(offsets) == 0:
        return np.array([]), np.array([])
    else:
        diff = onsets[1:] - offsets[:-1]
        indices = diff > min_distance
        merged_onsets = onsets[np.hstack([True, indices])]
        merged_offsets = offsets[np.hstack([indices, True])]
        return merged_onsets, merged_offsets

    
def remove_events(onsets, offsets, min_duration, max_duration=None):
    """Remove events that are too short or too long.

    If the length of an event, i.e. `offset` (offset, trough, or negative
    threshold crossing) minus `onset` (onset, peak, or positive threshold crossing),
    is shorter than `min_duration` or longer than `max_duration`, then this event is
    removed.
    
    Parameters
    ----------
    onsets: 1-D array
        The onsets (peaks, or positive threshold crossings) of the events
        as indices or times.
    offsets: 1-D array
        The offsets (troughs, or negative threshold crossings) of the events
        as indices or times.
    min_duration: int, float, or None
        The minimum duration of events. If the event offset minus the event onset
        is less than `min_duration`, then the event is removed from the lists.
        If the event onsets and offsets are given in indices than
        `min_duration` is also in indices. If `None` then this test is skipped.
    max_duration: int, float, or None
        The maximum duration of events. If the event offset minus the event onset
        is larger than `max_duration`, then the event is removed from the lists.
        If the event onsets and offsets are given in indices than
        `max_duration` is also in indices. If `None` then this test is skipped.

    Returns
    -------
    onsets: 1-D array
        The onsets (peaks, or positive threshold crossings) of the events
        with too short and too long events removed as indices or times according to onsets.
    offsets: 1-D array
        The offsets (troughs, or negative threshold crossings) of the events
        with too short and too long events removed as indices or times according to offsets.
    """
    onsets, offsets = trim_to_peak(onsets, offsets)
    if len(onsets) == 0 or len(offsets) == 0:
        return np.array([]), np.array([])
    elif min_duration is not None or max_duration is not None:
        diff = offsets - onsets
        if min_duration is not None and max_duration is not None:
            indices = (diff > min_duration) & (diff < max_duration)
        elif min_duration is not None:
            indices = diff > min_duration
        else:
            indices = diff < max_duration
        onsets = onsets[indices]
        offsets = offsets[indices]
    return onsets, offsets


def widen_events(onsets, offsets, max_time, duration):
    """Enlarge events on both sides without overlap.

    Subtracts `duration` from the `onsets` and adds `duration` to the offsets.
    If two succeeding events are separated by less than two times the `duration`,
    then the offset of the previous event and the onset of the following event are
    set at the center between the two events.
    
    Parameters
    ----------
    onsets: 1-D array
        The onsets (peaks, or positive threshold crossings) of the events
        as indices or times.
    offsets: 1-D array
        The offsets (troughs, or negative threshold crossings) of the events
        as indices or times.
    max_time: int or float
        The maximum value for the end of the last event.
        If the event onsets and offsets are given in indices than
        max_time is the maximum possible index, i.e. the len of the
        data array on which the events where detected.
    duration: int or float
        The number of indices or the time by which the events should be enlarged.
        If the event onsets and offsets are given in indices than
        duration is also in indices. 

    Returns
    -------
    onsets: 1-D array
        The onsets (peaks, or positive threshold crossings) of the enlarged events.
    offsets: 1-D array
        The offsets (troughs, or negative threshold crossings) of the enlarged events.
    """
    new_onsets = []
    new_offsets = []
    if len(onsets) > 0:
        on_idx = onsets[0]
        new_onsets.append( on_idx - duration if on_idx >= duration else 0 )
    for off_idx, on_idx in zip(offsets[:-1], onsets[1:]):
        if on_idx - off_idx < 2*duration:
            mid_idx = (on_idx + off_idx)//2
            new_offsets.append(mid_idx)
            new_onsets.append(mid_idx)
        else:
            new_offsets.append(off_idx + duration)
            new_onsets.append(on_idx - duration)
    if len(offsets) > 0:
        off_idx = offsets[-1]
        new_offsets.append(off_idx + duration if off_idx + duration < max_time else max_time)
    return new_onsets, new_offsets

    
def std_threshold(data, samplerate=None, win_size=None, th_factor=5.):
    """Esimates a threshold for detect_peaks() based on the standard deviation of the data.

    The threshold is computed as the standard deviation of the data multiplied with th_factor.

    If samplerate and win_size is given, then the threshold is computed for
    each non-overlapping window of duration win_size separately.
    In this case the returned threshold is an array of the same size as data.
    Without a samplerate and win_size a single threshold value determined from
    the whole data array is returned.

    Parameters
    ----------
    data: 1-D array
        The data to be analyzed.
    samplerate: float or None
        Sampling rate of the data in Hz.
    win_size: float or None
        Size of window in which a threshold value is computed.
    th_factor: float
        Factor by which the standard deviation is multiplied to set the threshold.

    Returns
    -------
    threshold: float or 1-D array
        The computed threshold.
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
                   nbins=100, hist_height=1.0/np.sqrt(np.e)):
    """Esimate a threshold for detect_peaks() based on a histogram of the data.

    The standard deviation of the data is estimated from the
    width of the histogram of the data at hist_height relative height.

    If samplerate and win_size is given, then the threshold is computed for
    each non-overlapping window of duration win_size separately.
    In this case the returned threshold is an array of the same size as data.
    Without a samplerate and win_size a single threshold value determined from
    the whole data array is returned.

    Parameters
    ----------
    data: 1-D array
        The data to be analyzed.
    samplerate: float or None
        Sampling rate of the data in Hz.
    win_size: float or None
        Size of window in which a threshold value is computed in sec.
    th_factor: float
        Factor by which the width of the histogram is multiplied to set the threshold.
    nbins: int or list of floats
        Number of bins or the bins for computing the histogram.
    hist_height: float
        Height between 0 and 1 at which the width of the histogram is computed.

    Returns
    -------
    threshold: float or 1-D array
        The computed threshold.
    center: float or 1-D array
        The center (mean) of the width of the histogram.
    """

    if samplerate and win_size:
        threshold = np.zeros(len(data))
        centers = np.zeros(len(data))
        win_size_indices = int(win_size * samplerate)

        for inx0 in range(0, len(data), win_size_indices):
            inx1 = inx0 + win_size_indices
            std, center = hist_threshold(data[inx0:inx1], samplerate=None, win_size=None,
                                         th_factor=th_factor, nbins=nbins,
                                         hist_height=hist_height)
            threshold[inx0:inx1] = std
            centers[inx0:inx1] = center
        return threshold, centers
    else:
        maxd = np.max(data)
        mind = np.min(data)
        contrast = np.abs((maxd - mind)/(maxd + mind))
        if contrast > 1e-8:
            hist, bins = np.histogram(data, nbins, density=False)
            inx = hist > np.max(hist) * hist_height
            lower = bins[0:-1][inx][0]
            upper = bins[1:][inx][-1]  # needs to return the next bin
            center = 0.5 * (lower + upper)
            std = 0.5 * (upper - lower)
        else:
            std = np.std(data)
            center = np.mean(data)
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

    Parameters
    ----------
    data: 1-D array
        The data to be analyzed.
    samplerate: float or None
        Sampling rate of the data in Hz.
    win_size: float or None
        Size of window in which a threshold value is computed.
    th_factor: float
        Factor by which the difference between minimum and maximum data value
        is multiplied to set the threshold.

    Returns
    -------
    threshold: float or 1-D array
        The computed threshold.
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

    Parameters
    ----------
    data: 1-D array
        The data to be analyzed.
    samplerate: float or None
        Sampling rate of the data in Hz.
    win_size: float or None
        Size of window in which a threshold value is computed.
    percentile: int
        The interpercentile range is computed at percentile and 100.0-percentile.
    th_factor: float
        Factor by which the inter-percentile range of the data is multiplied to set the threshold.

    Returns
    -------
    threshold: float or 1-D array
        The computed threshold.
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

    Parameters
    ----------
    data: 1-D array
        Data array from which snippets are extracted.
    indices: list of int
        Indices around which snippets are cut out.
    start: int
        Each snippet starts at index + start.
    stop: int
        Each snippet ends at index + stop.
        
    Returns
    -------
    snippet_data: 2-D array
        The snippets: first index number of snippet, second index time.
    """
    idxs = indices[(indices>=-start) & (indices<len(data)-stop)]
    snippet_data = np.empty((len(idxs), stop-start))
    for k, idx in enumerate(idxs):
        snippet_data[k] = data[idx+start:idx+stop]
    return snippet_data


def detect_dynamic_peaks(data, threshold, min_thresh, tau, time=None,
                         check_peak_func=None, check_trough_func=None, **kwargs):
    """
    Detect peaks and troughs using a relative threshold according to
    Bryan S. Todd and David C. Andrews (1999): The identification of peaks in physiological signals.
    Computers and Biomedical Research 32, 322-335.
    The threshold decays dynamically towards min_thresh with time constant tau.
    Use `check_peak_func` or `check_trough_func` to reset the threshold to an appropriate size.

    Parameters
    ----------
    data: array
        An 1-D array of input data where peaks are detected.
    threshold: float
        A positive number setting the minimum distance between peaks and troughs.
    min_thresh: float
        The minimum value the threshold is allowed to assume.
    tau: float
        The time constant of the the decay of the threshold value
        given in indices (`time` is None) or time units (`time` is not `None`).
    time: array
        The (optional) 1-D array with the time corresponding to the data values.
    check_peak_func: function
        An optional function to be used for further evaluating and analysing a peak.
        The signature of the function is
        ```
        r, th = check_peak_func(time, data, peak_inx, index, min_inx, threshold, **kwargs)
        ```
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
    check_trough_func: function
        An optional function to be used for further evaluating and analysing a trough.
        The signature of the function is
        ```
        r, th = check_trough_func(time, data, trough_inx, index, max_inx, threshold, **kwargs)
        ```
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
    kwargs: key-word arguments
        Arguments passed on to `check_peak_func` and `check_trough_func`.
    
    Returns 
    -------
    peak_list: np.array
        A list of peaks.
    trough_list: np.array
        A list of troughs.
    If time is `None` and no `check_peak_func` is given,
    then these are lists of the indices where the peaks/troughs occur.
    If `time` is given and no `check_peak_func`/`check_trough_func` is given,
    then these are lists of the times where the peaks/troughs occur.
    If `check_peak_func` or `check_trough_func` is given,
    then these are lists of whatever `check_peak_func`/`check_trough_func` return.

    Raises
    ------
    ValueError: If `threshold <= 0` or `min_thresh <= 0` or `tau <= 0`.
    IndexError: If `data` and `time` arrays differ in length.
    """

    if threshold <= 0:
        raise ValueError('input argument threshold must be positive!')
    if min_thresh <= 0:
        raise ValueError('input argument min_thresh must be positive!')
    if tau <= 0:
        raise ValueError('input argument tau must be positive!')
    if time is not None and len(data) != len(time):
        raise IndexError('input arrays time and data must have same length!')

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

        # decaying threshold (first order low pass filter):
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


def accept_peak_size_threshold(time, data, event_inx, index, min_inx, threshold,
                               min_thresh, tau, thresh_ampl_fac=0.75, thresh_weight=0.02):
    """Accept each detected peak/trough and return its index or time.
    Adjust the threshold to the size of the detected peak.
    To be passed to the detect_dynamic_peaks() function.

    Parameters
    ----------
    time: array
        Time values, can be `None`.
    data: array
        The data in wich peaks and troughs are detected.
    event_inx: int
        Index of the current peak/trough.
    index: int
        Current index.
    min_inx: int
        Index of the previous trough/peak.
    threshold: float
        Threshold value.
    min_thresh: float
        The minimum value the threshold is allowed to assume..
    tau: float
        The time constant of the the decay of the threshold value
        given in indices (`time` is `None`) or time units (`time` is not `None`).
    thresh_ampl_fac: float
        The new threshold is `thresh_ampl_fac` times the size of the current peak.
    thresh_weight: float
        New threshold is weighted against current threshold with `thresh_weight`.

    Returns 
    -------
    index: int
        Index of the peak/trough if `time` is `None`.
    time: int
        Time of the peak/trough if `time` is not `None`.
    threshold: float
        The new threshold to be used.
    """
    size = data[event_inx] - data[min_inx]
    threshold += thresh_weight * (thresh_ampl_fac * size - threshold)
    if time is None:
        return event_inx, threshold
    else:
        return time[event_inx], threshold


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    print("Checking eventetection module ...")
    print('')
    # generate data:
    dt = 0.001
    time = np.arange(0.0, 10.0, dt)
    f = 2.0
    data = (0.5 * np.sin(2.0 * np.pi * f * time) + 0.5) ** 4.0
    data += -0.1 * time * (time - 10.0)
    data += 0.1 * np.random.randn(len(data))

    print("generated waveform with %d peaks" % int(np.round(time[-1] * f)))
    plt.plot(time, data)

    print('')
    print('check detect_peaks(data, 1.0)...')
    peaks, troughs = detect_peaks(data, 1.0)
    # print peaks:
    print('detected %d peaks with period %g that differs from the real frequency by %g' % (
        len(peaks), np.mean(np.diff(peaks)), f - 1.0 / np.mean(np.diff(peaks)) / np.mean(np.diff(time))))
    # print troughs:
    print('detected %d troughs with period %g that differs from the real frequency by %g' % (
        len(troughs), np.mean(np.diff(troughs)), f - 1.0 / np.mean(np.diff(troughs)) / np.mean(np.diff(time))))

    # plot peaks and troughs:
    plt.plot(time[peaks], data[peaks], '.r', ms=20)
    plt.plot(time[troughs], data[troughs], '.g', ms=20)

    # detect threshold crossings:
    onsets, offsets = threshold_crossings(data, 3.0)
    onsets, offsets = merge_events(onsets, offsets, int(0.5/f/dt))
    plt.plot(time, 3.0*np.ones(len(time)), 'k')
    plt.plot(time[onsets], data[onsets], '.c', ms=20)
    plt.plot(time[offsets], data[offsets], '.b', ms=20)

    plt.ylim(-0.5, 4.0)
    plt.show()
