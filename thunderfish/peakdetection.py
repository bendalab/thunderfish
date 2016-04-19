import sys
import numpy as np

def detect_peaks(data, threshold, time=None, check_func=None, check_conditions=None):
    """
    Tod & Andrews 1999 peak detection algorithm
    """

    if not np.isscalar(threshold):
        sys.exit('detect_peaks(): input argument threshold must be a scalar!')

    if threshold <= 0:
        sys.exit('detect_peaks(): input argument threshold must be positive!')

    if time and len(data) != len(time):
        sys.exit('detect_peaks(): input arrays time and data must have same length!')
    
    if not check_conditions:
        check_conditions = dict()
        
    event_list = list()

    # initialize:
    dir = 0
    min_inx = 0
    max_inx = 0
    min_value = data[0]
    max_value = min_value
    trough_inx = 0

    # loop through the new read data
    for index, value in enumerate(data):

        # rising?
        if dir > 0:
            # if the new value is bigger than the old maximum: set it as new maximum
            if max_value < value:
                max_inx = index  # maximum element
                max_value = value

            # otherwise, if the maximum value is bigger than the new value plus the threshold:
            # this is a local maximum!
            elif max_value >= value + threshold:
                # there was a peak:
                event_inx = max_inx

                # check and update event with this magic function
                if check_func:
                    r = check_func(time, data, event_inx, index, trough_inx, min_inx, threshold, check_conditions)
                    if len( r ) > 0 :
                        # this really is an event:
                        event_list.append( r )
                else:
                    # this really is an event:
                    if time :
                        event_list.append(time[event_inx])
                    else :
                        event_list.append(event_inx)

                # change direction:
                min_inx = index  # minimum element
                min_value = value
                dir = -1

        # falling?
        elif dir < 0:
            if value < min_value:
                min_inx = index  # minimum element
                min_value = value
                trough_inx = index

            elif value >= min_value + threshold:
                # there was a trough:
                # change direction:
                max_inx = index  # maximum element
                max_value = value
                dir = 1

        # don't know!
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
                trough_inx = index

    return np.array( event_list )


def accept_psd_peaks( freqs, data, peak_inx, index, trough_inx, min_inx, threshold, check_conditions ) :
    """
    Accept each detected peak and compute its size and width.

    Args:
        freqs (array): frequencies of the power spectrum
        data (array): the power spectrum
        peak_inx: index of the current peak
        index: current index (first minimum after peak at threshold below)
        trough_inx: index of the previous trough
        min_inx: index of previous minimum
        threshold: threshold value
        check_conditions: not used
    
    Returns: 
        freq (float): frequency of the peak
        power (float): power of the peak (value of data at the peak)
        size (float): size of the peak (peak minus previous trough)
        width (float): width of the peak at 0.75*size
        count (float): zero
    """
    size = data[peak_inx] - data[trough_inx]
    wthresh = data[trough_inx] + 0.75*size
    width = 0.0
    for k in xrange( peak_inx, trough_inx, -1 ) :
        if data[k] < wthresh :
            width = freqs[peak_inx] - freqs[k]
            break
    for k in xrange( peak_inx, index ) :
        if data[k] < wthresh :
            width += freqs[k] - freqs[peak_inx]
            break
    return [ freqs[peak_inx], data[peak_inx], size, width, 0.0 ]


def detect_peaks_troughs(data, threshold, time=None,
                         check_peaks=None, check_troughs=None, check_conditions=None):
    """
    Tod & Andrews 1999 peak detection algorithm
    """

    if not np.isscalar(threshold):
        sys.exit('detect_peaks(): input argument threshold must be a scalar!')

    if threshold <= 0:
        sys.exit('detect_peaks(): input argument threshold must be positive!')

    if time and len(data) != len(time):
        sys.exit('detect_peaks(): input arrays time and data must have same length!')
    
    if not check_conditions:
        check_conditions = dict()
        
    peaks_list = list()
    troughs_list = list()

    # initialize:
    dir = 0
    min_inx = 0
    max_inx = 0
    min_value = data[0]
    max_value = min_value
    peak_inx = 0
    trough_inx = 0

    # loop through the new read data
    for index, value in enumerate(data):

        # rising?
        if dir > 0:
            # if the new value is bigger than the old maximum: set it as new maximum
            if max_value < value:
                max_inx = index  # maximum element
                max_value = value
                peak_inx = index

            # otherwise, if the maximum value is bigger than the new value plus the threshold:
            # this is a local maximum!
            elif max_value >= value + threshold:
                # there was a peak:

                # check and update event with this magic function
                if check_peak:
                    r = check_peak(time, data, peak_inx, index, trough_inx, min_inx, threshold, check_conditions)
                    if len( r ) > 0 :
                        # this really is an event:
                        peaks_list.append( r )
                else:
                    # this really is an event:
                    if time :
                        peaks_list.append(time[event_inx])
                    else :
                        peaks_list.append(event_inx)

                # change direction:
                min_inx = index  # minimum element
                min_value = value
                dir = -1

        # falling?
        elif dir < 0:
            if value < min_value:
                min_inx = index  # minimum element
                min_value = value
                trough_inx = index

            elif value >= min_value + threshold:
                # there was a trough:

                # check and update event with this magic function
                if check_trough:
                    r = check_trough(time, data, trough_inx, index, peak_inx, min_inx, threshold, check_conditions)
                    if len( r ) > 0 :
                        # this really is an event:
                        troughs_list.append( r )
                else:
                    # this really is an event:
                    if time :
                        troughs_list.append(time[event_inx])
                    else :
                        troughs_list.append(event_inx)

                # change direction:
                max_inx = index  # maximum element
                max_value = value
                dir = 1

        # don't know!
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
                trough_inx = index

    return np.array(peaks_list), np.array(troughs_list)


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

