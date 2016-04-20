import sys
import numpy as np

def detect_peaks(data, threshold, time=None, check_peak_func=None, check_conditions=None):
    """
    Detect peaks using a relative threshold according to
    Bryan S. Todd and David C. Andrews (1999): The identification of peaks in physiological signals.
    Computers and Biomedical Research 32, 322-335.

    Args:
        data (array): an 1-D array of input data where peaks are detected
        threshold (float): a positive number setting the minimum distance between peaks and troughs
        time (array): the (optional) 1-D array with the time corresponding to the data values
        check_peak_func (function): an optional function to be used for further evaluating and analyzing a peak.
          The signature of the function is
          r = check_peak_func(time, data, peak_inx, index, trough_inx, min_inx, threshold, check_conditions)
          with
            time (array): the full time array that might be None
            data (array): the full data array
            peak_inx (int): the index of the  detected peak
            index (int): the current index (a trough)
            trough_inx (int): the index of the trough preceeding the peak (might be 0)
            min_inx (int): the index of the current local minimum
            threshold (float): the threshold value
            check_conditions (dict): dictionary with further user supplied parameter
            r (scalar or np.array): a single number or an array with properties of the peak
        check_conditions (dict): an optional dictionary for supplying further parameter to check_peak_func
    
    Returns: 
        peak_list (np.array): a list of peaks
          if time is None and no check_peak_func is given, then this is a list of the indices where the peaks occur.
          if time is given and no check_peak_func is given, then this is a list of the times where the peaks occur.
          if check_peak_func is given, then this is a list of whatever check_peak_func returns.
    """

    if not np.isscalar(threshold):
        sys.exit('detect_peaks(): input argument threshold must be a scalar!')

    if threshold <= 0:
        sys.exit('detect_peaks(): input argument threshold must be positive!')

    if time is not None and len(data) != len(time):
        sys.exit('detect_peaks(): input arrays time and data must have same length!')
    
    if not check_conditions:
        check_conditions = dict()
        
    peak_list = list()

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
                peak_inx = max_inx

                # check and update peak with check_peak_func function:
                if check_peak_func :
                    r = check_peak_func(time, data, peak_inx, index,
                                        trough_inx, min_inx, threshold,
                                        check_conditions)
                    if len( r ) > 0 :
                        # this really is an peak:
                        peak_list.append( r )
                else:
                    # this is an peak:
                    if time is None :
                        peak_list.append(peak_inx)
                    else :
                        peak_list.append(time[peak_inx])

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
                trough_inx = index

    return np.array(peak_list)


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
                         check_peak_func=None, check_trough_func=None, check_conditions=None):
    """
    Detect peaks and troughs using a relative threshold according to
    Bryan S. Todd and David C. Andrews (1999): The identification of peaks in physiological signals.
    Computers and Biomedical Research 32, 322-335.

    Args:
        data (array): an 1-D array of input data where peaks are detected
        threshold (float): a positive number setting the minimum distance between peaks and troughs
        time (array): the (optional) 1-D array with the time corresponding to the data values
        check_peak_func (function): an optional function to be used for further evaluating and analysing a peak
          The signature of the function is
          r = check_peak_func(time, data, peak_inx, index, trough_inx, min_inx, threshold, check_conditions)
          with
            time (array): the full time array that might be None
            data (array): the full data array
            peak_inx (int): the index of the  detected peak
            index (int): the current index (a trough)
            trough_inx (int): the index of the trough preceeding the peak (might be 0)
            min_inx (int): the index of the current local minimum
            threshold (float): the threshold value
            check_conditions (dict): dictionary with further user supplied parameter
            r (scalar or np.array): a single number or an array with properties of the peak
        check_trough_func (function): an optional function to be used for further evaluating and analysing a trough
          The signature of the function is
          r = check_trough_func(time, data, trough_inx, index, peak_inx, max_inx, threshold, check_conditions)
          with
            time (array): the full time array that might be None
            data (array): the full data array
            trough_inx (int): the index of the  detected trough
            index (int): the current index (a peak)
            peak_inx (int): the index of the peak preceeding the trough (might be 0)
            max_inx (int): the index of the current local maximum
            threshold (float): the threshold value
            check_conditions (dict): dictionary with further user supplied parameter
            r (scalar or np.array): a single number or an array with properties of the trough
        check_conditions (dict): an optional dictionary for supplying further parameter to check_peak_func and  check_trough_func
    
    Returns: 
        peak_list (np.array): a list of peaks
        trough_list (np.array): a list of troughs
          if time is None and no check_peak_func is given, then these are lists of the indices where the peaks/troughs occur.
          if time is given and no check_peak_func/check_trough_func is given, then these are lists of the times where the peaks/troughs occur.
          if check_peak_func or check_trough_func is given, then these are lists of whatever check_peak_func/check_trough_func return.
    """

    if not np.isscalar(threshold):
        sys.exit('detect_peaks(): input argument threshold must be a scalar!')

    if threshold <= 0:
        sys.exit('detect_peaks(): input argument threshold must be positive!')

    if time is not None and len(data) != len(time):
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

                # check and update peak with the check_peak_func function:
                if check_peak_func :
                    r = check_peak_func(time, data, peak_inx, index,
                                        trough_inx, min_inx, threshold,
                                        check_conditions)
                    if len( r ) > 0 :
                        # this really is an peak:
                        peaks_list.append(r)
                else:
                    # this is an peak:
                    if time is None :
                        peaks_list.append(peak_inx)
                    else :
                        peaks_list.append(time[peak_inx])

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

                # check and update trough with the check_trough function:
                if check_trough_func :
                    r = check_trough_func(time, data, trough_inx,
                                          index, peak_inx, max_inx, threshold,
                                          check_conditions)
                    if len( r ) > 0 :
                        # this really is an trough:
                        troughs_list.append(r)
                else:
                    # this is an trough:
                    if time is None :
                        troughs_list.append(trough_inx)
                    else :
                        troughs_list.append(time[trough_inx])

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


if __name__ == "__main__":
    print("Checking peakdetection module ...")
    import matplotlib.pyplot as plt
    # generate data:
    time = np.arange(0.0, 10.0, 0.01)
    f = 2.0
    ampl = (0.5*np.sin(2.0*np.pi*f*time)+0.5)**4.0
    #ampl += 0.2*np.random.randn(len(ampl))
    print("generated waveform with %d peaks" % int(np.round(time[-1]*f)))
    #plt.plot(time, ampl)
    #plt.show()

    print
    print('check peakdet(ampl, 0.5, time)...')
    maxtab, maxidx, mintab, minidx = peakdet(ampl, 0.5, time)
    print maxtab
    print len(maxtab)
    print maxidx
    print np.diff(maxidx)
    print np.mean(np.diff(maxidx))
    print f-1.0/np.mean(np.diff(maxidx))
    print mintab
    print len(mintab)
    print minidx
    print np.diff(minidx)
    print np.mean(np.diff(minidx))
    print f-1.0/np.mean(np.diff(minidx))
    
    ## print
    ## print('check peakdet(ampl, 0.5)...')
    ## maxtab, maxidx, mintab, minidx = peakdet(ampl, 0.5)
    ## print maxidx
    ## print np.diff(maxidx)
    ## print np.mean(np.diff(maxidx))
    ## print f-1.0/np.mean(np.diff(maxidx))/np.mean(np.diff(time))
    ## print minidx
    ## print np.diff(minidx)
    ## print np.mean(np.diff(minidx))
    ## print f-1.0/np.mean(np.diff(minidx))/np.mean(np.diff(time))
    
    print
    print('check detect_peaks_troughs(ampl, 0.5, time)...')
    peaks, troughs = detect_peaks_troughs(ampl, 0.5, time)
    print peaks
    print len(peaks)
    print np.diff(peaks)
    print np.mean(np.diff(peaks))
    print f-1.0/np.mean(np.diff(peaks))
    print troughs
    print len(troughs)
    print np.diff(troughs)
    print np.mean(np.diff(troughs))
    print f-1.0/np.mean(np.diff(troughs))
    
    print
    print('check detect_peaks_troughs(ampl, 0.5)...')
    peaks, troughs = detect_peaks_troughs(ampl, 0.5)
    print peaks
    print len(peaks)
    print np.diff(peaks)
    print np.mean(np.diff(peaks))
    print f-1.0/np.mean(np.diff(peaks))/np.mean(np.diff(time))
    print troughs
    print len(troughs)
    print np.diff(troughs)
    print np.mean(np.diff(troughs))
    print f-1.0/np.mean(np.diff(troughs))/np.mean(np.diff(time))
    
    
