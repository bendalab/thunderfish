import sys
import numpy as np


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
          r, th = check_peak_func(time, data, peak_inx, index, min_inx, threshold, check_conditions)
          with
            time (array): the full time array that might be None
            data (array): the full data array
            peak_inx (int): the index of the  detected peak
            index (int): the current index
            min_inx (int): the index of the trough preceeding the peak (might be 0)
            threshold (float): the threshold value
            check_conditions (dict): dictionary with further user supplied parameter
            r (scalar or np.array): a single number or an array with properties of the peak or None to skip the peak
            th (float): a new value for the threshold or None (to keep the original value)
        check_trough_func (function): an optional function to be used for further evaluating and analysing a trough
          The signature of the function is
          r, th = check_trough_func(time, data, trough_inx, index, max_inx, threshold, check_conditions)
          with
            time (array): the full time array that might be None
            data (array): the full data array
            trough_inx (int): the index of the  detected trough
            index (int): the current index
            max_inx (int): the index of the peak preceeding the trough (might be 0)
            threshold (float): the threshold value
            check_conditions (dict): dictionary with further user supplied parameter
            r (scalar or np.array): a single number or an array with properties of the trough or None to skip the trough
            th (float): a new value for the threshold or None (to keep the original value)            
        check_conditions (dict): an optional dictionary for supplying further parameter to check_peak_func and check_trough_func
    
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

    # loop through the data:
    for index, value in enumerate(data):

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
                if check_peak_func :
                    r, th = check_peak_func(time, data, max_inx, index,
                                            min_inx, threshold, check_conditions)
                    if r is not None :
                        # this really is a peak:
                        peaks_list.append(r)
                    if th is not None :
                        threshold = th
                else:
                    # this is a peak:
                    if time is None :
                        peaks_list.append(max_inx)
                    else :
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
                if check_trough_func :
                    r, th = check_trough_func(time, data, min_inx, index,
                                              max_inx, threshold, check_conditions)
                    if r is not None :
                        # this really is a trough:
                        troughs_list.append(r)
                    if th is not None :
                        threshold = th
                else:
                    # this is a trough:
                    if time is None :
                        troughs_list.append(min_inx)
                    else :
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

    return np.array(peaks_list), np.array(troughs_list)


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
          r = check_peak_func(time, data, peak_inx, index, min_inx, threshold, check_conditions)
          with
            time (array): the full time array that might be None
            data (array): the full data array
            peak_inx (int): the index of the  detected peak
            index (int): the current index
            min_inx (int): the index of the trough preceeding the peak (might be 0)
            threshold (float): the threshold value
            check_conditions (dict): dictionary with further user supplied parameter
            r (scalar or np.array): a single number or an array with properties of the peak or None to skip the peak
            th (float): a new value for the threshold or None (to keep the original value)
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

    # loop through the data:
    for index, value in enumerate(data):

        # rising?
        if dir > 0:
            # if the new value is bigger than the old maximum: set it as new maximum
            if value > max_value:
                max_inx = index  # maximum element
                max_value = value

            # otherwise, if the maximum value is bigger than the new value plus the threshold:
            # this is a local maximum!
            elif max_value >= value + threshold:
                # check and update peak with check_peak_func function:
                if check_peak_func :
                    r, th = check_peak_func(time, data, max_inx, index,
                                            min_inx, threshold, check_conditions)
                    if r is not None :
                        # this really is a peak:
                        peak_list.append( r )
                    if th is not None :
                        threshold = th
                else:
                    # this is a peak:
                    if time is None :
                        peak_list.append(max_inx)
                    else :
                        peak_list.append(time[max_inx])

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

    return np.array(peak_list)


def accept_peak(time, data, event_inx, index, min_inx, threshold, check_conditions) :
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
        index (int): index of the peak/trough
        time (float): time of the peak/trough if time is not None
        value (float): value of data at the peak
    """
    size = data[event_inx]
    if time is None :
        return [event_inx, size], None
    else :
        return [event_inx, time[event_inx], size], None


def accept_psd_peaks(freqs, data, peak_inx, index, min_inx, threshold, check_conditions) :
    """
    Accept each detected peak and compute its size and width.

    Args:
        freqs (array): frequencies of the power spectrum
        data (array): the power spectrum
        peak_inx: index of the current peak
        index: current index
        min_inx: index of the previous trough
        threshold: threshold value
        check_conditions: not used
    
    Returns: 
        freq (float): frequency of the peak
        power (float): power of the peak (value of data at the peak)
        size (float): size of the peak (peak minus previous trough)
        width (float): width of the peak at 0.75*size
        count (float): zero
    """
    
    pfac = 0.75
    if check_conditions is not None :
        pfac = check_conditions.get('peakfac', 0.75)

    size = data[peak_inx] - data[min_inx]
    wthresh = data[min_inx] + pfac*size
    width = 0.0
    for k in xrange( peak_inx, min_inx, -1 ) :
        if data[k] < wthresh :
            width = freqs[peak_inx] - freqs[k]
            break
    for k in xrange( peak_inx, index ) :
        if data[k] < wthresh :
            width += freqs[k] - freqs[peak_inx]
            break
    return [ freqs[peak_inx], data[peak_inx], size, width, 0.0 ], None


if __name__ == "__main__":
    print("Checking peakdetection module ...")
    import matplotlib.pyplot as plt

    print
    # generate data:
    time = np.arange(0.0, 10.0, 0.01)
    f = 2.0
    ampl = (0.5*np.sin(2.0*np.pi*f*time)+0.5)**4.0
    ampl += -0.1*time*(time-10.0)
    ampl += 0.1*np.random.randn(len(ampl))
    print("generated waveform with %d peaks" % int(np.round(time[-1]*f)))
    plt.plot(time, ampl)
    
    print
    print('check detect_peaks_troughs(ampl, 0.5, time)...')
    peaks, troughs = detect_peaks_troughs(ampl, 0.5, time)
    #print peaks
    print('detected %d peaks with period %g that differs from the real frequency by %g' % (len(peaks), np.mean(np.diff(peaks)), f-1.0/np.mean(np.diff(peaks))))
    #print troughs
    print('detected %d troughs with period %g that differs from the real frequency by %g' % (len(troughs), np.mean(np.diff(troughs)), f-1.0/np.mean(np.diff(troughs))))
    
    print
    print('check detect_peaks_troughs(ampl, 0.5)...')
    peaks, troughs = detect_peaks_troughs(ampl, 0.5)
    #print peaks
    print('detected %d peaks with period %g that differs from the real frequency by %g' % (len(peaks), np.mean(np.diff(peaks)), f-1.0/np.mean(np.diff(peaks))/np.mean(np.diff(time))))
    #print troughs
    print('detected %d troughs with period %g that differs from the real frequency by %g' % (len(troughs), np.mean(np.diff(troughs)), f-1.0/np.mean(np.diff(troughs))/np.mean(np.diff(time))))
        
    print
    print('check detect_peaks_troughs(ampl, 0.5, time, accept_peak, accept_peak)...')
    peaks, troughs = detect_peaks_troughs(ampl, 0.5, time, accept_peak, accept_peak)
    #print peaks
    print('detected %d peaks with period %g that differs from the real frequency by %g' % (len(peaks), np.mean(np.diff(peaks[:,1])), f-1.0/np.mean(np.diff(peaks[:,1]))))
    #print troughs
    print('detected %d troughs with period %g that differs from the real frequency by %g' % (len(troughs), np.mean(np.diff(troughs[:,1])), f-1.0/np.mean(np.diff(troughs[:,1]))))
    plt.plot(peaks[:,1], peaks[:,2], '.r', ms=20)
    plt.plot(troughs[:,1], troughs[:,2], '.g', ms=20)
    
    print
    print('check detect_peaks(ampl, 0.5, time)...')
    peaks = detect_peaks(ampl, 0.5, time)
    #print peaks
    print('detected %d peaks with period %g that differs from the real frequency by %g' % (len(peaks), np.mean(np.diff(peaks)), f-1.0/np.mean(np.diff(peaks))))
    
    print
    print('check detect_peaks(ampl, 0.5)...')
    peaks = detect_peaks(ampl, 0.5)
    #print peaks
    print('detected %d peaks with period %g that differs from the real frequency by %g' % (len(peaks), np.mean(np.diff(peaks)), f-1.0/np.mean(np.diff(peaks))/np.mean(np.diff(time))))
        
    print
    print('check detect_peaks(ampl, 0.5, time, accept_peak)...')
    peaks = detect_peaks(ampl, 0.5, time, accept_peak)
    #print peaks
    print('detected %d peaks with period %g that differs from the real frequency by %g' % (len(peaks), np.mean(np.diff(peaks[:,1])), f-1.0/np.mean(np.diff(peaks[:,1]))))
    #plt.plot(peaks[:,0], peaks[:,1], '.r', ms=20)

    plt.ylim(-0.5,4.0)
    plt.show()
    
