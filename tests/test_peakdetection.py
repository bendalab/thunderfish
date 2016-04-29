import numpy as np
from nose.tools import assert_true

import thunderfish.peakdetection as pd


def test_peakdetection():
    """Tests whether peakdetection works corretly"""

    # generate data:
    time = np.arange(0.0, 10.0, 0.01)
    data = np.zeros(time.shape)
    pt_indices = np.random.randint(5,len(data)-10, size=40)
    pt_indices.sort()
    while np.any(np.diff(pt_indices).min() < 5):
        pt_indices = np.random.randint(0,len(data), size=40)
        pt_indices.sort()
    peak_indices = pt_indices[0::2]
    trough_indices = pt_indices[1::2]
    n = pt_indices[0]
    data[0:n] = 0.1+0.9*np.arange(0.0, n)/n
    up = False
    for i in xrange(0,len(pt_indices)-1) :
        n = pt_indices[i+1]-pt_indices[i]
        if up :
            data[pt_indices[i]:pt_indices[i+1]] = np.arange(0.0, n)/n
        else :
            data[pt_indices[i]:pt_indices[i+1]] = 1.0 - np.arange(0.0, n)/n
        up = not up
    n = len(data)-pt_indices[-1]
    if up :
        data[pt_indices[-1]:] = 0.8*np.arange(0.0, n)/n
    else :
        data[pt_indices[-1]:] = 1.0 - 0.8*np.arange(0.0, n)/n
    up = not up
    data += -0.025*time*(time-10.0)
    peak_times = time[peak_indices]
    trough_times = time[trough_indices]
    threshold = 0.5
    
    peaks, troughs = pd.detect_peaks_troughs(data, threshold)
    assert_true(np.all(peaks == peak_indices),
                "detect_peaks_troughs(data, threshold) did not correctly detect peaks")
    assert_true(np.all(troughs == trough_indices),
                "detect_peaks_troughs(data, threshold) did not correctly detect troughs")
    
    peaks, troughs = pd.detect_peaks_troughs(data, threshold, time)
    assert_true(np.all(peaks == peak_times),
                "detect_peaks_troughs(data, threshold, time) did not correctly detect peaks")
    assert_true(np.all(troughs == trough_times),
                "detect_peaks_troughs(data, threshold, time) did not correctly detect troughs")
        
    peaks, troughs = pd.detect_peaks_troughs(data, threshold, time,
                                             pd.accept_peak, pd.accept_peak)
    assert_true(np.all(peaks[:,0] == peak_indices),
                "detect_peaks_troughs(data, threshold, time, accept_peak, accept_peak) did not correctly detect peaks")
    assert_true(np.all(troughs[:,0] == trough_indices),
                "detect_peaks_troughs(data, threshold, time, accept_peak, accept_peak) did not correctly detect troughs")
    assert_true(np.all(peaks[:,1] == peak_times),
                "detect_peaks_troughs(data, threshold, time, accept_peak, accept_peak) did not correctly detect peaks")
    assert_true(np.all(troughs[:,1] == trough_times),
                "detect_peaks_troughs(data, threshold, time, accept_peak, accept_peak) did not correctly detect troughs")
    
    peaks = pd.detect_peaks(data, threshold)
    assert_true(np.all(peaks == peak_indices),
                "detect_peaks(data, threshold) did not correctly detect peaks")
    
    peaks = pd.detect_peaks(data, threshold, time)
    assert_true(np.all(peaks == peak_times),
                "detect_peaks(data, threshold, time) did not correctly detect peaks")
        
    peaks = pd.detect_peaks(data, threshold, time, pd.accept_peak)
    assert_true(np.all(peaks[:,0] == peak_indices),
                "detect_peaks(data, threshold, time, accept_peak) did not correctly detect peaks")
    assert_true(np.all(peaks[:,1] == peak_times),
                "detect_peaks(data, threshold, time, accept_peak) did not correctly detect peaks")
    
