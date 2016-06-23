from nose.tools import assert_true
import numpy as np
import thunderfish.peakdetection as pd


def test_detect_peaks():
    # generate data:
    time = np.arange(0.0, 10.0, 0.01)
    data = np.zeros(time.shape)
    pt_indices = np.random.randint(5, len(data) - 10, size=40)
    pt_indices.sort()
    while np.any(np.diff(pt_indices).min() < 5):
        pt_indices = np.random.randint(5, len(data) - 10, size=40)
        pt_indices.sort()
    peak_indices = pt_indices[0::2]
    trough_indices = pt_indices[1::2]
    n = pt_indices[0]
    data[0:n] = 0.1 + 0.9 * np.arange(0.0, n) / n
    up = False
    for i in xrange(0, len(pt_indices) - 1):
        n = pt_indices[i + 1] - pt_indices[i]
        if up:
            data[pt_indices[i]:pt_indices[i + 1]] = np.arange(0.0, n) / n
        else:
            data[pt_indices[i]:pt_indices[i + 1]] = 1.0 - np.arange(0.0, n) / n
        up = not up
    n = len(data) - pt_indices[-1]
    if up:
        data[pt_indices[-1]:] = 0.8 * np.arange(0.0, n) / n
    else:
        data[pt_indices[-1]:] = 1.0 - 0.8 * np.arange(0.0, n) / n
    up = not up
    data += -0.025 * time * (time - 10.0)
    peak_times = time[peak_indices]
    trough_times = time[trough_indices]
    threshold = 0.5
    min_thresh = 0.3

    peaks, troughs = pd.detect_peaks(data, 0.0)
    assert_true(np.all(peaks == np.array([])) and np.all(troughs == np.array([])),
                "detect_peaks(data, threshold) did not handle zero threshold")

    peaks, troughs = pd.detect_peaks(data, -1.0)
    assert_true(np.all(peaks == np.array([])) and np.all(troughs == np.array([])),
                "detect_peaks(data, threshold) did not handle negative threshold")

    peaks, troughs = pd.detect_peaks(data, threshold, time[:len(time) / 2])
    assert_true(np.all(peaks == np.array([])) and np.all(troughs == np.array([])),
                "detect_peaks(data, threshold) did not handle wrong time array")

    peaks, troughs = pd.detect_peaks(data, threshold)
    assert_true(np.all(peaks == peak_indices),
                "detect_peaks(data, threshold) did not correctly detect peaks")
    assert_true(np.all(troughs == trough_indices),
                "detect_peaks(data, threshold) did not correctly detect troughs")

    peaks, troughs = pd.detect_peaks(data, threshold, time)
    assert_true(np.all(peaks == peak_times),
                "detect_peaks(data, threshold, time) did not correctly detect peaks")
    assert_true(np.all(troughs == trough_times),
                "detect_peaks(data, threshold, time) did not correctly detect troughs")

    peaks, troughs = pd.detect_peaks(data, threshold, time,
                                     pd.accept_peak, pd.accept_peak)
    assert_true(np.all(peaks[:, 0] == peak_indices),
                "detect_peaks(data, threshold, time, accept_peak, accept_peak) did not correctly detect peaks")
    assert_true(np.all(troughs[:, 0] == trough_indices),
                "detect_peaks(data, threshold, time, accept_peak, accept_peak) did not correctly detect troughs")
    assert_true(np.all(peaks[:, 1] == peak_times),
                "detect_peaks(data, threshold, time, accept_peak, accept_peak) did not correctly detect peaks")
    assert_true(np.all(troughs[:, 1] == trough_times),
                "detect_peaks(data, threshold, time, accept_peak, accept_peak) did not correctly detect troughs")

    peaks, troughs = pd.detect_peaks(data, threshold, time,
                                     pd.accept_peaks_size_width)
    assert_true(np.all(peaks[:, 0] == peak_times),
                "detect_peaks(data, threshold, time, accept_peaks_size_width) did not correctly detect peaks")
    assert_true(np.all(troughs == trough_times),
                "detect_peaks(data, threshold, time, accept_peak, accept_peaks_size_width) did not correctly detect troughs")

    peaks, troughs = pd.detect_peaks(data, threshold, None,
                                     pd.accept_peak, pd.accept_peak)
    assert_true(np.all(peaks[:, 0] == peak_indices),
                "detect_peaks(data, threshold, time, accept_peak, accept_peak) did not correctly detect peaks")
    assert_true(np.all(troughs[:, 0] == trough_indices),
                "detect_peaks(data, threshold, time, accept_peak, accept_peak) did not correctly detect troughs")


def test_detect_dynamic_peaks():
    # generate data:
    time = np.arange(0.0, 10.0, 0.01)
    data = np.zeros(time.shape)
    pt_indices = np.random.randint(5, len(data) - 10, size=40)
    pt_indices.sort()
    while np.any(np.diff(pt_indices).min() < 5):
        pt_indices = np.random.randint(5, len(data) - 10, size=40)
        pt_indices.sort()
    peak_indices = pt_indices[0::2]
    trough_indices = pt_indices[1::2]
    n = pt_indices[0]
    data[0:n] = 0.1 + 0.9 * np.arange(0.0, n) / n
    up = False
    for i in xrange(0, len(pt_indices) - 1):
        n = pt_indices[i + 1] - pt_indices[i]
        if up:
            data[pt_indices[i]:pt_indices[i + 1]] = np.arange(0.0, n) / n
        else:
            data[pt_indices[i]:pt_indices[i + 1]] = 1.0 - np.arange(0.0, n) / n
        up = not up
    n = len(data) - pt_indices[-1]
    if up:
        data[pt_indices[-1]:] = 0.8 * np.arange(0.0, n) / n
    else:
        data[pt_indices[-1]:] = 1.0 - 0.8 * np.arange(0.0, n) / n
    up = not up
    data += -0.025 * time * (time - 10.0)
    peak_times = time[peak_indices]
    trough_times = time[trough_indices]
    threshold = 0.5
    min_thresh = 0.3

    peaks, troughs = pd.detect_dynamic_peaks(data, 0.0, min_thresh, 0.5, time,
                                             pd.accept_peak_size_threshold)
    assert_true(np.all(peaks == np.array([])) and np.all(troughs == np.array([])),
                "detect_dynamic_peaks(data, threshold) did not handle zero threshold")

    peaks, troughs = pd.detect_dynamic_peaks(data, -1.0, min_thresh, 0.5, time,
                                             pd.accept_peak_size_threshold)
    assert_true(np.all(peaks == np.array([])) and np.all(troughs == np.array([])),
                "detect_dynamic_peaks(data, threshold) did not handle negative threshold")

    peaks, troughs = pd.detect_dynamic_peaks(data, threshold, 0.0, 0.5, time,
                                             pd.accept_peak_size_threshold)
    assert_true(np.all(peaks == np.array([])) and np.all(troughs == np.array([])),
                "detect_dynamic_peaks(data, threshold) did not handle zero min_thresh")

    peaks, troughs = pd.detect_dynamic_peaks(data, threshold, -1.0, 0.5, time,
                                             pd.accept_peak_size_threshold)
    assert_true(np.all(peaks == np.array([])) and np.all(troughs == np.array([])),
                "detect_dynamic_peaks(data, threshold) did not handle negative min_thresh")

    peaks, troughs = pd.detect_dynamic_peaks(data, threshold, min_thresh, 0.0, time,
                                             pd.accept_peak_size_threshold)
    assert_true(np.all(peaks == np.array([])) and np.all(troughs == np.array([])),
                "detect_dynamic_peaks(data, threshold) did not handle zero tau")

    peaks, troughs = pd.detect_dynamic_peaks(data, threshold, min_thresh, -1.0, time,
                                             pd.accept_peak_size_threshold)
    assert_true(np.all(peaks == np.array([])) and np.all(troughs == np.array([])),
                "detect_dynamic_peaks(data, threshold) did not handle negative tau")

    peaks, troughs = pd.detect_dynamic_peaks(data, threshold, min_thresh, 0.5, time[:len(time) / 2],
                                             pd.accept_peak_size_threshold)
    assert_true(np.all(peaks == np.array([])) and np.all(troughs == np.array([])),
                "detect_dynamic_peaks(data, threshold) did not handle wrong time array")

    peaks, troughs = pd.detect_dynamic_peaks(data, threshold, min_thresh, 0.5, time,
                                             pd.accept_peak_size_threshold)
    assert_true(np.all(peaks == peak_times),
                "detect_dynamic_peaks(data, threshold, time, accept_peak_size_threshold) did not correctly detect peaks")
    assert_true(np.all(troughs == trough_times),
                "detect_dynamic_peaks(data, threshold, time, accept_peak_size_threshold) did not correctly detect troughs")

    peaks, troughs = pd.detect_dynamic_peaks(data, threshold, min_thresh, 0.5, None,
                                             pd.accept_peak_size_threshold,
                                             thresh_ampl_fac=0.9, thresh_weight=0.1)
    assert_true(np.all(peaks == peak_indices),
                "detect_dynamic_peaks(data, threshold, time, accept_peak_size_threshold) did not correctly detect peaks")
    assert_true(np.all(troughs == trough_indices),
                "detect_dynamic_peaks(data, threshold, time, accept_peak_size_threshold) did not correctly detect troughs")


def test_trim():
    # generate peak and trough indices (same length, peaks first):
    pt_indices = np.random.randint(5, 1000, size=40)
    pt_indices.sort()
    peak_indices = pt_indices[0::2]
    trough_indices = pt_indices[1::2]

    # peak first, same length:
    p_inx, t_inx = pd.trim(peak_indices, trough_indices)
    assert_true(len(p_inx) == len(peak_indices) and np.all(p_inx == peak_indices),
                "trim(peak_indices, trough_indices) failed on peaks")
    assert_true(len(t_inx) == len(trough_indices) and np.all(t_inx == trough_indices),
                "trim(peak_indices, trough_indices) failed on troughs")

    # trough first, same length:
    p_inx, t_inx = pd.trim(peak_indices[1:], trough_indices[:-1])
    assert_true(len(p_inx) == len(peak_indices[1:]) and np.all(p_inx == peak_indices[1:]),
                "trim(peak_indices[1:], trough_indices[:-1]) failed on peaks")
    assert_true(len(t_inx) == len(trough_indices[:-1]) and np.all(t_inx == trough_indices[:-1]),
                "trim(peak_indices[1:], trough_indices[:-1]) failed on troughs")

    # peak first, more peaks:
    p_inx, t_inx = pd.trim(peak_indices, trough_indices[:-2])
    assert_true(len(p_inx) == len(peak_indices[:-2]) and np.all(p_inx == peak_indices[:-2]),
                "trim(peak_indices, trough_indices[:-2]) failed on peaks")
    assert_true(len(t_inx) == len(trough_indices[:-2]) and np.all(t_inx == trough_indices[:-2]),
                "trim(peak_indices, trough_indices[:-2]) failed on troughs")

    # trough first, more troughs:
    p_inx, t_inx = pd.trim(peak_indices[1:-2], trough_indices)
    assert_true(len(p_inx) == len(peak_indices[1:-2]) and np.all(p_inx == peak_indices[1:-2]),
                "trim(peak_indices[1:-2], trough_indices) failed on peaks")
    assert_true(len(t_inx) == len(trough_indices[:-3]) and np.all(t_inx == trough_indices[:-3]),
                "trim(peak_indices[1:-2], trough_indices) failed on troughs")


def test_trim_to_peak():
    # generate peak and trough indices (same length, peaks first):
    pt_indices = np.random.randint(5, 1000, size=40)
    pt_indices.sort()
    peak_indices = pt_indices[0::2]
    trough_indices = pt_indices[1::2]

    # peak first, same length:
    p_inx, t_inx = pd.trim_to_peak(peak_indices, trough_indices)
    assert_true(len(p_inx) == len(peak_indices) and np.all(p_inx == peak_indices),
                "trim_to_peak(peak_indices, trough_indices) failed on peaks")
    assert_true(len(t_inx) == len(trough_indices) and np.all(t_inx == trough_indices),
                "trim_to_peak(peak_indices, trough_indices) failed on troughs")

    # trough first, same length:
    p_inx, t_inx = pd.trim_to_peak(peak_indices[1:], trough_indices[:-1])
    assert_true(len(p_inx) == len(peak_indices[1:-1]) and np.all(p_inx == peak_indices[1:-1]),
                "trim_to_peak(peak_indices[1:], trough_indices[:-1]) failed on peaks")
    assert_true(len(t_inx) == len(trough_indices[1:-1]) and np.all(t_inx == trough_indices[1:-1]),
                "trim_to_peak(peak_indices[1:], trough_indices[:-1]) failed on troughs")

    # peak first, more peaks:
    p_inx, t_inx = pd.trim_to_peak(peak_indices, trough_indices[:-2])
    assert_true(len(p_inx) == len(peak_indices[:-2]) and np.all(p_inx == peak_indices[:-2]),
                "trim_to_peak(peak_indices, trough_indices[:-2]) failed on peaks")
    assert_true(len(t_inx) == len(trough_indices[:-2]) and np.all(t_inx == trough_indices[:-2]),
                "trim_to_peak(peak_indices, trough_indices[:-2]) failed on troughs")

    # trough first, more troughs:
    p_inx, t_inx = pd.trim_to_peak(peak_indices[1:-2], trough_indices)
    assert_true(len(p_inx) == len(peak_indices[1:-2]) and np.all(p_inx == peak_indices[1:-2]),
                "trim_to_peak(peak_indices[1:-2], trough_indices) failed on peaks")
    assert_true(len(t_inx) == len(trough_indices[1:-2]) and np.all(t_inx == trough_indices[1:-2]),
                "trim_to_peak(peak_indices[1:-2], trough_indices) failed on troughs")


def test_trim_closest():
    # generate peak and trough indices (same length, peaks first):
    pt_indices = np.random.randint(5, 100, size=40) * 10
    pt_indices.sort()
    peak_indices = pt_indices

    trough_indices = peak_indices - np.random.randint(1, 5, size=len(peak_indices))
    p_inx, t_inx = pd.trim_closest(peak_indices, trough_indices)
    assert_true(len(p_inx) == len(peak_indices) and np.all(p_inx == peak_indices),
                "trim_closest(peak_indices, peak_indices-5) failed on peaks")
    assert_true(len(t_inx) == len(trough_indices) and np.all(t_inx == trough_indices),
                "trim_closest(peak_indices, peak_indices-5) failed on troughs")

    p_inx, t_inx = pd.trim_closest(peak_indices[1:], trough_indices[:-1])
    assert_true(len(p_inx) == len(peak_indices[1:-1]) and np.all(p_inx == peak_indices[1:-1]),
                "trim_closest(peak_indices[1:], peak_indices-5) failed on peaks")
    assert_true(len(t_inx) == len(trough_indices[1:-1]) and np.all(t_inx == trough_indices[1:-1]),
                "trim_closest(peak_indices[1:], peak_indices-5) failed on troughs")

    trough_indices = peak_indices + np.random.randint(1, 5, size=len(peak_indices))
    p_inx, t_inx = pd.trim_closest(peak_indices, trough_indices)
    assert_true(len(p_inx) == len(peak_indices) and np.all(p_inx == peak_indices),
                "trim_closest(peak_indices, peak_indices+5) failed on peaks")
    assert_true(len(t_inx) == len(trough_indices) and np.all(t_inx == trough_indices),
                "trim_closest(peak_indices, peak_indices+5) failed on troughs")

    p_inx, t_inx = pd.trim_closest(np.array([]), np.array([]))
    assert_true(len(p_inx) == 0,
                "trim_closest([], []) failed on peaks")
    assert_true(len(t_inx) == 0,
                "trim_closest([], []) failed on troughs")
