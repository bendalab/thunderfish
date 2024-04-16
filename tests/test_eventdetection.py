import pytest
import numpy as np
import thunderfish.eventdetection as ed


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
    for i in range(len(pt_indices) - 1):
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

    with pytest.raises(ValueError):
        ed.detect_peaks(data, 0.0)
    with pytest.raises(ValueError):
        ed.detect_peaks(data, -1.0)
    
    peaks, troughs = ed.detect_peaks(data, threshold)
    assert np.all(peaks == peak_indices), "detect_peaks(data, threshold) did not correctly detect peaks"
    assert np.all(troughs == trough_indices), "detect_peaks(data, threshold) did not correctly detect troughs"

    threshs = np.ones(len(data))*threshold
    peaks, troughs = ed.detect_peaks(data, threshs)
    assert np.all(peaks == peak_indices), "detect_peaks(data, threshs) did not correctly detect peaks"
    assert np.all(troughs == trough_indices), "detect_peaks(data, threshs) did not correctly detect troughs"
    
    with pytest.raises(IndexError):
        ed.detect_peaks(data, threshs[:10])
    threshs[10] = -1
    with pytest.raises(ValueError):
        ed.detect_peaks(data, threshs)


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
    for i in range(len(pt_indices) - 1):
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

    with pytest.raises(ValueError):
        ed.detect_dynamic_peaks(data, 0.0, min_thresh, 0.5, time,
                                ed.accept_peak_size_threshold)

    with pytest.raises(ValueError):
        ed.detect_dynamic_peaks(data, -1.0, min_thresh, 0.5, time,
                                ed.accept_peak_size_threshold)

    with pytest.raises(ValueError):
        ed.detect_dynamic_peaks(data, threshold, 0.0, 0.5, time,
                                ed.accept_peak_size_threshold)

    with pytest.raises(ValueError):
        ed.detect_dynamic_peaks(data, threshold, -1.0, 0.5, time,
                                ed.accept_peak_size_threshold)

    with pytest.raises(ValueError):
        ed.detect_dynamic_peaks(data, threshold, min_thresh, 0.0, time,
                                ed.accept_peak_size_threshold)

    with pytest.raises(ValueError):
        ed.detect_dynamic_peaks(data, threshold, min_thresh, -1.0, time,
                                ed.accept_peak_size_threshold)

    with pytest.raises(IndexError):
        ed.detect_dynamic_peaks(data, threshold, min_thresh, 0.5,
                                time[:len(time)//2],
                                ed.accept_peak_size_threshold)

    peaks, troughs = ed.detect_dynamic_peaks(data, threshold,
                                             min_thresh, 0.5, time,
                                             ed.accept_peak_size_threshold)
    assert np.all(peaks == peak_times), "detect_dynamic_peaks(data, threshold, time, accept_peak_size_threshold) did not correctly detect peaks"
    assert np.all(troughs == trough_times), "detect_dynamic_peaks(data, threshold, time, accept_peak_size_threshold) did not correctly detect troughs"

    peaks, troughs = ed.detect_dynamic_peaks(data, threshold, min_thresh, 0.5, None,
                                             ed.accept_peak_size_threshold,
                                             thresh_ampl_fac=0.9, thresh_weight=0.1)
    assert np.all(peaks == peak_indices), "detect_dynamic_peaks(data, threshold, time, accept_peak_size_threshold) did not correctly detect peaks"
    assert np.all(troughs == trough_indices), "detect_dynamic_peaks(data, threshold, time, accept_peak_size_threshold) did not correctly detect troughs"


def test_threshold_crossings():
    # generate data:
    time = np.arange(0.0, 10.0, 0.01)
    data = np.zeros(time.shape)
    pt_indices = np.random.randint(5, len(data) - 10, size=40)
    pt_indices.sort()
    while np.any(np.diff(pt_indices).min() < 5):
        pt_indices = np.random.randint(5, len(data) - 10, size=40)
        pt_indices.sort()
    up_indices = pt_indices[0::2]
    down_indices = pt_indices[1::2]
    up = True
    for i in range(len(pt_indices) - 1):
        if up:
            data[pt_indices[i]:pt_indices[i + 1]] = 1.0
        else:
            data[pt_indices[i]:pt_indices[i + 1]] = 0.0
        up = not up
    if up:
        data[pt_indices[-1]:] = 1.0

    threshold = 0.5
    up, down = ed.threshold_crossings(data, threshold)
    assert np.all(up == up_indices-1), "threshold_crossings(data, threshold) did not correctly detect up crossings"
    assert np.all(down == down_indices-1), "threshold_crossings(data, threshold) did not correctly detect down crossings"

    upt, downt = ed.threshold_crossing_times(time, data, threshold, up, down)
    assert len(upt) == len(up) and len(downt) == len(down), 'threshold_crossings_time failed'

    threshold = 0.1 + 0.8/10.0*time
    with pytest.raises(IndexError):
        ed.threshold_crossings(data, threshold[1:])
    up, down = ed.threshold_crossings(data, threshold)
    assert np.all(up == up_indices-1), "threshold_crossings(data, threshold) did not correctly detect up crossings"
    assert np.all(down == down_indices-1), "threshold_crossings(data, threshold) did not correctly detect down crossings"

    
def test_thresholds():
    # generate noise data:
    data = np.random.randn(10000)
    std_th = ed.std_threshold(data, thresh_fac=1.0)
    prc_th = ed.percentile_threshold(data, thresh_fac=1.0, percentile=16.0)
    assert std_th == pytest.approx(1.0, abs=0.1), 'std_threshold %g esimate failed' % std_th
    assert np.abs(prc_th-2.0) < 0.1, 'percentile_threshold %g esimate failed' % prc_th

    std_ths = ed.std_threshold(data, 50, thresh_fac=1.0)
    assert len(std_ths) == len(data), 'std_threshold array'
    assert np.all(np.abs(std_ths - 1) < 1), 'threshold array estimated of std failed'

    prc_ths = ed.percentile_threshold(data, 50, thresh_fac=1.0, percentile=16.0)
    assert len(prc_ths) == len(data), 'percentile threshold array'

    hist_th, center = ed.hist_threshold(data, thresh_fac=1.0)
    assert hist_th == pytest.approx(1.0, abs=1), 'hist_threshold %g esimate failed' % hist_th
    assert center == pytest.approx(0.0, abs=1), 'hist_threshold center %g esimate failed' % center
    hist_th, center = ed.hist_threshold(1e-8*data + 10, thresh_fac=1.0)
    hist_ths, centers = ed.hist_threshold(data, 500, thresh_fac=1.0)
    assert len(hist_ths) == len(data), 'hist_threshold array thresholds'
    assert len(centers) == len(data), 'hist_threshold array centers'

    median_th = ed.median_std_threshold(data, thresh_fac=1.0)
    assert median_th == pytest.approx(1.0, abs=0.1), 'median_std_threshold %g esimate failed' % median_th
    median_th = ed.median_std_threshold(data, win_size=0, thresh_fac=1.0)

    # generate sine wave:
    time = np.arange(0.0, 10.0, 0.01)
    data = np.sin(2.0*np.pi*21.7*time)
    mm_th = ed.minmax_threshold(data, thresh_fac=1.0)
    assert mm_th == pytest.approx(2.0, abs=0.01), 'minmax_threshold %g esimate failed' % mm_th
    mm_ths = ed.minmax_threshold(data, 500, thresh_fac=1.0)
    assert len(mm_ths) == len(data), 'minmax threshold array'
    prc_th = ed.percentile_threshold(data, thresh_fac=1.0, percentile=0.1)
    assert np.abs(prc_th-2.0) < 0.1, 'percentile_threshold %g esimate failed' % prc_th
    


def test_trim():
    # generate peak and trough indices (same length, peaks first):
    pt_indices = np.unique(np.random.randint(5, 1000, size=40))
    n = (len(pt_indices)//2)*2
    peak_indices = pt_indices[0:n:2]
    trough_indices = pt_indices[1:n:2]

    # peak first, same length:
    p_inx, t_inx = ed.trim(peak_indices, trough_indices)
    assert len(p_inx) == len(peak_indices) and np.all(p_inx == peak_indices), "trim(peak_indices, trough_indices) failed on peaks"
    assert len(t_inx) == len(trough_indices) and np.all(t_inx == trough_indices), "trim(peak_indices, trough_indices) failed on troughs"

    # trough first, same length:
    p_inx, t_inx = ed.trim(peak_indices[1:], trough_indices[:-1])
    assert len(p_inx) == len(peak_indices[1:]) and np.all(p_inx == peak_indices[1:]), "trim(peak_indices[1:], trough_indices[:-1]) failed on peaks"
    assert len(t_inx) == len(trough_indices[:-1]) and np.all(t_inx == trough_indices[:-1]), "trim(peak_indices[1:], trough_indices[:-1]) failed on troughs"

    # peak first, more peaks:
    p_inx, t_inx = ed.trim(peak_indices, trough_indices[:-2])
    assert len(p_inx) == len(peak_indices[:-2]) and np.all(p_inx == peak_indices[:-2]), "trim(peak_indices, trough_indices[:-2]) failed on peaks"
    assert len(t_inx) == len(trough_indices[:-2]) and np.all(t_inx == trough_indices[:-2]), "trim(peak_indices, trough_indices[:-2]) failed on troughs"

    # trough first, more troughs:
    p_inx, t_inx = ed.trim(peak_indices[1:-2], trough_indices)
    assert len(p_inx) == len(peak_indices[1:-2]) and np.all(p_inx == peak_indices[1:-2]), "trim(peak_indices[1:-2], trough_indices) failed on peaks"
    assert len(t_inx) == len(trough_indices[:-3]) and np.all(t_inx == trough_indices[:-3]), "trim(peak_indices[1:-2], trough_indices) failed on troughs"

    
def test_trim_to_peak():
    # generate peak and trough indices (same length, peaks first):
    pt_indices = np.unique(np.random.randint(5, 1000, size=40))
    n = (len(pt_indices)//2)*2
    peak_indices = pt_indices[0:n:2]
    trough_indices = pt_indices[1:n:2]

    # peak first, same length:
    p_inx, t_inx = ed.trim_to_peak(peak_indices, trough_indices)
    assert len(p_inx) == len(peak_indices) and np.all(p_inx == peak_indices), "trim_to_peak(peak_indices, trough_indices) failed on peaks"
    assert len(t_inx) == len(trough_indices) and np.all(t_inx == trough_indices), "trim_to_peak(peak_indices, trough_indices) failed on troughs"

    # trough first, same length:
    p_inx, t_inx = ed.trim_to_peak(peak_indices[1:], trough_indices[:-1])
    assert len(p_inx) == len(peak_indices[1:-1]) and np.all(p_inx == peak_indices[1:-1]), "trim_to_peak(peak_indices[1:], trough_indices[:-1]) failed on peaks"
    assert len(t_inx) == len(trough_indices[1:-1]) and np.all(t_inx == trough_indices[1:-1]), "trim_to_peak(peak_indices[1:], trough_indices[:-1]) failed on troughs"

    # peak first, more peaks:
    p_inx, t_inx = ed.trim_to_peak(peak_indices, trough_indices[:-2])
    assert len(p_inx) == len(peak_indices[:-2]) and np.all(p_inx == peak_indices[:-2]), "trim_to_peak(peak_indices, trough_indices[:-2]) failed on peaks"
    assert len(t_inx) == len(trough_indices[:-2]) and np.all(t_inx == trough_indices[:-2]), "trim_to_peak(peak_indices, trough_indices[:-2]) failed on troughs"

    # trough first, more troughs:
    p_inx, t_inx = ed.trim_to_peak(peak_indices[1:-2], trough_indices)
    assert len(p_inx) == len(peak_indices[1:-2]) and np.all(p_inx == peak_indices[1:-2]), "trim_to_peak(peak_indices[1:-2], trough_indices) failed on peaks"
    assert len(t_inx) == len(trough_indices[1:-2]) and np.all(t_inx == trough_indices[1:-2]), "trim_to_peak(peak_indices[1:-2], trough_indices) failed on troughs"


def test_trim_closest():
    # generate peak and trough indices (same length, peaks first):
    pt_indices = np.unique(np.random.randint(5, 1000, size=40))
    n = (len(pt_indices)//2)*2
    peak_indices = pt_indices[0:n:2]
    trough_indices = pt_indices[1:n:2]

    trough_indices = peak_indices - np.random.randint(1, 5, size=len(peak_indices))
    p_inx, t_inx = ed.trim_closest(peak_indices, trough_indices)
    assert len(p_inx) == len(peak_indices) and np.all(p_inx == peak_indices), "trim_closest(peak_indices, peak_indices-5) failed on peaks"
    assert len(t_inx) == len(trough_indices) and np.all(t_inx == trough_indices), "trim_closest(peak_indices, peak_indices-5) failed on troughs"

    p_inx, t_inx = ed.trim_closest(peak_indices[1:], trough_indices[:-1])
    assert len(p_inx) == len(peak_indices[1:-1]) and np.all(p_inx == peak_indices[1:-1]), "trim_closest(peak_indices[1:], peak_indices-5) failed on peaks"
    assert len(t_inx) == len(trough_indices[1:-1]) and np.all(t_inx == trough_indices[1:-1]), "trim_closest(peak_indices[1:], peak_indices-5) failed on troughs"

    trough_indices = peak_indices + np.random.randint(1, 5, size=len(peak_indices))
    p_inx, t_inx = ed.trim_closest(peak_indices, trough_indices)
    assert len(p_inx) == len(peak_indices) and np.all(p_inx == peak_indices), "trim_closest(peak_indices, peak_indices+5) failed on peaks"
    assert len(t_inx) == len(trough_indices) and np.all(t_inx == trough_indices), "trim_closest(peak_indices, peak_indices+5) failed on troughs"

    p_inx, t_inx = ed.trim_closest(np.array([]), np.array([]))
    assert len(p_inx) == 0, "trim_closest([], []) failed on peaks"
    assert len(t_inx) == 0, "trim_closest([], []) failed on troughs"

    
def test_peak_width_algorithm():
    time = np.arange(-10.0, 10.0, 0.1)
    for i in range(1000):
        m = -3.0 + 6.0*np.random.rand()
        sd = 0.2 + 3.0*np.random.rand()
        peak_frac = 0.1+0.8*np.random.rand()
        data = np.exp(-0.5*((time-m)/sd)**2.0) - 0.2*np.exp(-0.5*((time-m)/5.0/sd)**2.0)
        pix, tix = ed.detect_peaks(data, 0.1)
        assert len(pix) == 1, 'only a single peak should be detected'
        assert len(tix) <= 2, 'less than three troughs should be detected'
        # we need a trough before and after each peak:
        peak_inx = np.asarray(pix, dtype=int)
        trough_inx = np.asarray(tix, dtype=int)
        if len(trough_inx) == 0 or peak_inx[0] < trough_inx[0]:
             trough_inx = np.hstack((0, trough_inx))
        if peak_inx[-1] > trough_inx[-1]:
             trough_inx = np.hstack((trough_inx, len(data)-1))
        assert len(trough_inx) == 2, 'we need two troughs around the peak'
        # width of peaks:
        widths = []
        for j in range(len(peak_inx)):
            li = trough_inx[j]
            ri = trough_inx[j+1]
            base = max(data[li], data[ri])
            thresh = base*(1.0-peak_frac) + data[peak_inx[j]]*peak_frac
            assert base < data[peak_inx[j]], 'base should be smaller than peak'
            assert thresh < data[peak_inx[j]], 'threshold should be smaller than peak'
            assert base < thresh, 'base should be smaller than threshold'
            inx0 = li + np.argmax(data[li:ri] > thresh)
            assert inx0 <= peak_inx[j], 'left index should be smaller than peak index'
            assert thresh >= data[inx0-1], 'thresh should be larger than inx0-1'
            assert thresh <= data[inx0], 'thresh should be smaller than inx0-1'
            ti0 = np.interp(thresh, data[inx0-1:inx0+1], time[inx0-1:inx0+1])
            assert ti0>=time[inx0-1], 'left time should be larger than inx0-1'
            assert ti0<=time[inx0], 'left time should be smaller than inx0'
            assert (thresh-data[inx0-1])/(data[inx0]-data[inx0-1]) == pytest.approx((ti0-time[inx0-1])/(time[inx0]-time[inx0-1]), abs=1e-5), 'left thresh fraction should equal time fraction'
            inx1 = ri - np.argmax(data[ri:li:-1] > thresh)
            assert inx1 >= peak_inx[j], 'right index should be larger than peak index'
            assert thresh >= data[inx1+1], 'thresh should be larger than inx1+1'
            assert thresh <= data[inx1], 'thresh should be smaller than inx1'
            ti1 = np.interp(thresh, data[inx1+1:inx1-1:-1], time[inx1+1:inx1-1:-1])
            assert ti1>=time[inx1], 'right time should be larger than inx1'
            assert ti1<=time[inx1+1], 'rigth time should be smaller than inx1+1'
            assert (thresh-data[inx1])/(data[inx1+1]-data[inx1]) == pytest.approx((ti1-time[inx1])/(time[inx1+1]-time[inx1]), abs=1e-5), 'right thresh fraction should equal time fraction'
            width = ti1 - ti0
            assert width>0.0, 'width should be larger than zero'
            widths.append(width)
        edwidths = ed.peak_width(time, data, pix, tix, peak_frac, 'max')
        assert np.all(widths == edwidths), 'widths should be the same'
        
        for base in ['left', 'right', 'min', 'max', 'mean', 'closest']:
            edwidths = ed.peak_width(time, data, pix, tix, peak_frac, base)
            assert len(edwidths) == len(pix), 'as many widths as peaks'
        with pytest.raises(ValueError):
            ed.peak_width(time, data, pix, tix, peak_frac, 'xxx')

        for base in ['left', 'right', 'min', 'max', 'mean', 'closest']:
            edpeaks = ed.peak_size_width(time, data, pix, tix, peak_frac, base)
            assert len(edpeaks) == len(pix), 'as many peaks as peaks'
        with pytest.raises(ValueError):
            ed.peak_size_width(time, data, pix, tix, peak_frac, 'xxx')
        edpeaks = ed.peak_size_width(time, data, [], [], peak_frac, base)
        assert len(edpeaks) == 0, 'no peaks'
        
                
def test_event_manipulation():
    # generate peak and trough indices (same length, peaks first):
    pt_indices = np.unique(np.random.randint(5, 1000, size=40))
    n = (len(pt_indices)//2)*2
    onsets = pt_indices[0:n:2]
    offsets = pt_indices[1:n:2]
    
    p, t = ed.merge_events(onsets, offsets, 10)
    assert len(p) > 0 and len(t) > 0 and len(p) == len(t), 'merged events'
    p, t = ed.merge_events([], offsets, 10)
    assert len(p) == 0 and len(t) == 0, 'no events'

    p, t = ed.widen_events(onsets, offsets, 1010, 10)
    assert len(p) > 0 and len(t) > 0 and len(p) == len(t), 'widened events'

    p, t = ed.remove_events(onsets, offsets, 10, 100)
    assert len(p) > 0 and len(t) > 0 and len(p) == len(t) and np.all(t-p >= 10) and np.all(t-p <= 100), 'removed events'
    p, t = ed.remove_events(onsets, offsets, 10)
    assert len(p) > 0 and len(t) > 0 and len(p) == len(t) and np.all(t-p >= 10), 'removed minimum events'
    p, t = ed.remove_events(onsets, offsets, None, 100)
    assert len(p) > 0 and len(t) > 0 and len(p) == len(t) and np.all(t-p <= 100), 'removed maximum events'
    p, t = ed.remove_events(onsets, [], 10, 100)
    assert len(p) == 0 and len(t) == 0, 'no removed events'
    
                
def test_snippets():
    # generate data:
    data = np.random.randn(10000)
    indices = np.unique(np.random.randint(50, 10000-50, size=200))

    snips = ed.snippets(data, indices)
    assert len(snips) == len(indices), 'snippets()'

                
def test_main():
    ed.main()

    
