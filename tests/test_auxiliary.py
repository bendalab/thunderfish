import numpy as np
from nose.tools import assert_true

from thunderfish.Auxiliary import peakdet


def test_peakdet():
    """Tests whether peakdet works corretly"""
    x = np.zeros(1000)
    peaks = np.random.randint(0,len(x), size=20)
    peaks.sort()
    while np.any(np.diff(peaks).min() < 2):
        peaks = np.random.randint(0,len(x), size=20)
        peaks.sort()
    x[peaks] = 5
    maxtab, maxidx, _, _ = peakdet(x, 2)
    assert_true(np.all(maxidx == peaks), "Peaks were not detected correctly")
