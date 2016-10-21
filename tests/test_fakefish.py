from nose.tools import assert_true, assert_equal, assert_almost_equal
import numpy as np
import thunderfish.fakefish as ff


def test_harmonic_groups():

    # generate data:
    samplerate = 44100.0
    df = 0.5
    eodfs = np.array([123.0, 321.0, 666.0, 668.0])
    fish1 = ff.generate_wavefish(eodfs[0], samplerate, duration=8.0, noise_std=0.01,
                                 amplitudes=[1.0, 0.5, 0.2, 0.1, 0.05])
    fish2 = ff.generate_wavefish(eodfs[1], samplerate, duration=8.0, noise_std=0.01,
                                 amplitudes=[1.0, 0.7, 0.2, 0.1])
    fish3 = ff.generate_wavefish(eodfs[2], samplerate, duration=8.0, noise_std=0.01,
                                 amplitudes=[10.0, 5.0, 1.0])
    fish4 = ff.generate_wavefish(eodfs[3], samplerate, duration=8.0, noise_std=0.01,
                                 amplitudes=[6.0, 3.0, 1.0])

    # check:
    assert_true(np.all(np.abs(eodfs-fundamentals) < df),
                'harmonic_groups() did not correctly detect all fundamental frequencies')
