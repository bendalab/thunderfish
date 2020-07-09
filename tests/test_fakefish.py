from nose.tools import assert_true, assert_equal, assert_almost_equal
import numpy as np
import thunderfish.fakefish as ff


def test_fakefish():

    # generate data:
    samplerate = 44100.0
    duration = 1.
    
    # generate data:
    time = np.arange(0, duration, 1./samplerate)

    # wavefish with fixed frequency:
    eodf = 300.0
    data = ff.wavefish_eods(([1.0, 0.5, 0.0, 0.0001], [0.0, 0.0, 0.0, 0.0]),
                            eodf, samplerate, duration=duration, noise_std=0.02)
    assert_equal(len(time), len(data), 'wavefish_eods(tuple) failed')

    data = ff.wavefish_eods('Alepto', eodf, samplerate, duration=duration)
    assert_equal(len(time), len(data), 'wavefish_eods(Alepto) failed')

    data = ff.wavefish_eods('Eigenmannia', eodf, samplerate, duration=duration)
    assert_equal(len(time), len(data), 'wavefish_eods(Eigenmannia) failed')
    
    # wavefish with frequency modulation:
    eodf = 500.0 - time/duration*400.0
    data = ff.wavefish_eods('Eigenmannia', eodf, samplerate, duration=duration, noise_std=0.02)
    assert_equal(len(time), len(data), 'wavefish_eods(frequency ramp) failed')

    # pulse fishes:
    data = ff.pulsefish_eods(([0.0, 0.0003], [1.0, -0.3], [0.0001, 0.0002]),
                             80.0, samplerate, duration=duration,
                             noise_std=0.02, jitter_cv=0.1)
    assert_equal(len(time), len(data), 'pulsefish_eods() failed')
    
    data = ff.pulsefish_eods('Monophasic', 80., samplerate, duration=duration)
    assert_equal(len(time), len(data), 'pulsefish_eods(Monophasic) failed')

    data = ff.pulsefish_eods('Biphasic', 80., samplerate, duration=duration)
    assert_equal(len(time), len(data), 'pulsefish_eods(Biphasic) failed')

    data = ff.pulsefish_eods('Triphasic', 80., samplerate, duration=duration)
    assert_equal(len(time), len(data), 'pulsefish_eods(Triphasic) failed')

    # communication signals:
    eodf, ampl = ff.chirps(600.0, samplerate, duration=duration, chirp_kurtosis=1.0)
    assert_equal(len(time), len(eodf), 'chirps() failed')
    assert_equal(len(time), len(ampl), 'chirps() failed')

    data = ff.rises(600.0, samplerate, duration=duration, rise_size=20.0)
    assert_equal(len(time), len(data), 'rises() failed')

    
