from nose.tools import assert_true, assert_equal, assert_almost_equal
import numpy as np
import thunderfish.fakefish as ff


def test_harmonic_groups():

    # generate data:
    samplerate = 44100.0
    duration = 1.
    
    # generate data:
    time = np.arange(0, duration, 1./samplerate)

    # wavefish with fixed frequency:
    eodf = 300.0
    data = ff.generate_wavefish(eodf, samplerate, duration=duration, noise_std=0.02, 
                                amplitudes=[1.0, 0.5, 0.0, 0.0001],
                                phases=[0.0, 0.0, 0.0, 0.0])
    assert_equal(len(time), len(data), 'generate_wavefish() failed')

    data = ff.generate_alepto(eodf, samplerate, duration=duration)
    assert_equal(len(time), len(data), 'generate_alepto() failed')

    data = ff.generate_eigenmannia(eodf, samplerate, duration=duration)
    assert_equal(len(time), len(data), 'generate_eigenmannia() failed')
    
    
    # wavefish with frequency modulation:
    eodf = 500.0 - time/duration*400.0
    data = ff.generate_wavefish(eodf, samplerate, duration=duration, noise_std=0.02, 
                                amplitudes=[1.0, 0.5, 0.0, 0.0001],
                                phases=[0.0, 0.0, 0.0, 0.0])
    assert_equal(len(time), len(data), 'generate_wavefish() failed')

    data = ff.generate_alepto(eodf, samplerate, duration=duration)
    assert_equal(len(time), len(data), 'generate_alepto() failed')

    data = ff.generate_eigenmannia(eodf, samplerate, duration=duration)
    assert_equal(len(time), len(data), 'generate_eigenmannia() failed')

    # pulse fishes:
    data = ff.generate_pulsefish(80., samplerate, duration=duration,
                                 noise_std=0.02, jitter_cv=0.1,
                                 peak_stds=[0.0001, 0.0002],
                                 peak_amplitudes=[1.0, -0.3],
                                 peak_times=[0.0, 0.0003])
    assert_equal(len(time), len(data), 'generate_pulsefish() failed')
    
    data = ff.generate_monophasic_pulses(80., samplerate, duration=duration)
    assert_equal(len(time), len(data), 'generate_monophasic_pulsefish() failed')

    data = ff.generate_biphasic_pulses(80., samplerate, duration=duration)
    assert_equal(len(time), len(data), 'generate_biphasic_pulsefish() failed')

    data = ff.generate_triphasic_pulses(80., samplerate, duration=duration)
    assert_equal(len(time), len(data), 'generate_triphasic_pulsefish() failed')

    # communication signals:
    data = ff.chirps_frequency(600.0, samplerate, duration=duration, chirp_kurtosis=1.0)
    assert_equal(len(time), len(data), 'chirps_frequency() failed')

    data = ff.rises_frequency(600.0, samplerate, duration=duration, rise_size=20.0)
    assert_equal(len(time), len(data), 'rises_frequency() failed')
    
    
