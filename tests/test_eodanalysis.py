from nose.tools import assert_true, assert_equal, assert_almost_equal
import numpy as np
import thunderfish.fakefish as ff
import thunderfish.eodanalysis as ea


def test_pulsefish():
    samplerate = 44100.0
    data = ff.generate_biphasic_pulses(200.0, samplerate, 5.0, noise_std=0.02)
    mean_eod, eod_times = ea.eod_waveform(data, samplerate)
    mean_eod, props, peaks, power, intervals = ea.analyze_pulse(mean_eod, eod_times)

def test_wavefish():
    samplerate = 44100.0
    data = ff.generate_alepto(800.0, samplerate, duration=10.0, noise_std=0.01)
    mean_eod, eod_times = ea.eod_waveform(data, samplerate)
    mean_eod, props, power = ea.analyze_wave(mean_eod, 800.0)
