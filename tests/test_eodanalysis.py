from nose.tools import assert_true, assert_equal, assert_almost_equal
import os
import numpy as np
import matplotlib.pyplot as plt
from thunderfish.fakefish import generate_biphasic_pulses, generate_alepto
from thunderfish.eventdetection import detect_peaks
import thunderfish.eodanalysis as ea


def test_pulsefish():
    samplerate = 44100.0
    data = generate_biphasic_pulses(200.0, samplerate, 5.0, noise_std=0.02)
    pi, _ = detect_peaks(data, 1.0)
    mean_eod, eod_times = ea.eod_waveform(data, samplerate, pi/samplerate)
    mean_eod, props, peaks, power = ea.analyze_pulse(mean_eod, eod_times)
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ea.pulse_spectrum_plot(power, props, ax)
    fig.savefig('pulse.png')
    assert_true(os.path.exists('pulse.png'), 'plotting failed')
    os.remove('pulse.png')

def test_wavefish():
    samplerate = 44100.0
    EODf = 800.0
    data = generate_alepto(EODf, samplerate, duration=10.0, noise_std=0.01)
    eod_times = np.arange(0.01, 9.95, 1.0/EODf)
    mean_eod, eod_times = ea.eod_waveform(data, samplerate, eod_times)
    mean_eod, props, power, es = ea.analyze_wave(mean_eod, 800.0)
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ea.eod_waveform_plot(mean_eod, [], ax)
    fig.savefig('wave.png')
    assert_true(os.path.exists('wave.png'), 'plotting failed')
    os.remove('wave.png')
