import pytest
import os
import numpy as np
import matplotlib.pyplot as plt
from thunderlab.eventdetection import detect_peaks
from thunderfish.fakefish import wavefish_eods, pulsefish_eods
import thunderfish.eodanalysis as ea


def test_pulsefish():
    samplerate = 44100.0
    data = pulsefish_eods('Biphasic', 200.0, samplerate, 5.0, noise_std=0.02)
    pi, _ = detect_peaks(data, 1.0)
    mean_eod, eod_times = ea.eod_waveform(data, samplerate, pi/samplerate)
    mean_eod, props, peaks, power = ea.analyze_pulse(mean_eod, eod_times)
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ea.plot_pulse_spectrum(ax, power, props)
    fig.savefig('pulse.png')
    assert os.path.exists('pulse.png'), 'plotting failed'
    os.remove('pulse.png')

def test_wavefish():
    samplerate = 44100.0
    EODf = 800.0
    data = wavefish_eods('Alepto', EODf, samplerate, duration=10.0, noise_std=0.01)
    eod_times = np.arange(0.01, 9.95, 1.0/EODf)
    mean_eod, eod_times = ea.eod_waveform(data, samplerate, eod_times)
    mean_eod, props, power, es = ea.analyze_wave(mean_eod, 800.0)
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ea.plot_eod_waveform(ax, mean_eod, props)
    fig.savefig('wave.png')
    assert os.path.exists('wave.png'), 'plotting failed'
    os.remove('wave.png')
