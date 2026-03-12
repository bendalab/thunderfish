import pytest
import os
import numpy as np
import matplotlib.pyplot as plt
from thunderlab.eventdetection import detect_peaks
from thunderfish.fakefish import wavefish_eods, pulsefish_eods
import thunderfish.pulseanalysis as pa
import thunderfish.waveanalysis as wa
import thunderfish.eodanalysis as ea


def test_pulsefish():
    rate = 44100.0
    data = pulsefish_eods('Biphasic', 200.0, rate, 5.0, noise_std=0.02)
    pi, _ = detect_peaks(data, 1.0)
    mean_eod, eod_times = ea.eod_waveform(data, rate, pi/rate)
    mean_eod, props, peaks, pulses, power = \
        pa.analyze_pulse(mean_eod, None, eod_times)
    fig, (axw, axp) = plt.subplots(1, 2, layout='constrained')
    pa.plot_pulse_eod(axw, mean_eod, props)
    pa.plot_pulse_spectrum(axp, power, props)
    fig.savefig('pulse.png')
    assert os.path.exists('pulse.png'), 'plotting failed'
    os.remove('pulse.png')

    
def test_wavefish():
    rate = 44100.0
    EODf = 800.0
    data = wavefish_eods('Alepto', EODf, rate, duration=10.0, noise_std=0.01)
    eod_times = np.arange(0.01, 9.95, 1.0/EODf)
    mean_eod, eod_times = ea.eod_waveform(data, rate, eod_times)
    mean_eod, props, phases, spec = wa.analyze_wave(mean_eod, None, 800.0)
    fig, (axw, axa, axp) = plt.subplots(1, 3, layout='constrained')
    wa.plot_wave_eod(axw, mean_eod, props)
    wa.plot_wave_spectrum(axa, axp, spec, props)
    fig.savefig('wave.png')
    assert os.path.exists('wave.png'), 'plotting failed'
    os.remove('wave.png')
