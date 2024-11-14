import pytest
import os
import sys
from io import StringIO
import numpy as np
import thunderfish.fakefish as ff


def test_musical_intervals():
    r = 0.0
    for i, k in enumerate(ff.musical_intervals):
        x = ff.musical_intervals[k]
        assert len(x) == 4, 'musical_intervals wrong number of item elements'
        assert x[3] == i, 'musical_intervals wrong index'
        assert x[0] > r, 'musical_intervals ratio too small'
        assert x[1]/x[2] == x[0], 'musical_intervals wrong ratio'
        r = x[0]

        
def test_wavefish():
    # generate data:
    rate = 44100.0
    duration = 1.
    
    # generate data:
    time = np.arange(0, duration, 1./rate)

    # wavefish with fixed frequency:
    eodf = 300.0
    with pytest.raises(IndexError):
        ff.wavefish_eods(([1.0, 0.5, 0.0, 0.0001], [0.0, 0.0]), eodf,
                         rate, duration=duration)

    with pytest.raises(KeyError):
        ff.wavefish_eods('Amicro', eodf, rate, duration=duration)
    
    data = ff.wavefish_eods(([1.0, 0.5, 0.0, 0.0001], [0.0, 0.0, 0.0, 0.0]),
                            eodf, rate, duration=duration, noise_std=0.02)
    assert len(time) == len(data), 'wavefish_eods(tuple) failed'

    data = ff.wavefish_eods(ff.Apteronotus_leptorhynchus_harmonics,
                            eodf, rate, duration=duration)
    assert len(time) == len(data), 'wavefish_eods(Alepto_leptorhynchus_harmonics) failed'

    data = ff.wavefish_eods('Alepto', eodf, rate, duration=duration)
    assert len(time) == len(data), 'wavefish_eods(Alepto) failed'

    data = ff.wavefish_eods('Eigenmannia', eodf, rate, duration=duration)
    assert len(time) == len(data), 'wavefish_eods(Eigenmannia) failed'
    
    # wavefish with frequency modulation:
    eodf = 500.0 - time/duration*400.0
    data = ff.wavefish_eods('Eigenmannia', eodf, rate, duration=duration, noise_std=0.02)
    assert len(time) == len(data), 'wavefish_eods(frequency ramp) failed'

    # normalize:
    for key in ff.wavefish_harmonics:
        ff.normalize_wavefish(key, 'peak')
        ff.normalize_wavefish(key, 'zero')
        ff.export_wavefish(key, f'{key}_fish', sys.stdout)
    ff.export_wavefish('Alepto', 'test_fish')
    ff.export_wavefish('Alepto', 'test_fish', 'testfile.txt')
    os.remove('testfile.txt')

    
def test_communication():
    # generate data:
    rate = 44100
    duration = 10.
    
    eodf, ampl = ff.chirps(600.0, rate, duration=duration, chirp_kurtosis=1.0)
    assert duration*rate == len(eodf), 'chirps() failed'
    assert duration*rate == len(ampl), 'chirps() failed'

    data = ff.rises(600.0, rate, duration=10,
                    rise_freq=0.5, rise_size=20.0)
    assert duration*rate == len(data), 'rises() failed'


def test_pulsefish():
    # generate data:
    rate = 44100.0
    duration = 1.
    time = np.arange(0, duration, 1./rate)

    # pulse fishes:
    data = ff.pulsefish_eods(([0.0, 0.0003], [1.0, -0.3], [0.0001, 0.0002]),
                             80.0, rate, duration=duration,
                             noise_std=0.02, jitter_cv=0.1)
    assert len(time) == len(data), 'pulsefish_eods() failed'

    for key in ff.pulsefish_eodpeaks:
        data = ff.pulsefish_eods(key, 80., rate, duration=duration)
        assert len(time) == len(data), f'pulsefish_eods({key}) failed'
        ff.normalize_pulsefish(key)
    
    data = ff.pulsefish_eods(ff.Biphasic_peaks, 80., rate,
                             duration=duration)
    assert len(time) == len(data), f'pulsefish_eods({key}) failed'

    with pytest.raises(KeyError):
        ff.pulsefish_eods('Quad', 80.0, rate, duration=duration)
    with pytest.raises(IndexError):
        ff.pulsefish_eods(([0.0, 0.0003], [1.0], [0.0001, 0.0002]),
                          80.0, rate, duration=duration)

    ff.export_pulsefish('Biphasic', 'test_fish')
    ff.export_pulsefish('Triphasic', 'test_fish', 'testfile.txt')
    ff.export_pulsefish('Monophasic', 'test_fish')
    time = np.random.rand(10)*2 - 0.5
    amps = np.random.rand(10)*2 - 0.5
    stdevs = np.random.rand(10)*2 - 0.5
    ff.export_pulsefish((time, amps, stdevs), 'test_fish')
    os.remove('testfile.txt')

    
def test_speciesnames():
    s = ff.abbrv_genus('Apteronotus albifrons')
    assert s == 'A. albifrons', 'abbrv_genus() failed'


def test_generate_waveform():
    ips = '44100\n10\n1\n1\nSPEC\n800\n1\nn\n'
    for wave_species in ['a', 'e']:
        sys.stdin = StringIO(ips.replace('SPEC', wave_species))
        ff.generate_waveform('testeod.wav')
    for pulse_species in ['1', '2', '3']:
        sys.stdin = StringIO(ips.replace('SPEC', pulse_species) + '0.2\n')
        ff.generate_waveform('testeod.wav')
    cips = ips[:-2].replace('SPEC', 'a') + 'c\n2\n85\n19\n'
    sys.stdin = StringIO(cips)
    ff.generate_waveform('testeod.wav')
    rips = ips[:-2].replace('SPEC', 'a') + 'r\n0.5\n20\n0.1\n1.5\n'
    sys.stdin = StringIO(rips)
    ff.generate_waveform('testeod.wav')
    os.remove('testeod.wav')
    os.remove('testeod.inp')

    
def test_main():
    ff.main(['-h'])
    ff.main()
