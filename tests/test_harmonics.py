import pytest
import numpy as np
from thunderlab.configfile import ConfigFile
import thunderlab.powerspectrum as ps
import thunderfish.fakefish as ff
import thunderfish.harmonics as hg


def test_harmonic_groups():

    # generate data:
    rate = 44100.0
    df = 0.5
    eodfs = np.array([123.0, 321.0, 666.0, 668.0])
    fish1 = ff.wavefish_eods(([1.0, 0.5, 0.2, 0.1, 0.05], [0.0, 0.0, 0.0, 0.0, 0.0]),
                             eodfs[0], rate, duration=8.0, noise_std=0.01)
    fish2 = ff.wavefish_eods('Eigenmannia', eodfs[1], rate, duration=8.0, noise_std=0.0)
    fish3 = ff.wavefish_eods('Alepto', eodfs[2], rate, duration=8.0, noise_std=0.0)
    fish4 = ff.wavefish_eods('Arostratus', eodfs[3], rate, duration=8.0, noise_std=0.0)
    data = fish1 + fish2 + fish3 + fish4

    # analyse:
    psd_data = ps.psd(data, rate, freq_resolution=df)
    groups = hg.harmonic_groups(psd_data[0], psd_data[1], max_db_diff=20.0, verbose=10)[0]
    fundamentals = hg.fundamental_freqs(groups)
    fdbs = hg.fundamental_freqs_and_power(groups)
    # check:
    assert np.all(np.abs(eodfs-fundamentals) < 1.5*df), 'harmonic_groups() did not correctly detect all fundamental frequencies'

    fundamentals = hg.fundamental_freqs([groups, [groups[1], groups[3]]])
    fdbs = hg.fundamental_freqs_and_power([groups, [groups[1], groups[3]]], 3)
    # check:
    assert np.all(np.abs(eodfs-fundamentals[0]) < df), 'harmonic_groups() did not correctly detect all fundamental frequencies'

    fundamentals = hg.fundamental_freqs([[groups, [groups[1], groups[3]]]])
    fdbs = hg.fundamental_freqs_and_power([[groups, [groups[1], groups[3]]]], 10, True)
    # check:
    assert np.all(np.abs(eodfs-fundamentals[0][0]) < df), 'harmonic_groups() did not correctly detect all fundamental frequencies'


def test_config():
    cfg = ConfigFile()
    hg.add_harmonic_groups_config(cfg)
    hg.add_psd_peak_detection_config(cfg)
    assert type(hg.harmonic_groups_args(cfg)) is dict, 'harmonic_groups_args()'
    assert type(hg.psd_peak_detection_args(cfg)) is dict, 'psd_peak_detection_args()'
    

def test_main():
    hg.main()
        
