from nose.tools import assert_true, assert_equal, assert_almost_equal
import numpy as np
import thunderfish.fakefish as ff
import thunderfish.powerspectrum as ps
import thunderfish.harmonicgroups as hg


def test_harmonic_groups():

    # generate data:
    samplerate = 44100.0
    df = 0.5
    eodfs = np.array([123.0, 321.0, 666.0, 668.0])
    fish1 = ff.generate_wavefish(eodfs[0], samplerate, duration=8.0, noise_std=0.01,
                                 amplitudes=[1.0, 0.5, 0.2, 0.1, 0.05],
                                 phases=[0.0, 0.0, 0.0, 0.0, 0.0])
    fish2 = ff.generate_wavefish(eodfs[1], samplerate, duration=8.0, noise_std=0.0,
                                 amplitudes=[1.0, 0.7, 0.2, 0.1],
                                 phases=[0.0, 0.0, 0.0, 0.0])
    fish3 = ff.generate_wavefish(eodfs[2], samplerate, duration=8.0, noise_std=0.0,
                                 amplitudes=[10.0, 5.0, 1.0],
                                 phases=[0.0, 0.0, 0.0])
    fish4 = ff.generate_wavefish(eodfs[3], samplerate, duration=8.0, noise_std=0.0,
                                 amplitudes=[6.0, 3.0, 1.0],
                                 phases=[0.0, 0.0, 0.0])
    data = fish1 + fish2 + fish3 + fish4

    # analyse:
    psd_data = ps.psd(data, samplerate, fresolution=df)
    groups = hg.harmonic_groups(psd_data[1], psd_data[0])[0]
    fundamentals = hg.fundamental_freqs(groups)
    fdbs = hg.fundamental_freqs_and_db(groups)

    # check:
    assert_true(np.all(np.abs(eodfs-fundamentals) < df),
                'harmonic_groups() did not correctly detect all fundamental frequencies')
