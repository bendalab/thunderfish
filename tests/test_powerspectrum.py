from nose.tools import assert_true, assert_equal, assert_almost_equal
import numpy as np
import thunderfish.thunderfish.powerspectrum as ps

def test_powerspectrum():
    # generate data
    fundamental = 300.  # Hz
    samplerate = 100000
    time = np.linspace(0, 8 - 1 / samplerate, 8 * samplerate)
    data = np.sin(time * 2 * np.pi * fundamental[0]) + np.sin(time * 2 * np.pi * fundamental[1])

    psd_data = ps.multi_resolution_psd(data, samplerate, fresolution=[0.5, 1], verbose=1)

    assert_equal(psd_data[0][1][psd_data[0][0].argsort()[-1]], fundamental, 'peak in PSD is not the fundamental '
                                                                            'frequency given.')

    psd_data = ps.multi_resolution_psd(data, samplerate, fresolution=0.5, verbose=0)

    assert_equal(psd_data[1][psd_data[0].argsort()[-1]], fundamental, 'peak in PSD is not the fundamental frequency'
                                                                      ' given.')