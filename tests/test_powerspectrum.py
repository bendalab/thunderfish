from nose.tools import assert_equal, assert_true
import numpy as np
import thunderfish.powerspectrum as ps

# run this with "nosetests tests/test_powerspectrum.py" in the first thunderfish folder.

def test_powerspectrum():
    # generate data
    fundamental = 300.  # Hz
    samplerate = 100000
    time = np.linspace(0, 8 - 1 / samplerate, 8 * samplerate)
    data = np.sin(time * 2 * np.pi * fundamental)

    # run multi_resolution_psd with 2 fresolutions (list)
    psd_data = ps.multi_resolution_psd(data, samplerate, fresolution=[0.5, 1])

    # test the results
    assert_equal(round(psd_data[0][1][np.argmax(psd_data[0][0])]), fundamental,
                 'peak in PSD is not the fundamental frequency given.')
    assert_equal(round(psd_data[1][1][np.argmax(psd_data[1][0])]), fundamental,
                 'peak in PSD is not the fundamental frequency given.')
    # run multi_resolution_psd with 1 fresolutions (float)
    psd_data = ps.multi_resolution_psd(data, samplerate, fresolution=0.5)

    # test the result
    assert_equal(round(psd_data[1][np.argmax(psd_data[0])]), fundamental,
                 'peak in PSD is not the fundamental frequency given.')


def test_peak_freqs():
    # generate data:
    dt = 0.001
    time = np.arange(0.0, 10.0, dt)
    data = np.zeros(len(time))
    w = len(data)//10
    freqs = 20.0 + 80.0*np.random.rand(10)
    onsets = []
    offsets = []
    for k in range(10):
        i0 = k*w
        i1 = i0 + w
        data[i0:i1] = np.sin(2.0*np.pi*freqs[k]*time[i0:i1])
        onsets.append(i0+w//10)
        offsets.append(i1-w//10)
    df = 0.5
    mfreqs = ps.peak_freqs(onsets, offsets, data, 1.0/dt, freq_resolution=df)
    assert_true(np.all(np.abs(freqs - mfreqs) <= 2.0*df), "peak_freqs() failed")
    
