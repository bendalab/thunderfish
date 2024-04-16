import pytest
import numpy as np
import thunderfish.powerspectrum as ps
from thunderfish.configfile import ConfigFile
import matplotlib.pyplot as plt

        
def test_decibel():
    data = np.random.randn(1000)**2
    for ref in [0.1, 1.0, None, 'peak']:
        l = ps.decibel(data, ref, 1e-10)
        assert data.shape == l.shape, 'decibel()'
        if type(ref) is float:
            p = ps.power(l, ref)
            assert p.shape == l.shape, 'power()'


def test_powerspectrum():
    # generate data
    fundamental = 300.  # Hz
    samplerate = 100000
    time = np.arange(0, 8, 1/samplerate)
    data = np.sin(time * 2 * np.pi * fundamental)

    # run multi_psd with two windows:
    psd_data = ps.multi_psd(data, samplerate, freq_resolution=0.5,
                            num_windows=2)
    assert round(psd_data[0][np.argmax(psd_data[0][:,1]),0]) == fundamental, 'peak in PSD is not the fundamental frequency given.'
    assert round(psd_data[1][np.argmax(psd_data[1][:,1]),0]) == fundamental, 'peak in PSD is not the fundamental frequency given.'
    
    # run multi_psd with one window:
    psd_data = ps.multi_psd(data, samplerate, freq_resolution=0.5)
    assert round(psd_data[0][np.argmax(psd_data[0][:,1]),0]) == fundamental, 'peak in PSD is not the fundamental frequency given.'

    # check detrend:
    for detrend in ['none', 'constant', 'mean', 'linear']:
        psd_data = ps.multi_psd(data, samplerate, freq_resolution=0.5,
                                detrend=detrend)

    # run multi_psd with two windows with mlab:
    ps.psdscipy = False
    psd_data = ps.multi_psd(data, samplerate, freq_resolution=0.5,
                            num_windows=2)
    assert round(psd_data[0][np.argmax(psd_data[0][:,1]),0]) == fundamental, 'peak in PSD is not the fundamental frequency given.'
    assert round(psd_data[1][np.argmax(psd_data[1][:,1]),0]) == fundamental, 'peak in PSD is not the fundamental frequency given.'

    # check detrend with mlab:
    for detrend in ['constant', 'mean', 'linear']:
        psd_data = ps.multi_psd(data, samplerate, freq_resolution=0.5,
                                detrend=detrend)


def test_spectrogram():
    # generate data
    fundamental = 300.  # Hz
    samplerate = 100000
    time = np.arange(0, 8, 1/samplerate)
    data = np.sin(time * 2 * np.pi * fundamental)
    n = 4
    datan = np.zeros((len(time), n))
    for k in range(n):
        datan[:,k] = np.sin(time * 2 * np.pi * (k+1) * fundamental)

    freqs, time, spec = ps.spectrogram(data, samplerate, freq_resolution=2)
    idx = np.argmax(spec, 0)
    assert spec.shape[0] == len(freqs), 'spectrogram() frequency dimension'
    assert spec.shape[1] == len(time), 'spectrogram() time dimension'
    assert np.all(idx == idx[0]), 'spectrogram() peak positions'
    assert np.abs(freqs[idx[0]] - fundamental) < 3, 'peak in spectrogram() is not the fundamental frequency given.'

    freqs, time, spec = ps.spectrogram(datan, samplerate, freq_resolution=2)
    assert spec.shape[0] == len(freqs), 'spectrogram() frequency dimension'
    assert spec.shape[1] == len(time), 'spectrogram() time dimension'
    assert spec.shape[2] == n, 'spectrogram() channel dimension'

    ps.specgramscipy = False
    freqs, time, spec = ps.spectrogram(data, samplerate, freq_resolution=2)
    idx = np.argmax(spec, 0)
    assert spec.shape[1] == len(time), 'spectrogram() time dimension'
    assert spec.shape[0] == len(freqs), 'spectrogram() frequency dimension'
    assert np.all(idx == idx[0]), 'spectrogram() peak positions'
    assert np.abs(freqs[idx[0]] - fundamental) < 3, 'peak in spectrogram() is not the fundamental frequency given.'
    
    freqs, time, spec = ps.spectrogram(datan, samplerate, freq_resolution=2)
    assert spec.shape[0] == len(freqs), 'spectrogram() frequency dimension'
    assert spec.shape[1] == len(time), 'spectrogram() time dimension'
    assert spec.shape[2] == n, 'spectrogram() channel dimension'

    
def test_plot_decibel_psd():
    # generate data
    fundamental = 300.  # Hz
    samplerate = 100000
    time = np.arange(0, 8, 1/samplerate)
    data = np.sin(time * 2 * np.pi * fundamental)
    freqs, power = ps.psd(data, samplerate, freq_resolution=1)
    
    fig, ax = plt.subplots()
    ps.plot_decibel_psd(ax, freqs, power)
    ps.plot_decibel_psd(ax, freqs, power, max_freq=0)
    ps.plot_decibel_psd(ax, freqs, power, log_freq=True)
    ps.plot_decibel_psd(ax, freqs, power, log_freq=True, min_freq=0)

    
def test_peak_freqs():
    # generate data:
    dt = 0.001
    time = np.arange(0.0, 10.0, dt)
    data = np.zeros(len(time))
    w = len(data)//10
    freqs = 20.0 + 4*np.random.randint(0, 20, 10)
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
    assert np.all(np.abs(freqs - mfreqs) <= 2.0*df), "peak_freqs() failed"

    mfreqs = ps.peak_freqs(onsets, offsets, data, 1.0/dt, freq_resolution=df,
                           thresh=10)
    assert np.all(np.abs(freqs - mfreqs) <= 2.0*df), "peak_freqs() with threshold failed"

    mfreqs = ps.peak_freqs(onsets, offsets, data, 1.0/dt, freq_resolution=df,
                           thresh=1000000)
    assert np.all(np.isnan(mfreqs)), "peak_freqs() returned no nans"
    
    mfreqs = ps.peak_freqs(onsets, offsets, data, 1.0/dt, freq_resolution=df,
                           max_nfft=2**12)

    
def test_config():
    cfg = ConfigFile()
    ps.add_multi_psd_config(cfg)
    ps.multi_psd_args(cfg)

    
def test_main():
    ps.main()
    
