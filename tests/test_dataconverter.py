from nose.tools import assert_equal, assert_greater, assert_greater_equal, assert_less, assert_raises
import os
import numpy as np
import thunderfish.datawriter as dw
import thunderfish.convertdata as cd


def write_data_file(filename):
    samplerate = 44100.0
    duration = 10.0
    channels = 2
    t = np.arange(0.0, duration, 1.0/samplerate)
    data = np.sin(2.0*np.pi*880.0*t) * t/duration
    data = data.reshape((-1, 1))
    for k in range(data.shape[1], channels):
        data = np.hstack((data, data[:,0].reshape((-1, 1))/k))
    encoding = 'PCM_16'
    dw.write_data(filename, data, samplerate, encoding=encoding)


def test_main():
    filename = 'test.npz'
    destfile = 'test2'
    write_data_file(filename)
    assert_raises(SystemExit, cd.main, '-h')
    assert_raises(SystemExit, cd.main, '--help')
    assert_raises(SystemExit, cd.main, '--version')
    cd.main('-l')
    cd.main('-f', 'pickle', '-l')
    cd.main('-f', 'pickle', '-o', destfile, filename)
    assert_raises(SystemExit, cd.main, '-f', 'xxx', '-o', destfile, filename)
    cd.main('-o', destfile, filename)
    cd.main('-f', 'wav', '-o', destfile, filename)
    cd.main('-e', 'PCM_16', '-o', destfile + '.pkl', filename)
    cd.main('-f', 'pickle', '-e', 'PCM_32', '-o', destfile, '-v', filename)
    cd.main('-f', 'pickle', '-e', 'float', '-o', destfile, filename)
    cd.main('-e', 'float', '-o', destfile + '.npz', filename)
    os.remove(filename)
    os.remove(destfile+'.pkl')
    os.remove(destfile+'.npz')
