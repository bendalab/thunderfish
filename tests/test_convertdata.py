from nose.tools import assert_equal, assert_greater, assert_greater_equal, assert_less, assert_raises
import os
import numpy as np
import thunderfish.dataloader as dl
import thunderfish.datawriter as dw
import thunderfish.convertdata as cd


def write_data_file(filename, channels=2, samplerate = 44100):
    duration = 10.0
    t = np.arange(0.0, duration, 1.0/samplerate)
    data = np.sin(2.0*np.pi*880.0*t) * t/duration
    data = data.reshape((-1, 1))
    for k in range(data.shape[1], channels):
        data = np.hstack((data, data[:,0].reshape((-1, 1))/k))
    encoding = 'PCM_16'
    dw.write_numpy(filename, data, samplerate, encoding=encoding)


def test_main():
    filename = 'test.npz'
    filename1 = 'test1.npz'
    destfile = 'test2'
    write_data_file(filename)
    assert_raises(SystemExit, cd.main, '-h')
    assert_raises(SystemExit, cd.main, '--help')
    assert_raises(SystemExit, cd.main, '--version')
    cd.main('-l')
    cd.main('-f', 'npz', '-l')
    cd.main('-f', 'pkl', '-o', destfile, filename)
    cd.main('-u', '-f', 'pkl', '-o', destfile, filename)
    cd.main('-u', '0.8', '-f', 'pkl', '-o', destfile, filename)
    cd.main('-U', '0.8', '-f', 'pkl', '-o', destfile, filename)
    assert_raises(SystemExit, cd.main, 'prog', '-f', 'xxx', '-o', destfile, filename)
    assert_raises(SystemExit, cd.main, '-o', destfile, filename)
    cd.main('-f', 'pkl', '-o', destfile, filename)
    cd.main('-e', 'PCM_32', '-o', destfile + '.pkl', filename)
    cd.main('-f', 'pkl', '-e', 'PCM_32', '-o', destfile, '-v', filename)
    assert_raises(SystemExit, cd.main)
    destfile += '.pkl'
    write_data_file(filename1, 4)
    cd.main('-c', '1', '-o', destfile, filename1)
    cd.main('-c', '0-2', '-o', destfile, filename1)
    cd.main('-c', '0-1,3', '-o', destfile, filename1)
    assert_raises(SystemExit, cd.main, '-o', destfile, filename, filename1)
    write_data_file(filename1, 2, 20000)
    assert_raises(SystemExit, cd.main, '-o', destfile, filename, filename1)
    write_data_file(filename1)
    cd.main('-o', destfile, filename, filename1)
    xdata, xrate, xunit = dl.load_data(filename)
    n = len(xdata)
    xdata, xrate, xunit = dl.load_data(filename1)
    n += len(xdata)
    xdata, xrate, xunit = dl.load_data(destfile)
    assert_equal(len(xdata), n, 'len of merged files')
    os.remove(filename)
    os.remove(filename1)
    os.remove(destfile)
