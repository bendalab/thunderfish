import pytest
import os
import shutil
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
    md = dict(Amplifier='Teensy_Amp', Num=42)
    dw.write_numpy(filename, data, samplerate, metadata=md, encoding=encoding)


def test_main():
    filename = 'test.npz'
    filename1 = 'test1.npz'
    destfile = 'test2'
    write_data_file(filename)
    with pytest.raises(SystemExit):
        cd.main('-h')
    with pytest.raises(SystemExit):
        cd.main('--help')
    with pytest.raises(SystemExit):
        cd.main('--version')
    cd.main('-l')
    cd.main('-f', 'npz', '-l')
    cd.main('-f', 'pkl', '-o', destfile, filename)
    with pytest.raises(SystemExit):
        cd.main()
    with pytest.raises(SystemExit):
        cd.main('')
    with pytest.raises(SystemExit):
        cd.main('-f', 'xxx', '-l')
    with pytest.raises(SystemExit):
        cd.main('-f', 'xxx', '-o', destfile, filename)
    with pytest.raises(SystemExit):
        cd.main('-o', 'test.xxx', filename)
    with pytest.raises(SystemExit):
        cd.main('-f', 'xyz123', filename)
    with pytest.raises(SystemExit):
        cd.main(filename)
    with pytest.raises(SystemExit):
        cd.main('-o', filename, filename)
    with pytest.raises(SystemExit):
        cd.main('-o', destfile, filename)
    with pytest.raises(SystemExit):
        cd.main('-o', destfile)
    cd.main('-a', 'Artist=John Doe', '-f', 'pkl', '-o', destfile, filename)
    cd.main('-r', 'Amplifier', '-o', destfile + '.wav', filename)
    os.remove(destfile + '.wav')
    cd.main('-u', '-f', 'pkl', '-o', destfile, filename)
    cd.main('-u', '0.8', '-f', 'pkl', '-o', destfile, filename)
    cd.main('-U', '0.8', '-f', 'pkl', '-o', destfile, filename)
    cd.main('-s', '0.1', '-f', 'pkl', '-o', destfile, filename)
    cd.main('-f', 'pkl', '-o', destfile, filename)
    cd.main('-e', 'PCM_32', '-o', destfile + '.pkl', filename)
    cd.main('-f', 'pkl', '-e', 'PCM_32', '-o', destfile, '-v', filename)
    destfile += '.pkl'
    write_data_file(filename1, 4)
    cd.main('-c', '1', '-o', destfile, filename1)
    cd.main('-c', '0-2', '-o', destfile, filename1)
    cd.main('-c', '0-1,3', '-o', destfile, filename1)
    with pytest.raises(SystemExit):
        cd.main('-o', destfile, filename, filename1)
    write_data_file(filename1, 2, 20000)
    with pytest.raises(SystemExit):
        cd.main('-o', destfile, filename, filename1)
    write_data_file(filename1)
    with pytest.raises(SystemExit):
        cd.main('-n', '1', '-o', destfile[:-4], filename, filename1)
    cd.main('-n', '1', '-f', 'wav', '-o', destfile[:-4], filename, filename1)
    shutil.rmtree(destfile[:-4])
    cd.main('-vv', '-o', destfile, filename, filename1)
    xdata, xrate, xunit, xamax = dl.load_data(filename)
    n = len(xdata)
    xdata, xrate, xunit, xamax = dl.load_data(filename1)
    n += len(xdata)
    xdata, xrate, xunit, xamax = dl.load_data(destfile)
    assert len(xdata) == n, 'len of merged files'
    cd.main('-d', '4', '-o', destfile, filename)
    xdata, xrate, xunit, xamax = dl.load_data(filename)
    ydata, yrate, yunit, xamax = dl.load_data(destfile)
    assert len(ydata) == len(xdata)//4, 'decimation data'
    assert yrate*4 == xrate, 'decimation rate'
    cd.main('-o', 'test{Num}.npz', filename)
    os.remove('test42.npz')
    os.remove(filename)
    os.remove(filename1)
    os.remove(destfile)
