from nose.tools import assert_true, with_setup
import os
import numpy as np
import thunderfish.dataloader as dl


filename = 'testd'

    
def write_relacs(path, data, samplerate):
    os.mkdir(path)
    for c in range(data.shape[1]):
        df = open(os.path.join(path, 'trace-%d.raw' % (c+1)), 'wb')
        df.write(np.array(data[:, c], dtype=np.float32).tostring())
        df.close()
    df = open(os.path.join(path, 'stimuli.dat'), 'w')
    df.write('# analog input traces:\n')
    for c in range(data.shape[1]):
        df.write('#     identifier%d      : V-%d\n' % (c+1, c+1))
        df.write('#     data file%d       : trace-%d.raw\n' % (c+1, c+1))
        df.write('#     sample interval%d : %.4fms\n' % (c+1, 1000.0/samplerate))
        df.write('#     sampling rate%d   : %.2fHz\n' % (c+1, samplerate))
        df.write('#     unit%d            : V\n' % (c+1))
    df.write('# event lists:\n')
    df.write('#      event file1: stimulus-events.dat\n')
    df.write('#      event file2: restart-events.dat\n')
    df.write('#      event file3: recording-events.dat\n') 
    df.close()


def write_fishgrid(path, data, samplerate):
    os.mkdir(path)
    df = open(os.path.join(path, 'traces-grid1.raw'), 'wb')
    df.write(np.array(data, dtype=np.float32).tostring())
    df.close()
    df = open(os.path.join(path, 'fishgrid.cfg'), 'w')
    df.write('*FishGrid\n')
    df.write('  Grid &1\n')
    df.write('     Used1      : true\n')
    df.write('     Columns    : 2\n')
    df.write('     Rows       : %d\n' % (data.shape[1]//2))
    df.write('  Hardware Settings\n')
    df.write('    DAQ board\n')
    df.write('      AISampleRate: %.3fkHz\n' % (0.001*samplerate))
    df.write('      AIMaxVolt   : 10.0mV\n')
    df.write('    Amplifier:\n')
    df.write('      AmplName: "16-channel-EPM-module"\n')
    df.close()

    
def remove_files():
    path = filename
    if os.path.isdir(path):
        for root, dirs, files in os.walk(path, topdown=False):
            for name in files:
                os.remove(os.path.join(root, name))
        os.rmdir(os.path.join(path))
        

def check_reading(filename, data):
    tolerance = 2.0**(-15)

        # load full data:
    full_data, rate, unit = dl.load_data(filename, -1)
    assert_true(np.all(np.abs(data[:-2, :] - full_data)<tolerance), 'full load failed')

    # load on demand:
    data = dl.DataLoader(filename, -1, 10.0, 2.0)

    nframes = int(1.5*data.samplerate)
    # check access:
    ntests = 1000
    step = int(len(data)/ntests)
    success = -1
    print('  check random single frame access...')
    for inx in np.random.randint(0, len(full_data), ntests):
        if np.any(np.abs(full_data[inx] - data[inx]) > tolerance):
            success = inx
            break
    assert_true(success < 0, 'single random frame access failed at index %d' % (success))
    print('  check random frame slice access...')
    for inx in np.random.randint(0, len(full_data)-nframes, ntests):
        if np.any(np.abs(full_data[inx:inx+nframes] - data[inx:inx+nframes]) > tolerance):
            success = inx
            break
    assert_true(success < 0, 'random frame slice access failed at index %d' % (success))
    print('  check forward slice access...')
    for inx in range(0, len(full_data)-nframes, step):
        if np.any(np.abs(full_data[inx:inx+nframes] - data[inx:inx+nframes]) > tolerance):
            success = inx
            break
    assert_true(success < 0, 'frame slice access forward failed at index %d' % (success))
    print('  check backward slice access...')
    for inx in range(len(full_data)-nframes, 0, -step):
        if np.any(np.abs(full_data[inx:inx+nframes] - data[inx:inx+nframes]) > tolerance):
            success = inx
            break
    assert_true(success < 0, 'frame slice access backward failed at index %d' % (success))

    data.close()
    

@with_setup(None, remove_files)
def test_dataloader_relacs():
    # generate data:
    samplerate = 44100.0
    duration = 100.0
    channels = 4
    t = np.arange(int(duration*samplerate))/samplerate
    data = np.sin(2.0*np.pi*880.0*t) * t/duration
    data = data.reshape((-1, 1))
    for k in range(data.shape[1], channels):
        data = np.hstack((data, data[:,0].reshape((-1, 1))/k))

    write_relacs(filename, data, samplerate)
    check_reading(filename, data)


@with_setup(None, remove_files)
def test_dataloader_fishgrid():
    # generate data:
    samplerate = 44100.0
    duration = 100.0
    channels = 4
    t = np.arange(int(duration*samplerate))/samplerate
    data = np.sin(2.0*np.pi*880.0*t) * t/duration
    data = data.reshape((-1, 1))
    for k in range(data.shape[1], channels):
        data = np.hstack((data, data[:,0].reshape((-1, 1))/k))

    write_fishgrid(filename, data, samplerate)
    check_reading(filename, data)
