import pytest
import os
import sys
import numpy as np
import thunderfish.datawriter as dw
import thunderfish.dataloader as dl


relacs_path = 'test_relacs'
fishgrid_path = 'test_fishgrid'


def generate_data():
    samplerate = 44100.0
    duration = 100.0
    channels = 4
    amax = 20.0
    t = np.arange(int(duration*samplerate))/samplerate
    data = 18*np.sin(2.0*np.pi*880.0*t) * t/duration
    data = data.reshape((-1, 1))
    for k in range(data.shape[1], channels):
        data = np.hstack((data, data[:,0].reshape((-1, 1))/k))
    info = dict(Comment='good',
                Recording=dict(Experimenter='John Doe',
                               Temperature='23.8°C'),
                Subject=dict(Species='Apteronotus leptorhynchus',
                             Sex='Female', Size='12cm'),
                Weather='bad')
    return data, samplerate, amax, info


def generate_markers(maxi):
    locs = np.random.randint(10, maxi-10, (5, 2))
    locs = locs[np.argsort(locs[:,0]),:]
    locs[:,1] = np.random.randint(0, 20, len(locs))
    labels = np.zeros((len(locs), 2), dtype=np.object_)
    for i in range(len(labels)):
        labels[i,0] = chr(ord('a') + i % 26)
        labels[i,1] = chr(ord('A') + i % 26)*5
    return locs, labels


def remove_files(path):
    if os.path.isdir(path):
        for root, dirs, files in os.walk(path, topdown=False):
            for name in files:
                os.remove(os.path.join(root, name))
        os.rmdir(os.path.join(path))

        
@pytest.fixture   
def remove_relacs_files():
    yield
    remove_files(relacs_path)

    
@pytest.fixture   
def remove_fishgrid_files():
    yield
    remove_files(fishgrid_path)


def check_reading(filename, data):
    # load full data:
    full_data, rate, unit, rmax = dl.load_data(filename)
    tolerance = rmax*2.0**(-15)
    assert np.all(data.shape == full_data.shape), 'full load failed: shape'
    assert np.all(np.abs(data - full_data)<tolerance), 'full load failed: data'

    # load on demand:
    data = dl.DataLoader(filename, 10.0, 2.0)
    tolerance = data.ampl_max*2.0**(-15)

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
    assert success < 0, 'single random frame access failed at index %d' % (success)
    print('  check random frame slice access...')
    for inx in np.random.randint(0, len(full_data)-nframes, ntests):
        if np.any(np.abs(full_data[inx:inx+nframes] - data[inx:inx+nframes]) > tolerance):
            success = inx
            break
    assert success < 0, 'random frame slice access failed at index %d' % (success)
    print('  check forward slice access...')
    for inx in range(0, len(full_data)-nframes, step):
        if np.any(np.abs(full_data[inx:inx+nframes] - data[inx:inx+nframes]) > tolerance):
            success = inx
            break
    assert success < 0, 'frame slice access forward failed at index %d' % (success)
    print('  check backward slice access...')
    for inx in range(len(full_data)-nframes, 0, -step):
        if np.any(np.abs(full_data[inx:inx+nframes] - data[inx:inx+nframes]) > tolerance):
            success = inx
            break
    assert success < 0, 'frame slice access backward failed at index %d' % (success)

    data.close()

    
def test_container():
    data, samplerate, amax, info = generate_data()
    locs, labels = generate_markers(len(data))
    # pickle:
    for encoding in dw.encodings_pickle():
        filename = dw.write_pickle('test', data, samplerate, amax, 'mV', info,
                                   locs, labels, encoding=encoding)
        check_reading(filename, data)
        md = dl.metadata(filename)
        assert info == md, 'pickle metadata'
        os.remove(filename)
    filename = dw.write_data('test', data, samplerate, amax, 'mV', info,
                             locs, labels, format='pkl')
    full_data, rate, unit, rmax = dl.load_data(filename)
    tolerance = rmax*2.0**(-15)
    assert np.all(np.abs(data - full_data)<tolerance), 'full pickle load failed'
    llocs, llabels = dl.markers(filename)
    assert np.all(locs == llocs), 'pickle same locs'
    assert np.all(labels == llabels), 'pickle same labels'
    with dl.DataLoader(filename) as sf:
        llocs, llabels = sf.markers()
        assert np.all(locs == llocs), 'pickle same locs'
        assert np.all(labels == llabels), 'pickle same labels'
    os.remove(filename)

    # numpy:
    for encoding in dw.encodings_numpy():
        filename = dw.write_numpy('test', data, samplerate, amax, 'mV',
                                  info, locs, labels, encoding=encoding)
        check_reading(filename, data)
        md = dl.metadata(filename)
        assert info == md, 'numpy metadata'
        os.remove(filename)
    filename = dw.write_data('test', data, samplerate, amax, 'mV',
                             info, locs, labels, format='npz')
    full_data, rate, unit, rmax = dl.load_data(filename)
    tolerance = rmax*2.0**(-15)
    assert np.all(np.abs(data - full_data)<tolerance), 'full numpy load failed'
    llocs, llabels = dl.markers(filename)
    assert np.all(locs == llocs), 'numpy same locs'
    assert np.all(labels == llabels), 'numpy same labels'
    with dl.DataLoader(filename) as sf:
        llocs, llabels = sf.markers()
        assert np.all(locs == llocs), 'numpy same locs'
        assert np.all(labels == llabels), 'numpy same labels'
    os.remove(filename)

    # mat:
    for encoding in dw.encodings_mat():
        filename = dw.write_mat('test', data, samplerate, amax, 'mV', info,
                                locs, labels, encoding=encoding)
        check_reading(filename, data)
        md = dl.metadata(filename)
        assert info == md, 'mat metadata'
        os.remove(filename)
    filename = dw.write_data('test', data, samplerate, amax, 'mV',
                             info, locs, labels, format='mat')
    full_data, rate, unit, rmax = dl.load_data(filename)
    tolerance = rmax*2.0**(-15)
    assert np.all(np.abs(data - full_data)<tolerance), 'full mat load failed'
    llocs, llabels = dl.markers(filename)
    assert np.all(locs == llocs), 'mat same locs'
    assert np.all(labels == llabels), 'mat same labels'
    with dl.DataLoader(filename) as sf:
        llocs, llabels = sf.markers()
        assert np.all(locs == llocs), 'mat same locs'
        assert np.all(labels == llabels), 'mat same labels'
    os.remove(filename)
    
    
def test_relacs(remove_relacs_files):
    data, samplerate, amax, info = generate_data()
    dw.write_metadata_text(sys.stdout, info)
    dw.write_relacs(relacs_path, data, samplerate, amax, 'mV', metadata=info)
    dl.metadata_relacs(relacs_path + '/info.dat')
    check_reading(relacs_path, data)
    remove_files(relacs_path)
    dw.write_relacs(relacs_path, data[:,0], samplerate, amax, 'mV',
                    metadata=info)
    check_reading(relacs_path, data[:,:1])


def test_fishgrid(remove_fishgrid_files):
    data, samplerate, amax, info = generate_data()
    locs, labels = generate_markers(len(data))
    dw.write_fishgrid(fishgrid_path, data, samplerate, amax, 'mV',
                      metadata=info, locs=locs, labels=labels)
    check_reading(fishgrid_path, data)
    llocs, llabels = dl.markers(fishgrid_path)
    assert np.all(locs == llocs), 'fishgrid same locs'
    assert np.all(labels == llabels), 'fishgrid same labels'
    with dl.DataLoader(fishgrid_path) as sf:
        llocs, llabels = sf.markers()
        assert np.all(locs == llocs), 'fishgrid same locs'
        assert np.all(labels == llabels), 'fishgrid same labels'
    remove_files(fishgrid_path)
    dw.write_fishgrid(fishgrid_path, data[:,0], samplerate, amax, 'mV',
                      metadata=info)
    check_reading(fishgrid_path, data[:,:1])

    
def test_audioio():
    data, samplerate, amax, info = generate_data()
    filename = dw.write_audioio('test.wav', data, samplerate, amax, 'mV',
                                metadata=info)
    full_data, rate, unit, rmax = dl.load_data(filename)
    tolerance = rmax*2.0**(-15)
    assert np.all(np.abs(data - full_data)<tolerance), 'full audio load failed'
    os.remove(filename)

    info['gain'] = f'{amax:g}mV'
    filename = dw.write_audioio('test.wav', data, samplerate, None, None,
                                metadata=info)
    full_data, rate, unit, rmax = dl.load_data(filename)
    assert unit == 'mV'
    check_reading(filename, data)
    os.remove(filename)

    
def test_main(remove_fishgrid_files):
    data, samplerate, amax, info = generate_data()
    filename = dw.write_fishgrid(fishgrid_path, data[:10*int(samplerate)],
                                 samplerate, amax, 'mV', info)
    dl.main(filename)
    dl.main('-p', filename)
    
