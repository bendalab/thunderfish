import os.path
import numpy as np

# see http://nbviewer.jupyter.org/github/mgeier/python-audio/blob/master/audio-files/index.ipynb
# for an overview on available python modules
    
def load_wave(filename, channel=0, verbose=0) :
    """
    Load wav file using wave module (from pythons standard libray).
    Documentation: https://docs.python.org/2/library/wave.html

    Args:
        filepath (string): the full path and name of the file to load
        channel (int): the single channel to be returned
        verbose (int): if >0 show detailed error/warning messages
                       if 2 print information about soundfile

    Returns:
        freq (float): the sampling rate of the data in Hz
        data (array): the data trace as a 1-D numpy array
        unit (string): the unit of the data

    Exceptions:
        ImportError: if the wave module is not installed
        *: if loading of the data failed
    """
    try:
        import wave
    except ImportError:
        print 'python module "wave" is not installed.'
        raise ImportError

    data = np.array([])
    freq = 0.0
    wf = wave.open(filename, 'r')   # 'with' is not supported by wave
    (nchannels, sampwidth, freq, nframes, comptype, compname) = wf.getparams()
    if verbose > 1 :
        print('channels        : %d' % nchannels)
        print('bytes           : %d' % sampwidth)
        print('sampling rate   : %g' % freq)
        print('frames          : %d' % nframes)
        print('compression type: %s' % comptype)
        print('compression name: %s' % compname)
    buffer = wf.readframes(nframes)
    format = 'i%d' % sampwidth
    data = np.fromstring(buffer, dtype=format).reshape(-1, nchannels)  # read data
    wf.close()
    if len(data.shape) == 1 :
        if channel >= 1 :
            print('number of channels in file %s is 1, but requested channel %d' %
                  (filename, channel))
        return freq, data/2.0**(sampwidth*8-1), 'a.u.'
    else :
        tracen = data.shape[1]
        if channel >= tracen :
            print('number of channels in file %s is %d, but requested channel %d' %
                  (filename, tracen, channel))
            channel = tracen-1
        return freq, data[:,channel]/2.0**(sampwidth*8-1), 'a.u.'

    
def load_ewave(filename, channel=0, verbose=0) :
    """
    Load wav file using ewave module.
    https://github.com/melizalab/py-ewave
    Installation:
    git clone https://github.com/melizalab/py-ewave
    python setup.py build
    sudo python setup.py install

    Args:
        filepath (string): the full path and name of the file to load
        channel (int): the single channel to be returned
        verbose (int): if >0 show detailed error/warning messages (not used)

    Returns:
        freq (float): the sampling rate of the data in Hz
        data (array): the data trace as a 1-D numpy array
        unit (string): the unit of the data

    Exceptions:
        ImportError: if the ewave module is not installed
        *: if loading of the data failed
    """
    try:
        import ewave
    except ImportError:
        print 'python module "ewave" is not installed.'
        raise ImportError

    data = np.array([])
    freq = 0.0
    with ewave.open(filename, 'r') as wf :
        freq = wf.sampling_rate
        buffer = wf.read()
        data = ewave.rescale(buffer, 'float')
    if len(data.shape) == 1 :
        if channel >= 1 :
            print('number of channels in file %s is 1, but requested channel %d' %
                  (filename, channel))
        return freq, data, 'a.u.'
    else :
        tracen = data.shape[1]
        if channel >= tracen :
            print('number of channels in file %s is %d, but requested channel %d' %
                  (filename, tracen, channel))
            channel = tracen-1
        return freq, data[:,channel], 'a.u.'

    
def load_wavfile(filename, channel=0, verbose=0) :
    """
    Load wav file using scipy.io.wavfile.
    Documentation: http://docs.scipy.org/doc/scipy/reference/io.html
    
    Args:
        filepath (string): the full path and name of the file to load
        channel (int): the single channel to be returned
        verbose (int): if >0 show detailed error/warning messages

    Returns:
        freq (float): the sampling rate of the data in Hz
        data (array): the data trace as a 1-D numpy array
        unit (string): the unit of the data

    Exceptions:
        ImportError: if the scipy.io module is not installed
        *: if loading of the data failed
    """
    try:
        from scipy.io import wavfile
    except ImportError:
        print 'python module "scipy.io" is not installed.'
        raise ImportError

    if verbose < 2 :
        warnings.filterwarnings("ignore")
    freq, data = wavfile.read(filename)
    if verbose < 2 :
        warnings.filterwarnings("always")
    if len(data.shape) == 1 :
        if channel >= 1 :
            print('number of channels in file %s is 1, but requested channel %d' %
                  (filename, channel))
        return freq, data/2.0**15, 'a.u.'
    else :
        tracen = data.shape[1]
        if channel >= tracen :
            print('number of channels in file %s is %d, but requested channel %d' %
                  (filename, tracen, channel))
            channel = tracen-1
        return freq, data[:,channel]/2.0**15, 'a.u.'
        

def load_soundfile(filename, channel=0, verbose=0) :
    """
    Load audio file using pysoundfile (based on libsndfile).

    Installation of pysoundfile:
    sudo apt-get install libffi-dev
    sudo pip install pysoundfile
    For documentation see http://pysoundfile.readthedocs.org .
    Block processing is possible.

    Args:
        filepath (string): the full path and name of the file to load
        channel (int): the single channel to be returned
        verbose (int): if >0 show detailed error/warning messages (not used)

    Returns:
        freq (float): the sampling rate of the data in Hz
        data (array): the data trace as a 1-D numpy array
        unit (string): the unit of the data

    Exceptions:
        ImportError: if the soundfile module is not installed
        *: if loading of the data failed
    """
    try:
        import soundfile
    except ImportError:
        print 'python module "soundfile" is not installed.'
        raise ImportError

    data = np.array([])
    samplerate = 0.0
    with soundfile.SoundFile(filename, 'r') as sf :
        tracen = sf.channels
        if channel >= tracen :
            print('number of channels in file %s is %d, but requested channel %d' %
                  (filename, tracen, channel))
            channel = tracen-1
        samplerate = sf.samplerate
        data = sf.read(always_2d=True)[:,channel]
    return samplerate, data, 'a.u.'


def load_wavefile(filename, channel=0, verbose=0) :
    """
    Load audio file using wavefile (based on libsndfile).

    Installation: sudo pip install wavefile
    Website: https://github.com/vokimon/python-wavefile
    Block processing is possible.

    Args:
        filepath (string): the full path and name of the file to load
        channel (int): the single channel to be returned
        verbose (int): if >0 show detailed error/warning messages (not used)

    Returns:
        freq (float): the sampling rate of the data in Hz
        data (array): the data trace as a 1-D numpy array
        unit (string): the unit of the data

    Exceptions:
        ImportError: if the wavefile module is not installed
        *: if loading of the data failed
    """
    try:
        import wavefile
    except ImportError:
        print 'python module "wavefile" is not installed.'
        raise ImportError

    data = np.array([])
    samplerate, buffer = wavefile.load(filename)
    tracen = buffer.shape[0]
    if channel >= tracen :
        print('number of channels in file %s is %d, but requested channel %d' %
              (filename, tracen, channel))
        channel = tracen-1
    return samplerate, buffer[channel], 'a.u.'


def load_audiolab(filename, channel=0, verbose=0) :
    """
    Load audio file using scikits.audiolab (based on libsndfile).

    Installation: sudo pip install scikits.audiolab
    Website: http://cournape.github.io/audiolab/
    Block processing is possible.

    Args:
        filepath (string): the full path and name of the file to load
        channel (int): the single channel to be returned
        verbose (int): if >0 show detailed error/warning messages (not used)

    Returns:
        freq (float): the sampling rate of the data in Hz
        data (array): the data trace as a 1-D numpy array
        unit (string): the unit of the data

    Exceptions:
        ImportError: if the scikits.audiolab module is not installed
        *: if loading of the data failed
    """
    try:
        import scikits.audiolab as audiolab
    except ImportError:
        print 'python module "scikits.audiolab" is not installed.'
        raise ImportError

    data = np.array([])
    af = audiolab.Sndfile(filename)
    samplerate = af.samplerate
    tracen = af.channels
    if channel >= tracen :
        print('number of channels in file %s is %d, but requested channel %d' %
              (filename, tracen, channel))
        channel = tracen-1
    buffer = af.read_frames(af.nframes)
    if len(buffer.shape) == 1 :
        return samplerate, buffer, 'a.u.'
    else :
        return samplerate, buffer[:,channel], 'a.u.'


def load_audio(filename, channel=0, verbose=0) :
    """
    Load audio file using audioread.
    https://github.com/sampsyo/audioread
    This is not available in python x,y.

    Args:
        filepath (string): the full path and name of the file to load
        channel (int): the single channel to be returned
        verbose (int): if >0 show detailed error/warning messages (not used)

    Returns:
        freq (float): the sampling rate of the data in Hz
        data (array): the data trace as a 1-D numpy array
        unit (string): the unit of the data

    Exceptions:
        ImportError: if the audioread module is not installed
        *: if loading of the data failed
    """
    try:
        import audioread
    except ImportError:
        print 'python module "audioread" is not installed.'
        raise ImportError
    
    data = np.array([])
    samplerate = 0.0
    with audioread.audio_open(filename) as af :
        tracen = af.channels
        if channel >= tracen :
            print('number of channels in file %s is %d, but requested channel %d' %
                  (filename, tracen, channel))
            channel = tracen-1
        samplerate = af.samplerate
        data = np.zeros(np.ceil(af.samplerate*af.duration), dtype="<i2")
        index = 0
        for buffer in af:
            fulldata = np.fromstring(buffer, dtype='<i2').reshape(-1, af.channels)
            n = fulldata.shape[0]
            if index+n > len( data ) :
                n = len( data ) - index
            if n > 0 :
                data[index:index+n] = fulldata[:n,channel]
                index += n
            else :
                break
    return samplerate, data/2.0**15, 'a.u.'


def load_pickle(filename, channel=0, verbose=0) :
    """
    Load Joerg's pickle files.

    Args:
        filepath (string): the full path and name of the file to load
        channel (int): the single channel to be returned
        verbose (int): if >0 show detailed error/warning messages (not used)

    Returns:
        freq (float): the sampling rate of the data in Hz
        data (array): the data trace as a 1-D numpy array
        unit (string): the unit of the data
    """
    import pickle
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    time = data['time_trace']
    freq = 1000.0/(time[1]-time[0])
    tracen = data['raw_data'].shape[1]
    if channel >= tracen :
        print('number of channels in file %s is %d, but requested channel %d' %
              (filename, tracen, channel))
        channel = tracen-1
    return freq, data['raw_data'][:,channel], 'mV'

audio_loader = [
    ['wave', load_wave],
    ['scipy.io.wavfile', load_wavfile],
    ['soundfile', load_soundfile],
    ['scikits.audiolab', load_audiolab],
    ['audioread', load_audio],
    ['ewave', load_ewave],
    ['wavefile', load_wavefile]
    ]

def load_data(filepath, channel=0, verbose=0) :
    """
    Call this function to load a single trace of data from a file.
    This function tries different modules to load an audio file.

    Args:
        filepath (string): the full path and name of the file to load
        channel (int): the single channel to be returned
        verbose (int): if >0 show detailed error/warning messages

    Returns:
        freq (float): the sampling rate of the data in Hz
        data (array): the data trace as a 1-D numpy array
        unit (string): the unit of the data
    """
    # check types:
    if not isinstance(filepath, basestring) :
        raise NameError('load_data(): input argument filepath must be a string!')
    if not isinstance(channel, int) :
        raise NameError('load_data(): input argument channel must be an int!')

    # check values:
    freq = 0.0
    data = np.array([])
    unit = ''
    if len(filepath) == 0 :
        print('load_data(): input argument filepath is empty string!')
        return freq, data, unit
    if not os.path.isfile(filepath) :
        print('load_data(): input argument filepath=%s does not indicate an existing file!' % filepath)
        return freq, data, unit
    if os.path.getsize(filepath) <= 0:
        print('load_data(): input argument filepath=%s indicates file of size 0!' % filepath)
        return freq, data, unit
    if channel < 0 :
        print('load_data(): input argument channel=%d is negative!' % channel)
        channel = 0

    # load data:
    ext = filepath.split('.')[-1]
    if ext == 'pkl' :
        freq, data, unit = load_pickle(filepath, channel)
    else :
        # load an audio file by trying various modules:
        for lib, load_file in audio_loader :
            try:
                freq, data, unit = load_file(filepath, channel, verbose)
            except:
                if verbose > 0 :
                    print('failed to load data from file "%s" with %s' %
                          (filepath, lib))
            if len(data) > 0 :
                if verbose > 0 :
                    print('loaded data from file "%s" using %s:' %
                          (filepath, lib))
                    print('  sampling rate: %g Hz' % freq)
                    print('  unit         : %s' % unit)
                    print('  data values  : %d' % len(data))
                break
    return freq, data, unit


if __name__ == "__main__":
    import sys
    print("Checking dataloader module ...")
    filepath = sys.argv[-1]
    channel = 0
    print
    print("try load_data:")
    freq, data, unit = load_data(filepath, channel, 2)
    print
    for lib, load_file in audio_loader :
        try:
            freq, data, unit = load_file(filepath, channel, 1)
            print('loaded data from file "%s" with %s' %
                    (filepath, lib))
        except:
            print('failed to load data from file "%s" with %s' %
                    (filepath, lib))
