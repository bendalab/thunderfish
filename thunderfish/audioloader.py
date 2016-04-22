import os.path
import numpy as np

"""
Numerous functions for loading data from audio files.

data, samplingrate = load_audio(filepath, channel)
tries different functions until it succeeds to load the data.

For an overview on available python modules see
http://nbviewer.jupyter.org/github/mgeier/python-audio/blob/master/audio-files/index.ipynb
"""
 
def load_wave(filename, channel=0, verbose=0) :
    """
    Load wav file using wave module (from pythons standard libray).
    Documentation: https://docs.python.org/2/library/wave.html

    Args:
        filepath (string): the full path and name of the file to load
        channel (int): the single channel to be returned (0, 1, 2, ...)
                       if negative, then all channels are returned
        verbose (int): if >0 show detailed error/warning messages
                       if 2 print information about soundfile

    Returns:
        data (array): channel>=0: the single data trace as an 1-D numpy array
                      channel<0: all data traces as an 2-D numpy array
        freq (float): the sampling rate of the data in Hz

    Exceptions:
        ImportError: if the wave module is not installed
        *: if loading of the data failed
    """
    try:
        import wave
    except ImportError:
        print 'python module "wave" is not installed.'
        raise ImportError

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
    data /= 2.0**(sampwidth*8-1)
    channels = 1
    if len(data.shape) > 1 :
        channels = data.shape[1]
    if channel >= channels :
        print('number of channels in file %s is %d, but requested channel %d' %
              (filename, channels, channel))
        channel = channels-1
    if channels == 1 :
        data = np.reshape(data,(-1, 1))
    if channel < 0 :
        return data, freq
    else :
        return data[:,channel], freq

    
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
        channel (int): the single channel to be returned (0, 1, 2, ...)
                       if negative, then all channels are returned
        verbose (int): if >0 show detailed error/warning messages (not used)

    Returns:
        data (array): channel>=0: the single data trace as an 1-D numpy array
                      channel<0: all data traces as an 2-D numpy array
        freq (float): the sampling rate of the data in Hz

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
    channels = 1
    if len(data.shape) > 1 :
        channels = data.shape[1]
    if channel >= channels :
        print('number of channels in file %s is %d, but requested channel %d' %
              (filename, channels, channel))
        channel = channels-1
    if channels == 1 :
        data = np.reshape(data,(-1, 1))
    if channel < 0 :
        return data, freq
    else :
        return data[:,channel], freq

    
def load_wavfile(filename, channel=0, verbose=0) :
    """
    Load wav file using scipy.io.wavfile.
    Documentation: http://docs.scipy.org/doc/scipy/reference/io.html
    
    Args:
        filepath (string): the full path and name of the file to load
        channel (int): the single channel to be returned (0, 1, 2, ...)
                       if negative, then all channels are returned
        verbose (int): if >0 show detailed error/warning messages

    Returns:
        data (array): channel>=0: the single data trace as an 1-D numpy array
                      channel<0: all data traces as an 2-D numpy array
        freq (float): the sampling rate of the data in Hz

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
    data /= 2.0**15
    channels = 1
    if len(data.shape) > 1 :
        channels = data.shape[1]
    if channel >= channels :
        print('number of channels in file %s is %d, but requested channel %d' %
              (filename, channels, channel))
        channel = channels-1
    if channels == 1 :
        data = np.reshape(data,(-1, 1))
    if channel < 0 :
        return data, freq
    else :
        return data[:,channel], freq
        

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
        channel (int): the single channel to be returned (0, 1, 2, ...)
                       if negative, then all channels are returned
        verbose (int): if >0 show detailed error/warning messages (not used)

    Returns:
        data (array): channel>=0: the single data trace as an 1-D numpy array
                      channel<0: all data traces as an 2-D numpy array
        freq (float): the sampling rate of the data in Hz

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
    freq = 0.0
    with soundfile.SoundFile(filename, 'r') as sf :
        channels = sf.channels
        if channel >= channels :
            print('number of channels in file %s is %d, but requested channel %d' %
                  (filename, channels, channel))
            channel = channels-1
        freq = sf.samplerate
        data = sf.read(always_2d=True)
    if channel < 0 :
        return data, freq
    else :
        return data[:,channel], freq


def load_wavefile(filename, channel=0, verbose=0) :
    """
    Load audio file using wavefile (based on libsndfile).

    Installation: sudo pip install wavefile
    Website: https://github.com/vokimon/python-wavefile
    Block processing is possible.

    Args:
        filepath (string): the full path and name of the file to load
        channel (int): the single channel to be returned (0, 1, 2, ...)
                       if negative, then all channels are returned
        verbose (int): if >0 show detailed error/warning messages (not used)

    Returns:
        data (array): channel>=0: the single data trace as an 1-D numpy array
                      channel<0: all data traces as an 2-D numpy array
        freq (float): the sampling rate of the data in Hz

    Exceptions:
        ImportError: if the wavefile module is not installed
        *: if loading of the data failed
    """
    try:
        import wavefile
    except ImportError:
        print 'python module "wavefile" is not installed.'
        raise ImportError

    freq, data = wavefile.load(filename)
    channels = data.shape[0]
    if channel >= channels :
        print('number of channels in file %s is %d, but requested channel %d' %
              (filename, channels, channel))
        channel = channels-1
    if channels < 0 :
        return data.T, freq
    else :
        return data[channel], freq


def load_audiolab(filename, channel=0, verbose=0) :
    """
    Load audio file using scikits.audiolab (based on libsndfile).

    Installation: sudo pip install scikits.audiolab
    Website: http://cournape.github.io/audiolab/
    Block processing is possible.

    Args:
        filepath (string): the full path and name of the file to load
        channel (int): the single channel to be returned (0, 1, 2, ...)
                       if negative, then all channels are returned
        verbose (int): if >0 show detailed error/warning messages (not used)

    Returns:
        data (array): channel>=0: the single data trace as an 1-D numpy array
                      channel<0: all data traces as an 2-D numpy array
        freq (float): the sampling rate of the data in Hz

    Exceptions:
        ImportError: if the scikits.audiolab module is not installed
        *: if loading of the data failed
    """
    try:
        import scikits.audiolab as audiolab
    except ImportError:
        print 'python module "scikits.audiolab" is not installed.'
        raise ImportError

    af = audiolab.Sndfile(filename)
    freq = af.samplerate
    channels = af.channels
    if channel >= channels :
        print('number of channels in file %s is %d, but requested channel %d' %
              (filename, channels, channel))
        channel = channels-1
    data = af.read_frames(af.nframes)
    if len(data.shape) == 1 :
        data = np.reshape(data,(-1, 1))
    if channel < 0 :
        return data, freq
    else :
        return data[:,channel], freq


def load_audioread(filename, channel=0, verbose=0) :
    """
    Load audio file using audioread.
    https://github.com/sampsyo/audioread
    This is not available in python x,y.

    Args:
        filepath (string): the full path and name of the file to load
        channel (int): the single channel to be returned (0, 1, 2, ...)
                       if negative, then all channels are returned
        verbose (int): if >0 show detailed error/warning messages (not used)

    Returns:
        data (array): channel>=0: the single data trace as an 1-D numpy array
                      channel<0: all data traces as an 2-D numpy array
        freq (float): the sampling rate of the data in Hz

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
    freq = 0.0
    with audioread.audio_open(filename) as af :
        channels = af.channels
        if channel >= channels :
            print('number of channels in file %s is %d, but requested channel %d' %
                  (filename, channels, channel))
            channel = channels-1
        freq = af.samplerate
        data = np.zeros((np.ceil(af.samplerate*af.duration), channels),
                        dtype="<i2")
        index = 0
        for buffer in af:
            fulldata = np.fromstring(buffer, dtype='<i2').reshape(-1, channels)
            n = fulldata.shape[0]
            if index+n > len( data ) :
                n = len( data ) - index
            if n > 0 :
                data[index:index+n,:] = fulldata[:n,:]
                index += n
            else :
                break
    if channel < 0 :
        return data/2.0**15, freq
    else :
        return data[:,channel]/2.0**15, freq


audio_loader = [
    ['wave', load_wave],
    ['scipy.io.wavfile', load_wavfile],
    ['soundfile', load_soundfile],
    ['audioread', load_audioread],
    ['scikits.audiolab', load_audiolab],
    ['ewave', load_ewave],
    ['wavefile', load_wavefile]
    ]

def load_audio(filepath, channel=0, verbose=0) :
    """
    Call this function to load a single channel of audio data from a file.
    This function tries different python modules to load the audio file.

    Args:
        filepath (string): the full path and name of the file to load
        channel (int): the single channel to be returned (0, 1, 2, ...)
                       if negative, then all channels are returned
        verbose (int): if >0 show detailed error/warning messages

    Returns:
        data (array): channel>=0: the single data trace as an 1-D numpy array
                      channel<0: all data traces as an 2-D numpy array
        freq (float): the sampling rate of the data in Hz
    """
    # check types:
    if not isinstance(filepath, basestring) :
        raise NameError('load_audio(): input argument filepath must be a string!')
    if not isinstance(channel, int) :
        raise NameError('load_audio(): input argument channel must be an int!')

    # check values:
    freq = 0.0
    data = np.array([])
    if len(filepath) == 0 :
        print('load_audio(): input argument filepath is empty string!')
        return data, freq
    if not os.path.isfile(filepath) :
        print('load_audio(): input argument filepath=%s does not indicate an existing file!' % filepath)
        return data, freq
    if os.path.getsize(filepath) <= 0:
        print('load_audio(): input argument filepath=%s indicates file of size 0!' % filepath)
        return data, freq

    # load data:
    # load an audio file by trying various modules:
    for lib, load_file in audio_loader :
        try:
            data, freq = load_file(filepath, channel, verbose)
            if len(data) > 0 :
                if verbose > 0 :
                    print('loaded data from file "%s" using %s:' %
                          (filepath, lib))
                    print('  sampling rate: %g Hz' % freq)
                    print('  data values  : %d' % len(data))
                break
        except:
            if verbose > 0 :
                print('failed to load data from file "%s" with %s' %
                      (filepath, lib))
    return data, freq


if __name__ == "__main__":
    import sys
    print("Checking audioloader module ...")
    filepath = sys.argv[-1]
    channel = 0
    print
    print("try load_audio:")
    data, freq = load_audio(filepath, channel, 2)
    print
    for lib, load_file in audio_loader :
        try:
            data, freq = load_file(filepath, channel, 1)
            print('loaded data from file "%s" with %s' %
                    (filepath, lib))
        except:
            print('failed to load data from file "%s" with %s' %
                  (filepath, lib))
