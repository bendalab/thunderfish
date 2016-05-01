import warnings
import os.path
import numpy as np

"""
Numerous functions for loading data from audio files.

data, samplingrate = load_audio(filepath)
tries different functions until it succeeds to load the data.

For an overview on available python modules see
http://nbviewer.jupyter.org/github/mgeier/python-audio/blob/master/audio-files/index.ipynb
"""
 
def load_wave(filename, verbose=0) :
    """
    Load wav file using wave module (from pythons standard libray).
    Documentation: https://docs.python.org/2/library/wave.html

    Args:
        filepath (string): the full path and name of the file to load
        verbose (int): if >0 show detailed error/warning messages
                       if 2 print information about soundfile

    Returns:
        data (array): all data traces as an 2-D numpy array,
                      first dimension is time, second is channel
        rate (float): the sampling rate of the data in Hz

    Exceptions:
        ImportError: if the wave module is not installed
        *: if loading of the data failed
    """
    try:
        import wave
    except ImportError:
        warnings.warn('python module "wave" is not installed.')
        raise ImportError

    wf = wave.open(filename, 'r')   # 'with' is not supported by wave
    (nchannels, sampwidth, rate, nframes, comptype, compname) = wf.getparams()
    if verbose > 1 :
        print('channels        : %d' % nchannels)
        print('bytes           : %d' % sampwidth)
        print('sampling rate   : %g' % rate)
        print('frames          : %d' % nframes)
        print('compression type: %s' % comptype)
        print('compression name: %s' % compname)
    buffer = wf.readframes(nframes)
    format = 'i%d' % sampwidth
    data = np.fromstring(buffer, dtype=format).reshape(-1, nchannels)  # read data
    wf.close()
    data /= 2.0**(sampwidth*8-1)
    if len(data.shape) == 1 :
        data = np.reshape(data,(-1, 1))
    return data, float(rate)

    
def load_ewave(filename, verbose=0) :
    """
    Load wav file using ewave module.
    https://github.com/melizalab/py-ewave
    Installation:
    git clone https://github.com/melizalab/py-ewave
    python setup.py build
    sudo python setup.py install

    Args:
        filepath (string): the full path and name of the file to load
        verbose (int): if >0 show detailed error/warning messages (not used)

    Returns:
        data (array): all data traces as an 2-D numpy array,
                      first dimension is time, second is channel
        rate (float): the sampling rate of the data in Hz

    Exceptions:
        ImportError: if the ewave module is not installed
        *: if loading of the data failed
    """
    try:
        import ewave
    except ImportError:
        warnings.warn('python module "ewave" is not installed.')
        raise ImportError

    data = np.array([])
    rate = 0.0
    with ewave.open(filename, 'r') as wf :
        rate = wf.sampling_rate
        buffer = wf.read()
        data = ewave.rescale(buffer, 'float')
    if len(data.shape) == 1 :
        data = np.reshape(data,(-1, 1))
    return data, float(rate)

    
def load_wavfile(filename, verbose=0) :
    """
    Load wav file using scipy.io.wavfile.
    Documentation: http://docs.scipy.org/doc/scipy/reference/io.html
    
    Args:
        filepath (string): the full path and name of the file to load
        verbose (int): if >0 show detailed error/warning messages

    Returns:
        data (array): all data traces as an 2-D numpy array,
                      first dimension is time, second is channel
        rate (float): the sampling rate of the data in Hz

    Exceptions:
        ImportError: if the scipy.io module is not installed
        *: if loading of the data failed
    """
    try:
        from scipy.io import wavfile
    except ImportError:
        warnings.warn('python module "scipy.io" is not installed.')
        raise ImportError

    if verbose < 2 :
        warnings.filterwarnings("ignore")
    rate, data = wavfile.read(filename)
    if verbose < 2 :
        warnings.filterwarnings("always")
    data /= 2.0**15
    if len(data.shape) == 1 :
        data = np.reshape(data,(-1, 1))
    return data, float(rate)
        

def load_soundfile(filename, verbose=0) :
    """
    Load audio file using pysoundfile (based on libsndfile).

    Installation of pysoundfile:
    sudo apt-get install libffi-dev
    sudo pip install pysoundfile
    For documentation see http://pysoundfile.readthedocs.org .
    Block processing is possible.

    Args:
        filepath (string): the full path and name of the file to load
        verbose (int): if >0 show detailed error/warning messages (not used)

    Returns:
        data (array): all data traces as an 2-D numpy array,
                      first dimension is time, second is channel
        rate (float): the sampling rate of the data in Hz

    Exceptions:
        ImportError: if the soundfile module is not installed
        *: if loading of the data failed
    """
    try:
        import soundfile
    except ImportError:
        warnings.warn('python module "soundfile" is not installed.')
        raise ImportError

    data = np.array([])
    rate = 0.0
    with soundfile.SoundFile(filename, 'r') as sf :
        rate = sf.samplerate
        data = sf.read(always_2d=True)
    return data, float(rate)


def load_wavefile(filename, verbose=0) :
    """
    Load audio file using wavefile (based on libsndfile).

    Installation: sudo pip install wavefile
    Website: https://github.com/vokimon/python-wavefile
    Block processing is possible.

    Args:
        filepath (string): the full path and name of the file to load
        verbose (int): if >0 show detailed error/warning messages (not used)

    Returns:
        data (array): all data traces as an 2-D numpy array,
                      first dimension is time, second is channel
        rate (float): the sampling rate of the data in Hz

    Exceptions:
        ImportError: if the wavefile module is not installed
        *: if loading of the data failed
    """
    try:
        import wavefile
    except ImportError:
        warnings.warn('python module "wavefile" is not installed.')
        raise ImportError

    rate, data = wavefile.load(filename)
    return data.T, float(rate)


def load_audiolab(filename, verbose=0) :
    """
    Load audio file using scikits.audiolab (based on libsndfile).

    Installation: sudo pip install scikits.audiolab
    Website: http://cournape.github.io/audiolab/
    Block processing is possible.

    Args:
        filepath (string): the full path and name of the file to load
        verbose (int): if >0 show detailed error/warning messages (not used)

    Returns:
        data (array): all data traces as an 2-D numpy array,
                      first dimension is time, second is channel
        rate (float): the sampling rate of the data in Hz

    Exceptions:
        ImportError: if the scikits.audiolab module is not installed
        *: if loading of the data failed
    """
    try:
        import scikits.audiolab as audiolab
    except ImportError:
        warnings.warn('python module "scikits.audiolab" is not installed.')
        raise ImportError

    af = audiolab.Sndfile(filename)
    rate = af.samplerate
    data = af.read_frames(af.nframes)
    if len(data.shape) == 1 :
        data = np.reshape(data,(-1, 1))
    return data, float(rate)


def load_audioread(filename, verbose=0) :
    """
    Load audio file using audioread.
    https://github.com/sampsyo/audioread
    This is not available in python x,y.

    Args:
        filepath (string): the full path and name of the file to load
        verbose (int): if >0 show detailed error/warning messages (not used)

    Returns:
        data (array): all data traces as an 2-D numpy array,
                      first dimension is time, second is channel
        rate (float): the sampling rate of the data in Hz

    Exceptions:
        ImportError: if the audioread module is not installed
        *: if loading of the data failed
    """
    try:
        import audioread
    except ImportError:
        warnings.warn('python module "audioread" is not installed.')
        raise ImportError
    
    data = np.array([])
    rate = 0.0
    with audioread.audio_open(filename) as af :
        rate = af.samplerate
        data = np.zeros((np.ceil(af.samplerate*af.duration), af.channels),
                        dtype="<i2")
        index = 0
        for buffer in af:
            fulldata = np.fromstring(buffer, dtype='<i2').reshape(-1, af.channels)
            n = fulldata.shape[0]
            if index+n > len( data ) :
                n = len( data ) - index
            if n > 0 :
                data[index:index+n,:] = fulldata[:n,:]
                index += n
            else :
                break
    return data/2.0**15, float(rate)


audio_loader = [
    ['wave', load_wave],
    ['scipy.io.wavfile', load_wavfile],
    ['soundfile', load_soundfile],
    ['audioread', load_audioread],
    ['scikits.audiolab', load_audiolab],
    ['ewave', load_ewave],
    ['wavefile', load_wavefile]
    ]

def load_audio(filepath, verbose=0) :
    """
    Call this function to load all channels of audio data from a file.
    This function tries different python modules to load the audio file.

    Args:
        filepath (string): the full path and name of the file to load
        verbose (int): if >0 show detailed error/warning messages

    Returns:
        data (array): all data traces as an 2-D numpy array,
                      first dimension is time, second is channel
        rate (float): the sampling rate of the data in Hz
    """
    # check types:
    if not isinstance(filepath, basestring) :
        raise NameError('load_audio(): input argument filepath must be a string!')

    # check values:
    rate = 0.0
    data = np.array([])
    if len(filepath) == 0 :
        warnings.warn('input argument filepath is empty string!')
        return data, rate
    if not os.path.isfile(filepath) :
        warnings.warn('input argument filepath=%s does not indicate an existing file!' % filepath)
        return data, rate
    if os.path.getsize(filepath) <= 0:
        warnings.warn('load_audio(): input argument filepath=%s indicates file of size 0!' % filepath)
        return data, rate

    # load data:
    # load an audio file by trying various modules:
    for lib, load_file in audio_loader :
        try:
            data, rate = load_file(filepath, verbose)
            if len(data) > 0 :
                if verbose > 0 :
                    print('loaded data from file "%s" using %s:' %
                          (filepath, lib))
                    print('  sampling rate: %g Hz' % rate)
                    print('  data values  : %d' % len(data))
                break
        except:
            warnings.warn('failed to load data from file "%s" with %s' % (filepath, lib))
    return data, rate


if __name__ == "__main__":
    import sys
    print("Checking audioloader module ...")
    filepath = sys.argv[-1]
    print('')
    print("try load_audio:")
    data, rate = load_audio(filepath, 2)
    print('')
    for lib, load_file in audio_loader :
        try:
            data, rate = load_file(filepath, 1)
            print('loaded data from file "%s" with %s' %
                    (filepath, lib))
        except:
            print('failed to load data from file "%s" with %s' %
                  (filepath, lib))
