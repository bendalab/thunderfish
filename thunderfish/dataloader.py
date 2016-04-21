import os.path
import numpy as np

    
def load_wavfile(filename, channel=0, verbose=0) :
    """
    Load wav file using scipy io.wavfile.

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
        
    
def load_wave(filename, channel=0) :
    """
    Load wav file using wave module.

    Args:
        filepath (string): the full path and name of the file to load
        channel (int): the single channel to be returned

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
    with wave.open(filename, 'r') as wf :
        (nchannels, sampwidth, freq, nframes, comptype, compname) = wf.getparams()
        print nchannels, sampwidth, freq, nframes, comptype, compname
        buffer = wf.readframes(nframes)
        format = 'i%d' % sampwidth
        data = np.fromstring(buffer, dtype=format).reshape(-1, nchannels)  # read data
    if len(data.shape) == 1 :
        if channel >= 1 :
            print('number of channels in file %s is 1, but requested channel %d' %
                  (filename, channel))
        return freq, data/2.0**(sampwidth*8-1), ''
    else :
        tracen = data.shape[1]
        if channel >= tracen :
            print('number of channels in file %s is %d, but requested channel %d' %
                  (filename, tracen, channel))
            channel = tracen-1
        return freq, data[:,channel]/2.0**(sampwidth*8-1), 'a.u.'

    
def load_audio(filename, channel=0) :
    """
    Load wav file using audioread.
    This is not available in python x,y.

    Args:
        filepath (string): the full path and name of the file to load
        channel (int): the single channel to be returned

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
    with audioread.audio_open(filename) as af :
        tracen = af.channels
        if channel >= tracen :
            print('number of channels in file %s is %d, but requested channel %d' %
                  (filename, tracen, channel))
            channel = tracen-1
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
    return af.samplerate, data/2.0**15, 'a.u.'


def load_pickle(filename, channel=0) :
    """
    Load Joerg's pickle files.

    Args:
        filepath (string): the full path and name of the file to load
        channel (int): the single channel to be returned

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
        print('load_data(): input argument filepath=%s does not indicate an existing file!', filepath)
        return freq, data, unit
    if os.path.getsize(filepath) <= 0:
        print('load_data(): input argument filepath=%s indicates file of size 0!', filepath)
        return freq, data, unit
    if channel < 0 :
        print('load_data(): input argument channel=%d is negative!' % channel)
        channel = 0

    # load data:
    ext = filepath.split('.')[-1]
    if ext == 'pkl' :
        freq, data, unit = load_pickle(filepath, channel)
    else :
        try:
            freq, data, unit = load_wavfile(filepath, channel, verbose)
        except:
            if verbose > 0 :
                print 'failed to load data from file "%s" with scipy.io.wavfile' % filepath
            try:
                freq, data, unit = load_wave(filepath, channel)
            except:
                if verbose > 0 :
                    print 'failed to load data from file "%s" with wave' % filepath
                try:
                    freq, data, unit = load_audio(filepath, channel)
                except:
                    if verbose > 0 :
                        print 'failed to load data from file "%s" with audioread' % filepath
    return freq, data, unit


if __name__ == "__main__":
    import sys
    print("Checking dataloader module ...")
    filepath = sys.argv[-1]
    channel = 0
    freq, data, unit = load_data(filepath, channel, 1)
    print('loaded data with sampling rate %g Hz, unit %s, and %d data values' % (freq, unit, len(data)))
