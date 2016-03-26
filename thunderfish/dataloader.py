import numpy as np

###############################################################################
## load data:

def load_pickle( filename, trace=0 ) :
    """
    load Joerg's pickle files
    """
    import pickle
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    time = data['time_trace']
    freq = 1000.0/(time[1]-time[0])
    tracen = data['raw_data'].shape[1]
    if trace >= tracen :
        print 'number of traces in file is', tracen
        quit()
    return freq, data['raw_data'][:,trace], 'mV'
    
def load_wavfile( filename, trace=0 ) :
    """
    load wav file using scipy io.wavfile
    """
    from scipy.io import wavfile
    freq, data = wavfile.read( filename )
    if len( data.shape ) == 1 :
        if trace >= 1 :
            print 'number of traces in file is', 1
            quit()
        return freq, data/2.0**15, ''
    else :
        tracen = data.shape[1]
        if trace >= tracen :
            print 'number of traces in file is', tracen
            quit()
        return freq, data[:,trace]/2.0**15, 'a.u.'

def load_wave( filename, trace=0 ) :
    """
    load wav file using wave module
    """
    try:
        import wave
    except ImportError:
        print 'python module "wave" is not installed.'
        return load_wavfile( filename, trace )

    wf = wave.open( filename, 'r' )
    (nchannels, sampwidth, freq, nframes, comptype, compname) = wf.getparams()
    print nchannels, sampwidth, freq, nframes, comptype, compname
    buffer = wf.readframes( nframes )
    format = 'i%d' % sampwidth
    data = np.fromstring( buffer, dtype=format ).reshape( -1, nchannels )  # read data
    wf.close()
    print data.shape
    if len( data.shape ) == 1 :
        if trace >= 1 :
            print 'number of traces in file is', 1
            quit()
        return freq, data/2.0**(sampwidth*8-1), ''
    else :
        tracen = data.shape[1]
        if trace >= tracen :
            print 'number of traces in file is', tracen
            quit()
        return freq, data[:,trace]/2.0**(sampwidth*8-1), 'a.u.'

    
def load_audio( filename, trace=0 ) :
    """
    load wav file using audioread.
    This is not available in python x,y.
    """
    try:
        import audioread
    except ImportError:
        print 'python module "audioread" is not installed.'
        return load_wave( filename, trace )
    
    data = np.array( [] )
    with audioread.audio_open( filename ) as af :
        tracen = af.channels
        if trace >= tracen :
            print 'number of traces in file is', tracen
            quit()
        data = np.zeros( np.ceil( af.samplerate*af.duration ), dtype="<i2" )
        index = 0
        for buffer in af:
            fulldata = np.fromstring( buffer, dtype='<i2' ).reshape( -1, af.channels )
            n = fulldata.shape[0]
            if index+n > len( data ) :
                n = len( data ) - index
            if n > 0 :
                data[index:index+n] = fulldata[:n,trace]
                index += n
            else :
                break
    return af.samplerate, data/2.0**15, 'a.u.'


def load_data(filename, trace=0) :
    """
    Call this function to load a single trace of data.

    Args:
        filename (string): the full path and name of the file to load
        trace (int): the trace/channel to be returned

    Returns:
        freq (float): the sampling rate of the data in Hz
        data (array): the data trace
        unit (string): the unit of the data
    """
    ext = filename.split( '.' )[-1]
    if ext == 'pkl' :
        freq, data, unit = dl.load_pickle( filepath, channel )
    else :
        #freq, data, unit = dl.load_wavfile( filepath, channel )
        #freq, data, unit = dl.load_wave( filepath, channel )
        freq, data, unit = dl.load_audio( filepath, channel )
    return freq, data, unit
