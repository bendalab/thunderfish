"""
Functions for loading data from audio files.

data, samplingrate = load_audio(filepath)
Loads the whole file by trying different modules until it succeeds to load the data.

data = AudioLoader(filepath, 60.0)
Create an AudioLoader object that loads chuncks of 60 seconds long data on demand.
data can be used like a read-only numpy array.

list_modules() and available_modules() let you query which audio modules
are installed and available.
For further information and installing instructions of missing modules,
see the documentation of the respective load_*() functions.

For an overview on available python modules see
http://nbviewer.jupyter.org/github/mgeier/python-audio/blob/master/audio-files/index.ipynb
"""
 
import warnings
import os.path
import numpy as np


# probe for available audio modules:
audio_modules = {}

try:
    import wave
    audio_modules['wave'] = True
except ImportError:
    audio_modules['wave'] = False

try:
    import ewave
    audio_modules['ewave'] = True
except ImportError:
    audio_modules['ewave'] = False

try:
    from scipy.io import wavfile
    audio_modules['scipy.io.wavfile'] = True
except ImportError:
    audio_modules['scipy.io.wavfile'] = False

try:
    import soundfile
    audio_modules['soundfile'] = True
except ImportError:
    audio_modules['soundfile'] = False

try:
    import wavefile
    audio_modules['wavefile'] = True
except ImportError:
    audio_modules['wavefile'] = False

try:
    import scikits.audiolab as audiolab
    audio_modules['scikits.audiolab'] = True
except ImportError:
    audio_modules['scikits.audiolab'] = False
        
try:
    import audioread
    audio_modules['audioread'] = True
except ImportError:
    audio_modules['audioread'] = False


def available_modules():
    """Returns:
         mods (list): list of installed audio modules.
    """
    mods = []
    for module, available in audio_modules.items():
        if available:
            mods.append(module)
    return mods


def list_modules():
    """Print list of all modules the audioloader module is bale to use
    and whether they are installed or not.
    """
    for module, available in audio_modules.items():
        if available:
            print('%-16s is     installed' % module)
        else:
            print('%-16s is not installed' % module)


def load_wave(filename, verbose=0):
    """
    Load wav file using the wave module from pythons standard libray.
    
    Documentation:
        https://docs.python.org/2/library/wave.html

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
    if not audio_modules['wave']:
        raise ImportError

    wf = wave.open(filename, 'r')   # 'with' is not supported by wave
    (nchannels, sampwidth, rate, nframes, comptype, compname) = wf.getparams()
    if verbose > 1:
        print('channels       : %d' % nchannels)
        print('bytes          : %d' % sampwidth)
        print('sampling rate  : %g' % rate)
        print('frames         : %d' % nframes)
        print('compression type: %s' % comptype)
        print('compression name: %s' % compname)
    buffer = wf.readframes(nframes)
    format = 'i%d' % sampwidth
    data = np.fromstring(buffer, dtype=format).reshape(-1, nchannels)  # read data
    wf.close()
    data /= 2.0**(sampwidth*8-1)
    if len(data.shape) == 1:
        data = np.reshape(data,(-1, 1))
    return data, float(rate)

    
def load_ewave(filename, verbose=0):
    """
    Load wav file using ewave module.

    Documentation:
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
    if not audio_modules['ewave']:
        raise ImportError

    data = np.array([])
    rate = 0.0
    with ewave.open(filename, 'r') as wf:
        rate = wf.sampling_rate
        buffer = wf.read()
        data = ewave.rescale(buffer, 'float')
    if len(data.shape) == 1:
        data = np.reshape(data,(-1, 1))
    return data, float(rate)

    
def load_wavfile(filename, verbose=0):
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
    if not audio_modules['scipy.io.wavfile']:
        raise ImportError

    if verbose < 2:
        warnings.filterwarnings("ignore")
    rate, data = wavfile.read(filename)
    if verbose < 2:
        warnings.filterwarnings("always")
    data = data / 2.0**15
    if len(data.shape) == 1:
        data = np.reshape(data,(-1, 1))
    return data, float(rate)
        

def load_soundfile(filename, verbose=0):
    """
    Load audio file using pysoundfile (based on libsndfile).

    Documentation:
        http://pysoundfile.readthedocs.org
        http://www.mega-nerd.com/libsndfile
        
    Installation:
        sudo apt-get install libsndfile1 libsndfile1-dev libffi-dev
        sudo pip install pysoundfile

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
    if not audio_modules['soundfile']:
        raise ImportError

    data = np.array([])
    rate = 0.0
    with soundfile.SoundFile(filename, 'r') as sf:
        rate = sf.samplerate
        data = sf.read(always_2d=True)
    return data, float(rate)


def load_wavefile(filename, verbose=0):
    """
    Load audio file using wavefile (based on libsndfile).

    Documentation:
        https://github.com/vokimon/python-wavefile

    Installation:
        sudo pip install wavefile

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
    if not audio_modules['wavefile']:
        raise ImportError

    rate, data = wavefile.load(filename)
    return data.T, float(rate)


def load_audiolab(filename, verbose=0):
    """
    Load audio file using scikits.audiolab (based on libsndfile).

    Documentation:
        http://cournape.github.io/audiolab/
        
    Installation:
        sudo pip install scikits.audiolab

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
    if not audio_modules['scikits.audiolab']:
        raise ImportError

    af = audiolab.Sndfile(filename)
    rate = af.samplerate
    data = af.read_frames(af.nframes)
    if len(data.shape) == 1:
        data = np.reshape(data,(-1, 1))
    return data, float(rate)


def load_audioread(filename, verbose=0):
    """
    Load audio file using audioread.

    Documentation:
        https://github.com/sampsyo/audioread

    Installation:
        sudo apt-get install libav-tools python-audioread
        
    audioread is not available in python x,y.

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
    if not audio_modules['audioread']:
        raise ImportError
    
    data = np.array([])
    rate = 0.0
    with audioread.audio_open(filename) as af:
        rate = af.samplerate
        data = np.zeros((int(np.ceil(af.samplerate*af.duration)), af.channels),
                        dtype="<i2")
        index = 0
        for buffer in af:
            fulldata = np.fromstring(buffer, dtype='<i2').reshape(-1, af.channels)
            n = fulldata.shape[0]
            if index+n > len( data ):
                n = len( data ) - index
            if n > 0:
                data[index:index+n,:] = fulldata[:n,:]
                index += n
            else:
                break
    return data/2.0**15, float(rate)


# list of implemented load functions:
audio_loader = [
    ['wave', load_wave],
    ['scipy.io.wavfile', load_wavfile],
    ['soundfile', load_soundfile],
    ['audioread', load_audioread],
    ['scikits.audiolab', load_audiolab],
    ['ewave', load_ewave],
    ['wavefile', load_wavefile]
    ]

def load_audio(filepath, verbose=0):
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
    # check values:
    rate = 0.0
    data = np.zeros((0, 0))
    if len(filepath) == 0:
        warnings.warn('input argument filepath is empty string!')
        return data, rate
    if not os.path.isfile(filepath):
        warnings.warn('input argument filepath=%s does not indicate an existing file!' % filepath)
        return data, rate
    if os.path.getsize(filepath) <= 0:
        warnings.warn('load_audio(): input argument filepath=%s indicates file of size 0!' % filepath)
        return data, rate

    # load an audio file by trying various modules:
    for lib, load_file in audio_loader:
        if not audio_modules[lib]:
            continue
        try:
            data, rate = load_file(filepath, verbose)
            if len(data) > 0:
                if verbose > 0:
                    print('loaded data from file "%s" using %s:' %
                          (filepath, lib))
                    print('  sampling rate: %g Hz' % rate)
                    print('  data values : %d' % len(data))
                break
        except:
            warnings.warn('failed to load data from file "%s" with %s' % (filepath, lib))
    return data, rate


class AudioLoader:
    """Buffered reading of audio data.
    """
    
    def __init__(self, filename=None, buffersize=10.0, backsize=0.0, verbose=0):
        self.sf = None
        self.samplerate = 0.0
        self.channels = 0
        self.frames = 0
        self.shape = (0, 0)
        self.offset = 0
        self.buffersize = 0
        self.backsize = 0
        self.buffer = np.zeros((0,0))
        self.verbose = verbose
        if filename is not None:
            self.open(filename, buffersize, backsize)

    def __del__(self):
        self.close()

    def __enter__(self):
        return self
        
    def __exit__(self, type, value, tb):
        self.__del__()
        return value
        
    def __len__(self):
        return self.frames

    def __getitem__(self, key):
        if hasattr(key, '__len__'):
            index = key[0]
        else:
            index = key
        if isinstance(index, slice):
            start = index.start
            stop = index.stop
            step = index.step
            if start is None:
                start=0
            if start < 0:
                start += len(self)
            if stop is None:
                stop = len(self)
            if stop < 0:
                stop += len(self)
            if stop > self.frames:
                stop = self.frames
            if step is None:
                step = 1
            self._update_buffer(start, stop)
            newindex = slice(start-self.offset, stop-self.offset, step)
        else:
            if index > self.frames:
                raise IndexError
            self._update_buffer(index, index+1)
            newindex = start-self.offset
        if hasattr(key, '__len__'):
            newkey = (newindex,) + key[1:]
            return self.buffer[newkey]
        else:
            return self.buffer[newindex]
        return 0

    def _read_indices(self, start, stop):
        """ Compute position and size for next read from file. """ 
        if self.offset < 0:
            self.offset = 0
        offset = start
        size = stop-start
        if size < self.buffersize:
            back = self.backsize
            if self.buffersize - size < back:
                back = self.buffersize - size
            offset -= back
            size = self.buffersize
            if offset < 0:
                offset = 0
            if offset + size > self.frames:
                offset = self.frames - size
                if offset < 0:
                    offset = 0
                    size = self.frames - offset
        return offset, size

    # wave interface:        
    def _open_wave(self, filename, buffersize=10.0, backsize=0.0):
        """Open audio file for reading using the wave module.

        Args:
          filename (string): name of the file
          buffersize (float): size of internal buffer in seconds
          backsize (float): part of the buffer to be loaded before the requested start index in seconds
        """
        print('_open_wave(filename)')
        if not audio_modules['wave']:
            self.samplerate = 0.0
            self.channels = 0
            self.frames = 0
            self.shape = (0, 0)
            self.offset = 0
            raise ImportError
        if self.sf is not None:
            self._close_wave()
        self.sf = wave.open(filename, 'r')
        self.samplerate = self.sf.getframerate()
        self.format = 'i%d' % self.sf.getsampwidth()
        self.factor = 1.0/2.0**(self.sf.getsampwidth()*8-1)
        self.channels = self.sf.getnchannels()
        self.frames = self.sf.getnframes()
        self.shape = (self.frames, self.channels)
        self.buffersize = int(buffersize*self.samplerate)
        self.backsize = int(backsize*self.samplerate)
        self.offset = -self.buffersize  # we have no data loaded yet
        self.close = self._close_wave
        self._update_buffer = self._update_buffer_wave

    def _close_wave(self):
        """ Close the audio file using the wave module. """
        if self.sf is not None:
            self.sf.close()
            self.sf = None

    def _update_buffer_wave(self, start, stop):
        """Make sure that the buffer contains the data between
        start and stop using the wave module.
        """
        if start < self.offset or stop > self.offset + self.buffersize:
            offset, size = self._read_indices(start, stop)
            # backwards:
            if offset < self.offset :
                self.sf.rewind()
                self.offset = 0
            # read forward to requested position:
            while self.offset+self.buffersize < offset:
                #print('readframes')
                self.sf.readframes(self.buffersize)
                self.offset += self.buffersize
            if self.offset < offset:
                #print('readframes')
                self.sf.readframes(offset - self.offset)
                self.offset += offset - self.offset
            # read buffer:
            #print 'read', size, 'frames at', self.offset
            buffer = self.sf.readframes(size)
            buffer = np.fromstring(buffer, dtype=self.format).reshape((-1, self.channels))
            self.buffer = buffer * self.factor
            self.offset = offset
            if self.verbose > 0:
                print('loaded %d frames from %d up to %d'
                      % (self.buffer.shape[0], self.offset, self.offset+self.buffer.shape[0]))
        

    # ewave interface:        
    def _open_ewave(self, filename, buffersize=10.0, backsize=0.0):
        """Open audio file for reading using the ewave module.

        Args:
          filename (string): name of the file
          buffersize (float): size of internal buffer in seconds
          backsize (float): part of the buffer to be loaded before the requested start index in seconds
        """
        print('_open_ewave(filename)')
        if not audio_modules['ewave']:
            self.samplerate = 0.0
            self.channels = 0
            self.frames = 0
            self.shape = (0, 0)
            self.offset = 0
            raise ImportError
        if self.sf is not None:
            self._close_ewave()
        self.sf = ewave.open(filename, 'r')
        self.samplerate = self.sf.sampling_rate
        self.channels = self.sf.nchannels
        self.frames = self.sf.nframes
        self.shape = (self.frames, self.channels)
        self.buffersize = int(buffersize*self.samplerate)
        self.backsize = int(backsize*self.samplerate)
        self.offset = -self.buffersize  # we have no data loaded yet
        self.close = self._close_ewave
        self._update_buffer = self._update_buffer_ewave

    def _close_ewave(self):
        """ Close the audio file using the ewave module. """
        if self.sf is not None:
            del self.sf
            self.sf = None

    def _update_buffer_ewave(self, start, stop):
        """Make sure that the buffer contains the data between
        start and stop using the ewave module.
        """
        if start < self.offset or stop > self.offset + self.buffersize:
            offset, size = self._read_indices(start, stop)
            # read buffer:
            buffer = self.sf.read(frames=size, offset=offset, memmap='r')
            self.buffer = ewave.rescale(buffer, 'float')
            if len(self.buffer.shape) == 1:
                self.buffer = np.reshape(self.buffer,(-1, 1))
            self.offset = offset
            if self.verbose > 0:
                print('loaded %d frames from %d up to %d'
                      % (self.buffer.shape[0], self.offset, self.offset+self.buffer.shape[0]))

            
    # pysound file interface:        
    def _open_soundfile(self, filename, buffersize=10.0, backsize=0.0):
        """Open audio file for reading using the pysoundfile module.

        Args:
          filename (string): name of the file
          buffersize (float): size of internal buffer in seconds
          backsize (float): part of the buffer to be loaded before the requested start index in seconds
        """
        print('_open_soundfile(filename)')
        if not audio_modules['soundfile']:
            self.samplerate = 0.0
            self.channels = 0
            self.frames = 0
            self.shape = (0, 0)
            self.offset = 0
            raise ImportError
        if self.sf is not None:
            self._close_soundfile()
        self.sf = soundfile.SoundFile(filename, 'r')
        self.samplerate = self.sf.samplerate
        self.channels = self.sf.channels
        self.frames = 0
        if self.sf.seekable():
            self.frames = self.sf.seek(0, soundfile.SEEK_END)
            self.sf.seek(0, soundfile.SEEK_SET)
        self.shape = (self.frames, self.channels)
        self.buffersize = int(buffersize*self.samplerate)
        self.backsize = int(backsize*self.samplerate)
        self.offset = -self.buffersize  # we have no data loaded yet
        self.close = self._close_soundfile
        self._update_buffer = self._update_buffer_soundfile

    def _close_soundfile(self):
        """ Close the audio file using the pysoundfile module. """
        if self.sf is not None:
            self.sf.close()
            self.sf = None

    def _update_buffer_soundfile(self, start, stop):
        """Make sure that the buffer contains the data between
        start and stop using the pysoundfile module.
        """
        if start < self.offset or stop > self.offset + self.buffersize:
            offset, size = self._read_indices(start, stop)
            self.sf.seek(offset, soundfile.SEEK_SET)
            self.buffer = self.sf.read(size, always_2d=True)
            self.offset = offset
            if self.verbose > 0:
                print('loaded %d frames from %d up to %d'
                      % (self.buffer.shape[0], self.offset, self.offset+self.buffer.shape[0]))


    def open(self, filename, buffersize=10.0, backsize=0.0):
        """Open audio file for reading.

        Args:
          filename (string): name of the file
          buffersize (float): size of internal buffer in seconds
          backsize (float): part of the buffer to be loaded before the requested start index in seconds
        """
        # list of implemented open functions:
        audio_open = [
            ['ewave', self._open_ewave],
            ['wave', self._open_wave],
            ['soundfile', self._open_soundfile]
            #['audioread', self._open_audioread],
            #['scikits.audiolab', self._open_audiolab],
            #['wavefile', self._open_wavefile]
            ]
        # open an audio file by trying various modules:
        for lib, open_file in audio_open:
            if not audio_modules[lib]:
                continue
            try:
                open_file(filename, buffersize, backsize)
                if self.verbose > 0:
                    print('opened audio file "%s" using %s:' %
                          (filepath, lib))
                    print('  sampling rate: %g Hz' % self.samplerate)
                    print('  data values : %d' % self.frames)
                break
            except:
                warnings.warn('failed to open audio file "%s" with %s' % (filepath, lib))


open = AudioLoader
                

if __name__ == "__main__":
    import sys
    print("Checking audioloader module ...")
    print('')
    list_modules()
    print('')
    print('available modules:')
    print('  %s' % '\n  '.join(available_modules()))
    
    filepath = sys.argv[-1]
    print('')
    print("try load_audio:")
    data, rate = load_audio(filepath, 2)
    print('')
    for lib, load_file in audio_loader:
        if not audio_modules[lib]:
            continue
        try:
            data, rate = load_file(filepath, 1)
            print('loaded data from file "%s" with %s' %
                    (filepath, lib))
        except:
            print('failed to load data from file "%s" with %s' %
                  (filepath, lib))
    print('')

    print("try AudioLoader:")
    print('')
    with open(filepath, 8.0, 3.0, 1) as data:
        print('samplerate: %g' % data.samplerate)
        print('channels: %d %d' % (data.channels, data.shape[1]))
        print('frames: %d %d' % (len(data), data.shape[0]))
        nframes = int(3.0*data.samplerate)
        # forward:
        for i in range(0, len(data), nframes):
            print('\nforward %d-%d' % (i, i+nframes))
            x = data[i:i+nframes,0]
        # and backwards:
        for i in reversed(range(0, len(data), nframes)):
            print('\nbackward %d-%d' % (i, i+nframes))
            x = data[i:i+nframes,0]
