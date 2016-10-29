import os.path
import glob
import numpy as np
import audioio as aio


def relacs_samplerate_unit(filename, channel=0):
    """
    Opens the corresponding stimuli.dat file and reads the sampling rate and unit.

    Parameters
    ----------
    filename: string
        path to a relacs data directory, a file in a relacs data directory,
        or a relacs trace-*.raw files.
    channel: int
        the channel (trace) number, if filename does not specify a trace-*.raw file.

    Returns
    -------
    samplerate: float
        the sampling rate in Hertz
    unit: string
        the unit of the trace, can be empty if not found

    Raises
    ------
    IOError/FileNotFoundError:
        If the stimuli.dat file does not exist.
    ValueError:
        stimuli.dat file does not contain sampling rate.
    """

    trace = channel+1
    relacs_dir = filename
    # check for relacs data directory:
    if not os.path.isdir(filename):
        relacs_dir = os.path.dirname(filename)
        bn = os.path.basename(filename)
        if (len(bn) > 5 and bn[0:5] == 'trace' and bn[-4:] == '.raw'):
            trace = int(bn[6:].replace('.raw', ''))

    # retreive sampling rate and unit from stimuli.dat file:
    samplerate = None
    unit = ""
    stimuli_file = os.path.join(relacs_dir, 'stimuli.dat')
    with open(stimuli_file, 'r') as sf:
        for line in sf:
            if len(line) == 0 or line[0] != '#':
                break
            if "unit%d" % trace in line:
                unit = line.split(':')[1].strip()
            if "sampling rate%d" % trace in line:
                value = line.split(':')[1].strip()
                samplerate = float(value.replace('Hz',''))

    if samplerate is not None:
        return samplerate, unit
    raise ValueError('could not retrieve sampling rate from ' + stimuli_file)


def relacs_metadata(filename):
    """
    Reads header of a relacs *.dat file.

    Parameters
    ----------
    filename: string
        a relacs *.dat file.

    Returns
    -------
    data: dict
        dictionary with the content of the file header.
        
    Raises
    ------
    IOError/FileNotFoundError:
        If filename cannot be opened.
    """
    
    data = {}
    with open(filename, 'r') as sf:
        for line in sf:
            if len(line) == 0 or line[0] != '#':
                break
            words = line.split(':')
            if len(words) >= 2:
                key = words[0].strip('# ')
                value = ':'.join(words[1:]).strip()
                data[key] = value
    return data


def check_relacs(filepathes):
    """
    Check whether filepathes are relacs files.

    Parameters
    ----------
    filepathes: string or list of strings
        path to a relacs data directory, a file in a relacs data directory,
        or relacs trace-*.raw files.

    Returns
    -------
    If filepathes is a single path, then returns True if it is a file in
    a valid relacs data directory.
    If filepathes are more than one path, then returns True if filepathes
    are trace-*.raw files in a valid relacs data directory.
    """

    path = filepathes
    # filepathes must be trace-*.raw:
    if type(filepathes) is list:
        if len(filepathes) > 1:
            for file in filepathes:
                bn = os.path.basename(file)
                if len(bn) <= 5 or bn[0:5] != 'trace' or bn[-4:] != '.raw':
                    return False
        path = filepathes[0]
    # relacs data directory:
    relacs_dir = path
    if not os.path.isdir(path):
        relacs_dir = os.path.dirname(path)
    # check for a valid relacs data directory:
    if (os.path.isfile(os.path.join(relacs_dir, 'stimuli.dat')) and
        os.path.isfile(os.path.join(relacs_dir, 'trace-1.raw'))):
        return True
    else:
        return False
    

def load_relacs(filepathes, verbose=0):
    """
    Load traces (trace-*.raw files) that have been recorded with relacs (www.relacs.net).

    Parameters
    ----------
    filepathes: string or list of string
        path to a relacs data directory, a relacs stimuli.dat file, a relacs info.dat file,
        or relacs trace-*.raw files.

    Returns
    -------
    data: 2-D array
        the data, first dimension time, second dimension channel
    samplerate: float
        the sampling rate of the data in Hz
    unit: string
        the unit of the data

    Raises
    ------
    ValueError:
        - Invalid name for relacs trace-*.raw file.
        - Sampling rates of traces differ.
        - Unit of traces differ.
    """
    
    # fix pathes:
    if type(filepathes) is not list:
        filepathes = [filepathes]
    if len(filepathes) == 1:
        if os.path.isdir(filepathes[0]):
            filepathes = glob.glob(os.path.join(filepathes[0], 'trace-*.raw'))
        else:
            bn = os.path.basename(filepathes[0])
            if len(bn) <= 5 or bn[0:5] != 'trace' or bn[-4:] != '.raw':
                filepathes = glob.glob(os.path.join(os.path.dirname(filepathes[0]),
                                                    'trace-*.raw'))
                
    # load trace*.raw files:
    nchannels = len(filepathes)
    data = None
    nrows = 0
    samplerate = None
    unit = ""
    for n, path in enumerate(filepathes):
        bn = os.path.basename(path)
        if len(bn) <= 5 or bn[0:5] != 'trace' or bn[-4:] != '.raw':
            raise ValueError('invalid name %s of relacs trace file', path)
        x = np.fromfile(path, np.float32)
        if verbose > 0:
            print( 'loaded %s' % path)
        if data is None:
            nrows = len(x)-2
            data = np.empty((nrows, nchannels))
        data[:,n] = x[:nrows]
        # retrieve sampling rate and unit:
        rate, us = relacs_samplerate_unit(path)
        if samplerate is None:
            samplerate = rate
        elif rate != samplerate:
            raise ValueError('sampling rates of traces differ')
        if len(unit) == 0:
            unit = us
        elif us != unit:
            raise ValueError('unit of traces differ')
    return data, samplerate, unit


def check_pickle(filepath):
    """
    Check if filepath is a pickle file.
    
    Returns
    -------
    True, if fielpath is a pickle file.
    """
    ext = os.path.splitext(filepath)[1]
    return ext == 'pkl'


def load_pickle(filename, channel=0):
    """
    Load Joerg's pickle files.

    Args:
        filepath (string): the full path and name of the file to load
        channel (int): the single channel to be returned

    Returns:
        data (array): the data trace as a 1-D numpy array
        freq (float): the sampling rate of the data in Hz
        unit (string): the unit of the data
    """
    import pickle
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    time = data['time_trace']
    freq = 1000.0 / (time[1] - time[0])
    tracen = data['raw_data'].shape[1]
    if channel >= tracen:
        print('number of channels in file %s is %d, but requested channel %d' %
              (filename, tracen, channel))
        channel = tracen - 1
    return data['raw_data'][:, channel], freq, 'mV'


def load_data(filepath, channel=0, verbose=0):
    """
    Call this function to load a single trace of data from a file.

    Parameters
    ----------
    filepath: string
        the full path and name of the file to load
    channel: int
        the single channel to be returned
    verbose: int
        if >0 show detailed error/warning messages

    Returns
    -------
    data: array
        the data trace as a 1-D numpy array
    samplerate: float
        the sampling rate of the data in Hz
    unit: string
        the unit of the data

    Raise
    -----
    ValueError:
        input argument filepath is empty string or list.
    """
    
    # check values:
    data = np.array([])
    samplerate = 0.0
    unit = ''
    if len(filepath) == 0:
        raise ValueError('input argument filepath is empty string or list.')

    # load data:
    if check_relacs(filepath):
        data, samplerate, unit = load_relacs(filepath, verbose)
        channels = data.shape[1]
        if channel >= channels:
            print('number of channels in file %s is %d, but requested channel %d' %
                  (filepath, channels, channel))
            channel = channels - 1
        if channel >= 0:
            data = data[:, channel]
        return data, samplerate, unit
    elif check_pickle(filepath):
        data, samplerate, unit = load_pickle(filepath, channel)
    else:
        data, samplerate = aio.load_audio(filepath, verbose)
        channels = data.shape[1]
        if channel >= channels:
            print('number of channels in file %s is %d, but requested channel %d' %
                  (filepath, channels, channel))
            channel = channels - 1
        if channel >= 0:
            data = data[:, channel]
        unit = 'a.u.'
    return data, samplerate, unit


class DataLoader(aio.AudioLoader):
    """
    """

    def __init__(self, filepath=None, channel=0, buffersize=10.0, backsize=0.0, verbose=0):
        """Initialize the DataLoader instance. If filepath is not None open the file.

        Args:
          filepath (string): name of the file
          channel (int): the single channel to be worked on
          buffersize (float): size of internal buffer in seconds
          backsize (float): part of the buffer to be loaded before the requested start index in seconds
          verbose (int): if >0 show detailed error/warning messages
        """
        super(DataLoader, self).__init__(filepath, buffersize, backsize, verbose)
        if channel < 0:
            channel = 0
        if channel > self.channels:
            channel = self.channels - 1
        self.channel = channel
        self.unit = 'a.u.'

    def __getitem__(self, key):
        if type(key) is tuple:
            raise IndexError
        return super(DataLoader, self).__getitem__((key, self.channel))
 
    def __next__(self):
        return super(DataLoader, self).__next__()[self.channel]
 
    def open(self, filepath, channel=0, buffersize=10.0, backsize=0.0, verbose=0):
        """Open data file for reading.

        Args:
          filepath (string): name of the file
          channel (int): the single channel to be worked on
          buffersize (float): size of internal buffer in seconds
          backsize (float): part of the buffer to be loaded before the requested start index in seconds
          verbose (int): if >0 show detailed error/warning messages
        """
        super(DataLoader, self).open(filepath, buffersize, backsize, verbose)
        if channel < 0:
            channel = 0
        if channel > self.channels:
            channel = self.channels - 1
        self.channel = channel
        self.unit = 'a.u.'
        self.shape = (self.frames,)
        return self


open_data = DataLoader

if __name__ == "__main__":
    import sys
    import matplotlib.pyplot as plt
    
    data, rate, unit = load_relacs(sys.argv[1:])
    print(data, rate, unit)
    exit()


    print("Checking dataloader module ...")
    print('')
    print('Usage:')
    print('  python dataloader.py [-p] <datafile>')
    print('  -p: plot data')
    print('')

    filepath = sys.argv[-1]
    channel = 0
    plot = False
    if len(sys.argv) > 1 and sys.argv[1] == '-p':
        plot = True

    print("try load_data:")
    data, rate, unit = load_data(filepath, channel, 2)
    if plot:
        plt.plot(np.arange(len(data)) / rate, data)
        plt.show()

    print('')
    print("try DataLoader:")
    with open_data(filepath, 0, 2.0, 1.0, 1) as data:
        print('samplerate: %g' % data.samplerate)
        print('frames: %d %d' % (len(data), data.shape[0]))
        nframes = int(1.0 * data.samplerate)
        # forward:
        for i in range(0, len(data), nframes):
            print('forward %d-%d' % (i, i + nframes))
            x = data[i:i + nframes]
            if plot:
                plt.plot((i + np.arange(len(x))) / rate, x)
                plt.show()
        # and backwards:
        for i in reversed(range(0, len(data), nframes)):
            print('backward %d-%d' % (i, i + nframes))
            x = data[i:i + nframes]
            if plot:
                plt.plot((i + np.arange(len(x))) / rate, x)
                plt.show()
