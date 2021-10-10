"""
Load time-series data from files.

```
data, samplingrate, unit = load_data('data/file.wav')
```
Loads the whole time-series from the file as a numpy array of floats.

```
data = DataLoader('data/file.wav', 0, 60.0)
```
or
```
with open_data('data/file.wav', 0, 60.0) as data:
```
Create an `DataLoader` object that loads chuncks of 60 seconds long data on demand.
data can be used like a read-only numpy array of floats.

`relacs_metadata()` reads key-value pairs from relacs *.dat file headers.
"""

import os
import glob
import numpy as np
from audioio.audioloader import load_audio, AudioLoader


def relacs_samplerate_unit(filename, channel=0):
    """
    Retrieve sampling rate and unit from a relacs stimuli.dat file.

    Parameters
    ----------
    filename: string
        path to a relacs data directory, a file in a relacs data directory,
        or a relacs trace-*.raw file.
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
        dictionary with the key-value pairs of the file header.
        
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
    is_relacs: boolean
      If filepathes is a single path, then returns `True` if it is a or is a file in
      a valid relacs data directory.
      If filepathes are more than one path, then returns `True` if filepathes
      are 'trace-*.raw' files in a valid relacs data directory.
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

    
def relacs_files(filepathes, channel):
    """
    Expand file pathes for relacs data to appropriate trace*.raw file names.

    Parameters
    ----------
    filepathes: string or list of strings
        path to a relacs data directory, a file in a relacs data directory,
        or relacs trace-*.raw files.
    channel: int
        The data channel. If negative all channels are selected.
        
    Returns
    -------
    filepathes: list of strings
        List of relacs trace*.raw files.

    Raises
    ------
    ValueError: invalid name of relacs trace file
    """
    if type(filepathes) is not list:
        filepathes = [filepathes]
    if len(filepathes) == 1:
        if os.path.isdir(filepathes[0]):
            if channel < 0:
                relacs_dir = filepathes[0]
                filepathes = []
                for k in range(10000):
                    file = os.path.join(relacs_dir, 'trace-%d.raw'%(k+1))
                    if os.path.isfile(file):
                        filepathes.append(file)
                    else:
                        break
            else:
                filepathes[0] = os.path.join(filepathes[0], 'trace-%d.raw' % (channel+1))
        else:
            bn = os.path.basename(filepathes[0])
            if len(bn) <= 5 or bn[0:5] != 'trace' or bn[-4:] != '.raw':
                if channel < 0:
                    relacs_dir = os.path.dirname(filepathes[0])
                    filepathes = []
                    for k in range(10000):
                        file = os.path.join(relacs_dir, 'trace-%d.raw'%(k+1))
                        if os.path.isfile(file):
                            filepathes.append(file)
                        else:
                            break
                else:
                    filepathes[0] = os.path.join(os.path.dirname(filepathes[0]),
                                                 'trace-%d.raw' % (channel+1))
    for path in filepathes:
        bn = os.path.basename(path)
        if len(bn) <= 5 or bn[0:5] != 'trace' or bn[-4:] != '.raw':
            raise ValueError('invalid name %s of relacs trace file', path)
        
    return filepathes

        
def load_relacs(filepathes, channel=-1, verbose=0):
    """
    Load traces (trace-*.raw files) that have been recorded with relacs (www.relacs.net).

    Parameters
    ----------
    filepathes: string or list of strings
        path to a relacs data directory, a file in a relacs data directory,
        or relacs trace-*.raw files.
    channel: int
        The data channel. If negative all channels are selected.
    verbose: int
        if > 0 show detailed error/warning messages

    Returns
    -------
    data: 1-D or 2-D array
        If channel is negative or more than one trace file is specified,
        a 2-D array with data of all channels is returned,
        where first dimension is time and second dimension is channel number.
        Otherwise an 1-D array with the data of that channel is returned.
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
    filepathes = relacs_files(filepathes, channel)
    if len(filepathes) > 1:
        channel = -1
                
    # load trace*.raw files:
    nchannels = len(filepathes)
    data = None
    nrows = 0
    samplerate = None
    unit = ""
    for n, path in enumerate(filepathes):
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
    if channel < 0:
        return data, samplerate, unit
    else:
        return data[:, 0], samplerate, unit


def fishgrid_samplerate(filename):
    """
    Retrieve the sampling rate from a fishgrid.cfg file.

    Parameters
    ----------
    filename: string
        path to a fishgrid data directory, a file in a fishgrid data directory,
        or a fishgrid traces-*.raw file.

    Returns
    -------
    samplerate: float
        the sampling rate in Hertz

    Raises
    ------
    IOError/FileNotFoundError:
        If the fishgrid.cfg file does not exist.
    ValueError:
        fishgrid.cfg file does not contain sampling rate.
    """
    # check for fishgrid data directory:
    fishgrid_dir = filename
    if not os.path.isdir(filename):
        fishgrid_dir = os.path.dirname(filename)

    # retreive sampling rate from fishgrid.cfg file:
    samplerate = None
    fishgrid_file = os.path.join(fishgrid_dir, 'fishgrid.cfg')
    with open(fishgrid_file, 'r') as sf:
        for line in sf:
            if "AISampleRate" in line:
                value = line.split(':')[1].strip()
                samplerate = float(value.replace('kHz',''))*1000.0

    if samplerate is not None:
        return samplerate
    raise ValueError('could not retrieve sampling rate from ' + fishgrid_file)


def fishgrid_spacings(filename):
    fishgrid_dir = filename
    if not os.path.isdir(filename):
        fishgrid_dir = os.path.dirname(filename)

    # retreive grids from fishgrid.cfg file:
    grids_dist = []
    rows_dist = None
    cols_dist = None
    fishgrid_file = os.path.join(fishgrid_dir, 'fishgrid.cfg')
    with open(fishgrid_file, 'r') as sf:
        for line in sf:
            if "Grid" in line:
                if rows_dist is not None and cols_dist is not None:
                    grids_dist.append((rows_dist, cols_dist))
                rows_dist = None
                cols_dist = None
            elif "ColumnDistance1" in line:
                cols_dist = int(line.split(':')[1].strip().split('.')[0])
            elif "RowDistance1" in line:
                rows_dist = int(line.split(':')[1].strip().split('.')[0])
        if rows_dist is not None and cols_dist is not None:
            grids_dist.append((rows_dist, cols_dist))
    return grids_dist


def fishgrid_grids(filename):
    """ Retrieve grid sizes from a fishgrid.cfg file.

    Parameters
    ----------
    filename: string
        path to a fishgrid data directory, a file in a fishgrid data directory,
        or a fishgrid traces-*.raw file.

    Returns
    -------
    grids: list of tuples of ints
        For each grid the number of rows and columns.

    Raises
    ------
    IOError/FileNotFoundError:
        If the fishgrid.cfg file does not exist.
    """
    # check for fishgrid data directory:
    fishgrid_dir = filename
    if not os.path.isdir(filename):
        fishgrid_dir = os.path.dirname(filename)

    # retreive grids from fishgrid.cfg file:
    grids = []
    rows = None
    cols = None
    fishgrid_file = os.path.join(fishgrid_dir, 'fishgrid.cfg')
    with open(fishgrid_file, 'r') as sf:
        for line in sf:
            if "Grid" in line:
                if rows is not None and cols is not None:
                    grids.append((rows, cols))
                rows = None
                cols = None
            elif "Columns" in line:
                cols = int(line.split(':')[1].strip())
            elif "Rows" in line:
                rows = int(line.split(':')[1].strip())
        if rows is not None and cols is not None:
            grids.append((rows, cols))
    return grids


def check_fishgrid(filepathes):
    """
    Check whether filepathes are valid fishgrid files (https://github.com/bendalab/fishgrid).

    Parameters
    ----------
    filepathes: string or list of strings
        path to a fishgrid data directory, a file in a fishgrid data directory,
        or fishgrid traces-*.raw files.

    Returns
    -------
    is_fishgrid: bool
        If filepathes is a single path, then returns `True` if it is a file in
        a valid fishgrid data directory.
        If filepathes are more than one path, then returns `True` if filepathes
        are 'trace-*.raw' files in a valid fishgrid data directory.
    """
    path = filepathes
    # filepathes must be traces-*.raw:
    if type(filepathes) is list:
        if len(filepathes) > 1:
            for file in filepathes:
                bn = os.path.basename(file)
                if len(bn) <= 7 or bn[0:7] != 'traces-' or bn[-4:] != '.raw':
                    return False
        path = filepathes[0]
    # relacs data directory:
    fishgrid_dir = path
    if not os.path.isdir(path):
        fishgrid_dir = os.path.dirname(path)
    # check for a valid relacs data directory:
    if (os.path.isfile(os.path.join(fishgrid_dir, 'fishgrid.cfg')) and
        os.path.isfile(os.path.join(fishgrid_dir, 'traces-grid1.raw'))):
        return True
    else:
        return False

    
def fishgrid_files(filepathes, channel, grid_sizes):
    """
    Expand file pathes for fishgrid data to appropriate traces-*.raw file names.

    Parameters
    ----------
    filepathes: string or list of strings
        path to a fishgrid data directory, a file in a fishgrid data directory,
        or fishgrid traces-*.raw files.
    channel: int
        The data channel. If negative all channels are selected.
    grid_sizes: list of int
        The number of channels of each grid.
        
    Returns
    -------
    filepathes: list of strings
        List of fishgrid traces-*.raw files.

    Raises
    ------
    IndexError: invalid channel.
    """
    # find grids:
    grid = -1
    if channel >= 0:
        grid = -1
        gs = 0
        for g, s in enumerate(grid_sizes):
            gs += s
            if channel < gs:
                grid = g
                break
        if grid < 0:
            raise IndexError("invalid channel")
            
    if type(filepathes) is not list:
        filepathes = [filepathes]
    if len(filepathes) == 1:
        if os.path.isdir(filepathes[0]):
            if grid < 0:
                fishgrid_dir = filepathes[0]
                filepathes = []
                for k in range(10000):
                    file = os.path.join(fishgrid_dir, 'traces-grid%d.raw'%(k+1))
                    if os.path.isfile(file):
                        filepathes.append(file)
                    else:
                        break
            else:
                filepathes[0] = os.path.join(filepathes[0], 'traces-grid%d.raw' % (grid+1))
        else:
            bn = os.path.basename(filepathes[0])
            if len(bn) <= 7 or bn[0:7] != 'traces-' or bn[-4:] != '.raw':
                if grid < 0:
                    fishgrid_dir = os.path.dirname(filepathes[0])
                    filepathes = []
                    for k in range(10000):
                        file = os.path.join(fishgrid_dir, 'traces-grid%d.raw'%(k+1))
                        if os.path.isfile(file):
                            filepathes.append(file)
                        else:
                            break
                else:
                    filepathes[0] = os.path.join(os.path.dirname(filepathes[0]),
                                                 'traces-grid%d.raw' % (grid+1))
    for path in filepathes:
        bn = os.path.basename(path)
        if len(bn) <= 7 or bn[0:7] != 'traces-' or bn[-4:] != '.raw':
            raise ValueError('invalid name %s of fishgrid traces file', path)

    return filepathes

        
def load_fishgrid(filepathes, channel=-1, verbose=0):
    """
    Load traces (traces-grid*.raw files) that have been recorded with fishgrid (https://github.com/bendalab/fishgrid).

    Parameters
    ----------
    filepathes: string or list of string
        path to a fishgrid data directory, a fishgrid.cfg file,
        or fidhgrid traces-grid*.raw files.
     channel: int
        The data channel. If negative all channels are selected.
    verbose: int
        if > 0 show detailed error/warning messages

    Returns
    -------
    data: 1-D or 2-D array
        If channel is negative or more than one trace file is specified,
        a 2-D array with data of all channels is returned,
        where first dimension is time and second dimension is channel number.
        Otherwise an 1-D array with the data of that channel is returned.
    samplerate: float
        the sampling rate of the data in Hz
    unit: string
        the unit of the data
    """
    if type(filepathes) is not list:
        filepathes = [filepathes]
    grids = fishgrid_grids(filepathes[0])
    grid_sizes = [r*c for r,c in grids]
    filepathes = fishgrid_files(filepathes, channel, grid_sizes)
    if len(filepathes) > 1:
        channel = -1
                
    # load traces-grid*.raw files:
    grid_channels = []
    nchannels = 0
    for path in filepathes:
        g = int(os.path.basename(path)[11:].replace('.raw', '')) - 1
        grid_channels.append(grid_sizes[g])
        nchannels += grid_sizes[g]
    data = None
    nrows = 0
    n = 0
    samplerate = None
    if len(filepathes) > 0:
        samplerate = fishgrid_samplerate(filepathes[0])
    unit = "V"
    for path, channels in zip(filepathes, grid_channels):
        x = np.fromfile(path, np.float32).reshape((-1, channels))
        if verbose > 0:
            print( 'loaded %s' % path)
        if data is None:
            nrows = len(x)-2
            data = np.empty((nrows, nchannels))
        data[:,n:n+channels] = x[:nrows,:]
    if channel < 0:
        return data, samplerate, unit
    else:
        gs = 0
        for s in grid_sizes:
            if channel < gs + s:
                break
            gs += s
        return data[:, channel-gs], samplerate, unit


def check_pickle(filepath):
    """ Check if filepath is a pickle file.
    
    Returns
    -------
    is_pickle: bool
        `True`, if filepath is a pickle file.
    """
    ext = os.path.splitext(filepath)[1]
    return ext == 'pkl'


def load_pickle(filename, channel=-1, verbose=0):
    """ Load Joerg's pickle files.

    Parameters
    ----------
    filepath: string
        The full path and name of the file to load.
    channel: int
        The data channel. If negative all channels are selected.
    verbose: int
        if > 0 show detailed error/warning messages

    Returns
    -------
    data: 1-D or 2-D array
        If channel is negative, a 2-D array with data of all channels is returned,
        where first dimension is time and second dimension is channel number.
        Otherwise an 1-D array with the data of that channel is returned.
    samplerate: float
        The sampling rate of the data in Hz.
    unit: string
        The unit of the data.
    """
    import pickle
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    if verbose > 0:
        print( 'loaded %s' % filename)
    time = data['time_trace']
    samplerate = 1000.0 / (time[1] - time[0])
    if channel >= 0:
        if channel >= data.shape[1]:
            raise IndexError('invalid channel number %d requested' % channel)
        data = data[:, channel]
        return data['raw_data'][:, channel], samplerate, 'mV'
    return data['raw_data'], samplerate, 'mV'


def load_data(filepath, channel=-1, verbose=0):
    """ Call this function to load time-series data from a file of arbitrary format.

    Parameters
    ----------
    filepath: string or list of strings
        The full path and name of the file to load. For some file
        formats several files can be provided.
    channel: int
        The data channel. If negative all channels are selected.
    verbose: int
        if > 0 show detailed error/warning messages

    Returns
    -------
    data: 1-D or 2-D array
        If channel is negative, a 2-D array with data of all channels is returned,
        where first dimension is time and second dimension is channel number.
        Otherwise an 1-D array with the data of that channel is returned.
    samplerate: float
        the sampling rate of the data in Hz
    unit: string
        the unit of the data

    Raises
    ------
    ValueError:
        Input argument filepath is empty string or list.
    IndexError:
        Invalid channel requested.
    """
    # check values:
    data = np.array([])
    samplerate = 0.0
    unit = ''
    if len(filepath) == 0:
        raise ValueError('input argument filepath is empty string or list.')

    # load data:
    if check_relacs(filepath):
        return load_relacs(filepath, channel, verbose)
    elif check_fishgrid(filepath):
        return load_fishgrid(filepath, channel, verbose)
    else:
        if type(filepath) is list:
            filepath = filepath[0]
        if check_pickle(filepath):
            return load_pickle(filepath, channel, verbose)
        else:
            data, samplerate = load_audio(filepath, verbose)
            if channel >= 0:
                if channel >= data.shape[1]:
                    raise IndexError('invalid channel number %d requested' % channel)
                data = data[:, channel]
            unit = 'a.u.'
        return data, samplerate, unit


class DataLoader(AudioLoader):
    """ Buffered reading of time-series data for random access of the data in the file.
    
    This allows for reading very large data files that do not fit into memory.
    An `DataLoader` instance can be used like a huge read-only numpy array, i.e.
    ```
    data = DataLoader('path/to/data/file.dat')
    x = data[10000:20000,0]
    ```
    The first index specifies the frame, the second one the channel.

    `DataLoader` first determines the format of the data file and then opens
    the file (first line). It then reads data from the file
    as necessary for the requested data (second line).

    Supported file formats are relacs trace*.raw files (www.relacs.net),
    fishgrid traces-*.raw files, and audio files via `audioio.AudioLoader`.

    Reading sequentially through the file is always possible. If previous data
    are requested, then the file is read from the beginning. This might slow down access
    to previous data considerably. Use the `backsize` argument to the open functions to
    make sure some data are loaded before the requested frame. Then a subsequent access
    to the data within backsize seconds before that frame can still be handled without
    the need to reread the file from the beginning.

    Usage:
    ------
    ```
    import thunderfish.dataloader as dl
    with dl.open_data(filepath, -1, 60.0, 10.0) as data:
        # do something with the content of the file:
        x = data[0:10000, 0]
        y = data[10000:20000, 0]
        z = x + y
    ```
    
    Normal open and close:
    ```
    data = dl.DataLoader(filepath, 0, 60.0)
    x = data[:]  # read the whole file
    data.close()
    ```    
    that is the same as:
    ```
    data = dl.DataLoader()
    data.open(filepath, 0, 60.0)
    ```

    Member variables:
    -----------------
    samplerate (float): the sampling rate of the data in Hertz.
    channels (int): the number of channels that are read in.
    channel (int): the channel of which the trace is returned.
                   If negative, all channels are returned.
    frames (int): the number of frames in the file.
    shape (tuple): number of frames and channels of the data.
    unit (string): unit of the data.

    Some member functions:
    ----------------------
    len(): the number of frames
    open(): open a data file.
    open_*(): open a data file of a specific format.
    close(): close the file.
    """

    def __init__(self, filepath=None, channel=-1, buffersize=10.0, backsize=0.0, verbose=0):
        """ Initialize the DataLoader instance. If filepath is not None open the file.

        Parameters
        ----------
        filepath: string
            Name of the file.
        channel: int
            The single channel to be worked on.
        buffersize: float
            Size of internal buffer in seconds.
        backsize: float
            Part of the buffer to be loaded before the requested start index in seconds.
        verbose: int
            If > 0 show detailed error/warning messages.
        """
        super(DataLoader, self).__init__(None, buffersize, backsize, verbose)
        if filepath is not None:
            self.open(filepath, channel, buffersize, backsize, verbose)

    def __getitem__(self, key):
        if self.channel >= 0:
            if type(key) is tuple:
                raise IndexError
            return super(DataLoader, self).__getitem__((key, self.channel))
        else:
            return super(DataLoader, self).__getitem__(key)
 
    def __next__(self):
        if self.channel >= 0:
            return super(DataLoader, self).__next__()[self.channel]
        else:
            return super(DataLoader, self).__next__()

    
    # relacs interface:        
    def open_relacs(self, filepathes, channel=-1, buffersize=10.0, backsize=0.0, verbose=0):
        """ Open relacs data files (www.relacs.net) for reading.

        Parameters
        ----------
        filepathes: string or list of string
            Path to a relacs data directory, a relacs stimuli.dat file, a relacs info.dat file,
            or relacs trace-*.raw files.
        channel: int
            The requested data channel. If negative all channels are selected.
        buffersize: float
            Size of internal buffer in seconds.
        backsize: float
            Part of the buffer to be loaded before the requested start index in seconds.
        verbose: int
            If > 0 show detailed error/warning messages.
        """
        self.verbose = verbose
        
        if self.sf is not None:
            self._close_relacs()

        filepathes = relacs_files(filepathes, channel)
        if len(filepathes) > 1:
            channel = -1
        else:
            channel = 0

        # open trace files:
        self.sf = []
        self.frames = None
        self.samplerate = None
        self.unit = ""
        for path in filepathes:
            file = open(path, 'rb')
            self.sf.append(file)
            if verbose > 0:
                print( 'opened %s' % path)
            # file size:
            file.seek(0, os.SEEK_END)
            frames = file.tell()//4
            if self.frames is None:
                self.frames = frames
            elif self.frames != frames:
                diff = self.frames - frames
                if diff > 1 or diff < -2:
                    raise ValueError('number of frames of traces differ')
                elif diff >= 0:
                    self.frames = frames
            file.seek(0)
            # retrieve sampling rate and unit:
            rate, us = relacs_samplerate_unit(path)
            if self.samplerate is None:
                self.samplerate = rate
            elif rate != self.samplerate:
                raise ValueError('sampling rates of traces differ')
            if len(self.unit) == 0:
                self.unit = us
            elif us != self.unit:
                raise ValueError('unit of traces differ')
        self.channels = len(self.sf)
        self.channel = channel
        if self.channel >= 0:
            self.shape = (self.frames,)
        else:
            self.shape = (self.frames, self.channels)
        self.buffersize = int(buffersize*self.samplerate)
        self.backsize = int(backsize*self.samplerate)
        self._init_buffer()
        self.offset = 0
        self.close = self._close_relacs
        self._update_buffer = self._update_buffer_relacs
        return self

    def _close_relacs(self):
        """ Close the relacs data files.
        """
        if self.sf is not None:
            for file in self.sf:
                file.close()
            self.sf = None

    def _update_buffer_relacs(self, start, stop):
        """ Make sure that the buffer contains the data between
        start and stop for relacs files.
        """
        if start < self.offset or stop > self.offset + self.buffer.shape[0]:
            offset, size = self._read_indices(start, stop)
            r_offset, r_size = self._recycle_buffer(offset, size)
            # read buffer:
            for i, file in enumerate(self.sf):
                file.seek(r_offset*4)
                buffer = file.read(r_size*4)
                self.buffer[r_offset-offset:r_offset+r_size-offset, i] = np.fromstring(buffer, dtype=np.float32)
            self.offset = offset
            if self.verbose > 1:
                print('  read %6d frames at %d' % (r_size, r_offset))
            if self.verbose > 0:
                print('  loaded %d frames from %d up to %d'
                      % (self.buffer.shape[0], self.offset, self.offset+self.buffer.shape[0]))
        
    
    # fishgrid interface:        
    def open_fishgrid(self, filepathes, channel=-1, buffersize=10.0, backsize=0.0, verbose=0):
        """ Open fishgrid data files (https://github.com/bendalab/fishgrid) for reading.

        Parameters
        ----------
        filepathes: string or list of string
            Path to a fishgrid data directory, a fishgrid.cfg file,
            or fishgrid trace-*.raw files.
        channel: int
            The requested data channel. If negative all channels are selected.
        buffersize: float
            Size of internal buffer in seconds.
        backsize: float
            Part of the buffer to be loaded before the requested start index in seconds.
        verbose: int
            If > 0 show detailed error/warning messages.
        """
        self.verbose = verbose
        
        if self.sf is not None:
            self._close_fishgrid()

        if type(filepathes) is not list:
            filepathes = [filepathes]
        grids = fishgrid_grids(filepathes[0])
        grid_sizes = [r*c for r,c in grids]
        filepathes = fishgrid_files(filepathes, channel, grid_sizes)
        if len(filepathes) > 1:
            channel = -1

        # open grid files:
        self.channels = 0
        for path in filepathes:
            g = int(os.path.basename(path)[11:].replace('.raw', '')) - 1
            self.channels += grid_sizes[g]
        self.sf = []
        self.grid_channels = []
        self.grid_offs = []
        offs = 0
        self.frames = None
        self.samplerate = None
        if len(filepathes) > 0:
            self.samplerate = fishgrid_samplerate(filepathes[0])
        self.unit = "V"
        for path in filepathes:
            file = open(path, 'rb')
            self.sf.append(file)
            if verbose > 0:
                print( 'opened %s' % path)
            # grid channels:
            g = int(os.path.basename(path)[11:].replace('.raw', '')) - 1
            self.grid_channels.append(grid_sizes[g])
            self.grid_offs.append(offs)
            offs += grid_sizes[g]
            # file size:
            file.seek(0, os.SEEK_END)
            frames = file.tell()//4//grid_sizes[g]
            if self.frames is None:
                self.frames = frames
            elif self.frames != frames:
                diff = self.frames - frames
                if diff > 1 or diff < -2:
                    raise ValueError('number of frames of traces differ')
                elif diff >= 0:
                    self.frames = frames
            file.seek(0)
        gs = 0
        for s in grid_sizes:
            if channel < gs + s:
                break
            gs += s
        self.channel = channel - gs
        if self.channel >= 0:
            self.shape = (self.frames,)
        else:
            self.shape = (self.frames, self.channels)
        self.buffersize = int(buffersize*self.samplerate)
        self.backsize = int(backsize*self.samplerate)
        self._init_buffer()
        self.offset = 0
        self.close = self._close_fishgrid
        self._update_buffer = self._update_buffer_fishgrid
        return self

    def _close_fishgrid(self):
        """ Close the fishgrid data files.
        """
        if self.sf is not None:
            for file in self.sf:
                file.close()
            self.sf = None

    def _update_buffer_fishgrid(self, start, stop):
        """ Make sure that the buffer contains the data between
        start and stop for fishgrid files.
        """
        if start < self.offset or stop > self.offset + self.buffer.shape[0]:
            offset, size = self._read_indices(start, stop)
            r_offset, r_size = self._recycle_buffer(offset, size)
            # read buffer:
            for file, gchannels, goffset in zip(self.sf, self.grid_channels, self.grid_offs):
                file.seek(r_offset*4*gchannels)
                buffer = file.read(r_size*4*gchannels)
                self.buffer[r_offset-offset:r_offset+r_size-offset, goffset:goffset+gchannels] = np.fromstring(buffer, dtype=np.float32).reshape((-1, gchannels))
            self.offset = offset
            if self.verbose > 1:
                print('  read %6d frames at %d' % (r_size, r_offset))
            if self.verbose > 0:
                print('  loaded %d frames from %d up to %d'
                      % (self.buffer.shape[0], self.offset, self.offset+self.buffer.shape[0]))
        

    def open(self, filepath, channel=0, buffersize=10.0, backsize=0.0, verbose=0):
        """ Open file with time-series data for reading.

        Parameters
        ----------
        filepathes: string or list of string
            Path to a data files or directory.
        channel: int
            The requested data channel. If negative all channels are selected.
        buffersize: float
            Size of internal buffer in seconds.
        backsize: float
            Part of the buffer to be loaded before the requested start index in seconds.
        verbose: int
            If > 0 show detailed error/warning messages.
        """
        if check_relacs(filepath):
            self.open_relacs(filepath, channel, buffersize, backsize, verbose)
        elif check_fishgrid(filepath):
            self.open_fishgrid(filepath, channel, buffersize, backsize, verbose)
        else:
            if type(filepath) is list:
                filepath = filepath[0]
            super(DataLoader, self).open(filepath, buffersize, backsize, verbose)
            if channel > self.channels:
                raise IndexError('invalid channel number %d' % channel)
            self.channel = channel
            if self.channel >= 0:
                self.shape = (self.frames,)
            else:
                self.shape = (self.frames, self.channels)
            self.unit = 'a.u.'
        return self


open_data = DataLoader


if __name__ == "__main__":
    import sys
    import matplotlib.pyplot as plt

    print("Checking dataloader module ...")
    print('')
    print('Usage:')
    print('  python dataloader.py [-p] [-c <channel>] <datafile> <channel>')
    print('  -p: plot data')
    print('')

    n = 1
    plot = False
    if len(sys.argv) > n and sys.argv[n] == '-p':
        plot = True
        n += 1
    channel = 0
    if len(sys.argv) > n+1 and sys.argv[n] == '-c':
        channel = int(sys.argv[n+1])
        n += 2
    filepath = sys.argv[n:]
        
    print("try load_data:")
    data, samplerate, unit = load_data(filepath, channel, 2)
    if plot:
        if channel < 0:
            time = np.arange(len(data)) / samplerate
            for c in range(data.shape[1]):
                plt.plot(time, data[:, c])
        else:
            plt.plot(np.arange(len(data)) / samplerate, data)
        plt.xlabel('Time [s]')
        plt.ylabel('[' + unit + ']')
        plt.show()

    print('')
    print("try DataLoader for channel=%d:" % channel)
    with open_data(filepath, channel, 2.0, 1.0, 1) as data:
        print('samplerate: %g' % data.samplerate)
        print('frames: %d %d' % (len(data), data.shape[0]))
        nframes = int(1.0 * data.samplerate)
        # forward:
        for i in range(0, len(data), nframes):
            print('forward %d-%d' % (i, i + nframes))
            if channel < 0:
                x = data[i:i + nframes, 0]
            else:
                x = data[i:i + nframes]
            if plot:
                plt.plot((i + np.arange(len(x))) / data.samplerate, x)
                plt.xlabel('Time [s]')
                plt.ylabel('[' + data.unit + ']')
                plt.show()
        # and backwards:
        for i in reversed(range(0, len(data), nframes)):
            print('backward %d-%d' % (i, i + nframes))
            if channel < 0:
                x = data[i:i + nframes, 0]
            else:
                x = data[i:i + nframes]
            if plot:
                plt.plot((i + np.arange(len(x))) / data.samplerate, x)
                plt.xlabel('Time [s]')
                plt.ylabel('[' + data.unit + ']')
                plt.show()
                
