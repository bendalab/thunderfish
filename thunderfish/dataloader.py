"""Load time-series data from files.

```
data, samplingrate, unit = load_data('data/file.wav')
```
Loads the whole time-series from the file as a numpy array of floats.

```
data = DataLoader('data/file.wav', 60.0)
```
or
```
with DataLoader('data/file.wav', 60.0) as data:
```
Create an `DataLoader` object that loads chuncks of 60 seconds long data
on demand. `data` can be used like a read-only numpy array of floats.


## Supported file formats

- python pickle files
- numpy .npz files
- matlab .mat files
- audio files via `audioio` package
- relacs trace*.raw files (www.relacs.net)
- fishgrid traces-*.raw files


## Metadata

Many file formats allow to store metadata that further describe the
stored time series data. We handle them as nested list of key-value
pairs. Load them with the `load_metadata()` function:
```
metadata = metadata('data/file.mat')
```

## Markers

Some file formats also allow to store markers that mark specific
positions in the time series data. Load marker positions and spans (in
the 2-D array `locs`) and label and text strings (in the 2-D array
`labels`) with the `load_markers()` function:
```
locs, labels = markers('data.wav')
```

## Aditional functions

- `relacs_samplerate_unit()`: retrieve sampling rate and unit from a relacs stimuli.dat file.
- `relacs_header()`: read key-value pairs from relacs *.dat file headers.
- `fishgrid_samplerate()`: retrieve the sampling rate from a fishgrid.cfg file.
- `fishgrid_grids()`: retrieve grid sizes from a fishgrid.cfg file.
- `fishgrid_spacings()`: spacing between grid electrodes.

"""

import os
import sys
import glob
import gzip
import numpy as np
try:
    import matplotlib.pyplot as plt
except ImportError:
    pass
from audioio import load_audio, AudioLoader, unflatten_metadata
from audioio import metadata as audioio_metadata
from audioio import markers as audioio_markers


def relacs_samplerate_unit(filepath, channel=0):
    """Retrieve sampling rate and unit from a relacs stimuli.dat file.

    Parameters
    ----------
    filepath: string
        Path to a relacs data directory, a file in a relacs data directory,
        or a relacs trace-*.raw file. Files can be .gz files.
    channel: int
        Channel (trace) number, if `filepath` does not specify a
        trace-*.raw file.

    Returns
    -------
    samplerate: float
        Sampling rate in Hertz
    unit: string
        Unit of the trace, can be empty if not found

    Raises
    ------
    IOError/FileNotFoundError:
        If the stimuli.dat file does not exist.
    ValueError:
        stimuli.dat file does not contain sampling rate.
    """
    trace = channel + 1
    relacs_dir = filepath
    # check for relacs data directory:
    if not os.path.isdir(filepath):
        relacs_dir = os.path.dirname(filepath)
        bn = os.path.basename(filepath).lower()
        i = bn.find('.raw')
        if len(bn) > 5 and bn[0:5] == 'trace' and i > 6:
            trace = int(bn[6:i])

    # retreive sampling rate and unit from stimuli.dat file:
    samplerate = None
    sampleinterval = None
    unit = ""

    lines = []
    stimuli_file = os.path.join(relacs_dir, 'stimuli.dat')
    if os.path.isfile(stimuli_file + '.gz'):
        stimuli_file += '.gz'
    if stimuli_file[-3:] == '.gz':
        with gzip.open(stimuli_file, 'r') as sf:
            for line in sf:
                line = line.decode('latin-1').strip()
                if len(line) == 0 or line[0] != '#':
                    break
                lines.append(line)
    else:
        with open(stimuli_file, 'r', encoding='latin-1') as sf:
            for line in sf:
                if len(line) == 0 or line[0] != '#':
                    break
                lines.append(line)
        
    for line in lines:
        if "unit%d" % trace in line:
            unit = line.split(':')[1].strip()
        if "sampling rate%d" % trace in line:
            value = line.split(':')[1].strip()
            samplerate = float(value.replace('Hz',''))
        elif "sample interval%d" % trace in line:
            value = line.split(':')[1].strip()
            sampleinterval = float(value.replace('ms',''))

    if samplerate is not None:
        return samplerate, unit
    if sampleinterval is not None:
        return 1000/sampleinterval, unit
    raise ValueError(f'could not retrieve sampling rate from {stimuli_file}')


def relacs_header(filepath, store_empty=False, first_only=False,
                  lower_keys=False, flat=False,
                  add_sections=False):
    """Read key-value pairs from a relacs *.dat file header.

    Parameters
    ----------
    filepath: string
        A relacs *.dat file, can be also a zipped .gz file.
    store_empty: bool
        If `False` do not add meta data with empty values.
    first_only: bool
        If `False` only store the first element of a list.
    lower_keys: bool
        Make all keys lower case.
    flat: bool
        Do not make a nested dictionary.
        Use this option also to read in very old relacs metadata with
        ragged left alignment.
    add_sections: bool
        If `True`, prepend keys with sections names separated by
        '.' to make them unique.

    Returns
    -------
    data: dict
        Nested dictionary with key-value pairs of the file header.
        
    Raises
    ------
    IOError/FileNotFoundError:
        If `filepath` cannot be opened.
    """
    # read in header from file:
    lines = []
    if os.path.isfile(filepath + '.gz'):
        filepath += '.gz'
    if filepath[-3:] == '.gz':
        with gzip.open(filepath, 'r') as sf:
            for line in sf:
                line = line.decode('latin-1')
                if len(line) == 0 or line[0] != '#':
                    break
                lines.append(line)
    else:
        with open(filepath, 'r', encoding='latin-1') as sf:
            for line in sf:
                if len(line) == 0 or line[0] != '#':
                    break
                lines.append(line)
    # parse:
    data = {}
    cdatas = [data]
    sections = ['']
    ident_offs = None
    ident = None
    for line in lines:
        if len(line) == 0 or line[0] != '#':
            break
        words = line.split(':')
        value = ':'.join(words[1:]).strip() if len(words) > 1 else ''
        if len(words) >= 1:
            key = words[0].strip('#')
            # get section level:
            level = 0
            if not flat or len(value) == 0:
                nident = len(key) - len(key.lstrip())
                if ident_offs is None:
                    ident_offs = nident
                elif ident is None:
                    if nident > ident_offs:
                        ident = nident - ident_offs
                        level = 1
                else:
                    level = (nident - ident_offs)//ident
                # close sections:
                if not flat:
                    while len(cdatas) > level + 1:
                        cdatas[-1][sections.pop()] = cdatas.pop()
                else:
                    while len(sections) > level + 1:
                        sections.pop()
            # key:
            key = key.strip().strip('"')
            if lower_keys:
                key = key.lower()
            skey = key
            if add_sections:
                key = '.'.join(sections[1:] + [key])
            if len(value) == 0:
                # new sub-section:
                if flat:
                    if store_empty:
                        cdatas[-1][key] = None
                else:
                    cdatas.append({})
                sections.append(skey)
            else:
                # key-value pair:
                value = value.strip('"')
                if len(value) > 0 or value != '-' or store_empty:
                    if len(value) > 0 and value[0] == '[' and value[-1] == ']':
                        value = [v.strip() for v in value.lstrip('[').rstrip(']').split(',')]
                        if first_only:
                            value = value[0]
                    cdatas[-1][key] = value
    while len(cdatas) > 1:
        cdatas[-1][sections.pop()] = cdatas.pop()
    return data


def check_relacs(file_paths):
    """Check whether file_paths are relacs files.

    Parameters
    ----------
    file_paths: string or list of strings
        Path to a relacs data directory, a file in a relacs data directory,
        or relacs trace-*.raw or trace-*.raw.gz files.

    Returns
    -------
    is_relacs: boolean
      If `file_paths` is a single path, then returns `True` if it is a
      valid relacs directory or is a file in a valid relacs data
      directory.
      If file_paths are more than one path, then returns `True` if `file_paths`
      are 'trace-*.raw' files in a valid relacs data directory.
    """
    path = file_paths
    # file_paths must be trace-*.raw:
    if isinstance(file_paths, (list, tuple, np.ndarray)):
        if len(file_paths) > 1:
            for file in file_paths:
                bn = os.path.basename(file)
                if len(bn) <= 5 or bn[0:5] != 'trace' or bn[6:].find('.raw') < 0:
                    return False
        path = file_paths[0]
    # relacs data directory:
    relacs_dir = path
    if not os.path.isdir(path):
        relacs_dir = os.path.dirname(path)
    # check for a valid relacs data directory:
    has_stimuli = False
    has_trace = False
    for fname in ['stimuli.dat', 'stimuli.dat.gz']:
        if os.path.isfile(os.path.join(relacs_dir, fname)):
            has_stimuli = True
    for fname in ['trace-1.raw', 'trace-1.raw.gz']:
        if os.path.isfile(os.path.join(relacs_dir, fname)):
            has_trace = True
    return has_stimuli and has_trace

    
def relacs_files(file_paths, channel):
    """Expand file paths for relacs data to appropriate trace*.raw file names.

    Parameters
    ----------
    file_paths: string or list of strings
        Path to a relacs data directory, a file in a relacs data directory,
        or relacs trace-*.raw or trace-*.raw.gz files.
    channel: int
        The data channel. If negative all channels are selected.
        
    Returns
    -------
    file_paths: list of strings
        List of relacs trace*.raw files.

    Raises
    ------
    ValueError: invalid name of or non-existing relacs trace file
    """
    if not isinstance(file_paths, (list, tuple, np.ndarray)):
        file_paths = [file_paths]
    if len(file_paths) == 1:
        relacs_dir = file_paths[0]
        if not os.path.isdir(relacs_dir):
            relacs_dir = os.path.dirname(file_paths[0])
        if channel < 0:
            file_paths = []
            for k in range(10000):
                fname = os.path.join(relacs_dir, f'trace-{k+1}.raw')
                if os.path.isfile(fname):
                    file_paths.append(fname)
                elif os.path.isfile(fname + '.gz'):
                    file_paths.append(fname + '.gz')
                else:
                    break
        else:
            file_paths[0] = os.path.join(relacs_dir, f'trace-{channel+1}.raw')
    data_paths = []
    for path in file_paths:
        bn = os.path.basename(path)
        if len(bn) <= 5 or bn[0:5] != 'trace' or bn[6:].find('.raw') < 0:
            raise ValueError(f'invalid name {path} of relacs trace file')
        if not os.path.isfile(path):
            path += '.gz'
            if not os.path.isfile(path):
                raise ValueError(f'relacs file {path} does not exist')
        data_paths.append(path)
    return data_paths

        
def load_relacs(file_paths, channel=-1, verbose=0):
    """Load traces (trace-*.raw files) that have been recorded with relacs (www.relacs.net).

    Parameters
    ----------
    file_paths: string or list of strings
        Path to a relacs data directory, a file in a relacs data directory,
        or relacs trace-*.raw files.
    channel: int
        The data channel. If negative all channels are selected.
    verbose: int
        if > 0 show detailed error/warning messages

    Returns
    -------
    data: 1-D or 2-D array
        If `channel` is negative or more than one trace file is specified,
        a 2-D array with data of all channels is returned,
        where first dimension is time and second dimension is channel number.
        Otherwise an 1-D array with the data of that channel is returned.
    samplerate: float
        Sampling rate of the data in Hz
    unit: string
        Unit of the data

    Raises
    ------
    ValueError:
        - Invalid name for relacs trace-*.raw file.
        - Sampling rates of traces differ.
        - Unit of traces differ.
    """
    file_paths = relacs_files(file_paths, channel)
    if len(file_paths) > 1:
        channel = -1
                
    # load trace*.raw files:
    nchannels = len(file_paths)
    data = None
    nrows = 0
    samplerate = None
    unit = ""
    for n, path in enumerate(file_paths):
        if path[-3:] == '.gz':
            with gzip.open(path, 'rb') as sf:
                x = np.frombuffer(sf.read(), dtype=np.float32)
        else:
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


def metadata_relacs(filepath, store_empty=False, first_only=False,
                    lower_keys=False, flat=False, add_sections=False):
    """ Read meta-data of a relacs data set.

    Parameters
    ----------
    filepath: string
        A relacs data directory or a file therein.
    store_empty: bool
        If `False` do not add meta data with empty values.
    first_only: bool
        If `False` only store the first element of a list.
    lower_keys: bool
        Make all keys lower case.
    flat: bool
        Do not make a nested dictionary.
        Use this option also to read in very old relacs metadata with
        ragged left alignment.
    add_sections: bool
        If `True`, prepend keys with sections names separated by
        '.' to make them unique.

    Returns
    -------
    data: nested dict
        Nested dictionary with key-value pairs of the meta data.
    """
    path = filepath
    if isinstance(filepath, (list, tuple, np.ndarray)):
        path = filepath[0]
    relacs_dir = path
    if not os.path.isdir(path):
        relacs_dir = os.path.dirname(path)
    info_path = os.path.join(relacs_dir, 'info.dat')
    if not os.path.exists(info_path):
        return dict(), []
    data = relacs_header(info_path, store_empty, first_only,
                         lower_keys, flat, add_sections)
    return data


def fishgrid_samplerate(filepath):
    """Retrieve the sampling rate from a fishgrid.cfg file.

    Parameters
    ----------
    filepath: string
        Path to a fishgrid data directory, a file in a fishgrid data
        directory, or a fishgrid traces-*.raw file.

    Returns
    -------
    samplerate: float
        Sampling rate in Hertz

    Raises
    ------
    IOError/FileNotFoundError:
        If the fishgrid.cfg file does not exist.
    ValueError:
        fishgrid.cfg file does not contain sampling rate.
    """
    # check for fishgrid data directory:
    fishgrid_dir = filepath
    if not os.path.isdir(filepath):
        fishgrid_dir = os.path.dirname(filepath)

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
    raise ValueError(f'could not retrieve sampling rate from {fishgrid_file}')


def fishgrid_spacings(filepath):
    """Spacing between grid electrodes.

    Parameters
    ----------
    filepath: string
        Path to a fishgrid data directory, a file in a fishgrid data
        directory, or a fishgrid traces-*.raw file.

    Returns
    -------
    grid_dist: list of tuples of floats
        For each grid the distances between rows and columns.
    """
    fishgrid_dir = filepath
    if not os.path.isdir(filepath):
        fishgrid_dir = os.path.dirname(filepath)

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


def fishgrid_grids(filepath):
    """Retrieve grid sizes from a fishgrid.cfg file.

    Parameters
    ----------
    filepath: string
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
    fishgrid_dir = filepath
    if not os.path.isdir(filepath):
        fishgrid_dir = os.path.dirname(filepath)

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


def check_fishgrid(file_paths):
    """Check whether file_paths are valid fishgrid files (https://github.com/bendalab/fishgrid).

    Parameters
    ----------
    file_paths: string or list of strings
        Path to a fishgrid data directory, a file in a fishgrid data directory,
        or fishgrid traces-*.raw files.

    Returns
    -------
    is_fishgrid: bool
        If `file_paths` is a single path, then returns `True` if it is a file in
        a valid fishgrid data directory.
        If `file_paths` are more than one path, then returns `True` if `file_paths`
        are 'trace-*.raw' files in a valid fishgrid data directory.
    """
    path = file_paths
    # file_paths must be traces-*.raw:
    if isinstance(file_paths, (list, tuple, np.ndarray)):
        if len(file_paths) > 1:
            for file in file_paths:
                bn = os.path.basename(file).lower()
                if len(bn) <= 7 or bn[0:7] != 'traces-' or bn[-4:] != '.raw':
                    return False
        path = file_paths[0]
    # fishgrid data directory:
    fishgrid_dir = path
    if not os.path.isdir(path):
        fishgrid_dir = os.path.dirname(path)
    # check for a valid fishgrid data directory:
    return (os.path.isfile(os.path.join(fishgrid_dir, 'fishgrid.cfg')) and
            os.path.isfile(os.path.join(fishgrid_dir, 'traces-grid1.raw')))

    
def fishgrid_files(file_paths, channel, grid_sizes):
    """Expand file paths for fishgrid data to appropriate traces-*.raw file names.

    Parameters
    ----------
    file_paths: string or list of strings
        Path to a fishgrid data directory, a file in a fishgrid data directory,
        or fishgrid traces-*.raw files.
    channel: int
        The data channel. If negative all channels are selected.
    grid_sizes: list of int
        The number of channels of each grid.
        
    Returns
    -------
    file_paths: list of strings
        List of fishgrid traces-*.raw files.

    Raises
    ------
    IndexError:
        Invalid channel.
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
            
    if not isinstance(file_paths, (list, tuple, np.ndarray)):
        file_paths = [file_paths]
    if len(file_paths) == 1:
        fishgrid_dir = file_paths[0]
        if not os.path.isdir(fishgrid_dir):
            fishgrid_dir = os.path.dirname(file_paths[0])
        if grid < 0:
            file_paths = []
            for k in range(10000):
                file = os.path.join(fishgrid_dir, f'traces-grid{k+1}.raw')
                if os.path.isfile(file):
                    file_paths.append(file)
                else:
                    break
        else:
            file_paths[0] = os.path.join(fishgrid_dir, f'traces-grid{grid+1}.raw')
    for path in file_paths:
        bn = os.path.basename(path)
        if len(bn) <= 7 or bn[0:7] != 'traces-' or bn[-4:] != '.raw':
            raise ValueError(f'invalid name {path} of fishgrid traces file')
    return file_paths

        
def load_fishgrid(file_paths, channel=-1, verbose=0):
    """Load traces (traces-grid*.raw files) that have been recorded with fishgrid (https://github.com/bendalab/fishgrid).

    Parameters
    ----------
    file_paths: string or list of string
        Path to a fishgrid data directory, a fishgrid.cfg file,
        or fidhgrid traces-grid*.raw files.
     channel: int
        The data channel. If negative all channels are selected.
    verbose: int
        If > 0 show detailed error/warning messages.

    Returns
    -------
    data: 1-D or 2-D array
        If `channel` is negative or more than one trace file is specified,
        a 2-D array with data of all channels is returned,
        where first dimension is time and second dimension is channel number.
        Otherwise an 1-D array with the data of that channel is returned.
    samplerate: float
        Sampling rate of the data in Hz.
    unit: string
        Unit of the data.
    """
    if not isinstance(file_paths, (list, tuple, np.ndarray)):
        file_paths = [file_paths]
    grids = fishgrid_grids(file_paths[0])
    grid_sizes = [r*c for r,c in grids]
    file_paths = fishgrid_files(file_paths, channel, grid_sizes)
    if len(file_paths) > 1:
        channel = -1
                
    # load traces-grid*.raw files:
    grid_channels = []
    nchannels = 0
    for path in file_paths:
        g = int(os.path.basename(path)[11:].replace('.raw', '')) - 1
        grid_channels.append(grid_sizes[g])
        nchannels += grid_sizes[g]
    data = None
    nrows = 0
    n = 0
    samplerate = None
    if len(file_paths) > 0:
        samplerate = fishgrid_samplerate(file_paths[0])
    unit = "V"
    for path, channels in zip(file_paths, grid_channels):
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


def metadata_fishgrid(filepath, store_empty=False, first_only=False,
                      lower_keys=False, flat=False, add_sections=False):
    """ Read meta-data of a fishgrid data set.

    Parameters
    ----------
    filepath: string
        A fishgrid data directory or a file therein.
    store_empty: bool
        If `False` do not add meta data with empty values.
    first_only: bool
        If `False` only store the first element of a list.
    lower_keys: bool
        Make all keys lower case.
    flat: bool
        Do not make a nested dictionary.
        Use this option also to read in very old relacs metadata with
        ragged left alignment.
    add_sections: bool
        If `True`, prepend keys with sections names separated by
        '.' to make them unique.

    Returns
    -------
    data: nested dict
        Nested dictionary with key-value pairs of the meta data.
    """
    path = filepath
    if isinstance(filepaths, (list, tuple, np.ndarray)):
        path = filepath[0]
    relacs_dir = path
    if not os.path.isdir(path):
        relacs_dir = os.path.dirname(path)
    info_path = os.path.join(relacs_dir, 'fishgrid.cfg')
    if not os.path.exists(info_path):
        return dict(), []
    data = relacs_header(info_path, store_empty, first_only,
                         lower_keys, flat, add_sections)
    return data


def check_container(filepath):
    """Check if file is a generic container file.

    Supported file formats are:

    - python pickle files (.pkl)
    - numpy files (.npz)
    - matlab files (.mat)

    Parameters
    ----------
    filepath: string
        Path of the file to check.
    
    Returns
    -------
    is_container: bool
        `True`, if `filepath` is a supported container format.
    """
    ext = os.path.splitext(filepath)[1]
    return ext.lower() in ('.pkl', '.npz', '.mat')


def load_container(filepath, channel=-1, verbose=0, datakey=None,
                   samplekey=['rate', 'Fs', 'fs'],
                   timekey=['time'], unitkey='unit'):
    """Load data from a generic container file.

    Supported file formats are:

    - python pickle files (.pkl)
    - numpy files (.npz)
    - matlab files (.mat)

    Parameters
    ----------
    filepath: string
        Path of the file to load.
    channel: int
        The data channel. If negative all channels are selected.
    verbose: int
        if > 0 show detailed error/warning messages
    datakey: None, string, or list of string
        Name of the variable holding the data.  If `None` take the
        variable that is an 2D array and has the largest number of
        elements.
    samplekey: string or list of string
        Name of the variable holding the sampling rate.
    timekey: string or list of string
        Name of the variable holding sampling times.
        If no sampling rate is available, the samplingrate is retrieved
        from the sampling times.
    unitkey: string
        Name of the variable holding the unit of the data.
        If `unitkey` is not a valid key, then return `unitkey` as the `unit`.

    Returns
    -------
    data: 1-D or 2-D array of floats
        If `channel` is negative, a 2-D array with data of all channels
        is returned, where first dimension is time and second
        dimension is channel number.  Otherwise an 1-D array with the
        data of that channel is returned.
    samplerate: float
        Sampling rate of the data in Hz.
    unit: string
        Unit of the data.

    Raises
    ------
    IndexError:
        Invalid channel requested
    ValueError:
        Invalid key requested.
    """
    # load data:
    data = {}
    ext = os.path.splitext(filepath)[1]
    if ext == '.pkl':
        import pickle
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
    elif ext == '.npz':
        data = np.load(filepath)
    elif ext == '.mat':
        from scipy.io import loadmat
        data = loadmat(filepath, squeeze_me=True)
    if verbose > 0:
        print( 'loaded %s' % filepath)
    # extract format data:
    if not isinstance(samplekey, (list, tuple, np.ndarray)):
        samplekey = (samplekey,)
    if not isinstance(timekey, (list, tuple, np.ndarray)):
        timekey = (timekey,)
    samplerate = 0.0
    for skey in samplekey:
        if skey in data:
            samplerate = float(data[skey])
            break
    if samplerate == 0.0:
        for tkey in timekey:
            if tkey in data:
                samplerate = 1.0/(data[tkey][1] - data[tkey][0])
                break
    if samplerate == 0.0:
        raise ValueError(f"invalid keys {', '.join(samplekey)} and {', '.join(timekey)} for requesting sampling rate or sampling times")
    unit = ''
    if unitkey in data:
        unit = data[unitkey]
    elif unitkey != 'unit':
        unit = unitkey
    unit = str(unit)
    # get data array:
    raw_data = np.array([])
    if datakey:
        # try data keys:
        if not isinstance(datakey, (list, tuple, np.ndarray)):
            datakey = (datakey,)
        for dkey in datakey:
            if dkey in data:
                raw_data = data[dkey]
                break
        if np.prod(raw_data.shape) == 0:
            raise ValueError(f"invalid key(s) {', '.join(datakey)} for requesting data")
    else:
        # find largest 2D array:
        for d in data:
            if hasattr(data[d], 'shape'):
                if 1 <= len(data[d].shape) <= 2 and \
                   np.prod(data[d].shape) > np.prod(raw_data.shape):
                    raw_data = data[d]
    if np.prod(raw_data.shape) == 0:
        raise ValueError('no data found')
    # make 2D:
    if len(raw_data.shape) == 1:
        raw_data = raw_data.reshape(-1, 1)
    # transpose if necessary:
    if np.argmax(raw_data.shape) > 0:
        raw_data = raw_data.T
    if channel >= 0:
        if channel >= raw_data.shape[1]:
            raise IndexError(f'invalid channel number {channel} requested')
        raw_data = raw_data[:,channel]
    # recode:
    dtype = raw_data.dtype
    data = raw_data.astype(float)
    if dtype == np.dtype('int16'):
        data /= 2**15
    elif dtype == np.dtype('int32'):
        data /= 2**31
    return data, samplerate, unit


def metadata_container(filepath, metadatakey=['metadata', 'info']):
    """ Read meta-data of a container file.

    Parameters
    ----------
    filepath: string
        A container file.
    metadatakey: string or list of string
        Name of the variable holding the metadata.

    Returns
    -------
    data: nested dict
        Nested dictionary with key-value pairs of the meta data.
    """
    # load data:
    data = {}
    ext = os.path.splitext(filepath)[1]
    if ext == '.pkl':
        import pickle
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
    elif ext == '.npz':
        data = np.load(filepath)
    elif ext == '.mat':
        from scipy.io import loadmat
        data = loadmat(filepath, squeeze_me=True)
    if not isinstance(metadatakey, (list, tuple, np.ndarray)):
        metadatakey = (metadatakey,)
    # get single metadata dictionary:
    for mkey in metadatakey:
        if mkey in data:
            return data[mkey]
    # collect all keys starting with metadatakey:
    metadata = {}
    for mkey in metadatakey:
        mkey += '.'
        for dkey in data:
            if dkey[:len(mkey)] == mkey:
                metadata[dkey[len(mkey):]] = data[dkey]
        if len(metadata) > 0:
            return unflatten_metadata(metadata)
    return metadata

    
def load_data(filepath, channel=-1, verbose=0, **kwargs):
    """Load time-series data from a file of arbitrary format.

    Parameters
    ----------
    filepath: string or list of strings
        The full path and name of the file to load. For some file
        formats several files can be provided in a list.
    channel: int
        The data channel. If negative all channels are selected.
    verbose: int
        If > 0 show detailed error/warning messages.
    **kwargs: dict
        Further keyword arguments that are passed on to the 
        format specific loading functions.

    Returns
    -------
    data: 1-D or 2-D array
        If `channel` is negative, a 2-D array with data of all
        channels is returned, where first dimension is time and second
        dimension is channel number.  Otherwise an 1-D array with the
        data of that channel is returned.
    samplerate: float
        Sampling rate of the data in Hz.
    unit: string
        Unit of the data.

    Raises
    ------
    ValueError:
        Input argument `filepath` is empty string or list.
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
        if isinstance(filepath, (list, tuple, np.ndarray)):
            filepath = filepath[0]
        if check_container(filepath):
            return load_container(filepath, channel, verbose=verbose, **kwargs)
        else:
            data, samplerate = load_audio(filepath, verbose)
            if channel >= 0:
                if channel >= data.shape[1]:
                    raise IndexError(f'invalid channel number {channel} requested')
                data = data[:, channel]
            unit = 'a.u.'
        return data, samplerate, unit


def metadata(filepath, store_empty=False, first_only=False, **kwargs):
    """ Read meta-data from a data file.

    Parameters
    ----------
    filepath: string or list of strings
        The full path and name of the file to load. For some file
        formats several files can be provided in a list.
    store_empty: bool
        If `False` do not add meta data with empty values.
    first_only: bool
        If `False` only store the first element of a list.
    **kwargs: dict
        Further keyword arguments that are passed on to the 
        format specific loading functions.

    Returns
    -------
    meta_data: nested dict
        Meta data contained in the file.  Keys of the nested
        dictionaries are always strings.  If the corresponding
        values are dictionaries, then the key is the section name
        of the metadata contained in the dictionary. All other
        types of values are values for the respective key. In
        particular they are strings, or list of strings. But other
        simple types like ints or floats are also allowed.
    """
    if check_relacs(filepath):
        return metadata_relacs(filepath, store_empty, first_only, **kwargs)
    elif check_fishgrid(filepath):
        return metadata_fishgrid(filepath, store_empty, first_only, **kwargs)
    else:
        if isinstance(filepath, (list, tuple, np.ndarray)):
            filepath = filepath[0]
        if check_container(filepath):
            return metadata_container(filepath, **kwargs)
        else:
            return audioio_metadata(filepath, store_empty)


def markers(filepath):
    """ Read markers of a data file.

    Parameters
    ----------
    filepath: string or file handle
        The data file.

    Returns
    -------
    locs: 2-D array of ints
        Marker positions (first column) and spans (second column)
        for each marker (rows).
    labels: 2-D array of string objects
        Labels (first column) and texts (second column)
        for each marker (rows).
    """
    return audioio_markers(filepath)


class DataLoader(AudioLoader):
    """Buffered reading of time-series data for random access of the data in the file.
    
    This allows for reading very large data files that do not fit into
    memory.  An `DataLoader` instance can be used like a huge
    read-only numpy array, i.e.
    ```
    data = DataLoader('path/to/data/file.dat')
    x = data[10000:20000,0]
    ```
    The first index specifies the frame, the second one the channel.

    `DataLoader` first determines the format of the data file and then
    opens the file (first line). It then reads data from the file as
    necessary for the requested data (second line).

    Supported file formats are

    - relacs trace*.raw files (www.relacs.net)
    - fishgrid traces-*.raw files
    - audio files via `audioio` package

    Reading sequentially through the file is always possible. If
    previous data are requested, then the file is read from the
    beginning. This might slow down access to previous data
    considerably. Use the `backsize` argument to the open functions to
    make sure some data are loaded before the requested frame. Then a
    subsequent access to the data within backsize seconds before that
    frame can still be handled without the need to reread the file
    from the beginning.

    Usage:
    ------
    ```
    import thunderfish.dataloader as dl
    with dl.open_data(filepath, 60.0, 10.0) as data:
        # do something with the content of the file:
        x = data[0:10000,0]
        y = data[10000:20000,0]
        z = x + y
    ```
    
    Normal open and close:
    ```
    data = dl.DataLoader(filepath, 60.0)
    x = data[:,:]  # read the whole file
    data.close()
    ```    
    that is the same as:
    ```
    data = dl.DataLoader()
    data.open(filepath, 60.0)
    ```

    Attributes
    ----------
    samplerate: float
        The sampling rate of the data in Hertz.
    channels: int
        The number of channels that are read in.
    channel: int
        The channel of which the trace is returned.
        If negative, all channels are returned.
    frames: int
        The number of frames in the file.
    shape: tuple
        Number of frames and channels of the data.
    ndim: int
        Number of dimensions. One of only a single channel is read in,
        two otherwise (frames, channels).
    unit: string
        Unit of the data.
    ampl_min: float
        Minimum amplitude the file format supports.
    ampl_max: float
        Maximum amplitude the file format supports.

    Methods
    -------

    - `len()`: the number of frames
    - `open()`: open a data file.
    - `open_*()`: open a data file of a specific format.
    - `close()`: close the file.
    - `metadata()`: metadata of the file.
    - `markers()`: markers of the file.

    """

    def __init__(self, filepath=None, buffersize=10.0, backsize=0.0,
                 verbose=0, channel=-1):
        """Initialize the DataLoader instance.

        If filepath is not None open the file.

        Parameters
        ----------
        filepath: string
            Name of the file.
        buffersize: float
            Size of internal buffer in seconds.
        backsize: float
            Part of the buffer to be loaded before the requested start index in seconds.
        verbose: int
            If > 0 show detailed error/warning messages.
        channel: int
            The single channel to be worked on or all channels if negative.
        """
        super(DataLoader, self).__init__(None, buffersize, backsize, verbose)
        if filepath is not None:
            self.open(filepath, buffersize, backsize, verbose, channel)

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
    def open_relacs(self, file_paths, buffersize=10.0, backsize=0.0,
                    verbose=0, channel=-1):
        """Open relacs data files (www.relacs.net) for reading.

        Parameters
        ----------
        file_paths: string or list of string
            Path to a relacs data directory, a relacs stimuli.dat file, a relacs info.dat file,
            or relacs trace-*.raw files.
        buffersize: float
            Size of internal buffer in seconds.
        backsize: float
            Part of the buffer to be loaded before the requested start index in seconds.
        verbose: int
            If > 0 show detailed error/warning messages.
        channel: int
            The requested data channel. If negative all channels are selected.

        Raises
        ------
        ValueError: .gz files not supported.
        """
        self.verbose = verbose
        
        if self.sf is not None:
            self._close_relacs()

        file_paths = relacs_files(file_paths, channel)

        # open trace files:
        self.sf = []
        self.frames = None
        self.samplerate = None
        self.unit = ""
        self.filepath = None
        if len(file_paths) > 0:
            self.filepath = os.path.dirname(file_paths[0])
        for path in file_paths:
            if path[-3:] == '.gz':
                raise ValueError('.gz files not supported')
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
        self.ndim = len(self.shape)
        self.buffersize = int(buffersize*self.samplerate)
        self.backsize = int(backsize*self.samplerate)
        self._init_buffer()
        self.offset = 0
        self.close = self._close_relacs
        self.load_buffer = self._load_buffer_relacs
        self.metadata = self._metadata_relacs
        self.markers = self._markers_relacs
        self.ampl_min = -np.inf
        self.ampl_max = +np.inf
        return self

    def _close_relacs(self):
        """Close the relacs data files.
        """
        if self.sf is not None:
            for file in self.sf:
                file.close()
            self.sf = None

    def _load_buffer_relacs(self, r_offset, r_size, buffer):
        """Load new data from relacs data file.

        Parameters
        ----------
        r_offset: int
           First frame to be read from file.
        r_size: int
           Number of frames to be read from file.
        buffer: ndarray
           Buffer where to store the loaded data.
        """
        for i, file in enumerate(self.sf):
            file.seek(r_offset*4)
            data = file.read(r_size*4)
            buffer[:, i] = np.fromstring(data, dtype=np.float32)
        

    def _metadata_relacs(self, store_empty=False, first_only=False):
        """ Read meta-data of a relacs data set.
        """
        info_path = os.path.join(self.filepath, 'info.dat')
        if not os.path.exists(info_path):
            return dict()
        data = relacs_header(info_path, store_empty, first_only)
        return data


    def _markers_relacs(self):
        """ Read markers of a relacs data set.
        """
        # Not implemented yet!
        return None, None

    
    # fishgrid interface:        
    def open_fishgrid(self, file_paths, buffersize=10.0, backsize=0.0,
                      verbose=0, channel=-1):
        """Open fishgrid data files (https://github.com/bendalab/fishgrid) for reading.

        Parameters
        ----------
        file_paths: string or list of string
            Path to a fishgrid data directory, a fishgrid.cfg file,
            or fishgrid trace-*.raw files.
        buffersize: float
            Size of internal buffer in seconds.
        backsize: float
            Part of the buffer to be loaded before the requested start index in seconds.
        verbose: int
            If > 0 show detailed error/warning messages.
        channel: int
            The requested data channel. If negative all channels are selected.
        """
        self.verbose = verbose
        
        if self.sf is not None:
            self._close_fishgrid()

        if not isinstance(file_paths, (list, tuple, np.ndarray)):
            file_paths = [file_paths]
        grids = fishgrid_grids(file_paths[0])
        grid_sizes = [r*c for r,c in grids]
        file_paths = fishgrid_files(file_paths, channel, grid_sizes)
        self.filepath = None
        if len(file_paths) > 0:
            self.filepath = os.path.dirname(file_paths[0])

        # open grid files:
        self.channels = 0
        for path in file_paths:
            g = int(os.path.basename(path)[11:].replace('.raw', '')) - 1
            self.channels += grid_sizes[g]
        self.sf = []
        self.grid_channels = []
        self.grid_offs = []
        offs = 0
        self.frames = None
        self.samplerate = None
        if len(file_paths) > 0:
            self.samplerate = fishgrid_samplerate(file_paths[0])
        self.unit = "V"
        for path in file_paths:
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
        self.ndim = len(self.shape)
        self.buffersize = int(buffersize*self.samplerate)
        self.backsize = int(backsize*self.samplerate)
        self._init_buffer()
        self.offset = 0
        self.close = self._close_fishgrid
        self.load_buffer = self._load_buffer_fishgrid
        self.metadata = self._metadata_fishgrid
        self.markers = self._markers_fishgrid
        self.ampl_min = -np.inf
        self.ampl_max = +np.inf
        return self

    def _close_fishgrid(self):
        """Close the fishgrid data files.
        """
        if self.sf is not None:
            for file in self.sf:
                file.close()
            self.sf = None

    def _load_buffer_fishgrid(self, r_offset, r_size, buffer):
        """Load new data from relacs data file.

        Parameters
        ----------
        r_offset: int
           First frame to be read from file.
        r_size: int
           Number of frames to be read from file.
        buffer: ndarray
           Buffer where to store the loaded data.
        """
        for file, gchannels, goffset in zip(self.sf, self.grid_channels, self.grid_offs):
            file.seek(r_offset*4*gchannels)
            data = file.read(r_size*4*gchannels)
            buffer[:, goffset:goffset+gchannels] = np.fromstring(data, dtype=np.float32).reshape((-1, gchannels))
        

    def _metadata_fishgrid(self, store_empty=False, first_only=False):
        """ Read meta-data of a fishgrid data set.
        """
        info_path = os.path.join(self.filepath, 'fishgrid.cfg')
        if not os.path.exists(info_path):
            return dict()
        data = relacs_header(info_path, store_empty, first_only)
        return data


    def _markers_fishgrid(self):
        """ Read markers of a fishgrid data set.
        """
        # Not implemented yet!
        return None, None


    def open(self, filepath, buffersize=10.0, backsize=0.0,
             verbose=0, channel=-1):
        """Open file with time-series data for reading.

        Parameters
        ----------
        filepath: string or list of string
            Path to a data files or directory.
        buffersize: float
            Size of internal buffer in seconds.
        backsize: float
            Part of the buffer to be loaded before the requested start index in seconds.
        verbose: int
            If > 0 show detailed error/warning messages.
        channel: int
            The requested data channel. If negative all channels are selected.
        """
        if check_relacs(filepath):
            self.open_relacs(filepath, buffersize, backsize, verbose, channel)
        elif check_fishgrid(filepath):
            self.open_fishgrid(filepath, buffersize, backsize, verbose, channel)
        else:
            if isinstance(filepath, (list, tuple, np.ndarray)):
                filepath = filepath[0]
            if check_container(filepath):
                raise ValueError('file format not supported')
            super(DataLoader, self).open(filepath, buffersize, backsize, verbose)
            if channel > self.channels:
                raise IndexError(f'invalid channel number {channel}')
            self.channel = channel
            if self.channel >= 0:
                self.shape = (self.frames,)
            else:
                self.shape = (self.frames, self.channels)
            self.ndim = len(self.shape)
            self.unit = 'a.u.'
        return self


def demo(filepath, plot=False, channel=-1):
    print("try load_data:")
    data, samplerate, unit = load_data(filepath, channel, verbose=2)
    if plot:
        time = np.arange(len(data))/samplerate
        if channel < 0:
            for c in range(data.shape[1]):
                plt.plot(time, data[:,c])
        else:
            plt.plot(time, data)
        plt.xlabel('Time [s]')
        plt.ylabel('[' + unit + ']')
        plt.show()
        return

    print('')
    print("try DataLoader for channel=%d:" % channel)
    with DataLoader(filepath, channel, 2.0, 1.0, 1) as data:
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
                

    
def main(*cargs):
    """Call demo with command line arguments.

    Parameters
    ----------
    cargs: list of strings
        Command line arguments as provided by sys.argv[1:]
    """
    import argparse
    parser = argparse.ArgumentParser(description=
                                     'Checking thunderfish.dataloader module.')
    parser.add_argument('-p', dest='plot', action='store_true',
                        help='plot loaded data')
    parser.add_argument('-c', dest='channel', default=-1, type=int,
                        help='channel to be loaded')
    parser.add_argument('file', nargs=1, default='', type=str,
                        help='name of data file')
    args = parser.parse_args(cargs)
    demo(args.file[0], args.plot, args.channel)
    

if __name__ == "__main__":
    main(*sys.argv[1:])
