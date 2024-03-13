"""Load time-series data from files.

```
data, samplingrate, unit = load_data('data/file.wav')
```
Loads the whole time-series from the file as a numpy array of floats.
First dimension is frames, second is channels. In contrast to the
`audioio.load_audio()` function, the values of the data array are not
restricted between -1 and 1. They can assume any value with the unit
that is also returned.

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
- audio files via [`audioio`](https://github.com/bendalab/audioio) package
- relacs trace*.raw files (https://www.relacs.net)
- fishgrid traces-*.raw files (https://github.com/bendalab/fishgrid)


## Metadata

Many file formats allow to store metadata that further describe the
stored time series data. We handle them as nested dictionary of key-value
pairs. Load them with the `metadata()` function:
```
metadata = metadata('data/file.mat')
```

## Markers

Some file formats also allow to store markers that mark specific
positions in the time series data. Load marker positions and spans (in
the 2-D array `locs`) and label and text strings (in the 2-D array
`labels`) with the `markers()` function:
```
locs, labels = markers('data.wav')
```

## Aditional, format specific functions

- `relacs_samplerate_unit()`: retrieve sampling rate and unit from a relacs stimuli.dat file.
- `relacs_header()`: read key-value pairs from relacs *.dat file headers.
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
from audioio import get_number_unit, get_number, get_int, get_bool, get_gain
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
        with gzip.open(stimuli_file, 'r', encoding='latin-1') as sf:
            for line in sf:
                line = line.strip()
                if len(line) == 0 or line[0] != '#':
                    break
                lines.append(line)
    else:
        with open(stimuli_file, 'r', encoding='latin-1') as sf:
            for line in sf:
                line = line.strip()
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
        with gzip.open(filepath, 'r', encoding='latin-1') as sf:
            for line in sf:
                line = line.strip()
                if len(line) == 0 or line[0] != '#':
                    break
                lines.append(line)
    else:
        with open(filepath, 'r', encoding='latin-1') as sf:
            for line in sf:
                line = line.strip()
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

    
def relacs_files(file_paths):
    """Expand file paths for relacs data to appropriate trace*.raw file names.

    Parameters
    ----------
    file_paths: string or list of strings
        Path to a relacs data directory, a file in a relacs data directory,
        or relacs trace-*.raw or trace-*.raw.gz files.
        
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
        file_paths = []
        for k in range(10000):
            fname = os.path.join(relacs_dir, f'trace-{k+1}.raw')
            if os.path.isfile(fname):
                file_paths.append(fname)
            elif os.path.isfile(fname + '.gz'):
                file_paths.append(fname + '.gz')
            else:
                break
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

        
def load_relacs(file_paths):
    """Load traces (trace-*.raw files) that have been recorded with relacs (www.relacs.net).

    Parameters
    ----------
    file_paths: string or list of strings
        Path to a relacs data directory, a file in a relacs data directory,
        or relacs trace-*.raw files.

    Returns
    -------
    data: 2-D array
        All data traces as an 2-D numpy array, even for single channel data.
        First dimension is time, second is channel.
    samplerate: float
        Sampling rate of the data in Hz
    unit: string
        Unit of the data
    amax: float
        Maximum amplitude of data range.

    Raises
    ------
    ValueError:
        - Invalid name for relacs trace-*.raw file.
        - Sampling rates of traces differ.
        - Unit of traces differ.
    """
    file_paths = relacs_files(file_paths)
    # load trace*.raw files:
    nchannels = len(file_paths)
    data = None
    nrows = 0
    samplerate = None
    unit = ""
    for c, path in enumerate(sorted(file_paths)):
        if path[-3:] == '.gz':
            with gzip.open(path, 'rb') as sf:
                x = np.frombuffer(sf.read(), dtype=np.float32)
        else:
            x = np.fromfile(path, np.float32)
        if data is None:
            nrows = len(x)
            data = np.zeros((nrows, nchannels))
        n = min(len(x), nrows)
        data[:n,c] = x[:n]
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
    return data, samplerate, unit, np.inf


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


def fishgrid_spacings(metadata, unit='m'):
    """Spacing between grid electrodes.

    Parameters
    ----------
    metadata: dict
        Fishgrid metadata obtained from `metadata_fishgrid()`.
    unit: str
        Unit in which to return the spacings.

    Returns
    -------
    grid_dist: list of tuple of float
        For each grid the distances between rows and columns in `unit`.
    """
    grids_dist = []
    for k in range(4):
        row_dist = get_number(metadata, unit, f'RowDistance{k+1}', default=0)
        col_dist = get_number(metadata, unit, f'ColumnDistance{k+1}', default=0)
        rows = get_int(metadata, f'Rows{k+1}', default=0)
        cols = get_int(metadata, f'Columns{k+1}', default=0)
        if get_bool(metadata, f'Used{k+1}', default=False) or \
           cols > 0 and rows > 0:
            grids_dist.append((row_dist, col_dist))
    return grids_dist


def fishgrid_grids(metadata):
    """Retrieve grid sizes from a fishgrid.cfg file.

    Parameters
    ----------
    metadata: dict
        Fishgrid metadata obtained from `metadata_fishgrid()`.

    Returns
    -------
    grids: list of tuple of int
        For each grid the number of rows and columns.
    """
    grids = []
    for k in range(4):
        rows = get_int(metadata, f'Rows{k+1}', default=0)
        cols = get_int(metadata, f'Columns{k+1}', default=0)
        if get_bool(metadata, f'Used{k+1}', default=False) or \
           cols > 0 and rows > 0:
            grids.append((rows, cols))
    return grids


def check_fishgrid(file_paths):
    """Check whether file_paths are valid fishgrid files (https://github.com/bendalab/fishgrid).

    Parameters
    ----------
    file_paths: string or list of strings
        Path to a fishgrid data directory, a file in a fishgrid data directory,
        or fishgrid traces*.raw files.

    Returns
    -------
    is_fishgrid: bool
        If `file_paths` is a single path, then returns `True` if it is a file in
        a valid fishgrid data directory.
        If `file_paths` are more than one path, then returns `True` if `file_paths`
        are 'trace*.raw' files in a valid fishgrid data directory.
    """
    path = file_paths
    # file_paths must be traces*.raw:
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
            (os.path.isfile(os.path.join(fishgrid_dir, 'traces-grid1.raw')) or
             os.path.isfile(os.path.join(fishgrid_dir, 'traces.raw'))))

    
def fishgrid_trace_files(file_paths):
    """Expand file paths for fishgrid data to appropriate traces*.raw file names.

    Parameters
    ----------
    file_paths: string or list of strings
        Path to a fishgrid data directory, a file in a fishgrid data directory,
        or fishgrid traces*.raw files.
        
    Returns
    -------
    file_paths: list of strings
        List of fishgrid traces*.raw files.

    Raises
    ------
    FileNotFoundError:
        Invalid fishgrid file.
    """
    # find grids:
    if not isinstance(file_paths, (list, tuple, np.ndarray)):
        file_paths = [file_paths]
    if len(file_paths) == 1:
        fishgrid_dir = file_paths[0]
        if not os.path.isdir(fishgrid_dir):
            fishgrid_dir = os.path.dirname(file_paths[0])
        file_paths = []
        for k in range(10000):
            file = os.path.join(fishgrid_dir, f'traces-grid{k+1}.raw')
            if os.path.isfile(file):
                file_paths.append(file)
            else:
                break
        if len(file_paths) == 0:
            file = os.path.join(fishgrid_dir, f'traces.raw')
            if os.path.isfile(file):
                file_paths.append(file)
    for path in file_paths:
        bn = os.path.basename(path)
        if len(bn) <= 6 or bn[0:6] != 'traces' or bn[-4:] != '.raw':
            raise FileNotFoundError(f'invalid name {path} of fishgrid traces file')
    return sorted(file_paths)

        
def load_fishgrid(file_paths):
    """Load traces (traces-grid*.raw files) that have been recorded with fishgrid (https://github.com/bendalab/fishgrid).

    Parameters
    ----------
    file_paths: string or list of string
        Path to a fishgrid data directory, a fishgrid.cfg file,
        or fidhgrid traces-grid*.raw files.

    Returns
    -------
    data: 2-D array
        All data traces as an 2-D numpy array, even for single channel data.
        First dimension is time, second is channel.
    samplerate: float
        Sampling rate of the data in Hz.
    unit: string
        Unit of the data.
    amax: float
        Maximum amplitude of data range.

    Raises
    ------
    FileNotFoundError:
        Invalid or not existing fishgrid files.
    """
    if not isinstance(file_paths, (list, tuple, np.ndarray)):
        file_paths = [file_paths]
    file_paths = fishgrid_trace_files(file_paths)
    if len(file_paths) == 0:
        raise FileNotFoundError(f'no fishgrid files specified')
    md = metadata_fishgrid(file_paths[0])
    grids = fishgrid_grids(md)
    grid_sizes = [r*c for r,c in grids]
                
    # load traces-grid*.raw files:
    grid_channels = []
    nchannels = 0
    for g, path in enumerate(file_paths):
        grid_channels.append(grid_sizes[g])
        nchannels += grid_sizes[g]
    data = None
    nrows = 0
    c = 0
    samplerate = get_number(md, 'Hz', 'AISampleRate')
    for path, channels in zip(file_paths, grid_channels):
        x = np.fromfile(path, np.float32).reshape((-1, channels))
        if data is None:
            nrows = len(x)
            data = np.zeros((nrows, nchannels))
        n = min(len(x), nrows)
        data[:n,c:c+channels] = x[:n,:]
        c += channels
    amax, unit = get_number_unit(md, 'AIMaxVolt')
    return data, samplerate, unit, amax


def metadata_fishgrid(filepath):
    """ Read meta-data of a fishgrid data set.

    Parameters
    ----------
    filepath: string
        A fishgrid data directory or a file therein.

    Returns
    -------
    data: nested dict
        Nested dictionary with key-value pairs of the meta data.
    """
    path = filepath
    if isinstance(filepath, (list, tuple, np.ndarray)):
        path = filepath[0]
    if 'trace' in os.path.basename(path):
        path = os.path.dirname(path)
    if os.path.isdir(path):
        path = os.path.join(path, 'fishgrid.cfg')
    # read in header from file:
    lines = []
    if os.path.isfile(path + '.gz'):
        info_path += '.gz'
    if not os.path.exists(path):
        return {}
    if path[-3:] == '.gz':
        with gzip.open(path, 'r', encoding='latin-1') as sf:
            for line in sf:
                lines.append(line)
    else:
        with open(path, 'r', encoding='latin-1') as sf:
            for line in sf:
                lines.append(line)
    # parse:
    data = {}
    cdatas = [data]
    ident_offs = None
    ident = None
    old_style = False
    grid1 = False
    for line in lines:
        if len(line.strip()) == 0:
            continue
        if line[0] == '*':
            key = line[1:].strip()
            data[key] = {}
            cdatas = [data, data[key]]
        elif '----' in line:
            old_style = True
            key = line.strip().strip(' -').replace('&', '')
            if key.upper() == 'SETUP':
                key = 'Grid 1'
            grid1 = (key == 'Grid 1')
            cdatas = cdatas[:2]
            cdatas[1][key] = {}
            cdatas.append(cdatas[1][key])
        else:
            words = line.split(':')
            key = words[0].strip().strip('"')
            value = None
            if len(words) > 1 and len(words[1].strip()) > 0:
                value = ':'.join(words[1:]).strip().strip('"')
            if old_style:
                if value is None:
                    cdatas = cdatas[:3]
                    cdatas[2][key] = {}
                    cdatas.append(cdatas[2][key])            
                else:
                    if grid1 and key[-1] != '1':
                        key = key + '1'
                    cdatas[-1][key] = value
            else:
                # get section level:
                level = 0
                nident = len(line) - len(line.lstrip())
                if ident_offs is None:
                    ident_offs = nident
                elif ident is None:
                    if nident > ident_offs:
                        ident = nident - ident_offs
                        level = 1
                else:
                    level = (nident - ident_offs)//ident
                # close sections:
                cdatas = cdatas[:2 + level]
                if value is None:
                    # new section:
                    cdatas[-1][key] = {}
                    cdatas.append(cdatas[-1][key])
                else:
                    # key-value pair:
                    cdatas[-1][key] = value
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


def extract_container_data(data_dict, datakey=None,
                           samplekey=['rate', 'Fs', 'fs'],
                           timekey=['time'], amplkey=['amax'], unitkey='unit'):
    """Extract data from dictionary loaded from a container file.

    Parameters
    ----------
    data_dict: dict
        Dictionary of the data items contained in the container.
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
    amplkey: string or list of string
        Name of the variable holding the amplitude range of the data.
    unitkey: string
        Name of the variable holding the unit of the data.
        If `unitkey` is not a valid key, then return `unitkey` as the `unit`.

    Returns
    -------
    data: 2-D array of floats
        All data traces as an 2-D numpy array, even for single channel data.
        First dimension is time, second is channel.
    samplerate: float
        Sampling rate of the data in Hz.
    unit: string
        Unit of the data.
    amax: float
        Maximum amplitude of data range in `unit`.

    Raises
    ------
    ValueError:
        Invalid key requested.
    """
    # extract format data:
    if not isinstance(samplekey, (list, tuple, np.ndarray)):
        samplekey = (samplekey,)
    if not isinstance(timekey, (list, tuple, np.ndarray)):
        timekey = (timekey,)
    if not isinstance(amplkey, (list, tuple, np.ndarray)):
        amplkey = (amplkey,)
    samplerate = 0.0
    for skey in samplekey:
        if skey in data_dict:
            samplerate = float(data_dict[skey])
            break
    if samplerate == 0.0:
        for tkey in timekey:
            if tkey in data_dict:
                samplerate = 1.0/(data_dict[tkey][1] - data_dict[tkey][0])
                break
    if samplerate == 0.0:
        raise ValueError(f"invalid keys {', '.join(samplekey)} and {', '.join(timekey)} for requesting sampling rate or sampling times")
    amax = 1.0
    for akey in amplkey:
        if akey in data_dict:
            amax = float(data_dict[akey])
            break
    unit = ''
    if unitkey in data_dict:
        unit = data_dict[unitkey]
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
            if dkey in data_dict:
                raw_data = data_dict[dkey]
                break
        if np.prod(raw_data.shape) == 0:
            raise ValueError(f"invalid key(s) {', '.join(datakey)} for requesting data")
    else:
        # find largest 2D array:
        for d in data_dict:
            if hasattr(data_dict[d], 'shape'):
                if 1 <= len(data_dict[d].shape) <= 2 and \
                   np.prod(data_dict[d].shape) > np.prod(raw_data.shape):
                    raw_data = data_dict[d]
    if np.prod(raw_data.shape) == 0:
        raise ValueError('no data found')
    # make 2D:
    if len(raw_data.shape) == 1:
        raw_data = raw_data.reshape(-1, 1)
    # transpose if necessary:
    if np.argmax(raw_data.shape) > 0:
        raw_data = raw_data.T
    # recode:
    if raw_data.dtype == np.dtype('int16'):
        data = raw_data.astype('float32')
        data *= amax/2**15
    elif raw_data.dtype == np.dtype('int32'):
        data = raw_data.astype(float)
        data *= amax/2**31
    elif raw_data.dtype == np.dtype('int64'):
        data = raw_data.astype(float)
        data *= amax/2**63
    else:
        data = raw_data
    return data, samplerate, unit, amax


def load_container(filepath, datakey=None,
                   samplekey=['rate', 'Fs', 'fs'],
                   timekey=['time'], amplkey=['amax'], unitkey='unit'):
    """Load data from a generic container file.

    Supported file formats are:

    - python pickle files (.pkl)
    - numpy files (.npz)
    - matlab files (.mat)

    Parameters
    ----------
    filepath: string
        Path of the file to load.
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
    amplkey: string
        Name of the variable holding the amplitude range of the data.
    unitkey: string
        Name of the variable holding the unit of the data.
        If `unitkey` is not a valid key, then return `unitkey` as the `unit`.

    Returns
    -------
    data: 2-D array of floats
        All data traces as an 2-D numpy array, even for single channel data.
        First dimension is time, second is channel.
    samplerate: float
        Sampling rate of the data in Hz.
    unit: string
        Unit of the data.
    amax: float
        Maximum amplitude of data range.

    Raises
    ------
    ValueError:
        Invalid key requested.
    """
    # load data:
    data_dict = {}
    ext = os.path.splitext(filepath)[1]
    if ext == '.pkl':
        import pickle
        with open(filepath, 'rb') as f:
            data_dict = pickle.load(f)
    elif ext == '.npz':
        data_dict = np.load(filepath)
    elif ext == '.mat':
        from scipy.io import loadmat
        data_dict = loadmat(filepath, squeeze_me=True)
    return extract_container_data(data_dict, datakey, samplekey,
                                  timekey, amplkey, unitkey)


def extract_container_metadata(data_dict, metadatakey=['metadata', 'info']):
    """ Extract metadata from dictionary loaded from a container file.

    Parameters
    ----------
    data_dict: dict
        Dictionary of the data items contained in the container.
    metadatakey: string or list of string
        Name of the variable holding the metadata.

    Returns
    -------
    metadata: nested dict
        Nested dictionary with key-value pairs of the meta data.
    """
    if not isinstance(metadatakey, (list, tuple, np.ndarray)):
        metadatakey = (metadatakey,)
    # get single metadata dictionary:
    for mkey in metadatakey:
        if mkey in data_dict:
            return data_dict[mkey]
    # collect all keys starting with metadatakey:
    metadata = {}
    for mkey in metadatakey:
        mkey += '__'
        for dkey in data_dict:
            if dkey[:len(mkey)] == mkey:
                v = data_dict[dkey]
                if hasattr(v, 'size') and v.ndim == 0:
                    v = v.item()
                metadata[dkey[len(mkey):]] = v
        if len(metadata) > 0:
            return unflatten_metadata(metadata, sep='__')
    return metadata


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
    metadata: nested dict
        Nested dictionary with key-value pairs of the meta data.
    """
    # load data:
    data_dict = {}
    ext = os.path.splitext(filepath)[1]
    if ext == '.pkl':
        import pickle
        with open(filepath, 'rb') as f:
            data_dict = pickle.load(f)
    elif ext == '.npz':
        data_dict = np.load(filepath)
    elif ext == '.mat':
        from scipy.io import loadmat
        data_dict = loadmat(filepath, squeeze_me=True)
    return extract_container_metadata(data_dict, metadatakey)


def load_audioio(filepath, verbose=0, gainkey=['gain'], sep='.'):
    """Load data from an audio file.

    See the
    [`load_audio()`](https://bendalab.github.io/audioio/api/audioloader.html#audioio.audioloader.load_audio)
    function of the [`audioio`](https://github.com/bendalab/audioio)
    package for more infos.

    Parameters
    ----------
    filepath: str
        Path of the file to load.
    verbose: int
        If > 0 show detailed error/warning messages.
    gainkey: str or list of str
        Key in the file's metadata that holds some gain information.
        If found, the data will be multiplied with the gain,
        and if available, the corresponding unit is returned.
        See the [audioio.get_gain()](https://bendalab.github.io/audioio/api/audiometadata.html#audioio.audiometadata.get_gain) function for details.
    sep: str
        String that separates section names in `gainkey`.

    Returns
    -------
    data: 2-D array of floats
        All data traces as an 2-D numpy array, even for single channel data.
        First dimension is time, second is channel.
    samplerate: float
        Sampling rate of the data in Hz.
    unit: string
        Unit of the data if found in the metadata (see `gainkey`),
        otherwise "a.u.".
    amax: float
        Maximum amplitude of data range.
    """
    # get gain:
    md = audioio_metadata(filepath)
    amax, unit = get_gain(md, gainkey, sep)
    # load data:
    data, samplerate = load_audio(filepath, verbose)
    if amax != 1.0:
        data *= amax
    return data, samplerate, unit, amax

    
def load_data(filepath, verbose=0, **kwargs):
    """Load time-series data from a file.

    Parameters
    ----------
    filepath: string or list of strings
        The full path and name of the file to load. For some file
        formats several files can be provided in a list.
    verbose: int
        If > 0 show detailed error/warning messages.
    **kwargs: dict
        Further keyword arguments that are passed on to the 
        format specific loading functions.

    Returns
    -------
    data: 2-D array
        All data traces as an 2-D numpy array, even for single channel data.
        First dimension is time, second is channel.
    samplerate: float
        Sampling rate of the data in Hz.
    unit: string
        Unit of the data.
    amax: float
        Maximum amplitude of data range.

    Raises
    ------
    ValueError:
        Input argument `filepath` is empty string or list.
    """
    def print_verbose(verbose, data, rate, unit, amax, filepath, lib):
        if verbose > 0:
            if isinstance(filepath, (list, tuple, np.ndarray)):
                filepath = filepath[0]
            print(f'loaded data from file "{filepath}" using open_{lib}()')
            if verbose > 1:
                print(f'  sampling rate: {rate:g} Hz')
                print(f'  channels     : {data.shape[1]}')
                print(f'  frames       : {len(data)}')
                print(f'  unit         : {amax:g}{unit}')
        
    # check values:
    if len(filepath) == 0:
        raise ValueError('input argument filepath is empty string or list.')

    # load data:
    if check_relacs(filepath):
        data, rate, unit, amax = load_relacs(filepath)
        print_verbose(verbose, data, rate, unit, amax, filepath, 'relacs')
        return data, rate, unit, amax
    elif check_fishgrid(filepath):
        data, rate, unit, amax = load_fishgrid(filepath)
        print_verbose(verbose, data, rate, unit, amax, filepath, 'fishgrid')
        return data, rate, unit, amax
    else:
        if isinstance(filepath, (list, tuple, np.ndarray)):
            filepath = filepath[0]
        if check_container(filepath):
            data, rate, unit, amax = load_container(filepath, **kwargs)
            print_verbose(verbose, data, rate, unit, amax, filepath,
                          'container')
            return data, rate, unit, amax
        else:
            data, rate, unit, amax = load_audioio(filepath, verbose, **kwargs)
            return data, rate, unit, amax


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
    memory.  A `DataLoader` instance can be used like a huge
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

    - audio files via `audioio` package
    - relacs trace*.raw files (www.relacs.net)
    - fishgrid traces-*.raw files

    Reading sequentially through the file is always possible. If
    previous data are requested, then the file is read from the
    beginning. This might slow down access to previous data
    considerably. Use the `backsize` argument to the open functions to
    make sure some data are loaded before the requested frame. Then a
    subsequent access to the data within `backsize` seconds before that
    frame can still be handled without the need to reread the file
    from the beginning.

    Usage:
    ------
    ```
    import thunderfish.dataloader as dl
    with dl.DataLoader(filepath, 60.0, 10.0) as data:
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
    
    Parameters
    ----------
    filepath: string
        Name of the file.
    buffersize: float
        Size of internal buffer in seconds.
    backsize: float
        Part of the buffer to be loaded before the requested start index in seconds.
    verbose: int
        If larger than zero show detailed error/warning messages.
    meta_kwargs: dict
        Keyword arguments that are passed on to the _load_metadata() function.

    Attributes
    ----------
    samplerate: float
        The sampling rate of the data in Hertz.
    channels: int
        The number of channels that are read in.
    frames: int
        The number of frames in the file.
    shape: tuple
        Number of frames and channels of the data.
    ndim: int
        Number of dimensions: always 2 (frames and channels).
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
    - `set_unwrap()`: Set parameters for unwrapping clipped data.

    """

    def __init__(self, filepath=None, buffersize=10.0, backsize=0.0,
                 verbose=0, **meta_kwargs):
        super(DataLoader, self).__init__(None, buffersize, backsize,
                                         verbose, **meta_kwargs)
        if filepath is not None:
            self.open(filepath, buffersize, backsize, verbose)

    def __getitem__(self, key):
        return super(DataLoader, self).__getitem__(key)
 
    def __next__(self):
        return super(DataLoader, self).__next__()

    
    # relacs interface:        
    def open_relacs(self, file_paths, buffersize=10.0, backsize=0.0,
                    verbose=0):
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

        Raises
        ------
        ValueError: .gz files not supported.
        """
        self.verbose = verbose
        
        if self.sf is not None:
            self._close_relacs()

        file_paths = relacs_files(file_paths)

        # open trace files:
        self.sf = []
        self.frames = None
        self.samplerate = None
        self.unit = ""
        self.filepath = None
        if len(file_paths) > 0:
            self.filepath = os.path.dirname(file_paths[0])
        for path in sorted(file_paths):
            if path[-3:] == '.gz':
                raise ValueError('.gz files not supported')
            file = open(path, 'rb')
            self.sf.append(file)
            if verbose > 0:
                print(f'open_relacs(filepath) with filepath={path}')
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
        self.shape = (self.frames, self.channels)
        self.size = self.frames * self.channels
        self.ndim = len(self.shape)
        self.buffersize = int(buffersize*self.samplerate)
        self.backsize = int(backsize*self.samplerate)
        self._init_buffer()
        self.offset = 0
        self.close = self._close_relacs
        self.load_buffer = self._load_buffer_relacs
        self.ampl_min = -np.inf
        self.ampl_max = +np.inf
        self._load_metadata = self._metadata_relacs
        # TODO: load markers:
        self._locs = np.zeros((0, 2), dtype=int)
        self._labels = np.zeros((0, 2), dtype=object)
        self._load_markers = None
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
        """ Load meta-data of a relacs data set.
        """
        info_path = os.path.join(self.filepath, 'info.dat')
        if not os.path.exists(info_path):
            return {}
        return relacs_header(info_path, store_empty, first_only)

    
    # fishgrid interface:        
    def open_fishgrid(self, file_paths, buffersize=10.0, backsize=0.0,
                      verbose=0):
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
        """
        self.verbose = verbose
        
        if self.sf is not None:
            self._close_fishgrid()

        if not isinstance(file_paths, (list, tuple, np.ndarray)):
            file_paths = [file_paths]
        file_paths = fishgrid_trace_files(file_paths)
        self.filepath = None
        if len(file_paths) > 0:
            self.filepath = os.path.dirname(file_paths[0])
        self._load_metadata = metadata_fishgrid
        self._load_markers = None  # TODO

        # open grid files:
        grids = fishgrid_grids(self.metadata())
        grid_sizes = [r*c for r,c in grids]
        self.channels = 0
        for g, path in enumerate(file_paths):
            self.channels += grid_sizes[g]
        self.sf = []
        self.grid_channels = []
        self.grid_offs = []
        offs = 0
        self.frames = None
        self.samplerate = get_number(self.metadata(), 'Hz', 'AISampleRate')
        v, self.unit = get_number_unit(self.metadata(), 'AIMaxVolt')
        if v is not None:
            self.ampl_min = -v
            self.ampl_max = +v
            
        for g, path in enumerate(file_paths):
            file = open(path, 'rb')
            self.sf.append(file)
            if verbose > 0:
                print(f'open_fishgrid(filepath) with filepath={path}')
            # grid channels:
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
        self.shape = (self.frames, self.channels)
        self.size = self.frames * self.channels
        self.ndim = len(self.shape)
        self.buffersize = int(buffersize*self.samplerate)
        self.backsize = int(backsize*self.samplerate)
        self._init_buffer()
        self.offset = 0
        self.close = self._close_fishgrid
        self.load_buffer = self._load_buffer_fishgrid
        # TODO: load markers:
        self._locs = np.zeros((0, 2), dtype=int)
        self._labels = np.zeros((0, 2), dtype=object)
        self._load_markers = None
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


    # container interface:
    def open_container(self, file_path, buffersize=10.0,
                       backsize=0.0, verbose=0, datakey=None,
                       samplekey=['rate', 'Fs', 'fs'],
                       timekey=['time'], amplkey=['amax'], unitkey='unit',
                       metadatakey=['metadata', 'info']):
        """Open generic container file.

        Supported file formats are:

        - python pickle files (.pkl)
        - numpy files (.npz)
        - matlab files (.mat)

        Parameters
        ----------
        file_path: string
            Path to a container file.
        buffersize: float
            Size of internal buffer in seconds.
        backsize: float
            Part of the buffer to be loaded before the requested start index in seconds.
        verbose: int
            If > 0 show detailed error/warning messages.
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
        amplkey: string or list of string
            Name of the variable holding the amplitude range of the data.
        unitkey: string
            Name of the variable holding the unit of the data.
            If `unitkey` is not a valid key, then return `unitkey` as the `unit`.
        metadatakey: string or list of string
            Name of the variable holding the metadata.

        Raises
        ------
        ValueError:
            Invalid key requested.
        """
        self.verbose = verbose
        data_dict = {}
        ext = os.path.splitext(file_path)[1]
        if ext == '.pkl':
            import pickle
            with open(file_path, 'rb') as f:
                data_dict = pickle.load(f)
        elif ext == '.npz':
            data_dict = np.load(file_path)
        elif ext == '.mat':
            from scipy.io import loadmat
            data_dict = loadmat(file_path, squeeze_me=True)
        self.buffer, self.samplerate, self.unit, amax = \
            extract_container_data(data_dict, datakey, samplekey,
                                   timekey, amplkey, unitkey)
        self.filepath = file_path
        self.channels = self.buffer.shape[1]
        self.frames = self.buffer.shape[0]
        self.shape = self.buffer.shape
        self.ndim = self.buffer.ndim
        self.size = self.buffer.size
        self.ampl_min = -amax
        self.ampl_max = +amax
        self.offset = 0
        self.buffersize = self.frames
        self.backsize = 0
        self.close = self._close_container
        self.load_buffer = self._load_buffer_container
        self._metadata = extract_container_metadata(data_dict, metadatakey)
        self._load_metadata = None
        # TODO: load markers:
        self._locs = np.zeros((0, 2), dtype=int)
        self._labels = np.zeros((0, 2), dtype=object)
        self._load_markers = None

    def _close_container(self):
        """Close container. """
        pass

    def _load_buffer_container(self, r_offset, r_size, buffer):
        """Load new data from container."""
        pass
            
    
    # audioio interface:        
    def open_audioio(self, file_path, buffersize=10.0, backsize=0.0,
                     verbose=0, gainkey=['gain'], sep='.'):
        """Open an audio file.

        See the [audioio](https://github.com/bendalab/audioio) package
        for details.

        Parameters
        ----------
        file_path: string
            Path to an audio file.
        buffersize: float
            Size of internal buffer in seconds.
        backsize: float
            Part of the buffer to be loaded before the requested start index
            in seconds.
        verbose: int
            If > 0 show detailed error/warning messages.
        gainkey: str or list of str
            Key in the file's metadata that holds some gain information.
            If found, the data will be multiplied with the gain,
            and if available, the corresponding unit is returned.
            See the [audioio.get_gain()](https://bendalab.github.io/audioio/api/audiometadata.html#audioio.audiometadata.get_gain) function for details.
        sep: str
            String that separates section names in `gainkey`.

        """
        self.verbose = verbose
        super(DataLoader, self).open(file_path, buffersize, backsize, verbose)
        md = self.metadata()
        fac, unit = get_gain(md, gainkey, sep)
        self.gain_fac = fac
        if self.gain_fac != 1.0:
            self._load_buffer_audio_org = self.load_buffer
            self.load_buffer = self._load_buffer_audioio
        self.ampl_min *= self.gain_fac
        self.ampl_max *= self.gain_fac
        self.unit = unit
        return self
    
    def _load_buffer_audioio(self, r_offset, r_size, buffer):
        """Load and scale new data from an audio file.

        Parameters
        ----------
        r_offset: int
           First frame to be read from file.
        r_size: int
           Number of frames to be read from file.
        buffer: ndarray
           Buffer where to store the loaded data.
        """
        self._load_buffer_audio_org(r_offset, r_size, buffer)
        buffer *= self.gain_fac

        
    def open(self, filepath, buffersize=10.0, backsize=0.0,
             verbose=0, **kwargs):
        """Open file with time-series data for reading.

        Parameters
        ----------
        filepath: string or list of string
            Path to a data files or directory.
        buffersize: float
            Size of internal buffer in seconds.
        backsize: float
            Part of the buffer to be loaded before the requested start index
            in seconds.
        verbose: int
            If > 0 show detailed error/warning messages.
        **kwargs: dict
            Further keyword arguments that are passed on to the 
            format specific opening functions.
        """
        if check_relacs(filepath):
            self.open_relacs(filepath, buffersize, backsize, verbose)
        elif check_fishgrid(filepath):
            self.open_fishgrid(filepath, buffersize, backsize, verbose)
        else:
            if isinstance(filepath, (list, tuple, np.ndarray)):
                filepath = filepath[0]
            if check_container(filepath):
                self.open_container(filepath, buffersize, backsize,
                                    verbose, **kwargs)
            else:
                self.open_audioio(filepath, buffersize, backsize,
                                  verbose, **kwargs)
        return self


def demo(filepath, plot=False):
    print("try load_data:")
    data, samplerate, unit, amax = load_data(filepath, verbose=2)
    if plot:
        fig, ax = plt.subplots()
        time = np.arange(len(data))/samplerate
        for c in range(data.shape[1]):
            ax.plot(time, data[:,c])
        ax.set_xlabel('Time [s]')
        ax.set_ylabel(f'[{unit}]')
        if amax is not None and np.isfinite(amax):
            ax.set_ylim(-amax, +amax)
        plt.show()
        return

    print('')
    print("try DataLoader:")
    with DataLoader(filepath, 2.0, 1.0, 1) as data:
        print('samplerate: %g' % data.samplerate)
        print('frames: %d %d' % (len(data), data.shape[0]))
        nframes = int(1.0 * data.samplerate)
        # forward:
        for i in range(0, len(data), nframes):
            print('forward %d-%d' % (i, i + nframes))
            x = data[i:i + nframes, 0]
            if plot:
                fig, ax = plt.subplots()
                ax.plot((i + np.arange(len(x)))/data.samplerate, x)
                ax.set_xlabel('Time [s]')
                ax.set_ylabel(f'[{data.unit}]')
                plt.show()
        # and backwards:
        for i in reversed(range(0, len(data), nframes)):
            print('backward %d-%d' % (i, i + nframes))
            x = data[i:i + nframes, 0]
            if plot:
                fig, ax = plt.subplots()
                ax.plot((i + np.arange(len(x)))/data.samplerate, x)
                ax.set_xlabel('Time [s]')
                ax.set_ylabel(f'[{data.unit}]')
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
    parser.add_argument('file', nargs=1, default='', type=str,
                        help='name of data file')
    args = parser.parse_args(cargs)
    demo(args.file[0], args.plot)
    

if __name__ == "__main__":
    main(*sys.argv[1:])
