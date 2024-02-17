"""Writing numpy arrays of floats to data files.

- `write_data()`: write data into a file.
- `available_formats()`: supported data and audio file formats.
- `available_encodings()`: encodings of a data file format.
- `format_from_extension()`: deduce data file format from file extension.
- `write_metadata_text()`: write meta data into a text/yaml file.
- `recode_array()`: recode array of floats.
"""

import os
import sys

data_modules = {}
"""Dictionary with availability of various modules needed for writing data.
Keys are the module names, values are booleans.
"""

try:
    import pickle
    data_modules['pickle'] = True
except ImportError:
    data_modules['pickle'] = False

try:
    import numpy as np
    data_modules['numpy'] = True
except ImportError:
    data_modules['numpy'] = False

try:
    import scipy.io as sio
    data_modules['scipy'] = True
except ImportError:
    data_modules['scipy'] = False

try:
    import audioio.audiowriter as aw
    import audioio.audiometadata as am
    from audioio import write_metadata_text, flatten_metadata
    data_modules['audioio'] = True
except ImportError:
    data_modules['audioio'] = False


def format_from_extension(filepath):
    """Deduce data file format from file extension.

    Parameters
    ----------
    filepath: string
        Name of the data file.

    Returns
    -------
    format: string
        Data format deduced from file extension.
    """
    if not filepath:
        return None
    ext = os.path.splitext(filepath)[1]
    if not ext:
        return None
    if ext[0] == '.':
        ext = ext[1:]
    if not ext:
        return None
    ext = ext.upper()
    if data_modules['audioio']:
        ext = aw.format_from_extension(filepath)
    return ext


def recode_array(data, encoding):
    """Recode array of floats.

    Parameters
    ----------
    data: array of floats
        Data array with values ranging between -1 and 1
    encoding: str
        Encoding, one of PCM_16, PCM_32, PCM_64, FLOAT or DOUBLE.

    Returns
    -------
    buffer: array
        The data recoded according to `encoding`.
    """
    
    encodings = {'PCM_16': (2, 'i2'),
                 'PCM_32': (4, 'i4'),
                 'PCM_64': (8, 'i8'),
                 'FLOAT': (4, 'f'),
                 'DOUBLE': (8, 'd')}

    if not encoding in encodings:
        return data
    dtype = encodings[encoding][1]
    if dtype[0] == 'i':
        sampwidth = encodings[encoding][0]
        factor = 2**(sampwidth*8-1)
        buffer = np.floor(data * factor).astype(dtype)
        buffer[data >= 1.0] = factor - 1
    else:
        buffer = data.astype(dtype, copy=False)
    return buffer

    
def formats_relacs():
    """Data format of the relacs file format.

    Returns
    -------
    formats: list of strings
        List of supported file formats as strings.
    """
    return ['RELACS']


def encodings_relacs(format=None):
    """Encodings of the relacs file format.

    Parameters
    ----------
    format: str
        The file format.

    Returns
    -------
    encodings: list of strings
        List of supported encodings as strings.
    """
    if not format:
        format = 'RELACS'
    if format.upper() != 'RELACS':
        return []
    else:
        return ['FLOAT']
    
    
def write_relacs(filepath, data, samplerate, unit=None,
                 metadata=None, format=None, encoding=None):
    """Write data as relacs raw files.

    Parameters
    ----------
    filepath: string
        Full path of folder where to write relacs files.
    data: 1-D or 2-D array of floats
        Array with the data (first index time, optional second index channel).
    samplerate: float
        Sampling rate of the data in Hertz.
    unit: string
        Unit of the data.
    metadata: nested dict
        Additional metadata saved into `info.dat`.
    format: string or None
        File format, only None or 'RELACS' are supported.
    encoding: string or None
        Encoding of the data. Only None or 'FLOAT' are supported.

    Returns
    -------
    filepath: string or None
        On success, the actual file name used for writing the data.

    Raises
    ------
    ValueError
        Invalid `filepath`.
    ValueError
        File format or encoding not supported.
    """
    if not filepath:
        raise ValueError('no file specified!')
    if format is None:
        format = 'RELACS'
    if format.upper() != 'RELACS':
        raise ValueError(f'file format {format} not supported by relacs file format')
    if encoding is None:
        encoding = 'FLOAT'
    if encoding.upper() != 'FLOAT':
        raise ValueError(f'file encoding {format} not supported by relacs file format')
    os.mkdir(filepath)
    # write data:
    for c in range(data.shape[1]):
        df = open(os.path.join(filepath, f'trace-{c+1}.raw'), 'wb')
        df.write(np.array(data[:, c], dtype=np.float32).tostring())
        df.close()
    if unit is None:
        unit = 'V'
    # write data format:
    filename = os.path.join(filepath, 'stimuli.dat')
    df = open(filename, 'w')
    df.write('# analog input traces:\n')
    for c in range(data.shape[1]):
        df.write(f'#     identifier{c+1}      : V-{c+1}\n')
        df.write(f'#     data file{c+1}       : trace-{{c+1}}.raw\n')
        df.write(f'#     sample interval{c+1} : {1000.0/samplerate:.4f}ms\n')
        df.write(f'#     sampling rate{c+1}   : {samplerate:.2f}Hz\n')
        df.write(f'#     unit{c+1}            : {unit}\n')
    df.write('# event lists:\n')
    df.write('#      event file1: stimulus-events.dat\n')
    df.write('#      event file2: restart-events.dat\n')
    df.write('#      event file3: recording-events.dat\n')
    df.close()
    # write empty event files:
    for events in ['Recording', 'Restart', 'Stimulus']:
        df = open(os.path.join(filepath, f'{events.lower()}-events.dat'), 'w')
        df.write(f'# events: {events}\n\n')
        df.write('#Key\n')
        if events == 'Stimulus':
            df.write('# t    duration\n')
            df.write('# sec  s\n')
            df.write('#   1         2\n')
        else:
            df.write('# t\n')
            df.write('# sec\n')
            df.write('# 1\n')
            if events == 'Recording':
                df.write('  0.0\n')
        df.close()
    # write metadata:
    if metadata:
        write_metadata_text(os.path.join(filepath, 'info.dat'),
                            metadata, prefix='# ')
    return filename

    
def formats_fishgrid():
    """Data format of the fishgrid file format.

    Returns
    -------
    formats: list of strings
        List of supported file formats as strings.
    """
    return ['FISHGRID']


def encodings_fishgrid(format=None):
    """Encodings of the fishgrid file format.

    Parameters
    ----------
    format: str
        The file format.

    Returns
    -------
    encodings: list of strings
        List of supported encodings as strings.
    """
    if not format:
        format = 'FISHGRID'
    if format.upper() != 'FISHGRID':
        return []
    else:
        return ['FLOAT']
    
    
def write_fishgrid(filepath, data, samplerate, unit=None, metadata=None,
                   format=None, encoding=None):
    """Write data as fishgrid raw files.

    Parameters
    ----------
    filepath: string
        Full path of the folder where to write fishgrid files.
    data: 1-D or 2-D array of floats
        Array with the data (first index time, optional second index channel).
    samplerate: float
        Sampling rate of the data in Hertz.
    unit: string
        Unit of the data.
    metadata: nested dict
        Additional metadata saved into the `fishgrid.cfg`.
    format: string or None
        File format, only None or 'FISHGRID' are supported.
    encoding: string or None
        Encoding of the data. Only None or 'FLOAT' are supported.

    Returns
    -------
    filepath: string or None
        On success, the actual file name used for writing the data.

    Raises
    ------
    ValueError
        Invalid `filepath`.
    ValueError
        File format or encoding not supported.
    """
    if not filepath:
        raise ValueError('no file specified!')
    if format is None:
        format = 'FISHGRID'
    if format.upper() != 'FISHGRID':
        raise ValueError(f'file format {format} not supported by fishgrid file format')
    if encoding is None:
        encoding = 'FLOAT'
    if encoding.upper() != 'FLOAT':
        raise ValueError(f'file encoding {format} not supported by fishgrid file format')
    os.mkdir(filepath)
    # write data:
    df = open(os.path.join(filepath, 'traces-grid1.raw'), 'wb')
    df.write(np.array(data, dtype=np.float32).tostring())
    df.close()
    # write metadata:
    if unit is None:
        unit = 'mV'
    filename = os.path.join(filepath, 'fishgrid.cfg')
    df = open(filename, 'w')
    df.write('*FishGrid\n')
    df.write('  Grid &1\n')
    df.write('     Used1      : true\n')
    df.write('     Columns    : 2\n')
    df.write(f'     Rows       : {data.shape[1]//2}\n')
    df.write('  Hardware Settings\n')
    df.write('    DAQ board:\n')
    df.write(f'      AISampleRate: {0.001*samplerate:.3f}kHz\n')
    df.write(f'      AIMaxVolt   : 10.0{unit}\n')
    df.write('    Amplifier:\n')
    df.write('      AmplName: "16-channel-EPM-module"\n')
    if metadata:
        df.write('*Recording\n')
        write_metadata_text(df, metadata, prefix='  ')
    df.close()
    return filename

    
def formats_pickle():
    """Data formats supported by pickle.dump().

    Returns
    -------
    formats: list of strings
        List of supported file formats as strings.
    """
    if not data_modules['pickle']:
        return []
    else:
        return ['PKL']


def encodings_pickle(format=None):
    """Encodings of the pickle format.

    Parameters
    ----------
    format: str
        The file format.

    Returns
    -------
    encodings: list of strings
        List of supported encodings as strings.
    """
    if not format:
        format = 'PKL'
    if format.upper() != 'PKL':
        return []
    else:
        return ['PCM_16', 'PCM_32', 'FLOAT', 'DOUBLE']

    
def write_pickle(filepath, data, samplerate, unit=None, metadata=None,
                 format=None, encoding=None):
    """Write data into python pickle file.
    
    Documentation
    -------------
    https://docs.python.org/3/library/pickle.html

    Parameters
    ----------
    filepath: string
        Full path and name of the file to write.
    data: 1-D or 2-D array of floats
        Array with the data (first index time, optional second index channel).
        Stored under the key "data".
    samplerate: float
        Sampling rate of the data in Hertz.
        Stored under the key "rate".
    unit: string
        Unit of the data.
        Stored under the key "unit".
    metadata: nested dict
        Additional metadata saved into the pickle.
        Stored under the key "metadata".
    format: string or None
        File format, only None or 'PKL' are supported.
    encoding: string or None
        Encoding of the data.

    Returns
    -------
    filepath: string or None
        On success, the actual file name used for writing the data.

    Raises
    ------
    ImportError
        The pickle module is not available.
    ValueError
        Invalid `filepath`.
    ValueError
        File format or encoding not supported.
    """
    if not data_modules['pickle']:
        raise ImportError
    if not filepath:
        raise ValueError('no file specified!')
    if format is None:
        format = 'PKL'
    if format.upper() != 'PKL':
        raise ValueError(f'file format {format} not supported by pickle file format')
    ext = os.path.splitext(filepath)[1]
    if len(ext) <= 1 or ext[1].upper() != 'P':
        filepath += os.extsep + 'pkl'
    if encoding is None:
        encoding = 'DOUBLE'
    encoding = encoding.upper()
    if not encoding in encodings_pickle(format):
        raise ValueError(f'file encoding {format} not supported by pickle file format')
    buffer = recode_array(data, encoding)
    ddict = dict(data=buffer, rate=samplerate)
    if unit:
        ddict['unit'] = unit
    if metadata:
        ddict['metadata'] = metadata
    with open(filepath, 'wb') as df:
        pickle.dump(ddict, df)
    return filepath


def formats_numpy():
    """Data formats supported by numpy.savez().

    Returns
    -------
    formats: list of strings
        List of supported file formats as strings.
    """
    if not data_modules['numpy']:
        return []
    else:
        return ['NPZ']


def encodings_numpy(format=None):
    """Encodings of the numpy file format.

    Parameters
    ----------
    format: str
        The file format.

    Returns
    -------
    encodings: list of strings
        List of supported encodings as strings.
    """
    if not format:
        format = 'NPZ'
    if format.upper() !=  'NPZ':
        return []
    else:
        return ['PCM_16', 'PCM_32', 'FLOAT', 'DOUBLE']


def write_numpy(filepath, data, samplerate, unit=None, metadata=None,
                format=None, encoding=None):
    """Write data into numpy npz file.
    
    Documentation
    -------------
    https://numpy.org/doc/stable/reference/generated/numpy.savez.html

    Parameters
    ----------
    filepath: string
        Full path and name of the file to write.
    data: 1-D or 2-D array of floats
        Array with the data (first index time, optional second index channel).
        Stored under the key "data".
    samplerate: float
        Sampling rate of the data in Hertz.
        Stored under the key "rate".
    unit: string
        Unit of the data.
        Stored under the key "unit".
    metadata: nested dict
        Additional metadata saved into the numpy file.
        Flattened dictionary entries stored under keys
        starting with "metadata__".
    format: string or None
        File format, only None or 'NPZ' are supported.
    encoding: string or None
        Encoding of the data.

    Returns
    -------
    filepath: string or None
        On success, the actual file name used for writing the data.

    Raises
    ------
    ImportError
        The numpy module is not available.
    ValueError
        Invalid `filepath`.
    ValueError
        File format or encoding not supported.
    """
    if not data_modules['numpy']:
        raise ImportError
    if not filepath:
        raise ValueError('no file specified!')
    if format is None:
        format = 'NPZ'
    if format.upper() not in formats_numpy():
        raise ValueError(f'file format {format} not supported by numpy file format')
    ext = os.path.splitext(filepath)[1]
    if len(ext) <= 1 or ext[1].upper() != 'N':
        filepath += os.extsep + 'npz'
    if encoding is None:
        encoding = 'DOUBLE'
    encoding = encoding.upper()
    if not encoding in encodings_numpy(format):
        raise ValueError(f'file encoding {format} not supported by numpy file format')
    buffer = recode_array(data, encoding)
    ddict = dict(data=buffer, rate=samplerate)
    if unit:
        ddict['unit'] = unit
    if metadata:
        fmeta = flatten_metadata(metadata, True, sep='__')
        for k in list(fmeta):
            fmeta['metadata__'+k] = fmeta.pop(k)
        ddict.update(fmeta)
    np.savez(filepath, **ddict)
    return filepath


def formats_mat():
    """Data formats supported by scipy.io.savemat().

    Returns
    -------
    formats: list of strings
        List of supported file formats as strings.
    """
    if not data_modules['scipy']:
        return []
    else:
        return ['MAT']


def encodings_mat(format=None):
    """Encodings of the matlab format.

    Parameters
    ----------
    format: str
        The file format.

    Returns
    -------
    encodings: list of strings
        List of supported encodings as strings.
    """
    if not format:
        format = 'MAT'
    if format.upper() != 'MAT':
        return []
    else:
        return ['PCM_16', 'PCM_32', 'FLOAT', 'DOUBLE']


def write_mat(filepath, data, samplerate, unit=None, metadata=None,
              format=None, encoding=None):
    """Write data into matlab file.
    
    Documentation
    -------------
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.io.savemat.html

    Parameters
    ----------
    filepath: string
        Full path and name of the file to write.
        Stored under the key "data".
    data: 1-D or 2-D array of floats
        Array with the data (first index time, optional second index channel).
        Stored under the key "data".
    samplerate: float
        Sampling rate of the data in Hertz.
        Stored under the key "rate".
    unit: string
        Unit of the data.
        Stored under the key "unit".
    metadata: nested dict
        Additional metadata saved into the mat file.
        Stored under the key "metadata".
    format: string or None
        File format, only None or 'MAT' are supported.
    encoding: string or None
        Encoding of the data.

    Returns
    -------
    filepath: string or None
        On success, the actual file name used for writing the data.

    Raises
    ------
    ImportError
        The scipy.io module is not available.
    ValueError
        Invalid `filepath`.
    ValueError
        File format or encoding not supported.
    """
    if not data_modules['scipy']:
        raise ImportError
    if not filepath:
        raise ValueError('no file specified!')
    if format is None:
        format = 'MAT'
    if format.upper() not in formats_mat():
        raise ValueError(f'file format {format} not supported by matlab file format')
    ext = os.path.splitext(filepath)[1]
    if len(ext) <= 1 or ext[1].upper() != 'M':
        filepath += os.extsep + 'mat'
    if encoding is None:
        encoding = 'DOUBLE'
    encoding = encoding.upper()
    if not encoding in encodings_mat(format):
        raise ValueError(f'file encoding {format} not supported by matlab file format')
    buffer = recode_array(data, encoding)
    ddict = dict(data=buffer, rate=samplerate)
    if unit:
        ddict['unit'] = unit
    if metadata:
        fmeta = flatten_metadata(metadata, True, sep='__')
        for k in list(fmeta):
            fmeta['metadata__'+k] = fmeta.pop(k)
        ddict.update(fmeta)
    sio.savemat(filepath, ddict)
    return filepath


def formats_audioio():
    """Data formats supported by audioio.

    Returns
    -------
    formats: list of strings
        List of supported file formats as strings.
    """
    if not data_modules['audioio']:
        return []
    else:
        return aw.available_formats()


def encodings_audio(format):
    """Encodings of any audio format.

    Parameters
    ----------
    format: str
        The file format.

    Returns
    -------
    encodings: list of strings
        List of supported encodings as strings.
    """
    if not data_modules['audioio']:
        return []
    else:
        return aw.available_encodings(format)


def write_audioio(filepath, data, samplerate, unit=None, metadata=None,
                  format=None, encoding=None,
                  gainkey=['Gain', 'Scale', 'Unit'], sep='__'):
    """Write data into audio file.

    If a gain setting is available in the metadata, then the data are divided
    by the gain before they are stored in the audio file.
    After this operation, the data values need to range between -1 and 1,
    in particular if the data are encoded as integers
    (i.e. PCM_16, PCM_32 and PCM_64).
    Note, that this function does not check for this requirement!
    
    Documentation
    -------------
    https://bendalab.github.io/audioio/

    Parameters
    ----------
    filepath: string
        Full path and name of the file to write.
    data: 1-D or 2-D array of floats
        Array with the data (first index time, optional second index channel).
    samplerate: float
        Sampling rate of the data in Hertz.
    unit: string
        Unit of the data. If supplied and a gain is found in the metadata it
        has to match the unit of the gain. If no gain is found in the metadata
        and metadata is not None, then a gain of one with this unit is added
        to the metadata using the first key in `gainkey`.
    metadata: nested dict
        Metadata saved into the audio file. If it contains a gain,
        the gain factor is used to divide the data down into a
        range between -1 and 1.
    format: string or None
        File format. If None deduce file format from filepath.
        See `available_formats()` for possible values.
    encoding: string or None
        Encoding of the data. See `available_encodings()` for possible values.
        If None or empty string use 'PCM_16'.
    gainkey: str or list of str
        Key in the file's metadata that holds some gain information.
        If found, the data will be multiplied with the gain,
        and if available, the corresponding unit is returned.
        See the [audioio.get_gain()](https://bendalab.github.io/audioio/api/audiometadata.html#audioio.audiometadata.get_gain) function for details.
    sep: str
        String that separates section names in `gainkey`.

    Returns
    -------
    filepath: string or None
        On success, the actual file name used for writing the data.

    Raises
    ------
    ImportError
        The audioio module is not available.
    ValueError
        Invalid `filepath` or `unit` does not match gain in metadata.
    """
    if not data_modules['audioio']:
        raise ImportError
    if not filepath:
        raise ValueError('no file specified!')
    fac, u = am.get_gain(metadata, gainkey, sep)
    if unit and unit != 'a.u.' and u != 'a.u.' and unit != u:
        raise ValueError(f'unit "{unit}" does not match gain unit "{u}" in metadata')
    if fac != 1.0:
        data = data / fac
    elif not metadata is None and unit and unit != 'a.u.':
        gk = gainkey[0] if len(gainkey) > 0 else 'Gain'
        m, k = am.find_key(metadata, gk)
        if k in m:
            m[k] = f'1{unit}'
        elif 'INFO' in metadata and gk.upper() == 'GAIN':
            metadata['INFO'][gk] = f'1{unit}'
        else:
            metadata[gk] = f'1{unit}'
    aw.write_audio(filepath, data, samplerate, metadata)
    return filepath


data_formats_funcs = (
    ('relacs', None, formats_relacs),
    ('fishgrid', None, formats_fishgrid),
    ('pickle', 'pickle', formats_pickle),
    ('numpy', 'numpy', formats_numpy),
    ('matlab', 'scipy', formats_mat),
    ('audio', 'audioio', formats_audioio)
    )
"""List of implemented formats functions.

Each element of the list is a tuple with the format's name, the
module's name in `data_modules` or None, and the formats function.
"""


def available_formats():
    """Data and audio file formats supported by any of the installed modules.

    Returns
    -------
    formats: list of strings
        List of supported file formats as strings.
    """
    formats = set()
    for fmt, lib, formats_func in data_formats_funcs:
        if not lib or data_modules[lib]:
            formats |= set(formats_func())
    return sorted(list(formats))


data_encodings_funcs = (
    ('relacs', encodings_relacs),
    ('fishgrid', encodings_fishgrid),
    ('pickle', encodings_pickle),
    ('numpy', encodings_numpy),
    ('matlab', encodings_mat),
    ('audio', encodings_audio)
    )
""" List of implemented encodings functions.

Each element of the list is a tuple with the module's name and the encodings function.
"""


def available_encodings(format):
    """Encodings of a data file format.

    Parameters
    ----------
    format: str
        The file format.

    Returns
    -------
    encodings: list of strings
        List of supported encodings as strings.
    """
    encodings = set()
    for module, encodings_func in data_encodings_funcs:
        encs = encodings_func(format)
        encodings |= set(encs)
    return sorted(list(encodings))


data_writer_funcs = {
    'relacs': write_relacs,
    'fishgrid': write_fishgrid,
    'pickle': write_pickle,
    'numpy': write_numpy,
    'matlab':  write_mat,
    'audio': write_audioio
    }
"""Dictionary of implemented write functions.

Keys are the format's name and values the corresponding write
function.
"""


def write_data(filepath, data, samplerate, unit=None, metadata=None,
               format=None, encoding=None, verbose=0, **kwargs):
    """Write data into a file.

    Parameters
    ----------
    filepath: string
        Full path and name of the file to write.
        File format is determined from extension.
    data: 1-D or 2-D array of floats
        Array with the data (first index time, second index channel).
    samplerate: float
        Sampling rate of the data in Hertz.
    unit: string
        Unit of the data.
    metadata: nested dict
        Additional metadata.
    format: string or None
        File format. If None deduce file format from filepath.
        See `available_formats()` for possible values.
    encoding: string or None
        Encoding of the data. See `available_encodings()` for possible values.
        If None or empty string use 'PCM_16'.
    verbose: int
        If >0 show detailed error/warning messages.
    kwargs: dict
        Additional, file format specific keyword arguments.

    Returns
    -------
    filepath: string or None
        On success, the actual file name used for writing the data.

    Raises
    ------
    ValueError
        `filepath` is empty string or unspecified format.
    IOError
        Requested file format not supported.

    Example
    -------
    ```
    import numpy as np
    from thunderfish.datawriter import write_data
    
    samplerate = 28000.0
    freq = 800.0
    time = np.arange(0.0, 1.0, 1/samplerate)     # one second
    data = 2.5*np.sin(2.0*np.p*freq*time)        # 800Hz sine wave
    md = dict(Artist='underscore_')          # metadata
    write_data('audio/file.npz', data, samplerate, 'mV', md)
    ```
    """
    if not filepath:
        raise ValueError('no file specified!')

    if not format:
        format = format_from_extension(filepath)
    if not format:
        raise ValueError('unspecified file format')
    for fmt, lib, formats_func in data_formats_funcs:
        if lib and not data_modules[lib]:
            continue
        if format.upper() in formats_func():
            writer_func = data_writer_funcs[fmt]
            filepath = writer_func(filepath, data, samplerate, unit, metadata,
                                   format=format, encoding=encoding, **kwargs)
            if verbose > 0:
                print(f'wrote data to file "{filepath}" using {fmt} format')
                if verbose > 1:
                    print(f'  sampling rate: {samplerate:g}Hz')
                    print(f'  channels     : {data.shape[1] if len(data.shape) > 1 else 1}')
                    print(f'  frames       : {len(data)}')
                    print(f'  unit         : {unit}')
            return filepath
    raise IOError(f'file format "{format.upper()}" not supported.') 


def demo(file_path, channels=2, format=None):
    """Demo of the datawriter functions.

    Parameters
    ----------
    file_path: string
        File path of a data file.
    format: string or None
        File format to be used.
    """
    print('generate data ...')
    samplerate = 44100.0
    t = np.arange(0.0, 1.0, 1.0/samplerate)
    data = np.zeros((len(t), channels))
    for c in range(channels):
        data[:,c] = 0.1*(channels-c)*np.sin(2.0*np.pi*(440.0+c*8.0)*t)
        
    print(f"write_data('{file_path}') ...")
    write_data(file_path, data, samplerate, 'mV', format=format, verbose=2)

    print('done.')
    

def main(*cargs):
    """Call demo with command line arguments.

    Parameters
    ----------
    cargs: list of strings
        Command line arguments as provided by sys.argv[1:]
    """
    import argparse
    parser = argparse.ArgumentParser(description=
                                     'Checking thunderfish.datawriter module.')
    parser.add_argument('-c', dest='channels', default=2, type=int,
                        help='number of channels to be written')
    parser.add_argument('-f', dest='format', default=None, type=str,
                        help='file format')
    parser.add_argument('file', nargs=1, default='test.npz', type=str,
                        help='name of data file')
    args = parser.parse_args(cargs)
    demo(args.file[0], args.channels, args.format)
    

if __name__ == "__main__":
    main(*sys.argv[1:])

    

