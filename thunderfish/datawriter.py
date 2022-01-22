"""
Writing numpy arrays of floats to data files.
"""

import os

data_modules = {}
""" Dictionary with availability of various modules needed for writing data.
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
    if ext == 'PKL':
        return 'PICKLE'
    if data_modules['audioio']:
        ext = aw.format_from_extension(filepath)
    return ext

    
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
        return ['PICKLE']

    
def write_pickle(filepath, data, samplerate, unit=None, meta=None):
    """Write data into python pickle file.
    
    Documentation
    -------------
    https://docs.python.org/3/library/pickle.html

    Parameters
    ----------
    filepath: string
        Full path and name of the file to write.
    data: 1-D or 2-D array of floats
        Array with the data (first index time, second index channel).
    samplerate: float
        Sampling rate of the data in Hertz.
    unit: string
        Unit of the data.
    meta: dict
        Additional metadata saved into the pickle.

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
    """
    if not data_modules['pickle']:
        raise ImportError
    if not filepath:
        raise ValueError('no file specified!')
    ext = os.path.splitext(filepath)[1]
    if len(ext) <= 1 or ext[1].upper() != 'P':
        filepath += os.extsep + 'pkl'
    ddict = dict(data=data, rate=samplerate)
    if unit:
        ddict['unit'] = unit
    if meta:
        ddict.update(meta)
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


def write_numpy(filepath, data, samplerate, unit=None, meta=None):
    """Write data into numpy npz file.
    
    Documentation
    -------------
    https://numpy.org/doc/stable/reference/generated/numpy.savez.html

    Parameters
    ----------
    filepath: string
        Full path and name of the file to write.
    data: 1-D or 2-D array of floats
        Array with the data (first index time, second index channel).
    samplerate: float
        Sampling rate of the data in Hertz.
    unit: string
        Unit of the data.
    meta: dict
        Additional metadata saved into the numpy file.

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
    """
    if not data_modules['numpy']:
        raise ImportError
    if not filepath:
        raise ValueError('no file specified!')
    ext = os.path.splitext(filepath)[1]
    if len(ext) <= 1 or ext[1].upper() != 'N':
        filepath += os.extsep + 'npz'
    ddict = dict(data=data, rate=samplerate)
    if unit:
        ddict['unit'] = unit
    if meta:
        ddict.update(meta)
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


def write_mat(filepath, data, samplerate, unit=None, meta=None):
    """Write data into matlab file.
    
    Documentation
    -------------
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.io.savemat.html

    Parameters
    ----------
    filepath: string
        Full path and name of the file to write.
    data: 1-D or 2-D array of floats
        Array with the data (first index time, second index channel).
    samplerate: float
        Sampling rate of the data in Hertz.
    unit: string
        Unit of the data.
    meta: dict
        Additional metadata saved into the mat file.

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
    """
    if not data_modules['scipy']:
        raise ImportError
    if not filepath:
        raise ValueError('no file specified!')
    ext = os.path.splitext(filepath)[1]
    if len(ext) <= 1 or ext[1].upper() != 'M':
        filepath += os.extsep + 'mat'
    ddict = dict(data=data, rate=samplerate)
    if unit:
        ddict['unit'] = unit
    if meta:
        ddict.update(meta)
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


def write_audioio(filepath, data, samplerate, unit=None, meta=None):
    """Write data into audio file.
    
    Documentation
    -------------
    https://bendalab.github.io/audioio/

    Parameters
    ----------
    filepath: string
        Full path and name of the file to write.
    data: 1-D or 2-D array of floats
        Array with the data (first index time, second index channel).
    samplerate: float
        Sampling rate of the data in Hertz.
    unit: string
        Unit of the data.
        Currently ignored.
    meta: dict
        Additional metadata saved into the audio file.
        Currently ignored.

    Returns
    -------
    filepath: string or None
        On success, the actual file name used for writing the data.

    Raises
    ------
    ImportError
        The audioio module is not available.
    ValueError
        Invalid `filepath`.
    """
    if not data_modules['audioio']:
        raise ImportError
    if not filepath:
        raise ValueError('no file specified!')
    aw.audio_writer(filepath, data, samplerate)
    return filepath


data_formats_funcs = (
    ('pickle', 'pickle', formats_pickle),
    ('numpy', 'numpy', formats_numpy),
    ('matlab', 'scipy', formats_mat),
    ('audio', 'audioio', formats_audioio)
    )
"""List of implemented formats functions.

Each element of the list is a tuple with the format's name, the
module's name in `data_modules`, and the formats function.
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
        if data_modules[lib]:
            formats |= set(formats_func())
    return sorted(list(formats))


data_writer_funcs = {
    'pickle': write_pickle,
    'numpy': write_numpy,
    'matlab':  write_mat,
    'audio': write_audioio
    }
"""Dictionary of implemented write functions.

Keys are the format's name and values the corresponding write
function.
"""


def write_data(filepath, data, samplerate, unit=None, meta=None,
               format=None, verbose=0):
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
    meta: dict
        Additional metadata saved into the pickle.
    format: string or None
        File format. If None deduce file format from filepath.
        See `available_formats()` for possible values.
    verbose: int
        If >0 show detailed error/warning messages.

    Returns
    -------
    filepath: string or None
        On success, the actual file name used for writing the data.

    Raises
    ------
    ValueError
        `filepath` is empty string.
    IOError
        Writing of the data failed.

    Example
    -------
    ```
    import numpy as np
    from thunderfish.datawriter import write_data
    
    samplerate = 28000.0
    freq = 800.0
    time = np.arange(0.0, 1.0, 1/samplerate)     # one second
    data = 2.5*np.sin(2.0*np.p*freq*time)        # 800Hz sine wave
    write_data('audio/file.npz', data, samplerate, 'mV')
    ```
    """
    if not filepath:
        raise ValueError('no file specified!')

    if not format:
        format = format_from_extension(filepath)
    for fmt, lib, formats_func in data_formats_funcs:
        if not data_modules[lib]:
            continue
        if format in formats_func():
            writer_func = data_writer_funcs[fmt]
            filepath = writer_func(filepath, data, samplerate, unit, meta)
            if verbose > 0:
                print('wrote data to file "%s" using %s module' %
                      (filepath, lib))
                if verbose > 1:
                    print('  sampling rate: %g Hz' % samplerate)
                    print('  channels     : %d' % (data.shape[1] if len(data.shape) > 1 else 1))
                    print('  frames       : %d' % len(data))
                    print('  unit         : %s' % unit)
            return filepath
    raise IOError('file format "%s" not supported.' % format) 


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
        
    print("write_data('%s') ..." % file_path)
    write_data(file_path, data, samplerate, 'mV', format=format, verbose=2)

    print('done.')
    

def main(cargs):
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
    if args.format:
        args.format = args.format.upper()
    demo(args.file[0], args.channels, args.format)
    

if __name__ == "__main__":
    import sys
    main(sys.argv[1:])

    

