"""Load EODs from Hopkins files.

Carl Hopkins and John Sullivan stored only a few cut-out EOD waveforms
of Mormyrid EODs in a specific mat file. These recordings are
available at the Macaulay library.

## Functions

- `load_hopkins()`: load a Hopkins file containing a few EOD pulses.
- `analyse_hopkins()`: analyze the content of Hopkins files.

"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from audioio import print_metadata


def load_hopkins(file_path):
    """ Load a Hopkins file containing a few EOD pulses.

    Parameters
    ----------
    file_path: str
        The mat file with the data.

    Returns
    -------
    data: list of 2-D ndarrays
        A list of single EOD pulses.
        First column is time in seconds, second column EOD waveform.
    md: nested dict
        Metadata.
    """
    x = loadmat(file_path, squeeze_me=True)
    y = x['eod'].reshape(x['eod'].size)
    # assemble data:
    data = []
    for k in range(len(y['wave'])):
        eod = np.zeros((len(y['wave'][k]), 2))
        eod[:, 0] = y['time'][k]
        eod[:, 1] = (y['wave'][k]).astype(float)
        data.append(eod)
    # assemble metadata:
    md = {}
    eod_md = []
    for n in y.dtype.names:
        t = type(y[n][0])
        if not t is np.ndarray:
            # some metadata may or may not differ between EODs:
            for k in range(len(y[n])):
                if y[n][k] and n != 'eodnum' and y[n][k] != y[n][0]:
                    while len(eod_md) < len(y[n]):
                        eod_md.append({})
                    for k in range(len(y[n])):
                        v = y[n][k]
                        if isinstance(v, str):
                            v = v.replace('Date:', '')
                            v = v.replace('Time:', '')
                            v = v.replace('Time', '')
                            v = v.strip()
                        eod_md[k][n] = v
                    break
            else:
                v = y[n][0]
                if isinstance(v, str):
                    v = v.replace('Date:', '')
                    v = v.replace('Time:', '')
                    v = v.replace('Time', '')
                    v = v.replace(' T: ', 'T')
                    v = v.strip()
                md[n] = v
    for k in range(len(eod_md)):
        md[f'EOD{k}'] = eod_md[k]
    return data, md


def analyse_hopkins(pathes):
    """ Analyze the content of Hopkins files.

    Prints out some statistics about the field names and types.

    Parameters
    ----------
    pathes: list of str
        Files to be analyzed.
    """
    keys = {}
    types = {}
    data_types = {}
    for file_path in pathes:
        x = loadmat(file_path, squeeze_me=True)
        y = x['eod'].reshape(x['eod'].size)
        for n in y.dtype.names:
            c = keys.get(n, 0)
            keys[n] = c + 1
            t = type(y[n][0])
            c = types.get(t, 0)
            types[t] = c + 1
        t = y['wave'][0].dtype
        c = data_types.get(t, 0)
        data_types[t] = c + 1

    # each file contains several "wave" and "time" arrays for plotting the EODs.
    # within a file they might differ in size!

    # print all keys found in the data with their frequency:
    print('keys:')
    for k in keys:
        print(f'  {100*keys[k]/len(pathes):3.0f}%', k)
    print()

    # print all wave data types with their frequency:
    print('data types:')
    for t in data_types:
        print(f'  {data_types[t]:5d}', t)
    print()
    #  226 float64
    #   57 int16

    # print types of all fields with their frequency:
    print('field types:')
    for t in types:
        print(f'  {types[t]:5d}', t)
    print()
    # 2845 <class 'int'>
    # 4002 <class 'numpy.ndarray'>
    #14681 <class 'str'>
    # 1478 <class 'float'>


if __name__ == '__main__':

    analyse_hopkins(sys.argv[1:])

    for file_path in sys.argv[1:]:
        print(file_path)
        data, md = load_hopkins(file_path)
        print_metadata(md, '  ')
        fig, ax = plt.subplots()
        ax.set_title(md.get('speciesIDweb', ''))
        for k in range(len(data)):
            ax.plot(1000*data[k][:, 0], data[k][:, 1])
            ax.set_xlabel('Time [ms]')
        plt.show()
        print()
