import os.path
import numpy as np
import audioloader

def load_pickle(filename, channel=0) :
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
    freq = 1000.0/(time[1]-time[0])
    tracen = data['raw_data'].shape[1]
    if channel >= tracen :
        print('number of channels in file %s is %d, but requested channel %d' %
              (filename, tracen, channel))
        channel = tracen-1
    return data['raw_data'][:,channel], freq, 'mV'


def load_data(filepath, channel=0, verbose=0) :
    """
    Call this function to load a single trace of data from a file.

    Args:
        filepath (string): the full path and name of the file to load
        channel (int): the single channel to be returned
        verbose (int): if >0 show detailed error/warning messages

    Returns:
        data (array): the data trace as a 1-D numpy array
        freq (float): the sampling rate of the data in Hz
        unit (string): the unit of the data
    """
    # check types:
    if not isinstance(filepath, basestring) :
        raise NameError('load_data(): input argument filepath must be a string!')
    if not isinstance(channel, int) :
        raise NameError('load_data(): input argument channel must be an int!')

    # check values:
    data = np.array([])
    freq = 0.0
    unit = ''
    if len(filepath) == 0 :
        print('load_data(): input argument filepath is empty string!')
        return data, freq, unit
    if not os.path.isfile(filepath) :
        print('load_data(): input argument filepath=%s does not indicate an existing file!' % filepath)
        return data, freq, unit
    if os.path.getsize(filepath) <= 0:
        print('load_data(): input argument filepath=%s indicates file of size 0!' % filepath)
        return data, freq, unit
    if channel < 0 :
        print('load_data(): input argument channel=%d is negative!' % channel)
        channel = 0

    # load data:
    ext = filepath.split('.')[-1]
    if ext == 'pkl' :
        data, freq, unit = load_pickle(filepath, channel)
    else :
        data, freq = audioloader.load_audio(filepath, verbose)
        channels = data.shape[1]
        if channel >= channels :
            print('number of channels in file %s is %d, but requested channel %d' %
                  (filename, channels, channel))
            channel = channels-1
        if channel >= 0 :
            data = data[:,channel]
        unit = 'a.u.'
    return data, freq, unit


if __name__ == "__main__":
    import sys
    print("Checking dataloader module ...")
    filepath = sys.argv[-1]
    channel = 0
    print('')
    print("try load_data:")
    freq, data, unit = load_data(filepath, channel, 2)
