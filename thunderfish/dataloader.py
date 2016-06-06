import os.path
import numpy as np
import audioio as aio


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
    if not isinstance(filepath, basestring):
        raise NameError('load_data(): input argument filepath must be a string!')
    if not isinstance(channel, int):
        raise NameError('load_data(): input argument channel must be an int!')

    # check values:
    data = np.array([])
    freq = 0.0
    unit = ''
    if len(filepath) == 0:
        print('load_data(): input argument filepath is empty string!')
        return data, freq, unit
    if not os.path.isfile(filepath):
        print('load_data(): input argument filepath=%s does not indicate an existing file!' % filepath)
        return data, freq, unit
    if os.path.getsize(filepath) <= 0:
        print('load_data(): input argument filepath=%s indicates file of size 0!' % filepath)
        return data, freq, unit
    if channel < 0:
        print('load_data(): input argument channel=%d is negative!' % channel)
        channel = 0

    # load data:
    ext = filepath.split('.')[-1]
    if ext == 'pkl':
        data, freq, unit = load_pickle(filepath, channel)
    else:
        data, freq = aio.load_audio(filepath, verbose)
        channels = data.shape[1]
        if channel >= channels:
            print('number of channels in file %s is %d, but requested channel %d' %
                  (filepath, channels, channel))
            channel = channels - 1
        if channel >= 0:
            data = data[:, channel]
        unit = 'a.u.'
    return data, freq, unit


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
        if hasattr(key, '__len__'):
            raise IndexError
        return super(DataLoader, self).__getitem__((key, self.channel))

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
        print('channels: %d %d' % (data.channels, data.shape[1]))
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
