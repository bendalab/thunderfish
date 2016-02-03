from collections import namedtuple
import inspect
from numpy import fft
import numpy as np
from scipy.signal import butter, filtfilt
import types
from matplotlib.pyplot import subplots, show


def butter_lowpass(highcut, samplingrate, order=5):
    nyq = 0.5 * samplingrate
    high = highcut / nyq
    b, a = butter(order, high, btype='low')
    return b, a


def butter_lowpass_filter(data, highcut, samplingrate, order=5):
    b, a = butter_lowpass(highcut, samplingrate, order=order)
    y = filtfilt(b, a, data)
    return y

def butter_highpass(lowcut, samplingrate, order=5):
    nyq = 0.5 * samplingrate
    high = lowcut / nyq
    b, a = butter(order, high, btype='high')
    return b, a


def butter_highpass_filter(data, lowcut, samplingrate, order=5):
    b, a = butter_lowpass(lowcut, samplingrate, order=order)
    y = filtfilt(b, a, data)
    return y


def fraction_amplitude_above(x, samplingrate, lowcut):
    w = fft.fftfreq(len(x), 1./samplingrate)
    ampl_spec = np.abs(fft.fft(x))
    return np.sum(ampl_spec[np.abs(w) > lowcut])/np.sum(ampl_spec)

class ProcessorFactory:

    def __init__(self, **kwargs):
        self.globals = kwargs

    def __getitem__(self, item):
        return self.globals.__getitem__(item)

    def __setitem__(self, key, value):
        return self.globals.__setitem__(key, value)

    def __call__(self, f, **kwargs):
        kw_names, _, _, defaults = inspect.getargspec(f)

        if defaults is not None:
            n = len(defaults)
            nkwargs = dict(zip(kw_names[-n:], defaults[-n:]))
        else:
            nkwargs = {}


        for arg in kw_names[1:]:
            if arg in self.globals:
                nkwargs[arg] = self.globals[arg]

            if arg in kwargs:
                nkwargs[arg] = kwargs[arg]
        sdiff = set(kw_names[1:]).difference(nkwargs.keys())
        assert len(sdiff) == 0, "Cannot call %s, because arguments %s are missing" % (f.__name__, ', '.join(sdiff))
        return lambda x: f(x, **nkwargs)

class Chain:

    def __init__(self, *args):
        self._items = args

    def __add__(self, other):
        return Chain( *(self._items + other._items) )

    def __call__(self, x):
        for i, (name, condition, alt1, alt2) in enumerate(self._items):
            print 'Step {0:d}: {1:s}'.format(i, name)
            if (condition if type(condition) == types.BooleanType else condition(x)):
                print "\t Alternative 1"
                x = alt1(x)
            elif alt2 is not None:
                print "\t Alternative 2"
                x = alt2(x)
        return x


if __name__=="__main__":

    PF = ProcessorFactory(samplingrate = 20000, lowcut = 2000., highcut = 2000.)

    standardize = Chain(
        ('Center', True, lambda x: x - np.mean(x), None),
        ('Normalize', True, lambda x: x/np.std(x, ddof=1), None),
    )

    preprocessing = Chain(
        ('Filter', lambda x: PF(fraction_amplitude_above)(x) > 0.9,
                   PF(butter_highpass_filter),
                   PF(butter_lowpass_filter)
        ),
        ('Square', True, lambda x: x**2, None),
        ('Standardize', True, standardize, None)
    )

    x = np.random.randn(20000)
    y = preprocessing(x)
    fig, ax = subplots()
    ax.plot(fft.fftfreq(len(x), 1./PF['samplingrate']), abs(fft.fft(y)))
    show()
