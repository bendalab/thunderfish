"""

"""

import numpy as np
import matplotlib.mlab as mlab

def psd(data, samplerate, fresolution, detrend=mlab.detrend_none,
    window=mlab.window_hanning, overlap=0.5, pad_to=None,
    sides='default', scale_by_freq=None):
    """
    Calculates a Powerspecturm.

    This function takes a data array, its samplerate and a frequencyresolution for the powerspectrum.
    With this input it first calculates a nfft value and later a powerspectrum.

    (for further argument information see numpy.psd documentation)
    :param data:                (1-D array) data array you want to calculate a psd of.
    :param samplerate:          (float) sampling rate of the data that you want to calculate a psd of.
    :param fresolution:         (float) frequency resolution of the psd.
    :return:                    (2-D array) contains the power and frequency calculated in the powerspectrum.
    """

    nfft = int(np.round(2 ** (np.floor(np.log(samplerate / fresolution) / np.log(2.0)) + 1.0)))
    if nfft < 16:
        nfft = 16
    noverlap = nfft*overlap
    power, freqs = mlab.psd(data, NFFT=nfft, noverlap=noverlap, Fs=samplerate, detrend=detrend, window=window,
                            pad_to=pad_to, sides=sides, scale_by_freq=scale_by_freq)

    return np.asarray([power, freqs])

def plot_decibel_psd(power, freqs, ax, fs, max_freq=3000, color='blue', alpha=1., verbose=0):
    """
    Plots a powerspectum.

    :param power:               (1-D array) power array of a psd.
    :param freqs:               (1-D array) frequency array of a psd.
    :param ax:                  (axis for plot) empty axis that is filled with content in the function.
    :param fs:                  (int) fontsize for the plot.
    :param max_freq:            (float) maximum frequency that shall appear in the plot.
    :param color:               (string) color that shall be used for the plot.
    :param alpha:               (float) transparency of the plot.
    :param verbose:             (int) when the value is 1 you get additional shell output.
    :return ax:                 (axis for plot) axis that is ready for plotting containing the powerspectrum.
    """
    if verbose >=1:
        print('create PSD plot...')
    power_cp = power.copy()
    power_cp[power_cp < 1e-20] = np.nan

    decibel_psd = 10.0 * np.log10(power_cp)
    ax.plot(freqs, decibel_psd, color=color, alpha=alpha)
    ax.set_ylabel('power [dB]', fontsize=fs)
    ax.set_xlabel('frequency [Hz]', fontsize=fs)
    ax.set_xlim([0, max_freq])

def multi_resolution_psd(data, samplerate, fresolution=0.5, detrend=mlab.detrend_none, window=mlab.window_hanning,
                         overlap=0.5, pad_to=None, sides='default', scale_by_freq=None, verbose=0):
    """
    This function is performing the steps to calculate a powerspectrum on the basis of a given dataset, a given
    samplingrate and a given frequencyresolution for the psd. Therefore two other functions are called to first
    calculate the nfft value and second calculate the powerspectrum.

    (for further argument information see numpy.psd documentation)
    :param data:                (1-D array) data array you want to calculate a psd of.
    :param samplerate:          (float) sampling rate of the data that you want to calculate a psd of.
    :param fresolution:         (1-D array) frequency resolutions for one or multiple psds.
    :param plot_data_func:      (function) function (powerspectrum_plot()) that is used to create a axis for later
                                plotting containing the calculated powerspectrum.
    :param **kwargs:            additional arguments that are passed to the plot_data_func().
    :return multi_psd_data:     (3-D or 2-D array) if the psd shall only be calculated for one frequency resolution
                                this Outupt is a 2-D array ( psd_data[power, freq] )
                                If the psd shall be calculated for multiple frequency resolutions its a 3-D array
                                (psd_data[frequency_resolution][power, freq])
    :return ax:                 (axis for plot) axis that is ready for plotting containing a figure that shows what the
                                modul did.
    """
    if verbose >=1:
        print('Coumputing powerspectrum ...')

    return_list = True
    if not hasattr(fresolution, '__len__'):
        return_list = False
        fresolution = [fresolution]

    multi_psd_data = []
    for fres in fresolution:
        psd_data = psd(data, samplerate, fres, detrend, window, overlap, pad_to, sides, scale_by_freq)
        multi_psd_data.append(psd_data)

    if not return_list:
        multi_psd_data = multi_psd_data[0]

    return np.asarray(multi_psd_data)

if __name__ == '__main__':
    import matplotlib.pyplot as plt

    print('Computes powerspectrum of a created signal of two wavefish (300 and 450 Hz)')
    print('')
    print('Usage:')
    print('  python powerspectrum.py')
    print('')

    fundamental = [300, 450]  # Hz
    samplerate = 100000
    time = np.linspace(0, 8 - 1 / samplerate, 8 * samplerate)
    data = np.sin(time * 2 * np.pi * fundamental[0]) + np.sin(time * 2 * np.pi * fundamental[1])

    psd_data = multi_resolution_psd(data, samplerate, fresolution=[0.5, 1], verbose=1)

    fig, ax = plt.subplots()
    plot_decibel_psd( psd_data[0][0], psd_data[0][1], ax=ax, fs = 12, color='firebrick', alpha=0.9, verbose= 1)
    plt.show()
