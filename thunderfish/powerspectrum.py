import numpy as np
import matplotlib.mlab as ml

def calc_nfft(samplingrate, fresolution):
    """
    This function calcualtes a nfft value depending on samplingrate and frequencyresolution of the powerspectrum.

    :param samplingrate:        (float) sampling rate of the data that you want to calculate a psd of.
    :param fresolution:         (float) frequency resolution of the psd
    :return nfft:               (float) the value that defines the resolution if the psd.
    """
    nfft = int(np.round(2 ** (np.floor(np.log(samplingrate / fresolution) / np.log(2.0)) + 1.0)))
    if nfft < 16:
        nfft = 16
    return nfft

def calc_psd(data, samplingrate, nfft):
    """
    This function is calcualting a powerspectrum of a given data-array when nfft and samplingrate is given as argument.

    :param data:                (1-D array) data array you want to calculate a psd of.
    :param samplingrate:        (float) sampling rate of the data that you want to calculate a psd of.
    :param nfft:                (float) the value that defines the resolution if the psd.
    :return power:              (1-D array) power array of the psd.
    :return freqs:              (1-D array) psd array of the psd.
    """
    power, freqs = ml.psd(data, NFFT=nfft, noverlap=nfft / 2, Fs=samplingrate, detrend=ml.detrend_mean)
    return power, freqs

def powerspectrumplot(power, freqs, ax):
    """
    Plots a powerspectum.

    :param power:               (1-D array) power array of a psd.
    :param freqs:               (1-D array) frequency array of a psd.
    :param ax:                  (axis for plot) empty axis that is filled with content in the function.
    :return ax:                 (axis for plot) axis that is ready for plotting containing the powerspectrum.
    """
    ax.plot(freqs, power)
    ax.set_ylabel('power')
    ax.set_xlabel('frequency [Hz]')
    ax.set_xlim([0, 3000])
    return ax

def powerspectrum_main(data, samplingrate, fresolution=0.5, plot_data_func=None, **kwargs):
    """
    This function is performing the steps to calculate a powerspectrum on the basis of a given dataset, a given
    samplingrate and a given frequencyresolution for the psd. Therefore two other functions are called to first
    calculate the nfft value and second calculate the powerspectrum.

    :param data:                (1-D array) data array you want to calculate a psd of.
    :param samplingrate:        (float) sampling rate of the data that you want to calculate a psd of.
    :param fresolution:         (float) frequency resolution of the psd
    :param plot_data_func:      (function) function (powerspectrumplot()) that is used to create a axis for later plotting containing the calculated powerspectrum.
    :param **kwargs:            additional arguments that are passed to the plot_data_func().
    :return power:              (1-D array) power array of the psd.
    :return freqs:              (1-D array) psd array of the psd.
    :return ax:                 (axis for plot) axis that is ready for plotting containing a figure that shows what the modul did.
    """

    # print("calculating powerspecturm ...")

    nfft = calc_nfft(samplingrate, fresolution)
    power, freqs = calc_psd(data, samplingrate, nfft)

    if plot_data_func:
        ax = plot_data_func(power, freqs, **kwargs)
        return power, freqs, ax
    else:
        return power, freqs

if __name__ == '__main__':
    import sys

    print('Computes powerspectrum of a created signal of two wavefish (300 and 450 Hz)')
    print('')
    print('Usage:')
    print('  python powerspectrum.py')
    print('')

    fundamental = [300, 450] # Hz
    samplingrate = 100000
    time = np.linspace(0, 8-1/samplingrate, 8*samplingrate)
    data = np.sin(time * 2 * np.pi* fundamental[0]) + np.sin(time * 2 * np.pi* fundamental[1])

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    power, freqs, ax = powerspectrum_main(data, samplingrate, plot_data_func=powerspectrumplot, ax=ax)
    plt.show()
