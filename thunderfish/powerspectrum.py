import numpy as np
import matplotlib.mlab as ml

def psd(data, samplerate, fresolution):
    """
    Calculates a Powerspecturm.

    This function takes a data array, its samplerate and a frequencyresolution for the powerspectrum.
    With this input it first calculates a nfft value and later a powerspectrum.

    :param data:                (1-D array) data array you want to calculate a psd of.
    :param samplerate:          (float) sampling rate of the data that you want to calculate a psd of.
    :param fresolution:         (float) frequency resolution of the psd.
    :return:                    (2-D array) contains the power and frequency calculated in the powerspectrum.
    """

    nfft = int(np.round(2 ** (np.floor(np.log(samplerate / fresolution) / np.log(2.0)) + 1.0)))
    if nfft < 16:
        nfft = 16
    power, freqs = ml.psd(data, NFFT=nfft, noverlap=nfft / 2, Fs=samplerate, detrend=ml.detrend_mean)
    return [power, freqs]

def powerspectrum_plot(power, freqs, ax):
    """
    Plots a powerspectum.

    :param power:               (1-D array) power array of a psd.
    :param freqs:               (1-D array) frequency array of a psd.
    :param ax:                  (axis for plot) empty axis that is filled with content in the function.
    :return ax:                 (axis for plot) axis that is ready for plotting containing the powerspectrum.
    """
    ax.plot(freqs, 10.0 * np.log10(power))
    ax.set_ylabel('power [dB]')
    ax.set_xlabel('frequency [Hz]')
    ax.set_xlim([0, 3000])

def powerspectrum(data, samplerate, fresolution=[0.5], plot_data_func=None, **kwargs):
    """
    This function is performing the steps to calculate a powerspectrum on the basis of a given dataset, a given
    samplingrate and a given frequencyresolution for the psd. Therefore two other functions are called to first
    calculate the nfft value and second calculate the powerspectrum.

    :param data:                (1-D array) data array you want to calculate a psd of.
    :param samplerate:          (float) sampling rate of the data that you want to calculate a psd of.
    :param fresolution:         (1-D array) frequency resolutions for one or multiple psds.
    :param plot_data_func:      (function) function (powerspectrumplot()) that is used to create a axis for later
                                plotting containing the calculated powerspectrum.
    :param **kwargs:            additional arguments that are passed to the plot_data_func().
    :return power:              (1-D array) power array of the psd.
    :return freqs:              (1-D array) psd array of the psd.
    :return multi_psd_data:     (3-D or 2-D array) if the psd shall only be calculated for one frequency resolution
                                this Outupt is a 2-D array ( psd_data[power, freq] )
                                If the psd shall be calculated for multiple frequency resolutions its a 3-D array
                                (psd_data[frequency_resolution][power, freq])
    :return ax:                 (axis for plot) axis that is ready for plotting containing a figure that shows what the
                                modul did.
    """
    print('\nCoumputing powerspectrum for %0.f frequency resolutions ...' % len(fresolution))

    multi_psd_data = []
    for fres in fresolution:
        psd_data = psd(data, samplerate, fres)
        multi_psd_data.append(psd_data)

    if plot_data_func:
        plot_data_func(multi_psd_data[0][0], multi_psd_data[0][1], **kwargs)

    if len(multi_psd_data) == 1:
        multi_psd_data = multi_psd_data[0]

    if plot_data_func:
        plot_data_func(psd_data[0], psd_data[1], **kwargs)

    return multi_psd_data

if __name__ == '__main__':

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
    psd_data = powerspectrum(data, samplingrate, plot_data_func=powerspectrum_plot, ax=ax)
    plt.show()
