import numpy as np
import matplotlib.mlab as ml

def calc_nfft(samplingrate, fresolution):
    """
    This function calcualtes a nfft value depending on samplingrate and frequencyresolution of the powerspectrum.

    :param samplingrate: (float)
    :param fresolution: (float)
    :return nfft: (float)
    """
    nfft = int(np.round(2 ** (np.floor(np.log(samplingrate / fresolution) / np.log(2.0)) + 1.0)))
    if nfft < 16:
        nfft = 16
    return nfft

def calc_psd(data, samplingrate, nfft):
    """
    This function is calcualting a powerspectrum of a given data-array when nfft and samplingrate is given as argument.

    :param data: (1-D array)
    :param samplingrate: (int)
    :param nfft: (float)
    :return power: (1-D array)
    :return freqs: (1-D array)
    """
    power, freqs = ml.psd(data, NFFT=nfft, noverlap=nfft / 2, Fs=samplingrate, detrend=ml.detrend_mean)
    return power, freqs

def powerspectrum_main(data, samplingrate, fresolution=0.5):
    """
    This function is performing the steps to calculate a powerspectrum on the basis of a given dataset, a given
    samplingrate and a given frequencyresolution for the psd. Therefore two other functions are called to first
    calculate the nfft value and second calculate the powerspectrum.

    :param data: (1-D array)
    :param samplingrate: (int)
    :param fresolution: (float)
    :return power: (1-D array)
    :return freqs:(1-D array)
    """
    print("calculating powerspecturm ...")

    nfft = calc_nfft(samplingrate, fresolution)
    power, freqs = calc_psd(data, samplingrate, nfft)
    return power, freqs

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import sys

    print('Calculating powerspectrum of a created signal of two wavefish (300 and 450 Hz)')
    print('')
    print('Usage:')
    print('  python powerspectrum.py [-p]')
    print('  -p: plot data')
    print('')

    plot = False
    if len(sys.argv) > 1 and sys.argv[1] == '-p':
        plot = True

    fundamental = [300, 450] # Hz
    samplingrate = 100000
    time = np.linspace(0, 8-1/samplingrate, 8*samplingrate)
    data = np.sin(time * 2 * np.pi* fundamental[0]) + np.sin(time * 2 * np.pi* fundamental[1])

    power, freqs = powerspectrum_main(data, samplingrate)

    if plot:
        fig, ax = plt.subplots()
        ax.plot(freqs, power)
        ax.set_ylabel('power')
        ax.set_xlabel('frequency [Hz]')
        ax.set_xlim([0, 3000])
        plt.show()