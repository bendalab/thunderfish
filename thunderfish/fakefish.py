import numpy as np
from IPython import embed

import matplotlib.pyplot as plt


def generate_wavefish(freq, samplerate, time_len=20., harmonics=3):
    """

    :param freq: (float). Frequency of the fish in Hz.
    :param samplerate: (float). Sampling rate in Hz.
    :param time_len: (float). Length of the recording in sec.
    :param harmonics: (int). Number of harmonics of freq to be included.

    :return: (array). Data of a wavefish with given frequency.
    """
    time = np.arange(0, time_len, 1./samplerate)

    data = np.sin(2 * np.pi * time * freq)
    for har in range(1, harmonics):
        data += (np.sin(2*np.pi*time*(har*freq))) * 0.1*har

    # Insert some noise
    data += 0.05 * np.random.randn(len(data))

    return data


def generate_pulsefish(freq, samplerate, time_len=20., noise_fac=0.1,
                       pk_std=0.001, pk_amplitude=1.,
                       tr_std=0.001, tr_amplitude=0.5,
                       tr_time=0.005):
    """

    :param freq: (float). Frequency of the fish in Hz.
    :param samplerate: (float). Sampling Rate in Hz
    :param time_len: (float). Length of the recording in sec.
    :param noise_fac: (float). Factor by which random gaussian distributed noise is inserted.
    :param pk_std: (float). std of the positive part of the pulse in sec.
    :param pk_amplitude: (float). Factor for regulating the positive part of the pulse.
    :param tr_std: (float). std of the negative part of the pulse in sec.
    :param tr_amplitude: (float). Factor for regulating the negative part of the pulse.
    :param tr_time:

    :return: (array). Data with pulses at the given frequency.
    """
    # Create a Gaussian Distribution; one for the peak and another for the trough

    x = np.arange(-4.*pk_std, tr_time + 4.*tr_std, 1./samplerate)

    pulse = np.exp(-0.5 * (x/pk_std)**2) * pk_amplitude  # Peak Gaussian
    pulse -= np.exp(-0.5 * ((x-tr_time)/tr_std)**2) * tr_amplitude  # Trough Gaussian

    # Now we need to set the pulse into some baseline with noise.
    time = np.arange(0, time_len, 1. / samplerate)

    data = np.random.randn(len(time)) * noise_fac

    period = int(samplerate / freq)
    for s in range(period / 2, len(data) - len(pulse), period):
        data[s:s + len(pulse)] += pulse

    return data


if __name__ == '__main__':

    samplerate = 20000.  # in Hz
    fs = 16.  # Font size
    rec_length = 20.  # in sec
    inset_len = 0.02  # in sec

    time = np.arange(0, rec_length, 1./samplerate)

    pulsefish = generate_pulsefish(80., samplerate, time_len=rec_length,
                                   noise_fac=0.02,
                                   pk_std=0.001, pk_amplitude=1.,
                                   tr_std=0.001, tr_amplitude=0.5, tr_time=0.002)

    wavefish = generate_wavefish(300., samplerate, time_len=rec_length, harmonics=3)

    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(19, 10))

    # ax[0][0] is complete wavefish
    ax[0][0].plot(time, wavefish, color='dodgerblue', alpha=0.7, lw=2)
    ax[0][0].set_title('Fake-wavefish-RECORDING', fontsize=fs+2)

    # ax[0][1] is wavefish inset
    ax[0][1].plot(time[:samplerate*inset_len], wavefish[:samplerate*inset_len], '-o',
                  lw=3, color='dodgerblue', ms=10, mec='k', mew=1.5)
    ax[0][1].set_title('Fake-wavefish-INSET', fontsize=fs + 2)

    # ax[1][0] is complete pulsefish
    ax[1][0].plot(time, pulsefish, color='forestgreen', alpha=0.7, lw=2)
    ax[1][0].set_title('Fake-pulsefish-RECORDING', fontsize=fs+2)

    # get proper ylim
    ymin = np.min(pulsefish)
    ymax = np.max(pulsefish)
    ws_fac = 1.2  # Whitespace factor
    ax[1][0].set_ylim([ymin*ws_fac, ymax*ws_fac])

    # ax[1][1] is pulsefish inset
    ax[1][1].plot(time[:samplerate * inset_len], pulsefish[:samplerate * inset_len], '-o',
                  lw=3, color='forestgreen', ms=10, mec='k', mew=1.5)
    ax[1][1].set_title('Fake-pulsefish-INSET', fontsize=fs + 2)

    # get proper ylim
    ymin = np.min(pulsefish)
    ymax = np.max(pulsefish)
    yrange = np.abs(ymin) + np.abs(ymax)
    ws_fac = 0.1  # Whitespace factor (between 0. and 1.; preferably a small number)
    ax[1][0].set_ylim([ymin - yrange*ws_fac, ymax + yrange*ws_fac])
    ax[1][1].set_ylim([ymin - yrange*ws_fac, ymax + yrange*ws_fac])

    for row in ax:
        for c_ax in row:
            c_ax.set_xlabel('Time [sec]', fontsize=fs)
            c_ax.set_ylabel('Amplitude [a.u.]', fontsize=fs)
            c_ax.tick_params(axis='both', which='major', labelsize=fs - 2)

    plt.tight_layout()
    plt.show()
