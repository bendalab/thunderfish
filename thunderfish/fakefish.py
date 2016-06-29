import numpy as np
import matplotlib.pyplot as plt
from IPython import embed


def generate_wavefish(freq, samplerate, time_len=20., harmonics=3):

    # generate time
    """

    :param freq: (float). Frequency of the fish in Hz.
    :param samplerate: (float). Sampling rate in Hz.
    :param time_len: (float). Length of the recording in sec.
    :param harmonics: (int). Number of harmonics of freq to be included.

    :return: (array). Data of a wavefish with given frequency.
    """
    time = np.arange(0, time_len, 1./samplerate)

    data = np.sin(2 * np.pi * time * freq)
    for har in range(harmonics):
        if har > 1:
            data += (np.sin(2*np.pi*time*(har*freq))) * 0.1*har

    # Insert some noise
    data += 0.05 * np.random.randn(len(data))

    return data


def generate_pulsefish(freq, samplerate, time_len=20., noise_fac=0.1, wave_cut=0.6,
                       pk_len=0.001, pk_std=0.1, peak_fac=2.,
                       tr_len=0.001, tr_std=0.12, tr_fac=1.2):

    """

    :param freq: (float). Frequency of the fish in Hz.
    :param samplerate: (float). Sampling Rate in Hz
    :param time_len: (float). Length of the recording in sec.
    :param noise_fac: (float). Factor by which random gaussian distributed noise is inserted.
    :param wave_cut: (float). Ideally between 0.5 and 1. marks the percentage at which the distribution is cut.
    :param pk_len: (float). length of the positive part of the pulse.
    :param pk_std: (float). std of the positive part of the pulse.
    :param peak_fac: (float). Factor for regulating the positive part of the pulse.
    :param tr_len: (float). length of the negative part of the pulse.
    :param tr_std: (float). std of the negative part of the pulse.
    :param tr_fac: (float). Factor for regulating the negative part of the pulse.

    :return: (array). Data with pulses at the given frequency.
    """

    def gaussian(x, mu, si):
        """
        Standard Gaussian function taken from Wikipedia.

        :param x: (array-like). An arange of numbers.
        :param mu: (float). The Mean, which in this case should be preferably 0.
        :param si: (float). The Standard Deviation.
        :return:
        """
        return (1./np.sqrt(2*si**2*np.pi)) * np.exp(-(x-mu)**2/(2*si**2))

    def get_slope(peak, trough, length_percent=0.05):
        """Calculates the slope that connects the positive part of the pulse with the negative one.

        :param peak: (array-like). the positive values of the pulse
        :param trough: (array-like). the negative values of the pulse
        :param length_percent: (float). How long (in percent) the slope should be compared to the whole pulse length
        :return: (np.array). returns the slope values
        """
        slope_len = (len(peak) + len(trough)) * length_percent  # length of slope respective to length of entire pulse
        s = np.linspace(peak[-1], trough[0], slope_len)
        if len(s) < 3:
            s = np.array([(peak[-1] + trough[0])/2.])
        return s

    def insert_pulses(freq, pulse, time_len, noise_fac):

        """Insert pulses into noisy baseline at a given frequency

        :param freq: s.a.
        :param pulse: s.a.
        :param time_len: s.a.
        :param noise_fac: s.a.
        :return: (array). Data with pulses at the given frequency.
        """
        time = np.arange(0, time_len, 1. / samplerate)

        noise = (np.random.randn(len(time)) * noise_fac)
        dat = np.zeros(len(time)) * noise

        bl_idx = np.arange(len(dat))
        pulse_start = bl_idx[bl_idx % (samplerate/freq) == 0.]

        for s in pulse_start:
            dat[s:s+len(pulse)] = pulse

        return dat

    # Create a pulse inset

    # Create a Gaussian Distribution; one for the peak and another for the trough
    mu = 0.  # Fix parameter, should not be changed
    pk_gaus = gaussian(np.arange(-0.5, 0.5, 1./(samplerate * pk_len)), mu, pk_std)
    tr_gaus = -gaussian(np.arange(-0.5, 0.5, 1./(samplerate * tr_len)), mu, tr_std)

    pk_end = len(pk_gaus)*wave_cut
    tr_start = len(tr_gaus)*(1-wave_cut)

    peak = pk_gaus[:pk_end] * peak_fac
    trough = tr_gaus[tr_start:] * tr_fac

    slope = get_slope(peak, trough)

    pulse = np.hstack((np.hstack((peak, slope)), trough))  # This is a single pulse

    # Now we need to set the pulse into some baseline with noise.
    data = insert_pulses(freq, pulse, time_len, noise_fac)

    return data


if __name__ == '__main__':

    samplerate = 20000.  # in Hz
    fs = 16.  # Font size
    rec_length = 20.  # in sec
    inset_len = 0.02  # in sec

    time = np.arange(0, rec_length, 1./samplerate)

    pulsefish = generate_pulsefish(80., samplerate, time_len=rec_length,
                                   noise_fac=0.1, wave_cut=0.6,
                                   pk_len=0.001, pk_std=0.1, peak_fac=2.,
                                   tr_len=0.001, tr_std=0.12, tr_fac=1.2)

    wavefish = generate_wavefish(300., samplerate, time_len=rec_length, harmonics=3)

    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(19, 10))

    # embed()
    # quit()

    # ax[0] is complete wavefish
    ax[0][0].plot(time, wavefish, color='dodgerblue', alpha=0.7, lw=2)
    ax[0][0].set_title('Fake-wavefish-RECORDING', fontsize=fs+2)

    # ax[1] is wavefish inset
    ax[0][1].plot(time[:samplerate*inset_len], wavefish[:samplerate*inset_len], '-o',
                  lw=3, color='dodgerblue', ms=10, mec='k', mew=1.5)
    ax[0][1].set_title('Fake-wavefish-INSET', fontsize=fs + 2)

    for wave_ax in [ax[0][0], ax[0][1]]:
        wave_ax.set_xlabel('Time [sec]', fontsize=fs)
        wave_ax.set_ylabel('Amplitude [a.u.]', fontsize=fs)
        wave_ax.tick_params(axis='both', which='major', labelsize=fs - 2)

    plt.show()
