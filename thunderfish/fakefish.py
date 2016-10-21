"""
Generate artificial waveforms of weakly electric fish.

The two main functions are

generate_wavefish()
generate_pulsefish()

for generating EODs of wave-type and pulse_type electric fish, respectively.

The following functions use the two functions to generate waveforms of specific fishes:

generate_alepto(): mimicks the wave-type fish Apteronotus leptorhynchus, 
generate_eigenmannia(): mimicks the wave-type fish Eigenmannia, 

generate_monophasic_pulses(): mimicks a monophasic pulsefish,
generate_biphasic_pulses(): mimicks a biphasic pulsefish,
generate_triphasic_pulses(): mimicks a triphasic pulsefish.
"""

import numpy as np


def generate_wavefish(frequency=100.0, samplerate=44100., duration=1., noise_std=0.05,
                      amplitudes=1.0, phases=0.0):
    """
    Generate EOD of a wave-type fish.

    The waveform is constructed by superimposing sinewaves of integral multiples of
    the fundamental frequency - the fundamental and its harmonics.
    The fundamental frequency of the EOD is given by frequency. The amplitude of the
    fundamental is given by the first element in amplitudes. The amplitudes and
    relative phases of higher harmonics are give by optional further elements of
    the amplitudes and phases lists.

    The generated waveform is duration seconds long and is sampled with samplerate Hertz.
    Gaussian white noise with a standard deviation of noise_std is added to the generated
    waveform.

    Parameters
    ----------
    frequency: float or array of floats
        EOD frequency of the fish in Hz. Either fixed number or array for
        time-dependent frequencies.
    samplerate: float
        Sampling rate in Hz.
    duration: float
        Duration of the generated data in seconds. Only used if frequency is float.
    noise_std: float
        Standard deviation of additive Gaussian white noise.
    amplitudes: float or list of floats
        Amplitudes of fundamental and optional harmonics.
    phases: float or list of floats
        Relative phases of fundamental and optional harmonics in radians.

    Returns
    -------
    data: array of floats
        Generated data of a wave-type fish.

    Raises
    ------
    IndexError: amplitudes and phases differ in length.
    """
    
    # compute phase:
    if np.isscalar(frequency):
        phase = np.arange(0, duration, 1./samplerate)
        phase *= frequency
    else:
        phase = np.cumsum(frequency)/samplerate
    # fix amplitudes and phases:
    if np.isscalar(amplitudes):
        amplitudes = [amplitudes]
    if np.isscalar(phases):
        phases = [phases]
    if len(amplitudes) != len(phases):
        raise IndexError('need exactly as many phases as amplitudes')
    # generate EOD:
    data = np.zeros(len(phase))
    for har, (ampl, phi) in enumerate(zip(amplitudes, phases)):
        data += ampl * np.sin(2*np.pi*(har+1)*phase+phi)
    # add noise:
    data += noise_std * np.random.randn(len(data))
    return data


def generate_alepto(frequency=100.0, samplerate=44100., duration=1., noise_std=0.01):
    """Generate EOD of a Apteronotus leptorhynchus.

    See generate_wavefish() for details.
    """
    return generate_wavefish(frequency=frequency, samplerate=samplerate, duration=duration,
                             noise_std=noise_std, amplitudes=[1.0, 0.5, 0.0, 0.01],
                             phases=[0.0, 0.0, 0.0, 0.0])


def generate_eigenmannia(frequency=100.0, samplerate=44100., duration=1., noise_std=0.01):
    """Generate EOD of an Eigenmannia.

    See generate_wavefish() for details.
    """
    return generate_wavefish(frequency=frequency, samplerate=samplerate, duration=duration,
                             noise_std=noise_std, amplitudes=[1.0, 0.25, 0.0, 0.01],
                             phases=[0.0, 0.5*np.pi, 0.0, 0.0])



def generate_pulsefish(frequency=100.0, samplerate=44100., duration=1., noise_std=0.01,
                       jitter_cv=0.1, peak_stds=0.001, peak_amplitudes=1.0, peak_times=0.0):
    """
    Generate EOD of a pulse-type fish.

    Pulses are spaced by 1/frequency, jittered as determined by jitter_cv. Each pulse is
    a combination of Gaussian peaks, whose widths, amplitudes, and positions are given by
    their standard deviation peak_stds, peak_amplitudes, and peak_times, respectively.

    The generated waveform is duration seconds long and is sampled with samplerate Hertz.
    Gaussian white noise with a standard deviation of noise_std is added to the generated
    pulse train.

    Parameters
    ----------
    frequency: float
        EOD frequency of the fish in Hz.
    samplerate: float
        Sampling Rate in Hz.
    duration: float
        Duration of the generated data in seconds.
    noise_std: float
        Standard deviation of additive Gaussian white noise.
    jitter_cv: float
        Gaussian distributed jitter of pulse times as coefficient of variation of inter-pulse intervals.
    peak_stds: float or list of floats
        Standard deviation of Gaussian shaped peaks in seconds.
    peak_amplitudes: float or list of floats
        Amplitude of each peak (positive and negative).
    peak_times: float or list of floats
        Position of each Gaussian peak in seconds.

    Returns
    -------
    data: array of floats
        Generated data of a pulse-type fish.

    Raises
    ------
    IndexError: peak_stds or peak_amplitudes or peak_times differ in length.
    """

    # make sure peak properties are in a list:
    if np.isscalar(peak_stds):
        peak_stds = [peak_stds]
    if np.isscalar(peak_amplitudes):
        peak_amplitudes = [peak_amplitudes]
    if np.isscalar(peak_times):
        peak_times = [peak_times]
    if len(peak_stds) != len(peak_amplitudes) or len(peak_stds) != len(peak_times):
        raise IndexError('need exactly as many peak_stds as peak_amplitudes and peak_times')

    # time axis for single pulse:
    min_time_inx = np.argmin(peak_times)
    max_time_inx = np.argmax(peak_times)
    x = np.arange(-4.*peak_stds[min_time_inx] + peak_times[min_time_inx],
                  4.*peak_stds[max_time_inx] + peak_times[max_time_inx], 1.0/samplerate)
    pulse_duration = x[-1] - x[0]
    
    # generate a single pulse:
    pulse = np.zeros(len(x))
    for time, ampl, std in zip(peak_times, peak_amplitudes, peak_stds):
        pulse += ampl * np.exp(-0.5*((x-time)/std)**2) 

    # paste the pulse into the noise floor:
    time = np.arange(0, duration, 1. / samplerate)
    data = np.random.randn(len(time)) * noise_std
    period = 1.0/frequency
    jitter_std = period * jitter_cv
    first_pulse = np.max(pulse_duration, 3.0*jitter_std)
    pulse_times = np.arange(first_pulse, duration, period )
    pulse_times += np.random.randn(len(pulse_times)) * jitter_std
    pulse_indices = np.round(pulse_times * samplerate).astype(np.int)
    for inx in pulse_indices[(pulse_indices >= 0) & (pulse_indices < len(data)-len(pulse)-1)]:
        data[inx:inx + len(pulse)] += pulse

    return data


def generate_monophasic_pulses(frequency=100.0, samplerate=44100., duration=1.,
                               noise_std=0.01, jitter_cv=0.1):
    """Generate EOD of a monophasic pulse-type fish.

    See generate_pulsefish() for details.
    """
    return generate_pulsefish(frequency=frequency, samplerate=samplerate, duration=duration,
                              noise_std=noise_std, jitter_cv=jitter_cv,
                              peak_stds=0.0003, peak_amplitudes=1.0, peak_times=0.0)


def generate_biphasic_pulses(frequency=100.0, samplerate=44100., duration=1.,
                              noise_std=0.01, jitter_cv=0.1):
    """Generate EOD of a biphasic pulse-type fish.

    See generate_pulsefish() for details.
    """
    return generate_pulsefish(frequency=frequency, samplerate=samplerate, duration=duration,
                              noise_std=noise_std, jitter_cv=jitter_cv,
                              peak_stds=[0.0001, 0.0002],
                              peak_amplitudes=[1.0, -0.3],
                              peak_times=[0.0, 0.0003])


def generate_triphasic_pulses(frequency=100.0, samplerate=44100., duration=1.,
                              noise_std=0.01, jitter_cv=0.1):
    """Generate EOD of a triphasic pulse-type fish.

    See generate_pulsefish() for details.
    """
    return generate_pulsefish(frequency=frequency, samplerate=samplerate, duration=duration,
                              noise_std=noise_std, jitter_cv=jitter_cv,
                              peak_stds=[0.0001, 0.0001, 0.0002],
                              peak_amplitudes=[1.0, -0.8, 0.1],
                              peak_times=[0.0, 0.00015, 0.0004])

def generate_alepto_with_chirps():
    samplerate = 40000.0
    duration = 6.0
    feod = 820.0
    frequency = feod * np.ones(int(duration*samplerate))
    chirp_times = np.arange(0.5, 5.5, 1.5)
    chirp_width = 0.005
    chirp_t = np.arange(-4.0*chirp_width, 4.0*chirp_width, 1./samplerate)
    chirp = 100.0 * np.exp(-0.5*(chirp_t/chirp_width)**2.0)
    for c in chirp_times:
        index = int(c*samplerate)
        frequency[index:index+len(chirp)] += chirp
    phase = np.cumsum(frequency)/samplerate
    raw_data = np.sin(2.0*np.pi*phase)
    raw_data += 0.5*np.sin(2.0*np.pi*2.0*phase)
    raw_data += 0.1*np.sin(2.0*np.pi*3.0*phase)
    raw_data += 0.1*np.random.randn(len(raw_data))
    plt.psd(raw_data, Fs=samplerate, NFFT=4096*4)
    plt.show()


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    samplerate = 40000.  # in Hz
    rec_length = 1.  # in sec
    inset_len = 0.01  # in sec
    inset_indices = int(inset_len*samplerate)
    ws_fac = 0.1  # whitespace factor or ylim (between 0. and 1.; preferably a small number)

    # generate data:
    time = np.arange(0, rec_length, 1./samplerate)

    eodf = 300.0
    eodf = 500.0 - time/rec_length*400.0
    wavefish = generate_wavefish(eodf, samplerate, duration=rec_length, noise_std=0.02, 
                                 amplitudes=[1.0, 0.5, 0.0, 0.0001],
                                 phases=[0.0, 0.0, 0.0, 0.0])
    # wavefish = generate_alepto(eodf, samplerate, duration=rec_length)
    # wavefish = generate_eigenmannia(eodf, samplerate, duration=rec_length)

    pulsefish = generate_pulsefish(80., samplerate, duration=rec_length,
                                   noise_std=0.02, jitter_cv=0.1,
                                   peak_stds=[0.0001, 0.0002, 0.0002],
                                   peak_amplitudes=[1.0, -0.3, -0.1],
                                   peak_times=[0.0, 0.0003, -0.0004])
    # pulsefish = generate_monophasic_pulses(80., samplerate, duration=rec_length)
    # pulsefish = generate_biphasic_pulses(80., samplerate, duration=rec_length)
    pulsefish = generate_triphasic_pulses(80., samplerate, duration=rec_length)

    # plot:
    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(19, 10))

    # get proper wavefish ylim
    ymin = np.min(wavefish)
    ymax = np.max(wavefish)
    dy = ws_fac*(ymax - ymin)
    ymin -= dy
    ymax += dy

    # complete wavefish:
    ax[0][0].set_title('Wavefish')
    ax[0][0].set_ylim(ymin, ymax)
    ax[0][0].plot(time, wavefish)

    # wavefish zoom in:
    ax[0][1].set_title('Wavefish ZOOM IN')
    ax[0][1].set_ylim(ymin, ymax)
    ax[0][1].plot(time[:inset_indices], wavefish[:inset_indices], '-o')

    # get proper pulsefish ylim
    ymin = np.min(pulsefish)
    ymax = np.max(pulsefish)
    dy = ws_fac*(ymax - ymin)
    ymin -= dy
    ymax += dy

    # complete pulsefish:
    ax[1][0].set_title('Pulsefish')
    ax[1][0].set_ylim(ymin, ymax)
    ax[1][0].plot(time, pulsefish)

    # pulsefish zoom in:
    ax[1][1].set_title('Pulsefish ZOOM IN')
    ax[1][1].set_ylim(ymin, ymax)
    ax[1][1].plot(time[:inset_indices/2], pulsefish[:inset_indices/2], '-o')
    
    for row in ax:
        for c_ax in row:
            c_ax.set_xlabel('Time [sec]')
            c_ax.set_ylabel('Amplitude [a.u.]')

    plt.tight_layout()
    plt.show()
