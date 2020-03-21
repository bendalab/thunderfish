"""
Simulate EOD waveforms.

- `wavefish_eods()`: simulate EOD waveform of a wave-type fish.
- `normalize_wavefish()`: normalize amplitudes and phases of EOD wave-type waveform.
- `export_wavefish()`: serialize wavefish parameter to file.
- `chirps()`: simulate frequency trace with chirps.
- `rises()`: simulate frequency trace with rises.
- `pulsefish_eods(): simulate EOD waveform of a pulse-type fish.
- `normalize_pulsefish()`: normalize times and stdevs of pulse-type EOD waveform.
- `export_pulsefish()`: serialize pulsefish parameter to file.
- `generate_waveform()`: interactively generate audio file with simulated EOD waveforms.
"""

from __future__ import print_function
import sys
import numpy as np


""" Amplitudes and phases of various wavefish species. """
Apteronotus_leptorhynchus_harmonics = dict(amplitudes=(0.90062, 0.15311, 0.072049, 0.012609, 0.011708),
                        phases=(1.3623, 2.3246, 0.9869, 2.6492, -2.6885))
Apteronotus_rostratus_harmonics = dict(amplitudes=(0.64707, 0.43874, 0.063592, 0.07379, 0.040199, 0.023073, 0.0097678),
                                       phases=(2.2988, 0.78876, -1.316, 2.2416, 2.0413, 1.1022, -2.0513))
Eigenmannia_harmonics = dict(amplitudes=(1.0087, 0.23201, 0.060524, 0.020175, 0.010087, 0.0080699),
                             phases=(1.3414, 1.3228, 2.9242, 2.8157, 2.6871, -2.8415))
Sternopygus_dariensis_harmonics = dict(amplitudes=(0.98843, 0.41228, 0.047848, 0.11048, 0.022801, 0.030706, 0.019018),
                                       phases=(1.4153, 1.3141, 3.1062, -2.3961, -1.9524, 0.54321, 1.6844))

""" Amplitudes and phases of EOD waveforms of various species of wave-type electric fish. """
wavefish_harmonics = dict(Alepto=Apteronotus_leptorhynchus_harmonics,
                          Arostratus=Apteronotus_rostratus_harmonics,
                          Eigenmannia=Eigenmannia_harmonics,
                          Sternopygus=Sternopygus_dariensis_harmonics)


def wavefish_eods(fish='Eigenmannia', frequency=100.0, samplerate=44100.0,
                  duration=1.0, phase0=0.0, noise_std=0.05):
    """
    Simulate EOD waveform of a wave-type fish.

    The waveform is constructed by superimposing sinewaves of integral
    multiples of the fundamental frequency - the fundamental and its
    harmonics.  The fundamental frequency of the EOD (EODf) is given by
    frequency. With 'fish' relative amplitudes and phases of the
    fundamental and its harmonics are specified.

    The generated waveform is duration seconds long and is sampled with
    samplerate Hertz.  Gaussian white noise with a standard deviation of
    noise_std is added to the generated waveform.

    Parameters
    ----------
    fish: string, dict or tuple of lists/arrays
        Specify relative amplitudes and phases of the fundamental and its harmonics.
        If string then take amplitudes and phases from the `wavefish_harmonics` dictionary.
        If dictionary then take amplitudes and phases from the 'amlitudes' and 'phases' keys.
        If tuple then the first element is the list of amplitudes and
        the second one the list of relative phases in radians.
    frequency: float or array of floats
        EOD frequency of the fish in Hertz. Either fixed number or array for
        time-dependent frequencies.
    samplerate: float
        Sampling rate in Hertz.
    duration: float
        Duration of the generated data in seconds. Only used if frequency is scalar.
    phase0: float
-        Phase offset of the EOD waveform in radians.
    noise_std: float
        Standard deviation of additive Gaussian white noise.

    Returns
    -------
    data: array of floats
        Generated data of a wave-type fish.

    Raises
    ------
    KeyError: unknown fish.
    IndexError: amplitudes and phases differ in length.
    """
    # get relative amplitude and phases:
    if isinstance(fish, (tuple, list)):
        amplitudes = fish[0]
        phases = fish[1]
    elif isinstance(fish, dict):
        amplitudes = fish['amplitudes']
        phases = fish['phases']
    else:
        if not fish in wavefish_harmonics:
            raise KeyError('unknown wavefish. Choose one of ' +
                           ', '.join(wavefish_harmonics.keys()))
        amplitudes = wavefish_harmonics[fish]['amplitudes']
        phases = wavefish_harmonics[fish]['phases']
    if len(amplitudes) != len(phases):
        raise IndexError('need exactly as many phases as amplitudes')
    # compute phase:
    if np.isscalar(frequency):
        phase = np.arange(0, duration, 1.0/samplerate)
        phase *= frequency
    else:
        phase = np.cumsum(frequency)/samplerate
    # generate EOD:
    data = np.zeros(len(phase))
    for har, (ampl, phi) in enumerate(zip(amplitudes, phases)):
        data += ampl * np.sin(2*np.pi*(har+1)*phase + phi - (har+1)*phase0)
    # add noise:
    data += noise_std * np.random.randn(len(data))
    return data


def normalize_wavefish(fish):
    """ Normalize amplitudes and phases of wave-type EOD waveform.

    The amplitudes and phases of the Fourier components are adjusted such
    that the resulting EOD waveform has a peak-to-peak amplitude of two
    and the peak of the waveform is a t time zero.

    Parameters
    ----------
    fish: string, dict or tuple of lists/arrays
        Specify relative amplitudes and phases of the fundamental and its harmonics.
        If string then take amplitudes and phases from the `wavefish_harmonics` dictionary.
        If dictionary then take amplitudes and phases from the 'amlitudes' and 'phases' keys.
        If tuple then the first element is the list of amplitudes and
        the second one the list of relative phases in radians.

    Returns
    -------
    fish: dict
        Dictionary with adjusted amplitudes and phases.
    """
    # get relative amplitude and phases:
    if isinstance(fish, (tuple, list)):
        amplitudes = fish[0]
        phases = fish[1]
    elif isinstance(fish, dict):
        amplitudes = fish['amplitudes']
        phases = fish['phases']
    else:
        if not fish in wavefish_harmonics:
            raise KeyError('unknown wavefish. Choose one of ' +
                           ', '.join(wavefish_harmonics.keys()))
        amplitudes = wavefish_harmonics[fish]['amplitudes']
        phases = wavefish_harmonics[fish]['phases']
    # generate waveform:
    eodf = 100.0
    rate = 100000.0
    data = wavefish_eods(fish, eodf, rate, 2.0/eodf, noise_std=0.0)
    # normalize amplitudes:
    ampl = 0.5*(np.max(data) - np.min(data))
    newamplitudes = np.array(amplitudes)/ampl
    # shift phases:
    deltat = np.argmax(data[:int(rate/eodf)])/rate
    deltap = 2.0*np.pi*deltat*eodf
    newphases = np.array([p+(k+1)*deltap for k, p in enumerate(phases)])
    newphases %= 2.0*np.pi
    newphases[newphases>np.pi] -= 2.0*np.pi
    # store and return:
    harmonics = dict(amplitudes=newamplitudes,
                     phases=newphases)
    return harmonics


def export_wavefish(fish, name="Unknown_harmonics", file=None):
    """ Serialize wavefish parameter to file.

    Add output to the wavefish_harmonics dictionary!

    Parameters
    ----------
    fish: string, dict or tuple of lists/arrays
        Specify relative amplitudes and phases of the fundamental and its harmonics.
        If string then take amplitudes and phases from the `wavefish_harmonics` dictionary.
        If dictionary then take amplitudes and phases from the 'amlitudes' and 'phases' keys.
        If tuple then the first element is the list of amplitudes and
        the second one the list of relative phases in radians.
    name: string
        Name of the dictionary to be written.
    file: string or file or None
        File name or open file object where to write wavefish dictionary.

    Returns
    -------
    fish: dict
        Dictionary with amplitudes and phases.
    """
    # get relative amplitude and phases:
    if isinstance(fish, (tuple, list)):
        amplitudes = fish[0]
        phases = fish[1]
    elif isinstance(fish, dict):
        amplitudes = fish['amplitudes']
        phases = fish['phases']
    else:
        if not fish in wavefish_harmonics:
            raise KeyError('unknown wavefish. Choose one of ' +
                           ', '.join(wavefish_harmonics.keys()))
        amplitudes = wavefish_harmonics[fish]['amplitudes']
        phases = wavefish_harmonics[fish]['phases']
    # write out dictionary:
    if file is None:
        file = sys.stdout
    try:
        file.write('')
        closeit = False
    except AttributeError:
        file = open(file, 'w')
        closeit = True
    ds = name + ' = dict('
    file.write(ds + 'amplitudes=(')
    file.write(', '.join(['%.5g' % a for a in amplitudes]))
    file.write('),\n')
    file.write(' ' * len(ds) + 'phases=(')
    file.write(', '.join(['%.5g' % p for p in phases]))
    file.write('))\n')
    if closeit:
        file.close()
    # return dictionary:
    harmonics = dict(amplitudes=amplitudes,
                     phases=phases)
    return harmonics


def chirps(eodf=100.0, samplerate=44100.0, duration=1.0, chirp_freq=5.0,
           chirp_size=100.0, chirp_width=0.01, chirp_kurtosis=1.0, chirp_contrast=0.05):
    """
    Simulate frequency trace with chirps.

    A chirp is modeled as a Gaussian frequency modulation.
    The first chirp is placed at 0.5/chirp_freq.

    Parameters
    ----------
    eodf: float
        EOD frequency of the fish in Hertz.
    samplerate: float
        Sampling rate in Hertz.
    duration: float
        Duration of the generated data in seconds.
    chirp_freq: float
        Frequency of occurance of chirps in Hertz.
    chirp_size: float
        Size of the chirp (maximum frequency increase above eodf) in Hertz.
    chirp_width: float
        Width of the chirp at 10% height in seconds.
    chirp_kurtosis: float:
        Shape of the chirp. =1: Gaussian, >1: more rectangular, <1: more peaked.
    chirp_contrast: float
        Maximum amplitude reduction of EOD during chirp.

    Returns
    -------
    frequency: array of floats
        Generated frequency trace that can be passed on to wavefish_eods().
    amplitude: array of floats
        Generated amplitude modulation that can be used to multiply the trace generated by
        wavefish_eods().
    """
    # baseline eod frequency and amplitude modulation:
    n = int(duration*samplerate)
    frequency = eodf * np.ones(n)
    am = np.ones(n)
    # time points for chirps:
    chirp_period = 1.0/chirp_freq
    chirp_times = np.arange(0.5*chirp_period, duration, chirp_period)
    # chirp frequency waveform:
    chirp_t = np.arange(-2.0*chirp_width, 2.0*chirp_width, 1./samplerate)
    chirp_sig = 0.5*chirp_width / (2.0*np.log(10.0))**(0.5/chirp_kurtosis)
    gauss = np.exp(-0.5*((chirp_t/chirp_sig)**2.0)**chirp_kurtosis)
    # add chirps on baseline eodf:
    for ct in chirp_times:
        index = int(ct*samplerate)
        i0 = index - len(gauss)//2
        i1 = i0 + len(gauss)
        gi0 = 0
        gi1 = len(gauss)
        if i0 < 0:
            gi0 -= i0
            i0 = 0
        if i1 >= len(frequency):
            gi1 -= i1 - len(frequency)
            i1 = len(frequency)
        frequency[i0:i1] += chirp_size * gauss[gi0:gi1]
        am[i0:i1] -= chirp_contrast * gauss[gi0:gi1]
    return frequency, am


def rises(eodf=100.0, samplerate=44100.0, duration=1.0, rise_freq=0.1,
          rise_size=10.0, rise_tau=1.0, decay_tau=10.0):
    """
    Simulate frequency trace with rises.

    A rise is modeled as a double exponential frequency modulation.

    Parameters
    ----------
    eodf: float
        EOD frequency of the fish in Hertz.
    samplerate: float
        Sampling rate in Hertz.
    duration: float
        Duration of the generated data in seconds.
    rise_freq: float
        Frequency of occurance of rises in Hertz.
    rise_size: float
        Size of the rise (frequency increase above eodf) in Hertz.
    rise_tau: float
        Time constant of the frequency increase of the rise in seconds.
    decay_tau: float
        Time constant of the frequency decay of the rise in seconds.

    Returns
    -------
    data: array of floats
        Generated frequency trace that can be passed on to wavefish_eods().
    """
    # baseline eod frequency:
    frequency = eodf * np.ones(int(duration*samplerate))
    # time points for rises:
    rise_period = 1.0/rise_freq
    rise_times = np.arange(0.5*rise_period, duration, rise_period)
    # rise frequency waveform:
    rise_t = np.arange(0.0, 5.0*decay_tau, 1./samplerate)
    rise = rise_size * (1.0-np.exp(-rise_t/rise_tau)) * np.exp(-rise_t/decay_tau)
    # add rises on baseline eodf:
    for r in rise_times:
        index = int(r*samplerate)
        if index+len(rise) > len(frequency):
            rise_index = len(frequency)-index
            frequency[index:index+rise_index] += rise[:rise_index]
            break
        else:
            frequency[index:index+len(rise)] += rise
    return frequency


""" Positions, amplitudes and standard deviations of monophasic EOD waveforms. """
monophasic_peaks = dict(times=(0.0,), amplitudes=(1.0,), stdevs=(0.0003,))

""" Positions, amplitudes and standard deviations of binophasic EOD waveforms. """
biphasic_peaks = dict(times=(1e-05, 0.00031),
                      amplitudes=(1.1053, -0.33158),
                      stdevs=(0.0001, 0.0002))

""" Positions, amplitudes and standard deviations of trinophasic EOD waveforms. """
triphasic_peaks = dict(times=(3e-05, 0.00018, 0.00043),
                       amplitudes=(1.2382, -0.9906, 0.12382),
                       stdevs=(0.0001, 0.0001, 0.0002))

""" Standard deviations, amplitudes and positions of Gaussians that make up
    EOD waveforms of pulse-type electric fish. """
pulsefish_peaks = dict(monophasic=monophasic_peaks,
                       biphasic=biphasic_peaks,
                       triphasic=triphasic_peaks)
                              

def pulsefish_eods(fish='biphasic', frequency=100.0, samplerate=44100.0, duration=1.0,
                   noise_std=0.01, jitter_cv=0.1, first_pulse=None):
    """
    Simulate EOD waveform of a pulse-type fish.

    Pulses are spaced by 1/frequency, jittered as determined by jitter_cv. Each pulse is
    a combination of Gaussian peaks, whose positions, amplitudes and widths are
    given by 'fish'.

    The generated waveform is duration seconds long and is sampled with samplerate Hertz.
    Gaussian white noise with a standard deviation of noise_std is added to the generated
    pulse train.

    Parameters
    ----------
    fish: string, dict or tuple of floats/lists/arrays
        Specify positions, amplitudes and standard deviations Gaussians peaks that are
        superimposed to simulate EOD waveforms of pulse-type electric fishes. 
        If string then take positions, amplitudes and standard deviations 
        from the `pulsefish_peaks` dictionary.
        If dictionary then take pulse properties from the 'times', 'amlitudes'
        and 'stdevs' keys.
        If tuple then the first element is the list of peak positions,
        the second is the list of corresponding amplitudes, and
        the third one the list of corresponding standard deviations.
    frequency: float
        EOD frequency of the fish in Hz.
    samplerate: float
        Sampling Rate in Hz.
    duration: float
        Duration of the generated data in seconds.
    noise_std: float
        Standard deviation of additive Gaussian white noise.
    jitter_cv: float
        Gaussian distributed jitter of pulse times as coefficient of variation
        of inter-pulse intervals.
    first_pulse: float or None
        The position of the first pulse. If None it is choosen automatically
        depending on pulse width, jitter, and frequency.

    Returns
    -------
    data: array of floats
        Generated data of a pulse-type fish.

    Raises
    ------
    KeyError: unknown fish.
    IndexError: peak positions, amplitudes, or standard deviations differ in length.
    """
    # get peak properties:
    if isinstance(fish, (tuple, list)):
        peak_times = fish[0]
        peak_amplitudes = fish[1]
        peak_stds = fish[2]
    elif isinstance(fish, dict):
        peak_times = fish['times']
        peak_amplitudes = fish['amplitudes']
        peak_stdevs = fish['stdevs']
    else:
        if not fish in pulsefish_peaks:
            raise KeyError('unknown pulse-type fish. Choose one of ' +
                           ', '.join(pulsefish_peaks.keys()))
        peak_times = pulsefish_peaks[fish]['times']
        peak_amplitudes = pulsefish_peaks[fish]['amplitudes']
        peak_stdevs = pulsefish_peaks[fish]['stdevs']
    if len(peak_stdevs) != len(peak_amplitudes) or len(peak_stdevs) != len(peak_times):
        raise IndexError('need exactly as many standard deviations as amplitudes and times')

    # time axis for single pulse:
    min_time_inx = np.argmin(peak_times)
    max_time_inx = np.argmax(peak_times)
    tmax = max(np.abs(peak_times[min_time_inx]-4.0*peak_stdevs[min_time_inx]),
               np.abs(peak_times[max_time_inx]+4.0*peak_stdevs[max_time_inx]))
    x = np.arange(-tmax, tmax, 1.0/samplerate)
    pulse_duration = x[-1] - x[0]
    
    # generate a single pulse:
    pulse = np.zeros(len(x))
    for time, ampl, std in zip(peak_times, peak_amplitudes, peak_stdevs):
        pulse += ampl * np.exp(-0.5*((x-time)/std)**2)
    poffs = len(pulse)//2

    # paste the pulse into the noise floor:
    time = np.arange(0, duration, 1.0/samplerate)
    data = np.random.randn(len(time)) * noise_std
    period = 1.0/frequency
    jitter_std = period * jitter_cv
    if first_pulse is None:
        first_pulse = np.max([pulse_duration, 3.0*jitter_std])
    pulse_times = np.arange(first_pulse, duration, period )
    pulse_times += jitter_std*np.random.randn(len(pulse_times))
    pulse_indices = np.round(pulse_times * samplerate).astype(np.int)
    for inx in pulse_indices[(pulse_indices>=poffs)&(pulse_indices-poffs+len(pulse)<len(data))]:
        data[inx-poffs:inx-poffs+len(pulse)] += pulse
    return data


def normalize_pulsefish(fish):
    """ Normalize times and stdevs of pulse-type EOD waveform.

    The positions and amplitudes of Gaussian peaks are adjusted such
    that the resulting EOD waveform has a maximum peak amplitude of one
    and has the largest peak at time zero.

    Parameters
    ----------
    fish: string, dict or tuple of floats/lists/arrays
        Specify positions, amplitudes and standard deviations Gaussians peaks that are
        superimposed to simulate EOD waveforms of pulse-type electric fishes. 
        If string then take positions, amplitudes and standard deviations 
        from the `pulsefish_peaks` dictionary.
        If dictionary then take pulse properties from the 'times', 'amlitudes'
        and 'stdevs' keys.
        If tuple then the first element is the list of peak positions,
        the second is the list of corresponding amplitudes, and
        the third one the list of corresponding standard deviations.

    Returns
    -------
    fish: dict
        Dictionary with adjusted times and standard deviations.
    """
    # get peak properties:
    if isinstance(fish, (tuple, list)):
        peak_times = fish[0]
        peak_amplitudes = fish[1]
        peak_stds = fish[2]
    elif isinstance(fish, dict):
        peak_times = fish['times']
        peak_amplitudes = fish['amplitudes']
        peak_stdevs = fish['stdevs']
    else:
        if not fish in pulsefish_peaks:
            raise KeyError('unknown pulse-type fish. Choose one of ' +
                           ', '.join(pulsefish_peaks.keys()))
        peak_times = pulsefish_peaks[fish]['times']
        peak_amplitudes = pulsefish_peaks[fish]['amplitudes']
        peak_stdevs = pulsefish_peaks[fish]['stdevs']
    # generate waveform:
    eodf = 10.0
    rate = 100000.0
    first_pulse = 0.5/eodf
    data = pulsefish_eods(fish, eodf, rate, 1.0/eodf, noise_std=0.0,
                          jitter_cv=0.0, first_pulse=first_pulse)
    # maximum peak:
    idx = np.argmax(np.abs(data))
    # normalize amplitudes:
    ampl = data[idx]
    newamplitudes = np.array(peak_amplitudes)/ampl
    # shift times:
    deltat = idx/rate - first_pulse
    newtimes = np.array(peak_times) - deltat
    # store and return:
    peaks = dict(times=newtimes,
                 amplitudes=newamplitudes,
                 stdevs=peak_stdevs)
    return peaks


def export_pulsefish(fish, name="Unknown_peaks", file=None):
    """ Serialize pulsefish parameter to file.

    Add output to the wavefish_harmonics dictionary!

    Parameters
    ----------
    fish: string, dict or tuple of floats/lists/arrays
        Specify positions, amplitudes and standard deviations Gaussians peaks that are
        superimposed to simulate EOD waveforms of pulse-type electric fishes. 
        If string then take positions, amplitudes and standard deviations 
        from the `pulsefish_peaks` dictionary.
        If dictionary then take pulse properties from the 'times', 'amlitudes'
        and 'stdevs' keys.
        If tuple then the first element is the list of peak positions,
        the second is the list of corresponding amplitudes, and
        the third one the list of corresponding standard deviations.
    name: string
        Name of the dictionary to be written.
    file: string or file or None
        File name or open file object where to write pulsefish dictionary.

    Returns
    -------
    fish: dict
        Dictionary with peak times, amplitudes and standard deviations.
    """
    # get peak properties:
    if isinstance(fish, (tuple, list)):
        peak_times = fish[0]
        peak_amplitudes = fish[1]
        peak_stds = fish[2]
    elif isinstance(fish, dict):
        peak_times = fish['times']
        peak_amplitudes = fish['amplitudes']
        peak_stdevs = fish['stdevs']
    else:
        if not fish in pulsefish_peaks:
            raise KeyError('unknown pulse-type fish. Choose one of ' +
                           ', '.join(pulsefish_peaks.keys()))
        peak_times = pulsefish_peaks[fish]['times']
        peak_amplitudes = pulsefish_peaks[fish]['amplitudes']
        peak_stdevs = pulsefish_peaks[fish]['stdevs']
    # write out dictionary:
    if file is None:
        file = sys.stdout
    try:
        file.write('')
        closeit = False
    except AttributeError:
        file = open(file, 'w')
        closeit = True
    ds = name + ' = dict('
    file.write(ds + 'times=(')
    file.write(', '.join(['%.5g' % t for t in peak_times]))
    file.write('),\n')
    file.write(' ' * len(ds) + 'amplitudes=(')
    file.write(', '.join(['%.5g' % a for a in peak_amplitudes]))
    file.write('),\n')
    file.write(' ' * len(ds) + 'stdevs=(')
    file.write(', '.join(['%.5g' % a for a in peak_stdevs]))
    file.write('))\n')
    if closeit:
        file.close()
    # return dictionary:
    peaks = dict(times=peak_times,
                 amplitudes=peak_amplitudes,
                 stdevs=peak_stdevs)
    return peaks


def generate_waveform(filename):
    """ Interactively generate audio file with simulated EOD waveforms.

    Parameters needed to generate EOD waveforms are take from console input.

    Parameters
    ----------
    filename: string
        Name of file inclusively extension (e.g. '.wav')
        used to store the simulated EOD waveforms.
    """
    import os
    from audioio import write_audio
    from .consoleinput import read, select, save_inputs
    # generate file:
    samplerate = read('Sampling rate in Hz', '44100', float, 1.0)
    duration = read('Duration in seconds', '10', float, 0.001)
    nfish = read('Number of fish', '1', int, 1)
    ndata = read('Number of electrodes', '1', int, 1)
    fish_spread = 1
    if ndata > 1:
        fish_spread = read('Number of electrodes fish are spread over', '2', int, 1)
    data = np.random.randn(int(duration*samplerate), ndata)*0.01
    fish_indices = np.random.randint(ndata, size=nfish)
    eodt = 'a'
    eodf = 800.0
    eoda = 1.0
    eodsig = 'n'
    pulse_jitter = 0.1
    chirp_freq = 5.0
    chirp_size = 100.0
    chirp_width = 0.01
    chirp_kurtosis = 1.0            
    rise_freq = 0.1
    rise_size = 10.0
    rise_tau = 1.0
    rise_decay_tau = 10.0
    for k in range(nfish):
        print('')
        fish = 'Fish %d: ' % (k+1)
        eodt = select(fish + 'EOD type', eodt, ['a', 'e', '1', '2', '3'],
                      ['Apteronotus', 'Eigenmannia',
                       'monophasic pulse', 'biphasic pulse', 'triphasic pulse'])
        eodf = read(fish + 'EOD frequency in Hz', '%g'%eodf, float, 1.0, 3000.0)
        eoda = read(fish + 'EOD amplitude', '%g'%eoda, float, 0.0, 10.0)
        if eodt in 'ae':
            eodsig = select(fish + 'Add communication signals', eodsig, ['n', 'c', 'r'],
                      ['fixed EOD', 'chirps', 'rises'])
            eodfreq = eodf
            if eodsig == 'c':
                chirp_freq = read('Number of chirps per second', '%g'%chirp_freq, float, 0.001)
                chirp_size = read('Size of chirp in Hz', '%g'%chirp_size, float, 1.0)
                chirp_width = 0.001*read('Width of chirp in ms', '%g'%(1000.0*chirp_width), float, 1.0)
                eodfreq, _ = chirps(eodf, samplerate, duration,
                                    chirp_freq, chirp_size, chirp_width, chirp_kurtosis)
            elif eodsig == 'r':
                rise_freq = read('Number of rises per second', '%g'%rise_freq, float, 0.00001)
                rise_size = read('Size of rise in Hz', '%g'%rise_size, float, 0.01)
                rise_tau = read('Time-constant of rise onset in seconds', '%g'%rise_tau, float, 0.01)
                rise_decay_tau = read('Time-constant of rise decay in seconds', '%g'%rise_decay_tau, float, 0.01)
                eodfreq = rises_frequency(eodf, samplerate, duration,
                                          rise_freq, rise_size, rise_tau, rise_decay_tau)
            if eodt == 'a':
                fishdata = eoda*wavefish_eods('Alepto', eodfreq, samplerate, duration,
                                              phase0=0.0, noise_std=0.0)
            elif eodt == 'e':
                fishdata = eoda*wavefish_eods('Eigenmannia', eodfreq, samplerate,
                                              duration, phase0=0.0, noise_std=0.0)
        else:
            pulse_jitter = read(fish + 'CV of pulse jitter', '%g'%pulse_jitter, float, 0.0, 2.0)
            if eodt == '1':
                fishdata = eoda*pulsefish_eods('monophasic', eodf, samplerate, duration,
                                               jitter_cv=pulse_jitter, noise_std=0.0)
            elif eodt == '2':
                fishdata = eoda*pulsefish_eods('biphasic', eodf, samplerate, duration,
                                               jitter_cv=pulse_jitter, noise_std=0.0)
            elif eodt == '3':
                fishdata = eoda*pulsefish_eods('triphasic', eodf, samplerate, duration,
                                               jitter_cv=pulse_jitter, noise_std=0.0)
        i = fish_indices[k]
        for j in range(fish_spread):
            data[:, (i+j)%ndata] += fishdata*(0.2**j)

    maxdata = np.max(np.abs(data))
    write_audio(filename, 0.9*data/maxdata, samplerate)
    input_file = os.path.splitext(filename)[0] + '.inp' 
    save_inputs(input_file)
    print('\nWrote fakefish data to file "%s".' % filename)
            

def demo():
    import matplotlib.pyplot as plt
    samplerate = 40000.0 # in Hz
    duration = 10.0      # in sec

    inset_len = 0.01    # in sec
    inset_indices = int(inset_len*samplerate)
    ws_fac = 0.1         # whitespace factor or ylim (between 0. and 1.)

    # generate data:
    eodf = 400.0
    wavefish = wavefish_eods('Alepto', eodf, samplerate, duration, noise_std=0.02)
    eodf = 650.0
    wavefish += 0.5*wavefish_eods('Eigenmannia', eodf, samplerate, duration)

    pulsefish = pulsefish_eods('biphasic', 80.0, samplerate, duration,
                               noise_std=0.02, jitter_cv=0.1, first_pulse=inset_len/2)
    time = np.arange(len(wavefish))/samplerate

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
    ax[1][1].plot(time[:inset_indices], pulsefish[:inset_indices], '-o')

    for row in ax:
        for c_ax in row:
            c_ax.set_xlabel('Time [sec]')
            c_ax.set_ylabel('Amplitude')

    plt.tight_layout()

    # chirps:
    chirps_freq = chirps(600.0, samplerate, duration)
    chirps_data = wavefish_eods('Alepto', chirps_freq, samplerate)

    # rises:
    rises_freq = rises(600.0, samplerate, duration, rise_size=20.0)
    rises_data = wavefish_eods('Alepto', rises_freq, samplerate)

    nfft = 256
    fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(19, 10))
    ax[0].set_title('Chirps')
    ax[0].specgram(chirps_data, Fs=samplerate, NFFT=nfft, noverlap=nfft//16)
    time = np.arange(len(chirps_freq))/samplerate
    ax[0].plot(time[:-nfft//2], chirps_freq[nfft//2:], '-k', lw=2)
    ax[0].set_ylim(0.0, 3000.0)
    ax[0].set_ylabel('Frequency [Hz]')

    nfft = 4096
    ax[1].set_title('Rises')
    ax[1].specgram(rises_data, Fs=samplerate, NFFT=nfft, noverlap=nfft//2)
    time = np.arange(len(rises_freq))/samplerate
    ax[1].plot(time[:-nfft/4], rises_freq[nfft/4:], '-k', lw=2)
    ax[1].set_ylim(500.0, 700.0)
    ax[1].set_ylabel('Frequency [Hz]')
    ax[1].set_xlabel('Time [s]')
    plt.tight_layout()

    plt.show()


def main():
    import sys
    
    if len(sys.argv) > 1:
        if len(sys.argv) == 2 or sys.argv[1] != '-s':
            print('usage: fakefish [-h|--help] [-s audiofile]')
            print('')
            print('Without arguments, run a demo for illustrating fakefish functionality.')
            print('')
            print('-s audiofile: writes audiofile with user defined simulated electric fishes.')
            print('')
            print('by bendalab (2020)')
        else:
            generate_waveform(sys.argv[2])
    else:
        demo()

            
if __name__ == '__main__':
    main()
