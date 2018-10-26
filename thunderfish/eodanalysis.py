"""
Detects EODs in a given dataset and computes their mean waveform.

eod_waveform(): calculates a mean EOD of a given dataset.
"""

import numpy as np
from .peakdetection import percentile_threshold, detect_peaks, snippets


def eod_waveform(data, samplerate, th_factor=0.6, percentile=0.1, start=None, stop=None):
    """Detects EODs in the given data, extracts data snippets around each EOD and computes a mean waveform with standard
    deviation.

    :param data: (1-D array) the data to be analysed.
    :param samplerate: (float) samplerate of the data in Hertz.
    :param percentile: (int). percentile parameter for the peakdetection.percentile_threshold() function used to
    estimate thresholds for detecting EOD peaks in the data.
    :param th_factor: (float). th_factor parameter for the peakdetection.percentile_threshold() function used to
    estimate thresholds for detecting EOD peaks in the data.
    :param start: (float or None) start time of EOD snippets relative to peak.
    :param stop: (float or None) stop time of EOD snippets relative to peak.
    :return mean_eod (1-D array) Average of the EOD snippets.
    :return std_eod (1-D array) Standard deviation of the averaged snippets.
    :return time (1-D array) Time axis for mean_eod and std_eod.
    :return eod_times (1-D array) Times of EOD peaks in seconds.
    """
    # threshold for peak detection:
    threshold = percentile_threshold(data, th_factor=th_factor, percentile=percentile)

    # detect peaks:
    eod_idx, _ = detect_peaks(data, threshold)

    # eod times:
    eod_times = eod_idx / samplerate

    # start and stop times:
    if start is None or stop is None:
        period = np.mean(np.diff(eod_times))
        if start is None:
            start = -period
        if stop is None:
            stop = period
    # start and stop indices:
    start_inx = int(start * samplerate)
    stop_inx = int(stop * samplerate)

    # extract snippets:
    eod_snippets = snippets(data, eod_idx, start_inx, stop_inx)

    # mean and std of snippets:    
    mean_eod = np.mean(eod_snippets, axis=0)
    std_eod = np.std(eod_snippets, axis=0, ddof=1)

    # time axis:
    time = (np.arange(len(mean_eod)) + start_inx) / samplerate

    return mean_eod, std_eod, time, eod_times


def eod_waveform_plot(time, mean_eod, std_eod, ax, unit='a.u.'):
    """Plot mean eod and its standard deviation.

    :param time: (1-D array) Time of the mean EOD.
    :param mean_eod: (1-D array) Mean EOD waveform.
    :param std_eod: (1-D array) Sandard deviation of EOD waveform.
    :param ax: (axis for plot).
    :param unit: (string) Unit of the data.
    """
    ax.plot(1000.0*time, mean_eod, lw=2, color='r')
    ax.fill_between(1000.0*time, mean_eod + std_eod, mean_eod - std_eod,
                    color='grey', alpha=0.3)
    ax.set_xlabel('Time [msec]')
    ax.set_ylabel('Amplitude [%s]' % unit)


if __name__ == '__main__':
    import sys
    import matplotlib.pyplot as plt
    from .fakefish import generate_biphasic_pulses
    from .dataloader import load_data
    from .bestwindow import best_window

    print('Analysis of EOD waveforms.')
    print('')
    print('Usage:')
    print('  python eodanalysis.py [<audiofile>]')
    print('')

    # data:
    if len(sys.argv) <= 1:
        samplerate = 44100.0
        data = generate_biphasic_pulses(80.0, samplerate, 4.0, noise_std=0.05)
        unit = 'mV'
    else:
        rawdata, samplerate, unit = load_data(sys.argv[1], 0)
        data, _ = best_window(rawdata, samplerate)

    # analyse EOD:
    mean_eod, std_eod, time, eod_times = eod_waveform(data, samplerate,
                                                      start=-0.002, stop=0.002)

    # plot:
    fig, ax = plt.subplots()
    eod_waveform_plot(time, mean_eod, std_eod, ax, unit=unit)
    plt.show()
