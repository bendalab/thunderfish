import sys
import numpy as np
import peakdetection as pkd


def eod_snippets(bwin_data, samplerate, win_shift=0.5):
    """
    Detects peaks and collects data arround these detected peaks for later comparison.

    This function detects EODs and creates a list of lists each list containing the data around a detected eod.
    For the eod detection a threshold is defined. When the data rises above this threshold an EOD is detected.
    The threshold is defined as the difference between the mean and the maximum of the data added to the mean of the
    data.

    :param bwin_data:           (1-D array) Data array that shall be analyzed.
    :param samplerate:          (float) samleate of the data that shall be analyzed.
    :param win_shift:           (float) Time shift in seconds between windows.
    :param pulse_window:        (float) data window around the detected EODs that shall be extracted in sec.
    :return eod_data:           (2-D array) An array of lists. Each list contains the data around a EOD.
    """
    # th = np.mean(bwin_data) + ( (np.max(bwin_data)-np.mean(bwin_data)) / 2.0 )
    threshold = np.zeros(len(bwin_data))
    win_shift_indices = int(win_shift * samplerate)

    for inx0 in range(0, len(bwin_data) - win_shift_indices, win_shift_indices):
        inx1 = inx0 + win_shift_indices
        threshold[inx0:inx1] = np.percentile(bwin_data[inx0:inx1], 99)

    eod_data = []

    eod_idx = pkd.detect_peaks(bwin_data, threshold)[0]

    eod_idx_diff = [(eod_idx[i + 1] - eod_idx[i]) for i in range(len(eod_idx) - 1)]
    mean_th_idx_diff = np.mean(eod_idx_diff)

    for idx in eod_idx:
        if int(idx - (mean_th_idx_diff / (3. / 2))) >= 0 and int(idx + (mean_th_idx_diff / (3. / 2))) <= len(bwin_data):
            eod_data.append(
                bwin_data[int(idx - (mean_th_idx_diff / (3. / 2))): int(idx + (mean_th_idx_diff / (3. / 2)))])

    return np.asarray(eod_data), eod_idx_diff


def eod_analysis_plot(time, mean_eod, std_eod, ax):
    """
    Creates an axis for plotting  to visualize a mean eod and its standard deviation.

    :param time:                (1-D array) containing the time of the mean eod.
    :param mean_eod:            (1-D array) containing the data of the mean eod.
    :param std_eod:             (1-D array) containing the data of the std eod.
    :param ax:                  (axis for plot) empty axis that is filled with content in the function.
    """
    ax.plot(time, mean_eod, lw=2, color='firebrick', alpha=0.7, label='mean EOD')
    ax.fill_between(time, mean_eod + std_eod, mean_eod - std_eod, color='grey', alpha=0.3)
    ax.set_xlabel('time [msec]')
    ax.set_ylabel('amplitude (mV)')
    ax.set_xlim([min(time), max(time)])


def eod_analysis(bwin_data, samplerate, verbose=0, plot_data_func=None, **kwargs):
    """
    Detects EODs in the given data and tries to build a mean eod.

    This function takes the data array of a soundfile, the samplerate and several other arguments to detect EODs and
    build a mean eod with the help of several other functions.

    :param bwin_data:           (1-D array) containing the data that shall be analysed.
    :param samplerate:          (float) samplerate of the data that shall be analysed.
    :param verbose:             (int) when the value is 1 you get additional shell output.
    :param plot_data_func:      (function) function (eod_analysis_plot()) that is used to create a axis for later plotting containing a figure to
                                visualice what the modul did.
    :param kwargs:              additional arguments that are passed to the plot_data_func().
    :return eod_idx_diff        (1-D array) Containing the index differences between the detected eods.
    """
    if verbose >= 1:
        print('analyse EOD waveform ...')
        
    eod_data, eod_idx_diff = eod_snippets(bwin_data, samplerate)
    mean_eod = np.mean(eod_data, axis=0)
    std_eod = np.std(eod_data, axis=0, ddof=1)

    time = 1000.0 * (np.arange(len(mean_eod)) - len(mean_eod)/2) / samplerate

    if plot_data_func:
        plot_data_func(time, mean_eod, std_eod, **kwargs)

    return eod_idx_diff


if __name__ == '__main__':
    try:
        import dataloader as dl
        import bestwindow as bw
        import psdtype as pt
        import powerspectrum as ps
    except ImportError:
        'Import error !!!'
        quit()


    def load_example_data(audio_file):
        """
        This function shows in part the same components of the thunderfish.py project. Here several modules are used to load
        some data to display the functionality of the eodanalysis.py module.

        :param audio_file:      (string) filepath of a audiofile that shall be used for the analysis.
        :return power:          (1-D array) power array of a psd.
        :return freqs:          (1-D array) frequency array of a psd.
        """

        # load data using dataloader module
        print('loading example data ...\n')
        data, samplerate, unit = dl.load_data(audio_file)

        # calculate best_window
        print('calculating best window ...\n')
        bwin_data, clip = bw.best_window(data, samplerate)

        print('calculating powerspectrum ...\n')
        power, freqs = ps.multi_resolution_psd(data, samplerate)

        pulse_psd, proportion = pt.psd_assignment(power, freqs)

        return bwin_data, samplerate, pulse_psd


    print(
    'This modul detects single eods and tries to create a mean eod. Here this is shown as example for a defined file.')
    print('')
    print('Usage:')
    print('  python eodanalysis.py <audiofile>')
    print('')

    if len(sys.argv) <= 1:
        quit()

    bwin_data, samplerate, pulse_psd = load_example_data(sys.argv[1])

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()

    eod_idx_diff = eod_analysis(bwin_data, samplerate, plot_data_func=eod_analysis_plot, ax=ax)
    plt.legend()
    plt.show()
