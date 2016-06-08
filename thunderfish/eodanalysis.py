import numpy as np
import sys

def eod_extracting(bwin_data, samplerate, fish_type, psd_type):
    """
    Detects peaks and collects data arround these detected peaks for later comparison.

    This function detects EODs and creates a list of lists each list containing the data around a detected eod.
    For the eod detection a threshold is defined. When the data rises above this threshold an EOD is detected.
    The threshold is defined as the difference between the mean and the maximum of the data added to the mean of the
    data.

    :param bwin_data:           (1-D array) Data array that shall be analyzed.
    :param samplerate:          (float) samleate of the data that shall be analyzed.
    :param window:              (float) time around the threshold that shall be saved for further EOD comparison. [in s]
    :return eod_data:           (2-D array) An array of lists. Each list contains the data around a EOD.
    """
    # ToDo: replace this with the peakdetection modul as soon as it is trustworthy.
    th = np.mean(bwin_data) + ( (np.max(bwin_data)-np.mean(bwin_data)) / 2.0 )
    th_idx = []
    eod_data = []

    ### could be replaced be the peakdetection modul when working properly ###
    for idx in np.arange(len(bwin_data)-1)+1:
        if bwin_data[idx] > th and bwin_data[idx-1] <= th:
            th_idx.append(idx)

    th_idx_diff = [(th_idx[i+1] - th_idx[i]) for i in np.arange(len(th_idx)-1)]
    mean_th_idx_diff = np.mean(th_idx_diff)

    if fish_type is 'pulse' and psd_type is 'pulse':
        window = 0.006
        for idx in th_idx:
            if int(idx - samplerate * window / 2) >= 0 and int(idx + samplerate * window / 2) <= len(bwin_data):
                eod_data.append(bwin_data[int(idx - samplerate * window / 2): int(idx + samplerate * window / 2)])
    else:
        for idx in th_idx:
            if int(idx - (mean_th_idx_diff / (3./2) )) >= 0 and int(idx + (mean_th_idx_diff / (3./2))) <= len(bwin_data):
                eod_data.append(bwin_data[int(idx - (mean_th_idx_diff / (3./2)) ): int(idx + (mean_th_idx_diff / (3./2)))])

    return eod_data

def eod_mean(eod_data):
    """
    Creates an array containing the mean eod data.

    This function gets a 2-D array as input. This input cantains of lists of data with the same length. Each list
    represents the data around a EOD. To analyse the structure in this function the mean EOD and its standard deviation
    is calculated.

    :param eod_data:            (2-D array) An array of lists. Each list contains the data around a EOD.
    :return mean_eod:           (1-D array) Array of the mean data calculated from the 2-D eod_data array.
    :return std_eod:            (1-D array) Array of the std data calculated from the 2-D eod_data array.S
    """
    mean_eod = np.zeros(len(eod_data[0]))
    std_eod = np.zeros(len(eod_data[0]))

    for i in np.arange(len(eod_data[0])):
        mean_eod[i] = np.mean([eod_data[eod][i] for eod in np.arange(len(eod_data))])
        std_eod[i] = np.std([eod_data[eod][i] for eod in np.arange(len(eod_data))], ddof=1)

    return mean_eod, std_eod

def eodanalysisplot(time, mean_eod, std_eod, ax):
    """
    Creates a axis for plotting  to visualize a mean eod and its standard deviation.

    :param time:                (1-D array) containing the time of the mean eod.
    :param mean_eod:            (1-D array) containing the data of the mean eod.
    :param std_eod:             (1-D array) containing the data of the std eod.
    :param ax:                  (axis for plot) empty axis that is filled with content in the function.
    :return:                    (axis for plot) axis that is ready for plotting explaining what the modul did.
    """
    u_std = [mean_eod[i] + std_eod[i] for i in np.arange(len(mean_eod))]
    l_std = [mean_eod[i] - std_eod[i] for i in np.arange(len(mean_eod))]
    ax.plot(time, mean_eod, lw=2, color='firebrick', alpha=0.7, label='mean EOD')
    ax.fill_between(time, u_std, l_std, color='grey', alpha=0.3)
    ax.set_xlabel('time [sec]')
    ax.set_ylabel('Amplitude (mV)')
    ax.set_xlim([min(time), max(time)])
    return ax

def load_example_data(audio_file, channel=0, verbose=None):
    """
    This function shows in part the same components of the thunderfish.py poject. Here several moduls are used to load
    some data to dispay the functionality of the eodanalysis.py modul.

    :param audio_file:      (string) filepath of a audiofile that shall be used for the analysis.
    :return power:          (1-D array) power array of a psd.
    :return freqs:          (1-D array) frequency array of a psd.
    """
    cfg = ct.get_config_dict()

    if verbose is not None:
        cfg['verboseLevel'][0] = verbose
    channel = channel

    # load data using dataloader module
    print('loading example data ...\n')
    data, samplrate, unit = dl.load_data(audio_file)

    # calculate best_window
    print('calculating best window ...\n')
    bwin_data, clip = bw.best_window(data, samplrate)

    print('calculating powerspectrum ...\n')
    power, freqs = ps.powerspectrum(data, samplrate)


    psd_type, proportion = pt.psd_assignment(power, freqs)

    return bwin_data, samplrate, psd_type

def eod_analysis(bwin_data, samplerate, fish_type, psd_type, plot_data_func=None, **kwargs):
    """
    Detects EODs in the given data and tries to build a mean eod.

    This function takes the data array of a soundfile, the samplerate and several other arguments to detect EODs and
    build a mean eod with the help of several other functions.

    :param bwin_data:           (1-D array) containing the data that shall be analysed.
    :param samplerate:          (float) samplerate of the data that shall be analysed.
    :param fish_type:           (string) result of the "sortfishtype.py" modul.
    :param psd_type:            (string) result of the "psdtype.py" mudule.
    :param plot_data_func:      (function) function (eodanalysisplot()) that is used to create a axis for later plotting containing a figure to
                                visualice what the modul did.
    :param kwargs:              additional arguments that are passed to the plot_data_func().
    :return mean_eod:           (1-D array) containing the data of the mean eod.
    :return std_eod:            (1-D array) containing the data of the std eod.
    :return time:               (1-D array) containing the time of the mean eod. (same length as mean_eod or std_eod)
    :return ax:                 (axis for plot) axis that is ready for plotting explaining what the modul does.
    """

    print('\nAnalysing EOD structures ...')
    pulse_data = eod_extracting(bwin_data, samplerate, fish_type, psd_type)
    mean_eod, std_eod = eod_mean(pulse_data)

    time = ((np.arange(len(mean_eod)) * 1.0 / samplerate) - 0.5 * len(mean_eod) / samplerate) * 1000.0

    if plot_data_func:
        ax = plot_data_func(time, mean_eod, std_eod, **kwargs)
        return mean_eod, std_eod, time, ax
    else:
        return mean_eod, std_eod, time

if __name__ == '__main__':
    try:
        import config_tools as ct
        import dataloader as dl
        import bestwindow as bw
        import psdtype as pt
        import powerspectrum as ps
    except ImportError:
        'Import error !!!'
        quit()

    print('This modul detects single eods and tries to create a mean eod. Here this is shown as example for a defined file.')
    print('')
    print('Usage:')
    print('  python eodanalysis.py <audiofile>')
    print('')

    if len(sys.argv) <= 1:
        quit()

    bwin_data, samplrate, psd_type = load_example_data(sys.argv[1])

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()

    mean_eod, std_eod, time, ax = eod_analysis(bwin_data, samplrate, fish_type='wave', psd_type=psd_type,
                                               plot_data_func=eodanalysisplot, ax=ax)
    plt.legend()
    plt.show()