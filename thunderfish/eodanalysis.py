import numpy as np

def eod_extracting(bwin_data, samplerate, window):
    th = np.mean(bwin_data) + ( (np.max(bwin_data)-np.mean(bwin_data)) / 2.0 )
    th_idx = []
    eod_data = []

    ### could be replaced be the peakdetection modul when working properly ###
    for idx in np.arange(len(bwin_data)-1)+1:
        if bwin_data[idx] > th and bwin_data[idx-1] <= th:
            th_idx.append(idx)

    for idx in th_idx:
        if int(idx - samplerate * window / 2) >= 0 and idx + samplerate * window / 2 <= len(bwin_data):
            eod_data.append(bwin_data[int(idx - samplerate * window / 2) : idx + samplerate * window / 2])
    return eod_data

def eod_mean(eod_data):
    mean_eod = np.zeros(len(eod_data[0]))
    std_eod = np.zeros(len(eod_data[0]))

    for i in np.arange(len(eod_data[0])):
        mean_eod[i] = np.mean([eod_data[eod][i] for eod in np.arange(len(eod_data[0]))])
        std_eod[i] = np.std([eod_data[eod][i] for eod in np.arange(len(eod_data[0]))], ddof=1)

    return mean_eod, std_eod


def eod_analysis_main(bwin_data, samplerate, fish_type, psd_type):
    pulse_data = eod_extracting(bwin_data, samplerate, window= 0.006)
    mean_eod, std_eod = eod_mean(pulse_data)


    time = np.arange(len(mean_eod)) * 1.0 / samplrate

    return mean_eod, std_eod, time

def load_example_data(audio_file= '../../../raab_data/colombia_2013/data/recordings_cano_rubiano_RAW/31129L11.WAV',
                      channel=0, verbose=None):
    """
    This function shows in part the same components of the thunderfish.py poject. Here several moduls are used to load
    some data to dispay the functionality of the psdtype.py modul.

    :param audio_file:      (string) filepath of a audiofile that shall be used for the analysis.
    :return power:          (1-D array) power array of a psd.
    :return freqs:          (1-D array) frequency array of a psd.
    """
    cfg = ct.get_config_dict()

    if verbose is not None:  # ToDo: Need to document the whole cfg-dict thing.
        cfg['verboseLevel'][0] = verbose
    channel = channel

    # load data using dataloader module
    print('loading example data ...\n')
    data, samplrate, unit = dl.load_data(audio_file)

    # calculate best_window
    print('calculating best window ...\n')
    bwin_data, clip = bw.best_window(data, samplrate)

    return bwin_data, samplrate

if __name__ == '__main__':
    try:
        import config_tools as ct
        import dataloader as dl
        import bestwindow as bw
    except ImportError:
        'Import error !!!'
        quit()

    bwin_data, samplrate = load_example_data()

    mean_eod, std_eod, time = eod_analysis_main(bwin_data, samplrate, fish_type='pulse', psd_type='pulse')

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    ax.plot(time, mean_eod, '-')
    plt.show()