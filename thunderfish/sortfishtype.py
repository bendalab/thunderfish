import numpy as np
from IPython import embed

import peakdetection as pkd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


def type_detector(data, samplerate, win_shift=0.5, pulse_thres=0.1, plot_data_func=None):
    # ToDo: The peak-ratio docu is mistaken! The float should be between 0. and 1. but it doesn't and Idk why!!

    """ Detects if fish is pulse or wave by calculating the proportion of the time distance between a peak and its
     following trough, relative to the time between consecutive peaks.

    :param data: (1-D array) The data to be analyzed (Usually the best window already)
    :param samplerate: (float). Sampling rate of the data in Hz
    :param win_shift: (float). Time shift in seconds between windows.
    :param pulse_thres: (float): a positive number setting the minimum distance between peaks and troughs
    :param plot_data_func: Function for plotting the data with detected peaks and troughs, an inset and the distribution
    of peak-ratios.
        plot_data_func(data, rate, peak_idx, trough_idx, idx0, idx1,
                       win_start_times, cv_interv, mean_ampl, cv_ampl, clipped_frac, cost,
                       thresh, valid_wins, **kwargs)
        :param data (array): the raw data.
        :param rate (float): the sampling rate of the data.
        :param peak_idx (array): indices into raw data indicating detected peaks.
        :param trough_idx (array): indices into raw data indicating detected troughs.
    :return: suggestion: (str). Returns the suggested fish-type ("pulse"/"wave").
    :return: peak_ratio: (float). Returns a float between 0. and 1. which gives the proportion of peak-2-trough,
                            from peak-2-peak time distance. (Wave-fishes should have larger values than pulse-fishes)
    """

    print('\nAnalyzing Fish-Type...')

    # threshold for peak detection:
    threshold = np.zeros(len(data))
    win_shift_indices = int(win_shift * samplerate)

    for inx0 in xrange(0, len(data) - win_shift_indices / 2, win_shift_indices):
        inx1 = inx0 + win_shift_indices
        threshold[inx0:inx1] = np.percentile(data[inx0:inx1], 99)

    # detect large peaks and troughs:
    peak_idx, trough_idx = pkd.detect_peaks(data, threshold)
    peak_idx, trough_idx = pkd.trim_closest(peak_idx, trough_idx)

    # get times of peaks and troughs
    pk_times = peak_idx/samplerate
    tr_times = trough_idx/samplerate

    pk_2_pk = np.diff(pk_times)
    pk_2_tr = np.abs(pk_times - tr_times)[:-1]

    # get the proportion of peak-2-trough, from peak-2-peak time distance
    r_tr = pk_2_tr / pk_2_pk
    r_tr[r_tr > 0.5] = 1 - r_tr[r_tr > 0.5]
    # ToDo: Some values are not between 0. and 1. as they should! Need to find a solution!

    peak_ratio = np.median(r_tr)
    suggestion = 'pulse' if peak_ratio < pulse_thres else 'wave'

    if plot_data_func:
        plot_data_func(data, samplerate, peak_idx, trough_idx, threshold, pulse_thres, suggestion, r_tr)

    print('\nFish-type is %s. r-value = %.3f' % (suggestion, peak_ratio))
    return suggestion, peak_ratio


def plot_type_detection(data, rate, peak_idx, trough_idx, peakdet_th, r_value_th, type_suggestion, r_vals):
    """ Plots the best-window, a zoomed index of it and the r-values that determine whether fish is pulse or wave-type.

    :param data: (array-like). amplitude data
    :param rate: (float). sample-rate in Hz
    :param peak_idx: (array of int). Array with peak indices
    :param trough_idx: (array of int). Array with trough indices
    :param peakdet_th: (float or array-like). threshold for peak detection
    :param r_value_th: (float). If the median r-value is smaller than this threshold, the fish is pulse-; else wave-type
    :param type_suggestion: (str). Fish-type suggested by the algorithm
    :param r_vals: (array-like). Array with r-values (peak2trough / peak2peak)
    """

    t = np.arange(len(data)) / rate
    fs = 14

    # Draw Figure with subplots
    fig, ax = plt.subplots(nrows=3, ncols=1, figsize=(8., 12.))

    # First and second subplots: raw-data with peak-trough-detection
    for enu, c_axis in enumerate(ax):
        if enu < 2:
            c_axis.plot(t, data, lw=2.5, color='purple', alpha=0.7)
            c_axis.plot(t[peak_idx], data[peak_idx], 'o', ms=11, mec='black', mew=1.5, color='crimson', alpha=0.7)
            c_axis.plot(t[trough_idx], data[trough_idx], 'o', ms=11, mec='black', mew=1.5, color='lime', alpha=0.7)
            c_axis.plot(t[::10], peakdet_th[::10], '-k', lw=2., rasterized=True, alpha=0.7)

    # Define Inset Boundaries and plot them
    inset_width = 20.  # in msec
    start_in = int((len(data)/2. - (inset_width//2.*rate*0.001)))  # the 0.001 converts msec to sec
    end_in = int((len(data)/2. + (inset_width//2.*rate*0.001)))
    max_amp = np.max(data)
    min_amp = np.min(data)

    ax[0].add_patch(Rectangle((t[start_in], min_amp),  # (x,y)
                              t[end_in]-t[start_in],  # width
                              max_amp + abs(min_amp),  # height
                              edgecolor='black', facecolor='white', lw=3))

    # Cosmetics
    ax[0].set_title('Best window with peak detection', fontsize=fs+2)
    ax[0].set_xlabel('Time [sec]', fontsize=fs)
    ax[0].set_ylabel('Amplitude [a.u.]', fontsize=fs)
    ax[0].tick_params(axis='both', which='major', labelsize=fs - 2)

    # Second Plot: Window inset
    ax[1].set_xlim(t[start_in], t[end_in])  # just set proper xlim!

    # Cosmetics
    ax[1].set_title('Best window Inset. Fish-type suggestion is %s' % type_suggestion, fontsize=fs + 2)
    ax[1].set_xlabel('Time [sec]', fontsize=fs)
    ax[1].set_ylabel('Amplitude [a.u.]', fontsize=fs)
    ax[1].tick_params(axis='both', which='major', labelsize=fs - 2)

    # Third Plot: r-values and Fish-type threshold
    hist, bins = np.histogram(r_vals, bins=50)
    width = 0.7 * (bins[1] - bins[0])
    center = (bins[:-1] + bins[1:]) / 2
    ax[2].bar(center, hist, align='center', facecolor='darkgreen', edgecolor='black', lw=2.5, alpha=0.7, width=width)
    ax[2].plot([r_value_th, r_value_th], [0, np.max(hist)], '--k', lw=2.5, alpha=0.7)

    # Cosmetics
    ax[2].set_xlabel('Peak2Trough / Peak2Peak', fontsize=fs)
    ax[2].set_ylabel('Counts', fontsize=fs)
    ax[2].tick_params(axis='both', which='major', labelsize=fs - 2)
    ax[2].set_title('Distribution of r-values', fontsize=fs + 2)

    pass


if __name__ == "__main__":
    print("\nChecking sortfishtype module ...\n")
    import sys
    import matplotlib.pyplot as plt

    if len(sys.argv) < 2:
        # generate data:
        print("Generating artificial waveform...")
        rate = 40000.0
        time = np.arange(0.0, 10.0, 1./rate)
        f1 = 100.0
        data0 = (0.5*np.sin(2.0*np.pi*f1*time)+0.5)**20.0
        amf1 = 0.3
        data1 = data0*(1.0-np.cos(2.0*np.pi*amf1*time))
        data1 += 0.2
        f2 = f1*2.0*np.pi
        data2 = 0.1*np.sin(2.0*np.pi*f2*time)
        amf3 = 0.15
        data3 = data2*(1.0-np.cos(2.0*np.pi*amf3*time))
        #data = data1+data3
        data = data2
        data += 0.01*np.random.randn(len(data))

    else:  # load data given by the user
        import dataloader as dl

        file_path = sys.argv[1]
        print("loading %s ...\n" % file_path)
        data, rate, unit = dl.load_data(sys.argv[1], 0)

    # run fish-type detector
    type_suggestion, r_val = type_detector(data, rate, plot_data_func=plot_type_detection)
    if len(sys.argv) >= 2:
        filename = file_path.split('/')[-1]
        title = 'Fish # %s is %s' % (filename, type_suggestion)
        fig = plt.gcf()
        fig.canvas.set_window_title(title)
    plt.tight_layout()
    plt.show()
    plt.close()
    quit()