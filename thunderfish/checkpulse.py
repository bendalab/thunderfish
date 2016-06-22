
"""
This module checks if the recorded signal corresponds to a wave- or a pulse-fish using two different approaches:
One checks for the width of an EOD compared to the distance to the next EOD. The second performs a power-spectrum-
analysis.
The key function for the pulse-width approach is check_pulse_width.
The key function for the power-spectrum-analysis approach is XXXX.
"""

import numpy as np
import peakdetection as pkd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

def psd_type_plot(freqs, power, proportions, percentiles, ax, fs, max_freq = 3000):
    """
    Makes a plot of what the rest of the modul is doing.

    This function takes the frequency and power array of a powerspectrum as well as the calculated percentiles array
    of the frequency bins (see: get_bin_percentiles()) and the proportions array calculated from these percentiles. With
    all these arrays this function is plotting what the rest of the modul doing.

    :param freqs:           (1-D array) frequency array of a psd.
    :param power:           (1-D array) power array of a psd.
    :param proportions:     (1-D array) proportions of the single psd bins.
    :param percentiles:     (2-D array) for every bin four values are calulated and stored in separate lists. These four
                            values are percentiles of the respective bins.
    :param ax:              (axis for plot) empty axis that is filled with content in the function.
    :param fs:              (int) fontsize for the plot.
    :param max_freq:        (float) maximum frequency that shall appear in the plot.
    """
    ax.plot(freqs[:int(max_freq / (freqs[-1] / len(freqs)))],
                10.0 * np.log10(power[:int(3000 / (freqs[-1] / len(freqs)))]), '-', alpha=0.5)
    for bin in range(len(proportions)):
        ax.fill_between([bin * 125, (bin + 1) * 125], percentiles[bin][0], percentiles[bin][1], color='red',
                        alpha=0.7)
        ax.fill_between([bin * 125, (bin + 1) * 125], percentiles[bin][1], percentiles[bin][2], color='green',
                        alpha=0.7)
        ax.fill_between([bin * 125, (bin + 1) * 125], percentiles[bin][2], percentiles[bin][3], color='red',
                        alpha=0.7)
    ax.set_xlim([0, 3000])
    ax.set_xlabel('Frequency', fontsize=fs)
    ax.set_ylabel('Power [dB]', fontsize=fs)

def psd_assignment(power, freqs, proportion_th = 0.27, freq_bins = 125, max_freq = 3000, outer_percentile= 1,
                   inner_percentile = 25, verbose=0, plot_data_func=None, **kwargs):
    """
    Function that is called when you got a PSD and want to find out from what fishtype this psd is. With the help of
    several other function it analysis the structur of the EOD and can with this approach tell what type of fish the PSD
    belongs to.

    :param power:           (1-D array) power array of a psd.
    :param freqs:           (1-D array) frequency array of a psd.
    :param proportion_th:   (float) Proportion of the data that defines if the psd belongs to a wave or a pulsefish.
    :param freq_bins:       (float) width of frequency bins in which the psd shall be divided (Hz).
    :param max_freq:        (float) maximum frequency that shall be provided in the separated power array.
    :param outer_percentile:(float) ((100-outer_percentile) - outer_percentile) / ((100-inner_percentile) - inner_percentile)
                            is the proportion that leeds to the decision if the psd belongs to a wave or pulsetype fish.
    :param inner_percentile:(float) ((100-outer_percentile) - outer_percentile) / ((100-inner_percentile) - inner_percentile)
                            is the proportion that leeds to the decision if the psd belongs to a wave or pulsetype fish.
    :param verbose:         (int) when the value is 1 you get additional shell output.
    :param plot_data_func:  (function) function (psdtypeplot()) that is used to create a axis for later plotting about the process of psd
                            type detection.
    :param **kwargs:        additional arguments that are passed to the plot_data_func().
    :return psd_type:       (string) "wave" or "pulse" depending on the proportion of the psd.
    :return proportions:    (1-D array) proportions of the single psd bins.
    """

    if verbose >= 1:
        print('Checking if pulse-PSD ...')
    res = np.mean(np.diff(freqs))

    # Take a 1-D array of powers (from powerspectrums), transforms it into dB and divides it into several bins.
    proportions = []
    all_percentiles = []
    for trial in range(int(max_freq / freq_bins)):
        tmp_power_db = 10.0 * np.log10(power[trial * int(freq_bins / res) : (trial +1) * int(freq_bins / res)])
        # calculates 4 percentiles for each powerbin
        percentiles = np.percentile(tmp_power_db, [outer_percentile, inner_percentile, 100-inner_percentile,
                                                  100-outer_percentile])
        all_percentiles.append(percentiles)
        proportions.append((percentiles[1] - percentiles[2]) / (percentiles[0] - percentiles[3]))

    pulse_psd = np.mean(proportions) > proportion_th

    if verbose >= 1:
        f_type = 'pulse' if pulse_psd else 'wave'
        print ('PSD-type is %s. proportion = %.3f' % (f_type, float(np.mean(proportions))))

    if plot_data_func:
        plot_data_func(freqs, power, np.asarray(proportions), np.asarray(all_percentiles), **kwargs)

    return pulse_psd, np.mean(proportions)

def check_pulse_width(data, samplerate, percentile_th=1., th_factor=0.8,
                      win_shift=0.5, pulse_thres=0.1, verbose=0, plot_data_func=None, **kwargs):

    """ Detects if fish is pulse or wave by calculating the proportion of the time distance between a peak and its
     following trough, relative to the time between consecutive peaks.

     WARNING! This function does not detect monophasic pulse-fishes such as Brachyhypopomus bennetti or Electrophorus.

    :param data: (1-D array). The data to be analyzed (Usually the best window already)
    :param samplerate: (float). Sampling rate of the data in Hz
    :param percentile_th: (float between 0. and 50.). The inter-percentile range TODO
    :param th_factor: (float). The threshold for peak detection is the inter-percentile-range multiplied by this factor.
    :param win_shift: (float). Time shift in seconds between windows.
    :param pulse_thres: (float). a positive number setting the minimum distance between peaks and troughs
    :param verbose: (int). if > 1, print information in the command line.
    :param plot_data_func: Function for plotting the data with detected peaks and troughs, an inset and the distribution
    of peak-ratios.
        plot_data_func(data, samplerate, peak_idx, trough_idx, peakdet_th, pulse_th, type_suggestion, pvt_dist, tvp_dist)
        :param data: (array). the raw data.
        :param samplerate: (float). the sampling rate of the data.
        :param peak_idx: (array). indices into raw data indicating detected peaks.
        :param trough_idx: (array). indices into raw data indicating detected troughs.
        :param peakdet_th: (float). a positive number setting the minimum distance between peaks and troughs.
        :param pulse_th: (float). If the median r-value is smaller than this threshold, the fish is pulse-; else wave-type
        :param type_suggestion: (bool). Fish-type suggested by the algorithm; True if pulse-fish, False if wave-fish
        :param pvt_dist: (array-like). distribution of r-values [pk2tr/pk2pk]
        :param tvp_dist: (array-like). distribution of r-values [tr2pk/tr2tr]
    :return: suggestion: (bool). True if pulse-fish, False if wave-fish
    :return: peak_ratio: (float). Returns a float between 0. and 1. which gives the proportion of peak-2-trough,
                            from peak-2-peak time distance. (Wave-fishes should have larger values than pulse-fishes)
    """

    def ratio(peak_idx, trough_idx):
        """ Calculates (peak-trough) / (peak-peak)

        :return: peak-ratio (float). The median of (peak-trough) / (peak-peak)
        :return: r_tr (array). The distribution of (peak-trough) / (peak-peak)
        """
        peaks, troughs = pkd.trim_to_peak(peak_idx, trough_idx)

        # get times of peaks and troughs, pk_times need to be floats!
        pk_times = peaks / float(samplerate)  # Actually there is no need to divide by samplerate.
        tr_times = troughs / float(samplerate)  # Time differences can be calculated using indices only.

        pk_2_pk = np.diff(pk_times)
        pk_2_tr = (tr_times - pk_times)[:-1]

        # get the proportion of peak-2-trough, from peak-2-peak time distance
        r_tr = pk_2_tr / pk_2_pk
        r_tr[r_tr > 0.5] = 1 - r_tr[r_tr > 0.5]  # fix for cases where trough of eod comes before peak

        peak_ratio = np.median(r_tr)

        return peak_ratio, r_tr

    if verbose > 1:
        print('Analyzing Fish-Type...')

    # threshold for peak detection:
    threshold = np.zeros(len(data))
    win_shift_indices = int(win_shift * samplerate)

    for inx0 in range(0, len(data), win_shift_indices):
        inx1 = inx0 + win_shift_indices
        threshold[inx0:inx1] = np.diff(np.percentile(
            data[inx0:inx1], [percentile_th, 100. - percentile_th]))*th_factor

    # detect large peaks and troughs:
    peak_idx, trough_idx = pkd.detect_peaks(data, threshold)

    pr_pvt, pvt_dist = ratio(peak_idx, trough_idx)
    pr_tvp, tvp_dist = ratio(trough_idx, peak_idx)

    peak_ratio = np.mean([pr_pvt, pr_tvp])

    suggestion = peak_ratio < pulse_thres

    if plot_data_func:
        plot_data_func(data, samplerate, peak_idx, trough_idx, threshold, pulse_thres, suggestion, pr_pvt, pr_tvp,
                       **kwargs)

    if verbose > 0:
        f_type = 'pulse' if suggestion else 'wave'
        print('Fish-type is %s. r-value = %.3f' % (f_type, peak_ratio))
    return suggestion, peak_ratio


def plot_width_period_ratio(data, samplerate, peak_idx, trough_idx, peakdet_th, pulse_th, type_suggestion,
                            pvt_dist, tvp_dist, ax, fs=14):
    """ Plots the data, a zoomed index of it and the r-values that determine whether fish is pulse or wave-type.

    :param data: (array-like). amplitude data
    :param samplerate: (float). sample-rate in Hz
    :param peak_idx: (array of int). Array with peak indices
    :param trough_idx: (array of int). Array with trough indices
    :param peakdet_th: (float). threshold for peak detection
    :param pulse_th: (float). If the median r-value is smaller than this threshold, the fish is pulse-; else wave-type
    :param type_suggestion: (bool). Fish-type suggested by the algorithm (True for pulse-type)
    :param pvt_dist: (array-like). Array with r-values (peak2trough / peak2peak)
    :param tvp_dist: (array-like). Array with r-values (trough2peak/ trough2trough)
    """

    def plot_peak_trough_hist(vals, ax, hist_color, plot_label, label_size):
        hist, bins = np.histogram(vals, bins=50)
        width = 0.7 * (bins[1] - bins[0])
        center = (bins[:-1] + bins[1:]) / 2
        ax.bar(center, hist, align='center', facecolor=hist_color, edgecolor='black', lw=2.5, alpha=0.4, width=width,
               label=plot_label)
        ax.plot([pulse_th, pulse_th], [0, np.max(hist)], '--k', lw=2.5, alpha=0.7)
        ax.legend(frameon=False, loc='best', fontsize=label_size-2)
        pass

    t = np.arange(len(data)) / samplerate

    # First and second subplots: raw-data with peak-trough-detection
    for enu, c_axis in enumerate(ax):
        if enu < 2:
            c_axis.plot(t, data, lw=2.5, color='purple', alpha=0.7)
            c_axis.plot(t[peak_idx], data[peak_idx], 'o', ms=11, mec='black', mew=1.5, color='crimson', alpha=0.7)
            c_axis.plot(t[trough_idx], data[trough_idx], 'o', ms=11, mec='black', mew=1.5, color='lime', alpha=0.7)
            c_axis.plot(t[::10], peakdet_th[::10], '--k', lw=2., rasterized=True, alpha=0.7, label='peakdet-threshold')
            c_axis.plot(t[::10], np.zeros(len(t[::10])), '--k', lw=2., rasterized=True, alpha=0.7)

    # Define Inset Boundaries and plot them
    inset_width = 20.  # in msec
    start_in = int((len(data) / 2. - (inset_width // 2. * samplerate * 0.001)))  # the 0.001 converts msec to sec
    end_in = int((len(data) / 2. + (inset_width // 2. * samplerate * 0.001)))
    max_amp = np.max(data)
    min_amp = np.min(data)

    ax[0].add_patch(Rectangle((t[start_in], min_amp),  # (x,y)
                              t[end_in]-t[start_in],  # width
                              max_amp + abs(min_amp),  # height
                              edgecolor='black', facecolor='white', lw=3))

    # Cosmetics
    ax[0].set_title('Raw-data with peak detection', fontsize=fs+2)
    ax[0].set_xlabel('Time [sec]', fontsize=fs)
    ax[0].set_ylabel('Amplitude [a.u.]', fontsize=fs)
    ax[0].tick_params(axis='both', which='major', labelsize=fs - 2)

    # Second Plot: Window inset
    ax[1].set_xlim(t[start_in], t[end_in])  # just set proper xlim!

    # Cosmetics
    f_type = 'pulse' if type_suggestion else 'wave'
    ax[1].set_title('Inset of plot above. Fish-type suggestion is %s' % f_type, fontsize=fs + 2)
    ax[1].set_xlabel('Time [sec]', fontsize=fs)
    ax[1].set_ylabel('Amplitude [a.u.]', fontsize=fs)
    ax[1].tick_params(axis='both', which='major', labelsize=fs - 2)
    ax[1].legend(frameon=False, loc='best')

    # Third Plot: r-values and Fish-type threshold
    plot_peak_trough_hist(pvt_dist, ax[2], hist_color='darkgreen', plot_label='Peak2Trough', label_size=fs)
    plot_peak_trough_hist(tvp_dist, ax[2], hist_color='cornflowerblue', plot_label='Trough2Peak', label_size=fs)

    # Cosmetics
    ax[2].set_xlabel('Peak2Trough / Peak2Peak', fontsize=fs)
    ax[2].set_ylabel('Counts', fontsize=fs)
    ax[2].tick_params(axis='both', which='major', labelsize=fs - 2)
    ax[2].set_title('Distribution of r-values', fontsize=fs + 2)

    pass


if __name__ == "__main__":
    print("\nChecking sortfishtype module ...\n")
    import sys
    import bestwindow as bw
    import powerspectrum as ps

    if len(sys.argv) < 2:
        # generate data:
        # ToDo: Make a parameter for the user to choose whether pulse- or wave-fish data should be generated!!
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
        data = data0
        data += 0.01*np.random.randn(len(data))

    else:  # load data given by the user
        import dataloader as dl

        file_path = sys.argv[1]
        print("loading %s ...\n" % file_path)
        data, rate, unit = dl.load_data(sys.argv[1], 0)

    bwin_data, clip = bw.best_window(data, rate)
    psd_data = ps.multi_resolution_psd(bwin_data, rate)

    # Draw Figure with subplots
    fig, ax = plt.subplots(nrows=3, ncols=1, figsize=(8., 12.))

    # run fish-type detector
    type_suggestion, r_val = check_pulse_width(data, rate, plot_data_func=plot_width_period_ratio, ax=ax)

    if len(sys.argv) >= 2:
        filename = file_path.split('/')[-1]
        f_type = 'pulse' if type_suggestion else 'wave'
        title = 'Fish # %s is %s' % (filename, f_type)
        fig = plt.gcf()
        fig.canvas.set_window_title(title)
    plt.tight_layout()

    fig2, ax2 = plt.subplots()
    psd_type, proportions = psd_assignment(psd_data[0], psd_data[1], verbose=1, plot_data_func=psd_type_plot, ax=ax2, fs=12)
    plt.tight_layout()
    plt.show()
