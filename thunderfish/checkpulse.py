"""
# Check for pulse-type weakly electric fish
Functions for checking whether a pulse-type or a wave-type weakly electric fish
is present in a recording.

# Main functions
- `check_pulse_width()`: checks for a pulse-type fish based on the width of detected peaks.
- `check_pulse_psd()`: checks for pulse-type fish based on its signature on the power sepctrum.

## Visualization
- `plot_width_period_ratio()`: visualizes the check_pulse_width() algorithm.
- `plot_psd_proportion()`: visualizes the check_pulse_psd() algorithm.

## Configuration parameter
- `add_check_pulse_width_config()': add parameters for check_pulse_width() to configuration.
- `check_pulse_width_args()`: retrieve parameters for check_pulse_width() from configuration.
- `add_check_pulse_psd_config()`: add parameters for check_pulse_psd() to configuration.
- `check_pulse_psd_args()`: retrieve parameters for check_pulse_psd() from configuration.

"""

import numpy as np
from .eventdetection import percentile_threshold, detect_peaks, trim_to_peak
from .powerspectrum import decibel
try:
    from matplotlib.patches import Rectangle
except ImportError:
    pass


def check_pulse_width(data, samplerate, th_factor=0.6, percentile=10.0,
                      pulse_thresh=0.1, verbose=0, plot_data_func=None, **kwargs):
    """Detects if a fish is pulse- or wave-type based on the proportion of the time distance
    between a peak and its following trough, relative to the time between consecutive peaks.


    Parameters
    ----------
    data: 1-D array
         The data to be analyzed (Usually the best window already).
    samplerate: float
         Sampling rate of the data in Hertz.
    percentile: float
         The interpercentile range is computed at percentile and 100.0-percentile.
    th_factor: float
         The threshold for peak detection is the inter-percentile-range
         multiplied by this factor.
    pulse_thresh: float
         A positive number setting the minimum distance between peaks and troughs.
    verbose: int
         If > 1, print information in the command line.
    plot_data_func(data, samplerate, peak_idx, trough_idx, peakdet_th, pulse_th, pulse_fish, pvt_dist, tvp_dist):
        data: array
             The raw data.
        samplerate: float
             The sampling rate of the data.
        peak_idx: array
             Indices into raw data indicating detected peaks.
        trough_idx: array
             Indices into raw data indicating detected troughs.
        peakdet_th: float
             A positive number setting the minimum distance between peaks and troughs.
        pulse_th: float
             If the median r-value is smaller than this threshold, the fish is pulse-; else wave-type
        pulse_fish: bool
             True if algorithm suggests a pulse-type fish.
        pvt_dist: array-like
             Distribution of r-values [pk2tr/pk2pk]
        tvp_dist: array-like
             Distribution of r-values [tr2pk/tr2tr]

    Returns
    -------
    pulse_fish: bool
        True if algorithm suggests a pulse-type fish.
    peak_ratio: float
        Returns a float between 0. and 1. which gives the proportion of peak-2-trough,
        from peak-2-peak time distance, i.e. pulse width relative to pulse interval.
    interval: float
        The mean inter-pulse-interval.
    """

    def ratio(peak_idx, trough_idx):
        """ Calculates (peak-trough) / (peak-peak)

        Returns
        -------

        peak-ratio: float
            The median of (peak-trough) / (peak-peak)
        r_tr: array
            The distribution of (peak-trough) / (peak-peak)
        """
        peaks, troughs = trim_to_peak(peak_idx, trough_idx)
        if len(peaks) < 2:
            return 1.0, np.array([])

        pk_2_pk = np.diff(peaks).astype(np.float)
        pk_2_tr = (troughs - peaks)[:-1].astype(np.float)

        # get the proportion of peak-2-trough, from peak-2-peak time distance
        r_tr = pk_2_tr / pk_2_pk
        r_tr[r_tr > 0.5] = 1.0 - r_tr[r_tr > 0.5]  # fix for cases where trough of eod comes before peak

        peak_ratio = np.median(r_tr)

        return peak_ratio, r_tr

    # threshold for peak detection:
    threshold = percentile_threshold(data, th_factor=th_factor, percentile=percentile)
    if verbose > 0:
        print('check_pulse_width: threshold for pulse detection is %g' % threshold)
    
    # detect large peaks and troughs:
    peak_idx, trough_idx = detect_peaks(data, threshold)

    pr_pvt, pvt_dist = ratio(peak_idx, trough_idx)
    pr_tvp, tvp_dist = ratio(trough_idx, peak_idx)
    
    peak_ratio = np.mean([pr_pvt, pr_tvp])

    pulse_fish = peak_ratio < pulse_thresh

    if len(peak_idx) > 1:
        interval = np.mean(np.diff(peak_idx)/samplerate)
    else:
        interval = -1.0
    
    if plot_data_func:
        plot_data_func(data, samplerate, peak_idx, trough_idx, threshold, pulse_thresh,
                       pulse_fish, pr_pvt, pr_tvp, **kwargs)

    if verbose > 0:
        f_type = 'pulse' if pulse_fish else 'wave'
        print('  fish-type is %s. pulse-width-ratio is %.3f' % (f_type, peak_ratio))
        
    return pulse_fish, peak_ratio, interval


def check_pulse_psd(power, freqs, proportion_th=0.27, freq_bin_width=125.0, max_freq=3000.0,
                    outer_percentile=1.0, inner_percentile=25.0, verbose=0,
                    plot_data_func=None, **kwargs):
    """Detects if a fish is pulse- or wave-type based on the inter-quartile range
    relative to the inter-percentile range in the power-spectrum.

    Parameters
    ----------
    power: 1-D array
        Power array of a power spectrum.
    freqs: 1-D array
        Frequency array of a power spectrum.
    proportion_th: float
        Proportion of the data that defines if the psd belongs to a wave or a pulsefish.
    freq_bin_width: float
        Width of frequency bins in which the psd shall be divided (Hz).
    max_freq: float
        Maximum frequency that shall be provided in the separated power array.
    outer_percentile: float
        ((100-outer_percentile) - outer_percentile) / ((100-inner_percentile) - inner_percentile)
        is the proportion that leeds to the decision if the psd belongs to a wave
        or pulsetype fish.
    inner_percentile: float
        ((100-outer_percentile) - outer_percentile) / ((100-inner_percentile) - inner_percentile)
        is the proportion that leeds to the decision if the psd belongs to a wave
        or pulsetype fish.
    verbose: int
        When the value is 1 you get additional shell output.
    plot_data_func: function
        Visualize the algorithm.
    **kwargs:
        

    Returns
    -------
    pulse_fish: bool
        True if algorithm suggests a pulse-type fish.
    proportions: 1-D array
        Proportions of the single psd bins.
    """

    if verbose >= 1:
        print('checking for pulse-type fish in power spectrum ...')
    res = np.mean(np.diff(freqs))

    # Take a 1-D array of powers (from powerspectrums), transforms it into dB and divides it into several bins.
    proportions = []
    all_percentiles = []
    for trial in range(int(max_freq / freq_bin_width)):
        tmp_power_db = decibel(power[trial * int(freq_bin_width / res): (trial + 1) * int(freq_bin_width / res)])
        # calculates 4 percentiles for each powerbin
        percentiles = np.percentile(tmp_power_db, [outer_percentile, inner_percentile,
                                                   100 - inner_percentile,
                                                   100 - outer_percentile])
        all_percentiles.append(percentiles)
        proportions.append((percentiles[1] - percentiles[2]) / (percentiles[0] - percentiles[3]))

    percentile_ratio = np.mean(proportions)

    pulse_fish = percentile_ratio > proportion_th

    if verbose > 0:
        f_type = 'pulse' if pulse_fish else 'wave'
        print ('PSD-type is %s. proportion = %.3f' % (f_type, float(np.mean(proportions))))

    if plot_data_func:
        plot_data_func(freqs, power, proportions, all_percentiles, pulse_fish, **kwargs)

    return pulse_fish, percentile_ratio


def plot_width_period_ratio(data, samplerate, peak_idx, trough_idx, peakdet_th, pulse_th,
                            pulse_fish, pvt_dist, tvp_dist, ax, fs=14):
    """Plots the data, a zoomed index of it and the peak-width versus peak-period ratios
    that determine whether fish is pulse or wave-type.

    Parameters
    ----------
    data: array-like
        Amplitude data.
    samplerate: float
        Sampling rate in Hz.
    peak_idx: array of int
        Array with peak indices.
    trough_idx: array of int
        Array with trough indices.
    peakdet_th: float
        Threshold for peak detection.
    pulse_th: float
        If the median r-value is smaller than this threshold, the fish is pulse-;
        else wave-type.
    pulse_fish: bool
        Fish-type suggested by the algorithm (True for pulse-type).
    pvt_dist: array-like
        Array with r-values (peak2trough / peak2peak).
    tvp_dist: array-like
        Array with r-values (trough2peak/ trough2trough).
    """

    def plot_peak_trough_hist(vals, ax, hist_color, plot_label, label_size):
        hist, bins = np.histogram(vals, bins=50)
        width = 0.7 * (bins[1] - bins[0])
        center = (bins[:-1] + bins[1:]) / 2
        ax.bar(center, hist, align='center', facecolor=hist_color, edgecolor='black', lw=2.5, alpha=0.4, width=width,
               label=plot_label)
        ax.plot([pulse_th, pulse_th], [0, np.max(hist)], '--k', lw=2.5, alpha=0.7)
        ax.legend(frameon=False, loc='best', fontsize=label_size - 2)
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

    # define inset boundaries and plot them:
    inset_width = 20.  # in msec
    start_in = int((len(data) / 2. - (inset_width // 2. * samplerate * 0.001)))  # the 0.001 converts msec to sec
    end_in = int((len(data) / 2. + (inset_width // 2. * samplerate * 0.001)))
    max_amp = np.max(data)
    min_amp = np.min(data)

    ax[0].add_patch(Rectangle((t[start_in], min_amp),  # (x,y)
                              t[end_in] - t[start_in],  # width
                              max_amp + abs(min_amp),  # height
                              edgecolor='black', facecolor='white', lw=3))

    # Cosmetics
    ax[0].set_title('Raw-data with peak detection', fontsize=fs + 2)
    ax[0].set_xlabel('Time [sec]', fontsize=fs)
    ax[0].set_ylabel('Amplitude [a.u.]', fontsize=fs)
    ax[0].tick_params(axis='both', which='major', labelsize=fs - 2)

    # Second Plot: Window inset
    ax[1].set_xlim(t[start_in], t[end_in])  # just set proper xlim!

    # Cosmetics
    f_type = 'pulse' if pulse_fish else 'wave'
    ax[1].set_title('Inset of plot above. Suggested fish-type is %s' % f_type, fontsize=fs + 2)
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
    ax[2].set_title('Distribution of peak-width versus peak interval ratios', fontsize=fs + 2)


def plot_psd_proportion(freqs, power, proportions, percentiles, pulse_fish,
                        ax, fs, max_freq = 3000):
    """Visualizes the check_pulse_psd() algorithm.

    This function takes the frequency and power array of a powerspectrum
    as well as the calculated percentiles array of the frequency bins
    and the proportions array calculated from these percentiles.
    
    Parameters
    ----------
    freqs: 1-D array
        Frequency array of a psd.
    power: 1-D array
        Power array of a psd.
    proportions: 1-D array
        Proportions of the single psd bins.
    percentiles: 2-D array
        For every bin four values are calulated and stored in separate lists. These four
        values are percentiles of the respective bins.
    ax: axis for plot
        Empty axis that is filled with content in the function.
    fs: int
        Fontsize for the plot.
    max_freq: float
        Maximum frequency that shall appear in the plot.
    """
    f_type = 'pulse' if pulse_fish else 'wave'
    ax.set_title('Suggested fish-type is %s' % f_type, fontsize=fs + 2)
    ax.plot(freqs[:int(max_freq / (freqs[-1] / len(freqs)))],
            decibel(power[:int(3000 / (freqs[-1] / len(freqs)))]), '-', alpha=0.5)
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


def add_check_pulse_width_config(cfg, th_factor=0.6, percentile=10.0, pulse_thresh=0.1):
    """ Add parameter needed for check_pulse_width() as
    a new section to a configuration.

    Parameters
    ----------
    cfg: ConfigFile
        The configuration.
    See check_pulse_width() for details on the remaining arguments.
    """

    cfg.add_section('Check pulse widths:')
    cfg.add('pulseWidthPercentile', percentile, '%', 'The variance of the data is measured as the interpercentile range.')
    cfg.add('pulseWidthThresholdFactor', th_factor, '', 'The threshold for peak detection is this factor multiplied with the interpercentile range.')
    cfg.add('pulseWidthThresholdRatio', pulse_thresh, '', 'A pulsefish is detected if the width of the pulses relative to the intervals is smaller than this threshold.')


def check_pulse_width_args(cfg):
    """ Translates a configuration to the
    respective parameter names of the function check_pulse_width().
    The return value can then be passed as key-word arguments to this function.

    Parameters
    ----------
    cfg: ConfigFile
        The configuration.

    Returns
    -------
    a: dict
        Dictionary with names of arguments of the check_pulse_width() function
        and their values as supplied by `cfg`.
    """
    a = cfg.map({'th_factor': 'pulseWidthThresholdFactor',
                 'percentile': 'pulseWidthPercentile',
                 'pulse_thresh': 'pulseWidthThresholdRatio'})
    return a


def add_check_pulse_psd_config(cfg, proportion_th=0.27, freq_bin_width=125.0, max_freq=3000.0,
                               outer_percentile=1.0, inner_percentile=25.0):
    """ Add parameter needed for check_pulse_psd() as
    a new section to a configuration.

    Parameters
    ----------
    cfg: ConfigFile
        The configuration.
    See check_pulse_psd() for details on the remaining arguments.
    """

    cfg.add_section('Check for pulse fish in power spectrum:')
    cfg.add('pulsePSDProportionThreshold', proportion_th, '', 'Proportion of the data that defines if the power spectrum is the one of a pulsefish.')
    cfg.add('pulsePSDFrequencyBinWidth', freq_bin_width, 'Hz', 'Width of frequency bins used for analyzing power distribution.')
    cfg.add('pulsePSDMaximumFrequency', max_freq, 'Hz', 'Maximum frequency up to which the power spectrum is analyzed.')
    cfg.add('pulsePSDOuterPercentile', outer_percentile, '%', 'The interpercentile range of powers used as reference.')
    cfg.add('pulsePSDInnerPercentile', inner_percentile, '%', 'The interpercentile range of powers that is divided by the out percentile range.')


def check_pulse_psd_args(cfg):
    """ Translates a configuration to the
    respective parameter names of the function check_pulse_psd().
    The return value can then be passed as key-word arguments to this function.

    Parameters
    ----------
    cfg: ConfigFile
        The configuration.

    Returns
    -------
    a: dict
        Dictionary with names of arguments of the check_pulse_psd() function
        and their values as supplied by `cfg`.
    """
    a = cfg.map({'proportion_th': 'pulsePSDProportionThreshold',
                 'freq_bin_width': 'pulsePSDFrequencyBins',
                 'max_freq': 'pulsePSDMaximumFrequency',
                 'outer_percentile': 'pulsePSDOuterPercentile',
                 'inner_percentile': 'pulsePSDInnerPercentile'})
    return a

    
if __name__ == "__main__":
    import sys
    import matplotlib.pyplot as plt
    from .bestwindow import best_window
    from .powerspectrum import multi_resolution_psd
    from .fakefish import generate_monophasic_pulses, generate_biphasic_pulses, generate_triphasic_pulses, generate_alepto

    print("\nChecking checkpulse module ...\n")
    
    # generate data:
    rate = 44100.0
    if len(sys.argv) < 2:
        data = generate_biphasic_pulses(80.0, rate, 8.0)
    elif sys.argv[1] == '-w':
        data = generate_alepto(600.0, rate, 8.0)
    elif sys.argv[1] == '-m':
        data = generate_monophasic_pulses(80.0, rate, 8.0)
    elif sys.argv[1] == '-b':
        data = generate_biphasic_pulses(80.0, rate, 8.0)
    elif sys.argv[1] == '-t':
        data = generate_triphasic_pulses(80.0, rate, 8.0)
    else:  # load data given by the user
        from .dataloader import load_data

        file_path = sys.argv[1]
        print("loading %s ...\n" % file_path)
        rawdata, rate, unit = load_data(sys.argv[1], 0)
        data, _ = best_window(rawdata, rate)

    # draw figure with subplots:
    fig1, ax1 = plt.subplots(nrows=3, ncols=1, figsize=(8., 12.))

    # run pulse-width-based detector:
    pulse_fish, r_val = check_pulse_width(data, rate,
                                          plot_data_func=plot_width_period_ratio, ax=ax1)
    plt.tight_layout()

    fig2, ax2 = plt.subplots()
    psd_data = multi_resolution_psd(data, rate)
    psd_type, proportions = check_pulse_psd(psd_data[0], psd_data[1], verbose=1,
                                            plot_data_func=plot_psd_proportion, ax=ax2, fs=12)
    plt.tight_layout()
    plt.show()
