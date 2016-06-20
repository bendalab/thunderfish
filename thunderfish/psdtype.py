import numpy as np
import sys

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


    ################### doesnt work ###########################
    pulse_psd = np.mean(proportions) > proportion_th

    # TODO check Juan how to print pulse versus wave
    if verbose >= 1:
        print ('Pulse-PSD is %s. proportion = %.3f' % (pulse_psd, float(np.mean(proportions))))

    if plot_data_func:
        plot_data_func(freqs, power, np.asarray(proportions), np.asarray(all_percentiles), **kwargs)

    return pulse_psd, float(np.mean(proportions))

if __name__ == '__main__':
    def get_example_data(audio_file):
        """
        This function shows in part the same components of the thunderfish.py poject. Here several moduls are used to load
        some data to dispay the functionality of the psdtype.py modul.

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

        print('calculation powerspecturm ...\n')
        psd_data = ps.multi_resolution_psd(bwin_data, samplerate)

        return psd_data[0], psd_data[1]

    try:
        import powerspectrum as ps
        import dataloader as dl
        import bestwindow as bw
    except ImportError:
        print('')
        print('This modul is dependent on the moduls')
        print('"powerspectrum.py", "dataloader.py" and "bestwindow.py"\n')
        print('If you want to download them please visit:')
        print('https://github.com/bendalab/thunderfish')
        print('')
        quit()

    print('Algorithm that analysis the structure of a powerspectrum to tell if it belongs to a wave of pulsetype fish.')
    print('')
    print('Usage:')
    print('  python psdtype.py <audiofile>')
    print('')

    if len(sys.argv) == 2:
        file = sys.argv[-1]
        power, freqs = get_example_data(file)
    else:
        print('need <audiofile> !!!')
        quit()

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    psd_type, proportions = psd_assignment(power, freqs, verbose=1, plot_data_func=psd_type_plot, ax=ax, fs=12)
    plt.show()
