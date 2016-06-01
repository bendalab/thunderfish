import numpy as np
import sys

def bin_it(power, freq_bins, max_freq, res):
    """
    Take a 1-D array of powers (from powerspectrums), transforms it into dB and devides it into several bins.

    :param power: (1-D array)
    :param freq_bins: (float)
    :param max_freq: (float)
    :param res: (float)
    :return power_db: (2-D array)
    """
    power_db = []
    for trial in np.arange(max_freq / freq_bins):
        tmp_power_db = 10.0 * np.log10(power[trial * int( freq_bins / (res) ) : (trial +1) * int( freq_bins / (res) ) - 1])
        power_db.append(tmp_power_db)
    return power_db

def get_bin_percentiles(power_db):
    """
    Takes a 2-D array if lists. For every list the function calculates several percentails that are later returned.

    :param power_db: (2-D array)
    :return: (4-D array)
    """
    power_db_top = np.ones(len(power_db))
    power_db_upper_middle = np.ones(len(power_db))
    power_db_lower_middle = np.ones(len(power_db))
    power_db_bottom = np.ones(len(power_db))

    for fbin in np.arange(len(power_db)):
        power_db_top[fbin] = np.percentile(power_db[fbin], 99)
        power_db_upper_middle[fbin] = np.percentile(power_db[fbin],75)
        power_db_lower_middle[fbin] = np.percentile(power_db[fbin], 25)
        power_db_bottom[fbin] = np.percentile(power_db[fbin], 1)

    return [power_db_top, power_db_upper_middle, power_db_lower_middle, power_db_bottom]


def psd_type_main(power, freqs, freq_bins=125, max_freq = 3000, return_percentiles= False):
    """
    Function that is called when you got a PSD and want to find out from what fishtype this psd is. It with the help of
    several other function it analysis the structur of the EOD and can with this approach tell what type of fish the PSD
    comes from.

    :param power: (1-D array)
    :param freqs: (1-D array)
    :param freq_bins: (float)
    :param max_freq: (float)
    :param return_percentiles: (boolean)
    :return psd_type: (string)
    :return proportions: (1-D array)
    :return percentiles: (2-D array)

    """
    print('try to figure out psd type ...')
    res = freqs[-1]/len(freqs) # resolution

    power_db = bin_it(power, freq_bins, max_freq, res)

    percentiles= get_bin_percentiles(power_db)

    proportions = [(percentiles[1][i] - percentiles[2][i]) / (percentiles[0][i] - percentiles[3][i])
                   for i in np.arange(len(power_db))]

    if np.mean(proportions) < 0.27:
        psd_type = 'wave'
    else:
        psd_type = 'pulse'
    print ('The PSD belongs to a %s-fish (%.2f)' % (psd_type, float(np.mean(proportions))))

    if return_percentiles:
        return psd_type, proportions, percentiles
    else:
        return psd_type, proportions

def get_example_data(audio_file, channel=0, verbose=None):
    """
    This function shows in part the same components of the thunderfish.py poject. Here several moduls are used to load
    some data to dispay the functionality of the psdtype.py modul.

    :param audio_file:
    :return power:
    :return freqs:
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

    print('calculation powerspecturm ...\n')
    power, freqs = ps.powerspectrum_main(bwin_data, samplrate)

    return power, freqs

if __name__ == '__main__':
    try:
        import powerspectrum as ps
        import config_tools as ct
        import dataloader as dl
        import bestwindow as bw
    except ImportError:
        print('')
        print('This modul is dependent on the moduls')
        print('"powerspectrum.py", "config_tools.py", "dataloader.py" and "bestwindow.py"\n')
        print('If you want to download them please visit:')
        print('https://github.com/bendalab/thunderfish')
        print('')
        quit()

    print('Algorithm that analysis the structur of a powerspectrum to tell if it belongs to a wave of pulsetype fish.')
    print('')
    print('Usage:')
    print('  python psdtype.py [-p] <audiofile>')
    print('  -p: plot data')
    print('')

    if len(sys.argv) <=1:
        quit()

    plot = False
    if len(sys.argv) > 2 and sys.argv[1] == '-p':
        plot = True

    file = sys.argv[-1]
    power, freqs = get_example_data(file)

    psd_type, proportions, percentiles = psd_type_main(power, freqs, return_percentiles=True)

    if plot:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        ax.plot(freqs[:int(3000 / (freqs[-1] / len(freqs)))],
                10.0 * np.log10(power[:int(3000 / (freqs[-1] / len(freqs)))]), '-', alpha=0.5)
        for bin in np.arange(len(proportions)):
            ax.fill_between([bin * 125, (bin + 1) * 125], percentiles[0][bin], percentiles[1][bin], color='red',
                            alpha=0.7)
            ax.fill_between([bin * 125, (bin + 1) * 125], percentiles[1][bin], percentiles[2][bin], color='green',
                            alpha=0.7)
            ax.fill_between([bin * 125, (bin + 1) * 125], percentiles[2][bin], percentiles[3][bin], color='red',
                            alpha=0.7)
        ax.set_xlim([0, 3000])
        ax.set_xlabel('Frequency')
        ax.set_ylabel('Power [dB]')
        plt.show()
