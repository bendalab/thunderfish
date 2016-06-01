import numpy as np
import sys

def bin_it(power, freq_bins, max_freq, res):
    power_db = []
    for trial in np.arange(max_freq / freq_bins):
        tmp_power_db = 10.0 * np.log10(power[trial * int( freq_bins / (res) ) : (trial +1) * int( freq_bins / (res) ) - 1])
        power_db.append(tmp_power_db)
    return power_db

def get_proportions(power_db, return_all=False):
    power_db_top = np.ones(len(power_db))
    power_db_upper_middle = np.ones(len(power_db))
    power_db_lower_middle = np.ones(len(power_db))
    power_db_bottom = np.ones(len(power_db))

    for fbin in np.arange(len(power_db)):
        power_db_top[fbin] = np.percentile(power_db[fbin], 99)
        power_db_upper_middle[fbin] = np.percentile(power_db[fbin],75)
        power_db_lower_middle[fbin] = np.percentile(power_db[fbin], 25)
        power_db_bottom[fbin] = np.percentile(power_db[fbin], 1)

    proportions = [(power_db_upper_middle[i] - power_db_lower_middle[i]) / (power_db_top[i] - power_db_bottom[i])
                   for i in np.arange(len(power_db))]

    if return_all:
        return proportions, [power_db_top, power_db_upper_middle, power_db_lower_middle, power_db_bottom]
    else:
        return proportions

def psd_type_main(power, freqs, freq_bins=125, max_freq = 3000, return_all= False):
    print('try to figure out psd type ...')
    res = freqs[-1]/len(freqs)

    power_db = bin_it(power, freq_bins, max_freq, res)

    if return_all:
        proportions, percentiles= get_proportions(power_db, return_all)
    else:
        proportions= get_proportions(power_db)

    if np.mean(proportions) < 0.27:
        psd_type = 'wave'
    else:
        psd_type = 'pulse'

    if return_all:
        return psd_type, proportions, percentiles
    else:
        return psd_type, proportions

def get_example_data(audio_file, channel=0, verbose=None):
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
        print('"powerspectrum.py", "config_tools.py", "dataloader.py" and "bestwindow.py"')
        print('If you want do download this modul please visit:')
        print('https://github.com/bendalab/thunderfish')
        print('')
        quit()

    plot = False
    if sys.argv > 2 and sys.argv[1] == '-p':
        plot = True

    file = sys.argv[-1]
    power, freqs = get_example_data(file)

    psd_type, proportions, percentiles = psd_type_main(power, freqs, return_all=True)
    print ('The PSD belongs to a %s-fish (%.2f)' %(psd_type, np.mean(proportions)))

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

        plt.show()
