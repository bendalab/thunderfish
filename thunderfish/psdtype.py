import numpy as np

def bin_it(power, freq_bins, max_freq, res):
    power_db = []
    for trial in np.arange(max_freq / freq_bins):
        tmp_power_db = 10.0 * np.log10(power[trial * int( freq_bins / (res) ) : (trial +1) * int( freq_bins / (res) ) - 1])
        power_db.append(tmp_power_db)
    return power_db

def get_proportions(power_db):
    power_db_top = np.ones(len(power_db))
    power_db_upper_middle = np.ones(len(power_db))
    power_db_lower_middle = np.ones(len(power_db))
    power_db_bottom = np.ones(len(power_db))

    for fbin in np.arange(len(power_db)):
        power_db_top[fbin] = np.percentile(power_db, 99)
        power_db_upper_middle[fbin] = np.percentile(power_db,75)
        power_db_lower_middle[fbin] = np.percentile(power_db, 25)
        power_db_bottom[fbin] = np.percentile(power_db, 1)

    proportions = [(power_db_upper_middle[i] - power_db_lower_middle[i]) / (power_db_top[i] - power_db_bottom[i])
                   for i in np.arange(len(power_db))]

    return proportions

def psd_type_main(power, freqs, freq_bins=125, max_freq = 3000):
    print('try to figure out psd type ...')
    res = freqs[-1]/len(freqs)

    power_db = bin_it(power, freq_bins, max_freq, res)

    proportions = get_proportions(power_db)

    if np.mean(proportions) < 0.27:
        psd_type = 'wave'
    else:
        psd_type = 'pulse'

    return psd_type, np.mean(proportions)

if __name__ == '__main__':
    import powerspectrum as ps

    fundamental = [300, 450] # Hz
    samplingrate = 100000
    time = np.linspace(0, 8-1/samplingrate, 8*samplingrate)
    data = np.sin(time * 2 * np.pi* fundamental[0]) + np.sin(time * 2 * np.pi* fundamental[1])

    power, freqs = ps.powerspectrum_main(data, samplingrate)

    psd_type, mean_proportions = psd_type_main(power, freqs)
    print(psd_type, '-fish', mean_proportions)