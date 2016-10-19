import numpy as np
from .dataloader import load_data
from .consistentfishes import consistent_fishes
from .harmonicgroups import harmonic_groups
from .powerspectrum import multi_resolution_psd


def extract_main_freqs_and_db(data, samplerate):
    psd_data = multi_resolution_psd(data, samplerate, fresolution=[0.5, 2 * 0.5, 4 * 0.5])

    # find the fishes in the different powerspectra and extract fund. frequency and power
    fishlists = []
    for i in range(len(psd_data)):
        fishlist = harmonic_groups(psd_data[i][1], psd_data[i][0])[0]
        fishlists.append(fishlist)

    filtered_fishlist = consistent_fishes(fishlists)
    tmp = np.vstack([filtered_fishlist[e][0] for e in range(len(filtered_fishlist))])

    freqs = tmp[:, 0]
    pows = tmp[:, 1]
    decibel = 10.0 * np.log10(pows)  # calculate decibel using 1 as P0
    return freqs, decibel


def write_csv(filename, freq, decibel):
    header = ['fundamental frequency', 'dB']
    # convert to strings
    freq = np.array(['%.3f' % e for e in freq])
    decibel = np.array(['%.3f' % e for e in decibel])

    with open(filename, 'wb') as fin:
        fin.write(str.encode(','.join(header)))
        fin.write(str.encode('\n'))
        for row_id in range(len(freq)):

            fin.write(str.encode(','.join([freq[row_id], decibel[row_id]])))
            fin.write(str.encode('\n'))
    pass


if __name__ == '__main__':
    print("\nChecking csvmaker module ...")
    import os
    import sys
    from .fakefish import generate_alepto

    if len(sys.argv) == 1:  # user did not give a specific file to analyze
        filename = 'csvmaker_testfile.csv'
        # Generate data with four fishes.
        duration = 20.
        samplerate = 44100.0
        fishfreqs = [400., 650, 924, 1270]
        for idx in range(len(fishfreqs)):
            if idx == 0:
                data = generate_alepto(fishfreqs[idx], samplerate, duration)
            else:
                data += generate_alepto(fishfreqs[idx], samplerate, duration)

    elif len(sys.argv) == 2:  # user specified a file to be analized
        filename = os.path.splitext(os.path.basename(sys.argv[1]))[0] + '.csv'
        data, samplerate, unit = load_data(sys.argv[1], 0)

    fund_freqs, db = extract_main_freqs_and_db(data, samplerate)
    write_csv(filename, fund_freqs, db)

    print('\ncsv_file created in %s' % filename)
