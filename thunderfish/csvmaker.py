import numpy as np
from .dataloader import load_data
from .consistentfishes import consistent_fishes
from .harmonicgroups import harmonic_groups
from .powerspectrum import multi_resolution_psd


def extract_main_freqs_and_db(filtered_grouplist):

    if len(filtered_grouplist) == 0:
        fundamentals = np.array([])
    elif hasattr(filtered_grouplist[0][0][0], '__len__'):
        fundamentals = []
        pows = []
        for groups in filtered_grouplist:
            fundamentals.append([harmonic_group[0][0] for harmonic_group in groups])
            pows.append([harmonic_group[0][1] for harmonic_group in groups])
    else:
        fundamentals = [harmonic_group[0][0] for harmonic_group in filtered_grouplist]
        pows = [harmonic_group[0][1] for harmonic_group in filtered_grouplist]

    decibel = 10.0 * np.log10(pows)  # calculate decibel using 1 as P0

    return np.array(fundamentals), decibel


def write_csv(filename, csv_header, data_matrix):

    # Check if header has same row length as data_matrix
    if len(csv_header) != len(data_matrix[0]):
        raise ValueError('The length of the header does not match the length of the data matrix!')

    with open(filename, 'wb') as fin:
        fin.write(str.encode(','.join(csv_header)))
        fin.write(str.encode('\n'))
        for row_id in range(len(data_matrix)):
            fin.write(str.encode(','.join(['%.3f' % e for e in data_matrix[row_id]])))  # convert to strings!
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
        filename = os.path.splitext(os.path.basename(sys.argv[1]))[0] + '-eodfs.csv'
        data, samplerate, unit = load_data(sys.argv[1], 0)

    # calculate powerspectrums with different frequency resolutions
    psd_data = multi_resolution_psd(data, samplerate, fresolution=[0.5, 2 * 0.5, 4 * 0.5])

    # find the fishes in the different powerspectra and extract fund. frequency and power
    fishlists = []
    for i in range(len(psd_data)):
        fishlist = harmonic_groups(psd_data[i][1], psd_data[i][0])[0]
        fishlists.append(fishlist)
    filtered_fishlist = consistent_fishes(fishlists)

    fund_freqs, db = extract_main_freqs_and_db(filtered_fishlist)
    csv_matrix = np.column_stack((fund_freqs, db))
    header = ['fundamental frequency', 'dB']
    write_csv(filename, header, csv_matrix)
    
    print('\ncsv_file created in %s' % filename)
