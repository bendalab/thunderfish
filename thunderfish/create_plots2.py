__author__ = 'raab'
from Auxiliary import *
import sys
import glob
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os


def load_data(glob_ls):
    """
    loads all .npy data (numpy.array) and converts it to a dictionary using part from filenames as keys.

    :param glob_ls: list of .npy files
    :return: data (list of frequencies)
    """
    print '\nloading data ...'
    # load data as a dictionary using either pulsefish or wavefish as key and the np.array as value.
    data = {curr_file.split('_')[-1].split('.npy')[0]: np.load(curr_file) for curr_file in glob_ls}
    print 'data loaded successfully\n'
    return data


def create_histo(data):
    """
    gets a list of data and creates an histogram of this data.

    :param data: dictionary with fishtype as keys and np.array with EOD-Frequencies as values.
    """
    print 'creating histogramm ...'

    inch_factor = 2.54
    sns.set_context("poster")
    sns.axes_style('white')
    sns.set_style("ticks")
    fig, ax = plt.subplots(figsize=(15. / inch_factor, 10. / inch_factor))
    colors = ['salmon', 'cornflowerblue']

    for enu, curr_fishtype in enumerate(data.keys()):
        if len(data[curr_fishtype]) >= 4:
            hist, bins = np.histogram(data[curr_fishtype], bins=len(data[curr_fishtype]) // 4)
            width = 0.7 * (bins[1] - bins[0])
            center = (bins[:-1] + bins[1:]) / 2
            ax.bar(center, hist, align='center', width=width, alpha=0.8, facecolor=colors[enu], label=curr_fishtype)

    ax.set_ylabel('Counts', fontsize=14)
    ax.set_xlabel('Frequency [Hz]', fontsize=14)
    ax.set_xticks(np.arange(0, max(np.hstack(data.values())) + 100, 250))
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.set_title('Distribution of EOD-Frequencies', fontsize=16)
    ax.legend(frameon=False, loc='best', fontsize=12)
    sns.despine(fig=fig, ax=ax, offset=10)
    fig.tight_layout()
    fig.savefig('figures/histo_of_eod_freqs.pdf')
    plt.close()


def fishtype_barplot(data):
    """ This function creates a bar plot showing the distribution of wave-fishes vs. pulse-fishes.

    :param data: dictionary with fish-type as keys and array of EODfs as values.
    """

    # Read the keys of the dictionary and use them to get the count of pulse- and wave-type fishes.
    keys = np.array(data.keys())
    bool_wave = np.array(['wave' in e for e in keys], dtype=bool)
    bool_pulse = np.array(['puls' in e for e in keys], dtype=bool)

    count_wave = len(data[keys[bool_wave][0]])
    count_pulse = len(data[keys[bool_pulse][0]])

    inch_factor = 2.54
    sns.set_context("poster")
    sns.axes_style('white')
    sns.set_style("ticks")
    fig, ax = plt.subplots(figsize=(10. / inch_factor, 10. / inch_factor))
    width = 0.5
    ax.bar(1 - width / 2., count_wave, width=width, facecolor='cornflowerblue', alpha=0.8)
    ax.bar(2 - width / 2., count_pulse, width=width, facecolor='salmon', alpha=0.8)
    ax.set_xticks([1, 2])
    ax.set_xticklabels(['Wave-type', 'Pulse-type'])
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.set_ylabel('Number of Fishes', fontsize=14)
    ax.set_title('Distribution of Fish-types', fontsize=16)
    sns.despine(fig=fig, ax=ax, offset=10)
    fig.tight_layout()
    fig.savefig('figures/fishtype_barplot.pdf')
    plt.close()


def main():
    # Load data
    wave_file = sys.argv[1]

    # npy_files = glob.glob('*.npy')
    # data = load_data(npy_files)

    # create histogram of EOD-frequencies
    create_histo(data)

    # create histogram of all possible beat-frequencies
    wave_file = ('fish_wave.npy')
    wave_freqs = np.load(wave_file[0])
    dfs = df_histogram(wave_freqs)
    plot_dfs_histogram(dfs)

    # Plot barplot with fish-type distribution
    fishtype_barplot(data)

    print 'code finished'
    # if len(sys.argv) == 3:
    #     response = raw_input('Do you want to create a .pdf file with the data and the figures processed ? [y/n]')
    #     if response == 'y':
    #         os.system('python create_tex.py %s.npy %s.npy' % (sys.argv[1], sys.argv[2]))


if __name__ == '__main__':
    main()
