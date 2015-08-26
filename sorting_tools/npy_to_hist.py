__author__ = 'raab'
import sys
import glob
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from IPython import embed

def load_data(glob_ls):
    """
    loads all .npy data (numpy.array) and converts it to a dictionary using part from filenames as keys.

    :param glob_ls: list of .npy files
    :return: data (list of frequencies)
    """
    print 'loading data ...'
    # load data as a dictionary using either pulsefish or wavefish as key and the np.array as value.
    data = {curr_file.split('_')[-1].split('.npy')[0]: np.load(curr_file) for curr_file in glob_ls}
    print 'data loaded successfully'
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
    fig, ax = plt.subplots(figsize=(20./inch_factor, 15./inch_factor))
    colors = ['salmon', 'cornflowerblue']

    for enu, curr_fishtype in enumerate(data.keys()):
        hist, bins = np.histogram(data[curr_fishtype], bins=len(data[curr_fishtype])//4)
        width = 0.7 * (bins[1] - bins[0])
        center = (bins[:-1] + bins[1:]) / 2
        ax.bar(center, hist, align='center', width=width, alpha=0.8, facecolor=colors[enu], label=curr_fishtype)

    ax.set_ylabel('Counts', fontsize=16)
    ax.set_xlabel('Frequency [Hz]', fontsize=16)
    ax.set_xticks(np.arange(0, max(np.hstack(data.values()))+100, 250))
    ax.tick_params(axis='both', which='major')
    ax.set_title('Distribution of EOD-Frequencies')
    ax.legend(frameon=False, loc='best', fontsize=14)
    sns.despine(fig=fig, ax=ax, offset=10)
    fig.tight_layout()
    fig.savefig('histo_of_eod_freqs.pdf')
    plt.close()

def main():
    npy_files = glob.glob('*.npy')
    data = load_data(npy_files)

    create_histo(data)

    print 'code finished'

if __name__ == '__main__':
    main()