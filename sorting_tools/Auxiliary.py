__author__ = 'juan'
# Imports
import numpy as np
import matplotlib.pyplot as plt
import sys
import seaborn as sns
from IPython import embed


def df_histogram(freqs_array):
    """ This Function takes an array of wave-fish fundamental frequencies and calculates all possible
    difference-frequencies between all combinations of EODFs.

    :rtype : array with all possible dfs
    :param freqs_array: array of fish fundamental frequencies.
    """
    all_diffs = np.hstack([freqs_array - e for e in freqs_array])
    ret = all_diffs[all_diffs != 0.]
    return ret


def plot_dfs_histogram(dfs_array, binwidth='FD'):
    """ Plots a histogram of the difference frequencies

    :param binwidth: select the size of the binwidth. use 'FD' for Freedman-Diaconis rule
    :param dfs_array: array-like. list of difference frequencies.
    """
    q75, q25 = np.percentile(abs(dfs_array), [75, 25])

    inch_factor = 2.54
    sns.set_context("poster")
    sns.axes_style('white')
    sns.set_style("ticks")
    fig, ax = plt.subplots(figsize=(15./inch_factor, 10./inch_factor))

    if binwidth == 'FD':
        ax.hist(dfs_array, bins=int(2*(q75-q25) * len(dfs_array)**(-1./3.)),
                facecolor='cornflowerblue', alpha=0.8)  # Freedman-Diaconis rule for binwidth
    else:
        ax.hist(dfs_array, bins=binwidth, color='cornflowerblue', alpha=0.8)

    # Plot Cosmetics

    ax.set_ylabel('Counts', fontsize=16)
    ax.set_xlabel('Possible Beat-Frequencies [Hz]', fontsize=14)
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.set_title('Distribution of Beat-Frequencies', fontsize=16)
    sns.despine(fig=fig, ax=ax, offset=10)
    fig.tight_layout()
    fig.savefig('figures/histo_of_dfs.pdf')
    plt.close()


if __name__ == '__main__':
    # Test the df_histogram function
    a = np.load('%s' % sys.argv[1])
    dfs = df_histogram(a)
    plot_dfs_histogram(dfs)