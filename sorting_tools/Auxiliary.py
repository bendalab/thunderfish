__author__ = 'juan'
# Imports
import numpy as np
import matplotlib.pyplot as plt
import sys
from IPython import embed


def df_histogram(freqs_ls):
    """ This Function takes a list of wave-fish fundamental frequencies and calculates all possible
    difference-frequencies between all combinations of EODFs.

    :param freqs_ls: array of fish fundamental frequencies.
    """
    all_diffs = np.hstack([freqs_ls - e for e in freqs_ls])
    ret = all_diffs[all_diffs != 0.]
    return ret


def plot_histogram(dfs_ls, binwidth='FD'):
    """ Plots a histogram of the difference frequencies

    :param binwidth: select the size of the binwidth. use 'FD' for Freedman-Diaconis rule
    :param dfs_ls: array-like. list of difference frequencies.
    """
    q75, q25 = np.percentile(abs(dfs_ls), [75, 25])
    fig, ax = plt.subplots()
    if binwidth == 'FD':
        ax.hist(dfs, bins=int(2*(q75-q25) * len(dfs)**(-1./3.)))  # Freedman-Diaconis rule for binwidth
    else:
        ax.hist(dfs, bins=binwidth)

    plt.show()


if __name__ == '__main__':
    # Test the df_histogram function
    a = np.load('%s' % sys.argv[1])
    dfs = df_histogram(a)
    plot_histogram(dfs)