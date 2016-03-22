import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from IPython import embed


def get_diff_freqs(freqs_array):
    """ This Function takes an array of fundamental frequencies and calculates all possible
    difference-frequencies between all combinations.

    :rtype : array with all possible dfs
    :param freqs_array: array of fish fundamental frequencies.
    """
    all_diffs = np.hstack([freqs_array - e for e in freqs_array])
    ret = all_diffs[all_diffs != 0.]
    return ret


def plot_freq_histogram(freqs_array, x_label, bins='FD', savefig=False, save_path_and_name='./histo_fig.pdf'):
    """ Plots a histogram of frequencies.

    :param freqs_array: array-like. list of frequencies.
    :param x_label: string. Sets the label of x-axis
    :param bins: float. select the number of bins. Exception: string 'FD' for Freedman-Diaconis rule (default)
    :param savefig: boolean. True if you want to save the figure.
    :param save_path_and_name: string. path + fig-name where you want to save the file. Default is './histo_fig.pdf'.
    Don't forget the '.pdf' at the end of the save_path_and_name string!!
    """
    q75, q25 = np.percentile(abs(freqs_array), [75, 25])

    inch_factor = 2.54
    sns.set_context("poster")
    sns.axes_style('white')
    sns.set_style("ticks")
    fig, ax = plt.subplots(figsize=(20./inch_factor, 15./inch_factor))

    if bins == 'FD':
        ax.hist(freqs_array, bins=int(2 * (q75 - q25) * len(freqs_array) ** (-1. / 3.)),
                facecolor='cornflowerblue', alpha=0.8, lw=1.5)  # Freedman-Diaconis rule for binwidth
    else:
        ax.hist(freqs_array, bins=bins, color='cornflowerblue', alpha=0.8)

    # Plot Cosmetics

    ax.set_ylabel('Counts', fontsize=16)
    ax.set_xlabel(x_label + ' [Hz]', fontsize=14)
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.set_title('Distribution of %s' % x_label, fontsize=16)
    sns.despine(fig=fig, ax=ax, offset=10)
    fig.tight_layout()
    if savefig:
        fig.savefig(save_path_and_name)
        plt.close()

if __name__ == '__main__':

    # Test the plot_freq_histogram() function
    file_path = 'data/recordings_cano_rubiano_RAW/fish_wave.npy'
    freqs = np.load(file_path)
    freqs = np.unique(freqs)

    dfs = get_diff_freqs(freqs)
    dfs = np.unique(dfs)

    envelopes = get_diff_freqs(dfs)
    envelopes = np.unique(envelopes)

    plot_freq_histogram(freqs, 'Fundamental EOD-Frequencies')
    plot_freq_histogram(dfs, 'Beat Frequencies', bins=2000)
    plot_freq_histogram(envelopes, 'Envelope Frequencies', bins=500)

    plt.show()