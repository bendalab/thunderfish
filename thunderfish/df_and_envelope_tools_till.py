import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from IPython import embed
import sys


def get_diff_freqs(freqs_array):
    """ This Function takes an array of fundamental frequencies and calculates all possible
    difference-frequencies between all combinations.
    :rtype : array with all possible dfs
    :param freqs_array: array of fish fundamental frequencies.
    """
    all_diffs = np.hstack([freqs_array - e for e in freqs_array])
    ret = all_diffs[all_diffs != 0.]
    return ret


def plot_freq_histogram(m1, m2, m3, m4, s1, s2, s3, s4, x_label, x_limits, binwidth, savefig=False, save_path_and_name='./histo_fig.pdf'):
    """ Plots a histogram of frequencies.
    :param freqs_array: array-like. list of frequencies.
    :param x_label: string. Sets the label of x-axis
    :param bins: float. select the number of bins. Exception: string 'FD' for Freedman-Diaconis rule (default)
    :param savefig: boolean. True if you want to save the figure.
    :param save_path_and_name: string. path + fig-name where you want to save the file. Default is './histo_fig.pdf'.
    Don't forget the '.pdf' at the end of the save_path_and_name string!!
    """

    inch_factor = 2.54
    sns.set_context("poster")
    sns.axes_style('white')
    sns.set_style("ticks")
    fig = plt.figure(figsize= (40./inch_factor, 40./inch_factor))

    ax1 = fig.add_subplot(4, 2, 1)
    ax1.hist(m1, bins=range(x_limits[0], x_limits[1]+100, binwidth), color='firebrick', alpha=0.8)
    ax1.set_xlim(x_limits)
    ax1.set_ylabel('n' , fontsize=18)
    ax1.set_xlabel(x_label + ' [Hz]', fontsize=18)
    ax1.tick_params(axis='both', which='major', labelsize=16)
    ax1.set_title('\nDay 1: large Group', fontsize=20)
    # sns.despine(fig=fig, ax=ax1, offset=10)
    fig.tight_layout()

    ax2 = fig.add_subplot(4, 2, 3)
    ax2.hist(m2, bins=range(x_limits[0], x_limits[1]+100, binwidth), color='firebrick', alpha=0.8)
    ax2.set_xlim(x_limits)
    ax2.set_ylabel('n' , fontsize=18)
    ax2.set_xlabel(x_label + ' [Hz]', fontsize=18)
    ax2.tick_params(axis='both', which='major', labelsize=16)
    ax2.set_title('\nDay 2: large Group', fontsize=20)
    # sns.despine(fig=fig, ax=ax1, offset=10)
    fig.tight_layout()

    ax3 = fig.add_subplot(4, 2, 5)
    ax3.hist(m3, bins=range(x_limits[0], x_limits[1]+100, binwidth), color='firebrick', alpha=0.8)
    ax3.set_xlim(x_limits)
    ax3.set_ylabel('n' , fontsize=18)
    ax3.set_xlabel(x_label + ' [Hz]', fontsize=18)
    ax3.tick_params(axis='both', which='major', labelsize=16)
    ax3.set_title('\nDay 3: large Group', fontsize=20)
    # sns.despine(fig=fig, ax=ax1, offset=10)
    fig.tight_layout()

    ax4 = fig.add_subplot(4, 2, 7)
    ax4.hist(m4, bins=range(x_limits[0], x_limits[1]+100, binwidth), color='firebrick', alpha=0.8)
    ax4.set_xlim(x_limits)
    ax4.set_ylabel('n' , fontsize=18)
    ax4.set_xlabel(x_label + ' [Hz]', fontsize=18)
    ax4.tick_params(axis='both', which='major', labelsize=16)
    ax4.set_title('\nDay 4: large Group', fontsize=20)
    # sns.despine(fig=fig, ax=ax1, offset=10)
    fig.tight_layout()

    ax5 = fig.add_subplot(4, 2, 2)
    ax5.hist(s1, bins=range(x_limits[0], x_limits[1]+100, binwidth), color='firebrick', alpha=0.8)
    ax5.set_xlim(x_limits)
    ax5.set_ylabel('n' , fontsize=18)
    ax5.set_xlabel(x_label + ' [Hz]', fontsize=18)
    ax5.tick_params(axis='both', which='major', labelsize=16)
    ax5.set_title('\nDay1: single', fontsize=20)
    # sns.despine(fig=fig, ax=ax1, offset=10)
    fig.tight_layout()

    ax6 = fig.add_subplot(4, 2, 4)
    ax6.hist(s2, bins=range(x_limits[0], x_limits[1]+100, binwidth), color='firebrick', alpha=0.8)
    ax6.set_xlim(x_limits)
    ax6.set_ylabel('n' , fontsize=18)
    ax6.set_xlabel(x_label + ' [Hz]', fontsize=18)
    ax6.tick_params(axis='both', which='major', labelsize=16)
    ax6.set_title('\nDay 2: single', fontsize=20)
    # sns.despine(fig=fig, ax=ax1, offset=10)
    fig.tight_layout()

    ax7 = fig.add_subplot(4, 2, 6)
    ax7.hist(s3, bins=range(x_limits[0], x_limits[1]+100, binwidth), color='firebrick', alpha=0.8)
    ax7.set_xlim(x_limits)
    ax7.set_ylabel('n' , fontsize=18)
    ax7.set_xlabel(x_label + ' [Hz]', fontsize=18)
    ax7.tick_params(axis='both', which='major', labelsize=16)
    ax7.set_title('\nDay 3: single', fontsize=20)
    # sns.despine(fig=fig, ax=ax1, offset=10)
    fig.tight_layout()

    ax8 = fig.add_subplot(4, 2, 8)
    ax8.hist(s4, bins=range(x_limits[0], x_limits[1]+100, binwidth), color='firebrick', alpha=0.8)
    ax8.set_xlim(x_limits)
    ax8.set_ylabel('n' , fontsize=18)
    ax8.set_xlabel(x_label + ' [Hz]', fontsize=18)
    ax8.tick_params(axis='both', which='major', labelsize=16)
    ax8.set_title('\nDay 4: single', fontsize=20)
    # sns.despine(fig=fig, ax=ax1, offset=10)
    fig.tight_layout()


    sns.despine(fig=fig, ax=[ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8])
    ## DESPINE !!!
    if savefig:
        fig.savefig(save_path_and_name)
        plt.close()

if __name__ == '__main__':

    # Test the plot_freq_histogram() function
    # file_path = sys.argv[1]
    # freqs = np.load(file_path)
    # freqs = np.unique(freqs)

    m1 = np.load('20140521_RioCanita/multi/fish_wave.npy')
    m1 = np.unique(m1)
    m2 = np.load('20140522_RioCanita/multi/fish_wave.npy')
    m1 = np.unique(m2)
    m3 = np.load('20140523_RioCanita/multi/fish_wave.npy')
    m1 = np.unique(m3)
    m4 = np.load('20140524_RioCanita/multi/fish_wave.npy')
    m1 = np.unique(m4)
    s1 = np.load('20140521_RioCanita/single/fish_wave.npy')
    m1 = np.unique(s1)
    s2 = np.load('20140522_RioCanita/single/fish_wave.npy')
    m1 = np.unique(s2)
    s3 = np.load('20140523_RioCanita/single/fish_wave.npy')
    m1 = np.unique(s3)
    s4 = np.load('20140524_RioCanita/single/fish_wave.npy')
    m1 = np.unique(s4)

    df_m1 = get_diff_freqs(m1)
    df_m1 = np.unique(df_m1)

    df_m2 = get_diff_freqs(m2)
    df_m2 = np.unique(df_m2)

    df_m3 = get_diff_freqs(m3)
    df_m3 = np.unique(df_m3)

    df_m4 = get_diff_freqs(m4)
    df_m4 = np.unique(df_m4)

    df_s1 = get_diff_freqs(s1)
    df_s1 = np.unique(df_s1)

    df_s2 = get_diff_freqs(s2)
    df_s2 = np.unique(df_s2)

    df_s3 = get_diff_freqs(s3)
    df_s3 = np.unique(df_s3)

    df_s4 = get_diff_freqs(s4)
    df_s4 = np.unique(df_s4)

    # envelopes = get_diff_freqs(dfs)
    # envelopes = np.unique(envelopes)

    # bins=range(min(data), max(data) + binwidth, binwidth)

    plot_freq_histogram(m1, m2, m3, m4, s1, s2, s3, s4, 'Fundamental EOD-Frequencies', binwidth= 10, x_limits=[0, 1000])
    plot_freq_histogram(df_m1,df_m2,df_m3,df_m4,df_s1,df_s2,df_s3,df_s4, 'Beat Frequencies', binwidth= 20, x_limits=[-1000, 1000])
    # plot_freq_histogram(envelopes, 'Envelope Frequencies', binwidth= 20, x_limits=[-1000, 1000])

    plt.show()