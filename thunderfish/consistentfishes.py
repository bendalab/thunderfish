"""
Functions to compare fishlists created by the harmonicgroups.py module in order to create a create a fishlist with only
these fishes available in all fishlists.

consistent_fishes(): Compares a list of fishlists and builds a consistent fishlist.
"""

import numpy as np
import harmonicgroups as hg


def find_consistency(fundamentals, df_th=1.):
    """
    Compares lists of floats to find these values consistent in every list. (with a certain threshold)

    Every value of the first list is compared to the values of the other lists. The consistency_help array consists in
    the beginning of ones, has the same length as the first list and is used to keep the information about values that
    are available in several lists. If the difference between a value of the first list and a value of another list is
    below the threshold the entry of the consistency_help array is added 1 to at the index of the value in the value in
    the first list. The indices of the consistency_help array that are equal to the amount of lists compared are the
    indices of the values of the first list that are consistent in all lists. The consistent value array and the indices
    are returned.


    :param fundamentals:    (2-D array) list of lists containing the fundamentals of a fishlist.
                            fundamentals = [ [f1, f1, ..., f1, f1], [f2, f2, ..., f2, f2], ..., [fn, fn, ..., fn, fn] ]
    :param df_th:           (float) Frequency threshold for the comparison of different fishlists. If the fundamental
                            frequencies of two fishes from different fishlists vary less than this threshold they are
                            assigned as the same fish.
    :return consistent_fundamentals: (1-D array) List containing all values that are available in all given lists.
    :return index:          (1-D array) Indices of the values that are in every list relating to the fist list in fishlists.
    """
    consistency_help = np.ones(len(fundamentals[0]), dtype=int)

    for enu, fundamental in enumerate(fundamentals[0]):
        for list in range(1, len(fundamentals)):
            if np.sum(np.abs(fundamentals[list] - fundamental) < df_th) > 0:
                consistency_help[enu] += 1

    index = np.arange(len(fundamentals[0]))[consistency_help == len(fundamentals)]
    consistent_fundamentals = fundamentals[0][index]

    return consistent_fundamentals, index


def consistent_fishes_plot(fishlists, filtered_fishlist, ax, fs):
    """
    Creates an axis for plotting all lists and the consistent values marked with a bar.

    :param filtered_fishlist: (3-D array) Contains power and frequency of these fishes that hve been detected in
                            several powerspectra using different resolutions.
    :param fishlists:       (4-D array or 3-D array) List of or single fishlists with harmonics and each frequency and power.
                            fishlists[fishlist][fish][harmonic][frequency, power]
                            fishlists[fish][harmonic][frequency, power]
    :param ax:              (axis for plot) empty axis that is filled with content in the function.
    :param fs:              (int) fontsize for the plot.
    """
    for list in np.arange(len(fishlists)):
        for fish in np.arange(len(fishlists[list])):
            ax.plot(list + 1, fishlists[list][fish][0][0], 'k.', markersize=10)

    for fish in np.arange(len(filtered_fishlist)):
        x = np.arange(len(fishlists)) + 1
        y = [filtered_fishlist[fish][0][0] for i in range(len(fishlists))]
        if fish == 0:
            ax.plot(x, y, '-r', linewidth=10, alpha=0.5, label='consistent in all lists')
        else:
            ax.plot(x, y, '-r', linewidth=10, alpha=0.5)
    ax.set_xlim([0, len(fishlists) + 1])
    ax.set_ylabel('value', fontsize=fs)
    ax.set_xlabel('list no.', fontsize=fs)


def consistent_fishes_psd_plot(filtered_fishlist, ax, max_freq= 3000):
    """
    Creates an axis for plotting the power of the four most powerful detected fishes with its fundamental and harmonics.

    The axis passed to this function should already contain the plotted power-spectrum in dB.

    :param filtered_fishlist: (3-D array) Contains power and frequency of these fishes that hve been detected in
                            several powerspectra using different resolutions.
    :param ax:              (axis for plot) empty axis that is filled with content in the function.
    """
    fundamental_power = [filtered_fishlist[fish][0][1] for fish in range(len(filtered_fishlist))]
    if len(filtered_fishlist) >= 4:
        idx_maxpower = np.argsort(fundamental_power)[-4:]
    else:
        idx_maxpower = np.argsort(fundamental_power)[:]

    for fish in idx_maxpower:
        x = np.array([filtered_fishlist[fish][harmonic][0] for harmonic in range(len(filtered_fishlist[fish]))])
        y = np.array([filtered_fishlist[fish][harmonic][1] for harmonic in range(len(filtered_fishlist[fish]))])
        y[y < 1e-20] = np.nan
        ax.plot(x[x < max_freq], 10.0 * np.log10(y[x < max_freq]), 'o', markersize=8, label='%.1f' % filtered_fishlist[fish][0][0])
        ax.legend(frameon=False, numpoints=1, bbox_to_anchor=(1.2, 1))


def consistent_fishes(fishlists, verbose=0, plot_data_func=None, **kwargs):
    """
    Compares several fishlists to create a fishlist only containing these fishes present in all these fishlists.

    Therefore several functions are used to first extract the fundamental frequencies of every fish in each fishlist,
    before comparing them and building a fishlist only containing these fishes present in all fishlists.

    :param fishlists:       (4-D array) List of fishlists with harmonics and each frequency and power.
                            fishlists[fishlist][fish][harmonic][frequency, power]
    :param plot_data_func:  (function) function (consistentfishesplot()) that is used to create a axis for later plotting containing a figure to
                            visualice what the modul did.
    :param verbose:         (int) when the value is 1 you get additional shell output.
    :param **kwargs:        additional arguments that are passed to the plot_data_func().
    :return filtered_fishlist:(3-D array) New fishlist with the same structure as a fishlist in fishlists only
                            containing these fishes that are available in every fishlist in fishlists.
                            fishlist[fish][harmonic][frequency, power]
    """
    if verbose >= 1:
        print('Finding consistent fishes out of %0.f fishlists ...' % len(fishlists))

    fundamentals = hg.extract_fundamental_freqs(fishlists)

    consistent_fundamentals, index = find_consistency(fundamentals)

    # creates a filtered fishlist only containing the data of the fishes consistent in several fishlists.
    filtered_fishlist = []
    for idx in index:
        filtered_fishlist.append(fishlists[0][idx])

    if plot_data_func:
        plot_data_func(fishlists, filtered_fishlist, **kwargs)

    return filtered_fishlist


if __name__ == '__main__':
    print('Creating one fishlist containing only the fishes that are consistant in several fishlists.')
    print('The input structur locks like this fishlists[list][fish][harmonic][frequency, power]')
    print('')
    print('Usage:')
    print('  python consistentfishes.py')
    print('')

    # example 4-D array containing of 4 fishlists all haveing 3 fishes with 1 harmonic with frequency and power
    fishlists = [[np.array([np.array([350, 0])]), np.array([np.array([700.2, 0])]), np.array([np.array([1000, 0])])],
                 [np.array([np.array([350.1, 0])]), np.array([np.array([699.8, 0])]), np.array([np.array([250.2, 0])])],
                 [np.array([np.array([349.7, 0])]), np.array([np.array([700.4, 0])]),
                  np.array([np.array([1000.2, 0])])],
                 [np.array([np.array([349.8, 0])]), np.array([np.array([700.5, 0])]),
                  np.array([np.array([1000.3, 0])])]]
    #
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    filtered_fishlist = consistent_fishes(fishlists, verbose=1, plot_data_func=consistent_fishes_plot, ax=ax, fs=12)
    plt.show()
