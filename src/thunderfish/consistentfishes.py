"""
Compare fishlists created by the harmonics module in order to create a fishlist with
fishes present in all fishlists.

- `consistent_fishes()`: Compares a list of fishlists and builds a consistent fishlist.
- `plot_consistent_fishes()`: Visualize the algorithm of consisten_fishes().
"""

import numpy as np
from .harmonics import fundamental_freqs


def find_consistency(fundamentals, df_th=1.0):
    """
    Compares lists of floats to find these values consistent in every list.
    (with a certain threshold)

    Every value of the first list is compared to the values of the other
    lists. The consistency_help array consists in the beginning of ones,
    has the same length as the first list and is used to keep the
    information about values that are available in several lists. If the
    difference between a value of the first list and a value of another
    list is below the threshold the entry of the consistency_help array
    is added 1 to at the index of the value in the value in the first
    list. The indices of the consistency_help array that are equal to
    the amount of lists compared are the indices of the values of the
    first list that are consistent in all lists. The consistent value
    array and the indices are returned.


    Parameters
    ----------
    fundamentals: 2-D array
        List of lists containing the fundamentals of a fishlist.
        fundamentals = [ [f1, f1, ..., f1, f1], [f2, f2, ..., f2, f2], ..., [fn, fn, ..., fn, fn] ]
    df_th: float
        Frequency threshold for the comparison of different fishlists in Hertz. If the fundamental
        frequencies of two fishes from different fishlists vary less than this threshold they are
        assigned as the same fish.

    Returns
    -------
    consistent_fundamentals: 1-D array
        List containing all values that are available in all given lists.
    index: 1-D array
        Indices of the values that are in every list relating to the fist list in fishlists.
    """
    consistency_help = np.ones(len(fundamentals[0]), dtype=int)

    for enu, fundamental in enumerate(fundamentals[0]):
        for list in range(1, len(fundamentals)):
            if np.sum(np.abs(fundamentals[list] - fundamental) < df_th) > 0:
                consistency_help[enu] += 1

    index = np.arange(len(fundamentals[0]))[consistency_help == len(fundamentals)]
    consistent_fundamentals = fundamentals[0][index]

    return consistent_fundamentals, index


def consistent_fishes(fishlists, verbose=0, plot_data_func=None, df_th=1.0, **kwargs):
    """
    Compares several fishlists to create a fishlist only containing these fishes present in all these fishlists.

    Therefore several functions are used to first extract the fundamental frequencies of every fish in each fishlist,
    before comparing them and building a fishlist only containing these fishes present in all fishlists.

    Parameters
    ----------
    fishlists: list of list of 2D array
        List of fishlists with harmonics and each frequency and power.
        fishlists[fishlist][fish][harmonic][frequency, power]
    plot_data_func:
        Function that visualizes what consistent_fishes() did.
    verbose: int
        When the value larger than one you get additional shell output.
    df_th: float
        Frequency threshold for the comparison of different fishlists in Hertz. If the fundamental
        frequencies of two fishes from different fishlists vary less than this threshold they are
        assigned as the same fish.
    **kwargs: dict
        Passed on to plot function.

    Returns
    -------
    filtered_fishlist: list of 2-D arrays
        New fishlist with the same structure as a fishlist in fishlists only
        containing these fishes that are available in every fishlist in fishlists.
        fishlist[fish][harmonic][frequency, power]
    """
    if verbose >= 1:
        print('Finding consistent fishes out of %d fishlists ...' % len(fishlists))

    fundamentals = fundamental_freqs(fishlists)
    if len(fundamentals) == 0:
        return []

    consistent_fundamentals, index = find_consistency(fundamentals)

    # creates a filtered fishlist only containing the data of the fishes consistent in several fishlists.
    filtered_fishlist = []
    for idx in index:
        filtered_fishlist.append(fishlists[0][idx])

    if plot_data_func:
        plot_data_func(fishlists, filtered_fishlist, **kwargs)

    return filtered_fishlist


def plot_consistent_fishes(fishlists, filtered_fishlist, ax, fs):
    """
    Creates an axis for plotting all lists and the consistent values marked with a bar.

    Parameters
    ----------
    filtered_fishlist: 3-D array
        Contains power and frequency of these fishes that hve been detected in
        several powerspectra using different resolutions.
    fishlists: 4-D array or 3-D array
        List of or single fishlists with harmonics and each frequency and power.
        fishlists[fishlist][fish][harmonic][frequency, power]
        fishlists[fish][harmonic][frequency, power]
    ax: axis for plot
        Empty axis that is filled with content in the function.
    fs: int
        Fontsize for the plot.
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


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    
    print('Creating one fishlist containing only the fishes that are consistant in several fishlists.')
    print('The input structure looks like this:')
    print('  fishlists[list][fish][harmonic][frequency, power]')
    print('')
    print('Usage:')
    print('  python -m thunderfish.consistentfishes')
    print('')

    # example 4-D array containing of 4 fishlists all haveing 3 fishes with 1 harmonic with frequency and power
    fishlists = [[np.array([np.array([350, 0])]), np.array([np.array([700.2, 0])]), np.array([np.array([1050, 0])])],
                 [np.array([np.array([350.1, 0])]), np.array([np.array([699.8, 0])]), np.array([np.array([250.2, 0])])],
                 [np.array([np.array([349.7, 0])]), np.array([np.array([700.4, 0])]),
                  np.array([np.array([1050.2, 0])])],
                 [np.array([np.array([349.8, 0])]), np.array([np.array([700.5, 0])]),
                  np.array([np.array([1050.3, 0])])]]
    #
    
    fig, ax = plt.subplots()
    filtered_fishlist = consistent_fishes(fishlists, verbose=1, plot_data_func=plot_consistent_fishes, ax=ax, fs=12)
    plt.show()

    # check almost empty fishlist:
    fishlists = [[], [np.array([[349.8, 0], [700.5, 0], [1050.3, 0]])], []]
    filtered_fishlist = consistent_fishes(fishlists, verbose=1)
    print(filtered_fishlist)

    # check empty fishlist:
    fishlists = [[], []]
    filtered_fishlist = consistent_fishes(fishlists, verbose=1)
    print(filtered_fishlist)

    # check really empty fishlist:
    fishlists = []
    filtered_fishlist = consistent_fishes(fishlists, verbose=1)
    print(filtered_fishlist)
    
