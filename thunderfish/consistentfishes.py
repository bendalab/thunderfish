import numpy as np

def extract_fundamentals(fishlists):
    """
    Extracts the fundamental frequencies of multiple fishlists created by the harmonicgroups modul.

    This function gets a 4-D array as input. This input consists of multiple fishlists from the harmonicgroups modul
    lists up (fishlists[list][fish][harmonic][frequency, power]). The amount of lists doesn't matter. With a for-loop
    this function collects all fundamental frequencies of every fishlist. In the end the output is a 2-D array
    containing the fundamentals of each list (fundamentals[list][fundamental_frequencies]).

    :param fishlists:       (4-D array) List of fishlists with harmonics and each frequency and power.
                            fishlists[fishlist][fish][harmonic][frequency, power]
    :return fundamentals:   (2-D array) list of lists containing the fundamentals of a fishlist.
                            fundamentals = [ [f1, f1, ..., f1, f1], [f2, f2, ..., f2, f2], ..., [fn, fn, ..., fn, fn] ]
    """
    fundamentals = [[] for i in range(len(fishlists))]
    for fishlist in np.arange(len(fishlists)):
        for fish in np.arange(len(fishlists[fishlist])):
            fundamentals[fishlist].append(fishlists[fishlist][fish][0][0])

    return fundamentals

def find_consistency(fundamentals, df_th = 1.):
    """
    Compares several lists to find values that are, with a certain threshhold, available in every list.

    This function gets a list of lists conatining values as input. For every value in the first list this function is
    looking in the other lists if there is a entry with the same value +- threshold. The vector consistancy_help
    has the length of the first list of fundamentals and it's entries are all ones. When a value from the first list is
    also available in another list this function adds 1 to that particular entry in the consostany_help vector. After
    comparina all values from the first list with all values of the other lists the consostancy_help vector got values
    between 1 and len(fundamentals). The indices where this vector is equal to len(fundamentals) are also the indices of
    fishes in the first fishlist that are available in every fishlist.


    :param fundamentals:    (2-D array) list of lists containing the fundamentals of a fishlist.
                            fundamentals = [ [f1, f1, ..., f1, f1], [f2, f2, ..., f2, f2], ..., [fn, fn, ..., fn, fn] ]
    :param df_th:           (float) Frequency threshold for the comparison of different fishlists. If the fundamental
                            frequencies of two fishes from different fishlists vary less than this threshold they are
                            assigned as the same fish.
    :return consistent_fundamentals: (1-D array) List containing all values that are available in all given lists.
    :return index:          (1-D array) Indices of the values that are in every list relating to the fist list in fishlists.
    """
    consistency_help = np.ones_like(fundamentals[0])

    index = []
    consistent_fundamentals = []

    for enu, fundamental in enumerate(fundamentals[0]):
        for list in np.arange(len(fundamentals)-1)+1:
            freq_diff = [fundamentals[list][i] - fundamental for i in np.arange(len(fundamentals[list]))]
            for i in np.arange(len(freq_diff)):
                if freq_diff[i] >= -df_th and freq_diff[i] <= df_th:
                    consistency_help[enu] += 1
                    break

    for idx in np.arange(len(consistency_help)):
        if consistency_help[idx] == len(fundamentals):
            index.append(idx)
            consistent_fundamentals.append(fundamentals[0][idx])

    return consistent_fundamentals, index

def consistent_fishes_plot(filtered_fishlist, ax):
    """
    Creates an axis for plotting to visualize what this modul did.

    :param filtered_fishlist: (3-D array) Contains power and frequency of these fishes that hve been detected in
                            several powerspectra using different resolutions.
    :param ax:              (axis for plot) empty axis that is filled with content in the function.
    """
    for list in np.arange(len(fishlists)):
        for fish in np.arange(len(fishlists[list])):
            ax.plot(list+1, fishlists[list][fish][0][0], 'k.', markersize= 10)

    for fish in np.arange(len(filtered_fishlist)):
        x = np.arange(len(fishlists))+1
        y = [filtered_fishlist[fish][0][0] for i in np.arange(len(fishlists))]
        if fish == 0:
            ax.plot(x, y, '-r', linewidth= 10, alpha=0.5, label='consistent in all lists')
        else:
            ax.plot(x, y, '-r', linewidth= 10, alpha=0.5)
    ax.set_xlim([0, len(fishlists)+1])
    ax.set_ylabel('value')
    ax.set_xlabel('list no.')

def consistentfishes(fishlists, plot_data_func=None, **kwargs):
    """
    This function gets several fishlists, compares them, and gives back one fishlist that only contains these fishes
    that are available in every given fishlist. This is the main function that calls the other functions in the code.

    :param fishlists:       (4-D array) List of fishlists with harmonics and each frequency and power.
                            fishlists[fishlist][fish][harmonic][frequency, power]
    :param plot_data_func:  (function) function (consistentfishesplot()) that is used to create a axis for later plotting containing a figure to
                            visualice what the modul did.
    :param **kwargs:        additional arguments that are passed to the plot_data_func().
    :return filtered_fishlist:(3-D array) New fishlist with the same structure as a fishlist in fishlists only
                            containing these fishes that are available in every fishlist in fishlists.
                            fishlist[fish][harmonic][frequency, power]
    """
    print('\nFinding consistent fishes out of %0.f fishlists ...' % len(fishlists))

    fundamentals = extract_fundamentals(fishlists)
    consistant_fundamentals, index = find_consistency(fundamentals)

    # creates a filtered fishlist only containing the data of the fishes consistent in several fishlists.
    filtered_fishlist = []
    for idx in index:
        filtered_fishlist.append(fishlists[0][idx])

    if plot_data_func:
        plot_data_func(filtered_fishlist, **kwargs)

    return filtered_fishlist

if __name__ == '__main__':
    print('Creating one fishlist containing only the fishes that are consistant in several fishlists.')
    print('The input structur locks like this fishlists[list][fish][harmonic][frequency, power]')
    print('')
    print('Usage:')
    print('  python consistentfishes.py')
    print('')

    # example 4-D array containing of 4 fishlists all haveing 3 fishes with 1 harmonic with frequency and power
    fishlists = [ [[[350, 0]], [[700.2, 0]], [[1000, 0]]],
                  [[[350.1, 0]], [[699.8, 0]], [[250.2, 0]]],
                  [[[349.7, 0]], [[700.4, 0]], [[1000.2, 0]]],
                  [[[349.8, 0]], [[700.5, 0]], [[1000.3, 0]]]]

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    filtered_fishlist = consistentfishes(fishlists, plot_data_func=consistent_fishes_plot, ax=ax)
    plt.show()
