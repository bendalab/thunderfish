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
    # TODO no np.aranges in loops:
    # TODO Test it!
    # TODO: single and lists of fishlist hasattr(list, 'len')
    fundamentals = []
    for fishlist in range(len(fishlists)):
        fundamentals.append(np.array([fish[0][0] for fish in fishlists[fishlist]]))  # TODO: Docu np.array

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
    consistency_help = np.ones(len(fundamentals[0]), dtype=int)

    for enu, fundamental in enumerate(fundamentals[0]):
        for list in range(1, len(fundamentals)):
            if np.sum(np.abs(fundamentals[list] - fundamental) < df_th) > 0:
                consistency_help[enu] += 1

    index = np.arange(len(fundamentals[0]))[consistency_help == len(fundamentals)]
    consistent_fundamentals = fundamentals[0][consistency_help == len(fundamentals)]

    return consistent_fundamentals, index

def consistent_fishes_plot(fishlists, filtered_fishlist, ax, fs):
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
    ax.set_ylabel('value', fontsize=12)
    ax.set_xlabel('list no.', fontsize=12)

def consistent_fishes_psd_plot(filtered_fishlist, ax, fs):
    """
    This function can be passed to the consistent fishes plot as plot_data_func. It's purpose is to plot the four most
    powerfull fundamental frequencies into a powerspectrum plot.

    :param filtered_fishlist: (3-D array) Contains power and frequency of these fishes that hve been detected in
                            several powerspectra using different resolutions.
    :param ax:              (axis for plot) empty axis that is filled with content in the function.
    """
    fundamental_power = [filtered_fishlist[fish][0][1] for fish in np.arange(len(filtered_fishlist))]
    if len(filtered_fishlist) >=4:
        idx_maxpower = np.argsort(fundamental_power)[-4:]
    else:
        idx_maxpower = np.argsort(fundamental_power)[:]

    for fish in idx_maxpower:
        x = np.array([filtered_fishlist[fish][harmonic][0] for harmonic in range(len(filtered_fishlist[fish]))])
        y = np.array([filtered_fishlist[fish][harmonic][1] for harmonic in range(len(filtered_fishlist[fish]))])
        y[y < 1e-20] = np.nan
        ax.plot(x, 10.0 * np.log10(y), 'o', markersize=8, label='%.1f' % filtered_fishlist[fish][0][0])

    ax.legend(loc= 1, numpoints=1, frameon=False, fontsize= 10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()

def consistent_fishes(fishlists, verbose=0, plot_data_func = None, **kwargs):
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
    if verbose >= 1:
        print('Finding consistent fishes out of %0.f fishlists ...' % len(fishlists))

    fundamentals = extract_fundamentals(fishlists)

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
    # TODO numpy arrays
    fishlists = [ [[[350, 0]], [[700.2, 0]], [[1000, 0]]],
                  [[[350.1, 0]], [[699.8, 0]], [[250.2, 0]]],
                  [[[349.7, 0]], [[700.4, 0]], [[1000.2, 0]]],
                  [[[349.8, 0]], [[700.5, 0]], [[1000.3, 0]]]]

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    filtered_fishlist = consistent_fishes(fishlists, verbose= 1, plot_data_func=consistent_fishes_plot, ax =ax, fs=12)
    plt.show()
