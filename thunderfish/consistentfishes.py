import numpy as np

def extract_fundamentals(fishlists):
    """
    Extracts the fundamental frequencies of multiple fishlists created by the harmonicgroups modul.

    This function gets a 4-D array as input. This input consists of multiple fishlists from the harmonicgroups modul
    lists up (fishlists[list][fish][harmonic][frequency, power]). The amount of lists doesn't matter. With a for-loop
    this function collects all fundamental frequencies of every fishlist. In the end the output is a 2-D array
    containing the fundamentals of each list (fundamentals[list][fundamental_frequencies]).

    :param fishlists: (4-D array)
    :return fundamentals: (2-D array)
    """
    fundamentals = [[] for i in np.arange(len(fishlists))]
    for fishlist in np.arange(len(fishlists)):
        for fish in np.arange(len(fishlists[fishlist])):
            fundamentals[fishlist].append(fishlists[fishlist][fish][0][0])

    return fundamentals

def find_consistency(fundamentals):
    """
    Compares several lists to find values that are, with a certain threshhold, available in every list.

    This function gets a list of lists conatining values as input. For every value in the first list this function is
    looking in the other lists if there is a entry with the same value +- threshold. The vector consistancy_help
    has the length of the first list of fundamentals and it's entries are all ones. When a value from the first list is
    also available in another list this function adds 1 to that particular entry in the consostany_help vector. After
    comparina all values from the first list with all values of the other lists the consostancy_help vector got values
    between 1 and len(fundamentals). The indices where this vector is equal to len(fundamentals) are also the indices of
    fishes in the first fishlist that are available in every fishlist.


    :param fundamentals: (2-D array)
    :return consistent_fundamentals: (1-D array)
    :return index: (1-D array)
    """
    consistancy_help = [1 for i in np.arange(len(fundamentals[0]))]
    index = []
    consistent_fundamentals = []

    for enu, fundamental in enumerate(fundamentals[0]):
        for list in np.arange(len(fundamentals)-1)+1:
            freq_diff = [fundamentals[list][i] - fundamental for i in np.arange(len(fundamentals[list]))]
            for i in np.arange(len(freq_diff)):
                if freq_diff[i] >= -1.0 and freq_diff[i] <= 1:
                    consistancy_help[enu] += 1
                    break

    for idx in np.arange(len(consistancy_help)):
        if consistancy_help[idx] == len(fundamentals):
            index.append(idx)
            consistent_fundamentals.append(fundamentals[0][idx])

    return consistent_fundamentals, index

def consistent_fishlist(index, fishlists):
    """
    This function gets the 4-D array of fishlists and the indices of the fishes from the first fishlist that are also
    available in all other fishlists. It gives back a 3-D array (filterd_fishlist) that only contains the information of
    the fishes that are in all lists (structur: filtered_fishlist[fish][harmonic][frequency, power])

    :param index: (1-D array)
    :param fishlists: (4-D array)
    :return filtered_fishlist: (3-D array)
    """
    filtered_fishlist = []
    for idx in index:
        filtered_fishlist.append(fishlists[0][idx])

    return filtered_fishlist

def consistentfishes_main(fishlists):
    """
    This function gets several fishlists, compares them, and gives back one fishlist that only contains these fishes
    that are available in every given fishlist. This is the main function that calls the other functions in the code.

    :param fishlists:
    :return filtered_fishlist:
    """
    print('comparing different lists ...')

    fundamentals = extract_fundamentals(fishlists)
    consistant_fundamentals, index = find_consistency(fundamentals)
    filtered_fishlist = consistent_fishlist(index, fishlists)

    print('Done !')
    print('')

    return filtered_fishlist

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import sys

    print('Creating one fishlist containing only the fishes that are consistant in several fishlists.')
    print('The input structur locks like this fishlists[list][fish][harmonic][frequency, power]')
    print('')
    print('Usage:')
    print('  python consistentfishes.py [-p]')
    print('  -p: plot data')
    print('')

    plot = False
    if len(sys.argv) > 1 and sys.argv[1] == '-p':
        plot = True

    # example 4-D array containing of 4 fishlists all haveing 3 fishes with 1 harmonic with frequency and power
    fishlists = [ [[[350, 0]], [[700.2, 0]], [[1000, 0]]],
                  [[[350.1, 0]], [[699.8, 0]], [[250.2, 0]]],
                  [[[349.7, 0]], [[700.4, 0]], [[1000.2, 0]]],
                  [[[349.8, 0]], [[700.5, 0]], [[1000.3, 0]]]]
    filtered_fishlist = consistentfishes_main(fishlists)

    if plot:
        print('plotting example data ...')
        fig, ax = plt.subplots()
        for list in np.arange(len(fishlists)):
            for fish in np.arange(len(fishlists[list])):
                ax.plot(list+1, fishlists[list][fish][0][0], 'k.', markersize= 10)

        for fish in np.arange(len(filtered_fishlist)):
            if fish == 0:
                ax.plot(np.arange(len(fishlists))+1, [filtered_fishlist[fish][0][0] for i in np.arange(len(fishlists))],
                        '-r', linewidth= 10, alpha=0.5, label='consistent in all lists')
            else:
                ax.plot(np.arange(len(fishlists))+1, [filtered_fishlist[fish][0][0] for i in np.arange(len(fishlists))],
                        '-r', linewidth= 10, alpha=0.5)
        ax.set_xlim([0, len(fishlists)+1])
        ax.set_ylabel('value')
        ax.set_xlabel('list no.')
        plt.legend(loc='upper right', fontsize= 12)
        plt.show()
