import numpy as np

def extract_fundamentals(fishlists):
    fundamentals = [[] for i in np.arange(len(fishlists))]
    for fishlist in np.arange(len(fishlists)):
        for fish in np.arange(len(fishlists[fishlist])):
            fundamentals[fishlist].append(fishlists[fishlist][fish][0][0])

    return fundamentals

def find_consistency(fundamentals):
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
    filtered_fishlist = []
    for idx in index:
        filtered_fishlist.append(fishlists[0][idx])

    return filtered_fishlist

def consistentfishes_main(fishlists):
    print('comparing different fishlists ...')

    fundamentals = extract_fundamentals(fishlists)
    consistant_fundamentals, index = find_consistency(fundamentals)
    filtered_fishlist = consistent_fishlist(index, fishlists)

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

    fishlists = [ [[[350, 0]], [[700.2, 0]], [[1000, 0]]],
                  [[[350.1, 0]], [[699.8, 0]], [[250.2, 0]]],
                  [[[349.7, 0]], [[700.4, 0]], [[1000.2, 0]]] ]
    filtered_fishlist = consistentfishes_main(fishlists)

    if len(sys.argv) > 1 and sys.argv[1] == '-p':
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
        ax.set_xlim([0, 4])
        ax.set_ylabel('value')
        ax.set_xlabel('list no.')
        plt.legend(loc='upper right', fontsize= 12)
        plt.show()


    # print filtered_fishlist