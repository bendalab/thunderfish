import numpy as np

def extract_fundamentals(fishlists):
    fundamentals = [[] for i in np.arange(len(fishlists))]
    for fishlist in np.arange(len(fishlists)):
        for fish in np.arange(len(fishlists[fishlist])):
            fundamentals[fishlist].append(fishlists[fishlist][fish][0][0])
    return fundamentals

def find_consistency(fundamentals):
    consistent_fundamentals = []
    index = []
    return consistent_fundamentals, index

def consistent_fishlist(index, fishlists):
    return consistent_fishlist

def consistentfishes_main(fishlists):
    print('comparing different fishlists ...')
    fundamentals = extract_fundamentals(fishlists)
    consistant_fundamentals, index = find_consistency(fundamentals)
    filtered_fishlist = consistent_fishlist(index, fishlists)
    return filtered_fishlist

if __name__ == '__main__':

    fishlists = []
    filtered_fishlist = consistentfishes_main(fishlists)