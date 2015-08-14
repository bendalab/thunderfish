__author__ = 'raab'
import sys
import numpy as np
import matplotlib.pyplot as plt

def load_npy_to_list():
    """
    loads an .npy data (numpy.array) and converts it to a list.

    :return: data (list of frequencies)
    """
    print 'loading data ...'
    data = np.load('%s.npy' %sys.argv[1])
    data = data.tolist()
    print 'data loaded successfully'
    return data

def creat_histo(data):
    """
    gets a list of data and creates an histogram of this data.

    :param data:
    """
    print 'creating histogramm ...'
    hist, bins = np.histogram(data, bins= len(data)//4)
    width = 0.7 * (bins[1] - bins[0])
    center = (bins[:-1] + bins[1:]) / 2

    fig, ax = plt.subplots(facecolor='white')
    ax.bar(center, hist, align='center', width=width)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.tick_params(axis='both', direction='out')
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    ax.set_ylabel('counts')
    ax.set_xlabel('frequency')
    plt.xticks(np.arange(0, max(data)+100, 250))
    plt.title('Histogram')
    plt.show()

def main():
    data = load_npy_to_list()

    creat_histo(data)

    print 'code finished'

if __name__ == '__main__':
    main()