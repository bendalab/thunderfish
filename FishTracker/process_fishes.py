import sys
import numpy as np
import matplotlib.pyplot as plt
import pickle


def plot_fundamentals(fishes, ax):
    for fish in range(len(fishes)):
        ax.plot((np.arange(len(fishes[fish])) * 1.024) / 60., fishes[fish], '.k', markersize=2)
    ax.set_xlabel('time [min]')
    ax.set_ylabel('frequency [Hz]')

def clear_fishes(fishes):
    clean_fishes = []

    for fish in range(len(fishes)):
        if len(fishes[fish][~np.isnan(fishes[fish])]) >= 50:
            tmp_fish = fishes[fish][~np.isnan(fishes[fish])]

            clean_fishes.append(np.asarray(tmp_fish))

    return clean_fishes

def combine_fishes(all_fishes):
    # all_fishes[snippet][fish array]
    fishes = all_fishes[0]

    for snippet in range(1, len(all_fishes)):
        for fish in range(len(all_fishes[snippet])):
            fish_last_f = np.array([fishes[i][-1] for i in range(len(fishes))])
            diff = abs(fish_last_f - all_fishes[snippet][fish][0])
            if diff[np.argsort(diff)[0]] < 3.:
                fishes[np.argsort(diff)[0]] = np.concatenate((fishes[np.argsort(diff)[0]], all_fishes[snippet][fish]))
            else:
                fishes.append(all_fishes[snippet][fish])

    return fishes


def main(filename):
    fig, ax = plt.subplots()
    f = open(filename, 'rb')
    fishes = pickle.load(f)

    if fishes == []:
        print("file doesn't contain any fishes")
        quit()
    plot_fundamentals(fishes, ax)

    clean_fishes = clear_fishes(fishes)

    plt.show()

if __name__ == '__main__':
    file_name = sys.argv[1]
    main(file_name)