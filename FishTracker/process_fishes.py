import sys
import numpy as np
import matplotlib.pyplot as plt
import pickle
import glob


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


def main(folder):
    fig, ax = plt.subplots()

    list = glob.glob(folder + '*.p')
    for i in range(len(list)):
        list[i] = int(list[i].split('_')[-1].split('.')[0])
    endings = np.unique(list)

    for ending in endings:
        fishes_file = glob.glob(folder+'*fishes*'+'_'+str(ending)+'*.p')[0]
        time_file = glob.glob(folder+'*all_times*'+'_'+ str(ending)+'*.p')[0]
        chirp_freq_file = glob.glob(folder+'*chirp_freq*'+'_'+ str(ending)+'*.p')[0]
        chirp_time_file = glob.glob(folder+'*chirp_time*'+'_'+ str(ending)+'*.p')[0]

        f = open(fishes_file, 'rb')
        fishes = pickle.load(f)

        t = open(time_file, 'rb')
        time = pickle.load(t)

        cf = open(chirp_freq_file, 'rb')
        chirp_freq = pickle.load(cf)

        ct = open(chirp_time_file, 'rb')
        chirp_time = pickle.load(ct)

        for fish in range(len(fishes)):
            if len(fishes[fish][~np.isnan(fishes[fish])]) >= 50:
                ax.plot(time[:len(fishes[fish])]/60., fishes[fish])
        ax.plot(chirp_time, chirp_freq, '.', color='red', markersize=20)

        del f, t, cf, ct, fishes, time, chirp_freq, chirp_time

        plt.draw()
        plt.pause(0.001)
    plt.show()


    # fig, ax = plt.subplots()
    # f = open(filename, 'rb')
    # fishes = pickle.load(f)
    #
    # if fishes == []:
    #     print("file doesn't contain any fishes")
    #     quit()
    # plot_fundamentals(fishes, ax)
    #
    # clean_fishes = clear_fishes(fishes)
    #
    # plt.show()

if __name__ == '__main__':
    folder = sys.argv[1]
    main(folder)