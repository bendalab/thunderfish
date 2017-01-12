import argparse
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as scp
from scipy.optimize import curve_fit
from IPython import embed


def multi_file_rise_analysis(file_path):
    def load_and_join_data(file_path):
        rise_dt = []
        rise_df = []

        subdirectories = np.array([x[0] for x in os.walk(file_path)])[2:]

        for folder in subdirectories:
            all_rises = None
            all_times = None
            for file in os.listdir(folder):
                if file.endswith('final_rises.npy'):
                    all_rises = np.load(os.path.join(folder, file))
                elif file.endswith('final_times.npy'):
                    all_times = np.load(os.path.join(folder, file))
            if all_rises == None or all_times == None:
                print('missing data ...')
                print(folder)

            for fish in range(len(all_rises)):
                if all_rises[fish] == []:
                    continue
                else:
                    for rise in range(len(all_rises[fish])):
                        if all_rises[fish][rise][1][0] - all_rises[fish][rise][1][1] >= 1.5:
                            rise_df.append(all_rises[fish][rise][1][0] - all_rises[fish][rise][1][1])
                            rise_dt.append(all_times[all_rises[fish][rise][0][1]] - all_times[all_rises[fish][rise][0][0]])
        return rise_df, rise_dt

    ## Rise duration vs. delta freq ##
    rise_df, rise_dt = load_and_join_data(file_path)

    slope, intercept, r_val, p_val, stderr = scp.linregress(rise_dt, rise_df)
    x = np.arange(np.min(rise_dt), np.max(rise_dt), 0.01)
    y = slope * x + intercept

    fig1, ax1 = plt.subplots(facecolor='white')
    ax1.plot(rise_dt, rise_df, '.')
    ax1.plot(x, y)

    ax1.set_ylabel('delta frequency [Hz]')
    ax1.set_xlabel('duration [s]')
    ax1.set_title('Single rise df/dt (R=%.2f; p=%.3f; n = %.0f)' % (r_val, p_val, len(rise_dt)))

    ## df distribution ##
    fig2, ax2 = plt.subplots(facecolor='white')
    bins = np.arange(0, np.ceil(np.max(rise_df)), 0.1)
    n, edges = np.histogram(rise_df, bins)
    ax2.bar(edges[:-1], n, 0.1)
    ax2.set_ylabel('count')
    ax2.set_xlabel('delta frequency [Hz]')
    ax2.set_title('Rise delta frequency histogram (bw = 0.1Hz)')

    ## dt distribution ##
    fig3, ax3 = plt.subplots(facecolor='white')
    bins = np.arange(0, np.ceil(np.max(rise_dt)), 1)
    n, edges = np.histogram(rise_dt, bins)
    ax3.bar(edges[:-1], n, 1)
    ax3.set_ylabel('count')
    ax3.set_xlabel('delta time [s]')
    ax3.set_title('Rise delta frequency histogram (bw = 1s)')

    plt.show()


def rise_train_analysis(file_path):
    def exponenial_func(t, c1, tau, c2):
        return c1 * np.exp(-t / tau) + c2

    def load_rise_data(file_path):
        fishes = None
        times = None
        all_rises = None

        for file in os.listdir(file_path):
            if file.endswith("final_fishes.npy"):
                fishes = np.load(os.path.join(file_path, file))
            elif file.endswith("final_times.npy"):
                times = np.load(os.path.join(file_path, file))
            elif file.endswith("final_rises.npy"):
                all_rises = np.load(os.path.join(file_path, file))
            else:
                continue

        if fishes == None or times == None or all_rises == None:
            print('missing data files.')
            quit()

        return fishes, times, all_rises

    fishes, times, all_rises = load_rise_data(file_path)

    #### exponential fit test ####
    # fish 12 rise -6

    fish = 12
    rise = -6

    for fish in range(len(all_rises)):
        if all_rises[fish] == []:
            continue
        else:
            for rise in range(len(all_rises[fish])):


                test_rise = all_rises[fish][rise]

                test_fish = fishes[fish][test_rise[0][0] : test_rise[0][1]+1]
                test_fish = test_fish[~np.isnan(test_fish)]

                test_time = times[test_rise[0][0] : test_rise[0][1]+1]
                test_time = test_time[~np.isnan(test_fish)]
                test_time -= test_time[0]

                popt, pcov = curve_fit(exponenial_func, test_time, test_fish)

                yy = exponenial_func(test_time, popt[0], popt[1], popt[2])

                all_rises[fish][rise].append(popt[1])
                # fig, ax = plt.subplots()
                # ax.plot(test_time, test_fish, '.')
                # ax.plot(test_time, yy, '--')
                # plt.draw()
                # plt.pause(2)
                # plt.close()
    embed()
    quit()

    ##############################
    time_vec = []
    tau_vac = []
    rise_times = []
    for fish in range(len(all_rises)):
        rise_times.append([])
        for rise in range(len(all_rises[fish])):
            rise_times[-1].append(np.mean([times[all_rises[fish][rise][0][1]], times[all_rises[fish][rise][0][0]]]))
            time_vec.append(np.mean([times[all_rises[fish][rise][0][1]], times[all_rises[fish][rise][0][0]]]))
##############################################################################


def load_data(folder):
    all_rises = None
    all_times = None
    fishes = None


    for file in os.listdir(folder):
        if file.endswith('final_rises.npy'):
            all_rises = np.load(os.path.join(folder, file))
        elif file.endswith('final_times.npy'):
            all_times = np.load(os.path.join(folder, file))
        elif file.endswith('final_fishes.npy'):
            fishes = np.load(os.path.join(folder, file))

    if all_rises == None or all_times == None or fishes == None:
        print('missing data ...')
        print(folder)
        quit()

    return fishes, all_times, all_rises


def get_rise_params(fishes, all_times, all_rises, calculate_tau=False):
    def exponenial_func(t, c1, tau, c2):
        return c1 * np.exp(-t / tau) + c2

    def compute_tau(all_rises, fishes, all_times, rise, fish, dpm):
        test_rise = all_rises[fish][rise]
        test_fish = fishes[fish][test_rise[0][0]: test_rise[0][1] + np.floor(dpm / 2)]
        test_time = all_times[test_rise[0][0]: test_rise[0][1] + np.floor(dpm/ 2)]

        test_time = test_time[~np.isnan(test_fish)]
        test_fish = test_fish[~np.isnan(test_fish)]
        test_time -= test_time[0]


        c1 = all_rises[fish][rise][1][0] - all_rises[fish][rise][1][1]
        tau = (all_times[all_rises[fish][rise][0][1]] - all_times[all_rises[fish][rise][0][0]]) * 0.3
        c2 = all_rises[fish][rise][1][1]
        popt, pcov = curve_fit(exponenial_func, test_time, test_fish, p0=(c1, tau, c2))
        return popt

    detection_time_diff = all_times[1] - all_times[0]
    dpm = 60. / detection_time_diff  # detections per minute

    rise_f = [] # rise base freq
    rise_t = [] # rise time of occurrence
    rise_tau = [] # tau of rise
    rise_dt = [] # rise duration
    rise_df = [] # delta frequency of rise

    counter1 = 0
    counter2 = 0
    for fish in range(len(all_rises)):
        if all_rises[fish] == []:
            continue
        else:
            for rise in range(len(all_rises[fish])):
                if all_rises[fish][rise][1][0] - all_rises[fish][rise][1][1] <= 1.5:
                    counter1 += 1
                    continue

                end_idx_plus = all_rises[fish][rise][0][1] + dpm / 2
                other_start_idx = np.array([all_rises[fish][x][0][0] for x in np.arange(rise+1, len(all_rises[fish]))])
                larger_than = end_idx_plus > other_start_idx
                # print larger_than

                if calculate_tau:
                    if True in larger_than:
                        counter2 += 1
                        continue
                    popt = compute_tau(all_rises, fishes, all_times, rise, fish, dpm)
                    if popt[1] >= 200:
                        continue

                    rise_tau.append(popt[1])
                    rise_df.append(all_rises[fish][rise][1][0] - exponenial_func(popt[1] * 2, *popt))
                else:
                    rise_df.append(all_rises[fish][rise][1][0] - all_rises[fish][rise][1][1])

                rise_f.append(all_rises[fish][rise][1][1])
                rise_t.append(all_times[all_rises[fish][rise][0][0]])
                rise_dt.append(all_times[all_rises[fish][rise][0][1]] - all_times[all_rises[fish][rise][0][0]])

    total_rise_count = np.sum([len(all_rises[x]) for x in range(len(all_rises))])
    c1p = 100. * counter1 / total_rise_count
    c2p = 100. * counter2 / total_rise_count
    print('\n# excluded because df to low:          %.1f percent' % c1p)
    print('\n# excluded because not able to fit:    %.1f percent' %c2p)
    print('')

    return rise_f, rise_t, rise_tau, rise_dt, rise_df


def fist_level_analysis(folders):
    rise_f, rise_t, rise_tau, rise_dt, rise_df = [], [], [], [], []

    for folder in folders:
        tmp_fishes, all_times, all_rises = load_data(folder)
        f, t, tau, dt, df = get_rise_params(tmp_fishes, all_times, all_rises, calculate_tau=True)

        rise_f += f
        rise_t += t
        rise_tau += tau
        rise_dt += dt
        rise_df += df

    rise_tau = np.array(rise_tau)

    ## Rise tau vs. delta freq ##
    slope, intercept, r_val, p_val, stderr = scp.linregress(rise_tau, rise_df)
    x = np.arange(np.min(rise_tau), np.max(rise_tau), 0.01)
    y = slope * x + intercept

    fig1, ax1 = plt.subplots(facecolor='white')

    ax1.plot(rise_tau, rise_df, '.')
    ax1.plot(x, y)

    ax1.set_ylabel('delta frequency [Hz]')
    ax1.set_xlabel('tau [s]')
    ax1.set_title('Single rise df/tau (R=%.2f; p=%.3f; n = %.0f)' % (r_val, p_val, len(rise_tau)))
    plt.draw()
    plt.pause(0.001)

    ## df distribution ##
    fig2, ax2 = plt.subplots(facecolor='white')
    bins = np.arange(0, np.ceil(np.max(rise_df)), 0.1)
    n, edges = np.histogram(rise_df, bins)
    ax2.bar(edges[:-1], n, 0.1)
    ax2.set_ylabel('count')
    ax2.set_xlabel('delta frequency [Hz]')
    ax2.set_title('Rise delta frequency histogram (bw = 0.1Hz)')
    plt.draw()
    plt.pause(0.001)

    ## tau distribution ##
    fig3, ax3 = plt.subplots(facecolor='white')
    bins = np.arange(0, np.ceil(np.max(rise_tau)), 1)
    n, edges = np.histogram(rise_tau, bins)
    ax3.bar(edges[:-1], n, 1)
    ax3.set_ylabel('count')
    ax3.set_xlabel('tau [s]')
    ax3.set_title('Rise tau histogram (bw = 1s)')

    plt.draw()
    plt.pause(0.001)

    return tmp_fishes, all_times, all_rises

def IRIs(all_rises):
    iri = []


    embed()
    quit()


def rise_analysis(file_path):

    folders = np.array([x[0] for x in os.walk(file_path)])
    if len(folders) > 1:
        folders = folders[1:]

    fishes, all_times, all_rises = fist_level_analysis(folders)

    if len(folders) == 1:
        IRIs(all_rises)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Analyse long single- or multi electrode EOD recordings of weakly electric fish.',
        epilog='by bendalab (2015-2016)')
    parser.add_argument('file_path', nargs='?', default='', type=str, help='folder containing the data to analyse')

    args = parser.parse_args()
    rise_analysis(args.file_path)