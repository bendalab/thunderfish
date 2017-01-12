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
    rise_iri = []

    counter1 = 0
    counter2 = 0
    for fish in range(len(all_rises)):
        rise_f.append([])
        rise_t.append([])
        rise_tau.append([])
        rise_dt.append([])
        rise_df.append([])
        rise_iri.append([])
        #
        # tmp_t = []
        if all_rises[fish] == []:
            # rise_iri.append(np.array([]))
            continue
        else:
            for rise in range(len(all_rises[fish])):
                if all_rises[fish][rise][1][0] - all_rises[fish][rise][1][1] <= 1.5:
                    counter1 += 1
                    continue

                # tmp_t.append(all_rises[fish][rise][0][0])
                rise_f[-1].append(all_rises[fish][rise][1][1])
                rise_t[-1].append(all_times[all_rises[fish][rise][0][0]])
                rise_dt[-1].append(all_times[all_rises[fish][rise][0][1]] - all_times[all_rises[fish][rise][0][0]])


                end_idx_plus = all_rises[fish][rise][0][1] + dpm / 2
                other_start_idx = np.array([all_rises[fish][x][0][0] for x in np.arange(rise+1, len(all_rises[fish]))])
                larger_than = end_idx_plus > other_start_idx
                # print larger_than

                if calculate_tau:
                    if True in larger_than:
                        rise_tau[-1].append(np.nan)
                        counter2 += 1
                        continue
                    popt = compute_tau(all_rises, fishes, all_times, rise, fish, dpm)
                    if popt[1] >= 200:
                        rise_tau[-1].append(np.nan)
                        continue

                    rise_tau[-1].append(popt[1])
                    rise_df[-1].append(all_rises[fish][rise][1][0] - exponenial_func(popt[1] * 2, *popt))
                else:
                    rise_tau[-1].append(np.nan)
                    rise_df[-1].append(all_rises[fish][rise][1][0] - all_rises[fish][rise][1][1])

                # rise_f[-1].append(all_rises[fish][rise][1][1])
                # rise_t[-1].append(all_times[all_rises[fish][rise][0][0]])
                # rise_dt[-1].append(all_times[all_rises[fish][rise][0][1]] - all_times[all_rises[fish][rise][0][0]])
            rise_iri[-1] = np.diff(rise_t[-1])

    total_rise_count = np.sum([len(all_rises[x]) for x in range(len(all_rises))])
    c1p = 100. * counter1 / total_rise_count
    c2p = 100. * counter2 / total_rise_count
    print('\n# excluded because df to low:          %.1f percent' % c1p)
    print('\n# excluded because not able to fit:    %.1f percent' %c2p)
    print('')

    return rise_f, rise_t, rise_tau, rise_dt, rise_df, rise_iri


def fist_level_analysis(rise_tau, rise_df):
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
    # plt.draw()
    # plt.pause(0.001)

    ## df distribution ##
    fig2, ax2 = plt.subplots(facecolor='white')
    bins = np.arange(0, np.ceil(np.max(rise_df)), 0.1)
    n, edges = np.histogram(rise_df, bins)
    ax2.bar(edges[:-1], n, 0.1)
    ax2.set_ylabel('count')
    ax2.set_xlabel('delta frequency [Hz]')
    ax2.set_title('Rise delta frequency histogram (bw = 0.1Hz)')
    # plt.draw()
    # plt.pause(0.001)

    ## tau distribution ##
    fig3, ax3 = plt.subplots(facecolor='white')
    bins = np.arange(0, np.ceil(np.max(rise_tau)), 1)
    n, edges = np.histogram(rise_tau, bins)
    ax3.bar(edges[:-1], n, 1)
    ax3.set_ylabel('count')
    ax3.set_xlabel('tau [s]')
    ax3.set_title('Rise tau histogram (bw = 1s)')

    plt.show()


def IRIs(rise_iri):
    # serial correlation#
    lags = np.arange(1, 5)
    corrkof = []
    p_val = []

    for fish in range(len(rise_iri)):
        if len(rise_iri[fish]) <= lags[-1] + 1:
            corrkof.append(np.array([]))
            p_val.append(np.array([]))
            continue
        corrkof.append(np.ones(len(lags)+1))
        p_val.append(np.zeros(len(lags)+1))
        for lag in lags:
            corrkof[fish][lag], p_val[fish][lag] = scp.pearsonr(rise_iri[fish][lag:], rise_iri[fish][:-lag])

    for fish in range(len(corrkof)):
        if len(corrkof[fish]) == 0:
            continue
        fig, ax = plt.subplots()
        ax.plot(np.arange(5), corrkof[fish], marker='o')
        plt.show()

def rise_analysis(file_path):
    folders = np.array([x[0] for x in os.walk(file_path)])
    if len(folders) > 1:
        folders = folders[1:]

    f_vec, t_vec, tau_vec, dt_vec, df_vec = np.array([]), np.array([]), np.array([]), np.array([]), np.array([])

    for folder in folders:
        tmp_fishes, all_times, all_rises = load_data(folder)
        # ToDo: which variables shall be calculated if tau is not calculatable !!! np.nans ?
        f, t, tau, dt, df, iri = get_rise_params(tmp_fishes, all_times, all_rises, calculate_tau=True)

        # params of the rises larger than 1.5 Hz and where tau was possible to calculate.
        # --IRIs are NOT dependant on the ability of tau calculation--
        f_vec = np.concatenate((f_vec, np.hstack(f)))  # rise base freq
        t_vec = np.concatenate((t_vec, np.hstack(t)))  # rise time of occurrence
        tau_vec = np.concatenate((tau_vec, np.hstack(tau)))  # tau of rise
        dt_vec = np.concatenate((dt_vec, np.hstack(dt)))  # rise duration
        df_vec = np.concatenate((df_vec, np.hstack(df)))  # delta frequency of rise

    fist_level_analysis(tau_vec, df_vec)

    embed()
    quit()
    if len(folders) == 1:
        IRIs(iri)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Analyse long single- or multi electrode EOD recordings of weakly electric fish.',
        epilog='by bendalab (2015-2016)')
    parser.add_argument('file_path', nargs='?', default='', type=str, help='folder containing the data to analyse')

    args = parser.parse_args()
    rise_analysis(args.file_path)