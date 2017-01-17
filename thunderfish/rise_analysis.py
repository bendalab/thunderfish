import argparse
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as scp
from scipy.optimize import curve_fit
from scipy.signal import correlate

from IPython import embed


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
        rise_iri.append(np.array([]))
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
                        rise_df[-1].append(all_rises[fish][rise][1][0] - all_rises[fish][rise][1][1])
                        counter2 += 1
                        continue
                    popt = compute_tau(all_rises, fishes, all_times, rise, fish, dpm)
                    if popt[1] >= 200:
                        rise_tau[-1].append(np.nan)
                        rise_df[-1].append(all_rises[fish][rise][1][0] - all_rises[fish][rise][1][1])
                        continue

                    rise_tau[-1].append(popt[1])
                    rise_df[-1].append(all_rises[fish][rise][1][0] - exponenial_func(popt[1] * 2, *popt))
                else:
                    rise_tau[-1].append(np.nan)
                    rise_df[-1].append(all_rises[fish][rise][1][0] - all_rises[fish][rise][1][1])

            rise_iri[-1] = np.diff(rise_t[-1])

    total_rise_count = np.sum([len(all_rises[x]) for x in range(len(all_rises))])
    c1p = 100. * counter1 / total_rise_count
    c2p = 100. * counter2 / total_rise_count
    print('\n# excluded because df to low:          %.1f percent' % c1p)
    print('\n# excluded because not able to fit:    %.1f percent' %c2p)
    print('')

    return rise_f, rise_t, rise_tau, rise_dt, rise_df, rise_iri


def fist_level_analysis(rise_tau, rise_df, show_plot=False):
    ## Rise tau vs. delta freq ##
    slope, intercept, r_val, p_val, stderr = scp.linregress(rise_tau, rise_df)
    x = np.arange(np.min(rise_tau), np.max(rise_tau), 0.01)
    y = slope * x + intercept

    if show_plot:
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
    lags = np.arange(1, 5)

    ser_IRI_data = []

    for fish in range(len(rise_iri)):
        if len(rise_iri[fish]) <= lags[-1] + 1:
            ser_IRI_data.append([[], [], []])
            continue

        # real serial correlation ###
        corrcof = np.ones(len(lags)+1)
        for lag in lags:
            corrcof[lag] = scp.pearsonr(rise_iri[fish][lag:], rise_iri[fish][:-lag])[0]

        # bootstrapt serial correlation ###
        boot_corrcof = []
        for i in range(2000):
            boot_corrcof.append(np.ones(len(lags)+1))
            tmp_iri = np.copy(rise_iri[fish])
            np.random.shuffle(tmp_iri)

            for lag in lags:
                boot_corrcof[-1][lag] = scp.pearsonr(tmp_iri[lag:], tmp_iri[:-lag])[0]

        # calculate percentiles
        p975 = np.ones(np.size(boot_corrcof, axis=1))
        p025 = np.ones(np.size(boot_corrcof, axis=1))

        for i in range(np.size(boot_corrcof, axis=1)):
            p975[i], p025[i] = np.percentile(np.array(boot_corrcof)[:, i], (97.5, 2.5))

        ser_IRI_data.append([corrcof, p975, p025])

    for fish in range(len(ser_IRI_data)):
        if ser_IRI_data[fish][0] == []:
            continue

        fig, ax = plt.subplots()
        ax.fill_between(np.arange(len(lags)+1), ser_IRI_data[fish][1], ser_IRI_data[fish][2], alpha=0.5)
        ax.plot(np.arange(len(lags)+1), ser_IRI_data[fish][0], color='red', marker='o')
        ax.set_ylabel('corrkoef')
        ax.set_xlabel('lag')
        ax.set_title('fish %.0f; rise n = %.0f' % (fish, len(rise_iri[fish])+1))
    plt.show()


def cross_t(rise_t):
    def gauss_kernal(t, mu, sigma):
        return 1/(np.sqrt(2 * np.pi) *sigma) * np.exp( -1. * (t - mu)**2 / (2*(sigma**2))  )

    for fish in range(len(rise_t)-1):
        if rise_t[fish] == []:
            continue
        # todo one vs all
        for comp_fish in range(fish+1, len(rise_t)):
            if rise_t[comp_fish] == []:
                continue

            # ToDo: bootstrap, gauss kernal, usw.
            rel_t = np.arange(-180, 180, 1)
            sumed_gaus = np.zeros(len(rel_t))

            rel_possition = []
            for t0 in rise_t[fish]:
                for t1 in rise_t[comp_fish]:
                    rel_possition.append(t1 - t0)
            counter = 0
            for mu in rel_possition:
                if (mu > -200) & (mu < 200):
                    counter += 1
                    sumed_gaus += gauss_kernal(rel_t, mu, 10.)

            boot_sumed_gaus = []
            for i in range(2000):
                boot_sumed_gaus.append(np.zeros(len(rel_t)))
                rise_t_fish_cp = np.copy(rise_t[fish])
                rise_t_comp_fish_cp = np.copy(rise_t[comp_fish])

                dt_fish = np.diff(np.append(0, rise_t_fish_cp))
                dt_comp = np.diff(np.append(0, rise_t_comp_fish_cp))

                np.random.shuffle(dt_fish)
                np.random.shuffle(dt_comp)

                new_t_fish = np.cumsum(dt_fish)
                new_t_cfish = np.cumsum(dt_comp)

                rel_possition = []
                for t0 in new_t_fish:
                    for t1 in new_t_cfish:
                        rel_possition.append(t1 - t0)

                for mu in rel_possition:
                    if (mu > -200) & (mu < 200):
                        boot_sumed_gaus[-1] += gauss_kernal(rel_t, mu, 10.)


            p975, p025 = np.percentile(np.array(boot_sumed_gaus), (97.5, 2.5), axis=0)
            # embed()
            # quit()

            if counter >= 5:
                fig, ax = plt.subplots()
                ax.fill_between(rel_t, p975, p025, alpha=0.5)
                ax.plot(rel_t, sumed_gaus, color='red')
                ax.set_xlabel('time [s]')
                ax.set_title('fish %.0f (%.0f); comp_fish %.0f (%.0f); n= %.0f' % (fish, len(rise_t[fish]), comp_fish, len(rise_t[comp_fish]), counter))
    plt.show()

            # if counter >= 5:
            #     fig, ax = plt.subplots()
            #     ax.plot(rel_t, sumed_gaus)
            #     ax.set_xlabel('time [s]')
            #     ax.set_title('fish %.0f (%.0f); comp_fish %.0f (%.0f); n= %.0f' % (fish, len(rise_t[fish]), comp_fish, len(rise_t[comp_fish]), counter))
            #     plt.show()

def rise_analysis(file_path):
    folders = np.array([x[0] for x in os.walk(file_path)])
    if len(folders) > 1:
        folders = folders[1:]

    f_vec, t_vec, tau_vec, dt_vec, df_vec = np.array([]), np.array([]), np.array([]), np.array([]), np.array([])

    for folder in folders:
        tmp_fishes, all_times, all_rises = load_data(folder)
        f, t, tau, dt, df, iri = get_rise_params(tmp_fishes, all_times, all_rises, calculate_tau=True)

        f_vec = np.concatenate((f_vec, np.hstack(f)))  # rise base freq
        t_vec = np.concatenate((t_vec, np.hstack(t)))  # rise time of occurrence
        tau_vec = np.concatenate((tau_vec, np.hstack(tau)))  # tau of rise
        dt_vec = np.concatenate((dt_vec, np.hstack(dt)))  # rise duration
        df_vec = np.concatenate((df_vec, np.hstack(df)))  # delta frequency of rise

    fist_level_analysis(tau_vec[~np.isnan(tau_vec)], df_vec[~np.isnan(tau_vec)], show_plot=False)

    if len(folders) == 1:
        # IRIs(iri)

        cross_t(t)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Analyse long single- or multi electrode EOD recordings of weakly electric fish.',
        epilog='by bendalab (2015-2016)')
    parser.add_argument('file_path', nargs='?', default='', type=str, help='folder containing the data to analyse')

    args = parser.parse_args()
    rise_analysis(args.file_path)