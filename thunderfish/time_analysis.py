import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as scp

from IPython import embed

def load_temp_and_time(all_times, log_infos):
    dt = all_times[1] - all_times[0]
    dpm = 60./ dt

    if log_infos == False:
        return [], []
    else:
        start_time_str = False
        for i in range(len(log_infos)):
            if log_infos[i][2] == 'begin of recording':
                start_time_str = ' ' + log_infos[i][1]
                start_date_str = log_infos[i][0]

    if start_time_str:
        start_time = [int(start_time_str.split(':')[i]) for i in range(2)]
        duration = all_times[-1]
        add_h = duration // 3600.
        duration -= add_h * 3600.

        add_m = duration // 60.
        duration -= add_m * 60.

        end_time = [int(start_time[0] + add_h), int(start_time[1] + add_m)]
        if end_time[1] >= 60:
            end_time[1] -= 60
            end_time[0] += 1
        if end_time[0] >= 24:
            end_time[0] -= 24
        end_time = [str(end_time[i]) for i in range(len(end_time))]
        for t in range(len(end_time)):
            if len(end_time[t]) == 1:
                end_time[t] = '0' + end_time[t]
        end_time_str = ' ' +  end_time[0] + ':' + end_time[1]

    # load temperature
    hobo_logger1 = pd.read_csv('../../raab_data/colombia_2016/hobo_logger_data/logger1.csv')
    hobo_logger2 = pd.read_csv('../../raab_data/colombia_2016/hobo_logger_data/logger2.csv')

    temp = []
    temp_idx = []
    for hobo_logger in [hobo_logger1, hobo_logger2]:
        start_idx = hobo_logger['DateTime'][(hobo_logger['DateTime'].str.contains(start_date_str)) &
                                        (hobo_logger['DateTime'].str.contains(start_time_str))].index

        end_idx = hobo_logger['DateTime'][hobo_logger['DateTime'].str.contains(end_time_str)].index

        # embed()
        if len(end_idx) == 0:
            continue
        else:
            if len(start_idx) == 0:
                start_idx = 0
            else:
                start_idx = start_idx[0]

            end_idx = end_idx[end_idx > start_idx]
            if len(end_idx) == 0:
                continue
            else:
                end_idx = end_idx[end_idx > start_idx][0]

            temp = hobo_logger['Temp'][start_idx:end_idx+1].values
            temp = np.array([float(temp[i].replace(' ', '.')) for i in range(len(temp))])

            first_temp_time = hobo_logger['DateTime'][start_idx].split(' ')[-1].split(':')[:-1]
            first_temp_time = [int(first_temp_time[0]), int(first_temp_time[1])]
            if first_temp_time[0] < start_time[0]:
                first_temp_time[0] += 24

            mindiff = (first_temp_time[0] * 60 +  first_temp_time[1]) - (start_time[0] * 60 + start_time[1])

            temp_idx = np.asarray((np.arange(len(temp))+mindiff) * dpm, dtype=int)

            interp_temp = np.interp( np.arange(len(all_times)), temp_idx, temp)
            break

    if temp == []:
        return [], []
    else:
        temp = temp[temp_idx < len(all_times)]
        temp_idx = temp_idx[temp_idx < len(all_times)]

        return interp_temp, start_time_str
        # return temp, temp_idx, start_time_str


def load_dat_file(dat_file):
    log_infos = []
    for line in dat_file:
        if 'Num' in line:
            log_infos.append([])
        if 'Date' in line:
            date = line.split(': ')[-1].strip().replace('"', '').split('-')
            date = date[2] + '/' + date[1] + '/' + date[0][2:]
            log_infos[-1].append(date)
        if 'Time' in line:
            log_infos[-1].append(line.split(': ')[-1].strip().split('.')[0].replace('"', '')[:-3])
        if 'Comment' in line:
            log_infos[-1].append(line.split(': ')[-1].strip().replace('"', ''))
    return log_infos


def rises_per_hour(fishes, all_times, all_rises, temp, slope):
    rises_ph_n_f = []
    for fish in range(len(fishes)):
        # median frequency at 25 deg C
        cp_fish = np.copy(fishes[fish])
        cp_temp = np.copy(temp) - 25.
        cp_fish += cp_temp * slope[fish]
        if len(cp_fish[~np.isnan(cp_fish)]) <= 1:
            continue
        first = all_times[~np.isnan(fishes[fish])][0]
        last = all_times[~np.isnan(fishes[fish])][-1]
        if (last - first) < 1800.:
            continue

        med_fish_freq = np.median(cp_fish[~np.isnan(cp_fish)])

        rise_count = 0
        for rise in all_rises[fish]:
            if rise[1][0] - rise[1][1] > 1.:
                rise_count += 1

        occure_h = (all_times[~np.isnan(cp_fish)][-1] - all_times[~np.isnan(cp_fish)][0]) / 3600.

        rises_ph_n_f.append([1.0* rise_count / occure_h, med_fish_freq])

    return rises_ph_n_f

def plot_fishes(fishes, all_times, all_rises, temp, start_time_str, hz_p_deg    ):

    fig, ax = plt.subplots(facecolor='white', figsize=(11.6, 8.2))

    # determine x resolution
    if all_times[-1] <= 120:
        time_factor = 1.
    elif all_times[-1] > 120 and all_times[-1] < 7200:
        time_factor = 60.
    else:
        time_factor = 3600.

    # plot fishes
    for fish in range(len(fishes)):
        color = np.random.rand(3, 1)
        ax.plot(all_times[~np.isnan(fishes[fish])] / time_factor, fishes[fish][~np.isnan(fishes[fish])],
                color=color, marker='.')

    # plot rises
    legend_in = False
    for fish in range(len(all_rises)):
        for rise in all_rises[fish]:
            if rise[1][0] - rise[1][1] > 1.5:
                if legend_in == False:
                    ax.plot(all_times[rise[0][0]] / time_factor, rise[1][0], 'o', color='red', markersize=7,
                            markerfacecolor='None', label='rise begin')
                    ax.plot(all_times[rise[0][1]] / time_factor, rise[1][1], 's', color='green', markersize=7,
                            markerfacecolor='None', label='rise end')
                    legend_in = True
                    ax.legend(loc=4, numpoints=1, frameon=False, fontsize=12)
                else:
                    ax.plot(all_times[rise[0][0]] / time_factor, rise[1][0], 'o', color='red', markersize=7,
                            markerfacecolor='None')
                    ax.plot(all_times[rise[0][1]] / time_factor, rise[1][1], 's', color='green', markersize=7,
                            markerfacecolor='None')

    # plot cosmetics for fish plot
    maxy = np.max(np.array([np.mean(fishes[fish][~np.isnan(fishes[fish])]) for fish in range(len(fishes))]))
    miny = np.min(np.array([np.mean(fishes[fish][~np.isnan(fishes[fish])]) for fish in range(len(fishes))]))
    if miny < 0:
        miny =0

    ax.set_ylim([miny - 150, maxy + 150])
    ax.set_ylabel('Frequency [Hz]', fontsize=14)
    if time_factor == 1.:
        plt.xlabel('Time [sec]', fontsize=14)
    elif time_factor == 60.:
        plt.xlabel('Time [min]', fontsize=14)
    else:
        plt.xlabel('Time [h]', fontsize=14)

    ax.spines['top'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()

    # temp plot
    if temp != []:
        ax2 = ax.twinx()
        ax2.plot(all_times / time_factor, temp, '-', color='red', linewidth=2, label='temperature [C]')
        temp_ylim = np.diff(ax.get_ylim())[0] // hz_p_deg /2
        ax2.set_ylim([30 - temp_ylim, 30 + temp_ylim])
        ax2.set_ylabel('temperature [C]', fontsize=14)
        ax2.legend(loc=1, numpoints=1, frameon=False, fontsize=12)
    else:
        ax.spines['right'].set_visible(False)

    if start_time_str:
        old_ticks = ax.get_xticks()
        start_time = start_time_str.split(':')

        if time_factor == 3600.:
            new_ticks = [str(int((int(start_time[0])+ old_ticks[i]) % 24)) + ':' + str(int(start_time[1])) for i in range(len(old_ticks))]
            plt.xticks(old_ticks, new_ticks)
            ax.set_xlabel('Time')
    plt.tight_layout()
    plt.show()



def plot_freq_vs_temp(fishes, temp):
    # embed()
    for fish in fishes:
        fig, ax = plt.subplots()
        ax.scatter(temp[~np.isnan(fish)], fish[~np.isnan(fish)])
        plt.show()

def main(file_path, show_fish_plot=True):
    # load data and get fishcount per half an hour...
    rise_phnf= []
    folders = np.array([x[0] for x in os.walk(file_path)])
    if len(folders) > 1:
        folders = folders[1:]

    for folder in sorted(folders):
        print folder
        # load data
        dat_file = False
        fishes = False
        all_rises = False
        all_times = False
        for file in os.listdir(folder):
            if file.endswith('final_times.npy'):
                all_times = np.load(os.path.join(folder, file))
            elif file.endswith('final_fishes.npy'):
                fishes = np.load(os.path.join(folder, file))
            elif file.endswith('final_rises.npy'):
                all_rises = np.load(os.path.join(folder, file))
            elif file.endswith('.dat'):
                dat_file = open(os.path.join(folder, file), 'r')

        # load log file and hobo temp
        log_infos = False
        temp = False
        hz_p_deg = False
        start_time_str = False
        if dat_file:
            log_infos = load_dat_file(dat_file)

        if log_infos:
            # temp, temp_idx, start_time_str = load_temp_and_time(all_times, log_infos)
            temp, start_time_str = load_temp_and_time(all_times, log_infos)

            # calculate freq change per deg C
            if temp != []:
                slopes = []
                for fish in fishes:
                    slope, intercept, _, _, _ = scp.linregress(temp[~np.isnan(fish)], fish[~np.isnan(fish)])
                    slopes.append(slope)
                hz_p_deg = np.median(slopes[slopes > 0])

            # rises_per_hour(fishes, all_times, all_rises, temp, slopes)

        # plot fishes
        if show_fish_plot:
            plot_fishes(fishes, all_times, all_rises, temp, start_time_str, hz_p_deg)

        if log_infos and temp != []:
            rises_ph_n_f = rises_per_hour(fishes, all_times, all_rises, temp, slopes)
            rise_phnf += rises_ph_n_f
            # plt.close()

    rise_base_freq = np.array([rise_phnf[n][1] for n in range(len(rise_phnf))])
    rise_counts = np.array([rise_phnf[n][0] for n in range(len(rise_phnf))])

    bins = np.arange(455, 1055, 10)
    bins = bins.tolist()
    rise_freq_counts = []
    for bin_c in bins:
        rise_freq_counts.append(rise_counts[(rise_base_freq >= bin_c -5) & (rise_base_freq < bin_c +5)])

    for bin in reversed(range(len(bins))):
        if len(rise_freq_counts[bin]) == 0:
            rise_freq_counts.pop(bin)
            bins.pop(bin)
    bins = np.array(bins)

    # male plot
    inch_factor = 2.54
    fig, ax = plt.subplots(facecolor='white', figsize=(20. / inch_factor, 12. / inch_factor))
    ax.boxplot(np.array(rise_freq_counts)[(bins >= 800) & (bins < 1050)], sym='')
    old_ticks = ax.get_xticks()
    plt.xticks(old_ticks, bins[(bins >= 800) & (bins < 1050)], rotation=45)

    male_base_freq = rise_base_freq[(rise_base_freq >= 800) & (rise_base_freq < 1050)]
    male_rise_count = rise_counts[(rise_base_freq >= 800) & (rise_base_freq < 1050)]
    r_val, p_val = scp.pearsonr(male_base_freq, male_rise_count)

    ax.set_ylim([0, 10])
    ax.set_xlabel('frequency [Hz]')
    ax.set_ylabel('rise count per h [n/h]')
    ax.set_title('Risecounts vs. frequency in Males \n(pearsonr; r= %.3f; p= %.3f)' % (r_val, p_val))
    plt.tight_layout()


    # neutron plot
    inch_factor = 2.54
    fig, ax = plt.subplots(facecolor='white', figsize=(20. / inch_factor, 12. / inch_factor))
    ax.boxplot(np.array(rise_freq_counts)[(bins >= 700) & (bins < 800)], sym='')
    old_ticks = ax.get_xticks()
    plt.xticks(old_ticks, bins[(bins >= 700) & (bins < 800)], rotation=45)
    ax.set_ylim([0, 10])

    # neutron plot
    inch_factor = 2.54
    fig, ax = plt.subplots(facecolor='white', figsize=(20. / inch_factor, 12. / inch_factor))
    ax.boxplot(np.array(rise_freq_counts)[(bins >= 450) & (bins < 700)], sym='')
    old_ticks = ax.get_xticks()
    plt.xticks(old_ticks, bins[(bins >= 450) & (bins < 700)], rotation=45)

    female_base_freq = rise_base_freq[(rise_base_freq >= 450) & (rise_base_freq < 700)]
    female_rise_count = rise_counts[(rise_base_freq >= 450) & (rise_base_freq < 700)]
    r_val, p_val = scp.pearsonr(female_base_freq, female_rise_count)

    ax.set_ylim([0, 10])
    ax.set_xlabel('frequency [Hz]')
    ax.set_ylabel('rise count per h [n/h]')
    ax.set_title('Risecounts vs. frequency in Females \n(pearsonr; r= %.3f; p= %.3f)' % (r_val, p_val))
    plt.tight_layout()

    # male female neurtron plot
    f_rise_counts = rise_counts[(rise_base_freq >= 450) & (rise_base_freq < 700)]
    n_rise_counts = rise_counts[(rise_base_freq >= 700) & (rise_base_freq < 800)]
    m_rise_counts = rise_counts[(rise_base_freq >= 800) & (rise_base_freq < 1050)]

    fig, ax = plt.subplots(facecolor='white', figsize=(20. / inch_factor, 12. / inch_factor))
    ax.boxplot([f_rise_counts, n_rise_counts, m_rise_counts], sym='')
    old_ticks = ax.get_xticks()
    plt.xticks(old_ticks, ['female', 'neutron', 'male'])
    plt.show()

    # fig, ax = plt.subplots()
    # ax.scatter([rise_phnf[n][1] for n in range(len(rise_phnf))],
    #            [rise_phnf[n][0] for n in range(len(rise_phnf))])
    #
    # plt.show()
    ######################### old stuff ##################################

    # ### create a clock array ###
    # full_clock = ['0:15']
    # while True:
    #     h = int(full_clock[-1].split(':')[0])
    #     m = int(full_clock[-1].split(':')[1])
    #     if m is 15:
    #         full_clock.append(str(h) + ':' + str(m+30))
    #     else:
    #         full_clock.append(str(h+1) + ':' + str(m-30))
    #     if full_clock[-1] == '23:45':
    #         break
    # full_clock = np.array(full_clock)
    # ####

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Analyse long single- or multi electrode EOD recordings of weakly electric fish.',
        epilog='by bendalab (2015-2016)')
    parser.add_argument('file_path', nargs='?', default='', type=str, help='folder containing the data to analyse')

    args = parser.parse_args()
    main(args.file_path)