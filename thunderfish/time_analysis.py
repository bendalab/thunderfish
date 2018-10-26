import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as scp

from IPython import embed


def load_temp_and_time(hobo_path, all_times, start_time, start_time_str, end_time_str, start_date_str, end_date_str):
    dt = all_times[1] - all_times[0]
    dpm = 60./ dt
    # embed()
    # quit()
    hobo_logger_files = []
    for file in os.listdir(hobo_path):
        if file.endswith('.csv'):
            hobo_logger_files.append(pd.read_csv(os.path.join(hobo_path, file)))

    interp_temp = []
    interp_lux = []
    # for hobo_logger in [hobo_logger1, hobo_logger2]:
    for hobo_logger in hobo_logger_files:
        start_idx = hobo_logger['DateTime'][(hobo_logger['DateTime'].str.contains(start_date_str)) &
                                            (hobo_logger['DateTime'].str.contains(start_time_str))].index

        end_idx = hobo_logger['DateTime'][(hobo_logger['DateTime'].str.contains(end_date_str)) &
                                          (hobo_logger['DateTime'].str.contains(end_time_str))].index

        # get start and end index in temp array
        if len(end_idx) == 0:
            continue

        if len(start_idx) == 0:
            start_idx = 0
        else:
            start_idx = start_idx[0]

        end_idx = end_idx[end_idx > start_idx]
        if len(end_idx) == 0:
            continue
        else:
            end_idx = end_idx[end_idx > start_idx][0]

        # get correct temp array
        temp = hobo_logger['Temp'][start_idx:end_idx+1].values
        temp = np.array([float(temp[i].replace(' ', '.')) for i in range(len(temp))])

        lux = hobo_logger['Lux'][start_idx:end_idx+1].values
        lux = np.array([lux[i].replace('.', '') for i in range(len(lux))])
        lux = np.array([float(lux[i].replace(' ', '.')) for i in range(len(lux))])

        first_temp_time = hobo_logger['DateTime'][start_idx].split(' ')[-1].split(':')[:-1]
        first_temp_time = [int(first_temp_time[0]), int(first_temp_time[1])]
        if first_temp_time[0] < start_time[0]:
            first_temp_time[0] += 24

        # interpolate temp arrray
        mindiff = (first_temp_time[0] * 60 +  first_temp_time[1]) - (start_time[0] * 60 + start_time[1])

        temp_idx = np.asarray((np.arange(len(temp))+mindiff) * dpm, dtype=int)

        interp_temp = np.interp( np.arange(len(all_times)), temp_idx, temp)
        interp_lux = np.interp( np.arange(len(all_times)), temp_idx, lux)
        break

    return interp_temp, interp_lux


def extract_times(log_infos, all_times):
    # get start time and date from log file
    start_time_str = False
    start_date_str = False
    for i in range(len(log_infos)):
        if log_infos[i][2] == 'begin of recording':
            start_time_str = ' ' + log_infos[i][1]
            start_date_str = log_infos[i][0]

    # calulate end time and date
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
    end_time_str = ' ' + end_time[0] + ':' + end_time[1]

    if int(start_time_str.split(':')[0]) < int(end_time_str.split(':')[0]):
        end_date_str = start_date_str
    else:
        new_day = int(start_date_str.split('/')[0]) + 1
        if new_day < 10:
            end_date_str = '0' + str(new_day) + start_date_str[2:]
        else:
            end_date_str = str(new_day) + start_date_str[2:]

    return start_time, start_time_str, end_time_str, start_date_str, end_date_str


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
        if not np.isnan(slope[fish]):
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


def fish_n_rise_count_per_time(fishes, all_times, start_time, temp, all_rises):
    # create clock array..
    full_clock = ['0:15']
    while True:
        h = int(full_clock[-1].split(':')[0])
        m = int(full_clock[-1].split(':')[1])
        if m is 15:
            full_clock.append(str(h) + ':' + str(m+30))
        else:
            full_clock.append(str(h+1) + ':' + str(m-30))
        if full_clock[-1] == '23:45':
            break
    full_clock = np.array(full_clock)
    ###
    fish_clock_counts = np.full(len(full_clock), np.nan)
    m_clock_counts = np.full(len(full_clock), np.nan)
    f_clock_counts = np.full(len(full_clock), np.nan)

    rise_clock_counts = np.full(len(full_clock), np.nan)
    m_rise_clock_counts = np.full(len(full_clock), np.nan)
    f_rise_clock_counts = np.full(len(full_clock), np.nan)

    rise_rate_clock = np.full(len(full_clock), np.nan)
    m_rise_rate_clock = np.full(len(full_clock), np.nan)
    f_rise_rate_clock = np.full(len(full_clock), np.nan)

    mean_clock_temp = np.full(len(full_clock), np.nan)

    beat_f = []

    dt = all_times[1] - all_times[0]
    dpm = 60./ dt

    if start_time[1] <= 30:
        start_idx = np.floor((30 - start_time[1]) * dpm)
        start_clock = [start_time[0], 45]
        start_clock_str = str(start_clock[0]) + ':' + str(start_clock[1])
    else:
        start_idx = np.floor((60 - start_time[1]) * dpm)
        start_clock = [start_time[0] + 1, 15]
        start_clock_str = str(start_clock[0]) + ':' + str(start_clock[1])

    clock_idx = np.where(full_clock == start_clock_str)[0][0]

    while True:
        end_idx = np.floor(start_idx + 30. * dpm)
        if end_idx >= len(fishes[0]):
            break

        count = 0
        m_count = 0
        f_count = 0

        rise_count = 0
        m_rise_count = 0
        f_rise_count = 0

        fish_time = 0
        male_time = 0
        female_time = 0

        snippet_freq = []

        for enu, fish in enumerate(fishes):
            snipped_fish = fish[start_idx:end_idx]
            snipped_time = all_times[start_idx:end_idx]

            if len(snipped_fish[~np.isnan(snipped_fish)]) > 0:
                snippet_freq.append(np.median(snipped_fish[~np.isnan(snipped_fish)]))

                fish_dur = snipped_time[~np.isnan(snipped_fish)][-1] - snipped_time[~np.isnan(snipped_fish)][0]
                count += 1
                fish_time += fish_dur
                if np.median(snipped_fish[~np.isnan(snipped_fish)]) >= 730 and np.median(
                        snipped_fish[~np.isnan(snipped_fish)]) < 1050:
                    m_count += 1
                    male_time += fish_dur

                if np.median(snipped_fish[~np.isnan(snipped_fish)]) >= 550 and np.median(
                        snipped_fish[~np.isnan(snipped_fish)]) < 730:
                    f_count += 1
                    female_time += fish_dur

            for rise in all_rises[enu]:
                if rise[0][0] >= start_idx and rise[0][0] < end_idx:
                    if rise[1][0] - rise[1][1] > 1.:
                        rise_count += 1
                        if rise[1][1] >= 730 and rise[1][1] < 1050:
                            m_rise_count += 1
                        if rise[1][1] >= 550 and rise[1][1] < 730:
                            f_rise_count += 1

        snippet_freq = np.array(snippet_freq)
        snippet_freq = snippet_freq[(snippet_freq > 550) & (snippet_freq < 1050)]
        for i in range(len(snippet_freq)-1):
            df = snippet_freq[i+1:] - snippet_freq[i]
            beat_f += df.tolist()

        fish_clock_counts[clock_idx] = count
        m_clock_counts[clock_idx] = m_count
        f_clock_counts[clock_idx] = f_count

        rise_clock_counts[clock_idx] = rise_count
        m_rise_clock_counts[clock_idx] = m_rise_count
        f_rise_clock_counts[clock_idx] = f_rise_count

        rise_rate_clock[clock_idx] = rise_count / (fish_time / 3600.)
        m_rise_rate_clock[clock_idx] = m_rise_count / (male_time / 3600.)
        f_rise_rate_clock[clock_idx] = f_rise_count / (female_time / 3600.)

        if temp != []:
            mean_clock_temp[clock_idx] = np.mean(temp[start_idx:end_idx])

        clock_idx += 1
        if clock_idx == len(full_clock):
            clock_idx = 0
        start_idx = end_idx

    return fish_clock_counts, m_clock_counts, f_clock_counts, rise_clock_counts, m_rise_clock_counts, f_rise_clock_counts, \
           rise_rate_clock, m_rise_rate_clock, f_rise_rate_clock, mean_clock_temp, full_clock, beat_f


def get_presence_time(fishes, all_times, start_time):
    presence_time = []
    vanish_time = []
    presence_freq = []
    appear_time_h = []
    vanish_time_h = []

    start_min = start_time[0] * 60 + start_time[1]

    for fish in fishes:
        presence_time.append((all_times[~np.isnan(fish)][-1] - all_times[~np.isnan(fish)][0]) / 60.)
        vanish_time.append(start_min + all_times[~np.isnan(fish)][-1] / 60.)
        presence_freq.append(np.median(fish[~np.isnan(fish)]))

        if all_times[~np.isnan(fish)][0] / 60. >= 300.:
            appear_time_h.append(start_min + all_times[~np.isnan(fish)][0] / 60.)
        if all_times[~np.isnan(fish)][-1] / 60. <=  all_times[-1] / 60. - 300.:
            vanish_time_h.append(start_min + all_times[~np.isnan(fish)][-1] / 60.)

    for i in range(len(vanish_time)):
        if vanish_time[i] >= 1440:
            vanish_time[i] -= 1440
    for i in range(len(appear_time_h)):
        if appear_time_h[i] >= 1440:
            appear_time_h[i] -= 1440
    for i in range(len(vanish_time_h)):
        if vanish_time_h[i] >= 1440:
            vanish_time_h[i] -= 1440

    return np.array(presence_time), np.array(vanish_time), np.array(presence_freq), np.array(appear_time_h), np.array(vanish_time_h)


def get_rise_characteristic(fishes, all_rises, all_times):
    dt = all_times[1] - all_times[0]
    dpm = 60./dt

    m_rise_chars = []
    f_rise_chars = []
    df_male = []
    df_female = []

    for fish in range(len(fishes)):
        for rise in all_rises[fish]:
            if rise[1][0] - rise[1][1] < 1.:
                continue
            rise_idx = rise[0][1]
            rise_freq = rise[1][1]

            if rise_freq >= 730 and rise_freq < 1050:
                f0 = 730
                f1 = 1050
            elif rise_freq >= 550 and rise_freq < 730:
                f0 = 550
                f1 = 730
            else:
                continue

            idx0 = rise_idx - dpm if rise_idx - dpm >= 0 else 0
            idx1 = rise_idx + dpm

            upper_count = 0
            lower_count = 0
            dfs = []
            for enu, comp_fish in enumerate(fishes):
                if enu == fish:
                    continue
                if len(comp_fish[idx0:idx1][~np.isnan(comp_fish[idx0:idx1])]) > 1:
                    comp_fish_med = np.median(comp_fish[idx0:idx1][~np.isnan(comp_fish[idx0:idx1])])
                    if comp_fish_med >= f0 and comp_fish_med < f1:
                        dfs.append(np.diff([rise_freq, comp_fish_med])[0])
                        if comp_fish_med > rise_freq:
                            upper_count += 1
                        if comp_fish_med < rise_freq:
                            lower_count += 1
            dfs = np.array(dfs)
            if rise_freq >= 730 and rise_freq < 1050:
                m_rise_chars.append(np.array([upper_count, lower_count]))
                df_male.append(dfs)
            elif rise_freq >= 550 and rise_freq < 730:
                f_rise_chars.append(np.array([upper_count, lower_count]))
                df_female.append(dfs)

    m_rise_chars = np.array(m_rise_chars)
    f_rise_chars = np.array(f_rise_chars)

    all_m_at_rise = m_rise_chars[:, 0] + m_rise_chars[:, 1]
    m_below_ration = 1. * m_rise_chars[:, 1] / all_m_at_rise
    m_below_ration = m_below_ration[all_m_at_rise >= 5]

    all_f_at_rise = f_rise_chars[:, 0] + f_rise_chars[:, 1]
    f_below_ration = 1. * f_rise_chars[:, 1] / all_f_at_rise
    f_below_ration = f_below_ration[all_f_at_rise >= 5]

    bin_borders = np.arange(0, 1.1, 0.1)
    hist, bins = np.histogram(m_below_ration, bins = bin_borders)
    m_rise_proportions = hist

    hist, bins = np.histogram(f_below_ration, bins = bin_borders)
    f_rise_proportions = hist

    return m_rise_proportions, f_rise_proportions, np.array(df_male), np.array(df_female)


def plot_presence_time(all_presence_time, all_vanish_time, all_presence_freq, day_recording, start_dates, show_all_days = True):
    fs = 14
    colors = ['#BA2D22', '#F47F17', '#53379B', '#3673A4', '#AAB71B', '#DC143C', '#1E90FF', 'magenta']

    if show_all_days:
        day_night_rec = [np.arange(len(all_presence_time))[day_recording], np.arange(len(all_presence_time))[~day_recording]]

        for j in range(len(day_night_rec)):
            inch_factor = 2.54

            fig, ax = plt.subplots(facecolor='white', figsize=(20. / inch_factor, 12. / inch_factor))

            boxplots_day = []
            day_labels = []
            boxplots_night = []
            night_labels = []
            for enu, i in enumerate(day_night_rec[j]):
                presence_time_v = all_presence_time[i]
                vanish_time_v = all_vanish_time[i]
                presence_freq_v = all_presence_freq[i]

                male_day_presence_t = presence_time_v[(vanish_time_v > 360) & (vanish_time_v <= 1080) &
                                                      (presence_freq_v >= 730) & (presence_freq_v < 1050)]
                female_day_presence_t = presence_time_v[(vanish_time_v > 360) & (vanish_time_v <= 1080) &
                                                        (presence_freq_v >= 550) & (presence_freq_v < 730)]
                boxplots_day.append(male_day_presence_t)
                # day_labels.append('m_' + str(enu+1))
                boxplots_day.append(female_day_presence_t)
                # day_labels.append('f_' + str(enu + 1))
                day_labels.append(start_dates[i])

                male_night_presence_t = presence_time_v[(vanish_time_v <= 360) | (vanish_time_v > 1080) &
                                                        (presence_freq_v >= 730) & (presence_freq_v < 1050)]
                female_night_presence_t = presence_time_v[(vanish_time_v <= 360) | (vanish_time_v > 1080) &
                                                          (presence_freq_v >= 550) & (presence_freq_v < 730)]
                boxplots_night.append(male_night_presence_t)
                # night_labels.append('m_' + str(enu +1))
                boxplots_night.append(female_night_presence_t)
                # night_labels.append('f_' + str(enu +1))
                night_labels.append(start_dates[i])

            if j == 0:
                bp = ax.boxplot(boxplots_day, sym='', patch_artist=True)
                for enu, box in enumerate(bp['boxes']):
                    if enu % 2 == 0:
                        box.set(facecolor = colors[6])
                    else:
                        box.set(facecolor = colors[1])
                ax.set_xticks([1.5 + i*2 for i in range(13)])
                # day_labels = ['day %.0f' % (i+1) for i in range(13)]
                old_ticks = ax.get_xticks()
                plt.xticks(old_ticks, day_labels, rotation=70)
                plt.title('day recordings', fontsize= fs + 2)
            else:
                bp = ax.boxplot(boxplots_night, sym='', patch_artist=True)
                for enu, box in enumerate(bp['boxes']):
                    if enu % 2 == 0:
                        box.set(facecolor = colors[6])
                    else:
                        box.set(facecolor = colors[1])
                ax.set_xticks([1.5 + i*2 for i in range(15)])
                # night_labels = ['night %.0f' % (i + 1) for i in range(15)]
                old_ticks = ax.get_xticks()
                plt.xticks(old_ticks, night_labels, rotation=70)
                plt.title('night recordings', fontsize= fs + 2)
            ax.set_ylabel('presence time [min]', fontsize= fs)
            ax.tick_params(labelsize= fs - 2)

            plt.tight_layout()
        plt.show()


    presence_time_v = np.hstack(all_presence_time)
    vanish_time_v = np.hstack(all_vanish_time)
    presence_freq_v = np.hstack(all_presence_freq)

    day_presence_times = presence_time_v[(vanish_time_v > 360) & (vanish_time_v <= 1080)]
    night_presence_times = presence_time_v[(vanish_time_v <= 360) | (vanish_time_v > 1080)]

    male_day_presence_t = presence_time_v[(vanish_time_v > 360) & (vanish_time_v <= 1080) &
                                            (presence_freq_v >= 730) & (presence_freq_v < 1050)]
    female_day_presence_t = presence_time_v[(vanish_time_v > 360) & (vanish_time_v <= 1080) &
                                              (presence_freq_v >= 550) & (presence_freq_v < 730)]

    male_night_presence_t = presence_time_v[(vanish_time_v <= 360) | (vanish_time_v > 1080) &
                                              (presence_freq_v >= 730) & (presence_freq_v < 1050)]
    female_night_presence_t = presence_time_v[(vanish_time_v <= 360) | (vanish_time_v > 1080) &
                                                (presence_freq_v >= 550) & (presence_freq_v < 730)]

    _, dn_p = scp.mannwhitneyu(day_presence_times, night_presence_times)
    _, dnm_p = scp.mannwhitneyu(male_day_presence_t, male_night_presence_t)
    _, dnf_p = scp.mannwhitneyu(female_day_presence_t, female_night_presence_t)

    _, dmf_p = scp.mannwhitneyu(male_day_presence_t, female_day_presence_t)
    _, nmf_p = scp.mannwhitneyu(male_night_presence_t, female_night_presence_t)
    print ('\n Presence duration:')
    print('Day - Night - all:     p=%.3f' % (dn_p * 5.))
    print('Day - Night - male:    p=%.3f' % (dnm_p * 5.))
    print('Day - Night - female:  p=%.3f' % (dnf_p * 5.))
    print('Day - male - female:   p=%.3f' % (dmf_p * 5.))
    print('Night - male - female: p=%.3f' % (nmf_p * 5.))

    inch_factor = 2.54
    fig, ax = plt.subplots(facecolor='white', figsize=(20. / inch_factor, 12. / inch_factor))
    bp = ax.boxplot([day_presence_times, night_presence_times, male_day_presence_t, male_night_presence_t,
                     female_day_presence_t, female_night_presence_t], sym='', patch_artist=True)
    box_colors = ['white', 'grey', colors[6], colors[6], colors[1], colors[1]]
    for enu, box in enumerate(bp['boxes']):
        box.set(facecolor=box_colors[enu])

    ax.plot([1, 2], [700, 700], color='k', linewidth=2)
    ax.text(1.4, 712, '***', fontsize=fs)

    ax.plot([3, 4], [700, 700], color='k', linewidth=2)
    ax.text(3.4, 702, '***', fontsize=fs)

    ax.plot([5, 6], [700, 700], color='k', linewidth=2)
    ax.text(5.4, 702, '***', fontsize=fs)

    ax.plot([4, 6], [750, 750], color='k', linewidth=2)
    ax.text(4.9, 752, '**', fontsize=fs)

    old_ticks = ax.get_xticks()
    plt.xticks(old_ticks, ['day', 'night', 'male\nday', 'male\nnight', 'female\nday', 'female\nnight'])
    plt.ylabel('presence time [min]', fontsize= fs)
    # plt.title('presence duration', fontsize= fs + 2)
    ax.tick_params(labelsize = fs -2)
    ax.set_ylim([0, 775])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()

    plt.tight_layout()
    plt.show()


def plot_fishes(fishes, all_times, all_rises, temp, lux, start_time_str, start_date_str):

    # fig, ax = plt.subplots(facecolor='white', figsize=(11.6, 8.2), sharex=True)
    inch_factor = 2.54
    fs = 14
    fig = plt.figure(facecolor='white',  figsize=(20. / inch_factor, 14. / inch_factor))
    if temp != []:
        ax2 = fig.add_axes([0.1, 0.75, 0.8, 0.15])
        ax = fig.add_axes([0.1, 0.1, 0.8, 0.65])  # left bottom width height
    else:
        ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])


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
            if rise[1][0] - rise[1][1] > 1.:
                if legend_in == False:
                    ax.plot(all_times[rise[0][0]] / time_factor, rise[1][0], 'o', color='red', markersize=7,
                            markerfacecolor='None', label='rise peak')
                    ax.plot(all_times[rise[0][1]] / time_factor, rise[1][1], 's', color='green', markersize=7,
                            markerfacecolor='None', label='rise end')
                    legend_in = True
                    ax.legend(loc=1, numpoints=1, frameon=False, fontsize=fs-4)
                else:
                    ax.plot(all_times[rise[0][0]] / time_factor, rise[1][0], 'o', color='red', markersize=7,
                            markerfacecolor='None')
                    ax.plot(all_times[rise[0][1]] / time_factor, rise[1][1], 's', color='green', markersize=7,
                            markerfacecolor='None')

    ax.set_ylim([450, 950])
    ax.set_ylabel('frequency [Hz]', fontsize=fs)
    ax.yaxis.set_label_coords(-0.08, 0.5)
    if time_factor == 1.:
        plt.xlabel('time [sec]', fontsize=fs)
    elif time_factor == 60.:
        plt.xlabel('time [min]', fontsize=fs)
    else:
        plt.xlabel('time [h]', fontsize=fs)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()

    if start_time_str:
        old_ticks = ax.get_xticks()
        start_time = start_time_str.split(':')

        if time_factor == 3600.:
            new_ticks = ['%02.0f:%02.0f' % ((float(old_ticks[i]) + float(start_time[0])) % 24,  float(start_time[1]) ) for i in range(len(old_ticks))]
            # new_ticks = [str(int((int(start_time[0])+ old_ticks[i]) % 24)) + ':' + str(int(start_time[1])) for i in range(len(old_ticks))]
            plt.xticks(old_ticks, new_ticks)
            ax.set_xlabel('time', fontsize= fs)
            ax.tick_params(labelsize = fs-2)

    # temp plot
    if temp != []:
        ax2.plot(all_times / time_factor, temp, '-', color='red', linewidth=2)
        ax2.set_ylim([24, 27])
        ax2.set_ylabel('temp [$^\circ$C]', fontsize=fs)
        ax2.yaxis.set_label_coords(-0.08, 0.5)
        # ax2.legend(loc=1, numpoints=1, frameon=False, fontsize=12)

        ax2.get_xaxis().set_visible(False)
        ax2.spines['top'].set_visible(False)
        ax2.spines['bottom'].set_visible(False)
        ax2.set_yticks([24, 25, 26, 27])
        ax2.tick_params(bottom='off', top='off', right='off', labelsize=fs - 2)

        ax2.spines['right'].set_visible(False)

    fig.suptitle(start_date_str, fontsize = fs + 2)
    plt.show()

    if start_date_str == '09/04/16':
        # rd = np.random.RandomState(200000)
        rd = np.random.RandomState(200010)
        colors = ['#BA2D22', '#F47F17', '#53379B', '#3673A4', '#AAB71B', '#DC143C', '#1E90FF']
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharex=True, facecolor='white',
                                                     figsize=(20. / inch_factor, 12. / inch_factor))
        axx = [ax1, ax2, ax3, ax4]
        xx = [23400, 25700, 20700, 28700]
        yy = [585, 802, 877, 577]
        for enu, ax in enumerate(axx):
            for fish in range(len(fishes)):
                color = np.random.rand(3, 1)
                ax.plot(all_times[~np.isnan(fishes[fish])] - xx[enu], fishes[fish][~np.isnan(fishes[fish])],
                        color=colors[rd.randint(0, len(colors))], marker='.')
            ax.set_ylim([yy[enu], yy[enu]+ 15.])
            # ax.set_xlim([xx[enu], xx[enu]+ 800.])
            ax.set_xlim([0, 800])
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.get_xaxis().tick_bottom()
            ax.get_yaxis().tick_left()
            ax.set_xticks([0, 200, 400, 600, 800])
            ax.tick_params(labelsize=fs - 2)
        fig.subplots_adjust(hspace=0.2, wspace=0.2)
        ax1.set_ylabel('frequency [Hz]', fontsize=fs)
        ax3.set_ylabel('frequency [Hz]', fontsize=fs)

        ax3.set_xlabel('time [sec]', fontsize=fs)
        ax4.set_xlabel('time [sec]', fontsize=fs)

        plt.show()
        # embed()
        # quit()



def plot_freq_vs_temp(fishes, temp):
    # embed()
    for fish in fishes:
        fig, ax = plt.subplots()
        ax.scatter(temp[~np.isnan(fish)], fish[~np.isnan(fish)])
        plt.show()


def comb_fish_plot(ax_all, fishes, all_times, start_time_str, last_end_time, last_t, old_colors):
    new_colors = []
    if not last_end_time:
        start_t = 0
    else:
        last_end_time = [int(last_end_time.split(':')[n]) for n in range(2)]
        start_time = [int(start_time_str.split(':')[n]) for n in range(2)]

        if start_time[1] < last_end_time[1]:
            start_time[1] += 60
            last_end_time[0] += 1
        if start_time[0] < last_end_time[0]:
            start_time[0] += 24

        ds = (start_time[1] - last_end_time[1]) * 60. + (start_time[0] - last_end_time[0]) * 3600.

        start_t = last_t + ds

    for fish in range(len(fishes)):
        if all_times[~np.isnan(fishes[fish])][0] - all_times[0] <= 1800:
            if old_colors and old_colors != []:
                # embed()
                freq_diff = np.array([old_colors[n][0] for n in range(len(old_colors))]) - fishes[fish][~np.isnan(fishes[fish])][0]
                sorted_idx = np.argsort(np.abs(freq_diff))
                # embed()
                for min_freq_diff_idx in sorted_idx:
                    if freq_diff[min_freq_diff_idx] <= 2.:
                        color = old_colors[min_freq_diff_idx][1]
                        old_colors.pop(min_freq_diff_idx)
                        break
                    else:
                        color = np.random.rand(3, 1)
            else:
                color = np.random.rand(3, 1)
        else:
            color = np.random.rand(3, 1)

        ax_all.plot((all_times[~np.isnan(fishes[fish])] + start_t) / 3600, fishes[fish][~np.isnan(fishes[fish])],
                    color=color, marker='.', zorder=2)

        if all_times[-1] - all_times[~np.isnan(fishes[fish])][-1] <= 1800:
            new_colors.append([fishes[fish][~np.isnan(fishes[fish])][-1], color])

    last_t = all_times[-1] + start_t
    return last_t, new_colors


def plot_rise_phnf(rise_freq_counts, rise_base_freq, rise_counts, bins):
    # embed()
    # quit()
    colors = ['#BA2D22', '#F47F17', '#53379B', '#3673A4', '#AAB71B', '#DC143C', '#1E90FF', 'magenta']

    # male plot
    inch_factor = 2.54
    fig, ax = plt.subplots(facecolor='white', figsize=(20. / inch_factor, 12. / inch_factor))
    bp = ax.boxplot(np.array(rise_freq_counts)[(bins >= 730) & (bins < 1050)], sym='', patch_artist=True)
    for box in bp['boxes']:
        box.set(facecolor=colors[6])
    old_ticks = ax.get_xticks()
    plt.xticks(old_ticks, bins[(bins >= 730) & (bins < 1050)], rotation=45)

    male_base_freq = rise_base_freq[(rise_base_freq >= 730) & (rise_base_freq < 1050)]
    male_rise_count = rise_counts[(rise_base_freq >= 730) & (rise_base_freq < 1050)]
    r_val, p_val = scp.pearsonr(male_base_freq, male_rise_count)
    # r_val, p_val = scp.spearmanr(male_base_freq, male_rise_count)


    ax.set_ylim([0, 8])
    ax.set_xlabel('frequency [Hz]')
    ax.set_ylabel('rise count per h [n/h]')
    ax.set_title('Risecounts vs. frequency in Males \n(spearmanr; r= %.3f; p= %.3f)' % (r_val, p_val *2.))
    plt.tight_layout()

    # female plot
    inch_factor = 2.54
    fig, ax = plt.subplots(facecolor='white', figsize=(20. / inch_factor, 12. / inch_factor))
    bp = ax.boxplot(np.array(rise_freq_counts)[(bins >= 550) & (bins < 730)], sym='', patch_artist=True)
    for box in bp['boxes']:
        box.set(facecolor=colors[1])
    old_ticks = ax.get_xticks()
    plt.xticks(old_ticks, bins[(bins >= 550) & (bins < 730)], rotation=45)

    female_base_freq = rise_base_freq[(rise_base_freq >= 550) & (rise_base_freq < 730)]
    female_rise_count = rise_counts[(rise_base_freq >= 550) & (rise_base_freq < 730)]
    r_val, p_val = scp.pearsonr(female_base_freq, female_rise_count)
    # r_val, p_val = scp.spearmanr(female_base_freq, female_rise_count)

    ax.set_ylim([0, 8])
    ax.set_xlabel('frequency [Hz]')
    ax.set_ylabel('rise count per h [n/h]')
    ax.set_title('Risecounts vs. frequency in Females \n(spearmanr; r= %.3f; p= %.3f)' % (r_val, p_val *2.))
    plt.tight_layout()

    # male female plot
    f_rise_counts = rise_counts[(rise_base_freq >= 550) & (rise_base_freq < 730)]
    m_rise_counts = rise_counts[(rise_base_freq >= 730) & (rise_base_freq < 1050)]

    _, p = scp.mannwhitneyu(m_rise_counts, f_rise_counts)
    print('\n rise counts per hour')
    print('male - female: p = %.3f' % p)

    box_colors = [colors[6], colors[1]]
    fig, ax = plt.subplots(facecolor='white', figsize=(20. / inch_factor, 12. / inch_factor))
    bp = ax.boxplot([m_rise_counts, f_rise_counts], sym='', patch_artist=True)
    for enu, box in enumerate(bp['boxes']):
        box.set(facecolor=box_colors[enu])

    ax.plot([1, 2], [6, 6], color='k', linewidth=2)
    ax.text(1.47, 6, '***', fontsize=14)

    ax.set_ylabel('rise count per h [n/h]')
    ax.set_ylim([0, 6.5])
    old_ticks = ax.get_xticks()
    plt.xticks(old_ticks, ['male', 'female'])
    ax.set_ylabel('rise counts per h')
    plt.tight_layout()


    ### new plot ###
    # embed()
    # quit()
    fs = 14
    fig, ax = plt.subplots(facecolor='white', figsize=(20. / inch_factor, 12. / inch_factor))
    bp = ax.boxplot(np.array(rise_freq_counts)[(bins >= 550) & (bins < 980)], sym='', patch_artist=True)
    for enu, box in enumerate(bp['boxes']):
        if enu >= 18:
            box.set(facecolor=colors[6])
        else:
            box.set(facecolor=colors[1])

    ax.set_xticks([3 + i*2.5 for i in range(17)])
    new_ticks = [str(575 + i*25) for i in range(17)]
    ax.set_xticklabels(new_ticks, fontsize= fs - 2)

    ax.set_xlabel('EOD$f$ at 25$^{\circ}C$ [Hz]')
    ax.set_ylabel('rate rate [n/h]')

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()

    ax.tick_params(labelsize = fs -2)
    ax.set_ylim([0, 6])

    plt.tight_layout()
    plt.show()
    # embed()
    # quit()


def plot_clock_fish_counts(all_clock_counts, all_m_clock_counts, all_f_clock_counts, full_clock, all_mean_clock_temp):
    ### combined fishcount per time plot ###
    all_clock_counts = np.array(all_clock_counts)
    all_m_clock_counts = np.array(all_m_clock_counts)
    all_f_clock_counts = np.array(all_f_clock_counts)

    all_mean_clock_temp = np.array(all_mean_clock_temp)

    mean_clock_counts = np.full(len(full_clock), np.nan)
    std_clock_counts = np.full(len(full_clock), np.nan)

    mean_m_clock_counts = np.full(len(full_clock), np.nan)
    std_m_clock_counts = np.full(len(full_clock), np.nan)

    mean_f_clock_counts = np.full(len(full_clock), np.nan)
    std_f_clock_counts = np.full(len(full_clock), np.nan)

    mean_clock_temp = np.full(len(full_clock), np.nan)
    std_clock_temp = np.full(len(full_clock), np.nan)

    for i in range(len(full_clock)):
        clock_fish_counts = all_clock_counts[:, i]
        clock_m_counts = all_m_clock_counts[:, i]
        clock_f_counts = all_f_clock_counts[:, i]
        clock_temp = all_mean_clock_temp[:, i]

        mean_clock_counts[i] = np.mean(clock_fish_counts[~np.isnan(clock_fish_counts)])
        std_clock_counts[i] = np.std(clock_fish_counts[~np.isnan(clock_fish_counts)], ddof= 1)

        mean_m_clock_counts[i] = np.mean(clock_m_counts[~np.isnan(clock_m_counts)])
        std_m_clock_counts[i] = np.std(clock_m_counts[~np.isnan(clock_m_counts)], ddof=1)

        mean_f_clock_counts[i] = np.mean(clock_f_counts[~np.isnan(clock_f_counts)])
        std_f_clock_counts[i] = np.std(clock_f_counts[~np.isnan(clock_f_counts)], ddof=1)

        mean_clock_temp[i] = np.mean(clock_temp[~np.isnan(clock_temp)])
        std_clock_temp[i] = np.std(clock_temp[~np.isnan(clock_temp)], ddof= 1)

    # embed()
    # quit()

    inch_factor = 2.54
    fs = 14
    colors = ['#BA2D22', '#F47F17', '#53379B', '#3673A4', '#AAB71B', '#DC143C', '#1E90FF', 'magenta']

    fig_c, (ax_c, ax2) = plt.subplots(2, 1, sharex=True, facecolor='white', figsize=(20. / inch_factor, 14. / inch_factor))

    ax_c.plot(np.arange(len(mean_clock_counts)) / 2., mean_clock_counts, marker='.', color=colors[4], linewidth = 2)
    ax_c.fill_between(np.arange(len(mean_clock_counts)) / 2., mean_clock_counts + std_clock_counts,
                      mean_clock_counts - std_clock_counts, color=colors[4], alpha=0.4)
    ax_c.fill_between([0, 5.75], [8, 8], [24, 24], color='grey', alpha=0.4, zorder=1)
    ax_c.fill_between([17.75, 23.5], [8, 8], [24, 24], color='grey', alpha=0.4, zorder=1)
    ax_c.set_ylabel('n', fontsize = fs)

    ax_c.spines['top'].set_visible(False)
    ax_c.spines['right'].set_visible(False)
    ax_c.get_xaxis().tick_bottom()
    ax_c.get_yaxis().tick_left()

    ax2.plot(np.arange(len(mean_m_clock_counts)) / 2., mean_m_clock_counts, marker='.', color= colors[6], linewidth = 2)
    ax2.fill_between(np.arange(len(mean_m_clock_counts)) / 2., mean_m_clock_counts + std_m_clock_counts,
                      mean_m_clock_counts - std_m_clock_counts, color=colors[6], alpha=0.4)

    ax2.plot(np.arange(len(mean_f_clock_counts)) / 2., mean_f_clock_counts, marker='.', color= colors[1], linewidth = 2)
    ax2.fill_between(np.arange(len(mean_f_clock_counts)) / 2., mean_f_clock_counts + std_f_clock_counts,
                     mean_f_clock_counts - std_f_clock_counts, color=colors[1], alpha=0.4)
    ax2.fill_between([0, 5.75], [0, 0], [14, 14], color='grey', alpha=0.4, zorder=1)
    ax2.fill_between([17.75, 23.5], [0, 0], [14, 14], color='grey', alpha=0.4, zorder=1)

    ax2.set_xlim([0, 23.5])
    ax2.set_xlabel('time', fontsize=fs)
    ax2.set_ylabel('n', fontsize=fs)

    plt.xticks([0.75 + i for i in range(23)], ['%02.0f:00' % i for i in np.arange(23) + 1], rotation=70)
    ax2.tick_params(labelsize= fs -2)

    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.get_xaxis().tick_bottom()
    ax2.get_yaxis().tick_left()

    # plt.suptitle('Fishcounts', fontsize = fs + 2)
    plt.tight_layout()
    # plt.show(fig_c)

    ### box plot ####
    day_idx = np.arange(len(full_clock))[np.array([(int(full_clock[i].split(':')[0]) >= 6) & (int(full_clock[i].split(':')[0]) < 18) for i in range(len(full_clock))])]
    night_idx = np.setdiff1d(np.arange(len(full_clock)), day_idx)

    day_counts = np.hstack(all_clock_counts[:, day_idx])
    night_counts = np.hstack(all_clock_counts[:, night_idx])

    day_m_counts = np.hstack(all_m_clock_counts[:, day_idx])
    day_f_counts = np.hstack(all_f_clock_counts[:, day_idx])

    night_m_counts = np.hstack(all_m_clock_counts[:, night_idx])
    night_f_counts = np.hstack(all_f_clock_counts[:, night_idx])

    _, dn_p = scp.mannwhitneyu(day_counts[~np.isnan(day_counts)], night_counts[~np.isnan(night_counts)])
    _, dnm_p = scp.mannwhitneyu(day_m_counts[~np.isnan(day_m_counts)], night_m_counts[~np.isnan(night_m_counts)])
    _, dnf_p = scp.mannwhitneyu(day_f_counts[~np.isnan(day_f_counts)], night_f_counts[~np.isnan(night_f_counts)])

    _, dmf_p = scp.mannwhitneyu(day_m_counts[~np.isnan(day_m_counts)], day_f_counts[~np.isnan(day_f_counts)])
    _, nmf_p = scp.mannwhitneyu(night_m_counts[~np.isnan(night_m_counts)], night_f_counts[~np.isnan(night_f_counts)])

    print('\n Fish counts:')
    print('Day - Night - all:     p = %.3f' % (dn_p * 5.))
    print('Day - Night - males:   p = %.3f' % (dnm_p * 5.))
    print('Day - Night - female:  p = %.3f' % (dnf_p * 5.))
    print('Day - male - female:   p = %.3f' % (dmf_p * 5.))
    print('Night - male - female: p = %.3f' % (nmf_p * 5.))

    # embed()
    # quit()
    fig, ax = plt.subplots(facecolor='white', figsize=(20. / inch_factor, 12. / inch_factor))
    bp = ax.boxplot([day_counts[~np.isnan(day_counts)], night_counts[~np.isnan(night_counts)],
                     day_m_counts[~np.isnan(day_m_counts)], night_m_counts[~np.isnan(night_m_counts)],
                     day_f_counts[~np.isnan(day_f_counts)], night_f_counts[~np.isnan(night_f_counts)]],
                    sym = '', patch_artist=True)
    # ax.text(0.9, np.median(day_counts[~np.isnan(day_counts)]), '%.0f' % np.median(day_counts[~np.isnan(day_counts)]), fontsize=fs)

    box_colors = ['white', 'grey', colors[6], colors[6], colors[1], colors[1]]
    for enu, box in enumerate(bp['boxes']):
        box.set(facecolor=box_colors[enu])

    ax.plot([1, 2], [26, 26], color='k', linewidth=2)
    ax.text(1.4, 26.1, '*', fontsize=fs)

    ax.plot([3, 4], [26, 26], color='k', linewidth=2)
    ax.text(3.4, 26.1, '***', fontsize=fs)

    ax.plot([3, 5], [28, 28], color='k', linewidth=2)
    ax.text(3.9, 28.1, '***', fontsize=fs)

    ax.plot([4, 6], [30, 30], color='k', linewidth=2)
    ax.text(4.9, 30.1, '***', fontsize=fs)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()

    old_ticks = ax.get_xticks()
    plt.xticks(old_ticks, ['day', 'night', 'male\nday', 'male\nnight', 'female\nday', 'female\nnight'])

    ax.set_ylabel('n')
    ax.set_ylim([0, 32])
    plt.show()


def plot_clock_rise_counts(all_rise_clock_counts, all_m_rise_clock_counts, all_f_rise_clock_counts, all_clock_counts,
                           all_m_clock_counts, all_f_clock_counts, all_rise_rate_clock, all_m_rise_rate_clock,
                           all_f_rise_rate_clock, full_clock):

    all_rise_rate_clock = np.array(all_rise_rate_clock)
    all_m_rise_rate_clock = np.array(all_m_rise_rate_clock)
    all_f_rise_rate_clock = np.array(all_f_rise_rate_clock)

    all_rise_clock_counts = np.array(all_rise_clock_counts) * 2.
    all_m_rise_clock_counts = np.array(all_m_rise_clock_counts) * 2.
    all_f_rise_clock_counts = np.array(all_f_rise_clock_counts) * 2.

    all_clock_counts = np.array(all_clock_counts)
    all_m_clock_counts = np.array(all_m_clock_counts)
    all_f_clock_counts = np.array(all_f_clock_counts)

    mean_rise_per_fish_clock_counts = np.full(len(full_clock), np.nan)
    mean_m_rise_per_fish_clock_counts = np.full(len(full_clock), np.nan)
    mean_f_rise_per_fish_clock_counts = np.full(len(full_clock), np.nan)

    std_rise_per_fish_clock_counts = np.full(len(full_clock), np.nan)
    std_m_rise_per_fish_clock_counts = np.full(len(full_clock), np.nan)
    std_f_rise_per_fish_clock_counts = np.full(len(full_clock), np.nan)

    inch_factor = 2.54
    colors = ['#BA2D22', '#F47F17', '#53379B', '#3673A4', '#AAB71B', '#DC143C', '#1E90FF', 'magenta']

    fig = plt.figure(facecolor='white', figsize=(20. / inch_factor, 20. / inch_factor))
    ax1 = fig.add_subplot(311)
    ax2 = fig.add_subplot(334)
    ax3 = fig.add_subplot(335)
    ax4 = fig.add_subplot(336)

    ax5 = fig.add_subplot(337)
    ax6 = fig.add_subplot(338)
    ax7 = fig.add_subplot(339)

    fish_counts = np.hstack(all_clock_counts)[~np.isnan(np.hstack(all_clock_counts))]
    male_counts = np.hstack(all_m_clock_counts)[~np.isnan(np.hstack(all_m_clock_counts))]
    female_counts = np.hstack(all_f_clock_counts)[~np.isnan(np.hstack(all_f_clock_counts))]

    rise_counts = np.hstack(all_rise_clock_counts)[~np.isnan(np.hstack(all_rise_clock_counts))]
    male_r_counts = np.hstack(all_m_rise_clock_counts)[~np.isnan(np.hstack(all_m_rise_clock_counts))]
    female_r_counts = np.hstack(all_f_rise_clock_counts)[~np.isnan(np.hstack(all_f_rise_clock_counts))]

    rise_rate = np.hstack(all_rise_rate_clock)[~np.isnan(np.hstack(all_rise_rate_clock))]
    m_rise_rate = np.hstack(all_m_rise_rate_clock)[~np.isnan(np.hstack(all_m_rise_rate_clock))]
    f_rise_rate = np.hstack(all_f_rise_rate_clock)[~np.isnan(np.hstack(all_f_rise_rate_clock))]

    # embed()
    # quit()

    # heatmap, xedges, yedges = np.histogram2d(fish_counts, rise_counts / fish_counts)
    heatmap, xedges, yedges = np.histogram2d(fish_counts, rise_rate)
    v0 = 0
    v1 = np.max(heatmap)

    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    # ax1.imshow(heatmap.T, extent=extent, origin='lower', aspect='auto', vmin= v0, vmax=v1, cmap='afmhot')
    ax1.imshow(heatmap.T, extent=extent, origin='lower', aspect='auto', vmin= v0, vmax=v1, cmap='Oranges')
    # ax1.scatter(fish_counts, rise_counts / fish_counts, color=colors[4], alpha=0.5, s=1)
    ax1.scatter(fish_counts, rise_rate, color='k', alpha=0.5, s=1)
    # sl, interc, r, p, _ = scp.linregress(fish_counts, rise_counts / fish_counts)
    sl, interc, r, p, _ = scp.linregress(fish_counts, rise_rate)
    # r, p = scp.spearmanr(fish_counts, rise_counts / fish_counts)
    r, p = scp.spearmanr(fish_counts, rise_rate)

    print('\n rise rate vs. fish count')
    print ('a rise -- a #: p = %.3f; r = %.2f' % (p * 7., r))
    if p < 0.05:
        ax1.plot([xedges[0], xedges[-1]], interc + sl * np.array([xedges[0], xedges[-1]]), color='red')
        ax1.set_xlim([xedges[0], xedges[-1]])
        ax1.set_ylim([yedges[0], yedges[-1]])
    ax1.set_ylabel('rise rate [n/h]')
    ax1.set_xlabel('total fish count')

    # heatmap, xedges, yedges = np.histogram2d(male_counts, male_r_counts / male_counts)
    heatmap, xedges, yedges = np.histogram2d(male_counts, m_rise_rate)
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    ax2.imshow(heatmap.T, extent=extent, origin='lower', aspect='auto', vmin= v0, vmax=v1, cmap='Oranges')
    # ax2.scatter(male_counts, male_r_counts / male_counts, color=colors[6], alpha=0.5, s=1)
    ax2.scatter(male_counts, m_rise_rate, color='k', alpha=0.5, s=1)
    # sl, interc, r, p, _ = scp.linregress(male_counts, male_r_counts / male_counts)
    sl, interc, r, p, _ = scp.linregress(male_counts, m_rise_rate)
    # r, p = scp.spearmanr(male_counts, male_r_counts / male_counts)
    r, p = scp.spearmanr(male_counts, m_rise_rate)
    print ('m rise -- m #: p = %.3f; r = %.2f' % (p * 7., r))
    if p < 0.05:
        ax2.plot([xedges[0], xedges[-1]], interc + sl * np.array([xedges[0], xedges[-1]]), color='red')
        ax2.set_xlim([xedges[0], xedges[-1]])
        ax2.set_ylim([yedges[0], yedges[-1]])
    ax2.set_ylabel('male rise rate [n/h]')
    ax2.get_xaxis().set_visible(False)

    # heatmap, xedges, yedges = np.histogram2d(female_counts, male_r_counts / male_counts)
    heatmap, xedges, yedges = np.histogram2d(female_counts, m_rise_rate)
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    ax3.imshow(heatmap.T, extent=extent, origin='lower', aspect='auto', vmin= v0, vmax=v1, cmap='Oranges')
    # ax3.scatter(female_counts, male_r_counts / male_counts, color = colors[6], alpha = 0.5, s=1)
    ax3.scatter(female_counts, m_rise_rate, color ='k', alpha = 0.5, s=1)
    # sl, interc, r, p, _ = scp.linregress(female_counts, male_r_counts / male_counts)
    sl, interc, r, p, _ = scp.linregress(female_counts, m_rise_rate)
    # r, p = scp.spearmanr(female_counts, male_r_counts / male_counts)
    r, p = scp.spearmanr(female_counts, m_rise_rate)
    print ('m rise -- f #: p = %.3f; r = %.2f' % (p * 7., r))
    if p < 0.05:
        ax3.plot([xedges[0], xedges[-1]], interc + sl * np.array([xedges[0], xedges[-1]]), color='red')
        ax3.set_xlim([xedges[0], xedges[-1]])
        ax3.set_ylim([yedges[0], yedges[-1]])
    ax3.get_xaxis().set_visible(False)
    ax3.get_yaxis().set_visible(False)

    # heatmap, xedges, yedges = np.histogram2d(fish_counts, male_r_counts / male_counts)
    heatmap, xedges, yedges = np.histogram2d(fish_counts, m_rise_rate)
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    ax4.imshow(heatmap.T, extent=extent, origin='lower', aspect='auto', vmin= v0, vmax=v1, cmap='Oranges')
    # ax4.scatter(fish_counts, male_r_counts / male_counts, color = colors[6], alpha = 0.5, s=1)
    ax4.scatter(fish_counts, m_rise_rate, color ='k', alpha = 0.5, s=1)
    # sl, interc, r, p, _ = scp.linregress(fish_counts, male_r_counts / male_counts)
    sl, interc, r, p, _ = scp.linregress(fish_counts, m_rise_rate)
    # r, p = scp.spearmanr(fish_counts, male_r_counts / male_counts)
    r, p = scp.spearmanr(fish_counts, m_rise_rate)
    print ('m rise -- a #: p = %.3f; r = %.2f' % (p * 7., r))
    if p < 0.05:
        ax4.plot([xedges[0], xedges[-1]], interc + sl * np.array([xedges[0], xedges[-1]]), color='red')
        ax4.set_xlim([xedges[0], xedges[-1]])
        ax4.set_ylim([yedges[0], yedges[-1]])
    ax4.get_xaxis().set_visible(False)
    ax4.get_yaxis().set_visible(False)

    # heatmap, xedges, yedges = np.histogram2d(male_counts, female_r_counts / female_counts)
    heatmap, xedges, yedges = np.histogram2d(male_counts, f_rise_rate)
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    ax5.imshow(heatmap.T, extent=extent, origin='lower', aspect='auto', vmin= v0, vmax=v1, cmap='Oranges')
    # ax5.scatter(male_counts, female_r_counts / female_counts, color = colors[1], alpha = 0.5, s=1)
    ax5.scatter(male_counts, f_rise_rate, color ='k', alpha = 0.5, s=1)
    # sl, interc, r, p, _ = scp.linregress(male_counts, female_r_counts / female_counts)
    sl, interc, r, p, _ = scp.linregress(male_counts, f_rise_rate)
    # r, p = scp.spearmanr(male_counts, female_r_counts / female_counts)
    r, p = scp.spearmanr(male_counts, f_rise_rate)
    print ('f rise -- m #: p = %.3f; r = %.2f' % (p * 7., r))
    if p < 0.05:
        ax5.plot([xedges[0], xedges[-1]], interc + sl * np.array([xedges[0], xedges[-1]]), color='red')
        ax5.set_xlim([xedges[0], xedges[-1]])
        ax5.set_ylim([yedges[0], yedges[-1]])
    ax5.set_ylabel('female rise rate [n/h]')
    ax5.set_xlabel('male counts')

    # heatmap, xedges, yedges = np.histogram2d(female_counts, female_r_counts / female_counts)
    heatmap, xedges, yedges = np.histogram2d(female_counts, f_rise_rate)
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    ax6.imshow(heatmap.T, extent=extent, origin='lower', aspect='auto', vmin= v0, vmax=v1, cmap='Oranges')
    # ax6.scatter(female_counts, female_r_counts / female_counts, color = colors[1], alpha = 0.5, s=1)
    ax6.scatter(female_counts, f_rise_rate, color ='k', alpha = 0.5, s=1)
    # sl, interc, r, p, _ = scp.linregress(female_counts, female_r_counts / female_counts)
    sl, interc, r, p, _ = scp.linregress(female_counts, f_rise_rate)
    # r, p = scp.spearmanr(female_counts, female_r_counts / female_counts)
    r, p = scp.spearmanr(female_counts, f_rise_rate)
    print ('f rise -- f #: p = %.3f; r = %.2f' % (p * 7., r))
    if p < 0.05:
        ax6.plot([xedges[0], xedges[-1]], interc + sl * np.array([xedges[0], xedges[-1]]), color='red')
        ax6.set_xlim([xedges[0], xedges[-1]])
        ax6.set_ylim([yedges[0], yedges[-1]])
    ax6.set_xlabel('female counts')
    ax6.get_yaxis().set_visible(False)

    # heatmap, xedges, yedges = np.histogram2d(fish_counts, female_r_counts / female_counts)
    heatmap, xedges, yedges = np.histogram2d(fish_counts, f_rise_rate)
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    ax7.imshow(heatmap.T, extent=extent, origin='lower', aspect='auto', vmin= v0, vmax=v1, cmap='Oranges')
    # ax7.scatter(fish_counts, female_r_counts / female_counts, color = colors[1], alpha = 0.5, s=1)
    ax7.scatter(fish_counts, f_rise_rate, color ='k', alpha = 0.5, s=1)
    # sl, interc, r, p, _ = scp.linregress(fish_counts, female_r_counts / female_counts)
    sl, interc, r, p, _ = scp.linregress(fish_counts, f_rise_rate)
    # r, p = scp.spearmanr(fish_counts, female_r_counts / female_counts)
    r, p = scp.spearmanr(fish_counts, f_rise_rate)
    print ('f rise -- a #: p = %.3f; r = %.2f' % (p * 7., r))
    if p < 0.05:
        ax7.plot([xedges[0], xedges[-1]], interc + sl * np.array([xedges[0], xedges[-1]]), color='red')
        ax7.set_xlim([xedges[0], xedges[-1]])
        ax7.set_ylim([yedges[0], yedges[-1]])
    ax7.set_xlabel('total fish counts')
    ax7.get_yaxis().set_visible(False)

    for ax in [ax1, ax2, ax3, ax4, ax5, ax6, ax7]:
        ax.set_ylim([0, 3])

    plt.subplots_adjust(hspace = 0.2, wspace=0.2)
    plt.tight_layout()
    plt.show()

    # embed()
    # quit()

    for i in range(len(full_clock)):
        clock_rise_counts = all_rise_clock_counts[:, i]
        clock_m_rise_counts = all_m_rise_clock_counts[:, i]
        clock_f_rise_counts = all_f_rise_clock_counts[:, i]

        clock_fish_counts = all_clock_counts[:, i]
        clock_male_counts = all_m_clock_counts[:, i]
        clock_female_counts = all_f_clock_counts[:, i]

        # rise_count_per_fish = clock_rise_counts / clock_fish_counts
        # m_rise_count_per_fish = clock_m_rise_counts / clock_male_counts
        # f_rise_count_per_fish = clock_f_rise_counts / clock_female_counts

        rise_count_per_fish = all_rise_rate_clock[:, i]
        m_rise_count_per_fish = all_m_rise_rate_clock[:, i]
        f_rise_count_per_fish = all_f_rise_rate_clock[:, i]

        mean_rise_per_fish_clock_counts[i] = np.mean(rise_count_per_fish[~np.isnan(rise_count_per_fish)])
        std_rise_per_fish_clock_counts[i] = np.std(rise_count_per_fish[~np.isnan(rise_count_per_fish)], ddof= 1)

        mean_m_rise_per_fish_clock_counts[i] = np.mean(m_rise_count_per_fish[~np.isnan(m_rise_count_per_fish)])
        std_m_rise_per_fish_clock_counts[i] = np.std(m_rise_count_per_fish[~np.isnan(m_rise_count_per_fish)], ddof=1)

        mean_f_rise_per_fish_clock_counts[i] = np.mean(f_rise_count_per_fish[~np.isnan(f_rise_count_per_fish)])
        std_f_rise_per_fish_clock_counts[i] = np.std(f_rise_count_per_fish[~np.isnan(f_rise_count_per_fish)], ddof=1)

    inch_factor = 2.54
    fs = 14
    colors = ['#BA2D22', '#F47F17', '#53379B', '#3673A4', '#AAB71B', '#DC143C', '#1E90FF', 'magenta']

    # fig_c, ax_c = plt.subplots(facecolor='white', figsize=(20. / inch_factor, 12. / inch_factor))
    fig_c, (ax_c, ax2) = plt.subplots(2, 1, sharex=True, facecolor='white',
                                      figsize=(20. / inch_factor, 14. / inch_factor))
    ax_c.plot(np.arange(len(mean_rise_per_fish_clock_counts)) / 2., mean_rise_per_fish_clock_counts, marker='.', color=colors[4], linewidth=2)
    ax_c.fill_between(np.arange(len(mean_rise_per_fish_clock_counts)) / 2., mean_rise_per_fish_clock_counts + std_rise_per_fish_clock_counts,
                      mean_rise_per_fish_clock_counts - std_rise_per_fish_clock_counts, color=colors[4], alpha=0.4)
    ax_c.fill_between([0, 5.75], [0, 0], [6, 6], color='grey', alpha=0.4, zorder=1)
    ax_c.fill_between([17.75, 23.5], [0, 0], [6, 6], color='grey', alpha=0.4, zorder=1)

    # ax_c.set_ylabel('n', fontsize=fs)
    # ax_c.yaxis.set_label_coords(-0.05, 0.5)
    ax_c.set_xlim([0, 23.5])
    ax_c.set_ylabel('rise rate [n/h]', fontsize=fs)

    ax_c.spines['top'].set_visible(False)
    ax_c.spines['right'].set_visible(False)
    ax_c.get_xaxis().tick_bottom()
    ax_c.get_yaxis().tick_left()

    ax2.plot(np.arange(len(mean_m_rise_per_fish_clock_counts)) / 2., mean_m_rise_per_fish_clock_counts, marker='.', color = colors[6], linewidth = 2)
    ax2.fill_between(np.arange(len(mean_m_rise_per_fish_clock_counts)) / 2., mean_m_rise_per_fish_clock_counts + std_m_rise_per_fish_clock_counts,
                     mean_m_rise_per_fish_clock_counts - std_m_rise_per_fish_clock_counts, color=colors[6], alpha=0.4)

    ax2.plot(np.arange(len(mean_f_rise_per_fish_clock_counts)) / 2., mean_f_rise_per_fish_clock_counts, marker='.', color = colors[1], linewidth = 2)
    ax2.fill_between(np.arange(len(mean_f_rise_per_fish_clock_counts)) / 2., mean_f_rise_per_fish_clock_counts + std_f_rise_per_fish_clock_counts,
                     mean_f_rise_per_fish_clock_counts - std_f_rise_per_fish_clock_counts, color=colors[1], alpha=0.4)
    ax2.fill_between([0, 5.75], [0, 0], [6, 6], color='grey', alpha=0.4, zorder=1)
    ax2.fill_between([17.75, 23.5], [0, 0], [6, 6], color='grey', alpha=0.4, zorder=1)

    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.get_xaxis().tick_bottom()
    ax2.get_yaxis().tick_left()
    ax2.set_xlabel('time', fontsize=fs)
    ax2.set_ylabel('rise rate [n/h]', fontsize=fs)

    # fig.text(0.06, 0.5, 'rise rate / fishcount', va='center', rotation='vertical', fontsize=fs)

    plt.xticks([0.75 + i for i in range(23)], ['%02.0f:00' % i for i in np.arange(23) + 1], rotation=70)

    ax_c.set_ylim([0, 6])
    ax2.set_ylim([0, 6])

    ax_c.tick_params(labelsize=fs - 2)
    ax2.tick_params(labelsize=fs - 2)
    # plt.title('Rise counts per fish', fontsize=fs + 2)
    # plt.suptitle('Rise counts per fish', fontsize=fs + 2)
    plt.tight_layout()
    # plt.show(fig_c)


    ### box plot ####
    day_idx = np.arange(len(full_clock))[np.array([(int(full_clock[i].split(':')[0]) >= 6) & (int(full_clock[i].split(':')[0]) < 18) for i in range(len(full_clock))])]
    night_idx = np.setdiff1d(np.arange(len(full_clock)), day_idx)

    # day_r_counts = np.hstack(all_rise_clock_counts[:, day_idx]) / np.hstack(all_clock_counts[:, day_idx])
    day_r_counts = np.hstack(all_rise_rate_clock[:, day_idx])
    # night_r_counts = np.hstack(all_rise_clock_counts[:, night_idx]) / np.hstack(all_clock_counts[:, night_idx])
    night_r_counts = np.hstack(all_rise_rate_clock[:, night_idx])

    # day_m_r_counts = np.hstack(all_m_rise_clock_counts[:, day_idx]) / np.hstack(all_m_clock_counts[:, day_idx])
    day_m_r_counts = np.hstack(all_m_rise_rate_clock[:, day_idx])
    # day_f_r_counts = np.hstack(all_f_rise_clock_counts[:, day_idx]) / np.hstack(all_f_clock_counts[:, day_idx])
    day_f_r_counts = np.hstack(all_f_rise_rate_clock[:, day_idx])

    # night_m_r_counts = np.hstack(all_m_rise_clock_counts[:, night_idx]) / np.hstack(all_m_clock_counts[:, night_idx])
    night_m_r_counts = np.hstack(all_m_rise_rate_clock[:, night_idx])
    # night_f_r_counts = np.hstack(all_f_rise_clock_counts[:, night_idx]) / np.hstack(all_f_clock_counts[:, night_idx])
    night_f_r_counts = np.hstack(all_f_rise_rate_clock[:, night_idx])

    _, dn_p = scp.mannwhitneyu(day_r_counts[~np.isnan(day_r_counts)], night_r_counts[~np.isnan(night_r_counts)])
    _, dnm_p = scp.mannwhitneyu(day_m_r_counts[~np.isnan(day_m_r_counts)], night_m_r_counts[~np.isnan(night_m_r_counts)])
    _, dnf_p = scp.mannwhitneyu(day_f_r_counts[~np.isnan(day_f_r_counts)], night_f_r_counts[~np.isnan(night_f_r_counts)])

    _, dmf_p = scp.mannwhitneyu(day_m_r_counts[~np.isnan(day_m_r_counts)], day_f_r_counts[~np.isnan(day_f_r_counts)])
    _, nmf_p = scp.mannwhitneyu(night_m_r_counts[~np.isnan(night_m_r_counts)], night_f_r_counts[~np.isnan(night_f_r_counts)])

    print('\n # rise rate / fish')
    print('Day - Night - all:     p = %.3f' % (dn_p * 5.))
    print('Day - Night - male:    p = %.3f' % (dnm_p * 5.))
    print('Day - Night - female:  p = %.3f' % (dnf_p * 5.))
    print('Day - male - female:   p = %.3f' % (dmf_p * 5.))
    print('Night - male - female: p = %.3f' % (dmf_p * 5.))

    # embed()
    # quit()

    fig, ax = plt.subplots(facecolor='white', figsize=(20. / inch_factor, 12. / inch_factor))
    bp = ax.boxplot([day_r_counts[~np.isnan(day_r_counts)], night_r_counts[~np.isnan(night_r_counts)],
                     day_m_r_counts[~np.isnan(day_m_r_counts)], night_m_r_counts[~np.isnan(night_m_r_counts)],
                     day_f_r_counts[~np.isnan(day_f_r_counts)], night_f_r_counts[~np.isnan(night_f_r_counts)]],
                    sym='', patch_artist=True)

    box_colors = ['white', 'grey', colors[6], colors[6], colors[1], colors[1]]
    for enu, box in enumerate(bp['boxes']):
        box.set(facecolor=box_colors[enu])

    ax.plot([1, 2], [5.5, 5.5], color='k', linewidth=2)
    ax.text(1.4, 5.5, '***', fontsize=fs)

    ax.plot([3, 4], [5.5, 5.5], color='k', linewidth=2)
    ax.text(3.4, 5.5, '***', fontsize=fs)

    ax.plot([3, 5], [5.85, 5.85], color='k', linewidth=2)
    ax.text(3.9, 5.85, '***', fontsize=fs)

    ax.plot([4, 6], [6.2, 6.2], color='k', linewidth=2)
    ax.text(4.9, 6.2, '***', fontsize=fs)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()

    ax.set_ylim([0, 6.7])
    old_ticks = ax.get_xticks()
    plt.xticks(old_ticks, ['day', 'night', 'male\nday', 'male\nnight', 'female\nday', 'female\nnight'])

    ax.set_ylabel('rise rate [n/h]', fontsize=fs)
    ax.tick_params(labelsize=fs - 2)

    plt.show()

    # embed()
    # quit()


def plot_appear_n_vanish_time(all_appear_time_h, all_vanish_time_h):
    colors = ['#BA2D22', '#F47F17', '#53379B', '#3673A4', '#AAB71B', '#DC143C', '#1E90FF', 'magenta']

    fs = 14
    inch_factor = 2.54
    # fig, ax = plt.subplots(facecolor='white', figsize=(20. / inch_factor, 12. / inch_factor))
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, facecolor='white', figsize=(20. / inch_factor, 12. / inch_factor))
    bin_boreders = np.arange(0, 24.25, 0.25)

    hist, bins = np.histogram(np.hstack(all_appear_time_h) / 60., bins=bin_boreders)
    width = bins[1] - bins[0]
    center = (bins[:-1] + bins[1:]) / 2
    b1 = ax1.bar(center, np.roll(hist, -4*8), align='center', width=width, color=colors[4])
    # b1 = ax1.bar(center, hist, align='center', width=width, color=colors[4])
    ax1.fill_between([10, 22], [0, 0], [50, 50], color='grey', alpha=0.4, zorder=1)

    ax1.set_ylim([0, 50])
    ax1.set_xlim([0, 24])

    ax1.set_xticks([0, 4, 8, 12, 16, 20, 24])
    old_ticks = ax1.get_xticks()
    new_ticks = []
    for i in range(len(old_ticks)):
        new_ticks.append('%02.0f:00' % ((old_ticks[i] + 8) % 24))

    ax1.set_xticklabels(new_ticks, fontsize=fs - 2)
    ax1.tick_params(labelsize= fs -2)

    hist, bins = np.histogram(np.hstack(all_vanish_time_h) / 60., bins=bin_boreders)
    width = bins[1] - bins[0]
    center = (bins[:-1] + bins[1:]) / 2
    b2 = ax2.bar(center, np.roll(hist, -4*8), align='center', width=width, color=colors[3])
    # b2 = ax2.bar(center, hist, align='center', width=width, color=colors[3])
    ax2.fill_between([10, 22], [0, 0], [50, 50], color='grey', alpha=0.4, zorder=1)

    ax2.set_ylim([0, 50])
    ax2.tick_params(labelsize=fs - 2)
    ax2.invert_yaxis()

    plt.xlabel('Time', fontsize= fs)
    fig.text(0.06, 0.5, 'n', va='center', rotation='vertical', fontsize= fs)

    ax1.legend([b1, b2], ['appear', 'disappear'], 'upper left', frameon=False, fontsize=fs -4)

    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.get_xaxis().tick_bottom()
    ax1.get_yaxis().tick_left()

    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.get_xaxis().tick_bottom()
    ax2.get_yaxis().tick_left()

    fig.subplots_adjust(hspace=0)
    plt.show()


def plot_slope_freq_dependancy(all_freq_at_25, all_slopes):
    inch_factor = 2.54
    fs = 14
    colors = ['#BA2D22', '#F47F17', '#53379B', '#3673A4', '#AAB71B', '#DC143C', '#1E90FF', 'magenta']

    # fig, ax = plt.subplots(facecolor='white', figsize=(20. / inch_factor, 12. / inch_factor))
    fig = plt.figure(facecolor='white', figsize=(20. / inch_factor, 12. / inch_factor))
    ax3 = fig.add_axes([0.85, 0.1, 0.1, 0.65]) # right plot
    ax2 = fig.add_axes([0.1, 0.75, 0.75, 0.2]) # upper plot
    ax = fig.add_axes([0.1, 0.1, 0.75, 0.65])  # left bottom width height   normal plot

    ax.scatter(np.hstack(all_freq_at_25), np.hstack(all_slopes), color=colors[0])

    f_25 = np.hstack(all_freq_at_25)[~np.isnan(np.hstack(all_freq_at_25))]
    s_tf = np.hstack(all_slopes)[~np.isnan(np.hstack(all_slopes))]

    r, p = scp.spearmanr(f_25, s_tf)
    r_m, p_m = scp.spearmanr(f_25[f_25 > 730], s_tf[f_25 > 730])
    r_f, p_f = scp.spearmanr(f_25[f_25 <= 730], s_tf[f_25 <= 730])

    print('\n base frequency vs. freq-temp adaption')
    print('all:    n = %.0f; r = %.2f; p = %.3f' % (len(f_25), r, p))
    print('male:   n = %.0f; r = %.2f; p = %.3f' % (len(f_25[f_25 > 730]), r_m, p_m))
    print('female: n = %.0f; r = %.2f; p = %.3f' % (len(f_25[f_25 <= 730]), r_f, p_f))

    sl, inter, _, _, _ = scp.linregress(f_25, s_tf)
    sl_m, inter_m, _, _, _ = scp.linregress(f_25[f_25 > 730], s_tf[f_25 > 730])
    sl_f, inter_f, _, _, _ = scp.linregress(f_25[f_25 <= 730], s_tf[f_25 <= 730])

    ax.plot(np.arange(550, 900), np.arange(550, 900) * sl + inter, '-', color=colors[4], linewidth=2,
            label='r=%.2f  p=%.3f  all' % (r, p))
    ax.plot(np.arange(730, 900), np.arange(730, 900) * sl_m + inter_m, '--', color=colors[6], linewidth=3,
            label='r=%.2f  p=%.3f  male' % (r_m, p_m))
    ax.plot(np.arange(550, 730), np.arange(550, 730) * sl_f + inter_f, '--', color=colors[1], linewidth=3,
            label='r=%.2f  p=%.3f  female' % (r_f, p_f))

    # ax.set_title('Frequency temperature relation', fontsize=fs + 2)
    ax.set_ylim([0, 60])
    ax.set_xlim([550, 900])
    ax.set_xlabel('EOD$f$ at 25$^{\circ}C$ [Hz]', fontsize=fs)
    ax.set_ylabel('$\Delta$EOD$f$ / $\Delta$temp [Hz/K]', fontsize=fs)
    ax.tick_params(labelsize=fs - 2)

    # ax.spines['top'].set_visible(False)
    # ax.spines['right'].set_visible(False)
    # ax.get_xaxis().tick_bottom()
    # ax.get_yaxis().tick_left()

    plt.legend(loc=4, fontsize=fs - 4, frameon=False)

    bin_borders = np.arange(550, 905, 5)
    hist, bins = np.histogram(np.hstack(all_freq_at_25), bins=bin_borders, normed = True)
    width = bins[1] - bins[0]
    center = (bins[:-1] + bins[1:]) / 2
    ax2.bar(center[center > 730], hist[center > 730], align='center', width=width, color=colors[6])
    ax2.bar(center[center <= 730], hist[center <= 730], align='center', width=width, color=colors[1])
    ax2.set_xlim([550, 900])

    ax2.set_frame_on(False)
    ax2.get_xaxis().set_visible(False)
    ax2.get_yaxis().set_visible(False)

    bin_borders = np.arange(0, 62, 2)
    hist, bins = np.histogram(np.hstack(all_slopes), bins=bin_borders, normed=True)
    width = bins[1] - bins[0]
    center = (bins[:-1] + bins[1:]) / 2
    # ax3.barh(center, hist, align='center', width=width, color=colors[4])
    ax3.barh(bins[:-1], hist, height=2., color = colors[4])
    ax3.set_ylim([0, 60])

    ax3.set_frame_on(False)
    ax3.get_xaxis().set_visible(False)
    ax3.get_yaxis().set_visible(False)

    # ax2.tick_params(bottom='off', top='off', right='off', left='off')

    # plt.tight_layout()
    plt.show()
    # embed()
    # quit()


def get_rise_proportions(fishes, all_times, male_rises, female_rises):
    dt = all_times[1] - all_times[0]
    dpm = 60. / dt

    proportions = [[], []]
    all_dfs = [[], []]

    for enu, rises in enumerate([male_rises, female_rises]):
        [f0, f1] = [730, 1050] if enu == 0 else [550, 730]

        for rise in rises:
            freq = fishes[rise[0]][rise[1]]
            idx0 = rise[1] - dpm if rise[1] - dpm >= 0 else 0
            idx1 = rise[1] + dpm

            higher = 0
            lower = 0
            dfs = []
            for enu2, comp_fish in enumerate(fishes):
                if enu2 == rise[0]:
                    continue
                if len(comp_fish[idx0:idx1][~np.isnan(comp_fish[idx0:idx1])]) > 1:
                    comp_fish_med = np.median(comp_fish[idx0:idx1][~np.isnan(comp_fish[idx0:idx1])])
                    dfs.append(comp_fish_med - freq)
                    if comp_fish_med >= f0 and comp_fish_med < f1:
                        if comp_fish_med > freq:
                            higher += 1
                        elif comp_fish_med < freq:
                            lower += 1

            if lower + higher >= 3:
                proportions[enu].append(1.* lower / (lower + higher))

            dfs = np.array(dfs)

            min_df = np.min(np.abs(dfs)) if len(dfs) > 0 else np.nan
            min_up_df = np.min(dfs[dfs > 0]) if len(dfs[dfs > 0]) > 0 else np.nan
            min_down_df = np.min(np.abs(dfs[dfs < 0])) if len(dfs[dfs < 0]) > 0 else np.nan

            all_dfs[enu].append( np.array([ min_df, min_up_df, min_down_df ]) )

    bin_borders = np.arange(0, 1.1, 0.1)
    hist, bins = np.histogram(proportions[0], bins=bin_borders)
    m_rise_proportions = 1.* hist / np.sum(hist)

    hist, bins = np.histogram(proportions[1], bins=bin_borders)
    f_rise_proportions = 1.* hist / np.sum(hist)

    return m_rise_proportions, f_rise_proportions, np.array(all_dfs[0]), np.array(all_dfs[1])


def boot_rise_characteristics(rng, fishes, all_times, b_counts):
    dt = all_times[1] - all_times[0]
    dpm = 60./dt

    df_males = []
    df_females = []

    m_rise_chars = []
    f_rise_chars = []
    while len(df_males) < b_counts or len(df_females) < b_counts:
        fish = rng.randint(0, len(fishes))
        non_nan_idx = np.arange(len(fishes[fish]))[~np.isnan(fishes[fish])]

        rise_idx = non_nan_idx[rng.randint(0, len(non_nan_idx))]
        rise_freq = fishes[fish][rise_idx]

        if rise_freq >= 730 and rise_freq < 1050 and len(df_males) < b_counts:
            f0 = 730
            f1 = 1050
        elif rise_freq >= 550 and rise_freq < 730 and len(df_females) < b_counts:
            f0 = 550
            f1 = 730
        else:
            continue
        idx0 = rise_idx - dpm if rise_idx - dpm >= 0 else 0
        idx1 = rise_idx + dpm

        upper_count = 0
        lower_count = 0
        dfs = []

        for enu, comp_fish in enumerate(fishes):
            if enu == fish:
                continue
            if len(comp_fish[idx0:idx1][~np.isnan(comp_fish[idx0:idx1])]) > 1:
                comp_fish_med = np.median(comp_fish[idx0:idx1][~np.isnan(comp_fish[idx0:idx1])])
                if comp_fish_med >= f0 and comp_fish_med < f1:
                    dfs.append(np.diff([rise_freq, comp_fish_med])[0])
                    if comp_fish_med > rise_freq:
                        upper_count += 1
                    if comp_fish_med < rise_freq:
                        lower_count += 1
        dfs = np.array(dfs)
        if rise_freq >= 730 and rise_freq < 1050:
            df_males.append(dfs)
            m_rise_chars.append(np.array([upper_count, lower_count]))
        elif rise_freq >= 550 and rise_freq < 730:
            df_females.append(dfs)
            f_rise_chars.append(np.array([upper_count, lower_count]))

    m_rise_chars = np.array(m_rise_chars)
    f_rise_chars = np.array(f_rise_chars)

    all_m_at_rise = m_rise_chars[:, 0] + m_rise_chars[:, 1]
    m_below_ration = 1. * m_rise_chars[:, 1] / all_m_at_rise
    m_below_ration = m_below_ration[all_m_at_rise >= 5]

    all_f_at_rise = f_rise_chars[:, 0] + f_rise_chars[:, 1]
    f_below_ration = 1. * f_rise_chars[:, 1] / all_f_at_rise
    f_below_ration = f_below_ration[all_f_at_rise >= 5]

    bin_borders = np.arange(0, 1.1, 0.1)
    hist, bins = np.histogram(m_below_ration, bins=bin_borders)
    m_rise_proportions = hist

    hist, bins = np.histogram(f_below_ration, bins=bin_borders)
    f_rise_proportions = hist

    return m_rise_proportions, f_rise_proportions, np.array(df_males), np.array(df_females)


def plot_rise_characteristics(all_m_rise_chars, all_f_rise_chars):
    all_m_below_ration = []
    all_f_below_ration = []

    for i in range(len(all_m_rise_chars)):
        all_m_at_rise = all_m_rise_chars[i][:, 0] + all_m_rise_chars[i][:, 1]
        m_below_ration = 1. * all_m_rise_chars[i][:, 1] / all_m_at_rise
        all_m_below_ration.append(m_below_ration[all_m_at_rise >= 5])

    for i in range(len(all_f_rise_chars)):
        all_f_at_rise = all_f_rise_chars[i][:, 0] + all_f_rise_chars[i][:, 1]
        f_below_ration = 1. * all_f_rise_chars[i][:, 1] / all_f_at_rise
        all_f_below_ration.append(f_below_ration[all_f_at_rise >= 5])

    m_hists = []
    f_hists = []
    bin_borders = np.arange(0, 1.1, 0.1)

    for i in range(len(all_m_below_ration)):
        hist, bins = np.histogram(all_m_below_ration[i], bins = bin_borders)
        # hist = 1.0 * hist / np.sum(hist)
        m_hists.append(hist)

        hist, bins = np.histogram(all_f_below_ration[i], bins = bin_borders)
        # hist = 1.0 * hist / np.sum(hist)
        f_hists.append(hist)
        center = (bins[:-1] + bins[1:]) / 2

    m_hists = np.array(m_hists)
    f_hists = np.array(f_hists)

    m_p25 = np.zeros(len(center))
    m_p75 = np.zeros(len(center))
    f_p25 = np.zeros(len(center))
    f_p75 = np.zeros(len(center))
    m_med_hists = np.zeros(len(center))
    f_med_hists = np.zeros(len(center))
    for i in range(len(center)):
        m_p25[i], m_p75[i] = np.percentile(m_hists[:, i][~np.isnan(m_hists[:, i])], (25, 75))
        f_p25[i], f_p75[i] = np.percentile(f_hists[:, i][~np.isnan(f_hists[:, i])], (25, 75))
        m_med_hists[i] = np.median(m_hists[:, i][~np.isnan(m_hists[:, i])])
        f_med_hists[i] = np.median(f_hists[:, i][~np.isnan(f_hists[:, i])])

    colors = ['#BA2D22', '#F47F17', '#53379B', '#3673A4', '#AAB71B', '#DC143C', '#1E90FF', 'magenta']
    inch_factor = 2.54
    fig, (ax, ax2) = plt.subplots(2, 1, sharex=True, facecolor='white', figsize=(20. / inch_factor, 12. / inch_factor))
    ax.plot(center, m_med_hists, marker='.', color= colors[6])
    ax.fill_between(center, m_p25, m_p75, color= colors[6], alpha = 0.4)
    ax.set_ylabel('n')

    ax2.plot(center, f_med_hists, marker='.', color= colors[1])
    ax2.fill_between(center, f_p25, f_p75, color= colors[1], alpha = 0.4)
    ax2.set_ylabel('n')
    ax2.set_xlabel('ratio of m/f below rising fish')

    plt.show()


def plot_rise_df(rise_base_f_and_df):
    rise_base_f_and_df = np.array(rise_base_f_and_df)
    fs = 14

    colors = ['#BA2D22', '#F47F17', '#53379B', '#3673A4', '#AAB71B', '#DC143C', '#1E90FF', 'magenta']
    inch_factor = 2.54

    all_day = rise_base_f_and_df[1][(rise_base_f_and_df[2] >= 360) & (rise_base_f_and_df[2] < 1080)]
    all_night = rise_base_f_and_df[1][(rise_base_f_and_df[2] < 360) | (rise_base_f_and_df[2] >= 1080)]

    m_day = rise_base_f_and_df[1][(rise_base_f_and_df[0] >=730) & (rise_base_f_and_df[0] < 1050) &
                                  (rise_base_f_and_df[2] >= 360) & (rise_base_f_and_df[2] < 1080)]
    m_night = rise_base_f_and_df[1][(rise_base_f_and_df[0] >=730) & (rise_base_f_and_df[0] < 1050) &
                                    (rise_base_f_and_df[2] < 360) | (rise_base_f_and_df[2] >= 1080)]

    f_day = rise_base_f_and_df[1][(rise_base_f_and_df[0] >= 550) & (rise_base_f_and_df[0] < 730) &
                                  (rise_base_f_and_df[2] >= 360) & (rise_base_f_and_df[2] < 1080)]
    f_night = rise_base_f_and_df[1][(rise_base_f_and_df[0] >= 550) & (rise_base_f_and_df[0] < 730) &
                                    (rise_base_f_and_df[2] < 360) | (rise_base_f_and_df[2] >= 1080)]


    _, dn_p = scp.mannwhitneyu(all_day, all_night)
    _, dnm_p = scp.mannwhitneyu(m_day, m_night)
    _, dnf_p = scp.mannwhitneyu(f_day, f_night)
    _, dmf_p = scp.mannwhitneyu(m_day, f_day)
    _, nmf_p = scp.mannwhitneyu(m_night, f_night)

    print('\nrise highth')
    print('Day - Night - all:     p = %.3f' % (dn_p * 5.))
    print('Day - Night - male:    p = %.3f' % (dnm_p * 5.))
    print('Day - Night - female:  p = %.3f' % (dnf_p * 5.))
    print('Day - male - female:   p = %.3f' % (dmf_p * 5.))
    print('Night - male - female: p = %.3f' % (nmf_p * 5.))

    fig, ax = plt.subplots(facecolor='white', figsize=(20. / inch_factor, 12. / inch_factor))
    box_colors = ['white', 'grey', colors[6], colors[6], colors[1], colors[1]]

    bp = ax.boxplot([all_day, all_night, m_day, m_night, f_day, f_night], sym='', patch_artist=True)

    for enu, box in enumerate(bp['boxes']):
        box.set(facecolor=box_colors[enu])

    ax.set_ylabel('rise size [Hz]', fontsize=fs)
    ax.set_ylim([0, 5.5])
    old_ticks = ax.get_xticks()
    plt.xticks(old_ticks, ['day', 'night', 'male\nday', 'male\nnight', 'female\nday', 'female\nnight'])

    x_lims = ax.get_xlim()
    ax.plot([x_lims[0], x_lims[1]], [1, 1], '--', color='k', linewidth=2, label='detection threshold')
    plt.legend(loc = 4, frameon=False, fontsize=fs - 4)

    ax.plot([1, 2], [4.6, 4.6], color='k', linewidth=2)
    ax.text(1.47, 4.6, '**', fontsize=14)

    ax.plot([5, 6], [4.6, 4.6], color='k', linewidth=2)
    ax.text(5.47, 4.6, '***', fontsize=14)

    ax.plot([3, 5], [4.9, 4.9], color='k', linewidth=2)
    ax.text(3.97, 4.9, '*', fontsize=14)

    ax.plot([4, 6], [5.2, 5.2], color='k', linewidth=2)
    ax.text(4.97, 5.2, '**', fontsize=14)

    ax.tick_params(labelsize=fs - 2)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()

    bin_borders = np.arange(1, 15.2, 0.2)
    fig, (ax, ax2) = plt.subplots(1, 2, facecolor='white', figsize=(20. / inch_factor, 12. / inch_factor))

    hist, bins = np.histogram(rise_base_f_and_df[1], bins=bin_borders, normed=True)

    width = bins[1] - bins[0]
    center = (bins[:-1] + bins[1:]) / 2

    # embed()
    # quit()
    p_exp = scp.expon.pdf(center - 1, scale=np.mean(rise_base_f_and_df[1]) - 1.)
    ax.bar(center, hist, align='center', width=width, color=colors[4])
    ax.plot(center, p_exp, lw=3, color=colors[5], alpha=0.8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()

    ax.set_xlabel('rise size [Hz]', fontsize = fs)
    ax.set_ylabel('pdf', fontsize = fs)
    ax.set_ylim([0, np.max(hist) + 0.1])
    ax.set_xlim([0, 14])
    ax.tick_params(labelsize=fs - 2)


    ax2.scatter(center, hist, color=colors[4])
    ax2.plot(center, p_exp, lw=3, color=colors[5], alpha=0.8)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.get_xaxis().tick_bottom()
    ax2.get_yaxis().tick_left()

    ax2.set_xlabel('rise size [Hz]', fontsize = fs)
    ax2.set_ylabel('pdf', fontsize = fs)
    ax2.set_ylim([0.001, np.max(hist) + 0.1])
    ax2.set_xlim([0, 14])
    ax2.tick_params(labelsize=fs - 2)
    ax2.set_yscale('log', fontsize = fs)

    fig.subplots_adjust(wspace=0.4)

    plt.show()

    # embed()
    # quit()

def plot_dfs(all_beat_f):
    colors = ['#BA2D22', '#F47F17', '#53379B', '#3673A4', '#AAB71B', '#DC143C', '#1E90FF', 'magenta']
    inch_factor = 2.54
    fs = 14

    bin_borders = np.arange(0, 402, 2)
    hist, bins = np.histogram(np.abs(all_beat_f), bins=bin_borders)

    width = bins[1] - bins[0]
    center = (bins[:-1] + bins[1:]) / 2

    # p_exp = scp.expon.pdf(center - 1, scale=np.mean(rise_base_f_and_df[1]) - 1.)
    fig, ax = plt.subplots(facecolor='white', figsize=(20. / inch_factor, 12. / inch_factor))
    ax.bar(center, hist, align='center', width=width, color=colors[4])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    ax.tick_params(labelsize=fs-2)
    ax.set_ylabel('n')
    ax.set_xlabel('beat EOD$f$ [Hz]')

    plt.show()
    # ax.plot(center, p_exp, lw=3, color=colors[5], alpha=0.8)



def main(file_path, hobo_path, show_fish_plot=False, show_combined_fishes=False, rise_characteristics= False,
         plot_freq_temp_slope=False):

    rng = np.random.RandomState(15235412311)

    all_beat_f = []

    rise_phnf= []  # rises per hour and frequency.
    all_clock_counts = []  # num of fishes available in 30 min bins
    all_m_clock_counts = []  # num of males available in 30 min bins
    all_f_clock_counts = []  # num of females available in 30 min bins

    all_rise_clock_counts = []  # number of rises in 30 min bins
    all_m_rise_clock_counts = []  # number of rises in 30 min bins
    all_f_rise_clock_counts = []  # number of rises in 30 min bins

    all_rise_rate_clock = []
    all_m_rise_rate_clock = []
    all_f_rise_rate_clock = []

    all_mean_clock_temp = []  # mean temp of 30 min bins
    all_presence_time = []  # presence time of all fishes
    all_vanish_time = []  # time when da fish disappears
    all_presence_freq = []  # mean freq of all fishes

    all_appear_time_h = []  # appear time of fishes available for more than 30 min
    all_vanish_time_h = []  # vanish time of fishes available for more than 30 min

    all_slopes = []  # slope of temperature dependant frequency
    all_freq_at_25 = []  # frequency at 25 degree for the array above

    day_recording = []  # True if the recording was started in the morning
    start_dates = []  # strings containing the date when the recording started

    all_m_rise_proportions = []
    all_f_rise_proportions = []
    all_df_male = []
    all_df_female = []
    all_b_df_male = []
    all_b_df_female = []

    rise_base_f_and_df = [[], [], []]

    temp_plot_count = 0
    if plot_freq_temp_slope:
        inch_factor = 2.54
        fs = 14
        fig_t, ((axt0, axt1), (axt2, axt3)) = plt.subplots(2, 2, facecolor='white', figsize=(20. / inch_factor, 12. / inch_factor))
        axt = [axt0, axt1, axt2, axt3]
    # identify folders
    folders = np.array([x[0] for x in os.walk(file_path)])
    if len(folders) > 1:
        folders = folders[1:]

    if show_combined_fishes:
        inch_factor = 2.54
        fig_all, ax_all = plt.subplots(facecolor='white', figsize=(20. / inch_factor, 12. / inch_factor))
        last_t = 0
        last_end_time = False
        first_start_time = False
        old_colors = False

    for enu, folder in enumerate(sorted(folders)):
        print(folder)
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
        if dat_file == False:
            print('dat_file missing...')
            quit()

        # extract dat_file infos
        log_infos = load_dat_file(dat_file)

        # get start and end time strings
        start_time, start_time_str, end_time_str, start_date_str, end_date_str = extract_times(log_infos, all_times)
        start_dates.append(start_date_str)

        # load temp
        temp, lux = load_temp_and_time(hobo_path, all_times, start_time, start_time_str, end_time_str, start_date_str, end_date_str)

        # calculate freq change per deg C and rises per hour
        if temp != []:
            slopes = []
            freq_at_25 = []
            for fish in fishes:
                slope, intercept, _, _, _ = scp.linregress(temp[~np.isnan(fish)], fish[~np.isnan(fish)])
                if all_times[~np.isnan(fish)][-1] - all_times[~np.isnan(fish)][0] >= 1800 and \
                                np.max(temp[~np.isnan(fish)]) - np.min(temp[~np.isnan(fish)]) >= 1. and \
                                        intercept + slope * 25. > 550. and slope > 5.:
                    if plot_freq_temp_slope and temp_plot_count < 4 and enu >=2:
                        colors = ['#BA2D22', '#F47F17', '#53379B', '#3673A4', '#AAB71B', '#DC143C', '#1E90FF',
                                  'magenta']
                        min_t = np.min(temp[~np.isnan(fish)])
                        max_t = np.max(temp[~np.isnan(fish)])
                        axt[temp_plot_count].scatter(temp[~np.isnan(fish)], fish[~np.isnan(fish)], marker='.', color = '#AAB71B')
                        axt[temp_plot_count].plot([min_t, max_t], np.array([min_t, max_t]) * slope + intercept, color='red', linewidth=2, label='%.2f Hz/K' % slope)
                        axt[temp_plot_count].set_xticks([24.5, 25, 25.5, 26, 26.5])
                        # print slope
                        axt[temp_plot_count].legend(loc=2, numpoints=1, frameon=False, fontsize=fs-4)
                        temp_plot_count += 1
                        if temp_plot_count == 4:
                            axt0.set_ylabel('frequency [Hz]', fontsize=fs)
                            axt2.set_ylabel('frequency [Hz]', fontsize=fs)
                            axt2.set_xlabel('temp [$^{\circ}C$]', fontsize=fs)
                            axt3.set_xlabel('temp [$^{\circ}C$]', fontsize=fs)
                            for ax in axt:
                                ax.spines['top'].set_visible(False)
                                ax.spines['right'].set_visible(False)
                                ax.get_xaxis().tick_bottom()
                                ax.get_yaxis().tick_left()
                                ax.tick_params(labelsize=fs-2)
                            plt.tight_layout()
                            plt.show(fig_t)
                    slopes.append(slope)
                    freq_at_25.append(intercept + slope * 25.)
                else:
                    slopes.append(np.nan)
                    freq_at_25.append(np.nan)

            slopes = np.array(slopes)
            freq_at_25 = np.array(freq_at_25)

            all_slopes.append(slopes)
            all_freq_at_25.append(freq_at_25)

            rises_ph_n_f = rises_per_hour(fishes, all_times, all_rises, temp, slopes)
            rise_phnf += rises_ph_n_f

        # get rise characteristics
        if rise_characteristics:
            male_rises = []
            female_rises = []
            for fish in range(len(all_rises)):
                for rise in all_rises[fish]:
                    if rise[1][0] - rise[1][1] < 1.:
                        continue

                    if rise[1][1] >= 730 and rise[1][1] < 1050:
                        male_rises.append([fish, rise[0][0]])
                    elif rise[1][1] >= 550 and rise[1][1] < 730:
                        female_rises.append([fish, rise[0][0]])
                    else:
                        continue

            m_proportions, f_proportions, m_dfs, f_dfs = get_rise_proportions(fishes, all_times, male_rises, female_rises)

            bm_proportions = []
            bf_proportions = []

            bm_dfs = []
            bf_dfs = []

            for i in range(1000):
                fake_male_rises = []
                fake_female_rises = []

                while len(fake_male_rises) < len(male_rises) or len(fake_female_rises) < len(female_rises):
                    fish = rng.randint(0, len(fishes))
                    freq = np.median(fishes[fish][~np.isnan(fishes[fish])])

                    non_nan_idx = np.arange(len(fishes[fish]))[~np.isnan(fishes[fish])]
                    idx = non_nan_idx[rng.randint(0, len(non_nan_idx))]

                    if freq >= 730 and freq < 1050 and len(fake_male_rises) < len(male_rises):
                        fake_male_rises.append([fish, idx])
                    elif freq >= 550 and freq < 730 and len(fake_female_rises) < len(female_rises):
                        fake_female_rises.append([fish, idx])
                bm_proportion, bf_proportion, bm_df, bf_df = get_rise_proportions(fishes, all_times, fake_male_rises, fake_female_rises)
                bm_dfs.append(bm_df)
                bf_dfs.append(bf_df)

                bm_proportions.append(np.array(bm_proportion))
                bf_proportions.append(np.array(bf_proportion))

            bm_proportions = np.array(bm_proportions)
            bf_proportions = np.array(bf_proportions)

            m_p975 = np.zeros(len(m_proportions))
            m_p025 = np.zeros(len(m_proportions))
            f_p975 = np.zeros(len(m_proportions))
            f_p025 = np.zeros(len(m_proportions))

            for i in range(len(m_proportions)):
                m_p025[i], m_p975[i] = np.percentile(bm_proportions[:, i], (2.5, 97.5))
                f_p025[i], f_p975[i] = np.percentile(bf_proportions[:, i], (2.5, 97.5))

            all_m_rise_proportions.append(np.array([m_p025, m_proportions, m_p975]))
            all_f_rise_proportions.append(np.array([f_p025, f_proportions, f_p975]))

            all_df_male.append(m_dfs)
            all_df_female.append(f_dfs)
            all_b_df_male.append(bm_dfs)
            all_b_df_female.append(bf_dfs)

        ################################################################################
        # calculate fish count per hour.
        clock_counts, m_clock_counts, f_clock_counts, rise_clock_counts, m_rise_clock_counts, f_rise_clock_counts, \
        rise_rate_clock, m_rise_rate_clock, f_rise_rate_clock, mean_clock_temp, full_clock, beat_f = \
            fish_n_rise_count_per_time(fishes, all_times, start_time, temp, all_rises)

        all_beat_f += beat_f

        all_clock_counts.append(clock_counts)
        all_m_clock_counts.append(m_clock_counts)
        all_f_clock_counts.append(f_clock_counts)

        all_rise_clock_counts.append(rise_clock_counts)
        all_m_rise_clock_counts.append(m_rise_clock_counts)
        all_f_rise_clock_counts.append(f_rise_clock_counts)

        all_rise_rate_clock.append(rise_rate_clock)
        all_m_rise_rate_clock.append(m_rise_rate_clock)
        all_f_rise_rate_clock.append(f_rise_rate_clock)


        all_mean_clock_temp.append(mean_clock_temp)

        presence_time, vanish_time, presence_freq, appear_time_h, vanish_time_h = get_presence_time(fishes, all_times, start_time)
        all_presence_time.append(presence_time)
        all_vanish_time.append(vanish_time)
        all_presence_freq.append(presence_freq)
        all_appear_time_h.append(appear_time_h)
        all_vanish_time_h.append(vanish_time_h)

        dt = all_times[1] - all_times[0]
        dpm = 60. / dt
        for fish in range(len(all_rises)):
            for rise in all_rises[fish]:
                if rise[1][0] - rise[1][1] >= 1.:
                    rise_base_f_and_df[0].append(rise[1][1])
                    rise_base_f_and_df[1].append(rise[1][0] - rise[1][1])
                    rise_base_f_and_df[2].append((start_time[0] * 60 + start_time[1] + np.floor(rise[0][1] / dpm)) % 1440 )

        if start_time[0] >= 13:
            day_recording.append(False)
        else:
            day_recording.append(True)

        # plot stuff
        if show_fish_plot:
            plot_fishes(fishes, all_times, all_rises, temp, lux, start_time_str, start_date_str)

        if show_combined_fishes:
            last_t, old_colors = comb_fish_plot(ax_all, fishes, all_times, start_time_str, last_end_time, last_t, old_colors)
            last_end_time = end_time_str

            if not first_start_time:
                first_start_time = start_time_str

    if show_combined_fishes:
        fs = 14
        first_start_time = [int(first_start_time.split(':')[n]) for n in range(2)]
        morning_time = [6, 0]

        if morning_time[1] < first_start_time[1]:
            morning_time[1] += 60
            first_start_time[0] += 1
        if morning_time[0] < first_start_time[0]:
            morning_time[0] += 24

        dh = (morning_time[1] - first_start_time[1]) / 60. + (morning_time[0] - first_start_time[0])

        xlimits = ax_all.get_xlim()
        ylimits = ax_all.get_ylim()

        mornings = np.arange(dh, xlimits[1], 24)
        # embed()
        # quit()
        for morning in mornings:
            evening = morning - 12.
            if evening < 0:
                evening = 0

            ax_all.fill_between([evening, morning], [ylimits[0], ylimits[0]], [ylimits[1], ylimits[1]], color='grey',
                                alpha= 0.4, zorder= 1)

        ax_all.set_ylim([400, 950])
        ax_all.set_xlim([0, 355])

        dates = ['10.04.', '11.04.', '12.04.', '13.04.', '14.04.', '15.04.', '16.04.', '17.04.', '18.04.', '19.04.',
                 '20.04.', '21.04.', '22.04.', '23.04.', '24.04.']
        ax_all.set_xticks(mornings[:15] + 6.)
        # old_ticks = ax_all.get_xticks()
        ax_all.set_xticklabels(dates, rotation=70)
        ax_all.set_xlabel('date', fontsize=fs)
        ax_all.set_ylabel('frequency [Hz]', fontsize=fs)
        ax_all.tick_params(labelsize=fs - 2)
        plt.tight_layout()
        plt.show()

    if rise_characteristics:
        np.save('m_rise_proportions.npy', np.array(all_m_rise_proportions))
        np.save('f_rise_proportions.npy', np.array(all_f_rise_proportions))
        np.save('m_df.npy', np.array(all_df_male))
        np.save('f_df.npy', np.array(all_df_female))
        np.save('mb_df.npy', np.array(all_b_df_male))
        np.save('fb_df.npy', np.array(all_b_df_female))

    plot_dfs(all_beat_f)

    # temp and frequency couses
    plot_slope_freq_dependancy(all_freq_at_25, all_slopes)

    # frequency courses
    plot_appear_n_vanish_time(all_appear_time_h, all_vanish_time_h)

    day_recording = np.array(day_recording)
    plot_presence_time(all_presence_time, all_vanish_time, all_presence_freq, day_recording, start_dates)

    plot_clock_fish_counts(all_clock_counts, all_m_clock_counts, all_f_clock_counts, full_clock, all_mean_clock_temp)

    # rise analysis

    plot_rise_df(rise_base_f_and_df)

    plot_clock_rise_counts(all_rise_clock_counts, all_m_rise_clock_counts, all_f_rise_clock_counts, all_clock_counts,
                           all_m_clock_counts, all_f_clock_counts, all_rise_rate_clock, all_m_rise_rate_clock,
                           all_f_rise_rate_clock, full_clock)


    if rise_phnf != []:
        # get fish base freq and rise counts per hour
        rise_base_freq = np.array([rise_phnf[n][1] for n in range(len(rise_phnf))])
        rise_counts = np.array([rise_phnf[n][0] for n in range(len(rise_phnf))])

        # set freq bins and get consecutive rise counts per hour
        bins = np.arange(455, 1055, 10)
        bins = bins.tolist()
        rise_freq_counts = []
        for bin_c in bins:
            rise_freq_counts.append(rise_counts[(rise_base_freq >= bin_c -5) & (rise_base_freq < bin_c +5)])

        # delete entries == 0 to not destroy statistics
        # for bin in reversed(range(len(bins))):
        #     if len(rise_freq_counts[bin]) == 0:
        #         rise_freq_counts.pop(bin)
        #         bins.pop(bin)
        bins = np.array(bins)

        plot_rise_phnf(rise_freq_counts, rise_base_freq, rise_counts, bins)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Analyse long single- or multi electrode EOD recordings of weakly electric fish.',
        epilog='by bendalab (2015-2016)')
    parser.add_argument('file_path', nargs='?', default='', type=str, help='folder containing the data to analyse')
    parser.add_argument('hobo_path', nargs='?', default='', type=str, help='folder containing the hobo datalogger files')

    args = parser.parse_args()
    main(args.file_path, args.hobo_path)
