import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from IPython import embed


def main(file_path, plot_final_fishes = True):
    # load data and get fishcount per half an hour...
    folders = np.array([x[0] for x in os.walk(file_path)])
    if len(folders) > 1:
        folders = folders[1:]

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
    # embed()
    # quit()

    all_counts = np.array([])
    all_clock = np.array([])
    for folder in sorted(folders):
        print folder
        for file in os.listdir(folder):

            if file.endswith('final_times.npy'):
                all_times = np.load(os.path.join(folder, file))
            elif file.endswith('final_fishes.npy'):
                fishes = np.load(os.path.join(folder, file))
            elif file.endswith('final_rises.npy'):
                all_rises = np.load(os.path.join(folder, file))

        if plot_final_fishes:
            fig, ax = plt.subplots(facecolor='white', figsize=(11.6, 8.2))
            time_factor = 1.
            # if all_times[-1] <= 120:
            #     time_factor = 1.
            # elif all_times[-1] > 120 and all_times[-1] < 7200:
            #     time_factor = 60.
            # else:
            #     time_factor = 3600.

            for fish in range(len(fishes)):
                color = np.random.rand(3, 1)
                ax.plot(all_times[~np.isnan(fishes[fish])] / time_factor, fishes[fish][~np.isnan(fishes[fish])],
                        color=color, marker='.')
                #
                # for rise in all_rises[fish]:
                #     ax.plot(all_times[rise[0][0]] / time_factor, rise[1][0], 'o', color=color, markersize=7)
                #     ax.plot(all_times[rise[0][1]] / time_factor, rise[1][1], 's', color=color, markersize=7)

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
                            plt.legend(loc=1, numpoints=1, frameon=False, fontsize=12)
                        else:
                            ax.plot(all_times[rise[0][0]] / time_factor, rise[1][0], 'o', color='red', markersize=7,
                                    markerfacecolor='None')
                            ax.plot(all_times[rise[0][1]] / time_factor, rise[1][1], 's', color='green', markersize=7,
                                    markerfacecolor='None')

            maxy = np.max(np.array([np.mean(fishes[fish][~np.isnan(fishes[fish])]) for fish in range(len(fishes))]))
            miny = np.min(np.array([np.mean(fishes[fish][~np.isnan(fishes[fish])]) for fish in range(len(fishes))]))

            plt.ylim([miny - 150, maxy + 150])
            plt.ylabel('Frequency [Hz]', fontsize=14)
            if time_factor == 1.:
                plt.xlabel('Time [sec]', fontsize=14)
            elif time_factor == 60.:
                plt.xlabel('Time [min]', fontsize=14)
            else:
                plt.xlabel('Time [h]', fontsize=14)

            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.get_xaxis().tick_bottom()
            ax.get_yaxis().tick_left()

            plt.show()

            # fig, ax = plt.subplots()
            # for fish in fishes:
            #
            #     ax.plot(all_times[~np.isnan(fish)] / 3600., fish[~np.isnan(fish)], color=np.random.rand(3, 1), marker='.')
            # plt.title(folder)
            # plt.show()


        start_h = int(folder[-5:-3])
        start_min = int(folder[-2:])

        dt = all_times[1]-all_times[0]
        dpm = 60./ dt

        if start_min > 30:
            start_idx = np.floor((60. - start_min) * dpm)
            start_min = 0
            start_h += 1
        elif start_min <= 30:
            start_idx = np.floor((30. - start_min) * dpm)
            start_min = 30

        start_time_min = start_h * 60 + start_min

        loops = int((len(all_times) - start_idx) // (dpm * 30))
        counts = np.zeros(loops)

        for i in range(loops):
            end_idx = start_idx + dpm * 30
            for fish in fishes:
                if len(fish[start_idx:end_idx][~np.isnan(fish[start_idx:end_idx])]) >= dpm:
                    counts[i] += 1
            start_idx += dpm * 30

        # embed()
        # quit()
        plot_time_min = np.arange(start_time_min + 15, start_time_min + 15 + loops * 30, 30)
        plot_time_str = []
        for i in range(len(plot_time_min)):
            if plot_time_min[i] // 60 < 24:
                plot_time_str.append(str(plot_time_min[i] // 60) + ':' + str(plot_time_min[i] % 60))
            else:
                plot_time_str.append(str((plot_time_min[i] - 24 * 60) // 60) + ':' + str(plot_time_min[i] % 60))

        if all_counts != []:
            t0 = np.where(full_clock == all_clock[-1])[0]
            t1 = np.where(full_clock == plot_time_str[0])[0]
            if t1 > t0:
                fill_up_times = full_clock[t0+1:t1]
            elif t0 > t1:
                fill_up_times = np.append(full_clock[t0+1:], full_clock[:t1])
            else:
                print('error !!!')
                quit()
            all_clock = np.append(all_clock, fill_up_times)
            all_counts = np.append(all_counts, np.full(len(fill_up_times), np.nan))

        all_counts = np.append(all_counts, counts)
        all_clock = np.append(all_clock, plot_time_str)

    # calculation of median and percentiles
    med_clock_count = np.empty(len(full_clock))
    p75_clock_count = np.empty(len(full_clock))
    p25_clock_count = np.empty(len(full_clock))

    for i in range(len(full_clock)):
        idx = np.where(all_clock == full_clock[i])[0]
        med_clock_count[i] = np.median(all_counts[idx][~np.isnan(all_counts[idx])])
        p25_clock_count[i], p75_clock_count[i] = np.percentile(all_counts[idx][~np.isnan(all_counts[idx])], (25, 75))

    # Median fishcount plot per time
    sunrise_idx = np.where(full_clock == '5:15')[0]
    sunset_idx = np.where(full_clock == '19:15')[0]

    fig, ax = plt.subplots(figsize=(16., 6.), facecolor='white')
    ax.fill_between(np.arange(sunrise_idx, sunset_idx+1), np.ones(len(np.arange(sunrise_idx, sunset_idx+1)))*
                    np.min(p25_clock_count), np.ones(len(np.arange(sunrise_idx, sunset_idx+1))) *
                    np.max(p75_clock_count), color='y', alpha=0.2)
    ax.fill_between(np.arange(sunrise_idx+1), np.ones(len(np.arange(sunrise_idx+1)))* np.min(p25_clock_count),
                    np.ones(len(np.arange(sunrise_idx+1)))* np.max(p75_clock_count), color='k', alpha=0.2)
    ax.fill_between(np.arange(sunset_idx, len(full_clock)), np.ones(len(np.arange(sunset_idx, len(full_clock))))*
                    np.min(p25_clock_count), np.ones(len(np.arange(sunset_idx, len(full_clock))))*
                    np.max(p75_clock_count), color='k', alpha=0.2 )

    x = np.arange(len(full_clock))
    ax.plot(x, med_clock_count, color='red', marker='o', linewidth=2)
    ax.fill_between(x, p25_clock_count, p75_clock_count, alpha=0.4)


    ax.set_xlim([-1, len(x)])
    plt.xticks(x[::2], full_clock[::2], rotation='vertical')

    ax.set_xlabel('Time')
    ax.set_ylabel('fish count')
    ax.set_title('Median fish count during the day')
    plt.tight_layout()

    # Fishcount plot over all days

    midnight_idx = np.where(all_clock == '0:15')[0]
    sunrise_idx = np.where(all_clock == '5:15')[0]
    sunset_idx = np.where(all_clock == '19:15')[0]


    fig, ax = plt.subplots(figsize=(16., 6.), facecolor='white')
    x = np.arange(len(all_counts))
    ax.plot(x, all_counts)

    min_counts = np.min(all_counts[~np.isnan(all_counts)])
    max_counts = np.max(all_counts[~np.isnan(all_counts)])

    ax.plot([midnight_idx, midnight_idx], [min_counts, max_counts], color='k', linewidth=1)
    ax.set_xlim([-1, len(x)])

    plt.xticks(x[::10], all_clock[::10], rotation='vertical')
    ax.set_xlabel('Time')
    ax.set_ylabel('fish count')
    ax.set_title('Fish count during time of recording')
    plt.tight_layout()

    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Analyse long single- or multi electrode EOD recordings of weakly electric fish.',
        epilog='by bendalab (2015-2016)')
    parser.add_argument('file_path', nargs='?', default='', type=str, help='folder containing the data to analyse')

    args = parser.parse_args()
    main(args.file_path)