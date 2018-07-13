import matplotlib.pyplot as plt
import numpy as np
import os
from IPython import embed
import time

def loaddata(datafile):
    print('loading datafile: %s' % datafile)
    fund_v=np.load(datafile+"/fund_v.npy")
    ident_v=np.load(datafile+"/ident_v.npy")
    idx_v=np.load(datafile+"/idx_v.npy")
    times=np.load(datafile+"/times.npy")
    sign_v=np.load(datafile+"/sign_v.npy")
    times_v=times[idx_v]
    return fund_v,ident_v,idx_v,times_v,sign_v

def create_plot(datafile, shift, fish_nr_in_rec, colors, saving_folder):
    print('plotting traces')
    fig, ax = plt.subplots(facecolor='white', figsize=(20. / 2.54, 12. / 2.54))
    for datei_nr in range(len(datafile)):
        fund_v, ident_v, idx_v, times_v, sign_v = loaddata(datafile[datei_nr])
        times_v += shift[datei_nr]

        for fish_nr in range(len(fish_nr_in_rec)):
            if np.isnan(fish_nr_in_rec[fish_nr][datei_nr]):
                continue

            f = fund_v[ident_v == fish_nr_in_rec[fish_nr][datei_nr]]
            t = times_v[ident_v == fish_nr_in_rec[fish_nr][datei_nr]]

            ax.plot(t, f, color=colors[fish_nr])

    ax.set_ylabel('frequency [Hz]')
    ax.set_xlabel('time')
    ticks = np.arange(110 * 60 + 9 * 60 * 60, times_v[-1], 96*60*60)
    tick_labels = ['24:00\n05.05.18', '24:00\n09.05.18','24:00\n13.05.18', '24:00\n17.05.18','24:00\n21.05.18', '24:00\n25.05.18', '24:00\n29.05.18']
    ax.set_xticks(ticks)
    ax.set_xticklabels(tick_labels)
    plt.tight_layout()
    fig.savefig(saving_folder + 'all_freq_traces.pdf')
    plt.close()



def extract_bin_freqs(datafile, shift, fish_nr_in_rec, bin_start = 0, bw = 300):
    print('extracting bin frequencies')
    current_bin_freq = [[] for f in fish_nr_in_rec]
    bin_freq = [[] for f in fish_nr_in_rec]
    centers = []

    for datei_nr in range(len(datafile)):
        fund_v, ident_v, idx_v, times_v, sign_v = loaddata(datafile[datei_nr])
        times_v += shift[datei_nr]

        ###
        while True:
            for fish_nr in range(len(fish_nr_in_rec)):
                current_bin_freq[fish_nr].extend(fund_v[(ident_v == fish_nr_in_rec[fish_nr][datei_nr]) &
                                                        (times_v >= bin_start) &
                                                        (times_v < bin_start + bw)])
            if bin_start + bw > times_v[-1]:
                break
            else:
                for fish_nr in range(len(fish_nr_in_rec)):
                    bin_freq[fish_nr].append(current_bin_freq[fish_nr])
                    current_bin_freq[fish_nr] = []
                    # print([len(doi[i]) for i in range(len(doi))])
                centers.append(bin_start + bw / 2)
                bin_start += bw

    return bin_freq, centers

def extract_bin_sign(datafile, shift, fish_nr_in_rec, bin_start = 1200, bw = 1800):
    print('extracting bin signatures')
    current_bin_sign = [[] for f in fish_nr_in_rec    ]
    bin_sign = [[] for f in fish_nr_in_rec]
    centers = []

    for datei_nr in range(len(datafile)):
        fund_v, ident_v, idx_v, times_v, sign_v = loaddata(datafile[datei_nr])
        times_v += shift[datei_nr]

        while True:
            for fish_nr in range(len(fish_nr_in_rec)):
                current_bin_sign[fish_nr].extend(sign_v[(ident_v == fish_nr_in_rec[fish_nr][datei_nr]) &
                                                        (times_v >= bin_start) &
                                                        (times_v < bin_start + bw)])
            if bin_start + bw > times_v[-1]:
                break
            else:
                for fish_nr in range(len(fish_nr_in_rec)):
                    for line in np.arange(len(current_bin_sign[fish_nr])-1)+1:
                        if np.isnan(current_bin_sign[fish_nr][line][0]):
                            current_bin_sign[fish_nr][line] = current_bin_sign[fish_nr][line]

                    bin_sign[fish_nr].append(current_bin_sign[fish_nr])
                    current_bin_sign[fish_nr] = []
                    # print([len(doi[i]) for i in range(len(doi))])
                centers.append(bin_start + bw / 2)
                bin_start += bw

    return bin_sign, centers

def main():
    ################################################################################
    ### define datafiles, shifts for each datafile, fish numbers in recordings
    if os.path.exists('/home/raab'):
        path_start = '/home/raab'
    elif os.path.exists('/home/wurm'):
        path_start = '/home/wurm'
    elif os.path.exists('/home/linhart'):
        path_start = '/home/linhart'
    else:
        path_start = ''
        print('no data found ... new user ? contact Till Raab / check connection to server')
        quit()
    saving_folder = path_start + '/analysis/'
    datafile=[path_start + '/data/kraken_link/2018-05-04-13_10',
              path_start + '/data/kraken_link/2018-05-04-14:46',
              path_start + '/data/kraken_link/2018-05-04-16:44',
              path_start + '/data/kraken_link/2018-05-05-09:17',
              path_start + '/data/kraken_link/2018-05-06-12:14',
              path_start + '/data/kraken_link/2018-05-06-22:04',
              path_start + '/data/kraken_link/2018-05-07-08:55',
              path_start + '/data/kraken_link/2018-05-07-12:06',
              path_start + '/data/kraken_link/2018-05-07-16:14',
              path_start + '/data/kraken_link/2018-05-08-10:20',
              path_start + '/data/kraken_link/2018-05-08-17:50',
              path_start + '/data/kraken_link/2018-05-09-09:24',
              path_start + '/data/kraken_link/2018-05-09-16:53',
              path_start + '/data/kraken_link/2018-05-10-09:30',
              path_start + '/data/kraken_link/2018-05-10-20:04',
              path_start + '/data/kraken_link/2018-05-11-09:49',
              path_start + '/data/kraken_link/2018-05-11-20:21',
              path_start + '/data/kraken_link/2018-05-12-09:43',
              path_start + '/data/kraken_link/2018-05-12-19:25',
              path_start + '/data/kraken_link/2018-05-13-09:49',
              path_start + '/data/kraken_link/2018-05-13-20:08',
              path_start + '/data/kraken_link/2018-05-14-10:04',
              path_start + '/data/kraken_link/2018-05-22-14:21',
              path_start + '/data/kraken_link/2018-05-29-13:35']

    shift = [0, 5760, 12840, 72420, 169440, 204840, 243900, 255360, 270240, 335400, 362400, 418440, 445380, 505200, 543240, 592740, 630660, 678780, 713700, 765540, 802680, 852840, 1559460, 2161500]

    #  Datei            1        2         3      4       5       6      7       8        9      10      11     12      13       14     15      16      17      18      19      20       21      22      23     24
    # fish_nr_in_rec = [[7314  , 2071  , 107834, 157928, 8     , 2     , 18372 , 4     , 4     , 0     , 7     , 5     , 6     , 50283 , 21    , 28    , 7     , 11    , 19    , 76    , np.nan, 0     , 12    , 9     ],
    #                   [88    , 3541  , 107833, 158010, 16501 , 8     , 17287 , 26    , 32478 , 1     , 31    , 2     , 11    , 4     , 29496 , 6     , 19    , 37560 , 24    , 3     , np.nan, 4     , 123281, 164289],
    #                   [7315  , 9103  , 107256, 158179, 3     , 45    , 7     , 3     , 3     , 25208 , 32881 , 38054 , 47218 , 66437 , 9402  , 56948 , 6     , 50447 , 90962 , 45002 , np.nan, 3     , 4     , 31274 ],
    #                   [4627  , 9102  , 107832, 158205, 1     , 3     , 2514  , 2     , 10    , 32    , 47    , 25482 , 12638 , 66841 , 53    , 56949 , 25745 , 57594 , 24839 , 62328 , np.nan, 7     , 2     , 152249],
    #                   [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 38916 , 8     , 46503 , 15    , 26    , 9     , 57152 , 75735 , 45    , np.nan, 24409 , 8     , 3     ],
    #                   [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 23554 , 38328 , np.nan, 2     , 4     , 41729 , 55107 , 7     , 84    , np.nan, 3810  , 6     , 2     ],
    #                   [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 3155  , 2144  , 12    , 2     , 7     , np.nan, 1     , 124425, 164278],
    #                   [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 1104  , 18    , 5     , 10973 , 57578 , 42    , 81580 , np.nan, 21    , 72486 , 164288],
    #                   [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 4516  , 8     , 4     , 3     , 1     , 25    , 11411 , 3     , 57579 , 21618 , 247   , np.nan, 2     , 120610, 5     ],
    #                   [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 65093 , 59600 , 44    , 0     , 42932 , 6     , 108   , np.nan, 39100 , 5     , 54975 ],
    #                   [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 15004 , 24342 , 27327 , 34423 , 2     , 1099  , 4     , 31613 , 8     , 7865  , 4272  , 57593 , 3394  , 74472 , np.nan, 12    , 10    , 1     ],
    #                   [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 39778 , np.nan, 1227  , 2     , 6     , 59560 , 1878  , 81    , 57592 , np.nan, 29543 , np.nan, 37650 , 46043 , 56279 ],
    #                   [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 56947 , 38877 , 8     , 34    , 12405 , np.nan, 25536 , 15    , 0     ],
    #                   [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 17544 , 47    , 31    , 14    , 26840 , 10    , 63    , 48125 , 146   , 56950 , 39918 , 6     , 25858 , 6     , np.nan, 189   , 134   , 11    ]]

    fish_nr_in_rec = [[7314, 2071, 107834, 157928, 8, 2, 18372, 4, 4, 0, 7, 5, 6, 50283, 21, 28, 7, 11, 19, 76, 0, 0, 12, 9],
                      [88, 3541, 107833, 158010, 16501, 8, 17287, 26, 32478, 1, 31, 2, 11, 4, 29496, 6, 19, 37560, 24, 3, 37192, 4, 123281, 164289],
                      [7315, 9103, 107256, 158179, 3, 45, 7, 3, 3, 25208, 32881, 38054, 47218, 66437, 9402, 56948, 6, 50447, 90962, 45002, 217, 3, 4, 31274],
                      [4627, 9102, 107832, 158205, 1, 3, 2514, 2, 10, 32, 47, 25482, 12638, 66841, 53, 56949, 25745, 57594, 24839, 62328, 6, 24409, 8, 3],
                      [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 38916, 8, 46503, 15, 26, 9, 57152, 75735, 45, 24367, 7, 2, 152249],
                      [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 23554, 38328, np.nan, 2, 4, 41729, 55107, 7, 84, 16706, 3810, 6, 2],
                      [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 3155, 2144, 12, 2, 7, 117, 1, 124425, 164278],
                      [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 1104, 18, 5, 10973, 57578, 42, 81580, 86637, 21, 72486, 164288],
                      [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 4516, 8, 4, 3, 1, 25, 11411, 3, 57579, 21618, 247, 28786, 2, 120610, 5],
                      [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 65093, 59600, 44, 0, 42932, 6, 108, 8, 39100, 5, 54975],
                      [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 15004, 24342, 27327, 34423, 2, 1099, 4, 31613, 8, 7865, 4272, 57593, 3394, 74472, 3, 12, 10, 1],
                      [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 39778, np.nan, 1227, 2, 6, 59560, 1878, 81, 57592, np.nan, 29543, 16994, 37650, 46043, 56279],
                      [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 56947, 38877, 8, 34, 12405, 388, 25536, 15, 0],
                      [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 17544, 47, 31, 14, 26840, 10, 63, 48125, 146, 56950, 39918, 6, 25858, 6, 88393, 189, 134, 11]]

    colors = ['#BA2D22', '#53379B', '#F47F17', '#3673A4', '#AAB71B', '#DC143C', '#1E90FF', '#BA2D22', '#53379B', '#F47F17', '#3673A4', '#AAB71B', '#DC143C', '#1E90FF']
    ####################################################################################################################
    # datafile = datafile[:6]
    # shift = shift[:6]

    ###########################################################
    ### BEIDE ###
    create_plot(datafile, shift, fish_nr_in_rec, colors, saving_folder)
    # quit()
    ###########################################################
    ### ANNA ###

    bin_freq, bin_freq_centers = extract_bin_freqs(datafile, shift, fish_nr_in_rec, bin_start= 300, bw=900)

    bin_std = [[] for f in fish_nr_in_rec]

    for fish_nr in range(np.shape(bin_freq)[0]):
        for bin_nr in range(np.shape(bin_freq)[1]):
            if len(bin_freq[fish_nr][bin_nr]) == 0:
                bin_std[fish_nr].append(np.nan)
                continue
            current_fish_n_bin_std = np.std(bin_freq[fish_nr][bin_nr], ddof=1)
            bin_std[fish_nr].append(current_fish_n_bin_std)

    fig2, ax2 = plt.subplots(facecolor='white', figsize=(20/2.54, 12/2.54))
    for fish_nr in range(len(bin_std)):
        ax2.plot(bin_freq_centers, bin_std[fish_nr], color=colors[fish_nr])
        ax2.set_ylabel('rel. freq activity / 0.25h')
        ax2.set_xlabel('time [s]')
    plt.tight_layout()
    fig2.savefig(saving_folder + 'all_freq_activity.pdf')
    plt.close()


    dn_borders = np.arange(110*60, 2250000 + 12*60*60, 12*60*60)
    day_activity = []
    night_activity = []
    for i in range(len(dn_borders)-1):
        if i % 2 == 1:
            day_activity.append([])
        else:
            night_activity.append([])

        for fish_nr in range(len(bin_std)):
            std_oi = np.array(bin_std[fish_nr])[(bin_freq_centers >= dn_borders[i]) & (bin_freq_centers < dn_borders[i + 1])]
            if i % 2 == 1:
                day_activity[-1].extend(std_oi[~np.isnan(std_oi)])
            else:
                night_activity[-1].extend(std_oi[~np.isnan(std_oi)])

    day_activity_per_fishcount = [[] for f in range(6)]
    night_activity_per_fishcount = [[] for f in range(6)]

    for enu, act in enumerate(day_activity):
        if enu <= 2:
            day_activity_per_fishcount[0].extend(act)
        elif enu == 3:
            day_activity_per_fishcount[1].extend(act)
        elif enu == 4:
            day_activity_per_fishcount[2].extend(act)
        elif enu == 5:
            day_activity_per_fishcount[3].extend(act)
        elif enu == 6:
            day_activity_per_fishcount[4].extend(act)
        else:
            day_activity_per_fishcount[5].extend(act)

    for enu, act in enumerate(night_activity):
        if enu <= 1:
            night_activity_per_fishcount[0].extend(act)
        elif enu == 2:
            night_activity_per_fishcount[1].extend(act)
        elif enu == 3:
            night_activity_per_fishcount[2].extend(act)
        elif enu == 4:
            night_activity_per_fishcount[3].extend(act)
        elif enu == 5:
            night_activity_per_fishcount[4].extend(act)
        else:
            night_activity_per_fishcount[5].extend(act)

    fig, ax = plt.subplots(facecolor='white', figsize=(20/2.54, 12/2.54))
    ax.boxplot(day_activity_per_fishcount, sym = '')
    ax.set_title('day activity per fishcount')
    ax.set_xlabel('fishcount')
    ax.set_ylabel('frequency activity [Hz]')
    ax.set_xticks([1, 2, 3, 4, 5, 6])
    ax.set_xticklabels([4, 6, 8, 10, 12, 14])
    plt.tight_layout()
    fig.savefig(saving_folder + 'day_freq_activity_per_fishcount.pdf')
    plt.close()

    fig, ax = plt.subplots(facecolor='white', figsize=(20/2.54, 12/2.54))
    ax.boxplot(night_activity_per_fishcount, sym = '')
    ax.set_title('night activity per fishcount')
    ax.set_xlabel('fishcount')
    ax.set_ylabel('frequency activity [Hz]')
    ax.set_xticks([1, 2, 3, 4, 5, 6])
    ax.set_xticklabels([4, 6, 8, 10, 12, 14])
    plt.tight_layout()
    fig.savefig(saving_folder + 'night_freq_activity_per_fishcount.pdf')
    plt.close()

    fig, ax = plt.subplots(facecolor='white', figsize=(20/2.54, 12/2.54))
    ax.boxplot(day_activity, sym = '')
    ax.set_xlabel('day')
    ax.set_ylabel('frequency activity [Hz]')
    ax.set_title('day activity consecutive days')
    plt.tight_layout()
    fig.savefig(saving_folder + 'day_freq_activity_per_day.pdf')
    plt.close()


    fig, ax = plt.subplots(facecolor='white', figsize=(20/2.54, 12/2.54))
    ax.boxplot(night_activity, sym = '')
    ax.set_xlabel('day')
    ax.set_ylabel('frequency activity [Hz]')
    ax.set_title('night activity consecutive days')
    plt.tight_layout()
    fig.savefig(saving_folder + 'night_freq_activity_per_day.pdf')
    plt.close()

    percentiles = np.nanpercentile(bin_std, (25, 50, 75), axis=0)
    fig, ax = plt.subplots(facecolor='white', figsize=(20/2.54, 12/2.54))
    ax.plot(bin_freq_centers, percentiles[1], color='red')
    ax.fill_between(bin_freq_centers, percentiles[0], percentiles[2], color='cornflowerblue', alpha=0.5)
    ax.set_xlabel('time')
    ticks = np.arange(110 * 60 + 9 * 60 * 60, bin_freq_centers[-1], 96*60*60)
    tick_labels = ['24:00\n05.05.18', '24:00\n09.05.18','24:00\n13.05.18', '24:00\n17.05.18','24:00\n21.05.18', '24:00\n25.05.18', '24:00\n29.05.18']
    ax.set_xticks(ticks)
    ax.set_xticklabels(tick_labels)
    ax.set_ylabel('rel. freqeuncy activity')
    plt.tight_layout()
    fig.savefig(saving_folder + 'all_median_freq_activity.pdf')
    plt.close()

    unique_centers = np.unique(np.array(bin_freq_centers) % (24 * 60 * 60))
    day_std_p_f = [[[] for c in unique_centers] for f in fish_nr_in_rec]

    for fish_nr in range(len(fish_nr_in_rec)):
        for bin_nr in range(len(bin_std[fish_nr])):
            if ~np.isnan(bin_std[fish_nr][bin_nr]):
                day_std_p_f[fish_nr][int(bin_nr % len(unique_centers))].append(bin_std[fish_nr][bin_nr])

    for fish_nr in range(len(day_std_p_f)):
        fig, ax = plt.subplots(facecolor='white', figsize=(20/2.54, 12/2.54))
        ax.boxplot(day_std_p_f[fish_nr], sym='')
        ax.set_ylabel('frequency activity [Hz]')
        ax.set_xlabel('time')
        ax.set_title('fish Nr. %.0f' % fish_nr)
        ax.set_xticks(np.arange(4, len(day_std_p_f[fish_nr]) + 4, 4))
        ax.set_xticklabels(
            ['14:00', '15:00', '16:00', '17:00', '18:00', '19:00', '20:00', '21:00', '22:00', '23:00', '24:00', '01:00',
             '02:00', '03:00', '04:00', '05:00', '06:00', '07:00', '08:00', '09:00', '10:00', '11:00', '12:00',
             '13:00'],
            rotation=70)
        plt.tight_layout()
        fig.savefig(saving_folder + '24h_freq_activity_fish%.0f.pdf' % fish_nr)
        plt.close()
    # plt.show()

    day_std = [[] for c in unique_centers]
    day_std_m = [[] for c in unique_centers]
    day_std_f = [[] for c in unique_centers]

    mn = []
    md = []
    fn = []
    fd = []
    # embed()
    # quit()

    for fish_nr in range(len(fish_nr_in_rec)):
        for bin_nr in range(len(bin_std[0])):
            if not np.isnan(bin_std[fish_nr][bin_nr]):

                if fish_nr <= 5:
                    day_std_m[int(bin_nr % len(unique_centers))].append(bin_std[fish_nr][bin_nr])
                    if unique_centers[int(bin_nr % len(unique_centers))] >= 110 * 60 and unique_centers[int(bin_nr % len(unique_centers))] < 110 * 60 + 12 * 60 * 60:
                        mn.append(bin_std[fish_nr][bin_nr])
                    else:
                        md.append(bin_std[fish_nr][bin_nr])
                else:
                    day_std_f[int(bin_nr % len(unique_centers))].append(bin_std[fish_nr][bin_nr])
                    if unique_centers[int(bin_nr % len(unique_centers))] >= 110 * 60 and unique_centers[int(bin_nr % len(unique_centers))] < 110 * 60 + 12 * 60 * 60:
                        fn.append(bin_std[fish_nr][bin_nr])
                    else:
                        fd.append(bin_std[fish_nr][bin_nr])

                day_std[int(bin_nr % len(unique_centers))].append(bin_std[fish_nr][bin_nr])

    fig, ax = plt.subplots(facecolor='white', figsize=(20/2.54, 12/2.54))
    ax.boxplot(day_std, sym='')
    ax.set_xticks(np.arange(4, len(day_std)+4, 4))
    ax.set_xticklabels(['14:00', '15:00','16:00','17:00','18:00','19:00','20:00','21:00','22:00','23:00','24:00','01:00',
                        '02:00','03:00','04:00','05:00','06:00','07:00','08:00','09:00','10:00','11:00','12:00','13:00'],
                       rotation=70)
    ax.set_xlabel('time')
    ax.set_ylabel('frequency activity [Hz]')
    ax.set_title('frequency activity of all fish; n = 14')
    plt.tight_layout()
    fig.savefig(saving_folder + '24h_day_freq_act_male_and_female.pdf')
    plt.close()
    # embed()
    # quit()


    fig, ax = plt.subplots(facecolor='white', figsize=(20/2.54, 12/2.54))
    ax.boxplot(day_std_m, sym='')
    ax.set_xticks(np.arange(4, len(day_std)+4, 4))
    ax.set_xticklabels(['14:00', '15:00','16:00','17:00','18:00','19:00','20:00','21:00','22:00','23:00','24:00','01:00',
                        '02:00','03:00','04:00','05:00','06:00','07:00','08:00','09:00','10:00','11:00','12:00','13:00'],
                       rotation=70)
    ax.set_xlabel('time')
    ax.set_ylabel('frequency activity [Hz]')
    ax.set_title('frequency activity of all male fish; n = 6')
    plt.tight_layout()
    fig.savefig(saving_folder + '24h_day_freq_act_male.pdf')
    plt.close()

    fig, ax = plt.subplots(facecolor='white', figsize=(20/2.54, 12/2.54))
    ax.boxplot(day_std_f, sym='')
    ax.set_xticks(np.arange(4, len(day_std)+4, 4))
    ax.set_xticklabels(['14:00', '15:00','16:00','17:00','18:00','19:00','20:00','21:00','22:00','23:00','24:00','01:00',
                        '02:00','03:00','04:00','05:00','06:00','07:00','08:00','09:00','10:00','11:00','12:00','13:00'],
                       rotation=70)
    ax.set_xlabel('time')
    ax.set_ylabel('frequency activity [Hz]')
    ax.set_title('frequency activity of all female fish; n = 8')
    plt.tight_layout()
    fig.savefig(saving_folder + '24h_day_freq_act_female.pdf')
    plt.close()

    fig, ax = plt.subplots(facecolor='white', figsize=(20/2.54, 12/2.54))
    ax.boxplot([md, fd, mn, fn], sym='')
    ax.set_xticks([1, 2, 3, 4])
    ax.set_title('frequency activity')
    ax.set_ylabel('frequency activity [Hz]')
    ax.set_xticklabels(['male\nday', 'female\nday', 'male\nnight', 'female\nnight'])
    plt.tight_layout()
    fig.savefig(saving_folder + 'male_female_day_night_freq_activity.pdf')
    plt.close()


    # plt.show()

    ###########################################################
    ### LAURA ###

    bin_sign, bin_sign_centers = extract_bin_sign(datafile, shift, fish_nr_in_rec, bin_start= 1200, bw=1800)

    bin_poss = [[] for f in fish_nr_in_rec]

    for fish_nr in range(np.shape(bin_sign)[0]):
        for bin_nr in range(np.shape(bin_sign)[1]):
            if bin_sign[fish_nr][bin_nr] == []:
                bin_poss[fish_nr].append(np.array([]))
                continue

            current_fish_n_bin_max_elec = np.argmax(bin_sign[fish_nr][bin_nr], axis= 1)
            bin_poss[fish_nr].append(current_fish_n_bin_max_elec)

    activity_in_bin = [[] for f in fish_nr_in_rec]

    for fish_nr in range(len(bin_poss)):
        for bin_nr in range(len(bin_poss[fish_nr])):
            if len(bin_poss[fish_nr][bin_nr]) == 0:
                activity_in_bin[fish_nr].append(np.nan)
                continue
            electorde_nr_diff = np.diff(bin_poss[fish_nr][bin_nr])
            movement_activity = len(electorde_nr_diff[electorde_nr_diff > 0]) / len(electorde_nr_diff)
            activity_in_bin[fish_nr].append(movement_activity)

    fig2, ax2 = plt.subplots(facecolor='white', figsize=(20/2.54, 12/2.54))
    for fish_nr in range(len(activity_in_bin)):
        ax2.plot(bin_sign_centers, activity_in_bin[fish_nr], color=colors[fish_nr])
        ax2.set_ylabel('rel. movement activity / 0.5h')
        ax2.set_xlabel('time [s]')
    # plt.show()
    plt.tight_layout()
    fig2.savefig(saving_folder + 'all_movement_activity.pdf')
    plt.close()

    dn_borders = np.arange(110 * 60, 2250000 + 12 * 60 * 60, 12 * 60 * 60)
    day_activity = []
    night_activity = []
    for i in range(len(dn_borders) - 1):
        if i % 2 == 1:
            day_activity.append([])
        else:
            night_activity.append([])

        for fish_nr in range(len(activity_in_bin)):
            std_oi = np.array(activity_in_bin[fish_nr])[
                (bin_sign_centers >= dn_borders[i]) & (bin_sign_centers < dn_borders[i + 1])]
            if i % 2 == 1:
                day_activity[-1].extend(std_oi[~np.isnan(std_oi)])
            else:
                night_activity[-1].extend(std_oi[~np.isnan(std_oi)])

    day_activity_per_fishcount = [[] for f in range(6)]
    night_activity_per_fishcount = [[] for f in range(6)]

    for enu, act in enumerate(day_activity):
        if enu <= 2:
            day_activity_per_fishcount[0].extend(act)
        elif enu == 3:
            day_activity_per_fishcount[1].extend(act)
        elif enu == 4:
            day_activity_per_fishcount[2].extend(act)
        elif enu == 5:
            day_activity_per_fishcount[3].extend(act)
        elif enu == 6:
            day_activity_per_fishcount[4].extend(act)
        else:
            day_activity_per_fishcount[5].extend(act)

    for enu, act in enumerate(night_activity):
        if enu <= 1:
            night_activity_per_fishcount[0].extend(act)
        elif enu == 2:
            night_activity_per_fishcount[1].extend(act)
        elif enu == 3:
            night_activity_per_fishcount[2].extend(act)
        elif enu == 4:
            night_activity_per_fishcount[3].extend(act)
        elif enu == 5:
            night_activity_per_fishcount[4].extend(act)
        else:
            night_activity_per_fishcount[5].extend(act)

    fig, ax = plt.subplots(facecolor='white', figsize=(20/2.54, 12/2.54))
    ax.boxplot(day_activity_per_fishcount, sym='')
    ax.set_title('day activity per fishcount')
    ax.set_xlabel('fishcount')
    ax.set_ylabel('movement activity')
    ax.set_xticks([1, 2, 3, 4, 5, 6])
    ax.set_xticklabels([4, 6, 8, 10, 12, 14])
    plt.tight_layout()
    fig.savefig(saving_folder + 'day_movement_activity_per_fishcount.pdf')
    plt.close()

    fig, ax = plt.subplots(facecolor='white', figsize=(20/2.54, 12/2.54))
    ax.boxplot(night_activity_per_fishcount, sym='')
    ax.set_title('night activity per fishcount')
    ax.set_xlabel('fishcount')
    ax.set_ylabel('movement activity')
    ax.set_xticks([1, 2, 3, 4, 5, 6])
    ax.set_xticklabels([4, 6, 8, 10, 12, 14])
    plt.tight_layout()
    fig.savefig(saving_folder + 'night_movement_activity_per_fishcount.pdf')
    plt.close()

    fig, ax = plt.subplots(facecolor='white', figsize=(20/2.54, 12/2.54))
    ax.boxplot(day_activity, sym='')
    ax.set_xlabel('day')
    ax.set_ylabel('movement activity')
    ax.set_title('day activity consecutive days')
    plt.tight_layout()
    fig.savefig(saving_folder + 'day_movement_activity_per_day.pdf')
    plt.close()

    fig, ax = plt.subplots(facecolor='white', figsize=(20/2.54, 12/2.54))
    ax.boxplot(night_activity, sym='')
    ax.set_xlabel('day')
    ax.set_ylabel('movement activity')
    ax.set_title('night activity consecutive days')
    plt.tight_layout()
    fig.savefig(saving_folder + 'night_movement_activity_per_day.pdf')
    plt.close()

    ####new end ####

    percentile = np.nanpercentile(activity_in_bin, (25, 50, 75), axis=0)
    fig, ax = plt.subplots(facecolor='white', figsize=(20 / 2.54, 12 / 2.54))
    ax.plot(bin_sign_centers, percentile[1], color='red', label='Median')
    ax.fill_between(bin_sign_centers, percentile[0], percentile[2], color='cornflowerblue', alpha=0.5, label='Quartile')
    ax.set_xlabel('Zeit [Tagen]')
    ax.set_ylabel('relative Bewegungsaktivität / 0.5h')
    ax.set_title('gemittelte Bewegungsaktivität aller Fische auf die Zeit')
    plt.tight_layout()
    fig.savefig(saving_folder + 'all_median_movement_activity.pdf')
    plt.close()
    # ax.set_xticks([2, 4, 6, 8])
    # ax.set_xticklabels(['a', 'b', 'c', 'd'])

    ##Schwimmaktivitaet aller Fische in Dateien gemittelt mithilfe von  Median und Quartilen

    plt.legend()

    # plot anzeigen
    # plt.show()

    # fig

    unique_day_centers = np.unique(np.array(bin_sign_centers) % (24 * 60 * 60))  # centers von einem tag in unique

    # day_activity = [[] for bin in unique_day_centers] # leere liste fuer alle moeglichen centers an einem tag
    day_activity = [[[] for bin in unique_day_centers] for f in fish_nr_in_rec]
    all_day_activity = [[] for bin in unique_day_centers]  # fuer alle tier

    all_male_activity = [[] for bin in unique_day_centers]  # fuer alle tier
    all_female_activity = [[] for bin in unique_day_centers]  # fuer alle tier

    mn = []
    mt = []
    wn = []
    wt = []

    for f_all in range(len(fish_nr_in_rec)):
        for bin in range(len(activity_in_bin[f_all])):  ## loop ueber die bins von allen tagen fuer den ersten fish
            if np.isnan(activity_in_bin[f_all][bin]):
                continue
            day_activity[f_all][bin % len(unique_day_centers)].append(
                activity_in_bin[f_all][bin])  # umsortieren der aktivitats daten
            all_day_activity[bin % len(unique_day_centers)].append(
                activity_in_bin[f_all][bin])  # umsortieren der aktivitats daten

            if f_all <= 5:
                if unique_day_centers[bin % len(unique_day_centers)] >= 110 * 60 and unique_day_centers[
                    bin % len(unique_day_centers)] < 110 * 60 + 12 * 60 * 60:
                    mn.append(activity_in_bin[f_all][bin])
                else:
                    mt.append(activity_in_bin[f_all][bin])
                all_male_activity[bin % len(unique_day_centers)].append(activity_in_bin[f_all][bin])
            else:
                if unique_day_centers[bin % len(unique_day_centers)] >= 110 * 60 and unique_day_centers[
                    bin % len(unique_day_centers)] < 110 * 60 + 12 * 60 * 60:
                    wn.append(activity_in_bin[f_all][bin])
                else:
                    wt.append(activity_in_bin[f_all][bin])
                all_female_activity[bin % len(unique_day_centers)].append(activity_in_bin[f_all][bin])

        fig, ax = plt.subplots(facecolor='white', figsize=(20 / 2.54, 12 / 2.54))
        ax.boxplot(day_activity[f_all], sym = '')  ##schwimmaktivitaet von Fischen in einzelen Grafiken von allen gegebenen Datenfies (24h)

        ax.set_xlabel('Zeit [Stunden]')
        ax.set_ylabel('relative Bewegungsaktivität / 0.5h')
        ax.set_title('Fisch %.0f' % f_all)
        plt.tight_layout()
        fig.savefig(saving_folder + '24h_movement_activity_fish%.0f.pdf' % f_all)
        plt.close()
        # ax.set_xticks([2, 4, 6, 8])
        # ax.set_xticklabels(['a', 'b', 'c', 'd'])

        # plt.legend()

        # plot anzeigen
        # plt.show()
    # fig
    fig, ax = plt.subplots(facecolor='white', figsize=(20 / 2.54, 12 / 2.54))
    ax.boxplot(all_day_activity,
               sym='')  ##schwimmaktivitaet von allen Fischen zusammen in einer Grafik von allen gegebenen Datenfiles(24h)
    ax.set_xlabel('Zeit [Stunden]')
    ax.set_xticks(np.arange(4, len(day_std)+4, 4))
    ax.set_xticklabels(['14:00', '15:00','16:00','17:00','18:00','19:00','20:00','21:00','22:00','23:00','24:00','01:00',
                        '02:00','03:00','04:00','05:00','06:00','07:00','08:00','09:00','10:00','11:00','12:00','13:00'],
                       rotation=70)
    ax.set_ylabel('relative Bewegungsaktivität / 0.5h')
    ax.set_title('Bewegungsaktivität aller Fische gemittelt über 24h')
    plt.tight_layout()
    fig.savefig(saving_folder + '24h_day_move_act_male_and_female.pdf')
    plt.close()
    # ax.set_xticks([2, 4, 6, 8])
    # ax.set_xticklabels(['a', 'b', 'c', 'd'])

    fig, ax = plt.subplots(facecolor='white', figsize=(20 / 2.54, 12 / 2.54))
    ax.boxplot(all_male_activity, sym='')  ##
    ax.set_xlabel('Zeit [Stunden]')
    ax.set_xticks(np.arange(4, len(day_std)+4, 4))
    ax.set_xticklabels(['14:00', '15:00','16:00','17:00','18:00','19:00','20:00','21:00','22:00','23:00','24:00','01:00',
                        '02:00','03:00','04:00','05:00','06:00','07:00','08:00','09:00','10:00','11:00','12:00','13:00'],
                       rotation=70)
    ax.set_ylabel('relative Bewegungsaktivität / 0.5h')
    ax.set_title('Bewegungsaktivität aller männlichen Fische gemittelt auf 24h')
    plt.tight_layout()
    fig.savefig(saving_folder + '24h_day_move_act_male.pdf')
    plt.close()
    # ax.set_xticks([2, 4, 6, 8])
    # ax.set_xticklabels(['a', 'b', 'c', 'd'])

    fig, ax = plt.subplots(facecolor='white', figsize=(20 / 2.54, 12 / 2.54))
    ax.boxplot(all_female_activity, sym='')  ##
    ax.set_xlabel('Zeit [Stunden]')
    ax.set_xticks(np.arange(4, len(day_std)+4, 4))
    ax.set_xticklabels(['14:00', '15:00','16:00','17:00','18:00','19:00','20:00','21:00','22:00','23:00','24:00','01:00',
                        '02:00','03:00','04:00','05:00','06:00','07:00','08:00','09:00','10:00','11:00','12:00','13:00'],
                       rotation=70)
    ax.set_ylabel('relative Bewegungsaktivität / 0.5h')
    ax.set_title('Bewegungsaktivität aller weiblichen Fische gemittelt auf 24h')
    plt.tight_layout()
    fig.savefig(saving_folder + '24h_day_move_act_female.pdf')
    plt.close()
    # ax.set_xticks([2, 4, 6, 8])
    # ax.set_xticklabels(['a', 'b', 'c', 'd'])

    fig, ax = plt.subplots(facecolor='white', figsize=(20 / 2.54, 12 / 2.54))
    ax.boxplot([mt, wt, mn, wn], sym='')  ##
    ax.set_xlabel('unterschiedliches Geschlecht bei Tag/Nacht')
    ax.set_ylabel('relative Bewegungsaktivität / 0.5h')
    ax.set_title('Bewegungsaktivität der Geschlechter zu unterschiedlichen Lichtverhältnissen')
    ax.set_xticks([1, 2, 3, 4])
    ax.set_xticklabels(['male\nday', 'female\nday', 'male\nnight', 'female\nnight'])
    plt.tight_layout()
    fig.savefig(saving_folder + 'male_female_day_night_move_activity.pdf')
    plt.close()
    # ax.set_xticklabels(['mt', 'wt', 'mn', 'wn'])

    corr_bin_std = np.hstack(np.array(bin_std)[:, 2:][:, ::2])
    corr_bin_activity = np.hstack(activity_in_bin)

    m1 = ~np.isnan(corr_bin_activity)
    m2 = ~np.isnan(corr_bin_std)

    corr_bin_activity = corr_bin_activity[m1 & m2]
    corr_bin_std = corr_bin_std[m1 & m2]

    fig, ax = plt.subplots(facecolor='white', figsize=(20/2.54, 12/2.54))
    ax.plot(corr_bin_activity, corr_bin_std, '.')
    ax.set_xlabel('movement activtity')
    ax.set_ylabel('frequency activtity [Hz]')
    ax.set_title('correlation of frequency and movement activity')
    plt.tight_layout()
    fig.savefig(saving_folder + 'correlation_move_and_freq_activity.pdf')
    plt.close()


    # plt.show()


if __name__ == '__main__':
    main()