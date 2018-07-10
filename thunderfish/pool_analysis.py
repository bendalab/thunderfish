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

def create_plot(datafile, shift, fish_nr_in_rec, colors):
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
    ax.set_xlabel('time [s]')

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

    #  Datei   1        2         3      4       5       6      7       8        9      10      11     12      13       14     15      16      17      18      19      20       21      22      23     24
    fish_nr_in_rec = [[7314  , 2071  , 107834, 157928, 8     , 2     , 18372 , 4     , 4     , 0     , 7     , 5     , 6     , 50283 , 21    , 28    , 7     , 11    , 19    , 76    , np.nan, 0     , 12    , 9     ],
                      [88    , 3541  , 107833, 158010, 16501 , 8     , 17287 , 26    , 32478 , 1     , 31    , 2     , 11    , 4     , 29496 , 6     , 19    , 37560 , 24    , 3     , np.nan, 4     , 123281, 164289],
                      [7315  , 9103  , 107256, 158179, 3     , 45    , 7     , 3     , 3     , 25208 , 32881 , 38054 , 47218 , 66437 , 9402  , 56948 , 6     , 50447 , 90962 , 45002 , np.nan, 3     , 4     , 31274 ],
                      [4627  , 9102  , 107832, 158205, 1     , 3     , 2514  , 2     , 10    , 32    , 47    , 25482 , 12638 , 66841 , 53    , 56949 , 25745 , 57594 , 24839 , 62328 , np.nan, 7     , 2     , 152249],
                      [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 38916 , 8     , 46503 , 15    , 26    , 9     , 57152 , 75735 , 45    , np.nan, 24409 , 8     , 3     ],
                      [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 23554 , 38328 , np.nan, 2     , 4     , 41729 , 55107 , 7     , 84    , np.nan, 3810  , 6     , 2     ],
                      [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 3155  , 2144  , 12    , 2     , 7     , np.nan, 1     , 124425, 164278],
                      [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 1104  , 18    , 5     , 10973 , 57578 , 42    , 81580 , np.nan, 21    , 72486 , 164288],
                      [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 4516  , 8     , 4     , 3     , 1     , 25    , 11411 , 3     , 57579 , 21618 , 247   , np.nan, 2     , 120610, 5     ],
                      [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 65093 , 59600 , 44    , 0     , 42932 , 6     , 108   , np.nan, 39100 , 5     , 54975 ],
                      [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 15004 , 24342 , 27327 , 34423 , 2     , 1099  , 4     , 31613 , 8     , 7865  , 4272  , 57593 , 3394  , 74472 , np.nan, 12    , 10    , 1     ],
                      [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 39778 , np.nan, 1227  , 2     , 6     , 59560 , 1878  , 81    , 57592 , np.nan, 29543 , np.nan, 37650 , 46043 , 56279 ],
                      [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 56947 , 38877 , 8     , 34    , 12405 , np.nan, 25536 , 15    , 0     ],
                      [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 17544 , 47    , 31    , 14    , 26840 , 10    , 63    , 48125 , 146   , 56950 , 39918 , 6     , 25858 , 6     , np.nan, 189   , 134   , 11    ]]

    colors = ['#BA2D22', '#53379B', '#F47F17', '#3673A4', '#AAB71B', '#DC143C', '#1E90FF', '#BA2D22', '#53379B', '#F47F17', '#3673A4', '#AAB71B', '#DC143C', '#1E90FF']
    ####################################################################################################################
    datafile = datafile[:6]
    shift = shift[:6]

    ###########################################################
    ### BEIDE ###
    create_plot(datafile, shift, fish_nr_in_rec, colors)

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


    # fig, ax = plt.subplots(facecolor='white', figsize=(20/2.54, 12/2.54))
    # for fish_nr in range(len(bin_std)):
    #     ax.plot(bin_freq_centers, bin_std[fish_nr], color=colors[fish_nr])
    #     ax.set_ylabel('bin frequency standard deviation [Hz]')
    #     ax.set_xlabel('time [s]')

    percentiles = np.nanpercentile(bin_std, (25, 50, 75), axis=0)
    fig, ax = plt.subplots()
    ax.plot(bin_freq_centers, percentiles[1], color='red')
    ax.fill_between(bin_freq_centers, percentiles[0], percentiles[2], color='cornflowerblue', alpha=0.5)
    ax.set_xlabel('time[s]')
    ax.set_ylabel('rel. freqeuncy activity')

    unique_centers = np.unique(np.array(bin_freq_centers) % (24 * 60 * 60))
    day_std_p_f = [[[] for c in unique_centers] for f in fish_nr_in_rec]

    for fish_nr in range(len(fish_nr_in_rec)):
        for bin_nr in range(len(bin_std[fish_nr])):
            if ~np.isnan(bin_std[fish_nr][bin_nr]):
                day_std_p_f[fish_nr][int(bin_nr % len(unique_centers))].append(bin_std[fish_nr][bin_nr])

    for fish_nr in range(len(day_std_p_f)):
        fig, ax = plt.subplots(facecolor='white', figsize=(20/2.54, 12/2.54))
        ax.boxplot(day_std_p_f[fish_nr], sym='')
        ax.set_ylabel('rel. frequency activity')
        ax.set_xlabel('bin')
        ax.set_title('fish Nr. %.0f' % fish_nr)
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

    fig, ax = plt.subplots()
    ax.boxplot(day_std, sym='')

    fig, ax = plt.subplots()
    ax.boxplot(day_std_m, sym='')

    fig, ax = plt.subplots()
    ax.boxplot(day_std_f, sym='')

    fig, ax = plt.subplots()
    ax.boxplot([md, fd, mn, fn], sym='')

    plt.show()

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
    plt.show()

if __name__ == '__main__':
    main()