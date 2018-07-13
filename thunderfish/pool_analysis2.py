import matplotlib.pyplot as plt
import numpy as np
import os
from IPython import embed
import time
# from tqdm import tqdm

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

def extract_freq_and_pos_array(datafile, shift, fish_nr_in_rec, datafile_nr):
    fund_v, ident_v, idx_v, times_v, sign_v = loaddata(datafile)
    times_v += shift[datafile_nr]

    i_range = np.arange(0, np.nanmax(idx_v) + 1)
    # i_range = np.arange(len(np.unique(times_v)))

    fish_freqs = [np.full(len(i_range), np.nan) for i in range(len(fish_nr_in_rec))]
    fish_pos = [np.full(len(i_range), np.nan) for i in range(len(fish_nr_in_rec))]

    for fish_nr in range(len(np.array(fish_nr_in_rec)[:, datafile_nr])):
        if np.isnan(fish_nr_in_rec[fish_nr][datafile_nr]):
            continue

        freq = fund_v[ident_v == fish_nr_in_rec[fish_nr][datafile_nr]]
        pos = np.argmax(sign_v[ident_v == fish_nr_in_rec[fish_nr][datafile_nr]], axis=1)

        idx = idx_v[ident_v == fish_nr_in_rec[fish_nr][datafile_nr]]


        filled_f = np.interp(np.arange(idx[0], idx[-1] + 1), idx, freq)
        filled_p = np.interp(np.arange(idx[0], idx[-1] + 1), idx, pos)
        filled_p = np.round(filled_p, 0)

        fish_freqs[fish_nr][idx[0]:idx[-1] + 1] = filled_f
        fish_pos[fish_nr][idx[0]:idx[-1] + 1] = filled_p

    fish_freqs = np.array(fish_freqs)
    fish_pos = np.array(fish_pos)

    if len(fish_freqs[0]) != len(np.unique(times_v)):
        # ToDo: look into this ....
        # print('length of arrays dont match in %s' % datafile)
        # print('adjusting...')
        fish_freqs = fish_freqs[:, :len(np.unique(times_v))]
        fish_pos = fish_pos[:, :len(np.unique(times_v))]
        # fish_freqs = fish_freqs[:]

    return  fish_freqs, fish_pos, np.unique(times_v)

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

    #  Datei   1        2         3      4       5       6      7       8        9      10      11     12      13       14     15      16      17      18      19      20       21      22      23     24
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

    fish_nr_in_rec = [
        [7314, 2071, 107834, 157928, 8, 2, 18372, 4, 4, 0, 7, 5, 6, 50283, 21, 28, 7, 11, 19, 76, 0, 0, 12, 9],
        [88, 3541, 107833, 158010, 16501, 8, 17287, 26, 32478, 1, 31, 2, 11, 4, 29496, 6, 19, 37560, 24, 3, 37192, 4,
         123281, 164289],
        [7315, 9103, 107256, 158179, 3, 45, 7, 3, 3, 25208, 32881, 38054, 47218, 66437, 9402, 56948, 6, 50447, 90962,
         45002, 217, 3, 4, 31274],
        [4627, 9102, 107832, 158205, 1, 3, 2514, 2, 10, 32, 47, 25482, 12638, 66841, 53, 56949, 25745, 57594, 24839,
         62328, 6, 24409, 8, 3],
        [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 38916, 8, 46503, 15,
         26, 9, 57152, 75735, 45, 24367, 7, 2, 152249],
        [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 23554, 38328, np.nan,
         2, 4, 41729, 55107, 7, 84, 16706, 3810, 6, 2],
        [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan,
         np.nan, 3155, 2144, 12, 2, 7, 117, 1, 124425, 164278],
        [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 1104,
         18, 5, 10973, 57578, 42, 81580, 86637, 21, 72486, 164288],
        [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 4516, 8, 4, 3, 1, 25, 11411, 3, 57579,
         21618, 247, 28786, 2, 120610, 5],
        [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 65093,
         59600, 44, 0, 42932, 6, 108, 8, 39100, 5, 54975],
        [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 15004, 24342, 27327, 34423, 2, 1099, 4, 31613, 8, 7865, 4272,
         57593, 3394, 74472, 3, 12, 10, 1],
        [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 39778, np.nan, 1227, 2, 6, 59560, 1878,
         81, 57592, np.nan, 29543, 16994, 37650, 46043, 56279],
        [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan,
         np.nan, 56947, 38877, 8, 34, 12405, 388, 25536, 15, 0],
        [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 17544, 47, 31, 14, 26840, 10, 63, 48125, 146, 56950, 39918, 6,
         25858, 6, 88393, 189, 134, 11]]
    colors = ['#BA2D22', '#53379B', '#F47F17', '#3673A4', '#AAB71B', '#DC143C', '#1E90FF', '#BA2D22', '#53379B', '#F47F17', '#3673A4', '#AAB71B', '#DC143C', '#1E90FF']

    start_n = 110 * 60
    end_n = 110 * 60 + 12 * 60 * 60
    day_sec = 24 * 60 * 60

    habitats = [[12, 8], [14, 10], [11, 15], [9, 13], [0, 1, 2, 3, 4, 5, 6, 7]]
    hab_colors = ['k', 'grey', 'green', 'yellow', 'lightblue']


    for datafile_nr in range(len(datafile)):
        fish_f, fish_p, t = extract_freq_and_pos_array(datafile[datafile_nr], shift, fish_nr_in_rec, datafile_nr)

        if datafile_nr == 0:
            fish_freqs = fish_f
            fish_pos = fish_p
            times = t
            # print(len(t), len(fish_f[0]))
        else:
            fish_freqs = np.append(fish_freqs, fish_f, axis=1)
            fish_pos = np.append(fish_pos, fish_p, axis=1)
            times = np.append(times, t)

    clock_sec = (times % day_sec)


    night_mask = np.arange(len(clock_sec))[(clock_sec >= start_n) & (clock_sec < end_n)]
    day_mask = np.arange(len(clock_sec))[(clock_sec < start_n) | (clock_sec >= end_n)]
    # print('\n')
    # print(len([x in day_mask for x in night_mask]))
    # embed()
    # quit()

    ################################## ANNA ###############################################
    df_on_electrode = [[] for elec in range(16)]
    d_df_on_electrode = [[] for elec in range(16)]
    n_df_on_electrode = [[] for elec in range(16)]

    # for fish_nr in tqdm(range(len(fish_freqs))):
    for fish_nr in range(len(fish_freqs)):
        for fish_nr_comp in np.arange(fish_nr+1, len(fish_freqs)):
            for elec in range(16):
                fish_freq_oi = fish_freqs[fish_nr][(fish_pos[fish_nr] == elec) & (fish_pos[fish_nr_comp] == elec)]
                d_fish_freq_oi = fish_freqs[fish_nr][day_mask][ (fish_pos[fish_nr][day_mask] == elec) & (fish_pos[fish_nr_comp][day_mask] == elec)]
                n_fish_freq_oi = fish_freqs[fish_nr][night_mask][ (fish_pos[fish_nr][night_mask] == elec) & (fish_pos[fish_nr_comp][night_mask] == elec)]

                comp_fish_freq_oi = fish_freqs[fish_nr_comp][(fish_pos[fish_nr] == elec) & (fish_pos[fish_nr_comp] == elec)]
                d_comp_fish_freq_oi = fish_freqs[fish_nr_comp][day_mask][(fish_pos[fish_nr][day_mask] == elec) & (fish_pos[fish_nr_comp][day_mask] == elec)]
                n_comp_fish_freq_oi = fish_freqs[fish_nr_comp][night_mask][(fish_pos[fish_nr][night_mask] == elec) & (fish_pos[fish_nr_comp][night_mask] == elec)]

                df_on_electrode[elec].extend(np.abs(fish_freq_oi - comp_fish_freq_oi))
                d_df_on_electrode[elec].extend(np.abs(d_fish_freq_oi - d_comp_fish_freq_oi))
                n_df_on_electrode[elec].extend(np.abs(n_fish_freq_oi - n_comp_fish_freq_oi))

    df_in_habitat = [[] for habitat in habitats]
    d_df_in_habitat = [[] for habitat in habitats]
    n_df_in_habitat = [[] for habitat in habitats]

    for hab_nr in range(len(habitats)):
        for elec in habitats[hab_nr]:
            df_in_habitat[hab_nr].extend(df_on_electrode[elec])
            d_df_in_habitat[hab_nr].extend(d_df_on_electrode[elec])
            n_df_in_habitat[hab_nr].extend(n_df_on_electrode[elec])

    ################################## LAURA ###############################################
    dn_borders = np.arange(110 * 60, 2250000 + 12 * 60 * 60, 12 * 60 * 60)

    fish_counts_on_electrode = np.zeros((len(fish_nr_in_rec), 16), dtype = int)
    n_fish_counts_on_electrode = np.zeros((len(fish_nr_in_rec), 16), dtype = int)
    d_fish_counts_on_electrode = np.zeros((len(fish_nr_in_rec), 16), dtype = int)

    sep_fish_counts_on_electrode = np.array([fish_counts_on_electrode for i in dn_borders[:-1]])

    fish_counts_in_habitat = np.zeros((len(fish_nr_in_rec), 5), dtype=int)
    n_fish_counts_in_habitat = np.zeros((len(fish_nr_in_rec), 5), dtype=int)
    d_fish_counts_in_habitat = np.zeros((len(fish_nr_in_rec), 5), dtype=int)

    sep_fish_counts_in_habitat = np.array([fish_counts_in_habitat for i in dn_borders[:-1]])

    for fish_nr in range(len(fish_nr_in_rec)):
        for e_nr in range(16):
            for i in range(len(dn_borders)-1):
                sep_fish_counts_on_electrode[i][fish_nr][e_nr] += len(fish_pos[fish_nr][(fish_pos[fish_nr] == e_nr) & (times >= dn_borders[i]) & (times < dn_borders[i+1])])
            fish_counts_on_electrode[fish_nr][e_nr] += len(fish_pos[fish_nr][fish_pos[fish_nr] == e_nr])

            n_fish_counts_on_electrode[fish_nr][e_nr] += len(fish_pos[fish_nr][night_mask][fish_pos[fish_nr][night_mask] == e_nr])
            d_fish_counts_on_electrode[fish_nr][e_nr] += len(fish_pos[fish_nr][day_mask][fish_pos[fish_nr][day_mask] == e_nr])

    for fish_nr in range(len(fish_counts_on_electrode)):
        for habitat_nr in range(len(habitats)):
            count_in_habitat = 0
            count_in_habitat_n = 0
            count_in_habitat_d = 0
            for ele in habitats[habitat_nr]:
                count_in_habitat += fish_counts_on_electrode[fish_nr][ele]
                count_in_habitat_n += n_fish_counts_on_electrode[fish_nr][ele]
                count_in_habitat_d += d_fish_counts_on_electrode[fish_nr][ele]

            fish_counts_in_habitat[fish_nr][habitat_nr] = count_in_habitat
            d_fish_counts_in_habitat[fish_nr][habitat_nr] = count_in_habitat_d
            n_fish_counts_in_habitat[fish_nr][habitat_nr] = count_in_habitat_n

    for dn_nr in range(len(sep_fish_counts_on_electrode)):
        for fish_nr in range(len(sep_fish_counts_on_electrode[dn_nr])):
            for habitat_nr in range(len(habitats)):
                count_in_habitat = 0
                for ele in habitats[habitat_nr]:
                    count_in_habitat += sep_fish_counts_on_electrode[dn_nr][fish_nr][ele]
                sep_fish_counts_in_habitat[dn_nr][fish_nr][habitat_nr] = count_in_habitat

    fish_counts_in_habitat = np.array(fish_counts_in_habitat)
    d_fish_counts_in_habitat = np.array(d_fish_counts_in_habitat)
    n_fish_counts_in_habitat = np.array(n_fish_counts_in_habitat)

    sep_fish_counts_in_habitat = np.array(sep_fish_counts_in_habitat)
    rel_sep_fish_counts_in_habitat = np.zeros(np.shape(sep_fish_counts_in_habitat))

    for dn_nr in range(len(sep_fish_counts_in_habitat)):
        for fish_nr in range(len(sep_fish_counts_in_habitat[dn_nr])):
            if np.sum(sep_fish_counts_in_habitat[dn_nr][fish_nr]) == 0:
                pass
            else:
                # embed()
                # quit()
                rel_sep_fish_counts_in_habitat[dn_nr][fish_nr] = sep_fish_counts_in_habitat[dn_nr][fish_nr] / np.sum(sep_fish_counts_in_habitat[dn_nr][fish_nr])

    for fish_nr in range(np.shape(rel_sep_fish_counts_in_habitat)[1]):
        fig, ax = plt.subplots(1, 2, facecolor='white', figsize= (20/2.54, 12/2.54))
        for enu, dn_nr in enumerate(np.arange(0, len(rel_sep_fish_counts_in_habitat), 2)):
            upshift = 0
            for hab_nr in range(len(rel_sep_fish_counts_in_habitat[dn_nr][fish_nr])):
                ax[0].bar(enu, rel_sep_fish_counts_in_habitat[dn_nr][fish_nr][hab_nr], bottom=upshift, color=hab_colors[hab_nr])
                upshift+= rel_sep_fish_counts_in_habitat[dn_nr][fish_nr][hab_nr]
        ax[0].set_xlabel('night nr.')
        ax[0].set_ylabel('rel. occupation')
        ax[0].set_title('fish Nr. %.0f' % fish_nr)
        # ax[0].set_ylim([0, 1])

        for enu, dn_nr in enumerate(np.arange(1, len(rel_sep_fish_counts_in_habitat), 2)):
            upshift = 0
            for hab_nr in range(len(rel_sep_fish_counts_in_habitat[dn_nr][fish_nr])):
                ax[1].bar(enu, rel_sep_fish_counts_in_habitat[dn_nr][fish_nr][hab_nr], bottom=upshift, color=hab_colors[hab_nr])
                upshift+= rel_sep_fish_counts_in_habitat[dn_nr][fish_nr][hab_nr]
        ax[1].set_xlabel('day nr.')
        ax[1].set_ylabel('rel. ocupation')
        plt.tight_layout()
        # embed()
        # quit()
        fig.savefig(saving_folder + 'hab_occupation_fish%.0f.pdf' % fish_nr)
        plt.close()
        # ax[0].set_ylim([0, 1])
        # ax[1].set_title('fish Nr. %.0f' % fish_nr)

    # plt.show()


    # embed()
    # quit()

    rel_fish_counts_in_habitat = np.zeros(np.shape(fish_counts_in_habitat))
    rel_d_fish_counts_in_habitat = np.zeros(np.shape(d_fish_counts_in_habitat))
    rel_n_fish_counts_in_habitat = np.zeros(np.shape(n_fish_counts_in_habitat))
    # embed()
    # quit()

    for fish_nr in range(len(fish_counts_in_habitat)):
        if np.sum(fish_counts_in_habitat[fish_nr]) == 0:
            pass
        else:
            rel_fish_counts_in_habitat[fish_nr] = fish_counts_in_habitat[fish_nr] / np.sum(fish_counts_in_habitat[fish_nr])

        if np.sum(d_fish_counts_in_habitat[fish_nr]) == 0:
            pass
        else:
            rel_d_fish_counts_in_habitat[fish_nr] = d_fish_counts_in_habitat[fish_nr] / np.sum(d_fish_counts_in_habitat[fish_nr])

        if np.sum(n_fish_counts_in_habitat[fish_nr]) == 0:
            pass
        else:
            rel_n_fish_counts_in_habitat[fish_nr] = n_fish_counts_in_habitat[fish_nr] / np.sum(n_fish_counts_in_habitat[fish_nr])

    bw = 5.
    hab_names = ['stacked stones', 'stone canyon', 'plants', 'sand', 'water surface']
    for hab_nr in range(len(df_in_habitat)):
        # h, bin_edges = np.histogram(df_in_habitat[hab_nr], bins = np.arange(0, 250 + bw, bw))
        # centers = bin_edges[:-1] + ((bin_edges[1] - bin_edges[0]) / bw)
        # h = h / np.sum(h) / bw

        fig, ax = plt.subplots(facecolor='white', figsize=(20/2.54, 12/2.54))
        # ax.fill_between(centers, h, color='grey', alpha= 0.2, label='all dfs')

        h, bin_edges = np.histogram(d_df_in_habitat[hab_nr], bins=np.arange(0, 250 + bw, bw))
        centers = bin_edges[:-1] + ((bin_edges[1] - bin_edges[0]) / bw)
        h = h / np.sum(h) / bw
        d_med = np.median(d_df_in_habitat[hab_nr])

        ax.plot(centers, h, color = 'orange', label= 'day dfs')

        h, bin_edges = np.histogram(n_df_in_habitat[hab_nr], bins=np.arange(0, 250 + bw, bw))
        centers = bin_edges[:-1] + ((bin_edges[1] - bin_edges[0]) / bw)
        h = h / np.sum(h) / bw
        n_med = np.median(n_df_in_habitat[hab_nr])

        ax.plot(centers, h, color= 'k', label= 'night dfs')

        ylims = ax.get_ylim()
        ax.plot([d_med, d_med], [ylims[0], ylims[1]], '--', color='orange', label='day median')
        ax.plot([n_med, n_med], [ylims[0], ylims[1]], '--', color='k', label='night median')

        ax.set_title(hab_names[hab_nr])
        ax.set_ylabel('rel. occurance')
        ax.set_xlabel('frequency difference [Hz]')

        plt.legend()
        plt.tight_layout()
        fig.savefig(saving_folder + 'df_in_%s.pdf' % hab_names[hab_nr])
        plt.close()


    fig, ax = plt.subplots(facecolor='white', figsize=(20/2.54, 12/2.54))
    for fish_nr in range(len(rel_fish_counts_in_habitat)):
        # embed()
        upshift = 0
        if np.sum(rel_fish_counts_in_habitat[fish_nr]) == 0:
            continue
        for hab_nr in range(len(rel_fish_counts_in_habitat[fish_nr])):
            ax.bar(fish_nr, rel_fish_counts_in_habitat[fish_nr][hab_nr], bottom = upshift, color=hab_colors[hab_nr])
            upshift += rel_fish_counts_in_habitat[fish_nr][hab_nr]
        ax.set_title('total occupation of habitats')
        ax.set_xlabel('fish Nr.')
        ax.set_ylabel('rel. occurance in habitat')
        # ax.set_ylim([0, 1])
    plt.tight_layout()
    fig.savefig(saving_folder + 'total_occupation_in_habitats_all.pdf')
    plt.close()

    fig, ax = plt.subplots(facecolor='white', figsize=(20/2.54, 12/2.54))
    for fish_nr in range(len(rel_d_fish_counts_in_habitat)):
        # embed()
        upshift = 0
        if np.sum(rel_d_fish_counts_in_habitat[fish_nr]) == 0:
            continue
        for hab_nr in range(len(rel_d_fish_counts_in_habitat[fish_nr])):
            ax.bar(fish_nr, rel_d_fish_counts_in_habitat[fish_nr][hab_nr], bottom=upshift, color=hab_colors[hab_nr])
            upshift += rel_d_fish_counts_in_habitat[fish_nr][hab_nr]
        ax.set_title('day occupation of habitats')
        ax.set_xlabel('fish Nr.')
        ax.set_ylabel('rel. occurance in habitat')
        # ax.set_ylim([0, 1])
    plt.tight_layout()
    fig.savefig(saving_folder + 'total_occupation_in_habitats_day.pdf')
    plt.close()

    fig, ax = plt.subplots(facecolor='white', figsize=(20/2.54, 12/2.54))
    for fish_nr in range(len(rel_n_fish_counts_in_habitat)):
        # embed()
        upshift = 0
        if np.sum(rel_n_fish_counts_in_habitat[fish_nr]) == 0:
            continue
        for hab_nr in range(len(rel_n_fish_counts_in_habitat[fish_nr])):
            ax.bar(fish_nr, rel_n_fish_counts_in_habitat[fish_nr][hab_nr], bottom=upshift, color=hab_colors[hab_nr])
            upshift += rel_n_fish_counts_in_habitat[fish_nr][hab_nr]
        ax.set_title('night occupation of habitats')
        ax.set_xlabel('fish Nr.')
        ax.set_ylabel('rel. occurance in habitat')
        # ax.set_ylim([0, 1])
    plt.tight_layout()
    fig.savefig(saving_folder + 'total_occupation_in_habitats_night.pdf')
    plt.close()
    # plt.show()

if __name__ == '__main__':
    main()