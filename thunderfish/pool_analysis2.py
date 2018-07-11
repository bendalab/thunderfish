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

def extract_freq_and_pos_array(datafile, shift, fish_nr_in_rec, datafile_nr):
    fund_v, ident_v, idx_v, times_v, sign_v = loaddata(datafile)
    times_v += shift[datafile_nr]

    i_range = np.arange(0, np.nanmax(idx_v) + 1)
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
    # embed()
    # quit()
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

    start_n = 110 * 60
    end_n = 110 * 60 + 12 * 60 * 60
    day_sec = 24 * 60 * 60

    habitats = [[12, 8], [14, 10], [11, 15], [9, 13], [0, 1, 2, 3, 4, 5, 6, 7]]
    hab_colors = ['k', 'grey', 'green', 'yellow', 'lightblue']

    for datafile_nr in range(len(datafile)):
        fish_freqs, fish_pos, times = extract_freq_and_pos_array(datafile[datafile_nr], shift, fish_nr_in_rec, datafile_nr)
        clock_sec = (times % day_sec)

        night_mask = np.arange(len(clock_sec))[(clock_sec >= start_n) & (clock_sec < end_n)]
        day_mask = np.arange(len(clock_sec))[(clock_sec < start_n) | (clock_sec >= end_n)]

        # fish_counts_on_electrode = [[] for f in fish_nr_in_rec]
        df_on_electrode = [[] for elec in range(16)]
        for fish_nr in range(len(fish_freqs)):
            for fish_nr_comp in np.arange(fish_nr+1, len(fish_freqs)):
                for elec in range(16):
                    fish_freq_oi = fish_freqs[fish_nr][ (fish_pos[fish_nr] == elec) & (fish_pos[fish_nr_comp] == elec)]
                    comp_fish_freq_oi = fish_freqs[fish_nr_comp][(fish_pos[fish_nr] == elec) & (fish_pos[fish_nr_comp] == elec)]

                    df_on_electrode[elec].extend(np.abs(fish_freq_oi - comp_fish_freq_oi))
                    # print(len(fish_freq_oi), len(comp_fish_freq_oi))

        df_in_habitat = [[] for habitat in habitats]
        for hab_nr in range(len(habitats)):
            for elec in habitats[hab_nr]:
                df_in_habitat[hab_nr].extend(df_on_electrode[elec])

        if datafile_nr == 0:
            all_df_in_habitat = df_in_habitat
        else:
            for hab_nr in range(len(df_in_habitat)):
                all_df_in_habitat[hab_nr].extend(df_in_habitat[hab_nr])

        #
        # for dfs in df_in_habitat:
        #     fig, ax = plt.subplots()
        #     ax.hist(dfs)
        # plt.show()
        # embed()
        # quit()



        # continue

        ################################## LAURA ###############################################
        fish_counts_on_electrode = np.zeros((len(fish_nr_in_rec), 16), dtype = int)
        n_fish_counts_on_electrode = np.zeros((len(fish_nr_in_rec), 16), dtype = int)
        d_fish_counts_on_electrode = np.zeros((len(fish_nr_in_rec), 16), dtype = int)

        fish_counts_in_habitat = np.zeros((len(fish_nr_in_rec), 5), dtype=int)
        n_fish_counts_in_habitat = np.zeros((len(fish_nr_in_rec), 5), dtype=int)
        d_fish_counts_in_habitat = np.zeros((len(fish_nr_in_rec), 5), dtype=int)

        for fish_nr in range(len(fish_nr_in_rec)):
            for e_nr in range(16):
                # embed()
                # quit()
                fish_counts_on_electrode[fish_nr][e_nr] += len(fish_pos[fish_nr][fish_pos[fish_nr] == e_nr])

                n_fish_counts_on_electrode[fish_nr][e_nr] += len(fish_pos[fish_nr][night_mask][fish_pos[fish_nr][night_mask] == e_nr])
                d_fish_counts_on_electrode[fish_nr][e_nr] += len(fish_pos[fish_nr][day_mask][fish_pos[fish_nr][day_mask] == e_nr])

        for fish_nr in range(len(fish_counts_on_electrode)):
            for habitat_nr in range(len(habitats)):
                count_in_habitat = 0
                count_in_habitat_n = 0
                count_in_habitat_d = 0
                for ele in habitats[habitat_nr]:
                    # embed()
                    # quit()
                    count_in_habitat += fish_counts_on_electrode[fish_nr][ele]
                    count_in_habitat_n += n_fish_counts_on_electrode[fish_nr][ele]
                    count_in_habitat_d += d_fish_counts_on_electrode[fish_nr][ele]

                fish_counts_in_habitat[fish_nr][habitat_nr] = count_in_habitat
                d_fish_counts_in_habitat[fish_nr][habitat_nr] = count_in_habitat_d
                n_fish_counts_in_habitat[fish_nr][habitat_nr] = count_in_habitat_n

        fish_counts_in_habitat = np.array(fish_counts_in_habitat)
        d_fish_counts_in_habitat = np.array(d_fish_counts_in_habitat)
        n_fish_counts_in_habitat = np.array(n_fish_counts_in_habitat)

        if datafile_nr == 0:
            all_fish_counts_in_habitat = np.array(fish_counts_in_habitat)
            all_d_fish_counts_in_habitat = np.array(d_fish_counts_in_habitat)
            all_n_fish_counts_in_habitat = np.array(n_fish_counts_in_habitat)
        else:
            all_fish_counts_in_habitat += fish_counts_in_habitat
            all_d_fish_counts_in_habitat += d_fish_counts_in_habitat
            all_n_fish_counts_in_habitat += n_fish_counts_in_habitat

        # Todo: for single recording relativ calcs
        # rel_fish_counts_in_habitat = np.zeros(np.shape(fish_counts_in_habitat))
        # rel_d_fish_counts_in_habitat = np.zeros(np.shape(d_fish_counts_in_habitat))
        # rel_n_fish_counts_in_habitat = np.zeros(np.shape(n_fish_counts_in_habitat))
        #
        # for fish_nr in range(len(fish_counts_in_habitat)):
        #     if np.sum(fish_counts_in_habitat[fish_nr]) == 0:
        #         pass
        #     else:
        #         rel_fish_counts_in_habitat[fish_nr] = fish_counts_in_habitat[fish_nr] / np.sum(fish_counts_in_habitat[fish_nr])
        #
        #     if np.sum(d_fish_counts_in_habitat[fish_nr]) == 0:
        #         pass
        #     else:
        #         rel_d_fish_counts_in_habitat[fish_nr] = d_fish_counts_in_habitat[fish_nr] / np.sum(d_fish_counts_in_habitat[fish_nr])
        #
        #     if np.sum(n_fish_counts_in_habitat[fish_nr]) == 0:
        #         pass
        #     else:
        #         rel_n_fish_counts_in_habitat[fish_nr] = n_fish_counts_in_habitat[fish_nr] / np.sum(n_fish_counts_in_habitat[fish_nr])

    for df in all_df_in_habitat:
        fig, ax = plt.subplots()
        ax.hist(df)
    # plt.show()


    all_rel_fish_counts_in_habitat = np.zeros(np.shape(all_fish_counts_in_habitat))
    all_rel_d_fish_counts_in_habitat = np.zeros(np.shape(all_d_fish_counts_in_habitat))
    all_rel_n_fish_counts_in_habitat = np.zeros(np.shape(all_n_fish_counts_in_habitat))
    # embed()
    # quit()

    for fish_nr in range(len(all_fish_counts_in_habitat)):
        if np.sum(all_fish_counts_in_habitat[fish_nr]) == 0:
            pass
        else:
            all_rel_fish_counts_in_habitat[fish_nr] = all_fish_counts_in_habitat[fish_nr] / np.sum(all_fish_counts_in_habitat[fish_nr])

        if np.sum(all_d_fish_counts_in_habitat[fish_nr]) == 0:
            pass
        else:
            all_rel_d_fish_counts_in_habitat[fish_nr] = all_d_fish_counts_in_habitat[fish_nr] / np.sum(all_d_fish_counts_in_habitat[fish_nr])

        if np.sum(all_n_fish_counts_in_habitat[fish_nr]) == 0:
            pass
        else:
            all_rel_n_fish_counts_in_habitat[fish_nr] = all_n_fish_counts_in_habitat[fish_nr] / np.sum(all_n_fish_counts_in_habitat[fish_nr])

    fig, ax = plt.subplots()
    for fish_nr in range(len(all_rel_fish_counts_in_habitat)):
        # embed()
        upshift = 0
        if np.sum(all_rel_fish_counts_in_habitat[fish_nr]) == 0:
            continue
        for hab_nr in range(len(all_rel_fish_counts_in_habitat[fish_nr])):

            ax.bar(fish_nr, all_rel_fish_counts_in_habitat[fish_nr][hab_nr], bottom = upshift, color=hab_colors[hab_nr])
            upshift += all_rel_fish_counts_in_habitat[fish_nr][hab_nr]
        ax.set_title('all')

    fig, ax = plt.subplots()
    for fish_nr in range(len(all_rel_d_fish_counts_in_habitat)):
        # embed()
        upshift = 0
        if np.sum(all_rel_d_fish_counts_in_habitat[fish_nr]) == 0:
            continue
        for hab_nr in range(len(all_rel_d_fish_counts_in_habitat[fish_nr])):
            ax.bar(fish_nr, all_rel_d_fish_counts_in_habitat[fish_nr][hab_nr], bottom=upshift, color=hab_colors[hab_nr])
            upshift += all_rel_d_fish_counts_in_habitat[fish_nr][hab_nr]
        ax.set_title('day')

    fig, ax = plt.subplots()
    for fish_nr in range(len(all_rel_n_fish_counts_in_habitat)):
        # embed()
        upshift = 0
        if np.sum(all_rel_n_fish_counts_in_habitat[fish_nr]) == 0:
            continue
        for hab_nr in range(len(all_rel_n_fish_counts_in_habitat[fish_nr])):
            ax.bar(fish_nr, all_rel_n_fish_counts_in_habitat[fish_nr][hab_nr], bottom=upshift, color=hab_colors[hab_nr])
            upshift += all_rel_n_fish_counts_in_habitat[fish_nr][hab_nr]
        ax.set_title('night')
    plt.show()
        # for each fish calculate df to all fish
    #     df_of_all_to_all = []
    #     ds_of_all_to_all = []
    #     for fish_nr in range(len( np.array(fish_nr_in_rec)[:, datafile_nr])):
    #         df_of_all_to_all.append(np.full(np.shape(fish_freqs), np.nan))
    #         ds_of_all_to_all.append(np.full(np.shape(fish_freqs), np.nan))
    #         for fish_nr_comp in range(len(np.array(fish_nr_in_rec)[:, datafile_nr])):
    #             if fish_nr == fish_nr_comp:
    #                 continue
    #             df_of_all_to_all[fish_nr][fish_nr_comp] = fish_freqs[fish_nr_comp] - fish_freqs[fish_nr]
    #             ds = np.sqrt((electrode_loc[np.array(fish_pos[fish_nr][(~np.isnan(fish_pos[fish_nr])) & (~np.isnan(fish_pos[fish_nr_comp]))], dtype=int)][:, 0] - electrode_loc[np.array(fish_pos[fish_nr_comp][(~np.isnan(fish_pos[fish_nr])) & (~np.isnan(fish_pos[fish_nr_comp]))], dtype=int)][:, 0]) ** 2 +
    #                          (electrode_loc[np.array(fish_pos[fish_nr][(~np.isnan(fish_pos[fish_nr])) & (~np.isnan(fish_pos[fish_nr_comp]))], dtype=int)][:, 1] - electrode_loc[np.array(fish_pos[fish_nr_comp][(~np.isnan(fish_pos[fish_nr])) & (~np.isnan(fish_pos[fish_nr_comp]))], dtype=int)][:, 1]) ** 2 +
    #                          (electrode_loc[np.array(fish_pos[fish_nr][(~np.isnan(fish_pos[fish_nr])) & (~np.isnan(fish_pos[fish_nr_comp]))], dtype=int)][:, 2] - electrode_loc[np.array(fish_pos[fish_nr_comp][(~np.isnan(fish_pos[fish_nr])) & (~np.isnan(fish_pos[fish_nr_comp]))], dtype=int)][:, 2]) ** 2)
    #
    #             ds_of_all_to_all[fish_nr][fish_nr_comp][~np.isnan(df_of_all_to_all[fish_nr][fish_nr_comp])] = ds
    #
    #     bins = np.arange(0, 20 + 0.25, 0.25)
    #     binned_data = []
    #     for fish_nr in range(len(df_of_all_to_all)):
    #         for fish_nr_comp in range(len(df_of_all_to_all[fish_nr])):
    #             if fish_nr_comp <= fish_nr:
    #                 continue
    #             df_v = df_of_all_to_all[fish_nr][fish_nr_comp]
    #             ds_v = ds_of_all_to_all[fish_nr][fish_nr_comp]
    #             for i in range(len(bins) - 1):
    #                 binned_data.append(ds_v[~np.isnan(ds_v)][(df_v[~np.isnan(df_v)] >= bins[i]) & (df_v[~np.isnan(df_v)] < bins[i + 1])])
    #
    #             bin_med = [np.median(x) if len(x) > 0 else np.nan for x in binned_data]
    #             bin_center = bins[:-1]+(bins[1] - bins[0] / 2)
    #
    #             all_bin_med.extend(bin_med)
    #             all_bin_center.extend(bin_center)
    #             binned_data = []
    #
    #
    # embed()
    # quit()
    #     # fig, ax = plt.subplots()
    #     # ax.plot()
    #     # plt.show()
    #     # embed()
    #     # quit()
    #     #
    #     # # ax.scatter(df_v[~np.isnan(df_v)], ds_v[~np.isnan(ds_v)], color='k', alpha=0.2)
    #     #
    #     # plt.show()
    #     # embed()
    #     # quit()
    #
    #     # # todo:use this again
    #     # th_df = 10.
    #     # th_idx = 30
    #     # for fish_nr in range(len(df_of_all_to_all)):
    #     #     for fish_nr_comp in range(len(df_of_all_to_all)):
    #     #         if fish_nr == fish_nr_comp:
    #     #             continue
    #     #
    #     #         beginn_approach = i_range[1:][(np.abs(df_of_all_to_all[fish_nr][fish_nr_comp])[:-1] >= th_df) & (np.abs(df_of_all_to_all[fish_nr][fish_nr_comp])[1:] < th_df) ]
    #     #         end_approach = i_range[:-1][(np.abs(df_of_all_to_all[fish_nr][fish_nr_comp])[:-1] < th_df) & (np.abs(df_of_all_to_all[fish_nr][fish_nr_comp])[1:] >= th_df) ]
    #     #
    #     #         if beginn_approach[0] > end_approach[0]:
    #     #             end_approach = end_approach[1:]
    #     #
    #     #         if len(beginn_approach) != len(end_approach):
    #     #             beginn_approach = beginn_approach[:-1]
    #     #
    #     #         # for i in reversed(range(len(beginn_approach))):
    #     #         #     if end_approach[i] - beginn_approach[i] < th_idx:
    #     #         #
    #     #         # embed()
    #     #         # quit()
    #     #
    #     # # a = [i_range[:-1][( np.abs(df_of_all_to_all[0][1][i]) >= 10) and (np.abs(df_of_all_to_all[0][1][i+1]) < 10)] for i in range(len(df_of_all_to_all[0][1]))]
    #     #
    #     # embed()
    #     # quit()
    #     #
    #     # ### plotting stuff
    #     # for fish_nr in range(len(fish_freqs)):
    #     #     fig, ax = plt.subplots()
    #     #     ax.set_title('Fish Nr. %.0f' % fish_nr)
    #     #     ax.set_ylabel('df [Hz]')
    #     #     ax.set_xlabel('time [s]')
    #     #     for enu, df in enumerate(df_of_all_to_all[fish_nr]):
    #     #         ax.plot(times_v[i_range], df, color=colors[enu])
    #     #
    #     #
    #     # fig, ax = plt.subplots()
    #     # for enu, fish in enumerate(fish_freqs):
    #     #     ax.plot(times_v[i_range], fish, color=colors[enu])
    #     # # ax.set_title('Fish Nr. %.0f' % fish_nr)
    #     # ax.set_ylabel('freq [Hz]')
    #     # ax.set_xlabel('time [s]')
    #     # plt.show()

if __name__ == '__main__':
    main()