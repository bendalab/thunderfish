"""
Track wave-type electric fish frequencies over time.

fish_tracker(): load data and track fish.
"""
import sys
import os
import argparse
import numpy as np
import scipy.stats as scp
import multiprocessing
from functools import partial
from .version import __version__
from .configfile import ConfigFile
from .dataloader import open_data
from .powerspectrum import spectrogram, next_power_of_two, decibel
from .harmonicgroups import add_psd_peak_detection_config, add_harmonic_groups_config
from .harmonicgroups import harmonic_groups_args, psd_peak_detection_args
from .harmonicgroups import harmonic_groups, fundamental_freqs, plot_psd_harmonic_groups

from IPython import embed
import time
try:
    import matplotlib.pyplot as plt
except ImportError:
    pass


def snipped_fundamentals(data, samplerate, start_idx = 0, nffts_per_psd=2, fresolution=0.5,
                         overlap_frac=.9, plot_harmonic_groups=False, increase_start_idx = False, verbose=0, **kwargs):
    """
    For a long data array calculates spectograms of small data snippets, computes PSDs, extracts harmonic groups and
    extracts fundamental frequncies.

    :param data: (array) raw data.
    :param samplerate: (int) samplerate of data.

    :param start_time: (int) ########### ?????????????? ################

    :param data_snippet_secs: (float) duration of data snipped processed at once in seconds. Necessary because of memory issues.
    :param nffts_per_psd: (int) number of nffts used for calculating one psd.
    :param fresolution: (float) frequency resolution for the spectrogram.
    :param overlap_frac: (float) overlap of the nffts (0 = no overlap; 1 = total overlap).
    :param verbose: (int) with increasing value provides more output on console.
    :param kwargs: further arguments are passed on to harmonic_groups().
    :return all_fundamentals: (list) containing arrays with the fundamentals frequencies of fishes detected at a certain time.
    :return all_times: (array) containing time stamps of frequency detection. (  len(all_times) == len(fishes[xy])  )
    """

    electrode_fundamentals = []
    electrode_fund_power = []

    # def nfft size
    # fresolution *= 2.
    nfft = next_power_of_two(samplerate / fresolution)


    # spectrogram
    spectrum, freqs, time = spectrogram(data, samplerate, fresolution=fresolution, overlap_frac=overlap_frac)

    # psd and fish fundamentals frequency detection
    power = [np.array([]) for i in range(len(time)-(nffts_per_psd-1))]

    for t in range(len(time)-(nffts_per_psd-1)):
        power[t] = np.mean(spectrum[:, t:t+nffts_per_psd], axis=1)

    for p in range(len(power)):
        fishlist, _, mains, all_freqs, good_freqs, _, _, _ = harmonic_groups(freqs, power[p], **kwargs)

        fundamentals, fund_power = fundamental_freqs(fishlist, return_power=True)

        electrode_fundamentals.append(fundamentals)
        electrode_fund_power.append(fund_power)

        if plot_harmonic_groups:
            fs = 14
            colors = ['#BA2D22', '#F47F17', '#53379B', '#3673A4', '#AAB71B', '#DC143C', '#1E90FF']

            inch_factor = 2.54
            fig, ax = plt.subplots(facecolor='white', figsize=(20. / inch_factor, 12. / inch_factor))
            plot_power = 10.0 * np.log10(power[p])
            ax.plot(freqs[freqs <= 3000.0], plot_power[freqs <= 3000.0], color=colors[-1])

            power_order = np.argsort([fish[0][1] for fish in fishlist])[::-1]

            for enu, fish in enumerate(power_order):
                if enu == len(colors) - 1:
                    break
                for harmonic in range(len(fishlist[fish])):
                    if fishlist[fish][harmonic][0] >= 3000.0:
                        break
                    if harmonic == 0:
                        ax.plot(fishlist[fish][harmonic][0], 10.0 * np.log10(fishlist[fish][harmonic][1]), 'o',
                                color=colors[enu], markersize=9, alpha=0.9, label='%.1f' % fishlist[fish][harmonic][0])
                    else:
                        ax.plot(fishlist[fish][harmonic][0], 10.0 * np.log10(fishlist[fish][harmonic][1]), 'o',
                                color=colors[enu], markersize=9, alpha=0.9)
            ax.legend(loc=1, ncol=2, fontsize=fs - 4, frameon=False, numpoints=1)

            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.get_xaxis().tick_bottom()
            ax.get_yaxis().tick_left()

            ax.tick_params(labelsize=fs - 2)
            ax.set_ylim([np.min(plot_power) - 5., np.max(plot_power) + 15.])
            ax.set_xlabel('Frequency [Hz]', fontsize=fs)
            ax.set_ylabel('Power [dB]', fontsize=fs)

            ax.set_xlim([800, 850])

            fig.tight_layout()
            print(nfft)
            print(nffts_per_psd)
            print(samplerate)
            plt.show()

    if nffts_per_psd == 1:
        electrode_times = time - ((nfft / samplerate) / 2) + (start_idx / samplerate)
    else:
        electrode_times = time[:-(nffts_per_psd - 1)] - ((nfft / samplerate) / 2) + (start_idx / samplerate)

    if increase_start_idx:
        non_overlapping_idx = (1-overlap_frac) * nfft
        start_idx += len(time[:-(nffts_per_psd-1)]) * non_overlapping_idx

    return electrode_fundamentals, electrode_fund_power, electrode_times, start_idx, nfft, spectrum[freqs < 1500], freqs[freqs < 1500]


def estimate_error(a_error, f_error, t_error, a_error_distribution, f_error_distribution,
                   min_f_weight=0.4, max_f_weight=0.9, t_of_max_f_weight=2., max_t_error=10.):
    if t_error >= 2.:
        f_weight = max_f_weight
    else:
        f_weight = 1. * (max_f_weight - min_f_weight) / t_of_max_f_weight * t_error + min_f_weight
    a_weight = 1. - f_weight

    a_e = a_weight * len(a_error_distribution[a_error_distribution <= a_error]) / len(a_error_distribution)
    f_e = f_weight * len(f_error_distribution[f_error_distribution <= f_error]) / len(f_error_distribution)
    t_e = 0.5 * (1. * t_error / max_t_error) ** (1. / 3)  # when weight is 0.1 I end up in an endless loop somewhere

    return a_e + f_e + t_e


def freq_tracking_v2(fundamentals, signatures, positions, times, freq_tolerance, n_channels):

    # def estimate_error(a_error, f_error, t_error, a_error_distribution, f_error_distribution,
    #                    min_f_weight = 0.4, max_f_weight = 0.9, t_of_max_f_weight = 2., max_t_error = 10.):
    #
    #     if t_error >= 2.:
    #         f_weight = max_f_weight
    #     else:
    #         f_weight = 1. * (max_f_weight - min_f_weight) / t_of_max_f_weight * t_error + min_f_weight
    #     a_weight = 1. - f_weight
    #
    #     a_e = a_weight * len(a_error_distribution[a_error_distribution <= a_error]) / len(a_error_distribution)
    #     f_e = f_weight * len(f_error_distribution[f_error_distribution <= f_error]) / len(f_error_distribution)
    #     t_e = 0.5 * (1. * t_error / max_t_error)**(1./3)  # when weight is 0.1 I end up in an endless loop somewhere
    #
    #     return a_e + f_e + t_e

    detection_time_diff = times[1] - times[0]
    dps = 1. / detection_time_diff  # detections per second (temp. resolution of frequency tracking)

    # vector creation
    fund_v = np.hstack(fundamentals)  # fundamental frequencies
    ident_v = np.full(len(fund_v), np.nan)  # fish identities of frequencies
    idx_v = []  # temportal indices
    sign_v = []  # power of fundamentals on all electrodes
    for enu, funds in enumerate(fundamentals):
        idx_v.extend(np.ones(len(funds)) * enu)
        sign_v.extend(signatures[enu])
    idx_v = np.array(idx_v, dtype=int)
    sign_v = np.array(sign_v)

    # sorting parameters
    idx_comp_range = int(np.floor(dps * 10.))  # maximum compare range backwards for amplitude signature comparison
    low_freq_th = 400.  # min. frequency tracked
    high_freq_th = 1050.  # max. frequency tracked

    # get f and amp signature distribution ############### BOOT #######################
    a_error_distribution = np.zeros(20000)  # distribution of amplitude errors
    f_error_distribution = np.zeros(20000)  # distribution of frequency errors
    idx_of_distribution = np.zeros(20000)  # corresponding indices

    b = 0  # loop varialble
    next_message = 0.  # feedback
    while b < 20000:
        next_message = include_progress_bar(b, 20000, 'get f and sign dist', next_message)  # feedback

        while True: # finding compare indices to create initial amp and freq distribution
            r_idx0 = np.random.randint(np.max(idx_v[~np.isnan(idx_v)]))
            r_idx1 = r_idx0 + 1
            if len(sign_v[idx_v == r_idx0]) != 0 and len(sign_v[idx_v == r_idx1]) != 0:
                break

        r_idx00 = np.random.randint(len(sign_v[idx_v == r_idx0]))
        r_idx11 = np.random.randint(len(sign_v[idx_v == r_idx1]))

        s0 = sign_v[idx_v == r_idx0][r_idx00]  # amplitude signatures
        s1 = sign_v[idx_v == r_idx1][r_idx11]

        f0 = fund_v[idx_v == r_idx0][r_idx00]  # fundamentals
        f1 = fund_v[idx_v == r_idx1][r_idx11]

        if np.abs(f0 - f1) > freq_tolerance:  # frequency threshold
            continue

        idx_of_distribution[b] = r_idx0
        a_error_distribution[b] = np.sqrt(np.sum([(s0[k] - s1[k]) ** 2 for k in range(len(s0))]))
        f_error_distribution[b] = np.abs(f0 - f1)
        b += 1

    ### FREQUENCY SORTING ALGOITHM ###

    # get initial distance cube (3D-matrix) ##################### ERROR CUBE #######################
    error_cube = []  # [fundamental_list_idx, freqs_to_assign, target_freqs]
    i0_m = []
    i1_m = []

    print('\n ')
    next_message = 0.
    for i in range(idx_comp_range):
        next_message = include_progress_bar(i, idx_comp_range, 'initial error cube', next_message)
        i0_v = np.arange(len(idx_v))[idx_v == i]  # indices of fundamtenals to assign
        i1_v = np.arange(len(idx_v))[(idx_v > i) & (idx_v <= (i + idx_comp_range))]  # indices of possible targets

        i0_m.append(i0_v)
        i1_m.append(i1_v)

        if len(i0_v) == 0 or len(i1_v) == 0:  # if nothing to assign or no targets continue
            error_cube.append([[]])
            continue

        error_matrix = np.full((len(i0_v), len(i1_v)), np.nan)

        for enu0 in range(len(fund_v[i0_v])):
            if fund_v[i0_v[enu0]] < low_freq_th or fund_v[i0_v[enu0]] > high_freq_th:  # freq to assigne out of tracking range
                continue
            for enu1 in range(len(fund_v[i1_v])):
                if fund_v[i1_v[enu1]] < low_freq_th or fund_v[i1_v[enu1]] > high_freq_th:  # target freq out of tracking range
                    continue
                if np.abs(fund_v[i0_v[enu0]] - fund_v[i1_v[enu1]]) >= freq_tolerance:  # freq difference to high
                    continue

                a_error = np.sqrt(np.sum([(sign_v[i0_v[enu0]][j] - sign_v[i1_v[enu1]][j]) ** 2 for j in range(n_channels)]))
                f_error = np.abs(fund_v[i0_v[enu0]] - fund_v[i1_v[enu1]])
                t_error = 1. * np.abs(idx_v[i0_v[enu0]] - idx_v[i1_v[enu1]]) / dps

                error_matrix[enu0, enu1] = estimate_error(a_error, f_error, t_error, a_error_distribution, f_error_distribution)
        error_cube.append(error_matrix)

    cube_app_idx = len(error_cube)

    next_identity = 0  # next unassigned fish identity no.
    print('\n ')
    next_message = 0.  # feedback
    for i in range(int(len(fundamentals))):
        next_message = include_progress_bar(i, len(fundamentals), 'tracking', next_message)  # feedback

        # prepare error cube --> build the masks
        mask_cube = np.array([np.ones(np.shape(error_cube[n]), dtype=bool) for n in range(len(error_cube))])
        for j in reversed(range(len(error_cube))):
            tmp_mask = mask_cube[j] # 0 == perviouse nans; 1 == possible connection

            # create mask_matrix: only contains one 1 for each row and else 0... --> mask_cube
            mask_matrix = np.zeros(np.shape(mask_cube[j]), dtype=bool)
            # t0 = time.time()

            old_idx0 = np.nan
            old_idx1 = np.nan

            counter = 0
            while True:
                help_error_v = np.hstack(error_cube[j])
                help_mask_v = np.hstack(tmp_mask)
                # print(len(help_error_v[help_mask_v]))

                if len(help_error_v[help_mask_v][~np.isnan(help_error_v[help_mask_v])]) == 0:
                    break

                idx0s = np.where(error_cube[j] == np.min(help_error_v[help_mask_v][~np.isnan(help_error_v[help_mask_v])]))[0]
                idx1s = np.where(error_cube[j] == np.min(help_error_v[help_mask_v][~np.isnan(help_error_v[help_mask_v])]))[1]

                # try:
                if len(idx0s) == 1:
                    idx0 = idx0s[0]
                    idx1 = idx1s[0]
                    counter = 0
                else:
                    idx0 = idx0s[counter]
                    idx1 = idx1s[counter]
                    if counter + 1 >= len(idx0s):
                        counter = 0
                    else:
                        counter += 1
                # except:
                #     print('still not solved')
                #     embed()
                #     quit()

                if old_idx0 == idx0 and old_idx1 == idx1:
                    print ('\n indices did not change ... why ?')
                    embed()
                    quit()
                else:
                    old_idx1 = idx1
                    old_idx0 = idx0

                ioi = i1_m[j][idx1]  # index of interest
                ioi_mask = [ioi in i1_m[k] for k in range(j+1, len(i1_m))]  # true if ioi is target of others

                if len(ioi_mask) > 0:
                    masks_idxs_feat_ioi = np.arange(j + 1, len(error_cube))[np.array(ioi_mask)]
                else:
                    masks_idxs_feat_ioi = np.array([])

                other_errors_to_idx1 = []
                for mask in masks_idxs_feat_ioi:
                    if len(np.hstack(mask_cube[mask])) == 0:  #?
                        continue  #?

                    check_col = np.where(i1_m[mask] == ioi)[0][0]
                    row_mask = np.hstack(mask_cube[mask][:, check_col])
                    possible_error = np.hstack(error_cube[mask][:, check_col])[row_mask]

                    if len(possible_error) == 0:
                        continue

                    elif len(possible_error) == 1:
                        other_errors_to_idx1.append(possible_error[0])
                    else:
                        print('something strange that should not be possible occurred! ')
                        embed()
                        quit()

                # conditions !!!
                if 1. * np.abs(fund_v[i0_m[j][idx0]] - fund_v[i1_m[j][idx1]]) / (( idx_v[i1_m[j][idx1]] - idx_v[i0_m[j][idx0]]) / dps) > 2.5:
                    tmp_mask[idx0, idx1] = 0
                    continue


                if j > 0:
                    if np.any(np.array(other_errors_to_idx1) < error_cube[j][idx0, idx1]):
                        tmp_mask[idx0, idx1] = 0
                        continue
                    else:
                        mask_matrix[idx0, idx1] = 1
                        tmp_mask[idx0] = np.zeros(np.shape(tmp_mask[idx0]), dtype=bool)
                        tmp_mask[:, idx1] = np.zeros(len(tmp_mask), dtype=bool)
                else:
                    if np.any(np.array(other_errors_to_idx1) < error_cube[j][idx0, idx1]):
                        tmp_mask[idx0, idx1] = 0
                        continue
                    else:
                        tmp_mask[idx0] = np.zeros(np.shape(tmp_mask[idx0]), dtype=bool)
                        tmp_mask[:, idx1] = np.zeros(len(tmp_mask), dtype=bool)

                        if np.isnan(ident_v[i0_m[j][idx0]]):  # i0 doesnt have identity
                            if np.isnan(ident_v[i1_m[j][idx1]]):  # i1 doesnt have identity
                                ident_v[i0_m[j][idx0]] = next_identity
                                ident_v[i1_m[j][idx1]] = next_identity
                                next_identity += 1
                            else:  # i1 does have identity
                                if idx_v[i0_m[j][idx0]] in idx_v[ident_v == ident_v[i1_m[j][idx1]]]:  # i0 idx in i1 indices --> no combining
                                    ident_v[i0_m[j][idx0]] = next_identity
                                    next_identity += 1
                                else:  # i0 idx not in i1 indices --> append
                                    ident_v[i0_m[j][idx0]] = ident_v[i1_m[j][idx1]]


                        else:  # i0 does have identity
                            if np.isnan(ident_v[i1_m[j][idx1]]):  # i1 doesnt have identity
                                ident_v[i1_m[j][idx1]] = ident_v[i0_m[j][idx0]]
                            else:
                                ident0_idxs = idx_v[ident_v == ident_v[i0_m[j][idx0]]]
                                ident1_idxs = idx_v[ident_v == ident_v[i1_m[j][idx1]]]
                                if np.any([x in ident1_idxs for x in ident0_idxs]):
                                    continue
                                else:
                                    ident_v[ident_v == ident_v[i0_m[j][idx0]]] = ident_v[i1_m[j][idx1]]

                    ### combining plz

                    ### update distribution ### optional

            mask_cube[j] = mask_matrix
        # update error cube in second loop ############################

        i0_m.pop(0)
        i1_m.pop(0)
        error_cube.pop(0)

        i0_v = np.arange(len(idx_v))[idx_v == cube_app_idx]  # indices of fundamtenals to assign
        i1_v = np.arange(len(idx_v))[(idx_v > cube_app_idx) & (idx_v <= (cube_app_idx + idx_comp_range))]  # indices of possible targets

        i0_m.append(i0_v)
        i1_m.append(i1_v)

        if len(i0_v) == 0 or len(i1_v) == 0:  # if nothing to assign or no targets continue
            error_cube.append([[]])

        else:
            error_matrix = np.full((len(i0_v), len(i1_v)), np.nan)

            for enu0 in range(len(fund_v[i0_v])):
                if fund_v[i0_v[enu0]] < low_freq_th or fund_v[i0_v[enu0]] > high_freq_th:  # freq to assigne out of tracking range
                    continue
                for enu1 in range(len(fund_v[i1_v])):
                    if fund_v[i1_v[enu1]] < low_freq_th or fund_v[i1_v[enu1]] > high_freq_th:  # target freq out of tracking range
                        continue
                    if np.abs(fund_v[i0_v[enu0]] - fund_v[i1_v[enu1]]) >= freq_tolerance:  # freq difference to high
                        continue

                    a_error = np.sqrt(np.sum([(sign_v[i0_v[enu0]][j] - sign_v[i1_v[enu1]][j]) ** 2 for j in range(n_channels)]))
                    f_error = np.abs(fund_v[i0_v[enu0]] - fund_v[i1_v[enu1]])
                    t_error = 1. * np.abs(idx_v[i0_v[enu0]] - idx_v[i1_v[enu1]]) / dps

                    error_matrix[enu0, enu1] = estimate_error(a_error, f_error, t_error, a_error_distribution,
                                                              f_error_distribution)
            error_cube.append(error_matrix)

        cube_app_idx += 1

    print('reached the end')

    return fund_v, ident_v, idx_v, sign_v, a_error_distribution, f_error_distribution


def freq_tracking(fundamentals, signatures, positions, times, freq_tolerance, n_channels):
    detection_time_diff = times[1] - times[0]
    dps = 1. / detection_time_diff  # detections per second

    # vector creation
    idx_v = []
    sign_v = []
    for enu, funds in enumerate(fundamentals):
        idx_v.extend(np.ones(len(funds)) * enu)
        sign_v.extend(signatures[enu])
    idx_v = np.array(idx_v, dtype=int)
    sign_v = np.array(sign_v)
    fund_v = np.hstack(fundamentals)
    ident_v = np.full(len(fund_v), np.nan)
    a_error_v = np.full(len(fund_v), np.nan)
    f_error_v = np.full(len(fund_v), np.nan)

    # sorting parameters
    idx_comp_range = int(np.floor(dps * 10.))  # maximum compare range backwards for amplitude signature comparison
    low_freq_th = 400.
    high_freq_th = 1050.

    # get f and amp signature distribution
    a_error_distribution = np.zeros(20000)
    f_error_distribution = np.zeros(20000)
    idx_of_distribution = np.zeros(20000)
    b = 0
    next_message = 0.
    while b < 20000:
        next_message = include_progress_bar(b, 20000, 'get f and sign dist', next_message)
        while True:
            r_idx0 = np.random.randint(np.max(idx_v[~np.isnan(idx_v)]))
            r_idx1 = r_idx0 + 1
            if len(sign_v[idx_v == r_idx0]) != 0 and len(sign_v[idx_v == r_idx1]) != 0:
                break

        r_idx00 = np.random.randint(len(sign_v[idx_v == r_idx0]))
        r_idx11 = np.random.randint(len(sign_v[idx_v == r_idx1]))

        s0 = sign_v[idx_v == r_idx0][r_idx00]
        s1 = sign_v[idx_v == r_idx1][r_idx11]

        f0 = fund_v[idx_v == r_idx0][r_idx00]
        f1 = fund_v[idx_v == r_idx1][r_idx11]
        if np.abs(f0 - f1) > freq_tolerance:
            continue
        idx_of_distribution[b] = r_idx0
        a_error_distribution[b] = np.sqrt(np.sum([(s0[k] - s1[k]) ** 2 for k in range(len(s0))]))
        f_error_distribution[b] = np.abs(f0 - f1)
        b += 1

    # sorting loop
    next_identity = 0.
    next_message = 0.00

    for i in range(int(len(fundamentals))):
        next_message = include_progress_bar(i, len(fundamentals), 'tracking', next_message)
        # to assign frequency and the possible targets
        i0_v = np.arange(len(idx_v))[(idx_v < i) & (idx_v >= (i - idx_comp_range))]
        i1_v = np.arange(len(idx_v))[idx_v == i]

        if len(i0_v) == 0 or len(i1_v) == 0:
            continue

        # calculate amplitude error of freq to assign and targets
        # amp_distance = np.full((len(i0_v), len(i1_v)), np.nan)
        rel_amp_distance = np.full((len(i0_v), len(i1_v)), np.nan)
        # freq_distance = np.full((len(i0_v), len(i1_v)), np.nan)
        rel_freq_distance = np.full((len(i0_v), len(i1_v)), np.nan)

        for enu0, signature0 in enumerate(sign_v[i0_v]):
            if fund_v[i0_v[enu0]] < low_freq_th or fund_v[i0_v[enu0]] > high_freq_th:
                continue

            for enu1, signature1 in enumerate(sign_v[i1_v]):
                if fund_v[i1_v[enu1]] < low_freq_th or fund_v[i1_v[enu1]] > high_freq_th:
                    continue
                if np.abs(fund_v[i0_v[enu0]] - fund_v[i1_v[enu1]]) >= freq_tolerance:
                    continue

                else:
                    amp_error = np.sqrt(np.sum([(sign_v[i0_v[enu0]][i] - sign_v[i1_v[enu1]][i])**2 for i in range(n_channels)]))
                    # amp_distance[enu0, enu1] = amp_error
                    rel_amp_distance[enu0, enu1] = 1. * len(a_error_distribution[a_error_distribution >= amp_error]) / len(a_error_distribution)

                    f_error = np.abs(fund_v[i0_v[enu0]] - fund_v[i1_v[enu1]])
                    # freq_distance[enu0, enu1] = f_error
                    rel_freq_distance[enu0, enu1] = 1. * len(f_error_distribution[f_error_distribution >= f_error]) / len(f_error_distribution)

        while True:
            if np.size(amp_distance[~np.isnan(amp_distance)]) == 0:
                break

            add_fish = np.where(amp_distance == np.min(amp_distance[~np.isnan(amp_distance)]))[0][0]
            add_freq = np.where(amp_distance == np.min(amp_distance[~np.isnan(amp_distance)]))[1][0]

            if np.isnan(a_error_v[i0_v[add_fish]]) or a_error_v[i0_v[add_fish]] > amp_distance[add_fish, add_freq]:
            # if np.isnan(a_error_v[i0_v[add_fish]]):
                if np.isnan(ident_v[i0_v[add_fish]]):
                    ident_v[i1_v[add_freq]] = next_identity
                    ident_v[i0_v[add_fish]] = next_identity
                    a_error_v[i0_v[add_fish]] = amp_distance[add_fish, add_freq]
                    f_error_v[i0_v[add_fish]] = np.abs(fund_v[i1_v[add_freq]] - fund_v[i0_v[add_fish]])

                    next_identity += 1
                else:
                    if idx_v[i1_v[add_freq]] in idx_v[ident_v == ident_v[i0_v[add_fish]]]:
                        tmp_comp_idx = np.arange(len(ident_v))[(ident_v == ident_v[i0_v[add_fish]]) & (idx_v < idx_v[i1_v[add_freq]])][-1]

                        if np.abs(fund_v[tmp_comp_idx] - fund_v[i1_v[add_freq]]) >= np.percentile(f_error_v[~np.isnan(f_error_v)], 95):
                            amp_distance[add_fish, add_freq] = np.nan
                            continue

                        ident_v[i1_v[add_freq]] = next_identity
                        ident_v[(ident_v == ident_v[i0_v[add_fish]]) & (idx_v <= idx_v[i0_v[add_fish]])] = next_identity

                        a_error_v[tmp_comp_idx] = np.sqrt(np.sum([(sign_v[tmp_comp_idx][i] - sign_v[i1_v[add_freq]][i]) ** 2 for i in range(n_channels)]))
                        f_error_v[tmp_comp_idx] = np.abs(fund_v[i1_v[add_freq]] - fund_v[tmp_comp_idx])

                        next_identity += 1
                    else:
                        ident_v[i1_v[add_freq]] = ident_v[i0_v[add_fish]]
                        a_error_v[i0_v[add_fish]] = amp_distance[add_fish, add_freq]
                        f_error_v[i0_v[add_fish]] = np.abs(fund_v[i1_v[add_freq]] - fund_v[i0_v[add_fish]])

                amp_distance[:, add_freq] = np.full(len(amp_distance), np.nan)

            else:
                amp_distance[add_fish, add_freq] = np.nan


    # calculate down fishnumber
    for i in reversed(np.arange(int(np.max(ident_v[~np.isnan(ident_v)]))) + 1):
        if not i in ident_v:
            ident_v[(ident_v > i) & (~np.isnan(ident_v))] -= 1

    fig, ax = plt.subplots(facecolor='white', figsize=(20. / 2.54, 12. / 2.54))
    for i in range(int(np.max(ident_v[~np.isnan(ident_v)]))):
        c = np.random.rand(3)
        # c = colors[i % (len(colors)-1)]
        p_time = times[idx_v[ident_v == i]]
        p_freq = fund_v[ident_v == i]
        ax.plot(p_time, p_freq, marker='.', color=c)

    ax.set_title('amplitude sorting')
    ax.set_ylabel('frequency [Hz]')
    ax.set_xlabel('time [s]')

    plt.show()
    embed()
    quit()


def amp_signature_tracking(fundamentals, signatures, positions, times, freq_tolerance, n_channels):

    # ToDo: if np.unique(fund_v[ident_v == 'identity']) <= 10 ... zu wenig aenderung in frequeny... KILL IT

    colors = ['#BA2D22', '#F47F17', '#53379B', '#3673A4', '#AAB71B', '#DC143C', '#1E90FF', 'magenta']

    detection_time_diff = times[1] - times[0]
    dps = 1. / detection_time_diff  # detections per minutes

    idx_v = []
    sign_v = []
    for enu, funds in enumerate(fundamentals):
        idx_v.extend(np.ones(len(funds)) * enu)
        sign_v.extend(signatures[enu])
    idx_v = np.array(idx_v, dtype=int)
    sign_v = np.array(sign_v)
    fund_v = np.hstack(fundamentals)
    ident_v = np.full(len(fund_v), np.nan)


    steps = np.arange(int(np.floor(dps * 10.))) + 1.

    #assigned_v = np.zeros(len(fund_v))
    assigned_v = np.ones(len(fund_v)) * (steps[-1] + 1)
    #assigned2_v = np.zeros(len(fund_v))
    assigned2_v = np.ones(len(fund_v)) * (steps[-1] + 1)

    #steps = np.arange(50) + 1

    # ToDo: step one works fine... rest if fucked up xD

    # boot amp_distance
    max_idx_v = np.max(idx_v[~np.isnan(idx_v)])
    print('')
    next_message = 0.00
    boot_amp_distance = np.zeros(20000)
    for enu, b in enumerate(range(20000)):
        next_message = include_progress_bar(enu, 20000, 'boot delta amp', next_message)

        while True:
            r_idx0 = np.random.randint(0, max_idx_v)
            r_idx1 = r_idx0 + 1
            if len(sign_v[idx_v == r_idx0]) != 0 and len(sign_v[idx_v == r_idx1]) != 0:
                break
        try:
            s0 = sign_v[idx_v == r_idx0][np.random.randint(len(sign_v[idx_v == r_idx0]))]
            s1 = sign_v[idx_v == r_idx1][np.random.randint(len(sign_v[idx_v == r_idx1]))]
        except:
            embed()
            quit()


        boot_amp_distance[enu] = np.sqrt( np.sum([(s0[k] - s1[k])**2 for k in range(len(s0))]))

    print('')

    fig2, ax2 = plt.subplots(figsize = (20./2.54, 12./2.54), facecolor='white')

    hist, bins = np.histogram(boot_amp_distance, 200)
    width = bins[1] - bins[0]
    center = (bins[:-1] + bins[1:]) / 2.
    hist = 1. * hist / np.sum(hist) / width
    ax2.bar(center, hist, align='center', width=width, color=colors[3])


    #m_amp_distance = np.mean(boot_amp_distance)
    #s_amp_distance = np.std(boot_amp_distance, ddof=1)
    p_amp_distance = np.array(np.percentile(boot_amp_distance, (5, 10, 25, 50, 75, 100)))

    if np.any(np.array(p_amp_distance) < 1.):
        p_amp_distance = np.concatenate((np.array([1.]), p_amp_distance[p_amp_distance >= 1.]))

    tmp_ylim = ax2.get_ylim()
    for enu, p in enumerate(p_amp_distance):
        ax2.plot([p, p], [tmp_ylim[0], tmp_ylim[1]], lw = 2, color = colors[0])
    ax2.set_ylim([tmp_ylim[0], tmp_ylim[1]])
    ax2.set_xlim([bins[0], bins[-1]])
    ax2.set_xlabel('amplitude [dB]')
    ax2.set_ylabel('normed n')

    #plt.show()

    # ToDo: if I do it without amp_th this works perfectly fine... else i have to modify the assigning function alittle.
    # ToDo: in between connections have to be possible ...

    low_freq_th = 400.
    high_freq_th = 900.


    next_identity = 0.
    print ('')
    for enum, amp_th in enumerate(p_amp_distance):
        for step in steps:
            next_message = 0.00
            message_str = '%.0f/%.0f; %.2f dB th (%.0f/%.0f)' % (step, steps[-1], amp_th, enum+1, len(p_amp_distance))
            for i in range(int(len(fundamentals)-step)):
                #if i == 187:
                #    embed()

                next_message = include_progress_bar(i, len(fundamentals)-1, message_str, next_message)

                #m0 = [all(tup) for tup in zip(idx_v == i, assigned_v == 0.)]
                #i0_v = np.arange(len(idx_v))[m0]
                i0_v = np.arange(len(idx_v))[idx_v == i]

                #m1 = [all(tup) for tup in zip(idx_v == i+step, assigned2_v == 0.)]
                #i1_v = np.arange(len(idx_v))[m1]
                i1_v = np.arange(len(idx_v))[idx_v == i + step]

                amp_distance = np.full((len(i0_v), len(i1_v)), np.nan)
                for enu0, signature0 in enumerate(sign_v[i0_v]):
                    if fund_v[i0_v[enu0]] < low_freq_th or fund_v[i0_v[enu0]] > high_freq_th:
                        continue

                    for enu1, signature1 in enumerate(sign_v[i1_v]):
                        if fund_v[i1_v[enu1]] < low_freq_th or fund_v[i1_v[enu1]] > high_freq_th:
                            continue

                        if np.abs(fund_v[i0_v[enu0]] - fund_v[i1_v[enu1]]) >= freq_tolerance:
                            continue
                        else:
                            amp_distance[enu0, enu1] = np.sqrt(np.sum(
                                [(sign_v[i0_v[enu0]][i] - sign_v[i1_v[enu1]][i])**2 for i in range(n_channels)]))

                while True:
                    if np.size(amp_distance[~np.isnan(amp_distance)]) == 0:
                        break

                    add_fish = np.where(amp_distance == np.min(amp_distance[~np.isnan(amp_distance)]))[0][0]
                    add_freq = np.where(amp_distance == np.min(amp_distance[~np.isnan(amp_distance)]))[1][0]

                    # amp_th filter
                    if amp_distance[add_fish, add_freq] >= amp_th:
                        amp_distance[add_fish, add_freq] = np.nan
                        continue

                    # dont connect frequencies that would result in a slope larger than 1 Hz per sec
                    if np.abs(fund_v[i0_v[add_fish]] - fund_v[i1_v[add_freq]]) / (np.abs(idx_v[i0_v[add_fish]] - idx_v[i1_v[add_freq]]) / dps) >= 1.5:
                        amp_distance[add_fish, add_freq] = np.nan
                        continue

                    # see if there already is a closer connection
                    #embed()
                    #quit()

                    if assigned_v[i0_v[add_fish]] <= step or assigned2_v[i1_v[add_freq]] <= step:
                        amp_distance[add_fish, add_freq] = np.nan
                        continue

                    #assigned_v[i0_v[add_fish]] = 1.
                    #assigned2_v[i1_v[add_freq]] = 1.

                    if np.isnan(ident_v[i0_v[add_fish]]):
                        if idx_v[i0_v[add_fish]] in idx_v[ident_v == ident_v[i1_v[add_freq]]]:
                            amp_distance[add_fish, add_freq] = np.nan
                            continue

                        ident_v[i0_v[add_fish]] = next_identity
                        if np.isnan(ident_v[i1_v[add_freq]]):
                            ident_v[i1_v[add_freq]] = next_identity
                        else:
                            ident_v[ident_v == ident_v[i1_v[add_freq]]] = ident_v[i0_v[add_fish]]
                        next_identity += 1.
                    else:
                        if np.isnan(ident_v[i1_v[add_freq]]):
                            if idx_v[i1_v[add_freq]] in idx_v[ident_v == ident_v[i0_v[add_fish]]]:
                                amp_distance[add_fish, add_freq] = np.nan
                                continue
                            ident_v[i1_v[add_freq]] = ident_v[i0_v[add_fish]]
                        else:

                            # here is the only option where both connection points got an identity.
                            idx_add_fish = idx_v[ident_v == ident_v[i0_v[add_fish]]]
                            idx_add_freq = idx_v[ident_v == ident_v[i1_v[add_freq]]]

                            if len([x for x in idx_add_fish if x in idx_add_freq]) > 0:
                                amp_distance[add_fish, add_freq] = np.nan
                                continue

                            ident_v[ident_v == ident_v[i1_v[add_freq]]] = ident_v[i0_v[add_fish]]

                    assigned_v[i0_v[add_fish]] = step
                    assigned2_v[i1_v[add_freq]] = step

                    amp_distance[:, add_freq] = np.full(len(amp_distance), np.nan)
                    amp_distance[add_fish] = np.full(len(amp_distance[add_fish]), np.nan)

        plot_amp_diff = True
        if plot_amp_diff:

            plot_idents = np.unique(ident_v)

            fig3 = plt.figure(facecolor='white', figsize=(20. / 2.54, 20. / 2.54))
            ax3_1 = fig3.add_subplot(339)
            ax3_u = fig3.add_subplot(338)
            ax3_2 = fig3.add_subplot(337)

            ax3_3 = fig3.add_subplot(336)
            ax3_4 = fig3.add_subplot(335)
            ax3_5 = fig3.add_subplot(334)

            ax3_6 = fig3.add_subplot(333)
            ax3_7 = fig3.add_subplot(332)
            ax3_8 = fig3.add_subplot(331)
            axs = [ax3_1, ax3_2, ax3_3, ax3_4, ax3_5, ax3_6, ax3_7, ax3_8]

            for ident in plot_idents:
            #for enu0, ident in enumerate([rdm_ident, rdm_ident2]):
                if len(ident_v[ident_v == ident]) <= 5:
                    continue

                c = np.random.rand(3)

                for enu, c_ax in enumerate(axs):
                    sign_idx = np.arange(len(sign_v))[ident_v == ident]
                    c_ax.plot(times[idx_v[ident_v == ident]], sign_v[sign_idx][:, enu], color =c, marker = '.')

            y_lims = []
            for c_ax in axs:
                y_lims.extend(c_ax.get_ylim())
            for c_ax in axs:
                c_ax.set_ylim([np.min(y_lims), np.max(y_lims)])
            ax3_u.set_xlim(ax3_1.get_xlim())

            for ax in [ax3_3, ax3_4, ax3_5, ax3_6, ax3_7, ax3_8]:
                ax.set_xticks([])

            for ax in [ax3_u, ax3_1, ax3_3, ax3_4, ax3_6, ax3_7]:
                ax.set_yticks([])

            for ax in [ax3_2, ax3_5, ax3_8]:
                ax.set_ylabel('power [dB]')

            for ax in [ax3_1, ax3_u, ax3_2]:
                ax.set_xlabel('time [s]')

            plt.suptitle('Amplitude threshold: %.2f dB (max: %.2f dB)' % (amp_th, p_amp_distance[-1]))

            # plt.show()

    # calculate down fishnumber
    for i in reversed(np.arange(int(np.max(ident_v[~np.isnan(ident_v)]))) + 1):
        if not i in ident_v:
            ident_v[ident_v > i] -= 1

    fig, ax = plt.subplots(facecolor='white', figsize = (20. / 2.54, 12. / 2.54))
    for i in range(int(np.max(ident_v[~np.isnan(ident_v)]))):

        c = np.random.rand(3)
        #c = colors[i % (len(colors)-1)]
        p_time = times[idx_v[ident_v == i]]
        p_freq = fund_v[ident_v == i]
        ax.plot(p_time, p_freq, marker= '.', color=c)

    ax.set_title('amplitude sorting')
    ax.set_ylabel('frequency [Hz]')
    ax.set_xlabel('time [s]')

    ##########################
    #fig2, ax2 = plt.subplots()

    #for i in range(int(np.max(ident_v[~np.isnan(ident_v)]) + 1)):
    #    c = np.random.rand(3)
    #    ioi = np.arange(len(ident_v))[ident_v == i] # index of interest

    #    delta_t = np.diff(idx_v[ioi]) / dps
    #    delta_a = np.zeros(len(delta_t))

    #    for j in range(len(ioi) - 1):

    #        delta_a[j] = np.sqrt(np.sum([(sign_v[ioi[j]][k] - sign_v[ioi[j+1]][k])**2 for k in range(len(sign_v[ioi[j]]))]))

    #    ax2.plot(delta_t, delta_a, '.', color = c)


    #ax2.plot([0, 1], [m_amp_distance - s_amp_distance, m_amp_distance - s_amp_distance], '-', color='red')

    plt.show()

    embed()
    quit()


def first_level_fish_sorting(all_fundamentals, all_signatures, base_name, all_times, n_channels=64, positions=[], prim_time_tolerance=1., freq_tolerance = .5,
                             save_original_fishes=False, output_folder = '.', verbose=0):
    """
    Sorts fundamental frequencies of wave-type electric fish detected at certain timestamps to fishes.

    There is an array of fundamental frequencies for every timestamp (all_fundamentals). Each of these frequencies is
    compared to the last frequency of already detected fishes (last_fish_fundamentals). If the frequency difference
    between the new frequency and one or multiple already detected fishes the frequency is appended to the array
    containing all frequencies of the fish (fishes) that has been absent for the shortest period of time. If the
    frequency doesn't fit to one fish, a new fish array is created. If a fish has not been detected at one time-step
    a NaN is added to this fish array.

    The result is for each fish a array containing frequencies or nans with the same length than the time array
    (all_times). These fish arrays can be saved as .npy file to access the code after the time demanding step.

    :param all_fundamentals: (list) containing arrays with the fundamentals frequencies of fishes detected at a certain time.
    :param base_name: (string) filename.
    :param all_times: (array) containing time stamps of frequency detection. (  len(all_times) == len(fishes[xy])  )
    :param prim_time_tolerance: (int) time in minutes from when a certain fish is no longer tracked.
    :param freq_tolerance: (float) maximum frequency difference to assign a frequency to a certain fish.
    :param save_original_fishes: (boolean) if True saves the sorted fishes after the first level of fish sorting.
    :param verbose: (int) with increasing value provides more shell output.
    :return fishes: (list) containing arrays of sorted fish frequencies. Each array represents one fish.
    """
    def clean_up(fishes, fishes_x_pos, fishes_y_pos, last_fish_fundamentals, last_fish_signature, end_nans, dpm):
        """
        Delete fish arrays with too little data points to reduce memory usage.

        :param fishes: (list) containing arrays of sorted fish frequencies. Each array represents one fish.
        :param last_fish_fundamentals: (list) contains for every fish in fishes the last detected fundamental frequency.
        :param end_nans: (list) for every fish contains the counts of nans since the last fundamental detection.
        :return: fishes: (list) cleaned up input list.
        :return: last_fish_fundamentals: (list) cleaned up input list.
        :return: end_nans: (list) cleaned up input list.
        """
        min_occure_time = all_times[-1] * 0.01 / 60.
        if min_occure_time > 1.:
            min_occure_time = 1.

        for fish in reversed(range(len(fishes))):
            if len(np.array(fishes[fish])[~np.isnan(fishes[fish])]) <= dpm * min_occure_time:
                fishes.pop(fish)
                if fishes_x_pos != []:
                    fishes_x_pos.pop(fish)
                    fishes_y_pos.pop(fish)
                last_fish_fundamentals.pop(fish)
                end_nans.pop(fish)

        return fishes, fishes_x_pos, fishes_y_pos, last_fish_fundamentals, last_fish_signature, end_nans

    detection_time_diff = all_times[1] - all_times[0]
    dpm = 60. / detection_time_diff  # detections per minutes

    try:
        fishes = [np.full(len(all_fundamentals)+1, np.nan)]
        if positions != []:
            fishes_x_pos = [[0.]]
            fishes_y_pos = [[0.]]
        else:
            fishes_x_pos = []
            fishes_y_pos = []
    except AttributeError:
        fishes = [np.zeros(len(all_fundamentals)+1) / 0.]
        if positions != []:
            fishes_x_pos = [[0.]]
            fishes_y_pos = [[0.]]
        else:
            fishes_x_pos = []
            fishes_y_pos = []

    fishes[0][0] = 0.
    last_fish_fundamentals = [ 0. ]
    last_fish_signature = [np.zeros(n_channels)]
    end_nans = [0]

    # for every list of fundamentals ...
    clean_up_idx = int(30 * dpm)

    alpha = 0.02
    print ('')
    next_message = 0.
    for enu, fundamentals in enumerate(all_fundamentals):
        next_message = include_progress_bar(enu, len(all_fundamentals), '1st lvl freq sorting', next_message=next_message)

        if enu == clean_up_idx:
            # ToDo: add positions to clean up function !!!
            if verbose >= 3:
                print('cleaning up ...')
            fishes, fishes_x_pos, fishes_y_pos, last_fish_fundamentals, last_fish_signature, end_nans = \
                clean_up(fishes, fishes_x_pos, fishes_y_pos, last_fish_fundamentals, last_fish_signature, end_nans, dpm)
            clean_up_idx += int(30 * dpm)

        # ToDo: increase freq_tollerance so 10 Hz rises still get detected...
        diffs = []
        signature_diffs = []
        for idx in range(len(fundamentals)):
            diffs.append(np.abs(np.asarray(last_fish_fundamentals) - fundamentals[idx]))  # calulate diffs to previously detected fish
            c_signature = all_signatures[enu][idx]
            signature_diffs.append(np.array([np.sqrt(np.sum(
                [(last_fish_signature[f][i] - c_signature[i])**2 for i in range(len(c_signature))])) for f in range(len(last_fish_signature))]))

            signature_diffs[idx][diffs[idx] > freq_tolerance] = np.nan  # exclude fish with to large diffs
            diffs[idx][diffs[idx] > freq_tolerance] = np.nan  # exclude fish with to large diffs

            diffs[idx] = np.array([diffs[idx][i] + end_nans[i] / (dpm / 60.) * alpha for i in range(len(end_nans))])  # adapt delta to last detection of fish x

        diffs = np.array(diffs)
        signature_diffs = np.array(signature_diffs)
        assigned_freq_idx = []

        for freq_i in range(len(signature_diffs)):
            for fish_i in np.arange(len(signature_diffs[freq_i]))[~np.isnan(signature_diffs[freq_i])]:
                if end_nans[fish_i] >= dpm / 60. * 15.:
                    signature_diffs[freq_i][fish_i] = np.nan


        while True:
            only_amp = True
            if not only_amp:
                if np.size(diffs[~np.isnan(diffs)]) == 0:
                    break

                clear_identity = False
                for freq_i in range(len(diffs)):
                    if len(diffs[freq_i][~np.isnan(diffs[freq_i])]) == 1:
                        if len(diffs[:][freq_i][~np.isnan(diffs[:][freq_i])]) == 1:
                            add_freq = freq_i
                            add_fish = np.arange(len(diffs[:][freq_i]))[~np.isnan(diffs[:][freq_i])][0]
                            clear_identity = True

                if not clear_identity:
                    #sum_signature_diffs = np.array([np.array([np.sum(x[~np.isnan(x)]) for x in signature_diffs])]).T
                    #tmp_signature_diffs = signature_diffs / sum_signature_diffs

                    #sum_diffs = np.array([np.array([np.sum(x[~np.isnan(x)]) for x in diffs])]).T
                    #tmp_diffs = diffs / sum_diffs

                    #embed()
                    #quit()
                    if not np.size(signature_diffs[~np.isnan(signature_diffs)]) == 0:
                        add_freq = np.where(signature_diffs == np.min(signature_diffs[~np.isnan(signature_diffs)]))[0][0]
                        add_fish = np.where(signature_diffs == np.min(signature_diffs[~np.isnan(signature_diffs)]))[1][0]
                    else:
                        add_freq = np.where(diffs == np.min(diffs[~np.isnan(diffs)]))[0][0]
                        add_fish = np.where(diffs == np.min(diffs[~np.isnan(diffs)]))[1][0]

                    if fundamentals[add_freq] - last_fish_fundamentals[add_fish] >= 10.:
                        signature_diffs[add_freq][add_fish] = np.nan
                        continue
            else:
                if np.size(signature_diffs[~np.isnan(signature_diffs)]) == 0:
                    break

                add_freq = np.where(signature_diffs == np.min(signature_diffs[~np.isnan(signature_diffs)]))[0][0]
                add_fish = np.where(signature_diffs == np.min(signature_diffs[~np.isnan(signature_diffs)]))[1][0]

            # max frequency change per second set to 1Hz.
            #if all_times[enu] > 636 and all_times[enu] < 637.5 and fundamentals[add_freq] > 647. and fundamentals[
            #    add_freq] < 648.:
            #    embed()
            #    quit()

            if np.abs(fundamentals[add_freq] - last_fish_fundamentals[add_fish]) / ((end_nans[add_fish]+1) / (dpm / 60.)) >= 1.:
                #if all_times[enu] > 636 and all_times[enu] < 637.5 and fundamentals[add_freq] >647. and fundamentals[add_freq] < 648.:
                #    embed()
                #    quit()
                #if np.abs(fundamentals[add_freq] - last_fish_fundamentals[add_fish]) > 0.:
                #    embed()
                #    quit()
                diffs[add_freq][add_fish] = np.nan
                signature_diffs[add_freq][add_fish] = np.nan
                continue


            assigned_freq_idx.append(add_freq)

            fishes[add_fish][enu+1] = fundamentals[add_freq]
            if positions != []:
                fishes_x_pos[add_fish].append(positions[enu][add_freq][0])
                fishes_y_pos[add_fish].append(positions[enu][add_freq][1])

            last_fish_fundamentals[add_fish] = fundamentals[add_freq]
            last_fish_signature[add_fish] = all_signatures[enu][add_freq]
            end_nans[add_fish] = 0

            try:
                diffs[add_freq] = np.full(len(diffs[add_freq]), np.nan)
                signature_diffs[add_freq] = np.full(len(signature_diffs[add_freq]), np.nan)
            except AttributeError:
                diffs[add_freq] = np.zeros(len(diffs[add_freq])) / 0.
                signature_diffs[add_freq] = np.zeros(len(signature_diffs[add_freq])) / 0.

            for j in range(len(diffs)):
                diffs[j][add_fish] = np.nan
                signature_diffs[j][add_fish] = np.nan

        # frequencies that could not be assigned to an existing fish
        for i in range(len(diffs)):
            if i in assigned_freq_idx:
                continue
            else:
                try:
                    fishes.append(np.full(len(all_fundamentals) + 1, np.nan))
                    if positions != []:
                        fishes_x_pos.append([0.])
                        fishes_y_pos.append([0.])

                except AttributeError:
                    fishes.append(np.zeros(len(all_fundamentals)+1) / 0.)
                    if positions != []:
                        fishes_x_pos.append([0.])
                        fishes_y_pos.append([0.])

                fishes[-1][enu + 1] = fundamentals[i]
                if positions != []:
                    fishes_x_pos[-1].append(positions[enu][i][0])
                    fishes_y_pos[-1].append(positions[enu][i][1])

                last_fish_fundamentals.append(fundamentals[i])
                last_fish_signature.append(all_signatures[enu][i])
                end_nans.append(0)

        # stop tracking of fishes not detected in a long time
        for fish in range(len(fishes)):
            if end_nans[fish] >= prim_time_tolerance * dpm:
                last_fish_fundamentals[fish] = 0.
                last_fish_signature[fish] = np.zeros(n_channels)

            if np.isnan(fishes[fish][enu+1]):
                end_nans[fish] += 1

    print ('')
    if verbose >= 3:
        print('cleaning up ...')
    fishes, fishes_x_pos, fishes_y_pos, last_fish_fundamentals, last_fish_signature, end_nans = \
        clean_up(fishes, fishes_x_pos, fishes_y_pos, last_fish_fundamentals, last_fish_signature, end_nans, dpm)

    # reshape everything to arrays
    for fish in range(len(fishes)):
        fishes[fish] = np.array(fishes[fish])[1:]
        if positions != []:
            fishes_x_pos[fish] = np.array(fishes_x_pos[fish])[1:]
            fishes_y_pos[fish] = np.array(fishes_y_pos[fish])[1:]

    # if not removed be clean_up(): remove first fish because it has been used for the first comparison !
    if fishes[0][0] == 0.:
        fishes.pop(0)
        if positions != []:
            fishes_x_pos.pop(0)
            fishes_y_pos.pop(0)

    if save_original_fishes:
        print('saving...')
        # ToDo: also save posiotions
        np.save(os.path.join(output_folder, base_name) + '-fishes.npy', np.asarray(fishes))
        np.save(os.path.join(output_folder, base_name) + '-times.npy', all_times)

    return np.asarray(fishes), np.array(fishes_x_pos), np.array(fishes_y_pos)


def detect_rises(fishes, all_times, rise_f_th = .5, verbose = 0):
    """
    Detects rises in frequency arrays that belong to a certain fish.

    Single rises are detected with the function 'detect_single_rises()' and get appended to a list.
    When the function 'detect_single_rises()' detects a rise it returns some data about the rise and continues seaching
    for rises at that index in the data where the detected rise ended. (While-loop)

    :param fishes: (array) containing arrays of sorted fish frequencies. Each array represents one fish.
    :param all_times: (array) containing time stamps of frequency detection. (  len(all_times) == len(fishes[xy])  )
    :param rise_f_th: (float) minimum frequency difference between peak and base of a rise to be detected as such.
    :return all_rises: (list) contains a list for each fish which each contains a list for every detected rise. In this
                       last list there are two arrays containing the frequency and the index of start and end of the rise.
                       all_rises[ fish ][ rise ][ [idx_start, idx_end], [freq_start, freq_end] ]
    """

    def detect_single_rise(fish, non_nan_idx, rise_f_th, dpm):
        """
        Detects a single rise in an array of fish frequencies.

        At first and an index of the array is detected from where on in the next 10 seconds every frequency is lower.
        This index is at first assumed as the peak of the rise.
        Afterwards and index is searched for from where on in the next 30 seconds every frequency is larger.
        This index is assumed as the end of the rise.
        The other possibility to get an end index of a rise is when the frequency doesnt drop any longer.

        If during the process of finding the end and the peak of the rise, the time difference between those indices
        rise above a time threshold (10 min) or the frequency rises above the assumed peak frequency of the rise, both
        indices are withdrawn and the seach continues.

        When both a peak and a end index are detected the frequency difference between those indices have to be larger
        than n * frequency threshold. n is defined by the time difference between peak and end of the rise.

        In the end index and frequency of rise peak and end are part if the return as well as the non_nan_indices of the
        fish array that are larger than the end index of the detected rise.

        :param fish: (array) sorted fish frequencies-
        :param non_nan_idx: (array) Indices where the fish array is not Nan.
        :param f_th: (float) minimum frequency difference between peak and base of a rise to be detected as such.
        :param dpm: (float) delta-t of the fish array.
        :return: index and frequency of start and end of one detected rise.
                 [[start_idx, end_idx], [start_freq, end_freq]]
        :return: Indices where the fish array is not Nan only containing those values larger than the end_idx of the
                 detected rise.
        """
        loop_idxs = np.arange(len(non_nan_idx[non_nan_idx <= non_nan_idx[-1] - dpm/ 60. * 10]))
        for i in loop_idxs:
            help_idx = np.arange(len(non_nan_idx))[non_nan_idx < non_nan_idx[i] + dpm / 60. * 10][-1]

            idxs = non_nan_idx[i+1:help_idx]
            if len(idxs) < dpm / 60. * 1.:
                continue

            if len(fish[idxs][fish[idxs] < fish[non_nan_idx[i]]]) == len(fish[idxs]):
                for j in loop_idxs[loop_idxs > i]:

                    if fish[non_nan_idx[j]] >= fish[non_nan_idx[i]]:
                        break

                    if non_nan_idx[j] - non_nan_idx[i] >= dpm * 3.:
                        break

                    help_idx2 = np.arange(len(non_nan_idx))[non_nan_idx < non_nan_idx[j] + dpm / 60. * 30][-1]
                    idxs2 = non_nan_idx[j+1:help_idx2]

                    last_possibe = False

                    if fish[non_nan_idx[j]] - np.median(fish[idxs2]) < 0.025:
                        last_possibe = True

                    if len(fish[idxs2][fish[idxs2] >= fish[non_nan_idx[j]]]) == len(fish[idxs2]) or non_nan_idx[j] == non_nan_idx[-1] or last_possibe:
                        freq_th = rise_f_th + ((non_nan_idx[j] - non_nan_idx[i]) *1.) // (dpm /60. *30) * rise_f_th
                        if fish[non_nan_idx[i]] - fish[non_nan_idx[j]] >= freq_th:

                            return [[non_nan_idx[i], non_nan_idx[j]], [fish[non_nan_idx[i]], fish[non_nan_idx[j]]]], non_nan_idx[j+1:]
                        else:
                            break
        return [[], []], [non_nan_idx[-1]]

    detection_time_diff = all_times[1] - all_times[0]
    dpm = 60. / detection_time_diff
    all_rises = []

    print('')
    next_message = 0.00
    for enu, fish in enumerate(fishes):
        next_message = include_progress_bar(enu, len(fishes), 'detect rises', next_message)

        non_nan_idx = np.arange(len(fish))[~np.isnan(fish)]
        fish_rises = []
        while non_nan_idx[-1] - non_nan_idx[0] > (dpm / 60. * 10) + 1:
            rise_data, non_nan_idx = detect_single_rise(fish, non_nan_idx, rise_f_th, dpm)
            fish_rises.append(rise_data)
        if not fish_rises == []:
            if fish_rises[-1][0] == []:
                fish_rises.pop(-1)
        all_rises.append(fish_rises)
    print ('')

    return all_rises


def combine_fishes(fishes, fishes_x_pos, fishes_y_pos, all_times, all_rises, max_time_tolerance = 5., f_th = 5., plot_combi=False):
    """
    Combines array of electric fish fundamental frequencies which, based on frequency difference and time of occurrence
    likely belong to the same fish.

    Every fish is compared to the fishes that appeared before this fish. If the time of occurrence of two fishes overlap
    or differ by less than a certain time tolerance (10 min.) for each of these fishes a compare index is determined.
    For the fish that second this compare index is either the index of the end of a rise (when the fish array begins
    with a rise) of the first index of frequency detection (when the fish array doesn't begin with a rise). For the fish
    that occurred first the compare index is the first index of detection before the compare index of the second fish.

    If the frequency of the two fishes at the compare indices differ by less than the frequency threshold and the counts
    of detections at the same time is below threshold  a 'distance value' is calculated
    (frequency difference + alpha * time difference oc occur index). These 'distance values' are saved in a matrix.
    After all this matrix contains the 'distance values' for every fish to all other fishes of Nans if the fishes are
    not combinable. Ever fish and its 'distant values' to all other fishes is represented by a row in the matrix.
    (possible_combination_all_fish[fish][compare_fish] = 'distance value' between fish and compare_fish).

    In the next step the fish arrays get combined. Therefore the minimum 'distance value' in the whole matrix is located.
    The index of this value match the fishes that fit together the best. The values of the second fish (fish) get
    transfered into the array of the first fish (comp_fish). Furthermore in the 'distance value' matrix the values that
    pointed to the second fish now point to the first fish. Since the second fish can't anymore point to another fish
    its row in the 'distance value' matrix gets replaced the an array full of nans.
    This process is repeated until this 'distance value' matrix only consists of Nans.
    When a fish is combined with another its rise data also gets transfered.

    In the end the list of fish frequency arrays gets cleaned up as well as the rise array. (Resulting from the sorting
    process the fishes array contains arrays only consisting of Nans. These get deleated.)

    :param fishes: (array) containing arrays of sorted fish frequencies. Each array represents one fish.
    :param all_times: (array) containing time stamps of frequency detection. (  len(all_times) == len(fishes[xy])  )
    :param all_rises: (list) contains a list for each fish which each contains a list for every detected rise. In this
                      last list there are two arrays containing the frequency and the index of start and end of the rise.
                      all_rises[ fish ][ rise ][ [idx_start, idx_end], [freq_start, freq_end] ]
    :param max_time_tolerance: (float) maximum time difference in min. between two fishes to allow combination.
    :param f_th: (float) maximum frequency difference between two fishes to allow combination
    :return fishes: (array) containing arrays of sorted fish frequencies. Each array represents one fish.
    :return all_rises: (list) contains a list for each fish which each contains a list for every detected rise. In this
                       last list there are two arrays containing the frequency and the index of start and end of the rise.
                       all_rises[ fish ][ rise ][ [idx_start, idx_end], [freq_start, freq_end] ]
    """
    detection_time_diff = all_times[1] - all_times[0]
    dpm = 60. / detection_time_diff  # detections per minutes

    occure_idx = []
    delete_idx = []
    try:
        possible_combinations_all_fish = np.array([np.full(len(fishes), np.nan) for i in range(len(fishes))])
    except AttributeError:
        possible_combinations_all_fish = np.array([np.zeros(len(fishes)) / 0. for i in range(len(fishes))])

    for fish in range(len(fishes)):
        non_nan_idx = np.arange(len(fishes[fish]))[~np.isnan(fishes[fish])]
        first_and_last_idx = np.array([non_nan_idx[0], non_nan_idx[-1]])
        occure_idx.append(first_and_last_idx)

    occure_order = np.argsort(np.array([occure_idx[i][0] for i in range(len(fishes))]))

    for fish in reversed(occure_order):
        try:
            possible_freq_combinations = np.full(len(fishes), np.nan)
            possible_idx_combinations = np.full(len(fishes), np.nan)
            possible_combinations = np.full(len(fishes), np.nan)
        except AttributeError:
            possible_freq_combinations = np.zeros(len(fishes)) / 0.
            possible_idx_combinations = np.zeros(len(fishes)) / 0.
            possible_combinations = np.zeros(len(fishes)) / 0.

        for comp_fish in reversed(occure_order[:np.where(occure_order == fish)[0][0]]):

            combinable = False
            if occure_idx[fish][0] > occure_idx[comp_fish][0] and occure_idx[fish][0] < occure_idx[comp_fish][1]:
                combinable = True
                comp_fish_nnans_idxs = np.arange(len(fishes[comp_fish]))[~np.isnan(fishes[comp_fish])]
                if all_rises[fish] != []:
                    if occure_idx[fish][0] in [all_rises[fish][i][0][0] for i in range(len(all_rises[fish]))]:
                        x = np.where( np.array([all_rises[fish][i][0][0] for i in range(len(all_rises[fish]))]) == occure_idx[fish][0])[0][0]
                        compare_idxs = [all_rises[fish][x][0][0], comp_fish_nnans_idxs[comp_fish_nnans_idxs < all_rises[fish][x][0][0]][-1]]
                        compare_freq_idxs = [all_rises[fish][x][0][1], comp_fish_nnans_idxs[comp_fish_nnans_idxs < all_rises[fish][x][0][0]][-1]]
                    else:
                        compare_idxs = [occure_idx[fish][0], comp_fish_nnans_idxs[comp_fish_nnans_idxs < occure_idx[fish][0]][-1]]
                        compare_freq_idxs = [occure_idx[fish][0], comp_fish_nnans_idxs[comp_fish_nnans_idxs < occure_idx[fish][0]][-1]]
                else:
                    compare_idxs = [occure_idx[fish][0], comp_fish_nnans_idxs[comp_fish_nnans_idxs < occure_idx[fish][0]][-1]]
                    compare_freq_idxs = [occure_idx[fish][0], comp_fish_nnans_idxs[comp_fish_nnans_idxs < occure_idx[fish][0]][-1]]

            elif occure_idx[fish][0] > occure_idx[comp_fish][1] and occure_idx[fish][0] - occure_idx[comp_fish][1] < max_time_tolerance * dpm:
                combinable = True
                comp_fish_nnans_idxs = np.arange(len(fishes[comp_fish]))[~np.isnan(fishes[comp_fish])]
                if all_rises[fish] != []:
                    if occure_idx[fish][0] in [all_rises[fish][i][0][0] for i in range(len(all_rises[fish]))]:
                        x = np.where( np.array([all_rises[fish][i][0][0] for i in range(len(all_rises[fish]))]) == occure_idx[fish][0])[0][0]
                        compare_idxs = [all_rises[fish][x][0][0], comp_fish_nnans_idxs[comp_fish_nnans_idxs < all_rises[fish][x][0][0]][-1]]
                        compare_freq_idxs = [all_rises[fish][x][0][1], comp_fish_nnans_idxs[comp_fish_nnans_idxs < all_rises[fish][x][0][0]][-1]]
                    else:
                        compare_idxs = [occure_idx[fish][0], comp_fish_nnans_idxs[comp_fish_nnans_idxs < occure_idx[fish][0]][-1]]
                        compare_freq_idxs = [occure_idx[fish][0], comp_fish_nnans_idxs[comp_fish_nnans_idxs < occure_idx[fish][0]][-1]]
                else:
                    compare_idxs = [occure_idx[fish][0], comp_fish_nnans_idxs[comp_fish_nnans_idxs < occure_idx[fish][0]][-1]]
                    compare_freq_idxs = [occure_idx[fish][0], comp_fish_nnans_idxs[comp_fish_nnans_idxs < occure_idx[fish][0]][-1]]
            else:
                if occure_idx[comp_fish][0] > occure_idx[fish][0]:
                    from IPython import embed
                    print ('first embed')
                    embed()

            if combinable:
                if np.abs(fishes[fish][compare_freq_idxs[0]] - fishes[comp_fish][compare_freq_idxs[1]]) > f_th:
                    continue
                alpha = 0.01 # alpha cant be larger ... to many mistakes !!!
                nan_test = fishes[fish] + fishes[comp_fish]
                if len(nan_test[~np.isnan(nan_test)]) <= 50:
                    med_slope = []
                    if plot_combi:
                        fig, ax = plt.subplots()
                        ax.plot(all_times[~np.isnan(fishes[fish])], fishes[fish][~np.isnan(fishes[fish])], marker= '.', color = 'green')
                        ax.plot(all_times[~np.isnan(fishes[comp_fish])], fishes[comp_fish][~np.isnan(fishes[comp_fish])], marker= '.', color = 'red')

                    for h_fish in range(len(fishes)):
                        if h_fish == fish:
                            continue
                        h_fish_data = fishes[h_fish][int(np.floor(compare_freq_idxs[0] - dpm * 3.)):int(compare_freq_idxs[0])]
                        h_fish_time = all_times[int(np.floor(compare_freq_idxs[0] - dpm * 3.)):int(compare_freq_idxs[0])]
                        y = h_fish_data[~np.isnan(h_fish_data)]
                        x = h_fish_time[~np.isnan(h_fish_data)]
                        if len(y) >= 2:
                            slope_h_fish, interc, _, _, _ = scp.linregress(x, y)
                            med_slope.append(slope_h_fish)
                            if plot_combi:
                                ax.scatter(all_times[np.floor(compare_freq_idxs[0] - dpm * 3.):compare_freq_idxs[0]],
                                           h_fish_data, color = 'grey', alpha= 0.1)
                                ax.plot(x, interc + x * slope_h_fish, '-', linewidth=1, color='blue')

                    if len(med_slope) > 0:

                        pred_freq = fishes[comp_fish][compare_freq_idxs[1]] +\
                                    np.abs(compare_freq_idxs[0] - compare_freq_idxs[1]) * np.median(med_slope)
                        if plot_combi:
                            ax.plot([all_times[compare_freq_idxs[1]], all_times[compare_freq_idxs[0]]],
                                    [fishes[comp_fish][compare_freq_idxs[1]], pred_freq], linewidth= 2, color='k')
                    else:
                        continue

                    if plot_combi:
                        plt.show()

                    if np.abs(fishes[fish][compare_freq_idxs[0]] - pred_freq) > f_th:
                        continue
                    else:
                        possible_freq_combinations[comp_fish] = np.abs(fishes[fish][compare_freq_idxs[0]] - pred_freq)
                    # possible_freq_combinations[comp_fish] = np.abs(fishes[fish][compare_freq_idxs[0]] - fishes[comp_fish][compare_freq_idxs[1]])
                        possible_idx_combinations[comp_fish] = np.abs([compare_idxs[0] - compare_idxs[1]])

                        possible_combinations[comp_fish] = possible_freq_combinations[comp_fish] + possible_idx_combinations[comp_fish] / (dpm / 60.) * alpha

        if len(possible_combinations[~np.isnan(possible_combinations)]) > 0:
            possible_combinations_all_fish[fish] = possible_combinations

    combining_finished = False

    while combining_finished == False:
        if np.size(possible_combinations_all_fish[~np.isnan(possible_combinations_all_fish)]) == 0:
            combining_finished = True
            continue

        fish = np.where(possible_combinations_all_fish == np.min(possible_combinations_all_fish[~np.isnan(possible_combinations_all_fish)]))[0][0]
        comp_fish = np.where(possible_combinations_all_fish == np.min(possible_combinations_all_fish[~np.isnan(possible_combinations_all_fish)]))[1][0]

        nan_test2 = fishes[fish] +  fishes[comp_fish]
        if len(nan_test2[~np.isnan(nan_test2)]) >= 50:
            possible_combinations_all_fish[fish][comp_fish] = np.nan
            if np.size(possible_combinations_all_fish[~np.isnan(possible_combinations_all_fish)]) == 0:
                combining_finished = True
            continue
        # print('plotting')
        # fig, ax = plt.subplots()
        # for fishy in range(len(fishes)):
        #     ax.scatter(all_times[~np.isnan(fishes[fishy])], fishes[fishy][~np.isnan(fishes[fishy])], color='grey', alpha= 0.1)
        # ax.plot(all_times[~np.isnan(fishes[fish])], fishes[fish][~np.isnan(fishes[fish])], color='red', marker='.')
        # ax.plot(all_times[~np.isnan(fishes[comp_fish])], fishes[comp_fish][~np.isnan(fishes[comp_fish])], color='green', marker='.')
        # plt.show()


        non_nan_idx_fish = np.arange(len(fishes[fish]))[~np.isnan(fishes[fish])]
        non_nan_idx_comp_fish = np.arange(len(fishes[comp_fish]))[~np.isnan(fishes[comp_fish])]

        try:
            help_x_pos = np.full(len(fishes[comp_fish]), np.nan)
            help_y_pos = np.full(len(fishes[comp_fish]), np.nan)
        except:
            help_x_pos = np.zeros(len(fishes[comp_fish])) / 0.
            help_y_pos = np.zeros(len(fishes[comp_fish])) / 0.

        # fishes_x_pos[comp_fish][~np.isnan(fishes_x_pos[fish])] = fishes_x_pos[fish][~np.isnan(fishes_x_pos[fish])]
        # fishes_y_pos[comp_fish][~np.isnan(fishes_y_pos[fish])] = fishes_y_pos[fish][~np.isnan(fishes_y_pos[fish])]

        help_x_pos[non_nan_idx_comp_fish] = fishes_x_pos[comp_fish]
        help_x_pos[non_nan_idx_fish] = fishes_x_pos[fish]
        fishes_x_pos[comp_fish] = help_x_pos[~np.isnan(help_x_pos)]

        help_y_pos[non_nan_idx_comp_fish] = fishes_y_pos[comp_fish]
        help_y_pos[non_nan_idx_fish] = fishes_y_pos[fish]
        fishes_y_pos[comp_fish] = help_y_pos[~np.isnan(help_y_pos)]

        fishes[comp_fish][~np.isnan(fishes[fish])] = fishes[fish][~np.isnan(fishes[fish])]

        try:
            fishes[fish] = np.full(len(fishes[fish]), np.nan)
            # fishes_x_pos[fish] = np.full(len(fishes_x_pos[fish]), np.nan)
            # fishes_y_pos[fish] = np.full(len(fishes_y_pos[fish]), np.nan)
            fishes_x_pos[fish] = np.array([])
            fishes_y_pos[fish] = np.array([])
        except AttributeError:
            fishes[fish] = np.zeros(len(fishes[fish])) / 0.
            # fishes_x_pos[fish] = np.zeros(len(fishes_x_pos[fish])) / 0.
            # fishes_y_pos[fish] = np.zeros(len(fishes_y_pos[fish])) / 0.
            fishes_x_pos[fish] = np.array([])
            fishes_y_pos[fish] = np.array([])

        # clean up possible_combination all fish
        for i in range(len(possible_combinations_all_fish)):  # loop over all fishes ...
            if not np.isnan(possible_combinations_all_fish[i][fish]):  # if this fish points on 'fish'...
                if np.isnan(possible_combinations_all_fish[i][comp_fish]): # if this fish doesnt points on 'compfish'
                    # the loop fish points now on comp fish and no longer on fish
                    possible_combinations_all_fish[i][comp_fish] = possible_combinations_all_fish[i][fish]
                    possible_combinations_all_fish[i][fish] = np.nan

                elif possible_combinations_all_fish[i][fish] < possible_combinations_all_fish[i][comp_fish]:  # if this fish points on compfish
                    # the loop fish still pionts on compfish
                    possible_combinations_all_fish[i][comp_fish] = possible_combinations_all_fish[i][fish]
                    possible_combinations_all_fish[i][fish] = np.nan
                else:
                    possible_combinations_all_fish[i][fish] = np.nan
        try:
            possible_combinations_all_fish[fish] = np.full(len(possible_combinations_all_fish[fish]), np.nan)
        except AttributeError:
            possible_combinations_all_fish[fish] = np.zeros(len(possible_combinations_all_fish[fish])) / 0.

        if all_rises[fish] != []:
            for rise in range(len(all_rises[fish])):
                all_rises[comp_fish].append(all_rises[fish][rise])
        all_rises[fish] = []

        if np.size(possible_combinations_all_fish[~np.isnan(possible_combinations_all_fish)]) == 0:
            combining_finished = True

    for fish in reversed(range(len(fishes))):
        if len(fishes[fish][~np.isnan(fishes[fish])]) == 0:
            delete_idx.append(fish)
            all_rises.pop(fish)

    return_idxs = np.setdiff1d(np.arange(len(fishes)), np.array(delete_idx))

    return fishes[return_idxs], fishes_x_pos[return_idxs], fishes_y_pos[return_idxs], all_rises


def exclude_fishes(fishes, fishes_x_pos, fishes_y_pos, all_times, min_occure_time = 1.):
    """
    Delete fishes that are present for a to short period of time.

    :param fishes: (list) containing arrays of sorted fish frequencies. Each array represents one fish.
    :param all_times: (array) containing time stamps of frequency detection. (  len(all_times) == len(fishes[xy])  )
    :param min_occure_time (int) minimum duration a fish has to be available to not get excluded.
    :return fishes: (array) containing arrays of sorted fish frequencies. Each array represents one fish.
    """
    keep_idx = []
    detection_time_diff = all_times[1] - all_times[0]
    dpm = 60. / detection_time_diff # detections per minute

    for fish in range(len(fishes)):
        if len(fishes[fish][~np.isnan(fishes[fish])]) >= min_occure_time * dpm:
            keep_idx.append(fish)

    return fishes[keep_idx], fishes_x_pos[keep_idx], fishes_y_pos[keep_idx]


def cut_at_rises(fishes, fishes_x_pos, fishes_y_pos, all_rises, all_times, min_occure_time):
    """
    Cuts fish arrays at detected rise peaks. For each rise two fish arrays are created with the same length as the
    original fish array.

    This step is necessary because of wrong detections resulting from rises of fishes.

    :param fishes: (array) containing arrays of sorted fish frequencies. Each array represents one fish.
    :param all_rises: (array) containing time stamps of frequency detection. (  len(all_times) == len(fishes[xy])  )
    :return: (array) containing arrays of sorted fish frequencies. Each array represents one fish.
    """
    detection_time_diff = all_times[1] - all_times[0]
    dpm = 60. / detection_time_diff  # detections per minutes

    new_fishes = []
    new_fishes_x_pos = []
    new_fishes_y_pos = []

    delete_idx = []
    for fish in reversed(range(len(fishes))):

        for rise in reversed(range(len(all_rises[fish]))):
            #############################
            # ToDo: cut only if there is another fish
            di = all_rises[fish][rise][0][1] - all_rises[fish][rise][0][0]
            tmp_idx = [int(all_rises[fish][rise][0][0] - dpm * 5), int(all_rises[fish][rise][0][0] + dpm * 5)]

            if tmp_idx[0] < 0:
                tmp_idx[0] = 0
            if tmp_idx[-1] >= len(all_times):
                tmp_idx[-1] = len(all_times) - 1

            check_idx = np.arange(tmp_idx[0], tmp_idx[1])

            df = np.abs(all_rises[fish][rise][1][0] - all_rises[fish][rise][1][1])
            check_f = [all_rises[fish][rise][1][0] + df / 2, all_rises[fish][rise][1][1] - df / 2]

            there_is_another_fish = False
            for check_fish in range(len(fishes)):
                if check_fish == fish:
                    continue
                if len(fishes[check_fish][check_idx][~np.isnan(fishes[check_fish][check_idx])]) > 0:
                    check_freqs = fishes[check_fish][check_idx][~np.isnan(fishes[check_fish][check_idx])]
                    if len(check_freqs[(check_freqs < check_f[0]) & (check_freqs > check_f[1])]) > 0:
                        there_is_another_fish = True

            if not there_is_another_fish:
                # print('not cutting')
                continue
            # else:
            #     print('cutting')


            ################################
            cut_idx = all_rises[fish][rise][0][0]
            print('cutting')

            try:
                new_fishes.append(np.full(len(fishes[fish]), np.nan))
                # new_fishes_x_pos.append(np.full(len(fishes[fish]), np.nan))
                # new_fishes_y_pos.append(np.full(len(fishes[fish]), np.nan))
                new_fishes_x_pos.append([])
                new_fishes_y_pos.append([])
            except AttributeError:
                new_fishes.append(np.zeros(len(fishes[fish])) / 0.)
                # new_fishes_x_pos.append(np.zeros(len(fishes[fish])) / 0.)
                # new_fishes_y_pos.append(np.zeros(len(fishes[fish])) / 0.)
                new_fishes_x_pos.append([])
                new_fishes_y_pos.append([])

            new_fishes[-1][cut_idx:] = fishes[fish][cut_idx:]
            non_nans_before = len(fishes[fish][:cut_idx][~np.isnan(fishes[fish][:cut_idx])])

            # new_fishes_x_pos[-1][cut_idx:] = fishes_x_pos[fish][cut_idx:]
            # new_fishes_y_pos[-1][cut_idx:] = fishes_y_pos[fish][cut_idx:]
            new_fishes_x_pos[-1] = fishes_x_pos[fish][non_nans_before:]
            new_fishes_y_pos[-1] = fishes_y_pos[fish][non_nans_before:]

            try:
                fishes[fish][cut_idx:] = np.full(len(fishes[fish][cut_idx:]), np.nan)
                # fishes_x_pos[fish][cut_idx:] = np.full(len(fishes[fish][cut_idx:]), np.nan)
                # fishes_y_pos[fish][cut_idx:] = np.full(len(fishes[fish][cut_idx:]), np.nan)
                fishes_x_pos[fish] = fishes_x_pos[fish][:non_nans_before]
                fishes_y_pos[fish] = fishes_y_pos[fish][:non_nans_before]
            except AttributeError:
                fishes[fish][cut_idx:] = np.zeros(len(fishes[fish][cut_idx:])) / 0.
                # fishes_x_pos[fish][cut_idx:] = np.zeros(len(fishes[fish][cut_idx:])) / 0.
                # fishes_y_pos[fish][cut_idx:] = np.zeros(len(fishes[fish][cut_idx:])) / 0.
                fishes_x_pos[fish] = fishes_x_pos[fish][:non_nans_before]
                fishes_y_pos[fish] = fishes_y_pos[fish][:non_nans_before]

            # ToDo rises correkt uebertragen...
            new_rises = all_rises[fish][rise:]
            old_rises = all_rises[fish][:rise]
            all_rises.append(new_rises)
            all_rises[fish] = old_rises

            # all_rises.append([all_rises[fish][rise]])
            # all_rises[fish].pop(rise)
    for fish in reversed(range(len(fishes))):
        if len(fishes[fish][~np.isnan(fishes[fish])]) < min_occure_time * dpm:
            delete_idx.append(fish)
            all_rises.pop(fish)
    return_idx = np.setdiff1d(np.arange(len(fishes)), np.array(delete_idx))

    # embed()
    # quit()
    if len(new_fishes) == 0:
        return fishes, fishes_x_pos, fishes_y_pos, all_rises
    else:
        # return np.append(fishes[return_idx], new_fishes, axis=0), np.append(fishes_x_pos[return_idx], new_fishes_x_pos, axis=0),\
        #        np.append(fishes_y_pos[return_idx], new_fishes_y_pos, axis=0), all_rises
        return np.append(fishes[return_idx], new_fishes, axis=0), np.array(list(fishes_x_pos[return_idx]) + list(new_fishes_x_pos) ),\
               np.array(list(fishes_y_pos[return_idx]) + list(new_fishes_y_pos)), all_rises
    # return np.append(fishes[return_idx], new_fishes, axis=0), all_rises


def save_data(fishes, fishes_x_pos, fishes_y_pos, all_times, all_rises, base_name, output_folder):
    np.save(os.path.join(output_folder, base_name) + '-final_fishes.npy', np.asarray(fishes))
    np.save(os.path.join(output_folder, base_name) + '-final_x_pos.npy', np.asarray(fishes_x_pos))
    np.save(os.path.join(output_folder, base_name) + '-final_y_pos.npy', np.asarray(fishes_y_pos))
    np.save(os.path.join(output_folder, base_name) + '-final_times.npy', all_times)
    np.save(os.path.join(output_folder, base_name) + '-final_rises.npy', np.asarray(all_rises))


def plot_fishes(fishes, all_times, all_rises, base_name, save_plot, output_folder):
    """
    Plot shows the detected fish fundamental frequencies plotted against the time in hours.

    :param fishes: (list) containing arrays of sorted fish frequencies. Each array represents one fish.
    :param all_times: (array) containing time stamps of frequency detection. (  len(all_times) == len(fishes[xy])  )
    """
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
        ax.plot(all_times[~np.isnan(fishes[fish])] / time_factor, fishes[fish][~np.isnan(fishes[fish])], color=color, marker='.')
        #
        # for rise in all_rises[fish]:
        #     ax.plot(all_times[rise[0][0]] / time_factor, rise[1][0], 'o', color=color, markersize=7)
        #     ax.plot(all_times[rise[0][1]] / time_factor, rise[1][1], 's', color=color, markersize=7)

    #legend_in = False
    #for fish in range(len(all_rises)):
    #    for rise in all_rises[fish]:
    #        # if rise[1][0] - rise[1][1] > 1.5:
    #        if legend_in == False:
    #            ax.plot(all_times[rise[0][0]] / time_factor, rise[1][0], 'o', color='red', markersize= 7,
    #                    markerfacecolor='None', label='rise begin')
    #            ax.plot(all_times[rise[0][1]] / time_factor, rise[1][1], 's', color='green', markersize= 7,
    #                    markerfacecolor='None', label='rise end')
    #            legend_in = True
    #            plt.legend(loc=1, numpoints=1, frameon=False, fontsize = 12)
    #        else:
    #            ax.plot(all_times[rise[0][0]] / time_factor, rise[1][0], 'o', color='red', markersize=7,
    #                    markerfacecolor='None')
    #            ax.plot(all_times[rise[0][1]] / time_factor, rise[1][1], 's', color='green', markersize=7,
    #                    markerfacecolor='None')

    maxy = np.max(np.array([np.mean(fishes[fish][~np.isnan(fishes[fish])]) for fish in range(len(fishes))]))
    miny = np.min(np.array([np.mean(fishes[fish][~np.isnan(fishes[fish])]) for fish in range(len(fishes))]))

    plt.ylim([miny-150, maxy+150])
    plt.ylabel('Frequency [Hz]', fontsize=14)
    if time_factor == 1.:
        plt.xlabel('Time [sec]', fontsize=14)
    elif time_factor == 60.:
        plt.xlabel('Time [min]', fontsize=14)
    else:
        plt.xlabel('Time [h]', fontsize=14)
    plt.title(base_name, fontsize=16)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()

    if save_plot:
        plt.savefig(os.path.join(output_folder, base_name))
        plt.close(fig)
    else:
        plt.show()


def plot_positions(fishes, fishes_x_pos, fishes_y_pos, all_times):
    non_nan_idx = []
    for fish in fishes:
        non_nan_idx.append(np.arange(len(fish))[~np.isnan(fish)])

    plot_x = np.full((len(fishes), 3), np.nan)
    plot_y = np.full((len(fishes), 3), np.nan)

    f_handles = []
    l_handles = []
    i0 = 0

    plt.ion()
    fig = plt.figure(facecolor='white', figsize=(25. / 2.54, 25. / 2.54))
    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
    ax.set_xlim([0, 9])
    ax.set_ylim([0, 9])
    fig.canvas.draw()
    # plt.show(block=False)
    end_min = all_times[-1] // 60
    end_sec = all_times[-1] % 60

    colors = [[255./255., 140./255., 0./255.], [100./255., 149./255., 237./255.]] # orange, cornflowerblue
    alphas = np.linspace(0.1, 1, 3)

    plt.waitforbuttonpress()

    for i in range(len(fishes)):
        # embed()
        # quit()
        c = np.zeros((3, 4))
        if np.median(fishes[i][~np.isnan(fishes[i])]) <= 730:
            line_c = colors[1]
            c[:, 0:3] = colors[1]
        else:
            line_c = colors[0]
            c[:, 0:3] = colors[0]
        c[:, 3] = alphas

        if i0 in non_nan_idx[i]:
            plot_x[i][-1] = fishes_x_pos[i][non_nan_idx[i] == i0] + 1.
            plot_y[i][-1] = fishes_y_pos[i][non_nan_idx[i] == i0] + 1.
        # embed()
        # quit()
        f = ax.scatter(plot_x[i], plot_y[i], color=c, s=80)
        l, = ax.plot(plot_x[i], plot_y[i], color=line_c, alpha=0.5)

        current_min = all_times[i0] // 60
        current_sec = all_times[i0] % 60
        ax.set_title('%.0fmin %.0fsec of %.0fmin %.0fsec' % (current_min, current_sec, end_min, end_sec))

        f_handles.append(f)
        l_handles.append(l)
    fig.canvas.draw()
    t0 = time.time()
    # embed()
    # quit()

    while True:
        i0 += 1

        plot_x = np.roll(plot_x, -1)
        plot_y = np.roll(plot_y, -1)
        plot_x[:, 2] = np.nan
        plot_y[:, 2] = np.nan

        for i in range(len(fishes)):
            if i0 in non_nan_idx[i]:
                plot_x[i][-1] = fishes_x_pos[i][non_nan_idx[i] == i0] + 1.
                plot_y[i][-1] = fishes_y_pos[i][non_nan_idx[i] == i0] + 1.

            f_handles[i].set_offsets(np.c_[plot_x[i], plot_y[i]])
            l_handles[i].set_data(plot_x[i], plot_y[i])

        current_min = all_times[i0] // 60
        current_sec = all_times[i0] % 60
        ax.set_title('%.0fmin %.0fsec of %.0fmin %.0fsec' % (current_min, current_sec, end_min, end_sec))

        if time.time() - t0 < 0.335:
            time.sleep(0.335 - (time.time() - t0))
        # print (plot_x[i])
        fig.canvas.draw()
        t0 = time.time()
        if i0 == len(all_times) - 1:
            break

    plt.ioff()
    plt.show()


def add_tracker_config(cfg, data_snippet_secs = 30., nffts_per_psd = 2, fresolution =.5, overlap_frac = .9,
                       freq_tolerance = 20., rise_f_th = 0.5, prim_time_tolerance = 1., max_time_tolerance = 10., f_th=5.):
    """ Add parameter needed for fish_tracker() as
    a new section to a configuration.

    Parameters
    ----------
    cfg: ConfigFile
        the configuration
    data_snipped_secs: float
         duration of data snipped processed at once in seconds.
    nffts_per_psd: int
        nffts used for powerspectrum analysis.
    fresolution: float
        frequency resoltution of the spectrogram.
    overlap_frac: float
        overlap fraction of nffts for powerspectrum analysis.
    freq_tolerance: float
        frequency tollerance for combining fishes.
    rise_f_th: float
        minimum frequency difference between peak and base of a rise to be detected as such.
    prim_time_tolerance: float
        maximum time differencs in minutes in the first fish sorting step.
    max_time_tolerance: float
        maximum time difference in minutes between two fishes to combine these.
    f_th: float
        maximum frequency difference between two fishes to combine these in last combining step.
    """
    cfg.add_section('Fish tracking:')
    cfg.add('DataSnippedSize', data_snippet_secs, 's', 'Duration of data snipped processed at once in seconds.')
    cfg.add('NfftPerPsd', nffts_per_psd, '', 'Number of nffts used for powerspectrum analysis.')
    cfg.add('FreqResolution', fresolution, 'Hz', 'Frequency resolution of the spectrogram')
    cfg.add('OverlapFrac', overlap_frac, '', 'Overlap fraction of the nffts during Powerspectrum analysis')
    cfg.add('FreqTolerance', freq_tolerance, 'Hz', 'Frequency tolernace in the first fish sorting step.')
    cfg.add('RiseFreqTh', rise_f_th, 'Hz', 'Frequency threshold for the primary rise detection.')
    cfg.add('PrimTimeTolerance', prim_time_tolerance, 'min', 'Time tolerance in the first fish sorting step.')
    cfg.add('MaxTimeTolerance', max_time_tolerance, 'min', 'Time tolerance between the occurrance of two fishes to join them.')
    cfg.add('FrequencyThreshold', f_th, 'Hz', 'Maximum Frequency difference between two fishes to join them.')


def tracker_args(cfg):
    """ Translates a configuration to the
    respective parameter names of the function fish_tracker().
    The return value can then be passed as key-word arguments to this function.

    Parameters
    ----------
    cfg: ConfigFile
        the configuration

    Returns (dict): dictionary with names of arguments of the clip_amplitudes() function and their values as supplied by cfg.
    -------
    dict
        dictionary with names of arguments of the fish_tracker() function and their values as supplied by cfg.
    """
    return cfg.map({'data_snippet_secs': 'DataSnippedSize',
                    'nffts_per_psd': 'NfftPerPsd',
                    'fresolution': 'FreqResolution',
                    'overlap_frac': 'OverlapFrac',
                    'freq_tolerance': 'FreqTolerance',
                    'rise_f_th': 'RiseFreqTh',
                    'prim_time_tolerance': 'PrimTimeTolerance',
                    'max_time_tolerance': 'MaxTimeTolerance',
                    'f_th': 'FrequencyThreshold'})


def grid_fish_frequency_and_position(t_fundamentals, t_power, channels, coords, neighbours, plot_field=False):

    # def contains_neighbors(pos1, pos2, grid, n_tolance_e = 2):
    #     if not grid or grid > 3:
    #         n_x_elec, n_y_elec = 8, 8
    #     elif grid == 1:
    #         n_x_elec, n_y_elec = 3, 3
    #     elif grid == 2:
    #         n_x_elec, n_y_elec = 4, 4
    #     elif grid == 3:
    #         print ('should not have gotten to this point.')
    #         quit()
    #
    #     for p in pos1:
    #         x = p // n_y_elec
    #         y = p % n_y_elec
    #
    #         neighbor_coords = []
    #         for i in np.arange(-n_tolance_e, n_tolance_e+1):
    #             for j in np.arange(-n_tolance_e, n_tolance_e+1):
    #                 neighbor_coords.append([x+i, y+j])
    #
    #         for k in reversed(range(len(neighbor_coords))):
    #             if all((i >= 0) & (i <= 7) for i in neighbor_coords[k]):
    #                 continue
    #             else:
    #                 neighbor_coords.pop(k)
    #         neighbor_e = [n[0] * 8 + n[1] for n in neighbor_coords]
    #
    #         if set(neighbor_e).intersection(pos2):
    #             return True
    #
    #     return False

    #  unify fundamentals

    unique_fundamentals = np.unique(np.concatenate((t_fundamentals)))

    #  check for each fundamental in which electrodes they are detected
    f_pos_pow_ci = []
    for f in unique_fundamentals:
        pos_i = np.arange(len(t_fundamentals))[np.array([f in t_fundamentals[i] for i in range(len(t_fundamentals))])]
        pos_e = list(np.array(channels)[pos_i])
        pos_e_power = [t_power[p][t_fundamentals[p] == f][0] for p in pos_i]
        f_pos_pow_ci.append([f, pos_e, pos_e_power, pos_i])

    # check for potential combinations (similar frequencies) of neighboring electrodes
    pot_f = np.array([f_pos_pow_ci[i][0] for i in range(len(f_pos_pow_ci))])
    pot_combinations = []

    for i in range(len(pot_f) - 1):
        for j in range(i + 1, len(pot_f)):
            #if np.abs(pot_f[i] - pot_f[j]) <= 0.2:
            if np.abs(pot_f[i] - pot_f[j]) <= 1.:

                if len([x for x in f_pos_pow_ci[i][1] if x in f_pos_pow_ci[j][1]]) == 0: #no equal electrodes // misdetections
                    all_possible_neigbours = []
                    for nli in f_pos_pow_ci[i][3]:
                        all_possible_neigbours.extend(neighbours[nli])
                    if len([x for x in all_possible_neigbours if x in f_pos_pow_ci[j][1]]) >= 1:
                        pot_combinations.append([np.abs(pot_f[i] - pot_f[j]), i, j])
                #     if len(f_pos_pow[i][1]) == 1:
                #         if set(neighbours[f_pos_pow[i][1]]).intersection(f_pos_pow[j][1]):
                #             pot_combinations.append([np.abs(pot_f[i] - pot_f[j]), i, j])
                #     else:
                #         if set(np.concatenate((neighbours[f_pos_pow[i][1]]))).intersection((f_pos_pow[j][1])):
                #             pot_combinations.append([np.abs(pot_f[i] - pot_f[j]), i, j])

                # embed()
                # quit()
                # position f1 = f_pos_pow[i][1]   ;    positions f2 = f_pos_pow[j][1]

                # if contains_neighbors(f_pos_pow[i][1], f_pos_pow[j][1], grid):
                #     pot_combinations.append([np.abs(pot_f[i] - pot_f[j]), i, j])
    # combine those fuckers
    while len(pot_combinations) > 0:
        smalles_diff_idx = np.argmin([pot_combinations[i][0] for i in range(len(pot_combinations))])

        idxs = [pot_combinations[smalles_diff_idx][1], pot_combinations[smalles_diff_idx][2]]
        idx0 = idxs[0] if len(f_pos_pow_ci[idxs[0]][1]) >= len(f_pos_pow_ci[idxs[1]][1]) else idxs[1]
        idx1 = idxs[1] if len(f_pos_pow_ci[idxs[0]][1]) >= len(f_pos_pow_ci[idxs[1]][1]) else idxs[0]

        if len(set(f_pos_pow_ci[idx0][1]).intersection(f_pos_pow_ci[idx1][1])) == 0:
            # print 'combining fishes'
            f_pos_pow_ci[idx0][1] += f_pos_pow_ci[idx1][1]
            f_pos_pow_ci[idx0][2] += f_pos_pow_ci[idx1][2]
            f_pos_pow_ci[idx0][3] = np.concatenate((f_pos_pow_ci[idx0][3], f_pos_pow_ci[idx1][3]))

            f_pos_pow_ci[idx1] = np.nan

        pot_combinations.pop(smalles_diff_idx)

        for i in range(len(pot_combinations)):
            pot_combinations[i][1] = idx0 if pot_combinations[i][1] == idx1 else pot_combinations[i][1]
            pot_combinations[i][2] = idx0 if pot_combinations[i][2] == idx1 else pot_combinations[i][2]

    for i in reversed(range(len(f_pos_pow_ci))):
        if not hasattr(f_pos_pow_ci[i], '__len__'):
            f_pos_pow_ci.pop(i)

    # calculate relative position in grid
    freq_x_y = []

    for enu, fish in enumerate(f_pos_pow_ci):

        fish[1] = list(np.array(fish[1])[np.array(fish[2]) != 0])
        fish[3] = list(np.array(fish[3])[np.array(fish[2]) != 0])
        fish[2] = list(np.array(fish[2])[np.array(fish[2]) != 0])

        if len(fish[2]) < 1:
            continue

        root_power = np.sqrt(fish[2])
        max_p = np.max(root_power)
        min_p = np.min(root_power)
        # embed()
        # quit()

        x = np.array([coords[i][0] for i in fish[3]])
        y = np.array([coords[i][1] for i in fish[3]])
        #
        # if not grid or grid > 3:
        #     x, y = np.array(fish[1]) // 8, np.array(fish[1]) % 8
        # elif grid == 1:
        #     x, y = np.array(fish[1]) // 3, np.array(fish[1]) % 3
        # elif grid == 2:
        #     x, y = np.array(fish[1]) // 4, np.array(fish[1]) % 4
        # elif grid == 3:
        #     print('never ever should have gotten here...')
        # else:
        #     print ('strange...')
        #
        # if part_analysis:
        #     x = np.array(fish[1]) // 3
        #     y = np.array(fish[1]) % 3
        # else:
        #     x = np.array(fish[1]) // 8
        #     y = np.array(fish[1]) % 8

        triang_idxs = np.argsort(root_power)[-4:]
        # embed()
        # quit()

        x_pos = np.sum([x[i] * root_power[i] for i in triang_idxs]) / np.sum(root_power[triang_idxs])
        y_pos = np.sum([y[i] * root_power[i] for i in triang_idxs]) / np.sum(root_power[triang_idxs])
        freq_x_y.append([fish[0], x_pos, y_pos])


        if plot_field:
            fig, ax = plt.subplots()
            cmap = plt.get_cmap('jet')
            colors = cmap((root_power - min_p) / (max_p - min_p))
            ax.plot(x_pos + 1, y_pos + 1, 'o', color='k', alpha=0.5, markersize=15)
            ax.scatter(x + 1, y + 1, c=colors, alpha=0.8, edgecolor='black', linewidth='1.5', s=100, cmap=cmap,
                       vmin=min_p, vmax=max_p)

            ax.set_ylabel('upstream electrodes')
            ax.set_xlabel('riverwidth electrodes')
            ax.set_title('%.2f' % fish[0])
            ax.set_ylim([0, 9])
            ax.set_xlim([0, 9])
            fig.show()
            time.sleep(1.5)
            plt.close(fig)

    return_fundamentals = [freq_x_y[i][0] for i in range(len(freq_x_y))]
    return_positions = [[freq_x_y[i][1], freq_x_y[i][2]] for i in range(len(freq_x_y))]

    return return_fundamentals, return_positions


def get_grid_proportions(data, grid=False, n_tolerance_e=2, verbose=0):
    if verbose >= 1:
        print('')
        if not grid:
            print ('standard grid (8 x 8) or all electrodes')
        elif grid == 1:
            print ('small grid (3 x 3)')
        elif grid == 2:
            print ('medium grid (4 x 4)')
        elif grid == 3:
            print ('U.S. grid')
        else:
            print ('standard (8 x 8) or all electrodes')

    # get channels
    if not grid or grid >= 4:
        channels = range(data.shape[1]) if len(data.shape) > 1 else range(1)
        positions = np.array([[i // 8, i % 8] for i in channels])
        neighbours = []
        for x, y in positions:
            neighbor_coords = []
            for i in np.arange(-n_tolerance_e, n_tolerance_e+1):
                for j in np.arange(-n_tolerance_e, n_tolerance_e+1):
                    if i == 0 and j == 0:
                        continue
                    else:
                        neighbor_coords.append([x+i, y+j])

            for k in reversed(range(len(neighbor_coords))):
                if all((i >= 0) & (i <= 7) for i in neighbor_coords[k]):
                    continue
                else:
                    neighbor_coords.pop(k)
            neighbours.append(np.array([n[0] * 8 + n[1] for n in neighbor_coords]))

    elif grid == 1:
        channels = [18, 19, 20, 26, 27, 28, 34, 35, 36]
        positions = np.array([[i // 8, i % 8] for i in channels])
        neighbours = []

        for x, y in positions:
            neighbor_coords = []
            for i in np.arange(-n_tolerance_e, n_tolerance_e+1):
                for j in np.arange(-n_tolerance_e, n_tolerance_e+1):
                    if i == 0 and j == 0:
                        continue
                    else:
                        neighbor_coords.append([x+i, y+j])

            for k in reversed(range(len(neighbor_coords))):
                if all((i >= 2) & (i <= 4) for i in neighbor_coords[k]):
                    continue
                else:
                    neighbor_coords.pop(k)
            neighbours.append(np.array([n[0] * 8 + n[1] for n in neighbor_coords]))

    elif grid == 2:
        channels = [18, 19, 20, 21, 26, 27, 28, 29, 34, 35, 36, 37, 42, 43, 44, 45]
        positions = np.array([[i // 8, i % 8] for i in channels])
        neighbours = []

        for x, y in positions:
            neighbor_coords = []
            for i in np.arange(-n_tolerance_e, n_tolerance_e+1):
                for j in np.arange(-n_tolerance_e, n_tolerance_e+1):
                    if i == 0 and j == 0:
                        continue
                    else:
                        neighbor_coords.append([x+i, y+j])

            for k in reversed(range(len(neighbor_coords))):
                if all((i >= 2) & (i <= 5) for i in neighbor_coords[k]):
                    continue
                else:
                    neighbor_coords.pop(k)
            neighbours.append(np.array([n[0] * 8 + n[1] for n in neighbor_coords]))

    elif grid == 3:
        channels = range(data.shape[1])
        positions = np.array([[4, 2], [2, 2], [0, 2], [3, 1], [1, 1], [4, 0], [2, 0], [0, 0]])
        neighbours = []

        for i in range(len(positions)):
            tmp_neighbours = np.arange(len(positions))
            neighbours.append(tmp_neighbours[tmp_neighbours != i])
    else:
        'stange error...'
        quit()

    return channels, positions, np.array(neighbours)


def load_matfile(data_file):
    try:
        import h5py
        mat = h5py.File(data_file)
        data = np.array(mat['elec']['data']).transpose()
        samplerate = mat['elec']['meta']['Fs'][0][0]
    except:
        from scipy.io import loadmat
        mat = loadmat(data_file, variable_names=['elec'])
        data = np.array(mat['elec']['data'][0][0])
        samplerate = mat['elec']['meta'][0][0][0][0][1][0][0]

    return data, samplerate


def include_progress_bar(loop_v, loop_end, taskname ='', next_message=0.00):
    if len(taskname) > 30 or taskname == '':
        taskname = '        random task         ' # 30 characters
    else:
        taskname = ' ' * (30 - len(taskname)) + taskname

    if (1.*loop_v / loop_end) >= next_message:
        bar_factor = (1. * loop_v / loop_end) // 0.05
        bar = '[' + int(bar_factor)*'=' + (20- int(bar_factor)) * ' ' + ']'
        #bar = '[' + int(next_message * 20)*'=' + (20- int(next_message * 20)) * ' ' + ']'
        sys.stdout.write('\r' + bar + taskname)
        sys.stdout.flush()

        next_message = ((1. * loop_v / loop_end) // 0.05) * 0.05 + 0.05

        if next_message >= 1.:
            bar = '[' + 20 * '=' + (20 - int(next_message * 20)) * ' ' + ']'
            sys.stdout.write('\r' + bar + taskname)
            sys.stdout.flush()

    return next_message


def amplitude_distance(fundamentals, signatures, freq_tolerance):
    df = []
    da = []
    for t in range(len(fundamentals) -1 ):
        for f0 in range(len(fundamentals[t])):
            for f1 in range(len(fundamentals[t+1])):
                df.append(np.abs(fundamentals[t][f0] - fundamentals[t+1][f1]))
                da.append(np.sqrt(np.sum(
                    [(signatures[t][f0][e] - signatures[t+1][f1][e])**2 for e in range(len(signatures[t][f0]))]   )))

    df = np.array(df)
    da = np.array(da)
    f_bin_start = np.zeros(int(freq_tolerance / 0.05))
    amp_error_th = np.zeros(int(freq_tolerance / 0.05))

    for i in range(len(f_bin_start)):
        f_bin_start[i] = i * 0.05
        amp_error_th[i] = np.percentile(da[(df >= i*0.05) & (df < (i+1) * 0.05)], 33) if len(df[(df >= i*0.05) & (df < (i+1) * 0.05)]) > 0 else 0

    return f_bin_start, amp_error_th


def get_spectrum_funds_amp_signature(data, samplerate, channels, data_snippet_idxs, start_time, end_time, fresolution = 0.5,
                                     overlap_frac=.9, nffts_per_psd= 2, comp_min_freq= 0., comp_max_freq = 2000., plot_harmonic_groups=False,
                                     create_plotable_spectrogram=False, extract_funds_and_signature=True, **kwargs):
    fundamentals = []
    positions = []
    times = np.array([])
    signatures = []

    start_idx = int(start_time * samplerate)
    if end_time < 0.0:
        end_time = len(data) / samplerate
        end_idx = int(len(data) - 1)
    else:
        end_idx = int(end_time * samplerate)
        if end_idx >= int(len(data) - 1):
            end_idx = int(len(data) - 1)

    # increase_start_idx = False
    last_run = False

    print ('')
    init_idx = False
    if not init_idx:
        init_idx = start_idx
    next_message = 0.00

    # create spectra plot ####
    get_spec_plot_matrix = False
    # fig_xspan = 20.
    # fig_yspan = 12.
    # fig_dpi = 80.
    # no_x = fig_xspan * fig_dpi
    # no_y = fig_yspan * fig_dpi
    #
    # min_x = start_time
    # max_x = end_time
    #
    # min_y = 0.
    # max_y = 2000.
    #
    # x_borders = np.linspace(min_x, max_x, no_x * 2)
    # y_borders = np.linspace(min_y, max_y, no_y * 2)
    # # checked_xy_borders = False
    #
    # tmp_spectra = np.zeros((len(y_borders) - 1, len(x_borders) - 1))

    while start_idx <= end_idx:
        if create_plotable_spectrogram:
            next_message = include_progress_bar(start_idx - init_idx, end_idx - init_idx, 'get plotable spec',
                                                next_message)
        else:
            next_message = include_progress_bar(start_idx - init_idx, end_idx - init_idx, 'extract fundamentals',
                                                next_message)

        if start_idx >= end_idx - data_snippet_idxs:
            last_run = True

        # calulate spectogram ....
        core_count = multiprocessing.cpu_count()

        if plot_harmonic_groups:
            pool = multiprocessing.Pool(1)
        else:
            pool = multiprocessing.Pool(core_count // 2)
            # pool = multiprocessing.Pool(core_count - 1)

        nfft = next_power_of_two(samplerate / fresolution)

        func = partial(spectrogram, samplerate=samplerate, fresolution=fresolution, overlap_frac=overlap_frac)
        a = pool.map(func, [data[start_idx: start_idx + data_snippet_idxs, channel] for channel in
                            channels])  # ret: spec, freq, time

        spectra = [a[channel][0] for channel in range(len(a))]
        spec_freqs = a[0][1]
        spec_times = a[0][2]
        pool.terminate()

        comb_spectra = np.sum(spectra, axis=0)

        if nffts_per_psd == 1:
            tmp_times = spec_times - ((nfft / samplerate) / 2) + (start_idx / samplerate)
        else:
            tmp_times = spec_times[:-(nffts_per_psd - 1)] - ((nfft / samplerate) / 2) + (start_idx / samplerate)

        # etxtract reduced spectrum for plot
        plot_freqs = spec_freqs[spec_freqs < comp_max_freq]
        plot_spectra = np.sum(spectra, axis=0)[spec_freqs < comp_max_freq]

        if create_plotable_spectrogram:
            # if not checked_xy_borders:
            if not get_spec_plot_matrix:
                fig_xspan = 20.
                fig_yspan = 12.
                fig_dpi = 80.
                no_x = fig_xspan * fig_dpi
                no_y = fig_yspan * fig_dpi

                min_x = start_time
                max_x = end_time

                min_y = comp_min_freq
                max_y = comp_max_freq

                x_borders = np.linspace(min_x, max_x, no_x * 2)
                y_borders = np.linspace(min_y, max_y, no_y * 2)
                # checked_xy_borders = False

                tmp_spectra = np.zeros((len(y_borders) - 1, len(x_borders) - 1))

                recreate_matrix = False
                if (tmp_times[1] - tmp_times[0]) > (x_borders[1] - x_borders[0]):
                    x_borders = np.linspace(min_x, max_x, (max_x - min_x) // (tmp_times[1] - tmp_times[0]) + 1)
                    recreate_matrix = True
                if (spec_freqs[1] - spec_freqs[0]) > (y_borders[1] - y_borders[0]):
                    recreate_matrix = True
                    y_borders = np.linspace(min_y, max_y, (max_y - min_y) // (spec_freqs[1] - spec_freqs[0]) + 1)
                if recreate_matrix:
                    tmp_spectra = np.zeros((len(y_borders) - 1, len(x_borders) - 1))

                get_spec_plot_matrix = True
                # checked_xy_borders = True

            for i in range(len(y_borders) - 1):
                for j in range(len(x_borders) - 1):
                    if x_borders[j] > tmp_times[-1]:
                        break
                    if x_borders[j + 1] < tmp_times[0]:
                        continue

                    t_mask = np.arange(len(tmp_times))[(tmp_times >= x_borders[j]) & (tmp_times < x_borders[j + 1])]
                    f_mask = np.arange(len(plot_spectra))[(plot_freqs >= y_borders[i]) & (plot_freqs < y_borders[i + 1])]

                    if len(t_mask) == 0 or len(f_mask) == 0:
                        continue

                    tmp_spectra[i, j] = np.max(plot_spectra[f_mask[:, None], t_mask])


        # psd and fish fundamentals frequency detection
        if extract_funds_and_signature:
            power = [np.array([]) for i in range(len(spec_times) - (nffts_per_psd - 1))]

            for t in range(len(spec_times) - (nffts_per_psd - 1)):
                power[t] = np.mean(comb_spectra[:, t:t + nffts_per_psd], axis=1)

            if plot_harmonic_groups:
                pool = multiprocessing.Pool(1)
            else:
                pool = multiprocessing.Pool(core_count // 2)
                # pool = multiprocessing.Pool(core_count - 1)
            func = partial(harmonic_groups, spec_freqs, **kwargs)
            a = pool.map(func, power)
            pool.terminate()

            # get signatures
            # log_spectra = 10.0 * np.log10(np.array(spectra))
            log_spectra = decibel(np.array(spectra))

            for p in range(len(power)):
                tmp_fundamentals = fundamental_freqs(a[p][0])
                # tmp_fundamentals = a[p][0]
                fundamentals.append(tmp_fundamentals)

                if len(tmp_fundamentals) >= 1:
                    f_idx = np.array([np.argmin(np.abs(spec_freqs - f)) for f in tmp_fundamentals])
                    tmp_signatures = log_spectra[:, np.array(f_idx), p].transpose()
                else:
                    tmp_signatures = np.array([])

                signatures.append(tmp_signatures)
            pool.terminate()

        # if nffts_per_psd == 1:
        #     tmp_times = spec_times - ((nfft / samplerate) / 2) + (start_idx / samplerate)
        # else:
        #     tmp_times = spec_times[:-(nffts_per_psd - 1)] - ((nfft / samplerate) / 2) + (start_idx / samplerate)
        non_overlapping_idx = (1 - overlap_frac) * nfft
        start_idx += int((len(spec_times) - nffts_per_psd + 1) * non_overlapping_idx)
        times = np.concatenate((times, tmp_times))

        if start_idx >= end_idx or last_run:
            break

    if create_plotable_spectrogram and not extract_funds_and_signature:
        return tmp_spectra, times

    elif extract_funds_and_signature and not create_plotable_spectrogram:
        return fundamentals, signatures, positions, times

    else:
        return fundamentals, signatures, positions, times, tmp_spectra


# def decibel(spec, ref_power=1.0, min_power=1e-20):
#     """
#     Transforms power to decibel relative to ref_power.
#
#     decibel_psd = 10 * log10(power/ref_power)
#
#     Parameters
#     ----------
#     spec: array
#         the power values of the power spectrum or spectrogram.
#     ref_power: float
#         the reference power for computing decibel. If set to None the maximum power is used.
#     min_power: float
#         power values smaller than min_power are set to np.nan.
#
#     Returns
#     -------
#     decibel_psd: array
#         the power values in decibel
#     """
#     if ref_power is None:
#         ref_power = np.max(spec)
#
#     decibel_spec = spec.copy()
#     for i in range(len(spec)):
#         decibel_spec[i][decibel_spec[i] < min_power] = np.nan
#         decibel_spec[i][decibel_spec[i] >= min_power] = 10.0 * np.log10(decibel_spec[i][decibel_spec[i] >= min_power]/ref_power)
#     return decibel_spec


class Obs_tracker():
    # def __init__(self, data, samplerate, start_time, end_time, fresolution, overlap_frac, channels,
    #              nffts_per_psd, data_snippet_idxs, freq_tolerance, tmp_spectra=None, times=None, fund_v=None, ident_v=None, idx_v=None, sign_v=None, **kwargs):
    def __init__(self, data, samplerate, start_time, end_time, channels, data_snippet_idxs, **kwargs):

        # write input into self.
        self.data = data
        self.samplerate = samplerate
        self.start_time = start_time
        self.end_time = end_time
        if self.end_time < 0.0:
            self.end_time = len(self.data) / self.samplerate

        # self.fresolution = fresolution
        # self.overlap_frac = overlap_frac
        self.channels = channels
        # self.nffts_per_psd = nffts_per_psd
        self.data_snippet_idxs = data_snippet_idxs
        # self.freq_tolerance = freq_tolerance
        self.kwargs = kwargs

        # recalculate right start/end time and idx
        # self.start_idx = int(self.start_time * self.samplerate)
        # if self.end_time < 0.0:
        #     self.end_time = len(self.data) / self.samplerate
        #     self.end_idx = int(len(self.data) - 1)
        # else:
        #     self.end_idx = int(self.end_time * self.samplerate)
        #     if self.end_idx >= int(len(self.data) - 1):
        #         self.end_idx = int(len(self.data) - 1)

        self.kwargs['mains_freq'] = 0.
        self.kwargs['max_fill_ratio'] = 0.5
        self.kwargs['min_group_size'] = 2

        # primary tracking vectors
        self.fund_v = None
        self.ident_v = None
        self.idx_v = None
        self.sign_v = None

        # plot spectrum
        self.fundamentals = None
        self.times = None
        self.tmp_spectra = None
        self.part_spectra = None

        self.current_task = None
        self.current_idx = None
        self.x_zoom_0 = None
        self.x_zoom_1 = None
        self.y_zoom_0 = None
        self.y_zoom_1 = None

        # create plot environment
        self.main_fig = plt.figure(facecolor='white', figsize=(55. / 2.54, 30. / 2.54))

        # main window
        self.main_fig.canvas.mpl_connect('key_press_event', self.keypress)
        self.main_fig.canvas.mpl_connect('button_press_event', self.buttonpress)
        plt.rcParams['keymap.save'] = ''  # was s
        plt.rcParams['keymap.back'] = ''  # was c

        self.main_ax = self.main_fig.add_axes([0.1, 0.1, 0.8, 0.6])
        self.spec_img_handle = None

        self.tmp_plothandel_main = None  # red line
        self.trace_handles = []
        self.active_fundamental0_0 = None
        self.active_fundamental0_1 = None
        self.active_fundamental0_0_handle = None
        self.active_fundamental0_1_handle = None

        self.active_fundamental1_0 = None
        self.active_fundamental1_1 = None
        self.active_fundamental1_0_handle = None
        self.active_fundamental1_1_handle = None
        # self.plot_spectrum()

        # powerspectrum window and parameters
        self.ps_ax = None
        self.tmp_plothandel_ps = []
        self.tmp_harmonics_plot = None
        self.all_peakf_dots = None
        self.good_peakf_dots = None

        self.active_harmonic = None

        # get key options into plot
        self.text_handles_key = []
        self.text_handles_effect = []
        self.key_options()

        self.main_fig.canvas.draw()
        # print('i am in the main loop')

        # get prim spectrum and plot it...
        self.plot_spectrum()

        plt.show()


    def key_options(self):
        # for i in range(len(self.text_handles_key)):
        for i, j in zip(self.text_handles_key, self.text_handles_effect):
            self.main_fig.texts.remove(i)
            self.main_fig.texts.remove(j)
        self.text_handles_key = []
        self.text_handles_effect = []

        if True:
            t = self.main_fig.text(0.1, 0.85,  'h:')
            t1 = self.main_fig.text(0.15, 0.85,  'home (axis)')
            self.text_handles_key.append(t)
            self.text_handles_effect.append(t1)

            t = self.main_fig.text(0.1, 0.825, 'enter:')
            t1 = self.main_fig.text(0.15, 0.825, 'execute task')
            self.text_handles_key.append(t)
            self.text_handles_effect.append(t1)

            # t = self.main_fig.text(0.1, 0.8,  'e:')
            # t1 = self.main_fig.text(0.15, 0.8, 'embed')
            t = self.main_fig.text(0.1, 0.8, 'p:')
            t1 = self.main_fig.text(0.15, 0.8, 'calc/show PSD')
            self.text_handles_key.append(t)
            self.text_handles_effect.append(t1)

            t = self.main_fig.text(0.1, 0.775, 's:')
            t1 = self.main_fig.text(0.15, 0.775, 'create part spectrogram')
            self.text_handles_key.append(t)
            self.text_handles_effect.append(t1)

            t = self.main_fig.text(0.1, 0.75,  '(ctrl+)q:')
            t1 = self.main_fig.text(0.15, 0.75,  'close (all)/ powerspectrum')
            self.text_handles_key.append(t)
            self.text_handles_effect.append(t1)

            t = self.main_fig.text(0.1, 0.725, 'z')
            t1 = self.main_fig.text(0.15, 0.725, 'zoom')
            self.text_handles_key.append(t)
            self.text_handles_effect.append(t1)

        if self.ps_ax:
            if self.current_task == 'part_spec' or self.current_task == 'track_snippet':
                pass
            else:
                t = self.main_fig.text(0.3, 0.85, '(ctrl+)1:')
                t1 = self.main_fig.text(0.35, 0.85, '%.2f dB; rel. dB th for good Peaks' % (self.kwargs['high_threshold']))
                self.text_handles_key.append(t)
                self.text_handles_effect.append(t1)

                t = self.main_fig.text(0.3, 0.825, '(ctrl+)2:')
                t1 = self.main_fig.text(0.35, 0.825, '%.2f dB; rel. dB th for all Peaks' % (self.kwargs['low_threshold']))
                self.text_handles_key.append(t)
                self.text_handles_effect.append(t1)

                t = self.main_fig.text(0.3, 0.8, '(ctrl+)3:')
                t1 = self.main_fig.text(0.35, 0.8, '%.2f; x bin std = low Th' % (self.kwargs['noise_fac']))
                self.text_handles_key.append(t)
                self.text_handles_effect.append(t1)

                t = self.main_fig.text(0.3, 0.775, '(ctrl+)4:')
                t1 = self.main_fig.text(0.35, 0.775, '%.2f; peak_fac' % (self.kwargs['peak_fac']))
                self.text_handles_key.append(t)
                self.text_handles_effect.append(t1)

                t = self.main_fig.text(0.3, 0.75, '(ctrl+)5:')
                t1 = self.main_fig.text(0.35, 0.75, '%.2f dB; min Peak width' % (self.kwargs['min_peak_width']))
                self.text_handles_key.append(t)
                self.text_handles_effect.append(t1)

                t = self.main_fig.text(0.3, 0.725, '(ctrl+)6:')
                t1 = self.main_fig.text(0.35, 0.725, '%.2f X fresolution; max Peak width' % (self.kwargs['max_peak_width_fac']))
                self.text_handles_key.append(t)
                self.text_handles_effect.append(t1)

                t = self.main_fig.text(0.5, 0.85, '(ctrl+)7:')
                t1 = self.main_fig.text(0.55, 0.85, '%.0f; min group size' % (self.kwargs['min_group_size']))
                self.text_handles_key.append(t)
                self.text_handles_effect.append(t1)

                t = self.main_fig.text(0.5, 0.825, '(ctrl+)8:')
                t1 = self.main_fig.text(0.55, 0.825, '%.1f; * fresolution = max dif of harmonics' % (self.kwargs['freq_tol_fac']))
                self.text_handles_key.append(t)
                self.text_handles_effect.append(t1)

                t = self.main_fig.text(0.5, 0.8, '(ctrl+)9:')
                t1 = self.main_fig.text(0.55, 0.8, '%.0f; max divisor to check subharmonics' % (self.kwargs['max_divisor']))
                self.text_handles_key.append(t)
                self.text_handles_effect.append(t1)

                t = self.main_fig.text(0.5, 0.775, '(ctrl+)0:')
                t1 = self.main_fig.text(0.55, 0.775, '%.0f; max freqs to fill in' % (self.kwargs['max_upper_fill']))
                self.text_handles_key.append(t)
                self.text_handles_effect.append(t1)

                t = self.main_fig.text(0.5, 0.75, '(ctrl+)+:')
                t1 = self.main_fig.text(0.55, 0.75, '%.0f; 1 group max double used peaks' % (self.kwargs['max_double_use_harmonics']))
                self.text_handles_key.append(t)
                self.text_handles_effect.append(t1)

                t = self.main_fig.text(0.5, 0.725, '(ctrl+)#:')
                t1 = self.main_fig.text(0.55, 0.725, '%.0f; 1 Peak part of n groups' % (self.kwargs['max_double_use_count']))
                self.text_handles_key.append(t)
                self.text_handles_effect.append(t1)

        if self.current_task == 'part_spec':
            t = self.main_fig.text(0.3, 0.85, '(ctrl+)1:')
            t1 = self.main_fig.text(0.35, 0.85, '%.2f Hz; freuency resolution' % (self.kwargs['fresolution']))
            self.text_handles_key.append(t)
            self.text_handles_effect.append(t1)

            t = self.main_fig.text(0.3, 0.825, '(ctrl+)2:')
            t1 = self.main_fig.text(0.35, 0.825, '%.2f; overlap fraction of FFT windows' % (self.kwargs['overlap_frac']))
            self.text_handles_key.append(t)
            self.text_handles_effect.append(t1)

            t = self.main_fig.text(0.3, 0.8, '(ctrl+)3:')
            t1 = self.main_fig.text(0.35, 0.8, '%.0f; n fft widnows averaged for psd' % (self.kwargs['nffts_per_psd']))
            self.text_handles_key.append(t)
            self.text_handles_effect.append(t1)

            t = self.main_fig.text(0.3, 0.775, '')
            t1 = self.main_fig.text(0.35, 0.775, '%.0f; nfft' % (next_power_of_two(
                self.samplerate / self.kwargs['fresolution'])))
            self.text_handles_key.append(t)
            self.text_handles_effect.append(t1)

            t = self.main_fig.text(0.3, 0.75, '')
            t1 = self.main_fig.text(0.35, 0.75,
                                    '%.3f s; temporal resolution' % (next_power_of_two(
                                        self.samplerate / self.kwargs['fresolution']) / self.samplerate * (
                                        1.-self.kwargs['overlap_frac']) ))
            self.text_handles_key.append(t)
            self.text_handles_effect.append(t1)

        if self.current_task == 'check_tracking':
            if self.active_fundamental0_0 and self.active_fundamental0_1:
                a_error = np.sqrt(
                    np.sum([(self.signatures[self.active_fundamental0_0[0]][self.active_fundamental0_0[1]][k]
                             - self.signatures[self.active_fundamental0_1[0]][self.active_fundamental0_1[1]][k]) ** 2
                            for k in
                            range(len(self.signatures[self.active_fundamental0_0[0]][self.active_fundamental0_0[1]]))]))
                f_error = np.abs(self.fundamentals[self.active_fundamental0_0[0]][self.active_fundamental0_0[1]] -
                                 self.fundamentals[self.active_fundamental0_1[0]][self.active_fundamental0_1[1]])
                t_error = np.abs(self.times[self.active_fundamental0_0[0]] - self.times[self.active_fundamental0_1[0]])
                error = estimate_error(a_error, f_error, t_error, self.a_error_dist, self.f_error_dist)

                t = self.main_fig.text(0.3, 0.85, 'freq error:')
                t1 = self.main_fig.text(0.35, 0.85, '%.2f Hz' % (f_error))
                self.text_handles_key.append(t)
                self.text_handles_effect.append(t1)

                t = self.main_fig.text(0.3, 0.825, 'amp. error:')
                t1 = self.main_fig.text(0.35, 0.825, '%.2f dB' % (a_error))
                self.text_handles_key.append(t)
                self.text_handles_effect.append(t1)

                t = self.main_fig.text(0.3, 0.8, 'time error')
                t1 = self.main_fig.text(0.35, 0.8, '%.2f s' % (t_error))
                self.text_handles_key.append(t)
                self.text_handles_effect.append(t1)

                t = self.main_fig.text(0.3, 0.775, 'df / s')
                t1 = self.main_fig.text(0.35, 0.775, '%.2f s' % (f_error / t_error))
                self.text_handles_key.append(t)
                self.text_handles_effect.append(t1)

                t = self.main_fig.text(0.3, 0.725, 'error value')
                t1 = self.main_fig.text(0.35, 0.725, '%.3f' % (error))
                self.text_handles_key.append(t)
                self.text_handles_effect.append(t1)

            if self.active_fundamental1_0 and self.active_fundamental1_1:
                a_error = np.sqrt(
                    np.sum([(self.signatures[self.active_fundamental1_0[0]][self.active_fundamental1_0[1]][k]
                             - self.signatures[self.active_fundamental1_1[0]][self.active_fundamental1_1[1]][k]) ** 2
                            for k in
                            range(len(self.signatures[self.active_fundamental1_0[0]][self.active_fundamental1_0[1]]))]))
                f_error = np.abs(self.fundamentals[self.active_fundamental1_0[0]][self.active_fundamental1_0[1]] -
                                 self.fundamentals[self.active_fundamental1_1[0]][self.active_fundamental1_1[1]])
                t_error = np.abs(self.times[self.active_fundamental1_0[0]] - self.times[self.active_fundamental1_1[0]])
                error = estimate_error(a_error, f_error, t_error, self.a_error_dist, self.f_error_dist)

                t = self.main_fig.text(0.5, 0.85, 'freq error:')
                t1 = self.main_fig.text(0.55, 0.85, '%.2f Hz' % (f_error))
                self.text_handles_key.append(t)
                self.text_handles_effect.append(t1)

                t = self.main_fig.text(0.5, 0.825, 'amp. error:')
                t1 = self.main_fig.text(0.55, 0.825, '%.2f dB' % (a_error))
                self.text_handles_key.append(t)
                self.text_handles_effect.append(t1)

                t = self.main_fig.text(0.5, 0.8, 'time error')
                t1 = self.main_fig.text(0.55, 0.8, '%.2f s' % (t_error))
                self.text_handles_key.append(t)
                self.text_handles_effect.append(t1)

                t = self.main_fig.text(0.5, 0.775, 'df / s')
                t1 = self.main_fig.text(0.55, 0.775, '%.2f s' % (f_error / t_error))
                self.text_handles_key.append(t)
                self.text_handles_effect.append(t1)

                t = self.main_fig.text(0.5, 0.725, 'error value')
                t1 = self.main_fig.text(0.55, 0.725, '%.3f' % (error))
                self.text_handles_key.append(t)
                self.text_handles_effect.append(t1)

    def keypress(self, event):
        self.key_options()

        if event.key in 'h':
            if self.main_ax:
                self.main_ax.set_xlim([self.start_time, self.end_time])
                self.main_ax.set_ylim([0, 2000])
            if self.ps_ax:
                self.ps_ax.set_ylim([0, 2000])

            if hasattr(self.part_spectra, '__len__'):
                # self.main_fig.delaxes(self.main_ax)
                # self.main_ax = self.main_fig.add_axes([.1, .1, .8, .6])
                self.spec_img_handle.remove()
                self.spec_img_handle = self.main_ax.imshow(decibel(self.tmp_spectra)[::-1], extent=[self.start_time, self.end_time, 0, 2000],
                                    aspect='auto', alpha=0.7)
                self.main_ax.set_xlim([self.start_time, self.end_time])
                self.main_ax.set_ylim([0, 2000])
                self.main_ax.set_xlabel('time [s]')
                self.main_ax.set_ylabel('frequency [Hz]')

        if event.key in 'enter':
            if self.current_task == 'show_spectrum':
                if self.tmp_plothandel_main and self.ioi:
                    self.current_task = None
                    self.plot_ps()
                else:
                    print('\nmissing data')

            if self.current_task == 'update_hg':
                self.current_task = None
                self.update_hg()

            if self.current_task == 'zoom':
                self.current_task = None
                self.zoom()

            if self.current_task == 'track_snippet':
                self.current_task = None
                self.track_snippet()

            if self.current_task == 'part_spec':
                self.current_task = None
                self.plot_spectrum(part_spec=True)

        if event.key in 'e':
            embed()
            # quit()

        if event.key in 'p':
            self.current_task = 'show_spectrum'
            # print('\n%s' % self.current_task)

        if event.key in 't':
            self.current_task = 'track_snippet'

        if event.key == 'ctrl+q':
            plt.close(self.main_fig)
            # self.main_fig.close()
            return

        if event.key in 'q' and self.ps_ax:
            self.main_fig.delaxes(self.ps_ax)
            self.ps_ax = None
            self.tmp_plothandel_ps = []
            self.all_peakf_dots = None
            self.good_peakf_dots = None
            self.main_ax.set_position([.1, .1, .8, .6])

        if event.key in 'c':
            self.current_task = 'check_tracking'

        if event.key in 'z':
            self.current_task = 'zoom'

        if event.key in 's':
            self.current_task = 'part_spec'


        if self.current_task == 'part_spec':
            if event.key == '1':
                if self.kwargs['fresolution'] > 0.25:
                    self.kwargs['fresolution'] -= 0.25
                else:
                    self.kwargs['fresolution'] -= 0.05

            if event.key == 'ctrl+1':
                if self.kwargs['fresolution'] >= 0.25:
                    self.kwargs['fresolution'] += 0.25
                else:
                    self.kwargs['fresolution'] += 0.05

            if event.key == '2':
                self.kwargs['overlap_frac'] -= 0.05

            if event.key == 'ctrl+2':
                self.kwargs['overlap_frac'] += 0.05

            if event.key == '3':
                self.kwargs['nffts_per_psd'] -= 1

            if event.key == 'ctrl+3':
                self.kwargs['nffts_per_psd'] += 1

        else:
            if self.ps_ax:
                if event.key == '1':
                    self.kwargs['high_threshold'] -= 2.5
                    self.current_task = 'update_hg'
                if event.key == 'ctrl+1':
                    self.kwargs['high_threshold'] += 2.5
                    self.current_task = 'update_hg'

                if event.key == '2':
                    self.kwargs['low_threshold'] -= 2.5
                    self.current_task = 'update_hg'
                if event.key == 'ctrl+2':
                    self.kwargs['low_threshold'] += 2.5
                    self.current_task = 'update_hg'

                if event.key == '3':
                    self.kwargs['noise_fac'] -= 1
                    self.kwargs['low_threshold'] = 0.
                    self.kwargs['high_threshold'] = 0.
                    self.current_task = 'update_hg'
                if event.key == 'ctrl+3':
                    self.kwargs['noise_fac'] += 1
                    self.kwargs['low_threshold'] = 0.
                    self.kwargs['high_threshold'] = 0.
                    self.current_task = 'update_hg'

                if event.key == '4':
                    self.kwargs['peak_fac'] -= 0.1
                    self.kwargs['low_threshold'] = 0.
                    self.kwargs['high_threshold'] = 0.
                    self.current_task = 'update_hg'

                if event.key == 'ctrl+4':
                    self.kwargs['peak_fac'] += 0.1
                    self.kwargs['low_threshold'] = 0.
                    self.kwargs['high_threshold'] = 0.
                    self.current_task = 'update_hg'

                if event.key == '5':
                    self.kwargs['min_peak_width'] -= 0.5
                    self.current_task = 'update_hg'
                if event.key == 'ctrl+5':
                    self.kwargs['min_peak_width'] += 0.5
                    self.current_task = 'update_hg'

                if event.key == '6':
                    self.kwargs['max_peak_width_fac'] -= 1.
                    self.current_task = 'update_hg'
                if event.key == 'ctrl+6':
                    self.kwargs['max_peak_width_fac'] += 1.
                    self.current_task = 'update_hg'

                if event.key == '7':
                    self.kwargs['min_group_size'] -= 1.
                    self.current_task = 'update_hg'
                if event.key == 'ctrl+7':
                    self.kwargs['min_group_size'] += 1.
                    self.current_task = 'update_hg'

                if event.key == '8':
                    self.kwargs['freq_tol_fac'] -= .1
                    self.current_task = 'update_hg'
                if event.key == 'ctrl+8':
                    self.kwargs['freq_tol_fac'] += .1
                    self.current_task = 'update_hg'

                if event.key == '9':
                    self.kwargs['max_divisor'] -= 1.
                    self.current_task = 'update_hg'
                if event.key == 'ctrl+9':
                    self.kwargs['max_divisor'] += 1.
                    self.current_task = 'update_hg'

                if event.key == '0':
                    self.kwargs['max_upper_fill'] -= 1
                    self.current_task = 'update_hg'
                if event.key == 'ctrl+0':
                    self.kwargs['max_upper_fill'] += 1
                    self.current_task = 'update_hg'

                if event.key == '+':
                    self.kwargs['max_double_use_harmonics'] -= 1
                    self.current_task = 'update_hg'
                if event.key == 'ctrl+' + '+':
                    self.kwargs['max_double_use_harmonics'] += 1.
                    self.current_task = 'update_hg'

                if event.key == '#':
                    self.kwargs['max_double_use_count'] -= 1
                    self.current_task = 'update_hg'
                if event.key == 'ctrl+#':
                    self.kwargs['max_double_use_count'] += 1.
                    self.current_task = 'update_hg'

        self.key_options()
        self.main_fig.canvas.draw()
        # plt.show()


    def buttonpress(self, event):
        if event.button == 2:
            if event.inaxes != self.ps_ax:
                if self.tmp_plothandel_main:
                    self.tmp_plothandel_main.remove()
                    self.tmp_plothandel_main = None

            if self.tmp_harmonics_plot:
                self.tmp_harmonics_plot.remove()
                self.tmp_harmonics_plot = None
                self.active_harmonic = None

                if self.ps_ax:
                    ylims = self.main_ax.get_ylim()
                    self.ps_ax.set_ylim([ylims[0], ylims[1]])

            if self.active_fundamental0_0_handle:
                self.active_fundamental0_0 = None
                self.active_fundamental0_0_handle.remove()
                self.active_fundamental0_0_handle = None
            if self.active_fundamental0_1_handle:
                self.active_fundamental0_1 = None
                self.active_fundamental0_1_handle.remove()
                self.active_fundamental0_1_handle = None
            if self.active_fundamental1_0_handle:
                self.active_fundamental1_0 = None
                self.active_fundamental1_0_handle.remove()
                self.active_fundamental1_0_handle = None
            if self.active_fundamental1_1_handle:
                self.active_fundamental1_1 = None
                self.active_fundamental1_1_handle.remove()
                self.active_fundamental1_1_handle = None


        if event.inaxes == self.main_ax:
            if self.current_task == 'show_spectrum':
                if event.button == 1:
                    x = event.xdata
                    self.ioi = np.argmin(np.abs(self.times-x))

                    y_lims = self.main_ax.get_ylim()
                    if self.tmp_plothandel_main:
                        self.tmp_plothandel_main.remove()
                    self.tmp_plothandel_main, = self.main_ax.plot([self.times[self.ioi], self.times[self.ioi]], [y_lims[0], y_lims[1]], color='red', linewidth='2')

            if self.current_task == 'zoom':
                if event.button == 2:
                    self.x_zoom_0 = None
                    self.x_zoom_1 = None
                    self.y_zoom_0 = None
                    self.y_zoom_1 = None

                if event.inaxes == self.main_ax:
                    if event.button == 1:
                        if self.x_zoom_0:
                            self.x_zoom_1 = event.xdata
                        else:
                            self.x_zoom_0 = event.xdata
                    if event.button == 3:
                        if self.y_zoom_0:
                            self.y_zoom_1 = event.ydata
                        else:
                            self.y_zoom_0 = event.ydata

            if self.current_task == 'check_tracking' and hasattr(self.fundamentals, '__len__'):

                if event.button == 1:
                    if event.key == 'control':
                        x = event.xdata
                        y = event.ydata

                        funds_ioi = np.argsort(np.abs(self.times - x))[0]
                        fund_ioi = np.argsort(np.abs(self.fundamentals[funds_ioi] - y))[0]

                        self.active_fundamental0_0 = (funds_ioi, fund_ioi)

                        if self.active_fundamental0_0_handle:
                            self.active_fundamental0_0_handle.remove()
                        self.active_fundamental0_0_handle, = self.main_ax.plot(self.times[funds_ioi], self.fundamentals[funds_ioi][fund_ioi], 'o', color='red', markersize=4)
                    else:
                        x = event.xdata
                        y = event.ydata

                        funds_ioi = np.argsort(np.abs(self.times - x))[0]
                        fund_ioi = np.argsort(np.abs(self.fundamentals[funds_ioi] - y))[0]

                        self.active_fundamental0_1 = (funds_ioi, fund_ioi)

                        if self.active_fundamental0_1_handle:
                            self.active_fundamental0_1_handle.remove()

                        self.active_fundamental0_1_handle, = self.main_ax.plot(self.times[funds_ioi], self.fundamentals[funds_ioi][fund_ioi], 'o', color='red', markersize=4)

                if event.button == 3:
                    if event.key == 'control':
                        x = event.xdata
                        y = event.ydata

                        funds_ioi = np.argsort(np.abs(self.times - x))[0]
                        fund_ioi = np.argsort(np.abs(self.fundamentals[funds_ioi] - y))[0]

                        self.active_fundamental1_0 = (funds_ioi, fund_ioi)

                        if self.active_fundamental1_0_handle:
                            self.active_fundamental1_0_handle.remove()
                        self.active_fundamental1_0_handle, = self.main_ax.plot(self.times[funds_ioi],
                                                                               self.fundamentals[funds_ioi][fund_ioi],
                                                                               'o', color='green', markersize=4)
                    else:
                        x = event.xdata
                        y = event.ydata

                        funds_ioi = np.argsort(np.abs(self.times - x))[0]
                        fund_ioi = np.argsort(np.abs(self.fundamentals[funds_ioi] - y))[0]

                        self.active_fundamental1_1 = (funds_ioi, fund_ioi)

                        if self.active_fundamental1_1_handle:
                            self.active_fundamental1_1_handle.remove()

                        self.active_fundamental1_1_handle, = self.main_ax.plot(self.times[funds_ioi],
                                                                               self.fundamentals[funds_ioi][fund_ioi],
                                                                               'o', color='green', markersize=4)

        if self.ps_ax and event.inaxes == self.ps_ax:
            if not self.active_harmonic:
                self.active_harmonic = 1.

            if event.button == 1:
                plot_power = decibel(self.power)
                y = event.ydata
                active_all_freq = self.all_peakf[:, 0][np.argsort(np.abs(self.all_peakf[:, 0] - y))][0]

                plot_harmonics = np.arange(active_all_freq, 3000, active_all_freq)

                if self.tmp_harmonics_plot:
                    self.tmp_harmonics_plot.remove()

                self.tmp_harmonics_plot, = self.ps_ax.plot(np.ones(len(plot_harmonics)) * np.max(plot_power[self.freqs <= 3000.0]) + 10., plot_harmonics, 'o', color='k')

                current_ylim = self.ps_ax.get_ylim()
                self.ps_ax.set_ylim([current_ylim[0] + active_all_freq / self.active_harmonic, current_ylim[1] + active_all_freq / self.active_harmonic])
                self.active_harmonic += 1

            if event.button == 3:
                plot_power = decibel(self.power)
                y = event.ydata
                active_all_freq = self.all_peakf[:, 0][np.argsort(np.abs(self.all_peakf[:, 0] - y))][0]

                plot_harmonics = np.arange(active_all_freq, 3000, active_all_freq)

                if self.tmp_harmonics_plot:
                    self.tmp_harmonics_plot.remove()

                self.tmp_harmonics_plot, = self.ps_ax.plot(np.ones(len(plot_harmonics)) * np.max(plot_power[self.freqs <= 3000.0]) + 10., plot_harmonics, 'o', color='k')

                current_ylim = self.ps_ax.get_ylim()
                self.ps_ax.set_ylim([current_ylim[0] - active_all_freq / self.active_harmonic, current_ylim[1] - active_all_freq / self.active_harmonic])
                self.active_harmonic -= 1

        self.key_options()
        self.main_fig.canvas.draw()


    def plot_spectrum(self, part_spec = False):
        if part_spec:
            limitations = self.main_ax.get_xlim()
            min_freq = self.main_ax.get_ylim()[0]
            max_freq = self.main_ax.get_ylim()[1]

            self.part_spectra, self.part_times = get_spectrum_funds_amp_signature(
                self.data, self.samplerate, self.channels, self.data_snippet_idxs, limitations[0], limitations[1],
                comp_min_freq=min_freq, comp_max_freq=max_freq, create_plotable_spectrogram=True,
                extract_funds_and_signature=False, **self.kwargs)

                # self.main_fig.delaxes(self.main_ax)
            self.spec_img_handle.remove()

            # self.main_ax = self.main_fig.add_axes([.1, .1, .8, .6])
            self.spec_img_handle = self.main_ax.imshow(decibel(self.part_spectra)[::-1],
                                                       extent=[limitations[0], limitations[1], min_freq, max_freq],
                                                       aspect='auto', alpha=0.7)
            self.main_ax.set_xlabel('time [s]')
            self.main_ax.set_ylabel('frequency [Hz]')
        else:
            if not hasattr(self.tmp_spectra, '__len__'):
                self.tmp_spectra, self.times = get_spectrum_funds_amp_signature(
                    self.data, self.samplerate, self.channels, self.data_snippet_idxs, self.start_time, self.end_time,
                    create_plotable_spectrogram=True, extract_funds_and_signature=False,  **self.kwargs)

            self.spec_img_handle = self.main_ax.imshow(decibel(self.tmp_spectra)[::-1], extent=[self.start_time, self.end_time, 0, 2000],
                                aspect='auto', alpha=0.7)
            self.main_ax.set_xlabel('time [s]')
            self.main_ax.set_ylabel('frequency [Hz]')

    def track_snippet(self):
        if hasattr(self.fund_v, '__len__'):
            for i in reversed(range(len(self.trace_handles))):
                self.trace_handles[i].remove()
                self.trace_handles.pop(i)

            self.fund_v = None
            self.ident_v = None
            self.idx_v = None
            self.sign_v = None

        snippet_start, snippet_end = self.main_ax.get_xlim()

        reextract_fundamentals = True
        if self.fundamentals:
            if (snippet_start >= self.times[0]) and (snippet_end <= self.times[-1]):
                reextract_fundamentals = False

        if reextract_fundamentals:
            self.fundamentals, self.signatures, self.positions, self.times = \
                get_spectrum_funds_amp_signature(self.data, self.samplerate, self.channels, self.data_snippet_idxs,
                                                 snippet_start, snippet_end, create_plotable_spectrogram=False,
                                                 extract_funds_and_signature=True, **self.kwargs)

            self.fund_v, self.ident_v, self.idx_v, self.sign_v, self.a_error_dist, self.f_error_dist = \
                freq_tracking_v2(self.fundamentals, self.signatures, self.positions, self.times, self.kwargs['freq_tolerance'],
                                 n_channels = len(self.channels))

        else:
            mask = np.arange(len(self.times))[(self.times >= snippet_start) & (self.times <= snippet_end)]
            self.fund_v, self.ident_v, self.idx_v, self.sign_v, self.a_error_dist, self.f_error_dist = \
                freq_tracking_v2(np.array(self.fundamentals)[mask], np.array(self.signatures)[mask], self.positions,
                                 self.times[mask], self.kwargs['freq_tolerance'], n_channels=len(self.channels))

        self.plot_traces()

    def plot_traces(self):
        # self.main_ax.imshow(10.0 * np.log10(self.tmp_spectra)[::-1], extent=[self.start_time, self.end_time, 0, 2000], aspect='auto', alpha=0.7)

        possible_identities = np.unique(self.ident_v[~np.isnan(self.ident_v)])
        for ident in np.array(possible_identities):
            c = np.random.rand(3)
            h, =self.main_ax.plot(self.times[self.idx_v[self.ident_v == ident]], self.fund_v[self.ident_v == ident], marker='.', color=c)
            self.trace_handles.append(h)

    def plot_ps(self):
        # nfft = next_power_of_two(self.samplerate / self.fresolution)
        nfft = next_power_of_two(self.samplerate / self.kwargs['fresolution'])
        data_idx0 = int(self.times[self.ioi] * self.samplerate)
        data_idx1 = int(data_idx0 + nfft+1)

        all_c_spectra = []
        all_c_freqs = None

        for channel in self.channels:
            # c_spectrum, c_freqs, c_time = spectrogram(self.data[data_idx0: data_idx1, channel], self.samplerate,
            #                                           fresolution = self.fresolution, overlap_frac = self.overlap_frac)
            c_spectrum, c_freqs, c_time = spectrogram(self.data[data_idx0: data_idx1, channel], self.samplerate,
                                                      fresolution=self.kwargs['fresolution'], overlap_frac=self.kwargs['overlap_frac'])
            if not hasattr(all_c_freqs, '__len__'):
                all_c_freqs = c_freqs
            all_c_spectra.append(c_spectrum)

        comb_spectra = np.sum(all_c_spectra, axis=0)
        self.power = np.hstack(comb_spectra)
        self.freqs = all_c_freqs

        groups, _, _, self.all_peakf, self.good_peakf, self.kwargs['low_threshold'], self.kwargs['high_threshold'], self.psd_baseline = harmonic_groups(all_c_freqs, self.power, **self.kwargs)

        # plot_power = 10.0 * np.log10(self.power)
        plot_power = decibel(self.power)

        if not self.ps_ax:
            self.main_ax.set_position([.1, .1, .5, .6])
            self.ps_ax = self.main_fig.add_axes([.6, .1, .3, .6])
            # self.ps_ax.set_yticks([])
            self.ps_ax.yaxis.tick_right()
            self.ps_ax.yaxis.set_label_position("right")
            self.ps_ax.set_ylabel('frequency [Hz]')
            self.ps_ax.set_xlabel('power [dB]')
            self.ps_handle, =self.ps_ax.plot(plot_power[self.freqs <= 3000.0], self.freqs[self.freqs <= 3000.0],
                                             color='cornflowerblue')

            self.all_peakf_dots, = self.ps_ax.plot(np.ones(len(self.all_peakf[:, 0])) * np.max(plot_power[self.freqs <= 3000.0]) + 5., self.all_peakf[:, 0], 'o', color='red')
            self.good_peakf_dots, = self.ps_ax.plot(np.ones(len(self.good_peakf)) * np.max(plot_power[self.freqs <= 3000.0]) + 5., self.good_peakf, 'o', color='green')

        else:
            self.ps_handle.set_data(plot_power[all_c_freqs <= 3000.0], all_c_freqs[all_c_freqs <= 3000.0])
            self.all_peakf_dots.remove()
            self.good_peakf_dots.remove()
            self.all_peakf_dots, = self.ps_ax.plot(
                np.ones(len(self.all_peakf[:, 0])) * np.max(plot_power[all_c_freqs <= 3000.0]) +5., self.all_peakf[:, 0], 'o',
                color='red')
            self.good_peakf_dots, = self.ps_ax.plot(
                np.ones(len(self.good_peakf)) * np.max(plot_power[all_c_freqs <= 3000.0]) +5., self.good_peakf, 'o',
                color='green')

        for i in range(len(self.tmp_plothandel_ps)):
            self.tmp_plothandel_ps[i].remove()
        self.tmp_plothandel_ps = []

        for fish in range(len(groups)):
            c = np.random.rand(3)

            h, = self.ps_ax.plot(decibel(groups[fish][groups[fish][:, 0] < 3000., 1]),
                                 groups[fish][groups[fish][:, 0] < 3000., 0], 'o', color=c,
                                 markersize=7, alpha=0.9)
            self.tmp_plothandel_ps.append(h)

        ylims = self.main_ax.get_ylim()
        self.ps_ax.set_ylim([ylims[0], ylims[1]])

    def update_hg(self):
        # self.fundamentals = None
        # groups = harmonic_groups(self.freqs, self.power, **self.kwargs)
        groups, _, _, self.all_peakf, self.good_peakf, self.kwargs['low_threshold'], self.kwargs['high_threshold'], self.psd_baseline = \
            harmonic_groups(self.freqs, self.power, **self.kwargs)
        # print(self.psd_baseline)
        for i in range(len(self.tmp_plothandel_ps)):
            self.tmp_plothandel_ps[i].remove()
        self.tmp_plothandel_ps = []

        for fish in range(len(groups)):
            c = np.random.rand(3)

            h, = self.ps_ax.plot(decibel(groups[fish][groups[fish][:, 0] < 3000., 1]),
                                 groups[fish][groups[fish][:, 0] < 3000., 0], 'o', color=c,
                                 markersize=7, alpha=0.9)
            self.tmp_plothandel_ps.append(h)
        plot_power = decibel(self.power)
        self.all_peakf_dots.remove()
        self.good_peakf_dots.remove()
        self.all_peakf_dots, = self.ps_ax.plot(
            np.ones(len(self.all_peakf[:, 0])) * np.max(plot_power[self.freqs <= 3000.0]) + 5., self.all_peakf[:, 0], 'o',
            color='red')
        self.good_peakf_dots, = self.ps_ax.plot(
            np.ones(len(self.good_peakf)) * np.max(plot_power[self.freqs <= 3000.0]) + 5., self.good_peakf, 'o',
            color='green')

        ylims = self.main_ax.get_ylim()
        self.ps_ax.set_ylim([ylims[0], ylims[1]])

    def zoom(self):
        if self.x_zoom_0 and self.x_zoom_1:
            xlims = np.array([self.x_zoom_0, self.x_zoom_1])
            self.main_ax.set_xlim(xlims[np.argsort(xlims)])
        if self.y_zoom_0 and self.y_zoom_1:
            ylims = np.array([self.y_zoom_0, self.y_zoom_1])
            self.main_ax.set_ylim(ylims[np.argsort(ylims)])
            if self.ps_ax:
                self.ps_ax.set_ylim(ylims[np.argsort(ylims)])
        self.x_zoom_0 = None
        self.x_zoom_1 = None
        self.y_zoom_0 = None
        self.y_zoom_1 = None

def fish_tracker(data_file, start_time=0.0, end_time=-1.0, grid=False, data_snippet_secs=15., verbose=0, **kwargs):
    """
    Performs the steps to analyse long-term recordings of wave-type weakly electric fish including frequency analysis,
    fish tracking and more.

    In small data snippets spectrograms and power-spectra are calculated. With the help of the power-spectra harmonic
    groups and therefore electric fish fundamental frequencies can be detected. These fundamental frequencies are
    detected for every time-step throughout the whole file. Afterwards the fundamental frequencies get assigned to
    different fishes.

    :param data_file: (string) filepath of the analysed data file.
    :param data_snippet_secs: (float) duration of data snipped processed at once in seconds. Necessary because of memory issues.
    :param nffts_per_psd: (int) amount of nffts used to calculate one psd.
    :param start_time: (int) analyze data from this time on (in seconds).  XXX this should be a float!!!!
    :param end_time: (int) stop analysis at this time (in seconds).  XXX this should be a float!!!!
    :param plot_data_func: (function) if plot_data_func = plot_fishes creates a plot of the sorted fishes.
    :param save_original_fishes: (boolean) if True saves the sorted fishes after the first level of fish sorting.
    :param kwargs: further arguments are passed on to harmonic_groups().
    """
    if data_file.endswith('.mat'):
        if verbose >= 1:
            print ('loading mat file')
        data, samplerate = load_matfile(data_file)

    else:
        data = open_data(data_file, -1, 60.0, 10.0)
        samplerate = data.samplerate

    base_name = os.path.splitext(os.path.basename(data_file))[0]
    f0, f1 = 400, 1050

    channels, coords, neighbours = get_grid_proportions(data, grid, n_tolerance_e=2, verbose=verbose)
    #
    # if verbose >= 1:
    #     print('\nextract fundamentals...')
    #     if verbose >= 2:
    #         print('> frequency resolution = %.2f Hz' % fresolution)
    #         print('> nfft overlap fraction = %.2f' % overlap_frac)

    data_snippet_idxs = int(data_snippet_secs * samplerate)

    # start and end idx
    # start_idx = int(start_time * samplerate)
    # if end_time < 0.0:
    #     end_time = len(data) / samplerate
    #     end_idx = int(len(data) - 1)
    # else:
    #     end_idx = int(end_time * samplerate)
    #     if end_idx >= int(len(data) - 1):
    #         end_idx = int(len(data) - 1)

    #############################################################################################
    # fundamentals = []
    # positions = []
    # times = np.array([])
    # signatures = []
    #
    # # increase_start_idx = False
    # last_run = False
    #
    # print ('')
    # init_idx = False
    # if not init_idx:
    #     init_idx = start_idx
    # next_message = 0.00
    #
    # # create spectra plot ####
    # fig_xspan = 20.
    # fig_yspan = 12.
    # fig_dpi = 80.
    # no_x = fig_xspan * fig_dpi
    # no_y = fig_yspan * fig_dpi
    #
    # min_x = start_time
    # max_x = end_time
    #
    # min_y = 300.
    # max_y = 1200.
    #
    # x_borders = np.linspace(min_x, max_x, no_x * 4)
    # y_borders = np.linspace(min_y, max_y, no_y * 4)
    # checked_xy_borders = False
    #
    # tmp_spectra = np.zeros((len(y_borders) - 1, len(x_borders) - 1))
    #
    # while start_idx <= end_idx:
    #     next_message = include_progress_bar(start_idx - init_idx, end_idx - init_idx, 'extract fundamentals', next_message)
    #
    #     if start_idx >= end_idx - data_snippet_idxs:
    #         last_run = True
    #
    #     # calulate spectogram ....
    #     core_count = multiprocessing.cpu_count()
    #
    #     if plot_harmonic_groups:
    #         pool = multiprocessing.Pool(1)
    #     else:
    #         pool = multiprocessing.Pool(core_count - 1)
    #
    #     nfft = next_power_of_two(samplerate / fresolution)
    #
    #     func = partial(spectrogram, samplerate = samplerate, fresolution=fresolution, overlap_frac=overlap_frac)
    #     a = pool.map(func, [data[start_idx: start_idx + data_snippet_idxs, channel] for channel in channels])  # ret: spec, freq, time
    #
    #     spectra = [a[channel][0] for channel in range(len(a))]
    #     spec_freqs = a[0][1]
    #     spec_times = a[0][2]
    #     pool.terminate()
    #
    #     comb_spectra = np.sum(spectra, axis=0)
    #
    #     if nffts_per_psd == 1:
    #         tmp_times = spec_times - ((nfft / samplerate) / 2) + (start_idx / samplerate)
    #     else:
    #         tmp_times = spec_times[:-(nffts_per_psd - 1)] - ((nfft / samplerate) / 2) + (start_idx / samplerate)
    #
    #
    #     # etxtract reduced spectrum for plot
    #     plot_freqs = spec_freqs[spec_freqs < 2000.]
    #     plot_spectra = np.sum(spectra, axis=0)[spec_freqs < 2000.]
    #     if not checked_xy_borders:
    #         if (tmp_times[1] - tmp_times[0])  > (x_borders[1] - x_borders[0]):
    #             x_borders = np.linspace(min_x, max_x, (max_x - min_x) // (tmp_times[1] - tmp_times[0]) + 1)
    #             tmp_spectra = np.zeros((len(y_borders) - 1, len(x_borders) - 1))
    #
    #         checked_xy_borders = True
    #
    #     for i in range(len(y_borders)-1):
    #         for j in range(len(x_borders)-1):
    #             if x_borders[j] > tmp_times[-1]:
    #                 break
    #             if x_borders[j+1] < tmp_times[0]:
    #                 continue
    #
    #             t_mask = np.arange(len(tmp_times))[(tmp_times >= x_borders[j]) & (tmp_times < x_borders[j+1])]
    #             f_mask = np.arange(len(plot_spectra))[(plot_freqs >= y_borders[i]) & (plot_freqs < y_borders[i+1])]
    #
    #             if len(t_mask) == 0 or len(f_mask) == 0:
    #                 continue
    #             tmp_spectra[i, j] = np.max(plot_spectra[f_mask[:, None], t_mask])
    #
    #     # psd and fish fundamentals frequency detection
    #     power = [np.array([]) for i in range(len(spec_times) - (nffts_per_psd - 1))]
    #
    #     for t in range(len(spec_times) - (nffts_per_psd - 1)):
    #         power[t] = np.mean(comb_spectra[:, t:t + nffts_per_psd], axis=1)
    #
    #     if plot_harmonic_groups:
    #         pool = multiprocessing.Pool(1)
    #     else:
    #         pool = multiprocessing.Pool(core_count - 1)
    #     func = partial(harmonic_groups, spec_freqs, **kwargs)
    #     a = pool.map(func, power)
    #     pool.terminate()
    #
    #     # get signatures
    #     log_spectra = 10.0 * np.log10(np.array(spectra))
    #     for p in range(len(power)):
    #         tmp_fundamentals = fundamental_freqs(a[p][0])
    #         # tmp_fundamentals = a[p][0]
    #         fundamentals.append(tmp_fundamentals)
    #
    #         if len(tmp_fundamentals) >= 1:
    #             f_idx = np.array([np.argmin(np.abs(spec_freqs - f)) for f in tmp_fundamentals])
    #             tmp_signatures = log_spectra[:, np.array(f_idx), p].transpose()
    #         else:
    #             tmp_signatures = np.array([])
    #
    #         signatures.append(tmp_signatures)
    #     pool.terminate()
    #
    #     if nffts_per_psd == 1:
    #         tmp_times = spec_times - ((nfft / samplerate) / 2) + (start_idx / samplerate)
    #     else:
    #         tmp_times = spec_times[:-(nffts_per_psd - 1)] - ((nfft / samplerate) / 2) + (start_idx / samplerate)
    #
    #     non_overlapping_idx = (1 - overlap_frac) * nfft
    #     start_idx += int((len(spec_times) - nffts_per_psd+1) * non_overlapping_idx)
    #
    #     times = np.concatenate((times, tmp_times))
    #
    #     if start_idx >= end_idx or last_run:
    #         break

            ########## ERROR - single electrode fundamental extraction ###############
            # try:
            #     # fig, ax = plt.subplots(figsize=(10., 8.))
            #
            # func = partial(snipped_fundamentals, samplerate=samplerate, start_idx=start_idx, nffts_per_psd=nffts_per_psd,
            #                    fresolution=fresolution, overlap_frac=overlap_frac, plot_harmonic_groups=plot_harmonic_groups,
            #                    increase_start_idx = increase_start_idx, verbose=verbose, **kwargs)
            #
            #     a = pool.map(func, [data[start_idx: start_idx + data_snippet_idxs, channel] for channel in channels])
            #
            #     all_ele_fundamentals = [a[channel][0] for channel in range(len(a))]
            #     all_ele_fund_power = [a[channel][1] for channel in range(len(a))]
            #     ele_spectra = [a[channel][5] for channel in range(len(a))]
            #
            #     ele_spectrum_freqs = a[0][6]
            #     ele_times = a[0][2]
            #     start_idx = a[0][3]
            #     nfft = a[0][4]
            #     pool.terminate()
            #
            # except:
            #     pool.terminate()
            #     print ('Error in pool analysis... start_idx = %.0f' % start_idx)
            #     all_ele_fundamentals = []
            #     all_ele_fund_power = []
            #     ele_spectra = []
            #
            #     for channel in channels:
            #         try:
            #             electrode_fundamentals, electrode_fund_power, electrode_times, start_idx, nfft, spectrum, freqs= snipped_fundamentals(
            #                 data[start_idx: start_idx + data_snippet_idxs, channel], samplerate=samplerate,
            #                 start_idx=start_idx, nffts_per_psd=nffts_per_psd, fresolution=fresolution, overlap_frac=overlap_frac,
            #                 plot_harmonic_groups=plot_harmonic_groups, increase_start_idx = increase_start_idx,
            #                 verbose=verbose, **kwargs)
            #             all_ele_fundamentals.append(electrode_fundamentals)
            #             all_ele_fund_power.append(electrode_fund_power)
            #
            #             ele_spectra.append(spectrum)
            #             ele_spectrum_freqs = freqs
            #             ele_times = electrode_times
            #         except:
            #             print ('Error also in single channel analysis... start_idx = %.0f; channel = %.0f'
            #                    % (start_idx, channel))
            #             #all_ele_fundamentals.append([np.array([], dtype=float) for i in range(len(a[0][0]))])
            #             all_ele_fundamentals.append([np.array([], dtype=float) for i in range(len(electrode_fundamentals))])
            #
            #             #all_ele_fund_power.append([np.array([], dtype=float) for i in range(len(a[0][1]))])
            #             all_ele_fund_power.append([np.array([], dtype=float) for i in range(len(electrode_fund_power))])
            #
            #             #ele_spectra.append([np.full(len(spectrum), np.nan) for i in range(len(a[0][1]))])
            #             ele_spectra.append([np.full(len(spectrum[0]), np.nan) for i in range(len(spectrum))])
            #
            #             print('appended empty. Its dirty but works')
            #
            # ##############################################################################
            # non_overlapping_idx = (1 - overlap_frac) * nfft
            # start_idx += int(len(ele_times) * non_overlapping_idx)
            #
            # for t in range(np.shape(all_ele_fundamentals)[1]):
            #     current_fundamentals, current_positions = grid_fish_frequency_and_position(
            #         np.array(all_ele_fundamentals)[:, t], np.array(all_ele_fund_power)[:, t], channels, coords, neighbours)
            #     fundamentals.append(current_fundamentals)
            #     if len(current_fundamentals) >= 1:
            #         #tmp_signatures = np.zeros(( len(current_fundamentals), len(channels) ))
            #         f_idx = np.array([ np.argmin(np.abs(ele_spectrum_freqs - f)) for f in current_fundamentals])
            #
            #         #tmp_signatures = np.array(ele_spectra)[:, np.array(f_idx), t].transpose()
            #         tmp_signatures = 10.0 * np.log10(np.array(ele_spectra))[:, np.array(f_idx), t].transpose()
            #     else:
            #         tmp_signatures = np.array([])
            #
            #     signatures.append(tmp_signatures)
            #
            #     if not not_tracking:
            #         positions.append(current_positions)
            #
            # times = np.concatenate((times, ele_times))
            #
            # if start_idx >= end_idx or last_run:
            #     break

    Obs_tracker(data, samplerate, start_time, end_time, channels, data_snippet_idxs, **kwargs)


#     tmp_spectra, times = get_spectrum_funds_amp_signature(data, samplerate, channels, fresolution, overlap_frac, nffts_per_psd,
#                                                    data_snippet_idxs, start_time, start_idx, end_time, end_idx,
#                                                    plot_harmonic_groups, extract_funds_and_signature=False, **kwargs)
#
#     # tr = Obs_tracker(data, samplerate, times, tmp_spectra, start_time, end_time, fresolution, overlap_frac, channels,
#     #                  nffts_per_psd, **kwargs)
#     #
#     #
#     # print('returned')
#     # # embed()
#     # quit()
#
#
#     fundamentals, signatures, positions, times= \
#         get_spectrum_funds_amp_signature(data, samplerate, channels, fresolution, overlap_frac, nffts_per_psd,
#                                          data_snippet_idxs, start_time, start_idx, end_time, end_idx,
#                                          plot_harmonic_groups, create_plotable_spectrogram=False, **kwargs)
#
#     if verbose >= 1:
#         print('\nfirst level fish tracking ...')
#         if verbose >= 3:
#             print ('> frequency tolerance: %.2f Hz' % freq_tolerance)
#
#     old_tracking = False
#     if old_tracking:
#         fishes, fishes_x_pos, fishes_y_pos = first_level_fish_sorting(fundamentals, signatures, base_name, times,
#                                                                       n_channels= len(channels), positions=positions,
#                                                                       freq_tolerance=freq_tolerance,
#                                                                       save_original_fishes=save_original_fishes,
#                                                                       output_folder=output_folder, verbose=verbose)
#
#     else:
#         fund_v, ident_v, idx_v, sign_v = freq_tracking_v2(fundamentals, signatures, positions, times, freq_tolerance,
#                                                           n_channels = len(channels))
#
#         Obs_tracker(data, samplerate, times, start_time, end_time, fresolution, overlap_frac, channels, nffts_per_psd,
#                     tmp_spectra, fund_v, ident_v, idx_v, sign_v, **kwargs)
#
#     # embed()
#     quit()
# ####################################################################
#
#     plot_fishes(fishes, times, np.array([]), base_name, save_plot, output_folder)
#
#     ################################## continue no tracking ... ####################################
#     min_occure_time = times[-1] * 0.01 / 60.
#     if min_occure_time > 1.:
#         min_occure_time = 1.
#
#     if verbose >= 1:
#         print('\nexclude fishes...')
#         if verbose >= 2:
#             print('> minimum occur time: %.2f min' % min_occure_time)
#
#     fishes, fishes_x_pos, fishes_y_pos = exclude_fishes(fishes, fishes_x_pos, fishes_y_pos , times, min_occure_time)
#
#     if len(fishes) == 0:
#         print('excluded all fishes. Change parameters.')
#         quit()
#
#     if verbose >= 1:
#         print('\nrise detection...')
#         if verbose >= 2:
#             print('> rise frequency th = %.2f Hz' % rise_f_th)
#
#     all_rises = detect_rises(fishes, times, rise_f_th, verbose=verbose)
#
#     if verbose >= 1:
#         print('')
#         print('cut fishes at rises...')
#
#     # here somethins is fishy...
#     #fishes, fishes_x_pos, fishes_y_pos, all_rises = cut_at_rises(fishes, fishes_x_pos, fishes_y_pos, all_rises, times, min_occure_time)
#
#     if verbose >= 1:
#         print('\ncombining fishes...')
#         if verbose >= 2:
#             print('> maximum time difference: %.2f min' % max_time_tolerance)
#             print('> maximum frequency difference: %.2f Hz' % f_th)
#
#     fishes, fishes_x_pos, fishes_y_pos, all_rises = combine_fishes(fishes, fishes_x_pos, fishes_y_pos, times, all_rises, max_time_tolerance, f_th)
#
#     if 'plt' in locals() or 'plt' in globals():
#         #if not not_tracking:
#         #    plot_positions(fishes, fishes_x_pos, fishes_y_pos, times)
#         plot_fishes(fishes, times, all_rises, base_name, save_plot, output_folder)
#
#     if save_original_fishes:
#         if verbose >= 1:
#             print('')
#             print('saving data to ' + output_folder)
#         save_data(fishes, fishes_x_pos, fishes_y_pos, times, all_rises, base_name, output_folder)
#
#     if verbose >= 1:
#         print('')
#         print('Whole file processed.')

def main():
    # config file name:
    cfgfile = __package__ + '.cfg'

    # command line arguments:
    parser = argparse.ArgumentParser(
        description='Analyse long single- or multi electrode EOD recordings of weakly electric fish.',
        epilog='by bendalab (2015-2017)')
    parser.add_argument('--version', action='version', version=__version__)
    parser.add_argument('-v', action='count', dest='verbose', help='verbosity level')
    parser.add_argument('-c', '--save-config', nargs='?', default='', const=cfgfile,
                        type=str, metavar='cfgfile',
                        help='save configuration to file cfgfile (defaults to {0})'.format(cfgfile))
    parser.add_argument('file', nargs=1, default='', type=str, help='name of the file wih the time series data or the -fishes.npy file saved with the -s option')
    parser.add_argument('start_time', nargs='?', default=0.0, type=float, help='start time of analysis in min.')
    parser.add_argument('end_time', nargs='?', default=-1.0, type=float, help='end time of analysis in min.')
    # parser.add_argument('-g', dest='grid', action='store_true', help='sum up spectrograms of all channels available.')
    parser.add_argument('-g', action='count', dest='grid', help='grid information')
    parser.add_argument('-p', dest='save_plot', action='store_true', help='save output plot as png file')
    parser.add_argument('-s', dest='save_fish', action='store_true',
                        help='save fish EODs after first stage of sorting.')
    parser.add_argument('-f', dest='plot_harmonic_groups', action='store_true', help='plot harmonic group detection')
    parser.add_argument('-t', dest='not_tracking', action='store_true', help='dont track positions')
    parser.add_argument('-o', dest='output_folder', default=".", type=str,
                        help="path where to store results and figures")
    args = parser.parse_args()

    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)

    datafile = args.file[0]

    # set verbosity level from command line:
    verbose = 0
    if args.verbose != None:
        verbose = args.verbose

    # configuration options:
    cfg = ConfigFile()
    add_psd_peak_detection_config(cfg)
    add_harmonic_groups_config(cfg)
    add_tracker_config(cfg)
    
    # load configuration from working directory and data directories:
    cfg.load_files(cfgfile, datafile, 3, verbose)

    # save configuration:
    if len(args.save_config) > 0:
        ext = os.path.splitext(args.save_config)[1]
        if ext != os.extsep + 'cfg':
            print('configuration file name must have .cfg as extension!')
        else:
            print('write configuration to %s ...' % args.save_config)
            cfg.dump(args.save_config)
        return

    # work with previously sorted frequency traces saved as .npy file
    if os.path.splitext(datafile)[1] == '.npy':
        rise_f_th = .5
        max_time_tolerance = 10.
        f_th = 5.
        output_folder = args.output_folder

        a = np.load(sys.argv[1], mmap_mode='r+')
        fishes = a.copy()

        all_times = np.load(sys.argv[1].replace('-fishes', '-times'))

        min_occure_time = all_times[-1] * 0.01 / 60.
        if min_occure_time > 1.:
            min_occure_time = 1.

        if verbose >= 1:
            print('\nexclude fishes...')
            if verbose >= 2:
                print('> minimum occur time: %.2f min' % min_occure_time)
        fishes = exclude_fishes(fishes, all_times, min_occure_time=min_occure_time)

        if verbose >= 1:
            print('\nrise detection...')
            if verbose >= 2:
                print('> rise frequency th = %.2f Hz' % rise_f_th)
        all_rises = detect_rises(fishes, all_times, rise_f_th, verbose)

        if verbose >= 1:
            print('\ncut fishes at rises...')
        fishes, all_rises = cut_at_rises(fishes, all_rises, all_times, min_occure_time)

        if verbose >= 1:
            print('\ncombining fishes...')
            if verbose >= 2:
                print('> maximum time difference: %.2f min' % max_time_tolerance)
                print('> maximum frequency difference: %.2f Hz' % f_th)
        fishes, all_rises = combine_fishes(fishes, all_times, all_rises, max_time_tolerance, f_th)
        if verbose >= 1:
            print('%.0f fishes left' % len(fishes))

        base_name = os.path.splitext(os.path.basename(sys.argv[1]))[0]

        if 'plt' in locals() or 'plt' in globals():
            plot_fishes(fishes, all_times, all_rises, base_name, args.save_plot, args.output_folder)

        if args.save_fish:
            if verbose >= 1:
                print('saving data to ' + output_folder)
            save_data(fishes, all_times, all_rises, base_name, output_folder)

        if verbose >= 1:
            print('Whole file processed.')

    else:
        t_kwargs = psd_peak_detection_args(cfg)
        t_kwargs.update(harmonic_groups_args(cfg))
        t_kwargs.update(tracker_args(cfg))

        # fish_tracker(datafile, args.start_time*60.0, args.end_time*60.0,
        #              args.grid, args.save_plot, args.save_fish, output_folder=args.output_folder,
        #              plot_harmonic_groups=args.plot_harmonic_groups, verbose=verbose, not_tracking=args.not_tracking,
        #              part_analysis=False, **t_kwargs)
        fish_tracker(datafile, args.start_time * 60.0, args.end_time * 60.0,
                     args.grid, **t_kwargs)
        # data_file, start_time = 0.0, end_time = -1.0, grid = False, data_snippet_secs = 15., verbose = 0, ** kwargs

if __name__ == '__main__':
    # how to execute this code properly
    # python -m thunderfish.tracker_v2 <data file> [-v(vv)] [-g(ggg)]
    main()

