"""
Track wave-type electric fish frequencies over time.

fish_tracker(): load data and track fish.
"""
import sys
import os
import argparse
import numpy as np
import glob
import scipy.stats as scp
import multiprocessing
from sklearn.metrics import roc_curve, roc_auc_score
from functools import partial
from .version import __version__
from .configfile import ConfigFile
from .dataloader import open_data
from .powerspectrum import spectrogram, next_power_of_two, decibel
from .harmonicgroups import add_psd_peak_detection_config, add_harmonic_groups_config
from .harmonicgroups import harmonic_groups_args, psd_peak_detection_args
from .harmonicgroups import harmonic_groups, fundamental_freqs, plot_psd_harmonic_groups
from tqdm import tqdm

from IPython import embed
import time

try:
    import matplotlib.pyplot as plt
except ImportError:
    pass


def auto_connect_traces(fund_v, idx_v, ident_v, times, max_dt=120., max_df=2., max_overlap_n=0):
    """
    Connects EOD frequency traces that are less than 5 minutes, and more than 5 seconds apart. These traces are not
    connected because of the temporal resolution of the tracking algorithm. For each EOD frequency trace pair the
    the relative temporal difference and the relative frequency difference decide weather they belong to each other or
    not. (Maximum temporal difference is 300 sec.; maximum frequency difference is 2Hz; Maximum overlapping datapoints
    are 5)

    Parameters
    ----------
    fund_v: array
        detected fundamental frequencies throughout the recording. Result from frequency sorting algorithm (len = n).
    idx_v: array
        time indices of the respective frequency in fund_v (len = n).
    ident_v: array
        respective identity based on tracking algorithm of a signal (len = n).
    times: array
        time array for the whole recording (len = np.unique(idx_v)).
    max_dt: float
        maximum time difference to connect two identities in s.
    max_df: float
        maximum median frequency difference between two identities to be connecten in Hz.
    max_overlap_n: int
        maximum number of overlapping indices between two identities to still be connected.

    Returns
    -------
    ident_v: array
        updated identity array

    """

    only_bc = False
    # idents = np.unique(ident_v[~np.isnan(ident_v)])
    # # med_ident_freq = [np.median(fund_v[ident_v == ident]) for ident in idents]
    #
    # # embed()
    # # quit()
    # next_ident = np.max(ident_v[~np.isnan(ident_v)]) + 1
    # next_message = 0.00
    # for emu, ident in enumerate(idents):
    #     next_message = include_progress_bar(emu, len(idents), 'exclude outliers', next_message)
    #     a = fund_v[ident_v == ident]
    #     real_a_idx = np.arange(len(fund_v))[ident_v == ident]
    #     # plt.plot(a, color='k', alpha = 0.5)
    #     # plt.show()
    #     h_range = np.arange(len(a))
    #     diff_a = np.diff(a)
    #     id_a = np.zeros(len(a))
    #     n_id = 1
    #     jump_idx = np.arange(len(a))[1:][np.abs(diff_a) > 1]
    #
    #     for i in range(len(jump_idx)):
    #         for j in np.arange(i+1, len(jump_idx)):
    #             if jump_idx[j] - jump_idx[i] >= 15:
    #                 break
    #             if np.abs(a[jump_idx[i]-1] - a[jump_idx[j]]) <= np.abs(a[jump_idx[i]-1] - a[jump_idx[i]]) * 0.25 :
    #                 ioi = h_range[((h_range < jump_idx[i]) & (id_a == id_a[jump_idx[i]-1])) |
    #                               ((h_range >= jump_idx[j]) & (id_a == id_a[jump_idx[j]]))]
    #                 id_a[ioi] = n_id
    #                 n_id += 1
    #                 break
    #     # embed()
    #     # plt.close()
    #     # plt.plot(real_a_idx, a, color='grey')
    #     # plt.plot(real_a_idx[jump_idx], a[jump_idx], 'o', color='orange')
    #     for ii in np.unique(id_a):
    #         ident_v[real_a_idx[id_a == ii]] = next_ident
    #         next_ident += 1
    #         plt.plot(real_a_idx[id_a == ii], a[id_a == ii], marker = '.')
    #     # plt.show()

    if not only_bc:
        idents = np.unique(ident_v[~np.isnan(ident_v)])
        med_ident_freq = [np.median(fund_v[ident_v == ident]) for ident in idents]
        i0s = []
        i1s = []
        di = []
        df = []
        # embed()
        # quit()
        next_message = 0.0
        for enu in np.arange(len(idents) - 1):
            # for enu, i0 in enumerate(idents[:-1]):
            # print(enu)
            next_message = include_progress_bar(enu, len(idents) - 1, 'error traces', next_message)
            for enu1 in np.arange(enu + 1, len(idents)):
                # for enu1, i1 in enumerate(idents[enu+1:]):
                if np.abs(med_ident_freq[enu] - med_ident_freq[enu1]) > max_df:
                    continue

                if idx_v[ident_v == idents[enu]][-1] < idx_v[ident_v == idents[enu1]][0]:
                    if np.abs(times[idx_v[ident_v == idents[enu]][-1]] - times[
                        idx_v[ident_v == idents[enu1]][0]]) > max_dt:
                        continue
                    delta_idx = idx_v[ident_v == idents[enu1]][0] - idx_v[ident_v == idents[enu]][-1]

                    # i0 before i1
                    # print('wuff')
                elif idx_v[ident_v == idents[enu1]][-1] < idx_v[ident_v == idents[enu]][0]:
                    if np.abs(times[idx_v[ident_v == idents[enu1]][-1]] - times[
                        idx_v[ident_v == idents[enu]][0]]) > max_dt:
                        continue
                    delta_idx = idx_v[ident_v == idents[enu]][0] - idx_v[ident_v == idents[enu1]][-1]
                    # i1 before i1
                else:
                    delta_idx = 0
                    # overlapping

                i0s.append(idents[enu])
                i1s.append(idents[enu1])
                di.append(delta_idx)
                df.append(np.abs(med_ident_freq[enu] - med_ident_freq[enu1]))

        i0s = np.array(i0s)
        i1s = np.array(i1s)
        di = np.array(di)
        rel_di = (di - np.min(di)) / (np.max(di) - np.min(di))
        df = np.array(df)
        rel_df = (df - np.min(df)) / (np.max(df) - np.min(df))

        error = rel_di + rel_df
        # embed()
        # quit()
        next_message = 0.00
        for enu, i in enumerate(np.argsort(error)):
            next_message = include_progress_bar(enu, len(error), 'connecting traces', next_message)

            # if df[enu] >= max_df:
            #     continue
            if len(np.intersect1d(idx_v[ident_v == i0s[i]], idx_v[ident_v == i1s[i]])) > max_overlap_n:
                continue
            else:
                ident_v[ident_v == i1s[i]] = i0s[i]
                i0s[i0s == i1s[i]] = i0s[i]
                i1s[i1s == i1s[i]] = i0s[i]

    return ident_v


def boltzmann(t, alpha=0.25, beta=0.0, x0=4, dx=0.85):
    """
    Calulates a boltzmann function.

    Parameters
    ----------
    t: array
        time vector.
    alpha: float
        max value of the boltzmann function.
    beta: float
        min value of the boltzmann function.
    x0: float
        time where the turning point of the boltzmann function occurs.
    dx: float
        slope of the boltzman function.

    Returns
    -------
    array
        boltzmann function of the given time array base on the other parameters given.
    """

    boltz = (alpha - beta) / (1. + np.exp(- (t - x0) / dx)) + beta
    return boltz


def estimate_error(a_error, f_error, t_error, a_error_distribution, f_error_distribution,
                   min_f_weight=0.2, max_f_weight=0.5, t_of_max_f_weight=5.):
    """
    Cost function estimating the error between two fish signals at two different times using realative frequency
    difference and relative signal amplitude difference on n electrodes (relative because the values are compared to a
    given distribution of these values). With increasing time difference between the signals the impact of frequency
    error increases and the influence of amplitude error decreases due to potential changes caused by fish movement.

    Parameters
    ----------
    a_error: float
        MSE of amplitude difference of two electric signals recorded with n electodes.
    f_error: float
        absolute frequency difference between two electric signals.
    t_error: float
        temporal difference between two measured signals in s.
    a_error_distribution: array
        distribution of possible MSE of the amplitudes between random data points in the dataset.
    f_error_distribution: array
        distribution of possible frequency differences between random data points in the dataset.
    min_f_weight: float
        minimum proportion of the frequency impact to the error value.
    max_f_weight: float
        maximum proportion of the frequency impact to the error value.
    t_of_max_f_weight: float
        error value between two electric signals at two time points.

    Returns
    -------
    float
        error value between two electric signals at two time points
    """

    a_weight = 2. / 3
    f_weight = 1. / 3

    a_e = a_weight * len(a_error_distribution[a_error_distribution < a_error]) / len(a_error_distribution)
    f_e = f_weight * boltzmann(f_error, alpha=1, beta=0, x0=.25, dx=.15)

    return [a_e, f_e, 0]


def freq_tracking_v4(fundamentals, signatures, times, freq_tolerance, n_channels, return_tmp_idenities=False,
                     ioi_fti=False, fig=False, ax=False, freq_lims=(400, 1200), ioi_field=False):
    """
    Sorting algorithm which sorts fundamental EOD frequnecies detected in consecutive powespectra of single or
    multielectrode recordings using frequency difference and frequnency-power amplitude difference on the electodes.

    Signal tracking and identity assiginment is accomplished in four steps:
    1) Extracting possible frequency and amplitude difference distributions.
    2) Esitmate relative error between possible datapoint connections (relative amplitude and frequency error based on
    frequency and amplitude error distribution).
    3) For a data window covering the EOD frequencies detected 10 seconds before the accual datapoint to assigne
    identify temporal identities based on overall error between two datapoints from smalles to largest.
    4) Form tight connections between datapoints where one datapoint is in the timestep that is currently of interest.

    Repeat these steps until the end of the recording.
    The temporal identities are only updated when the timestep of current interest reaches the middle (5 sec.) of the
    temporal identities. This is because no tight connection shall be made without checking the temporal identities.
    The temnporal identities are used to check if the potential connection from the timestep of interest to a certain
    datsapoint is the possibly best or if a connection in the futur will be better. If a future connection is better
    the thight connection is not made.

    Parameters
    ----------
    fundamentals: 2d-arraylike / list
        list of arrays of fundemantal EOD frequnecies. For each timestep/powerspectrum contains one list with the
        respectivly detected fundamental EOD frequnecies.
    signatures: 3d-arraylike / list
        same as fundamentals but for each value in fundamentals contains a list of powers of the respective frequency
        detected of n electrodes used.
    times: array
        respective time vector.
    freq_tolerance: float
        maximum frequency difference between two datapoints to be connected in Hz.
    n_channels: int
        number of channels/electodes used in the analysis.,
    return_tmp_idenities: bool
        only returne temporal identities at a certain timestep. Dependent on ioi_fti and only used to check algorithm.
    ioi_fti: int
        Index Of Interest For Temporal Identities: respective index in fund_v to calculate the temporal identities for.
    a_error_distribution: array
        possible amplitude error distributions for the dataset.
    f_error_distribution: array
        possible frequency error distribution for the dataset.
    fig: mpl.figure
        figure to plot the tracking progress life.
    ax: mpl.axis
        axis to plot the tracking progress life.
    freq_lims: double
        minimum/maximum frequency to be tracked.

    Returns
    -------
    fund_v: array
        flattened fundamtantals array containing all detected EOD frequencies in the recording.
    ident_v: array
        respective assigned identites throughout the tracking progress.
    idx_v: array
        respective index vectro impliing the time of the detected frequency.
    sign_v: 2d-array
        for each fundamental frequency the power of this frequency on the used electodes.
    a_error_distribution: array
        possible amplitude error distributions for the dataset.
    f_error_distribution: array
        possible frequency error distribution for the dataset.
    idx_of_origin_v: array
        for each assigned identity the index of the datapoint on which basis the assignement was made.
    """

    def clean_up(fund_v, ident_v, idx_v, times):
        """
        deletes/replaces with np.nan those identities only consisting from little data points and thus are tracking
        artefacts. Identities get deleted when the proportion of the trace (slope, ratio of detected datapoints, etc.)
        does not fit a real fish.

        Parameters
        ----------
        fund_v: array
            flattened fundamtantals array containing all detected EOD frequencies in the recording.
        ident_v: array
            respective assigned identites throughout the tracking progress.
        idx_v: array
            respective index vectro impliing the time of the detected frequency.
        times: array
            respective time vector.

        Returns
        -------
        ident_v: array
            cleaned up identities vector.

        """
        # print('clean up')
        for ident in np.unique(ident_v[~np.isnan(ident_v)]):
            if np.median(np.abs(np.diff(fund_v[ident_v == ident]))) >= 0.25:
                ident_v[ident_v == ident] = np.nan
                continue

            if len(ident_v[ident_v == ident]) <= 10:
                ident_v[ident_v == ident] = np.nan
                continue

        return ident_v

    def get_tmp_identities(i0_m, i1_m, error_cube, fund_v, idx_v, i, ioi_fti, dps, idx_comp_range,
                           sign_v, a_error_distribution, f_error_distribution, ioi_field=False, fig=False, ax=False):
        """
        extract temporal identities for a datasnippted of 2*index compare range of the original tracking algorithm.
        for each data point in the data window finds the best connection within index compare range and, thus connects
        the datapoints based on their minimal error value until no connections are left or possible anymore.

        Parameters
        ----------
        i0_m: 2d-array
            for consecutive timestamps contains for each the indices of the origin EOD frequencies.
        i1_m: 2d-array
            respectively contains the indices of the targen EOD frequencies, laying within index compare range.
        error_cube: 3d-array
            error values for each combination from i0_m and the respective indices in i1_m.
        fund_v: array
            flattened fundamtantals array containing all detected EOD frequencies in the recording.
        idx_v: array
            respective index vectro impliing the time of the detected frequency.
        i: int
            loop variable and current index of interest for the assignment of tight connections.
        ioi_fti: int
            index of interest for temporal identities.
        dps: float
            detections per second. 1. / 'temporal resolution of the tracking'
        idx_comp_range: int
            index compare range for the assignment of two data points to each other.

        Returns
        -------
        tmp_ident_v: array
            for each EOD frequencies within the index compare range for the current time step of interest contains the
            temporal identity.
        errors_to_v: array
            for each assigned temporal identity contains the error value based on which this connection was made.

        """
        next_tmp_identity = 0
        # embed()
        # quit()
        # mask_cube = [np.ones(np.shape(error_cube[n]), dtype=bool) for n in range(len(error_cube))]

        max_shape = np.max([np.shape(layer) for layer in error_cube[1:]], axis=0)
        cp_error_cube = np.full((len(error_cube) - 1, max_shape[0], max_shape[1]), np.nan)
        for enu, layer in enumerate(error_cube[1:]):
            cp_error_cube[enu, :np.shape(error_cube[enu + 1])[0], :np.shape(error_cube[enu + 1])[1]] = layer

        # try:
        #     tmp_ident_v = np.full(len(fund_v), np.nan)
        #     errors_to_v = np.full(len(fund_v), np.nan)
        # except:
        #     tmp_ident_v = np.zeros(len(fund_v)) / 0.
        #     errors_to_v = np.zeros(len(fund_v)) / 0.

        min_i0 = np.min(np.hstack(i0_m))
        max_i1 = np.max(np.hstack(i1_m))
        tmp_ident_v = np.full(max_i1 - min_i0 + 1, np.nan)
        errors_to_v = np.full(max_i1 - min_i0 + 1, np.nan)
        tmp_fund_v = fund_v[min_i0:max_i1 + 1]

        i0_m = np.array(i0_m) - min_i0
        i1_m = np.array(i1_m) - min_i0

        layers, idx0s, idx1s = np.unravel_index(np.argsort(cp_error_cube, axis=None), np.shape(cp_error_cube))
        made_connections = np.zeros(np.shape(cp_error_cube))
        not_made_connections = np.zeros(np.shape(cp_error_cube))
        not_made_connections[~np.isnan(cp_error_cube)] = 1
        # made_connections[~np.isnan(cp_error_cube)] = 0

        layers = layers + 1

        # embed()
        # quit()
        plotted = False
        if hasattr(ioi_field, '__len__'):
            # if np.min(ioi_field) in np.hstack(i0_m[:int(len(i0_m)/2)]):
            if ioi_field[2] in np.hstack(i0_m[idx_comp_range:idx_comp_range * 2]):
                # embed()
                # quit()

                c_i = np.unique(np.concatenate((np.hstack(i0_m), np.hstack(i1_m))))
                c_i = c_i[idx_v[c_i] - idx_v[np.min(c_i)] <= idx_comp_range * 3]
                ax[0].scatter(times[idx_v[c_i]], fund_v[c_i], color='grey', alpha=0.5)
                ax[1].scatter(times[idx_v[c_i]], fund_v[c_i], color='grey', alpha=0.5)
                ax[2].scatter(times[idx_v[c_i]], fund_v[c_i], color='grey', alpha=0.5)

                max_t = np.max(times[idx_v[c_i]])
                ax[0].set_xlim([max_t - 39., max_t + 1.])
                ax[1].set_xlim([max_t - 39., max_t + 1.])
                ax[2].set_xlim([max_t - 39., max_t + 1.])

                ax[3].set_xlim([max_t - 39., max_t + 1.])
                ax[4].set_xlim([max_t - 39., max_t + 1.])
                ax[5].set_xlim([max_t - 39., max_t + 1.])
                ax[6].set_xlim([max_t - 39., max_t + 1.])

                # ax[0].set_ylim([885, 935])
                # ax[1].set_ylim([885, 935])
                # ax[2].set_ylim([885, 935])
                ax[0].set_ylim([907, 930])
                ax[1].set_ylim([907, 930])
                ax[2].set_ylim([907, 930])
                ax[3].set_ylim([907, 930])
                ax[4].set_ylim([907, 930])
                ax[5].set_ylim([907, 930])
                ax[6].set_ylim([907, 930])

                plotted = True
                c = 0
                for layer, idx0, idx1 in zip(layers, idx0s, idx1s):
                    c += 1
                    if np.isnan(cp_error_cube[layer - 1, idx0, idx1]):
                        break
                plot_idxs = [200, 500, int(np.floor(c))]
                # fig, ax = plt.subplots(3, 1, figsize=(20./2.54, 36./2.54), facecolor='white')
                # embed()
        counter = 0
        error_line_at = []

        i_non_nan = len(cp_error_cube[layers - 1, idx0s, idx1s][~np.isnan(cp_error_cube[layers - 1, idx0s, idx1s])])

        for layer, idx0, idx1 in zip(layers[:i_non_nan], idx0s[:i_non_nan], idx1s[:i_non_nan]):

            counter += 1
            # print(counter, i_non_nan)
            if hasattr(ioi_field, '__len__') and plotted:
                if counter in plot_idxs:
                    if counter >= plot_idxs[0] and counter < plot_idxs[1]:
                        for ident in np.unique(tmp_ident_v[~np.isnan(tmp_ident_v)]):
                            ax[0].plot(times[idx_v[tmp_ident_v == ident]], fund_v[tmp_ident_v == ident], color='k',
                                       marker='.', alpha=0.7)
                        error_line_at.append(last_error)
                    elif counter >= plot_idxs[1] and counter < plot_idxs[2]:
                        for ident in np.unique(tmp_ident_v[~np.isnan(tmp_ident_v)]):
                            ax[1].plot(times[idx_v[tmp_ident_v == ident]], fund_v[tmp_ident_v == ident], color='k',
                                       marker='.', alpha=0.7)
                        error_line_at.append(last_error)
                    else:
                        for ident in np.unique(tmp_ident_v[~np.isnan(tmp_ident_v)]):
                            ax[2].plot(times[idx_v[tmp_ident_v == ident]], fund_v[tmp_ident_v == ident], color='k',
                                       marker='.', alpha=0.7)
                        error_line_at.append(last_error)

                        # ax[5].plot(times[idx_v[tmp_ident_v == ident]], fund_v[tmp_ident_v == ident], color='k',
                        #            marker='.', alpha=0.7)

            if np.isnan(cp_error_cube[layer - 1, idx0, idx1]):
                break

            # _____ some control functions _____ ###
            if not ioi_fti:
                if idx_v[i1_m[layer][idx1]] - i > idx_comp_range * 3:
                    continue
            else:
                if idx_v[i1_m[layer][idx1]] - idx_v[ioi_fti] > idx_comp_range * 3:
                    continue

            # if fund_v[i0_m[layer][idx0]] > fund_v[i1_m[layer][idx1]]:
            #     if 1. * np.abs(fund_v[i0_m[layer][idx0]] - fund_v[i1_m[layer][idx1]]) / ((idx_v[i1_m[layer][idx1]] - idx_v[i0_m[layer][idx0]]) / dps) > 2.:
            #         continue
            # else:
            #     if 1. * np.abs(fund_v[i0_m[layer][idx0]] - fund_v[i1_m[layer][idx1]]) / ((idx_v[i1_m[layer][idx1]] - idx_v[i0_m[layer][idx0]]) / dps) > 2.:
            #         continue

            # ToDo:check if at least one direct neighbour of new connected has small delta f

            if np.isnan(tmp_ident_v[i0_m[layer][idx0]]):
                if np.isnan(tmp_ident_v[i1_m[layer][idx1]]):
                    if np.abs(tmp_fund_v[i0_m[layer][idx0]] - tmp_fund_v[i1_m[layer][idx1]]) > 0.5:
                        continue

                    tmp_ident_v[i0_m[layer][idx0]] = next_tmp_identity
                    tmp_ident_v[i1_m[layer][idx1]] = next_tmp_identity
                    errors_to_v[i1_m[layer][idx1]] = cp_error_cube[layer - 1][idx0, idx1]
                    # errors_to_v[i0_m[layer][idx0]] = error_cube[layer][idx0, idx1]
                    not_made_connections[layer - 1, idx0, idx1] = 0
                    made_connections[layer - 1, idx0, idx1] = 1

                    # errors_to_v[(tmp_ident_v == tmp_ident_v[i1_m[layer][idx1]]) & (np.isnan(errors_to_v))] = error_cube[layer][idx0, idx1]
                    next_tmp_identity += 1
                else:

                    mask = np.arange(len(tmp_ident_v))[tmp_ident_v == tmp_ident_v[i1_m[layer][idx1]]]
                    # if idx_v[i0_m[layer][idx0]] in idx_v[tmp_ident_v == tmp_ident_v[i1_m[layer][idx1]]]:
                    if idx_v[i0_m[layer][idx0]] in idx_v[mask]:
                        continue

                    same_id_idx = np.arange(len(tmp_ident_v))[tmp_ident_v == tmp_ident_v[i1_m[layer][idx1]]]
                    f_after = tmp_fund_v[same_id_idx[same_id_idx > i0_m[layer][idx0]]]
                    f_before = tmp_fund_v[same_id_idx[same_id_idx < i0_m[layer][idx0]]]
                    compare_freqs = []
                    if len(f_after) > 0:
                        compare_freqs.append(f_after[0])
                    if len(f_before) > 0:
                        compare_freqs.append(f_before[-1])
                    if len(compare_freqs) == 0:
                        continue
                    else:
                        if np.all(np.abs(np.array(compare_freqs) - tmp_fund_v[i0_m[layer][idx0]]) > 0.5):
                            continue

                    tmp_ident_v[i0_m[layer][idx0]] = tmp_ident_v[i1_m[layer][idx1]]

                    errors_to_v[i1_m[layer][idx1]] = cp_error_cube[layer - 1][idx0, idx1]
                    not_made_connections[layer - 1, idx0, idx1] = 0
                    made_connections[layer - 1, idx0, idx1] = 1

                    # errors_to_v[(tmp_ident_v == tmp_ident_v[i1_m[layer][idx1]]) & (np.isnan(errors_to_v))] = error_cube[layer][idx0, idx1]
                    # errors_to_v[tmp_ident_v == tmp_ident_v[i1_m[layer][idx1]]][0] = np.nan

            else:
                if np.isnan(tmp_ident_v[i1_m[layer][idx1]]):
                    mask = np.arange(len(tmp_ident_v))[tmp_ident_v == tmp_ident_v[i0_m[layer][idx0]]]
                    # if idx_v[i1_m[layer][idx1]] in idx_v[tmp_ident_v == tmp_ident_v[i0_m[layer][idx0]]]:
                    if idx_v[i1_m[layer][idx1]] in idx_v[mask]:
                        continue

                    same_id_idx = np.arange(len(tmp_ident_v))[tmp_ident_v == tmp_ident_v[i0_m[layer][idx0]]]
                    f_after = tmp_fund_v[same_id_idx[same_id_idx > i1_m[layer][idx1]]]
                    f_before = tmp_fund_v[same_id_idx[same_id_idx < i1_m[layer][idx1]]]
                    compare_freqs = []
                    if len(f_after) > 0:
                        compare_freqs.append(f_after[0])
                    if len(f_before) > 0:
                        compare_freqs.append(f_before[-1])
                    if len(compare_freqs) == 0:
                        continue
                    else:
                        if np.all(np.abs(np.array(compare_freqs) - tmp_fund_v[i1_m[layer][idx1]]) > 0.5):
                            continue

                    tmp_ident_v[i1_m[layer][idx1]] = tmp_ident_v[i0_m[layer][idx0]]
                    errors_to_v[i1_m[layer][idx1]] = cp_error_cube[layer - 1][idx0, idx1]
                    not_made_connections[layer - 1, idx0, idx1] = 0
                    made_connections[layer - 1, idx0, idx1] = 1

                    # errors_to_v[(tmp_ident_v == tmp_ident_v[i1_m[layer][idx1]]) & (np.isnan(errors_to_v))] = error_cube[layer][idx0, idx1]
                    # errors_to_v[tmp_ident_v == tmp_ident_v[i1_m[layer][idx1]]][0] = np.nan

                else:
                    if tmp_ident_v[i0_m[layer][idx0]] == tmp_ident_v[i1_m[layer][idx1]]:
                        if np.isnan(errors_to_v[i1_m[layer][idx1]]):
                            errors_to_v[i1_m[layer][idx1]] = cp_error_cube[layer - 1][idx0, idx1]
                        continue

                    # idxs_i0 = idx_v[tmp_ident_v == tmp_ident_v[i0_m[layer][idx0]]]
                    mask = np.arange(len(tmp_ident_v))[tmp_ident_v == tmp_ident_v[i0_m[layer][idx0]]]
                    idxs_i0 = idx_v[mask + min_i0]
                    # idxs_i1 = idx_v[tmp_ident_v == tmp_ident_v[i1_m[layer][idx1]]]
                    mask = np.arange(len(tmp_ident_v))[tmp_ident_v == tmp_ident_v[i1_m[layer][idx1]]]
                    idxs_i1 = idx_v[mask + min_i0]

                    if np.any(np.diff(sorted(np.concatenate((idxs_i0, idxs_i1)))) == 0):
                        continue
                    tmp_ident_v[tmp_ident_v == tmp_ident_v[i0_m[layer][idx0]]] = tmp_ident_v[i1_m[layer][idx1]]

                    if np.isnan(errors_to_v[i1_m[layer][idx1]]):
                        errors_to_v[i1_m[layer][idx1]] = cp_error_cube[layer - 1][idx0, idx1]
                        not_made_connections[layer - 1, idx0, idx1] = 0
                        made_connections[layer - 1, idx0, idx1] = 1

            last_error = cp_error_cube[layer - 1, idx0, idx1]
        # if hasattr(ioi_field, '__len__') and plotted:
        #     made_connection_dist = cp_error_cube[np.array(made_connections, dtype=bool)]
        #     n, h = np.histogram(made_connection_dist)
        #     ax[6].plot(h[:-1], n, color='k')
        #     ax[7].plot(h[:-1], n, color='k')
        #     ax[8].plot(h[:-1], n, color='k')
        #     y_lims = ax[6].get_ylim()
        #     print(error_line_at)
        #     ax[6].semilogx([error_line_at[0], error_line_at[0]], [y_lims[0], y_lims[1]], color='red')
        #     ax[7].semilogx([error_line_at[1], error_line_at[1]], [y_lims[0], y_lims[1]], color='red')
        #     ax[8].semilogx([error_line_at[2], error_line_at[2]], [y_lims[0], y_lims[1]], color='red')
        #     ax[6].set_ylim([y_lims[0], y_lims[1]])
        #     ax[7].set_ylim([y_lims[0], y_lims[1]])
        #     ax[8].set_ylim([y_lims[0], y_lims[1]])
        # embed()
        # quit()
        # if hasattr(ioi_field, '__len__') and plotted:
        # xl = ax[2].get_xlim()
        # for x in ax:
        #     x.set_ylim([885, 935])
        # x.set_xlim(xl)
        # plt.show(fig)
        # return tmp_ident_v, errors_to_v, fig, ax

        if len(error_line_at) == 3:
            total_i0v = np.hstack(i0_m)
            # total_i0v = total_i0v[~np.isnan(tmp_ident_v[total_i0v])]
            total_i1v = np.unique(np.hstack(i1_m))
            # total_i1v = total_i1v[~np.isnan(tmp_ident_v[total_i1v])]

            total_error_m = np.full((len(total_i0v), len(total_i1v)), np.nan)

            # ToDo:calculate full error matrix without nans ?!
            next_message = 0.0
            for i in range(np.shape(total_error_m)[0]):
                next_message = include_progress_bar(i, np.shape(total_error_m)[0], 'dada', next_message=next_message)
                for j in range(np.shape(total_error_m)[1]):
                    f_error = np.abs(fund_v[total_i0v[i]] - fund_v[total_i1v[j]])
                    a_error = np.sqrt(np.sum((sign_v[total_i0v[i]] - sign_v[total_i1v[j]]) ** 2))
                    e = estimate_error(a_error, f_error, 0, a_error_distribution, f_error_distribution)
                    total_error_m[i, j] = np.sum(e)

            # counter = 0
            # total_error_m2 = np.full((len(total_i0v), len(total_i1v)), np.nan)
            # for i in range(len(i0_m)):
            #     try:
            #         mask_bool = np.array([a in i1_m[i] for a in total_i1v])
            #         mask = np.arange(np.shape(total_error_m2)[1])[mask_bool]
            #     except:
            #         print('nope')
            #         embed()
            #         quit()
            #     for j in range(len(i0_m[i])):
            #         total_error_m2[counter][mask] = error_cube[i][j]
            #         counter += 1

            nonnan_i0v = np.array(~np.isnan(tmp_ident_v[total_i0v]))
            nonnan_i1v = np.array(~np.isnan(tmp_ident_v[total_i1v]))

            total_error_m = total_error_m[nonnan_i0v]
            # total_error_m2 = total_error_m2[nonnan_i0v]
            total_error_m = total_error_m[:, nonnan_i1v]
            # total_error_m2 = total_error_m2[:, nonnan_i1v]

            total_i0v = total_i0v[nonnan_i0v]
            total_i1v = total_i1v[nonnan_i1v]
            # plt.close()

            masked_array = np.ma.array(total_error_m, mask=np.isnan(total_error_m))
            cmap = plt.cm.jet
            cmap.set_bad('white', 1.)

            sorted_total_error_m = total_error_m[np.argsort(tmp_ident_v[total_i0v])]
            # sorted_total_error_m2 = total_error_m2[np.argsort(tmp_ident_v[total_i0v])]
            sorted_total_error_m = sorted_total_error_m[:, np.argsort(tmp_ident_v[total_i1v])]
            # sorted_total_error_m2 = sorted_total_error_m2[:, np.argsort(tmp_ident_v[total_i1v])]

            masked_array2 = np.ma.array(sorted_total_error_m, mask=np.isnan(sorted_total_error_m))
            # cmap2 = plt.cm.jet
            # cmap2.set_bad('white', 1.)

            # masked_array3 = np.ma.array(sorted_total_error_m2, mask=np.isnan(sorted_total_error_m2))
            # cmap3 = plt.cm.jet
            # cmap3.set_bad('white', 1.)

            fig = plt.figure(facecolor='white', figsize=(20. / 2.54, 12. / 2.54))
            ax0 = fig.add_axes([.05, .05, .4, .9])
            ax1 = fig.add_axes([.55, .05, .4, .9])
            ax0.set_ylabel('origin signal')
            ax0.set_xticks([])
            ax0.set_yticks([])
            ax0.set_xlabel('target signal')
            ax1.set_xlabel('target signal')
            ax1.set_xticks([])
            ax1.set_yticks([])

            sorted_id0 = tmp_ident_v[total_i0v][np.argsort(tmp_ident_v[total_i0v])]
            y_tick = []
            for ident in np.unique(sorted_id0):
                y_tick.append(np.median(np.arange(len(sorted_id0))[sorted_id0 == ident]))
            ax1.set_yticks(np.array(y_tick)[1::2])
            ax1.set_yticklabels(np.arange(len(y_tick))[1::2] + 1)
            # ax1.set_yticklabels(['fish #%.0f' % i+1 for i in np.arange(len(y_tick))])

            sorted_id1 = tmp_ident_v[total_i1v][np.argsort(tmp_ident_v[total_i1v])]
            x_tick = []
            for ident in np.unique(sorted_id0):
                x_tick.append(np.median(np.arange(len(sorted_id1))[sorted_id1 == ident]))
            ax1.set_xticks(np.array(x_tick)[1::2])
            ax1.set_xticklabels(np.arange(len(y_tick))[1::2] + 1)
            # ax1left = fig.add_axes([.5, .3, .4, .6])
            # ax1bottom = fig.add_axes([.55, .25, .4, .65])
            # ax2 = fig.add_axes([.1, .1, .8, .05])

            ax0.imshow(masked_array, cmap=cmap)
            cax = ax1.imshow(masked_array2, cmap=cmap)
            cbar = fig.colorbar(cax, ax=[ax0, ax1], orientation='horizontal')
            cbar.ax.set_xlabel('signal error')
            ax1.xaxis.tick_top()
            # ax2.colorbar()
            # ax[2].imshow(masked_array3, cmap = cmap3)

            # embed()
            # quit()

        tmp_ident_v_ret = np.full(len(fund_v), np.nan)
        tmp_ident_v_ret[min_i0:max_i1 + 1] = tmp_ident_v
        # embed()
        # quit()
        return tmp_ident_v_ret, errors_to_v, plotted

    def get_a_and_f_error_dist2(fund_v, idx_v, sign_v, start_idx, idx_comp_range, freq_lims, low_freq_th, high_freq_th,
                                freq_tolerance):
        f_error_distribution = []
        a_error_distribution = []

        # next_message = 0.0
        for i in range(start_idx, int(start_idx + idx_comp_range * 3)):
            # next_message = include_progress_bar(i - start_idx, int(idx_comp_range * 2), 'error dist init', next_message)
            i0_v = np.arange(len(idx_v))[
                (idx_v == i) & (fund_v >= freq_lims[0]) & (fund_v <= freq_lims[1])]  # indices of fundamtenals to assign
            i1_v = np.arange(len(idx_v))[
                (idx_v > i) & (idx_v <= (i + int(idx_comp_range))) & (fund_v >= freq_lims[0]) & (
                            fund_v <= freq_lims[1])]  # indices of possible targets

            if len(i0_v) == 0 or len(i1_v) == 0:  # if nothing to assign or no targets continue
                continue

            for enu0 in range(len(fund_v[i0_v])):
                if fund_v[i0_v[enu0]] < low_freq_th or fund_v[i0_v[enu0]] > high_freq_th:
                    continue
                for enu1 in range(len(fund_v[i1_v])):
                    if fund_v[i1_v[enu1]] < low_freq_th or fund_v[i1_v[enu1]] > high_freq_th:
                        continue
                    if np.abs(fund_v[i0_v[enu0]] - fund_v[i1_v[enu1]]) >= freq_tolerance:  # freq difference to high
                        continue
                    a_error_distribution.append(np.sqrt(np.sum(
                        [(sign_v[i0_v[enu0]][k] - sign_v[i1_v[enu1]][k]) ** 2 for k in
                         range(len(sign_v[i0_v[enu0]]))])))
                    f_error_distribution.append(np.abs(fund_v[i0_v[enu0]] - fund_v[i1_v[enu1]]))

        return np.array(a_error_distribution), np.array(f_error_distribution)

    colors = ['#BA2D22', '#53379B', '#F47F17', '#3673A4', '#AAB71B', '#DC143C', '#1E90FF']

    # _____ parameters and vectors _____ ###
    detection_time_diff = times[1] - times[0]
    dps = 1. / detection_time_diff
    fund_v = np.hstack(fundamentals)
    ident_v = np.full(len(fund_v), np.nan)  # fish identities of frequencies
    idx_of_origin_v = np.full(len(fund_v), np.nan)


    idx_v = []  # temportal indices
    sign_v = []  # power of fundamentals on all electrodes
    for enu, funds in enumerate(fundamentals):
        idx_v.extend(np.ones(len(funds)) * enu)
        sign_v.extend(signatures[enu])
    idx_v = np.array(idx_v, dtype=int)
    sign_v = np.array(sign_v)

    original_sign_v = sign_v
    if np.shape(sign_v)[1] > 2:
        sign_v = (sign_v - np.min(sign_v, axis=1).reshape(len(sign_v), 1)) / (
                    np.max(sign_v, axis=1).reshape(len(sign_v), 1) - np.min(sign_v, axis=1).reshape(len(sign_v), 1))

    idx_comp_range = int(np.floor(dps * 10.))  # maximum compare range backwards for amplitude signature comparison
    low_freq_th = 400.  # min. frequency tracked
    high_freq_th = 1050.  # max. frequency tracked

    error_cube = []  # [fundamental_list_idx, freqs_to_assign, target_freqs]
    i0_m = []
    i1_m = []

    next_message = 0.
    start_idx = 0 if not ioi_fti else idx_v[ioi_fti]  # Index Of Interest for temporal identities

    a_error_distribution, f_error_distribution = get_a_and_f_error_dist2(fund_v, idx_v, sign_v, start_idx,
                                                                         idx_comp_range, freq_lims, low_freq_th,
                                                                         high_freq_th, freq_tolerance)

    for i in range(start_idx, int(start_idx + idx_comp_range * 3)):

        next_message = include_progress_bar(i - start_idx, int(idx_comp_range * 2), 'initial error cube', next_message)
        i0_v = np.arange(len(idx_v))[
            (idx_v == i) & (fund_v >= freq_lims[0]) & (fund_v <= freq_lims[1])]  # indices of fundamtenals to assign
        i1_v = np.arange(len(idx_v))[(idx_v > i) & (idx_v <= (i + int(idx_comp_range))) & (fund_v >= freq_lims[0]) & (
                    fund_v <= freq_lims[1])]  # indices of possible targets

        i0_m.append(i0_v)
        i1_m.append(i1_v)

        if len(i0_v) == 0 or len(i1_v) == 0:  # if nothing to assign or no targets continue
            error_cube.append(np.array([[]]))
            continue
        try:
            error_matrix = np.full((len(i0_v), len(i1_v)), np.nan)
        except:
            error_matrix = np.zeros((len(i0_v), len(i1_v))) / 0.

        for enu0 in range(len(fund_v[i0_v])):
            if fund_v[i0_v[enu0]] < low_freq_th or fund_v[
                i0_v[enu0]] > high_freq_th:  # freq to assigne out of tracking range
                continue
            for enu1 in range(len(fund_v[i1_v])):
                if fund_v[i1_v[enu1]] < low_freq_th or fund_v[
                    i1_v[enu1]] > high_freq_th:  # target freq out of tracking range
                    continue
                if np.abs(fund_v[i0_v[enu0]] - fund_v[i1_v[enu1]]) >= freq_tolerance:  # freq difference to high
                    continue

                a_error = np.sqrt(
                    np.sum([(sign_v[i0_v[enu0]][j] - sign_v[i1_v[enu1]][j]) ** 2 for j in range(n_channels)]))
                f_error = np.abs(fund_v[i0_v[enu0]] - fund_v[i1_v[enu1]])
                t_error = 1. * np.abs(idx_v[i0_v[enu0]] - idx_v[i1_v[enu1]]) / dps

                error = estimate_error(a_error, f_error, t_error, a_error_distribution, f_error_distribution)
                error_matrix[enu0, enu1] = np.sum(error)
        error_cube.append(error_matrix)

    cube_app_idx = len(error_cube)

    next_identity = 0
    next_message = 0.00
    next_cleanup = int(idx_comp_range * 120)
    plotted = False
    plotting_finished = False

    # embed()
    # quit()

    step_plot = False
    if step_plot:
        # plt.close()
        # fig, ax = plt.subplots(figsize=(20./ 2.54, 12/2.54), facecolor='white')
        tmp_handle = []
        handle = []
        # plt.show(block=False)
    t0 = time.time()
    t00 = time.time()

    for enu, i in enumerate(np.arange(len(fundamentals))):
        # print(enu)
        if time.time() - t00 >= 300:
            print('%.2f speed' % (((i - start_idx) / dps) / (time.time() - t0)))
            t00 = time.time()
        # print(i)
        if i >= next_cleanup:  # clean up every 10 minutes
            ident_v = clean_up(fund_v, ident_v, idx_v, times)
            next_cleanup += int(idx_comp_range * 120)

        if not return_tmp_idenities:
            next_message = include_progress_bar(i, len(fundamentals), 'tracking', next_message)  # feedback

        if enu % idx_comp_range == 0:
            # t0 = time.time()
            # print('\ndist')
            a_error_distribution, f_error_distribution = get_a_and_f_error_dist2(fund_v, idx_v, sign_v, start_idx,
                                                                                 idx_comp_range, freq_lims, low_freq_th,
                                                                                 high_freq_th, freq_tolerance)
            # print('\ntmp idents')
            tmp_ident_v, errors_to_v, plotted = get_tmp_identities(i0_m, i1_m, error_cube, fund_v, idx_v, i, ioi_fti,
                                                                   dps, idx_comp_range, sign_v, a_error_distribution,
                                                                   f_error_distribution, ioi_field, fig, ax)

            if step_plot:
                for h in tmp_handle:
                    h.remove()
                tmp_handle = []
                for ident in np.unique(tmp_ident_v[~np.isnan(tmp_ident_v)]):
                    h, = ax.plot(times[idx_v[tmp_ident_v == ident]], fund_v[tmp_ident_v == ident], color='grey', lw=4,
                                 zorder=0)
                    tmp_handle.append(h)
                # ax.set_title('initial tmp_idents')
                # plt.draw()
                # plt.waitforbuttonpress()
                fig.canvas.draw()
                # plt.pause(0.5)

            if enu == 0:
                for ident in np.unique(tmp_ident_v[~np.isnan(tmp_ident_v)]):
                    ident_v[(tmp_ident_v == ident) & (idx_v <= i + idx_comp_range)] = next_identity
                    next_identity += 1
                    # h, = ax.plot(idx_v[tmp_ident_v == ident], fund_v[tmp_ident_v == ident], color='grey', lw = 4)
                    # tmp_handle.append(h)
                if step_plot:
                    for ident in np.unique(ident_v[~np.isnan(ident_v)]):
                        h, = ax.plot(times[idx_v[ident_v == ident]], fund_v[ident_v == ident], marker='.', zorder=1)
                        handle.append(h)
                    # ax.set_title('initial idents')
                    # plt.draw()
                    # plt.waitforbuttonpress()
                    fig.canvas.draw()

            else:
                tmptimes = times[idx_v[~np.isnan(tmp_ident_v)]]
                if step_plot:
                    ax.set_xlim(np.min(tmptimes) - 10, np.max(tmptimes) + 10)
                    fig.canvas.draw()

            max_shape = np.max([np.shape(layer) for layer in error_cube], axis=0)
            cp_error_cube = np.full((len(error_cube), max_shape[0], max_shape[1]), np.nan)
            for enu, layer in enumerate(error_cube):
                cp_error_cube[enu, :np.shape(error_cube[enu])[0], :np.shape(error_cube[enu])[1]] = layer

            layers, idx0s, idx1s = np.unravel_index(np.argsort(cp_error_cube[:idx_comp_range], axis=None),
                                                    np.shape(cp_error_cube[:idx_comp_range]))
            # layers, idx0s, idx1s = np.unravel_index(np.argsort(cp_error_cube[:idx_comp_range-1], axis=None), np.shape(cp_error_cube[:idx_comp_range-1]))

            if plotted:
                for ident in np.unique(ident_v[~np.isnan(ident_v)]):
                    c = colors[int(ident % len(colors))]
                    # ax[3].plot(times[idx_v[ident_v == ident]], fund_v[ident_v == ident], marker='.', color=c, zorder = 2)
                    ax[4].plot(times[idx_v[ident_v == ident]], fund_v[ident_v == ident], marker='.', color=c, zorder=2)
                    ax[5].plot(times[idx_v[ident_v == ident]], fund_v[ident_v == ident], marker='.', color=c, zorder=2)
                    ax[9].plot(times[idx_v[ident_v == ident]], fund_v[ident_v == ident], marker='.', color=c, zorder=2)
                    ax[10].plot(times[idx_v[ident_v == ident]], fund_v[ident_v == ident], marker='.', color=c, zorder=2)
                    # ax[6].plot(times[idx_v[ident_v == ident]], fund_v[ident_v == ident], marker='.', color=c, zorder = 2)

                for ident in np.unique(tmp_ident_v[~np.isnan(tmp_ident_v)]):
                    c = colors[int(ident % len(colors))]

                    # ax[3].plot(times[idx_v[tmp_ident_v == ident]], fund_v[tmp_ident_v == ident], color='grey', lw = 5, zorder=1)
                    ax[7].set_xlim([ax[3].get_xlim()[0], ax[3].get_xlim()[1]])
                    ax[7].set_ylim([0, 1])
                    ax[8].set_xlim([ax[3].get_xlim()[0], ax[3].get_xlim()[1]])
                    ax[8].set_ylim([0, 1])

                    ax[7].fill_between([times[i], times[i + idx_comp_range * 3]], [.925, .925], [.95, .95],
                                       color='grey', alpha=0.7)
                    ax[7].fill_between([times[i + idx_comp_range], times[i + idx_comp_range * 2]], [.925, .925],
                                       [.95, .95], color='k')

                    ax[7].plot([times[i], times[i]], [.55, 0.925], color='k', lw=.5)
                    ax[7].plot([times[i + idx_comp_range], times[i + idx_comp_range]], [.55, 0.925], color='k', lw=.5)
                    ax[7].plot([times[i + idx_comp_range * 2], times[i + idx_comp_range * 2]], [.55, 0.925], color='k',
                               lw=.5)
                    ax[7].plot([times[i + idx_comp_range * 3], times[i + idx_comp_range * 3]], [.55, 0.925], color='k',
                               lw=.5)
                    ax[7].plot([times[i], times[i]], [.1, 0.45], color='k', lw=.5)
                    ax[7].plot([times[i + idx_comp_range], times[i + idx_comp_range]], [.1, 0.45], color='k', lw=.5)
                    ax[7].plot([times[i + idx_comp_range * 2], times[i + idx_comp_range * 2]], [.1, 0.45], color='k',
                               lw=.5)
                    ax[7].plot([times[i + idx_comp_range * 3], times[i + idx_comp_range * 3]], [.1, 0.45], color='k',
                               lw=.5)

                    ax[7].spines['right'].set_visible(False)
                    ax[7].spines['left'].set_visible(False)
                    ax[7].spines['top'].set_visible(False)
                    ax[7].spines['bottom'].set_visible(False)
                    ax[7].set_yticks([])
                    ax[7].set_xticks([])
                    ax[7].patch.set_alpha(.0)

                    ax[8].fill_between([times[i], times[i + idx_comp_range * 3]], [.925, .925], [.95, .95],
                                       color='grey', alpha=0.7)
                    ax[8].fill_between([times[i + idx_comp_range], times[i + idx_comp_range * 2]], [.925, .925],
                                       [.95, .95], color='k')

                    ax[8].plot([times[i], times[i]], [.55, 0.925], color='k', lw=.5)
                    ax[8].plot([times[i + idx_comp_range], times[i + idx_comp_range]], [.55, 0.925], color='k', lw=.5)
                    ax[8].plot([times[i + idx_comp_range * 2], times[i + idx_comp_range * 2]], [.55, 0.925], color='k',
                               lw=.5)
                    ax[8].plot([times[i + idx_comp_range * 3], times[i + idx_comp_range * 3]], [.55, 0.925], color='k',
                               lw=.5)
                    ax[8].plot([times[i], times[i]], [.1, 0.45], color='k', lw=.5)
                    ax[8].plot([times[i + idx_comp_range], times[i + idx_comp_range]], [.1, 0.45], color='k', lw=.5)
                    ax[8].plot([times[i + idx_comp_range * 2], times[i + idx_comp_range * 2]], [.1, 0.45], color='k',
                               lw=.5)
                    ax[8].plot([times[i + idx_comp_range * 3], times[i + idx_comp_range * 3]], [.1, 0.45], color='k',
                               lw=.5)

                    ax[8].spines['right'].set_visible(False)
                    ax[8].spines['left'].set_visible(False)
                    ax[8].spines['top'].set_visible(False)
                    ax[8].spines['bottom'].set_visible(False)
                    ax[8].set_yticks([])
                    ax[8].set_xticks([])
                    ax[8].patch.set_alpha(.0)

                    # ax[3].plot([times[i + idx_comp_range], times[i + idx_comp_range]], [400, 1100], '--', lw = 2, color='k')
                    # ax[3].plot([times[i + idx_comp_range * 2], times[i + idx_comp_range *2]], [400, 1100], '--', lw=2, color='k')

                    # ax[4].plot([times[i + idx_comp_range], times[i + idx_comp_range]], [400, 1100], '--', lw=2, color='k')
                    # ax[4].plot([times[i + idx_comp_range * 2], times[i + idx_comp_range * 2]], [400, 1100], '--', lw=2, color='k')

                    ax[3].plot(times[idx_v[(tmp_ident_v == ident)]], fund_v[(tmp_ident_v == ident)], lw=5, color=c,
                               zorder=1, alpha=.4)
                    # ax[3].plot(times[idx_v[(tmp_ident_v == ident) & (idx_v > i + idx_comp_range) & (idx_v <= i + idx_comp_range*2)]], fund_v[(tmp_ident_v == ident) & (idx_v > i + idx_comp_range) & (idx_v <= i + idx_comp_range*2)], '.', color=c, zorder=2)
                    ax[3].plot(times[idx_v[
                        (tmp_ident_v == ident) & (idx_v > i + idx_comp_range) & (idx_v <= i + idx_comp_range * 2)]],
                               fund_v[(tmp_ident_v == ident) & (idx_v > i + idx_comp_range) & (
                                           idx_v <= i + idx_comp_range * 2)],
                               lw=5, color=c, zorder=2)

                    ax[4].plot(times[idx_v[(tmp_ident_v == ident)]], fund_v[(tmp_ident_v == ident)], lw=5, color=c,
                               zorder=1, alpha=.4)
                    # ax[4].plot(times[idx_v[(tmp_ident_v == ident) & (idx_v > i + idx_comp_range) & (idx_v <= i + idx_comp_range*2)]], fund_v[(tmp_ident_v == ident) & (idx_v > i + idx_comp_range) & (idx_v <= i + idx_comp_range*2)], '.', color=c, zorder=2)
                    ax[4].plot(times[idx_v[
                        (tmp_ident_v == ident) & (idx_v > i + idx_comp_range) & (idx_v <= i + idx_comp_range * 2)]],
                               fund_v[(tmp_ident_v == ident) & (idx_v > i + idx_comp_range) & (
                                           idx_v <= i + idx_comp_range * 2)],
                               lw=5, color=c, zorder=2)

                    ax[5].plot(times[idx_v[(tmp_ident_v == ident)]], fund_v[(tmp_ident_v == ident)], lw=5, color=c,
                               zorder=1, alpha=.4)
                    # ax[5].plot(times[idx_v[(tmp_ident_v == ident) & (idx_v > i + idx_comp_range) & (idx_v <= i + idx_comp_range*2)]], fund_v[(tmp_ident_v == ident) & (idx_v > i + idx_comp_range) & (idx_v <= i + idx_comp_range*2)], '.', color=c, zorder=2)
                    ax[5].plot(times[idx_v[
                        (tmp_ident_v == ident) & (idx_v > i + idx_comp_range) & (idx_v <= i + idx_comp_range * 2)]],
                               fund_v[(tmp_ident_v == ident) & (idx_v > i + idx_comp_range) & (
                                           idx_v <= i + idx_comp_range * 2)],
                               lw=5, color=c, zorder=2)

                    # ax[6].plot(times[idx_v[(tmp_ident_v == ident)]], fund_v[(tmp_ident_v == ident)], lw=5, color=c, zorder=1, alpha=.6)
                    # ax[6].plot(times[idx_v[(tmp_ident_v == ident) & (idx_v > i + idx_comp_range) & (idx_v <= i + idx_comp_range*2)]], fund_v[(tmp_ident_v == ident) & (idx_v > i + idx_comp_range) & (idx_v <= i + idx_comp_range*2)], '.', color=c, zorder=2)

                # embed()
                # quit()

            #######
            i_non_nan = len(cp_error_cube[layers - 1, idx0s, idx1s][~np.isnan(cp_error_cube[layers - 1, idx0s, idx1s])])
            min_i0 = np.min(np.hstack(i0_m))
            max_i1 = np.max(np.hstack(i1_m))

            p_ident_v = ident_v[min_i0:max_i1 + 1]
            p_tmp_ident_v = tmp_ident_v[min_i0:max_i1 + 1]
            p_idx_v = idx_v[min_i0:max_i1 + 1]
            p_fund_v = fund_v[min_i0:max_i1 + 1]

            p_i0_m = np.array(i0_m) - min_i0
            p_i1_m = np.array(i1_m) - min_i0

            ################################################################################

            # poss_ident_v = p_ident_v[~np.isnan(p_ident_v)]
            # poss_tmp_ident_v = p_tmp_ident_v[~np.isnan(p_tmp_ident_v)]
            #
            # max_len_i0_m = np.max([len(i0_m[i]) for i in range(len(i0_m))])
            #
            # squared_i0_m = np.full((len(i0_m), max_len_i0_m), np.nan)
            # for enu, l in enumerate(i0_m):
            #     squared_i0_m[enu][:len(l)] = l
            #
            # max_len_i1_m = np.max([len(i1_m[i]) for i in range(len(i1_m))])
            # squared_i1_m = np.full((len(i1_m), max_len_i1_m), np.nan)
            # for enu, l in enumerate(i1_m):
            #     squared_i1_m[enu][:len(l)] = l
            #
            # embed()
            # quit()
            #
            # connect_ident = np.full((len(poss_ident_v), len(poss_tmp_ident_v)), np.nan)
            # for i in range(len(poss_ident_v)):
            #     for j in range(len(poss_tmp_ident_v)):
            #         try:
            #             errors_of_interest = cp_error_cube[layers, idx0s, idx1s][
            #                 (p_ident_v[np.array(squared_i0_m, dtype=int)[layers, idx0s]] == poss_ident_v[i]) &
            #                 (p_tmp_ident_v[np.array(squared_i1_m, dtype=int)[layers, idx1s]] == poss_tmp_ident_v[j]) &
            #                 (p_idx_v[np.array(squared_i1_m, dtype=int)[layers, idx1s]] > idx_comp_range) &
            #                 (p_idx_v[np.array(squared_i1_m, dtype=int)[layers, idx1s]] <= idx_comp_range * 2)]
            #         except:
            #             embed()
            #             quit()
            #
            #         connect_ident[i, j] = np.percentile(errors_of_interest, 10) if len(errors_of_interest) >=  1 else np.nan
            #
            # ident_id, tmp_ident_id = np.unravel_index(np.argsort(connect_ident, axis=None), np.shape(connect_ident))
            #
            # for i, j in zip(ident_id, tmp_ident_id):
            #     idxs_i0 = p_idx_v[(p_ident_v == poss_ident_v[i]) & (p_idx_v > i + idx_comp_range) & (p_idx_v <= i + idx_comp_range * 2)]
            #     idxs_i1 = p_idx_v[(p_tmp_ident_v == poss_tmp_ident_v[j]) & (p_idx_v > i + idx_comp_range) & (p_idx_v <= i + idx_comp_range * 2)]
            #
            #     if np.any(np.diff(sorted(np.concatenate((idxs_i0, idxs_i1)))) == 0):
            #         continue
            #
            #     p_ident_v[(p_tmp_ident_v == poss_tmp_ident_v[j]) & (p_idx_v > i + idx_comp_range) & (p_idx_v <= i + idx_comp_range * 2)] = poss_ident_v[i]

            ##################################################################################

            for layer, idx0, idx1 in zip(layers[:i_non_nan], idx0s[:i_non_nan], idx1s[:i_non_nan]):
                # idents_to_assigne = ident_v[~np.isnan(tmp_ident_v) & (idx_v > i + idx_comp_range) & (idx_v <= i + idx_comp_range*2)]
                idents_to_assigne = p_ident_v[
                    ~np.isnan(p_tmp_ident_v) & (p_idx_v > i + idx_comp_range) & (p_idx_v <= i + idx_comp_range * 2)]

                if len(idents_to_assigne[np.isnan(idents_to_assigne)]) == 0:
                    if step_plot:
                        print('alle tmp identities vergeben')
                    # embed()
                    # quit()
                    break

                if np.isnan(cp_error_cube[layer, idx0, idx1]):
                    if step_plot:
                        print('nan connection')
                    break

                if ~np.isnan(p_ident_v[p_i1_m[layer][idx1]]):
                    continue

                if np.isnan(p_tmp_ident_v[p_i1_m[layer][idx1]]):
                    continue

                # if p_i1_m[layer][idx1] < enu + idx_comp_range:
                if p_i1_m[layer][idx1] < idx_comp_range:
                    if p_i1_m[layer][idx1] >= idx_comp_range * 2.:
                        # if p_i1_m[layer][idx1] >= enu + idx_comp_range *2.:
                        print('impossible')
                        embed()
                        quit()
                    continue

                if freq_lims:
                    if p_fund_v[p_i0_m[layer][idx0]] > freq_lims[1] or p_fund_v[p_i0_m[layer][idx0]] < freq_lims[0]:
                        continue
                    if p_fund_v[p_i1_m[layer][idx1]] > freq_lims[1] or p_fund_v[p_i1_m[layer][idx1]] < freq_lims[0]:
                        continue

                if np.isnan(p_ident_v[p_i0_m[layer][idx0]]):
                    continue

                idxs_i0 = p_idx_v[(p_ident_v == p_ident_v[p_i0_m[layer][idx0]]) & (p_idx_v > i + idx_comp_range) & (
                            p_idx_v <= i + idx_comp_range * 2)]
                idxs_i1 = p_idx_v[(p_tmp_ident_v == p_tmp_ident_v[p_i1_m[layer][idx1]]) & (np.isnan(p_ident_v)) & (
                            p_idx_v > i + idx_comp_range) & (p_idx_v <= i + idx_comp_range * 2)]

                if np.any(np.diff(sorted(np.concatenate((idxs_i0, idxs_i1)))) == 0):
                    continue

                if step_plot:
                    ax.plot([times[p_idx_v[p_i0_m[layer][idx0]]], times[p_idx_v[p_i1_m[layer][idx1]]]],
                            [p_fund_v[p_i0_m[layer][idx0]], p_fund_v[p_i1_m[layer][idx1]]], color='white', lw=2,
                            zorder=10)
                # print(p_tmp_ident_v[p_i1_m[layer][idx1]])

                p_ident_v[(p_tmp_ident_v == p_tmp_ident_v[p_i1_m[layer][idx1]]) & (np.isnan(p_ident_v)) & (
                            p_idx_v > i + idx_comp_range) & (p_idx_v <= i + idx_comp_range * 2)] = p_ident_v[
                    p_i0_m[layer][idx0]]

                if plotted:
                    # embed()
                    # quit()
                    if ioi_field[0] in np.arange(len(idx_v))[tmp_ident_v == tmp_ident_v[i1_m[layer][idx1]]]:
                        ax[5].plot(times[idx_v[i1_m[layer][idx1]]], fund_v[i1_m[layer][idx1]], marker='o', color='k',
                                   zorder=3)
                        ax[5].plot(times[idx_v[i0_m[layer][idx0]]], fund_v[i0_m[layer][idx0]], marker='o', color='gold',
                                   zorder=3)

                        help = \
                        np.arange(len(idx_v))[(ident_v == ident_v[ioi_field[1]]) & (idx_v <= i + idx_comp_range)][-1]
                        ax[5].plot(times[idx_v[help]], fund_v[help], marker='o', color='forestgreen', zorder=3)
                        ax[5].plot(times[idx_v[help]], fund_v[help], marker='x', color='red', zorder=3, markersize=8)

                        import matplotlib.patches as patches
                        style = "Simple,tail_width=0.5,head_width=4,head_length=8"
                        kw = dict(arrowstyle=style, color="k")
                        a = patches.FancyArrowPatch((times[idx_v[i0_m[layer][idx0]]], fund_v[i0_m[layer][idx0]]),
                                                    (times[idx_v[i1_m[layer][idx1]]], fund_v[i1_m[layer][idx1]]),
                                                    connectionstyle="arc3,rad=.7", zorder=6, **kw)

                        ax[5].add_patch(a)

                        ioi_field[0] = i1_m[layer][idx1]
                        ioi_field[1] = help
                        ioi_field[2] = i0_m[layer][idx0]
                        plotted = False
                        plotting_finished = True

            ident_v[min_i0:max_i1 + 1] = p_ident_v

            if plotting_finished:
                for ident in np.unique(ident_v[~np.isnan(ident_v)]):
                    c = colors[int(ident % len(colors))]
                    ax[6].plot(times[idx_v[ident_v == ident]], fund_v[ident_v == ident], marker='.', color=c, zorder=2)

                return ioi_field

            if step_plot:
                for h in handle:
                    h.remove()
                handle = []

                for ident in np.unique(ident_v[~np.isnan(ident_v)]):
                    h, = ax.plot(times[idx_v[ident_v == ident]], fund_v[ident_v == ident], marker='.', zorder=1)
                    handle.append(h)
                fig.canvas.draw()
                # ax.set_title('added idents')
                # plt.draw()
                # plt.waitforbuttonpress()

            for ident in np.unique(p_tmp_ident_v[~np.isnan(p_tmp_ident_v)]):
                if len(p_ident_v[p_tmp_ident_v == ident][~np.isnan(p_ident_v[p_tmp_ident_v == ident])]) == 0:
                    p_ident_v[(p_tmp_ident_v == ident) & (p_idx_v > i + idx_comp_range) & (
                                p_idx_v <= i + idx_comp_range * 2)] = next_identity
                    next_identity += 1

            if step_plot:
                for h in handle:
                    h.remove()
                handle = []

                for ident in np.unique(ident_v[~np.isnan(ident_v)]):
                    h, = ax.plot(times[idx_v[ident_v == ident]], fund_v[ident_v == ident], marker='.', zorder=1)
                    handle.append(h)
                fig.canvas.draw()

            ######################################################
            # for idx0, idx1 in zip(idx0s, idx1s):
            #     if np.isnan(error_cube[0][idx0, idx1]):
            #         break
            #
            #     if freq_lims:
            #         if fund_v[i0_m[0][idx0]] > freq_lims[1] or fund_v[i0_m[0][idx0]] < freq_lims[0]:
            #             continue
            #         if fund_v[i1_m[0][idx1]] > freq_lims[1] or fund_v[i1_m[0][idx1]] < freq_lims[0]:
            #             continue
            #
            #     if not np.isnan(ident_v[i1_m[0][idx1]]):
            #         continue
            #
            #     if not np.isnan(errors_to_v[i1_m[0][idx1]]):
            #         if errors_to_v[i1_m[0][idx1]] < error_cube[0][idx0, idx1]:
            #             continue
            #
            #     if np.isnan(ident_v[i0_m[0][idx0]]):  # i0 doesnt have identity
            #         # if 1. * np.abs(fund_v[i0_m[0][idx0]] - fund_v[i1_m[0][idx1]]) / ((idx_v[i1_m[0][idx1]] - idx_v[i0_m[0][idx0]]) / dps) > 2.:
            #         #     continue
            #
            #         if np.isnan(ident_v[i1_m[0][idx1]]):  # i1 doesnt have identity
            #             ident_v[i0_m[0][idx0]] = next_identity
            #             ident_v[i1_m[0][idx1]] = next_identity
            #             next_identity += 1
            #         else:  # i1 does have identity
            #             continue
            #
            #     else:  # i0 does have identity
            #         if np.isnan(ident_v[i1_m[0][idx1]]):  # i1 doesnt have identity
            #             if idx_v[i1_m[0][idx1]] in idx_v[ident_v == ident_v[i0_m[0][idx0]]]:
            #                 continue
            #             # _____ if either idx0-idx1 is not a direct connection or ...
            #             # _____ idx1 is not the new last point of ident[idx0] check ...
            #             if not idx_v[i0_m[0][idx0]] == idx_v[ident_v == ident_v[i0_m[0][idx0]]][-1]:  # if i0 is not the last ...
            #                 if len(ident_v[(ident_v == ident_v[i0_m[0][idx0]]) & (idx_v > idx_v[i0_m[0][idx0]]) & (idx_v < idx_v[i1_m[0][idx1]])]) == 0:  # zwischen i0 und i1 keiner
            #                     next_idx_after_new = np.arange(len(ident_v))[(ident_v == ident_v[i0_m[0][idx0]]) & (idx_v > idx_v[i1_m[0][idx1]])][0]
            #                     if tmp_ident_v[next_idx_after_new] != tmp_ident_v[i1_m[0][idx1]]:
            #                         continue
            #                 elif len(ident_v[(ident_v == ident_v[i0_m[0][idx0]]) & (idx_v > idx_v[i1_m[0][idx1]])]) == 0:  # keiner nach i1
            #                     last_idx_before_new = np.arange(len(ident_v))[(ident_v == ident_v[i0_m[0][idx0]]) & (idx_v < idx_v[i1_m[0][idx1]])][-1]
            #                     if tmp_ident_v[last_idx_before_new] != tmp_ident_v[i1_m[0][idx1]]:
            #                         continue
            #                 else:  # sowohl als auch
            #                     next_idx_after_new = np.arange(len(ident_v))[(ident_v == ident_v[i0_m[0][idx0]]) & (idx_v > idx_v[i1_m[0][idx1]])][0]
            #                     last_idx_before_new = np.arange(len(ident_v))[(ident_v == ident_v[i0_m[0][idx0]]) & (idx_v < idx_v[i1_m[0][idx1]])][-1]
            #                     if tmp_ident_v[last_idx_before_new] != tmp_ident_v[i1_m[0][idx1]] or tmp_ident_v[next_idx_after_new] != tmp_ident_v[i1_m[0][idx1]]:
            #                         continue
            #
            #             ident_v[i1_m[0][idx1]] = ident_v[i0_m[0][idx0]]
            #         else:
            #             continue
            #
            #     idx_of_origin_v[i1_m[0][idx1]] = i0_m[0][idx0]
            #
            #     if fig:
            #         if not hasattr(ioi_field, '__len__'):
            #             for handle in life_handels:
            #                 handle.remove()
            #             if life0:
            #                 life0.remove()
            #                 life1.remove()
            #
            #             life_handels = []
            #
            #             life0, = ax.plot(times[idx_v[i0_m[0][idx0]]], fund_v[i0_m[0][idx0]], color='red', marker='o')
            #             life1, = ax.plot(times[idx_v[i1_m[0][idx1]]], fund_v[i1_m[0][idx1]], color='red', marker='o')
            #
            #             xlims = ax.get_xlim()
            #             for ident in np.unique(ident_v[~np.isnan(ident_v)]):
            #                 # embed()
            #                 # quit()
            #                 plot_times = times[idx_v[ident_v == ident]]
            #                 plot_freqs = fund_v[ident_v == ident]
            #
            #                 # h, = ax.plot(times[idx_v[ident_v == ident]], fund_v[ident_v == ident ], color='k', marker = '.', markersize=5)
            #                 h, = ax.plot(plot_times[(plot_times >= xlims[0] - 1)],
            #                              plot_freqs[(plot_times >= xlims[0] - 1)], color='k', marker='.', markersize=5)
            #                 life_handels.append(h)
            #
            #                 if times[idx_v[ident_v == ident]][-1] > xlims[1]:
            #                     # xlim = ax.get_xlim()
            #                     ax.set_xlim([xlims[0] + 10, xlims[1] + 10])
            #
            #             fig.canvas.draw()

            # fund_v[min_i0:max_i1+1] = p_fund_v

        # sort_time += time.time()-t0
        if plotted and np.min(ioi_field) in np.array(i0_m[0]):
            # for ident in ident_v[~np.isnan(ident_v)]:
            #     # ax[3].plot(times[idx_v[(ident_v == ident) & (idx_v <= idx_v[np.min(ioi_field)]) ]], fund_v[(ident_v == ident)  & (idx_v <= idx_v[np.min(ioi_field)])], marker='.', color=np.random.rand(3))
            #     ax[4].plot(times[idx_v[(ident_v == ident) & (idx_v <= idx_v[np.min(ioi_field)]) ]], fund_v[(ident_v == ident)  & (idx_v <= idx_v[np.min(ioi_field)])], marker='.', color=np.random.rand(3))
            #     # ax[5].plot(times[idx_v[(ident_v == ident) & (idx_v < idx_v[np.min(ioi_field)]) ]], fund_v[(ident_v == ident)  & (idx_v < idx_v[np.min(ioi_field)])], marker='.', color=np.random.rand(3))
            #     ax[5].plot(times[idx_v[ident_v == ident]], fund_v[ident_v == ident], marker='.', color=np.random.rand(3))
            #
            # ax[3].plot(times[idx_v[ioi_field[0]]], fund_v[ioi_field[0]], color='green', marker='o', alpha = 0.5)
            # ax[4].plot(times[idx_v[ioi_field[0]]], fund_v[ioi_field[0]], color='green', marker='o', alpha = 0.5)
            # ax[5].plot(times[idx_v[ioi_field[0]]], fund_v[ioi_field[0]], color='green', marker='o', alpha = 0.5)
            #
            # ax[3].plot(times[idx_v[ioi_field[1]]], fund_v[ioi_field[1]], color='orange', marker='o', alpha = 0.5)
            # ax[4].plot(times[idx_v[ioi_field[1]]], fund_v[ioi_field[1]], color='orange', marker='o', alpha = 0.5)
            # ax[5].plot(times[idx_v[ioi_field[1]]], fund_v[ioi_field[1]], color='orange', marker='o', alpha = 0.5)
            #
            # ax[3].plot(times[idx_v[ioi_field[2]]], fund_v[ioi_field[2]], color='red', marker='o', alpha = 0.5)
            # ax[4].plot(times[idx_v[ioi_field[2]]], fund_v[ioi_field[2]], color='red', marker='o', alpha = 0.5)
            # ax[5].plot(times[idx_v[ioi_field[2]]], fund_v[ioi_field[2]], color='red', marker='o', alpha = 0.5)
            # print('im here')
            return fund_v, ident_v, idx_v, sign_v, a_error_distribution, f_error_distribution, idx_of_origin_v

        i0_m.pop(0)
        i1_m.pop(0)
        error_cube.pop(0)

        i0_v = np.arange(len(idx_v))[(idx_v == cube_app_idx) & (fund_v >= freq_lims[0]) & (
                    fund_v <= freq_lims[1])]  # indices of fundamtenals to assign
        i1_v = np.arange(len(idx_v))[
            (idx_v > cube_app_idx) & (idx_v <= (cube_app_idx + idx_comp_range)) & (fund_v >= freq_lims[0]) & (
                        fund_v <= freq_lims[1])]  # indices of possible targets

        i0_m.append(i0_v)
        i1_m.append(i1_v)

        if len(i0_v) == 0 or len(i1_v) == 0:  # if nothing to assign or no targets continue
            error_cube.append(np.array([[]]))

        else:
            try:
                error_matrix = np.full((len(i0_v), len(i1_v)), np.nan)
            except:
                error_matrix = np.zeros((len(i0_v), len(i1_v))) / 0.

            for enu0 in range(len(fund_v[i0_v])):
                if fund_v[i0_v[enu0]] < low_freq_th or fund_v[
                    i0_v[enu0]] > high_freq_th:  # freq to assigne out of tracking range
                    continue

                for enu1 in range(len(fund_v[i1_v])):
                    if fund_v[i1_v[enu1]] < low_freq_th or fund_v[
                        i1_v[enu1]] > high_freq_th:  # target freq out of tracking range
                        continue
                    if np.abs(fund_v[i0_v[enu0]] - fund_v[i1_v[enu1]]) >= freq_tolerance:  # freq difference to high
                        continue

                    a_error = np.sqrt(
                        np.sum([(sign_v[i0_v[enu0]][j] - sign_v[i1_v[enu1]][j]) ** 2 for j in range(n_channels)]))
                    f_error = np.abs(fund_v[i0_v[enu0]] - fund_v[i1_v[enu1]])
                    t_error = 1. * np.abs(idx_v[i0_v[enu0]] - idx_v[i1_v[enu1]]) / dps

                    error = estimate_error(a_error, f_error, t_error, a_error_distribution, f_error_distribution)
                    error_matrix[enu0, enu1] = np.sum(error)
            error_cube.append(error_matrix)

        cube_app_idx += 1
    ident_v = clean_up(fund_v, ident_v, idx_v, times)

    if step_plot:
        for h in tmp_handle:
            h.remove()
        tmp_handle = []

        ax.set_xlim([times[idx_v[~np.isnan(ident_v)]][0] - 1, times[idx_v[~np.isnan(ident_v)]][-1] + 1])
        fig.canvas.draw()

    # plt.close()
    # fig, ax = plt.subplots()
    #
    # uni_t = np.unique(all_false_dt)
    # uni_a_rel_false = [np.array(all_rel_a_error_false)[np.array(all_false_dt) == t] for t in uni_t]
    # uni_a_rel_true = [np.array(all_rel_a_error_true)[np.array(all_true_dt) == t] for t in uni_t]
    #
    # mean_false = [np.mean(uni_a_rel_false[i]) for i in range(len(uni_a_rel_false))]
    # std_false =  [np.std(uni_a_rel_false[i]) for i in range(len(uni_a_rel_false))]
    # mean_true = [np.mean(uni_a_rel_true[i]) for i in range(len(uni_a_rel_true))]
    # std_true = [np.std(uni_a_rel_true[i]) for i in range(len(uni_a_rel_true))]
    #
    # ax.plot(uni_t, mean_false, color='red')
    # ax.fill_between(uni_t, np.array(mean_false)+np.array(std_false), np.array(mean_false)-np.array(std_false), color='red', alpha=0.3)
    # ax.plot(uni_t, mean_true, color='green')
    # ax.fill_between(uni_t, np.array(mean_true)+np.array(std_true), np.array(mean_true)-np.array(std_true), color='green', alpha=0.3)

    # embed()
    # quit()

    return fund_v, ident_v, idx_v, sign_v, a_error_distribution, f_error_distribution, idx_of_origin_v, original_sign_v


# def freq_tracking_v3(fundamentals, signatures, times, freq_tolerance, n_channels, return_tmp_idenities=False,
#                      ioi_fti=False, a_error_distribution=False, f_error_distribution=False, fig = False, ax = False,
#                      freq_lims=(400, 1200), ioi_field=False):
#     """
#     Sorting algorithm which sorts fundamental EOD frequnecies detected in consecutive powespectra of single or
#     multielectrode recordings using frequency difference and frequnency-power amplitude difference on the electodes.
#
#     Signal tracking and identity assiginment is accomplished in four steps:
#     1) Extracting possible frequency and amplitude difference distributions.
#     2) Esitmate relative error between possible datapoint connections (relative amplitude and frequency error based on
#     frequency and amplitude error distribution).
#     3) For a data window covering the EOD frequencies detected 10 seconds before the accual datapoint to assigne
#     identify temporal identities based on overall error between two datapoints from smalles to largest.
#     4) Form tight connections between datapoints where one datapoint is in the timestep that is currently of interest.
#
#     Repeat these steps until the end of the recording.
#     The temporal identities are only updated when the timestep of current interest reaches the middle (5 sec.) of the
#     temporal identities. This is because no tight connection shall be made without checking the temporal identities.
#     The temnporal identities are used to check if the potential connection from the timestep of interest to a certain
#     datsapoint is the possibly best or if a connection in the futur will be better. If a future connection is better
#     the thight connection is not made.
#
#     Parameters
#     ----------
#     fundamentals: 2d-arraylike / list
#         list of arrays of fundemantal EOD frequnecies. For each timestep/powerspectrum contains one list with the
#         respectivly detected fundamental EOD frequnecies.
#     signatures: 3d-arraylike / list
#         same as fundamentals but for each value in fundamentals contains a list of powers of the respective frequency
#         detected of n electrodes used.
#     times: array
#         respective time vector.
#     freq_tolerance: float
#         maximum frequency difference between two datapoints to be connected in Hz.
#     n_channels: int
#         number of channels/electodes used in the analysis.,
#     return_tmp_idenities: bool
#         only returne temporal identities at a certain timestep. Dependent on ioi_fti and only used to check algorithm.
#     ioi_fti: int
#         Index Of Interest For Temporal Identities: respective index in fund_v to calculate the temporal identities for.
#     a_error_distribution: array
#         possible amplitude error distributions for the dataset.
#     f_error_distribution: array
#         possible frequency error distribution for the dataset.
#     fig: mpl.figure
#         figure to plot the tracking progress life.
#     ax: mpl.axis
#         axis to plot the tracking progress life.
#     freq_lims: double
#         minimum/maximum frequency to be tracked.
#
#     Returns
#     -------
#     fund_v: array
#         flattened fundamtantals array containing all detected EOD frequencies in the recording.
#     ident_v: array
#         respective assigned identites throughout the tracking progress.
#     idx_v: array
#         respective index vectro impliing the time of the detected frequency.
#     sign_v: 2d-array
#         for each fundamental frequency the power of this frequency on the used electodes.
#     a_error_distribution: array
#         possible amplitude error distributions for the dataset.
#     f_error_distribution: array
#         possible frequency error distribution for the dataset.
#     idx_of_origin_v: array
#         for each assigned identity the index of the datapoint on which basis the assignement was made.
#     """
#     def clean_up(fund_v, ident_v, idx_v, times):
#         """
#         deletes/replaces with np.nan those identities only consisting from little data points and thus are tracking
#         artefacts. Identities get deleted when the proportion of the trace (slope, ratio of detected datapoints, etc.)
#         does not fit a real fish.
#
#         Parameters
#         ----------
#         fund_v: array
#             flattened fundamtantals array containing all detected EOD frequencies in the recording.
#         ident_v: array
#             respective assigned identites throughout the tracking progress.
#         idx_v: array
#             respective index vectro impliing the time of the detected frequency.
#         times: array
#             respective time vector.
#
#         Returns
#         -------
#         ident_v: array
#             cleaned up identities vector.
#
#         """
#         # print('clean up')
#         for ident in np.unique(ident_v[~np.isnan(ident_v)]):
#             if np.median(np.abs(np.diff(fund_v[ident_v == ident]))) >= 0.25:
#                 ident_v[ident_v == ident] = np.nan
#                 continue
#
#             if len(ident_v[ident_v == ident]) <= 10:
#                 ident_v[ident_v == ident] = np.nan
#                 continue
#
#         return ident_v
#
#     def get_a_and_f_error_dist(fund_v, idx_v, sign_v):
#         """
#         get the distribution of possible frequency and amplitude errors for the tracking.
#
#         Parameters
#         ----------
#         fund_v: array
#             flattened fundamtantals array containing all detected EOD frequencies in the recording.
#         idx_v: array
#             respective index vectro impliing the time of the detected frequency.
#         sign_v: 2d-array
#             for each fundamental frequency the power of this frequency on the used electodes.
#
#         Returns
#         -------
#         f_error_distribution: array
#             possible frequency error distribution for the dataset.
#         a_error_distribution: array
#             possible amplitude error distributions for the dataset.
#         """
#         # get f and amp signature distribution ############### BOOT #######################
#         a_error_distribution = np.zeros(20000)  # distribution of amplitude errors
#         f_error_distribution = np.zeros(20000)  # distribution of frequency errors
#         idx_of_distribution = np.zeros(20000)  # corresponding indices
#
#         b = 0  # loop varialble
#         next_message = 0.  # feedback
#
#         while b < 20000:
#             next_message = include_progress_bar(b, 20000, 'get f and sign dist', next_message)  # feedback
#
#             while True:  # finding compare indices to create initial amp and freq distribution
#                 # r_idx0 = np.random.randint(np.max(idx_v[~np.isnan(idx_v)]))
#                 r_idx0 = np.random.randint(np.max(idx_v[~np.isnan(idx_v)]))
#                 r_idx1 = r_idx0 + 1
#                 if len(sign_v[idx_v == r_idx0]) != 0 and len(sign_v[idx_v == r_idx1]) != 0:
#                     break
#
#             r_idx00 = np.random.randint(len(sign_v[idx_v == r_idx0]))
#             r_idx11 = np.random.randint(len(sign_v[idx_v == r_idx1]))
#
#             s0 = sign_v[idx_v == r_idx0][r_idx00]  # amplitude signatures
#             s1 = sign_v[idx_v == r_idx1][r_idx11]
#
#             f0 = fund_v[idx_v == r_idx0][r_idx00]  # fundamentals
#             f1 = fund_v[idx_v == r_idx1][r_idx11]
#
#             # if np.abs(f0 - f1) > freq_tolerance:  # frequency threshold
#             if np.abs(f0 - f1) > 10.:  # frequency threshold
#                 continue
#
#             idx_of_distribution[b] = r_idx0
#             a_error_distribution[b] = np.sqrt(np.sum([(s0[k] - s1[k]) ** 2 for k in range(len(s0))]))
#             f_error_distribution[b] = np.abs(f0 - f1)
#             b += 1
#
#         return f_error_distribution, a_error_distribution
#
#     def get_tmp_identities(i0_m, i1_m, error_cube, fund_v, idx_v, i, ioi_fti, dps, idx_comp_range,
#                            sign_v, a_error_distribution, f_error_distribution, ioi_field = False, fig=False, ax=False):
#         """
#         extract temporal identities for a datasnippted of 2*index compare range of the original tracking algorithm.
#         for each data point in the data window finds the best connection within index compare range and, thus connects
#         the datapoints based on their minimal error value until no connections are left or possible anymore.
#
#         Parameters
#         ----------
#         i0_m: 2d-array
#             for consecutive timestamps contains for each the indices of the origin EOD frequencies.
#         i1_m: 2d-array
#             respectively contains the indices of the targen EOD frequencies, laying within index compare range.
#         error_cube: 3d-array
#             error values for each combination from i0_m and the respective indices in i1_m.
#         fund_v: array
#             flattened fundamtantals array containing all detected EOD frequencies in the recording.
#         idx_v: array
#             respective index vectro impliing the time of the detected frequency.
#         i: int
#             loop variable and current index of interest for the assignment of tight connections.
#         ioi_fti: int
#             index of interest for temporal identities.
#         dps: float
#             detections per second. 1. / 'temporal resolution of the tracking'
#         idx_comp_range: int
#             index compare range for the assignment of two data points to each other.
#
#         Returns
#         -------
#         tmp_ident_v: array
#             for each EOD frequencies within the index compare range for the current time step of interest contains the
#             temporal identity.
#         errors_to_v: array
#             for each assigned temporal identity contains the error value based on which this connection was made.
#
#         """
#         next_tmp_identity = 0
#         # mask_cube = [np.ones(np.shape(error_cube[n]), dtype=bool) for n in range(len(error_cube))]
#
#         max_shape = np.max([np.shape(layer) for layer in error_cube[1:]], axis=0)
#         cp_error_cube = np.full((len(error_cube)-1, max_shape[0], max_shape[1]), np.nan)
#         for enu, layer in enumerate(error_cube[1:]):
#             cp_error_cube[enu, :np.shape(error_cube[enu+1])[0], :np.shape(error_cube[enu+1])[1]] = layer
#
#         try:
#             tmp_ident_v = np.full(len(fund_v), np.nan)
#             errors_to_v = np.full(len(fund_v), np.nan)
#         except:
#             tmp_ident_v = np.zeros(len(fund_v)) / 0.
#             errors_to_v = np.zeros(len(fund_v)) / 0.
#
#         layers, idx0s, idx1s = np.unravel_index(np.argsort(cp_error_cube, axis=None), np.shape(cp_error_cube))
#         made_connections = np.zeros(np.shape(cp_error_cube))
#         not_made_connections = np.zeros(np.shape(cp_error_cube))
#         not_made_connections[~np.isnan(cp_error_cube)] = 1
#         # made_connections[~np.isnan(cp_error_cube)] = 0
#
#         layers = layers+1
#
#         # embed()
#         # quit()
#         plotted = False
#         if hasattr(ioi_field, '__len__'):
#             if np.min(ioi_field) in np.hstack(i0_m[:int(len(i0_m)/3)]):
#                 # embed()
#                 # quit()
#
#                 c_i = np.unique(np.concatenate((np.hstack(i0_m), np.hstack(i1_m))))
#                 # c_i = c_i[idx_v[c_i] - idx_v[np.min(c_i)] <= idx_comp_range * 3]
#                 ax[0].scatter(times[idx_v[c_i]], fund_v[c_i], color='grey', alpha= 0.5)
#                 ax[1].scatter(times[idx_v[c_i]], fund_v[c_i], color='grey', alpha= 0.5)
#                 ax[2].scatter(times[idx_v[c_i]], fund_v[c_i], color='grey', alpha= 0.5)
#
#                 max_t = np.max(times[idx_v[c_i]])
#                 ax[0].set_xlim([max_t - 39., max_t + 1.])
#                 ax[1].set_xlim([max_t - 39., max_t + 1.])
#                 ax[2].set_xlim([max_t - 39., max_t + 1.])
#
#                 ax[3].set_xlim([max_t - 39., max_t + 1.])
#                 ax[4].set_xlim([max_t - 39., max_t + 1.])
#                 ax[5].set_xlim([max_t - 39., max_t + 1.])
#
#                 ax[0].set_ylim([885, 935])
#                 ax[1].set_ylim([885, 935])
#                 ax[2].set_ylim([885, 935])
#                 ax[3].set_ylim([907, 930])
#                 ax[4].set_ylim([907, 930])
#                 ax[5].set_ylim([907, 930])
#
#                 plotted = True
#                 c = 0
#                 for layer, idx0, idx1 in zip(layers, idx0s, idx1s):
#                     c += 1
#                     if np.isnan(cp_error_cube[layer-1, idx0, idx1]):
#                         break
#                 plot_idxs = [200, 500, int(np.floor(c))]
#                 # fig, ax = plt.subplots(3, 1, figsize=(20./2.54, 36./2.54), facecolor='white')
#                 # embed()
#         counter = 0
#         error_line_at = []
#         for layer, idx0, idx1 in zip(layers, idx0s, idx1s):
#             counter += 1
#             if hasattr(ioi_field, '__len__') and plotted:
#                 if counter in plot_idxs:
#                     if counter >= plot_idxs[0] and counter < plot_idxs[1]:
#                         for ident in np.unique(tmp_ident_v[~np.isnan(tmp_ident_v)]):
#                             ax[0].plot(times[idx_v[tmp_ident_v == ident]], fund_v[tmp_ident_v == ident], color='k',
#                                        marker='.', alpha=0.7)
#                         error_line_at.append(last_error)
#                     elif counter >= plot_idxs[1] and counter < plot_idxs[2]:
#                         for ident in np.unique(tmp_ident_v[~np.isnan(tmp_ident_v)]):
#                             ax[1].plot(times[idx_v[tmp_ident_v == ident]], fund_v[tmp_ident_v == ident], color='k',
#                                        marker='.', alpha=0.7)
#                         error_line_at.append(last_error)
#                     else:
#                         for ident in np.unique(tmp_ident_v[~np.isnan(tmp_ident_v)]):
#                             ax[2].plot(times[idx_v[tmp_ident_v == ident]], fund_v[tmp_ident_v == ident], color='k',
#                                        marker='.', alpha=0.7)
#                             ax[3].plot(times[idx_v[tmp_ident_v == ident]], fund_v[tmp_ident_v == ident], color='k',
#                                        marker='.', alpha=0.7)
#                             ax[4].plot(times[idx_v[(tmp_ident_v == ident) & (idx_v > idx_v[np.min(ioi_field)])]],
#                                        fund_v[(tmp_ident_v == ident) & (idx_v > idx_v[np.min(ioi_field)])], color='k',
#                                        marker='.', alpha=0.7)
#                             ax[5].plot(times[idx_v[(tmp_ident_v == ident) & (idx_v > idx_v[np.min(ioi_field)])]],
#                                        fund_v[(tmp_ident_v == ident) & (idx_v > idx_v[np.min(ioi_field)])], color='k',
#                                        marker='.', alpha=0.7)
#                         error_line_at.append(last_error)
#
#                             # ax[5].plot(times[idx_v[tmp_ident_v == ident]], fund_v[tmp_ident_v == ident], color='k',
#                             #            marker='.', alpha=0.7)
#
#
#             if np.isnan(cp_error_cube[layer-1, idx0, idx1]):
#                 break
#
#             # _____ some control functions _____ ###
#             if not ioi_fti:
#                 if idx_v[i1_m[layer][idx1]] - i > idx_comp_range*2:
#                     continue
#             else:
#                 if idx_v[i1_m[layer][idx1]] - idx_v[ioi_fti] > idx_comp_range*2:
#                     continue
#
#             if fund_v[i0_m[layer][idx0]] > fund_v[i1_m[layer][idx1]]:
#                 if 1. * np.abs(fund_v[i0_m[layer][idx0]] - fund_v[i1_m[layer][idx1]]) / ((idx_v[i1_m[layer][idx1]] - idx_v[i0_m[layer][idx0]]) / dps) > 2.:
#                     continue
#             else:
#                 if 1. * np.abs(fund_v[i0_m[layer][idx0]] - fund_v[i1_m[layer][idx1]]) / ((idx_v[i1_m[layer][idx1]] - idx_v[i0_m[layer][idx0]]) / dps) > 2.:
#                     continue
#
#             if np.isnan(tmp_ident_v[i0_m[layer][idx0]]):
#                 if np.isnan(tmp_ident_v[i1_m[layer][idx1]]):
#                     tmp_ident_v[i0_m[layer][idx0]] = next_tmp_identity
#                     tmp_ident_v[i1_m[layer][idx1]] = next_tmp_identity
#                     errors_to_v[i1_m[layer][idx1]] = cp_error_cube[layer-1][idx0, idx1]
#                     # errors_to_v[i0_m[layer][idx0]] = error_cube[layer][idx0, idx1]
#                     not_made_connections[layer-1, idx0, idx1] = 0
#                     made_connections[layer-1, idx0, idx1] = 1
#
#                     # errors_to_v[(tmp_ident_v == tmp_ident_v[i1_m[layer][idx1]]) & (np.isnan(errors_to_v))] = error_cube[layer][idx0, idx1]
#                     next_tmp_identity += 1
#                 else:
#                     if idx_v[i0_m[layer][idx0]] in idx_v[tmp_ident_v == tmp_ident_v[i1_m[layer][idx1]]]:
#                         continue
#                     tmp_ident_v[i0_m[layer][idx0]] = tmp_ident_v[i1_m[layer][idx1]]
#                     errors_to_v[i1_m[layer][idx1]] = cp_error_cube[layer-1][idx0, idx1]
#                     not_made_connections[layer-1, idx0, idx1] = 0
#                     made_connections[layer-1, idx0, idx1] = 1
#
#                     # errors_to_v[(tmp_ident_v == tmp_ident_v[i1_m[layer][idx1]]) & (np.isnan(errors_to_v))] = error_cube[layer][idx0, idx1]
#                     # errors_to_v[tmp_ident_v == tmp_ident_v[i1_m[layer][idx1]]][0] = np.nan
#
#             else:
#                 if np.isnan(tmp_ident_v[i1_m[layer][idx1]]):
#                     if idx_v[i1_m[layer][idx1]] in idx_v[tmp_ident_v == tmp_ident_v[i0_m[layer][idx0]]]:
#                         continue
#                     tmp_ident_v[i1_m[layer][idx1]] = tmp_ident_v[i0_m[layer][idx0]]
#                     errors_to_v[i1_m[layer][idx1]] = cp_error_cube[layer-1][idx0, idx1]
#                     not_made_connections[layer-1, idx0, idx1] = 0
#                     made_connections[layer-1, idx0, idx1] = 1
#
#                     # errors_to_v[(tmp_ident_v == tmp_ident_v[i1_m[layer][idx1]]) & (np.isnan(errors_to_v))] = error_cube[layer][idx0, idx1]
#                     # errors_to_v[tmp_ident_v == tmp_ident_v[i1_m[layer][idx1]]][0] = np.nan
#
#                 else:
#                     if tmp_ident_v[i0_m[layer][idx0]] == tmp_ident_v[i1_m[layer][idx1]]:
#                         if np.isnan(errors_to_v[i1_m[layer][idx1]]):
#                             errors_to_v[i1_m[layer][idx1]] = cp_error_cube[layer-1][idx0, idx1]
#                         continue
#
#                     idxs_i0 = idx_v[tmp_ident_v == tmp_ident_v[i0_m[layer][idx0]]]
#                     idxs_i1 = idx_v[tmp_ident_v == tmp_ident_v[i1_m[layer][idx1]]]
#
#                     if np.any(np.diff(sorted(np.concatenate((idxs_i0, idxs_i1)))) == 0):
#                         continue
#                     tmp_ident_v[tmp_ident_v == tmp_ident_v[i0_m[layer][idx0]]] = tmp_ident_v[i1_m[layer][idx1]]
#
#                     if np.isnan(errors_to_v[i1_m[layer][idx1]]):
#                         errors_to_v[i1_m[layer][idx1]] = cp_error_cube[layer-1][idx0, idx1]
#                         not_made_connections[layer-1, idx0, idx1] = 0
#                         made_connections[layer-1, idx0, idx1] = 1
#
#             last_error = cp_error_cube[layer-1, idx0, idx1]
#         # if hasattr(ioi_field, '__len__') and plotted:
#         #     made_connection_dist = cp_error_cube[np.array(made_connections, dtype=bool)]
#         #     n, h = np.histogram(made_connection_dist)
#         #     ax[6].plot(h[:-1], n, color='k')
#         #     ax[7].plot(h[:-1], n, color='k')
#         #     ax[8].plot(h[:-1], n, color='k')
#         #     y_lims = ax[6].get_ylim()
#         #     print(error_line_at)
#         #     ax[6].semilogx([error_line_at[0], error_line_at[0]], [y_lims[0], y_lims[1]], color='red')
#         #     ax[7].semilogx([error_line_at[1], error_line_at[1]], [y_lims[0], y_lims[1]], color='red')
#         #     ax[8].semilogx([error_line_at[2], error_line_at[2]], [y_lims[0], y_lims[1]], color='red')
#         #     ax[6].set_ylim([y_lims[0], y_lims[1]])
#         #     ax[7].set_ylim([y_lims[0], y_lims[1]])
#         #     ax[8].set_ylim([y_lims[0], y_lims[1]])
#         # embed()
#         # quit()
#         # if hasattr(ioi_field, '__len__') and plotted:
#             # xl = ax[2].get_xlim()
#             # for x in ax:
#             #     x.set_ylim([885, 935])
#                 # x.set_xlim(xl)
#             # plt.show(fig)
#             # return tmp_ident_v, errors_to_v, fig, ax
#
#
#
#         return tmp_ident_v, errors_to_v, plotted
#
#     def get_a_and_f_error_dist2(fund_v, idx_v, sign_v, start_idx, idx_comp_range, freq_lims, low_freq_th, high_freq_th, freq_tolerance, a_error_distribution = np.array([]), f_error_distribution= np.array([])):
#         f_error_distribution = list(f_error_distribution)
#         a_error_distribution = list(a_error_distribution)
#
#         # next_message = 0.0
#         for i in range(start_idx, int(start_idx + idx_comp_range * 2)):
#             # next_message = include_progress_bar(i - start_idx, int(idx_comp_range * 2), 'error dist init', next_message)
#             i0_v = np.arange(len(idx_v))[(idx_v == i) & (fund_v >= freq_lims[0]) & (fund_v <= freq_lims[1])]  # indices of fundamtenals to assign
#             i1_v = np.arange(len(idx_v))[(idx_v > i) & (idx_v <= (i + int(idx_comp_range))) & (fund_v >= freq_lims[0]) & (fund_v <= freq_lims[1])]  # indices of possible targets
#
#             if len(i0_v) == 0 or len(i1_v) == 0:  # if nothing to assign or no targets continue
#                 continue
#
#             for enu0 in range(len(fund_v[i0_v])):
#                 if fund_v[i0_v[enu0]] < low_freq_th or fund_v[i0_v[enu0]] > high_freq_th:
#                     continue
#                 for enu1 in range(len(fund_v[i1_v])):
#                     if fund_v[i1_v[enu1]] < low_freq_th or fund_v[i1_v[enu1]] > high_freq_th:
#                         continue
#                     if np.abs(fund_v[i0_v[enu0]] - fund_v[i1_v[enu1]]) >= freq_tolerance:  # freq difference to high
#                         continue
#                     a_error_distribution.append(np.sqrt(np.sum(
#                         [(sign_v[i0_v[enu0]][k] - sign_v[i1_v[enu1]][k]) ** 2 for k in
#                          range(len(sign_v[i0_v[enu0]]))])))
#                     f_error_distribution.append(np.abs(fund_v[i0_v[enu0]] - fund_v[i1_v[enu1]]))
#
#         return np.array(a_error_distribution)[-5000:], np.array(f_error_distribution)[-5000:]
#
#     # total_t0 = time.time()
#
#     # _____ plot environment for live tracking _____ ###
#     if fig:
#         if not hasattr(ioi_field, '__len__'):
#             xlim = ax.get_xlim()
#             ax.set_xlim(xlim[0], xlim[0]+20)
#             fig.canvas.draw()
#             life_handels = []
#             tmp_handles = []
#             life0 = None
#             life1 = None
#
#     # _____ exclude frequencies with lower dFs than 0.5Hz from algorythm ______ ###
#     # ToDo choose the one with the bigger power
#     # for i in range(len(fundamentals)):
#     #     # include_progress_bar(i, len(fundamentals), 'clear dubble deltections', next_message)
#     #     mask = np.zeros(len(fundamentals[i]), dtype=bool)
#     #     order = np.argsort(fundamentals[i])
#     #     fundamentals[i][order[np.arange(len(mask)-1)[np.diff(sorted(fundamentals[i])) < 0.5]+1]] = 0
#
#     pre_sort_time = 0.
#     sort_time = 0.
#     # _____ parameters and vectors _____ ###
#     detection_time_diff = times[1] - times[0]
#     dps = 1. / detection_time_diff
#     fund_v = np.hstack(fundamentals)
#     try:
#         ident_v = np.full(len(fund_v), np.nan)  # fish identities of frequencies
#         idx_of_origin_v = np.full(len(fund_v), np.nan)
#     except:
#         ident_v = np.zeros(len(fund_v)) / 0.  # fish identities of frequencies
#         idx_of_origin_v = np.zeros(len(fund_v)) / 0.
#
#     idx_v = []  # temportal indices
#     sign_v = []  # power of fundamentals on all electrodes
#     for enu, funds in enumerate(fundamentals):
#         idx_v.extend(np.ones(len(funds)) * enu)
#         sign_v.extend(signatures[enu])
#     idx_v = np.array(idx_v, dtype=int)
#     sign_v = np.array(sign_v)
#
#     # sign_v = (10.**sign_v) / 10.
#     sign_v = (sign_v - np.min(sign_v, axis =1).reshape(len(sign_v), 1)) / (np.max(sign_v, axis=1).reshape(len(sign_v), 1) - np.min(sign_v, axis=1).reshape(len(sign_v), 1))
#     # embed()
#     # quit()
#
#     idx_comp_range = int(np.floor(dps * 10.))  # maximum compare range backwards for amplitude signature comparison
#     low_freq_th = 400.  # min. frequency tracked
#     high_freq_th = 1050.  # max. frequency tracked
#
#     error_cube = []  # [fundamental_list_idx, freqs_to_assign, target_freqs]
#     i0_m = []
#     i1_m = []
#
#     next_message = 0.
#     start_idx = 0 if not ioi_fti else idx_v[ioi_fti] # Index Of Interest for temporal identities
#
#     # _____ get amp and freq error distribution
#     # if hasattr(a_error_distribution, '__len__') and hasattr(f_error_distribution, '__len__'):
#     #     pass
#     # else:
#     #     f_error_distribution, a_error_distribution = get_a_and_f_error_dist(fund_v, idx_v, sign_v)
#
#     a_error_distribution, f_error_distribution = get_a_and_f_error_dist2(fund_v, idx_v, sign_v, start_idx, idx_comp_range, freq_lims, low_freq_th, high_freq_th, freq_tolerance)
#
#     # _____ create initial error cube _____ ###
#
#
#     for i in range(start_idx, int(start_idx + idx_comp_range*2)):
#         next_message = include_progress_bar(i - start_idx, int(idx_comp_range*2), 'initial error cube', next_message)
#         i0_v = np.arange(len(idx_v))[(idx_v == i) & (fund_v >= freq_lims[0]) & (fund_v <= freq_lims[1])]  # indices of fundamtenals to assign
#         i1_v = np.arange(len(idx_v))[(idx_v > i) & (idx_v <= (i + int(idx_comp_range))) & (fund_v >= freq_lims[0]) & (fund_v <= freq_lims[1])]  # indices of possible targets
#
#         i0_m.append(i0_v)
#         i1_m.append(i1_v)
#
#         if len(i0_v) == 0 or len(i1_v) == 0:  # if nothing to assign or no targets continue
#             error_cube.append(np.array([[]]))
#             continue
#         try:
#             error_matrix = np.full((len(i0_v), len(i1_v)), np.nan)
#         except:
#             error_matrix = np.zeros((len(i0_v), len(i1_v))) / 0.
#
#         for enu0 in range(len(fund_v[i0_v])):
#             if fund_v[i0_v[enu0]] < low_freq_th or fund_v[
#                 i0_v[enu0]] > high_freq_th:  # freq to assigne out of tracking range
#                 continue
#             for enu1 in range(len(fund_v[i1_v])):
#                 if fund_v[i1_v[enu1]] < low_freq_th or fund_v[
#                     i1_v[enu1]] > high_freq_th:  # target freq out of tracking range
#                     continue
#                 if np.abs(fund_v[i0_v[enu0]] - fund_v[i1_v[enu1]]) >= freq_tolerance:  # freq difference to high
#                     continue
#
#                 a_error = np.sqrt(
#                     np.sum([(sign_v[i0_v[enu0]][j] - sign_v[i1_v[enu1]][j]) ** 2 for j in range(n_channels)]))
#                 f_error = np.abs(fund_v[i0_v[enu0]] - fund_v[i1_v[enu1]])
#                 t_error = 1. * np.abs(idx_v[i0_v[enu0]] - idx_v[i1_v[enu1]]) / dps
#
#                 error = estimate_error(a_error, f_error, t_error, a_error_distribution, f_error_distribution)
#                 error_matrix[enu0, enu1] = np.sum(error)
#         error_cube.append(error_matrix)
#
#     cube_app_idx = len(error_cube)
#
#     # _____ accual tracking _____ ###
#     next_identity = 0
#     next_message = 0.00
#     plotted = False
#     for enu, i in enumerate(np.arange(len(fundamentals))):
#         # print(i)
#         if i != 0 and (i % int(idx_comp_range * 120)) == 0: # clean up every 10 minutes
#             ident_v = clean_up(fund_v, ident_v, idx_v, times)
#
#         if not return_tmp_idenities:
#             next_message = include_progress_bar(i, len(fundamentals), 'tracking', next_message)  # feedback
#
#         if enu % idx_comp_range == 0:
#             # t0 = time.time()
#             a_error_distribution, f_error_distribution = get_a_and_f_error_dist2(fund_v, idx_v, sign_v, start_idx,idx_comp_range, freq_lims, low_freq_th,high_freq_th, freq_tolerance, a_error_distribution = a_error_distribution, f_error_distribution = f_error_distribution)
#             tmp_ident_v, errors_to_v, plotted = get_tmp_identities(i0_m, i1_m, error_cube, fund_v, idx_v, i, ioi_fti, dps, idx_comp_range, sign_v, a_error_distribution, f_error_distribution, ioi_field, fig, ax)
#
#             if fig:
#                 if not hasattr(ioi_field, '__len__'):
#                     for handle in tmp_handles:
#                         handle.remove()
#                     tmp_handles = []
#
#                     for ident in np.unique(tmp_ident_v[~np.isnan(tmp_ident_v)]):
#                         plot_times = times[idx_v[tmp_ident_v == ident]]
#                         plot_freqs = fund_v[tmp_ident_v == ident]
#
#                         # h, = ax.plot(times[idx_v[ident_v == ident]], fund_v[ident_v == ident ], color='k', marker = '.', markersize=5)
#                         h, = ax.plot(plot_times, plot_freqs, color='white', linewidth=4)
#                         tmp_handles.append(h)
#
#                     fig.canvas.draw()
#
#         if ioi_fti and return_tmp_idenities:
#             return fund_v, tmp_ident_v, idx_v
#
#         # mask_matrix = np.ones(np.shape(error_cube[0]), dtype=bool)
#
#         # t0 = time.time()
#
#         idx0s, idx1s = np.unravel_index(np.argsort(error_cube[0], axis=None), np.shape(error_cube[0]))
#
#         # if ioi_field:
#
#         # embed()
#         # quit()
#         # if plotted and np.min(ioi_field) in np.array(i0_m[0]):
#         #     for ident in ident_v[~np.isnan(ident_v)]:
#         #         ax[4].plot(times[idx_v[ident_v == ident]], fund_v[ident_v == ident], marker='.', color=np.random.rand(3))
#
#         for idx0, idx1 in zip(idx0s, idx1s):
#             if np.isnan(error_cube[0][idx0, idx1]):
#                 break
#
#             if freq_lims:
#                 if fund_v[i0_m[0][idx0]] > freq_lims[1] or fund_v[i0_m[0][idx0]] < freq_lims[0]:
#                     continue
#                 if fund_v[i1_m[0][idx1]] > freq_lims[1] or fund_v[i1_m[0][idx1]] < freq_lims[0]:
#                     continue
#
#             if not np.isnan(ident_v[i1_m[0][idx1]]):
#                 continue
#
#             if not np.isnan(errors_to_v[i1_m[0][idx1]]):
#                 if errors_to_v[i1_m[0][idx1]] < error_cube[0][idx0, idx1]:
#                     continue
#
#             if np.isnan(ident_v[i0_m[0][idx0]]):  # i0 doesnt have identity
#                 # if 1. * np.abs(fund_v[i0_m[0][idx0]] - fund_v[i1_m[0][idx1]]) / ((idx_v[i1_m[0][idx1]] - idx_v[i0_m[0][idx0]]) / dps) > 2.:
#                 #     continue
#
#                 if np.isnan(ident_v[i1_m[0][idx1]]):  # i1 doesnt have identity
#                     ident_v[i0_m[0][idx0]] = next_identity
#                     ident_v[i1_m[0][idx1]] = next_identity
#                     next_identity += 1
#                 else:  # i1 does have identity
#                     continue
#
#             else:  # i0 does have identity
#                 if np.isnan(ident_v[i1_m[0][idx1]]):  # i1 doesnt have identity
#                     if idx_v[i1_m[0][idx1]] in idx_v[ident_v == ident_v[i0_m[0][idx0]]]:
#                         continue
#                     # _____ if either idx0-idx1 is not a direct connection or ...
#                     # _____ idx1 is not the new last point of ident[idx0] check ...
#                     if not idx_v[i0_m[0][idx0]] == idx_v[ident_v == ident_v[i0_m[0][idx0]]][-1]:  # if i0 is not the last ...
#                         if len(ident_v[(ident_v == ident_v[i0_m[0][idx0]]) & (idx_v > idx_v[i0_m[0][idx0]]) & (idx_v < idx_v[i1_m[0][idx1]])]) == 0:  # zwischen i0 und i1 keiner
#                             next_idx_after_new = np.arange(len(ident_v))[(ident_v == ident_v[i0_m[0][idx0]]) & (idx_v > idx_v[i1_m[0][idx1]])][0]
#                             if tmp_ident_v[next_idx_after_new] != tmp_ident_v[i1_m[0][idx1]]:
#                                 continue
#                         elif len(ident_v[(ident_v == ident_v[i0_m[0][idx0]]) & (idx_v > idx_v[i1_m[0][idx1]])]) == 0:  # keiner nach i1
#                             last_idx_before_new = np.arange(len(ident_v))[(ident_v == ident_v[i0_m[0][idx0]]) & (idx_v < idx_v[i1_m[0][idx1]])][-1]
#                             if tmp_ident_v[last_idx_before_new] != tmp_ident_v[i1_m[0][idx1]]:
#                                 continue
#                         else:  # sowohl als auch
#                             next_idx_after_new = np.arange(len(ident_v))[(ident_v == ident_v[i0_m[0][idx0]]) & (idx_v > idx_v[i1_m[0][idx1]])][0]
#                             last_idx_before_new = np.arange(len(ident_v))[(ident_v == ident_v[i0_m[0][idx0]]) & (idx_v < idx_v[i1_m[0][idx1]])][-1]
#                             if tmp_ident_v[last_idx_before_new] != tmp_ident_v[i1_m[0][idx1]] or tmp_ident_v[next_idx_after_new] != tmp_ident_v[i1_m[0][idx1]]:
#                                 continue
#
#                     ident_v[i1_m[0][idx1]] = ident_v[i0_m[0][idx0]]
#                 else:
#                     continue
#
#             idx_of_origin_v[i1_m[0][idx1]] = i0_m[0][idx0]
#
#             if fig:
#                 if not hasattr(ioi_field, '__len__'):
#                     for handle in life_handels:
#                         handle.remove()
#                     if life0:
#                         life0.remove()
#                         life1.remove()
#
#                     life_handels = []
#
#                     life0, = ax.plot(times[idx_v[i0_m[0][idx0]]], fund_v[i0_m[0][idx0]], color='red', marker='o')
#                     life1, = ax.plot(times[idx_v[i1_m[0][idx1]]], fund_v[i1_m[0][idx1]], color='red', marker='o')
#
#                     xlims = ax.get_xlim()
#                     for ident in np.unique(ident_v[~np.isnan(ident_v)]):
#                         # embed()
#                         # quit()
#                         plot_times = times[idx_v[ident_v == ident]]
#                         plot_freqs = fund_v[ident_v == ident]
#
#                         # h, = ax.plot(times[idx_v[ident_v == ident]], fund_v[ident_v == ident ], color='k', marker = '.', markersize=5)
#                         h, = ax.plot(plot_times[(plot_times >= xlims[0] - 1)],
#                                      plot_freqs[(plot_times >= xlims[0] - 1)], color='k', marker='.', markersize=5)
#                         life_handels.append(h)
#
#                         if times[idx_v[ident_v == ident]][-1] > xlims[1]:
#                             # xlim = ax.get_xlim()
#                             ax.set_xlim([xlims[0] + 10, xlims[1] + 10])
#
#                     fig.canvas.draw()
#
#         # sort_time += time.time()-t0
#         if plotted and np.min(ioi_field) in np.array(i0_m[0]):
#             for ident in ident_v[~np.isnan(ident_v)]:
#                 # ax[3].plot(times[idx_v[(ident_v == ident) & (idx_v <= idx_v[np.min(ioi_field)]) ]], fund_v[(ident_v == ident)  & (idx_v <= idx_v[np.min(ioi_field)])], marker='.', color=np.random.rand(3))
#                 ax[4].plot(times[idx_v[(ident_v == ident) & (idx_v <= idx_v[np.min(ioi_field)]) ]], fund_v[(ident_v == ident)  & (idx_v <= idx_v[np.min(ioi_field)])], marker='.', color=np.random.rand(3))
#                 # ax[5].plot(times[idx_v[(ident_v == ident) & (idx_v < idx_v[np.min(ioi_field)]) ]], fund_v[(ident_v == ident)  & (idx_v < idx_v[np.min(ioi_field)])], marker='.', color=np.random.rand(3))
#                 ax[5].plot(times[idx_v[ident_v == ident]], fund_v[ident_v == ident], marker='.', color=np.random.rand(3))
#
#             ax[3].plot(times[idx_v[ioi_field[0]]], fund_v[ioi_field[0]], color='green', marker='o', alpha = 0.5)
#             ax[4].plot(times[idx_v[ioi_field[0]]], fund_v[ioi_field[0]], color='green', marker='o', alpha = 0.5)
#             ax[5].plot(times[idx_v[ioi_field[0]]], fund_v[ioi_field[0]], color='green', marker='o', alpha = 0.5)
#
#             ax[3].plot(times[idx_v[ioi_field[1]]], fund_v[ioi_field[1]], color='orange', marker='o', alpha = 0.5)
#             ax[4].plot(times[idx_v[ioi_field[1]]], fund_v[ioi_field[1]], color='orange', marker='o', alpha = 0.5)
#             ax[5].plot(times[idx_v[ioi_field[1]]], fund_v[ioi_field[1]], color='orange', marker='o', alpha = 0.5)
#
#             ax[3].plot(times[idx_v[ioi_field[2]]], fund_v[ioi_field[2]], color='red', marker='o', alpha = 0.5)
#             ax[4].plot(times[idx_v[ioi_field[2]]], fund_v[ioi_field[2]], color='red', marker='o', alpha = 0.5)
#             ax[5].plot(times[idx_v[ioi_field[2]]], fund_v[ioi_field[2]], color='red', marker='o', alpha = 0.5)
#                 # print('im here')
#             return fund_v, ident_v, idx_v, sign_v, a_error_distribution, f_error_distribution, idx_of_origin_v
#
#         i0_m.pop(0)
#         i1_m.pop(0)
#         error_cube.pop(0)
#
#         i0_v = np.arange(len(idx_v))[(idx_v == cube_app_idx) & (fund_v >= freq_lims[0]) & (fund_v <= freq_lims[1])]  # indices of fundamtenals to assign
#         i1_v = np.arange(len(idx_v))[(idx_v > cube_app_idx) & (idx_v <= (cube_app_idx + idx_comp_range)) & (fund_v >= freq_lims[0]) & (fund_v <= freq_lims[1])]  # indices of possible targets
#
#         i0_m.append(i0_v)
#         i1_m.append(i1_v)
#
#         if len(i0_v) == 0 or len(i1_v) == 0:  # if nothing to assign or no targets continue
#             error_cube.append(np.array([[]]))
#
#         else:
#             try:
#                 error_matrix = np.full((len(i0_v), len(i1_v)), np.nan)
#             except:
#                 error_matrix = np.zeros((len(i0_v), len(i1_v))) / 0.
#
#             for enu0 in range(len(fund_v[i0_v])):
#                 if fund_v[i0_v[enu0]] < low_freq_th or fund_v[i0_v[enu0]] > high_freq_th:  # freq to assigne out of tracking range
#                     continue
#
#                 for enu1 in range(len(fund_v[i1_v])):
#                     if fund_v[i1_v[enu1]] < low_freq_th or fund_v[i1_v[enu1]] > high_freq_th:  # target freq out of tracking range
#                         continue
#                     if np.abs(fund_v[i0_v[enu0]] - fund_v[i1_v[enu1]]) >= freq_tolerance:  # freq difference to high
#                         continue
#
#                     a_error = np.sqrt(
#                         np.sum([(sign_v[i0_v[enu0]][j] - sign_v[i1_v[enu1]][j]) ** 2 for j in range(n_channels)]))
#                     f_error = np.abs(fund_v[i0_v[enu0]] - fund_v[i1_v[enu1]])
#                     t_error = 1. * np.abs(idx_v[i0_v[enu0]] - idx_v[i1_v[enu1]]) / dps
#
#                     error = estimate_error(a_error, f_error, t_error, a_error_distribution, f_error_distribution)
#                     error_matrix[enu0, enu1] = np.sum(error)
#             error_cube.append(error_matrix)
#
#         cube_app_idx += 1
#     ident_v = clean_up(fund_v, ident_v, idx_v, times)
#
#     # plt.close()
#     # fig, ax = plt.subplots()
#     #
#     # uni_t = np.unique(all_false_dt)
#     # uni_a_rel_false = [np.array(all_rel_a_error_false)[np.array(all_false_dt) == t] for t in uni_t]
#     # uni_a_rel_true = [np.array(all_rel_a_error_true)[np.array(all_true_dt) == t] for t in uni_t]
#     #
#     # mean_false = [np.mean(uni_a_rel_false[i]) for i in range(len(uni_a_rel_false))]
#     # std_false =  [np.std(uni_a_rel_false[i]) for i in range(len(uni_a_rel_false))]
#     # mean_true = [np.mean(uni_a_rel_true[i]) for i in range(len(uni_a_rel_true))]
#     # std_true = [np.std(uni_a_rel_true[i]) for i in range(len(uni_a_rel_true))]
#     #
#     # ax.plot(uni_t, mean_false, color='red')
#     # ax.fill_between(uni_t, np.array(mean_false)+np.array(std_false), np.array(mean_false)-np.array(std_false), color='red', alpha=0.3)
#     # ax.plot(uni_t, mean_true, color='green')
#     # ax.fill_between(uni_t, np.array(mean_true)+np.array(std_true), np.array(mean_true)-np.array(std_true), color='green', alpha=0.3)
#
#     # embed()
#     # quit()
#
#     return fund_v, ident_v, idx_v, sign_v, a_error_distribution, f_error_distribution, idx_of_origin_v


def add_tracker_config(cfg, data_snippet_secs=15., nffts_per_psd=1, fresolution=.25, overlap_frac=.95,
                       freq_tolerance=10., rise_f_th=0.5, prim_time_tolerance=1., max_time_tolerance=10., f_th=2.):
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
    cfg.add('MaxTimeTolerance', max_time_tolerance, 'min',
            'Time tolerance between the occurrance of two fishes to join them.')
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


def load_matfile(data_file):
    """
    loads matlab files trying two possible methodes.

    Parameters
    ----------
    data_file: str
        datapath.

    Returns
    -------
    data: array
        nd-array with the recorded data. Dimensions vary with the used  number of recording electrodes.
    samplerate: int
        samplerate of the data

    """
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


def include_progress_bar(loop_v, loop_end, taskname='', next_message=0.00):
    """
    creates based on the progress of a loop a progressbar in a linux shell-

    Parameters
    ----------
    loop_v: int
        current loop variable.
    loop_end: int
        last loop.
    taskname: str
        tasename.
    next_message: float
        the proportion when the next message shall be posted.

    Returns
    -------
    next_message: float
        last next_message + 0.05

    """
    if len(taskname) > 30 or taskname == '':
        taskname = '        random task         '  # 30 characters
    else:
        taskname = ' ' * (30 - len(taskname)) + taskname

    if (1. * loop_v / loop_end) >= next_message:
        next_message = ((1. * loop_v / loop_end) // 0.05) * 0.05 + 0.05

        if next_message >= 1.:
            bar = '[' + 20 * '=' + ']'
            sys.stdout.write('\r' + bar + taskname)
            sys.stdout.flush()
        else:
            bar_factor = (1. * loop_v / loop_end) // 0.05
            bar = '[' + int(bar_factor) * '=' + (20 - int(bar_factor)) * ' ' + ']'
            sys.stdout.write('\r' + bar + taskname)
            sys.stdout.flush()

    return next_message


def get_spectrum_funds_amp_signature(data, samplerate, channels, data_snippet_idxs, start_time, end_time,
                                     fresolution=0.5,
                                     overlap_frac=.9, nffts_per_psd=2, comp_min_freq=0., comp_max_freq=2000.,
                                     plot_harmonic_groups=False,
                                     create_plotable_spectrogram=False, extract_funds_and_signature=True,
                                     create_fill_spec=False, noice_cancel=False, filename=None, **kwargs):
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

    if create_fill_spec:
        first_run = True
        pre_save_spectra = np.array([])

    while start_idx <= end_idx:
        if create_fill_spec:
            fill_spec_str = None
            memmap = None
            next_message = include_progress_bar(start_idx - init_idx + data_snippet_idxs, end_idx - init_idx,
                                                'get refill spec', next_message)
        else:
            if create_plotable_spectrogram and not extract_funds_and_signature:
                next_message = include_progress_bar(start_idx - init_idx + data_snippet_idxs, end_idx - init_idx,
                                                    'get plotable spec', next_message)
            elif not create_plotable_spectrogram and extract_funds_and_signature:
                next_message = include_progress_bar(start_idx - init_idx + data_snippet_idxs, end_idx - init_idx,
                                                    'extract fundamentals', next_message)
            else:
                next_message = include_progress_bar(start_idx - init_idx + data_snippet_idxs, end_idx - init_idx,
                                                    'extract funds and spec', next_message)

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

        if create_fill_spec:
            fresolution = 0.5
            overlap_frac = .8

        func = partial(spectrogram, samplerate=samplerate, fresolution=fresolution, overlap_frac=overlap_frac)

        if noice_cancel:
            # print('denoiced')
            denoiced_data = np.array([data[start_idx: start_idx + data_snippet_idxs, channel] for channel in channels])
            # print(denoiced_data.shape)
            mean_data = np.mean(denoiced_data, axis=0)
            # mean_data.shape = (len(mean_data), 1)
            denoiced_data -= mean_data

            a = pool.map(func, denoiced_data)
        # self.data = self.data - mean_data
        else:
            if len(np.shape(data)) == 1:
                a = pool.map(func, [data[start_idx: start_idx + data_snippet_idxs]])  # ret: spec, freq, time
            else:
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
                    f_mask = np.arange(len(plot_spectra))[
                        (plot_freqs >= y_borders[i]) & (plot_freqs < y_borders[i + 1])]

                    if len(t_mask) == 0 or len(f_mask) == 0:
                        continue

                    tmp_spectra[i, j] = np.max(plot_spectra[f_mask[:, None], t_mask])

        # psd and fish fundamentals frequency detection
        if extract_funds_and_signature:
            power = [np.array([]) for i in range(len(spec_times) - (int(nffts_per_psd) - 1))]

            for t in range(len(spec_times) - (int(nffts_per_psd) - 1)):
                power[t] = np.mean(comb_spectra[:, t:t + nffts_per_psd], axis=1)

            if plot_harmonic_groups:
                pool = multiprocessing.Pool(1)
            else:
                pool = multiprocessing.Pool(core_count // 2)
                # pool = multiprocessing.Pool(core_count - 1)
            func = partial(harmonic_groups, spec_freqs, **kwargs)
            a = pool.map(func, power)
            # pool.terminate()

            # get signatures
            # log_spectra = 10.0 * np.log10(np.array(spectra))
            log_spectra = decibel(np.array(spectra))
            for p in range(len(power)):
                tmp_fundamentals = fundamental_freqs(a[p][0])
                # tmp_fundamentals = a[p][0]
                fundamentals.append(tmp_fundamentals)

                if len(tmp_fundamentals) >= 1:
                    f_idx = np.array([np.argmin(np.abs(spec_freqs - f)) for f in tmp_fundamentals])
                    # embed()
                    # quit()
                    tmp_signatures = log_spectra[:, np.array(f_idx), p].transpose()
                else:
                    tmp_signatures = np.array([])

                signatures.append(tmp_signatures)

                # embed()
                # quit()
            pool.terminate()

        if create_fill_spec:
            # embed()
            # fill_spec_str = os.path.join(os.path.split(filename)[0], 'fill_spec.npy')
            fill_spec_str = os.path.join('/home/raab/analysis', 'fill_spec.npy')
            # fill_spec_str2 = os.path.join(os.path.split(filename)[0], 'fill_spec2.npy')
            if first_run:
                # embed()
                # quit()
                first_run = False
                # t0 = time.time()
                fill_spec = np.memmap(fill_spec_str, dtype='float', mode='w+',
                                      shape=(len(comb_spectra), len(tmp_times)), order='F')
                # print('create %.1f' % (time.time() - t0))
                # t0 = time.time()

                fill_spec[:, :] = comb_spectra
                # mem1 = False
            else:
                if len(pre_save_spectra) == 0:
                    pre_save_spectra = comb_spectra
                else:
                    pre_save_spectra = np.append(pre_save_spectra, comb_spectra, axis=1)
                    # embed()
                if np.shape(pre_save_spectra)[1] >= 500:
                    old_len = np.shape(fill_spec)[1]
                    # embed()
                    fill_spec = np.memmap(fill_spec_str, dtype='float', mode='r+', shape=(
                    np.shape(pre_save_spectra)[0], np.shape(pre_save_spectra)[1] + old_len), order='F')
                    # embed()
                    # quit()
                    fill_spec[:, old_len:] = pre_save_spectra
                    pre_save_spectra = np.array([])

        # print(len(fundamentals))
        # print(len(fundamentals))
        # print(fundamentals)
        non_overlapping_idx = (1 - overlap_frac) * nfft
        start_idx += int((len(spec_times) - nffts_per_psd + 1) * non_overlapping_idx)
        times = np.concatenate((times, tmp_times))

        if start_idx >= end_idx or last_run:
            break

    # print(len(fundamentals))
    # print(len(signatures))
    # embed()
    # quit()
    if create_fill_spec:
        # embed()
        # quit()
        # if mem1:
        #     fill_spec = np.memmap(fill_spec_str, dtype='float', mode='w+', shape=(np.shape(fill_spec2)[0], np.shape(fill_spec2)[1]))
        #     fill_spec[:, :] = fill_spec2
        #
        #     del fill_spec2
        # # embed()
        # # quit()
        # os.remove(fill_spec_str2)

        np.save(os.path.join(os.path.split(filename)[0], 'fill_spec_shape.npy'), np.array(np.shape(fill_spec)))
        return times, spec_freqs
    else:
        if create_plotable_spectrogram and not extract_funds_and_signature:
            return tmp_spectra, times

        elif extract_funds_and_signature and not create_plotable_spectrogram:
            return fundamentals, signatures, positions, times

        else:
            return fundamentals, signatures, positions, times, tmp_spectra


def grid_config_update(cfg):
    cfg['mains_freq'] = 0.
    cfg['max_fill_ratio'] = .75
    cfg['min_group_size'] = 2
    cfg['low_thresh_factor'] = 4
    cfg['high_thresh_factor'] = 8
    cfg['min_peak_width'] = 0.5
    cfg['max_peak_width_fac'] = 9.5

    return cfg


class Obs_tracker():
    def __init__(self, data, samplerate, start_time, end_time, channels, data_snippet_idxs, data_file, auto, fill_spec,
                 **kwargs):

        # write input into self.
        self.data = data
        self.auto = auto
        self.fill_spec = fill_spec
        self.data_file = data_file
        self.samplerate = samplerate
        self.start_time = start_time
        self.end_time = end_time
        if self.end_time < 0.0:
            self.end_time = len(self.data) / self.samplerate

        # embed()
        # quit()
        self.channels = np.array(channels, dtype=int)
        if len(channels) == 64:
            self.grid_prop = (8, 8)
        elif len(channels) == 8:
            self.grid_prop = (4, 2)
        elif len(channels) == 16:
            self.grid_prop = (4, 4)
        elif len(channels) == 1:
            self.grid_prop = (1, 1)
        else:
            pass
        # embed()
        # quit()
        self.data_snippet_idxs = data_snippet_idxs
        self.kwargs = kwargs
        self.verbose = 0
        # embed()
        # quit()

        # primary tracking vectors
        self.fund_v = None
        self.ident_v = None
        self.last_ident_v = None
        self.idx_v = None
        self.sign_v = None
        self.original_sign_v = None
        self.f_error_dist = None
        self.a_error_dist = None

        self.idx_of_origin_v = None

        # plot spectrum
        self.fundamentals = None
        self.times = None
        self.tmp_spectra = None # ToDo: rename... what is this ?
        self.part_spectra = None #  ToDO: rename... what is this ?
        self.spec_shift = 0
        self.ps_handle = None

        self.current_task = None
        self.current_idx = None
        self.x_zoom_0 = None
        self.x_zoom_1 = None
        self.y_zoom_0 = None
        self.y_zoom_1 = None

        self.last_xy_lims = None

        self.live_tracking = False

        # task lists
        self.t_tasks = ['track_snippet', 'track_snippet_show', 'track_snippet_live'] # ToDo: rename tasks
        # ['track_snippet', 'track_snippet_show', 'track_snippet_live', 'plot_tmp_identities', 'check_tracking']

        self.c_tasks = ['cut_trace', 'connect_trace', 'group_connect', 'auto connect_traces', 'fill_trace', 'group_reassign']
        #  ['cut_trace', 'connect_trace', 'group_connect', 'auto connect_traces', 'fill_trace', 'group_reassign']

        self.d_tasks = ['delete_trace', 'group_delete', 'delete_noise']
        self.p_tasks = ['part_spec', 'show_powerspectum', 'hide_spectogram', 'show_spectogram', 'part_spec_from_file',
                        'show_fields']
        self.s_tasks = ['save_plot', 'save_traces']


        if self.fill_spec:
            snippet_start = self.start_time
            snippet_end = self.end_time

            fill_times, fill_freqs = get_spectrum_funds_amp_signature(self.data, self.samplerate, self.channels,
                                                                      self.data_snippet_idxs,
                                                                      snippet_start, snippet_end,
                                                                      create_plotable_spectrogram=False,
                                                                      extract_funds_and_signature=False,
                                                                      create_fill_spec=fill_spec,
                                                                      filename=self.data_file, **self.kwargs)

            np.save(os.path.join(os.path.split(self.data_file)[0], 'fill_times.npy'), fill_times)
            np.save(os.path.join(os.path.split(self.data_file)[0], 'fill_freqs.npy'), fill_freqs)
            print('finished')
            quit()

        if self.auto:
            self.main_fig = None
            self.main_ax = None
            self.track_snippet()
            self.save_traces()
            print('finished')
            quit()

        else:

            # create plot environment
            self.main_fig = plt.figure(facecolor='white', figsize=(55. / 2.54, 30. / 2.54))

            # main window
            self.main_fig.canvas.mpl_connect('key_press_event', self.keypress)
            self.main_fig.canvas.mpl_connect('button_press_event', self.buttonpress)
            self.main_fig.canvas.mpl_connect('button_release_event', self.buttonrelease)

            # keymap.fullscreen : f, ctrl+f       # toggling
            # keymap.home : h, r, home            # home or reset mnemonic
            # keymap.back : left, c, backspace    # forward / backward keys to enable
            # keymap.forward : right, v           #   left handed quick navigation
            # keymap.pan : p                      # pan mnemonic
            # keymap.zoom : o                     # zoom mnemonic
            # keymap.save : s                     # saving current figure
            # keymap.quit : ctrl+w, cmd+w         # close the current figure
            # keymap.grid : g                     # switching on/off a grid in current axes
            # keymap.yscale : l                   # toggle scaling of y-axes ('log'/'linear')
            # keymap.xscale : L, k                # toggle scaling of x-axes ('log'/'linear')
            # keymap.all_axes : a                 # enable all axes

            plt.rcParams['keymap.save'] = ''  # was s
            plt.rcParams['keymap.back'] = ''  # was c
            plt.rcParams['keymap.forward'] = ''
            plt.rcParams['keymap.yscale'] = ''
            plt.rcParams['keymap.pan'] = ''
            plt.rcParams['keymap.home'] = ''
            plt.rcParams['keymap.fullscreen'] = ''

            self.main_ax = self.main_fig.add_axes([0.1, 0.1, 0.8, 0.6])
            self.add_ax = [None, [None, None, None], [None, None]] # [extra plot, [field plots, ...], [error plots]]
            self.add_ax[0] = self.main_fig.add_axes([.6, .1, .3, .6])
            self.add_ax[0].set_visible(False)
            self.add_ax[1][0] = self.main_fig.add_axes([.55, .225, .2, .3])
            self.add_ax[1][0].set_visible(False)
            self.add_ax[1][1] = self.main_fig.add_axes([.775, .05, .2, .3])
            self.add_ax[1][1].set_visible(False)
            self.add_ax[1][2] = self.main_fig.add_axes([.775, .4, .2, .3])
            self.add_ax[1][2].set_visible(False)
            self.add_ax[2][0] = self.main_fig.add_axes([.6, .75, 0.15, 0.15])
            self.add_ax[2][0].set_visible(False)
            self.add_ax[2][1] = self.main_fig.add_axes([.8, .75, 0.15, 0.15])
            self.add_ax[2][1].set_visible(False)


            self.spec_img_handle = None

            self.tmp_plothandel_main = None  # red line
            self.add_tmp_plothandel = None  # red line
            self.trace_handles = []
            self.tmp_trace_handels = []

            self.life_trace_handles = []

            self.active_idx0 = None
            self.active_idx_handle0 = None
            self.active_idx1 = None
            self.active_idx_handle1 = None

            self.active_ident0 = None
            self.active_ident_handle0 = None
            self.active_ident1 = None
            self.active_ident_handle1 = None

            self.active_indices = []
            self.active_indices_handle = []
            self.ioi_field = [None, None, None]
            self.ioi_field_handle = [None, None, None]
            self.ioi_field_marker = [None, None, None]

            self.ioi_a_error_line = [[None, None], [None, None]]
            self.ioi_f_error_line = [[None, None], [None, None]]
            self.ioi_t_error_line = [[None, None], [None, None]]
            self.error_text = [None, None]

            # powerspectrum window and parameters
            # self.add_ax = None
            # self.add_ax = [None, [None, None, None], [None, None]]

            self.add_tmp_plothandel = []
            self.tmp_harmonics_plot = None
            self.all_peakf_dots = None
            self.good_peakf_dots = None

            self.active_harmonic = None

            # self.f_error_ax = None
            # self.a_error_ax = None
            # self.t_error_ax = None

            # get key options into plot
            self.text_handles_key = []
            self.text_handles_effect = []
            self.key_options()

            self.main_fig.canvas.draw()
            # print('i am in the main loop')

            # get prim spectrum and plot it...
            self.plot_spectrum()

            self.get_clock_time()

            plt.show()

    def key_options(self):
        # for i in range(len(self.text_handles_key)):
        for i, j in zip(self.text_handles_key, self.text_handles_effect):
            self.main_fig.texts.remove(i)
            self.main_fig.texts.remove(j)
        self.text_handles_key = []
        self.text_handles_effect = []

        if True:
            if self.current_task:
                t = self.main_fig.text(0.1, 0.925, 'task: ')
                t1 = self.main_fig.text(0.2, 0.925, '%s' % self.current_task)
                self.text_handles_key.append(t)
                self.text_handles_effect.append(t1)

            t = self.main_fig.text(0.1, 0.875, 'enter:')
            t1 = self.main_fig.text(0.15, 0.875, 'execute task')
            self.text_handles_key.append(t)
            self.text_handles_effect.append(t1)

            t = self.main_fig.text(0.1, 0.85, 'ctrl+t:')
            t1 = self.main_fig.text(0.15, 0.85, 'tracking tasks')
            self.text_handles_key.append(t)
            self.text_handles_effect.append(t1)

            t = self.main_fig.text(0.1, 0.825, 'c:')
            t1 = self.main_fig.text(0.15, 0.825, 'correction tasks')
            self.text_handles_key.append(t)
            self.text_handles_effect.append(t1)

            # t = self.main_fig.text(0.1, 0.8,  'e:')
            # t1 = self.main_fig.text(0.15, 0.8, 'embed')
            t = self.main_fig.text(0.1, 0.8, 'p:')
            t1 = self.main_fig.text(0.15, 0.8, 'spectral tasks')
            self.text_handles_key.append(t)
            self.text_handles_effect.append(t1)

            t = self.main_fig.text(0.1, 0.775, 's/l:')
            t1 = self.main_fig.text(0.15, 0.775, 'save/load arrays or plots')
            self.text_handles_key.append(t)
            self.text_handles_effect.append(t1)

            t = self.main_fig.text(0.1, 0.75, '(ctrl+)q:')
            t1 = self.main_fig.text(0.15, 0.75, 'close (all)/ powerspectrum')
            self.text_handles_key.append(t)
            self.text_handles_effect.append(t1)

            t = self.main_fig.text(0.1, 0.725, 'z')
            t1 = self.main_fig.text(0.15, 0.725, 'zoom')
            self.text_handles_key.append(t)
            self.text_handles_effect.append(t1)

        if self.add_ax[0]:
            # if self.current_task == 'part_spec' or self.current_task == 'track_snippet':
            #     pass
            if self.current_task in  ['track_snippet', 'track_snippet_show', 'track_snippet_live', 'show_powerspectum', 'update_hg']:
                if hasattr(self.fundamentals, '__len__'):
                    pass
                else:
                    t = self.main_fig.text(0.3, 0.85, '(ctrl+)1:')
                    t1 = self.main_fig.text(0.35, 0.85,
                                            '%.2f dB; rel. dB th for good Peaks' % (self.kwargs['high_threshold']))
                    self.text_handles_key.append(t)
                    self.text_handles_effect.append(t1)

                    t = self.main_fig.text(0.3, 0.825, '(ctrl+)2:')
                    t1 = self.main_fig.text(0.35, 0.825,
                                            '%.2f dB; rel. dB th for all Peaks' % (self.kwargs['low_threshold']))
                    self.text_handles_key.append(t)
                    self.text_handles_effect.append(t1)

                    t = self.main_fig.text(0.3, 0.8, '(ctrl+)3:')
                    t1 = self.main_fig.text(0.35, 0.8, '%.2f; low th fac' % (self.kwargs['low_thresh_factor']))
                    self.text_handles_key.append(t)
                    self.text_handles_effect.append(t1)

                    t = self.main_fig.text(0.3, 0.775, '(ctrl+)4:')
                    t1 = self.main_fig.text(0.35, 0.775, '%.2f; high th fac' % (self.kwargs['high_thresh_factor']))
                    self.text_handles_key.append(t)
                    self.text_handles_effect.append(t1)

                    t = self.main_fig.text(0.3, 0.75, '(ctrl+)5:')
                    t1 = self.main_fig.text(0.35, 0.75, '%.2f dB; min Peak width' % (self.kwargs['min_peak_width']))
                    self.text_handles_key.append(t)
                    self.text_handles_effect.append(t1)

                    t = self.main_fig.text(0.3, 0.725, '(ctrl+)6:')
                    t1 = self.main_fig.text(0.35, 0.725,
                                            '%.2f X fresolution; max Peak width' % (self.kwargs['max_peak_width_fac']))
                    self.text_handles_key.append(t)
                    self.text_handles_effect.append(t1)

                    t = self.main_fig.text(0.5, 0.85, '(ctrl+)7:')
                    t1 = self.main_fig.text(0.55, 0.85, '%.0f; min group size' % (self.kwargs['min_group_size']))
                    self.text_handles_key.append(t)
                    self.text_handles_effect.append(t1)

                    t = self.main_fig.text(0.5, 0.825, '(ctrl+)8:')
                    t1 = self.main_fig.text(0.55, 0.825,
                                            '%.1f; * fresolution = max dif of harmonics' % (self.kwargs['freq_tol_fac']))
                    self.text_handles_key.append(t)
                    self.text_handles_effect.append(t1)

                    t = self.main_fig.text(0.5, 0.8, '(ctrl+)9:')
                    t1 = self.main_fig.text(0.55, 0.8,
                                            '%.0f; max divisor to check subharmonics' % (self.kwargs['max_divisor']))
                    self.text_handles_key.append(t)
                    self.text_handles_effect.append(t1)

                    t = self.main_fig.text(0.5, 0.775, '(ctrl+)0:')
                    t1 = self.main_fig.text(0.55, 0.775, '%.0f; max freqs to fill in' % (self.kwargs['max_upper_fill']))
                    self.text_handles_key.append(t)
                    self.text_handles_effect.append(t1)

                    t = self.main_fig.text(0.5, 0.75, '(ctrl+)+:')
                    t1 = self.main_fig.text(0.55, 0.75, '%.0f; 1 group max double used peaks' % (
                    self.kwargs['max_double_use_harmonics']))
                    self.text_handles_key.append(t)
                    self.text_handles_effect.append(t1)

                    t = self.main_fig.text(0.5, 0.725, '(ctrl+)#:')
                    t1 = self.main_fig.text(0.55, 0.725,
                                            '%.0f; 1 Peak part of n groups' % (self.kwargs['max_double_use_count']))
                    self.text_handles_key.append(t)
                    self.text_handles_effect.append(t1)

        if self.current_task == 'part_spec':
            t = self.main_fig.text(0.3, 0.85, '(ctrl+)1:')
            t1 = self.main_fig.text(0.35, 0.85, '%.2f Hz; freuency resolution' % (self.kwargs['fresolution']))
            self.text_handles_key.append(t)
            self.text_handles_effect.append(t1)

            t = self.main_fig.text(0.3, 0.825, '(ctrl+)2:')
            t1 = self.main_fig.text(0.35, 0.825,
                                    '%.2f; overlap fraction of FFT windows' % (self.kwargs['overlap_frac']))
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
                                                                             1. - self.kwargs['overlap_frac'])))
            self.text_handles_key.append(t)
            self.text_handles_effect.append(t1)

        # if self.current_task == 'check_tracking':
        #     if self.active_fundamental0_0 and self.active_fundamental0_1:
        #         a_error = np.sqrt(np.sum([(self.sign_v[self.active_fundamental0_0][k] -
        #                                    self.sign_v[self.active_fundamental0_1][k]) ** 2
        #                                   for k in range(len(self.sign_v[self.active_fundamental0_0]))]))
        #
        #         f_error = np.abs(self.fund_v[self.active_fundamental0_0] - self.fund_v[self.active_fundamental0_1])
        #
        #         t_error = np.abs(self.times[self.idx_v[self.active_fundamental0_0]] - self.times[
        #             self.idx_v[self.active_fundamental0_1]])
        #
        #         error = estimate_error(a_error, f_error, t_error, self.a_error_dist, self.f_error_dist)
        #
        #         t = self.main_fig.text(0.3, 0.85, 'freq error:')
        #         t1 = self.main_fig.text(0.35, 0.85, '%.2f Hz (%.2f; %.2f); %.2f' % (
        #             f_error, self.fund_v[self.active_fundamental0_0], self.fund_v[self.active_fundamental0_1],
        #             1. * len(self.f_error_dist[self.f_error_dist < f_error]) / len(self.f_error_dist)))
        #         self.text_handles_key.append(t)
        #         self.text_handles_effect.append(t1)
        #
        #         t = self.main_fig.text(0.3, 0.825, 'amp. error:')
        #         t1 = self.main_fig.text(0.35, 0.825, '%.2f dB; %.2f' % (
        #         a_error, 1. * len(self.a_error_dist[self.a_error_dist < a_error]) / len(self.a_error_dist)))
        #         self.text_handles_key.append(t)
        #         self.text_handles_effect.append(t1)
        #
        #         t = self.main_fig.text(0.3, 0.8, 'time error')
        #         t1 = self.main_fig.text(0.35, 0.8, '%.2f s (%.2f, %.2f)' % (
        #         t_error, self.times[self.idx_v[self.active_fundamental0_0]],
        #         self.times[self.idx_v[self.active_fundamental0_1]]))
        #         self.text_handles_key.append(t)
        #         self.text_handles_effect.append(t1)
        #
        #         t = self.main_fig.text(0.3, 0.775, 'df / s')
        #         t1 = self.main_fig.text(0.35, 0.775, '%.2f s' % (f_error / t_error))
        #         self.text_handles_key.append(t)
        #         self.text_handles_effect.append(t1)
        #
        #         t = self.main_fig.text(0.3, 0.725, 'error value')
        #         t1 = self.main_fig.text(0.35, 0.725, '%.3f' % (np.sum(error)))
        #         self.text_handles_key.append(t)
        #         self.text_handles_effect.append(t1)
        #
        #     if self.active_fundamental1_0 and self.active_fundamental1_1:
        #         a_error = np.sqrt(np.sum([(self.sign_v[self.active_fundamental1_0][k] -
        #                                    self.sign_v[self.active_fundamental1_1][k]) ** 2
        #                                   for k in range(len(self.sign_v[self.active_fundamental1_0]))]))
        #
        #         f_error = np.abs(self.fund_v[self.active_fundamental1_0] - self.fund_v[self.active_fundamental1_1])
        #
        #         t_error = np.abs(self.times[self.idx_v[self.active_fundamental1_0]] - self.times[
        #             self.idx_v[self.active_fundamental1_1]])
        #
        #         error = estimate_error(a_error, f_error, t_error, self.a_error_dist, self.f_error_dist)
        #
        #         t = self.main_fig.text(0.5, 0.85, 'freq error:')
        #         t1 = self.main_fig.text(0.55, 0.85, '%.2f Hz (%.2f; %.2f); %.2f' % (
        #             f_error, self.fund_v[self.active_fundamental1_0], self.fund_v[self.active_fundamental1_1],
        #             1. * len(self.f_error_dist[self.f_error_dist < f_error]) / len(self.f_error_dist)))
        #         self.text_handles_key.append(t)
        #         self.text_handles_effect.append(t1)
        #
        #         t = self.main_fig.text(0.5, 0.825, 'amp. error:')
        #         t1 = self.main_fig.text(0.55, 0.825, '%.2f dB; %.2f' % (
        #         a_error, 1. * len(self.a_error_dist[self.a_error_dist < a_error]) / len(self.a_error_dist)))
        #         self.text_handles_key.append(t)
        #         self.text_handles_effect.append(t1)
        #
        #         t = self.main_fig.text(0.5, 0.8, 'time error')
        #         t1 = self.main_fig.text(0.55, 0.8, '%.2f s (%.2f; %.2f)' % (
        #         t_error, self.times[self.idx_v[self.active_fundamental1_0]],
        #         self.times[self.idx_v[self.active_fundamental1_1]]))
        #         self.text_handles_key.append(t)
        #         self.text_handles_effect.append(t1)
        #
        #         t = self.main_fig.text(0.5, 0.775, 'df / s')
        #         t1 = self.main_fig.text(0.55, 0.775, '%.2f s' % (f_error / t_error))
        #         self.text_handles_key.append(t)
        #         self.text_handles_effect.append(t1)
        #
        #         t = self.main_fig.text(0.5, 0.725, 'error value')
        #         t1 = self.main_fig.text(0.55, 0.725, '%.3f' % (np.sum(error)))
        #         self.text_handles_key.append(t)
        #         self.text_handles_effect.append(t1)

    def keypress(self, event):
        self.key_options()
        # self.main_ax.set_position([.1, .1, .8, .6])
        # self.add_ax[1][0].set_visible(False)
        # self.add_ax[1][1].set_visible(False)
        # self.add_ax[1][2].set_visible(False)


        if event.key == 'm':
            self.current_task = 'method_figure'

        if event.key == 'ctrl+s':
            self.current_task = 'saving traces ... please wait'
            self.key_options()
            self.main_fig.canvas.draw()

            self.save_traces()

            self.current_task = None
            self.key_options()
            self.main_fig.canvas.draw()

        if event.key == 'ctrl+backspace':
            self.ident_v = self.last_ident_v
            self.plot_traces(clear_traces=True)

        if event.key == 'backspace':
            if hasattr(self.last_xy_lims, '__len__'):
                self.main_ax.set_xlim(self.last_xy_lims[0][0], self.last_xy_lims[0][1])
                self.main_ax.set_ylim(self.last_xy_lims[1][0], self.last_xy_lims[1][1])
                if self.add_ax:
                    self.add_ax[0].set_ylim(self.last_xy_lims[1])
                self.get_clock_time()
        if event.key == 'up':
            ylims = self.main_ax.get_ylim()
            self.main_ax.set_ylim(ylims[0] + 0.5 * (ylims[1] - ylims[0]), ylims[1] + 0.5 * (ylims[1] - ylims[0]))
            if self.add_ax:
                self.add_ax[0].set_ylim(ylims[0] + 0.5 * (ylims[1] - ylims[0]), ylims[1] + 0.5 * (ylims[1] - ylims[0]))

        if event.key == 'down':
            ylims = self.main_ax.get_ylim()
            self.main_ax.set_ylim(ylims[0] - 0.5 * (ylims[1] - ylims[0]), ylims[1] - 0.5 * (ylims[1] - ylims[0]))
            if self.add_ax:
                self.add_ax[0].set_ylim(ylims[0] - 0.5 * (ylims[1] - ylims[0]), ylims[1] - 0.5 * (ylims[1] - ylims[0]))

        if event.key == 'right':
            xlims = self.main_ax.get_xlim()[:]
            self.main_ax.set_xlim(xlims[0] + 0.5 * (xlims[1] - xlims[0]), xlims[1] + 0.5 * (xlims[1] - xlims[0]))
            self.get_clock_time()

        if event.key == 'left':
            xlims = self.main_ax.get_xlim()[:]
            self.main_ax.set_xlim(xlims[0] - 0.5 * (xlims[1] - xlims[0]), xlims[1] - 0.5 * (xlims[1] - xlims[0]))
            self.get_clock_time()

        if event.key in 'b':
            self.current_task = 'save_plot'

        if event.key in 'h':
            self.current_task = None

            if self.main_ax:
                self.main_ax.set_xlim([self.start_time, self.end_time])
                self.main_ax.set_ylim([0, 2000])
            if self.add_ax:
                self.add_ax[0].set_ylim([0, 2000])

            if hasattr(self.part_spectra, '__len__'):
                # self.main_fig.delaxes(self.main_ax)
                # self.main_ax = self.main_fig.add_axes([.1, .1, .8, .6])
                self.spec_img_handle.remove()
                self.spec_img_handle = self.main_ax.imshow(decibel(self.tmp_spectra)[::-1],
                                                           extent=[self.start_time, self.end_time, 0, 2000],
                                                           aspect='auto', alpha=0.7, cmap='jet',
                                                           interpolation='gaussian')
                self.main_ax.set_xlim([self.start_time, self.end_time])
                self.main_ax.set_ylim([0, 2000])
                self.main_ax.set_xlabel('time')
                self.main_ax.set_ylabel('frequency [Hz]')

            self.get_clock_time()

        if event.key in 'e':
            embed()

        if event.key == 'ctrl+e':
            for ident in np.unique(self.ident_v[~np.isnan(self.ident_v)]):
                if len(self.ident_v[self.ident_v == ident]) <= 1000:
                    self.main_ax.plot(self.times[self.idx_v[self.ident_v == ident]], self.fund_v[self.ident_v == ident],
                                      marker='o', markersize=10, alpha=0.3, color='red')
            self.main_fig.canvas.draw()

        if event.key in 'd':
            self.current_task = self.d_tasks[0]
            self.d_tasks = np.roll(self.d_tasks, -1)

        if event.key in 'p':
            self.current_task = self.p_tasks[0]
            self.p_tasks = np.roll(self.p_tasks, -1)
            if self.current_task == 'show_fields':
                self.main_ax.set_position([.1, .1, .4, .6])
                self.add_ax[0].set_visible(False)
                self.add_ax[1][0].set_visible(True)
                self.add_ax[1][1].set_visible(True)
                self.add_ax[1][2].set_visible(True)
            elif self.current_task == 'show_powerspectum':
                self.main_ax.set_position([.1, .1, .6, .6])
                self.add_ax[0].set_visible(True)
                self.add_ax[1][0].set_visible(False)
                self.add_ax[1][1].set_visible(False)
                self.add_ax[1][2].set_visible(False)
            else:
                self.main_ax.set_position([.1, .1, .8, .6])
                self.add_ax[0].set_visible(False)
                self.add_ax[1][0].set_visible(False)
                self.add_ax[1][1].set_visible(False)
                self.add_ax[1][2].set_visible(False)


        if event.key == 'ctrl+t':
            self.current_task = self.t_tasks[0]
            self.t_tasks = np.roll(self.t_tasks, -1)

        if event.key == 'c':
            self.current_task = self.c_tasks[0]
            self.c_tasks = np.roll(self.c_tasks, -1)

        if event.key == 'f':
            self.current_task = 'fish_hist'

        if event.key == 'ctrl+q':
            plt.close(self.main_fig)
            # self.main_fig.close()
            return

        if event.key == 'v':
            self.verbose += 1
            # print(self.verbose)
            print('verbosity: %.0f' % self.verbose)
        if event.key == 'ctrl+v':
            self.verbose -= 1
            print('verbosity: %.0f' % self.verbose)

        if event.key in 'q' and self.add_ax[0]:
            # self.main_fig.delaxes(self.add_ax)
            # self.add_ax = None
            self.main_ax.set_position([.1, .1, .8, .6])
            self.add_ax[0].set_visible(False)
            self.add_ax[1][0].set_visible(False)
            self.add_ax[1][1].set_visible(False)
            self.add_ax[1][2].set_visible(False)
            self.add_tmp_plothandel = []
            self.all_peakf_dots = None
            self.good_peakf_dots = None
            if hasattr(self.a_error_dist, '__len__') and hasattr(self.f_error_dist, '__len__'):
                self.plot_error()

        if event.key in 'z':
            self.current_task = 'zoom'

        if event.key in 's':
            self.current_task = self.s_tasks[0]
            self.s_tasks = np.roll(self.s_tasks, -1)
            # self.current_task = 'part_spec'

        if event.key in 'l':
            self.current_task = 'load_traces'

        if self.current_task == 'part_spec_from_file':
            if event.key == '+':
                self.spec_shift += 1
            if event.key == '-':
                self.spec_shift -= 1
            print(self.spec_shift)

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
                self.kwargs['overlap_frac'] -= 0.01

            if event.key == 'ctrl+2':
                self.kwargs['overlap_frac'] += 0.01

            if event.key == '3':
                self.kwargs['nffts_per_psd'] -= 1

            if event.key == 'ctrl+3':
                self.kwargs['nffts_per_psd'] += 1

        else:
            if self.add_ax[0] and not self.current_task == 'delete_noise':
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
                    self.kwargs['low_thresh_factor'] -= 1
                    self.kwargs['low_threshold'] = 0.
                    self.kwargs['high_threshold'] = 0.
                    self.current_task = 'update_hg'
                if event.key == 'ctrl+3':
                    self.kwargs['low_thresh_factor'] += 1
                    self.kwargs['low_threshold'] = 0.
                    self.kwargs['high_threshold'] = 0.
                    self.current_task = 'update_hg'

                if event.key == '4':
                    self.kwargs['high_thresh_factor'] -= 1
                    self.kwargs['low_threshold'] = 0.
                    self.kwargs['high_threshold'] = 0.
                    self.current_task = 'update_hg'

                if event.key == 'ctrl+4':
                    self.kwargs['high_thresh_factor'] += 1
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

        if event.key == 'enter':
            if self.current_task == 'method_figure':
                self.method_figure()

            if self.current_task == 'fill_trace':
                if hasattr(self.part_spectra, '__len__'):
                    self.fill_trace()

            if self.current_task == 'part_spec_from_file':
                self.plot_spectrum(part_spec_from_file=True)

            if self.current_task == 'fish_hist':
                self.fish_hist()

            if self.current_task == 'group_connect':
                self.last_ident_v = np.copy(self.ident_v)
                self.group_connect()

                self.plot_traces(post_group_connection=True)

            if self.current_task == 'group_reassign':
                self.last_ident_v = np.copy(self.ident_v)
                self.group_reassign()

                self.plot_traces(post_group_reassign=True)

            if self.current_task == 'group_delete':
                self.last_ident_v = np.copy(self.ident_v)
                self.group_delete()
                # self.current_task = None
                # self.plot_traces(clear_traces=True)
                self.plot_traces(post_group_delete=True)

            if self.current_task == 'auto connect_traces':
                self.ident_v = auto_connect_traces(self.fund_v, self.idx_v, self.ident_v, self.times)
                self.current_task = None
                self.plot_traces(clear_traces=True)

            if self.current_task == 'hide_spectogram':
                if self.spec_img_handle:
                    self.spec_img_handle.remove()
                    self.spec_img_handle = None

            if self.current_task == 'show_spectogram':
                if hasattr(self.tmp_spectra, '__len__'):
                    if self.spec_img_handle:
                        self.spec_img_handle.remove()
                    self.spec_img_handle = self.main_ax.imshow(decibel(self.tmp_spectra)[::-1],
                                                               extent=[self.start_time, self.end_time, 0, 2000],
                                                               aspect='auto', alpha=0.7, cmap='jet',
                                                               interpolation='gaussian')
                    self.main_ax.set_xlabel('time', fontsize=12)
                    self.main_ax.set_ylabel('frequency [Hz]', fontsize=12)

            if self.current_task == 'delete_noise':
                self.main_ax.set_position([.1, .1, .6, .6])
                self.add_ax[0].set_visible(True)
                self.add_ax[1][0].set_visible(False)
                self.add_ax[1][1].set_visible(False)
                self.add_ax[1][2].set_visible(False)

                if self.active_indices_handle:
                    self.last_ident_v = np.copy(self.ident_v)
                    self.ident_v[self.active_indices] = np.nan
                    self.active_indices_handle.remove()
                    self.active_indices_handle = None
                    # self.plot_traces(clear_traces=True)
                    self.plot_traces(post_group_delete=True)

                min_x, max_x = self.main_ax.get_xlim()
                min_y, max_y = self.main_ax.get_ylim()
                self.iois = np.arange(len(self.fund_v))[
                    (self.times[self.idx_v] >= min_x) & (self.times[self.idx_v] < max_x) & (self.fund_v >= min_y) & (
                                self.fund_v < max_y)]
                self.iois = self.iois[~np.isnan(self.ident_v[self.iois])]
                self.min_max = np.max(self.original_sign_v[self.iois], axis=1) - np.min(self.original_sign_v[self.iois], axis=1)

                if self.add_ax[0]:
                    # self.main_fig.delaxes(self.add_ax)
                    # self.add_ax = None
                    self.add_tmp_plothandel = []
                    self.all_peakf_dots = None
                    self.good_peakf_dots = None
                    # self.main_ax.set_position([.1, .1, .8, .6])
                # self.main_ax.set_position([.1, .1, .5, .6])
                # self.add_ax[0] = self.main_fig.add_axes([.6, .1, .3, .6])
                # self.ps_ax.set_yticks([])
                self.add_ax[0].yaxis.tick_right()
                self.add_ax[0].yaxis.set_label_position("right")
                self.add_ax[0].set_ylabel('frequency [Hz]', fontsize=12)
                self.add_ax[0].set_xlabel('max - min power', fontsize=12)
                self.ps_handle, = self.add_ax[0].plot(self.min_max, self.fund_v[self.iois], '.')
                self.add_ax[0].set_ylim(self.main_ax.get_ylim())

            if self.current_task == 'load_traces':
                self.load_trace()
                self.current_task = None

            if self.current_task == 'save_traces':
                self.save_traces()
                self.current_task = None

            if self.current_task == 'connect_trace':
                if self.active_ident_handle0 and self.active_ident_handle1:
                    self.last_ident_v = np.copy(self.ident_v)
                    self.connect_trace()

            if self.current_task == 'cut_trace':
                if self.active_ident0 and self.active_idx0:
                    self.last_ident_v = np.copy(self.ident_v)
                    self.cut_trace()

            if self.current_task == 'delete_trace':
                if self.active_ident_handle0:
                    self.last_ident_v = np.copy(self.ident_v)
                    self.delete_trace()

            if self.current_task == 'save_plot':
                self.current_task = None
                self.save_plot()

            if self.current_task == 'show_powerspectum':
                if self.tmp_plothandel_main and self.ioi:
                    # self.current_task = None
                    self.plot_ps()

                else:
                    print('\nmissing data')

            if self.current_task == 'update_hg':
                self.current_task = None
                self.update_hg()

            if self.current_task == 'track_snippet_show':
                self.track_snippet()
                self.plot_error()
                # embed()

                # self.main_ax.set_position([.1, .1, .4, .6])
                # self.add_ax[1][0] = self.main_fig.add_axes([.55, .225, .2, .3])
                # self.add_ax[1][1] = self.main_fig.add_axes([.775, .05, .2, .3])
                # self.add_ax[1][2] = self.main_fig.add_axes([.775, .4, .2, .3])
                self.main_fig.canvas.draw()

            if self.current_task == 'track_snippet':
                self.current_task = None
                self.track_snippet()
                self.plot_error()

                self.main_fig.canvas.draw()
                ############################################
                show_a_diff = False
                if show_a_diff:
                    plt.close()

                    shift = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
                    s_ids = []
                    ns_ids = []
                    fig, ax = plt.subplots()
                    for s in shift:
                        s_id = []
                        ns_id = []
                        for i in np.unique(self.idx_v)[:-s]:
                            idents = self.ident_v[self.idx_v == i]
                            c_idents = self.ident_v[self.idx_v == i + s]
                            for id in idents[~np.isnan(idents)]:
                                for c_id in c_idents[~np.isnan(c_idents)]:
                                    s0 = self.sign_v[(self.idx_v == i) & (self.ident_v == id)]
                                    s1 = self.sign_v[(self.idx_v == i + s) & (self.ident_v == c_id)]
                                    if id == c_id:
                                        s_id.append(np.sqrt(np.sum((s0 - s1) ** 2)))
                                    else:
                                        ns_id.append(np.sqrt(np.sum((s0 - s1) ** 2)))
                        s_ids.append(s_id)
                        ns_ids.append(ns_id)

                    # embed()
                    # quit()
                    ax.boxplot(s_ids, positions=np.arange(len(s_ids)) * 2 - 1, sym='')
                    ax.boxplot(ns_ids, positions=np.arange(len(ns_ids)) * 2, sym='')
                    plt.show()

            if self.current_task == 'track_snippet_live':
                self.current_task = None
                self.live_tracking = True
                self.track_snippet()
                self.live_tracking = False

            if self.current_task == 'part_spec':
                self.current_task = None
                self.plot_spectrum(part_spec=True)

        self.key_options()
        self.main_fig.canvas.draw()
        # plt.show()

    def buttonpress(self, event):
        if event.button == 2:

            self.ioi_field = [None, None, None]
            for m in self.ioi_field_marker:
                if m != None:
                    m.remove()
            self.ioi_field_marker = [None, None, None]

            for i in range(len(self.ioi_a_error_line)):
                for j in range(len(self.ioi_a_error_line[i])):
                    if self.ioi_a_error_line[i][j] != None:
                        self.ioi_a_error_line[i][j].remove()
                    if self.ioi_f_error_line[i][j] != None:
                        self.ioi_f_error_line[i][j].remove()

            self.ioi_a_error_line = [[None, None], [None, None]]
            self.ioi_f_error_line = [[None, None], [None, None]]
            if self.error_text[0] != None:
                self.error_text[0].remove()
            if self.error_text[1] != None:
                self.error_text[1].remove()
            self.error_text = [None, None]

            self.ioi_t_error_line = [[None, None], [None, None]]

            if event.inaxes != self.add_ax:
                if self.tmp_plothandel_main:
                    self.tmp_plothandel_main.remove()
                    self.tmp_plothandel_main = None

            if self.tmp_harmonics_plot:
                self.tmp_harmonics_plot.remove()
                self.tmp_harmonics_plot = None
                self.active_harmonic = None

                if self.add_ax[0]:
                    ylims = self.main_ax.get_ylim()
                    self.add_ax[0].set_ylim([ylims[0], ylims[1]])

            if len(self.active_indices) > 0:
                self.active_indices = []

            if self.active_indices_handle:
                self.active_indices_handle.remove()
                self.active_indices_handle = None

            # if self.active_fundamental0_0_handle:
            #     self.active_fundamental0_0 = None
            #     self.active_fundamental0_0_handle.remove()
            #     self.active_fundamental0_0_handle = None
            # if self.active_fundamental0_1_handle:
            #     self.active_fundamental0_1 = None
            #     self.active_fundamental0_1_handle.remove()
            #     self.active_fundamental0_1_handle = None
            # if self.active_fundamental1_0_handle:
            #     self.active_fundamental1_0 = None
            #     self.active_fundamental1_0_handle.remove()
            #     self.active_fundamental1_0_handle = None
            # if self.active_fundamental1_1_handle:
            #     self.active_fundamental1_1 = None
            #     self.active_fundamental1_1_handle.remove()
            #     self.active_fundamental1_1_handle = None
            if self.active_idx_handle0:
                self.active_idx0 = None
                self.active_idx_handle0.remove()
                self.active_idx_handle0 = None

            if self.active_idx0:
                self.active_idx0 = None
                self.active_ident0 = None
            if self.active_ident_handle0:
                self.active_ident_handle0.remove()
                self.active_ident_handle0 = None

            if self.active_idx1:
                self.active_idx1 = None
                self.active_ident1 = None
            if self.active_ident_handle1:
                self.active_ident_handle1.remove()
                self.active_ident_handle1 = None

        if event.inaxes == self.main_ax:
            if self.current_task == 'show_powerspectum':
                if event.button == 1:
                    x = event.xdata
                    self.ioi = np.argmin(np.abs(self.times - x))

                    y_lims = self.main_ax.get_ylim()
                    if self.tmp_plothandel_main:
                        self.tmp_plothandel_main.remove()
                    self.tmp_plothandel_main, = self.main_ax.plot([self.times[self.ioi], self.times[self.ioi]],
                                                                  [y_lims[0], y_lims[1]], color='red', linewidth=2)

            # if self.current_task == 'check_tracking' and hasattr(self.fundamentals, '__len__'):
            #
            #     if event.button == 1:
            #         if event.key == 'control':
            #             x = event.xdata
            #             y = event.ydata
            #
            #             idx_searched = np.argsort(np.abs(self.times - x))[0]
            #             fund_searched = self.fund_v[self.idx_v == idx_searched][
            #                 np.argsort(np.abs(self.fund_v[(self.idx_v == idx_searched)] - y))[0]]
            #             current_idx = \
            #             np.arange(len(self.fund_v))[(self.idx_v == idx_searched) & (self.fund_v == fund_searched)][0]
            #
            #             self.active_fundamental0_0 = current_idx
            #             if self.active_fundamental0_0_handle:
            #                 self.active_fundamental0_0_handle.remove()
            #             self.active_fundamental0_0_handle, = self.main_ax.plot(self.times[self.idx_v[current_idx]],
            #                                                                    self.fund_v[current_idx], 'o',
            #                                                                    color='red', markersize=4)
            #
            #             if self.active_fundamental0_1_handle:
            #                 self.active_fundamental0_1_handle.remove()
            #                 self.active_fundamental0_1_handle = None
            #                 self.active_fundamental0_1 = None
            #
            #             if ~np.isnan(self.idx_of_origin_v[current_idx]):
            #                 self.active_fundamental0_1 = self.idx_of_origin_v[current_idx]
            #                 self.active_fundamental0_1_handle, = self.main_ax.plot(
            #                     self.times[self.idx_v[self.active_fundamental0_1]],
            #                     self.fund_v[self.active_fundamental0_1], 'o', color='red', markersize=4)
            #
            #         else:
            #             x = event.xdata
            #             y = event.ydata
            #
            #             idx_searched = np.argsort(np.abs(self.times - x))[0]
            #             fund_searched = self.fund_v[self.idx_v == idx_searched][
            #                 np.argsort(np.abs(self.fund_v[(self.idx_v == idx_searched)] - y))[0]]
            #             current_idx = \
            #             np.arange(len(self.fund_v))[(self.idx_v == idx_searched) & (self.fund_v == fund_searched)][0]
            #
            #             self.active_fundamental0_1 = current_idx
            #             if self.active_fundamental0_1_handle:
            #                 self.active_fundamental0_1_handle.remove()
            #
            #             self.active_fundamental0_1_handle, = self.main_ax.plot(self.times[self.idx_v[current_idx]],
            #                                                                    self.fund_v[current_idx], 'o',
            #                                                                    color='red', markersize=4)
            #
            #     if event.button == 3:
            #         if event.key == 'control':
            #             x = event.xdata
            #             y = event.ydata
            #
            #             idx_searched = np.argsort(np.abs(self.times - x))[0]
            #             fund_searched = self.fund_v[self.idx_v == idx_searched][
            #                 np.argsort(np.abs(self.fund_v[(self.idx_v == idx_searched)] - y))[0]]
            #             current_idx = \
            #             np.arange(len(self.fund_v))[(self.idx_v == idx_searched) & (self.fund_v == fund_searched)][0]
            #
            #             self.active_fundamental1_0 = current_idx
            #
            #             if self.active_fundamental1_0_handle:
            #                 self.active_fundamental1_0_handle.remove()
            #             self.active_fundamental1_0_handle, = self.main_ax.plot(self.times[self.idx_v[current_idx]],
            #                                                                    self.fund_v[current_idx], 'o',
            #                                                                    color='green', markersize=4)
            #
            #             if self.active_fundamental1_1_handle:
            #                 self.active_fundamental1_1_handle.remove()
            #                 self.active_fundamental1_1_handle = None
            #                 self.active_fundamental1_1 = None
            #
            #             if ~np.isnan(self.idx_of_origin_v[current_idx]):
            #                 self.active_fundamental1_1 = self.idx_of_origin_v[current_idx]
            #                 self.active_fundamental1_1_handle, = self.main_ax.plot(
            #                     self.times[self.idx_v[self.active_fundamental1_1]],
            #                     self.fund_v[self.active_fundamental1_1], 'o', color='green', markersize=4)
            #
            #         else:
            #             x = event.xdata
            #             y = event.ydata
            #
            #             idx_searched = np.argsort(np.abs(self.times - x))[0]
            #             fund_searched = self.fund_v[self.idx_v == idx_searched][
            #                 np.argsort(np.abs(self.fund_v[(self.idx_v == idx_searched)] - y))[0]]
            #             current_idx = \
            #             np.arange(len(self.fund_v))[(self.idx_v == idx_searched) & (self.fund_v == fund_searched)][0]
            #
            #             self.active_fundamental1_1 = current_idx
            #
            #             if self.active_fundamental1_1_handle:
            #                 self.active_fundamental1_1_handle.remove()
            #
            #             self.active_fundamental1_1_handle, = self.main_ax.plot(self.times[self.idx_v[current_idx]],
            #                                                                    self.fund_v[current_idx], 'o',
            #                                                                    color='green', markersize=4)

            # if self.current_task == 'plot_tmp_identities':
            #     if event.button == 1:
            #         x = event.xdata
            #         y = event.ydata
            #
            #         t_idx = np.argsort(np.abs(self.times - x))[0]
            #         f_idx = np.argsort(np.abs(self.fund_v[self.idx_v == t_idx] - y))[0]
            #
            #         self.active_idx0 = np.arange(len(self.fund_v))[
            #             (self.idx_v == t_idx) & (self.fund_v == self.fund_v[self.idx_v == t_idx][f_idx])][0]
            #         if self.active_idx_handle0:
            #             self.active_idx_handle0.remove()
            #         # self.active_vec_idx_handle, = self.main_ax.plot(self.time[self.idx_v[t_idx]], self.fund_v[self.idx_v == t_idx][f_idx], 'o', color='red', markersize=4)
            #         self.active_idx_handle0, = self.main_ax.plot(self.times[t_idx], self.fund_v[self.active_idx0],
            #                                                         'o', color='red', markersize=4)

            if self.current_task in ['connect_trace', 'delete_trace', 'zoom', 'cut_trace', 'group_delete',
                                     'group_connect', 'group_reassign', 'track_snippet_show', 'show_fields']:

                if self.current_task in ['group_delete', 'group_connect', 'group_reassign']:
                    if self.active_indices_handle:
                        self.active_indices_handle.remove()
                        self.active_indices_handle = None

                self.x = (event.xdata, 0)
                self.y = (event.ydata, 0)
                # embed()
                if self.current_task == 'show_fields':
                    self.main_ax.set_position([.1, .1, .4, .6])
                    # self.add_ax[1][0] = self.main_fig.add_axes([.55, .225, .2, .3])
                    # self.add_ax[1][1] = self.main_fig.add_axes([.775, .05, .2, .3])
                    # self.add_ax[1][2] = self.main_fig.add_axes([.775, .4, .2, .3])
                    self.main_fig.canvas.draw()

            if self.current_task == 'cut_trace':
                if event.button == 3:
                    x = event.xdata

                    trace_idxs = self.idx_v[self.ident_v == self.active_ident0]
                    current_idx = np.arange(len(self.fund_v))[(self.ident_v == self.active_ident0)][np.argsort(np.abs(self.times[trace_idxs] - x))[0]]

                    self.active_idx0 = current_idx
                    if self.active_idx_handle0:
                        self.active_idx_handle0.remove()
                    self.active_idx_handle0, = self.main_ax.plot(self.times[self.idx_v[current_idx]],
                                                                 self.fund_v[current_idx], 'o', color='red',
                                                                 markersize=4)

        if self.add_ax[0] and event.inaxes == self.add_ax:
            if self.current_task == 'delete_noise':
                if event.button == 1:
                    x = event.xdata
                    self.noi = np.argmin(np.abs(self.times - x))

                    y_lims = self.add_ax[0].get_ylim()
                    if self.add_tmp_plothandel:
                        self.add_tmp_plothandel.remove()
                    self.add_tmp_plothandel, = self.add_ax[0].plot([x, x], [y_lims[0], y_lims[1]], color='red', linewidth=2)
                # print('wuff')
                self.active_indices = self.iois[self.min_max <= x]
                if self.active_indices_handle:
                    self.active_indices_handle.remove()
                self.active_indices_handle, = self.main_ax.plot(self.times[self.idx_v[self.active_indices]],
                                                                self.fund_v[self.active_indices], 'o', color='orange')


            else:
                if not self.active_harmonic:
                    self.active_harmonic = 1.

                if event.button == 1:
                    plot_power = decibel(self.power)
                    y = event.ydata
                    active_all_freq = self.all_peakf[:, 0][np.argsort(np.abs(self.all_peakf[:, 0] - y))][0]

                    plot_harmonics = np.arange(active_all_freq, 3000, active_all_freq)

                    if self.tmp_harmonics_plot:
                        self.tmp_harmonics_plot.remove()

                    self.tmp_harmonics_plot, = self.add_ax[0].plot(
                        np.ones(len(plot_harmonics)) * np.max(plot_power[self.freqs <= 3000.0]) + 10., plot_harmonics,
                        'o', color='k')

                    current_ylim = self.add_ax[0].get_ylim()
                    self.add_ax[0].set_ylim([current_ylim[0] + active_all_freq / self.active_harmonic,
                                          current_ylim[1] + active_all_freq / self.active_harmonic])
                    self.active_harmonic += 1

                if event.button == 3:
                    plot_power = decibel(self.power)
                    y = event.ydata
                    active_all_freq = self.all_peakf[:, 0][np.argsort(np.abs(self.all_peakf[:, 0] - y))][0]

                    plot_harmonics = np.arange(active_all_freq, 3000, active_all_freq)

                    if self.tmp_harmonics_plot:
                        self.tmp_harmonics_plot.remove()

                    self.tmp_harmonics_plot, = self.add_ax[0].plot(
                        np.ones(len(plot_harmonics)) * np.max(plot_power[self.freqs <= 3000.0]) + 10., plot_harmonics,
                        'o', color='k')

                    current_ylim = self.add_ax[0].get_ylim()
                    self.add_ax[0].set_ylim([current_ylim[0] - active_all_freq / self.active_harmonic,
                                          current_ylim[1] - active_all_freq / self.active_harmonic])
                    self.active_harmonic -= 1

        self.key_options()
        self.main_fig.canvas.draw()

    def buttonrelease(self, event):
        # if event.inaxes == self.main_ax:
        if self.current_task in ['group_delete', 'group_connect', 'group_reassign']:
            if event.button == 1:
                self.x = (self.x[0], event.xdata)
                self.y = (self.y[0], event.ydata)

                self.active_indices = np.arange(len(self.fund_v))[
                    (self.fund_v >= np.min(self.y)) & (self.fund_v < np.max(self.y)) & (
                            self.times[self.idx_v] >= np.min(self.x)) & (self.times[self.idx_v] < np.max(self.x))]
                self.active_indices = self.active_indices[~np.isnan(self.ident_v[self.active_indices])]

                self.active_indices_handle, = self.main_ax.plot(self.times[self.idx_v[self.active_indices]],
                                                                self.fund_v[self.active_indices], 'o',
                                                                color='orange')

        if self.current_task == 'show_fields':
            if event.button == 1:
                self.x = (self.x[0], event.xdata)
                self.y = (self.y[0], event.ydata)
                self.active_idx0 = np.arange(len(self.fund_v))[
                    (self.fund_v >= np.min(self.y)) & (self.fund_v < np.max(self.y)) & (
                            self.times[self.idx_v] >= np.min(self.x)) & (
                            self.times[self.idx_v] < np.max(self.x))]
                if len(self.active_idx0) > 0:
                    self.active_idx0 = self.active_idx0[~np.isnan(self.ident_v[self.active_idx0])][0]
                else:
                    self.active_idx0 = None

                if self.ioi_field[0] == None:
                    self.ioi_field[0] = self.active_idx0
                    self.ioi_field_handle[0] = self.add_ax[1][0].imshow(
                        self.sign_v[self.ioi_field[0]].reshape(self.grid_prop).transpose()[::-1], cmap='jet',
                        interpolation='gaussian')
                    self.ioi_field_marker[0], = self.main_ax.plot(self.times[self.idx_v[self.ioi_field[0]]],
                                                                  self.fund_v[self.ioi_field[0]], marker='o',
                                                                  color='green', markersize=5)

                elif self.ioi_field[1] == None:
                    self.ioi_field[1] = self.active_idx0
                    self.ioi_field_handle[1] = self.add_ax[1][1].imshow(
                        self.sign_v[self.ioi_field[1]].reshape(self.grid_prop).transpose()[::-1], cmap='jet',
                        interpolation='gaussian')
                    self.ioi_field_marker[1], = self.main_ax.plot(self.times[self.idx_v[self.ioi_field[1]]],
                                                                  self.fund_v[self.ioi_field[1]], marker='o',
                                                                  color='orange', markersize=5)
                    if hasattr(self.a_error_dist, '__len__') and hasattr(self.f_error_dist, '__len__'):
                        a_e = np.sqrt(np.sum((self.sign_v[self.ioi_field[0]] - self.sign_v[self.ioi_field[1]]) ** 2))
                        f_e = np.abs(self.fund_v[self.ioi_field[0]] - self.fund_v[self.ioi_field[1]])

                        rel_a_e = len(self.a_error_dist[self.a_error_dist <= a_e]) / len(self.a_error_dist)
                        # rel_f_e = len(self.f_error_dist[self.f_error_dist <= f_e]) / len(self.f_error_dist)
                        # rel_f_e = boltzmann(f_e, alpha=1, beta=0, x0=2.5, dx=.6)
                        rel_f_e = boltzmann(f_e, alpha=1, beta=0, x0=.25, dx=.15)

                        error = estimate_error(a_e, f_e, np.abs(
                            self.times[self.idx_v[self.ioi_field[1]]] - self.times[self.idx_v[self.ioi_field[0]]]),
                                               self.a_error_dist, self.f_error_dist)
                        self.error_text[0] = self.main_fig.text(.55, .1,
                                                                'a_error: %.2f; f_error: %.2f; t_error: %.2f (%.1f s) \ntotal_error: %.2f' % (
                                                                    error[0], error[1], error[2], np.abs(
                                                                        self.times[self.idx_v[self.ioi_field[1]]] -
                                                                        self.times[self.idx_v[self.ioi_field[0]]]),
                                                                    error[0] * 2. / 3 + error[1] * 1. / 3),
                                                                color='orange')

                        self.ioi_a_error_line[0][0], = self.add_ax[2][1].plot([a_e, a_e], [0, rel_a_e], color='orange')
                        self.ioi_a_error_line[0][1], = self.add_ax[2][1].plot([0, a_e], [rel_a_e, rel_a_e],
                                                                            color='orange')

                        self.ioi_f_error_line[0][0], = self.add_ax[2][0].plot([f_e, f_e], [0, rel_f_e], color='orange')
                        self.ioi_f_error_line[0][1], = self.add_ax[2][0].plot([0, f_e], [rel_f_e, rel_f_e],
                                                                            color='orange')

                else:
                    self.ioi_field[2] = self.active_idx0
                    self.ioi_field_handle[2] = self.add_ax[1][2].imshow(
                        self.sign_v[self.ioi_field[2]].reshape(self.grid_prop).transpose()[::-1], cmap='jet',
                        interpolation='gaussian')
                    self.ioi_field_marker[2], = self.main_ax.plot(self.times[self.idx_v[self.ioi_field[2]]],
                                                                  self.fund_v[self.ioi_field[2]], marker='o',
                                                                  color='red',
                                                                  markersize=5)

                    if hasattr(self.a_error_dist, '__len__') and hasattr(self.f_error_dist, '__len__'):
                        a_e = np.sqrt(np.sum((self.sign_v[self.ioi_field[0]] - self.sign_v[self.ioi_field[2]]) ** 2))
                        f_e = np.abs(self.fund_v[self.ioi_field[0]] - self.fund_v[self.ioi_field[2]])

                        rel_a_e = len(self.a_error_dist[self.a_error_dist <= a_e]) / len(self.a_error_dist)
                        # rel_f_e = len(self.f_error_dist[self.f_error_dist <= f_e]) / len(self.f_error_dist)
                        rel_f_e = boltzmann(f_e, alpha=1, beta=0, x0=.25, dx=.15)

                        error = estimate_error(a_e, f_e, np.abs(
                            self.times[self.idx_v[self.ioi_field[2]]] - self.times[self.idx_v[self.ioi_field[0]]]),
                                               self.a_error_dist, self.f_error_dist)
                        self.error_text[1] = self.main_fig.text(.55, .55,
                                                                'a_error: %.2f; f_error: %.2f; t_error: %.2f (%.1f s) \ntotal_error: %.2f' % (
                                                                    error[0], error[1], error[2], np.abs(
                                                                        self.times[self.idx_v[self.ioi_field[2]]] -
                                                                        self.times[self.idx_v[self.ioi_field[0]]]),
                                                                    error[0] * 2. / 3 + error[1] * 1. / 3), color='red')

                        self.ioi_a_error_line[1][0], = self.add_ax[2][1].plot([a_e, a_e], [0, rel_a_e], color='red')
                        self.ioi_a_error_line[1][1], = self.add_ax[2][1].plot([0, a_e], [rel_a_e, rel_a_e],
                                                                            color='red')

                        self.ioi_f_error_line[1][0], = self.add_ax[2][0].plot([f_e, f_e], [0, rel_f_e], color='red')
                        self.ioi_f_error_line[1][1], = self.add_ax[2][0].plot([0, f_e], [rel_f_e, rel_f_e],
                                                                            color='red')

        if self.current_task == 'delete_trace':
            if event.button == 1:
                self.x = (self.x[0], event.xdata)
                self.y = (self.y[0], event.ydata)

                self.active_idx0 = np.arange(len(self.fund_v))[
                    (self.fund_v >= np.min(self.y)) & (self.fund_v < np.max(self.y)) & (
                                self.times[self.idx_v] >= np.min(self.x)) & (
                                self.times[self.idx_v] < np.max(self.x))]
                if len(self.active_idx0) > 0:
                    self.active_idx0 = self.active_idx0[~np.isnan(self.ident_v[self.active_idx0])][0]
                else:
                    self.active_idx0 = None

                self.active_ident0 = self.ident_v[self.active_idx0]

                if self.active_ident_handle0:
                    self.active_ident_handle0.remove()

                self.active_ident_handle0, = self.main_ax.plot(
                    self.times[self.idx_v[self.ident_v == self.active_ident0]],
                    self.fund_v[self.ident_v == self.active_ident0], color='orange', alpha=0.7, linewidth=4)

        if self.current_task == 'cut_trace':
            if event.button == 1:
                self.x = (self.x[0], event.xdata)
                self.y = (self.y[0], event.ydata)

                self.active_idx0 = np.arange(len(self.fund_v))[
                    (self.fund_v >= np.min(self.y)) & (self.fund_v < np.max(self.y)) & (
                                self.times[self.idx_v] >= np.min(self.x)) & (
                                self.times[self.idx_v] < np.max(self.x))]
                if len(self.active_idx0) > 0:
                    self.active_idx0 = self.active_idx0[~np.isnan(self.ident_v[self.active_idx0])][0]
                else:
                    self.active_idx0 = None

                self.active_ident0 = self.ident_v[self.active_idx0]

                if self.active_ident_handle0:
                    self.active_ident_handle0.remove()

                self.active_ident_handle0, = self.main_ax.plot(
                    self.times[self.idx_v[self.ident_v == self.active_ident0]],
                    self.fund_v[self.ident_v == self.active_ident0], color='orange', alpha=0.7, linewidth=4)

        if self.current_task == 'connect_trace':
            if event.button == 1:
                self.x = (self.x[0], event.xdata)
                self.y = (self.y[0], event.ydata)

                self.active_idx0 = np.arange(len(self.fund_v))[
                    (self.fund_v >= np.min(self.y)) & (self.fund_v < np.max(self.y)) & (
                            self.times[self.idx_v] >= np.min(self.x)) & (self.times[self.idx_v] < np.max(self.x))]
                if len(self.active_idx0) > 0:
                    self.active_idx0 = self.active_idx0[~np.isnan(self.ident_v[self.active_idx0])][0]
                else:
                    self.active_idx0 = None

                self.active_ident0 = self.ident_v[self.active_idx0]

                if self.active_ident_handle0:
                    self.active_ident_handle0.remove()

                self.active_ident_handle0, = self.main_ax.plot(
                    self.times[self.idx_v[self.ident_v == self.active_ident0]],
                    self.fund_v[self.ident_v == self.active_ident0], color='green', alpha=0.7, linewidth=4)

            if event.button == 3:
                self.x = (self.x[0], event.xdata)
                self.y = (self.y[0], event.ydata)

                if self.active_ident0:
                    self.active_idx1 = np.arange(len(self.fund_v))[(self.fund_v >= np.min(self.y)) &
                                                                   (self.fund_v < np.max(self.y)) &
                                                                   (self.times[self.idx_v] >= np.min(self.x)) &
                                                                   (self.times[self.idx_v] < np.max(self.x)) &
                                                                   (self.ident_v != self.active_ident0)]
                    if len(self.active_idx1) > 0:
                        self.active_idx1 = self.active_idx1[~np.isnan(self.ident_v[self.active_idx1])][
                            0]
                    else:
                        self.active_idx1 = None

                    self.active_ident1 = self.ident_v[self.active_idx1]

                    if self.active_ident_handle1:
                        self.active_ident_handle1.remove()

                    self.active_ident_handle1, = self.main_ax.plot(
                        self.times[self.idx_v[self.ident_v == self.active_ident1]],
                        self.fund_v[self.ident_v == self.active_ident1], color='red', alpha=0.7, linewidth=4)

        if self.current_task == 'zoom':
            self.last_xy_lims = [self.main_ax.get_xlim(), self.main_ax.get_ylim()]

            self.x = (self.x[0], event.xdata)
            self.y = (self.y[0], event.ydata)

            if event.button == 1:
                self.main_ax.set_xlim(np.array(self.x)[np.argsort(self.x)])

            self.main_ax.set_ylim(np.array(self.y)[np.argsort(self.y)])
            if self.add_ax[0]:
                self.add_ax[0].set_ylim(np.array(self.y)[np.argsort(self.y)])

            # embed()
            self.get_clock_time()
            # embed()

        self.key_options()
        self.main_fig.canvas.draw()

    def method_figure(self):
        #          brown      purple    orange      dark blue  green      wine red   light blue
        colors = ['#BA2D22', '#53379B', '#F47F17', '#3673A4', '#AAB71B', '#DC143C', '#1E90FF']
        fs = 12
        detection_time_diff = self.times[1] - self.times[0]
        dps = 1. / detection_time_diff
        idx_comp_range = int(np.floor(dps * 10.))
        shift = np.arange(idx_comp_range) + 1

        s_ids = []
        s_idsf = []
        s_idse = []
        ns_ids = []
        ns_idsf = []
        ns_idse = []
        next_message = 0.0
        for s in shift:
            next_message = include_progress_bar(s, shift[-1], 'shift error fig', next_message)
            s_id = []
            s_idf = []
            s_ide = []
            ns_id = []
            ns_idf = []
            ns_ide = []
            for i in np.unique(self.idx_v)[:-s]:
                idents = self.ident_v[self.idx_v == i]
                c_idents = self.ident_v[self.idx_v == i + s]
                for id in idents[~np.isnan(idents)]:
                    for c_id in c_idents[~np.isnan(c_idents)]:
                        s0 = self.sign_v[(self.idx_v == i) & (self.ident_v == id)]
                        f0 = self.fund_v[(self.idx_v == i) & (self.ident_v == id)][0]
                        s1 = self.sign_v[(self.idx_v == i + s) & (self.ident_v == c_id)]
                        f1 = self.fund_v[(self.idx_v == i + s) & (self.ident_v == c_id)][0]
                        if np.abs(f0 - f1) > 10:
                            continue

                        if id == c_id:
                            s_id.append(np.sqrt(np.sum((s0 - s1) ** 2)))
                            s_idf.append(np.abs(f0 - f1))
                            e = estimate_error(s_id[-1], s_idf[-1], 0, self.a_error_dist, self.f_error_dist)
                            s_ide.append(np.sum(e))
                        else:
                            ns_id.append(np.sqrt(np.sum((s0 - s1) ** 2)))
                            ns_idf.append(np.abs(f0 - f1))
                            e = estimate_error(ns_id[-1], ns_idf[-1], 0, self.a_error_dist, self.f_error_dist)
                            ns_ide.append(np.sum(e))

            s_ids.append(s_id)
            s_idsf.append(s_idf)
            s_idse.append(s_ide)
            ns_ids.append(ns_id)
            ns_idsf.append(ns_idf)
            ns_idse.append(ns_ide)

        p_s_ids = np.array([np.percentile(s_ids[i], (0, 25, 50, 75, 100)) for i in range(len(s_ids))])
        p_ns_ids = np.array([np.percentile(ns_ids[i], (0, 25, 50, 75, 100)) for i in range(len(s_ids))])
        p_s_idsf = np.array([np.percentile(s_idsf[i], (0, 25, 50, 75, 100)) for i in range(len(s_ids))])
        p_ns_idsf = np.array([np.percentile(ns_idsf[i], (0, 25, 50, 75, 100)) for i in range(len(s_ids))])

        # temp identities and tracking
        if True:
            limitations = self.main_ax.get_xlim()
            limitations = [limitations[0] - 20, limitations[1] + 20]
            # limitations[1] += 20.
            min_freq = self.main_ax.get_ylim()[0] - 100
            max_freq = self.main_ax.get_ylim()[1] + 100

            shape = np.load(os.path.join(os.path.split(self.data_file)[0], 'fill_spec_shape.npy'))
            spec = np.memmap(os.path.join(os.path.split(self.data_file)[0], 'fill_spec.npy'), dtype='float', mode='r',
                             shape=tuple(shape), order='F')
            freqs = np.load(os.path.join(os.path.split(self.data_file)[0], 'fill_freqs.npy'))
            times = np.load(os.path.join(os.path.split(self.data_file)[0], 'fill_times.npy'))

            part_f_lims = np.arange(len(freqs))[(freqs >= min_freq) & (freqs <= max_freq)]
            part_t_lims = np.arange(len(times))[(times >= limitations[0]) & (times <= limitations[1])]
            part_spectra = spec[part_f_lims[0]:part_f_lims[-1] + 1, part_t_lims[0]:part_t_lims[-1] + 1]

            fig = plt.figure(facecolor='white', figsize=(20. / 2.54, 24. / 2.54))
            ax0 = fig.add_subplot(311)
            ax1 = fig.add_subplot(312)
            ax2 = fig.add_subplot(313)

            fig = plt.figure(facecolor='white', figsize=(20. / 2.54, 14. / 2.54))
            ax3 = fig.add_axes([.1, .55, .35, .35])
            ax4 = fig.add_axes([.55, .55, .35, .35])
            ax5 = fig.add_axes([.1, .1, .35, .35])
            ax6 = fig.add_axes([.55, .1, .35, .35])
            ax7 = fig.add_axes([.1, .0, .35, 1])
            ax8 = fig.add_axes([.55, .0, .35, 1])
            # ax3 = fig.add_subplot(221)
            # ax4 = fig.add_subplot(222)
            # ax5 = fig.add_subplot(223)
            # ax6 = fig.add_subplot(224)

            # ax = np.hstack(ax)
            ax = [ax0, ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8]
            # ax = np.array([ax2, ax1, ax0, ax3, ax4, ax5, ax22, ax11, ax00])

            cax = ax[0].imshow(decibel(part_spectra)[::-1], extent=[limitations[0], limitations[1], min_freq, max_freq],
                               aspect='auto', alpha=0.7, cmap='jet', interpolation='gaussian', zorder=0)
            ax[1].imshow(decibel(part_spectra)[::-1], extent=[limitations[0], limitations[1], min_freq, max_freq],
                         aspect='auto', alpha=0.7, cmap='jet', interpolation='gaussian', zorder=0)
            ax[2].imshow(decibel(part_spectra)[::-1], extent=[limitations[0], limitations[1], min_freq, max_freq],
                         aspect='auto', alpha=0.7, cmap='jet', interpolation='gaussian', zorder=0)
            cbar = fig.colorbar(cax, ax=[ax[0], ax[1], ax[2]])
            cbar.ax.set_ylabel('dB')

            ax[3].imshow(decibel(part_spectra)[::-1], extent=[limitations[0], limitations[1], min_freq, max_freq],
                         aspect='auto', alpha=0.5, cmap='Greys', interpolation='gaussian', zorder=0)
            ax[4].imshow(decibel(part_spectra)[::-1], extent=[limitations[0], limitations[1], min_freq, max_freq],
                         aspect='auto', alpha=0.5, cmap='Greys', interpolation='gaussian', zorder=0)
            ax[5].imshow(decibel(part_spectra)[::-1], extent=[limitations[0], limitations[1], min_freq, max_freq],
                         aspect='auto', alpha=0.5, cmap='Greys', interpolation='gaussian', zorder=0)
            ax[6].imshow(decibel(part_spectra)[::-1], extent=[limitations[0], limitations[1], min_freq, max_freq],
                         aspect='auto', alpha=0.5, cmap='Greys', interpolation='gaussian', zorder=0)

            ax[2].set_xlabel('time [s]', fontsize=fs)
            ax[0].set_ylabel('frequency [Hz]', fontsize=fs)
            ax[1].set_ylabel('frequency [Hz]', fontsize=fs)
            ax[2].set_ylabel('frequency [Hz]', fontsize=fs)
            ax[0].set_xticks([])
            ax[1].set_xticks([])

            ax[3].set_ylabel('frequency [Hz]', fontsize=fs)
            ax[5].set_ylabel('frequency [Hz]', fontsize=fs)
            # ax[5].set_ylabel('frequency [Hz]', fontsize=fs)
            # ax[3].set_xlabel('time [s]', fontsize=fs)
            # ax[4].set_xlabel('time [s]', fontsize=fs)
            ax[5].set_xlabel('time [s]', fontsize=fs)
            ax[6].set_xlabel('time [s]', fontsize=fs)

            ax[0].tick_params(labelsize=fs - 2)
            ax[1].tick_params(labelsize=fs - 2)
            ax[2].tick_params(labelsize=fs - 2)
            ax[3].tick_params(labelsize=fs - 2)
            ax[4].tick_params(labelsize=fs - 2)
            ax[5].tick_params(labelsize=fs - 2)
            ax[6].tick_params(labelsize=fs - 2)

            ###
            # insert from next figure #
            fig, axx = plt.subplots(1, 2, facecolor='white', figsize=(20. / 2.54, 12 / 2.54))
            ax.extend([axx[0], axx[1]])
            ax[9].imshow(decibel(part_spectra)[::-1], extent=[limitations[0], limitations[1], min_freq, max_freq],
                         aspect='auto', alpha=0.7, cmap='jet', interpolation='gaussian')
            ax[10].imshow(decibel(part_spectra)[::-1], extent=[limitations[0], limitations[1], min_freq, max_freq],
                          aspect='auto', alpha=0.7, cmap='jet', interpolation='gaussian')
            ax[9].set_xlabel('time [s]', fontsize=fs)
            ax[10].set_xlabel('time [s]', fontsize=fs)
            ax[9].set_ylabel('frequency [Hz]', fontsize=fs)
            # ax[1].set_ylabel('frequency [Hz]', fontsize=fs)
            ax[9].tick_params(labelsize=fs - 2)
            ax[10].tick_params(labelsize=fs - 2)
            #####

            freq_lims = (400, 1200)
            snippet_start, snippet_end = self.main_ax.get_xlim()

            self.ioi_field = freq_tracking_v4(np.array(self.fundamentals), np.array(self.signatures), self.times,
                                              self.kwargs['freq_tolerance'], n_channels=len(self.channels),
                                              freq_lims=freq_lims,
                                              ioi_field=self.ioi_field, fig=fig, ax=ax)

            c_freq = np.mean([self.fund_v[i] for i in self.ioi_field])
            c_time = np.mean([self.times[self.idx_v[i]] for i in self.ioi_field])

            # plt.tight_layout()

            #####
            ax[9].plot(self.times[self.idx_v[self.ioi_field[0]]], self.fund_v[self.ioi_field[0]], marker='o', color='k',
                       markersize=5)
            ax[10].plot(self.times[self.idx_v[self.ioi_field[0]]], self.fund_v[self.ioi_field[0]], marker='o',
                        color='k', markersize=5)
            ax[9].plot(self.times[self.idx_v[self.ioi_field[1]]], self.fund_v[self.ioi_field[1]], marker='o',
                       color='forestgreen', markersize=5)
            ax[10].plot(self.times[self.idx_v[self.ioi_field[1]]], self.fund_v[self.ioi_field[1]], marker='o',
                        color='forestgreen', markersize=5)
            ax[9].plot(self.times[self.idx_v[self.ioi_field[2]]], self.fund_v[self.ioi_field[2]], marker='o',
                       color='gold', markersize=5)
            ax[10].plot(self.times[self.idx_v[self.ioi_field[2]]], self.fund_v[self.ioi_field[2]], marker='o',
                        color='gold', markersize=5)
            ax[9].set_xlim(limitations)
            ax[10].set_ylim([min_freq, max_freq])

            c_freq = np.mean([self.fund_v[i] for i in self.ioi_field])
            cf = [self.fund_v[i] for i in self.ioi_field]
            c_time = np.mean([self.times[self.idx_v[i]] for i in self.ioi_field])
            ct = [self.times[self.idx_v[i]] for i in self.ioi_field]
            ax[9].plot([np.min(ct) - 5, np.max(ct) + 5], [np.min(cf) - 5, np.min(cf) - 5], color='k', lw=2)
            ax[9].plot([np.min(ct) - 5, np.max(ct) + 5], [np.max(cf) + 5, np.max(cf) + 5], color='k', lw=2)
            ax[9].plot([np.min(ct) - 5, np.min(ct) - 5], [np.min(cf) - 5, np.max(cf) + 5], color='k', lw=2)
            ax[9].plot([np.max(ct) + 5, np.max(ct) + 5], [np.min(cf) - 5, np.max(cf) + 5], color='k', lw=2)

            ax[9].set_xlim([np.min(ct) - 10, np.max(ct) + 25])
            ax[9].set_ylim([np.min(cf) - 20, np.max(cf) + 20])

            ax[10].set_xlim([np.min(ct) - 5, np.max(ct) + 5])
            ax[10].set_ylim([np.min(cf) - 5, np.max(cf) + 5])
            # ax[1].set_ylim([c_freq - 5, c_freq + 5])

            lt1 = np.max([self.times[self.idx_v[i]] for i in self.ioi_field]) + 0.5
            lt2 = np.max([self.times[self.idx_v[i]] for i in self.ioi_field]) + 1.

            ax[10].plot([lt1, lt1], [self.fund_v[self.ioi_field[0]], self.fund_v[self.ioi_field[1]]], lw=4,
                        color='forestgreen', label='$\Delta f_0$')
            ax[10].plot([lt2, lt2], [self.fund_v[self.ioi_field[0]], self.fund_v[self.ioi_field[2]]], lw=4,
                        color='gold', label='$\Delta f_1$')
            ax[10].legend(loc=3, fontsize=fs - 2)
            # plt.tight_layout()

        # df plot
        # if True:
        #     fig, ax = plt.subplots(1, 2, facecolor='white', figsize=(20./2.54, 12/2.54))
        #
        #     ax[0].imshow(decibel(part_spectra)[::-1], extent=[limitations[0], limitations[1], min_freq, max_freq], aspect='auto', alpha=0.7, cmap='jet', interpolation='gaussian')
        #     ax[1].imshow(decibel(part_spectra)[::-1], extent=[limitations[0], limitations[1], min_freq, max_freq], aspect='auto', alpha=0.7, cmap='jet', interpolation='gaussian')
        #     ax[0].set_xlabel('time [s]', fontsize=fs)
        #     ax[1].set_xlabel('time [s]', fontsize=fs)
        #     ax[0].set_ylabel('frequency [Hz]', fontsize=fs)
        #     # ax[1].set_ylabel('frequency [Hz]', fontsize=fs)
        #     ax[0].tick_params(labelsize=fs-2)
        #     ax[1].tick_params(labelsize=fs-2)
        #
        #     for ident in np.unique(self.ident_v[~np.isnan(self.ident_v)]):
        #         c = np.random.rand(3)
        #         ax[0].plot(self.times[self.idx_v[self.ident_v == ident]], self.fund_v[self.ident_v == ident], color=c, marker='.')
        #         ax[1].plot(self.times[self.idx_v[self.ident_v == ident]], self.fund_v[self.ident_v == ident], color=c, marker='.')
        #
        #     ax[0].plot(self.times[self.idx_v[self.ioi_field[0]]], self.fund_v[self.ioi_field[0]], marker='o', color='k', markersize=5)
        #     ax[1].plot(self.times[self.idx_v[self.ioi_field[0]]], self.fund_v[self.ioi_field[0]], marker='o', color='k', markersize=5)
        #     ax[0].plot(self.times[self.idx_v[self.ioi_field[1]]], self.fund_v[self.ioi_field[1]], marker='o', color='forestgreen', markersize=5)
        #     ax[1].plot(self.times[self.idx_v[self.ioi_field[1]]], self.fund_v[self.ioi_field[1]], marker='o', color='forestgreen', markersize=5)
        #     ax[0].plot(self.times[self.idx_v[self.ioi_field[2]]], self.fund_v[self.ioi_field[2]], marker='o', color='gold', markersize=5)
        #     ax[1].plot(self.times[self.idx_v[self.ioi_field[2]]], self.fund_v[self.ioi_field[2]], marker='o', color='gold', markersize=5)
        #     ax[0].set_xlim(limitations)
        #     ax[0].set_ylim([min_freq, max_freq])
        #
        #     c_freq = np.mean([self.fund_v[i] for i in self.ioi_field])
        #     cf = [self.fund_v[i] for i in self.ioi_field]
        #     c_time = np.mean([self.times[self.idx_v[i]] for i in self.ioi_field])
        #     ct = [self.times[self.idx_v[i]] for i in self.ioi_field]
        #     ax[0].plot([np.min(ct)-5, np.max(ct)+5], [np.min(cf) - 5, np.min(cf) - 5], color='k', lw=2)
        #     ax[0].plot([np.min(ct)-5, np.max(ct)+5], [np.max(cf) + 5, np.max(cf) + 5], color='k', lw=2)
        #     ax[0].plot([np.min(ct)-5, np.min(ct)-5], [np.min(cf) - 5, np.max(cf) + 5], color='k', lw=2)
        #     ax[0].plot([np.max(ct)+5, np.max(ct)+5], [np.min(cf) - 5, np.max(cf) + 5], color='k', lw=2)
        #
        #     ax[0].set_xlim([np.min(ct) - 10, np.max(ct) + 25])
        #     ax[0].set_ylim([np.min(cf) - 20, np.max(cf) + 20])
        #
        #     ax[1].set_xlim([np.min(ct) - 5, np.max(ct) + 5])
        #     ax[1].set_ylim([np.min(cf) - 5, np.max(cf) + 5])
        #     # ax[1].set_ylim([c_freq - 5, c_freq + 5])
        #
        #     lt1 = np.max([self.times[self.idx_v[i]] for i in self.ioi_field]) + 0.5
        #     lt2 = np.max([self.times[self.idx_v[i]] for i in self.ioi_field]) + 1.
        #
        #     ax[1].plot([lt1, lt1], [self.fund_v[self.ioi_field[0]], self.fund_v[self.ioi_field[1]]], lw=4, color='forestgreen', label='$\Delta f_0$')
        #     ax[1].plot([lt2, lt2], [self.fund_v[self.ioi_field[0]], self.fund_v[self.ioi_field[2]]], lw=4, color='gold', label='$\Delta f_1$')
        #     ax[1].legend(loc=3, fontsize=fs-2)
        #     plt.tight_layout()

        # AUC figure
        if True:
            # freq: colors[6]; amp: color[5]; error = color[2]
            aucs_a = []
            aucs_f = []
            aucs_e = []
            fig = plt.figure(figsize=(20 / 2.54, 12 / 2.54), facecolor='white')
            ax0 = fig.add_axes([.1, .6, .8, .3])
            ax0.set_xlim([0, 10])
            # ax1 = fig.add_axes([.15, .1, .275, .35])
            # ax2 = fig.add_axes([.575, .1, .275, .35])
            ax1 = fig.add_axes([.1, .1, .233, .35])
            ax11 = fig.add_axes([.1 + .233 / 2 + 0.02, .1 + 0.01, .233 / 2 - 0.0275, .35 / 2 - 0.02])
            ax11.set_xticks([0., .2])
            ax11.set_yticks([.8, 1])

            ax2 = fig.add_axes([.383, .1, .233, .35])
            ax22 = fig.add_axes([.383 + .233 / 2 + 0.02, .1 + 0.01, .233 / 2 - 0.0275, .35 / 2 - 0.02])
            ax3 = fig.add_axes([.666, .1, .233, .35])
            ax33 = fig.add_axes([.666 + .233 / 2 + 0.02, .1 + 0.01, .233 / 2 - 0.0275, .35 / 2 - 0.02])

            for i in range(len(s_ids)):
                a = np.array(s_ids[i] + ns_ids[i])
                b = np.hstack([np.zeros(len(s_ids[i])), np.ones(len(ns_ids[i]))])
                auc = roc_auc_score(b, a)
                aucs_a.append(auc)
                if i == 0:
                    c = colors[0]
                elif i == 10:
                    c = colors[1]
                else:
                    c = colors[4]

                # c = np.random.rand(3)
                if i % 10 == 0:
                    roc = roc_curve(b, a)
                    # ax1.set_title('field error')
                    ax1.plot(roc[0], roc[1], label='auc = %.2f; shift = %.2fs' % (aucs_a[i], (i + 1) * dps), color=c)
                    ax11.plot(roc[0], roc[1], label='auc = %.2f; shift = %.2fs' % (aucs_a[i], (i + 1) * dps), color=c)
                    ax0.plot((i + 1) / dps, aucs_a[i], 'o', color=c, zorder=1)

                a = np.array(s_idsf[i] + ns_idsf[i])
                b = np.hstack([np.zeros(len(s_idsf[i])), np.ones(len(ns_idsf[i]))])
                auc = roc_auc_score(b, a)
                aucs_f.append(auc)

                if i % 10 == 0:
                    roc = roc_curve(b, a)
                    # ax2.set_title('frequency error')
                    ax2.plot(roc[0], roc[1], label='auc = %.2f; shift = %.2fs' % (aucs_f[i], (i + 1) * dps), color=c)
                    ax22.plot(roc[0], roc[1], label='auc = %.2f; shift = %.2fs' % (aucs_f[i], (i + 1) * dps), color=c)
                    ax0.plot((i + 1) / dps, aucs_f[i], 'o', color=c, zorder=1)

                a = np.array(s_idse[i] + ns_idse[i])
                b = np.hstack([np.zeros(len(s_idse[i])), np.ones(len(ns_idse[i]))])
                auc = roc_auc_score(b, a)
                aucs_e.append(auc)

                if i % 10 == 0:
                    roc = roc_curve(b, a)
                    # ax3.set_title('total error')
                    ax3.plot(roc[0], roc[1], label='auc = %.2f; shift = %.2fs' % (aucs_e[i], (i + 1) * dps), color=c)
                    ax33.plot(roc[0], roc[1], label='auc = %.2f; shift = %.2fs' % (aucs_e[i], (i + 1) * dps), color=c)
                    ax0.plot((i + 1) / dps, aucs_e[i], 'o', color=c, zorder=1)

            ax0.plot((np.arange(len(s_ids)) + 1) / dps, aucs_a, marker='.', color=colors[5], zorder=0, label='field')
            ax0.plot((np.arange(len(s_idsf)) + 1) / dps, aucs_f, marker='.', color=colors[6], zorder=0,
                     label='frequency')
            ax0.plot((np.arange(len(s_idse)) + 1) / dps, aucs_e, marker='.', color=colors[2], zorder=0, label='error')
            ax0.legend(bbox_to_anchor=(.25, 1.), ncol=3)
            ax0.set_ylabel('AUC')
            ax0.set_xlabel('$\Delta$t [s]')

            ax1.set_xticks([0, 1])
            ax1.set_yticks([0, 1])
            ax1.set_xlabel('False positive rate')
            ax1.set_ylabel('True positive rate')
            ax1.plot([0, 1], [0, 1], '--', color='grey')
            ax1.set_title('field', fontsize=fs - 2)

            ax2.set_xticks([0, 1])
            ax2.set_yticks([0, 1])
            ax2.set_xlabel('False positive rate')
            ax2.plot([0, 1], [0, 1], '--', color='grey')
            ax2.set_title('frequency', fontsize=fs - 2)
            # ax2.set_ylabel('True positive rate')
            # plt.tight_layout()
            ax3.set_xticks([0, 1])
            ax3.set_yticks([0, 1])
            ax3.set_xlabel('False positive rate')
            ax3.plot([0, 1], [0, 1], '--', color='grey')
            ax3.set_title('error', fontsize=fs - 2)

            ax11.set_xticks([0., .2])
            ax11.set_yticks([.8, 1])
            ax11.tick_params(labelsize=fs - 5)
            ax11.xaxis.tick_top()
            # ax11.tick_params(axis='x', top=True)
            ax11.set_xlim([-.01, .2])
            ax11.set_ylim([0.8, 1.01])

            ax22.set_xticks([0., .2])
            ax22.set_yticks([.8, 1])
            ax22.tick_params(labelsize=fs - 5)
            ax22.xaxis.tick_top()
            ax22.set_xlim([-0.01, .2])
            ax22.set_ylim([0.8, 1.01])

            ax33.set_xticks([0., .2])
            ax33.set_yticks([.8, 1])
            ax33.tick_params(labelsize=fs - 5)
            ax33.xaxis.tick_top()
            ax33.set_xlim([-0.01, .2])
            ax33.set_ylim([0.8, 1.01])

        # shift amplitude
        if True:
            fig = plt.figure(facecolor='white', figsize=(20 / 2.54, 12 / 2.54))
            ax = fig.add_axes([0.1, 0.1, 0.6, 0.8])
            ax2 = fig.add_axes([0.7, 0.1, 0.2, 0.8])

            ax.plot(shift / dps, p_s_ids[:, 2], color=colors[3], lw=2, label='same identity')
            ax.fill_between(shift / dps, p_s_ids[:, 1], p_s_ids[:, 3], color=colors[3], alpha=0.3)
            ax.plot(shift / dps, p_s_ids[:, 0], '--', color=colors[3], lw=2)
            ax.plot(shift / dps, p_s_ids[:, 4], '--', color=colors[3], lw=2)

            ax.plot(shift / dps, p_ns_ids[:, 2], color=colors[2], lw=2, label='other identity')
            ax.legend(bbox_to_anchor=(1, 1.1), ncol=2, fontsize=fs - 2)
            ax.fill_between(shift / dps, p_ns_ids[:, 1], p_ns_ids[:, 3], color='orange', alpha=0.3)
            ax.plot(shift / dps, p_ns_ids[:, 0], '--', color=colors[2], lw=2)
            ax.plot(shift / dps, p_ns_ids[:, 4], '--', color=colors[2], lw=2)

            ax2.boxplot([np.hstack(s_ids), np.hstack(ns_ids)], sym='')
            ax2.set_xticklabels(['same\nidentity', 'other\nidentity'])
            ax2.set_ylim([ax.get_ylim()[0], ax.get_ylim()[1]])
            ax2.set_yticks([])

            ax.set_xlabel('$\Delta$t [s]', fontsize=fs)
            ax.set_ylabel('field error [Hz]', fontsize=fs)
            ax.tick_params(labelsize=fs - 2)

        # shift frequency
        if True:
            fig = plt.figure(facecolor='white', figsize=(20 / 2.54, 12 / 2.54))
            ax = fig.add_axes([0.1, 0.1, 0.6, 0.8])
            ax2 = fig.add_axes([0.7, 0.1, 0.2, 0.8])

            ax.plot(shift / dps, p_s_idsf[:, 2], color=colors[3], lw=2, label='same identity')
            ax.fill_between(shift / dps, p_s_idsf[:, 1], p_s_idsf[:, 3], color=colors[3], alpha=0.3)
            ax.plot(shift / dps, p_s_idsf[:, 0], '--', color=colors[3], lw=2)
            ax.plot(shift / dps, p_s_idsf[:, 4], '--', color=colors[3], lw=2)

            ax.plot(shift / dps, p_ns_idsf[:, 2], color=colors[2], lw=2, label='other identity')
            ax.legend(bbox_to_anchor=(1, 1.1), ncol=2, fontsize=fs - 2)
            ax.fill_between(shift / dps, p_ns_idsf[:, 1], p_ns_idsf[:, 3], color='orange', alpha=0.3)
            ax.plot(shift / dps, p_ns_idsf[:, 0], '--', color=colors[2], lw=2)
            ax.plot(shift / dps, p_ns_idsf[:, 4], '--', color=colors[2], lw=2)

            ax2.boxplot([np.hstack(s_idsf), np.hstack(ns_idsf)], sym='')
            ax2.set_xticklabels(['same\nidentity', 'other\nidentity'])
            ax2.set_ylim([ax.get_ylim()[0], ax.get_ylim()[1]])
            ax2.set_yticks([])

            ax.set_xlabel('$\Delta$t [s]', fontsize=fs)
            ax.set_ylabel('frequency error [Hz]', fontsize=fs)
            ax.tick_params(labelsize=fs - 2)

        # freq vs. ampl
        if True:
            fig = plt.figure(figsize=(20. / 2.54, 20. / 2.54))
            ax1 = fig.add_axes([.75, .1, .15, .65])
            ax2 = fig.add_axes([.1, .75, .65, .15])
            ax = fig.add_axes([.1, .1, .65, .65])
            ax.plot(np.hstack(s_idsf), np.hstack(s_ids), '.', color=colors[6], markersize=8, alpha=0.5)
            ax.plot(np.hstack(ns_idsf), np.hstack(ns_ids), '.', color=colors[5], markersize=8, alpha=0.5)
            ax.set_xlabel('$\Delta$frequency [Hz]')
            ax.set_ylabel('$\Delta$field [a.u.]')

            ax1.spines['right'].set_visible(False)
            ax1.spines['left'].set_visible(False)
            ax1.spines['top'].set_visible(False)
            # ax1.spines['bottom'].set_visible(False)
            # ax1.set_xticks([])
            ax1.set_yticks([])

            ax2.spines['right'].set_visible(False)
            # ax2.spines['left'].set_visible(False)
            ax2.spines['top'].set_visible(False)
            ax2.spines['bottom'].set_visible(False)
            # ax2.set_xticks([])
            ax2.set_yticks([])

            n, b = np.histogram(np.hstack(s_idsf), 50)
            n = n / np.sum(n) / (b[1] - b[0])
            ax2.bar(b[1:] - (b[1] - b[0]) / 2., n, width=b[1] - b[0], color=colors[3], alpha=0.5)

            n, b = np.histogram(np.hstack(ns_idsf), 50)
            n = n / np.sum(n) / (b[1] - b[0])
            ax2.bar(b[1:] - (b[1] - b[0]) / 2., n, width=b[1] - b[0], color=colors[2], alpha=0.5)
            ax2.set_yscale("log")

            n, b = np.histogram(np.hstack(s_ids), 50)
            n = n / np.sum(n) / (b[1] - b[0])
            ax1.barh(b[1:] - (b[1] - b[0]) / 2., n, height=b[1] - b[0], color=colors[3], alpha=0.5)

            n, b = np.histogram(np.hstack(ns_ids), 50)
            n = n / np.sum(n) / (b[1] - b[0])
            ax1.barh(b[1:] - (b[1] - b[0]) / 2., n, height=b[1] - b[0], color=colors[2], alpha=0.5)

        # field comparison
        if True:
            # plt.close()
            # embed()
            # quit()
            fig = plt.figure(facecolor='white', figsize=(20 / 2.54, 14 / 2.54))
            ax4 = fig.add_axes([0, 0, 1, 1])
            ax4.spines['top'].set_visible(False)
            ax4.spines['bottom'].set_visible(False)
            ax4.spines['right'].set_visible(False)
            ax4.spines['left'].set_visible(False)
            ax4.set_xticks([])
            ax4.set_yticks([])

            ax1 = fig.add_axes([.05, .1, .35, .35])
            ax2 = fig.add_axes([.05, .55, .35, .35])
            ax22 = fig.add_axes([.4, .725, .25, .25])
            ax3 = fig.add_axes([.6, .325, .35, .35])
            ax33 = fig.add_axes([.4, 0.025, .25, .25])

            cm_ax = fig.add_axes([.65, .15, .3, .01])
            cm_ax.spines['right'].set_visible(False)
            cm_ax.spines['left'].set_visible(False)
            cm_ax.spines['top'].set_visible(False)
            cm_ax.spines['bottom'].set_visible(False)
            cm_ax.set_yticks([])
            cm_ax.set_xticks([])

            # ax4.arrow(.5, .66, .4, .4, width=.01, color='orange')
            ax4.arrow(.375, .625, .2, 0, width=.01, color='forestgreen')
            ax4.text(.45, .65, '$\Delta field_0$', color='forestgreen', fontsize=fs)
            # ax4.arrow(.5, .15, .2, .1, width=.01, color='gold')
            ax4.arrow(.375, .375, .2, 0, width=.01, color='gold')
            ax4.text(.45, .4, '$\Delta field_1$', color='gold', fontsize=fs)

            dsig0 = np.abs(self.sign_v[self.ioi_field[0]] - self.sign_v[self.ioi_field[1]])
            dsig1 = np.abs(self.sign_v[self.ioi_field[0]] - self.sign_v[self.ioi_field[2]])

            cax = ax1.imshow(self.sign_v[self.ioi_field[0]].reshape(self.grid_prop).transpose()[::-1], cmap='jet',
                             interpolation='gaussian', vmin=0, vmax=1)
            ax2.imshow(self.sign_v[self.ioi_field[1]].reshape(self.grid_prop).transpose()[::-1], cmap='jet',
                       interpolation='gaussian', vmin=0, vmax=1)
            ax22.imshow(dsig0.reshape(self.grid_prop).transpose()[::-1], cmap='jet', interpolation='gaussian', vmin=0, vmax=1)
            ax3.imshow(self.sign_v[self.ioi_field[2]].reshape(self.grid_prop).transpose()[::-1], cmap='jet',
                       interpolation='gaussian', vmin=0, vmax=1)
            ax33.imshow(dsig1.reshape(self.grid_prop).transpose()[::-1], cmap='jet', interpolation='gaussian', vmin=0, vmax=1)
            for ax in [ax1, ax2, ax3, ax22, ax33]:
                e0 = np.hstack([np.arange(8) for i in range(8)])
                e1 = np.hstack([np.ones(8) * i for i in range(8)])
                ax.plot(e0, e1, 'o', color='k', markersize=1)
                if ax not in [ax22, ax33]:
                    ax.set_xticks([0, 2, 4, 6])
                    ax.set_xticklabels([0, 1, 2, 3])
                    ax.set_yticks([0, 2, 4, 6])
                    ax.set_yticklabels([0, 1, 2, 3])
                    ax.tick_params(labelsize=fs - 2)
                    ax.set_ylabel('x [m]', fontsize=fs)
                    ax.set_xlabel('y [m]', fontsize=fs)
                else:
                    ax.set_xticks([])
                    ax.set_yticks([])
            cbar = fig.colorbar(cax, ax=[cm_ax], orientation='horizontal', fraction=2)
            cbar.ax.set_xlabel('field strength')
        # rel. field error
        if True:
            # fig, ax = plt.subplots(facecolor='white', figsize=(20/2.54, 12/2.54))
            # fig, ax = plt.subplots(1, 2, facecolor='white', figsize=(20/2.54, 14/2.54))
            fig = plt.figure(facecolor='white', figsize=(20 / 2.54, 14 / 2.54))
            ax0 = fig.add_axes([0.1, 0.55, .8, .35])
            ax1 = fig.add_axes([0.1, 0.1, .8, .35])
            ax = [ax0, ax1]

            n, h = np.histogram(self.a_error_dist, 5000)
            ax[0].plot(h[1:], np.cumsum(n) / np.sum(n), color=colors[5], linewidth=2)
            ax[0].set_xlabel('field error [a.u.]', fontsize=fs)
            ax[0].set_ylabel('rel. field error', fontsize=fs)

            a_e = np.sqrt(np.sum((self.sign_v[self.ioi_field[0]] - self.sign_v[self.ioi_field[1]]) ** 2))
            rel_a_e = len(self.a_error_dist[self.a_error_dist <= a_e]) / len(self.a_error_dist)
            ax[0].plot([a_e, a_e], [0, rel_a_e], color='gold', alpha=.7, lw=3,
                       label='$\Delta field_0$ = %.2f' % rel_a_e)
            ax[0].plot([0, a_e], [rel_a_e, rel_a_e], color='gold', alpha=.7, lw=3)

            a_e = np.sqrt(np.sum((self.sign_v[self.ioi_field[0]] - self.sign_v[self.ioi_field[2]]) ** 2))
            rel_a_e = len(self.a_error_dist[self.a_error_dist <= a_e]) / len(self.a_error_dist)
            ax[0].plot([a_e, a_e], [0, rel_a_e], color='forestgreen', alpha=.7, lw=3,
                       label='$\Delta field_1$ = %.2f' % rel_a_e)
            ax[0].plot([0, a_e], [rel_a_e, rel_a_e], color='forestgreen', alpha=.7, lw=3)
            ax[0].tick_params(labelsize=fs - 2)
            ax[0].legend(loc=4)
            # ax[0].legend(bbox_to_anchor=(1, 1.1), ncol=2, fontsize=fs-2)
            ax[0].set_xlim([0, h[-1]])
            ax[0].set_ylim([0, 1.01])

        # rel freq error
        if True:
            # fig, ax = plt.subplots(facecolor='white', figsize=(20/2.54, 12/2.54))
            f_v = np.arange(0, 10.05, 0.01)
            rel_f_v = boltzmann(f_v, alpha=1, beta=0, x0=.25, dx=.15)
            # ax.semilogx(f_v, rel_f_v, color=colors[6], lw=2)
            ax[1].plot(f_v, rel_f_v, color=colors[6], lw=2)
            ax[1].set_xlabel('frequency error [Hz]', fontsize=fs)
            ax[1].set_ylabel('rel. frequency error', fontsize=fs)
            ax[1].tick_params(labelsize=fs - 2)

            f_e = np.abs(self.fund_v[self.ioi_field[0]] - self.fund_v[self.ioi_field[1]])
            rel_f_e = boltzmann(f_e, alpha=1, beta=0, x0=.25, dx=.15)
            ax[1].plot([f_e, f_e], [0, rel_f_e], color='gold', alpha=.7, lw=3,
                       label='$\Delta frequency_0$ = %.2f' % rel_f_e)
            ax[1].plot([0, f_e], [rel_f_e, rel_f_e], color='gold', alpha=.7, lw=3)

            f_e = np.abs(self.fund_v[self.ioi_field[0]] - self.fund_v[self.ioi_field[2]])
            rel_f_e = boltzmann(f_e, alpha=1, beta=0, x0=.25, dx=.15)
            ax[1].plot([f_e, f_e], [0, rel_f_e], color='forestgreen', alpha=.7, lw=3,
                       label='$\Delta frequency_1$ = %.2f' % rel_f_e)
            ax[1].plot([0, f_e], [rel_f_e, rel_f_e], color='forestgreen', alpha=.7, lw=3)
            ax[1].legend(loc=4)
            # ax[1].legend(bbox_to_anchor=(1, 1.1), ncol=2, fontsize=fs-2)
            ax[1].set_xlim([0, 10])
            ax[1].set_ylim([0, 1.05])
            # plt.tight_layout()

        plt.show()
        embed()
        quit()

    def save_traces(self):
        folder = os.path.split(self.data_file)[0]
        np.save(os.path.join(folder, 'fund_v.npy'), self.fund_v)
        np.save(os.path.join(folder, 'sign_v.npy'), self.sign_v)
        np.save(os.path.join(folder, 'idx_v.npy'), self.idx_v)
        np.save(os.path.join(folder, 'ident_v.npy'), self.ident_v)
        np.save(os.path.join(folder, 'times.npy'), self.times)
        np.save(os.path.join(folder, 'meta.npy'), np.array([self.start_time, self.end_time]))
        # np.save(os.path.join(folder, 'a_error_dist.npy'), self.a_error_dist)
        # np.save(os.path.join(folder, 'f_error_dist.npy'), self.f_error_dist)

        np.save(os.path.join(folder, 'spec.npy'), self.tmp_spectra)

    def fish_hist(self):
        if not self.add_ax:
            self.main_ax.set_position([.1, .1, .5, .6])
            self.add_ax = self.main_fig.add_axes([.6, .1, .3, .6])
            # self.ps_ax.set_yticks([])
            self.add_ax[0].yaxis.tick_right()
            self.add_ax[0].yaxis.set_label_position("right")
            self.add_ax[0].set_ylabel('frequency [Hz]', fontsize=12)
            self.add_ax[0].set_xlabel('rel. n', fontsize=12)

        else:
            self.ps_handle.remove()

        fwi = self.fund_v[~np.isnan(self.ident_v)]
        h, be = np.histogram(fwi, bins=np.arange(np.min(fwi), np.max(fwi) + 1))
        centers = be[1:] - ((be[1] - be[0]) / 2.)
        h = h / np.sum(h)

        self.ps_handle, = self.add_ax[0].plot(h, centers)

        self.add_ax[0].set_ylim([self.main_ax.get_ylim()[0], self.main_ax.get_ylim()[1]])

    def load_trace(self):
        folder = os.path.split(self.data_file)[0]
        if os.path.exists(os.path.join(folder, 'fund_v.npy')):
            self.fund_v = np.load(os.path.join(folder, 'fund_v.npy'))
            self.sign_v = np.load(os.path.join(folder, 'sign_v.npy'))
            self.idx_v = np.load(os.path.join(folder, 'idx_v.npy'))
            self.ident_v = np.load(os.path.join(folder, 'ident_v.npy'))
            self.times = np.load(os.path.join(folder, 'times.npy'))
            self.tmp_spectra = np.load(os.path.join(folder, 'spec.npy'))
            self.start_time, self.end_time = np.load(os.path.join(folder, 'meta.npy'))
            # self.a_error_dist = np.load(os.path.join(folder, 'a_error_dist.npy'))
            # self.f_error_dist = np.load(os.path.join(folder, 'f_error_dist.npy'))

            if self.spec_img_handle:
                self.spec_img_handle.remove()
            self.spec_img_handle = self.main_ax.imshow(decibel(self.tmp_spectra)[::-1],
                                                       extent=[self.start_time, self.end_time, 0, 2000],
                                                       aspect='auto', alpha=0.7, cmap='jet', interpolation='gaussian')
            self.main_ax.set_xlabel('time', fontsize=12)
            self.main_ax.set_ylabel('frequency [Hz]', fontsize=12)
            self.main_ax.set_xlim([self.start_time, self.end_time])

            self.plot_traces(clear_traces=True)

            self.get_clock_time()

    def fill_trace(self):
        add_freqs = self.fill_freqs[self.part_f_lims][np.argmax(self.part_spectra, axis=0)]
        i0 = np.argmin(np.abs(self.fill_times[self.part_t_lims][0] - self.times))
        add_idx = np.arange(i0, i0 + 2 * len(add_freqs), 2)
        # embed()
        # quit()

        # self.main_ax.plot(self.times[add_idx], add_freqs)

        self.fund_v = np.append(self.fund_v, add_freqs)
        self.idx_v = np.append(self.idx_v, add_idx)
        self.ident_v = np.append(self.ident_v,
                                 np.ones(len(add_idx)) * np.max(np.unique(self.ident_v[~np.isnan(self.ident_v)])) + 1)
        self.sign_v = np.append(self.sign_v, np.full((len(add_idx), np.shape(self.sign_v)[1]), np.nan), axis=0)

        sorter = np.argsort(self.idx_v)

        self.fund_v = self.fund_v[sorter]
        self.ident_v = self.ident_v[sorter]
        self.sign_v = self.sign_v[sorter]
        self.idx_v = self.idx_v[sorter]

        self.plot_traces(post_refill=True)

    def group_reassign(self):
        target_ident = np.max(self.ident_v[~np.isnan(self.ident_v)]) + 1

        active_indices_list = list(self.active_indices)

        for i in reversed(range(len(active_indices_list))[1:]):
            if self.idx_v[active_indices_list[i]] == self.idx_v[active_indices_list[i - 1]]:
                active_indices_list.pop(i)

        self.active_indices = np.array(active_indices_list)

        self.ident_v[self.active_indices] = target_ident

        self.active_indices_handle.remove()
        self.active_indices_handle = None

    def group_connect(self):
        active_identities = np.unique(self.ident_v[self.active_indices])
        target_ident = self.ident_v[self.active_indices][0]

        for ai in active_identities:
            if ai == target_ident:
                continue

            overlapping_idxs = np.intersect1d(self.idx_v[self.ident_v == target_ident], self.idx_v[self.ident_v == ai])
            self.ident_v[(np.in1d(self.idx_v, np.array(overlapping_idxs))) & (self.ident_v == ai)] = np.nan
            self.ident_v[self.ident_v == ai] = target_ident
        self.active_indices_handle.remove()
        self.active_indices_handle = None

    def connect_trace(self):

        # overlapping_idxs = [x for x in self.idx_v[self.ident_v == self.active_ident0] if x in self.idx_v[self.ident_v == self.active_ident1]]
        overlapping_idxs = np.intersect1d(self.idx_v[self.ident_v == self.active_ident0],
                                          self.idx_v[self.ident_v == self.active_ident1])

        # self.ident_v[(self.idx_v == overlapping_idxs) & (self.ident_v == self.active_ident0)] = np.nan
        self.ident_v[(np.in1d(self.idx_v, np.array(overlapping_idxs))) & (self.ident_v == self.active_ident0)] = np.nan
        self.ident_v[self.ident_v == self.active_ident1] = self.active_ident0

        self.plot_traces(clear_traces=False, post_connect=True)

        self.active_ident_handle0.remove()
        self.active_ident_handle0 = None

        self.active_ident_handle1.remove()
        self.active_ident_handle1 = None

    def cut_trace(self):
        next_ident = np.max(self.ident_v[~np.isnan(self.ident_v)]) + 1
        self.ident_v[(self.ident_v == self.active_ident0) & (self.idx_v < self.idx_v[self.active_idx0])] = next_ident

        self.active_ident_handle0.remove()
        self.active_ident_handle0 = None

        self.active_idx_handle0.remove()
        self.active_idx_handle0 = None
        self.active_idx0 = None

        # self.plot_traces(clear_traces=True)
        self.plot_traces(clear_traces=False, post_cut=True)

    def delete_trace(self):
        self.ident_v[self.ident_v == self.active_ident0] = np.nan
        self.plot_traces(clear_traces=False, post_delete=True)
        self.active_ident0 = None
        self.active_ident_handle0.remove()
        self.active_ident_handle0 = None

        # self.plot_traces(clear_traces=True)

    def group_delete(self):
        self.ident_v[self.active_indices] = np.nan
        # self.active_indices = []
        self.active_indices_handle.remove()
        self.active_indices_handle = None

    def save_plot(self):
        self.main_ax.set_position([.1, .1, .8, .8])
        if self.add_ax:
            self.main_fig.delaxes(self.ps_as)
            self.add_ax = None
            self.all_peakf_dots = None
            self.good_peakf_dots = None
            # self.main_ax.set_position([.1, .1, .8, .8])

        for i, j in zip(self.text_handles_key, self.text_handles_effect):
            self.main_fig.texts.remove(i)
            self.main_fig.texts.remove(j)
        self.text_handles_key = []
        self.text_handles_effect = []

        for handle in self.trace_handles:
            handle[0].remove()
        self.trace_handles = []

        possible_identities = np.unique(self.ident_v[~np.isnan(self.ident_v)])
        for ident in np.array(possible_identities):
            c = np.random.rand(3)
            h, = self.main_ax.plot(self.times[self.idx_v[self.ident_v == ident]], self.fund_v[self.ident_v == ident],
                                   marker='.', color=c, markersize=1)
            self.trace_handles.append((h, ident))

        # time_extend = np.diff(self.main_ax.get_xlim())[0]
        # time_str = os.path.split(self.data_file)[0][-5:].replace('_', '').replace(':', '')
        # h = int(time_str[0:2])
        # m = int(time_str[2:])
        #
        # start_m = h * 60 + m
        #
        # first_h = h
        # first_m = m
        # first_s = 0
        # if time_extend <= 600.: # marker jede minute
        #     x_steps = 60
        #     pass
        # elif time_extend > 600. and time_extend <= 3600.: # marker alle 10 min
        #     x_steps = 600
        #     if m % 10 != 0:
        #         first_h = h
        #         first_m = m + (10 - m % 10)
        #         first_s = (10 - m % 10) * 60
        #         if first_m >= 60:
        #             first_m -= 60
        #             first_h += 1
        #         if first_h >= 24:
        #             first_h -= 24
        # else: # marker alle halbe stunde
        #     x_steps = 1800
        #     if m % 30 != 0:
        #         first_h = h
        #         first_m = m + (30 - m % 30)
        #         first_s = (30 - m % 30) * 60
        #         if first_m >= 60:
        #             first_m -= 60
        #             first_h += 1
        #         if first_h >= 24:
        #             first_h -= 24
        #
        # possible_timestamps = np.arange(first_s, self.times[-1], x_steps)
        # use_timestamps_s_origin = possible_timestamps[(possible_timestamps > self.main_ax.get_xlim()[0]) & (possible_timestamps < self.main_ax.get_xlim()[1])]
        # use_timestamps = use_timestamps_s_origin / 60. + start_m
        # use_timestamps[use_timestamps >= 3600.] -= 3600.
        #
        # x_ticks = ['%2.f:%2.f' %(x // 60, x % 60)  for x in use_timestamps]
        # x_ticks = [x.replace(' ', '0') for x in x_ticks]
        #
        # self.main_ax.set_xticks(use_timestamps_s_origin)
        # self.main_ax.set_xticklabels(x_ticks)
        # plt.tight_layout()

        # embed()
        # quit()

        # if self.add_ax[2][0]:
        #     self.main_fig.delaxes(self.add_ax[2][0])
        #     self.add_ax[2][0] = None
        # if self.add_ax[2][1]:
        #     self.main_fig.delaxes(self.add_ax[2][1])
        #     self.add_ax[2][1] = None
        # if self.t_error_ax:
        #     self.main_fig.delaxes(self.t_error_ax)
        #     self.t_error_ax = None

        self.main_fig.set_size_inches(20. / 2.54, 12. / 2.54)
        self.main_fig.canvas.draw()

        plot_nr = len(glob.glob('/home/raab/Desktop/plot*'))
        self.main_fig.savefig('/home/raab/Desktop/plot%.0f.pdf' % plot_nr)

        self.main_fig.set_size_inches(55. / 2.54, 30. / 2.54)
        self.main_ax.set_position([.1, .1, .8, .6])
        for handle in self.trace_handles:
            handle[0].remove()
        self.trace_handles = []

        possible_identities = np.unique(self.ident_v[~np.isnan(self.ident_v)])
        for ident in np.array(possible_identities):
            c = np.random.rand(3)
            h, = self.main_ax.plot(self.times[self.idx_v[self.ident_v == ident]], self.fund_v[self.ident_v == ident],
                                   marker='.', color=c)
            self.trace_handles.append((h, ident))

        self.main_fig.canvas.draw()

    def plot_spectrum(self, part_spec=False, part_spec_from_file=False):
        if part_spec_from_file:
            limitations = self.main_ax.get_xlim()
            min_freq = self.main_ax.get_ylim()[0]
            max_freq = self.main_ax.get_ylim()[1]

            # if os.path.exists(os.path.join(os.path.split(self.data_file)[0], 'fill_spec.npy')):
            if os.path.exists(os.path.join(os.path.split(self.data_file)[0], 'fill_freqs.npy')):
                # embed()
                # quit()
                self.fill_spec_shape = np.load(os.path.join(os.path.split(self.data_file)[0], 'fill_spec_shape.npy'))
                # ToDo: untill kraken is up load this from loacal harddrive...
                self.fill_spec = np.memmap(os.path.join(os.path.split(self.data_file)[0], 'fill_spec.npy'),
                                           dtype='float', mode='r', shape=tuple(self.fill_spec_shape), order='F')
                # self.fill_spec = np.memmap('/home/raab/analysis/fill_spec.npy', dtype='float', mode='r', shape=tuple(self.fill_spec_shape), order='F')
                self.fill_freqs = np.load(os.path.join(os.path.split(self.data_file)[0], 'fill_freqs.npy'))
                self.fill_times = np.load(os.path.join(os.path.split(self.data_file)[0], 'fill_times.npy'))
                # embed()
                # quit()

                self.spec_img_handle.remove()

                # f_lims = np.arange(len(self.fill_freqs)-1, -1, -1)[(self.fill_freqs >= min_freq) & (self.fill_freqs <= max_freq)]
                self.part_f_lims = np.arange(len(self.fill_freqs))[
                    (self.fill_freqs >= min_freq) & (self.fill_freqs <= max_freq)]
                self.part_t_lims = np.arange(len(self.fill_times))[
                    (self.fill_times >= limitations[0] + self.spec_shift) & (
                                self.fill_times <= limitations[1] + self.spec_shift)]
                self.part_spectra = self.fill_spec[self.part_f_lims[0]:self.part_f_lims[-1] + 1,
                                    self.part_t_lims[0]:self.part_t_lims[-1] + 1]

                self.spec_img_handle = self.main_ax.imshow(decibel(self.part_spectra)[::-1],
                                                           extent=[limitations[0], limitations[1], min_freq, max_freq],
                                                           aspect='auto', alpha=0.7, cmap='jet',
                                                           interpolation='gaussian')
                self.main_ax.set_xlabel('time', fontsize=12)
                self.main_ax.set_ylabel('frequency [Hz]', fontsize=12)
                self.main_ax.tick_params(labelsize=10)

            else:
                print('missing file')

        else:
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
                                                           aspect='auto', alpha=0.7, cmap='jet',
                                                           interpolation='gaussian')
                self.main_ax.set_xlabel('time', fontsize=12)
                self.main_ax.set_ylabel('frequency [Hz]', fontsize=12)
                self.main_ax.tick_params(labelsize=10)
            else:
                if not hasattr(self.tmp_spectra, '__len__'):
                    self.tmp_spectra, self.times = get_spectrum_funds_amp_signature(
                        self.data, self.samplerate, self.channels, self.data_snippet_idxs, self.start_time,
                        self.end_time,
                        create_plotable_spectrogram=True, extract_funds_and_signature=False, **self.kwargs)

                if not self.auto:
                    self.spec_img_handle = self.main_ax.imshow(decibel(self.tmp_spectra)[::-1],
                                                               extent=[self.start_time, self.end_time, 0, 2000],
                                                               aspect='auto', alpha=0.7, cmap='jet',
                                                               interpolation='gaussian')
                    self.main_ax.set_xlabel('time', fontsize=12)
                    self.main_ax.set_ylabel('frequency [Hz]', fontsize=12)
                    self.main_ax.tick_params(labelsize=10)

    def track_snippet(self):
        if hasattr(self.fund_v, '__len__'):
            for i in reversed(range(len(self.trace_handles))):
                self.trace_handles[i][0].remove()
                self.trace_handles.pop(i)

        if self.main_ax:
            snippet_start, snippet_end = self.main_ax.get_xlim()
        else:
            snippet_start = self.start_time
            snippet_end = self.end_time

        if not hasattr(self.fund_v, '__len__'):
            if not self.auto:
                self.fundamentals, self.signatures, self.positions, self.times = \
                    get_spectrum_funds_amp_signature(self.data, self.samplerate, self.channels, self.data_snippet_idxs,
                                                     snippet_start, snippet_end, create_plotable_spectrogram=False,
                                                     extract_funds_and_signature=True, **self.kwargs)
            else:
                self.fundamentals, self.signatures, self.positions, self.times, self.tmp_spectra = \
                    get_spectrum_funds_amp_signature(self.data, self.samplerate, self.channels, self.data_snippet_idxs,
                                                     snippet_start, snippet_end, create_plotable_spectrogram=True,
                                                     extract_funds_and_signature=True, **self.kwargs)
        else:
            mask = np.arange(len(self.idx_v))[
                (self.times[self.idx_v] >= snippet_start) & (self.times[self.idx_v] <= snippet_end)]
            self.fundamentals = []
            self.signatures = []

            fundamentas = []
            signatures = []
            for i in mask:
                if i > mask[0]:
                    if self.idx_v[i] != self.idx_v[i - 1]:
                        self.fundamentals.append(np.array(fundamentas))
                        fundamentas = []
                        self.signatures.append(np.array(signatures))
                        signatures = []

                    fundamentas.append(self.fund_v[i])
                    signatures.append(self.sign_v[i])
            self.fundamentals.append(np.array(fundamentas))
            self.signatures.append(np.array(signatures))

        mask = np.arange(len(self.times))[(self.times >= snippet_start) & (self.times <= snippet_end)]
        if self.live_tracking:
            self.fund_v, self.ident_v, self.idx_v, self.sign_v, self.a_error_dist, self.f_error_dist, self.idx_of_origin_v = \
                freq_tracking_v3(np.array(self.fundamentals)[mask], np.array(self.signatures)[mask],
                                 self.times[mask], self.kwargs['freq_tolerance'], n_channels=len(self.channels),
                                 fig=self.main_fig, ax=self.main_ax, freq_lims=self.main_ax.get_ylim())
        else:
            if not self.auto:
                freq_lims = self.main_ax.get_ylim()
            else:
                freq_lims = (400, 1200)

            self.fund_v, self.ident_v, self.idx_v, self.sign_v, self.a_error_dist, self.f_error_dist, self.idx_of_origin_v, self.original_sign_v = \
                freq_tracking_v4(np.array(self.fundamentals), np.array(self.signatures),
                                 self.times[mask], self.kwargs['freq_tolerance'], n_channels=len(self.channels),
                                 freq_lims=freq_lims, fig=self.main_fig, ax=self.main_ax)
            self.times = self.times[mask]

        if not self.auto:
            self.plot_traces(clear_traces=True)

    def plot_error(self):
        # if self.add_ax[0]:
            # self.main_fig.delaxes(self.add_ax[0])
            # self.add_ax[0] = None
            # self.add_tmp_plothandel = []
            # self.all_peakf_dots = None
            # self.good_peakf_dots = None

        # self.add_ax[2][0] = self.main_fig.add_axes([.6, .75, 0.15, 0.15])
        self.add_ax[2][0].plot(np.arange(0, 5, 0.02), boltzmann(np.arange(0, 5, 0.02), alpha=1, beta=0, x0=.25, dx=.15),
                             color='cornflowerblue', linewidth=2)
        self.add_ax[2][0].set_xlabel('frequency error [Hz]', fontsize=12)

        n, h = np.histogram(self.a_error_dist, 5000)
        # self.add_ax[2][1] = self.main_fig.add_axes([.8, .75, 0.15, 0.15])
        self.add_ax[2][1].plot(h[1:], np.cumsum(n) / np.sum(n), color='green', linewidth=2)
        self.add_ax[2][1].set_xlabel('amplitude error [a.u.]', fontsize=12)
        self.add_ax[2][0].set_visible(True)
        self.add_ax[2][1].set_visible(True)


    def plot_traces(self, clear_traces=False, post_connect=False, post_cut=False, post_delete=False,
                    post_group_delete=False, post_group_connection=False, post_refill=False, post_group_reassign=False):
        """
        shows/updates/deletes all frequency traces of individually tracked fish in a plot.

        :param clear_traces: (bool) if true removes all preiouly plotted traces before plotting the new ones
        :param post_connect: (bool) refreshes/deletes single identity traces previously selected and stored in class variables.
        """
        # self.main_ax.imshow(10.0 * np.log10(self.tmp_spectra)[::-1], extent=[self.start_time, self.end_time, 0, 2000], aspect='auto', alpha=0.7)

        if post_refill:
            new_ident = np.max(self.ident_v[~np.isnan(self.ident_v)])
            c = np.random.rand(3)
            h, = self.main_ax.plot(self.times[self.idx_v[self.ident_v == new_ident]],
                                   self.fund_v[self.ident_v == new_ident], marker='.', color=c)
            self.trace_handles.append((h, new_ident))

        if post_group_delete or post_group_connection or post_group_reassign:
            handle_idents = np.array([x[1] for x in self.trace_handles])
            effected_idents = np.unique(self.last_ident_v[self.active_indices])
            mask = np.array([x in effected_idents for x in handle_idents], dtype=bool)
            delete_handle_idx = np.arange(len(self.trace_handles))[mask]
            delete_handle = np.array(self.trace_handles)[mask]

            delete_afterwards = []
            for dhi, dh in zip(delete_handle_idx, delete_handle):
                dh[0].remove()
                if len(self.ident_v[self.ident_v == dh[1]]) >= 1:
                    c = np.random.rand(3)
                    h, = self.main_ax.plot(self.times[self.idx_v[self.ident_v == dh[1]]],
                                           self.fund_v[self.ident_v == dh[1]], marker='.', color=c)
                    self.trace_handles[dhi] = (h, dh[1])
                else:
                    delete_afterwards.append(dhi)

            for i in reversed(sorted(delete_afterwards)):
                self.trace_handles.pop(i)

            if post_group_reassign:
                c = np.random.rand(3)
                new_ident = np.max(np.unique(self.ident_v[~np.isnan(self.ident_v)]))
                h, = self.main_ax.plot(self.times[self.idx_v[self.ident_v == new_ident]],
                                       self.fund_v[self.ident_v == new_ident], marker='.', color=c)
                self.trace_handles.append((h, new_ident))

            self.active_indices = []

        if post_delete:
            handle_idents = np.array([x[1] for x in self.trace_handles])
            delete_handle_idx = np.arange(len(self.trace_handles))[handle_idents == self.active_ident0][0]
            delete_handle = np.array(self.trace_handles)[handle_idents == self.active_ident0][0]
            delete_handle[0].remove()

            self.trace_handles.pop(delete_handle_idx)

        if post_cut:
            handle_idents = np.array([x[1] for x in self.trace_handles])
            refresh_handle = np.array(self.trace_handles)[handle_idents == self.active_ident0][0]
            refresh_handle[0].remove()

            c = np.random.rand(3)
            h, = self.main_ax.plot(self.times[self.idx_v[self.ident_v == self.active_ident0]],
                                   self.fund_v[self.ident_v == self.active_ident0], marker='.', color=c)
            self.trace_handles[np.arange(len(self.trace_handles))[handle_idents == self.active_ident0][0]] = (
            h, self.active_ident0)

            new_ident = np.max(self.ident_v[~np.isnan(self.ident_v)])
            c = np.random.rand(3)
            h, = self.main_ax.plot(self.times[self.idx_v[self.ident_v == new_ident]],
                                   self.fund_v[self.ident_v == new_ident], marker='.', color=c)
            self.trace_handles.append((h, new_ident))

        if post_connect:
            handle_idents = np.array([x[1] for x in self.trace_handles])

            remove_handle = np.array(self.trace_handles)[handle_idents == self.active_ident1][0]
            remove_handle[0].remove()

            joined_handle = np.array(self.trace_handles)[handle_idents == self.active_ident0][0]
            joined_handle[0].remove()

            c = np.random.rand(3)
            # sorter = np.argsort(self.times[self.idx_v[self.ident_v == self.active_ident0]])
            h, = self.main_ax.plot(self.times[self.idx_v[self.ident_v == self.active_ident0]],
                                   self.fund_v[self.ident_v == self.active_ident0], marker='.', color=c)
            self.trace_handles[np.arange(len(self.trace_handles))[handle_idents == self.active_ident0][0]] = (
            h, self.active_ident0)
            # self.trace_handles.append((h, self.active_ident0))

            self.trace_handles.pop(np.arange(len(self.trace_handles))[handle_idents == self.active_ident1][0])

        if clear_traces:
            for handle in self.trace_handles:
                handle[0].remove()
            self.trace_handles = []

            possible_identities = np.unique(self.ident_v[~np.isnan(self.ident_v)])
            for ident in np.array(possible_identities):
                c = np.random.rand(3)
                h, = self.main_ax.plot(self.times[self.idx_v[self.ident_v == ident]],
                                       self.fund_v[self.ident_v == ident], marker='.', color=c)
                self.trace_handles.append((h, ident))

    def plot_ps(self):
        """
        calculates the powerspectrum of single or multiple datasnippets recorded with single or multiple electrodes at a time.
        If multiple electrode recordings are analysed the shown powerspectrum is the sum of all calculated powerspectra.
        """

        # if self.add_ax[2][0]:
        #     self.main_fig.delaxes(self.add_ax[2][0])
        #     self.add_ax[2][0] = None
        # if self.add_ax[2][1]:
        #     self.main_fig.delaxes(self.add_ax[2][1])
        #     self.add_ax[2][1] = None
        # if self.t_error_ax:
        #     self.main_fig.delaxes(self.t_error_ax)
        #     self.t_error_ax = None

        # nfft = next_power_of_two(self.samplerate / self.fresolution)
        nfft = next_power_of_two(self.samplerate / self.kwargs['fresolution'])
        data_idx0 = int(self.times[self.ioi] * self.samplerate)
        data_idx1 = int(data_idx0 + nfft + 1)

        all_c_spectra = []
        all_c_freqs = None

        # if self.kwargs['noice_cancel']:
        #     denoiced_data = np.array([self.data[data_idx0: data_idx1, channel] for channel in self.channels])
        #     # print(denoiced_data.shape)
        #     mean_data = np.mean(denoiced_data, axis=0)
        #     # mean_data.shape = (len(mean_data), 1)
        #     denoiced_data -= mean_data

        for channel in self.channels:
            # c_spectrum, c_freqs, c_time = spectrogram(self.data[data_idx0: data_idx1, channel], self.samplerate,
            #                                           fresolution = self.fresolution, overlap_frac = self.overlap_frac)
            # if self.kwargs['noice_cancel']:
            #     c_spectrum, c_freqs, c_time = spectrogram(denoiced_data[channel], self.samplerate,
            #                                               fresolution=self.kwargs['fresolution'],
            #                                               overlap_frac=self.kwargs['overlap_frac'])
            # else:
            c_spectrum, c_freqs, c_time = spectrogram(self.data[data_idx0: data_idx1, channel], self.samplerate,
                                                      fresolution=self.kwargs['fresolution'],
                                                      overlap_frac=self.kwargs['overlap_frac'])
            if not hasattr(all_c_freqs, '__len__'):
                all_c_freqs = c_freqs
            all_c_spectra.append(c_spectrum)

        comb_spectra = np.sum(all_c_spectra, axis=0)
        self.power = np.hstack(comb_spectra)
        self.freqs = all_c_freqs

        groups, _, _, self.all_peakf, self.good_peakf, self.kwargs['low_threshold'], self.kwargs[
            'high_threshold'], self.psd_baseline = harmonic_groups(all_c_freqs, self.power, verbose=self.verbose,
                                                                   **self.kwargs)
        self.verbose = 0
        # plot_power = 10.0 * np.log10(self.power)
        plot_power = decibel(self.power)

        if not self.ps_handle:
            # self.main_ax.set_position([.1, .1, .5, .6])
            # self.add_ax = self.main_fig.add_axes([.6, .1, .3, .6])
            # self.ps_ax.set_yticks([])
            self.add_ax[0].yaxis.tick_right()
            self.add_ax[0].yaxis.set_label_position("right")
            self.add_ax[0].set_ylabel('frequency [Hz]', fontsize=12)
            self.add_ax[0].set_xlabel('power [dB]', fontsize=12)
            self.ps_handle, = self.add_ax[0].plot(plot_power[self.freqs <= 3000.0], self.freqs[self.freqs <= 3000.0],
                                               color='cornflowerblue')

            self.all_peakf_dots, = self.add_ax[0].plot(
                np.ones(len(self.all_peakf[:, 0])) * np.max(plot_power[self.freqs <= 3000.0]) + 5.,
                self.all_peakf[:, 0], 'o', color='red')
            self.good_peakf_dots, = self.add_ax[0].plot(
                np.ones(len(self.good_peakf)) * np.max(plot_power[self.freqs <= 3000.0]) + 5., self.good_peakf, 'o',
                color='green')

        else:
            self.ps_handle.set_data(plot_power[all_c_freqs <= 3000.0], all_c_freqs[all_c_freqs <= 3000.0])
            self.all_peakf_dots.remove()
            self.good_peakf_dots.remove()
            self.all_peakf_dots, = self.add_ax[0].plot(
                np.ones(len(self.all_peakf[:, 0])) * np.max(plot_power[all_c_freqs <= 3000.0]) + 5.,
                self.all_peakf[:, 0], 'o',
                color='red')
            self.good_peakf_dots, = self.add_ax[0].plot(
                np.ones(len(self.good_peakf)) * np.max(plot_power[all_c_freqs <= 3000.0]) + 5., self.good_peakf, 'o',
                color='green')

        for i in range(len(self.add_tmp_plothandel)):
            self.add_tmp_plothandel[i].remove()
        self.add_tmp_plothandel = []

        for fish in range(len(groups)):
            c = np.random.rand(3)

            h, = self.add_ax[0].plot(decibel(groups[fish][groups[fish][:, 0] < 3000., 1]),
                                  groups[fish][groups[fish][:, 0] < 3000., 0], 'o', color=c,
                                  markersize=7, alpha=0.9)
            self.add_tmp_plothandel.append(h)

        ylims = self.main_ax.get_ylim()
        self.add_ax[0].set_ylim([ylims[0], ylims[1]])

    def update_hg(self):
        """
        reexecutes peak detection in a powerspectrum with changed parameters and updates the plot
        """
        # self.fundamentals = None
        # groups = harmonic_groups(self.freqs, self.power, **self.kwargs)
        groups, _, _, self.all_peakf, self.good_peakf, self.kwargs['low_threshold'], self.kwargs[
            'high_threshold'], self.psd_baseline = \
            harmonic_groups(self.freqs, self.power, **self.kwargs)
        # print(self.psd_baseline)
        for i in range(len(self.add_tmp_plothandel)):
            self.add_tmp_plothandel[i].remove()
        self.add_tmp_plothandel = []

        for fish in range(len(groups)):
            c = np.random.rand(3)

            h, = self.add_ax[0].plot(decibel(groups[fish][groups[fish][:, 0] < 3000., 1]),
                                  groups[fish][groups[fish][:, 0] < 3000., 0], 'o', color=c,
                                  markersize=7, alpha=0.9)
            self.add_tmp_plothandel.append(h)
        plot_power = decibel(self.power)
        self.all_peakf_dots.remove()
        self.good_peakf_dots.remove()
        self.all_peakf_dots, = self.add_ax[0].plot(
            np.ones(len(self.all_peakf[:, 0])) * np.max(plot_power[self.freqs <= 3000.0]) + 5., self.all_peakf[:, 0],
            'o',
            color='red')
        self.good_peakf_dots, = self.add_ax[0].plot(
            np.ones(len(self.good_peakf)) * np.max(plot_power[self.freqs <= 3000.0]) + 5., self.good_peakf, 'o',
            color='green')

        ylims = self.main_ax.get_ylim()
        self.add_ax[0].set_ylim([ylims[0], ylims[1]])

    def get_clock_time(self):
        if len(self.channels) > 2:

            # embed()
            # quit()
            time_extend = np.diff(self.main_ax.get_xlim())[0]
            time_str = os.path.split(self.data_file)[0][-5:].replace('_', '').replace(':', '')
            h = int(time_str[0:2])
            m = int(time_str[2:])

            start_m = h * 60 + m
            first_s = 0

            if time_extend <= 1200.:  # bis 20 min ... marker jede minute
                x_steps = 60
                pass
            elif time_extend > 1200. and time_extend <= 2700.:  # bis 90 min ... marker alle 5 min
                x_steps = 300
                if m % 5 != 0:
                    first_s = (5 - m % 5) * 60
            elif time_extend > 2700. and time_extend <= 5400.:  # bis 3h ... marker alle 10 min
                x_steps = 600
                if m % 10 != 0:
                    first_s = (10 - m % 10) * 60
            elif time_extend > 5400. and time_extend <= 21600.:  # bis 6h ... marker alle 15 min
                x_steps = 900
                if m % 15 != 0:
                    first_s = (15 - m % 15) * 60
            else:  # marker alle halbe stunde
                x_steps = 3600
                if m % 60 != 0:
                    first_s = (60 - m % 60) * 60

            possible_timestamps = np.arange(first_s, self.times[-1], x_steps)
            use_timestamps_s_origin = possible_timestamps[
                (possible_timestamps > self.main_ax.get_xlim()[0]) & (possible_timestamps < self.main_ax.get_xlim()[1])]
            use_timestamps = use_timestamps_s_origin / 60. + start_m
            use_timestamps[use_timestamps >= 3600.] -= 3600.

            x_ticks = ['%2.f:%2.f' % ((x // 60) % 24, x % 60) for x in use_timestamps]
            x_ticks = [x.replace(' ', '0') for x in x_ticks]

            self.main_ax.set_xticks(use_timestamps_s_origin)
            self.main_ax.set_xticklabels(x_ticks)


def fish_tracker(data_file, start_time=0.0, end_time=-1.0, grid=False, auto=False, fill_spec=False, transect_data=False,
                 data_snippet_secs=15., verbose=0, **kwargs):
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
        if transect_data:
            data = open_data(data_file, 0, 60.0, 10.0)
        else:
            data = open_data(data_file, -1, 60.0, 10.0)
        samplerate = data.samplerate
        # embed()
        # quit()

    channels, coords, neighbours = get_grid_proportions(data, grid, n_tolerance_e=2, verbose=verbose)

    data_snippet_idxs = int(data_snippet_secs * samplerate)

    Obs_tracker(data, samplerate, start_time, end_time, channels, data_snippet_idxs, data_file, auto, fill_spec,
                **kwargs)


def main():
    # config file name:
    cfgfile = __package__ + '.cfg'

    # command line arguments:
    parser = argparse.ArgumentParser(
        description='Analyse long single- or multi electrode EOD recordings of weakly electric fish.',
        epilog='by bendalab (2015-2018)')
    parser.add_argument('--version', action='version', version=__version__)
    parser.add_argument('-v', action='count', dest='verbose', help='verbosity level')
    parser.add_argument('-c', '--save-config', nargs='?', default='', const=cfgfile,
                        type=str, metavar='cfgfile',
                        help='save configuration to file cfgfile (defaults to {0})'.format(cfgfile))
    parser.add_argument('file', nargs=1, default='', type=str,
                        help='name of the file wih the time series data or the -fishes.npy file saved with the -s option')
    parser.add_argument('start_time', nargs='?', default=0.0, type=float, help='start time of analysis in min.')
    parser.add_argument('end_time', nargs='?', default=-1.0, type=float, help='end time of analysis in min.')
    # parser.add_argument('-g', dest='grid', action='store_true', help='sum up spectrograms of all channels available.')
    # parser.add_argument('-g', action='count', dest='grid', help='grid information')
    # parser.add_argument('-p', dest='save_plot', action='store_true', help='save output plot as png file')
    parser.add_argument('-a', dest='auto', action='store_true', help='automatically analyse data and save results')
    # parser.add_argument('-n', dest='noice_cancel', action='store_true',
    #                     help='cancsels noice by substracting mean of all electrodes from all electrodes')
    parser.add_argument('-s', dest='fill_spec', action='store_true',
                        help='compute whole spec--- CARE: big data requirement')
    parser.add_argument('-f', dest='plot_harmonic_groups', action='store_true', help='plot harmonic group detection')
    parser.add_argument('-t', dest='transect_data', action='store_true', help='adapt parameters for transect data')
    parser.add_argument('-o', dest='output_folder', default=".", type=str,
                        help="path where to store results and figures")
    args = parser.parse_args()

    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)

    datafile = args.file[0]
    data_snippet_secs = 15.

    # set verbosity level from command line:
    verbose = 0
    if args.verbose != None:
        verbose = args.verbose

    # configuration options:
    cfg = ConfigFile()
    add_psd_peak_detection_config(cfg)
    add_harmonic_groups_config(cfg, mains_freq=50)
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

    t_kwargs = psd_peak_detection_args(cfg)
    t_kwargs.update(harmonic_groups_args(cfg))
    t_kwargs.update(tracker_args(cfg))
    # t_kwargs['noice_cancel'] = args.noice_cancel

    t_kwargs = grid_config_update(t_kwargs)

    if True:
        t_kwargs['low_thresh_factor'] = 18.
        t_kwargs['high_thresh_factor'] = 20.
        t_kwargs['fresolution'] = 1.5
        t_kwargs['overlap_frac'] = .85



    if args.transect_data:
        t_kwargs['low_thresh_factor'] = 6.
        t_kwargs['high_thresh_factor'] = 8.
        # args.transect_data = False

    print('\nAnalysing %s' % datafile)
    if datafile.endswith('.mat'):
        if verbose >= 1:
            print ('loading mat file')
        data, samplerate = load_matfile(datafile)

    else:
        if args.transect_data:
            data = open_data(datafile, 0, 60.0, 10.0)
        else:
            data = open_data(datafile, -1, 60.0, 10.0)
        samplerate = data.samplerate

    channels = range(data.shape[1]) if len(data.shape) > 1 else range(1)

    data_snippet_idxs = int(data_snippet_secs * samplerate)

    Obs_tracker(data, samplerate, args.start_time * 60., args.end_time * 60., channels, data_snippet_idxs, datafile, args.auto, args.fill_spec,
                **t_kwargs)



if __name__ == '__main__':
    main()

