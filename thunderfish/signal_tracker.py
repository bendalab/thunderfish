import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
from IPython import embed
from tqdm import tqdm


def freq_tracking_v5(fundamentals, signatures, times, freq_tolerance= 10., n_channels=64, max_dt=10., ioi_fti=False,
                     freq_lims=(400, 1200)):
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

    def clean_up(fund_v, ident_v):
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

    def get_tmp_identities(i0_m, i1_m, error_cube, fund_v, idx_v, i, ioi_fti, idx_comp_range):
        """
        extract temporal identities for a datasnippted of 2*index compare range of the original tracking algorithm.
        for each data point in the data window finds the best connection within index compare range and, thus connects
        the datapoints based on their minimal error value until no connections are left or possible anymore.

        Parameters
        ----------
        i0_m: 2d-array
            for consecutive timestamps contains for each the indices of the origin EOD frequencies.
        i1_m: 2d-array
            respectively contains the indices of the target EOD frequencies, laying within index compare range.
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

        max_shape = np.max([np.shape(layer) for layer in error_cube[1:]], axis=0)
        cp_error_cube = np.full((len(error_cube) - 1, max_shape[0], max_shape[1]), np.nan)
        for enu, layer in enumerate(error_cube[1:]):
            cp_error_cube[enu, :np.shape(error_cube[enu + 1])[0], :np.shape(error_cube[enu + 1])[1]] = layer

        min_i0 = np.min(np.hstack(i0_m))
        max_i1 = np.max(np.hstack(i1_m))
        tmp_ident_v = np.full(max_i1 - min_i0 + 1, np.nan)
        errors_to_v = np.full(max_i1 - min_i0 + 1, np.nan)
        tmp_idx_v = idx_v[min_i0:max_i1 + 1]
        tmp_fund_v = fund_v[min_i0:max_i1 + 1]

        i0_m = np.array(i0_m) - min_i0
        i1_m = np.array(i1_m) - min_i0
        # tmp_idx_v -= min_i0

        layers, idx0s, idx1s = np.unravel_index(np.argsort(cp_error_cube, axis=None), np.shape(cp_error_cube))
        made_connections = np.zeros(np.shape(cp_error_cube))
        not_made_connections = np.zeros(np.shape(cp_error_cube))
        not_made_connections[~np.isnan(cp_error_cube)] = 1
        # made_connections[~np.isnan(cp_error_cube)] = 0

        layers = layers + 1

        i_non_nan = len(cp_error_cube[layers - 1, idx0s, idx1s][~np.isnan(cp_error_cube[layers - 1, idx0s, idx1s])])

        for layer, idx0, idx1 in zip(layers[:i_non_nan], idx0s[:i_non_nan], idx1s[:i_non_nan]):
            if np.isnan(cp_error_cube[layer - 1, idx0, idx1]):
                break

            # _____ some control functions _____ ###

            if not ioi_fti:
                if tmp_idx_v[i1_m[layer][idx1]] - i > idx_comp_range * 3:
                    continue
            else:
                if idx_v[i1_m[layer][idx1]] - idx_v[ioi_fti] > idx_comp_range * 3:
                    continue

            # ToDo:check if at least one direct neighbour of new connected has small delta f

            if np.isnan(tmp_ident_v[i0_m[layer][idx0]]):
                if np.isnan(tmp_ident_v[i1_m[layer][idx1]]):
                    if np.abs(tmp_fund_v[i0_m[layer][idx0]] - tmp_fund_v[i1_m[layer][idx1]]) > 0.5:
                        continue

                    tmp_ident_v[i0_m[layer][idx0]] = next_tmp_identity
                    tmp_ident_v[i1_m[layer][idx1]] = next_tmp_identity
                    errors_to_v[i1_m[layer][idx1]] = cp_error_cube[layer - 1][idx0, idx1]
                    not_made_connections[layer - 1, idx0, idx1] = 0
                    made_connections[layer - 1, idx0, idx1] = 1
                    next_tmp_identity += 1
                else:

                    mask = np.arange(len(tmp_ident_v))[tmp_ident_v == tmp_ident_v[i1_m[layer][idx1]]]  # idxs of target
                    if tmp_idx_v[i0_m[layer][idx0]] in tmp_idx_v[mask]:  # if goal already in target continue
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


            else:
                if np.isnan(tmp_ident_v[i1_m[layer][idx1]]):
                    mask = np.arange(len(tmp_ident_v))[tmp_ident_v == tmp_ident_v[i0_m[layer][idx0]]]
                    if tmp_idx_v[i1_m[layer][idx1]] in tmp_idx_v[mask]:
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


                else:
                    if tmp_ident_v[i0_m[layer][idx0]] == tmp_ident_v[i1_m[layer][idx1]]:
                        if np.isnan(errors_to_v[i1_m[layer][idx1]]):
                            errors_to_v[i1_m[layer][idx1]] = cp_error_cube[layer - 1][idx0, idx1]
                        continue

                    mask = np.arange(len(tmp_ident_v))[tmp_ident_v == tmp_ident_v[i0_m[layer][idx0]]]
                    idxs_i0 = tmp_idx_v[mask]
                    mask = np.arange(len(tmp_ident_v))[tmp_ident_v == tmp_ident_v[i1_m[layer][idx1]]]
                    idxs_i1 = tmp_idx_v[mask]

                    if np.any(np.diff(sorted(np.concatenate((idxs_i0, idxs_i1)))) == 0):
                        continue
                    tmp_ident_v[tmp_ident_v == tmp_ident_v[i0_m[layer][idx0]]] = tmp_ident_v[i1_m[layer][idx1]]

                    if np.isnan(errors_to_v[i1_m[layer][idx1]]):
                        errors_to_v[i1_m[layer][idx1]] = cp_error_cube[layer - 1][idx0, idx1]
                        not_made_connections[layer - 1, idx0, idx1] = 0
                        made_connections[layer - 1, idx0, idx1] = 1


        #### this is new and in progress ####
        # ToDo: check overlap of ident v...

        ids = np.unique(tmp_ident_v[~np.isnan(tmp_ident_v)])
        id_comb = []
        id_comb_df = []
        id_comb_overlap = []
        for id0 in range(len(ids)):
            id0_med_freq = np.median(tmp_fund_v[tmp_ident_v == ids[id0]])

            for id1 in range(id0 + 1, len(ids)):
                # print(id0, id1)
                id_comb.append((id0, id1))
                id1_med_freq = np.median(tmp_fund_v[tmp_ident_v == ids[id1]])
                id_comb_df.append(np.abs(id1_med_freq - id0_med_freq))

                if np.max(tmp_idx_v[tmp_ident_v == ids[id0]]) < np.min(tmp_idx_v[tmp_ident_v == ids[id1]]):
                    id_comb_overlap.append(0)  # ToDo: neg. values for time distance

                elif np.max(tmp_idx_v[tmp_ident_v == ids[id1]]) < np.min(tmp_idx_v[tmp_ident_v == ids[id0]]):
                    id_comb_overlap.append(0)

                elif (np.min(tmp_idx_v[tmp_ident_v == ids[id0]]) <= np.min(tmp_idx_v[tmp_ident_v == ids[id1]])) and (
                        np.max(tmp_idx_v[tmp_ident_v == ids[id0]]) >= np.min(tmp_idx_v[tmp_ident_v == ids[id1]])):
                    ioi = [np.min(tmp_idx_v[tmp_ident_v == ids[id0]]), np.max(tmp_idx_v[tmp_ident_v == ids[id0]]),
                           np.min(tmp_idx_v[tmp_ident_v == ids[id1]]), np.max(tmp_idx_v[tmp_ident_v == ids[id1]])]
                    ioi = np.array(ioi)[np.argsort(ioi)]
                    id_comb_overlap.append(ioi[2] - ioi[1] + 1)
                elif (np.min(tmp_idx_v[tmp_ident_v == ids[id1]]) <= np.min(tmp_idx_v[tmp_ident_v == ids[id0]])) and (
                        np.max(tmp_idx_v[tmp_ident_v == ids[id1]]) >= np.min(tmp_idx_v[tmp_ident_v == ids[id0]])):
                    ioi = [np.min(tmp_idx_v[tmp_ident_v == ids[id0]]), np.max(tmp_idx_v[tmp_ident_v == ids[id0]]),
                           np.min(tmp_idx_v[tmp_ident_v == ids[id1]]), np.max(tmp_idx_v[tmp_ident_v == ids[id1]])]
                    ioi = np.array(ioi)[np.argsort(ioi)]
                    id_comb_overlap.append(ioi[2] - ioi[1] + 1)
                else:
                    print('found a non existing cases')
                    embed()
                    quit()

        sorting_mask = np.argsort(id_comb_df)
        # id0, id1 = np.array(id_comb)[sorting_mask][0]
        for i, (id0, id1) in enumerate(np.array(id_comb)[sorting_mask]):
            # print(id0, id1, i)
            # if id_comb_df[i] > 5:
            #     continue
            comb_f = np.concatenate((tmp_fund_v[tmp_ident_v == ids[id0]], tmp_fund_v[tmp_ident_v == ids[id1]]))

            bins = np.arange((np.min(comb_f) // .1) * .1, (np.max(comb_f) // .1) * .1 + .1, .1)
            bc = bins[:-1] + (bins[1:] - bins[:-1]) / 2

            n0, bins = np.histogram(tmp_fund_v[tmp_ident_v == ids[id0]], bins=bins)

            n1, bins = np.histogram(tmp_fund_v[tmp_ident_v == ids[id1]], bins=bins)
            # n0 = n0 / np.sum(n0) / .1
            # n1 = n1 / np.sum(n1) / .1
            greater_mask = n0 >= n1
            # smaller_mask = n0 < n1

            overlapping_counts = np.sum(np.concatenate((n1[greater_mask], n0[~greater_mask])))

            pct_overlap = np.max([overlapping_counts / np.sum(n1), overlapping_counts / np.sum(n0)])

            if pct_overlap > .25:
                # embed()
                # quit()

                fig, ax = plt.subplots(1, 2, facecolor='white', figsize=(20 / 2.54, 12 / 2.54))
                for j in range(len(ids)):
                    if ids[j] == ids[id0]:
                        ax[0].plot(tmp_idx_v[tmp_ident_v == ids[j]], tmp_fund_v[tmp_ident_v == ids[j]], marker='.',
                                   color='red')
                    elif ids[j] == ids[id1]:
                        ax[0].plot(tmp_idx_v[tmp_ident_v == ids[j]], tmp_fund_v[tmp_ident_v == ids[j]], marker='.',
                                   color='blue')
                    else:
                        ax[0].plot(tmp_idx_v[tmp_ident_v == ids[j]], tmp_fund_v[tmp_ident_v == ids[j]], marker='.',
                                   color='grey')

                ax[1].set_title('%.2f' % pct_overlap)
                ax[1].bar(bc, n0, color='red', alpha=.5, width=.08)
                ax[1].bar(bc, n1, color='blue', alpha=.5, width=.08)
                plt.show(block=False)
                plt.waitforbuttonpress()
                plt.close(fig)

                if id_comb_overlap[sorting_mask[i]] > 0:
                    embed()
                    quit()
                len_id0 = len(tmp_ident_v[tmp_ident_v == ids[id0]])
                len_id1 = len(tmp_ident_v[tmp_ident_v == ids[id1]])

                overlapping_idx = list(set(tmp_idx_v[tmp_ident_v == ids[id0]]) & set(tmp_idx_v[tmp_ident_v == ids[id1]]))

        #### this is new and in progress --- end ####

        tmp_ident_v_ret = np.full(len(fund_v), np.nan)
        tmp_ident_v_ret[min_i0:max_i1 + 1] = tmp_ident_v

        return tmp_ident_v_ret, errors_to_v

    def get_a_and_f_error_dist(fund_v, idx_v, sign_v, start_idx, idx_comp_range, freq_lims, freq_tolerance):
        f_error_distribution = []
        a_error_distribution = []

        for i in range(start_idx, int(start_idx + idx_comp_range * 3)):
            i0_v = np.arange(len(idx_v))[
                (idx_v == i) & (fund_v >= freq_lims[0]) & (fund_v <= freq_lims[1])]  # indices of fundamtenals to assign
            i1_v = np.arange(len(idx_v))[
                (idx_v > i) & (idx_v <= (i + int(idx_comp_range))) & (fund_v >= freq_lims[0]) & (
                            fund_v <= freq_lims[1])]  # indices of possible targets

            if len(i0_v) == 0 or len(i1_v) == 0:  # if nothing to assign or no targets continue
                continue

            for enu0 in range(len(fund_v[i0_v])):
                if fund_v[i0_v[enu0]] < freq_lims[0] or fund_v[i0_v[enu0]] > freq_lims[1]:
                    continue
                for enu1 in range(len(fund_v[i1_v])):
                    if fund_v[i1_v[enu1]] < freq_lims[0] or fund_v[i1_v[enu1]] > freq_lims[1]:
                        continue
                    if np.abs(fund_v[i0_v[enu0]] - fund_v[i1_v[enu1]]) >= freq_tolerance:  # freq difference to high
                        continue
                    a_error_distribution.append(np.sqrt(np.sum(
                        [(sign_v[i0_v[enu0]][k] - sign_v[i1_v[enu1]][k]) ** 2 for k in
                         range(len(sign_v[i0_v[enu0]]))])))
                    f_error_distribution.append(np.abs(fund_v[i0_v[enu0]] - fund_v[i1_v[enu1]]))

        return np.array(a_error_distribution), np.array(f_error_distribution)

    def reshape_data():
        detection_time_diff = times[1] - times[0]
        dps = 1. / detection_time_diff
        fund_v = np.hstack(fundamentals)
        ident_v = np.full(len(fund_v), np.nan)  # fish identities of frequencies
        idx_of_origin_v = np.full(len(fund_v), np.nan)  # ToDo: necessary ? lets see

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

        idx_comp_range = int(
            np.floor(dps * max_dt))  # maximum compare range backwards for amplitude signature comparison

        return fund_v, ident_v, idx_v, sign_v, original_sign_v, idx_of_origin_v, idx_comp_range, dps

    def create_error_cube(i0_m, i1_m, error_cube, cube_app_idx, freq_lims, update=False):
        if update:
            i0_m.pop(0)
            i1_m.pop(0)
            error_cube.pop(0)
        else:
            error_cube = []  # [fundamental_list_idx, freqs_to_assign, target_freqs]
            i0_m = []
            i1_m = []

        if update:
            Citt = [cube_app_idx]
        else:
            Citt = np.arange(start_idx, int(start_idx + idx_comp_range * 3))

        for i in Citt:
            i0_v = np.arange(len(idx_v))[
                (idx_v == i) & (fund_v >= freq_lims[0]) & (fund_v <= freq_lims[1])]  # indices of fundamtenals to assign
            i1_v = np.arange(len(idx_v))[
                (idx_v > i) & (idx_v <= (i + int(idx_comp_range))) & (fund_v >= freq_lims[0]) & (
                            fund_v <= freq_lims[1])]  # indices of possible targets

            i0_m.append(i0_v)
            i1_m.append(i1_v)

            if len(i0_v) == 0 or len(i1_v) == 0:  # if nothing to assign or no targets continue
                error_cube.append(np.array([[]]))
                continue

            error_matrix = np.full((len(i0_v), len(i1_v)), np.nan)

            for enu0 in range(len(fund_v[i0_v])):
                if fund_v[i0_v[enu0]] < freq_lims[0] or fund_v[i0_v[enu0]] > freq_lims[1]:  # ToDo:should be obsolete
                    continue
                for enu1 in range(len(fund_v[i1_v])):
                    if fund_v[i1_v[enu1]] < freq_lims[0] or fund_v[i1_v[enu1]] > freq_lims[
                        1]:  # ToDo:should be obsolete
                        continue

                    if np.abs(fund_v[i0_v[enu0]] - fund_v[i1_v[enu1]]) >= freq_tolerance:  # freq difference to high
                        continue

                    a_error = np.sqrt(
                        np.sum([(sign_v[i0_v[enu0]][j] - sign_v[i1_v[enu1]][j]) ** 2 for j in range(n_channels)]))
                    f_error = np.abs(fund_v[i0_v[enu0]] - fund_v[i1_v[enu1]])
                    t_error = 1. * np.abs(idx_v[i0_v[enu0]] - idx_v[i1_v[enu1]]) / dps

                    error = estimate_error(a_error, f_error, a_error_distribution)
                    error_matrix[enu0, enu1] = np.sum(error)
            error_cube.append(error_matrix)

        if update:
            cube_app_idx += 1
        else:
            cube_app_idx = len(error_cube)

        return error_cube, i0_m, i1_m, cube_app_idx

    def assign_tmp_ids(ident_v, tmp_ident_v, idx_v, fund_v, error_cube, idx_comp_range, next_identity, i0_m, i1_m,
                       freq_lims):

        max_shape = np.max([np.shape(layer) for layer in error_cube], axis=0)
        cp_error_cube = np.full((len(error_cube), max_shape[0], max_shape[1]), np.nan)
        for enu, layer in enumerate(error_cube):
            cp_error_cube[enu, :np.shape(error_cube[enu])[0], :np.shape(error_cube[enu])[1]] = layer

        layers, idx0s, idx1s = np.unravel_index(np.argsort(cp_error_cube[:idx_comp_range], axis=None),
                                                np.shape(cp_error_cube[:idx_comp_range]))

        i_non_nan = len(cp_error_cube[layers - 1, idx0s, idx1s][~np.isnan(cp_error_cube[layers - 1, idx0s, idx1s])])
        min_i0 = np.min(np.hstack(i0_m))
        max_i1 = np.max(np.hstack(i1_m))

        p_ident_v = ident_v[min_i0:max_i1 + 1]
        p_tmp_ident_v = tmp_ident_v[min_i0:max_i1 + 1]
        p_idx_v = idx_v[min_i0:max_i1 + 1]
        p_fund_v = fund_v[min_i0:max_i1 + 1]

        p_i0_m = np.array(i0_m) - min_i0
        p_i1_m = np.array(i1_m) - min_i0

        already_assigned = []
        for layer, idx0, idx1 in zip(layers[:i_non_nan], idx0s[:i_non_nan], idx1s[:i_non_nan]):
            idents_to_assigne = p_ident_v[
                ~np.isnan(p_tmp_ident_v) & (p_idx_v > i + idx_comp_range) & (p_idx_v <= i + idx_comp_range * 2)]

            if len(idents_to_assigne[np.isnan(idents_to_assigne)]) == 0:
                break

            if np.isnan(cp_error_cube[layer, idx0, idx1]):
                break

            if ~np.isnan(p_ident_v[p_i1_m[layer][idx1]]):
                continue

            if np.isnan(p_tmp_ident_v[p_i1_m[layer][idx1]]):
                continue

            if p_i1_m[layer][idx1] < idx_comp_range:
                if p_i1_m[layer][idx1] >= idx_comp_range * 2.:
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

            idxs_i0 = p_idx_v[(p_ident_v == p_ident_v[p_i0_m[layer][idx0]]) & (p_idx_v > i + idx_comp_range) &
                              (p_idx_v <= i + idx_comp_range * 2)]
            idxs_i1 = p_idx_v[(p_tmp_ident_v == p_tmp_ident_v[p_i1_m[layer][idx1]]) & (np.isnan(p_ident_v)) &
                              (p_idx_v > i + idx_comp_range) & (p_idx_v <= i + idx_comp_range * 2)]

            if np.any(np.diff(sorted(np.concatenate((idxs_i0, idxs_i1)))) == 0):
                continue

            if p_i1_m[layer][idx1] in already_assigned:
                continue

            already_assigned.append(p_i1_m[layer][idx1])

            p_ident_v[(p_tmp_ident_v == p_tmp_ident_v[p_i1_m[layer][idx1]]) &
                      (np.isnan(p_ident_v)) & (p_idx_v > i + idx_comp_range) &
                      (p_idx_v <= i + idx_comp_range * 2)] = p_ident_v[p_i0_m[layer][idx0]]

        for ident in np.unique(p_tmp_ident_v[~np.isnan(p_tmp_ident_v)]):
            if len(p_ident_v[p_tmp_ident_v == ident][~np.isnan(p_ident_v[p_tmp_ident_v == ident])]) == 0:
                p_ident_v[(p_tmp_ident_v == ident) & (p_idx_v > i + idx_comp_range) & (
                        p_idx_v <= i + idx_comp_range * 2)] = next_identity
                next_identity += 1

        return ident_v, next_identity

    fund_v, ident_v, idx_v, sign_v, original_sign_v, idx_of_origin_v, idx_comp_range, dps = reshape_data()
    start_idx = 0 if not ioi_fti else idx_v[ioi_fti]  # Index Of Interest for temporal identities

    a_error_distribution, f_error_distribution = \
        get_a_and_f_error_dist(fund_v, idx_v, sign_v, start_idx, idx_comp_range, freq_lims,
                               freq_tolerance=freq_tolerance)
    # embed()
    # quit()
    error_cube, i0_m, i1_m, cube_app_idx = create_error_cube(i0_m=None, i1_m=None, error_cube=None, freq_lims=freq_lims,
                                                             cube_app_idx=None)

    next_identity = 0
    next_cleanup = int(idx_comp_range * 120)

    for i in tqdm(np.arange(len(fundamentals)), desc='tracking'):
        if len(np.hstack(i0_m)) == 0 or len(np.hstack(i0_m)) == 0:
            error_cube, i0_m, i1_m, cube_app_idx = create_error_cube(i0_m, i1_m, error_cube, cube_app_idx, freq_lims,
                                                                     update=True)
            start_idx += 1
            continue

        if i >= next_cleanup:  # clean up every 10 minutes
            ident_v = clean_up(fund_v, ident_v)
            next_cleanup += int(idx_comp_range * 120)

        if i % idx_comp_range == 0:  # next total sorting step
            a_error_distribution, f_error_distribution = \
                get_a_and_f_error_dist(fund_v, idx_v, sign_v, start_idx, idx_comp_range, freq_lims, freq_tolerance)

            tmp_ident_v, errors_to_v = get_tmp_identities(i0_m, i1_m, error_cube, fund_v, idx_v, i, ioi_fti,
                                                          idx_comp_range)

            if i == 0:
                for ident in np.unique(tmp_ident_v[~np.isnan(tmp_ident_v)]):
                    ident_v[(tmp_ident_v == ident) & (idx_v <= i + idx_comp_range)] = next_identity
                    next_identity += 1

            # assing tmp identities ##################################
            ident_v, next_identity = assign_tmp_ids(ident_v, tmp_ident_v, idx_v, fund_v, error_cube, idx_comp_range,
                                                    next_identity, i0_m, i1_m, freq_lims)

        error_cube, i0_m, i1_m, cube_app_idx = create_error_cube(i0_m, i1_m, error_cube, cube_app_idx, freq_lims,
                                                                 update=True)
        start_idx += 1

    ident_v = clean_up(fund_v, ident_v)

    return fund_v, ident_v, idx_v, sign_v, a_error_distribution, f_error_distribution, idx_of_origin_v, original_sign_v


def estimate_error(a_error, f_error, a_error_distribution):
    a_weight = 2. / 3
    f_weight = 1. / 3
    if len(a_error_distribution) > 0:
        a_e = a_weight * len(a_error_distribution[a_error_distribution < a_error]) / len(a_error_distribution)
    else:
        a_e = 1
    f_e = f_weight * boltzmann(f_error, alpha=1, beta=0, x0=.25, dx=.15)

    return [a_e, f_e, 0]


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


def load_example_data():
    folder = "/home/raab/data/2016-colombia/2016-04-10-11_12"

    if os.path.exists(os.path.join(folder, 'fund_v.npy')):
        fund_v = np.load(os.path.join(folder, 'fund_v.npy'))
        sign_v = np.load(os.path.join(folder, 'sign_v.npy'))
        idx_v = np.load(os.path.join(folder, 'idx_v.npy'))
        times = np.load(os.path.join(folder, 'times.npy'))
        start_time, end_time = np.load(os.path.join(folder, 'meta.npy'))
    else:
        fund_v, sign_v, idx_v, times, start_time, end_time  = [], [], [], [], [], []
        print('WARNING !!! files not found !')

    return fund_v, sign_v, idx_v, times, start_time, end_time


def back_shape_data(fund_v, sign_v, idx_v, times):
    # t0 = 17150
    t0 = 1000
    # t1 = 17250
    t1 = 1200
    mask = np.arange(len(idx_v))[(times[idx_v] >= t0) & (times[idx_v] <= t1)]
    fundamentals = []
    signatures = []

    f = []
    s = []
    for i in tqdm(mask, desc='reshape data'):
        if i == mask[0]:
            f.append(fund_v[i])
            s.append(sign_v[i])
        else:
            if idx_v[i] != idx_v[i - 1]:
                fundamentals.append(np.array(f))
                f = []
                signatures.append(np.array(s))
                s = []
            f.append(fund_v[i])
            s.append(sign_v[i])
    fundamentals.append(f)
    signatures.append(s)

    return fundamentals, signatures


def plot_tracked_traces(ident_v, fund_v, idx_v, times):
    fig = plt.figure(figsize=(20/2.54, 12/2.54), facecolor='white')
    gs = gridspec.GridSpec(1, 1)
    gs.update(left=0.1, bottom=0.1, right=.95, top=.95)
    ax = plt.subplot(gs[0, 0])

    for id in np.unique(ident_v[~np.isnan(ident_v)]):
        c = np.random.rand(3)
        ax.plot(times[idx_v[ident_v == id]], fund_v[ident_v == id], color = c, marker='.')

    plt.show()


def main():
    fund_v, sign_v, idx_v, times, start_time, end_time = \
        load_example_data()

    fundamentals, signatures = back_shape_data(fund_v, sign_v, idx_v, times)

    fund_v, ident_v, idx_v, sign_v, a_error_distribution, f_error_distribution, idx_of_origin_v, original_sign_v, = \
        freq_tracking_v5(fundamentals, signatures, times)

    plot_tracked_traces(ident_v, fund_v, idx_v, times)

if __name__ == '__main__':
    main()
