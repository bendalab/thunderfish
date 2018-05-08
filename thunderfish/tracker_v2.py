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

def auto_connect_traces(fund_v, idx_v, ident_v, times, max_dt=30., max_df=.5, max_overlap_n = 0):
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
    idents = np.unique(ident_v[~np.isnan(ident_v)])
    med_ident_freq = [np.median(fund_v[ident_v == ident]) for ident in idents]

    # embed()
    # quit()
    i0s = []
    i1s = []
    di = []
    df = []
    # embed()
    # quit()
    next_message = 0.0
    for enu in np.arange(len(idents)-1):
    # for enu, i0 in enumerate(idents[:-1]):
        # print(enu)
        next_message = include_progress_bar(enu, len(idents)-1, 'error traces', next_message)
        for enu1 in np.arange(enu+1, len(idents)):
        # for enu1, i1 in enumerate(idents[enu+1:]):
            if np.abs(med_ident_freq[enu] - med_ident_freq[enu1]) > max_df:
                continue

            if idx_v[ident_v == idents[enu]][-1] < idx_v[ident_v == idents[enu1]][0]:
                if np.abs(times[idx_v[ident_v == idents[enu]][-1]] - times[idx_v[ident_v == idents[enu1]][0]]) > max_dt:
                    continue
                delta_idx = idx_v[ident_v == idents[enu1]][0] - idx_v[ident_v == idents[enu]][-1]

                # i0 before i1
                # print('wuff')
            elif idx_v[ident_v == idents[enu1]][-1] < idx_v[ident_v == idents[enu]][0]:
                if np.abs(times[idx_v[ident_v == idents[enu1]][-1]] - times[idx_v[ident_v == idents[enu]][0]]) > max_dt:
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
        if len(np.intersect1d(idx_v[ident_v ==i0s[i]], idx_v[ident_v == i1s[i]])) > max_overlap_n:
            continue
        else:
            ident_v[ident_v == i1s[i]] = i0s[i]
            i0s[i0s == i1s[i]] = i0s[i]
            i1s[i1s == i1s[i]] = i0s[i]

    return ident_v


def estimate_error(a_error, f_error, t_error, a_error_distribution, f_error_distribution,
                   min_f_weight=0.4, max_f_weight=0.9, t_of_max_f_weight=2.):
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
    def boltzmann(t, alpha= 0.25, beta = 0.0, x0 = 4, dx = 0.85):
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

        boltz = (alpha - beta) / (1. + np.exp(- (t - x0 ) / dx)  ) + beta
        return boltz

    if t_error >= 2.:
        f_weight = max_f_weight
    else:
        f_weight = 1. * (max_f_weight - min_f_weight) / t_of_max_f_weight * t_error + min_f_weight
    a_weight = 1. - f_weight

    a_e = a_weight * len(a_error_distribution[a_error_distribution < a_error]) / len(a_error_distribution)
    f_e = f_weight * len(f_error_distribution[f_error_distribution < f_error]) / len(f_error_distribution)
    t_e = boltzmann(t_error)
    # t_e = 0.5 * (1. * t_error / max_t_error) ** (1. / 3)  # when weight is 0.1 I end up in an endless loop somewhere

    return a_e + f_e + t_e


def freq_tracking_v3(fundamentals, signatures, times, freq_tolerance, n_channels, return_tmp_idenities=False,
                     ioi_fti=False, a_error_distribution=False, f_error_distribution=False,fig = False, ax = False,
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

    def get_a_and_f_error_dist(fund_v, idx_v, sign_v, ):
        """
        get the distribution of possible frequency and amplitude errors for the tracking.

        Parameters
        ----------
        fund_v: array
            flattened fundamtantals array containing all detected EOD frequencies in the recording.
        idx_v: array
            respective index vectro impliing the time of the detected frequency.
        sign_v: 2d-array
            for each fundamental frequency the power of this frequency on the used electodes.

        Returns
        -------
        f_error_distribution: array
            possible frequency error distribution for the dataset.
        a_error_distribution: array
            possible amplitude error distributions for the dataset.
        """
        # get f and amp signature distribution ############### BOOT #######################
        a_error_distribution = np.zeros(20000)  # distribution of amplitude errors
        f_error_distribution = np.zeros(20000)  # distribution of frequency errors
        idx_of_distribution = np.zeros(20000)  # corresponding indices

        b = 0  # loop varialble
        next_message = 0.  # feedback

        while b < 20000:
            next_message = include_progress_bar(b, 20000, 'get f and sign dist', next_message)  # feedback

            while True:  # finding compare indices to create initial amp and freq distribution
                # r_idx0 = np.random.randint(np.max(idx_v[~np.isnan(idx_v)]))
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

            # if np.abs(f0 - f1) > freq_tolerance:  # frequency threshold
            if np.abs(f0 - f1) > 10.:  # frequency threshold
                continue

            idx_of_distribution[b] = r_idx0
            a_error_distribution[b] = np.sqrt(np.sum([(s0[k] - s1[k]) ** 2 for k in range(len(s0))]))
            f_error_distribution[b] = np.abs(f0 - f1)
            b += 1

        return f_error_distribution, a_error_distribution

    def get_tmp_identities(i0_m, i1_m, error_cube, fund_v, idx_v, i, ioi_fti, dps, idx_comp_range):
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
        # mask_cube = [np.ones(np.shape(error_cube[n]), dtype=bool) for n in range(len(error_cube))]

        max_shape = np.max([np.shape(layer) for layer in error_cube[1:]], axis=0)
        cp_error_cube = np.full((len(error_cube)-1, max_shape[0], max_shape[1]), np.nan)
        for enu, layer in enumerate(error_cube[1:]):
            cp_error_cube[enu, :np.shape(error_cube[enu+1])[0], :np.shape(error_cube[enu+1])[1]] = layer

        try:
            tmp_ident_v = np.full(len(fund_v), np.nan)
            errors_to_v = np.full(len(fund_v), np.nan)
        except:
            tmp_ident_v = np.zeros(len(fund_v)) / 0.
            errors_to_v = np.zeros(len(fund_v)) / 0.

        layers, idx0s, idx1s = np.unravel_index(np.argsort(cp_error_cube, axis=None), np.shape(cp_error_cube))

        layers = layers+1


        for layer, idx0, idx1 in zip(layers, idx0s, idx1s):
            if np.isnan(cp_error_cube[layer-1, idx0, idx1]):
                break

            # _____ some control functions _____ ###
            if not ioi_fti:
                if idx_v[i1_m[layer][idx1]] - i > idx_comp_range*2:
                    continue
            else:
                if idx_v[i1_m[layer][idx1]] - idx_v[ioi_fti] > idx_comp_range*2:
                    continue

            if fund_v[i0_m[layer][idx0]] > fund_v[i1_m[layer][idx1]]:
                if 1. * np.abs(fund_v[i0_m[layer][idx0]] - fund_v[i1_m[layer][idx1]]) / ((idx_v[i1_m[layer][idx1]] - idx_v[i0_m[layer][idx0]]) / dps) > 2.:
                    continue
            else:
                if 1. * np.abs(fund_v[i0_m[layer][idx0]] - fund_v[i1_m[layer][idx1]]) / ((idx_v[i1_m[layer][idx1]] - idx_v[i0_m[layer][idx0]]) / dps) > 2.:
                    continue

            if np.isnan(tmp_ident_v[i0_m[layer][idx0]]):
                if np.isnan(tmp_ident_v[i1_m[layer][idx1]]):
                    tmp_ident_v[i0_m[layer][idx0]] = next_tmp_identity
                    tmp_ident_v[i1_m[layer][idx1]] = next_tmp_identity
                    errors_to_v[i1_m[layer][idx1]] = cp_error_cube[layer-1][idx0, idx1]
                    # errors_to_v[i0_m[layer][idx0]] = error_cube[layer][idx0, idx1]

                    # errors_to_v[(tmp_ident_v == tmp_ident_v[i1_m[layer][idx1]]) & (np.isnan(errors_to_v))] = error_cube[layer][idx0, idx1]
                    next_tmp_identity += 1
                else:
                    if idx_v[i0_m[layer][idx0]] in idx_v[tmp_ident_v == tmp_ident_v[i1_m[layer][idx1]]]:
                        continue
                    tmp_ident_v[i0_m[layer][idx0]] = tmp_ident_v[i1_m[layer][idx1]]
                    errors_to_v[i1_m[layer][idx1]] = cp_error_cube[layer-1][idx0, idx1]

                    # errors_to_v[(tmp_ident_v == tmp_ident_v[i1_m[layer][idx1]]) & (np.isnan(errors_to_v))] = error_cube[layer][idx0, idx1]
                    # errors_to_v[tmp_ident_v == tmp_ident_v[i1_m[layer][idx1]]][0] = np.nan

            else:
                if np.isnan(tmp_ident_v[i1_m[layer][idx1]]):
                    if idx_v[i1_m[layer][idx1]] in idx_v[tmp_ident_v == tmp_ident_v[i0_m[layer][idx0]]]:
                        continue
                    tmp_ident_v[i1_m[layer][idx1]] = tmp_ident_v[i0_m[layer][idx0]]
                    errors_to_v[i1_m[layer][idx1]] = cp_error_cube[layer-1][idx0, idx1]

                    # errors_to_v[(tmp_ident_v == tmp_ident_v[i1_m[layer][idx1]]) & (np.isnan(errors_to_v))] = error_cube[layer][idx0, idx1]
                    # errors_to_v[tmp_ident_v == tmp_ident_v[i1_m[layer][idx1]]][0] = np.nan

                else:
                    if tmp_ident_v[i0_m[layer][idx0]] == tmp_ident_v[i1_m[layer][idx1]]:
                        if np.isnan(errors_to_v[i1_m[layer][idx1]]):
                            errors_to_v[i1_m[layer][idx1]] = cp_error_cube[layer-1][idx0, idx1]
                        continue

                    idxs_i0 = idx_v[tmp_ident_v == tmp_ident_v[i0_m[layer][idx0]]]
                    idxs_i1 = idx_v[tmp_ident_v == tmp_ident_v[i1_m[layer][idx1]]]

                    if np.any(np.diff(sorted(np.concatenate((idxs_i0, idxs_i1)))) == 0):
                        continue
                    tmp_ident_v[tmp_ident_v == tmp_ident_v[i0_m[layer][idx0]]] = tmp_ident_v[i1_m[layer][idx1]]

                    if np.isnan(errors_to_v[i1_m[layer][idx1]]):
                        errors_to_v[i1_m[layer][idx1]] = cp_error_cube[layer-1][idx0, idx1]

        return tmp_ident_v, errors_to_v

    # total_t0 = time.time()

    # _____ plot environment for live tracking _____ ###
    if fig and ax:
        xlim = ax.get_xlim()
        ax.set_xlim(xlim[0], xlim[0]+20)
        fig.canvas.draw()
        life_handels = []
        tmp_handles = []
        life0 = None
        life1 = None

    # _____ exclude frequencies with lower dFs than 0.5Hz from algorythm ______ ###
    # ToDo choose the one with the bigger power
    # for i in range(len(fundamentals)):
    #     # include_progress_bar(i, len(fundamentals), 'clear dubble deltections', next_message)
    #     mask = np.zeros(len(fundamentals[i]), dtype=bool)
    #     order = np.argsort(fundamentals[i])
    #     fundamentals[i][order[np.arange(len(mask)-1)[np.diff(sorted(fundamentals[i])) < 0.5]+1]] = 0

    pre_sort_time = 0.
    sort_time = 0.
    # _____ parameters and vectors _____ ###
    detection_time_diff = times[1] - times[0]
    dps = 1. / detection_time_diff
    fund_v = np.hstack(fundamentals)
    try:
        ident_v = np.full(len(fund_v), np.nan)  # fish identities of frequencies
        idx_of_origin_v = np.full(len(fund_v), np.nan)
    except:
        ident_v = np.zeros(len(fund_v)) / 0.  # fish identities of frequencies
        idx_of_origin_v = np.zeros(len(fund_v)) / 0.

    idx_v = []  # temportal indices
    sign_v = []  # power of fundamentals on all electrodes
    for enu, funds in enumerate(fundamentals):
        idx_v.extend(np.ones(len(funds)) * enu)
        sign_v.extend(signatures[enu])
    idx_v = np.array(idx_v, dtype=int)
    sign_v = np.array(sign_v)

    idx_comp_range = int(np.floor(dps * 5.))  # maximum compare range backwards for amplitude signature comparison
    low_freq_th = 400.  # min. frequency tracked
    high_freq_th = 1050.  # max. frequency tracked

    # _____ get amp and freq error distribution
    if hasattr(a_error_distribution, '__len__') and hasattr(f_error_distribution, '__len__'):
        pass
    else:
        f_error_distribution, a_error_distribution = get_a_and_f_error_dist(fund_v, idx_v, sign_v)

    # _____ create initial error cube _____ ###
    error_cube = []  # [fundamental_list_idx, freqs_to_assign, target_freqs]
    i0_m = []
    i1_m = []

    next_message = 0.
    start_idx = 0 if not ioi_fti else idx_v[ioi_fti] # Index Of Interest for temporal identities

    for i in range(start_idx, int(start_idx + idx_comp_range*2)):
        next_message = include_progress_bar(i - start_idx, int(idx_comp_range*2), 'initial error cube', next_message)
        i0_v = np.arange(len(idx_v))[(idx_v == i) & (fund_v >= freq_lims[0]) & (fund_v <= freq_lims[1])]  # indices of fundamtenals to assign
        i1_v = np.arange(len(idx_v))[(idx_v > i) & (idx_v <= (i + int(idx_comp_range))) & (fund_v >= freq_lims[0]) & (fund_v <= freq_lims[1])]  # indices of possible targets

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

                error_matrix[enu0, enu1] = estimate_error(a_error, f_error, t_error, a_error_distribution,
                                                          f_error_distribution)
        error_cube.append(error_matrix)

    cube_app_idx = len(error_cube)

    # _____ accual tracking _____ ###
    next_identity = 0
    next_message = 0.00
    for enu, i in enumerate(np.arange(len(fundamentals))):
        # print(i)
        if i != 0 and (i % int(idx_comp_range * 120)) == 0: # clean up every 10 minutes
            ident_v = clean_up(fund_v, ident_v, idx_v, times)

        if not return_tmp_idenities:
            next_message = include_progress_bar(i, len(fundamentals), 'tracking', next_message)  # feedback

        if enu % idx_comp_range == 0:
            # t0 = time.time()
            tmp_ident_v, errors_to_v = get_tmp_identities(i0_m, i1_m, error_cube, fund_v, idx_v, i, ioi_fti, dps, idx_comp_range)
            # pre_sort_time += time.time() - t0
            # print('got through')
            # embed()
            # quit()
            if fig and ax:
                for handle in tmp_handles:
                    handle.remove()
                tmp_handles = []

                for ident in np.unique(tmp_ident_v[~np.isnan(tmp_ident_v)]):
                    plot_times = times[idx_v[tmp_ident_v == ident]]
                    plot_freqs = fund_v[tmp_ident_v == ident]

                    # h, = ax.plot(times[idx_v[ident_v == ident]], fund_v[ident_v == ident ], color='k', marker = '.', markersize=5)
                    h, = ax.plot(plot_times, plot_freqs, color='white', linewidth=4)
                    tmp_handles.append(h)

                fig.canvas.draw()

        if ioi_fti and return_tmp_idenities:
            return fund_v, tmp_ident_v, idx_v

        # mask_matrix = np.ones(np.shape(error_cube[0]), dtype=bool)

        # t0 = time.time()

        idx0s, idx1s = np.unravel_index(np.argsort(error_cube[0], axis=None), np.shape(error_cube[0]))

        # embed()
        # quit()
        for idx0, idx1 in zip(idx0s, idx1s):
            if np.isnan(error_cube[0][idx0, idx1]):
                break

            if freq_lims:
                if fund_v[i0_m[0][idx0]] > freq_lims[1] or fund_v[i0_m[0][idx0]] < freq_lims[0]:
                    continue
                if fund_v[i1_m[0][idx1]] > freq_lims[1] or fund_v[i1_m[0][idx1]] < freq_lims[0]:
                    continue

            if not np.isnan(ident_v[i1_m[0][idx1]]):
                continue

            if not np.isnan(errors_to_v[i1_m[0][idx1]]):
                if errors_to_v[i1_m[0][idx1]] < error_cube[0][idx0, idx1]:
                    continue

            if np.isnan(ident_v[i0_m[0][idx0]]):  # i0 doesnt have identity
                if 1. * np.abs(fund_v[i0_m[0][idx0]] - fund_v[i1_m[0][idx1]]) / ((idx_v[i1_m[0][idx1]] - idx_v[i0_m[0][idx0]]) / dps) > 2.:
                    continue

                if np.isnan(ident_v[i1_m[0][idx1]]):  # i1 doesnt have identity
                    ident_v[i0_m[0][idx0]] = next_identity
                    ident_v[i1_m[0][idx1]] = next_identity
                    next_identity += 1
                else:  # i1 does have identity
                    continue

            else:  # i0 does have identity
                if np.isnan(ident_v[i1_m[0][idx1]]):  # i1 doesnt have identity
                    if idx_v[i1_m[0][idx1]] in idx_v[ident_v == ident_v[i0_m[0][idx0]]]:
                        continue
                    # _____ if either idx0-idx1 is not a direct connection or ...
                    # _____ idx1 is not the new last point of ident[idx0] check ...
                    if not idx_v[i0_m[0][idx0]] == idx_v[ident_v == ident_v[i0_m[0][idx0]]][-1]:  # if i0 is not the last ...
                        if len(ident_v[(ident_v == ident_v[i0_m[0][idx0]]) & (idx_v > idx_v[i0_m[0][idx0]]) & (idx_v < idx_v[i1_m[0][idx1]])]) == 0:  # zwischen i0 und i1 keiner
                            next_idx_after_new = np.arange(len(ident_v))[(ident_v == ident_v[i0_m[0][idx0]]) & (idx_v > idx_v[i1_m[0][idx1]])][0]
                            if tmp_ident_v[next_idx_after_new] != tmp_ident_v[i1_m[0][idx1]]:
                                continue
                        elif len(ident_v[(ident_v == ident_v[i0_m[0][idx0]]) & (idx_v > idx_v[i1_m[0][idx1]])]) == 0:  # keiner nach i1
                            last_idx_before_new = np.arange(len(ident_v))[(ident_v == ident_v[i0_m[0][idx0]]) & (idx_v < idx_v[i1_m[0][idx1]])][-1]
                            if tmp_ident_v[last_idx_before_new] != tmp_ident_v[i1_m[0][idx1]]:
                                continue
                        else:  # sowohl als auch
                            next_idx_after_new = np.arange(len(ident_v))[(ident_v == ident_v[i0_m[0][idx0]]) & (idx_v > idx_v[i1_m[0][idx1]])][0]
                            last_idx_before_new = np.arange(len(ident_v))[(ident_v == ident_v[i0_m[0][idx0]]) & (idx_v < idx_v[i1_m[0][idx1]])][-1]
                            if tmp_ident_v[last_idx_before_new] != tmp_ident_v[i1_m[0][idx1]] or tmp_ident_v[next_idx_after_new] != tmp_ident_v[i1_m[0][idx1]]:
                                continue

                    ident_v[i1_m[0][idx1]] = ident_v[i0_m[0][idx0]]
                else:
                    continue

            idx_of_origin_v[i1_m[0][idx1]] = i0_m[0][idx0]

            if fig and ax:
                for handle in life_handels:
                    handle.remove()
                if life0:
                    life0.remove()
                    life1.remove()

                life_handels = []

                life0, = ax.plot(times[idx_v[i0_m[0][idx0]]], fund_v[i0_m[0][idx0]], color='red', marker='o')
                life1, = ax.plot(times[idx_v[i1_m[0][idx1]]], fund_v[i1_m[0][idx1]], color='red', marker='o')

                xlims = ax.get_xlim()
                for ident in np.unique(ident_v[~np.isnan(ident_v)]):
                    # embed()
                    # quit()
                    plot_times = times[idx_v[ident_v == ident]]
                    plot_freqs = fund_v[ident_v == ident]

                    # h, = ax.plot(times[idx_v[ident_v == ident]], fund_v[ident_v == ident ], color='k', marker = '.', markersize=5)
                    h, = ax.plot(plot_times[(plot_times >= xlims[0] - 1)],
                                 plot_freqs[(plot_times >= xlims[0] - 1)], color='k', marker='.', markersize=5)
                    life_handels.append(h)

                    if times[idx_v[ident_v == ident]][-1] > xlims[1]:
                        # xlim = ax.get_xlim()
                        ax.set_xlim([xlims[0] + 10, xlims[1] + 10])

                fig.canvas.draw()

        # sort_time += time.time()-t0
        i0_m.pop(0)
        i1_m.pop(0)
        error_cube.pop(0)

        i0_v = np.arange(len(idx_v))[(idx_v == cube_app_idx) & (fund_v >= freq_lims[0]) & (fund_v <= freq_lims[1])]  # indices of fundamtenals to assign
        i1_v = np.arange(len(idx_v))[(idx_v > cube_app_idx) & (idx_v <= (cube_app_idx + idx_comp_range)) & (fund_v >= freq_lims[0]) & (fund_v <= freq_lims[1])]  # indices of possible targets

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
                if fund_v[i0_v[enu0]] < low_freq_th or fund_v[i0_v[enu0]] > high_freq_th:  # freq to assigne out of tracking range
                    continue

                for enu1 in range(len(fund_v[i1_v])):
                    if fund_v[i1_v[enu1]] < low_freq_th or fund_v[i1_v[enu1]] > high_freq_th:  # target freq out of tracking range
                        continue
                    if np.abs(fund_v[i0_v[enu0]] - fund_v[i1_v[enu1]]) >= freq_tolerance:  # freq difference to high
                        continue

                    a_error = np.sqrt(
                        np.sum([(sign_v[i0_v[enu0]][j] - sign_v[i1_v[enu1]][j]) ** 2 for j in range(n_channels)]))
                    f_error = np.abs(fund_v[i0_v[enu0]] - fund_v[i1_v[enu1]])
                    t_error = 1. * np.abs(idx_v[i0_v[enu0]] - idx_v[i1_v[enu1]]) / dps

                    error_matrix[enu0, enu1] = estimate_error(a_error, f_error, t_error, a_error_distribution,
                                                              f_error_distribution)
            error_cube.append(error_matrix)

        cube_app_idx += 1
    ident_v = clean_up(fund_v, ident_v, idx_v, times)

    return fund_v, ident_v, idx_v, sign_v, a_error_distribution, f_error_distribution, idx_of_origin_v


def add_tracker_config(cfg, data_snippet_secs = 15., nffts_per_psd = 1, fresolution =.25, overlap_frac = .95,
                       freq_tolerance = 5., rise_f_th = 0.5, prim_time_tolerance = 1., max_time_tolerance = 10., f_th=2.):
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


def include_progress_bar(loop_v, loop_end, taskname ='', next_message=0.00):
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
        taskname = '        random task         ' # 30 characters
    else:
        taskname = ' ' * (30 - len(taskname)) + taskname

    if (1.*loop_v / loop_end) >= next_message:
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


def get_spectrum_funds_amp_signature(data, samplerate, channels, data_snippet_idxs, start_time, end_time, fresolution = 0.5,
                                     overlap_frac=.9, nffts_per_psd= 2, comp_min_freq= 0., comp_max_freq = 2000., plot_harmonic_groups=False,
                                     create_plotable_spectrogram=False, extract_funds_and_signature=True, noice_cancel = False, **kwargs):
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

    while start_idx <= end_idx:
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

        func = partial(spectrogram, samplerate=samplerate, fresolution=fresolution, overlap_frac=overlap_frac)

        if noice_cancel:
            # print('denoiced')
            denoiced_data = np.array([data[start_idx: start_idx + data_snippet_idxs, channel] for channel in channels])
            # print(denoiced_data.shape)
            mean_data = np.mean(denoiced_data, axis = 0)
            # mean_data.shape = (len(mean_data), 1)
            denoiced_data -=mean_data


            a = pool.map(func, denoiced_data)
        # self.data = self.data - mean_data
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
                    f_mask = np.arange(len(plot_spectra))[(plot_freqs >= y_borders[i]) & (plot_freqs < y_borders[i + 1])]

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
    def __init__(self, data, samplerate, start_time, end_time, channels, data_snippet_idxs, data_file, auto, **kwargs):

        # write input into self.
        self.data = data
        self.auto = auto
        self.data_file = data_file
        self.samplerate = samplerate
        self.start_time = start_time
        self.end_time = end_time
        if self.end_time < 0.0:
            self.end_time = len(self.data) / self.samplerate

        self.channels = channels
        self.data_snippet_idxs = data_snippet_idxs
        self.kwargs = kwargs

        # primary tracking vectors
        self.fund_v = None
        self.ident_v = None
        self.idx_v = None
        self.sign_v = None
        self.idx_of_origin_v = None

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

        self.last_xy_lims = None

        self.live_tracking = False

        # task lists
        self.t_tasks = ['track_snippet', 'track_snippet_live', 'plot_tmp_identities', 'check_tracking']
        self.c_tasks = ['cut_trace', 'connect_trace', 'auto connect_traces']
        self.d_tasks = ['delete_trace', 'group_delete', 'delete_noise']
        self.p_tasks = ['part_spec', 'show_powerspectum', 'hide_spectogram', 'show_spectogram']
        self.s_tasks = ['save_plot', 'save_traces']

        self.f_error_dist = None
        self.a_error_dist = None

        if self.auto:
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
            plt.rcParams['keymap.yscale'] = ''
            plt.rcParams['keymap.pan'] = ''
            plt.rcParams['keymap.home'] = ''

            self.main_ax = self.main_fig.add_axes([0.1, 0.1, 0.8, 0.6])
            self.spec_img_handle = None

            self.tmp_plothandel_main = None  # red line
            self.tmp_plothandel_ps = None  # red line
            self.trace_handles = []
            self.tmp_trace_handels = []

            self.life_trace_handles = []

            self.active_fundamental0_0 = None
            self.active_fundamental0_1 = None
            self.active_fundamental0_0_handle = None
            self.active_fundamental0_1_handle = None

            self.active_fundamental1_0 = None
            self.active_fundamental1_1 = None
            self.active_fundamental1_0_handle = None
            self.active_fundamental1_1_handle = None
            # self.plot_spectrum()

            self.active_vec_idx = None
            self.active_vec_idx_handle = None
            self.active_vec_idx1 = None
            self.active_vec_idx_handle1 = None

            self.active_ident_handle0 = None
            self.active_ident0 = None
            self.active_ident_handle1 = None
            self.active_ident1 = None

            self.active_indices = []
            self.active_indices_handle = []

            # powerspectrum window and parameters
            self.ps_ax = None
            self.tmp_plothandel_ps = []
            self.tmp_harmonics_plot = None
            self.all_peakf_dots = None
            self.good_peakf_dots = None

            self.active_harmonic = None

            self.f_error_ax = None
            # self.f_error_dist = None
            self.a_error_ax = None
            # self.a_error_dist = None
            self.t_error_ax = None

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

            t = self.main_fig.text(0.1, 0.825,  'c:')
            t1 = self.main_fig.text(0.15, 0.825,  'correction tasks')
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
                a_error = np.sqrt( np.sum([ (self.sign_v[self.active_fundamental0_0][k] -
                                             self.sign_v[self.active_fundamental0_1][k])**2
                                            for k in range(len(self.sign_v[self.active_fundamental0_0]))  ]))

                f_error = np.abs(self.fund_v[self.active_fundamental0_0] - self.fund_v[self.active_fundamental0_1])

                t_error = np.abs(self.times[self.idx_v[self.active_fundamental0_0]] - self.times[self.idx_v[self.active_fundamental0_1]])

                error = estimate_error(a_error, f_error, t_error, self.a_error_dist, self.f_error_dist)

                t = self.main_fig.text(0.3, 0.85, 'freq error:')
                t1 = self.main_fig.text(0.35, 0.85, '%.2f Hz (%.2f; %.2f); %.2f' % (
                    f_error, self.fund_v[self.active_fundamental0_0], self.fund_v[self.active_fundamental0_1],
                    1.* len(self.f_error_dist[self.f_error_dist < f_error])/ len(self.f_error_dist)))
                self.text_handles_key.append(t)
                self.text_handles_effect.append(t1)

                t = self.main_fig.text(0.3, 0.825, 'amp. error:')
                t1 = self.main_fig.text(0.35, 0.825, '%.2f dB; %.2f' % (a_error, 1.* len(self.a_error_dist[self.a_error_dist < a_error]) / len(self.a_error_dist)))
                self.text_handles_key.append(t)
                self.text_handles_effect.append(t1)

                t = self.main_fig.text(0.3, 0.8, 'time error')
                t1 = self.main_fig.text(0.35, 0.8, '%.2f s (%.2f, %.2f)' % (t_error, self.times[self.idx_v[self.active_fundamental0_0]], self.times[self.idx_v[self.active_fundamental0_1]]))
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
                a_error = np.sqrt(np.sum([(self.sign_v[self.active_fundamental1_0][k] -
                                           self.sign_v[self.active_fundamental1_1][k]) ** 2
                                          for k in range(len(self.sign_v[self.active_fundamental1_0]))]))

                f_error = np.abs(self.fund_v[self.active_fundamental1_0] - self.fund_v[self.active_fundamental1_1])

                t_error = np.abs(self.times[self.idx_v[self.active_fundamental1_0]] - self.times[self.idx_v[self.active_fundamental1_1]])

                error = estimate_error(a_error, f_error, t_error, self.a_error_dist, self.f_error_dist)

                t = self.main_fig.text(0.5, 0.85, 'freq error:')
                t1 = self.main_fig.text(0.55, 0.85, '%.2f Hz (%.2f; %.2f); %.2f' % (
                    f_error, self.fund_v[self.active_fundamental1_0], self.fund_v[self.active_fundamental1_1],
                    1.* len(self.f_error_dist[self.f_error_dist < f_error])/ len(self.f_error_dist) ))
                self.text_handles_key.append(t)
                self.text_handles_effect.append(t1)

                t = self.main_fig.text(0.5, 0.825, 'amp. error:')
                t1 = self.main_fig.text(0.55, 0.825, '%.2f dB; %.2f' % (a_error, 1.* len(self.a_error_dist[self.a_error_dist < a_error]) / len(self.a_error_dist) ))
                self.text_handles_key.append(t)
                self.text_handles_effect.append(t1)

                t = self.main_fig.text(0.5, 0.8, 'time error')
                t1 = self.main_fig.text(0.55, 0.8, '%.2f s (%.2f; %.2f)' % (t_error, self.times[self.idx_v[self.active_fundamental1_0]], self.times[self.idx_v[self.active_fundamental1_1]]))
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
        if event.key == 'backspace':
            if hasattr(self.last_xy_lims, '__len__'):
                self.main_ax.set_xlim(self.last_xy_lims[0][0], self.last_xy_lims[0][1])
                self.main_ax.set_ylim(self.last_xy_lims[1][0], self.last_xy_lims[1][1])
                if self.ps_ax:
                    self.ps_ax.set_ylim(self.last_xy_lims[1])
        if event.key == 'up':
            ylims = self.main_ax.get_ylim()
            self.main_ax.set_ylim(ylims[0] + 0.5* (ylims[1]-ylims[0]), ylims[1] + 0.5* (ylims[1]-ylims[0]))
            if self.ps_ax:
                self.ps_ax.set_ylim(ylims[0] + 0.5* (ylims[1]-ylims[0]), ylims[1] + 0.5* (ylims[1]-ylims[0]))

        if event.key == 'down':
            ylims = self.main_ax.get_ylim()
            self.main_ax.set_ylim(ylims[0] - 0.5* (ylims[1]-ylims[0]), ylims[1] - 0.5* (ylims[1]-ylims[0]))
            if self.ps_ax:
                self.ps_ax.set_ylim(ylims[0] - 0.5* (ylims[1]-ylims[0]), ylims[1] - 0.5* (ylims[1]-ylims[0]))

        if event.key == 'right':
            xlims = self.main_ax.get_xlim()
            self.main_ax.set_xlim(xlims[0] + 0.5* (xlims[1]-xlims[0]), xlims[1] + 0.5* (xlims[1]-xlims[0]))

        if event.key == 'left':
            xlims = self.main_ax.get_xlim()
            self.main_ax.set_xlim(xlims[0] - 0.5* (xlims[1]-xlims[0]), xlims[1] - 0.5* (xlims[1]-xlims[0]))

        if event.key in 'b':
            self.current_task = 'save_plot'

        if event.key in 'h':
            self.current_task = None

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

        if event.key in 'e':
            embed()

        if event.key in 'd':
            self.current_task = self.d_tasks[0]
            self.d_tasks = np.roll(self.d_tasks, -1)

        if event.key in 'p':
            self.current_task = self.p_tasks[0]
            self.p_tasks = np.roll(self.p_tasks, -1)

        if event.key == 'ctrl+t':
            self.current_task = self.t_tasks[0]
            self.t_tasks = np.roll(self.t_tasks, -1)

        if event.key == 'c':
            self.current_task = self.c_tasks[0]
            self.c_tasks = np.roll(self.c_tasks, -1)

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
            if self.current_task == 'group_delete':
                self.group_delete()
                # self.current_task = None
                self.plot_traces(clear_traces=True)

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
                                                               aspect='auto', alpha=0.7)
                    self.main_ax.set_xlabel('time [s]', fontsize=12)
                    self.main_ax.set_ylabel('frequency [Hz]', fontsize=12)

            if self.current_task == 'delete_noise':
                if self.active_indices_handle:
                    self.ident_v[self.active_indices] = np.nan
                    self.active_indices_handle.remove()
                    self.active_indices_handle = None
                    self.plot_traces(clear_traces=True)

                min_x, max_x = self.main_ax.get_xlim()
                min_y, max_y = self.main_ax.get_ylim()
                self.iois = np.arange(len(self.fund_v))[(self.times[self.idx_v] >= min_x) & (self.times[self.idx_v] < max_x) & (self.fund_v >= min_y) & (self.fund_v < max_y)]
                self.iois = self.iois[~np.isnan(self.ident_v[self.iois])]
                self.min_max = np.max(self.sign_v[self.iois], axis=1) -  np.min(self.sign_v[self.iois], axis=1)

                if self.ps_ax:
                    self.main_fig.delaxes(self.ps_ax)
                    self.ps_ax = None
                    self.tmp_plothandel_ps = []
                    self.all_peakf_dots = None
                    self.good_peakf_dots = None
                    self.main_ax.set_position([.1, .1, .8, .6])
                self.main_ax.set_position([.1, .1, .5, .6])
                self.ps_ax = self.main_fig.add_axes([.6, .1, .3, .6])
                # self.ps_ax.set_yticks([])
                self.ps_ax.yaxis.tick_right()
                self.ps_ax.yaxis.set_label_position("right")
                self.ps_ax.set_ylabel('frequency [Hz]', fontsize=12)
                self.ps_ax.set_xlabel('max - min power', fontsize=12)
                self.ps_handle, = self.ps_ax.plot(self.min_max, self.fund_v[self.iois], '.')
                self.ps_ax.set_ylim(self.main_ax.get_ylim())

            if self.current_task == 'load_traces':
                self.load_trace()
                self.current_task = None

            if self.current_task == 'save_traces':
                self.save_traces()
                self.current_task = None

            if self.current_task == 'connect_trace':
                if self.active_ident_handle0 and self.active_ident_handle1:
                    self.connect_trace()

            if self.current_task == 'cut_trace':
                if self.active_ident_handle0 and self.active_fundamental0_0:
                    self.cut_trace()

            if self.current_task == 'delete_trace':
                if self.active_ident_handle0:
                    self.delete_trace()

            if self.current_task == 'save_plot':
                self.current_task = None
                self.save_plot()

            if self.current_task == 'show_powerspectum':
                if self.tmp_plothandel_main and self.ioi:
                    self.current_task = None
                    self.plot_ps()
                else:
                    print('\nmissing data')

            if self.current_task == 'update_hg':
                self.current_task = None
                self.update_hg()

            # if self.current_task == 'zoom':
            #     self.current_task = None
            #     self.zoom()

            if self.current_task == 'track_snippet':
                self.current_task = None
                self.track_snippet()

            if self.current_task == 'track_snippet_live':
                self.current_task = None
                self.live_tracking = True
                self.track_snippet()
                self.live_tracking = False

            if self.current_task == 'part_spec':
                self.current_task = None
                self.plot_spectrum(part_spec=True)

            if self.current_task == 'plot_tmp_identities':

                if self.active_vec_idx and hasattr(self.f_error_dist, '__len__') and hasattr(self.a_error_dist, '__len__'):
                    # tmp_fund_v, tmp_ident_v, tmp_idx_v = \
                        # freq_tracking_v2(self.fundamentals, self.signatures, self.positions, self.times,
                        #                  self.kwargs['freq_tolerance'], n_channels=len(self.channels),
                        #                  return_tmp_idenities=True, ioi_fti = self.active_vec_idx,
                        #                  a_error_distribution=self.a_error_dist, f_error_distribution=self.f_error_dist)
                    tmp_fund_v, tmp_ident_v, tmp_idx_v = \
                        freq_tracking_v3(self.fundamentals, self.signatures, self.times,
                                         self.kwargs['freq_tolerance'], n_channels=len(self.channels),
                                         return_tmp_idenities=True, ioi_fti=self.active_vec_idx,
                                         a_error_distribution=self.a_error_dist, f_error_distribution=self.f_error_dist)

                    for handle in self.tmp_trace_handels:
                        handle.remove()
                    self.tmp_trace_handels = []
                    possible_identities = np.unique(tmp_ident_v[~np.isnan(tmp_ident_v)])
                    for ident in np.array(possible_identities):
                        c = np.random.rand(3)
                        h, = self.main_ax.plot(self.times[tmp_idx_v[tmp_ident_v == ident]],
                                               tmp_fund_v[tmp_ident_v == ident], marker='o', color=c,
                                               linewidth=3, markersize=5)
                        self.tmp_trace_handels.append(h)

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

            if len(self.active_indices) > 0:
                self.active_indices = []

            if self.active_indices_handle:
                self.active_indices_handle.remove()
                self.active_indices_handle = None

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
            if self.active_vec_idx_handle:
                self.active_vec_idx = None
                self.active_vec_idx_handle.remove()
                self.active_vec_idx_handle = None

            if self.active_vec_idx:
                self.active_vec_idx = None
                self.active_ident0 = None
            if self.active_ident_handle0:
                self.active_ident_handle0.remove()
                self.active_ident_handle0 = None

            if self.active_vec_idx1:
                self.active_vec_idx1 = None
                self.active_ident1 = None
            if self.active_ident_handle1:
                self.active_ident_handle1.remove()
                self.active_ident_handle1 = None

        if event.inaxes == self.main_ax:
            if self.current_task == 'show_powerspectum':
                if event.button == 1:
                    x = event.xdata
                    self.ioi = np.argmin(np.abs(self.times-x))

                    y_lims = self.main_ax.get_ylim()
                    if self.tmp_plothandel_main:
                        self.tmp_plothandel_main.remove()
                    self.tmp_plothandel_main, = self.main_ax.plot([self.times[self.ioi], self.times[self.ioi]], [y_lims[0], y_lims[1]], color='red', linewidth='2')

            if self.current_task == 'check_tracking' and hasattr(self.fundamentals, '__len__'):

                if event.button == 1:
                    if event.key == 'control':
                        x = event.xdata
                        y = event.ydata

                        idx_searched = np.argsort(np.abs(self.times - x))[0]
                        fund_searched = self.fund_v[self.idx_v == idx_searched][np.argsort(np.abs(self.fund_v[(self.idx_v == idx_searched)] - y))[0]]
                        current_idx = np.arange(len(self.fund_v))[(self.idx_v == idx_searched) & (self.fund_v == fund_searched)][0]


                        self.active_fundamental0_0 = current_idx
                        if self.active_fundamental0_0_handle:
                            self.active_fundamental0_0_handle.remove()
                        self.active_fundamental0_0_handle, = self.main_ax.plot(self.times[self.idx_v[current_idx]], self.fund_v[current_idx], 'o', color='red', markersize=4)

                        if self.active_fundamental0_1_handle:
                            self.active_fundamental0_1_handle.remove()
                            self.active_fundamental0_1_handle = None
                            self.active_fundamental0_1 = None

                        if ~np.isnan(self.idx_of_origin_v[current_idx]):
                            self.active_fundamental0_1 = self.idx_of_origin_v[current_idx]
                            self.active_fundamental0_1_handle, = self.main_ax.plot(self.times[self.idx_v[self.active_fundamental0_1]], self.fund_v[self.active_fundamental0_1], 'o', color='red', markersize=4)

                    else:
                        x = event.xdata
                        y = event.ydata

                        idx_searched = np.argsort(np.abs(self.times - x))[0]
                        fund_searched = self.fund_v[self.idx_v == idx_searched][np.argsort(np.abs(self.fund_v[(self.idx_v == idx_searched)] - y))[0]]
                        current_idx = np.arange(len(self.fund_v))[(self.idx_v == idx_searched) & (self.fund_v == fund_searched)][0]

                        self.active_fundamental0_1 = current_idx
                        if self.active_fundamental0_1_handle:
                            self.active_fundamental0_1_handle.remove()

                        self.active_fundamental0_1_handle, = self.main_ax.plot(self.times[self.idx_v[current_idx]], self.fund_v[current_idx], 'o', color='red', markersize=4)

                if event.button == 3:
                    if event.key == 'control':
                        x = event.xdata
                        y = event.ydata

                        idx_searched = np.argsort(np.abs(self.times - x))[0]
                        fund_searched = self.fund_v[self.idx_v == idx_searched][np.argsort(np.abs(self.fund_v[(self.idx_v == idx_searched)] - y))[0]]
                        current_idx = np.arange(len(self.fund_v))[(self.idx_v == idx_searched) & (self.fund_v == fund_searched)][0]

                        self.active_fundamental1_0 = current_idx

                        if self.active_fundamental1_0_handle:
                            self.active_fundamental1_0_handle.remove()
                        self.active_fundamental1_0_handle, = self.main_ax.plot(self.times[self.idx_v[current_idx]], self.fund_v[current_idx], 'o', color='green', markersize=4)

                        if self.active_fundamental1_1_handle:
                            self.active_fundamental1_1_handle.remove()
                            self.active_fundamental1_1_handle = None
                            self.active_fundamental1_1 = None

                        if ~np.isnan(self.idx_of_origin_v[current_idx]):
                            self.active_fundamental1_1 = self.idx_of_origin_v[current_idx]
                            self.active_fundamental1_1_handle, = self.main_ax.plot(self.times[self.idx_v[self.active_fundamental1_1]], self.fund_v[self.active_fundamental1_1], 'o', color='green', markersize=4)

                    else:
                        x = event.xdata
                        y = event.ydata

                        idx_searched = np.argsort(np.abs(self.times - x))[0]
                        fund_searched = self.fund_v[self.idx_v == idx_searched][np.argsort(np.abs(self.fund_v[(self.idx_v == idx_searched)] - y))[0]]
                        current_idx = np.arange(len(self.fund_v))[(self.idx_v == idx_searched) & (self.fund_v == fund_searched)][0]

                        self.active_fundamental1_1 = current_idx

                        if self.active_fundamental1_1_handle:
                            self.active_fundamental1_1_handle.remove()

                        self.active_fundamental1_1_handle, = self.main_ax.plot(self.times[self.idx_v[current_idx]], self.fund_v[current_idx], 'o', color='green', markersize=4)

            if self.current_task == 'plot_tmp_identities':
                if event.button == 1:
                    x = event.xdata
                    y = event.ydata

                    t_idx = np.argsort(np.abs(self.times - x))[0]
                    f_idx = np.argsort(np.abs(self.fund_v[self.idx_v == t_idx] - y))[0]

                    self.active_vec_idx = np.arange(len(self.fund_v))[(self.idx_v == t_idx) & (self.fund_v == self.fund_v[self.idx_v == t_idx][f_idx])][0]
                    if self.active_vec_idx_handle:
                        self.active_vec_idx_handle.remove()
                    # self.active_vec_idx_handle, = self.main_ax.plot(self.time[self.idx_v[t_idx]], self.fund_v[self.idx_v == t_idx][f_idx], 'o', color='red', markersize=4)
                    self.active_vec_idx_handle, = self.main_ax.plot(self.times[t_idx], self.fund_v[self.active_vec_idx], 'o', color='red', markersize=4)

            if self.current_task in ['connect_trace', 'delete_trace', 'zoom', 'cut_trace', 'group_delete']:
                if self.current_task == 'group_delete':
                    if self.active_indices_handle:
                        self.active_indices_handle.remove()
                        self.active_indices_handle = None

                self.x = (event.xdata, 0)
                self.y = (event.ydata, 0)

            if self.current_task == 'cut_trace':
                if event.button == 3:
                    x = event.xdata

                    trace_idxs = self.idx_v[self.ident_v == self.active_ident0]
                    current_idx = np.arange(len(self.fund_v))[(self.ident_v == self.active_ident0)][np.argsort(np.abs(self.times[trace_idxs] - x))[0]]

                    self.active_fundamental0_0 = current_idx
                    if self.active_fundamental0_0_handle:
                        self.active_fundamental0_0_handle.remove()
                    self.active_fundamental0_0_handle, = self.main_ax.plot(self.times[self.idx_v[current_idx]],
                                                                           self.fund_v[current_idx], 'o', color='red',
                                                                           markersize=4)

        if self.ps_ax and event.inaxes == self.ps_ax:
            if self.current_task == 'delete_noise':
                if event.button == 1:
                    x = event.xdata
                    self.noi = np.argmin(np.abs(self.times-x))

                    y_lims = self.ps_ax.get_ylim()
                    if self.tmp_plothandel_ps:
                        self.tmp_plothandel_ps.remove()
                    self.tmp_plothandel_ps, = self.ps_ax.plot([x, x], [y_lims[0], y_lims[1]], color='red', linewidth='2')
                # print('wuff')
                self.active_indices = self.iois[self.min_max <=     x]
                if self.active_indices_handle:
                    self.active_indices_handle.remove()
                self.active_indices_handle, = self.main_ax.plot(self.times[self.idx_v[self.active_indices]], self.fund_v[self.active_indices], 'o', color='orange')


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

    def buttonrelease(self, event):
        if event.inaxes == self.main_ax:
            if self.current_task == 'group_delete':
                if event.button == 1:
                    self.x = (self.x[0], event.xdata)
                    self.y = (self.y[0], event.ydata)

                    # self.active_indices = np.arange(len(self.fund_v))[(self.fund_v >= np.min(self.y)) &
                    #                                                   (self.fund_v < np.max(self.y)) &
                    #                                                     (self.times[self.idx_v] >= np.min(self.x)),
                    #                                                   (self.times[self.idx_v] < np.max(self.x))]
                    self.active_indices = np.arange(len(self.fund_v))[
                        (self.fund_v >= np.min(self.y)) & (self.fund_v < np.max(self.y)) & (
                                self.times[self.idx_v] >= np.min(self.x)) & (self.times[self.idx_v] < np.max(self.x))]
                    self.active_indices = self.active_indices[~np.isnan(self.ident_v[self.active_indices])]

                    self.active_indices_handle, = self.main_ax.plot(self.times[self.idx_v[self.active_indices]], self.fund_v[self.active_indices], 'o', color='orange')


            if self.current_task == 'delete_trace':
                if event.button == 1:
                    self.x = (self.x[0], event.xdata)
                    self.y = (self.y[0], event.ydata)

                    self.active_vec_idx = np.arange(len(self.fund_v))[(self.fund_v >= np.min(self.y)) & (self.fund_v < np.max(self.y)) & (self.times[self.idx_v] >= np.min(self.x)) & (self.times[self.idx_v] < np.max(self.x))]
                    if len(self.active_vec_idx) > 0:
                        self.active_vec_idx = self.active_vec_idx[~np.isnan(self.ident_v[self.active_vec_idx])][0]
                    else:
                        self.active_vec_idx = None

                    self.active_ident0 = self.ident_v[self.active_vec_idx]

                    if self.active_ident_handle0:
                        self.active_ident_handle0.remove()

                    self.active_ident_handle0, = self.main_ax.plot(
                        self.times[self.idx_v[self.ident_v == self.active_ident0]],
                        self.fund_v[self.ident_v == self.active_ident0], color='orange', alpha=0.7, linewidth=4)

            if self.current_task == 'cut_trace':
                if event.button == 1:
                    self.x = (self.x[0], event.xdata)
                    self.y = (self.y[0], event.ydata)

                    self.active_vec_idx = np.arange(len(self.fund_v))[(self.fund_v >= np.min(self.y)) & (self.fund_v < np.max(self.y)) & (self.times[self.idx_v] >= np.min(self.x)) & (self.times[self.idx_v] < np.max(self.x))]
                    if len(self.active_vec_idx) > 0:
                        self.active_vec_idx = self.active_vec_idx[~np.isnan(self.ident_v[self.active_vec_idx])][0]
                    else:
                        self.active_vec_idx = None

                    self.active_ident0 = self.ident_v[self.active_vec_idx]

                    if self.active_ident_handle0:
                        self.active_ident_handle0.remove()

                    self.active_ident_handle0, = self.main_ax.plot(
                        self.times[self.idx_v[self.ident_v == self.active_ident0]],
                        self.fund_v[self.ident_v == self.active_ident0], color='orange', alpha=0.7, linewidth=4)

            if self.current_task == 'connect_trace':
                if event.button == 1:
                    self.x = (self.x[0], event.xdata)
                    self.y = (self.y[0], event.ydata)

                    self.active_vec_idx = np.arange(len(self.fund_v))[
                        (self.fund_v >= np.min(self.y)) & (self.fund_v < np.max(self.y)) & (
                                self.times[self.idx_v] >= np.min(self.x)) & (self.times[self.idx_v] < np.max(self.x))]
                    if len(self.active_vec_idx) > 0:
                        self.active_vec_idx = self.active_vec_idx[~np.isnan(self.ident_v[self.active_vec_idx])][0]
                    else:
                        self.active_vec_idx = None

                    self.active_ident0 = self.ident_v[self.active_vec_idx]

                    if self.active_ident_handle0:
                        self.active_ident_handle0.remove()

                    self.active_ident_handle0, = self.main_ax.plot(
                        self.times[self.idx_v[self.ident_v == self.active_ident0]],
                        self.fund_v[self.ident_v == self.active_ident0], color='green', alpha=0.7, linewidth=4)

                if event.button == 3:
                    self.x = (self.x[0], event.xdata)
                    self.y = (self.y[0], event.ydata)

                    if self.active_ident0:
                        self.active_vec_idx1 = np.arange(len(self.fund_v))[(self.fund_v >= np.min(self.y)) &
                                                                           (self.fund_v < np.max(self.y)) &
                                                                           (self.times[self.idx_v] >= np.min(self.x)) &
                                                                           (self.times[self.idx_v] < np.max(self.x)) &
                                                                           (self.ident_v != self.active_ident0)]
                        if len(self.active_vec_idx1) > 0:
                            self.active_vec_idx1 = self.active_vec_idx1[~np.isnan(self.ident_v[self.active_vec_idx1])][0]
                        else:
                            self.active_vec_idx1 = None

                        self.active_ident1 = self.ident_v[self.active_vec_idx1]

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
                if self.ps_ax:

                    self.ps_ax.set_ylim(np.array(self.y)[np.argsort(self.y)])



        self.key_options()
        self.main_fig.canvas.draw()

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
            self.spec_img_handle = self.main_ax.imshow(decibel(self.tmp_spectra)[::-1], extent=[self.start_time, self.end_time, 0, 2000],
                                aspect='auto', alpha=0.7)
            self.main_ax.set_xlabel('time [s]', fontsize=12)
            self.main_ax.set_ylabel('frequency [Hz]', fontsize=12)
            self.main_ax.set_xlim([self.start_time, self.end_time])

            self.plot_traces(clear_traces=True)

    def connect_trace(self):

        # overlapping_idxs = [x for x in self.idx_v[self.ident_v == self.active_ident0] if x in self.idx_v[self.ident_v == self.active_ident1]]
        overlapping_idxs = np.intersect1d(self.idx_v[self.ident_v == self.active_ident0], self.idx_v[self.ident_v == self.active_ident1])

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
        self.ident_v[(self.ident_v == self.active_ident0) & (self.idx_v < self.idx_v[self.active_fundamental0_0])] = next_ident

        self.active_ident_handle0.remove()
        self.active_ident_handle0 = None

        self.active_fundamental0_0_handle.remove()
        self.active_fundamental0_0_handle = None
        self.active_fundamental0_0 = None

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
        self.active_indices = []
        self.active_indices_handle.remove()
        self.active_indices_handle = None

    def save_plot(self):
        self.main_ax.set_position([.1, .1, .8, .8])
        if self.ps_ax:
            self.main_fig.delaxes(self.ps_as)
            self.ps_ax = None
            self.all_peakf_dots = None
            self.good_peakf_dots = None
            # self.main_ax.set_position([.1, .1, .8, .8])

        for i, j in zip(self.text_handles_key, self.text_handles_effect):
            self.main_fig.texts.remove(i)
            self.main_fig.texts.remove(j)
        self.text_handles_key = []
        self.text_handles_effect = []

        if self.f_error_ax:
            self.main_fig.delaxes(self.f_error_ax)
            self.f_error_ax = None
        if self.a_error_ax:
            self.main_fig.delaxes(self.a_error_ax)
            self.a_error_ax = None
        if self.t_error_ax:
            self.main_fig.delaxes(self.t_error_ax)
            self.t_error_ax = None

        self.main_fig.set_size_inches(20./2.54, 12./2.54)
        self.main_fig.canvas.draw()

        plot_nr = len(glob.glob('/home/raab/Desktop/plot*'))
        self.main_fig.savefig('/home/raab/Desktop/plot%.0f.pdf' % plot_nr)

        self.main_fig.set_size_inches(55. / 2.54, 30. / 2.54)
        self.main_ax.set_position([.1, .1, .8, .6])
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
            self.main_ax.set_xlabel('time [s]', fontsize=12)
            self.main_ax.set_ylabel('frequency [Hz]', fontsize=12)
            self.main_ax.tick_params(labelsize=10)
        else:
            if not hasattr(self.tmp_spectra, '__len__'):
                self.tmp_spectra, self.times = get_spectrum_funds_amp_signature(
                    self.data, self.samplerate, self.channels, self.data_snippet_idxs, self.start_time, self.end_time,
                    create_plotable_spectrogram=True, extract_funds_and_signature=False,  **self.kwargs)

            if not self.auto:
                self.spec_img_handle = self.main_ax.imshow(decibel(self.tmp_spectra)[::-1], extent=[self.start_time, self.end_time, 0, 2000],
                                    aspect='auto', alpha=0.7)
                self.main_ax.set_xlabel('time [s]', fontsize=12)
                self.main_ax.set_ylabel('frequency [Hz]', fontsize=12)
                self.main_ax.tick_params(labelsize=10)

    def track_snippet(self):
        if hasattr(self.fund_v, '__len__'):
            for i in reversed(range(len(self.trace_handles))):
                self.trace_handles[i].remove()
                self.trace_handles.pop(i)

            self.fund_v = None
            self.ident_v = None
            self.idx_v = None
            self.sign_v = None

        if self.main_ax:
            snippet_start, snippet_end = self.main_ax.get_xlim()
        else:
            snippet_start = self.start_time
            snippet_end = self.end_time

            # snippet_start = self.times[0]
            # snippet_end = self.times[-1]
        # else:
        #     snippet_start, snippet_end = self.main_ax.get_xlim()

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
        # embed()
        # quit()
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

            self.fund_v, self.ident_v, self.idx_v, self.sign_v, self.a_error_dist, self.f_error_dist, self.idx_of_origin_v= \
                freq_tracking_v3(np.array(self.fundamentals)[mask], np.array(self.signatures)[mask],
                             self.times[mask], self.kwargs['freq_tolerance'], n_channels=len(self.channels),
                             freq_lims= freq_lims)

        if not self.auto:
            self.plot_traces(clear_traces=True)

            self.plot_error()

    def plot_error(self):
        if self.ps_ax:
            self.main_fig.delaxes(self.ps_ax)
            self.ps_ax = None
            self.tmp_plothandel_ps = []
            self.all_peakf_dots = None
            self.good_peakf_dots = None

        self.main_ax.set_position([.1, .1, .6, .6])
        n, h = np.histogram(self.f_error_dist, 5000)
        self.f_error_ax = self.main_fig.add_axes([.75, .5, 0.15, 0.15])
        # self.f_error_ax.plot(h[:-1] + (h[1]- h[0]) / 2., n, '.', color='cornflowerblue')
        self.f_error_ax.plot(h[1:], np.cumsum(n) / np.sum(n), color='cornflowerblue', linewidth=2)
        self.f_error_ax.set_xlabel('frequency error [Hz]', fontsize=12)

        n, h = np.histogram(self.a_error_dist, 5000)
        self.a_error_ax = self.main_fig.add_axes([.75, .3, 0.15, 0.15])
        # self.a_error_ax.plot(h[:-1] + (h[1]- h[0]) / 2., n, '.', color='green')
        self.a_error_ax.plot(h[1:], np.cumsum(n) / np.sum(n), color='green', linewidth=2)
        self.a_error_ax.set_xlabel('amplitude error [a.u.]', fontsize=12)
        self.a_error_ax.set_ylabel('cumsum of error distribution', fontsize=12)


        self.t_error_ax = self.main_fig.add_axes([.75, .1, 0.15, 0.15])
        t = np.arange(0, 10, 0.0001)
        f = (0.25 - 0.0) / (1. + np.exp(- (t - 4) / 0.85)) + 0.0
        self.t_error_ax.plot(t, f, color='orange', linewidth=2)  # fucking hard coded
        self.t_error_ax.set_xlabel('time error [s]', fontsize=12)

    def plot_traces(self, clear_traces = False, post_connect= False, post_cut=False, post_delete=False):
        """
        shows/updates/deletes all frequency traces of individually tracked fish in a plot.

        :param clear_traces: (bool) if true removes all preiouly plotted traces before plotting the new ones
        :param post_connect: (bool) refreshes/deletes single identity traces previously selected and stored in class variables.
        """
        # self.main_ax.imshow(10.0 * np.log10(self.tmp_spectra)[::-1], extent=[self.start_time, self.end_time, 0, 2000], aspect='auto', alpha=0.7)
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
            h, = self.main_ax.plot(self.times[self.idx_v[self.ident_v == self.active_ident0]], self.fund_v[self.ident_v == self.active_ident0], marker='.', color=c)
            self.trace_handles[np.arange(len(self.trace_handles))[handle_idents == self.active_ident0][0]] = (h, self.active_ident0)

            new_ident = np.max(self.ident_v[~np.isnan(self.ident_v)])
            c = np.random.rand(3)
            h, = self.main_ax.plot(self.times[self.idx_v[self.ident_v == new_ident]], self.fund_v[self.ident_v == new_ident], marker='.', color=c)
            self.trace_handles.append((h, new_ident))

        if post_connect:
            handle_idents = np.array([x[1] for x in self.trace_handles])

            remove_handle = np.array(self.trace_handles)[handle_idents == self.active_ident1][0]
            remove_handle[0].remove()

            joined_handle = np.array(self.trace_handles)[handle_idents == self.active_ident0][0]
            joined_handle[0].remove()

            c = np.random.rand(3)
            h, = self.main_ax.plot(self.times[self.idx_v[self.ident_v == self.active_ident0]], self.fund_v[self.ident_v == self.active_ident0], marker='.', color=c)
            self.trace_handles[np.arange(len(self.trace_handles))[handle_idents == self.active_ident0][0]] = (h, self.active_ident0)
            # self.trace_handles.append((h, self.active_ident0))

            self.trace_handles.pop(np.arange(len(self.trace_handles))[handle_idents == self.active_ident1][0])

        if clear_traces:
            for handle in self.trace_handles:
                handle[0].remove()
            self.trace_handles = []


            possible_identities = np.unique(self.ident_v[~np.isnan(self.ident_v)])
            for ident in np.array(possible_identities):
                c = np.random.rand(3)
                h, =self.main_ax.plot(self.times[self.idx_v[self.ident_v == ident]], self.fund_v[self.ident_v == ident], marker='.', color=c)
                self.trace_handles.append((h, ident))

    def plot_ps(self):
        """
        calculates the powerspectrum of single or multiple datasnippets recorded with single or multiple electrodes at a time.
        If multiple electrode recordings are analysed the shown powerspectrum is the sum of all calculated powerspectra.
        """
        if self.f_error_ax:
            self.main_fig.delaxes(self.f_error_ax)
            self.f_error_ax = None
        if self.a_error_ax:
            self.main_fig.delaxes(self.a_error_ax)
            self.a_error_ax = None
        if self.t_error_ax:
            self.main_fig.delaxes(self.t_error_ax)
            self.t_error_ax = None

        # nfft = next_power_of_two(self.samplerate / self.fresolution)
        nfft = next_power_of_two(self.samplerate / self.kwargs['fresolution'])
        data_idx0 = int(self.times[self.ioi] * self.samplerate)
        data_idx1 = int(data_idx0 + nfft+1)

        all_c_spectra = []
        all_c_freqs = None

        if self.kwargs['noice_cancel']:
            denoiced_data = np.array([self.data[data_idx0: data_idx1, channel] for channel in self.channels])
            # print(denoiced_data.shape)
            mean_data = np.mean(denoiced_data, axis = 0)
            # mean_data.shape = (len(mean_data), 1)
            denoiced_data -=mean_data

        for channel in self.channels:
            # c_spectrum, c_freqs, c_time = spectrogram(self.data[data_idx0: data_idx1, channel], self.samplerate,
            #                                           fresolution = self.fresolution, overlap_frac = self.overlap_frac)
            if self.kwargs['noice_cancel']:
                c_spectrum, c_freqs, c_time = spectrogram(denoiced_data[channel], self.samplerate,
                                                      fresolution=self.kwargs['fresolution'], overlap_frac=self.kwargs['overlap_frac'])
            else:
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
            self.ps_ax.set_ylabel('frequency [Hz]', fontsize=12)
            self.ps_ax.set_xlabel('power [dB]', fontsize=12)
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
        """
        reexecutes peak detection in a powerspectrum with changed parameters and updates the plot
        """
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

def fish_tracker(data_file, start_time=0.0, end_time=-1.0, grid=False, auto = False, data_snippet_secs=15., verbose=0, **kwargs):
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

    channels, coords, neighbours = get_grid_proportions(data, grid, n_tolerance_e=2, verbose=verbose)

    data_snippet_idxs = int(data_snippet_secs * samplerate)

    Obs_tracker(data, samplerate, start_time, end_time, channels, data_snippet_idxs, data_file, auto, **kwargs)


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
    parser.add_argument('-a', dest='auto', action='store_true', help='automatically analyse data and save results')
    parser.add_argument('-n', dest='noice_cancel', action='store_true', help='cancsels noice by substracting mean of all electrodes from all electrodes')
    parser.add_argument('-s', dest='save_fish', action='store_true',
                        help='save fish EODs after first stage of sorting.')
    parser.add_argument('-f', dest='plot_harmonic_groups', action='store_true', help='plot harmonic group detection')
    parser.add_argument('-t', dest='transect_data', action='store_true', help='adapt parameters for transect data')
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

    t_kwargs = psd_peak_detection_args(cfg)
    t_kwargs.update(harmonic_groups_args(cfg))
    t_kwargs.update(tracker_args(cfg))
    t_kwargs['noice_cancel'] = args.noice_cancel

    t_kwargs = grid_config_update(t_kwargs)

    if args.transect_data:
        t_kwargs['low_thresh_factor'] = 6.
        t_kwargs['high_thresh_factor'] = 8.

    # embed()
    # quit()
    fish_tracker(datafile, args.start_time * 60.0, args.end_time * 60.0, args.grid, args.auto, **t_kwargs)

if __name__ == '__main__':
    # how to execute this code properly
    # python3 -m thunderfish.tracker_v2 <data file> [-v(vv)] [-g(ggg)]
    main()

