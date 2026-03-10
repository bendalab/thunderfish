"""
Extract and cluster EOD waverforms of pulse-type electric fish.

## Main function

- `extract_pulsefish()`: checks for pulse-type fish based on the EOD amplitude and shape.

"""

import numpy as np

try:
    import matplotlib.pyplot as plt
except ImportError:
    pass

from pathlib import Path
from scipy import stats
from scipy.interpolate import interp1d
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from sklearn.mixture import BayesianGaussianMixture
from sklearn.metrics import pairwise_distances

from thunderlab.eventdetection import detect_peaks, median_std_threshold
from thunderlab.fourier import fourier_coeffs

from .pulseplots import *

import warnings
def warn(*args, **kwargs):
    """
    Ignore all warnings.
    """
    pass
warnings.warn = warn

try:
    from numba import jit
except ImportError:
    def jit(*args, **kwargs):
        def decorator_jit(func):
            return func
        return decorator_jit


# upgrade numpy functions for backwards compatibility:

if not hasattr(np, 'isin'):
    np.isin = np.in1d

    
def unique_counts(ar):
    """ Find the unique elements of an array and their counts, ignoring shape.

    The code is condensed from numpy version 1.17.0.
    
    Parameters
    ----------
    ar : numpy array
        Input array

    Returns
    -------
    unique_vaulues : numpy array
        Unique values in array ar.
    unique_counts : numpy array
        Number of instances for each unique value in ar.
    """
    try:
        return np.unique(ar, return_counts=True)
    except TypeError:
        ar = np.asanyarray(ar).flatten()
        ar.sort()
        mask = np.empty(ar.shape, dtype=bool_)
        mask[:1] = True
        mask[1:] = ar[1:] != ar[:-1]
        idx = np.concatenate(np.nonzero(mask) + ([mask.size],))
        return ar[mask], np.diff(idx)  


###########################################################################


def extract_pulsefish(data, rate, frate=0.5e6, width_factor_shape=3,
                      width_factor_wave=8, width_factor_display=4,
                      verbose=0, plot_level=0, save_plots=False,
                      save_path='', ftype='png', return_data=[]):
    """Extract and cluster pulse-type fish EODs from data.
    
    Takes recording data containing an unknown number of pulsefish and
    extracts the mean EOD and EOD timepoints for each fish present in
    the recording.
    
    Parameters
    ----------
    data: 1-D array of float
        The data to be analysed.
    rate: float
        Sampling rate of the data in Hertz.
    frate: float
        Sampling rate used for the returned waveform estimates.
    width_factor_shape : float
        Width multiplier used for EOD shape analysis.
        EOD snippets are extracted based on width between the 
        peak and trough multiplied by the width factor.
    width_factor_wave : float
        Width multiplier used for wavefish detection.
    width_factor_display : float
        Width multiplier used for EOD mean extraction and display.
    verbose : int
        Verbosity level.
    plot_level : int
        Similar to verbosity levels, but with plots. 
        Only set to > 0 for debugging purposes.
    save_plots : bool
        Set to True to save the plots created by plot_level.
    save_path: Path or str
        Path for saving plots.
    ftype : str
        Define the filetype to save the plots in if save_plots is set to True.
        Options are: 'png', 'jpg', 'svg' ...
    return_data : list of str
        Specify data that should be logged and returned in a dictionary. Each clustering 
        step has a specific keyword that results in adding different variables to the log dictionary.
        Optional keys for return_data and the resulting additional key-value pairs to the log dictionary are:
        
        - 'all_eod_times':
            - 'all_times':  list of two lists of float.
                All peak (`all_times[0]`) and trough times (`all_times[1]`) extracted
                by the peak detection algorithm. Times are given in seconds.
            - 'eod_troughtimes': list of 1D arrays.
                The timepoints in seconds of each unique extracted EOD cluster,
                where each 1D array encodes one cluster.
        
        - 'peak_detection':
            - "data": 1D numpy array of float.
                Quadratically interpolated data which was used for peak detection.
            - "peaks_1": 1D numpy array of int.
                Peak indices on interpolated data after first peak detection step.
            - "troughs_1": 1D numpy array of int.
                Peak indices on interpolated data after first peak detection step.
            - "peaks_2": 1D numpy array of int.
                Peak indices on interpolated data after second peak detection step.
            - "troughs_2": 1D numpy array of int.
                Peak indices on interpolated data after second peak detection step.
            - "peaks_3": 1D numpy array of int.
                Peak indices on interpolated data after third peak detection step.
            - "troughs_3": 1D numpy array of int.
                Peak indices on interpolated data after third peak detection step.
            - "peaks_4": 1D numpy array of int.
                Peak indices on interpolated data after fourth peak detection step.
            - "troughs_4": 1D numpy array of int.
                Peak indices on interpolated data after fourth peak detection step.

        - 'all_cluster_steps':
            - 'rate': float.
                Sampling rate of interpolated data.
            - 'EOD_widths': list of three 1D numpy arrays.
                The first list entry gives the unique labels of all width clusters
                as a list of int.
                The second list entry gives the width values for each EOD in samples
                as a 1D numpy array of int.
                The third list entry gives the width labels for each EOD
                as a 1D numpy array of int.
            - 'EOD_heights': nested lists (2 layers) of three 1D numpy arrays.
                The first list entry gives the unique labels of all height clusters
                as a list of int for each width cluster.
                The second list entry gives the height values for each EOD
                as a 1D numpy array of floats for each width cluster.
                The third list entry gives the height labels for each EOD
                as a 1D numpy array of int for each width cluster.
            - 'EOD_shapes': nested lists (3 layers) of three 1D numpy arrays
                The first list entry gives the raw EOD snippets as a 2D numpy array
                for each height cluster in a width cluster.
                The second list entry gives the snippet PCA values for each EOD
                as a 2D numpy array of floats for each height cluster in a width cluster.
                The third list entry gives the shape labels for each EOD as a 1D numpy array
                of int for each height cluster in a width cluster.
            - 'discarding_masks': Nested lists (two layers) of 1D numpy arrays.
                The masks of EODs that are discarded by the discarding step of the algorithm.
                The masks are 1D boolean arrays where instances that are set to True are
                discarded by the algorithm. Discarding masks are saved in nested lists
                that represent the width and height clusters.
            - 'merge_masks': Nested lists (two layers) of 2D numpy arrays.
                The masks of EODs that are discarded by the merging step of the algorithm.
                The masks are 2D boolean arrays where for each sample point `i` either
                `merge_mask[i,0]` or `merge_mask[i,1]` is set to True. Here, merge_mask[:,0]
                represents the peak-centered clusters and `merge_mask[:,1]` represents the
                trough-centered clusters. Merge masks are saved in nested lists that
                represent the width and height clusters.

        - 'BGM_width':
            - 'BGM_width': dictionary
                - 'x': 1D numpy array of float.
                    BGM input values (in this case the EOD widths),
                - 'use_log': boolean.
                    True if the z-scored logarithm of the data was used as BGM input.
                - 'BGM': list of three 1D numpy arrays.
                    The first instance are the weights of the Gaussian fits.
                    The second instance are the means of the Gaussian fits.
                    The third instance are the variances of the Gaussian fits.
                - 'labels': 1D numpy array of int.
                    Labels defined by BGM model (before merging based on merge factor).
                - xlab': str.
                    Label for plot (defines the units of the BGM data).

        - 'BGM_height':
            This key adds a new dictionary for each width cluster.
            - 'BGM_height_*n*' : dictionary, where *n* defines the width cluster as an int.
                - 'x': 1D numpy array of float.
                    BGM input values (in this case the EOD heights),
                - 'use_log': boolean.
                    True if the z-scored logarithm of the data was used as BGM input.
                - 'BGM': list of three 1D numpy arrays.
                    The first instance are the weights of the Gaussian fits.
                    The second instance are the means of the Gaussian fits.
                    The third instance are the variances of the Gaussian fits.
                - 'labels': 1D numpy array of int.
                    Labels defined by BGM model (before merging based on merge factor).
                - 'xlab': str.
                    Label for plot (defines the units of the BGM data).

        - 'snippet_clusters':
            This key adds a new dictionary for each height cluster.
            - 'snippet_clusters*_n_m_p*' : dictionary, where *n* defines the width cluster
              (int), *m* defines the height cluster (int) and *p* defines shape clustering
              on peak or trough centered EOD snippets (str: 'peak' or 'trough').
                - 'raw_snippets': 2D numpy array (nsamples, nfeatures).
                    Raw EOD snippets.
                - 'snippets': 2D numpy array.
                    Normalized EOD snippets.
                - 'features': 2D numpy array.(nsamples, nfeatures)
                    PCA values for each normalized EOD snippet.
                - 'clusters': 1D numpy array of int.
                    Cluster labels.
                - 'rate': float.
                    Sampling rate of snippets.

        - 'eod_deletion':
            This key adds two dictionaries for each (peak centered) shape cluster,
            where *cluster* (int) is the unique shape cluster label.
            - 'mask_*cluster*' : list of four booleans.
                The mask for each cluster discarding step. 
                The first instance represents the artefact masks, where artefacts
                are set to True.
                The second instance represents the unreliable cluster masks,
                where unreliable clusters are set to True.
                The third instance represents the wavefish masks, where wavefish
                are set to True.
                The fourth instance represents the sidepeak masks, where sidepeaks
                are set to True.
            - 'vals_*cluster*' : list of lists.
                All variables that are used for each cluster deletion step.
                The first instance is a list of two 1D numpy arrays: the mean EOD and
                the FFT of that mean EOD.
                The second instance is a 1D numpy array with all EOD width to ISI ratios.
                The third instance is a list with three entries: 
                    The first entry is a 1D numpy array zoomed out version of the mean EOD.
                    The second entry is a list of two 1D numpy arrays that define the peak
                    and trough indices of the zoomed out mean EOD.
                    The third entry contains a list of two values that represent the
                    peak-trough pair in the zoomed out mean EOD with the largest height
                    difference.
            - 'rate' : float.
                EOD snippet sampling rate.

        - 'masks': 
            - 'masks' : 2D numpy array (4,N).
                Each row contains masks for each EOD detected by the EOD peakdetection step. 
                The first row defines the artefact masks, the second row defines the
                unreliable EOD masks, 
                the third row defines the wavefish masks and the fourth row defines
                the sidepeak masks.

        - 'moving_fish':
            - 'moving_fish': dictionary.
                - 'w' : list of float.
                    Median width for each width cluster that the moving fish algorithm is
                    computed on (in seconds).
                - 'T' : list of float.
                    Lenght of analyzed recording for each width cluster (in seconds).
                - 'dt' : list of float.
                    Sliding window size (in seconds) for each width cluster.
                - 'clusters' : list of 1D numpy int arrays.
                    Cluster labels for each EOD cluster in a width cluster.
                - 't' : list of 1D numpy float arrays.
                    EOD emission times for each EOD in a width cluster.
                - 'fishcount' : list of lists.
                    Sliding window timepoints and fishcounts for each width cluster.
                - 'ignore_steps' : list of 1D int arrays.
                    Mask for fishcounts that were ignored (ignored if True) in the
                    moving_fish analysis.
        
    Returns
    -------
    mean_eods: list of 2D array of float
        The average EOD for each detected fish. First column is time in seconds,
        second column the mean eod, third column the standard deviation.
    eod_times: list of 1D arrays
        For each detected fish the times of EOD peaks or troughs in seconds.
        Use these timepoints for EOD averaging.
    eod_peaktimes: list of 1D arrays
        For each detected fish the times of EOD peaks in seconds.
    zoom_window: tuple of float
        Start and endtime of suggested window for plotting EOD timepoints.
    log_dict: dictionary
        Dictionary with logged variables, where variables to log are specified
        by `return_data`.

    """
    if verbose > 0:
        print('')
        if verbose > 1:
            print(70*'#')
        print('##### extract_pulsefish', 46*'#')

    if save_plots and plot_level > 0:
        # create folder to save things in:
        save_path = Path(save_path)
        if not save_path.exists():
            save_path.mkdir(parents=True)
    else:
        save_path = ''

    mean_eods, eod_times, eod_peaktimes, zoom_window = [], [], [], []
    log_dict = {}

    # interpolate:
    i_rate = frate
    try:
        f = interp1d(np.arange(len(data))/rate, data, kind='quadratic')
        i_data = f(np.arange(0.0, (len(data)-1)/rate, 1.0/i_rate))
    except MemoryError:
        i_rate = rate
        i_data = data
    log_dict['data'] = i_data   # TODO: could be removed
    log_dict['rate'] = i_rate   # TODO: could be removed
    log_dict['i_data'] = i_data
    log_dict['i_rate'] = i_rate
                                         
    # standard deviation of data in small snippets:
    win_size = int(0.002*rate) # 2ms windows
    threshold = median_std_threshold(data, win_size)  # TODO make this a parameter
    
    # extract peaks:
    width_fac = max(width_factor_shape, width_factor_display, width_factor_wave)
    if 'peak_detection' in return_data:
        x_peak, x_trough, eod_heights, eod_widths, pd_log_dict = \
            detect_pulses(i_data, i_rate, threshold, width_fac=width_fac,
                          verbose=verbose, return_data=True)
        log_dict.update(pd_log_dict)
    else:
        x_peak, x_trough, eod_heights, eod_widths = \
            detect_pulses(i_data, i_rate, threshold, width_fac=width_fac,
                          verbose=verbose, return_data=False)
    
    if len(x_peak) > 0:
        # cluster:
        min_samples = 5  # TODO make parameter
        min_samples_frac = 0.05
        n_gaus_width = 3
        merge_thresh_width = 0.5
        n_gaus_height = 10
        merge_thresh_height = 0.1
        n_pca = 5
        shape_eps = 0.05
        clusters, x_merge, c_log_dict = \
            cluster(i_data, i_rate, x_peak, x_trough, eod_heights, eod_widths,
                    width_factor_shape, width_factor_wave,
                    min_samples=min_samples, min_samples_frac=min_samples_frac,
                    n_gaus_width=n_gaus_width,
                    merge_thresh_width=merge_thresh_width,
                    n_gaus_height=n_gaus_height,
                    merge_thresh_height=merge_thresh_height,
                    n_pca=n_pca, shape_eps=shape_eps,
                    verbose=verbose, plot_level=plot_level-1,
                    save_plots=save_plots, save_path=save_path,
                    ftype=ftype, return_data=return_data) 

        # extract mean eods and times:
        mean_eods, eod_times, eod_peaktimes, eod_troughtimes, cluster_labels = \
          extract_means(i_data, x_merge, x_peak, x_trough, eod_widths, clusters,
                        i_rate, width_factor_display, verbose=verbose)

        # determine clipped clusters (save them, but ignore in other steps):
        clusters, clipped_eods, clipped_times, clipped_peaktimes, clipped_troughtimes = \
          find_clipped_clusters(clusters, mean_eods, eod_times,
                                eod_peaktimes, eod_troughtimes,
                                cluster_labels, width_factor_display,
                                verbose=verbose)

        # delete moving fish:
        clusters, zoom_window, mf_log_dict = \
          delete_moving_fish(clusters, x_merge/i_rate, len(data)/rate,
                             eod_heights, eod_widths/i_rate, i_rate,
                             verbose=verbose, plot_level=plot_level-1,
                             save_plot=save_plots,
                             save_path=save_path, ftype=ftype,
                             return_data=return_data)
        
        if 'moving_fish' in return_data:
            log_dict['moving_fish'] = mf_log_dict

        clusters = remove_sparse_detections(clusters, eod_widths, i_rate,
                                            len(data)/rate, verbose=verbose)

        # extract mean eods
        mean_eods, eod_times, eod_peaktimes, eod_troughtimes, cluster_labels = \
          extract_means(i_data, x_merge, x_peak, x_trough, eod_widths,
                        clusters, i_rate, width_factor_display,
                        verbose=verbose)

        mean_eods.extend(clipped_eods)
        eod_times.extend(clipped_times)
        eod_peaktimes.extend(clipped_peaktimes)
        eod_troughtimes.extend(clipped_troughtimes)

        if plot_level > 0:
            plot_all(data, eod_peaktimes, eod_troughtimes, rate, mean_eods)
            if save_plots:
                plt.savefig('%sextract_pulsefish_results.%s' % (save_path, ftype))
        if save_plots:
            plt.close('all')
    
        if 'all_eod_times' in return_data:
            log_dict['all_times'] = [x_peak/i_rate, x_trough/i_rate]
            log_dict['eod_troughtimes'] = eod_troughtimes
        
        log_dict.update(c_log_dict)
        
    if verbose > 0:
        print('')

    return mean_eods, eod_times, eod_peaktimes, zoom_window, log_dict


def detect_pulses(data, rate, thresh, min_rel_slope_diff=0.25,
                  min_width=0.00005, max_width=0.01, width_fac=5.0,
                  verbose=0, return_data=False):
    """Detect pulses in data.

    Was `def extract_eod_times(data, rate, width_factor,
                      interp_freq=500000, max_peakwidth=0.01,
                      min_peakwidth=None, verbose=0, return_data=[],
                      save_path='')` before.

    Parameters
    ----------
    data: 1-D array of float
        The data to be analysed.
    rate: float
        Sampling rate of the data.
    thresh: float
        Threshold for peak and trough detection via `detect_peaks()`.
        Must be a positive number that sets the minimum difference
        between a peak and a trough.
    min_rel_slope_diff: float
        Minimum required difference between left and right slope (between
        peak and troughs) relative to mean slope for deciding which trough
        to take besed on slope difference.
    min_width: float
        Minimum width (peak-trough distance) of pulses in seconds.
    max_width: float
        Maximum width (peak-trough distance) of pulses in seconds.
    width_fac: float
        Pulses extend plus or minus `width_fac` times their width
        (distance between peak and assigned trough).
        Only pulses are returned that can fully be analysed with this width.
    verbose : int
        Verbosity level.
    return_data : bool
        If `True` data of this function is logged and returned (see
        extract_pulsefish()).

    Returns
    -------
    peak_indices: array of int
        Indices of EOD peaks in data.
    trough_indices: array of int
        Indices of EOD troughs in data. There is one x_trough for each x_peak.
    heights: array of float
        EOD heights for each x_peak.
    widths: array of int
        EOD widths for each x_peak (in samples).
    peak_detection_result : dictionary
        Key value pairs of logged data.
        This is only returned if `return_data` is `True`.

    """
    peak_detection_result = {}

    # detect peaks and troughs in the data:
    peak_indices, trough_indices = detect_peaks(data, thresh)
    if verbose > 0:
        print('Peaks/troughs detected in data:                      %5d %5d'
              % (len(peak_indices), len(trough_indices)))
    if return_data:
        peak_detection_result.update(peaks_1=np.array(peak_indices),
                                     troughs_1=np.array(trough_indices))
    if len(peak_indices) < 2 or \
       len(trough_indices) < 2 or \
       len(peak_indices) > len(data)/20:
        # TODO: if too many peaks increase threshold!
        if verbose > 0:
            print('No or too many peaks/troughs detected in data.')
        if return_data:
            return np.array([], dtype=int), np.array([], dtype=int), \
                np.array([]), np.array([], dtype=int), peak_detection_result
        else:
            return np.array([], dtype=int), np.array([], dtype=int), \
                np.array([]), np.array([], dtype=int)

    # assign troughs to peaks:
    peak_indices, trough_indices, heights, widths, slopes = \
      assign_side_peaks(data, peak_indices, trough_indices, min_rel_slope_diff)
    if verbose > 1:
        print('Number of peaks after assigning side-peaks:          %5d'
              % (len(peak_indices)))
    if return_data:
        peak_detection_result.update(peaks_2=np.array(peak_indices),
                                     troughs_2=np.array(trough_indices))

    # check widths:
    keep = ((widths>min_width*rate) & (widths<max_width*rate))
    peak_indices = peak_indices[keep]
    trough_indices = trough_indices[keep]
    heights = heights[keep]
    widths = widths[keep]
    slopes = slopes[keep]
    if verbose > 1:
        print('Number of peaks after checking pulse width:          %5d'
              % (len(peak_indices)))
    if return_data:
        peak_detection_result.update(peaks_3=np.array(peak_indices),
                                     troughs_3=np.array(trough_indices))

    # discard connected peaks:
    same = np.nonzero(trough_indices[:-1] == trough_indices[1:])[0]
    keep = np.ones(len(trough_indices), dtype=bool)
    for i in same:
        # same troughs at trough_indices[i] and trough_indices[i+1]:
        s = slopes[i:i+2]
        rel_slopes = np.abs(np.diff(s))[0]/np.mean(s)
        if rel_slopes > min_rel_slope_diff:
            keep[i+(s[1]<s[0])] = False
        else:
            keep[i+(heights[i+1]<heights[i])] = False
    peak_indices = peak_indices[keep]
    trough_indices = trough_indices[keep]
    heights = heights[keep]
    widths = widths[keep]
    if verbose > 1:
        print('Number of peaks after merging pulses:                %5d'
              % (len(peak_indices)))
    if return_data:
        peak_detection_result.update(peaks_4=np.array(peak_indices),
                                     troughs_4=np.array(trough_indices))
    if len(peak_indices) == 0:
        if verbose > 0:
            print('No peaks remain as pulse candidates.')
        if return_data:
            return np.array([], dtype=int), np.array([], dtype=int), \
                np.array([]), np.array([], dtype=int), peak_detection_result
        else:
            return np.array([], dtype=int), np.array([], dtype=int), \
                np.array([]), np.array([], dtype=int)
    
    # only take those where the maximum cutwidth does not cause issues -
    # if the width_fac times the width + x is more than length.
    keep = ((peak_indices - widths > 0) &
            (peak_indices + widths < len(data)) &
            (trough_indices - widths > 0) &
            (trough_indices + widths < len(data)))

    if verbose > 0:
        print('Remaining peaks after EOD extraction:                %5d'
              % (np.sum(keep)))
        print('')

    if return_data:
        return peak_indices[keep], trough_indices[keep], \
            heights[keep], widths[keep], peak_detection_result
    else:
        return peak_indices[keep], trough_indices[keep], \
            heights[keep], widths[keep]


@jit(nopython=True)
def assign_side_peaks(data, peak_indices, trough_indices,
                      min_rel_slope_diff=0.25):
    """Assign to each peak the trough resulting in a pulse with the steepest slope or largest height.

    The slope between a peak and a trough is computed as the height
    difference divided by the distance between peak and trough. If the
    slopes between the left and the right trough differ by less than
    `min_rel_slope_diff`, then just the heigths between and the two
    troughs relative to the peak are compared.

    Was `def detect_eod_peaks(data, main_indices, side_indices,
                              max_width=20, min_width=2, verbose=0)` before.

    Parameters
    ----------
    data: array of float
        Data in which the events were detected.
    peak_indices: array of int
        Indices of the detected peaks in the data time series.
    trough_indices: array of int
        Indices of the detected troughs in the data time series. 
    min_rel_slope_diff: float
        Minimum required difference of left and right slope relative
        to mean slope.

    Returns
    -------
    peak_indices: array of int
        Peak indices. Same as input `peak_indices` but potentially shorter
        by one or two elements.
    trough_indices: array of int
        Corresponding trough indices of trough to the left or right
        of the peaks.
    heights: array of float
        Peak heights (distance between peak and corresponding trough amplitude)
    widths: array of int
        Peak widths (distance between peak and corresponding trough indices)
    slopes: array of float
        Peak slope (height divided by width)
    """
    # is a main or side peak first?
    peak_first = int(peak_indices[0] < trough_indices[0])
    # is a main or side peak last?
    peak_last = int(peak_indices[-1] > trough_indices[-1])
    # ensure all peaks to have side peaks (troughs) at both sides,
    # i.e. troughs at same index and next index are before and after peak:
    peak_indices = peak_indices[peak_first:len(peak_indices)-peak_last]
    y = data[peak_indices]
    
    # indices of troughs on the left and right side of main peaks:
    l_indices = np.arange(len(peak_indices))
    r_indices = l_indices + 1

    # indices, distance to peak, height, and slope of left troughs:
    l_side_indices = trough_indices[l_indices]
    l_distance = np.abs(peak_indices - l_side_indices)
    l_height = np.abs(y - data[l_side_indices])
    l_slope = np.abs(l_height/l_distance)

    # indices, distance to peak, height, and slope of right troughs:
    r_side_indices = trough_indices[r_indices]
    r_distance = np.abs(r_side_indices - peak_indices)
    r_height = np.abs(y - data[r_side_indices])
    r_slope = np.abs(r_height/r_distance)

    # which trough to assign to the peak?
    # - either the one with the steepest slope, or
    # - when slopes are similar on both sides
    #   (within `min_rel_slope_diff` difference),
    #   the trough with the maximum height difference to the peak:
    rel_slopes = np.abs(l_slope-r_slope)/(0.5*(l_slope + r_slope))
    take_slopes = rel_slopes > min_rel_slope_diff
    take_left = l_height > r_height
    take_left[take_slopes] = l_slope[take_slopes] > r_slope[take_slopes]

    # assign troughs, heights, widths, and slopes:
    trough_indices = np.where(take_left,
                              trough_indices[:-1], trough_indices[1:])
    heights = np.where(take_left, l_height, r_height)
    widths = np.where(take_left, l_distance, r_distance)
    slopes = np.where(take_left, l_slope, r_slope)

    return peak_indices, trough_indices, heights, widths, slopes


def cluster(data, rate, eod_xp, eod_xt, eod_heights, eod_widths,
            width_factor_shape, width_factor_wave,
            min_samples=5, min_samples_frac=0.05,
            n_gaus_width=3, merge_thresh_width=0.6,
            n_gaus_height=10, merge_thresh_height=0.1,
            n_pca=5, shape_eps=0.05,
            verbose=0, plot_level=0, save_plots=False,
            save_path='', ftype='pdf', return_data=[]):
    """Cluster EODs.
    
    First cluster on EOD widths using a Bayesian Gaussian
    Mixture (BGM) model,  then cluster on EOD heights using a
    BGM model. Lastly, cluster on EOD waveform with DBSCAN.
    Clustering on EOD waveform is performed twice, once on
    peak-centered EODs and once on trough-centered EODs.
    Non-pulsetype EOD clusters are deleted, and clusters are
    merged afterwards.

    Parameters
    ----------
    data: array of float
        Data in which to detect pulse EODs.
    rate : float
        Sampling rate of `data`.
    eod_xp : list of int
        Location of EOD peaks in indices.
    eod_xt: list of int
        Locations of EOD troughs in indices.
    eod_heights: list of float
        EOD heights.
    eod_widths: list of int
        EOD widths in samples.
    width_factor_shape : float
        Multiplier for snippet extraction width. This factor is
        multiplied with the width between the peak and through of a
        single EOD.
    width_factor_wave : float
        Multiplier for wavefish extraction width.
    min_samples: int
        Minimum number of samples required for a valid cluster.
    min_samples_frac: float
        Cluster with less samples than this fraction of all the samples are removed.
    n_gaus_width : int
        Number of gaussians to use for the clustering based on EOD width.
    merge_thresh_width : float
        Threshold for merging clusters that are similar in width.
    n_gaus_height : int
        Number of gaussians to use for the clustering based on EOD height.
    merge_thresh_height : float
        Threshold for merging clusters that are similar in height.
    n_pca: int
        Number of PCs to use for PCA.
    shape_eps : float
        Epsilon to use for DBSCAN clustering of EOD shapes.
    verbose : int
        Verbosity level.
    plot_level : int
        Similar to verbosity levels, but with plots. 
        Only set to > 0 for debugging purposes.
    save_plots : bool
        Set to True to save created plots.
    save_path : str
        Path to save plots to. Only used if save_plots==True.
    ftype : str
        Filetype to save plot images in.
    return_data : list of str
        Keys that specify data to be logged. Keys that can be used to log data
        in this function are: 'all_cluster_steps', 'BGM_width', 'BGM_height',
        'snippet_clusters', 'eod_deletion' (see extract_pulsefish()).

    Returns
    -------
    labels : list of int
        EOD cluster labels based on height and EOD waveform.
    x_merge : list of int
        Locations of EODs in clusters.
    saved_data : dictionary
        Key value pairs of logged data. Data to be logged is specified
        by return_data.

    """
    saved_data = {}

    if plot_level > 0 or 'all_cluster_steps' in return_data:
        all_heightlabels = []
        all_shapelabels = []
        all_snippets = []
        all_features = []
        all_heights = []
        all_unique_heightlabels = []

    all_p_labels = -1*np.ones(len(eod_xp), dtype=int)
    all_t_labels = -1*np.ones(len(eod_xp), dtype=int)
    artefact_masks_p = np.ones(len(eod_xp), dtype=bool)
    artefact_masks_t = np.ones(len(eod_xp), dtype=bool)

    x_merge = -1*np.ones(len(eod_xp), dtype=int)

    # keep track of the labels so that no labels are overwritten:
    max_label_p = 0
    max_label_t = 0
    
    if verbose > 0:
        print('clusters generated based on EOD width:')

    # first cluster on width:
    width_labels, bgm_log_dict = BGM(1000*eod_widths/rate,
                                     min_samples=min_samples,
                                     min_samples_frac=min_samples_frac,
                                     merge_thresh=merge_thresh_width,
                                     n_gaus=n_gaus_width, use_log=False,
                                     xlabel='width [ms]',
                                     verbose=verbose - 1,
                                     plot_level=plot_level - 1,
                                     save_plot=save_plots,
                                     save_path=save_path,
                                     save_name='width', ftype=ftype,
                                     return_data='BGM_width' in return_data)
    if len(bgm_log_dict) > 0:
        saved_data['BGM_width'] = bgm_log_dict

    if verbose > 0:
        # report width clusters:
        for l in np.unique(width_labels):
            print(f'  {l:2d}: num={len(width_labels[width_labels == l]):5d}, width={np.mean(1000*eod_widths[width_labels == l]/rate):6.3f} +- {np.std(1000*eod_widths[width_labels == l]/rate):6.3f}ms')

    if plot_level > 1:
        plt.show()

    # loop over width clusters:
    unique_width_labels = np.unique(width_labels[width_labels != -1])
    for wi, width_label in enumerate(unique_width_labels):
        # select only features in one width cluster at a time:
        w_eod_widths = eod_widths[width_labels == width_label]
        w_eod_heights = eod_heights[width_labels == width_label]
        w_eod_xp = eod_xp[width_labels == width_label]
        w_eod_xt = eod_xt[width_labels == width_label]
        width = int(width_factor_shape*np.median(w_eod_widths))
        if width > w_eod_xp[0]:
            width = w_eod_xp[0]
        if width > w_eod_xt[0]:
            width = w_eod_xt[0]
        if width > len(data) - w_eod_xp[-1]:
            width = len(data) - w_eod_xp[-1]
        if width > len(data) - w_eod_xt[-1]:
            width = len(data) - w_eod_xt[-1]
        
        wp_labels = -1*np.ones(len(w_eod_xp), dtype=int)
        wt_labels = -1*np.ones(len(w_eod_xp), dtype=int)
        wartefact_mask = np.ones(len(w_eod_xp), dtype=int)

        # determine height labels:
        raw_p_snippets, p_snippets, p_features, p_bg_ratio = \
          extract_snippet_features(data, w_eod_xp, w_eod_widths,
                                   w_eod_heights, width, n_pca)
        raw_t_snippets, t_snippets, t_features, t_bg_ratio = \
          extract_snippet_features(data, w_eod_xt, w_eod_widths,
                                   w_eod_heights, width, n_pca)
            
        # TODO: rather keep the height threshold independent of slopes!
        #median_bg_ratios = np.median(np.min(np.vstack([p_bg_ratio, t_bg_ratio]),
        #                                  axis=0))
        #merge_thresh_height = min(merge_thresh_height, median_bg_ratios)

        if verbose > 0:
            print(f'  clusters generated based on EOD height in width cluster {width_label}:')
            
        height_labels, bgm_log_dict = \
          BGM(w_eod_heights, min_samples=min_samples, min_samples_frac=min_samples_frac,
              merge_thresh=merge_thresh_height, n_gaus=n_gaus_height,
              use_log=True, xlabel='height [a.u.]', 
              verbose=verbose - 1, plot_level=plot_level - 1,
              save_plot=save_plots, save_path=save_path,
              save_name = f'height_{wi}',
              ftype=ftype, return_data='BGM_height' in return_data)
        if len(bgm_log_dict) > 0:
            saved_data[f'BGM_height_{wi}'] = bgm_log_dict

        if verbose > 0:
            # report height clusters:
            for l in np.unique(height_labels):
                print(f'    {l:2d}: num={len(height_labels[height_labels == l]):5d}, height={np.mean(w_eod_heights[height_labels == l]):6.4g} +- {np.std(w_eod_heights[height_labels == l]):6.4g}')

        if plot_level > 0:
            prop_cycle = plt.rcParams['axes.prop_cycle']
            colors = prop_cycle.by_key()['color']
            if np.min(height_labels) == -1:
                colors = ['black'] + colors
            fig, axs = plt.subplots(2, 2, layout='constrained', sharex=True, sharey='col')
            axs[0, 0].set_title(f'Width cluster {width_label}: raw peak snippets')
            for l, c in zip(np.unique(height_labels), colors):
                ll = [f'{l}'] + [None]*(np.sum(height_labels == l) - 1)
                axs[0, 0].plot(raw_p_snippets[height_labels == l, :].T, label=ll, color=c, lw=0.2)
            axs[0, 1].set_title('Peak snippets')
            for l, c in zip(np.unique(height_labels), colors):
                ll = [f'{l}'] + [None]*(np.sum(height_labels == l) - 1)
                axs[0, 1].plot(p_snippets[height_labels == l, :].T, label=ll, color=c, lw=0.2)
            axs[0, 1].legend(title='height label')
            axs[1, 0].set_title(f'Width cluster {width_label}: raw trough snippets')
            for l, c in zip(np.unique(height_labels), colors):
                ll = [f'{l}'] + [None]*(np.sum(height_labels == l) - 1)
                axs[1, 0].plot(raw_t_snippets[height_labels == l, :].T, label=ll, color=c, lw=0.2)
            axs[1, 1].set_title('Trough snippets')
            for l, c in zip(np.unique(height_labels), colors):
                ll = [f'{l}'] + [None]*(np.sum(height_labels == l) - 1)
                axs[1, 1].plot(t_snippets[height_labels == l, :].T, label=ll, color=c, lw=0.2)
            axs[1, 1].legend(title='height label')
            axs[1, 0].set_xlabel('index')
            axs[1, 1].set_xlabel('index')
            plt.show()

        unique_height_labels = np.unique(height_labels)
        unique_height_labels = unique_height_labels[unique_height_labels != -1]

        if plot_level > 0 or 'all_cluster_steps' in return_data:
            all_heightlabels.append(height_labels)
            all_heights.append(w_eod_heights)
            all_unique_heightlabels.append(unique_height_labels)
            shape_labels = []
            cfeatures = []
            csnippets = []

        for hi, height_label in enumerate(unique_height_labels):

            h_eod_widths = w_eod_widths[height_labels == height_label]
            h_eod_heights = w_eod_heights[height_labels == height_label]
            h_eod_xp = w_eod_xp[height_labels == height_label]
            h_eod_xt = w_eod_xt[height_labels == height_label]
            
            if verbose > 0:
                print(f'    clusters generated based on EOD shape in width cluster {width_label}, height cluster {height_label}:')

            p_feats = p_features[height_labels == height_label]
            t_feats = t_features[height_labels == height_label]
            
            p_labels = cluster_on_shape(p_feats, shape_eps, min_samples,
                                        min_samples_frac)
            t_labels = cluster_on_shape(t_feats, shape_eps, min_samples,
                                        min_samples_frac)

            if plot_level > 0:
                p_snips = p_snippets[height_labels == height_label]
                t_snips = t_snippets[height_labels == height_label]
                prop_cycle = plt.rcParams['axes.prop_cycle']
                color_cycle = prop_cycle.by_key()['color']
                fig, axs = plt.subplots(2, 2, layout='constrained', sharex='col', sharey='col')
                axs[0, 0].set_title(f'Width cluster {width_label}, height cluster {height_label}: peak snippets')
                if np.min(p_labels) == -1:
                    colors = ['black'] + color_cycle
                else:
                    colors = color_cycle
                for l, c in zip(np.unique(p_labels), colors):
                    ll = [f'{l}'] + [None]*(np.sum(p_labels == l) - 1)
                    axs[0, 0].plot(p_feats[p_labels == l, 0], p_feats[p_labels == l, 1], 'o', label=ll, color=c)
                    axs[0, 1].plot(p_snips[p_labels == l, :].T, label=ll, color=c, lw=0.2)
                axs[0, 1].legend(title='shape label')
                if np.min(t_labels) == -1:
                    colors = ['black'] + color_cycle
                else:
                    colors = color_cycle
                axs[1, 0].set_title(f'Width cluster {width_label}, height cluster {height_label}: trough snippets')
                for l, c in zip(np.unique(t_labels), colors):
                    ll = [f'{l}'] + [None]*(np.sum(t_labels == l) - 1)
                    axs[1, 0].plot(t_feats[t_labels == l, 0], t_feats[t_labels == l, 1], 'o', label=ll, color=c)
                    axs[1, 1].plot(t_snips[t_labels == l, :].T, label=ll, color=c, lw=0.2)
                axs[1, 1].legend(title='shape label')
                plt.show()
            
            if False: #plot_level > 1:
                plot_feature_extraction(raw_p_snippets[height_labels == height_label],
                                        p_snippets[height_labels == height_label],
                                        p_features[height_labels == height_label],
                                        p_labels, 1/rate, 0)
                plt.savefig('%sDBSCAN_peak_w%i_h%i.%s' % (save_path, wi, hi, ftype))
                plot_feature_extraction(raw_t_snippets[height_labels == height_label],
                                        t_snippets[height_labels == height_label],
                                        t_features[height_labels == height_label],
                                        t_labels, 1/rate, 1)
                plt.savefig('%sDBSCAN_trough_w%i_h%i.%s' % (save_path, wi, hi, ftype))

            if 'snippet_clusters' in return_data:
                saved_data[f'snippet_clusters_{width_label}_{height_label}_peak'] = {
                    'raw_snippets': raw_p_snippets[height_labels == height_label],
                    'snippets': p_snippets[height_labels == height_label],
                    'features': p_features[height_labels == height_label],
                    'clusters': p_labels,
                    'rate': rate}
                saved_data['snippet_clusters_{width_label}_{height_label}_trough'] = {
                    'raw_snippets': raw_t_snippets[height_labels == height_label],
                    'snippets': t_snippets[height_labels == height_label],
                    'features': t_features[height_labels == height_label],
                    'clusters': t_labels,
                    'rate': rate}

            if plot_level > 0 or 'all_cluster_steps' in return_data:
                shape_labels.append([p_labels, t_labels])
                cfeatures.append([p_features[height_labels == height_label],
                                  t_features[height_labels == height_label]])
                csnippets.append([p_snippets[height_labels == height_label],
                                  t_snippets[height_labels == height_label]])

            p_labels[p_labels == -1] = -max_label_p - 1
            wp_labels[height_labels == height_label] = p_labels + max_label_p
            max_label_p = max(np.max(wp_labels), np.max(all_p_labels)) + 1

            t_labels[t_labels == -1] = -max_label_t - 1
            wt_labels[height_labels == height_label] = t_labels + max_label_t
            max_label_t = max(np.max(wt_labels), np.max(all_t_labels)) + 1

        if verbose > 0:
            if np.max(wp_labels) == -1:
                print(f'      none')
            else:
                unique_clusters = np.unique(wp_labels[wp_labels != -1])
                print(f'      num={len(unique_clusters):2d} different EOD shapes:',
                      str(unique_clusters).strip('[]'))
        
        if plot_level > 0 or 'all_cluster_steps' in return_data:
            all_shapelabels.append(shape_labels)
            all_snippets.append(csnippets)
            all_features.append(cfeatures)

        # for each cluster, save fft + label
        # so I end up with features for each label, and the masks.
        # then I can extract e.g. first artefact or wave etc.

        # remove artefacts here, based on the mean snippets ffts.
        artefact_masks_p[width_labels == width_label], sdict = \
          remove_artefacts(p_snippets, wp_labels, rate,
                           verbose=verbose-1, return_data=return_data)
        saved_data.update(sdict)
        artefact_masks_t[width_labels == width_label], _ = \
          remove_artefacts(t_snippets, wt_labels, rate,
                           verbose=verbose-1, return_data=return_data)

        # update maxlab so that no clusters are overwritten
        all_p_labels[width_labels == width_label] = wp_labels
        all_t_labels[width_labels == width_label] = wt_labels

    if verbose > 1:
        print()
    
    # remove all non-reliable clusters
    unreliable_fish_mask_p, saved_data = \
      delete_unreliable_fish(all_p_labels, eod_widths, eod_xp,
                             verbose=verbose-1, sdict=saved_data)
    unreliable_fish_mask_t, _ = \
      delete_unreliable_fish(all_t_labels, eod_widths, eod_xt, verbose=verbose-1)
    
    wave_mask_p, sidepeak_mask_p, saved_data = \
      delete_wavefish_and_sidepeaks(data, all_p_labels, eod_xp, eod_widths,
                                    width_factor_wave, verbose=verbose-1, sdict=saved_data)
    wave_mask_t, sidepeak_mask_t, _ = \
      delete_wavefish_and_sidepeaks(data, all_t_labels, eod_xt, eod_widths,
                                    width_factor_wave, verbose=verbose-1)  
        
    og_clusters = [np.copy(all_p_labels), np.copy(all_t_labels)]
    og_labels = np.copy(all_p_labels + all_t_labels)

    # go through all clusters and masks??
    all_p_labels[(artefact_masks_p | unreliable_fish_mask_p | wave_mask_p | sidepeak_mask_p)] = -1
    all_t_labels[(artefact_masks_t | unreliable_fish_mask_t | wave_mask_t | sidepeak_mask_t)] = -1

    # merge here:
    all_clusters, x_merge, mask = merge_clusters(np.copy(all_p_labels),
                                                 np.copy(all_t_labels),
                                                 eod_xp, eod_xt,
                                                 verbose=verbose - 1)
    
    if 'all_cluster_steps' in return_data or plot_level > 0:
        all_dmasks = []
        all_mmasks = []

        discarding_masks = \
          np.vstack(((artefact_masks_p | unreliable_fish_mask_p | wave_mask_p | sidepeak_mask_p),
                     (artefact_masks_t | unreliable_fish_mask_t | wave_mask_t | sidepeak_mask_t)))
        merge_mask = mask

        # save the masks in the same formats as the snippets
        for wi, (width_label, w_shape_label, heightlabels, unique_height_labels) in enumerate(zip(unique_width_labels, all_shapelabels, all_heightlabels, all_unique_heightlabels)):
            w_dmasks = discarding_masks[:,width_labels == width_label]
            w_mmasks = merge_mask[:,width_labels == width_label]

            wd_2 = []
            wm_2 = []

            for hi, (height_label, h_shape_label) in enumerate(zip(unique_height_labels, w_shape_label)):
               
                h_dmasks = w_dmasks[:,heightlabels == height_label]
                h_mmasks = w_mmasks[:,heightlabels == height_label]

                wd_2.append(h_dmasks)
                wm_2.append(h_mmasks)

            all_dmasks.append(wd_2)
            all_mmasks.append(wm_2)

        if plot_level > 0:
            plot_clustering(rate, [unique_width_labels, eod_widths, width_labels],
                            [all_unique_heightlabels, all_heights, all_heightlabels],
                            [all_snippets, all_features, all_shapelabels],
                            all_dmasks, all_mmasks)
            if save_plots:
                plt.savefig('%sclustering.%s' % (save_path, ftype))

        if 'all_cluster_steps' in return_data:
            saved_data = {'rate': rate,
                      'EOD_widths': [unique_width_labels, eod_widths, width_labels],
                      'EOD_heights': [all_unique_heightlabels, all_heights, all_heightlabels],
                      'EOD_shapes': [all_snippets, all_features, all_shapelabels],
                      'discarding_masks': all_dmasks,
                      'merge_masks': all_mmasks
                    }
    
    if 'masks' in return_data:
        saved_data = {'masks' : np.vstack(((artefact_masks_p & artefact_masks_t),
                                           (unreliable_fish_mask_p & unreliable_fish_mask_t),
                                           (wave_mask_p & wave_mask_t),
                                           (sidepeak_mask_p & sidepeak_mask_t),
                                           (all_p_labels+all_t_labels)))}

    if verbose > 0:
        print('clusters generated based on height, width and shape: ')
        for l in np.unique(all_clusters[all_clusters != -1]):
            print(f'  {l:2d}: num={len(all_clusters[all_clusters == l]):5d}')
             
    return all_clusters, x_merge, saved_data


def BGM(x, min_samples=5, min_samples_frac=0.05,
        merge_thresh=0.1, n_gaus=5, max_iter=200, n_init=5,
        use_log=False, xlabel='x [a.u.]', verbose=0, plot_level=0,
        save_plot=False, save_path='', save_name='', ftype='pdf',
        return_data=[]):
    """Use a Bayesian Gaussian Mixture Model to cluster one-dimensional data.

    The data are clustered on their z-scores or on the z-scores of the
    log-transformed data if `use_log`is true.
    Broad gaussian fits that cover one or more other gaussian fits are
    split by their intersections with the other gaussians.
    Clusters that are closer than `merge_thresh` on the original scale
    of the data (not the z-scores) are merged.
    
    Parameters
    ----------
    x: 1D numpy array
        Features to compute clustering on. 
    min_samples: int
        Minimum number of samples required for a valid cluster.
    min_samples_frac: float
        Cluster with less samples than this fraction of all the samples are removed.
    merge_thresh: float
        Ratio for merging nearby gaussians.
    n_gaus: int
        Maximum number of gaussians to fit on data.
    max_iter: int
        Maximum number of iterations for gaussian fit.
    n_init: int
        Number of initializations for the gaussian fit.
    use_log: boolean
        Set to True to compute the gaussian fit on the logarithm of x.
        Can improve clustering on features with nonlinear relationships such as peak height.
    xlabel: str
        Xlabel for displaying BGM plot.
    verbose: int
        Verbosity level.
    plot_level: int
        Similar to verbosity levels, but with plots. 
        Only set to > 0 for debugging purposes.
    save_plot: bool
        Set to True to save created plot.
    save_path: str
        Path to location where data should be saved. Only used if save_plot == True.
    save_name: str
        Filename of the saved plot. Usefull as usually multiple BGM models are generated.
    ftype: str
        Filetype of plot image if save_plots == True.
    return_data: bool
        True if additional data shouldbe returned in bgm_dict.

    Returns
    -------
    labels : 1D numpy array
        Cluster labels for each sample in x.
    bgm_dict : dictionary
        Key value pairs of logged data if `return_data` is True.

    """

    bgm_dict = {}

    if len(np.unique(x)) <= n_gaus:
        return np.zeros(len(x), dtype=int), bgm_dict
    
    if use_log:
        z = stats.zscore(np.log(x)).reshape(-1, 1)
    else:
        z = stats.zscore(x).reshape(-1, 1)
    BGM_model = BayesianGaussianMixture(n_components=n_gaus,
                                        max_iter=max_iter,
                                        n_init=n_init)
    labels = BGM_model.fit_predict(z)
    
    if not BGM_model.converged_ and verbose > 0:
        print('    !!! Bayesian Gaussian mixture did not converge !!!')

    labels_bgm = np.copy(labels)

    # separate gaussian clusters that can be split by other clusters:
    sidx_x = np.argsort(x)
    splits = x[sidx_x][1:][np.diff(labels[sidx_x]) != 0]
    labels[:] = 0
    for i, split in enumerate(splits):
        labels[x >= split] = i + 1

    labels_split = np.copy(labels)

    # merge gaussian clusters that are closer than merge_thresh:
    labels = merge_gaussians(x, labels, min_samples, min_samples_frac,
                             merge_thresh, verbose=verbose - 1)

    # sort model attributes by model.means_:
    sidx = np.argsort(BGM_model.means_[:, 0])
    means = BGM_model.means_[sidx, 0]
    variances = BGM_model.covariances_[sidx, 0, 0]
    weights = BGM_model.weights_[sidx]

    if plot_level > 0:
        all_labels = [labels_bgm, labels_split, labels]
        all_titles = ['BGM', 'split','merge']
        if False: #use_log:
            bins = np.geomspace(np.min(x), np.max(x), 100)
            xx = np.geomspace(np.min(x), np.max(x), 500)
        else:
            bins = np.linspace(np.min(x), np.max(x), 100)
            xx = np.linspace(np.min(x), np.max(x), 500)
        fig, axs = plt.subplots(3, 1, layout='constrained')
        for k in range(len(all_labels)):
            ax = axs[k]
            labs = all_labels[k]
            ax.set_title(all_titles[k])
            for l in np.unique(labs[labs != -1]):
                xl = x[labs == l]
                ax.hist(xl, bins, label=f'{l}')
            if k == len(all_labels) - 1:
                ax.set_xlabel(xlabel)
            ax.set_ylabel('counts')
            ax.set_ylim(bottom=0.3)
            if False: #use_log:
                ax.set_xscale('log')
            ax.set_yscale('log')
            ax.legend(title='labels')
        ax = axs[0].twinx()
        if use_log:
            means_logx = means*np.std(np.log(x)) + np.mean(np.log(x))
            stds_logx = np.sqrt(variances)*np.std(np.log(x))
            for m, s, w in zip(means_logx, stds_logx, weights):
                gg = np.exp(-0.5*((np.log(xx) - m)/s)**2)
                ax.plot(xx, w*gg, 'k')
        else:
            means_x = means*np.std(x) + np.mean(x)
            stds_x = np.sqrt(variances)*np.std(x)
            for m, s, w in zip(means_x, stds_x, weights):
                ax.plot(xx, w*np.exp(-0.5*((xx - m)/s)**2), 'k')
        ax.set_ylim(bottom=0)
        """
        plot_bgm(x, means, variances, weights, use_log, labels_split,
                 labels, xlabel)
        if save_plot:
            plt.savefig('%sBGM_%s.%s' % (save_path, save_name, ftype))
        """
        
    if return_data:
        bgm_dict = dict(x=x,
                        use_log=use_log,
                        BGM=[weights, means, variances],
                        labels=labels_split,
                        xlab=xlabel)

    return labels, bgm_dict


def merge_gaussians(x, labels, min_samples=5, min_samples_frac=0.05,
                    merge_thresh=0.1, verbose=0):
    """ Merge all clusters which have medians which are near.

    Only works in 1D.

    First, cluster with less than `min_samples` samples are removed -
    they get their label set to -1.
    Then, clusters where the difference between their medians relative to
    the larger of the two medians are smaller than `merge_thresh` are
    merged - all their members get the smaller of the two labels assignd. 

    Parameters
    ----------
    x : 1D array of int or float
        Features used for clustering.
    labels : 1D array of int
        Labels for each sample in `x`.
    min_samples: int
        Minimum number of samples required for a valid cluster.
    min_samples_frac: float
        Cluster with less samples than this fraction of all the samples are removed.
    merge_thresh : float
        Similarity threshold to merge clusters. The difference between
        median values relative to the larger median needs to be smaler
        than this threshold for merging.
    verbose: int
        Verbosity level.

    Returns
    -------
    labels : 1D array of int
        Merged labels for each sample in x.
    """
    # remove small clusters:
    u_labels, u_counts = unique_counts(labels[labels != -1])
    for l, c in zip(u_labels, u_counts):
        if c < min_samples:
            labels[labels == l] = -1
            if verbose > 0:
                print(f'      removed cluster {l:2d}: number of samples {c:2d} smaller than {min_samples:2d}')
        elif c/len(labels) < min_samples_frac:
            labels[labels == l] = -1
            if verbose > 0:
                print(f'      removed cluster {l:2d}: fraction of samples {100*c/len(labels):4.1f}% smaller than {100*min_samples_frac:4.1f}%')
    u_labels = u_labels[u_counts >= min_samples]
    if len(u_labels) == 0:
        return labels
    while len(u_labels) >= 2:
        # medians for each label:
        x_medians = np.array([np.median(x[labels == l]) for l in u_labels])
        sidx = np.argsort(x_medians)
        # fill a dict with label mappings:
        mapping = {}
        for k in range(len(sidx[:-1])):
            i0 = sidx[k]
            i1 = sidx[k + 1]
            label_1 = u_labels[i0]
            label_2 = u_labels[i1]
            median_1 = x_medians[i0]
            median_2 = x_medians[i1]
            rel_diff = np.abs(median_2 - median_1)/max(median_1, median_2)
            if rel_diff < merge_thresh:
                mapping[label_2] = label_1
                if verbose > 0:
                    print(f'      merge cluster {label_1:2d} (median={median_1:6.4g}) with cluster {label_2:2d} (median={median_2:6.4g}) because relative difference {rel_diff:4.2f} is smaller than {merge_thresh:4.2f}')
            elif verbose > 0:
                print(f'      keep  cluster {label_1:2d} (median={median_1:6.4g}) and  cluster {label_2:2d} (median={median_2:6.4g}) because relative difference {rel_diff:4.2f} is larger than {merge_thresh:4.2f}')
        # apply mapping:
        if len(mapping) > 0:
            for key in mapping:
                labels[labels == key] = mapping[key]
            u_labels = np.unique(labels[labels != -1])
        else:
            break
    return labels


def extract_snippet_features(data, eod_idx, eod_widths, eod_heights,
                             width, n_pca=5):
    """Extract, align, normalize, snippets from recording data, normalize them, and perform PCA.

    Parameters
    ----------
    data: 1D numpy array of float
        Recording data.
    eod_idx: 1D array of int
        Locations of EODs as indices.
    eod_widths: 1D array of int
        EOD widths (distance between peak and trough) in samples.
    eod_heights: 1D array of float
        EOD heights.
    width: int
        Width to cut out to each side in samples.
    n_pca: int
        Number of PCs to use for PCA.

    Returns
    -------
    raw_snippets : 2D numpy array (N, EOD_width)
        Raw extracted EOD snippets.
    snippets : 2D numpy array (N, EOD_width)
        Normalized EOD snippets
    features : 2D numpy array (N, n_pca)
        PC values of EOD snippets
    bg_ratio : 1D numpy array (N)
        Ratio of the background activity slopes compared to EOD height.

    """
    # extract snippets with corresponding width:
    xwidth = width + min(10, width//4)
    raw_snippets = np.zeros((len(eod_idx), 2*xwidth))
    w = min(eod_idx[0], xwidth)
    raw_snippets[0, xwidth - w:] = data[eod_idx[0] - w:eod_idx[0] + xwidth]
    for k, idx in enumerate(eod_idx[1:-1]):
        raw_snippets[k + 1, :] = data[idx - xwidth:idx + xwidth]
    if len(eod_idx) > 1:
        w = min(len(data) - eod_idx[-1], xwidth)
        raw_snippets[-1, :xwidth + w] = data[eod_idx[-1] - xwidth:eod_idx[-1] + w]

    # align snippets on phase of first Fourier coefficient:
    # (aligning on maximum of higher harmonics is much less robust)
    n = raw_snippets.shape[1]
    dist = int(np.median(eod_widths))
    freq = 1/(2*dist)
    dist = int(1.2*dist)
    coefs = np.zeros(len(raw_snippets),dtype=complex)
    for k in range(len(raw_snippets)):
        snippet = raw_snippets[k, n//2 - dist:n//2 + dist]
        m = len(snippet)
        coef = fourier_coeffs(snippet, np.arange(m) - m//2, freq, 1)[1]
        coefs[k] = coef/np.abs(coef)
    coefs *= np.conjugate(np.mean(coefs))
    ishifts = np.zeros(len(raw_snippets), dtype=int)
    for k in range(len(raw_snippets)):
        tshift = np.angle(coefs[k])/(2*np.pi*freq)
        ishift = int(np.round(tshift))
        ishifts[k] = ishift
        raw_snippets[k] = np.roll(raw_snippets[k], ishift)
    raw_snippets = raw_snippets[:, n//2 - width:n//2 + width]

    # subtract background slope:
    snippets, bg_ratio = subtract_slope(np.copy(raw_snippets), eod_heights)

    # scale so that their euclidian norm equals one:
    snippets = (snippets.T/np.linalg.norm(snippets, axis=1)).T

    # compute PCA features for clustering on waveform:
    pca = PCA(n_pca)
    features = pca.fit_transform(snippets)

    return raw_snippets, snippets, features, bg_ratio


def cluster_on_shape(features, epsilon=0.05,
                     min_samples=5, min_samples_frac=0.05):
    """Separate EODs by their shape using DBSCAN.

    Parameters
    ----------
    features : 2D numpy array of float (N, n_pc)
        PCA features of each EOD in a recording.
    epsilon : float
        Epsilon to use for DBSCAN clustering.
    min_samples: int
        Minimum number of samples required for a valid cluster.
    min_samples_frac: float
        Cluster with less samples than this fraction of all the samples are removed.

    Returns
    -------
    labels : 1D array of int
        Merged labels for each sample in x.
    """
    # minimum samples for core points:
    min_smpl = int(len(features)*min_samples_frac)
    if min_smpl > min_samples:
        min_samples = min_smpl

    return DBSCAN(eps=epsilon, min_samples=min_samples).fit(features).labels_


def subtract_slope(snippets, heights):
    """ Subtract underlying slope from all EOD snippets.

    Parameters
    ----------
    snippets: 2-D numpy array
        All EODs in a recorded stacked as snippets. 
        Shape = (number of EODs, EOD width)
    heights: 1D numpy array
        EOD heights.

    Returns
    -------
    snippets: 2-D numpy array
        EOD snippets with underlying slope subtracted.
    bg_ratio : 1-D numpy array
        EOD height/background activity height.
    """

    left_y = snippets[:, 0]
    right_y = snippets[:, -1]

    try:
        slopes = np.linspace(left_y, right_y, snippets.shape[1])
    except ValueError:
        delta = (right_y - left_y)/snippets.shape[1]
        slopes = np.arange(0, snippets.shape[1], dtype=snippets.dtype).reshape((-1,) + (1,) * np.ndim(delta))*delta + left_y
    
    return snippets - slopes.T, np.abs(left_y-right_y)/heights


def remove_artefacts(all_snippets, clusters, rate,
                     freq_low=20000, threshold=0.75,
                     verbose=0, return_data=[]):
    """ Create a mask for EOD clusters that result from artefacts, based on power in low frequency spectrum.

    Parameters
    ----------
    all_snippets: 2D array
        EOD snippets. Shape=(nEODs, EOD length)
    clusters: list of int
        EOD cluster labels
    rate : float
        Sampling rate of original recording data.
    freq_low: float
        Frequency up to which low frequency components are summed up. 
    threshold : float
        Minimum value for sum of low frequency components relative to
        sum overa ll spectrl amplitudes that separates artefact from
        clean pulsefish clusters.
    verbose : int
        Verbosity level.
    return_data : list of str
        Keys that specify data to be logged. The key that can be used to log data in this function is
        'eod_deletion' (see extract_pulsefish()).

    Returns
    -------
    mask: numpy array of booleans
        Set to True for every EOD which is an artefact.
    adict : dictionary
        Key value pairs of logged data. Data to be logged is specified by return_data.
    """
    adict = {}

    mask = np.zeros(clusters.shape, dtype=bool)

    for cluster in np.unique(clusters[clusters >= 0]):
        snippets = all_snippets[clusters == cluster]
        mean_eod = np.mean(snippets, axis=0)
        mean_eod = mean_eod - np.mean(mean_eod)
        mean_eod_fft = np.abs(np.fft.rfft(mean_eod))
        freqs = np.fft.rfftfreq(len(mean_eod), 1/rate)
        low_frequency_ratio = np.sum(mean_eod_fft[freqs<freq_low])/np.sum(mean_eod_fft)
        if low_frequency_ratio < threshold:  # TODO: check threshold!
            mask[clusters == cluster] = True
            
            if verbose > 0:
                print('Deleting cluster %i with low frequency ratio of %.3f (min %.3f)' % (cluster, low_frequency_ratio, threshold))

        if 'eod_deletion' in return_data:
            adict['vals_%d' % cluster] = [mean_eod, mean_eod_fft]
            adict['mask_%d' % cluster] = [np.any(mask[clusters == cluster])]
    
    return mask, adict


def delete_unreliable_fish(clusters, eod_widths, eod_x, verbose=0, sdict={}):
    """ Create a mask for EOD clusters that are either mixed with noise or other fish, or wavefish.
    
    This is the case when the ration between the EOD width and the ISI is too large.

    Parameters
    ----------
    clusters : list of int
        Cluster labels.
    eod_widths : list of float or int
        EOD widths in samples or seconds.
    eod_x : list of int or floats
        EOD times in samples or seconds.

    verbose : int
        Verbosity level.
    sdict : dictionary
        Dictionary that is used to log data. This is only used if a dictionary
        was created by remove_artefacts().
        For logging data in noise and wavefish discarding steps,
        see remove_artefacts().

    Returns
    -------
    mask : numpy array of booleans
        Set to True for every unreliable EOD.
    sdict : dictionary
        Key value pairs of logged data. Data is only logged if a dictionary
        was instantiated by remove_artefacts().
    """
    mask = np.zeros(clusters.shape, dtype=bool)
    for cluster in np.unique(clusters[clusters >= 0]):
        if len(eod_x[cluster == clusters]) < 2:
            mask[clusters == cluster] = True
            if verbose > 0:
                print('deleting unreliable cluster %i, number of EOD times %d < 2' % (cluster, len(eod_x[cluster == clusters])))
        elif np.max(np.median(eod_widths[clusters == cluster])/np.diff(eod_x[cluster == clusters])) > 0.5:
            if verbose > 0:
                print('deleting unreliable cluster %i, score=%f' % (cluster, np.max(np.median(eod_widths[clusters == cluster])/np.diff(eod_x[cluster == clusters]))))
            mask[clusters == cluster] = True
        if 'vals_%d' % cluster in sdict:
            sdict['vals_%d' % cluster].append(np.median(eod_widths[clusters == cluster])/np.diff(eod_x[cluster == clusters]))
            sdict['mask_%d' % cluster].append(any(mask[clusters == cluster]))
    return mask, sdict


def delete_wavefish_and_sidepeaks(data, clusters, eod_x, eod_widths,
                                  width_fac, max_slope_deviation=0.5,
                                  max_phases=4, verbose=0, sdict={}):
    """ Create a mask for EODs that are likely from wavefish, or sidepeaks of bigger EODs.

    Parameters
    ----------
    data : list of float
        Raw recording data.
    clusters : list of int
        Cluster labels.
    eod_x : list of int
        Indices of EOD times.
    eod_widths : list of int
        EOD widths in samples.
    width_fac : float
        Multiplier for EOD analysis width.

    max_slope_deviation: float
        Maximum deviation of position of maximum slope in snippets from
        center position in multiples of mean width of EOD.
    max_phases : int
        Maximum number of phases for any EOD. 
        If the mean EOD has more phases than this, it is not a pulse EOD.
    verbose : int 
        Verbosity level.
    sdict : dictionary
        Dictionary that is used to log data. This is only used if a dictionary
        was created by remove_artefacts().
        For logging data in noise and wavefish discarding steps, see remove_artefacts().

    Returns
    -------
    mask_wave: numpy array of booleans
        Set to True for every EOD which is a wavefish EOD.
    mask_sidepeak: numpy array of booleans
        Set to True for every snippet which is centered around a sidepeak of an EOD.
    sdict : dictionary
        Key value pairs of logged data. Data is only logged if a dictionary
        was instantiated by remove_artefacts().
    """
    mask_wave = np.zeros(clusters.shape, dtype=bool)
    mask_sidepeak = np.zeros(clusters.shape, dtype=bool)

    for i, cluster in enumerate(np.unique(clusters[clusters >= 0])):
        mean_width = np.mean(eod_widths[clusters == cluster])
        cutwidth = mean_width*width_fac
        current_x = eod_x[(eod_x>cutwidth) & (eod_x<(len(data)-cutwidth))]
        current_clusters = clusters[(eod_x>cutwidth) & (eod_x<(len(data)-cutwidth))]
        snippets = np.vstack([data[int(x-cutwidth):int(x+cutwidth)]
                              for x in current_x[current_clusters == cluster]])
        
        # extract information on main peaks and troughs:
        mean_eod = np.mean(snippets, axis=0)
        mean_eod = mean_eod - np.mean(mean_eod)

        # detect peaks and troughs on data + some maxima/minima at the
        # end, so that the sides are also considered for peak detection:
        pk, tr = detect_peaks(np.concatenate(([-10*mean_eod[0]], mean_eod, [10*mean_eod[-1]])),
                              np.std(mean_eod))
        pk = pk[(pk>0)&(pk<len(mean_eod))]
        tr = tr[(tr>0)&(tr<len(mean_eod))]

        if len(pk)>0 and len(tr)>0:
            idxs = np.sort(np.concatenate((pk, tr)))
            slopes = np.abs(np.diff(mean_eod[idxs]))
            m_slope = np.argmax(slopes)
            centered = np.min(np.abs(idxs[m_slope:m_slope+2] - len(mean_eod)//2))
            
            # compute all height differences of peaks and troughs within snippets.
            # if they are all similar, it is probably noise or a wavefish.
            idxs = np.sort(np.concatenate((pk, tr)))
            hdiffs = np.diff(mean_eod[idxs])

            if centered > max_slope_deviation*mean_width:  # TODO: check, factor was probably 0.16
                if verbose > 0:
                    print('Deleting cluster %i, which is a sidepeak' % cluster)
                mask_sidepeak[clusters == cluster] = True

            w_diff = np.abs(np.diff(np.sort(np.concatenate((pk, tr)))))

            if np.abs(np.diff(idxs[m_slope:m_slope+2])) < np.mean(eod_widths[clusters == cluster])*0.5 or len(pk) + len(tr)>max_phases or np.min(w_diff)>2*cutwidth/width_fac: #or len(hdiffs[np.abs(hdiffs)>0.5*(np.max(mean_eod)-np.min(mean_eod))])>max_phases:
                if verbose > 0:
                    print('Deleting cluster %i, which is a wavefish' % cluster)
                mask_wave[clusters == cluster] = True
        if 'vals_%d' % cluster in sdict:
            sdict['vals_%d' % cluster].append([mean_eod, [pk, tr],
                                               idxs[m_slope:m_slope+2]])
            sdict['mask_%d' % cluster].append(any(mask_wave[clusters == cluster]))
            sdict['mask_%d' % cluster].append(any(mask_sidepeak[clusters == cluster]))

    return mask_wave, mask_sidepeak, sdict


def merge_clusters(clusters_1, clusters_2, x_1, x_2, verbose=0): 
    """ Merge clusters resulting from two clustering methods.

    This method only works  if clustering is performed on the same EODs
    with the same ordering, where there  is a one to one mapping from
    clusters_1 to clusters_2. 

    Parameters
    ----------
    clusters_1: list of int
        EOD cluster labels for cluster method 1.
    clusters_2: list of int
        EOD cluster labels for cluster method 2.
    x_1: list of int
        Indices of EODs for cluster method 1 (clusters_1).
    x_2: list of int
        Indices of EODs for cluster method 2 (clusters_2).
    verbose : int
        Verbosity level.

    Returns
    -------
    clusters : list of int
        Merged clusters.
    x_merged : list of int
        Merged cluster indices.
    mask : 2d numpy array of int (N, 2)
        Mask for clusters that are selected from clusters_1 (mask[:,0]) and
        from clusters_2 (mask[:,1]).
    """
    if verbose > 0:
        print('\nmerge cluster:')

    # these arrays become 1 for each EOD that is chosen from that array
    c1_keep = np.zeros(len(clusters_1), dtype=int)
    c2_keep = np.zeros(len(clusters_2), dtype=int)

    # add n to one of the cluster lists to avoid overlap
    ovl = np.max(clusters_1) + 1
    clusters_2[clusters_2!=-1] = clusters_2[clusters_2!=-1] + ovl

    remove_clusters = [[]]
    keep_clusters = []
    og_clusters = [np.copy(clusters_1), np.copy(clusters_2)]
    
    # loop untill done
    while True:

        # compute unique clusters and cluster sizes
        # of cluster that have not been iterated over:
        c1_labels, c1_size = unique_counts(clusters_1[(clusters_1 != -1) & (c1_keep == 0)])
        c2_labels, c2_size = unique_counts(clusters_2[(clusters_2 != -1) & (c2_keep == 0)])

        # if all clusters are done, break from loop:
        if len(c1_size) == 0 and len(c2_size) == 0:
            break

        # if the biggest cluster is in c_p, keep this one and discard all clusters
        # on the same indices in c_t:
        elif np.argmax([np.max(np.append(c1_size, 0)), np.max(np.append(c2_size, 0))]) == 0:
            
            # remove all the mappings from the other indices
            cluster_mappings, _ = unique_counts(clusters_2[clusters_1 == c1_labels[np.argmax(c1_size)]])
            
            clusters_2[np.isin(clusters_2, cluster_mappings)] = -1
            
            c1_keep[clusters_1 == c1_labels[np.argmax(c1_size)]] = 1

            remove_clusters.append(cluster_mappings)
            keep_clusters.append(c1_labels[np.argmax(c1_size)])

            if verbose > 0:
                print(f'  group 1: keep cluster {c1_labels[np.argmax(c1_size)]:2d}, group 2: delete clusters {cluster_mappings[cluster_mappings != -1] - ovl}')

        # if the biggest cluster is in c_t, keep this one and discard all mappings in c_p
        elif np.argmax([np.max(np.append(c1_size, 0)), np.max(np.append(c2_size, 0))]) == 1:
            
            # remove all the mappings from the other indices
            cluster_mappings, _ = unique_counts(clusters_1[clusters_2 == c2_labels[np.argmax(c2_size)]])
            
            clusters_1[np.isin(clusters_1, cluster_mappings)] = -1

            c2_keep[clusters_2 == c2_labels[np.argmax(c2_size)]] = 1

            remove_clusters.append(cluster_mappings)
            keep_clusters.append(c2_labels[np.argmax(c2_size)])

            if verbose > 0:
                print(f'  group 2: keep cluster {c2_labels[np.argmax(c2_size)] - ovl:2d}, group 1: delete clusters {cluster_mappings[cluster_mappings!=-1]}')
    
    # combine results    
    clusters = (clusters_1 + 1)*c1_keep + (clusters_2 + 1)*c2_keep - 1
    x_merged = (x_1)*c1_keep + (x_2)*c2_keep

    return clusters.astype(int), x_merged, np.vstack([c1_keep, c2_keep])


def extract_means(data, eod_inx, eod_peak_inx, eod_tr_inx, eod_widths,
                  clusters, rate, width_fac, verbose=0):
    """ Extract mean EODs and EOD timepoints for each EOD cluster.

    Parameters
    ----------
    data: 1-D array of float
        Raw recording data.
    eod_inx: list of int
        Locations of EODs in samples.
    eod_peak_inx : list of int
        Locations of EOD peaks in samples.
    eod_tr_inx : list of int
        Locations of EOD troughs in samples.
    eod_widths: list of int
        EOD widths in samples.
    clusters: list of int
        EOD cluster labels
    rate: float
        Sampling rate of recording  
    width_fac : float
        Multiplication factor for window used to extract EOD.
    
    verbose : int
        Verbosity level.

    Returns
    -------
    mean_eods: list of 2D arrays
        The average EOD for each detected fish. First column is time in seconds,
        second column the mean eod, third column the standard deviation.
    eod_times: list of 1D arrays
        For each detected fish the times of EOD in seconds.
    eod_peak_times: list of 1D arrays
        For each detected fish the times of EOD peaks in seconds.
    eod_trough_times: list of 1D arrays
        For each detected fish the times of EOD troughs in seconds.
    eod_labels: list of int
        Cluster label for each detected fish.
    """
    mean_eods = []
    eod_times = []
    eod_peak_times = []
    eod_tr_times = []
    eod_heights = []
    cluster_labels = []

    for cluster in np.unique(clusters):
        if cluster != -1:
            cutwidth = np.mean(eod_widths[clusters == cluster])*width_fac
            current_inx = eod_inx[(eod_inx > cutwidth) & (eod_inx < (len(data) - cutwidth))]
            current_clusters = clusters[(eod_inx > cutwidth) & (eod_inx < (len(data)-cutwidth))]

            snippets = np.vstack([data[int(x-cutwidth):int(x+cutwidth)] for x in current_inx[current_clusters == cluster]])
            mean_eod = np.mean(snippets, axis=0)
            eod_time = np.arange(len(mean_eod))/rate - cutwidth/rate

            mean_eod = np.column_stack([eod_time,
                                        mean_eod,
                                        np.std(snippets, axis=0)])

            mean_eods.append(mean_eod)
            eod_times.append(eod_inx[clusters == cluster]/rate)
            eod_heights.append(np.max(mean_eod[:, 1]) - np.max(mean_eod[:, 1]))
            eod_peak_times.append(eod_peak_inx[clusters == cluster]/rate)
            eod_tr_times.append(eod_tr_inx[clusters == cluster]/rate)
            cluster_labels.append(cluster)

    sidx = np.argsort(eod_heights)
    return [mean_eods[i] for i in sidx], [eod_times[i] for i in sidx], [eod_peak_times[i] for i in sidx], [eod_tr_times[i] for i in sidx], [cluster_labels[i] for i in sidx]


def find_clipped_clusters(clusters, mean_eods, eod_times,
                          eod_peaktimes, eod_troughtimes,
                          cluster_labels, width_factor,
                          clip_threshold=0.9, verbose=0):
    """ Detect EODs that are clipped and set all clusterlabels of these clipped EODs to -1.
                          
    Also return the mean EODs and timepoints of these clipped EODs.

    Parameters
    ----------
    clusters: array of int
        Cluster labels for each EOD in a recording.
    mean_eods: list of numpy arrays
        Mean EOD waveform for each cluster.
    eod_times: list of numpy arrays
        EOD timepoints for each EOD cluster.
    eod_peaktimes
        EOD peaktimes for each EOD cluster.
    eod_troughtimes
        EOD troughtimes for each EOD cluster.
    cluster_labels: numpy array
        Unique EOD clusterlabels.
    clip_threshold: float
        Threshold for detecting clipped EODs.
    
    verbose: int
        Verbosity level.

    Returns
    -------
    clusters : array of int
        Cluster labels for each EOD in the recording, where clipped EODs have been set to -1.
    clipped_eods : list of numpy arrays
        Mean EOD waveforms for each clipped EOD cluster.
    clipped_times : list of numpy arrays
        EOD timepoints for each clipped EOD cluster.
    clipped_peaktimes : list of numpy arrays
        EOD peaktimes for each clipped EOD cluster.
    clipped_troughtimes : list of numpy arrays
        EOD troughtimes for each clipped EOD cluster.
    """
    clipped_eods, clipped_times, clipped_peaktimes, clipped_troughtimes, clipped_labels = [], [], [], [], []

    for mean_eod, eod_time, eod_peaktime, eod_troughtime,label in zip(mean_eods, eod_times, eod_peaktimes, eod_troughtimes, cluster_labels):
        
        if (np.count_nonzero(mean_eod[1]>clip_threshold) > len(mean_eod[1])/(width_factor*2)) or (np.count_nonzero(mean_eod[1] < -clip_threshold) > len(mean_eod[1])/(width_factor*2)):
            clipped_eods.append(mean_eod)
            clipped_times.append(eod_time)
            clipped_peaktimes.append(eod_peaktime)
            clipped_troughtimes.append(eod_troughtime)
            clipped_labels.append(label)
            if verbose > 0:
                print('clipped pulsefish')

    clusters[np.isin(clusters, clipped_labels)] = -1

    return clusters, clipped_eods, clipped_times, clipped_peaktimes, clipped_troughtimes


def delete_moving_fish(clusters, eod_t, T, eod_heights, eod_widths,
                       rate, min_dt=0.25, stepsize=0.05,
                       sliding_window_factor=2000, verbose=0,
                       plot_level=0, save_plot=False, save_path='',
                       ftype='pdf', return_data=[]):
    """
    Use a sliding window to detect the minimum number of fish detected simultaneously, 
    then delete all other EOD clusters. 

    Do this only for EODs within the same width clusters, as a
    moving fish will preserve its EOD width.

    Parameters
    ----------
    clusters: list of int
        EOD cluster labels.
    eod_t: list of float
        Timepoints of the EODs (in seconds).
    T: float
        Length of recording (in seconds).
    eod_heights: list of float
        EOD amplitudes.
    eod_widths: list of float
        EOD widths (in seconds).
    rate: float
        Recording data sampling rate.

    min_dt : float
        Minimum sliding window size (in seconds).
    stepsize : float
        Sliding window stepsize (in seconds).
    sliding_window_factor : float
        Multiplier for sliding window width,
        where the sliding window width = median(EOD_width)*sliding_window_factor.
    verbose : int
        Verbosity level.
    plot_level : int
        Similar to verbosity levels, but with plots. 
        Only set to > 0 for debugging purposes.
    save_plot : bool
        Set to True to save the plots created by plot_level.
    save_path : str
        Path to save data to. Only important if you wish to save data (save_data == True).
    ftype : str
        Define the filetype to save the plots in if save_plots is set to True.
        Options are: 'png', 'jpg', 'svg' ...
    return_data : list of str
        Keys that specify data to be logged. The key that can be used to log data
        in this function is 'moving_fish' (see extract_pulsefish()).

    Returns
    -------
    clusters : list of int
        Cluster labels, where deleted clusters have been set to -1.
    window : list of 2 floats
        Start and end of window selected for deleting moving fish in seconds.
    mf_dict : dictionary
        Key value pairs of logged data. Data to be logged is specified by return_data.
    """
    mf_dict = {}
    
    if len(np.unique(clusters[clusters != -1])) == 0:
        return clusters, [0, 1], {}

    all_keep_clusters = []
    width_classes = merge_gaussians(eod_widths, np.copy(clusters), 0.75)   

    all_windows = []
    all_dts = []
    ev_num = 0
    for iw, w in enumerate(np.unique(width_classes[clusters >= 0])):
        # initialize variables
        min_clusters = 100
        average_height = 0
        sparse_clusters = 100
        keep_clusters = []

        dt = max(min_dt, np.median(eod_widths[width_classes == w])*sliding_window_factor)
        window_start = 0
        window_end = dt

        wclusters = clusters[width_classes == w]
        weod_t = eod_t[width_classes == w]
        weod_heights = eod_heights[width_classes == w]
        weod_widths = eod_widths[width_classes == w]

        all_dts.append(dt)

        if verbose > 0:
            print('sliding window dt = %f'%dt)
        
        x = np.arange(0, T - dt + stepsize, stepsize)
        y = np.ones(len(x), dtype=int)

        # make W dependent on width??
        ignore_steps = np.zeros(len(x), dtype=int)

        for i, t in enumerate(x):
            current_clusters = wclusters[(weod_t>=t)&(weod_t<t+dt)&(wclusters!=-1)]
            if len(np.unique(current_clusters)) == 0:
                ignore_steps[i-int(dt/stepsize):i+int(dt/stepsize)] = 1
                if verbose > 0:
                    print('No pulsefish in recording at T=%.2f:%.2f' % (t, t+dt))

        running_sum = np.ones(len(x), dtype=int)
        ulabs = np.unique(wclusters[wclusters>=0])

        # sliding window
        for j, (t, ignore_step) in enumerate(zip(x, ignore_steps)):
            current_clusters = wclusters[(weod_t>=t)&(weod_t<t+dt)&(wclusters!=-1)]
            current_widths = weod_widths[(weod_t>=t)&(weod_t<t+dt)&(wclusters!=-1)]

            unique_clusters = np.unique(current_clusters)
            y[j] = len(unique_clusters)

            if (len(unique_clusters) <= min_clusters) and \
              (ignore_step == 0) and \
              (len(unique_clusters !=1)):

                current_labels = np.isin(wclusters, unique_clusters)
                current_height = np.mean(weod_heights[current_labels])

                # compute nr of clusters that are too sparse
                clusters_after_deletion = np.unique(remove_sparse_detections(np.copy(clusters[np.isin(clusters, unique_clusters)]), rate*eod_widths[np.isin(clusters, unique_clusters)], rate, T))
                current_sparse_clusters = len(unique_clusters) - len(clusters_after_deletion[clusters_after_deletion!=-1])
               
                if current_sparse_clusters <= sparse_clusters and \
                  ((current_sparse_clusters<sparse_clusters) or
                   (current_height > average_height) or
                   (len(unique_clusters) < min_clusters)):
                    
                    keep_clusters = unique_clusters
                    min_clusters = len(unique_clusters)
                    average_height = current_height
                    window_end = t+dt
                    sparse_clusters = current_sparse_clusters

        all_keep_clusters.append(keep_clusters)
        all_windows.append(window_end)
        
        if 'moving_fish' in return_data or plot_level > 0:
            if 'w' in mf_dict:
                mf_dict['w'].append(np.median(eod_widths[width_classes == w]))
                mf_dict['T'] = T
                mf_dict['dt'].append(dt)
                mf_dict['clusters'].append(wclusters)
                mf_dict['t'].append(weod_t)
                mf_dict['fishcount'].append([x+0.5*(x[1]-x[0]), y])
                mf_dict['ignore_steps'].append(ignore_steps)
            else:
                mf_dict['w'] = [np.median(eod_widths[width_classes == w])]
                mf_dict['T'] = [T]
                mf_dict['dt'] = [dt]
                mf_dict['clusters'] = [wclusters]
                mf_dict['t'] = [weod_t]
                mf_dict['fishcount'] = [[x+0.5*(x[1]-x[0]), y]]
                mf_dict['ignore_steps'] = [ignore_steps]

    if verbose > 0:
        print('Estimated nr of pulsefish in recording: %i'%len(all_keep_clusters))

    if plot_level > 0:
        plot_moving_fish(mf_dict['w'], mf_dict['dt'], mf_dict['clusters'],mf_dict['t'],
                         mf_dict['fishcount'], T, mf_dict['ignore_steps'])
        if save_plot:
            plt.savefig('%sdelete_moving_fish.%s' % (save_path, ftype))
        # empty dict
        if 'moving_fish' not in return_data:
            mf_dict = {}

    # delete all clusters that are not selected
    clusters[np.invert(np.isin(clusters, np.concatenate(all_keep_clusters)))] = -1

    return clusters, [np.max(all_windows)-np.max(all_dts), np.max(all_windows)], mf_dict


def remove_sparse_detections(clusters, eod_widths, rate, T,
                             min_density=0.0005, verbose=0):
    """ Remove all EOD clusters that are too sparse

    Parameters
    ----------
    clusters : list of int
        Cluster labels.
    eod_widths : list of int
        Cluster widths in samples.
    rate : float
        Sampling rate.
    T : float
        Lenght of recording in seconds.
    min_density : float
        Minimum density for realistic EOD detections.
    verbose : int
        Verbosity level.

    Returns
    -------
    clusters : list of int
        Cluster labels, where sparse clusters have been set to -1.
    """
    for c in np.unique(clusters):
        if c!=-1:

            n = len(clusters[clusters == c])
            w = np.median(eod_widths[clusters == c])/rate

            if n*w < T*min_density:
                if verbose > 0:
                    print('cluster %i is too sparse'%c)
                clusters[clusters == c] = -1
    return clusters
