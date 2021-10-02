"""
Extract and cluster EOD waverforms of pulse-type electric fish.

## Main function

- `extract_pulsefish()`: checks for pulse-type fish based on the EOD amplitude and shape.

"""

import os
import numpy as np
from scipy import stats
from scipy.interpolate import interp1d
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from sklearn.mixture import BayesianGaussianMixture
from sklearn.metrics import pairwise_distances
from .eventdetection import detect_peaks, median_std_threshold
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
        mask = np.empty(ar.shape, dtype=np.bool_)
        mask[:1] = True
        mask[1:] = ar[1:] != ar[:-1]
        idx = np.concatenate(np.nonzero(mask) + ([mask.size],))
        return ar[mask], np.diff(idx)  


###################################################################################


def extract_pulsefish(data, samplerate, width_factor_shape=3, width_factor_wave=8,
                      width_factor_display=4, verbose=0, plot_level=0, save_plots=False,
                      save_path='', ftype='png', return_data=[]):
    """ Extract and cluster pulse-type fish EODs from single channel data.
    
    Takes recording data containing an unknown number of pulsefish and extracts the mean 
    EOD and EOD timepoints for each fish present in the recording.
    
    Parameters
    ----------
    data: 1-D array of float
        The data to be analysed.
    samplerate: float
        Sampling rate of the data in Hertz.

    width_factor_shape : int or float (optional)
        Width multiplier used for EOD shape analysis.
        EOD snippets are extracted based on width between the 
        peak and trough multiplied by the width factor.
    width_factor_wave : int or float (optional)
        Width multiplier used for wavefish detection.
    width_factor_display :  int or float (optional)
        Width multiplier used for EOD mean extraction and display.
    verbose : int (optional)
        Verbosity level.
    plot_level : int (optional)
        Similar to verbosity levels, but with plots. 
        Only set to > 0 for debugging purposes.
    save_plots : bool (optional)
        Set to True to save the plots created by plot_level.
    save_path: string (optional)
        Path for saving plots.
    ftype : string (optional)
        Define the filetype to save the plots in if save_plots is set to True.
        Options are: 'png', 'jpg', 'svg' ...
    return_data : list of strings (optional)
        Specify data that should be logged and returned in a dictionary. Each clustering 
        step has a specific keyword that results in adding different variables to the log dictionary.
        Optional keys for return_data and the resulting additional key-value pairs to the log dictionary are:
        
        - 'all_eod_times':
            - 'all_times':  list of two lists of floats.
                All peak (`all_times[0]`) and trough times (`all_times[1]`) extracted
                by the peak detection algorithm. Times are given in seconds.
            - 'eod_troughtimes': list of 1D arrays.
                The timepoints in seconds of each unique extracted EOD cluster,
                where each 1D array encodes one cluster.
        
        - 'peak_detection':
            - "data": 1D numpy array of floats.
                Quadratically interpolated data which was used for peak detection.
            - "interp_f": float.
                Interpolation factor of raw data.
            - "peaks_1": 1D numpy array of ints.
                Peak indices on interpolated data after first peak detection step.
            - "troughs_1": 1D numpy array of ints.
                Peak indices on interpolated data after first peak detection step.
            - "peaks_2": 1D numpy array of ints.
                Peak indices on interpolated data after second peak detection step.
            - "troughs_2": 1D numpy array of ints.
                Peak indices on interpolated data after second peak detection step.
            - "peaks_3": 1D numpy array of ints.
                Peak indices on interpolated data after third peak detection step.
            - "troughs_3": 1D numpy array of ints.
                Peak indices on interpolated data after third peak detection step.
            - "peaks_4": 1D numpy array of ints.
                Peak indices on interpolated data after fourth peak detection step.
            - "troughs_4": 1D numpy array of ints.
                Peak indices on interpolated data after fourth peak detection step.

        - 'all_cluster_steps':
            - 'samplerate': float.
                Samplerate of interpolated data.
            - 'EOD_widths': list of three 1D numpy arrays.
                The first list entry gives the unique labels of all width clusters
                as a list of ints.
                The second list entry gives the width values for each EOD in samples
                as a 1D numpy array of ints.
                The third list entry gives the width labels for each EOD
                as a 1D numpy array of ints.
            - 'EOD_heights': nested lists (2 layers) of three 1D numpy arrays.
                The first list entry gives the unique labels of all height clusters
                as a list of ints for each width cluster.
                The second list entry gives the height values for each EOD
                as a 1D numpy array of floats for each width cluster.
                The third list entry gives the height labels for each EOD
                as a 1D numpy array of ints for each width cluster.
            - 'EOD_shapes': nested lists (3 layers) of three 1D numpy arrays
                The first list entry gives the raw EOD snippets as a 2D numpy array
                for each height cluster in a width cluster.
                The second list entry gives the snippet PCA values for each EOD
                as a 2D numpy array of floats for each height cluster in a width cluster.
                The third list entry gives the shape labels for each EOD as a 1D numpy array
                of ints for each height cluster in a width cluster.
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
                - 'x': 1D numpy array of floats.
                    BGM input values (in this case the EOD widths),
                - 'use_log': boolean.
                    True if the z-scored logarithm of the data was used as BGM input.
                - 'BGM': list of three 1D numpy arrays.
                    The first instance are the weights of the Gaussian fits.
                    The second instance are the means of the Gaussian fits.
                    The third instance are the variances of the Gaussian fits.
                - 'labels': 1D numpy array of ints.
                    Labels defined by BGM model (before merging based on merge factor).
                - xlab': string.
                    Label for plot (defines the units of the BGM data).

        - 'BGM_height':
            This key adds a new dictionary for each width cluster.
            - 'BGM_height_*n*' : dictionary, where *n* defines the width cluster as an int.
                - 'x': 1D numpy array of floats.
                    BGM input values (in this case the EOD heights),
                - 'use_log': boolean.
                    True if the z-scored logarithm of the data was used as BGM input.
                - 'BGM': list of three 1D numpy arrays.
                    The first instance are the weights of the Gaussian fits.
                    The second instance are the means of the Gaussian fits.
                    The third instance are the variances of the Gaussian fits.
                - 'labels': 1D numpy array of ints.
                    Labels defined by BGM model (before merging based on merge factor).
                - 'xlab': string.
                    Label for plot (defines the units of the BGM data).

        - 'snippet_clusters':
            This key adds a new dictionary for each height cluster.
            - 'snippet_clusters*_n_m_p*' : dictionary, where *n* defines the width cluster
              (int), *m* defines the height cluster (int) and *p* defines shape clustering
              on peak or trough centered EOD snippets (string: 'peak' or 'trough').
                - 'raw_snippets': 2D numpy array (nsamples, nfeatures).
                    Raw EOD snippets.
                - 'snippets': 2D numpy array.
                    Normalized EOD snippets.
                - 'features': 2D numpy array.(nsamples, nfeatures)
                    PCA values for each normalized EOD snippet.
                - 'clusters': 1D numpy array of ints.
                    Cluster labels.
                - 'samplerate': float.
                    Samplerate of snippets.

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
            - 'samplerate' : float.
                EOD snippet samplerate.

        - 'masks': 
            - 'masks' : 2D numpy array (4,N).
                Each row contains masks for each EOD detected by the EOD peakdetection step. 
                The first row defines the artefact masks, the second row defines the
                unreliable EOD masks, 
                the third row defines the wavefish masks and the fourth row defines
                the sidepeak masks.

        - 'moving_fish':
            - 'moving_fish': dictionary.
                - 'w' : list of floats.
                    Median width for each width cluster that the moving fish algorithm is
                    computed on (in seconds).
                - 'T' : list of floats.
                    Lenght of analyzed recording for each width cluster (in seconds).
                - 'dt' : list of floats.
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
    mean_eods: list of 2D arrays (3, eod_length)
        The average EOD for each detected fish. First column is time in seconds,
        second column the mean eod, third column the standard error.
    eod_times: list of 1D arrays
        For each detected fish the times of EOD peaks or troughs in seconds.
        Use these timepoints for EOD averaging.
    eod_peaktimes: list of 1D arrays
        For each detected fish the times of EOD peaks in seconds.
    zoom_window: tuple of floats
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

    if (save_plots and plot_level>0 and save_path):
        # create folder to save things in.
        if not os.path.exists(save_path):
            os.makedirs(save_path)
    else:
        save_path = ''

    mean_eods, eod_times, eod_peaktimes, zoom_window = [], [], [], []
    log_dict = {}
    
    # extract peaks and interpolated data
    x_peak, x_trough, eod_heights, eod_widths, i_samplerate, i_data, interp_f, pd_log_dict = \
      extract_eod_times(data, samplerate,
                        np.max([width_factor_shape, width_factor_display, width_factor_wave]),
                        verbose=verbose-1, return_data=return_data, save_path=save_path)
    
    if len(x_peak) > 0:
        # cluster
        clusters, x_merge, c_log_dict = cluster(x_peak, x_trough, eod_heights, eod_widths,
                                                i_data, i_samplerate, interp_f,
                                                width_factor_shape, width_factor_wave,
                                                verbose=verbose-1, plot_level=plot_level-1,
                                                save_plots=save_plots, save_path=save_path,
                                                ftype=ftype, return_data=return_data) 

        # extract mean eods and times
        mean_eods, eod_times, eod_peaktimes, eod_troughtimes, cluster_labels = \
          extract_means(i_data, x_merge, x_peak, x_trough, eod_widths, clusters,
                        i_samplerate, width_factor_display, verbose=verbose-1)

        # determine clipped clusters (save them, but ignore in other steps)
        clusters, clipped_eods, clipped_times, clipped_peaktimes, clipped_troughtimes = \
          find_clipped_clusters(clusters, mean_eods, eod_times, eod_peaktimes,
                                eod_troughtimes, cluster_labels, width_factor_display,
                                verbose=verbose-1)

        # delete the moving fish
        clusters, zoom_window, mf_log_dict = \
          delete_moving_fish(clusters, x_merge/i_samplerate, len(data)/samplerate,
                             eod_heights, eod_widths/i_samplerate, i_samplerate,
                             verbose=verbose-1, plot_level=plot_level-1, save_plot=save_plots,
                             save_path=save_path, ftype=ftype, return_data=return_data)
        
        if 'moving_fish' in return_data:
            log_dict['moving_fish'] = mf_log_dict

        clusters = remove_sparse_detections(clusters, eod_widths, i_samplerate,
                                            len(data)/samplerate, verbose=verbose-1)

        # extract mean eods
        mean_eods, eod_times, eod_peaktimes, eod_troughtimes, cluster_labels = \
          extract_means(i_data, x_merge, x_peak, x_trough, eod_widths,
                        clusters, i_samplerate, width_factor_display, verbose=verbose-1)

        mean_eods.extend(clipped_eods)
        eod_times.extend(clipped_times)
        eod_peaktimes.extend(clipped_peaktimes)
        eod_troughtimes.extend(clipped_troughtimes)

        if plot_level > 0:
            plot_all(data, eod_peaktimes, eod_troughtimes, samplerate, mean_eods)
            if save_plots:
                plt.savefig('%sextract_pulsefish_results.%s'%(save_path, ftype))
        if save_plots:
            plt.close('all')
    
        if 'all_eod_times' in return_data:
            log_dict['all_times'] = [x_peak/i_samplerate, x_trough/i_samplerate]
            log_dict['eod_troughtimes'] = eod_troughtimes
        
        log_dict.update(pd_log_dict)
        log_dict.update(c_log_dict)
        log_dict['samplerate'] = i_samplerate

    if plot_level > 0:
        # reset font family for the main thunderfish plot
        rcParams['font.family'] = 'sans-serif'

    return mean_eods, eod_times, eod_peaktimes, zoom_window, log_dict


def extract_eod_times(data, samplerate, width_factor, interp_freq=500000,
                      max_peakwidth=0.01, min_peakwidth=None, verbose=0,
                      return_data=[], save_path=''):
    """ Extract peaks from data which are potentially EODs.

    Parameters
    ----------
    data: 1-D array of float
        The data to be analysed.
    samplerate: int or float
        Sampling rate of the data
    width_factor: int or float
        Factor for extracting EOD shapes.
        Only EODs are extracted that can fully be analysed with this width.

    interp_freq: int or float (optional)
        Desired resolution in Hz. Data will be interpolated to match this resolution.
    max_peakwidth: float (optional)
        Maximum width for peak detection in seconds.
    min_peakwidth: int or None (optional)
        Minimum width for peak detection in seconds.
        If None, the minimum width is determined from the recording data
        resolution  (2/samplerate).

    verbose : int (optional)
        Verbosity level.
    return_data : list of strings (optional)
        Keys that specify data to be logged. If 'peak_detection' is in `return_data`, 
        data of this function is logged (see extract_pulsefish()).
    save_path : string (optional)
        Path to save data to. Only important if you wish to save data (save_data==True).

    Returns
    -------
    x_peak: array of ints
        Indices of EOD peaks in data.
    x_trough: array of ints
        Indices of EOD troughs in data. There is one x_trough for each x_peak.
    eod_heights: array of floats
        EOD heights for each x_peak.
    eod_widths: array of ints
        EOD widths for each x_peak (in samples).
    samplerate: int or float
        New samplerate (after interpolation).
    data: 1-D array of floats
        Interpolated data.
    interpolation_factor: float
        Factor used for interpolation.
    peak_detection_result : dictionary
        Key value pairs of logged data. Data to be logged is specified by return_data.
    """

    peak_detection_result = {}
    
    # standard deviation of data in small snippets:
    threshold = median_std_threshold(data, samplerate)  # TODO pass parameters

    try:
        interp_f = int(interp_freq/samplerate)
        f = interp1d(range(len(data)), data, kind='quadratic')
        data = f(np.arange(0, len(data)-1, 1/interp_f))
    except MemoryError:
        interp_f = 1

    orig_x_peaks, orig_x_troughs = detect_peaks(data, threshold)
    orig_x_peaks = orig_x_peaks.astype('int')
    orig_x_troughs = orig_x_troughs.astype('int')

    if verbose>0:
        print('Peaks extracted:                                        %5i'%(len(orig_x_peaks)))

    if len(orig_x_peaks)<2 or len(orig_x_troughs)<2 or len(orig_x_peaks)>samplerate:
        if verbose>0:
            print('No peaks detected.')
        return [], [], [], [], samplerate*interp_f, data, interp_f, peak_detection_result
    else:

        if min_peakwidth == None:
            min_peakwidth = interp_f*2
        else:
            min_peakwidth = min_peakwidth*interp_freq

        peaks, troughs, heights, widths, apeaks, atroughs, aheights, awidths = \
          detect_eod_peaks(orig_x_peaks, orig_x_troughs, data, max_peakwidth*interp_freq,
                           min_peakwidth, verbose=verbose-1)
        x_peaks, x_troughs, eod_heights, eod_widths = \
          discard_connecting_eods(peaks, troughs, heights, widths, verbose=verbose-1)
        
        if 'peak_detection' in return_data:
            peak_detection_result = {"data": data,
                                     "interp_f": interp_f,
                                     "peaks_1": orig_x_peaks,
                                     "troughs_1": orig_x_troughs,
                                     "peaks_2": apeaks,
                                     "troughs_2": atroughs,
                                     "peaks_3": peaks,
                                     "troughs_3": troughs,
                                     "peaks_4": x_peaks,
                                     "troughs_4": x_troughs
                                    }

        # only take those where the maximum cutwidth does not casue issues
        # so if the width_factor times the width + x is more than length.
        if len(eod_widths)==0:
            if verbose>0:
                print('No EOD peaks detected.')
            return [], [], [], [], samplerate*interp_f,data, interp_f, peak_detection_result

        cut_idx = ((x_peaks + np.max(eod_widths)*width_factor < len(data)) & (x_troughs + np.max(eod_widths)*width_factor < len(data)) & (x_peaks - np.max(eod_widths)*width_factor > 0) & (x_troughs - np.max(eod_widths)*width_factor > 0))
        
        if verbose>0:
            print('Remaining peaks after EOD extraction:                   %5i'%(len(cut_idx)))
            if verbose>1:
                print('Remaining peaks after deletion due to cutwidth:         %5i'%(len(cut_idx)))
            print('')

        return x_peaks[cut_idx], x_troughs[cut_idx], eod_heights[cut_idx], eod_widths[cut_idx], samplerate*interp_f, data, interp_f, peak_detection_result


def detect_eod_peaks(main_event_positions, side_event_positions, data,
                     max_width=20, min_width=2, verbose=0):
    """ Generate array of events that might be EODs of a pulse-type fish, using the location of peaks and troughs.

    Parameters
    ----------
    main_event_positions: array of int or float
        Positions of the detected peaks in the data time series.
    side_event_positions: array of int or float
        Positions of the detected troughs in the data time series. 
        The complimentary event to the main events.
    data: array of float
        The data in which the events were detected.
    max_width : int
        Maximum EOD width (in samples).
    min_width : int
        Minimum EOD width (in samples).
    verbose : int
        Verbosity level.

    Returns
    -------
    x_peak: numpy array of ints
        Peak indices.
    x_trough: numpy array of ints
        Trough indices.
    heights: numpy array of floats
        Peak heights (distance between peak and trough amplitude)
    widths: numpy array of ints
        Peak widths (distance between peak and trough indices)
    """

    # determine if there is a peak or through first (evaluates to 1 if there is a peak first):
    mainfirst = int((min(main_event_positions[0], side_event_positions[0]) <
                     side_event_positions[0]))
    # determine if there is a peak or through last (evaluates to 1 if there is a peak last):
    mainlast = int((max(main_event_positions[-1], side_event_positions[-1]) >
                    side_event_positions[-1]))

    x_peak = main_event_positions[mainfirst:len(main_event_positions)-mainlast]
    ind = np.arange(len(x_peak))
    y = data[x_peak]
    
    # find indices of troughs on the right and left side of peaks:
    l_side_ind = ind
    r_side_ind = l_side_ind + 1

    # compute x values, distance to peak and amplitude of right troughs:
    r_side_x = side_event_positions[r_side_ind]
    r_distance = np.abs(r_side_x - x_peak)
    r_side_y = data[r_side_x]

    # compute x values, distance to peak and amplitude of left troughs:
    l_side_x = side_event_positions[l_side_ind]
    l_distance = np.abs(x_peak - l_side_x)
    l_side_y = data[l_side_x]

    # compute slope of lines connecting the peaks to the nearest troughs on the right and left:
    l_slope = np.abs((y-l_side_y)/l_distance)
    r_slope = np.abs((y-r_side_y)/r_distance)

    # determine which trough to assign to the peak by taking either the steepest slope,
    # or, when slopes are similar on both sides (within 25% difference), take the trough 
    # with the maximum height difference to the peak.
    trough_idxs = np.argmax(np.vstack((np.abs(y-l_side_y), np.abs(y-r_side_y))), axis=0)
    slope_idxs = (np.abs(l_slope-r_slope)/(0.5*l_slope+0.5*r_slope) > 0.25)
    trough_idxs[slope_idxs] = np.argmax(np.array(np.vstack(np.array([l_slope[slope_idxs],
                                                                     r_slope[slope_idxs]]))),
                                                                     axis=0)

    # calculated using absolutes in case of for example troughs instead of peaks as main events:
    right_or_left = np.vstack([np.abs(trough_idxs-1), trough_idxs])
    heights = np.sum(np.vstack([np.abs(y-l_side_y), np.abs(y-r_side_y)])*right_or_left, axis=0)
    widths = np.sum(np.vstack([l_distance, r_distance])*right_or_left, axis=0)
    x_trough = np.sum((x_peak + np.vstack([-l_distance, r_distance]))*right_or_left, axis=0)

    keep_events = ((widths>min_width) & (widths<max_width))

    if verbose>0:
        print('Number of peaks after connecting to sidepeaks:          %5i'%(len(x_peak[keep_events])))

    return x_peak[keep_events], x_trough[keep_events], heights[keep_events], widths[keep_events], x_peak, x_trough, heights, widths


@jit(nopython=True)
def discard_connecting_eods(x_peak, x_trough, heights, widths, verbose=0):
    """ If two detected EODs share the same closest trough, keep only the highest peak.

    Parameters
    ----------
    x_peak: list of ints
        Indices of EOD peaks.
    x_trough: list of ints
        Indices of EOD troughs.
    heights: list of floats
        EOD heights.
    widths: list of ints
        EOD widths.
    verbose : int (optional)
        Verbosity level.

    Returns
    -------
    x_peak, x_trough, heights, widths : lists of ints and floats
        EOD location and features of the non-discarded EODs
    """
    keep_idxs = np.ones(len(x_peak))

    for tr in np.unique(x_trough):
        if len(x_trough[x_trough==tr]) > 1:
            slopes = heights[x_trough==tr]/widths[x_trough==tr]

            if (np.max(slopes)!=np.min(slopes)) and \
              (np.abs(np.max(slopes)-np.min(slopes))/(0.5*np.max(slopes)+0.5*np.min(slopes)) > 0.25):
                keep_idxs[np.where(x_trough==tr)[0][np.argmin(heights[x_trough==tr]/widths[x_trough==tr])]] = 0
            else:
                keep_idxs[np.where(x_trough==tr)[0][np.argmin(heights[x_trough==tr])]] = 0
            
    return x_peak[np.where(keep_idxs==1)[0]], x_trough[np.where(keep_idxs==1)[0]], heights[np.where(keep_idxs==1)[0]], widths[np.where(keep_idxs==1)[0]]


def cluster(eod_xp, eod_xt, eod_heights, eod_widths, data, samplerate, interp_f,
            width_factor_shape, width_factor_wave,
            n_gaus_height=10, merge_threshold_height=0.1, n_gaus_width=3,
            merge_threshold_width=0.5, minp=10,
            verbose=0, plot_level=0, save_plots=False, save_path='', ftype='pdf',
            return_data=[]):
    """ Cluster EODs.
    
    First cluster on EOD widths using a Bayesian Gaussian
    Mixture (BGM) model,  then cluster on EOD heights using a
    BGM model. Lastly, cluster on EOD waveform with DBSCAN.
    Clustering on EOD waveform is performed twice, once on
    peak-centered EODs and once on trough-centered EODs.
    Non-pulsetype EOD clusters are deleted, and clusters are
    merged afterwards.

    Parameters
    ----------
    eod_xp : list of ints
        Location of EOD peaks in indices.
    eod_xt: list of ints
        Locations of EOD troughs in indices.
    eod_heights: list of floats
        EOD heights.
    eod_widths: list of ints
        EOD widths in samples.
    data: list of floats
        Recording data.
    samplerate : int or float
        Sample rate of raw data.
    interp_f: float
        Interpolation factor used to obtain input data.
    width_factor_shape : int or float
        Multiplier for snippet extraction width. This factor is multiplied with the width
        between the peak and through of a single EOD.
    width_factor_wave : int or float
        Multiplier for wavefish extraction width.

    n_gaus_height : int (optional)
        Number of gaussians to use for the clustering based on EOD height.
    merge_threshold_height : float (optional)
        Threshold for merging clusters that are similar in height.
    n_gaus_width : int (optional)
        Number of gaussians to use for the clustering based on EOD width.
    merge_threshold_width : float (optional)
        Threshold for merging clusters that are similar in width.
    minp : int (optional)
        Minimum number of points for core clusters (DBSCAN).
   
    verbose : int (optional)
        Verbosity level.
    plot_level : int (optional)
        Similar to verbosity levels, but with plots. 
        Only set to > 0 for debugging purposes.
    save_plots : bool (optional)
        Set to True to save created plots.
    save_path : string (optional)
        Path to save plots to. Only used if save_plots==True.
    ftype : string (optional)
        Filetype to save plot images in.
    return_data : list of strings (optional)
        Keys that specify data to be logged. Keys that can be used to log data
        in this function are: 'all_cluster_steps', 'BGM_width', 'BGM_height',
        'snippet_clusters', 'eod_deletion' (see extract_pulsefish()).

    Returns
    -------
    labels : list of ints
        EOD cluster labels based on height and EOD waveform.
    x_merge : list of ints
        Locations of EODs in clusters.
    saved_data : dictionary
        Key value pairs of logged data. Data to be logged is specified by return_data.
    """
    saved_data = {}

    if plot_level>0 or 'all_cluster_steps' in return_data:
        all_heightlabels = []
        all_shapelabels = []
        all_snippets = []
        all_features = []
        all_heights = []
        all_unique_heightlabels = []

    all_p_clusters = np.ones(len(eod_xp))*-1
    all_t_clusters = np.ones(len(eod_xp))*-1
    artefact_masks_p = np.ones(len(eod_xp), dtype=bool)
    artefact_masks_t = np.ones(len(eod_xp), dtype=bool)

    x_merge = np.ones(len(eod_xp))*-1

    max_label_p = 0   # keep track of the labels so that no labels are overwritten
    max_label_t = 0

    # loop only over height clusters that are bigger than minp
    # first cluster on width
    width_labels, bgm_log_dict = BGM(1000*eod_widths/samplerate, merge_threshold_width,
                                     n_gaus_width, use_log=False, verbose=verbose-1,
                                     plot_level=plot_level-1, xlabel='width [ms]',
                                     save_plot=save_plots, save_path=save_path,
                                     save_name='width', ftype=ftype, return_data=return_data)
    saved_data.update(bgm_log_dict)

    if verbose>0:
        print('Clusters generated based on EOD width:')
        [print('N_{} = {:>4}      h_{} = {:.4f}'.format(l, len(width_labels[width_labels==l]), l, np.mean(eod_widths[width_labels==l]))) for l in np.unique(width_labels)]   


    w_labels, w_counts = unique_counts(width_labels)
    unique_width_labels = w_labels[w_counts>minp]

    for wi, width_label in enumerate(unique_width_labels):

        # select only features in one width cluster at a time
        w_eod_widths = eod_widths[width_labels==width_label]
        w_eod_heights = eod_heights[width_labels==width_label]
        w_eod_xp = eod_xp[width_labels==width_label]
        w_eod_xt = eod_xt[width_labels==width_label]
        wp_clusters = np.ones(len(w_eod_xp))*-1
        wt_clusters = np.ones(len(w_eod_xp))*-1
        wartefact_mask = np.ones(len(w_eod_xp))

        # determine height labels
        raw_p_snippets, p_snippets, p_features, p_bg_ratio = \
          extract_snippet_features(data, w_eod_xp, w_eod_widths, w_eod_heights,
                                   width_factor_shape)
        raw_t_snippets, t_snippets, t_features, t_bg_ratio = \
          extract_snippet_features(data, w_eod_xt, w_eod_widths, w_eod_heights,
                                   width_factor_shape)

        height_labels, bgm_log_dict = \
          BGM(w_eod_heights,
              min(merge_threshold_height, np.median(np.min(np.vstack([p_bg_ratio, t_bg_ratio]), axis=0))),
              n_gaus_height, use_log=True, verbose=verbose-1, plot_level=plot_level-1,
              xlabel='height [a.u.]', save_plot=save_plots, save_path=save_path,
              save_name=('height_'+str(wi)), ftype=ftype, return_data=return_data)
        saved_data.update(bgm_log_dict)

        if verbose>0:
            print('Clusters generated based on EOD height:')
            [print('N_{} = {:>4}      h_{} = {:.4f}'.format(l, len(height_labels[height_labels==l]), l, np.mean(w_eod_heights[height_labels==l]))) for l in np.unique(height_labels)]   

        h_labels, h_counts = unique_counts(height_labels)
        unique_height_labels = h_labels[h_counts>minp]

        if plot_level>0 or 'all_cluster_steps' in return_data:
            all_heightlabels.append(height_labels)
            all_heights.append(w_eod_heights)
            all_unique_heightlabels.append(unique_height_labels)
            shape_labels = []
            cfeatures = []
            csnippets = []

        for hi, height_label in enumerate(unique_height_labels):

            h_eod_widths = w_eod_widths[height_labels==height_label]
            h_eod_heights = w_eod_heights[height_labels==height_label]
            h_eod_xp = w_eod_xp[height_labels==height_label]
            h_eod_xt = w_eod_xt[height_labels==height_label]

            p_clusters = cluster_on_shape(p_features[height_labels==height_label],
                                          p_bg_ratio, minp, verbose=0)            
            t_clusters = cluster_on_shape(t_features[height_labels==height_label],
                                          t_bg_ratio, minp, verbose=0)            
            
            if plot_level>1:
                plot_feature_extraction(raw_p_snippets[height_labels==height_label],
                                        p_snippets[height_labels==height_label],
                                        p_features[height_labels==height_label],
                                        p_clusters, 1/samplerate, 0)
                plt.savefig('%sDBSCAN_peak_w%i_h%i.%s'%(save_path, wi, hi, ftype))
                plot_feature_extraction(raw_t_snippets[height_labels==height_label],
                                        t_snippets[height_labels==height_label],
                                        t_features[height_labels==height_label],
                                        t_clusters, 1/samplerate, 1)
                plt.savefig('%sDBSCAN_trough_w%i_h%i.%s'%(save_path, wi, hi, ftype))

            if 'snippet_clusters' in return_data:
                saved_data['snippet_clusters_%i_%i_peak'%(width_label, height_label)] = {
                    'raw_snippets':raw_p_snippets[height_labels==height_label],
                    'snippets':p_snippets[height_labels==height_label],
                    'features':p_features[height_labels==height_label],
                    'clusters':p_clusters,
                    'samplerate':samplerate}
                saved_data['snippet_clusters_%i_%i_trough'%(width_label, height_label)] = {
                    'raw_snippets':raw_t_snippets[height_labels==height_label],
                    'snippets':t_snippets[height_labels==height_label],
                    'features':t_features[height_labels==height_label],
                    'clusters':t_clusters,
                    'samplerate':samplerate}

            if plot_level>0 or 'all_cluster_steps' in return_data:
                shape_labels.append([p_clusters, t_clusters])
                cfeatures.append([p_features[height_labels==height_label],
                                  t_features[height_labels==height_label]])
                csnippets.append([p_snippets[height_labels==height_label],
                                  t_snippets[height_labels==height_label]])

            p_clusters[p_clusters==-1] = -max_label_p - 1
            wp_clusters[height_labels==height_label] = p_clusters + max_label_p
            max_label_p = max(np.max(wp_clusters), np.max(all_p_clusters)) + 1

            t_clusters[t_clusters==-1] = -max_label_t - 1
            wt_clusters[height_labels==height_label] = t_clusters + max_label_t
            max_label_t = max(np.max(wt_clusters), np.max(all_t_clusters)) + 1

        if verbose > 0:
            if np.max(wp_clusters) == -1:
                print('No EOD peaks in width cluster %i'%width_label)
            elif len(np.unique(wp_clusters[wp_clusters!=-1]))>1:
                print('%i different EOD peaks in width cluster %i'%(len(np.unique(wp_clusters[wp_clusters!=-1])), width_label))
        
            if np.max(wt_clusters) == -1:
                print('No EOD troughs in width cluster %i'%width_label)
            elif len(np.unique(wt_clusters[wt_clusters!=-1]))>1:
                print('%i different EOD troughs in width cluster %i'%(len(np.unique(wt_clusters[wt_clusters!=-1])), width_label))
        
        if plot_level>0 or 'all_cluster_steps' in return_data:
            all_shapelabels.append(shape_labels)
            all_snippets.append(csnippets)
            all_features.append(cfeatures)

        # for each cluster, save fft + label
        # so I end up with features for each label, and the masks.
        # then I can extract e.g. first artefact or wave etc.

        # remove artefacts here, based on the mean snippets ffts.
        artefact_masks_p[width_labels==width_label], sdict = \
          remove_artefacts(p_snippets, wp_clusters, interp_f, samplerate,
                           verbose=verbose-1, return_data=return_data)
        saved_data.update(sdict)
        artefact_masks_t[width_labels==width_label], _ = \
          remove_artefacts(t_snippets, wt_clusters, interp_f, samplerate,
                           verbose=verbose-1, return_data=return_data)

        # update maxlab so that no clusters are overwritten
        all_p_clusters[width_labels==width_label] = wp_clusters
        all_t_clusters[width_labels==width_label] = wt_clusters
    
    # remove all non-reliable clusters
    unreliable_fish_mask_p, saved_data = \
      delete_unreliable_fish(all_p_clusters, eod_widths, eod_xp,
                             verbose=verbose-1, sdict=saved_data)
    unreliable_fish_mask_t, _ = \
      delete_unreliable_fish(all_t_clusters, eod_widths, eod_xt, verbose=verbose-1)
    
    wave_mask_p, sidepeak_mask_p, saved_data = \
      delete_wavefish_and_sidepeaks(data, all_p_clusters, eod_xp, eod_widths, interp_f,
                                    width_factor_wave, verbose=verbose-1, sdict=saved_data)
    wave_mask_t, sidepeak_mask_t, _ = \
      delete_wavefish_and_sidepeaks(data, all_t_clusters, eod_xt, eod_widths, interp_f,
                                    width_factor_wave, verbose=verbose-1)  
        
    og_clusters = [np.copy(all_p_clusters), np.copy(all_t_clusters)]
    og_labels=np.copy(all_p_clusters+all_t_clusters)

    # go through all clusters and masks??
    all_p_clusters[(artefact_masks_p | unreliable_fish_mask_p | wave_mask_p | sidepeak_mask_p)] = -1
    all_t_clusters[(artefact_masks_t | unreliable_fish_mask_t | wave_mask_t | sidepeak_mask_t)] = -1

    # merge here.
    all_clusters, x_merge, mask = merge_clusters(np.copy(all_p_clusters),
                                                 np.copy(all_t_clusters), eod_xp, eod_xt,
                                                 verbose=verbose-1)
    
    if 'all_cluster_steps' in return_data or plot_level>0:
        all_dmasks = []
        all_mmasks = []

        discarding_masks = \
          np.vstack(((artefact_masks_p | unreliable_fish_mask_p | wave_mask_p | sidepeak_mask_p),
                     (artefact_masks_t | unreliable_fish_mask_t | wave_mask_t | sidepeak_mask_t)))
        merge_mask = mask

        # save the masks in the same formats as the snippets
        for wi, (width_label, w_shape_label, heightlabels, unique_height_labels) in enumerate(zip(unique_width_labels, all_shapelabels, all_heightlabels, all_unique_heightlabels)):
            w_dmasks = discarding_masks[:,width_labels==width_label]
            w_mmasks = merge_mask[:,width_labels==width_label]

            wd_2 = []
            wm_2 = []

            for hi, (height_label, h_shape_label) in enumerate(zip(unique_height_labels, w_shape_label)):
               
                h_dmasks = w_dmasks[:,heightlabels==height_label]
                h_mmasks = w_mmasks[:,heightlabels==height_label]

                wd_2.append(h_dmasks)
                wm_2.append(h_mmasks)

            all_dmasks.append(wd_2)
            all_mmasks.append(wm_2)

        if plot_level>0:
            plot_clustering(samplerate, [unique_width_labels, eod_widths, width_labels],
                            [all_unique_heightlabels, all_heights, all_heightlabels],
                            [all_snippets, all_features, all_shapelabels],
                            all_dmasks, all_mmasks)
            if save_plots:
                plt.savefig('%sclustering.%s'%(save_path, ftype))

        if 'all_cluster_steps' in return_data:
            saved_data = {'samplerate': samplerate,
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
                                           (all_p_clusters+all_t_clusters)))}

    if verbose>0:
        print('Clusters generated based on height, width and shape: ')
        [print('N_{} = {:>4}'.format(int(l), len(all_clusters[all_clusters==l]))) for l in np.unique(all_clusters[all_clusters!=-1])]
             
    return all_clusters, x_merge, saved_data


def BGM(x, merge_threshold=0.1, n_gaus=5, max_iter=200, n_init=5,
        use_log=False, verbose=0, plot_level=0, xlabel='x [a.u.]',
        save_plot=False, save_path='', save_name='', ftype='pdf', return_data=[]):
    """ Use a Bayesian Gaussian Mixture Model to cluster one-dimensional data.

    Additional steps are used to merge clusters that are closer than
    `merge_threshold`.  Broad gaussian fits that cover one or more other
    gaussian fits are split by their intersections with the other
    gaussians.

    Parameters
    ----------
    x : 1D numpy array
        Features to compute clustering on. 

    merge_threshold : float (optional)
        Ratio for merging nearby gaussians.
    n_gaus: int (optional)
        Maximum number of gaussians to fit on data.
    max_iter : int (optional)
        Maximum number of iterations for gaussian fit.
    n_init : int (optional)
        Number of initializations for the gaussian fit.
    use_log: boolean (optional)
        Set to True to compute the gaussian fit on the logarithm of x.
        Can improve clustering on features with nonlinear relationships such as peak height.
    verbose : int (optional)
        Verbosity level.
    plot_level : int (optional)
        Similar to verbosity levels, but with plots. 
        Only set to > 0 for debugging purposes.
    xlabel : string (optional)
        Xlabel for displaying BGM plot.
    save_plot : bool (optional)
        Set to True to save created plot.
    save_path : string (optional)
        Path to location where data should be saved. Only used if save_plot==True.
    save_name : string (optional)
        Filename of the saved plot. Usefull as usually multiple BGM models are generated.
    ftype : string (optional)
        Filetype of plot image if save_plots==True.
    return_data : list of strings (optional)
        Keys that specify data to be logged. Keys that can be used to log data
        in this function are: 'BGM_width' and/or 'BGM_height' (see extract_pulsefish()).

    Returns
    -------
    labels : 1D numpy array
        Cluster labels for each sample in x.
    bgm_dict : dictionary
        Key value pairs of logged data. Data to be logged is specified by return_data.
    """

    bgm_dict = {}

    if len(np.unique(x))>n_gaus:
        BGM_model = BayesianGaussianMixture(n_gaus, max_iter=max_iter, n_init=n_init)
        if use_log:
            labels = BGM_model.fit_predict(stats.zscore(np.log(x)).reshape(-1, 1))
        else:
            labels = BGM_model.fit_predict(stats.zscore(x).reshape(-1, 1))
    else:
        return np.zeros(len(x)), bgm_dict
    
    if verbose>0:
        if not BGM_model.converged_:
            print('!!! Gaussian mixture did not converge !!!')
    
    cur_labels = np.unique(labels)
    
    # map labels to be increasing for increasing values for x
    maxlab = len(np.unique(labels))
    aso = np.argsort([np.median(x[labels==l]) for l in cur_labels]) + 100
    for i, a in zip(cur_labels, aso):
        labels[labels==i] = a
    labels = labels - 100
    
    # separate gaussian clusters that can be split by other clusters
    splits = np.sort(np.copy(x))[1:][np.diff(labels[np.argsort(x)])!=0]

    labels[:] = 0
    for i, split in enumerate(splits):
        labels[x>=split] = i+1

    labels_before_merge = np.copy(labels)

    # merge gaussian clusters that are closer than merge_threshold
    labels = merge_gaussians(x, labels, merge_threshold)

    if 'BGM_'+save_name.split('_')[0] in return_data or plot_level>0:

        #sort model attributes by model_means_
        means = [m[0] for m in BGM_model.means_]
        weights = [w for w in BGM_model.weights_]
        variances = [v[0][0] for v in BGM_model.covariances_]
        weights = [w for _, w in sorted(zip(means, weights))]
        variances =  [v for _, v in sorted(zip(means, variances))]
        means =  sorted(means)
        
        if plot_level>0:
            plot_bgm(x, means, variances, weights, use_log, labels_before_merge,
                     labels, xlabel)
            if save_plot:
                plt.savefig('%sBGM_%s.%s'%(save_path, save_name, ftype))
    
        if 'BGM_'+save_name.split('_')[0] in return_data:
            bgm_dict['BGM_'+save_name] = {'x':x,
                                'use_log':use_log,
                                'BGM':[weights, means, variances],
                                'labels':labels_before_merge,
                                'xlab':xlabel}

    return labels, bgm_dict


def merge_gaussians(x, labels, merge_threshold=0.1):
    """ Merge all clusters which have medians which are near. Only works in 1D.

    Parameters
    ----------
    x : 1D array of ints or floats
        Features used for clustering.
    labels : 1D array of ints
        Labels for each sample in x.
    merge_threshold : float (optional)
        Similarity threshold to merge clusters.

    Returns
    -------
    labels : 1D array of ints
        Merged labels for each sample in x.
    """

    # compare all the means of the gaussians. If they are too close, merge them.
    unique_labels = np.unique(labels[labels!=-1])
    x_medians = [np.median(x[labels==l]) for l in unique_labels]

    # fill a dict with the label mappings
    mapping = {}
    for label_1, x_m1 in zip(unique_labels, x_medians):
        for label_2, x_m2 in zip(unique_labels, x_medians):
            if label_1!=label_2:
                if np.abs(np.diff([x_m1, x_m2]))/np.max([x_m1, x_m2]) < merge_threshold:
                    mapping[label_1] = label_2
    # apply mapping
    for map_key, map_value in mapping.items():
        labels[labels==map_key] = map_value

    return labels


def extract_snippet_features(data, eod_x, eod_widths, eod_heights, width_factor, n_pc=5):
    """ Extract snippets from recording data, normalize them, and perform PCA.

    Parameters
    ----------
    data : 1D numpy array of floats
        Recording data.
    eod_x : 1D array of ints
        Locations of EODs in indices.
    eod_widths : 1D array of ints
        EOD widths in samples.
    eod_heights: 1D array of floats
        EOD heights.
    width_factor: int or float
        Multiplier for extracting EOD snippets        

    n_pc : int (optional)
        Number of PCs to use for PCA.

    Returns
    -------
    raw_snippets : 2D numpy array (N, EOD_width)
        Raw extracted EOD snippets.
    snippets : 2D numpy array (N, EOD_width)
        Normalized EOD snippets
    features : 2D numpy array (N,n_pc)
        PC values of EOD snippets
    bg_ratio : 1D numpy array (N)
        Ratio of the background activity slopes compared to EOD height.
    """

    # extract snippets with corresponding width
    width = width_factor*np.median(eod_widths)
    raw_snippets = np.vstack([data[int(x-width):int(x+width)] for x in eod_x])

    # subtract the slope and normalize the snippets
    snippets, bg_ratio = subtract_slope(np.copy(raw_snippets), eod_heights)
    snippets = StandardScaler().fit_transform(snippets.T).T

    # scale so that the absolute integral = 1.
    snippets = (snippets.T/np.sum(np.abs(snippets), axis=1)).T

    # compute features for clustering on waveform
    features = PCA(n_pc).fit(snippets).transform(snippets)

    return raw_snippets, snippets, features, bg_ratio


def cluster_on_shape(features, bg_ratio, minp, percentile=80, max_epsilon=0.01,
                     slope_ratio_factor=4, min_cluster_fraction=0.01, verbose=0):
    """Separate EODs by their shape using DBSCAN.

    Parameters
    ----------
    features : 2D numpy array of floats (N, n_pc)
        PCA features of each EOD in a recording.
    bg_ratio : 1D array of floats
        Ratio of background activity slope the EOD is superimposed on.
    minp : int
        Minimum number of points for core cluster (DBSCAN).

    percentile : int (optional)
        Percentile of KNN distribution, where K=minp, to use as epsilon for DBSCAN.
    max_epsilon : float (optional)
        Maximum epsilon to use for DBSCAN clustering. This is used to avoid adding
        noisy clusters.
    slope_ratio_factor : int or float (optional)
        Influence of the slope-to-EOD ratio on the epsilon parameter.
        A slope_ratio_factor of 4 means that slope-to-EOD ratios >1/4
        start influencing epsilon.
    min_cluster_fraction : float (optional)
        Minimum fraction of all eveluated datapoint that can form a single cluster.
    verbose : int (optional)
        Verbosity level.

    Returns
    -------
    labels : 1D array of ints
        Merged labels for each sample in x.
    """

    # determine clustering threshold from data
    minpc = max(minp, int(len(features)*min_cluster_fraction))  
    knn = np.sort(pairwise_distances(features, features), axis=0)[minpc]
    eps = min(max(1, slope_ratio_factor*np.median(bg_ratio))*max_epsilon,
              np.percentile(knn, percentile))

    if verbose>1:
        print('epsilon = %f'%eps)
        print('Slope to EOD ratio = %f'%np.median(bg_ratio))

    # cluster on EOD shape
    return DBSCAN(eps=eps, min_samples=minpc).fit(features).labels_


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

    left_y = snippets[:,0]
    right_y = snippets[:,-1]

    try:
        slopes = np.linspace(left_y, right_y, snippets.shape[1])
    except ValueError:
        delta = (right_y - left_y)/snippets.shape[1]
        slopes = np.arange(0, snippets.shape[1], dtype=snippets.dtype).reshape((-1,) + (1,) * np.ndim(delta))*delta + left_y
    
    return snippets - slopes.T, np.abs(left_y-right_y)/heights


def remove_artefacts(all_snippets, clusters, int_f, samplerate, artefact_threshold=0.75,
                     verbose=0, return_data=[]):
    """ Create a mask for EOD clusters that result from artefacts, based on power in low frequency spectrum.

    Parameters
    ----------
    all_snippets: 2D array
        EOD snippets. Shape=(nEODs, EOD lenght)
    clusters: list of ints
        EOD cluster labels
    int_f : float
        Interpolation factor used for peak detection.
    samplerate : int or float
        Samplerate of original recording data.

    artefact_threshold : float (optional)
        Threshold that separates artefact from clean pulsefish clusters.
    verbose : int (optional)
        Verbosity level.
    return_data : list of strings (optional)
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

    for cluster in np.sort(np.unique(clusters[clusters>=0])):

        snippets = all_snippets[clusters==cluster]
        mean_eod = np.mean(snippets, axis=0)
        mean_eod = mean_eod - np.mean(mean_eod)
        cut_fft = int(len(np.fft.fft(mean_eod))/2)
        low_frequency_ratio = np.sum(np.abs(np.fft.fft(mean_eod))[:int(cut_fft/(2*int_f))])/np.sum(np.abs(np.fft.fft(mean_eod))[:int(cut_fft)])           
        freqs = np.linspace(0, samplerate, cut_fft)

        if low_frequency_ratio < artefact_threshold:
            mask[clusters==cluster] = True
            
            if verbose>0:
                print('Deleting cluster %i, which has a low frequency ratio of %f'%(cluster,low_frequency_ratio))

        if 'eod_deletion' in return_data:
            adict['vals_'+str(int(cluster))] = [mean_eod, np.abs(np.fft.fft(mean_eod))[:int(cut_fft/int_f)]]
            adict['mask_'+str(int(cluster))] = [any(mask[clusters==cluster])]
    
    return mask, adict


def delete_unreliable_fish(clusters, eod_widths, eod_x, verbose=0, sdict={}):
    """ Create a mask for EOD clusters that are either mixed with noise or other fish, or wavefish.
    
    This is the case when the ration between the EOD width and the ISI is too large.

    Parameters
    ----------
    clusters : list of ints
        Cluster labels.
    eod_widths : list of floats or ints
        EOD widths in samples or seconds.
    eod_x : list of ints or floats
        EOD times in samples or seconds.

    verbose : int (optional)
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
    for cluster in np.unique(np.sort(clusters[clusters>=0])):
        if len(eod_x[cluster==clusters]) < 2:
            mask[clusters==cluster] = True
            if verbose>0:
                print('deleting unreliable cluster %i, number of EOD times %d < 2'%(cluster, len(eod_x[cluster==clusters])))
        elif np.max(np.median(eod_widths[clusters==cluster])/np.diff(eod_x[cluster==clusters])) > 0.5:
            if verbose>0:
                print('deleting unreliable cluster %i, score=%f'%(cluster, np.max(np.median(eod_widths[clusters==cluster])/np.diff(eod_x[cluster==clusters]))))
            mask[clusters==cluster] = True
        if 'vals_'+str(int(cluster)) in sdict:
            sdict['vals_'+str(int(cluster))].append(np.median(eod_widths[clusters==cluster])/np.diff(eod_x[cluster==clusters]))
            sdict['mask_'+str(int(cluster))].append(any(mask[clusters==cluster]))
    return mask, sdict


def delete_wavefish_and_sidepeaks(data, clusters, eod_x, eod_widths, interp_f,
                                  w_factor, max_phases=4, verbose=0, sdict={}):
    """ Create a mask for EODs that are likely from wavefish, or sidepeaks of bigger EODs.

    Parameters
    ----------
    data : list of floats
        Raw recording data.
    clusters : list of ints
        Cluster labels.
    eod_x : list of ints
        Indices of EOD times.
    eod_widths : list of ints
        EOD widths in samples.
    interp_f : float
        Factor used to interpolate original data.
    w_factor : float or int
        Multiplier for EOD analysis width.

    max_phases : int (optional)
        Maximum number of phases for any EOD. 
        If the mean EOD has more phases than this, it is not a pulse EOD.
    verbose : int (optional) 
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

    for i, cluster in enumerate(np.sort(np.unique(clusters[clusters>=0]))):
        if cluster < 0:
            continue
        cutwidth = np.mean(eod_widths[clusters==cluster])*w_factor
        current_x = eod_x[(eod_x>cutwidth) & (eod_x<(len(data)-cutwidth))]
        current_clusters = clusters[(eod_x>cutwidth) & (eod_x<(len(data)-cutwidth))]
        
        snippets = np.vstack([data[int(x-cutwidth):int(x+cutwidth)]
                              for x in current_x[current_clusters==cluster]])
        
        # extract information on main peaks and troughs:
        mean_eod = np.mean(snippets, axis=0)
        mean_eod = mean_eod - np.mean(mean_eod)

        # detect peaks and troughs on data + some maxima/minima at the
        # end, so that the sides are also condidered for peak detection:
        pk, tr = detect_peaks(np.concatenate([[-10*mean_eod[0]], mean_eod, [10*mean_eod[-1]]]),
                              (np.std(mean_eod)))
        pk = pk[(pk>0)&(pk<len(mean_eod))]
        tr = tr[(tr>0)&(tr<len(mean_eod))]

        if len(pk)>0 and len(tr)>0:
            idxs = np.sort(np.concatenate((pk, tr)))
            slopes = np.abs(np.diff(mean_eod[idxs]))
            m_slope = np.argmax(slopes)
            centered = np.min(np.abs(idxs[m_slope:m_slope+2] - int(len(mean_eod)/2)))
            
            # compute all height differences of peaks and troughs within snippets.
            # if they are all similar, it is probably noise or a wavefish.
            idxs = np.sort(np.concatenate((pk, tr)))
            hdiffs = np.diff(mean_eod[idxs])

            if centered>interp_f*2:
                if verbose>0:
                    print('Deleting cluster %i, which is a sidepeak'%cluster)
                mask_sidepeak[clusters==cluster] = True

            w_diff = np.abs(np.diff(np.sort(np.concatenate((pk, tr)))))

            if np.abs(np.diff(idxs[m_slope:m_slope+2])) < np.mean(eod_widths[clusters==cluster])*0.5 or len(pk) + len(tr)>max_phases or np.min(w_diff)>2*cutwidth/w_factor: #or len(hdiffs[np.abs(hdiffs)>0.5*(np.max(mean_eod)-np.min(mean_eod))])>max_phases:
                if verbose>0:
                    print('Deleting cluster %i, which is a wavefish'%cluster)
                mask_wave[clusters==cluster] = True
        if 'vals_'+str(int(cluster)) in sdict:
            sdict['vals_'+str(int(cluster))].append([mean_eod, [pk, tr],
                                                     idxs[m_slope:m_slope+2]])
            sdict['mask_'+str(int(cluster))].append(any(mask_wave[clusters==cluster]))
            sdict['mask_'+str(int(cluster))].append(any(mask_sidepeak[clusters==cluster]))

    return mask_wave, mask_sidepeak, sdict


def merge_clusters(clusters_1, clusters_2, x_1, x_2, verbose=0): 
    """ Merge clusters resulting from two clustering methods.

    This method only works  if clustering is performed on the same EODs
    with the same ordering, where there  is a one to one mapping from
    clusters_1 to clusters_2. 

    Parameters
    ----------
    clusters_1: list of ints
        EOD cluster labels for cluster method 1.
    clusters_2: list of ints
        EOD cluster labels for cluster method 2.
    x_1: list of ints
        Indices of EODs for cluster method 1 (clusters_1).
    x_2: list of ints
        Indices of EODs for cluster method 2 (clusters_2).
    verbose : int (optional)
        Verbosity level.

    Returns
    -------
    clusters : list of ints
        Merged clusters.
    x_merged : list of ints
        Merged cluster indices.
    mask : 2d numpy array of ints (N, 2)
        Mask for clusters that are selected from clusters_1 (mask[:,0]) and
        from clusters_2 (mask[:,1]).
    """
    if verbose > 0:
        print('\nMerge cluster:')

    # these arrays become 1 for each EOD that is chosen from that array
    c1_keep = np.zeros(len(clusters_1))
    c2_keep = np.zeros(len(clusters_2))

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
        c1_labels, c1_size = unique_counts(clusters_1[(clusters_1!=-1) & (c1_keep == 0)])
        c2_labels, c2_size = unique_counts(clusters_2[(clusters_2!=-1) & (c2_keep == 0)])

        # if all clusters are done, break from loop:
        if len(c1_size) == 0 and len(c2_size) == 0:
            break

        # if the biggest cluster is in c_p, keep this one and discard all clusters
        # on the same indices in c_t:
        elif np.argmax([np.max(np.append(c1_size, 0)), np.max(np.append(c2_size, 0))]) == 0:
            
            # remove all the mappings from the other indices
            cluster_mappings, _ = unique_counts(clusters_2[clusters_1==c1_labels[np.argmax(c1_size)]])
            
            clusters_2[np.isin(clusters_2, cluster_mappings)] = -1
            
            c1_keep[clusters_1==c1_labels[np.argmax(c1_size)]] = 1

            remove_clusters.append(cluster_mappings)
            keep_clusters.append(c1_labels[np.argmax(c1_size)])

            if verbose > 0:
                print('Keep cluster %i of group 1, delete clusters %s of group 2'%(c1_labels[np.argmax(c1_size)], str(cluster_mappings[cluster_mappings!=-1] - ovl)))

        # if the biggest cluster is in c_t, keep this one and discard all mappings in c_p
        elif np.argmax([np.max(np.append(c1_size, 0)), np.max(np.append(c2_size, 0))]) == 1:
            
            # remove all the mappings from the other indices
            cluster_mappings, _ = unique_counts(clusters_1[clusters_2==c2_labels[np.argmax(c2_size)]])
            
            clusters_1[np.isin(clusters_1, cluster_mappings)] = -1

            c2_keep[clusters_2==c2_labels[np.argmax(c2_size)]] = 1

            remove_clusters.append(cluster_mappings)
            keep_clusters.append(c2_labels[np.argmax(c2_size)])

            if verbose > 0:
                print('Keep cluster %i of group 2, delete clusters %s of group 1'%(c2_labels[np.argmax(c2_size)] - ovl, str(cluster_mappings[cluster_mappings!=-1])))
    
    # combine results    
    clusters = (clusters_1+1)*c1_keep + (clusters_2+1)*c2_keep - 1
    x_merged = (x_1)*c1_keep + (x_2)*c2_keep

    return clusters, x_merged, np.vstack([c1_keep, c2_keep])


def extract_means(data, eod_x, eod_peak_x, eod_tr_x, eod_widths, clusters, samplerate,
                  w_factor, verbose=0):
    """ Extract mean EODs and EOD timepoints for each EOD cluster.

    Parameters
    ----------
    data: list of floats
        Raw recording data.
    eod_x: list of ints
        Locations of EODs in samples.
    eod_peak_x : list of ints
        Locations of EOD peaks in samples.
    eod_tr_x : list of ints
        Locations of EOD troughs in samples.
    eod_widths: list of ints
        EOD widths in samples.
    clusters: list of ints
        EOD cluster labels
    samplerate: float
        samplerate of recording  
    w_factor : float
        Multiplication factor for window used to extract EOD.
    
    verbose : int (optional)
        Verbosity level.

    Returns
    -------
    mean_eods: list of 2D arrays (3, eod_length)
        The average EOD for each detected fish. First column is time in seconds,
        second column the mean eod, third column the standard error.
    eod_times: list of 1D arrays
        For each detected fish the times of EOD in seconds.
    eod_peak_times: list of 1D arrays
        For each detected fish the times of EOD peaks in seconds.
    eod_trough_times: list of 1D arrays
        For each detected fish the times of EOD troughs in seconds.
    eod_labels: list of ints
        Cluster label for each detected fish.
    """
    mean_eods, eod_times, eod_peak_times, eod_tr_times, eod_heights, cluster_labels = [], [], [], [], [], []

    for cluster in np.unique(clusters):
        if cluster!=-1:
            cutwidth = np.mean(eod_widths[clusters==cluster])*w_factor
            current_x = eod_x[(eod_x>cutwidth) & (eod_x<(len(data)-cutwidth))]
            current_clusters = clusters[(eod_x>cutwidth) & (eod_x<(len(data)-cutwidth))]

            snippets = np.vstack([data[int(x-cutwidth):int(x+cutwidth)] for x in current_x[current_clusters==cluster]])
            mean_eod = np.mean(snippets, axis=0)
            eod_time = np.arange(len(mean_eod))/samplerate - cutwidth/samplerate

            mean_eod = np.vstack([eod_time, mean_eod, np.std(snippets, axis=0)])

            mean_eods.append(mean_eod)
            eod_times.append(eod_x[clusters==cluster]/samplerate)
            eod_heights.append(np.min(mean_eod)-np.max(mean_eod))
            eod_peak_times.append(eod_peak_x[clusters==cluster]/samplerate)
            eod_tr_times.append(eod_tr_x[clusters==cluster]/samplerate)
            cluster_labels.append(cluster)
           
    return [m for _, m in sorted(zip(eod_heights, mean_eods))], [t for _, t in sorted(zip(eod_heights, eod_times))], [pt for _, pt in sorted(zip(eod_heights, eod_peak_times))], [tt for _, tt in sorted(zip(eod_heights, eod_tr_times))], [c for _, c in sorted(zip(eod_heights, cluster_labels))]


def find_clipped_clusters(clusters, mean_eods, eod_times, eod_peaktimes, eod_troughtimes,
                          cluster_labels, width_factor, clip_threshold=0.9, verbose=0):
    """ Detect EODs that are clipped and set all clusterlabels of these clipped EODs to -1.
                          
    Also return the mean EODs and timepoints of these clipped EODs.

    Parameters
    ----------
    clusters: array of ints
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
    w_factor : float or int
        Factor used for mean EOD extraction.
    clip_threshold: float
        Threshold for detecting clipped EODs.
    
    verbose: int
        Verbosity level.

    Returns
    -------
    clusters : array of ints
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
            if verbose>0:
                print('clipped pulsefish')

    clusters[np.isin(clusters, clipped_labels)] = -1

    return clusters, clipped_eods, clipped_times, clipped_peaktimes, clipped_troughtimes


def delete_moving_fish(clusters, eod_t, T, eod_heights, eod_widths, samplerate,
                       min_dt=0.25, stepsize=0.05, sliding_window_factor=2000,
                       verbose=0, plot_level=0, save_plot=False, save_path='',
                       ftype='pdf', return_data=[]):
    """
    Use a sliding window to detect the minimum number of fish detected simultaneously, 
    then delete all other EOD clusters. 

    Do this only for EODs within the same width clusters, as a
    moving fish will preserve its EOD width.

    Parameters
    ----------
    clusters: list of ints
        EOD cluster labels.
    eod_t: list of floats
        Timepoints of the EODs (in seconds).
    T: float
        Length of recording (in seconds).
    eod_heights: list of floats
        EOD amplitudes.
    eod_widths: list of floats
        EOD widths (in seconds).
    samplerate: float
        Recording data samplerate.

    min_dt : float (optional)
        Minimum sliding window size (in seconds).
    stepsize : float (optional)
        Sliding window stepsize (in seconds).
    sliding_window_factor : int or float
        Multiplier for sliding window width,
        where the sliding window width = median(EOD_width)*sliding_window_factor.
    verbose : int (optional)
        Verbosity level.
    plot_level : int (optional)
        Similar to verbosity levels, but with plots. 
        Only set to > 0 for debugging purposes.
    save_plot : bool (optional)
        Set to True to save the plots created by plot_level.
    save_path : string (optional)
        Path to save data to. Only important if you wish to save data (save_data==True).
    ftype : string (optional)
        Define the filetype to save the plots in if save_plots is set to True.
        Options are: 'png', 'jpg', 'svg' ...
    return_data : list of strings (optional)
        Keys that specify data to be logged. The key that can be used to log data
        in this function is 'moving_fish' (see extract_pulsefish()).

    Returns
    -------
    clusters : list of ints
        Cluster labels, where deleted clusters have been set to -1.
    window : list of 2 floats
        Start and end of window selected for deleting moving fish in seconds.
    mf_dict : dictionary
        Key value pairs of logged data. Data to be logged is specified by return_data.
    """
    mf_dict = {}

    if len(np.unique(clusters[clusters!=-1])) == 0:
        return clusters, [0, 1], {}

    all_keep_clusters = []
    width_classes = merge_gaussians(eod_widths, np.copy(clusters), 0.75)   

    all_windows = []
    all_dts = []
    ev_num = 0
    wc_num = len(np.unique(width_classes[clusters>=0]))

    for iw, w in enumerate(np.unique(width_classes[clusters>=0])):

        # initialize variables
        min_clusters = 100
        average_height = 0
        sparse_clusters = 100
        keep_clusters = []

        dt = max(min_dt, np.median(eod_widths[width_classes==w])*sliding_window_factor)
        window_start = 0
        window_end = dt

        wclusters = clusters[width_classes==w]
        weod_t = eod_t[width_classes==w]
        weod_heights = eod_heights[width_classes==w]
        weod_widths = eod_widths[width_classes==w]

        all_dts.append(dt)

        if verbose>0:
            print('sliding window dt = %f'%dt)

        # make W dependent on width??
        ignore_steps = np.zeros(len(np.arange(0, T-dt+stepsize, stepsize)))

        for i, t in enumerate(np.arange(0, T-dt+stepsize, stepsize)):
            current_clusters = wclusters[(weod_t>=t)&(weod_t<t+dt)&(wclusters!=-1)]
            if len(np.unique(current_clusters))==0:
                ignore_steps[i-int(dt/stepsize):i+int(dt/stepsize)] = 1
                if verbose>0:
                    print('No pulsefish in recording at T=%.2f:%.2f'%(t, t+dt))

        
        x = np.arange(0, T-dt+stepsize, stepsize)
        y = np.ones(len(x))

        running_sum = np.ones(len(np.arange(0, T+stepsize, stepsize)))
        ulabs = np.unique(wclusters[wclusters>=0])

        # sliding window
        for j, (t, ignore_step) in enumerate(zip(x, ignore_steps)):
            current_clusters = wclusters[(weod_t>=t)&(weod_t<t+dt)&(wclusters!=-1)]
            current_widths = weod_widths[(weod_t>=t)&(weod_t<t+dt)&(wclusters!=-1)]

            y[j] = len(np.unique(current_clusters))

            if (len(np.unique(current_clusters)) <= min_clusters) and \
              (ignore_step==0) and \
              (len(np.unique(current_clusters) !=1)):

                current_labels = np.isin(wclusters, np.unique(current_clusters))
                current_height = np.mean(weod_heights[current_labels])

                # compute nr of clusters that are too sparse
                clusters_after_deletion = np.unique(remove_sparse_detections(np.copy(clusters[np.isin(clusters, np.unique(current_clusters))]), samplerate*eod_widths[np.isin(clusters, np.unique(current_clusters))], samplerate, T))
                current_sparse_clusters = len(np.unique(current_clusters)) - len(clusters_after_deletion[clusters_after_deletion!=-1])
               
                if current_sparse_clusters <= sparse_clusters and \
                  ((current_sparse_clusters<sparse_clusters) or
                   (current_height > average_height) or
                   (len(np.unique(current_clusters)) < min_clusters)):
                    
                    keep_clusters = np.unique(current_clusters)
                    min_clusters = len(np.unique(current_clusters))
                    average_height = current_height
                    window_end = t+dt
                    sparse_clusters = current_sparse_clusters

        all_keep_clusters.append(keep_clusters)
        all_windows.append(window_end)
        
        if 'moving_fish' in return_data or plot_level>0:
            if 'w' in mf_dict:
                mf_dict['w'].append(np.median(eod_widths[width_classes==w]))
                mf_dict['T'] = T
                mf_dict['dt'].append(dt)
                mf_dict['clusters'].append(wclusters)
                mf_dict['t'].append(weod_t)
                mf_dict['fishcount'].append([x+0.5*(x[1]-x[0]), y])
                mf_dict['ignore_steps'].append(ignore_steps)
            else:
                mf_dict['w'] = [np.median(eod_widths[width_classes==w])]
                mf_dict['T'] = [T]
                mf_dict['dt'] = [dt]
                mf_dict['clusters'] = [wclusters]
                mf_dict['t'] = [weod_t]
                mf_dict['fishcount'] = [[x+0.5*(x[1]-x[0]), y]]
                mf_dict['ignore_steps'] = [ignore_steps]

    if verbose>0:
        print('Estimated nr of pulsefish in recording: %i'%len(all_keep_clusters))

    if plot_level>0:
        plot_moving_fish(mf_dict['w'], mf_dict['dt'], mf_dict['clusters'],mf_dict['t'],
                         mf_dict['fishcount'], T, mf_dict['ignore_steps'])
        if save_plot:
            plt.savefig('%sdelete_moving_fish.%s'%(save_path, ftype))
        # empty dict
        if 'moving_fish' not in return_data:
            mf_dict = {}

    # delete all clusters that are not selected
    clusters[np.invert(np.isin(clusters, np.concatenate(all_keep_clusters)))] = -1

    return clusters, [np.max(all_windows)-np.max(all_dts), np.max(all_windows)], mf_dict


def remove_sparse_detections(clusters, eod_widths, samplerate, T,
                             min_density=0.0005, verbose=0):
    """ Remove all EOD clusters that are too sparse

    Parameters
    ----------
    clusters : list of ints
        Cluster labels.
    eod_widths : list of ints
        Cluster widths in samples.
    samplerate : int or float
        Samplerate.
    T : int or float
        Lenght of recording in seconds.
    min_density : float (optional)
        Minimum density for realistic EOD detections.
    verbose : int (optional)
        Verbosity level.

    Returns
    -------
    clusters : list of ints
        Cluster labels, where sparse clusters have been set to -1.
    """
    for c in np.unique(clusters):
        if c!=-1:

            n = len(clusters[clusters==c])
            w = np.median(eod_widths[clusters==c])/samplerate

            if n*w < T*min_density:
                if verbose>0:
                    print('cluster %i is too sparse'%c)
                clusters[clusters==c] = -1
    return clusters
