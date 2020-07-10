"""
# Extract pulse-type weakly electric fish
Extract all timepoints where pulsefish EODs are present for each separate pulsefish in a recording.

## Main function
- `extract_pulsefish()`: checks for pulse-type fish based on the EOD amplitude and shape.

Author: Liz Weerdmeester
Email: weerdmeester.liz@gmail.com

"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy import stats

try:
    from numba import jit
except ImportError:
    def jit(nopython):
        def decorator_jit(func):
            return func
        return decorator_jit

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from sklearn.mixture import BayesianGaussianMixture
from sklearn.metrics import pairwise_distances

from scipy.interpolate import interp1d

from .eventdetection import detect_peaks
from .pulse_tracker_helper import makeeventlist, discard_connecting_eods

import warnings
def warn(*args,**kwargs):
    pass
warnings.warn=warn

# upgrade numpy functions for backwards compatibility:
if not hasattr(np, 'isin'):
    np.isin = np.in1d

def unique_counts(ar):
    """
    Find the unique elements of an array and their counts, ignoring shape.

    The following code is condensed from numpy version 1.17.0.
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

def extract_pulsefish(data, samplerate, width_factor_shape=3, width_factor_wave=8, width_factor_display=4, verbose=0, plot_level=0, **kwargs):
    """ Extract and cluster pulse fish EODs from recording.
    
    Takes recording data containing an unknown number of pulsefish and extracts the mean 
    EOD and EOD timepoints for each fish present in the recording.
    
    Parameters
    ----------
    data: 1-D array of float
        The data to be analysed.
    samplerate: float
        Sampling rate of the data in Hertz.

    width_factor_shape : int or float
        Width multiplier used for EOD shape analysis.
        EOD snippets are extracted based on width between the 
        peak and trough multiplied by the width factor.
        Defaults to 3.
    width_factor_wave : int or float
        Width multiplier used for wavefish detection.
        Defaults to 8.
    width_factor_display :  int or float
        Width multiplier used for EOD mean extraction and display.
        Defaults to 4.
    verbose : int (optional)
        Verbosity level.
        Defaults to 0.
    plot_level : int (optional)
        Similar to verbosity levels, but with plots. 
        Only set to > 0 for debugging purposes.
        Defaults to 0.
    **peak_extraction_kwargs: (optional) 
        keyword arguments for clustering parameters (see 'extract_eod_times()')
    **cluster_kwargs: (optional) 
        keyword arguments for clustering parameters (see 'cluster()')
        
    Returns
    -------
    mean_eods: list of 2D arrays
        The average EOD for each detected fish. First column is time in seconds,
        second column the mean eod, third column the standard error.
    eod_times: list of 1D arrays
        For each detected fish the times of EOD peaks or troughs in seconds.
        Use these timepoints for EOD averaging.
    eod_peaktimes: list of 1D arrays
        For each detected fish the times of EOD peaks in seconds.
    zoom_window: tuple of floats
        Start and endtime of suggested window for plotting EOD timepoints.
    """
    if verbose > 0:
        print('')
        if verbose > 1:
            print(70*'#')
        print('##### extract_pulsefish', 46*'#')

    mean_eods, eod_times, eod_peaktimes, zoom_window = [], [], [], []
    
    # extract peaks and interpolated data
    x_peak, x_trough, eod_hights, eod_widths, i_samplerate, i_data, interp_f = extract_eod_times(data, samplerate, width_factor=np.max([width_factor_shape,width_factor_display,width_factor_wave]),verbose=verbose-1, plot_level=plot_level-1, **kwargs)
    
    if len(x_peak) > 0:

        # cluster on peaks
        if verbose > 0:
            print('\nCluster on peaks:')
        peak_clusters = cluster(x_peak, eod_hights, eod_widths, i_data, i_samplerate,
                                interp_f, width_factor_shape, width_factor_wave,verbose=verbose, plot_level=plot_level-1, **kwargs) 

        # cluster on troughs
        if verbose > 0:
            print('\nCluster on troughs:')
        trough_clusters = cluster(x_trough, eod_hights, eod_widths, i_data, i_samplerate,
                                 interp_f, width_factor_shape, width_factor_wave, verbose=verbose, plot_level=plot_level-1, **kwargs)

        # merge peak and trough clusters
        clusters, x_merge = merge_clusters(peak_clusters, trough_clusters, x_peak, x_trough, verbose=verbose)


        # extract mean eods and times
        mean_eods, eod_times, eod_peaktimes, eod_troughtimes, cluster_labels = extract_means(i_data, x_merge, x_peak, x_trough, eod_widths,
                                                              clusters, i_samplerate, width_factor_display, verbose=verbose-1)
        
        if plot_level > 1:
            plot_all(data, eod_peaktimes, eod_troughtimes, samplerate, mean_eods)


        # determine clipped clusters (save them, but ignore in other steps)
        clusters, clipped_eods, clipped_times, clipped_peaktimes, clipped_troughtimes = find_clipped_clusters(clusters, mean_eods, eod_times, eod_peaktimes, eod_troughtimes, cluster_labels, width_factor_display, verbose=verbose-1)


        # delete the moving fish
        clusters, zoom_window = delete_moving_fish(clusters, x_merge/i_samplerate, len(data)/samplerate,
                                      eod_hights, eod_widths/i_samplerate, i_samplerate, verbose=verbose-1)

        clusters = remove_sparse_detections(clusters,eod_widths,i_samplerate,len(data)/samplerate,verbose=verbose-1)

        # extract mean eods
        mean_eods, eod_times, eod_peaktimes, eod_troughtimes, cluster_labels = extract_means(i_data, x_merge, x_peak, x_trough, eod_widths,
                                                              clusters, i_samplerate, width_factor_display, verbose=verbose-1)

        if plot_level > 1:
            plot_all(data, eod_peaktimes, eod_troughtimes, samplerate, mean_eods)

        mean_eods.extend(clipped_eods)
        eod_times.extend(clipped_times)
        eod_peaktimes.extend(clipped_peaktimes)
        eod_troughtimes.extend(clipped_troughtimes)

        if plot_level > 0:
            plot_all(data, eod_peaktimes, eod_troughtimes, samplerate, mean_eods)
            plt.show()

    return mean_eods, eod_times, eod_peaktimes, zoom_window


def extract_eod_times(data, samplerate, width_factor,
                      interp_freq=500000, max_peakwidth=0.01, min_peakwidth=None,
                      verbose=0, plot_level=0):
    """ Extract peaks from data which are potentially EODs.

    Parameters
    ----------
    data: 1-D array of float
        The data to be analysed.
    samplerate: int or float
        Sampling rate of the data
    width_factor: int or float
        Factor for extracting EOD shapes. Only EODs are extracted that can fully be analysed with this width.

    interp_freq: int or float (optional)
        Desired resolution in Hz. Data will be interpolated to match this resolution.
        Defaults to 500 kHz
    max_peakwidth: float (optional)
        Maximum width for peak detection in seconds.
        Defaults to 10 ms.
    min_peakwidth: int (optional)
        Minimum width for peak detection in seconds.
        If no minimum width is specified, the minimum width is determined from the recording data resolution 
        (2/samplerate).
        Defaults to None.

    verbose : int (optional)
        Verbosity level.
        Default is zero.
    plot_level : int
        Set to >0 to plot intermediate steps. For debugging purposes only.
        Defaults is zero.

    Returns
    -------
    x_peak: array of ints
        Indices of EOD peaks in data.
    x_trough: array of ints
        Indices of EOD troughs in data. There is one x_trough for each x_peak.
    eod_hights: array of floats
        EOD hights for each x_peak.
    eod_widths: array of ints
        EOD widths for each x_peak (in samples).
    samplerate: int or float
        New samplerate (after interpolation).
    data: 
        Interpolated data.
    interpolation_factor: 
        Factor used for interpolation.
    """
    
    # standard deviation of data in small snippets:
    threshold = detect_threshold(np.array(data),samplerate)

    # try to do this on all channels at once to speed things up.
    try:
        interp_f = int(interp_freq/samplerate)
        f = interp1d(range(len(data)),data,kind='quadratic')
        data = f(np.arange(0,len(data)-1,1/interp_f))
    except MemoryError:
        interp_f = 1

    orig_x_peaks, orig_x_troughs = detect_peaks(data, threshold)
    orig_x_peaks = orig_x_peaks.astype('int')
    orig_x_troughs = orig_x_troughs.astype('int')

    if len(orig_x_peaks)==0 or len(orig_x_peaks)>samplerate:
        if verbose>0:
            print('No peaks detected.')
        return [], [], [], [], samplerate*interp_f,data, interp_f
    else:
        if min_peakwidth == None:
            min_peakwidth = interp_f*2
        else:
            min_peakwidth = min_peakwidth*interp_freq

        peaks, troughs, hights, widths = makeeventlist(orig_x_peaks, orig_x_troughs, data, max_peakwidth*interp_freq, min_peakwidth, verbose=verbose-1)
        x_peaks, x_troughs, eod_hights, eod_widths = discard_connecting_eods(peaks, troughs, hights, widths, verbose=verbose-1)
        
        if plot_level>0:
            plt.figure()
            plt.plot(data)
            plt.plot(orig_x_peaks,data[orig_x_peaks],'o',ms=10)
            plt.plot(orig_x_troughs,data[orig_x_troughs],'o',ms=10)
            plt.plot(peaks,data[peaks.astype('int')],'o')
            plt.plot(troughs,data[troughs.astype('int')],'o')
            plt.plot(x_peaks,data[x_peaks.astype('int')],'x',ms=10)
            plt.plot(x_troughs,data[x_troughs.astype('int')],'x',ms=10)        

        # only take those where the maximum cutwidth does not casue issues
        # so if the width_factor times the width + x is more than length.
        cut_idx = ((x_peaks + eod_widths*width_factor < len(data)) & (x_troughs + eod_widths*width_factor < len(data)) & (x_peaks - eod_widths*width_factor > 0) & (x_troughs - eod_widths*width_factor > 0))
        
        if verbose>0:
            print('Remaining peaks after EOD extraction                    %5i\n'%(len(cut_idx)))
            if verbose>1:
                print('Remaining peaks after deletion due to cutwidth          %5i\n'%(len(cut_idx)))

        return x_peaks[cut_idx], x_troughs[cut_idx], eod_hights[cut_idx], eod_widths[cut_idx], samplerate*interp_f, data, interp_f

    
@jit(nopython=True)
def detect_threshold(data, samplerate, win_size = 0.0005, n_stds = 1000, threshold_factor=6.0):
    """ Determine a suitable threshold for peak detection.
        The threshold is based on the median standard deviation of smaller sections of the recording data. 
        This method ensures a proper peak detection threshold for pulse-type EODs that are superimposed on slower waves.

        Parameters:
        -----------
        data: 1-D array of float
            The data to be analysed.
        samplerate: int or float
            Sampling rate of the data

        win_size: float (optional)
            Window size for determining peak detection threshold in seconds.
            Defaults to 0.5 ms.
        n_stds: int (optional)
            Number of standard deviations to make on data for determining peak detection threshold.
            Defaults to 1000.
        threshold_factor: float (optional)
            Multiplication factor for peak detection threshold.
            Defaults to 6.

        Returns
        -------
        threshold: float
            Suitable peak detection threshold for recording data.

    """
    win_size_indices = int(win_size * samplerate)
    if win_size_indices < 10:
        win_size_indices = 10
    step = len(data)//n_stds
    if step < win_size_indices//2:
        step = win_size_indices//2
    stds = [np.std(data[i:i+win_size_indices])
            for i in range(0, len(data)-win_size_indices, step)]

    return np.median(np.array(stds))*threshold_factor


def BGM(x, merge_threshold=0.1, n_gaus=5, max_iter=200, n_init=5, use_log=False,
        verbose=0, plot_level=0):
    """ Use a Bayesian Gaussian Mixture Model to cluster one-dimensional data. 
        Additional steps are used to merge clusters that are closer than merge_threshold.
        Broad gaussian fits that cover one or more other gaussian fits are split by their intersections with the other gaussians.

        Parameters
        ----------
        x : 1D numpy array
            Features to compute clustering on. 

        merge_threshold : float (optional)
            Ratio for merging nearby gaussians.
            Defaults to 0.1.
        n_gaus: int (optional)
            Maximum number of gaussians to fit on data.
            Defaults to 5.
        max_iter : int (optional)
            Maximum number of iterations for gaussian fit.
            Defaults to 200.
        n_init : int (optional)
            Number of initializations for the gaussian fit.
            Defaults to 5.
        use_log: boolean
            Set to True to compute the gaussian fit on the logarithm of x.
            Can improve clustering on features with nonlinear relationships such as peak hight.
            Defaults to False.
        verbose : int (optional)
            Verbosity level.
            Defaults to 0.
        plot_level : int (optional)
            Similar to verbosity levels, but with plots. 
            Only set to > 0 for debugging purposes.
            Defaults to 0.

        Returns
        -------
        labels : 1D numpy array
            Cluster labels for each sample in x.

    """

    if len(np.unique(x))>n_gaus:
        BGM_model = BayesianGaussianMixture(n_gaus, max_iter=max_iter, n_init=n_init)
        if use_log:
            labels = BGM_model.fit_predict(stats.zscore(np.log(x)).reshape(-1,1))
        else:
            labels = BGM_model.fit_predict(stats.zscore(x).reshape(-1,1))
    else:
        return np.zeros(len(x))
    
    if verbose>0:
        if not BGM_model.converged_:
            print('!!! Gaussian mixture did not converge !!!')

    # plot gaussian mixtures
    if plot_level>0:
        plt.figure()
        plt.subplot(1,4,1)
        plt.hist(x)
        plt.subplot(1,4,2)
        for l in np.unique(labels):
            plt.hist(x[labels==l],alpha=0.7)


    # merge gaussian clusters that are closer than merge_threshold
    labels = merge_gaussians(x,labels,merge_threshold)

    # plot labels after merging close clusters
    if plot_level>0:
        plt.subplot(1,4,3)
        for l in np.unique(labels):
            plt.hist(x[labels==l],alpha=0.7)
    
    # separate gaussian clusters that can be split by other clusters
    unique_labels = np.unique(labels)
    thresholds = np.vstack([[np.max(x[labels==l]),np.min(x[labels==l])] for l in unique_labels])
    all_thresholds = thresholds.flatten()

    for thresh, label in zip(thresholds,unique_labels):

        if len(all_thresholds[(all_thresholds>np.min(thresh)) & (all_thresholds<np.max(thresh))])>0:

            c_label = labels[labels==label]
            c_x = x[labels==label]
            maxlab = np.max(labels)+1
            for split in np.sort(all_thresholds[(all_thresholds>np.min(thresh)) & (all_thresholds<np.max(thresh))]):
                c_label[c_x>split] = maxlab
                maxlab = maxlab+1
            labels[labels==label] = c_label
    
    # plot labels after separating clusters
    if plot_level>0:
        plt.subplot(2,4,8)
        for l in np.unique(labels):
            plt.hist(x[labels==l],alpha=0.7)

    return labels


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
            Defaults to 0.1

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
    for label_1,x_m1 in zip(unique_labels,x_medians):
        for label_2,x_m2 in zip(unique_labels,x_medians):
            if label_1!=label_2:
                if np.abs(np.diff([x_m1,x_m2]))/np.max([x_m1,x_m2]) < merge_threshold:
                    mapping[label_1] = label_2
    # apply mapping
    for map_key,map_value in mapping.items():
        labels[labels==map_key] = map_value

    return labels


def cluster(eod_x, eod_hights, eod_widths, data, samplerate, interp_f,
            width_factor_shape, width_factor_wave, 
            n_gaus_hight=10, merge_threshold_hight=0.1, n_gaus_width=3,
            merge_threshold_width=0.5, 
            n_pc=5, minp=10, percentile=80, max_epsilon=0.01, slope_ratio_factor=4, 
            min_cluster_fraction=0.01, verbose=0, plot_level=0):
    """ Cluster EODs.

    First cluster on EOD hights using a Bayesian Gaussian Mixture model, 
    then cluster on EOD waveform with DBSCAN. Clustering on EOD waveform is performed
    twice, once on scaled EODs and once on non-scaled EODs. Clusters are merged afterwards.

    Parameters
    ----------
    eod_x: list of ints
        Locations of EODs in samples.
    eod_hights: list of floats
        EOD hights.
    eod_widths: list of ints
        EOD widths in samples.
    data: list of floats
        Raw recording data.
    samplerate : int or float
        Sample rate of raw data.
    interp_f: float
        Interpolation factor used to obtain input data.
    width_factor_shape : int or float
        Multiplier for snippet extraction width. This factor is multiplied with the width
        between the peak and through of a single EOD.
    width_factor_wave : int or float
        Multiplier for wavefish extraction width.

    n_gaus_hight (optional) : int
        Number of gaussians to use for the clustering based on EOD hight.
        Defaults to 10.
    merge_threshold_hight (optional) :
        Threshold for merging clusters that are similar in hight.
        Defaults to 0.1.
    n_gaus_width (optional) :
        Number of gaussians to use for the clustering based on EOD width.
        Defaults to 3.
    merge_threshold_width (optional) :
        Threshold for merging clusters that are similar in width.
        Defaults to 0.5.
    n_pc (optional): int
        Number of PCs to use for PCA.
        Defaults to 5.
    minp (optional) : int
        Minimum number of points for core cluster (DBSCAN).
        Defaults to 10.
    percentile (optional): int
        Percentile of KNN distribution, where K=minp, to use as epsilon for DBSCAN.
        Defaults to 75.
    max_epsilon (optional): float
        Maximum epsilon to use for DBSCAN clustering. This is used to avoid adding noisy clusters
        Defaults to 0.01.
    slope_ratio_factor (optional): int or float
        Influence of the slope-to-EOD ratio on the epsilon parameter.
        A slope_ratio_factor of 4 means that slope-to-EOD ratios >1/4 start influencing epsilon.
        Defaults to 4.
    min_cluster_fraction (optional): float
        Minimum fraction of all eveluated datapoint that can form a single cluster.
        Defaults to 1%.
   
    verbose : int (optional)
        Verbosity level.
        Defaults to 0.
    plot_level : int (optional)
        Similar to verbosity levels, but with plots. 
        Only set to > 0 for debugging purposes.
        Defaults to 0.

    Returns
    -------
    labels : list of ints
        EOD cluster labels based on hight and EOD waveform.
    """

    all_clusters = np.ones(len(eod_x))*-1
    width_labels = BGM(eod_widths,merge_threshold_width,n_gaus_width,verbose=verbose-1,plot_level=plot_level-1)

    if verbose > 0:
        wls = np.unique(width_labels)
        print('Clusters generated based on EOD width: %i' % len(wls))
        if verbose > 1:
            [print('  %2i: N=%4i   w=%7.1f samples' % (l, len(width_labels[width_labels==l]), np.mean(eod_widths[width_labels==l]))) for l in wls]   

    max_label = 0   # keep track of the labels so that no labels are overwritten

    # loop only over hight clusters that are bigger than minp
    w_labels, w_counts = unique_counts(width_labels)
    unique_width_labels = w_labels[w_counts>minp]

    for wi, width_label in enumerate(unique_width_labels):

        # select only features in one width cluster at a time
        w_eod_widths = eod_widths[width_labels==width_label]
        w_eod_hights = eod_hights[width_labels==width_label]
        w_eod_x = eod_x[width_labels==width_label]
        w_clusters = np.ones(len(w_eod_x))*-1

        # extract snippets with corresponding width
        width = width_factor_shape*np.median(w_eod_widths)
        snippets = np.vstack([data[int(x-width):int(x+width)] for x in w_eod_x])

        # subtract the slope and normalize the snippets
        snippets, slope_ratio = subtract_slope(snippets)
        snippets = StandardScaler().fit_transform(snippets.T).T

        # scale so that the absolute integral = 1.
        snippets = (snippets.T/np.sum(np.abs(snippets),axis=1)).T

        # compute features for clustering on waveform
        features = PCA(n_pc).fit(snippets).transform(snippets)

        # determine hight labels
        hight_labels = BGM(w_eod_hights,min(merge_threshold_hight,np.median(slope_ratio/w_eod_hights)),n_gaus_hight,use_log=True,verbose=verbose-1,plot_level=plot_level-1)
        
        if verbose > 0:
            hls = np.unique(hight_labels)
            print('Clusters generated in width cluster %i based on EOD hight: %i' % (width_label, len(hls)))
            if verbose > 1:
                [print('  %2i: N=%4i   h=%8.4f' % (l, len(hight_labels[hight_labels==l]), np.mean(w_eod_hights[hight_labels==l]))) for l in hls]   

        h_labels, h_counts = unique_counts(hight_labels)
        unique_hight_labels = h_labels[h_counts>minp]

        if plot_level>0:
            plt.figure()

        for hi,hight_label in enumerate(unique_hight_labels):

            h_eod_widths = w_eod_widths[hight_labels==hight_label]
            h_eod_hights = w_eod_hights[hight_labels==hight_label]
            h_eod_x = w_eod_x[hight_labels==hight_label]

            c_snippets = snippets[hight_labels==hight_label]
            c_features = features[hight_labels==hight_label]

            # determine clustering threshold from data
            minpc = max(minp,int(len(c_features)*min_cluster_fraction))  
            knn = np.sort(pairwise_distances(c_features,c_features),axis=0)[minpc]
            eps = min(max(1,slope_ratio_factor*np.median(slope_ratio/w_eod_hights))*max_epsilon,np.percentile(knn,percentile))

            if verbose > 2:
                print('  epsilon = %f' % eps)
                print('  slope to EOD ratio = %f' % np.median(slope_ratio/w_eod_hights))

            # cluster on EOD shape
            h_clusters = DBSCAN(eps=eps, min_samples=minpc).fit(c_features).labels_
            
            # plot clusters
            if plot_level>0:
                cols = ['b','r','g','y','m','c']

                plt.subplot(2,len(unique_hight_labels),hi+1)

                for j,c in enumerate(np.unique(h_clusters)):
                    if c==-1:
                        plt.scatter(c_features[h_clusters==c,0],c_features[h_clusters==c,1],alpha=0.1,c='k',label='-1')
                    else:
                        plt.scatter(c_features[h_clusters==c,0],c_features[h_clusters==c,1],alpha=0.1,c=cols[j%len(cols)],label=c+max_label)
                        plt.title('h = %.3f, w=%i'%(np.mean(h_eod_hights[h_clusters==c]),np.mean(h_eod_widths[h_clusters==c])))

                plt.subplot(2,len(unique_hight_labels),len(unique_hight_labels)+hi+1)

                for j,c in enumerate(np.unique(h_clusters)):
                    if c==-1:
                        plt.plot(c_snippets[h_clusters==c].T,alpha=0.1,c='k',label='-1')
                    else:
                        plt.plot(c_snippets[h_clusters==c].T,alpha=0.1,c=cols[j%len(cols)],label=c+max_label)
                        plt.title('h = %.3f, w=%i'%(np.mean(h_eod_hights[h_clusters==c]),np.mean(h_eod_widths[h_clusters==c])))

            h_clusters[h_clusters==-1] = -max_label - 1
            w_clusters[hight_labels==hight_label] = h_clusters + max_label
            max_label = max(np.max(w_clusters),np.max(all_clusters)) + 1

            if plot_level>0:
                plt.legend()

        if verbose > 1:
            if np.max(w_clusters) == -1:
                print('  No EODs in width cluster %2i' % width_label)
            elif len(np.unique(w_clusters[w_clusters!=-1])) > 1:
                print('  %i different EODs in width cluster %i' % (len(np.unique(w_clusters[w_clusters!=-1])),width_label))
        
        # remove artefacts here, based on the mean snippets ffts.
        w_clusters = remove_artefacts(snippets, w_clusters, interp_f, verbose=verbose-1)

        # update maxlab so that no clusters are overwritten
        all_clusters[width_labels==width_label] = w_clusters

    # remove all non-reliable clusters
    all_clusters = delete_unreliable_fish(all_clusters,eod_widths,eod_x,verbose=verbose-1)
    all_clusters = delete_wavefish_and_sidepeaks(data,all_clusters,eod_x,eod_widths,interp_f,width_factor_wave,verbose=verbose-1)

    if verbose > 0:
        ls = np.unique(all_clusters[all_clusters!=-1])
        print('Clusters generated based on hight, width and shape: %2i' % len(ls))
        if verbose > 1:
            [print('  %2i: N=%4i' % (int(l),len(all_clusters[all_clusters==l]))) for l in ls]

    return all_clusters


def subtract_slope(snippets):
    """ Subtract underlying slope from all EOD snippets.

    Parameters
    ----------
        snippets: 2-D numpy array
            All EODs in a recorded stacked as snippets. 
            Shape = (number of EODs, EOD width)
    Returns
    -------
        snippets: 2-D numpy array
            EOD snippets with underlying slope subtracted.
        slope_ratio : 1-D numpy array
            Ratio between the hight of the subtracted slope and the estimated EOD hight.
    """

    left_y = snippets[:,0]
    right_y = snippets[:,-1]

    try:
        slopes = np.linspace(left_y, right_y, snippets.shape[1])
    except ValueError:
        delta = (right_y - left_y)/snippets.shape[1]
        slopes = np.arange(0, snippets.shape[1], dtype=snippets.dtype).reshape((-1,) + (1,) * np.ndim(delta))*delta + left_y
    
    return snippets - slopes.T, np.abs(left_y-right_y)


def remove_artefacts(all_snippets, clusters, int_f, artefact_threshold=0.75,
                     cutoff_f=10000, verbose=0):
    """ Remove EOD clusters that result from artefacts based on power in low frequency spectrum.

    Parameters
    ----------
        all_snippets: 2D array
            EOD snippets. Shape=(nEODs, EOD lenght)
        clusters: list of ints
            EOD cluster labels
        int_f : float
            Interpolation factor used for peak detection.

        artefact_threshold (optional): float
            Threshold that separates artefact from clean pulsefish clusters.
            Defaults to 0.75.
        curoff_f (optional) : int or float
            Cut-off frequency for low frequency estimation.
            Defaults to 10kHz.
        verbose (optional): int
            Verbosity level.

    Returns
    -------
        clusters: list of ints
            Cluster labels, where noisy cluster and clusters with artefacts have been set to -1.
    """

    for cluster in np.unique(clusters):
        if cluster != -1:
            snippets = all_snippets[clusters==cluster]
            mean_eod = np.mean(snippets, axis=0)
            cut_fft = int(len(np.fft.fft(mean_eod))/2)
            low_frequency_ratio = np.sum(np.abs(np.fft.fft(mean_eod))[1:int(cut_fft/(2*int_f))])/np.sum(np.abs(np.fft.fft(mean_eod))[1:int(cut_fft)])   

            if low_frequency_ratio < artefact_threshold:
                clusters[clusters==cluster] = -1
                if verbose > 0:
                    print('  Deleting cluster %2i, which has a low frequency ratio  of %4.2f (min %4.2f)' % (cluster, low_frequency_ratio, artefact_threshold))
    return clusters


def delete_unreliable_fish(clusters, eod_widths, eod_x, verbose):
    """ Delete EOD clusters that are either mixed with noise or other fish, or wavefish.
        This is the case when the ration between the EODwidth and the ISI is too large.

        Parameters
        ----------
        clusters : list of ints
            Cluster labels.
        eod_widths : list of floats or ints
            EOD widths in samples or seconds.
        eod_x : list of ints or floats
            EOD times in samples or seconds.

        verbose (optional): int   
            Verbosity level.
            Defaults to 0.    

        Returns
        -------
        clusters : list of ints
            Cluster labels where unreliable clusters have been set to -1.

    """
    for cluster in np.unique(clusters[clusters!=-1]):
        unreliability = np.max(np.median(eod_widths[clusters==cluster])/np.diff(eod_x[cluster==clusters]))
        if unreliability > 0.5:
            if verbose > 0:
                print('Deleting cluster %2i, which has an unreliability score of %4.2f (max 0.5)' % (cluster, unreliability))
            clusters[clusters==cluster] = -1
    return clusters


def delete_wavefish_and_sidepeaks(data, clusters, eod_x, eod_widths, interp_f, w_factor,
                                  detailed_threshold_factor=0.01, max_phases=4, verbose=0):
    """Delete EODs that are likely from wavefish, or sidepeaks of bigger EODs.
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

        detailed_threshold_factor (optional): float
            Multiplier for peak and trough detection for determining EOD center alignment and slopes.
            Defaults to 0.01.
        max_phases : int
            Maximum number of phases for any EOD. 
            If the mean EOD has more phases than this, it is not a pulse EOD.
            Defaults to 4.
        verbose (optional): int   
            Verbosity level.
            Defaults to 0. 

        Returns
        -------
        clusters : list of ints
            Cluster labels, where wavefish and sidepeaks have been set to -1.

    """

    for cluster in np.unique(clusters):
        if cluster < 0:
            continue
        cutwidth = np.mean(eod_widths[clusters==cluster])*w_factor
        current_x = eod_x[(eod_x>cutwidth) & (eod_x<(len(data)-cutwidth))]
        current_clusters = clusters[(eod_x>cutwidth) & (eod_x<(len(data)-cutwidth))]
        
        snippets = np.vstack([data[int(x-cutwidth):int(x+cutwidth)] for x in current_x[current_clusters==cluster]])
        
        # extract information on main peaks and troughs
        mean_eod = np.mean(snippets, axis=0)
        mean_eod = mean_eod - np.mean(mean_eod)

        # detect peaks and troughs on data + some maxima/minima at the end, so that the sides are also condidered for peak detection.
        pk, tr = detect_peaks(np.concatenate([[-10*mean_eod[0]],mean_eod,[10*mean_eod[-1]]]), (np.std(mean_eod)))
        pk = pk[(pk>0)&(pk<len(mean_eod))]
        tr = tr[(tr>0)&(tr<len(mean_eod))]

        # determine if EOD is centered based on peak detection with lower threshold.
        small_peaks, small_troughs = detect_peaks(mean_eod, detailed_threshold_factor*(np.max(mean_eod)-np.min(mean_eod)))
        idxs = np.sort(np.concatenate((small_troughs, small_peaks)))
        slopes = np.abs(np.diff(mean_eod[idxs]))
        m_slope = np.argmax(slopes)
        centered = np.min(np.abs(idxs[m_slope:m_slope+2] - int(len(mean_eod)/2)))
        
        # compute all hight differences of peaks and troughs within snippets.
        # if they are all similar, it is probably noise or a wavefish.
        idxs = np.sort(np.concatenate((small_peaks, small_troughs)))
        hdiffs = np.diff(mean_eod[idxs])

        if centered > interp_f*2:
            if verbose>0:
                print('Deleting cluster %2i, which is a sidepeak' % cluster)
            clusters[clusters==cluster] = -1

        elif len(pk)>0 and len(tr)>0:
            w_diff = np.abs(np.diff(np.sort(np.concatenate((pk,tr)))))
            n_phases = np.abs(np.diff(idxs[m_slope:m_slope+2])) < np.mean(eod_widths[clusters==cluster])*0.5 or len(pk) + len(tr)>max_phases or np.min(w_diff)>2*cutwidth/w_factor or len(hdiffs[np.abs(hdiffs)>0.5*(np.max(mean_eod)-np.min(mean_eod))])
            if n_phases > max_phases:
                if verbose>0:
                    print('Deleting cluster %2i, which is a wavefish with %2d phases (max=%d)' % (cluster, n_phases, max_phases))
                clusters[clusters==cluster] = -1

    return clusters


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
        verbose (optional): int
            Verbosity level.

    Returns
    -------
        clusters : list of ints
            Merged clusters.
        x_merged : list of ints
            Merged cluster indices.
    """
    if verbose > 0:
        print('\nMerge cluster:')

    # these arrays become 1 for each EOD that is chosen from that array
    c1_keep = np.zeros(len(clusters_1))
    c2_keep = np.zeros(len(clusters_2))

    # add n to one of the cluster lists to avoid overlap
    ovl = np.max(clusters_1) + 1
    clusters_2[clusters_2!=-1] = clusters_2[clusters_2!=-1] + ovl
    
    # loop untill done
    while True:

        # compute unique clusters and cluster sizes
        # of cluster that have not been iterated over
        c1_labels, c1_size = unique_counts(clusters_1[(clusters_1!=-1) & (c1_keep == 0)])
        c2_labels, c2_size = unique_counts(clusters_2[(clusters_2!=-1) & (c2_keep == 0)])

        # if all clusters are done, break from loop
        if len(c1_size) == 0 and len(c2_size) == 0:
            break

        # if the biggest cluster is in c_p, keep this one and discard all clusters on the same indices in c_t
        elif np.argmax([np.max(np.append(c1_size,0)), np.max(np.append(c2_size,0))]) == 0:
            
            # remove all the mappings from the other indices
            cluster_mappings, _ = unique_counts(clusters_2[clusters_1==c1_labels[np.argmax(c1_size)]])
            
            clusters_2[np.isin(clusters_2, cluster_mappings)] = -1
            
            c1_keep[clusters_1==c1_labels[np.argmax(c1_size)]] = 1

            if verbose > 0:
                print('Keep cluster %i of group 1, delete clusters %s of group 2'%(c1_labels[np.argmax(c1_size)],str(cluster_mappings[cluster_mappings!=-1] - ovl)))

        # if the biggest cluster is in c_t, keep this one and discard all mappings in c_p
        elif np.argmax([np.max(np.append(c1_size, 0)), np.max(np.append(c2_size, 0))]) == 1:
            
            # remove all the mappings from the other indices
            cluster_mappings, _ = unique_counts(clusters_1[clusters_2==c2_labels[np.argmax(c2_size)]])
            
            clusters_1[np.isin(clusters_1, cluster_mappings)] = -1

            c2_keep[clusters_2==c2_labels[np.argmax(c2_size)]] = 1

            if verbose > 0:
                print('Keep cluster %i of group 2, delete clusters %s of group 1'%(c2_labels[np.argmax(c2_size)] - ovl, str(cluster_mappings[cluster_mappings!=-1])))

    # combine results    
    clusters = (clusters_1+1)*c1_keep + (clusters_2+1)*c2_keep - 1
    x_merged = (x_1)*c1_keep + (x_2)*c2_keep

    return clusters, x_merged


def extract_means(data, eod_x, eod_peak_x, eod_tr_x, eod_widths, clusters, samplerate,
                  w_factor, verbose=0):
    """ Extract mean EODs, EOD timepoints and unreliability score for each EOD cluster.

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
        
        verbose (optional): int   
            Verbosity level.
            Defaults to 0.           

    Returns
    -------
        mean_eods: list of 2D arrays
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

    mean_eods, eod_times, eod_peak_times, eod_tr_times, eod_hights, cluster_labels = [], [], [], [], [], []
    unreliability = []

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
            eod_hights.append(np.min(mean_eod)-np.max(mean_eod))
            eod_peak_times.append(eod_peak_x[clusters==cluster]/samplerate)
            eod_tr_times.append(eod_tr_x[clusters==cluster]/samplerate)
            cluster_labels.append(cluster)
           
    return [m for _,m in sorted(zip(eod_hights,mean_eods))], [t for _,t in sorted(zip(eod_hights,eod_times))], [pt for _,pt in sorted(zip(eod_hights,eod_peak_times))], [tt for _,tt in sorted(zip(eod_hights,eod_tr_times))], [c for _,c in sorted(zip(eod_hights,cluster_labels))]


def find_clipped_clusters(clusters, mean_eods, eod_times, eod_peaktimes, eod_troughtimes,
                          cluster_labels, width_factor, clip_threshold=0.9, verbose=0):
    """ Detect EODs that are clipped and set all clusterlabels of these clipped EODs to -1. 
    Also return the mean EODs and timepoints of these clipped EODs.

    Parameters:
    -----------
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
        Defaults to zero.

    Returns:
    --------
    clusters: array of ints
        Cluster labels for each EOD in the recording, where clipped EODs have been set to -1.
    clipped_eods: list of numpy arrays
        Mean EOD waveforms for each clipped EOD cluster.
    clipped_times: list of numpy arrays
        EOD timepoints for each clipped EOD cluster.
    clipped_peaktimes
        EOD peaktimes for each clipped EOD cluster.
    clipped_troughtimes
        EOD troughtimes for each clipped EOD cluster.

    """
    clipped_eods, clipped_times, clipped_peaktimes, clipped_troughtimes, clipped_labels = [], [], [], [], []

    for mean_eod, eod_time, eod_peaktime, eod_troughtime,label in zip(mean_eods, eod_times, eod_peaktimes, eod_troughtimes,cluster_labels):
        
        if (np.count_nonzero(mean_eod[1]>clip_threshold) > len(mean_eod[1])/(width_factor*2)) or (np.count_nonzero(mean_eod[1] < -clip_threshold) > len(mean_eod[1])/(width_factor*2)):
            clipped_eods.append(mean_eod)
            clipped_times.append(eod_time)
            clipped_peaktimes.append(eod_peaktime)
            clipped_troughtimes.append(eod_troughtime)
            clipped_labels.append(label)
            if verbose>0:
                print('clipped pulsefish')


    clusters[np.isin(clusters,clipped_labels)] = -1

    return clusters, clipped_eods, clipped_times, clipped_peaktimes, clipped_troughtimes


def delete_moving_fish(clusters, eod_t, T, eod_hights, eod_widths, samplerate,
                       dt=1, stepsize=0.05, verbose=0):
    """
    Use a sliding window to detect the minimum number of fish detected simultaneously, 
    then delete all other EOD clusters. 
    Do this only for EODs within the same width clusters, as a moving fish will contain its EOD width.

    Parameters
    ----------
        clusters: list of ints
            EOD cluster labels.
        eod_t: list of floats
            Timepoints of the EODs (in seconds).
        T: float
            Length of recording (in seconds).
        eod_hights: list of floats
            EOD amplitudes.
        eod_widths: list of ints
            EOD widths in samples.
        samplerate: float
            Recording data samplerate.

        dt (optional): float
            Sliding window size (in seconds).
            Defaults to 1.
        stepsize (optional): float
            Sliding window stepsize (in seconds).
            Defaults to 0.1.
        verbose (optional): int
            Verbosity level.
            Defaults to 0.

    Returns
    -------
        clusters : list of ints
            Cluster labels, where deleted clusters have been set to -1.
        window : list of 2 floats
            Start and end of window selected for deleting moving fish in seconds.
    """

    if len(np.unique(clusters[clusters!=-1])) == 0:
        return clusters, [0,1]

    all_keep_clusters = []

    width_classes = merge_gaussians(eod_widths,np.copy(clusters),0.75)   
    width_classes[np.isin(clusters,all_keep_clusters)] = -1 

    all_windows = []
    all_dts = []
    
    for w in np.unique(width_classes[width_classes!=-1]):

        # initialize variables
        min_clusters = 100
        average_hight = 0
        sparse_clusters = 100
        keep_clusters = []
        window_start = 0
        window_end = dt

        wclusters = clusters[width_classes==w]
        weod_t = eod_t[width_classes==w]
        weod_hights = eod_hights[width_classes==w]
        weod_widths = eod_widths[width_classes==w]

        dt = np.median(eod_widths[width_classes==w])*2000
        all_dts.append(dt)

        if verbose>0:
            print('dt = %f'%dt)

        # make W dependent on width??
        ignore_steps = np.zeros(len(np.arange(0, T-dt+stepsize, stepsize)))

        for i,t in enumerate(np.arange(0, T-dt+stepsize, stepsize)):
            current_clusters = wclusters[(weod_t>=t)&(weod_t<t+dt)&(wclusters!=-1)]
            if len(np.unique(current_clusters))==0:
                ignore_steps[i-int(dt/stepsize):i+int(dt/stepsize)] = 1
                if verbose>0:
                    print('No pulsefish in recording at T=%.2f:%.2f'%(t,t+dt))

        # sliding window
        for t,ignore_step in zip(np.arange(0, T-dt+stepsize, stepsize), ignore_steps):
            current_clusters = wclusters[(weod_t>=t)&(weod_t<t+dt)&(wclusters!=-1)]
            current_widths = weod_widths[(weod_t>=t)&(weod_t<t+dt)&(wclusters!=-1)]

            if (len(np.unique(current_clusters)) <= min_clusters) and (ignore_step==0) and (len(np.unique(current_clusters) !=1)):

                current_labels = np.isin(wclusters, np.unique(current_clusters))
                current_hight = np.mean(weod_hights[current_labels])

                # compute nr of clusters that are too sparse
                clusters_after_deletion = np.unique(remove_sparse_detections(np.copy(clusters[np.isin(clusters,np.unique(current_clusters))]),samplerate*eod_widths[np.isin(clusters,np.unique(current_clusters))],samplerate,T))
                current_sparse_clusters = len(np.unique(current_clusters)) - len(clusters_after_deletion[clusters_after_deletion!=-1])
               
                if current_sparse_clusters <= sparse_clusters and ((current_sparse_clusters<sparse_clusters) or (current_hight > average_hight) or (len(np.unique(current_clusters)) < min_clusters)):
                    
                    keep_clusters = np.unique(current_clusters)
                    min_clusters = len(np.unique(current_clusters))
                    average_hight = current_hight
                    window_end = t+dt
                    sparse_clusters = current_sparse_clusters

        all_keep_clusters.append(keep_clusters)
        all_windows.append(window_end)

    if verbose > 0:
        print('Estimated number of pulsefish: %2i' % min_clusters)


    # delete all clusters that are not selected
    clusters[np.invert(np.isin(clusters, np.concatenate(all_keep_clusters)))] = -1

    return clusters, [np.max(all_windows)-np.max(all_dts), np.max(all_windows)]


def remove_sparse_detections(clusters, eod_widths, samplerate, T,
                             min_density=0.0005, verbose=0):
    """ Remove all EOD clusters that are too sparse

        Parameters
        ----------
        clusters : list of ints
            Cluster labels.
        eod_widths : list of ints
            Cluster widths in samples
        samplerate : int or float
            Samplerate.
        T : int or float
            Lenght of recording in seconds
        min_density (optional) : float
            Minimum density for realistic EOD detections.
            Defaults to 0.05%
        verbose : int (optional)
            Verbosity level.
            Defaults to 0.

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


def plot_all(data, eod_p_times, eod_tr_times, fs, mean_eods):
    '''
    Quick way to view the output and intermediate steps of extract_pulsefish in a plot.

    Parameters:
    -----------
    data: array
        Recording data.
    eod_p_times: array of ints
        EOD peak indices.
    eod_tr_times: array of ints
        EOD trough indices.
    fs: float
        Samplerate.
    mean_eods: list of numpy arrays
        Mean EODs of each pulsefish found in the recording.
    '''

    try:
        cmap = plt.get_cmap("tab10")
    except ValueError:
        cmap = plt.get_cmap("jet")
    
    try:
        fig = plt.figure(constrained_layout=True,figsize=(10,5))
    except TypeError:
        fig = plt.figure(figsize=(10,5))
    if len(eod_p_times) > 0:
        gs = GridSpec(2, len(eod_p_times))
        ax = fig.add_subplot(gs[0,:])
        ax.plot(np.arange(len(data))/fs,data,c='k',alpha=0.3)
        
        for i,(pt,tt) in enumerate(zip(eod_p_times,eod_tr_times)):
            ax.plot(pt,data[(pt*fs).astype('int')],'o',label=i+1,ms=10,c=cmap(i))
            ax.plot(tt,data[(tt*fs).astype('int')],'o',label=i+1,ms=10,c=cmap(i))
            
        #for i,t in enumerate(eod_p_times):
        #    ax.plot(t,data[(t*fs).astype('int')],'o',label=i+1,c=cmap(i))
        ax.set_xlabel('time [s]')
        ax.set_ylabel('amplitude [V]')
        #ax.axis('off')

        for i, m in enumerate(mean_eods):
            ax = fig.add_subplot(gs[1,i])
            ax.plot(1000*m[0], 1000*m[1], c='k')
            ax.fill_between(1000*m[0],1000*(m[1]-m[2]),1000*(m[1]+m[2]),color=cmap(i))
            ax.set_xlabel('time [ms]')
            ax.set_ylabel('amplitude [mV]') 
    else:
        plt.plot(np.arange(len(data))/fs,data,c='k',alpha=0.3)
