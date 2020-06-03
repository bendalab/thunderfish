"""
# Extract pulse-type weakly electric fish
Extract all timepoints where pulsefish EODs are present for each separate pulsefish in a recording.

## Main function
- `extract_pulsefish()`: checks for pulse-type fish based on the EOD amplitude and shape.

Author: Liz Weerdmeester
Email: weerdmeester.liz@gmail.com

"""

# XXX there is something wrong with the BGM...
# maybe the merge_gaussian functions acts differently than I thought?
# seems to be random, maybe it has to do with the ordering of the unique labels?

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy import stats

from scipy.signal import argrelextrema

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from sklearn.mixture import BayesianGaussianMixture
from sklearn.metrics import pairwise_distances

from scipy.spatial import distance_matrix
from scipy.interpolate import interp1d

from .eventdetection import detect_peaks
from .pulse_tracker_helper import makeeventlist, discardnearbyevents, discard_connecting_eods

import warnings

def warn(*args,**kwargs):
    pass
warnings.warn=warn

# upgrade numpy functions for backwards compatibility:
if not hasattr(np, 'isin'):
    np.isin = np.in1d

def extract_pulsefish(data, samplerate, cutwidth=0.01, verbose=0, plot_level=1, **kwargs):
    """ Extract and cluster pulse fish EODs from recording.
    
    Takes recording data containing an unknown number of pulsefish and extracts the mean 
    EOD and EOD timepoints for each fish present in the recording.
    
    Parameters
    ----------
    data: 1-D array of float
        The data to be analysed.
    samplerate: float
        Sampling rate of the data in Hertz.
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

    mean_eods, eod_times, eod_peaktimes, zoom_window = [], [], [], []
    
    # extract peaks and interpolated data
    x_peak, x_trough, eod_hights, eod_widths, i_samplerate, i_data, interp_f = extract_eod_times(data, samplerate, cutwidth=cutwidth,verbose=verbose-1, plot_level=plot_level-1, **kwargs)
    
    if len(x_peak)>0:

        # cluster on peaks
        peak_clusters = cluster(x_peak, eod_hights, eod_widths, i_data, i_samplerate,
                                interp_f, cutwidth, verbose=verbose-1, plot_level=plot_level-1, **kwargs) 

        # cluster on troughs
        trough_clusters = cluster(x_trough, eod_hights, eod_widths, i_data, i_samplerate,
                                 interp_f, cutwidth, verbose=verbose-1, plot_level=plot_level-1, **kwargs)

        # merge peak and trough clusters
        clusters, x_merge = merge_clusters(peak_clusters, trough_clusters, x_peak, x_trough, verbose=verbose-1)

            
        # delete the moving fish
        clusters, zoom_window = delete_moving_fish(clusters, x_merge/i_samplerate, len(data)/samplerate,
                                      eod_hights, eod_widths, verbose=verbose-1)

        # extract mean eods
        mean_eods, eod_times, eod_peaktimes, eod_troughtimes = extract_means(i_data, x_merge, x_peak, x_trough, eod_widths,
                                                              clusters, i_samplerate, verbose=verbose-1)

        if plot_level>0:
            plot_all(data, eod_peaktimes, eod_troughtimes, samplerate, mean_eods)
    
    return mean_eods, eod_times, eod_peaktimes, zoom_window


def extract_eod_times(data, samplerate, interp_freq=500000, peakwidth=0.01, cutwidth=0.01, win_size = 0.0005, n_stds = 1000, threshold_factor=6, verbose=0, plot_level=0):
    """ Extract peaks from data which are potentially EODs.

    Parameters
    ----------
    data: 1-D array of float
        The data to be analysed.
    samplerate: int or float
        Sampling rate of the data

    interp_freq: int or float (optional)
        Desired resolution in Hz. Data will be interpolated to match this resolution.
        Defaults to 500 kHz
    peakwidth: int (optional)
        Maximum width for peak detection in seconds.
        Defaults to 10 ms.
    cutwidth: int (optional)
        Maximum width for extracting snippets for clustering based on EOD shape in seconds.
        Defaults to 10 ms
    win_size: float (optional)
        Window size for determining peak detection threshold in seconds.
        Defaults to 0.5 ms.
    n_stds: int (optional)
        Number of standard deviations to make on data for determining peak detection threshold.
        Defaults to 1000.
    threshold_factor: float (optional)
        Multiplication factor for peak detection threshold.
        Defaults to 6.

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
    win_size_indices = int(win_size * samplerate)
    step = len(data)//n_stds
    if step < 1:
        step = 1
    stds = [np.std(data[i:i+win_size_indices], ddof=1)
            for i in range(0, len(data)-win_size_indices, step)]

    threshold = np.median(stds) * threshold_factor

    try:
        interp_f = int(interp_freq/samplerate)
        f = interp1d(range(len(data)),data,kind='quadratic')
        data = f(np.arange(0,len(data)-1,1/interp_f))
    except MemoryError:
        interp_f = 1

    orig_x_peaks, orig_x_troughs = detect_peaks(data, threshold)

    if len(orig_x_peaks)==0 or len(orig_x_peaks)>samplerate:
        if verbose>0:
            print('No peaks detected.')
        return [], [], [], [], samplerate*interp_f,data, interp_f
    else:

        peaks = makeeventlist(orig_x_peaks, orig_x_troughs, data, peakwidth*samplerate*interp_f, 2*interp_f, verbose=verbose-1)
        x_peaks, x_troughs, eod_hights, eod_widths = discard_connecting_eods(peaks[0], peaks[1], peaks[3], peaks[4],verbose=verbose-1)
        
        if plot_level>0:
            plt.figure()
            plt.plot(data)
            plt.plot(orig_x_peaks,data[orig_x_peaks],'o',ms=10)
            plt.plot(orig_x_troughs,data[orig_x_troughs],'o',ms=10)
            plt.plot(peaks[0],data[peaks[0].astype('int')],'o')
            plt.plot(peaks[1],data[peaks[1].astype('int')],'o')
            plt.plot(x_peaks,data[x_peaks.astype('int')],'x',ms=10)
            plt.plot(x_troughs,data[x_troughs.astype('int')],'x',ms=10)        

        # only take those where the maximum cutwidth does not casue issues
        cut_idx = np.where((x_peaks>int(cutwidth*samplerate*interp_f)) & (x_peaks<(len(data)-int(cutwidth*samplerate*interp_f))) & (x_troughs>int(cutwidth*samplerate*interp_f)) & (x_troughs<(len(data)-int(cutwidth*samplerate*interp_f))))[0]
        
        if verbose>0:
            print('Remaining peaks after EOD extraction                    %5i\n'%(len(cut_idx)))
            if verbose>1:
                print('Remaining peaks after deletion due to cutwidth          %5i\n'%(len(cut_idx)))

        return x_peaks[cut_idx], x_troughs[cut_idx], eod_hights[cut_idx], eod_widths[cut_idx], samplerate*interp_f, data, interp_f

def BGM(x,merge_threshold=0.1,n_gaus=5,max_iter=200,n_init=5,verbose=1,plot_level=0):
    """ Use a Bayesian Gaussian Mixture Model to cluster one-dimensional data. 
        Additional steps are used to merge clusters that are closer than merge_percentage.
        Broad gaussian fits that cover one or more other gaussian fits are split.

        Parameters
        ----------
        x : 1D numpy array
            Features to compute clustering on. 

        merge_percentage : float (optional)
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
            plt.hist(x[labels==l],alpha=0.3)

    labels = merge_gaussians(x,labels,merge_threshold)

    # plot labels after merging close clusters
    if plot_level>0:
        plt.subplot(1,4,3)
        for l in np.unique(labels):
            plt.hist(x[labels==l],alpha=0.3)
    
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
            plt.hist(x[labels==l],alpha=0.3)

    return labels

def merge_gaussians(x,labels,merge_threshold=0.1):
    """ Merge all clusters have medians which are near. Only works in 1D.
        
        Parameters
        ----------
        x : 1D array of ints or floats
            Features used for clustering.
        labels : 1D array of ints
            Labels for each sample in x.
        merge_threshold : float (optional)
            Similarity threshold to merge clusters

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

def cluster(eod_x, eod_hights, eod_widths, data, samplerate, interp_f, cutwidth, width_factor=3, 
            n_gaus_hight=10, merge_threshold_hight=0.1, n_gaus_width=3, merge_threshold_width=0.5, 
            n_pc=5, minp=10, percentile=75, max_epsilon=0.02, verbose=0, plot_level=0):
    
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
    cutwidth: float
        Maximum width for extracting snippets for clustering based on EOD shape in seconds.
        Has to be the same cutwidth as used for timepoint extraction.
    
    width_factor (optional) : int or float
        Multiplier for snippet extraction width. This factor is multiplied with the width
        between the peak and through of a single EOD.
        Defaults to 3.
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

    # cap the widths at the maximum cutwidth value for a better gaussian fit.
    eod_widths[eod_widths>cutwidth*samplerate/width_factor] = cutwidth*samplerate/width_factor    

    all_clusters = np.ones(len(eod_x))*-1
    hight_labels = BGM(eod_hights,merge_threshold_hight,n_gaus_hight,plot_level=plot_level-1)

    if verbose>0:
        print('Clusters generated based on EOD amplitude:')
        [print('N_{} = {:>4}      h_{} = {:.4f}'.format(l,len(hight_labels[hight_labels==l]),l,np.mean(eod_hights[hight_labels==l]))) for l in np.unique(hight_labels)]   

    max_label = 0   # keep track of the labels so that no labels are overwritten

    # loop only over hight clusters that are bigger than minp
    hl, hlc = unique_counts(hight_labels)
    unique_hight_labels = hl[hlc>minp]

    for hi,hight_label in enumerate(unique_hight_labels):

        # select only features in one hight cluster at a time
        h_eod_widths = eod_widths[hight_labels==hight_label]
        h_eod_hights = eod_hights[hight_labels==hight_label]
        h_eod_x = eod_x[hight_labels==hight_label]
        h_clusters = np.ones(len(h_eod_x))*-1

        # determine width labels
        width_labels = BGM(h_eod_widths,merge_threshold_width,n_gaus_width,plot_level=plot_level-1)
        if verbose>0:
            print('Clusters generated based on EOD width:')
            [print('N_{} = {:>4}      h_{} = {:.4f}'.format(l,len(width_labels[width_labels==l]),l,np.mean(h_eod_widths[width_labels==l]))) for l in np.unique(width_labels)]   

        wl, wlc = unique_counts(width_labels)
        unique_width_labels = wl[wlc>minp]

        if plot_level>0:
            plt.figure()

        for wi,width_label in enumerate(unique_width_labels):

            w_eod_widths = h_eod_widths[width_labels==width_label]
            w_eod_hights = h_eod_hights[width_labels==width_label]
            w_eod_x = h_eod_x[width_labels==width_label]

            width = width_factor*np.median(w_eod_widths)
            snippets = np.vstack([data[int(x-width):int(x+width)] for x in w_eod_x])

            # subtract the slope and normalize the snippets
            snippets = StandardScaler().fit_transform(snippets.T).T
            snippets = subtract_slope(snippets)

            # scale so that the absolute integral = 1.
            snippets = (snippets.T/np.sum(np.abs(snippets),axis=1)).T

            # compute features for clustering on waveform
            features = PCA(n_pc).fit(snippets).transform(snippets)

            # determine clustering threshold from data
            minpc = max(minp,int(len(features)*0.01))  
            knn = np.sort(pairwise_distances(features,features),axis=0)[minpc] #[minpc]
            eps = min(max_epsilon,np.percentile(knn,percentile))

            # cluster on EOD shape
            w_clusters = DBSCAN(eps=eps, min_samples=minpc).fit(features).labels_
            
            # plot clusters
            if plot_level>0:
                cols = ['b','r','g','y','m','c']

                plt.subplot(2,len(unique_width_labels),wi+1)

                for j,c in enumerate(np.unique(w_clusters)):
                    if c==-1:
                        plt.scatter(features[w_clusters==c,0],features[w_clusters==c,1],alpha=0.1,c='k',label='-1')
                    else:
                        plt.scatter(features[w_clusters==c,0],features[w_clusters==c,1],alpha=0.1,c=cols[j%len(cols)],label=c+max_label)
                        plt.title('h = %.3f, w=%i'%(np.mean(w_eod_hights[w_clusters==c]),np.mean(w_eod_widths[w_clusters==c])))

                plt.subplot(2,len(unique_width_labels),len(unique_width_labels)+wi+1)

                for j,c in enumerate(np.unique(w_clusters)):
                    if c==-1:
                        plt.plot(snippets[w_clusters==c].T,alpha=0.1,c='k',label='-1')
                    else:
                        plt.plot(snippets[w_clusters==c].T,alpha=0.1,c=cols[j%len(cols)],label=c+max_label)
                        plt.title('h = %.3f, w=%i'%(np.mean(w_eod_hights[w_clusters==c]),np.mean(w_eod_widths[w_clusters==c])))

            # remove artefacts here, based on the mean snippets ffts.
            w_clusters = remove_artefacts(snippets, w_clusters, interp_f, verbose=verbose-1)

            w_clusters[w_clusters==-1] = -max_label - 1
            h_clusters[width_labels==width_label] = w_clusters + max_label
            max_label = max(np.max(h_clusters),np.max(all_clusters)) + 1

            if plot_level>0:
                plt.legend()

        if verbose > 0:
            if np.max(h_clusters) == -1:
                print('No EODs in hight cluster %i'%hight_label)
            elif len(np.unique(h_clusters[h_clusters!=-1]))>1:
                print('%i different EODs in hight cluster %i'%(len(np.unique(h_clusters[h_clusters!=-1])),hight_label))

        # update maxlab so that no clusters are overwritten
        all_clusters[hight_labels==hight_label] = h_clusters

    # remove all non-reliable clusters
    all_clusters = remove_sparse_detections(all_clusters,eod_widths,samplerate,len(data)/samplerate,verbose=verbose-1)
    all_clusters = delete_unreliable_fish(all_clusters,eod_widths,eod_x,verbose=verbose-1)
    all_clusters = delete_wavefish_and_sidepeaks(data,all_clusters,eod_x,eod_widths,verbose=verbose-1)
    
    for ac in np.unique(all_clusters[all_clusters!=-1]):
        snippets = np.vstack([data[int(x-width):int(x+width)] for x in eod_x[all_clusters==ac]])

    if verbose>0:
        print('Clusters generated based on hight, width and shape: ')
        [print('N_{} = {:>4}'.format(int(l),len(all_clusters[all_clusters==l]))) for l in np.unique(all_clusters[all_clusters!=-1])]
             
    return all_clusters


def extract_means(data, eod_x, eod_peak_x, eod_tr_x, eod_widths, clusters, samplerate,  w_factor=4, verbose=0):
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
        w_factor (optional): float
            Multiplication factor for window used to extract EOD.
            Defaults to 4.
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
        
    """

    mean_eods, eod_times, eod_peak_times, eod_tr_times, eod_hights = [], [], [], [], []
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
           
    return [m for _,m in sorted(zip(eod_hights,mean_eods))], [t for _,t in sorted(zip(eod_hights,eod_times))], [pt for _,pt in sorted(zip(eod_hights,eod_peak_times))], [tt for _,tt in sorted(zip(eod_hights,eod_tr_times))]

def delete_unreliable_fish(clusters,eod_widths,eod_x,verbose):
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
        
        if np.max(np.median(eod_widths[clusters==cluster])/np.diff(eod_x[cluster==clusters])) > 0.5:
            if verbose>0:
                print('deleting unreliable cluster %i, score=%f'%(cluster,np.max(np.median(eod_widths[clusters==cluster])/np.diff(eod_x[cluster==clusters]))))
            clusters[clusters==cluster] = -1
    return clusters


def delete_wavefish_and_sidepeaks(data, clusters, eod_x, eod_widths, w_factor=8, verbose=0):
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
        w_factor : float or int
            Multiplier for EOD analysis width.
            Defaults to 8.
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
        
        mean_eod = np.mean(snippets, axis=0)
        mean_eod = subtract_slope(mean_eod.reshape(1,-1)).flatten()
        pk, tr = detect_peaks(np.concatenate(([mean_eod[0]*10],mean_eod,[mean_eod[-1]*10])), 0.5*(np.max(mean_eod)-np.min(mean_eod)))

        pk = pk[(pk>0)&(pk<len(mean_eod))]
        tr = tr[(tr>0)&(tr<len(mean_eod))]

        pk2, tr2 = detect_peaks(np.concatenate(([mean_eod[0]*10],mean_eod,[mean_eod[-1]*10])), 0.01*(np.max(mean_eod)-np.min(mean_eod)))
        idxs = np.sort(np.concatenate((tr2,pk2)))
        slopes = np.abs(np.diff(mean_eod[idxs]))
        m_slope = np.argmax(slopes)
        centered = np.min(np.abs(idxs[m_slope:m_slope+2] - int(len(mean_eod)/2)))

        idxs = np.sort(np.concatenate((pk2,tr2)))
        hdiffs = np.diff(mean_eod[idxs])

        if len(pk)>0 and len(tr)>0:

            w_diff = np.abs(np.diff(np.sort(np.concatenate((pk,tr)))))

            if centered>10 or np.abs(np.diff(idxs[m_slope:m_slope+2])) < np.mean(eod_widths[clusters==cluster])*0.5 or len(pk) + len(tr)>5 or np.min(w_diff)>2*cutwidth/w_factor or len(hdiffs[np.abs(hdiffs)>0.5*(np.max(mean_eod)-np.min(mean_eod))])>=5:
                if verbose>0:
                    print('Deleting cluster %i, which is a wavefish'%cluster)
                clusters[clusters==cluster] = -1
        elif centered>10: # XXX make this a variable

            if verbose>0:
                print('Deleting cluster %i, which is a sidepeak'%cluster)
            clusters[clusters==cluster] = -1

    return clusters

def remove_sparse_detections(clusters, eod_widths, samplerate, T, min_density=0.0005, verbose=0):
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
            Defaults to 0.5%

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

    
def remove_artefacts(all_snippets, clusters, int_f, artefact_threshold=0.75, cutoff_f=10000, verbose=0):
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
            Defaults to 0.8.
        curoff_f (optional) : int or float
            Cut-off frequency for low frequency estimation.
        verbose (optional): int
            Verbosity level.

    Returns
    -------
        clusters: list of ints
            Cluster labels, where noisy cluster and clusters with artefacts have been set to -1.
    """

    for cluster in np.unique(clusters):
        if cluster!=-1:

            snippets = all_snippets[clusters==cluster]
            mean_eod = np.mean(snippets, axis=0)
            cut_fft = int(len(np.fft.fft(mean_eod))/2)
            low_frequency_ratio = np.sum(np.abs(np.fft.fft(mean_eod))[1:int(cut_fft/(2*int_f))])/np.sum(np.abs(np.fft.fft(mean_eod))[1:int(cut_fft)])   

            if low_frequency_ratio < artefact_threshold:
                clusters[clusters==cluster] = -1
                if verbose>0:
                    print('Deleting cluster %i, which has a low frequency ratio of %f'%(cluster,low_frequency_ratio))

    return clusters


def merge_clusters(clusters_1, clusters_2, x_1, x_2,verbose=0): 
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


def delete_moving_fish(clusters, eod_t, T, eod_hights, eod_widths, verbose=0, dt=1, stepsize=0.1):
    """
    Use a sliding window to detect the minimum number of fish detected simultaneously, 
    then delete all other EOD clusters. 
    Do this for various width clusters, as a moving fish will contain its EOD width.

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
    
    ignore_steps = np.zeros(len(np.arange(0, T-dt+stepsize, stepsize)))

    for i,t in enumerate(np.arange(0, T-dt+stepsize, stepsize)):
        current_clusters = clusters[(eod_t>=t)&(eod_t<t+dt)&(clusters!=-1)]
        if len(np.unique(current_clusters))==0:
            ignore_steps[i-int(dt/stepsize):i+int(dt/stepsize)] = 1
            if verbose>0:
                print('No pulsefish in recording at T=%.2f:%.2f'%(t,t+dt))

    width_classes = merge_gaussians(eod_widths,np.copy(clusters),0.5)
    all_keep_clusters = []
    all_windows = []
    
    for w in np.unique(width_classes[width_classes!=-1]):

        # initialize variables
        min_clusters = 100
        average_hight = 0
        keep_clusters = []
        window_start = 0
        window_end = dt

        wclusters = clusters[width_classes==w]
        weod_t = eod_t[width_classes==w]
        weod_hights = eod_hights[width_classes==w]

        # sliding window
        for t,ignore_step in zip(np.arange(0, T-dt+stepsize, stepsize), ignore_steps):
            current_clusters = wclusters[(weod_t>=t)&(weod_t<t+dt)&(wclusters!=-1)]

            if (len(np.unique(current_clusters)) <= min_clusters) and (ignore_step==0) and (len(np.unique(current_clusters) !=1)):

                current_labels = np.isin(wclusters, np.unique(current_clusters))
                current_hight = np.mean(weod_hights[current_labels])

                if (current_hight > average_hight) or (len(np.unique(current_clusters)) < min_clusters):
                    keep_clusters = np.unique(current_clusters)
                    min_clusters = len(np.unique(current_clusters))
                    average_hight = current_hight
                    window_start = t
                    window_end = t+dt

        all_keep_clusters.append(keep_clusters)
        all_windows.append([window_start,window_end])

    if verbose>0:
        print('Estimated nr of fish in recording: %i'%min_clusters)

    # delete all clusters that are not selected
    clusters[np.invert(np.isin(clusters, np.concatenate(all_keep_clusters)))] = -1

    # XXX what to do about the window???

    return clusters, [window_start, window_end]


def subtract_slope(snippets):
    """ Subtract underlying slope from all EOD snippets.
    
    Method still under revision.

    Parameters
    ----------
        snippets: 2-D numpy array
            All EODs in a recorded stacked as snippets. 
            Shape = (number of EODs, EOD width)
    Returns
    -------
        snippets: 2-D numpy array
            EOD snippets with underlying slope subtracted.
    """

    left_y = snippets[:,0]
    right_y = snippets[:,-1]

    try:
        slopes = np.linspace(left_y, right_y, snippets.shape[1])
    except ValueError:
        delta = (right_y - left_y)/snippets.shape[1]
        slopes = np.arange(0, snippets.shape[1], dtype=snippets.dtype).reshape((-1,) + (1,) * np.ndim(delta))*delta + left_y
    
    return snippets - slopes.T


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


def plot_all(data, eod_p_times, eod_tr_times, fs, mean_eods):
    '''
    Quick way to view the output of extract_pulsefish in a plot.
    '''
    
    cmap = plt.get_cmap("tab10")
    
    fig = plt.figure(constrained_layout=True,figsize=(10,5))
    if len(eod_p_times) > 0:
        gs = GridSpec(2, len(eod_p_times), figure=fig)
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