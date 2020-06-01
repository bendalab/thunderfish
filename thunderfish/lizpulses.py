"""
# Extract pulse-type weakly electric fish
Extract all timepoints where pulsefish EODs are present for each separate pulsefish in a recording.

## Main function
- `extract_pulsefish()`: checks for pulse-type fish based on the EOD amplitude and shape.

Author: Liz Weerdmeester
Email: weerdmeester.liz@gmail.com

"""

# NOTE rename pulse_unreliability
# or just remove at this point

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy import stats

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


def extract_pulsefish(data, samplerate, peakwidth=0.002, cutwidth=0.001, threshold_factor=2, verbose=0, plot_steps=False, **cluster_kwargs):
    """ Extract and cluster pulse fish EODs from recording.
    
    Takes recording data containing an unknown number of pulsefish and extracts the mean 
    EOD and EOD timepoints for each fish present in the recording.
    
    Parameters
    ----------
    data: 1-D array of float
        The data to be analysed.
    samplerate: float
        Sampling rate of the data in Hertz.
    peakwidth: float (optional)
        Peakwidth for peak detection in seconds.
    cutwidth: float (optional)
        Width for extracting snippets for clustering based on EOD shape.
    verbose : int (optional)
        Verbosity level.
    **cluster_kwargs: (optional) 
        keyword arguments for clustering parameters (see 'cluster()')
        
    Returns
    -------
    mean_eods: list of 2D arrays
        The average EOD for each detected fish. First column is time in seconds,
        second column the mean eod, third column the standard error.
    eod_times: list of 1D arrays
        For each detected fish the times of EOD peaks in seconds.
    eod_unreliability: list of floats
        For each detected fish, the unreliability score, where scores > 0.1
        signify unreliable EOD means (possibly wavefish or unreliable clusters).
    """

    mean_eods, eod_times, eod_peaktimes,eod_unreliability,zoom_window = [], [], [], [], []
    
    # extract peaks
    x_peak, x_trough, eod_hights, eod_widths, i_samplerate, i_data, interp_f = extract_eod_times(data, samplerate,
                                                                 peakwidth,cutwidth,threshold_factor,verbose-1)
    if len(x_peak)>0:

        # cluster on peaks
        peak_clusters = cluster(x_peak, eod_hights/eod_widths, eod_widths, i_data, i_samplerate,
                                cutwidth, interp_f, verbose-1, **cluster_kwargs) 

        # cluster on troughs
        trough_clusters = cluster(x_trough, eod_hights/eod_widths, eod_widths, i_data, i_samplerate,
                                  cutwidth, interp_f, verbose-1, **cluster_kwargs)

        # merge peak and trough clusters
        clusters, x_merge = merge_clusters(peak_clusters, trough_clusters, x_peak, x_trough, verbose-1)

            
        # delete the moving fish
        clusters, zoom_window = delete_moving_fish(clusters, x_merge/i_samplerate, len(data)/samplerate,
                                      eod_hights, verbose-1)

        # extract mean eods
        mean_eods, eod_times, eod_peaktimes, eod_unreliability = find_window(i_data, x_merge, x_peak, eod_widths,
                                                              clusters, i_samplerate, verbose-1)

        if plot_steps:
            plot_all(data, eod_times, samplerate, mean_eods, eod_unreliability)
    
    return mean_eods, eod_times, eod_peaktimes, eod_unreliability, zoom_window


def extract_eod_times(data, samplerate, peakwidth, cutwidth, threshold_factor=1,verbose=0,plot_steps=False,interp_freq=500000):
    """ Extract peaks from data which are potentially EODs.

    Parameters
    ----------
    data: 1-D array of float
        The data to be analysed.
    peakwidth: int
        Peakwidth for peak detection in samples.
    cutwidth: int
        Width for extracting snippets for clustering based on EOD shape in samples.
    threshold_factor: float (optional)
        Multiplication factor for peak detection threshold.
    verbose : int (optional)
        Verbosity level.

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
    """

    #NOTE: try to skip this.
    try:
        interp_f = int(interp_freq/samplerate)
        f = interp1d(range(len(data)),data,kind='quadratic')
        data = f(np.arange(0,len(data)-1,1/interp_f))
    except MemoryError:
        interp_f = 1

    orig_x_peaks = np.zeros(int(samplerate*interp_f))

    factor=0.01

    while len(orig_x_peaks)>len(data)*factor: 
        threshold = (np.mean(np.abs(data)))*threshold_factor
        orig_x_peaks, orig_x_troughs = detect_peaks(data, threshold)
        print('manual threshold:', threshold)
        threshold_factor = threshold_factor*2

    # standard deviation of data in small snippets:
    win_size = 0.001   # XXX make this a parameter
    win_size_indices = int(win_size * samplerate)
    if win_size_indices < 10:
        win_size_indices = 10
    n_stds = 1000      # XXX make this a parameter
    step = len(data)//n_stds
    if step < win_size_indices//2:
        step = win_size_indices//2
    stds = [np.std(data[i:i+win_size_indices], ddof=1)
            for i in range(0, len(data)-win_size_indices, step)]
    # the distribution of stds will be a Gaussian with mean given by
    # the standard deviation of the noise plus a tail introduced by
    # the EOD pulses. We are interested in the mean standard deviation
    # of the noise only. Taking the median of the standard deviations
    # seems to be a good guess.
    # XXX The factor 6 should be a parameter, i.e. threhsold_factor.
    threshold = np.median(stds) * 6.0
    print('median threshold:', threshold)
    orig_x_peaks, orig_x_troughs = detect_peaks(data, threshold)

    if len(orig_x_peaks)==0 or len(orig_x_peaks)>samplerate:
        if verbose>0:
            print('No peaks detected.')
        return [], [], [], [], samplerate*interp_f,data, interp_f
    else:
        peaks = makeeventlist(orig_x_peaks, orig_x_troughs, data, 10*peakwidth*samplerate*interp_f, 2*interp_f, verbose-1)
        
        if plot_steps:
            plt.figure()
            plt.plot(data)
            plt.plot(peaks[0],data[peaks[0].astype('int')],'o')
            plt.plot(peaks[1],data[peaks[1].astype('int')],'o')

        
        #peakindices, _, _ = discardnearbyevents(peaks[0], peaks[3], peaks[3]/peaks[4], peakwidth,verbose-1)
        #x_peaks, x_troughs, eod_hights, eod_widths = discard_connecting_eods(peaks[0][peakindices], peaks[1][peakindices], peaks[3][peakindices], peaks[4][peakindices],verbose-1)
        x_peaks, x_troughs, eod_hights, eod_widths = discard_connecting_eods(peaks[0], peaks[1], peaks[3], peaks[4],verbose-1)
        
        if plot_steps:
            plt.plot(x_peaks,data[x_peaks.astype('int')],'x',ms=10)
            plt.plot(x_troughs,data[x_troughs.astype('int')],'x',ms=10)
        
        
                
        peakindices = discardnearbyevents(x_peaks, x_troughs, eod_widths, eod_hights/eod_widths, 0.1*peakwidth*samplerate*interp_f, verbose-1)

        x_peaks=x_peaks[peakindices]
        x_troughs = x_troughs[peakindices]
        eod_hights = eod_hights[peakindices]
        eod_widths = eod_widths[peakindices]

        if plot_steps:
            plt.plot(x_peaks,data[x_peaks.astype('int')],'o',ms=10,alpha=0.5)
            plt.plot(x_troughs,data[x_troughs.astype('int')],'o',ms=10,alpha=0.5)
        

        # only take those where the cutwidth does not casue issues
        cut_idx = np.where((x_peaks>int(cutwidth*samplerate*interp_f/2)) & (x_peaks<(len(data)-int(cutwidth*samplerate*interp_f/2))) & (x_troughs>int(cutwidth*samplerate*interp_f/2)) & (x_troughs<(len(data)-int(cutwidth*samplerate*interp_f/2))))[0]
        
        if verbose==1:
            print('Remaining peaks after EOD extraction                    %5i\n'%(len(cut_idx)))
        elif verbose>0:
            print('Remaining peaks after deletion due to cutwidth          %5i\n'%(len(cut_idx)))


        return x_peaks[cut_idx], x_troughs[cut_idx], eod_hights[cut_idx], eod_widths[cut_idx], samplerate*interp_f, data, interp_f


def cluster(eod_x, eod_hights, eod_widths, data, samplerate, cutwidth, interp_f, verbose=0, minp=10,
            percentile=75, n_pc=2, n_gaus=6, n_init=5, max_iter=200, remove_slope=True,plot_steps=False):
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
    cutwidth: float
        Width for extracting snippets for clustering based on EOD shape.
    verbose (optional): int
        Verbosity level.
    minp (optional): int
        Minimun number of points for core cluster (DBSCAN).
        Defaults to 10.
    percentile (optional): float
        Percentile for determining epsilon from knn distribution where k=minp.
        Defaults to 75.
    n_pc (optional): int
        Number of PCs to use for PCA.
        Defaults to 3.
    n_gaus (optional): int
        Maximun number of gaussians to fit for clustering on EOD hights.
        Defaults to 4.
    n_init (optional): int
        Number of initializations for BayesianGaussianMixture method. 
        Increase for improved accuracy, decrease for improved computation speed.
        Defaults to 1.
    max_iter (optional): int
        Maximum number of iterations for BayesianGausssianMixture method.
    remove_slope (optional): boolean
        If True, subtract slope to improve clustering for pulse EODs superimposed on slow wavefish.
        Defaults to False, as method is still being tested.

    Returns
    -------
    labels : list of ints
        EOD cluster labels based on hight and EOD waveform.
    """

    # initiate labels based on hight

    hight_labels = np.ones(eod_hights.shape)
    
    if ((np.max(eod_hights) - np.min(eod_hights))/np.max(eod_hights)) > 0.25:

        # classify by height
        BGM_model = BayesianGaussianMixture(n_gaus, max_iter=max_iter, n_init=n_init)
        hight_labels = BGM_model.fit_predict(stats.zscore(eod_hights).reshape(-1, 1))
        
        if plot_steps:
            plt.figure()
            plt.hist(eod_hights)
            plt.figure()
            for h in np.unique(hight_labels):
                plt.hist(eod_hights[hight_labels==h])

        # set small clusters to -1.
        #for hl in np.unique(hight_labels):
        #    if len(hight_labels[hight_labels==hl])<minp:
        #        hight_labels[hight_labels==hl] = -1

        # 1. compare all the means of the gaussians. If they are too close, merge them.
        meds = []
        for l in np.unique(hight_labels):
            meds.append(np.median(eod_hights[hight_labels==l]))

        mapping = {}
        for l1,m1 in enumerate(meds):
            for l2,m2 in enumerate(meds):
                if m1!=m2:
                    
                    if np.abs(np.diff([m1,m2]))/np.max([m1,m2]) < 0.25:
                        mapping[l1] = l2
    

        for mk,mv in mapping.items():
            hight_labels[hight_labels==mk] = mv

        ul = np.unique(hight_labels)
        thresholds = np.vstack([[np.max(eod_hights[hight_labels==hl]),np.min(eod_hights[hight_labels==hl])] for hl in ul])

        for t,hl in zip(thresholds,ul):
            tr = thresholds.flatten()

            # is there any other time that splits the min and max?
            if len(tr[(tr>np.min(t)) & (tr<np.max(t))])>0:
                hlb = hight_labels[hight_labels==hl]
                c_eodh = eod_hights[hight_labels==hl]
                maxlab = np.max(hight_labels)+1
                for split in np.sort(tr[(tr>np.min(t)) & (tr<np.max(t))]):
                    hlb[c_eodh>split] = maxlab
                    maxlab = maxlab+1
                hight_labels[hight_labels==hl] = hlb

        # if any of the clusters merged have very little height difference, merge them.
        

             


        '''
        ul = np.unique(hight_labels)
        if len(np.unique(hight_labels))>1:
            for i, hight_label in enumerate(ul):
                for j, hight_label2 in enumerate(ul):
                    if hight_label!=hight_label2:
                        med1 = np.median(eod_hights[hight_labels==hight_label])
                        med2 = np.median(eod_hights[hight_labels==hight_label2])
        
        ul = np.unique(hight_labels)
        if len(np.unique(hight_labels))>1:
            for i, hight_label in enumerate(ul):
                if ((np.max(eod_hights[hight_labels!=hight_label]) - np.min(eod_hights[hight_labels!=hight_label]))/np.max(eod_hights[hight_labels!=hight_label])) < 0.25:
                    hight_labels[hight_labels!=hight_label] = np.max(hight_labels) + 1
        '''
        if plot_steps:
            plt.figure()
            for h in np.unique(hight_labels):
                plt.hist(eod_hights[hight_labels==h])

        if verbose>0:
            if not BGM_model.converged_:
                print('!!! Gaussian mixture did not converge !!!')
    if verbose>0:
        print('Clusters generated based on EOD amplitude:')
        [print('N_{} = {:>4}      h_{} = {:.4f}'.format(h,len(hight_labels[hight_labels==h]),h,np.mean(eod_hights[hight_labels==h]))) for h in np.unique(hight_labels)]

    # now cluster based on waveform
    labels = np.ones(len(hight_labels))*-1    
    
    # extract snippets
    snippets = np.vstack([data[int(x-cutwidth*samplerate/2):int(x+cutwidth*samplerate/2)] for x in eod_x]) 
    
    if remove_slope:
        snippets = subtract_slope(snippets)
    
    # keep track of the labels so that no labels are overwritten
    max_label = 0
    
    if plot_steps:
        plt.figure()

    for i,hight_label in enumerate(np.unique(hight_labels)):
        if len(hight_labels[hight_labels==hight_label]) > minp:
            
            # extract snippets, idxs and hs for this hight cluster
            current_snippets = snippets[hight_labels==hight_label]#StandardScaler().fit_transform(snippets[hight_labels==hight_label].T).T
            
            clusters = np.ones(len(current_snippets))*-1
            ceod_widths = eod_widths[hight_labels==hight_label]


            #fast_features = PCA(n_pc).fit(current_snippets[ceod_widths<3*np.median(eod_widths)]).transform(current_snippets)
            #slow_features = PCA(n_pc).fit(current_snippets[ceod_widths>=3*np.median(eod_widths)]).transform(current_snippets.T)
            fast_features = current_snippets[ceod_widths<3*np.median(eod_widths)]#PCA(n_pc).fit(current_snippets).transform(current_snippets)
            slow_features = current_snippets[ceod_widths>=3*np.median(eod_widths)]

            slow_clusters = np.ones(len(slow_features))*-1
            fast_clusters = np.ones(len(fast_features))*-1

            if len(fast_features)>minp:

                fast_features = PCA(n_pc).fit(fast_features).transform(fast_features)

                minpc = max(minp,int(len(fast_features)*0.05))
            
                # determine good epsilon for DBSCAN  
                knn = np.sort(pairwise_distances(fast_features, fast_features))[:,minpc]        
                eps = np.percentile(knn, percentile)

                # cluster by EOD shape
                fast_clusters = DBSCAN(eps=eps, min_samples=minp).fit(fast_features).labels_
                
                if plot_steps:
                    plt.subplot(4,len(np.unique(hight_labels)),i+1)

                    cols = ['b','r','g','y','m','c']
                
                    for j,c in enumerate(np.unique(fast_clusters)):
                        if c==-1:
                            plt.plot(fast_features[fast_clusters==c].T,alpha=0.1,c='k')
                        else:
                            plt.plot(fast_features[fast_clusters==c].T,alpha=0.1,c=cols[j%len(cols)])
                            plt.title('dbscan, h = %f'%(np.mean(eod_hights[hight_labels==hight_label])))              

            if len(slow_features)>minp:

                slow_features = PCA(n_pc).fit(slow_features).transform(slow_features)

                minpc = max(minp,int(len(slow_features)*0.05))
            
                # determine good epsilon for DBSCAN  
                knn = np.sort(pairwise_distances(slow_features, slow_features))[:,minpc]        
                eps = np.percentile(knn, percentile)

                # cluster by EOD shape
                slow_clusters = DBSCAN(eps=eps, min_samples=minp).fit(slow_features).labels_

                if plot_steps:              
                    plt.subplot(4,len(np.unique(hight_labels)),len(np.unique(hight_labels))+i+1)
                    cols = ['b','r','g','y','m','c']
                
                    for j,c in enumerate(np.unique(slow_clusters)):
                        if c==-1:
                            plt.plot(slow_features[slow_clusters==c].T,alpha=0.1,c='k')
                        else:
                            plt.plot(slow_features[slow_clusters==c].T,alpha=0.1,c=cols[j%len(cols)])
                            plt.title('dbscan, h = %f'%(np.mean(eod_hights[hight_labels==hight_label])))               
                
            if len(fast_clusters) > 0:
                slow_clusters[slow_clusters==-1] = -np.max(fast_clusters) - 2
                clusters[ceod_widths<3*np.median(eod_widths)] = fast_clusters
                clusters[ceod_widths>=3*np.median(eod_widths)] = slow_clusters + np.max(fast_clusters) + 1
            else:
                clusters[ceod_widths>=3*np.median(eod_widths)] = slow_clusters

            
            if plot_steps:
                plt.subplot(4,len(np.unique(hight_labels)),len(np.unique(hight_labels))*2+i+1)

                for j,c in enumerate(np.unique(clusters)):
                    if c==-1:
                        plt.plot(current_snippets[clusters==c].T,alpha=0.1,c='k')
                    else:
                        plt.plot(current_snippets[clusters==c].T,alpha=0.1,c=cols[j%len(cols)])
                        plt.title('dbscan, h = %f'%(np.mean(eod_hights[hight_labels==hight_label])))
            
            if verbose > 0:
                if np.max(clusters) == -1:
                    print('No EODs in hight cluster %i'%hight_label)
                elif len(np.unique(clusters[clusters!=-1]))>1:
                    print('%i different EODs in hight cluster %i'%(len(np.unique(clusters[clusters!=-1])),hight_label))
            
            # update maxlab so that no clusters are overwritten
            clusters[clusters==-1] = -max_label - 1
            labels[hight_labels==hight_label] = clusters + max_label
            max_label = np.max(labels) + 1

        elif verbose>0:
            print('Too few EODs in hight cluster %i'%hight_label)

    if verbose>0:
        print('Clusters generated based on hight and shape: ')
        [print('N_{} = {:>4}'.format(int(l),len(labels[labels==l]))) for l in np.unique(labels[labels!=-1])]

    # remove double detections
    labels = remove_double_detections(labels,eod_widths,eod_x)

    # remove noise
    labels = remove_noise_and_artefacts(data, eod_x, eod_widths, labels,
                                              int(cutwidth*samplerate), interp_f, verbose-1)

    labels = remove_sparse_detections(labels,eod_widths,samplerate,verbose-1)

    labels = delete_unreliable_fish(labels,eod_widths,eod_x,verbose-1)

    labels = delete_wavefish(data,labels,eod_x,eod_widths)

    # re-merge clusters incorrectly separated by hight
    #labels = remerge(labels,snippets)


    if plot_steps==True:
        f, axes = plt.subplots(1,len(np.unique(labels)))
        if len(np.unique(labels))>1:
            for ax,l in zip(axes,np.unique(labels)):
                ax.plot(snippets[labels==l].T,c='b',alpha=0.1)
                ax.set_title(l)
        else:
            axes.plot(snippets.T,c='r',alpha=0.1)
            axes.set_title('-1')

    # return the cluster labels             
    return labels

def remerge(labels,snippets):
    ul = np.unique(labels[labels!=-1])
    if len(ul)>0:
        means = np.stack([np.mean(snippets[labels==l],axis=0) for l in ul])
        print(means)
        mapping = {}
        for i,m1 in enumerate(means):
            for j,m2 in enumerate(means):
                #plt.figure()
                #plt.plot(m1)
                #plt.plot(m2)
                #plt.show()
                if i!=j:
                    merge = 0.5*(np.std(m1-m2)**2) / (np.std(m1)**2+np.std(m2)**2)
                    if merge<0.05:
                        mapping[ul[i]] = ul[j]
                    #if merge
        print('merge mapping')
        print(mapping)
        for mk,mv in mapping.items():
            labels[labels==mk] = mv
    return labels


def find_window(data, eod_x, eod_peak_x, eod_widths, clusters, samplerate, verbose=0, w_factor=4):
    """ Extract mean EODs, EOD timepoints and unreliability score for each EOD cluster.

    Parameters
    ----------
        data: list of floats
            Raw recording data.
        eod_x: list of ints
            Locations of EODs in samples.
        eod_widths: list of ints
            EOD widths in samples.
        clusters: list of ints
            EOD cluster labels
        samplerate: float
            samplerate of recording  
        verbose (optional): int   
            Verbosity level.           
        w_factor (optional): float
            Multiplication factor for window used to extract noise and artefacts.
            Defaults to 4.

    Returns
    -------
        mean_eods: list of 2D arrays
            The average EOD for each detected fish. First column is time in seconds,
            second column the mean eod, third column the standard error.
        eod_times: list of 1D arrays
            For each detected fish the times of EOD peaks in seconds.
        unreliability: list of floats
            Measure of unreliability of extracted fish. Numbers above 0.1 indicate 
            wavefish or unreliable pulsefish clusters.
    """

    mean_eods, eod_times, eod_peak_times, eod_hights = [], [], [], []
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

            # leave out unreliability, also in thunderfish!
            unreliability.append(0)#np.max(np.median(eod_widths[clusters==cluster])/np.diff(eod_x[cluster==clusters])))
           
            if verbose>0:
                print('unreliability score: %f'%unreliability[-1])

    return [m for _,m in sorted(zip(eod_hights,mean_eods))], [t for _,t in sorted(zip(eod_hights,eod_times))], [pt for _,pt in sorted(zip(eod_hights,eod_peak_times))], [ur for _,ur in sorted(zip(eod_hights,unreliability))]

def delete_unreliable_fish(clusters,eod_widths,eod_x,verbose):
    for cluster in np.unique(clusters[clusters!=-1]):
        
        if np.max(np.median(eod_widths[clusters==cluster])/np.diff(eod_x[cluster==clusters])) > 0.5:
            if verbose>0:
                print('deleting unreliable cluster %i, score=%f'%(cluster,np.max(np.median(eod_widths[clusters==cluster])/np.diff(eod_x[cluster==clusters]))))
            clusters[clusters==cluster] = -1
    return clusters


def delete_wavefish(data, clusters, eod_x, eod_widths, w_factor=8):
    for cluster in np.unique(clusters):
        if cluster < 0:
            continue
        cutwidth = np.mean(eod_widths[clusters==cluster])*w_factor
        current_x = eod_x[(eod_x>cutwidth) & (eod_x<(len(data)-cutwidth))]
        current_clusters = clusters[(eod_x>cutwidth) & (eod_x<(len(data)-cutwidth))]

        snippets = np.vstack([data[int(x-cutwidth):int(x+cutwidth)] for x in current_x[current_clusters==cluster]])
        mean_eod = np.mean(snippets, axis=0)
        pk, tr = detect_peaks(np.pad(mean_eod,(1,1), 'constant'),
                              0.25*(np.max(mean_eod)-np.min(mean_eod)))

        if len(pk) + len(tr)>4:
            clusters[clusters==cluster] = -1
    return clusters


def remove_double_detections(clusters,eod_widths,eod_x):
    for c in np.unique(clusters):
        if c < 0:
            continue
        cc = clusters[clusters==c]
        isi = np.append(np.diff(eod_x[clusters == c]),1000)

        cc[np.median(eod_widths[clusters==c])/isi > 0.9] = -1
        clusters[clusters==c] = cc
    return clusters

def remove_sparse_detections(clusters,eod_widths, samplerate, verbose, factor = 0.004):
    for c in np.unique(clusters):
        if c!=-1:
            n = len(clusters[clusters==c])
            w = np.median(eod_widths[clusters==c])/samplerate
            if n*w < factor:
                if verbose>0:
                    print('cluster %i is too sparse'%c)
                clusters[clusters==c] = -1
    return clusters

    
def remove_noise_and_artefacts(data, eod_x, eod_widths, clusters, original_cutwidth, int_f, verbose=0,
                               w_factor=2, noise_threshold=0.75, artefact_threshold=0.75, cutoff_f=10000):
    """ Remove EOD clusters that are too noisy or result from artefacts

    Parameters
    ----------
        data: list of floats
            Raw recording data.
        eod_x: list of ints
            Locations of EODs in samples.
        eod_widths: list of ints
            EOD widths in samples.
        clusters: list of ints
            EOD cluster labels
        original_cutwidth : int
            Width that was used for feature extraction.
        verbose (optional): int
            Verbosity level.
        w_factor (optional): float
            Multiplication factor for windowsize to use for noise and artefact detection.
            Defaults to 2.

        noise_threshold (optional): float
            Threshold that separates noisy clusters from clean clusters.
            Defaults to 0.003.
        artefact_threshold (optional): float
            Threshold that separates artefact from clean pulsefish clusters.
            Defaults to 0.8.

    Returns
    -------
        clusters: list of ints
            Cluster labels, where noisy cluster and clusters with artefacts have been set to zero.
    """

    for cluster in np.unique(clusters):
        if cluster!=-1:

            cutwidth = np.max([np.mean(eod_widths[clusters==cluster])*w_factor, int(original_cutwidth/2)])

            # extract snippets
            current_x = eod_x[(eod_x>cutwidth) & (eod_x<(len(data)-cutwidth))]
            current_clusters = clusters[(eod_x>cutwidth) & (eod_x<(len(data)-cutwidth))]

            snippets = np.vstack([data[int(x-cutwidth):int(x+cutwidth)] for x in current_x[current_clusters==cluster]])
            snippets = subtract_slope(snippets)

            mean_eod = np.mean(snippets, axis=0)
            eod_std = np.std(snippets, axis=0)
            noise_ratio = np.mean(np.abs(mean_eod)/np.mean(np.abs(np.diff(snippets,axis=0)),axis=0))
            noise_ratio2 = np.var(mean_eod)/(np.mean(eod_std))
            noise_ratio3 = np.mean(stats.sem(snippets,axis=0))#/np.mean(eod_std)

            cut_fft = int(len(np.fft.fft(mean_eod))/2)
            
            low_frequency_ratio = np.sum(np.abs(np.fft.fft(mean_eod))[1:int(cut_fft/(2*int_f))])/np.sum(np.abs(np.fft.fft(mean_eod))[1:int(cut_fft)])
            '''
            plt.figure()
            plt.plot(mean_eod)
            plt.figure()
            plt.plot(np.abs(np.fft.fft(mean_eod))[1:int(cut_fft)])
            plt.plot(np.abs(np.fft.fft(mean_eod))[1:int(cut_fft/(2*int_f))])
            
            plt.show()
            '''           

            if noise_ratio < noise_threshold: # or noise_ratio2>0.03:# or noise_ratio3 > 0.1:
                clusters[clusters==cluster] = -1
                if verbose>0:
                    print('Deleting cluster %i, which has an SNR of %f'%(cluster,noise_ratio))
                    #plt.figure()
                    #plt.plot(snippets.T,c='b',alpha=0.1)
                    #plt.title('Deleting cluster %i, which has an SNR of %f'%(cluster,noise_ratio))
            elif low_frequency_ratio < artefact_threshold:
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


def delete_moving_fish(clusters, eod_t, T, eod_hights, verbose=0, dt=1, stepsize=0.1, min_eod_factor=1):
    """
    Use a sliding window to detect the minimum number of fish detected simultaneously, 
    then delete all other EOD clusters.

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
        verbose (optional): int
            Verbosity level.
        dt (optional): float
            Sliding window size (in seconds).
        stepsize (optional): float
            Sliding window stepsize (in seconds).
        N (optional): int
            Minimum cluster size.

    Returns
    -------
        clusters: list of ints
            Cluster labels, where deleted clusters have been set to -1.
    """

    # initialize variables
    min_clusters = 100
    average_hight = 0
    keep_clusters = []
    ignore_steps = np.zeros(len(np.arange(0, T-dt+stepsize, stepsize)))
    window_start = 0
    window_end = dt

    # only compute on clusters with minimum length.
    labels, size = unique_counts(clusters)
    remove_clusters = np.isin(clusters,labels[size<T*min_eod_factor])
    clusters[remove_clusters] = -1

    for i,t in enumerate(np.arange(0, T-dt+stepsize, stepsize)):
        current_clusters = clusters[(eod_t>=t)&(eod_t<t+dt)&(clusters!=-1)]

        if len(np.unique(current_clusters))==0:
            # ignore all windows with 0 fish.
            # ignore current step, but also the next N
            ignore_steps[i-int(dt/stepsize):i+int(dt/stepsize)] = 1
            
            if verbose>0:
                print('Gap in recording at T=%.2f:%.2f'%(t,t+dt))
        #else:
        #    all_lens.append(len(np.unique(current_clusters)))

    # ignore couldnt be set for begin or end of recording. so only slide from 1 to 7?

    # sliding window
    for t,ignore_step in zip(np.arange(0, T-dt+stepsize, stepsize), ignore_steps):
        if t>1:
            current_clusters = clusters[(eod_t>=t)&(eod_t<t+dt)&(clusters!=-1)]

            if (len(np.unique(current_clusters)) <= min_clusters) and (ignore_step==0) and (len(np.unique(current_clusters) !=1)):
            
            #if len(np.unique(current_clusters)) == np.median(all_lens):
                # only update keep_clus if min_clus is less than the last save
                # or equal but the hights are higher.
                current_labels = np.isin(clusters, np.unique(current_clusters))
                current_hight = np.mean(eod_hights[current_labels])

                if (current_hight > average_hight) or (len(np.unique(current_clusters)) < min_clusters):
                    keep_clusters = np.unique(current_clusters)
                    min_clusters = len(np.unique(current_clusters))
                    average_hight = current_hight
                    window_start = t
                    window_end = t+dt
    
    if verbose>0:
        print('Estimated nr of fish in recording: %i'%min_clusters)

    # delete all clusters that are not selected
    clusters[np.invert(np.isin(clusters, keep_clusters))] = -1

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

def plot_all(data, eod_times, fs, mean_eods, rs):
    
    cmap = plt.get_cmap("tab10")
    
    fig = plt.figure(constrained_layout=True,figsize=(10,5))
    if len(eod_times) > 0:
        gs = GridSpec(2, len(eod_times), figure=fig)
        ax = fig.add_subplot(gs[0,:])
        ax.plot(np.arange(len(data))/fs,data,c='k',alpha=0.3)
        
        for i,t in enumerate(eod_times):
            ax.plot(t,data[(t*fs).astype('int')],'o',label=i+1,ms=10,c=cmap(i))
            
        #for i,t in enumerate(eod_p_times):
        #    ax.plot(t,data[(t*fs).astype('int')],'o',label=i+1,c=cmap(i))
        ax.set_xlabel('time [s]')
        ax.set_ylabel('amplitude [V]')
        #ax.axis('off')

        for i, (m,r) in enumerate(zip(mean_eods,rs)):
            ax = fig.add_subplot(gs[1,i])
            ax.plot(1000*m[0], 1000*m[1], c='k')
            ax.fill_between(1000*m[0],1000*(m[1]-m[2]),1000*(m[1]+m[2]),color=cmap(i))
            ax.set_xlabel('time [ms]')
            ax.set_ylabel('amplitude [mV]') 
            if r >0.1:
                ax.set_title('unreliable or wave')
            #ax.axis('off')
            #ax.set_ylim([np.min(data),np.max(data)])
            #ax.set_xlim([np.min(),np.max()])
    else:
        plt.plot(np.arange(len(data))/fs,data,c='k',alpha=0.3)
