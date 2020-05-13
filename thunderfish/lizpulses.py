import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
from sklearn.metrics import pairwise_distances, silhouette_score

from .eventdetection import detect_peaks
from .pulse_tracker_helper import makeeventlist, discardnearbyevents, discard_connecting_eods


# upgrade numpy fumctions:

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
        #ar = np.asanyarray(ar).flatten()
        ar.sort()
        mask = np.empty(ar.shape, dtype=np.bool_)
        mask[:1] = True
        mask[1:] = ar[1:] != ar[:-1]
        idx = np.concatenate(np.nonzero(mask) + ([mask.size],))
        return ar[mask], np.diff(idx)  


def extract_eod_times(data,peakwidth,cw,tf=2):
    '''
    Extract peaks from data which could be EODs.

    Input
    -------------------------------------------
    data: list of floats
        data to extract peaks from
    peakwidth: int
        maximum peakwidth for extracting single EODs
    cw: int
        cutwidth. Do not return peak indices that are don't
        allow full extraction of this EOD waveform

    Returns
    --------------------------------------------
    x_peak:     array of ints
        indices of peaks in data
    x_trough:   array of ints
        indices of troughs in data for each x_peak
    eod_hight:  array of floats
        EOD hights for each x_peak
    eod_width:  array of ints
        EOD widths for each x_peak (in samples)
    '''
    
    # make this std and make 2 a factor.
    #thresh = np.std(data)*tf
    thresh = np.mean(np.abs(data))*tf
    pk, tr = detect_peaks(data, thresh)

    if len(pk)==0:
        return [], [], [], []
    else:
        peaks = makeeventlist(pk,tr,data,peakwidth)
        peakindices, _, _ = discardnearbyevents(peaks[0],peaks[2],peakwidth)
        pk, tr, hs, ws = discard_connecting_eods(peaks[0][peakindices.astype('int')].astype('int'), peaks[-1][peakindices.astype('int')].astype('int'), peaks[2][peakindices.astype('int')], peaks[3][peakindices.astype('int')])
        
        # only take those where the cutwidth does not casue issues
        i = np.where((pk>int(cw/2)) & (pk<(len(data)-int(cw/2))) & (tr>int(cw/2)) & (tr<(len(data)-int(cw/2))))[0]

        return pk[i], tr[i], hs[i], ws[i]


def cluster(idx_arr,h_arr,w_arr,data,samplerate,cw,minp,percentile,npc,ngaus,n_init,m_iter=100):
    '''
    Cluster EODs. First cluster on EOD hights using a Bayesian Gaussian Mixture model, 
    then cluster on EOD wave shape using DBSCAN.

    Input
    ------------
    idx_arr : list of ints
        locations of EODs (as indices)
    h_arr : list of floats
        EOD hights
    w_arr: list of ints
        EOD widths
    data : list of floats
        raw recording data
    samplerate : int or float
        sample rate of raw data

    Parameters
    -----------------
    cw : float
        cut width for extracting snippets for clustering based on EOD shape
    percentile: float
        percentile for determining epsilon from knn distribution where k=minp
    npc : int
        number of PCs to use for PCA
    minp : int
        minimun number of point for core cluster (DBSCAN)
    ngaus : int
        maximun number of gaussians to fit for clustering on EOD hights
    n_init: int
        number of initializations for BayesianGaussianMixture method
    m_iter: int
        maximum number of iterations for BayesianGausssianMixture method

    Returns
    -------------------
    al : list of ints
        EOD cluster labels based on hight and EOD waveform 

    '''

    # initiate labels based on hight
    l = np.ones(h_arr.shape)
    
    if ((np.max(h_arr) - np.min(h_arr))/np.max(h_arr)) > 0.25:

        # classify by height
        bgm = BayesianGaussianMixture(ngaus,max_iter=m_iter,n_init=n_init)
        l = bgm.fit_predict(h_arr.reshape(-1,1))

        # if any of the clusters merged have very little height difference, merge them.
        if len(np.unique(l))>1:
            for ll in np.unique(l):
                if ((np.max(h_arr[l!=ll]) - np.min(h_arr[l!=ll]))/np.max(h_arr[l!=ll])) < 0.25:
                    l[l!=ll] = np.max(l)+1

    # now cluster based on waveform
    al = np.ones(len(l))*-1    
        
    # extract snippets
    snippets = np.vstack([data[int(idx-cw*samplerate/2):int(idx+cw*samplerate/2)] for idx in idx_arr]) 
    
    # keep track of the labels so that no labels are overwritten
    maxlab = 0
            
    for hl in np.unique(l):
        if len(l[l==hl])>minp:

            # extract snippets, idxs and hs for this hight cluster
            csnippets = StandardScaler().fit_transform(snippets[l==hl])
            
            # extract relevant snippet features
            pca = PCA(npc).fit(csnippets).transform(csnippets)
            
            # determine good epsilon  
            knn = np.sort(pairwise_distances(pca,pca))[:,minp]        
            eps = np.percentile(knn,percentile)

            # cluster by EOD shape
            c = DBSCAN(eps=eps, min_samples=minp).fit(pca).labels_

            # cluster again without scaling (sometimes this works better wrt scaling)
            csnippets_ns = snippets[l==hl]
            pca = PCA(npc).fit(csnippets_ns).transform(csnippets_ns)
            knn = np.sort(pairwise_distances(pca,pca))[:,minp]        
            eps = np.percentile(knn,percentile)
            c_ns = DBSCAN(eps=eps, min_samples=minp).fit(pca).labels_

            # remove noise and artefacts from ns clusters
            c_ns = remove_noise_and_artefacts(data,idx_arr[l==hl],w_arr[l==hl],c_ns,int(cw*samplerate))
            # remove noise and artefacts from scaled clusters
            c = remove_noise_and_artefacts(data,idx_arr[l==hl],w_arr[l==hl],c,int(cw*samplerate))
              
            # merge results for scaling and without scaling
            c, _ = merge_clusters(c,c_ns,idx_arr[l==hl],idx_arr[l==hl])

            # remove noise after merging
            c = remove_noise_and_artefacts(data,idx_arr[l==hl],w_arr[l==hl],c,int(cw*samplerate))

            # update maxlab so that no clusters are overwritten
            c[c==-1] = -maxlab-1
            al[l==hl] = c + maxlab
            maxlab = np.max(al)+1

    # return the overall clusters (al) and the clusters based on hight (l)                
    return al


def find_window(data,idx_arr,w_arr,c,samplerate):
    '''
    Select window for extracting mean EODs

    Input
    --------------
        data : list of floats
            raw recording data
        idx_arr : list of ints
            EOD indices
        w_arr : list of ints
            EOD widths
        c : list of ints
            EOD cluster labels
        samplerate : float
            samplerate of recording                

    Output
    --------------
        mean_eods: list of 2D arrays
            For each detected fish the average of the EOD snippets. First column is time in seconds,
            second column the mean eod, third column the standard error.
        eod_times: list of 1D arrays
            For each detected fish the times of EOD peaks in seconds.
    '''

    ms,ts = [],[]

    for l in np.unique(c):
        if l!=-1:
            cw = np.mean(w_arr[c==l])*4
            c_i = idx_arr[(idx_arr>cw) & (idx_arr<(len(data)-cw))]
            cc = c[(idx_arr>cw) & (idx_arr<(len(data)-cw))]

            w = np.vstack([data[int(idx-cw):int(idx+cw)] for idx in c_i[cc==l]])
            m = np.mean(w,axis=0)
            
            t = np.arange(len(m))/samplerate - cw/samplerate
            m = np.vstack([t,m,np.std(w,axis=0)])

            ms.append(m)
            ts.append(idx_arr[c==l]/samplerate)
                
    return ms, ts

    
def remove_noise_and_artefacts(data,idx_arr,w_arr,c,o_cw,nt=0.003,at=0.8):
    '''
    remove EOD clusters that are too noisy or result from artefacts

    Input
    --------------
        data : list of floats
            raw recording data
        idx_arr : list of ints
            EOD indices
        w_arr : list of ints
            EOD widths
        c : list of ints
            EOD cluster labels
        o_cw : int
            original cutwidth that was used for feature extraction

    Parameters
    ---------------
        nt : float
            noise threshold
        at : float
            artefact threshold

    Output
    --------------
        c : list of ints
            cluster labels, where noisy cluster and 
            clusters with artefacts have been set to zero
    '''

    for cc in np.unique(c):
        if cc!=-1:

            cw = np.max([np.mean(w_arr[c==cc])*2,int(o_cw/2)])

            # extract snippets
            c_i = idx_arr[(idx_arr>cw) & (idx_arr<(len(data)-cw))]
            cur_c = c[(idx_arr>cw) & (idx_arr<(len(data)-cw))]

            snippets = np.vstack([data[int(idx-cw):int(idx+cw)] for idx in c_i[cur_c==cc]])

            m = np.mean(snippets,axis=0)
            v = np.std(snippets,axis=0)
            r = np.var(m)/np.mean(v)
            stop = int(len(np.fft.fft(m))/2)

            if r<nt:
                c[c==cc] = -1
            elif np.sum(np.abs(np.fft.fft(m))[1:int(stop/2)])/np.sum(np.abs(np.fft.fft(m))[1:int(stop)]) < at:
                c[c==cc] = -1

    return c


def merge_clusters(c_p,c_t,idx_arr,idx_t_arr): 
    '''
    merge clusters of two clustering methods

    Input
    --------------
        c_p : list of ints
            EOD cluster labels for cluster method 1.
        c_t : list of ints
            EOD cluster labels for cluster method 2.
        idx_arr : list of ints
            indices of EODs for cluster method 1 (c_p)
        idx_t_arr : list of ints
            indices of EODs for cluster method 2 (c_t)

    Output
    --------------
        c : list of ints
            merged clusters
        idx_arr : list of ints
            merged cluster indices
    '''

    # these arrays become 1 for each EOD that is chosen from that array
    c_pd = np.zeros(len(c_p))
    c_td = np.zeros(len(c_t))

    # add n to one of the cluster lists to avoid overlap
    c_t[c_t!=-1] = c_t[c_t!=-1] + np.max(c_p) + 1
    
    # loop untill done
    while True:

        # compute unique clusters and cluster sizes
        # of cluster that have not been iterated over
        cu_p, clen_p = unique_counts(c_p[(c_p!=-1) & (c_pd == 0)])
        cu_t, clen_t = unique_counts(c_t[(c_t!=-1) & (c_td == 0)])

        # if all clusters are done, break from loop
        if len(clen_p) == 0 and len(clen_t) == 0:
            break

        # if the biggest cluster is in c_p, keep this one and discard all clusters on the same indices in c_t
        elif np.argmax([np.max(np.append(clen_p,0)), np.max(np.append(clen_t,0))]) == 0:
            # remove all the mappings from the other indices
            cm,ccount = unique_counts(c_t[c_p==cu_p[np.argmax(clen_p)]])
            for ccm in cm:
                c_t[c_t==ccm] = -1
            c_pd[c_p==cu_p[np.argmax(clen_p)]] = 1

        # if the biggest cluster is in c_t, keep this one and discard all mappings in c_p
        elif np.argmax([np.max(np.append(clen_p,0)), np.max(np.append(clen_t,0))]) == 1:
            cm, ccount = unique_counts(c_p[c_t==cu_t[np.argmax(clen_t)]])
            for ccm in cm:
                c_p[c_p==ccm] = -1
            c_td[c_t==cu_t[np.argmax(clen_t)]] = 1

    # combine results    
    c = (c_p+1)*c_pd + (c_t+1)*c_td - 1
    idx_arr = (idx_arr)*c_pd + (idx_t_arr)*c_td

    return c, idx_arr


def delete_moving_fish(c, ts, T, h_arr, dt=1, stepsize=0.1, N=8):
    '''
    Use a sliding window to detect the minimum number of fish detected simultaneously.
    Then delete all other EOD detections.

    Input
    --------------
        c : list of ints
            EOD cluster labels
        ts : list of floats
            timepoints of the EODs (in seconds)
        T : float
            length of recording (in seconds)
        h_arr : list of floats
            EOD amplitudes
    Parameters
    ------------------------------
        dt : float
            sliding window size (in seconds)
        stepsize : float
            sliding window stepsize (in seconds)
        N : int
            minimum cluster size

    Output
    --------------
        c : list of ints
            clusters, where deleted clusters have been set to -1
    '''

    # initialize variables
    min_clus = 100
    av_hight = 0
    keep_clus = []

    # only compute on clusters with minimum length of length N.
    u,uc = unique_counts(c)
    m = np.isin(c,u[uc<N])
    c[m] = -1

    # sliding window
    for t in np.arange(0,T-dt+stepsize,stepsize):
        current_clusters = c[(ts>=t)&(ts<t+dt)&(c!=-1)]
        if len(np.unique(current_clusters)) <= min_clus:
            
            # only update keep_clus if min_clus is less than the last save
            # or equal but the hights are higher.
            mask = np.isin(c,np.unique(current_clusters))
            c_hight = np.mean(h_arr[mask])

            if (c_hight > av_hight) or (len(np.unique(current_clusters)) < min_clus):
                keep_clus = np.unique(current_clusters)
                min_clus = len(np.unique(current_clusters))
                av_hight = c_hight

    # delete all clusters that are not selected
    for cc in np.unique(c):
        if cc not in keep_clus:
            c[c==cc] = -1

    return c


def extract_pulsefish(data, samplerate, pw=0.002, cw=0.001, percentile=75, npc=3, minp=10, ngaus=4, n_init=1, m_iter=1000):
    """
    Takes recording data containing one or more pulsefish 
    and extracts the mean EOD for each fish present in the recording.
    
    Parameters
    ----------
    data: 1-D array of float
        The data to be analysed.
    samplerate: float
        Sampling rate of the data in Hertz.

    OPTIONAL:
    pw : float
        peak width for peak detection in seconds.
    cw : float
        cut width for extracting snippets for clustering based on EOD shape
    percentile: float
        percentile for determining epsilon from knn distribution where k=minp
    npc : int
        number of PCs to use for PCA
    minp : int
        minimun number of point for core cluster (DBSCAN)
    ngaus : int
        maximun number of gaussians to fit for clustering on EOD hights
    n_init: int
        number of initializations for BayesianGaussianMixture method
    m_iter: int
        maximum number of iterations for BayesianGausssianMixture method
        
    Returns
    -------
    mean_eods: list of 2D arrays
        For each detected fish the average of the EOD snippets. First column is time in seconds,
        second column the mean eod, third column the standard error.
    eod_times: list of 1D arrays
        For each detected fish the times of EOD peaks in seconds.
    """
    
    mean_eods, eod_times = [], []
    
    # extract peaks
    idx_arr, idx_t_arr, h_arr, w_arr = extract_eod_times(data, pw*samplerate,cw*samplerate)

    if len(idx_arr)>0:

        # cluster on peaks
        c_p = cluster(idx_arr,h_arr,w_arr,data,samplerate,cw,minp,percentile,npc,ngaus,n_init,m_iter) 

        # cluster on troughs
        c_t = cluster(idx_t_arr,h_arr,w_arr,data,samplerate,cw,minp,percentile,npc,ngaus,n_init,m_iter)

        # merge peak and trough clusters
        c, c_idx_arr = merge_clusters(c_p,c_t,idx_arr,idx_t_arr)

        # remove noise from merged clusters
        c = remove_noise_and_artefacts(data,c_idx_arr,w_arr,c,int(cw*samplerate))

        # delete the moving fish
        c = delete_moving_fish(c,c_idx_arr/samplerate,len(data)/samplerate, h_arr)

        # extract mean eods
        mean_eods, eod_times = find_window(data,c_idx_arr,w_arr,c,samplerate)
    
    return mean_eods, eod_times
