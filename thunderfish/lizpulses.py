import sys
import numpy as np
import matplotlib.pyplot as plt
from thunderfish.dataloader import load_data
from thunderfish.bestwindow import best_window
import pulse_tracker_helper as pth
import thunderfish.eventdetection as ed

from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN

from matplotlib.gridspec import GridSpec

from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
from sklearn.metrics import pairwise_distances, silhouette_score
from sklearn.cluster import DBSCAN, OPTICS
import scipy.stats as stats

def extract_eod_times(data,peakwidth):
    
    thresh = np.mean(np.abs(data))*2
    pk, tr = ed.detect_peaks(data, thresh)

    if len(pk)==0:
        return [], [], [], []
    else:
        peaks = pth.makeeventlist(pk,tr,data,peakwidth)
        peakindices, _, _ = pth.discardnearbyevents(peaks[0],peaks[2],peakwidth)
        return peaks[0][peakindices.astype('int')].astype('int'), peaks[-1][peakindices.astype('int')].astype('int'), peaks[2][peakindices.astype('int')],peaks[3][peakindices.astype('int')] 

def cluster(idx_arr,h_arr,w_arr,data,cw,samplerate,minp,percentile,npc,ngaus,plot_steps=False):

    l = np.ones(h_arr.shape)
    
    if ((np.max(h_arr) - np.min(h_arr))/np.max(h_arr)) > 0.25:

        # classify by height
        bgm = BayesianGaussianMixture(ngaus,max_iter=100,n_init=10)
        l = bgm.fit_predict(h_arr.reshape(-1,1))

        # if any of the clusters merged have very little height difference, merge them.
        if len(np.unique(l))>1:
            for ll in np.unique(l):
                if ((np.max(h_arr[l!=ll]) - np.min(h_arr[l!=ll]))/np.max(h_arr[l!=ll])) < 0.25:
                    l[l!=ll] = np.max(l)+1

    if plot_steps == True:
        mean_eods, eod_times, _ = find_window(data,idx_arr,l,h_arr,rm=False)
        print('clustering based on hight')
        plot_all(data,ts,samplerate,ms,vs)

    # now cluster based on waveform
    al = np.ones(len(l))*-1    
        
    # extract snippets
    snippets = np.stack([data[int(idx-cw*samplerate/2):int(idx+cw*samplerate/2)] for idx in idx_arr]) 
    
    # keep track of the labels so that no labels are overwritten
    maxlab = 0
            
    for hl in np.unique(l):
        if len(l[l==hl])>minp:

            # extract snippets, idxs and hs for this hight cluster
            csnippets = StandardScaler().fit_transform(snippets[l==hl])
            cidx_arr = idx_arr[l==hl]
            ch_arr = h_arr[l==hl]

            # extract relevant snippet features
            pca = PCA(npc).fit(csnippets).transform(csnippets)
            
            # determine good epsilon  
            knn = np.sort(pairwise_distances(pca,pca))[:,minp]        
            eps = np.percentile(knn,percentile)

            # cluster by EOD shape
            c = DBSCAN(eps=eps, min_samples=minp).fit(pca).labels_

            if plot_steps == True:
                mean_eods, eod_times, _ = find_window(data,cidx_arr,c,ch_arr,rm=False)
                print('clustering on scaled eods')
                plot_all(data,ts,samplerate,ms,vs)

            # cluster again without scaling (sometimes this works better wrt scaling)
            csnippets_ns = snippets[l==hl]
            pca = PCA(npc).fit(csnippets_ns).transform(csnippets_ns)
            knn = np.sort(pairwise_distances(pca,pca))[:,minp]        
            eps = np.percentile(knn,percentile)
            c_ns = DBSCAN(eps=eps, min_samples=minp).fit(pca).labels_
            
            if plot_steps == True:
                mean_eods, eod_times = find_window(data,cidx_arr,c_ns,ch_arr,rm=False)
                print('clustering on non-scaled eods')
                plot_all(data,ts,samplerate,ms,vs)

            # merge results for scaling and without scaling
            _,_,_,c = merge_clusters(c,c_ns,cidx_arr,cidx_arr,ch_arr,data,samplerate)

            if plot_steps == True:
                mean_eods, eod_times = find_window(data,cidx_arr,c,ch_arr,rm=False)
                print('merged scale and non-scaled')
                plot_all(data,ts,samplerate,ms,vs)

            # update maxlab so that no clusters are overwritten
            c[c==-1] = -maxlab-1
            al[l==hl] = c + maxlab
            maxlab = np.max(al)+1

    # return the overall clusters (al) and the clusters based on hight (l)                
    return al, l
    
def find_window(data,idx_arr,c,h_arr,samplerate,sp=True,rm=True):
    
    remove_clusters = []
    ms,ts = [], []
    
    lw=10
    
    for l in np.unique(c):
        if l != -1:  

            rs = []

            for rw in range(10,100):

                # try different windows and different time shifts.
                # use only indexes that fit with the cutwidth

                c_i = idx_arr[c==l][(idx_arr[c==l]>lw) & (idx_arr[c==l]<(len(data)-rw))]

                w = np.stack([data[int(idx-lw):int(idx+rw)] for idx in c_i])

                m = np.mean(w,axis=0)
                v = np.std(w,axis=0)
                r = np.var(m)/np.mean(v)

                rs.append(r)

            rw = (np.argmax(rs) + 10)

            rs = []

            for lw in range(10,100):
                # try different windows and different time shifts.
                c_i = idx_arr[c== l][(idx_arr[c== l]>lw) & (idx_arr[c== l]<(len(data)-rw))]
                w = np.stack([data[int(idx-lw):int(idx+rw)] for idx in c_i])

                m = np.mean(w,axis=0)
                v = np.std(w,axis=0)
                r = np.var(m)/np.mean(v)

                rs.append(r)

            lw = (np.argmax(rs) + 10)
            
            sp_c = 0

            if sp == True:
                # check if any bigger peaks are within snippet selection
                # are there any idxs from a different clusters in the selected windows?
                c_i = idx_arr[c== l][(idx_arr[c== l]>lw) & (idx_arr[c== l]<(len(data)-rw))]
                cur_idxs = np.stack([np.arange(int(idx-lw*2),int(idx+rw*2)) for idx in c_i])
                alien_peaks = idx_arr[(c!=-1) & (c!=l)]

                for i,cii in enumerate(cur_idxs):
                    for ci in cii:
                        if ci in alien_peaks:

                            # check if this foreign peak is significantly higher.
                            if np.min(h_arr[i]/h_arr[(c!=-1) & (c!=l)][alien_peaks==ci]) < 0.2:
                                if rm == True:                                
                                    sp_c = sp_c + 1

                                #sidepeak = True

            # if the error is small enough, it is probably not noise
            c_i = idx_arr[c== l][(idx_arr[c== l]>lw*4) & (idx_arr[c== l]<(len(data)-rw*3))]
            w = np.stack([data[int(idx-lw*4):int(idx+rw*3)] for idx in c_i])
            m = np.mean(w,axis=0)
            stop = int(len(np.fft.fft(m))/2)

            # check if it is not either an artefact, noise or a sidepeak of a bigger EOD
            if (np.max(rs) > 0.005 and (sp_c<2) and (np.sum(np.abs(np.fft.fft(m))[1:int(stop/2)])/np.sum(np.abs(np.fft.fft(m))[1:int(stop)]) > 0.75)) or (rm==False):
                # time, mean, ste
                t = np.arange(len(m))/samplerate - lw*4/samplerate
                m = np.stack([t,m,stats.sem(w,axis=0)])

                ms.append(m)
                ts.append(idx_arr[c==l]/samplerate)
                
            else:
                remove_clusters.append(l)
                
    return ms, ts, remove_clusters
    
def merge_clusters(c_p,c_t,idx_arr,idx_t_arr, h_arr, data, samplerate): 

    # first remove all clusters with high variance or sidepeaks or artefacts from both clusters
    _,_,remove_clusters = find_window(data,idx_arr,c_p,h_arr,samplerate,sp=False)
    for rc in remove_clusters:
        c_p[c_p==rc] = -1
    _,_,remove_clusters = find_window(data,idx_t_arr,c_t,h_arr,samplerate,sp=False)
    for rc in remove_clusters:
        c_t[c_t==rc] = -1


    # now merge them.
    c_pd = np.zeros(len(c_p))
    c_td = np.zeros(len(c_t))

    try:
        c_t[c_t!=-1] = c_t[c_t!=-1] + np.max(c_p) + 1
    except:
        pass
    
    c_pd = np.zeros(len(c_p))
    c_td = np.zeros(len(c_t))   

    while True:
        cu_p, clen_p = np.unique(c_p[(c_p!=-1) & (c_pd == 0)],return_counts=True)
        cu_t, clen_t = np.unique(c_t[(c_t!=-1) & (c_td == 0)],return_counts=True)     

        if len(clen_p) == 0 and len(clen_t) == 0:
            break
        elif len(clen_p) == 0:
            cm, ccount = np.unique(c_p[c_t==cu_t[np.argmax(clen_t)]],return_counts=True)
            for ccm in cm:
                c_p[c_p==ccm] = -1
            c_td[c_t==cu_t[np.argmax(clen_t)]] = 1

        elif len(clen_t) == 0:
            # remove all the mappings from the other indices
            cm,ccount = np.unique(c_t[c_p==cu_p[np.argmax(clen_p)]],return_counts=True)
            for ccm in cm:
                c_t[c_t==ccm] = -1
            c_pd[c_p==cu_p[np.argmax(clen_p)]] = 1

        elif np.argmax([np.max(clen_p), np.max(clen_t)]) == 0:

            # remove all the mappings from the other indices
            cm,ccount = np.unique(c_t[c_p==cu_p[np.argmax(clen_p)]],return_counts=True)
            
            for ccm in cm:
                c_t[c_t==ccm] = -1
            c_pd[c_p==cu_p[np.argmax(clen_p)]] = 1

        else:
            cm, ccount = np.unique(c_p[c_t==cu_t[np.argmax(clen_t)]],return_counts=True)

            for ccm in cm:
                c_p[c_p==ccm] = -1
            c_td[c_t==cu_t[np.argmax(clen_t)]] = 1

    i = np.argsort(np.append(idx_arr[c_p != -1],idx_t_arr[c_t != -1]))
    c_idx_arr = np.append(idx_arr[c_p != -1],idx_t_arr[c_t != -1])[i]
    c = np.append(c_p[c_p != -1],c_t[c_t != -1])[i]
    c_h_arr = np.append(h_arr[c_p != -1],h_arr[c_t != -1])[i]
    
    c_o = (c_p+1)*c_pd + (c_t+1)*c_td - 1

    # now remove clusters that after merging have too much variance
    _,_,remove_clusters = find_window(data,c_idx_arr,c,c_h_arr,samplerate)
    for rc in remove_clusters:
        c[c==rc] = -1

    return c, c_idx_arr, c_h_arr, c_o

def delete_moving_fish(c, ts, T, h_arr, dt=1, stepsize=0.1):
    # initialize vars
    min_clus = 100
    av_hight = 0
    keep_clus = []

    # only compute on clusters with minimum length of 8.
    u,uc = np.unique(c,return_counts=True)
    m = np.isin(c,u[uc>=8])
    c = c[m]
    ts = ts[m]
    h_arr = h_arr[m]

    # sliding window
    for t in np.arange(0,T-dt+stepsize,stepsize):
        current_clusters = c[(ts>=t)&(ts<t+dt)&(c!=-1)]
        if len(np.unique(current_clusters)) <= min_clus:
            
            # only update keep_clus if min_clus is less than the last save
            # or equal but the hights are hihger.

            mask = np.isin(c,np.unique(current_clusters))
            c_hight = np.mean(h_arr[mask])

            if (c_hight > av_hight) or (len(np.unique(current_clusters)) < min_clus):
                keep_clus = np.unique(current_clusters)
                min_clus = len(np.unique(current_clusters))
                av_hight = c_hight
    
    return keep_clus


def plot_all(data, eod_times, fs, mean_eods):
    
    cmap = plt.get_cmap("tab10")
    
    fig = plt.figure(constrained_layout=True,figsize=(10,5))
    if len(eod_times) > 0:
        gs = GridSpec(2, len(eod_times), figure=fig)
        ax = fig.add_subplot(gs[0,:])
        ax.plot(np.arange(len(data))/fs,data,c='k',alpha=0.3)
        
        for i,t in enumerate(eod_times):
            ax.plot(t,data[(t*fs).astype('int')],'o',label=i+1)
        ax.set_xlabel('time [s]')
        ax.set_ylabel('amplitude [V]')
        #ax.axis('off')

        for i, m in enumerate(mean_eods):
            ax = fig.add_subplot(gs[1,i])
            ax.plot(1000*m[0], 1000*m[1], c='k')
            ax.fill_between(1000*m[0],1000*m[1]-m[2],1000*m[1]+m[2],color=cmap(i))
            ax.set_xlabel('time [ms]')
            ax.set_ylabel('amplitude [mV]')    
            #ax.axis('off')
            #ax.set_ylim([np.min(data),np.max(data)])
            #ax.set_xlim([np.min(),np.max()])
    else:
        plt.plot(np.arange(len(data))/fs,data,c='k',alpha=0.3)
    plt.show()


def extract_pulsefish(data, samplerate, percentile=75, npc=3, minp=10, ngaus=4, plot_steps=False):
    """
    This is what you should implement! Don't worry about wavefish for now.
    
    Parameters
    ----------
    data: 1-D array of float
        The data to be analysed.
    samplerate: float
        Sampling rate of the data in Hertz.
        
    Returns
    -------
    mean_eods: list of 2D arrays
        For each detected fish the average of the EOD snippets. First column is time in seconds,
        second column the mean eod, third column the standard error.
    eod_times: list of 1D arrays
        For each detected fish the times of EOD peaks in seconds.
    """
    
    mean_eods, eod_times = [], []
    
    # this would be the maximum pulsewidth. (used for peak extraction) in seconds
    pw=0.002
    
    # this is the cutwidth (used for snippet extraction) in seconds
    cw=0.001
    
    # 1. extract peaks
    idx_arr, idx_t_arr, h_arr, w_arr = extract_eod_times(data, pw*samplerate)

    if len(idx_arr)>0:
        i = np.where((idx_arr>int(cw*samplerate/2)) & (idx_arr<(len(data)-int(cw*samplerate/2))) & (idx_t_arr>int(cw*samplerate/2)) & (idx_t_arr<(len(data)-int(cw*samplerate/2))))[0]
        idx_arr = idx_arr[i]
        idx_t_arr = idx_t_arr[i]
        h_arr = h_arr[i]
        w_arr = w_arr[i]

        # cluster on peaks
        c_p, ch_p = cluster(idx_arr,h_arr,w_arr,data,cw,samplerate,minp,percentile,npc,ngaus,plot_steps) 
        if plot_steps == True:
            mean_eods, eod_times, _ = find_window(data,idx_arr,c_p,h_arr,samplerate,rm=False)
            print('clustering on peaks')
            plot_all(data,ts,samplerate,ms,vs)

        #cluster on troughs
        c_t, ch_t = cluster(idx_t_arr,h_arr,w_arr,data,cw,samplerate,minp,percentile,npc,ngaus,plot_steps)
        if plot_steps == True:
            mean_eods, eod_times, _ = find_window(data,idx_t_arr,c_t,h_arr,samplerate,rm=False)
            print('clustering on troughs')
            plot_all(data,ts,samplerate,ms,vs)


        # merge peak and trough clusters
        # windows can already be returned here.
        c, c_idx_arr, c_h_arr, _ = merge_clusters(c_p,c_t,idx_arr,idx_t_arr,h_arr,data,samplerate)
        if plot_steps==True:
            print(np.unique(c))
            mean_eods, eod_times, _ = find_window(data,c_idx_arr,c,c_h_arr,samplerate,rm=False)
            print('after merging')
            plot_all(data,ts,samplerate,ms,vs)

        # delete the moving fish
        keep_clus = delete_moving_fish(c,c_idx_arr/samplerate,len(data)/samplerate, c_h_arr)
        mask = np.isin(c,keep_clus)
        c = c[mask]
        c_idx_arr = c_idx_arr[mask]
        c_h_arr = c_h_arr[mask]

        # find windows for each of the clusters that are kept for visualizing them.
        # I could do this earlier, will save some computation time.
        mean_eods, eod_times, _ = find_window(data,c_idx_arr,c,c_h_arr,samplerate,rm=False)

        if plot_steps==True:
            print('after deleting moving fish')
    
    return mean_eods, eod_times

# load data:
filename = sys.argv[1]
channel = 0
raw_data, samplerate, unit = load_data(filename, channel)

# best_window:
data, clipped = best_window(raw_data, samplerate, win_size=8.0)

# plot the data you should analyze:
time = np.arange(len(data))/samplerate  # in seconds
plt.plot(time, data)
plt.show()

# pulse extraction:
mean_eods, eod_times = extract_pulsefish(data, samplerate)

# plot results
plot_all(data,eod_times,samplerate, mean_eods)