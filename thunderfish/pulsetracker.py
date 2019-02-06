"""
by Dexter Frueh
"""

import sys
import numpy as np
import copy
from scipy import stats
from scipy import signal
from scipy import optimize
import matplotlib
#from fish import ProgressFish
import matplotlib.pyplot as plt
from thunderfish.dataloader import open_data
from thunderfish.eventdetection import detect_peaks
from scipy.interpolate import interp1d
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from sklearn.cluster import AgglomerativeClustering
from collections import deque
import ntpath
import time
import os
from shutil import copy2

def makeeventlist(main_event_positions,side_event_positions,data,event_width=20):
    """
    Generate array of events that might be EODs of a pulse-type fish, using the location of peaks and troughs,
    the data and an optional width of an supposed EOD-event.
    The generated event-array contains location and height of such events.
    The height of the events is calculated by its height-difference to nearby troughs and main events that have no side events in a range closer than event_width are discarded and not considered as EOD event.

    Parameters
    ----------
    main_event_positions: array of int or float
        Positions of the detected main events in the data time series. Either peaks or troughs.
    side_event_positions: array of int or float
        Positions of the detected side events in the data time series. The complimentary event to the main events.
    data: array of float
        The given data.
    event_width: int or float, optional

    Returns
    -------
    EOD_events: ndarray
        2D array containing data with 'np.float' type, size (number_of_properties = 3, number_of_events).
        Generated and combined data of the detected events in an array with arrays of x, y and height along the first axis.

    """
    mainfirst = int((min(main_event_positions[0],side_event_positions[0])<side_event_positions[0]))  # determines if there is a peak or through first. Evaluates to 1 if there is a peak first.
    main_x = main_event_positions
    main_y = data[main_event_positions]
    # empty placeholders, filled in the next step while iterating over the properties of single main
    main_h = np.zeros(len(main_event_positions))
    main_real = np.ones(len(main_event_positions))
    # iteration over the properties of the single main
    for ind,(x, y, h, r) in enumerate(np.nditer([main_x, main_y, main_h, main_real], op_flags=[["readonly"],['readonly'],['readwrite'],['readwrite']])):
        l_side_ind = ind - mainfirst
        r_side_ind = l_side_ind + 1
        try:
            r_side_x = side_event_positions[r_side_ind]
            r_distance = r_side_x - x
            r_side_y = data[r_side_x]
        except:
            pass
        try:
            l_side_x = side_event_positions[l_side_ind]
            l_distance = x - l_side_x
            l_side_y = data[l_side_x]
        except:
            pass # ignore left or rightmost events which throw IndexError
        # calculate distances to the two side events next to the main event and mark all events where the next side events are not closer than maximum event_width as unreal. If an event might be an EOD, then calculate its height.
        if l_side_ind >= 0 and r_side_ind < len(side_event_positions):
            if min((l_distance),(r_distance)) > event_width:
                    r[...] = False
            elif max((l_distance),(r_distance)) <= event_width:
                    h[...] = max(abs(y-l_side_y),abs(y-r_side_y))  #calculated using absolutes in case of for example troughs instead of peaks as main events 
            else:
                    if (l_distance)<(r_distance): # evaluated only when exactly one side event is out of reach of the event width. Then the closer event will be the correct event
                        h[...] = abs(y-l_side_y)
                    else:
                        h[...] = abs(y-r_side_y)
        # check corner cases
        elif l_side_ind == -1:
            if r_distance > event_width:
                r[...] = False
            else:
                h[...] = y-r_side_y
        elif r_side_ind == len(side_event_positions):
            if l_distance> event_width:
                r[...] = False
            else:
                h[...] = y-l_side_y
    # generate return array and discard all events that are not marked as real
    EOD_events = np.array([main_x, main_y, main_h], dtype = np.float)[:,main_real==1]
    return EOD_events

def discardnearbyevents(event_locations, event_heights, min_distance):
    """
    Given a number of events with given location and heights, returns a selection
    of these events where  no event is closer than eventwidth to the next event.
    Among neighboring events closer than eventwidth the event with smaller height
    is discarded.
    Used to discard sidepeaks in detected multiple peaks of single EOD-pulses and
    only keep the largest event_height and the corresponding location as
    representative of the whole EOD pulse.

    Parameters
    ----------
    event_locations: array of int or float
        Positions of the given events in the data time series.
    event_heights: array of int or float
        Heights of the given events, indices refer to the same events as in
        event_locations.
    min_distance: int or float
        minimal distance between events before one of the events gets discarded.

    Returns
    -------
    event_locations: array of int or float
        Positions of the returned events in the data time series.
    event_heights: array of int or float
        Heights of the returned events, indices refer to the same events as in
        event_locations.

    """
    unchanged = False
    counter = 0
    event_indices = np.arange(0,len(event_locations)+1,1)
    while unchanged == False:# and counter<=200:
       x_diffs = np.diff(event_locations)
       events_delete = np.zeros(len(event_locations))
       for i, diff in enumerate(x_diffs):
           if diff < min_distance:
               if event_heights[i+1] > event_heights[i] :
                   events_delete[i] = 1
               else:
                   events_delete[i+1] = 1
       event_heights = event_heights[events_delete!=1]
       event_locations = event_locations[events_delete!=1]
       event_indices = event_indices[np.where(events_delete!=1)[0]]
       if np.count_nonzero(events_delete)==0:
           unchanged = True
       counter += 1
       if counter > 2000:
           print('Warning: unusual many discarding steps needed, unusually dense events')
           pass
    return event_indices, event_locations, event_heights

def crosscorrelation(sig, data):
    'returns crosscorrelation of two arrays, the first array should have a length equal to or smaller than the second array.'
    return signal.fftconvolve(data, sig[::-1],  mode='valid')

def interpol(data, kind):
    """
    interpolates the given data using scipy interpolation python package

    Parameters
    ----------
    data: array

    kind: string or int
        (‘linear’, ‘nearest’, ‘zero’, ‘slinear’, ‘quadratic’, ‘cubic’, ‘previous’, ‘next’), or integer of order of spline interpolation to be used

    Returns
    -------
    interpolation: function

    """
    width = len(data)
    x = np.linspace(0, width-1, num = width, endpoint = True)
    #return interp1d(x, data[0:width], kind, assume_sorted=True)
    return interp1d(x, data[0:width], kind)

def interpolated_array(data, kind, int_fact):
    """
    returns an interpolated array of the given dataarray.

    Parameters
    ----------
    data: array

    kind: string or int
        (‘linear’, ‘nearest’, ‘zero’, ‘slinear’, ‘quadratic’, ‘cubic’, ‘previous’, ‘next’), or integer of order of spline interpolation to be used

    int_fact: int
         factor by which the interpolated array is larger than the original array

    Returns
    -------
    interpolated array: array

    """
    return interpol(data,kind)(np.arange(0, len(data)-1, 1/int_fact))

def cut_snippets(data,event_locations,cut_width,int_met="linear",int_fact=10,max_offset = 1.5): 
    """
    cuts intervals from a data array, interpolates and aligns them and returns them in a list

    Parameters
    ----------
    data: array

    event_locations: array

    cut_width: [int, int]
        lower and upper limit of the intervals relative to the event locations.
        f.e. [-15,15] indicates an interval of 30 datapoints around each event location
s
    int_met: string or int
        method of interpolation. (‘linear’, ‘nearest’, ‘zero’, ‘slinear’, ‘quadratic’, ‘cubic’, ‘previous’, ‘next’), or integer of order of spline interpolation to be used

    int_fact: int
        factor by which the interpolated array is larger than the original

    max_offset: float
        maximal offset by which the interpolated intervals can be moved to be aligned with each other. offset relative to the datapoints of the original data.

    Returns
    -------
    aligned_snips: twodimensional nparray
        the processed intervals (interval#,intervallen)

    """
    snippets = []
    cut_width = [-cut_width, cut_width]
    alignwidth = int(np.ceil((max_offset) * int_fact))
    for pos in event_locations.astype('int'):
        snippets.append(data[pos+cut_width[0]:pos+cut_width[1]])
    ipoled_snips = np.empty((len(snippets), (cut_width[1]-cut_width[0])*int_fact-int_fact))
    for i, snip in enumerate(snippets):
       if len(snip) < ((cut_width[1]-cut_width[0])):
            if i == 0:
                snip = np.concatenate([np.zeros([((cut_width[1]-cut_width[0]) - len(snip))]),np.array(snip)])
            if i == len(snippets):
                snip = np.concatenate([snip, np.zeros([((cut_width[1]-cut_width[0])-len(snip))])])
            else:
                snip = np.zeros([(cut_width[1]-cut_width[0])])
       #f_interpoled = interpol(snip, int_met) #if len(snip) > 0 else np.zeros([(cut_width[1]-cut_width[0]-1)*int_fact ])
       interpoled_snip = interpolated_array(snip, int_met, 10)#f_interpoled(np.arange(0, len(snip)-1, 1/int_fact))
       intsnipheight   = np.max(interpoled_snip) - np.min(interpoled_snip)
       if intsnipheight == 0:
           intsnipheight = 1
       interpoled_snip = (interpoled_snip - max(interpoled_snip))* 1/intsnipheight
       ipoled_snips[i] = interpoled_snip
    mean = np.mean(ipoled_snips, axis = 0)
    aligned_snips = np.empty((len(snippets), (cut_width[1]-cut_width[0])* int_fact-(2*alignwidth)-int_fact))
    for i, interpoled_snip in enumerate(ipoled_snips):
        cc = crosscorrelation(interpoled_snip[alignwidth:-alignwidth], mean)
        #cc = crosscorrelation(interpoled_snip[15 + 10*-cut_width[0]-10*7:-15+ -10*cut_width[1]+ 31], mean[10*-cut_width[0]-10*7:-10*cut_width[1]+31])
        offset = -alignwidth + np.argmax(cc)
        aligned_snip = interpoled_snip[alignwidth-offset:-alignwidth-offset] if offset != -alignwidth else interpoled_snip[2*alignwidth:]
        if len(aligned_snip[~np.isnan(aligned_snip)])>0:
            aligned_snips[i] = aligned_snip
    return aligned_snips

def pc(dataset):
    """
    Calculates the principal components of a dataset using the python module scikit-learn's principal component analysis

    Parameters
    ----------
    dataset: ndarray
        dataset of which the principal components are to be calculated.
        twodimensional array of shape (observations, features)

    Returns
    -------
    pc_comp: ndarray
        principal components of the dataset

    """
    pc_comp= PCA().fit_transform(dataset)
    return pc_comp

def dbscan(pcs, events, order, eps, min_samples, takekm):
    """
    improve description, add parameter and returns

    calculates clusters of high spatial density of the given observations in their feature space. 
    #For example, the first few principal components of the data could be used as features for the classification.

    Parameters
    ----------
    pcs: ndarray
        %TODO
        shape(samples, features)
    ...

    Returns
    -------
    labels: ndarray
        labels of the clusters of each observation

    """
    # pcs (samples, features)
    # X (samples, features)
    try:
        X = pcs[:,:order]
    except:
        X = pcs[:,order]
    # #############################################################################
    # Compute DBSCAN
    clusters = DBSCAN(eps, min_samples).fit(X)
    #from sklearn.cluster import KMeans
    core_samples_mask = np.zeros_like(clusters.labels_, dtype=bool)
    core_samples_mask[clusters.core_sample_indices_] = True
    labels = clusters.labels_
    return labels

def cluster_events(features, events, order, eps, min_samples, takekm, method='DBSCAN'):
    """
    clusters the given events using the given feature space and the clustering algorithm of choice and appends the assigned cluster number to the event's properties.

    Parameters
    ----------

    Returns
    -------

    """
    ########################      function maybe could be even more generic, ? (dependant on datatype of "events" )
    if method == 'DBSCAN':
        labels = dbscan(features,events, order, eps, min_samples, takekm)
    elif method == 'kMean':
        pass
        # To be implemented
        #labels = kmeans([])
    return labels
    events = np.append(events,[labels], axis = 0)
    return events

class Peaklist(object):
    def __init__(self, peaklist):
        self.list = peaklist
        self.lastofclass = {}
        self.lastofclassx = {}
        self.classesnearby = []
        self.classesnearbyx = []
        self.classesnearbypccl = []
        self.classlist = []
        self.classamount = 0
        self.shapes = {}

def connect_blocks(oldblock):
    """
        used to connect blocks.
        transfers data from the previous analysis block to the current block
    """
    newblock = Peaklist([])
    newblock.lastofclass    = oldblock.lastofclass
    newblock.lastofclassx    = oldblock.lastofclassx
    newblock.classesnearby  = oldblock.classesnearby
    newblock.classesnearbypccl  = oldblock.classesnearbypccl
    newblock.classesnearbyx = [clnearbyx - oldblock.len for clnearbyx in oldblock.classesnearbyx]
    newblock.classamount = oldblock.classamount
    newblock.len = oldblock.len
    return newblock

def alignclusterlabels(labels, peaklist, peaks, data='test'):
    """
        used to connect blocks.
        changes the labels of clusters in the current block to fit with the labels of the previous block
    """
    overlapamount = len(peaks[:,peaks[0]<30000])
    if overlapamount == 0:
        return None
    old_peaklist = copy.deepcopy(peaklist)  #redundant 
    overlappeaks = copy.deepcopy(peaks[:,:overlapamount])
    overlap_peaklist = copy.deepcopy(old_peaklist)
  #  overlappeaks = np.append(overlappeaks,[labels], axis = 0)
    overlappeaks[3]=[-1]*len(overlappeaks[0])
    #overlap_peaklist = connect_blocks(old_peaklist)
    overlap_peaklist.classesnearbypccl = [-1]*len(overlap_peaklist.classesnearbypccl)
    classified_overlap = ampwalkclassify3_refactor(overlappeaks,overlap_peaklist)[0]
    #plot_events_on_data(classified_overlap,data)
    labeltranslator = {}
    for cl in np.unique(classified_overlap[4]):
        if len(labeltranslator) <= len(np.unique(labels)):
            labelindex = np.where(classified_overlap[4] == cl)[0]
            label = labels[labelindex]
            labelindex = labelindex[np.where(label == stats.mode(label)[0])[0][0]]
            newlabel = labels[labelindex] #waveform label belonging to the class cl in the new block
            try:
                oldlabel_ind= old_peaklist.classesnearby.index(cl)
                oldlabel = old_peaklist.classesnearbypccl[oldlabel_ind]
            #    oldlabel = old_peaklist.classesnearbypccl[::-1][old_peaklist.classesnearby[::-1].index(cl)] #last label belonging to cl in the old block
            except:
                oldlabel = -2
            try:
                labeltranslator[oldlabel]
            except KeyError:
                labeltranslator[oldlabel] = newlabel
    for lbl in old_peaklist.classesnearbypccl:
        try: labeltranslator[lbl]
        except KeyError: labeltranslator[lbl] = lbl
    peaklist.classesnearbypccl = [labeltranslator[lbl] for lbl in peaklist.classesnearbypccl]

def ampwalkclassify3_refactor(peaks,peaklist):
    """

        classifies peaks/EOD_events into different classes by their amplitude.

        Takes list of peaks and list of properties of the list of the last analysis block
        Classifies the single peaks in the direction of their occurence in time, based on their amplitude and
        their previously assigned class based on their waveform (... using the method cluster_events on the
        principal components of the snippets around the single peaks)

        Method:
        calculates differences in amplitude between the current peak and different amplitudeclasses that are nearby. creates new amplitudeclass if no class is close enough. creates no new class if the peaks's waveformclass is a noiseclass of the DBSCAN algorithm. Does not compare peaks of different Waveformclasses.

        --can be used without prior waveformclasses, resulting in classification solely on the amplitude development
        pcclclasses need to be set to the same class herefore, .... . not practical, but  should be possible to
        split up into more general functions
    """
    classamount = peaklist.classamount
    lastofclass = peaklist.lastofclass
    lastofclassx = peaklist.lastofclassx
    a=0
    elem = 0
    thresholder = []
    comperr = 1
    classesnearby = peaklist.classesnearby
    classesnearbyx = peaklist.classesnearbyx
    classesnearbypccl = peaklist.classesnearbypccl
    classes = np.zeros((len(peaks[0])))
    pcclasses = peaks[3]
    positions = peaks[0]
    heights = peaks[2]
    cl = 0
    maxdistance = 30000    #    Max distance to possibly belong to the same class
    factor = 1.6     # factor by which a peak fits into a class, f.E: classheight = 1 , factor = 2 => peaks accepted in range (0.5,2)
    c=0
    for peaknum, p in enumerate(peaks.T):
        if len(lastofclass) == 0:
            lastofclass[1] = deque()
            lastofclassx[1] = deque()
            lastofclass[1].append(heights[peaknum])
            lastofclassx[1].append(positions[peaknum])
            classesnearby.append(1)
            classesnearbyx.append(-1)
            classesnearbypccl.append(pcclasses[peaknum])
            classes[peaknum] = 1
            classamount += 1
            continue
        time1 = time.time()
        for i, cl in enumerate(classesnearby):
            if  (positions[peaknum] - classesnearbyx[i]) > maxdistance:
                classesnearby.pop(i)
                classesnearbyx.pop(i)
                classesnearbypccl.pop(i)
        lastofclassisis = []
        for i in classesnearby:
            lastofclassisis.append(np.median(np.diff(lastofclassx[i])))
        meanisi = np.mean(lastofclassisis)
        if 32000 > 20*meanisi> 6000:
            maxdistance = 20*meanisi
        cl = 0  # 'No class'
        comperr = 1
        clnrby = np.unique(classesnearby)

        for i in clnrby:
            classmean = np.mean(lastofclass[i])
            logerror = np.abs(np.log2(heights[peaknum])-np.log2(classmean))
            abserror = np.abs(heights[peaknum]-classmean)
            logthresh = np.log2(factor)
            #relerror = error 
            relerror = logerror
            
            if classesnearbypccl[classesnearby.index(i)] == pcclasses[peaknum] or pcclasses[peaknum] == -1:# or  
              if logerror < logthresh:     ## SameClass-Condition
                if relerror < comperr and (positions[peaknum]-classesnearbyx[classesnearby.index(i)])<maxdistance:
                    holdlastcl = cl
                    cl = i
                    holdlastcomperr = comperr
                    comperr = relerror
        time2 = time.time()
        time1 = time.time()
        if pcclasses[peaknum] != -1:
            if cl != 0 :
                if len(lastofclass[cl]) >= 3:
                    lastofclass[cl].popleft()
                if len(lastofclassx[cl]) >= 3:
                    lastofclassx[cl].popleft()
                lastofclass[cl].append(heights[peaknum])
                lastofclassx[cl].append(positions[peaknum])
                classes[peaknum] = cl
            else:                                    # Add new class
                cl = classamount+1
                classamount = cl
                lastofclass[cl] = deque()
                lastofclassx[cl] = deque()
                lastofclass[cl].append(heights[peaknum])
                lastofclassx[cl].append(positions[peaknum])
                classes[peaknum] = cl
                classesnearby.append(cl)
                classesnearbyx.append(positions[peaknum])
                classesnearbypccl.append(pcclasses[peaknum])
            if len(classesnearby) >= 12: #kacke implementiert? 
                minind = classesnearbyx.index(min(classesnearbyx))
                del lastofclass[classesnearby[minind]]
                del lastofclassx[classesnearby[minind]]
                classesnearby.pop(minind)
                classesnearbyx.pop(minind)
                classesnearbypccl.pop(minind)
    #        for ind, clnrby in enumerate(reversed(classesnearby)):
    #            classesnearbyx
    #            del lastofclass[classesnearby[ind]]
    #           # del lastofclassx[classesnearby[minind]]
    #            classesnearby.pop(minind)
    #            classesnearbyx.pop(minind)
            try:
                ind=classesnearby.index(cl)
                classesnearbyx[ind] = positions[peaknum]
            except ValueError:
                classesnearby.append(cl)
                classesnearbyx.append(positions[peaknum])
                classesnearbypccl.append(pcclasses[peaknum])
        else:
            if cl != 0:
                classes[peaknum] = cl
            else:
                cl = classamount+1
                classamount = cl
                lastofclass[cl] = deque()
                lastofclassx[cl] = deque()
                lastofclass[cl].append(heights[peaknum])
                lastofclassx[cl].append(positions[peaknum])
                classes[peaknum] = cl
                classesnearby.append(cl)
                classesnearbyx.append(positions[peaknum])
                classesnearbypccl.append(pcclasses[peaknum])
            if len(classesnearby) >= 12: #kacke implementiert? 
                minind = classesnearbyx.index(min(classesnearbyx))
                del lastofclass[classesnearby[minind]]
                del lastofclassx[classesnearby[minind]]
                classesnearby.pop(minind)
                classesnearbyx.pop(minind)
                classesnearbypccl.pop(minind)
    #        for ind, clnrby in enumerate(reversed(classesnearby)):
    #            classesnearbyx
    #            del lastofclass[classesnearby[ind]]
    #           # del lastofclassx[classesnearby[minind]]
    #            classesnearby.pop(minind)
    #            classesnearbyx.pop(minind)
            try:
                ind=classesnearby.index(cl)
                classesnearbyx[ind] = positions[peaknum]
            except ValueError:
                classesnearby.append(cl)
                classesnearbyx.append(positions[peaknum])
                classesnearbypccl.append(pcclasses[peaknum])
    peaklist.lastofclass = lastofclass
    peaklist.lastofclassx = lastofclassx
    peaklist.classesnearby = classesnearby
    peaklist.classesnearbyx = classesnearbyx
    peaklist.classesnearbypccl = classesnearbypccl
    peaklist.classlist =  classes # np.vectorize(lambda peak: peak.cl, otypes=[object])(peaklist.list)
    peaklist.classamount = classamount
    peaks = np.append(peaks,classes[None,:], axis = 0)
    return peaks, peaklist

def discard_wave_pulses(peaks, data):
     """
        discards events from a pulse_event list which are unusally wide (wider than a tenth of the inter pulse interval), which indicates a wave-type EOD instead of a pulse type
     """
     deleteclasses = []
     for cl in np.unique(peaks[3]):
         peaksofclass = peaks[:,peaks[3] == cl]
         isi = np.diff(peaksofclass[0])
         isi_mean = np.mean(isi)
         widepeaks = 0
         isi_tenth_area = lambda x, isi : np.arange(np.floor(x-0.1*isi),np.ceil(x+0.1*isi),1, dtype = np.int)
         for p in peaksofclass.T:
             data = np.array(data)
             try:
                 for dp_around in data[isi_tenth_area(p[0],isi_mean)]:
                    if dp_around <= p[1]-p[2]:
                       break
             except (IndexError,ValueError) as e:
                 pass
         else:
             widepeaks+=1
         if widepeaks > len(peaksofclass)*0.5:
             deleteclasses.append(cl)
     for cl in deleteclasses:
            peaks = peaks[:,peaks[3]!=cl]
     return peaks

def plot_events_on_data(peaks, data):
    """
        plots the detected events onto the data timeseries. If the events are classified, the classes are plotted in different colors and the class -1 (not belonging to a cluster) is plotted in black
    """
    plt.plot(range(len(data)),data, color = 'black')
    if len(peaks)>3:
        classlist =  np.array(peaks[3],dtype=np.int)
        if len(peaks) > 4:
            classlist = np.array(peaks[4],dtype=np.int)
        cmap = plt.get_cmap('jet')
        colors =cmap(np.linspace(0, 1.0, 3000)) #len(np.unique(classlist))))
        np.random.seed(22)
        np.random.shuffle(colors)
        colors = [colors[cl] for cl in np.unique(classlist)]
        for cl, color in zip(np.unique(classlist), colors):
            if cl == -1:
                color = 'black'
            peaksofclass = peaks[:,classlist == cl]
            plt.plot(peaksofclass[0],peaksofclass[1], '.', color = color,   ms =20)
            plt.scatter(peaksofclass[0], peaksofclass[2])
    else:
        plt.scatter(peaks[0],peaks[1], color = 'red')
    plt.show()
    plt.close()

def discard_short_classes(events, minlen):
    """ 
        returns all events despite events which are in classes with less than minlen members
    """
    classlist = events[3]
    smallclasses = [cl for cl in np.unique(classlist) if len(classlist[classlist
                                                                     == cl]) <
                 minlen]
    delete = np.zeros(len(classlist))
    for cl in smallclasses:
        delete[classlist == cl] == 1
        events = events[:,delete != 1]
    return events

def path_leaf(path):
    ntpath.basename("a/b/c")
    head, tail = ntpath.split(path)
    return tail or ntpath.basename(head)

def save_EOD_events_to_npmmp(EOD_Events,eods_len,startblock,datasavepath,mmpname='eods.npmmp'):
    n_EOD_Events = len(EOD_Events[0])
    savepath = datasavepath+"/"+mmpname
    if startblock:
                eods = np.memmap(savepath,
                                 dtype='float64', mode='w+',
                                 shape=(4,n_EOD_Events), order = 'F')
    else:
        dtypesize = 8#4 #float32 is 32bit = >4< bytes long  ---changed to float64 -> 8bit
        eods = np.memmap(savepath, dtype=
                         'float64', mode='r+', offset = dtypesize*eods_len*4,
                         shape=(4,n_EOD_Events), order = 'F')
    eods[:] = EOD_Events
    
def discard_small_EODs(EOD_Events, ultimate_threshold):
    return EOD_Events[:,np.where(EOD_Events[2]>ultimate_threshold)]
    return np.where(EOD_Events[2]>ultimate_threshold)

def analyze_pulse_data(filepath, deltat=30, thresh=0.04, starttime = 0, endtime = 0, savepath = False,save=False, npmmp = False, plot_steps=False, plot_result=False):
    """
    analyzes timeseries of a pulse fish EOD recording

    Parameters
    ----------
    filepath: WAV-file with the recorded timeseries

    deltat: int, optional
        time for a single analysisblock (recommended less than a minute, due to principal component clustering on the EOD-waveforms)

    thresh: float, optional
        minimum threshold for the peakdetection (if computing frequencies recommended a tiny bit lower than the wished threshold, and instead discard the EOD below the wished threshold after computing the frequencies for each EOD.)

    starttime: int or, str of int, optional
        time into the data from where to start the analysis, seconds.

    endtime: int or str of int, optional
        time into the data where to end the analysis, seconds, larger than starttime.

    savepath = Boolean or str, optional
        path to where to save results and intermediate result, only needed if save or npmmp is True.
        string to specify a relative path to the directory where results and intermediate results will bed
        or False to use preset savepath, which is ~/filepath/
        or True to specify savepath as input when the script is running

    save: Boolean, optional
        True to save the results into a npy file at the savepath

    npmmp: Boolean, optional
        True to save intermediate results into a npmmp at the savepath, only recommended in case of memory overflow

    plot_steps: Boolean, optional
        True to plot the results of each analysis block

    plot_results: Boolean, optional
        True to plot the results of the final analysis. Not recommended for long recordings due to %TODO

    Returns
    -------
    eods: numpy array
        2D numpy array. first axis: attributes of an EOD (x (datapoints), y (recorded voltage), height (difference from maximum to minimum), class), second axis: EODs in chronological order.
    """
    
    # parameters for the analysis
    thresh = 0.04 # minimal threshold for peakdetection
    peakwidth = 20 # width of a peak and minimal distance between two EODs
    # basic parameters for thunderfish.dataloader.open_data
    verbose = 0
    channel = 0
    ultimate_threshold = thresh+0.01
    startblock = 0
    # timeinterval to analyze other than the whole recording
    #starttime = 0
    #endtime = 0
    #timegiven =  0
    #save = int(save)
    #plot_steps = int(plot_steps)
    starttime = int(starttime)
    endtime = int(endtime)
    timegiven = False
    if endtime > starttime>=0:
        timegiven = True
    peaks = np.array([])
    troughs = np.array([])
    filename = path_leaf(filepath)
    eods_len = 0
    if savepath==False:
        datasavepath = filename[:-4]
    elif savepath==True:
        datasavepath = input('With the option npmmp enabled, a numpy memmap will be saved to: ').lower()
    else: datasavepath=savepath

    if save and (os.path.exists(datasavepath+"/eods8_"+filename[:-3]+"npy") or os.path.exists(datasavepath+"/eods5_"+filename[:-3]+"npy")):
        print('there already exists an analyzed file, aborting. Change the code if you don\'t want to abort')
        quit()
    if npmmp:
        #proceed = input('With the option npmmp enabled, a numpy memmap will be saved to ' + datasavepath + '. continue? [y/n] ').lower()
        proceed = 'y'
        if proceed != 'y':
             quit()
    # starting analysis
    with open_data(filepath, channel, deltat, 0.0, verbose) as data:

        samplerate = data.samplerate

        # selected time interval
        if timegiven == True:
            parttime1 = starttime*samplerate
            parttime2 = endtime*samplerate
            data = data[parttime1:parttime2]

        #split data into blocks
        nblock = int(deltat*samplerate)
        if len(data)%nblock != 0:
            blockamount = len(data)//nblock + 1
        else:
            blockamount = len(data)//nblock
        print('blockamount: ' , blockamount)
        progress = 0
        print(progress, '%' , flush = True, end = " ")
        #fish = ProgressFish(total = blockamount)
        for idx in range(0, blockamount):
            blockdata = data[idx*nblock:(idx+1)*nblock]
            if progress < (idx*100 //blockamount):
                progress = (idx*100)//blockamount
            progressstr = ' Filestatus: '
            #fish.animate(amount = idx, dexextra = progressstr)
            pk, tr = detect_peaks(blockdata, thresh)
            troughs = tr
            if len(pk) > 3:
                peaks = makeeventlist(pk,tr,blockdata,peakwidth)
                peakindices, peakx, peakh = discardnearbyevents(peaks[0],peaks[1],peakwidth)
                peaks = peaks[:,peakindices]
                if len(peaks) > 0:
                    if idx > startblock:
                        peaklist = connect_blocks(peaklist)
                    else:
                        peaklist = Peaklist([])
                    aligned_snips = cut_snippets(blockdata,peaks[0], 15, int_met = "cubic", int_fact = 10,max_offset = 1.5)
                    print(aligned_snips.shape)
                    pcs = pc(aligned_snips)#pc_refactor(aligned_snips)
                    order = 5
                    minpeaks = 3 if deltat < 2 else 10
                    labels = cluster_events(pcs, peaks, order, 0.4, minpeaks, False, method = 'DBSCAN')
                    peaks = np.append(peaks,[labels], axis = 0)
                    #plot_events_on_data(peaks, blockdata)
                    num = 1
                    if idx > startblock:
                      alignclusterlabels(labels, peaklist, peaks,data=blockdata)
                    peaks, peaklist = ampwalkclassify3_refactor(peaks, peaklist) # classification by amplitude
                    minlen = 6
                    peaks = discard_short_classes(peaks, minlen)
                    if len(peaks[0]) > 0:
                        peaks = discard_wave_pulses(peaks, blockdata)
                    if plot_steps == True:
                        plot_events_on_data(peaks, blockdata)
                        pass
                    peaklist.len = nblock
                    worldpeaks = np.copy(peaks)
                    worldpeaks[0] = worldpeaks[0] + (idx*nblock)
                    thisblock_eods = np.delete(worldpeaks,3,0)
                    if npmmp:
                        if idx == startblock:
                            if not os.path.exists(datasavepath):
                                os.makedirs(datasavepath)
                            mmpname = "eods_"+filename[:-3]+"npmmp"
                        # save the peaks of the current buffered part to a numpy-memmap on the disk
                        save_EOD_events_to_npmmp(thisblock_eods,eods_len,idx==startblock,datasavepath,mmpname)
                        eods_len += len(thisblock_eods[0])
                    else:
                        if idx > 0:
                            all_eods = np.concatenate((all_eods,thisblock_eods),axis = 1)
                        else:
                            all_eods = thisblock_eods
    #plot_events_on_data(all_eods,data)
    print('returnes analyzed EODS. Calculate frequencies using all of these but discard the data from the EODS within the lowest few percent of amplitude')
    if npmmp:
        all_eods = np.memmap(datasavepath+'/'+mmpname, dtype='float64', mode='r+', shape=(4,eods_len), order = 'F')
    if save == 1:
       path = filename[:-4]+"/"
       if not os.path.exists(path):
           os.makedirs(path)
       if eods_len > 0:
           np.save(datasavepath+"/eods8_"+filename[:-3]+"npy", all_eods)
           print('Saved!')
       else:
           print('not saved')
    return all_eods

def analyze_long_pulse_data_file(filepath,save=0,plot_steps=0,new=1,starttime = 0, endtime = 0):
    """
    analyzes timeseries of a pulse fish EOD recording
    """
    #    Script to detect and classify EODs in recordings of weakly electric pulse
    #    fish, Dexter Früh, 2018
    #
    #    results will be saved in workingdirectory/recording/
    #
    #    input:
    #      -  [Recorded Timeseries] recording.WAV
    #    outputs(optional):
    #      -  [Detected and Classified EODs]
    #            (Numpy Array with Shape (Number of EODs, 4 (Attributes of EODs)),
    #            with the EOD-Attributes
    #               -   x-location of the EOD
    #                       (time/x-coordinate/datapoint in recording)
    #               -   y-location of the EOD
    #                       (Amplitude of the positive peak of the pulse-EOD)
    #               -   height of the EOD(largest distance between peak and through in the EOD)
    #               -   class of the EOD
    #           eods_recording.npy
    #      -   [plots of the results of each analyse step for each
    #               analysepart (timeinterval of length = deltat) of the recording]
    #
    #    required command line arguments at function call
    #        - save  : if True, save the results to a numpy file (possibly
    #                                                          overwrite existing)
    #        - plot  : if True, plot results in each analysestep
    #        - new   : if True, do a new analysis of the recording, even if there
    #                       is an existing analyzed .npy file with the right name.
    #
    
    # parameters for the analysis
    deltat = 30.0  # seconds of buffer size
    thresh = 0.04 # minimal threshold for peakdetection
    peakwidth = 20 # width of a peak and minimal distance between two EODs
    # basic parameters for thunderfish.dataloader.open_data
    verbose = 0
    channel = 0
    ultimate_threshold = thresh+0.01
    startblock = 0
    # timeinterval to analyze other than the whole recording
    #starttime = 0
    #endtime = 0
    #timegiven =  0
    home = os.path.expanduser('~')
    os.chdir(home)
    new = int(sys.argv[4])
    save = int(sys.argv[2])
    plot = int(sys.argv[3])
    starttime = int(starttime)
    endtime = int(endtime)
    timegiven = False
    if endtime > starttime>=0:
        timegiven = True
    peaks = np.array([])
    troughs = np.array([])
    filename = path_leaf(filepath)
    datasavepath = filename[:-4]
    proceed = input('Currently operates in home directory. If given a pulsefish recording filename.WAV, then a folder filename/ will be created in the home directory and all relevant files will be stored there. continue? [y/n] ').lower()
    if proceed != 'y':
         quit()
    if not os.path.exists(datasavepath):
        os.makedirs(datasavepath)
    if save == 1:
         print('files will be saved to: ', datasavepath)
    eods_len = 0
    # starting analysis
    if new == 1 or not os.path.exists(filename[:-4]+"/eods5_"+filename[:-3]+"npy"):
        if filepath != home+ '/'+ datasavepath+'/'+filename:
            print(filepath, datasavepath+'/'+filename)
            proceed = input('Copy datafile to '+ datasavepath+ ' where all the other files will be stored? [y/n] ').lower()
            if proceed == 'y':
                copy2(filepath,datasavepath)
        # import data
        with open_data(filepath, channel, deltat, 0.0, verbose) as data:
            samplerate = data.samplerate
            nblock = int(deltat*data.samplerate)

            # selected time interval
            if timegiven == True:
                parttime1 = starttime*samplerate
                parttime2 = endtime*samplerate
                data = data[parttime1:parttime2]

            #split data into blocks
            if len(data)%nblock != 0:
                blockamount = len(data)//nblock + 1
            else:
                blockamount = len(data)//nblock

            # progress bar
            print('blockamount: ' , blockamount)
            progress = 0
            print(progress, '%' , flush = True, end = " ")
            #fish = ProgressFish(total = blockamount)

            # blockwise analysis 
            for idx in range(0, blockamount):
                blockdata = data[idx*nblock:(idx+1)*nblock]
                # progressbar
                if progress < (idx*100 //blockamount):
                    progress = (idx*100)//blockamount
                progressstr = ' Filestatus: '
                #fish.animate(amount = idx, dexextra = progressstr)
#---analysis-----------------------------------------------------------------------
                # step1: detect peaks in timeseries
                pk, tr = detect_peaks(blockdata, thresh)
                troughs = tr
                # continue with analysis only if multiple peaks are detected
                if len(pk) > 3:
                    peaks = makeeventlist(pk,tr,blockdata,peakwidth)

                    #plot_events_on_data(peaks, blockdata)
                    peakindices, peakx, peakh = discardnearbyevents(peaks[0],peaks[1],peakwidth)
                    peaks = peaks[:,peakindices]

                    if len(peaks) > 0:
                        # used to connect the results of the current block with the previous
                        if idx > startblock:
                            peaklist = connect_blocks(peaklist)
                        else:
                            peaklist = Peaklist([])
                        aligned_snips = cut_snippets(blockdata,peaks[0], 15, int_met = "cubic", int_fact = 10,max_offset = 1.5)
                        pcs = pc(aligned_snips)#pc_refactor(aligned_snips)
                        order = 5
                        minpeaks = 3 if deltat < 2 else 10
                        labels = cluster_events(pcs, peaks, order, 0.4, minpeaks, False, method = 'DBSCAN')
                        peaks = np.append(peaks,[labels], axis = 0)
                        #plot_events_on_data(peaks, blockdata)
                        num = 1
                        if idx > startblock:
                          alignclusterlabels(labels, peaklist, peaks,data=blockdata)
                        peaks, peaklist = ampwalkclassify3_refactor(peaks, peaklist) # classification by amplitude
                        minlen = 6   # >=1
                        peaks = discard_short_classes(peaks, minlen)
                        if len(peaks[0]) > 0:
                            peaks = discard_wave_pulses(peaks, blockdata)
                        # plots the data part and its detected and classified peaks
                        if plot_steps == True:
                            plot_events_on_data(peaks, blockdata)
                            pass
                    worldpeaks = np.copy(peaks)
                    # change peaks location in the buffered part to the location relative to the
                    peaklist.len = nblock
                    # peaklocations relative to whole recording 
                    worldpeaks[0] = worldpeaks[0] + (idx*nblock)
                    thisblock_eods = np.delete(peaks,3,0)
                    # save the peaks of the current buffered part to a numpy-memmap on the disk
                    mmpname = "eods_"+filename[:-3]+"npmmp"
                    save_EOD_events_to_npmmp(thisblock_eods,eods_len,idx==startblock,datasavepath,mmpname)
                    eods_len += len(thisblock_eods[0])
        # after the last buffered part has finished, save the memory mapped
        # numpy file of the detected and classified EODs to a .npy file to the
        # disk
        eods = np.memmap(datasavepath+"/eods_"+filename[:-3]+"npmmp", dtype='float64', mode='r+', shape=(4,eods_len), order = 'F')
        if save == 1:
           path = datasavepath+"/"
           if not os.path.exists(path):
               os.makedirs(path)
           if eods_len > 0:
               print('Saved!')
               np.save(datasavepath+"/eods8_"+datasavepath+"npy", eods)
           else:
               #np.save(filename[:-4]+"/eods5_"+filename[:-3]+"npy", thisblock_eods)

               print('not saved')
    else: # if there already has been a certain existing result file and 'new' was set to False
        print('already analyzed')
    print('returnes analyzed EODS. Calculate frequencies using all of these but discard the data from the EODS within the lowest few percent of amplitude')
    return eods

def main():
   # if len(sys.argv[:])<5:
   #     print('arguments missing. required: filepath to .WAV and whether to save results, plot each step and start a new analysis as 1 (True) or 0 (False) each')
   # elif len(sys.argv[:])>5:
   #     starttime = int(sys.argv[5])
   #     endtime = int(sys.argv[6])

   # #eods = analyze_pulse_data(sys.argv[1], starttime=0, endtime=50, save=0, new=1, plot_steps=0)
   # #eods = analyze_pulse_data(sys.argv[1], starttime=0, endtime=50, save=0, npmmp=True, plot_steps=0)
   # #eods = analyze_pulse_data(sys.argv[1], save=0, npmmp=True, plot_steps=0)
    print(sys.argv[1]) 
    eods = analyze_pulse_data(sys.argv[1], save=True, npmmp=True)
    print(eods)
#
#    colors = plt.get_cmap('jet')(np.linspace(0, 1.0, np.unique(eods[3])[-1]+1)) #len(np.unique(classlist))))
#    np.random.seed(22)
#    np.random.shuffle(colors)
#    colors = [colors[int(cl)] for cl in eods[3]]
#
#    plt.scatter(eods[0], eods[1], color = colors)
#    plt.show()
#
    #analyze_long_pulse_data_file(*sys.argv[1:])

if __name__ == '__main__':
    main()

