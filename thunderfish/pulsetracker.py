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

from collections import OrderedDict

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

def cut_snippets(data,event_locations,cut_width,int_met="linear",int_fact=10,max_offset = 1000000): 
    """
    cuts intervals from a data array, interpolates and aligns them and returns them in a list

    TODO: ALIGN THEM TO CAUSE LEAST SQUARE ERROR

    Parameters
    ----------
    data: array

    event_locations: array

    cut_width: [int, int]
        lower and upper limit of the intervals relative to the event locations.
        f.e. [-15,15] indicates an interval of 30 datapoints around each event location

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
    heights = np.zeros(len(event_locations))
    cut_width = [-cut_width, cut_width]
    #alignwidth = int(np.ceil((max_offset) * int_fact))
    alignwidth = 50

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
        try:
            heights[i] = np.max(interpoled_snip[alignwidth-offset:-alignwidth-offset]) - np.min(interpoled_snip[alignwidth-offset:-alignwidth-offset])
        except:
            heights[i] = np.max(interpoled_snip[2*alignwidth:]) - np.min(interpoled_snip[2*alignwidth:])
    return aligned_snips, heights

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
    pca = PCA(n_components=10)
    pc_comp = pca.fit_transform(dataset)

    return pc_comp #, pca

def chebyshev(dataset):
    x = range(len(dataset[0]))
    npol=5
    p = np.zeros((len(dataset),npol+1))
    for i,s in enumerate(dataset):
        cheb = np.polynomial.chebyshev.Chebyshev.fit(x,s,npol)
        p[i] = cheb.coef

    return p #, pca


def dbscan(pcs, events, eps, min_samples, takekm):
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
    #try:
    #    X = pcs[:,:order]
    #except:
    #    X = pcs[:,order]
    X = pcs
    
    # #############################################################################
    # Compute DBSCAN
    
    clusters = DBSCAN(eps, min_samples).fit(X)
    labels = clusters.labels_

    comp = clusters.components_

    comp_means = np.zeros((len(np.unique(labels)) - 1,comp.shape[1]))

    for i in range(len(np.unique(labels)) - 1):
        comp_means[i] = np.mean(pcs[labels==i],axis=0)

    return labels, comp_means

def cluster_events(features, events, eps, min_samples, takekm, method='DBSCAN'):
    """F
    clusters the given events using the given feature space and the clustering algorithm of choice and appends the assigned cluster number to the event's properties.

    Parameters
    ----------

    Returns
    -------

    """
    ########################      function maybe could be even more generic, ? (dependant on datatype of "events" )
    if method == 'DBSCAN':
        labels, clusters = dbscan(features,events, eps, min_samples, takekm)
    elif method == 'kMean':
        pass
        # To be implemented
        #labels = kmeans([])
    return labels, clusters

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

    # take first second of new peak data
    overlapamount = len(peaks[0,peaks[0]<30000])
    if overlapamount == 0:
        return None

    old_peaklist = copy.deepcopy(peaklist)  #redundant 
    overlappeaks = copy.deepcopy(peaks[:,:overlapamount])
    overlap_peaklist = copy.deepcopy(old_peaklist)
    
    # delete cluster classifications of the overlap class
    overlappeaks[3]=[-1]*len(overlappeaks[0])

    # set nearby pc classes to -1
    overlap_peaklist.classesnearbypccl = [-1]*len(overlap_peaklist.classesnearbypccl)

    # create peak labels using ampwalk classifier
    classified_overlap = ampwalkclassify3_refactor(overlappeaks,overlap_peaklist,glue=True)[0]

    # for each class
    for cl in np.unique(classified_overlap[4]):

        # indexes of the peaks with current class by ampwalk classification
        labelindex = np.where(classified_overlap[4] == cl)[0]

        # pc clustering labels that were originally given to those peaks
        label = labels[labelindex]
            
        # index of a peak with the most common translation from ampwalk class to pc clustering class
        labelindex = labelindex[np.where(label == stats.mode(label)[0])[0][0]]
            
        # pc clustering label belonging to the class cl in the new block
        newlabel = labels[labelindex]

        try:
            peaklist.classesnearbypccl[old_peaklist.classesnearby.index(cl)] = newlabel
        except:
            pass

def ampwalkclassify3_refactor(peaks,peaklist,glue=False):
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

    # loop through all the new peaks
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

        # classes nearby only count if they are within maxdistance
        for i, cl in enumerate(classesnearby):
            if  (positions[peaknum] - classesnearbyx[i]) > maxdistance:
                classesnearby.pop(i)
                classesnearbyx.pop(i)
                classesnearbypccl.pop(i)

        # compute mean isi of a class by taking the last 3 pulses in that class
        lastofclassisis = []
        for i in classesnearby:
            lastofclassisis.append(np.median(np.diff(lastofclassx[i])))
        
        meanisi = np.mean(lastofclassisis)

        # stop adding to a class if 40 isis have passed
        if 32000 > 40*meanisi> 6000:
            maxdistance = 20*meanisi
        
        cl = 0  # 'No class'
        comperr = 100
        clnrby = np.unique(classesnearby)

        last_err = 1000

        # TODO this assigns peaks with no class to the last clase if there are multiple candidates. 
        # can I fix this?
        for i in clnrby:

            # if the class of the current peak is equal to the current evaluated class or current peak has no class
            if classesnearbypccl[classesnearby.index(i)] == pcclasses[peaknum]: #or glue==True:   #pcclasses[peaknum] == -1: or  
              classmean = np.mean(lastofclass[i]) #mean hight of class
                
              # difference between current peak hight and average class hight
              logerror = np.abs(np.log2(heights[peaknum])-np.log2(classmean))
              logthresh = np.log2(factor)
              relerror = logerror
                
              # if the peak hights are similar
              if logerror < logthresh:     ## SameClass-Condition
                # if the peaks are close together in distance (20*isi)
                if relerror < comperr and (positions[peaknum]-classesnearbyx[classesnearby.index(i)])<maxdistance:
                  # keep the same class (or in case of no class assign that class)
                  cl = i
                  comperr = relerror
        time2 = time.time()
        time1 = time.time()
        
        # if a pc class is assigned to the peak
        if pcclasses[peaknum] != -1:
            # if the class is kept
            if cl != 0 :
                # append this peak to the peaklist for the right class (only keep last 3 peaks)
                if len(lastofclass[cl]) >= 3:
                    lastofclass[cl].popleft()
                if len(lastofclassx[cl]) >= 3:
                    lastofclassx[cl].popleft()
                lastofclass[cl].append(heights[peaknum])
                lastofclassx[cl].append(positions[peaknum])
                classes[peaknum] = cl
            else:             
                # if the class if not the same as any of the existing classes, create new class
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
            
            # if there are more than 12 classes, delete the class that is furthest away in proximity
            if len(classesnearby) >= 12: #kacke implementiert? 
                minind = classesnearbyx.index(min(classesnearbyx))
                del lastofclass[classesnearby[minind]]
                del lastofclassx[classesnearby[minind]]
                classesnearby.pop(minind)
                classesnearbyx.pop(minind)
                classesnearbypccl.pop(minind)

            # add position and class to peaklist
            try:
                ind=classesnearby.index(cl)
                classesnearbyx[ind] = positions[peaknum]
            except ValueError:
                classesnearby.append(cl)
                classesnearbyx.append(positions[peaknum])
                classesnearbypccl.append(pcclasses[peaknum])
        
        # if no pc class is assigned to the peak
        elif glue == True:
            # if a class is assigned through the peak amp method
            if cl != 0:
                # add this class to the peak
                classes[peaknum] = cl
            else:
                # create new class
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

def plot_events_on_data(peaks, data, colors):
    """
        plots the detected events onto the data timeseries. If the events are classified, the classes are plotted in different colors and the class -1 (not belonging to a cluster) is plotted in black
    """
    plt.plot(range(len(data)),data, color = 'black')
    if len(peaks)>3:
        classlist =  np.array(peaks[3],dtype=np.int)
        if len(peaks) > 4:
            classlist = np.array(peaks[4],dtype=np.int)
        #classlist=labels
        cmap = plt.get_cmap('jet')

        for cl in np.unique(classlist):

            if cl == -1:
                color = 'black'
            else:
                color = colors[cl]
            peaksofclass = peaks[:,classlist == cl]
            plt.plot(peaksofclass[0],peaksofclass[1], '.', color = color,   ms =20, label=cl)
            plt.legend()
    else:
        plt.scatter(peaks[0],peaks[1], color = 'red')
    plt.show()
    plt.close()

def discard_short_classes(events, minlen):
    """ 
        returns all events despite events which are in classes with less than minlen members
    """

    u, c = np.unique(events[-1],return_counts=True)

    smallclasses = u[c<minlen]
    classlist = events[-1]
    
    delete = np.zeros(len(classlist))

    for cl in smallclasses:
        delete[classlist == cl] = 1
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


def create_threshold_array(data,window,threshold):
    thr_array = np.zeros(data.shape)
    for i in range(int(len(data)/window)):
        thr_array[i*window:(i+1)*window] = np.std(data[i*window:(i+1)*window])*4
    thr_array[thr_array<threshold] = threshold
    return thr_array

def alignlabels(labels,clusters,old_labels,old_clusters,maxlabel):

    old_labels = old_labels[old_labels!=-1]

    labels_new = -1*np.ones(labels.shape)
    newclass = maxlabel
    
    for curlabel, cluster in enumerate(clusters):
        n = np.linalg.norm(old_clusters-cluster,axis=1)

        if np.min(n) < 0.1:
            labels_new[labels==curlabel] = old_labels[np.argmin(n)]
        else:
            labels_new[labels==curlabel] = newclass
            newclass = newclass + 1

    return labels_new


def analyze_pulse_data(filepath, deltat=10, thresh=0.04, starttime = 0, endtime = 0, cluster_thresh = 0.1, savepath = False,save=False, npmmp = False, plot_eods=False,plot_features=False,plot_steps=False, plot_result=False):

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
    
    cluster_thresh: float, optional
        threshold that decides the cluster density of the EOD waveform features.

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

    plot_eods: Boolean, optional
        True to plot the EOD waveforms for each analysis block

    plot_features: Boolean, optional
        True to plot the EOD waveform features for each analysis block

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
        #fish = ProgressFish(total = blockamount)

        pca_cur = 0
        progress = 0

        for idx in range(0, blockamount):
            print('BLOCK %i/%i'%(idx+1,blockamount))

            blockdata = data[idx*nblock:(idx+1)*nblock]
            if progress < (idx*100 //blockamount):
                progress = (idx*100)//blockamount
            progressstr = ' Filestatus: '

            # fish.animate(amount = idx, dexextra = progressstr)
             # delete peaks under absolute threshold
            

            #thresh_array = create_threshold_array(blockdata,30000,thresh)            
            pk, tr = detect_peaks(blockdata, thresh)
            troughs = tr

            if len(pk) > 3:
                
                peaks = makeeventlist(pk,tr,blockdata,peakwidth)
                peakindices, peakx, peakh = discardnearbyevents(peaks[0],peaks[1],peakwidth)
                peaks = peaks[:,peakindices]

                if len(peaks) > 0:

                    #if idx > startblock:
                    #    # adding a new block as copy of old list, only difference is peak indexing as it refers to last block
                    #    peaklist = connect_blocks(peaklist)
                    #else:
                    #    peaklist = Peaklist([])

                    aligned_snips, snip_heights = cut_snippets(blockdata,peaks[0], 30, int_met = "cubic", int_fact = 10,max_offset = 20)
                    
                    pols = chebyshev(aligned_snips)

                    feats = np.zeros((pols.shape[0],pols.shape[1]+1))
                    feats[:,:6] = pols
                    feats[:,-1] = snip_heights*0.1
                    #pcs, pca_cur = pc(aligned_snips) #pc_refactor(aligned_snips)
                    
                    minpeaks = 3 if deltat < 2 else 10
                    
                    labels, clusters = cluster_events(feats, peaks, cluster_thresh, minpeaks, False, method = 'DBSCAN')
                    peaks = np.append(peaks,[labels], axis = 0)
                    
                    if idx > startblock:
                      # instead of the peaklist I would have to add the previous cluster means
                      # alignclusterlabels(labels, peaklist, peaks,data=blockdata)
                      peaks[-1] = alignlabels(labels,clusters,old_labels,old_clusters,maxlabel)
                    
                    old_labels = np.unique(peaks[-1])
                    old_clusters = clusters
                      #I would want peaks updated here to have the right pc classes as well..

                    #peaks, peaklist = ampwalkclassify3_refactor(peaks, peaklist) # classification by amplitude

                    minlen = 5
                    peaks = discard_short_classes(peaks, minlen)
                    
                    if len(peaks[0]) > 0:
                        peaks = discard_wave_pulses(peaks, blockdata)
                        # delete peaks under absolute threshold
                        #thresh_array = create_threshold_array(blockdata,30000)
                        #peaks = peaks[:,peaks[1]>thresh_array[list(map(int,peaks[0]))]]
                    
                    cmap = plt.get_cmap('jet')
                    colors = cmap(np.linspace(0, 1.0, 10))

                    if plot_steps == True:
                        plot_events_on_data(peaks, blockdata, colors)
                        pass

                    for lab in np.unique(labels):
                        
                        if lab == -1:
                            c = 'k'
                            z=-1
                        else:
                            c=colors[lab]
                            z=1
                        if plot_eods==True:
                            plt.plot(range(aligned_snips.shape[1]),np.transpose(aligned_snips[labels == lab]),color=c,zorder=z,label=lab)
                    
                    if plot_eods==True:
                        plt.title('Detected and classified EODs')
                        plt.xlabel('time [ms]')
                        plt.ylabel('signal (normalized)')
                    
                        phandles, plabels = plt.gca().get_legend_handles_labels()
                        by_label = OrderedDict(zip(plabels, phandles))
                        plt.legend(by_label.values(), by_label.keys())
                        plt.show()

                    for lab in np.unique(labels):
                        
                        if lab == -1:
                            c = 'k'
                            z = -1
                        else:
                            c = colors[lab]
                            z=1

                        if plot_features==True:
                            plt.plot(np.squeeze(np.transpose(feats[labels == lab])),color=c,zorder=z,label=lab)
                
                    if plot_features==True:
                        plt.title('EOD Features')
                        plt.xlabel('feature [#]')
                        plt.ylabel('value [a.u.]')
                    
                        phandles, plabels = plt.gca().get_legend_handles_labels()
                        by_label = OrderedDict(zip(plabels, phandles))
                        plt.legend(by_label.values(), by_label.keys())
                        plt.show()

                    #peaklist.len = nblock
                    worldpeaks = np.copy(peaks)
                    worldpeaks[0] = worldpeaks[0] + (idx*nblock)
                    # delete the classification that only considers wave shape.
                    #thisblock_eods = np.delete(worldpeaks,3,0)
                    thisblock_eods = worldpeaks

                    if idx == startblock:
                        maxlabel = np.max(peaks[-1]) + 1
                    else:
                        maxlabel = np.max([maxlabel, (np.max(peaks[-1]) + 1)])

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
        if plot_steps == True:
            print('FINAL RESULTS')
            plot_events_on_data(all_eods, data, colors)
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


def main():

    eods = analyze_pulse_data(sys.argv[1], save=True, npmmp=True)
    print(eods)

if __name__ == '__main__':
    main()

