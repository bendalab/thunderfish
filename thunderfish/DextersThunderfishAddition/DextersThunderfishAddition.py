import sys
import numpy as np
import copy
#from scipy.stats import gmainan
from scipy import stats
from scipy import signal
from scipy import optimize
import matplotlib
from fish import ProgressFish
import matplotlib.pyplot as plt
#from thunderfish.dataloader import open_data
#from thunderfish.peakdetection import detect_main
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
#from sklearn import metrics
#from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import AgglomerativeClustering
from collections import deque
import ntpath
import nixio as nix
import time
import os
from shutil import copy2
from ownDataStructures import Peak, Tr, Peaklist
from IPython import embed

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
                h[...] = y- r_side_y
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
    print(event_indices)
    return event_indices, event_locations, event_heights
def crosscorrelation(sig, data):
    autocorr = signal.fftconvolve(data, sig[::-1],  mode='valid')
    return autocorr
def interpol(data, kind):
    #kind = 'linear' , 'cubic'
    width = len(data)
    x = np.linspace(0, width-1, num = width, endpoint = True)
    return interp1d(x, data[0:width], kind , assume_sorted=True)
def cut_snippets(data,event_locations,cut_width,int_met="linear",int_fact=10,max_offset = 1.5):
    snippets = []
    cut_width = [-cut_width, cut_width]
    alignwidth = int(np.ceil((max_offset) * int_fact))
    for pos in event_locations.astype('int'):
        snippets.append(data[pos+cut_width[0]:pos+cut_width[1]])
 #   scaled_snips = np.empty_like(snippets)
 #   for i, snip in enumerate(snippets):
 #       top = -cut_width[0]
 #       #plt.plot(snip)
 #       scaled_snips[i] = snip * 1/heights[i]
 #       #plt.plot(scaledsnips[i])
    ipoled_snips = np.empty((len(snippets), (cut_width[1]-cut_width[0])*int_fact-int_fact))
    for i, snip in enumerate(snippets):
       if len(snip) < ((cut_width[1]-cut_width[0])):
            if i == 0:
                snip = np.concatenate([np.zeros([((cut_width[1]-cut_width[0]) - len(snip))]),np.array(snip)])
            if i == len(snippets):
                snip = np.concatenate([snip, np.zeros([((cut_width[1]-cut_width[0])-len(snip))])])
            else:
                snip = np.zeros([(cut_width[1]-cut_width[0])])
       f_interpoled = interpol(snip, int_met) #if len(snip) > 0 else np.zeros([(cut_width[1]-cut_width[0]-1)*int_fact ])
       interpoled_snip = f_interpoled(np.arange(0, len(snip)-1, 1/int_fact))
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
    return snippets, aligned_snips
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
   # (observations, features) matrix
    pc_comp= PCA().fit_transform(dataset)
    return pc_comp

def dbscan(pcs, events, order, eps, min_samples, takekm, olddatalen):
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

def cluster_events(features, events, order, eps, min_samples, takekm, olddatalen, method):#**kwargs):
    """
    clusters the given events using the given feature space and the clustering algorithm of choice and appends the assigned cluster number to the event's properties.

    Parameters
    ----------

    Returns
    -------

    """
    ########################      function maybe could be even more generic, ? (dependant on datatype of "events" )
    if method == 'DBSCAN':
        labels = dbscan(features,events, order, eps, min_samples, takekm, olddatalen)
    elif method == 'kMean':
        pass
        # To be implemented
        #labels = kmeans([])
    events = np.append(events,[labels], axis = 0)
    return events


def ampwalkclassify3_refactor(peaks,peaklist, thresh):
    """

        Needs big change to be more general/generic ? maybe not even worth it?

        Takes List of peaks and list of properties of the list of the last analysis block
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
    # for i in range(start, len(peaks)-start):
    lastofclass = peaklist.lastofclass        # dict of a lists of the last few heightvalues of a class, f.E ((1,[0.7,0.68,0.71]), (5, [0.2, 0.21, 0.21]))
    lastofclassx = peaklist.lastofclassx      # dict of a list of the last few x-values of a class
    a=0
    elem = 0
    thresholder = []
    comperr = 1
    classesnearby = peaklist.classesnearby     # list of the classes of the last n peaks (currently 12)  f.E:[1,2,1,2,1,3,2,1,...]
    classesnearbyx = peaklist.classesnearbyx   # list of the x-values of the last n peaks,               f.E:[13300, 13460, 13587, 13690, 13701, ...]
    classesnearbypccl = peaklist.classesnearbypccl # list of the pc-classified classes of the last n peaks
    classes = np.zeros((len(peaks[0])))
    pcclasses = peaks[3]
    positions = peaks[0]
    heights = peaks[1]

   # #print('nearbyclasses at start:' ,classesnearby, classesnearbyx)
   # for peak in peaks:
   #          peak.cl = peak.pccl+2 
   # peaklist.classlist =  np.vectorize(lambda peak: peak.cl, otypes=[object])(peaklist.list)
   # return peaks
    cl = 0
    maxdistance = 30000    #    Max distance to possibly belong to the same class
    factor = 1.6     # factor by which a peak fits into a class, f.E: classheight = 1 , factor = 2 => peaks accepted in range (0.5,2)
    c=0
    peakamount = len(peaks.T)
    #fish = ProgressFish(total = peakamount)
    for peaknum, p in enumerate(peaks.T):
        perc = str((peaknum*100)//peakamount)
    #    fish.animate(amount = "", dexextra = 'Partstatus: '+ ' '*(3-len(perc)) +perc + ' % (' + ' '*(4-len(str(peaknum)))+str(peaknum) + '/' + ' ' *(4-len(str(peakamount)))+str(peakamount) + '), Filestatus:')
        awc_btime = []
        if len(lastofclass) == 0:    # Dict with all classes, containing the heights of the last few peaks
            lastofclass[1] = deque()
            lastofclassx[1]= deque()
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
            if  (positions[peaknum]-classesnearbyx[i]) > maxdistance:
                classesnearby.pop(i)
                classesnearbyx.pop(i)
                classesnearbypccl.pop(i)
        lastofclassisis = []
        for i in classesnearby:
          #  print(i, classesnearby)
            lastofclassisis.append(np.median(np.diff(lastofclassx[i])))
        meanisi = np.mean(lastofclassisis)
        if 32000 > 20*meanisi> 6000:
            maxdistance = 20*meanisi
            #print(meanisi, maxdistance , 'maxdistance ----------------------------------------------------------------------------------------------')

        time2 = time.time()
        awc_btime.append(time2-time1) #0
        cl = 0  # 'No class'
        comperr = 1
        ##print('classesnearby at a peak', classesnearby)
        clnrby = np.unique(classesnearby)
        time1 = time.time()
#        classmean = 0
       #     if pcclasses[peaknum] == -1:
       #         factor = 1.2
       #     else:
       #         factor = 1.6

        for i in clnrby:
            #print('cl: ', i)
          #  if classesnearbypccl[classesnearby.index(i)] == -1:
          #      factor = 2.2
          #  else: factor = 1.6
            classmean = np.mean(lastofclass[i])
            logerror = np.abs(np.log2(heights[peaknum])-np.log2(classmean)) 
            abserror = np.abs(heights[peaknum]-classmean) 
            logthresh = np.log2(factor) 
            #ä#print(np.std(lastofclass[i])) absthresh = 0.5*classmean #  #print('test log', np.abs(np.log2(np.array([0.4,0.5,1,1.5,2,2.4]))-np.log2(np.array([1,1,1,1,1,1]))) ) #   abs(classmean*0.5)
            #relerror = error 
            relerror = logerror
            relabserror = abserror/thresh
           # if 1140 < p.num < 1150:
           #     print(p.num)
           #     print('for classes at one peak: classmean, height, abserror, thresh',
           #       classmean,heights[peaknum], logerror, logthresh)
            #print(len(classesnearbypccl), len(classesnearby))
            #print(classmean, heights[peaknum], logerror, logthresh, pcclasses[peaknum], classesnearbypccl[classesnearby.index(i)])
            if classesnearbypccl[classesnearby.index(i)] == pcclasses[peaknum] or pcclasses[peaknum] == -1:# or  
              if logerror < logthresh:     ## SameClass-Condition
                if relerror < comperr and (positions[peaknum]-classesnearbyx[classesnearby.index(i)])<maxdistance:
                    holdlastcl = cl
                    cl = i
#                    if comperr/error < np.log2(1.02):
#                        factor = 1.5
                    holdlastcomperr = comperr
                    comperr = relerror
        #    else:
                #print('classfitsnot-------------------------------------------------------------------------------------------------')
        #else:
        #    if cl == 0:
                #print('noclassfits!')
#        if cl!=0 and holdlastcl != 0 and holdlastcomperr - relerror < 0.05 * holdlastcomperr:
#            if np.std(np.diff(lastofclassx[holdlastcl])) < 0.05 * np.mean(np.diff(lastofclassx[holdlastcl])) and np.std(np.diff(lastofclassx[cl])) < 0.05 * np.mean(np.diff(lastofclassx[cl])):
#
#                isicompaeecl =  np.mean(np.diff(lastofclassx[holdlastcl]))
#                isithiscl = np.mean(np.diff(lastofclassx[cl]))
#                isithis =  positions[peaknum] - lastofclassx[cl][-1]
#                isicompare = positions[peaknum]  -lastofclassx[holdlastcl][-1]
#                thisisierror = (isithis-isithiscl)/isithiscl
#                compareisierror = (isicompare-isicomparecl)/isicomparecl
#                #print('isidecision!', isithis, isithiscl, isicompare, isicomparecl)
#                if thisisierror > 2*compareisierror:
#                    cl = holdlastcl

        time2 = time.time()
        awc_btime.append(time2-time1) #1
        time1 = time.time()
        if pcclasses[peaknum] != -1:
            if cl != 0 :
                #print(cl)
                if len(lastofclass[cl]) >= 3:
                    lastofclass[cl].popleft()
                if len(lastofclassx[cl]) >= 3:
                    lastofclassx[cl].popleft()
                lastofclass[cl].append(heights[peaknum])
                lastofclassx[cl].append(positions[peaknum])
                classes[peaknum] = cl
            else:                                    # Add new class
                cl = classamount+1
                #print('existingclasses: ', classamount) 
                classamount = cl
                
                #print('newclass: ----------------------------------------------------------------', cl)
                lastofclass[cl] = deque()
                lastofclassx[cl] = deque()
                lastofclass[cl].append(heights[peaknum])
                lastofclassx[cl].append(positions[peaknum])
                classes[peaknum] = cl
                classesnearby.append(cl)
                classesnearbyx.append(positions[peaknum])
                classesnearbypccl.append(pcclasses[peaknum])
            ##print('tatsaechlich: ', cl)         
            if len(classesnearby) >= 12: #kacke implementiert? 
                minind = classesnearbyx.index(min(classesnearbyx))
                del lastofclass[classesnearby[minind]]
                del lastofclassx[classesnearby[minind]]
                #print(classesnearby[minind], 'del')
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
            #    #print(ind ,' --------------------------------------here -----------------------------')
            except ValueError:
                classesnearby.append(cl)
                classesnearbyx.append(positions[peaknum])
                classesnearbypccl.append(pcclasses[peaknum])
        else:
            if cl != 0:
                classes[peaknum] = cl
            else:
                cl = classamount+1
                #print('existingclasses: ', classamount) 
                classamount = cl
                #print('newclass: ', cl)
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
                #print(classesnearby[minind], 'del')
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
            #    #print(ind ,' --------------------------------------here -----------------------------')
            except ValueError:
                classesnearby.append(cl)
                classesnearbyx.append(positions[peaknum])
                classesnearbypccl.append(pcclasses[peaknum])
           #     #print('classesnearby after a peak', classesnearby)
  #     for clnum, cls in enumerate(classesnearby): ## deleting almost identical classes (< % difference in amplitude)
  #         if cls == False:
  #             continue
  #         if True:
  #             continue
  #         compare = np.mean(lastofclass[cls])
  #         for i in classesnearby[clnum:-1]:
  #             if i== False:
  #                 continue
  #             if i != cls and abs(compare - np.mean(lastofclass[i])) < compare*0.01:   ## 
  #              #   #print(compare)
  #              #   #print( np.mean(np.vectorize(lambda peak: peak.height)(lastofclass[i])))
  #                 clindex = classesnearby.index(cls)
  #                 classesnearby[clindex] = False
  #                 classesnearbyx[clindex] = False
  #                 del lastofclass[cls]
  #                 del lastofclassx[cls]
  #                # cl = holdlastcl
  #                # if cl == cls:
  #                     
  #                     
  #                 #print('combinedsomeclasses that were similar', cl, cls)
        time2 = time.time()
  #      awc_btime.append(time2-time1) #2
  #      classesnearby = [cls for cls in classesnearby if cls != False] 
  #      classesnearbyx = [clx for clx in classesnearbyx if clx != False]
  # 
  # 
        #print('awc_btime ', awc_btime , ' newpeak-------------------------------------------------------- :')
    peaklist.lastofclass = lastofclass
    peaklist.lastofclassx = lastofclassx
    peaklist.classesnearby = classesnearby
    peaklist.classesnearbyx = classesnearbyx
    peaklist.classlist =  classes # np.vectorize(lambda peak: peak.cl, otypes=[object])(peaklist.list)
    peaklist.classamount = classamount
    peaks = np.append(peaks,classes[None,:], axis = 0)
    return peaks, peaklist

def discard_wave_pulses(peaks, data):
     '''
        discards events from a pulse_event list which are unusally wide (wider than a tenth of the inter pulse interval), which indicates a wave-type EOD instead of a pulse type

        returns: "peaks"
     '''
     deleteclasses = []
     for cl in np.unique(peaks[3]):
         peaksofclass = peaks[:,peaks[3] == cl]
         isi = np.diff(peaksofclass[0])
         isi_mean = np.mean(isi)
         widepeaks = 0
         isi_tenth_area = lambda x, isi:np.arange(np.floor(x-0.1*isi),np.ceil(x+0.1*isi),1, dtype = np.int)
         for p in peaksofclass.T:
             data = np.array(data)
             try:
                 for dp_around in data[isi_tenth_area(p[0],isi_mean)]:
                    if dp_around <= p[1]-p[2]:
                       break
             except IndexError:
                 pass
         else:
             widepeaks+=1
         if widepeaks > len(peaksofclass)*0.5:
             deleteclasses.append(cl)
     for cl in deleteclasses:
            peaks = peaks[:,peaks[3]!=cl]
     return peaks


def plot_events_on_data(peaks, data):
    '''
        plots the detected events onto the data timeseries. If the events are classified, the classes are plotted in different colors and the class -1 (not belonging to a cluster) is plotted in black
    '''
    plt.plot(range(len(data)),data, color = 'black')
    if len(peaks) > 3:
        classlist = np.array(peaks[3],dtype=np.int)
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
    classlist = events[3]
    smallclasses = [cl for cl in np.unique(classlist) if len(classlist[classlist
                                                                     == cl]) <
                 minlen]
    delete = np.zeros(len(classlist))
    for cl in smallclasses:
        delete[classlist == cl] == 1
        events = events[:,delete != 1]
    return events


def analyze_data(path_to_data, path_to_save_results, save = 'n', plot = 'y', new = 'y', starttime = '0', endtime = '0'):
    #    Script to detect and classify EODs in recordings of weakly electric pulse
    #    fish, Dexter Früh, 2018
    # #    it is suggested to save the recording in
    #       workingdirectory/recording/recording.WAV

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
    #    call with:
    #    python3 scriptname.py save plot new (starttime endtime[sec] for only
    #                                                       partial analysis)
    #
    #   other parameters are behind imports and some hardcoded at the relevant
    #       codestep
    import sys
    import numpy as np
    import copy
    from scipy.stats import gmean
    from scipy import stats
    from scipy import signal
    from scipy import optimize
    import matplotlib
    from fish import ProgressFish
    import matplotlib.pyplot as plt
    from thunderfish.dataloader import open_data
    from thunderfish.peakdetection import detect_peaks
    from scipy.interpolate import interp1d
    from scipy.signal import savgol_filter
    from collections import deque
    import ntpath
    import nixio as nix
    import time
    import os
    from shutil import copy2
    from ownDataStructures import Peak, Tr, Peaklist
    import DextersThunderfishAddition as dta
    from IPython import embed
    # parameters for the analysis

    deltat = 30.0  # seconds of buffer size
    thresh = 0.04 # minimal threshold for peakdetection
    peakwidth = 20 # width of a peak and minimal distance between two EODs
    # basic parameters for thunderfish.dataloader.open_data
    verbose = 0
    channel = 0
    # timeinterval to analyze other than the whole recording
    #starttime = 0
    #endtime = 0
    #timegiven = False

    def main():         #  analyse_dex.py filename save plot new  (optional starttime endtime [sec])
        home = os.path.expanduser('~')
        os.chdir(home)
        # defaults for optional arguments
        timegiven = False
        plot_steps = False
        # parse command line arguments - filepath, save, plot, new (, starttime,
        filepath = sys.argv[1]
        if endtime != '0':
            timegiven = True
        peaks = np.array([])
        troughs = np.array([])
        cutsize = 20
        maxwidth = 50 #10
        ultimate_threshold = thresh+0.01
        filename = path_leaf(filepath)
        proceed = input('Currently operates in home directory. If given a pulsefish recording filename.WAV, then a folder filename/ will be created in the home directory and all relevant files will be stored there. continue? [y/n]').lower()
        if proceed == 'n':
         quit()
        elif proceed == 'y':
            pass
        #do something
        elif proceed != 'y':
             quit()
        datasavepath = filename[:-4]
        print(datasavepath)
        eods_len = 0
        ### ## starting analysis
        if new == 1 or not os.path.exists(filename[:-4]+"/eods5_"+filename[:-3]+"npy"):
            ### ##  import data
            with open_data(filepath, channel, deltat, 0.0, verbose) as data:
               if save == 1 or save == 0:
                   if not os.path.exists(datasavepath):
                       os.makedirs(datasavepath)
                       copy2(filepath, datasavepath)
               samplerate = data.samplerate
               ### ## split datalength into smaller blocks
               nblock = int(deltat*data.samplerate)
               if timegiven == True:
                   parttime1 = starttime*samplerate
                   parttime2 = endtime*samplerate
                   data = data[parttime1:parttime2]
               if len(data)%nblock != 0:
                   blockamount = len(data)//nblock + 1
               else:
                   blockamount = len(data)//nblock
               bigblock = []
               ### ## output first (0%) progress bar
               print('blockamount: ' , blockamount)
               progress = 0
               print(progress, '%' , end = " ", flush = True)
               fish = ProgressFish(total = blockamount)
               olddatalen = 0
               startblock = 0
               ## iterating through the blocks, detecting peaks in each block
               for idx in range(startblock, blockamount):
                   ### ## print progress
                   if progress < (idx*100 //blockamount):
                       progress = (idx*100)//blockamount
                   progressstr = 'Partstatus: '+ str(0) + ' '*2 + ' % (' + '0' + ' '*4+ '/' + '?'+' '*4+ '), Filestatus:'
                   fish.animate(amount = idx, dexextra = progressstr)
                   progressstr = 'Partstatus: '+ 'Part ' + '0'+ '/''5'+' Filestatus:'
                   fish.animate(amount = idx, dexextra = progressstr)
                   datx = data[idx*nblock:(idx+1)*nblock]
                   # ---------- analysis --------------------------------------------------------------------------
                   # step1: detect peaks in timeseries
                   pk, tr = detect_peaks(datx, thresh)
                   troughs = tr
                   # continue with analysis only if multiple peaks are detected
                   if len(pk) > 2:
                       peaks = dta.makeeventlist(pk,tr,datx,peakwidth)
                       #dta.plot_events_on_data(peaks, datx)
                       peakindices, peakx, peakh = dta.discardnearbyevents(peaks[0],peaks[1],peakwidth)
                       peaks = peaks[:,peakindices]
                       progressstr = 'Partstatus: '+ 'Part ' + '1'+ '/''5'+' Filestatus:'
                       fish.animate(amount = idx, dexextra = progressstr)
                       if len(peaks) > 0:
                           ### ## connects the current part with the one that came before, to allow for a continuous analysis
                           if idx >= startblock+1:
                               peaklist = connect_blocks(peaklist)
                           else:
                               peaklist = Peaklist([])
                           snips, aligned_snips = dta.cut_snippets(datx,peaks[0], 15, int_met = "cubic", int_fact = 10,max_offset = 1.5)
                           progressstr = 'Partstatus: '+ 'Part ' + '2'+ '/''5'+' Filestatus:'
                           fish.animate(amount = idx, dexextra = progressstr)
                           # calculates principal components
                           pcs = dta.pc(aligned_snips)#pc_refactor(aligned_snips)
                           #print('dbscan')
                           # clusters the features(principal components) using dbscan algorithm. clusterclasses are saved into the peak-object as Peak.pccl
                           order = 5
                           minpeaks = 3 if deltat < 2 else 10
                           peaks = dta.cluster_events(pcs, peaks, order, 0.4, minpeaks, False, olddatalen, method = 'DBSCAN')
                           
                           #dta.plot_events_on_data(peaks, datx)
                           olddatalen = len(datx)
                           num = 1
                           progressstr = 'Partstatus: '+ 'Part ' + '3'+ '/''5'+' Filestatus:'
                           fish.animate(amount = idx, dexextra = progressstr)
                           # classifies the peaks using the data from the clustered classes and a simple amplitude-walk which classifies peaks as different classes if their amplitude is too far from any other classes' last three peaks
                           peaks, peaklist = dta.ampwalkclassify3_refactor(peaks, peaklist, thresh) # classification by amplitude
                           #join_count=0
                         #  while True and joincc(peaklist, peaks) == True and join_count < 200:
                         #        join_count += 1
                         #        continue
                           # discards all classes that contain less than mincl EODs
                           minlen = 6   # >=1
                           peaks = dta.discard_short_classes(peaks, minlen)
                           if len(peaks[0]) > 0:
                               peaks = dta.discard_wave_pulses(peaks, datx)
                           # plots the data part and its detected and classified peaks
                           if plot_steps == True:
                               dta.plot_events_on_data(peaks, datx)
                               pass
                       # map the analyzed EODs of the buffer part to the whole
                       # recording
                       worldpeaks = np.copy(peaks)
                       # change peaks location in the buffered part to the location relative to the
                       idx = 1
                       # peaklocations relative to whole recording 
                       worldpeaks[0] = worldpeaks[0] + (idx*nblock)
                       peaklist.len = idx*nblock
                       thisblock_eods = np.delete(peaks,3,0)
                       thisblockeods_len = len(thisblock_eods[0])
                       progressstr = 'Partstatus: '+ 'Part ' + '4'+ '/''5'+' Filestatus:'
                       fish.animate(amount = idx, dexextra = progressstr)
                       # save the peaks of the current buffered part to a numpy-memmap on the disk
                       if thisblockeods_len> 0 and save == 1 or save == 0:
                           if idx == 0:
                                   eods = np.memmap(datasavepath+"/eods_"+filename[:-3]+"npmmp", dtype='float64', mode='w+', shape=(4,thisblockeods_len), order = 'F')
                           dtypesize = 8#4 #float32 is 32bit = >4< bytes long  ---changed to float64 -> 8bit
                           eods = np.memmap(datasavepath+"/eods_"+filename[:-3]+"npmmp", dtype='float64', mode='r+', offset = dtypesize*eods_len*4, shape=(4,thisblockeods_len), order = 'F')
                           eods[:] = thisblock_eods
                           eods_len += thisblockeods_len
                       # to clean the plt buffer...
                       plt.close()
                       # get and print the measured times of the algorithm parts for the
                       # current buffer
                       progressstr = 'Partstatus: '+ 'Part ' + '5'+ '/''5'+' Filestatus:'
                       fish.animate(amount = idx, dexextra = progressstr)
                     #  plt.show()
            # after the last buffered part has finished, save the memory mapped
            # numpy file of the detected and classified EODs to a .npy file to the
            # disk
            eods = np.memmap(datasavepath+"/eods_"+filename[:-3]+"npmmp", dtype='float64', mode='r+', shape=(4,eods_len), order = 'F')
            print('before final saving: print unique eodcl: ' , np.unique(eods[3]))
            if save == 1:
               # #print('eods', eods[3])
               path = filename[:-4]+"/"
               if not os.path.exists(path):
                   os.makedirs(path)
               if eods_len > 0:
                   print('Saved!')
                   np.save(filename[:-4]+"/eods8_"+filename[:-3]+"npy", eods)
               else:
                   #np.save(filename[:-4]+"/eods5_"+filename[:-3]+"npy", thisblock_eods)
                   print('not saved')

        else: # if there already has been a certain existing result file and 'new' was set to False
            print('already analyzed')


