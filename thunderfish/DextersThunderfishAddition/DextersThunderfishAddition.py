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
       event_indices = np.where(events_delete != 1)[0]
       if np.count_nonzero(events_delete)==0:
           unchanged = True
       counter += 1
       if counter > 2000:
           pass
           #print('Warning: unusual many discarding steps needed, unusually dense
                 #events')
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
    print(cut_width[1])
    alignwidth = int(np.ceil((max_offset) * int_fact))
    for pos in event_locations.view('int'):
        snippets.append(data[pos+cut_width[0]:pos+cut_width[1]])
 #   scaled_snips = np.empty_like(snippets)
 #   for i, snip in enumerate(snippets):
 #       top = -cut_width[0]
 #       #plt.plot(snip)
 #       scaled_snips[i] = snip * 1/heights[i]
 #       #plt.plot(scaledsnips[i])
 #   #plt.show()
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
        #plt.plot(interpoled_snip)
        if len(aligned_snip[~np.isnan(aligned_snip)])>0:
            aligned_snips[i] = aligned_snip
    #plt.show()
    return snippets, aligned_snips
