import numpy as np
from numba import jit
import matplotlib.pyplot as plt


# numba doesnt like lists.  convert everything to numpy.

def makeeventlist(main_event_positions, side_event_positions, data, event_width=20, min_width=2,verbose=0):
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
        Positions of the detected 
        side events in the data time series. The complimentary event to the main events.
    data: array of float
        The given data.
    event_width (optional): int
        Maximum EOD width (in samples).
    min_width (optional): int
        Minimum EOD width (in samples).
    verbose (optional): int
        Verbosity level.

    Returns
    -------
    EOD_events: ndarray
        2D array containing data with 'np.float' type, size (number_of_properties = 5, number_of_events).
        Generated and combined data of the detected events in an array with arrays of:
                x, y, height along the first axis, width and x of the most significant nearby trough.

    """
    mainfirst = int((min(main_event_positions[0],side_event_positions[0])<side_event_positions[0]))  # determines if there is a peak or through first. Evaluates to 1 if there is a peak first.
    mainlast = int((max(main_event_positions[-1],side_event_positions[-1])>side_event_positions[-1]))  # determines if there is a peak or through last. Evaluates to 1 if there is a peak last.
    xp = main_event_positions[mainfirst:len(main_event_positions)-mainlast]

    ind = np.arange(len(xp))

    #if len(xp)>len(side_event_positions):
    #    xp = xp[:-1]

    y = data[xp]
    
    l_side_ind = ind
    r_side_ind = l_side_ind + 1

    r_side_x = side_event_positions[r_side_ind]
    r_distance = np.abs(r_side_x - xp)
    r_side_y = data[r_side_x]

    l_side_x = side_event_positions[l_side_ind]
    l_distance = np.abs(xp - l_side_x)
    l_side_y = data[l_side_x]

    s_l = np.abs((y-l_side_y)/l_distance)
    s_r = np.abs((y-r_side_y)/r_distance)

    iw = np.argmax(np.vstack([np.abs(y-l_side_y),np.abs(y-r_side_y)]),axis=0)
    i = (np.abs(s_l-s_r)/(0.5*s_l+0.5*s_r) > 0.25)
    iw[i] = np.argmax(np.array(np.vstack([s_l[i],s_r[i]])),axis=0)

    mp = np.vstack([np.abs(iw-1),iw])
    h = np.sum(np.vstack([np.abs(y-l_side_y),np.abs(y-r_side_y)])*mp,axis=0) #calculated using absolutes in case of for example troughs instead of peaks as main events 

    w = np.sum(np.vstack([l_distance,r_distance])*mp,axis=0)
    xt = np.sum((xp + np.vstack([-l_distance,r_distance]))*mp,axis=0)

    print(min_width)
    print(event_width)

    r = ((w>min_width) & (w<event_width))

    return xp[r], xt[r], h[r], w[r]

#@jit(nopython=True)
def discardnearbyevents(event_locations, tr_locations, event_widths, event_slopes, min_distance,verbose=0):
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
    verbose (optional): int
        Verbosity level.

    Returns
    -------
    event_locations: array of int or float
        Positions of the returned events in the data time series.
    event_heights: array of int or float
        Heights of the returned events, indices refer to the same events as in
        event_locations.
    event_indices: array of int
        Indices of arrays which are not to be discarded
    """

    unchanged = False
    counter = 0
    event_indices = np.arange(0,len(event_locations)+1,1)
    while unchanged == False:

       print(len(event_locations))
       print(len(tr_locations))
       
       x_diffs = np.min(np.vstack([np.diff(event_locations),np.diff(tr_locations),np.abs(event_locations[1:]-tr_locations[:-1]),np.abs(event_locations[:-1]-tr_locations[1:])]),axis=0)
       events_delete = np.zeros(len(event_locations))

       for i, diff in enumerate(x_diffs):
           if diff < max(min_distance,max(event_widths[i],event_widths[i+1])*3):     
                if event_slopes[i+1] > event_slopes[i]:
                    # the width has to be at least 3*int_fact
                     events_delete[i] = 1
                else:
                     events_delete[i+1] = 1

       event_widths = event_widths[events_delete!=1]
       event_locations = event_locations[events_delete!=1]
       event_slopes = event_slopes[events_delete!=1]
       tr_locations = tr_locations[events_delete!=1]

       event_indices = event_indices[np.where(events_delete!=1)[0]]
       if np.count_nonzero(events_delete)==0:
           unchanged = True
       counter += 1
       if counter > 2000:
           print('Warning: unusual many discarding steps needed, unusually dense events')
           pass

    if verbose>0:
        print('Number of peaks after peak discarding:                  %5i'%(len(event_locations)))

    return event_indices

@jit(nopython=True)
def discard_connecting_eods(x_peak, x_trough, hights, widths, verbose=0):
    """
    If two detected EODs share the same closest trough, keep only the highest peak

    Parameters
    ----------
    x_peak: list of ints
        Indices of EOD peaks.
    x_trough: list of ints
        Indices of EOD troughs.
    hights: list of floats
        EOD hights.
    widths: list of ints
        EOD widths.
    verbose (optional): int
        Verbosity level.

    Returns
    -------
    x_peak, x_trough, hights, widths : lists of ints and floats
        EOD location and features of the non-discarded EODs
    """
    keep_idxs = np.ones(len(x_peak))

    for tr in np.unique(x_trough):
        if len(x_trough[x_trough==tr]) > 1:
            slopes = hights[x_trough==tr]/widths[x_trough==tr]

            if (np.max(slopes)!=np.min(slopes)) and (np.abs(np.max(slopes)-np.min(slopes))/(0.5*np.max(slopes)+0.5*np.min(slopes)) > 0.25):
                keep_idxs[np.where(x_trough==tr)[0][np.argmin(hights[x_trough==tr]/widths[x_trough==tr])]] = 0
            else:
                keep_idxs[np.where(x_trough==tr)[0][np.argmin(hights[x_trough==tr])]] = 0

    #if verbose>0:
        #print('Number of peaks after discarding connecting peaks:      %5i'%(len(keep_idxs)))
            
    return x_peak[np.where(keep_idxs==1)[0]], x_trough[np.where(keep_idxs==1)[0]], hights[np.where(keep_idxs==1)[0]], widths[np.where(keep_idxs==1)[0]]
