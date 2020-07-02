import numpy as np
try:
    from numba import jit
except ImportError:
    def jit(nopython):
        def decorator_jit(func):
            return func
        return decorator_jit
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

    r = ((w>min_width) & (w<event_width))

    return xp[r], xt[r], h[r], w[r]

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
