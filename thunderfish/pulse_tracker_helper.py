import numpy as np
try:
    from numba import jit
except ImportError:
    def jit(nopython):
        def decorator_jit(func):
            return func
        return decorator_jit
import matplotlib.pyplot as plt


def makeeventlist(main_event_positions, side_event_positions, data, max_width=20, min_width=2,verbose=0):
    """
    Generate array of events that might be EODs of a pulse-type fish, using the location of peaks and troughs,
    the data and the minimum and maximum width of an supposed EOD-event.
    The generated event-arrays include the location, heights and widths of such events.

    Parameters
    ----------
    main_event_positions: array of int or float
        Positions of the detected peaks in the data time series.
    side_event_positions: array of int or float
        Positions of the detected troughs in the data time series. 
        The complimentary event to the main events.
    data: array of float
        The given data.
    max_width (optional): int
        Maximum EOD width (in samples).
    min_width (optional): int
        Minimum EOD width (in samples).
    verbose (optional): int
        Verbosity level.

    Returns
    -------
    x_peak: numpy array
        Peak indices.
    x_trough: numpy array
        Trough indices.
    hights: numpy array
        Peak hights (distance between peak and trough amplitude)
    widths: numpy array
        Peak widths (distance between peak and trough indices)
    """

    # determine if there is a peak or through first. Evaluate to 1 if there is a peak first.
    mainfirst = int((min(main_event_positions[0],side_event_positions[0])<side_event_positions[0]))
    # determine if there is a peak or through last. Evaluate to 1 if there is a peak last.
    mainlast = int((max(main_event_positions[-1],side_event_positions[-1])>side_event_positions[-1]))

    x_peak = main_event_positions[mainfirst:len(main_event_positions)-mainlast]
    ind = np.arange(len(x_peak))
    y = data[x_peak]
    
    # find indices of troughs on the right and left side of peaks
    l_side_ind = ind
    r_side_ind = l_side_ind + 1

    # determine x values, distance to peak and amplitude of right troughs
    r_side_x = side_event_positions[r_side_ind]
    r_distance = np.abs(r_side_x - x_peak)
    r_side_y = data[r_side_x]

    # determine x values, distance to peak and amplitude of left troughs
    l_side_x = side_event_positions[l_side_ind]
    l_distance = np.abs(x_peak - l_side_x)
    l_side_y = data[l_side_x]

    # determine slope of lines connecting the peaks to the nearest troughs on the right and left.
    l_slope = np.abs((y-l_side_y)/l_distance)
    r_slope = np.abs((y-r_side_y)/r_distance)

    # determine which trough to assign to the peak by taking either the steepest slope,
    # or, when slopes are similar on both sides (within 25% difference), take the trough 
    # with the maximum hight difference to the peak.
    trough_idxs = np.argmax(np.vstack((np.abs(y-l_side_y),np.abs(y-r_side_y))),axis=0)
    slope_idxs = (np.abs(l_slope-r_slope)/(0.5*l_slope+0.5*r_slope) > 0.25)
    trough_idxs[slope_idxs] = np.argmax(np.array(np.vstack(np.array([l_slope[slope_idxs],r_slope[slope_idxs]]))),axis=0)

    #calculated using absolutes in case of for example troughs instead of peaks as main events 
    right_or_left = np.vstack([np.abs(trough_idxs-1),trough_idxs])
    hights = np.sum(np.vstack([np.abs(y-l_side_y),np.abs(y-r_side_y)])*right_or_left,axis=0)
    widths = np.sum(np.vstack([l_distance,r_distance])*right_or_left,axis=0)
    x_trough = np.sum((x_peak + np.vstack([-l_distance,r_distance]))*right_or_left,axis=0)

    keep_events = ((widths>min_width) & (widths<max_width))

    return x_peak[keep_events], x_trough[keep_events], hights[keep_events], widths[keep_events]

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
