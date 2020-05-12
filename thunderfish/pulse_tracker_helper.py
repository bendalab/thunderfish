import numpy as np

def makeeventlist(main_event_positions,side_event_positions,data,event_width=20, min_width=2):
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
    main_w = np.zeros(len(main_event_positions))
    main_xt = np.zeros(len(main_event_positions))

    main_real = np.ones(len(main_event_positions))
    # iteration over the properties of the single main
    for ind,(x, y, h, r, w, xt) in enumerate(np.nditer([main_x, main_y, main_h, main_real, main_w, main_xt], op_flags=[["readonly"],['readonly'],['readwrite'],['readwrite'],['readwrite'],['readwrite']])):
        
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
            if min((l_distance),(r_distance)) > event_width or min((l_distance),(r_distance)) <= min_width:
                    r[...] = False
            elif max((l_distance),(r_distance)) <= event_width:
                    h[...] = max(abs(y-l_side_y),abs(y-r_side_y))  #calculated using absolutes in case of for example troughs instead of peaks as main events 
                    w[...] = [l_distance,r_distance][np.argmax([abs(y-l_side_y),abs(y-r_side_y)])]
                    xt[...] = x + [-l_distance,r_distance][np.argmax([abs(y-l_side_y),abs(y-r_side_y)])]
            else:
                    if (l_distance)<(r_distance): # evaluated only when exactly one side event is out of reach of the event width. Then the closer event will be the correct event
                        h[...] = abs(y-l_side_y)
                        w[...] = l_distance
                        xt[...] = x + -l_distance
                    else:
                        h[...] = abs(y-r_side_y)
                        w[...] = r_distance
                        xt[...] = x + r_distance
        # check corner cases
        elif l_side_ind == -1:
            if r_distance > event_width:
                r[...] = False
            else:
                h[...] = y-r_side_y
                w[...] = r_distance
                xt[...] = x + r_distance
        elif r_side_ind == len(side_event_positions):
            if l_distance> event_width:
                r[...] = False
            else:
                h[...] = y-l_side_y
                w[...] = l_distance
                xt[...] = x - l_distance
    # generate return array and discard all events that are not marked as real
    EOD_events = np.array([main_x, main_y, main_h, main_w, main_xt], dtype = np.float)[:,main_real==1]
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
    while unchanged == False:
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
    return event_indices.astype('int'), event_locations, event_heights

def discard_connecting_eods(pks, trs, hs, ws):
    # pk should be unique but tr is maybe not
    # for each tr, select peaks that result in biggest hight.
    keep_idxs = np.ones(len(pks))
    for tr in np.unique(trs):
        if len(trs[trs==tr])>1:
            #keep only highest one.
            keep_idxs[np.where(trs==tr)[0][np.argmin(hs[trs==tr])]] = 0
    return pks[keep_idxs.astype('bool')].astype('int'), trs[keep_idxs.astype('bool')].astype('int'), hs[keep_idxs.astype('bool')], ws[keep_idxs.astype('bool')]
