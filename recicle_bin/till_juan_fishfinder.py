#!/usr/bin/python

# same as fishsorter

__author__ = 'raab'
def description_of_the_whole_code():
    """
    This code works different depending on how many arguments are give.

    If there are only 2 arguments:
        You can have a look on your data. You will get for each wave- and pulsefishes a spectogram, a plot that shows
        the fishes and their frequencies and a plot that shows the spectogram where the dots witch belong to one fish
        are connected.

        1st arg: code itself.
        2nd arg: Sound data that should be progressed.

    If there are 4 arguments
        Very useful when you want to combine the data of multiple sound files. The are .npy array witch contains the
        mean frequencies of each wave- and pulsefish.

        1st arg: code itself.
        2nd arg: Sound data(s) that should be progressed.
        3rd arg: str to name the .npy array for the wavefishes ([sys.argv[2]].npy) ( DO NOT give '.npy' as argument ).
        4th arg: str to name the .npy array for the pulsefishes ([sys.argv[3]].npy) ( DO NOT give '.npy' as argument ).
    """

import sys
import os
import argparse
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as ml
import matplotlib.colors as mc
from collections import OrderedDict
import pyaudio
import pandas as pd
from IPython import embed
from matplotlib.backends.backend_pdf import PdfPages
import subprocess

# check: import logging https://docs.python.org/2/howto/logging.html#logging-basic-tutorial

cfg = OrderedDict()
cfgsec = dict()

cfgsec['minPSDAverages'] = 'Power spectrum estimation:'
cfg['minPSDAverages'] = [ 3, '', 'Minimum number of fft averages for estimating the power spectrum.' ]
cfg['initialFrequencyResolution'] = [ 1.0, 'Hz', 'Initial frequency resolution of the power spectrum.' ]

cfgsec['lowThreshold'] = 'Thresholds for peak detection in power spectra:'
cfg['lowThreshold'] = [ 0.0, 'dB', 'Threshold for all peaks.\n If 0.0 estimate threshold from histogram.' ]
cfg['highThreshold'] = [ 0.0, 'dB', 'Threshold for good peaks. If 0.0 estimate threshold from histogram.' ]
#cfg['lowThreshold'][0] = 12. # panama
#cfg['highThreshold'][0] = 18. # panama

cfgsec['noiseFactor'] = 'Threshold estimation:\nIf no thresholds are specified they are estimated from the histogram of the decibel power spectrum.'
cfg['noiseFactor'] = [ 6.0, '', 'Factor for multiplying std of noise floor for lower threshold.' ]
cfg['peakFactor'] = [ 0.5, '', 'Fractional position of upper threshold above lower threshold.' ]

cfgsec['maxPeakWidthFac'] = 'Peak detection in decibel power spectrum:'
cfg['maxPeakWidthFac'] = [ 3.5, '', 'Maximum width of peaks at 0.75 hight in multiples of frequency resolution (might be increased)' ]
cfg['minPeakWidth'] = [ 1.0, 'Hz', 'Peaks do not need to be narrower than this.' ]

cfgsec['mainsFreq'] = 'Harmonic groups:'
cfg['mainsFreq'] = [ 60.0, 'Hz', 'Mains frequency to be excluded.' ]
cfg['maxDivisor'] = [ 4, '', 'Maximum ratio between the frequency of the largest peak and its fundamental' ]
cfg['freqTolerance'] = [ 0.7, '', 'Harmonics need be within this factor times the frequency resolution of the power spectrum. Needs to be higher than 0.5!' ]
cfg['maxUpperFill'] = [ 1, '', 'As soon as more than this number of harmonics need to be filled in conescutively stop searching for higher harmonics.' ]
cfg['maxFillRatio'] = [ 0.25, '', 'Maximum fraction of filled in harmonics allowed (usefull values are smaller than 0.5)' ]
cfg['maxDoubleUseHarmonics'] = [ 8, '', 'Maximum harmonics up to which double uses are penalized.' ]
cfg['maxDoubleUseCount'] = [ 1, '', 'Maximum overall double use count allowed.' ]
cfg['powerNHarmonics'] = [ 10, '', 'Compute total power over the first # harmonics.' ]

cfgsec['minimumGroupSize'] = 'Acceptance of best harmonic groups:'
cfg['minimumGroupSize'] = [ 3, '', 'Minimum required number of harmonics (inclusively fundamental) that are not filled in and are not used by other groups.' ]
#cfg['minimumGroupSize'][0] = 2 # panama
cfg['minimumFrequency'] = [ 20.0, 'Hz', 'Minimum frequency allowed for the fundamental.' ]
cfg['maximumFrequency'] = [ 2000.0, 'Hz', 'Maximum frequency allowed for the fundamental.' ]
cfg['maximumWorkingFrequency'] = [ 4000.0, 'Hz', 'Maximum frequency to be used to search for harmonic groups and to adjust fundamental frequency.' ]

cfg['maxHarmonics'] = [ 0, '', '0: keep all, >0 only keep the first # harmonics.' ]

cfgsec['displayHelp'] = 'Items to display:'
cfg['displayHelp'] = [ False, '', 'Display help on key bindings' ]
cfg['labelFrequency'] = [ True, '', 'Display the frequency of the peak' ]
cfg['labelHarmonic'] = [ True, '', 'Display the harmonic of the peak' ]
cfg['labelPower'] = [ True, '', 'Display the power of the peak' ]
cfg['labelWidth'] = [ True, '', 'Display the width of the peak' ]
cfg['labelDoubleUse'] = [ True, '', 'Display double-use count of the peak' ]

cfgsec['verboseLevel'] = 'Debugging:'
cfg['verboseLevel'] = [ 0, '', '0=off upto 4 very detailed' ]


###############################################################################
## load data:

def load_pickle( filename, trace=0 ) :
    """
    load Joerg's pickle files
    """
    import pickle
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    time = data['time_trace']
    freq = 1000.0/(time[1]-time[0])
    tracen = data['raw_data'].shape[1]
    if trace >= tracen :
        print 'number of traces in file is', tracen
        quit()
    return freq, data['raw_data'][:,trace], 'mV'


def load_wavfile( filename, trace=0 ) :
    """
    load wav file using scipy io.wavfile
    """
    from scipy.io import wavfile
    freq, data = wavfile.read( filename )
    if len( data.shape ) == 1 :
        if trace >= 1 :
            print 'number of traces in file is', 1
            quit()
        return freq, data/2.0**15, ''
    else :
        tracen = data.shape[1]
        if trace >= tracen :
            print 'number of traces in file is', tracen
            quit()
        return freq, data[:,trace]/2.0**15, 'a.u.'


def load_wave( filename, trace=0 ) :
    """
    load wav file using wave module
    """
    try:
        import wave
    except ImportError:
        print 'python module "wave" is not installed.'
        return load_wavfile( filename, trace )

    wf = wave.open( filename, 'r' )
    (nchannels, sampwidth, freq, nframes, comptype, compname) = wf.getparams()
    print nchannels, sampwidth, freq, nframes, comptype, compname
    buffer = wf.readframes( nframes )
    format = 'i%d' % sampwidth
    data = np.fromstring( buffer, dtype=format ).reshape( -1, nchannels )  # read data
    wf.close()
    print data.shape
    if len( data.shape ) == 1 :
        if trace >= 1 :
            print 'number of traces in file is', 1
            quit()
        return freq, data/2.0**(sampwidth*8-1), ''
    else :
        tracen = data.shape[1]
        if trace >= tracen :
            print 'number of traces in file is', tracen
            quit()
        return freq, data[:,trace]/2.0**(sampwidth*8-1), 'a.u.'


def load_audio( filename, trace=0 ) :
    """
    load wav file using audioread.
    This is not available in python x,y.
    """
    try:
        import audioread
    except ImportError:
        print 'python module "audioread" is not installed.'
        return load_wave( filename, trace )

    data = np.array( [] )
    with audioread.audio_open( filename ) as af :
        tracen = af.channels
        if trace >= tracen :
            print 'number of traces in file is', tracen
            quit()
        data = np.zeros( np.ceil( af.samplerate*af.duration ), dtype="<i2" )
        index = 0
        for buffer in af:
            fulldata = np.fromstring( buffer, dtype='<i2' ).reshape( -1, af.channels )
            n = fulldata.shape[0]
            if index+n > len( data ) :
                n = len( data ) - index
            if n > 0 :
                data[index:index+n] = fulldata[:n,trace]
                index += n
            else :
                break
    return af.samplerate, data/2.0**15, 'a.u.'


###############################################################################
## configuration file writing and loading:

def dump_config( filename, cfg, sections=None, header=None, maxline=60 ) :
    """
    Pretty print non-nested dicionary cfg into file.

    The keys of the dictionary are strings.

    The values of the dictionary can be single variables or lists:
    [value, unit, comment]
    Both unit and comment are optional.

    value can be any type of variable.

    unit is a string (that can be empty).

    Comments comment are printed out right before the key-value pair.
    Comments are single strings. Newline characters are intepreted as new paragraphs.
    Lines are folded if the character count exceeds maxline.

    Section comments can be added by the sections dictionary.
    It contains comment strings as values that are inserted right
    before the key-value pair with the same key. Section comments
    are formatted in the same way as comments for key-value pairs,
    but get two comment characters prependend ('##').

    A header can be printed initially. This is a simple string that is formatted
    like the section comments.

    Args:
        filename: The name of the file for writing the configuration.
        cfg (dict): Configuration keys, values, units, and comments.
        sections (dict): Comments describing secions of the configuration file.
        header (string): A string that is written as an introductory comment into the file.
        maxline (int): Maximum number of characters that fit into a line.
    """

    def write_comment( f, comment, maxline=60, cs='#' ) :
        # format comment:
        if len( comment ) > 0 :
            for line in comment.split( '\n' ) :
                f.write( cs + ' ' )
                cc = len( cs ) + 1  # character count
                for w in line.strip().split( ' ' ) :
                    # line too long?
                    if cc + len( w ) > maxline :
                        f.write( '\n' + cs + ' ' )
                        cc = len( cs ) + 1
                    f.write( w + ' ' )
                    cc += len( w ) + 1
                f.write( '\n' )

    with open( filename, 'w' ) as f :
        if header != None :
            write_comment( f, header, maxline, '##' )
        maxkey = 0
        for key in cfg.keys() :
            if maxkey < len( key ) :
                maxkey = len( key )
        for key, v in cfg.items() :
            # possible section entry:
            if sections != None and key in sections :
                f.write( '\n\n' )
                write_comment( f, sections[key], maxline, '##' )

            # get value, unit, and comment from v:
            val = None
            unit = ''
            comment = ''
            if hasattr(v, '__len__') and (not isinstance(v, str)) :
                val = v[0]
                if len( v ) > 1 :
                    unit = ' ' + v[1]
                if len( v ) > 2 :
                    comment = v[2]
            else :
                val = v

            # next key-value pair:
            f.write( '\n' )
            write_comment( f, comment, maxline, '#' )
            f.write( '{key:<{width}s}: {val}{unit:s}\n'.format( key=key, width=maxkey, val=val, unit=unit ) )


def load_config( filename, cfg ) :
    """
    Set values of dictionary cfg to values from key-value pairs read in from file.

    Args:
        filename: The name of the file from which to read in the configuration.
        cfg (dict): Configuration keys, values, units, and comments.
    """
    with open( filename, 'r' ) as f :
        for line in f :
            # do not process empty lines and comments:
            if len( line.strip() ) == 0 or line[0] == '#' or not ':' in line :
                continue
            key, val = line.split(':', 1)
            key = key.strip()
            if not key in cfg :
                continue
            cv = cfg[key]
            vals = val.strip().split( ' ' )
            if hasattr(cv, '__len__') and (not isinstance(cv, str)) :
                unit = ''
                if len( vals ) > 1 :
                    unit = vals[1]
                if unit != cv[1] :
                    print 'unit for', key, 'is', unit, 'but should be', cv[1]
                if type(cv[0]) == bool :
                    cv[0] = ( vals[0].lower() == 'true' or vals[0].lower() == 'yes' )
                else :
                    cv[0] = type(cv[0])(vals[0])
            else :
                if type(cv[0]) == bool :
                    cfg[key] = ( vals[0].lower() == 'true' or vals[0].lower() == 'yes' )
                else :
                    cfg[key] = type(cv)(vals[0])


###############################################################################
## peak detection:

def detect_peaks( time, data, threshold, check_func=None, check_conditions=None ):

    if not check_conditions:
        check_conditions = dict()

    event_list = list()

    # initialize:
    dir = 0
    min_inx = 0
    max_inx = 0
    min_value = data[0]
    max_value = min_value
    trough_inx = 0

    # loop through the new read data
    for index, value in enumerate(data):

        # rising?
        if dir > 0:
            # if the new value is bigger than the old maximum: set it as new maximum
            if max_value < value:
                max_inx = index  # maximum element
                max_value = value

            # otherwise, if the maximum value is bigger than the new value plus the threshold:
            # this is a local maximum!
            elif max_value >= value + threshold:
                # there was a peak:
                event_inx = max_inx

                # check and update event with this magic function
                if check_func:
                    r = check_func( time, data, event_inx, index, trough_inx, min_inx, threshold, check_conditions )
                    if len( r ) > 0 :
                        # this really is an event:
                        event_list.append( r )
                else:
                    # this really is an event:
                    event_list.append( time[event_inx] )

                # change direction:
                min_inx = index  # minimum element
                min_value = value
                dir = -1

        # falling?
        elif dir < 0:
            if value < min_value:
                min_inx = index  # minimum element
                min_value = value
                trough_inx = index

            elif value >= min_value + threshold:
                # there was a trough:
                # change direction:
                max_inx = index  # maximum element
                max_value = value
                dir = 1

        # don't know!
        else:
            if max_value >= value + threshold:
                dir = -1  # falling
            elif value >= min_value + threshold:
                dir = 1  # rising

            if max_value < value:
                max_inx = index  # maximum element
                max_value = value

            elif value < min_value:
                min_inx = index  # minimum element
                min_value = value
                trough_inx = index

    return np.array( event_list )


###############################################################################
## harmonic group extraction:

def build_harmonic_groups( freqs, more_freqs, deltaf, cfg ):

    verbose = cfg['verboseLevel'][0]

    # start at the strongest frequency:
    fmaxinx = np.argmax(freqs[:, 1])
    fmax = freqs[fmaxinx, 0]
    if verbose > 1:
        print
        print 70*'#'
        print 'freqs:     ', '[', ', '.join( [ '{:.2f}'.format( f ) for f in freqs[:,0] ] ), ']'
        print 'more_freqs:', '[', ', '.join( [ '{:.2f}'.format( f ) for f in more_freqs[:,0] if f < cfg['maximumFrequency'][0] ] ), ']'
        print '## fmax is: {0: .2f}Hz: {1:.5g} ##\n'.format(fmax, np.max(freqs[:, 1]))

    # container for harmonic groups
    best_group = list()
    best_moregroup = list()
    best_group_peaksum = 0.0
    best_group_fill_ins = 1000000000
    best_divisor = 0
    best_fzero = 0.0
    best_fzero_harmonics = 0

    freqtol = cfg['freqTolerance'][0] * deltaf

    # ###########################################
    # SEARCH FOR THE REST OF THE FREQUENCY GROUP
    # start with the strongest fundamental and try to gather the full group of available harmonics
    # In order to find the fundamental frequency of a fish harmonic group,
    # we divide fmax (the strongest frequency in the spectrum)
    # by a range of integer divisors.
    # We do this, because fmax could just be a strong harmonic of the harmonic group

    for divisor in xrange(1, cfg['maxDivisor'][0]+1):

        # define the hypothesized fundamental, which is compared to all higher frequencies:
        fzero = fmax / divisor
        # fzero is not allowed to be smaller than our chosen minimum frequency:
        #if divisor > 1 and fzero < cfg['minimumFrequency'][0]:   # XXX why not also for divisor=1???
        #    break
        fzero_harmonics = 1

        if verbose > 1:
            print '# divisor:', divisor, 'fzero=', fzero

        # ###########################################
        # SEARCH ALL DETECTED FREQUENCIES in freqs
        # this in the end only recomputes fzero!
        newgroup = list()
        npre = -1     # previous harmonics
        ndpre = 0.0   # difference of previous frequency
        connected = True
        for j in xrange(freqs.shape[0]):

            if verbose > 2 :
                print 'check freq {:3d} {:8.2f}'.format( j, freqs[j, 0] ),

            # IS THE CURRENT FREQUENCY AN INTEGRAL MULTIPLE OF FZERO?
            # divide the frequency-to-be-checked by fzero
            # to get the multiplication factor between freq and fzero
            n = np.round( freqs[j, 0] / fzero )
            if n == 0:
                if verbose > 2 :
                    print 'discarded: n == 0'
                continue

            # !! the difference between the current frequency, divided by the derived integer,
            # and fzero should be very very small: 1 resolution step of the fft
            # (freqs[j,0] / n) = should be fzero, plus minus a little tolerance,
            # which is the fft resolution
            nd = np.abs((freqs[j, 0] / n) - fzero)
            # ... compare it to our tolerance
            if nd > freqtol :
                if verbose > 2 :
                    print 'discarded: not a harmonic n=%2d d=%5.2fHz tol=%5.2fHz' % ( n, nd, freqtol )
                continue

            # two succeeding frequencies should also differ by
            # fzero plus/minus twice! the tolerance:
            if len( newgroup ) > 0 :
                nn = np.round( ( freqs[j,0] - freqs[newgroup[-1],0] )/fzero )
                if nn == 0:
                    # the current frequency is the same harmonic as the previous one
                    #print divisor, j, freqs[j,0], freqs[newgroup[-1],0]
                    if len( newgroup ) > 1 :
                        # check whether the current frequency is fzero apart from the previous harmonics:
                        nn = np.round( ( freqs[j,0] - freqs[newgroup[-2],0] )/fzero )
                        nnd = np.abs(((freqs[j,0] - freqs[newgroup[-2],0]) / nn) - fzero)
                        if nnd > 2.0*freqtol :
                            if verbose > 2 :
                                print 'discarded: distance to previous harmonics %2d %5.2f %5.2f %8.2f' % ( nn, nnd, freqtol, fzero )
                            continue
                    if ndpre < nd :
                        # the previous frequency is closer to the harmonics, keep it:
                        if verbose > 2 :
                            print 'discarded: previous harmonics is closer %2d %5.2f %5.2f %5.2f %8.2f' % ( n, nd, ndpre, freqtol, fzero )
                        continue
                    else :
                        # the current frequency is closer to the harmonics, remove the previous one:
                        newgroup.pop()
                else :
                    # check whether the current frequency is fzero apart from the previous harmonics:
                    nnd = np.abs(((freqs[j,0] - freqs[newgroup[-1],0]) / nn) - fzero)
                    if nnd > 2.0*freqtol :
                        if verbose > 2 :
                            print 'discarded: distance to previous harmonics %2d %5.2f %5.2f %8.2f' % ( nn, nnd, freqtol, fzero )
                        continue

            # take frequency:
            newgroup.append(j)  # append index of frequency
            if verbose > 2 :
                print 'append n={:.2f} d={:5.2f}Hz tol={:5.2f}Hz'.format( freqs[j, 0] / fzero, nd, freqtol )

            if npre >= 0 and n-npre > 1 :
                connected = False
            npre = n
            ndpre = nd

            if connected :
                # adjust fzero as we get more information from the higher frequencies:
                fzero = freqs[j,0]/n
                fzero_harmonics = int(n)
                if verbose > 2 :
                    print 'adjusted fzero to', fzero

        if verbose > 3:
            print 'newgroup:', divisor, fzero, newgroup

        newmoregroup = list()
        fill_ins = 0
        double_use = 0
        ndpre = 0.0   # difference of previous frequency

        # ###########################################
        # SEARCH ALL DETECTED FREQUENCIES in morefreqs
        for j in xrange(more_freqs.shape[0]):

            if verbose > 3 :
                print 'check more_freq %3d %8.2f' % ( j, more_freqs[j, 0] ),

            # IS FREQUENCY A AN INTEGRAL MULTIPLE OF FREQUENCY B?
            # divide the frequency-to-be-checked with fzero:
            # what is the multiplication factor between freq and fzero?
            n = np.round( more_freqs[j, 0] / fzero )
            if n == 0:
                if verbose > 3 :
                    print 'discarded: n == 0'
                continue

            # !! the difference between the detection, divided by the derived integer
            # , and fzero should be very very small: 1 resolution step of the fft
            # (more_freqs[j,0] / n) = should be fzero, plus minus a little tolerance,
            # which is the fft resolution
            nd = np.abs((more_freqs[j, 0] / n) - fzero)

            # ... compare it to our tolerance
            if nd > freqtol :
                if verbose > 3 :
                    print 'discarded: not a harmonic n=%2d d=%5.2fHz tol=%5.2fHz' % ( n, nd, freqtol )
                continue

            # two succeeding frequencies should also differ by fzero plus/minus tolerance:
            if len( newmoregroup ) > 0 :
                nn = np.round( ( more_freqs[j,0] - more_freqs[newmoregroup[-1],0] )/fzero )
                if nn == 0:
                    # the current frequency is close to the same harmonic as the previous one
                    #print n, newmoregroup[-1], ( more_freqs[j,0] - more_freqs[newmoregroup[-1],0] )/fzero
                    #print divisor, j, n, more_freqs[j,0], more_freqs[newmoregroup[-1],0], more_freqs[newmoregroup[-2],0], newmoregroup[-2]
                    if len( newmoregroup ) > 1 and newmoregroup[-2] >= 0 :
                        # check whether the current frequency is fzero apart from the previous harmonics:
                        nn = np.round( ( more_freqs[j,0] - more_freqs[newmoregroup[-2],0] )/fzero )
                        nnd = np.abs(((more_freqs[j,0] - more_freqs[newmoregroup[-2],0]) / nn) - fzero)
                        if nnd > 2.0*freqtol :
                            if verbose > 3 :
                                print 'discarded: distance to previous harmonics %2d %5.2f %5.2f %8.2f' % ( nn, nnd, freqtol, fzero )
                            continue
                    if ndpre < nd :
                        # the previous frequency is closer to the harmonics, keep it:
                        if verbose > 3 :
                            print 'discarded: previous harmonics is closer %2d %5.2f %5.2f %5.2f %8.2f' % ( n, nd, ndpre, freqtol, fzero )
                        continue
                    else :
                        # the current frequency is closer to the harmonics, remove the previous one:
                        newmoregroup.pop()
                else :
                    # check whether the current frequency is fzero apart from the previous harmonics:
                    nnd = np.abs(((more_freqs[j,0] - more_freqs[newmoregroup[-1],0]) / nn) - fzero)
                    if nnd > 2.0*freqtol :
                        if verbose > 3 :
                            print 'discarded: distance to previous harmonics %2d %5.2f %5.2f %8.2f' % ( nn, nnd, freqtol, fzero )
                        continue
            ndpre = nd

            # too many fill-ins upstream of fmax ?
            if more_freqs[j, 0] > fmax and n-1-len(newmoregroup) > cfg['maxUpperFill'][0] :
                # finish this group immediately
                if verbose > 3 :
                    print 'stopping group: too many upper fill-ins:', n-1-len(newmoregroup), '>', cfg['maxUpperFill'][0]
                break

            # fill in missing harmonics:
            while len(newmoregroup) < n-1:  # while some harmonics are missing ...
                newmoregroup.append(-1)  # ... add marker for non-existent harmonic
                fill_ins += 1

            # count double usage of frequency:
            if n <= cfg['maxDoubleUseHarmonics'][0] :
                double_use += more_freqs[j, 4]
                if verbose > 3 and more_freqs[j, 4] > 0 :
                    print 'double use of %.2fHz' % more_freqs[j, 0],

            # take frequency:
            newmoregroup.append(j)
            if verbose > 3 :
                print 'append'

        # double use of points:
        if double_use > cfg['maxDoubleUseCount'][0] :
            if verbose > 1:
                print 'discarded group because of double use:', double_use
            continue

        # ratio of total fill-ins too large:
        if float(fill_ins)/float(len(newmoregroup)) > cfg['maxFillRatio'][0] :
            if verbose > 1:
                print 'dicarded group because of too many fill ins! %d from %d (%g)' % ( fill_ins, len(newmoregroup), float(fill_ins)/float(len(newmoregroup)) ), newmoregroup
            continue

        # REASSEMBLE NEW GROUP BECAUSE FZERO MIGHT HAVE CHANGED AND
        # CALCULATE THE PEAKSUM, GIVEN THE UPPER LIMIT
        # DERIVED FROM morefreqs which can be low because of too many fill ins.
        # newgroup is needed to delete the right frequencies from freqs later on.
        newgroup = []
        fk = 0
        for j in xrange(len(newmoregroup)):
            if newmoregroup[j] >= 0 :
                # existing frequency peak:
                f = more_freqs[newmoregroup[j],0]
                # find this frequency in freqs:
                for k in xrange( fk, freqs.shape[0] ) :
                    if np.abs( freqs[k,0] - f ) < 1.0e-8 :
                        newgroup.append( k )
                        fk = k+1
                        break
                if fk >= freqs.shape[0] :
                    break

        # fmax might not be in our group, because we adjust fzero:
        if not fmaxinx in newgroup :
            if verbose > 1:
                print "discarded: lost fmax"
            continue

        n = cfg['powerNHarmonics'][0]
        newmoregroup_peaksum = np.sum(more_freqs[newmoregroup[:n], 1])
        fills = np.sum( np.asarray( newmoregroup[:len(best_moregroup)] ) < 0 )
        best_fills = np.sum( np.asarray( best_moregroup[:len(newmoregroup)] ) < 0 )
        takes = np.sum( np.asarray( newmoregroup ) >= 0 )
        best_takes = np.sum( np.asarray( best_moregroup ) >= 0 )

        if verbose > 1:
            print 'newgroup:      divisor={d}, fzero={fz:.2f}Hz, peaksum={ps}, fills={f}, takes={t}'.format( d=divisor, fz=fzero, ps=newmoregroup_peaksum, f=fills, t=takes ), newgroup
            print 'newmoregroup:  divisor={d}, fzero={fz:.2f}Hz, peaksum={ps}, fills={f}, takes={t}'.format( d=divisor, fz=fzero, ps=newmoregroup_peaksum, f=fills, t=takes ), newmoregroup
            if verbose > 2:
                print 'bestgroup:     divisor={d}, fzero={fz:.2f}Hz, peaksum={ps}, fills={f}, takes={t}'.format( d=best_divisor, fz=best_fzero, ps=best_group_peaksum, f=best_fills, t=best_takes ), best_group

        # TAKE THE NEW GROUP IF BETTER:
        # sum of peak power must be larger and
        # less fills. But if the new group has more takes,
        # this might compensate for more fills.
        if newmoregroup_peaksum > best_group_peaksum \
           and fills-best_fills <= 0.5*(takes-best_takes)  :

            best_group_peaksum = newmoregroup_peaksum
            if len( newgroup ) == 1 :
                best_group_fill_ins = np.max( ( 2, fill_ins ) )   # give larger groups a chance XXX we might reduce this!
            else :
                best_group_fill_ins = fill_ins
            best_group = newgroup
            best_moregroup = newmoregroup
            best_divisor = divisor
            best_fzero = fzero
            best_fzero_harmonics = fzero_harmonics

            if verbose > 2:
                print 'new bestgroup:     divisor={d}, fzero={fz:.2f}Hz, peaksum={ps}, fills={f}, takes={t}'.format( d=best_divisor, fz=best_fzero, ps=best_group_peaksum, f=best_fills, t=best_takes ), best_group
                print 'new bestmoregroup: divisor={d}, fzero={fz:.2f}Hz, peaksum={ps}, fills={f}, takes={t}'.format( d=best_divisor, fz=best_fzero, ps=best_group_peaksum, f=best_fills, t=best_takes ), best_moregroup
            elif verbose > 1:
                print 'took as new best group'

    # ##############################################################

    # no group found:
    if len( best_group ) == 0 :
        # erase fmax:
        freqs = np.delete( freqs, fmaxinx, axis=0 )
        group = np.zeros((0, 5))
        return freqs, more_freqs, group, 1, fmax

    # group found:
    if verbose > 2:
        print
        print '## best groups found for fmax={fm:.2f}Hz: fzero={fz:.2f}Hz, d={d:d}:'.format(fm=fmax, fz=best_fzero, d=best_divisor)
        print '## bestgroup:     ', best_group, '[', ', '.join( [ '{:.2f}'.format( f ) for f in freqs[best_group,0] ] ), ']'
        print '## bestmoregroup: ', best_moregroup, '[', ', '.join( [ '{:.2f}'.format( f ) for f in more_freqs[best_moregroup,0] ] ), ']'

    # fill up group:
    group = np.zeros((len(best_moregroup), 5))
    for i, inx in enumerate(best_moregroup):
        # increment double use counter:
        more_freqs[inx, 4] += 1.0
        if inx >= 0:
            group[i, :] = more_freqs[inx, :]
        # take adjusted peak frequencies:
        group[i, 0] = (i+1)*best_fzero

    if verbose > 1:
        refi = np.nonzero( group[:, 1] > 0.0 )[0][0]
        print
        print '# resulting harmonic group for fmax=', fmax
        for i in xrange(group.shape[0]):
            print '{0:8.2f}Hz n={1:5.2f}: p={2:10.3g} p/p0={3:10.3g}'.format( group[i, 0], group[i, 0]/group[0, 0], group[i, 1], group[i, 1]/group[refi, 1])

    # erase from freqs:
    for inx in reversed(best_group):
        freqs = np.delete(freqs, inx, axis=0)

    # freqs: removed all frequencies of bestgroup
    # more_freqs: updated double use count
    # group: the group
    # fmax: fmax
    return freqs, more_freqs, group, best_fzero_harmonics, fmax


def extract_fundamentals( good_freqs, all_freqs, deltaf, cfg ):
    """
    Extract fundamental frequencies from power-spectrum peaks.

    Returns:
        group_list (list): list of all harmonic groups found
        fzero_harmonics_list (list): the harmonics from which the fundamental frequencies were computed
        mains_list (2-d array): list of mains peaks found
    """

    verbose = cfg['verboseLevel'][0]
    if verbose > 0 :
        print

    # set double use count to zero:
    all_freqs[:,4] = 0.0

    freqtol = cfg['freqTolerance'][0] * deltaf
    mainsfreq = cfg['mainsFreq'][0]

    # remove power line harmonics from good_freqs:
    # XXX might be improved!!!
    if mainsfreq > 0.0 :
        pfreqtol = 1.0  # 1 Hz tolerance
        for inx in reversed(xrange(len(good_freqs))):
            n = np.round( good_freqs[inx, 0] / mainsfreq )
            nd = np.abs(good_freqs[inx, 0] - n*mainsfreq)
            if nd <= pfreqtol :
                if verbose > 1 :
                    print 'remove power line frequency', inx, good_freqs[inx, 0], np.abs(good_freqs[inx, 0]-n*mainsfreq)
                good_freqs = np.delete(good_freqs, inx, axis=0)

    group_list = list()
    fzero_harmonics_list = list()
    # as long as there are frequencies left in good_freqs:
    while good_freqs.shape[0] > 0:
        # we check for harmonic groups:
        good_freqs, all_freqs, harm_group, fzero_harmonics, fmax = \
          build_harmonic_groups( good_freqs, all_freqs, deltaf, cfg )

        if verbose > 1 :
            print

        # nothing found:
        if harm_group.shape[0] == 0 :
            if verbose > 0 :
                print 'Nothing found for fmax=%.2fHz' % fmax
            continue

        # count number of harmonics which have been detected, are not fill-ins,
        # and are not doubly used:
        group_size = np.sum((harm_group[:, 1] > 0.0) & (harm_group[:, 4] < 2.0))
        group_size_ok = ( group_size >= cfg['minimumGroupSize'][0] )

        # check frequency range of fundamental:
        fundamental_ok = ( harm_group[0, 0] >= cfg['minimumFrequency'][0] and
                           harm_group[0, 0] <= cfg['maximumFrequency'][0] )

        # check power hum (does this really ever happen???):
        mains_ok = ( ( mainsfreq == 0.0 ) |
                          ( np.abs(harm_group[0, 0]-mainsfreq) > freqtol ) )

        # check:
        if group_size_ok and fundamental_ok and mains_ok :
            if verbose > 0 :
                print 'Accepting harmonic group: {:.2f}Hz p={:10.8f}'.format(
                    harm_group[0, 0], np.sum(harm_group[:, 1]))

            group_list.append(harm_group[:,0:2])
            fzero_harmonics_list.append(fzero_harmonics)
        else :
            if verbose > 0 :
                print 'Discarded harmonic group: {:.2f}Hz p={:10.8f} g={:d} f={:} m={:}'.format(
                    harm_group[0, 0], np.sum(harm_group[:, 1]),
                    group_size, fundamental_ok, mains_ok )

    # sort groups by fundamental frequency:
    ffreqs = [ f[0,0] for f in group_list ]
    finx = np.argsort( ffreqs )
    group_list = [ group_list[fi] for fi in finx ]
    fzero_harmonics_list = [ fzero_harmonics_list[fi] for fi in finx ]

    # do not save more than n harmonics:
    maxharmonics = cfg['maxHarmonics'][0]
    if maxharmonics > 0 :
        for group in group_list:
            if group.shape[0] > maxharmonics :
                if verbose > 0 :
                    print 'Discarding some tailing harmonics'
                group = group[:maxharmonics, :]

    if verbose > 0 :
        print
        if len(group_list) > 0 :
            print '## FUNDAMENTALS FOUND: ##'
            for i in xrange(len(group_list)):
                power = group_list[i][:, 1]
                print '{:8.2f}Hz: {:10.8f} {:3d} {:3d}'.format( group_list[i][0, 0], np.sum( power ),
                                                            np.sum( power <= 0.0 ), fzero_harmonics_list[i] )
        else :
            print '## NO FUNDAMENTALS FOUND ##'

    # assemble mains frequencies from all_freqs:
    mains_list = []
    if mainsfreq > 0.0 :
        pfreqtol = 1.0
        for inx in xrange(len(all_freqs)):
            n = np.round( all_freqs[inx, 0] / mainsfreq )
            nd = np.abs(all_freqs[inx, 0] - n*mainsfreq)
            if nd <= pfreqtol :
                mains_list.append( all_freqs[inx] )
    return group_list, fzero_harmonics_list, np.array( mains_list )


def threshold_estimate( data, noise_factor, peak_factor ) :
    """
    Estimate noise standard deviation from histogram
    for usefull peak-detection thresholds.

    The standard deviation of the noise floor without peaks is estimated from
    the histogram of the data at 1/sqrt(e) relative height.

    Args:
        data: the data from which to estimate the thresholds
        noise_factor (float): multiplies the estimate of the standard deviation
                              of the noise to result in the low_threshold
        peak_factor (float): the high_threshold is the low_threshold plus
                             this fraction times the distance between largest peaks
                             and low_threshold plus half the low_threshold

    Returns:
        low_threshold (float): the threshold just above the noise floor
        high_threshold (float): the threshold for clear peaks
        center: (float): estimate of the median of the data without peaks
    """

    # estimate noise standard deviation:
    # XXX what about the number of bins for small data sets?
    hist, bins = np.histogram( data, 100, density=True )
    inx = hist > np.max( hist ) / np.sqrt( np.e )
    lower = bins[inx][0]
    upper = bins[inx][-1] # needs to return the next bin
    center = 0.5*(lower+upper)
    noisestd = 0.5*(upper-lower)

    # low threshold:
    lowthreshold = noise_factor*noisestd

    # high threshold:
    lowerth = center+0.5*lowthreshold
    cumhist = np.cumsum(hist)/np.sum(hist)
    upperpthresh = 0.95
    if bins[-2] >= lowerth :
        pthresh = cumhist[bins[:-1]>=lowerth][0]
        upperpthresh = pthresh+0.95*(1.0-pthresh)
    upperbins = bins[cumhist>upperpthresh]
    if len( upperbins ) > 0 :
        upperth = upperbins[0]
    else :
        upperth = bins[-1]
    highthreshold = lowthreshold + peak_factor*noisestd
    if upperth > lowerth + 0.1*noisestd :
        highthreshold = lowerth + peak_factor*(upperth - lowerth) + 0.5*lowthreshold - center

    return lowthreshold, highthreshold, center


def accept_psd_peaks( freqs, data, peak_inx, index, trough_inx, min_inx, threshold, check_conditions ) :
    """
    Accept each detected peak and compute its size and width.

    Args:
        freqs (array): frequencies of the power spectrum
        data (array): the power spectrum
        peak_inx: index of the current peak
        index: current index (first minimum after peak at threshold below)
        trough_inx: index of the previous trough
        min_inx: index of previous minimum
        threshold: threshold value
        check_conditions: not used

    Returns:
        freq (float): frequency of the peak
        power (float): power of the peak (value of data at the peak)
        size (float): size of the peak (peak minus previous trough)
        width (float): width of the peak at 0.75*size
        count (float): zero
    """
    size = data[peak_inx] - data[trough_inx]
    wthresh = data[trough_inx] + 0.75*size
    width = 0.0
    for k in xrange( peak_inx, trough_inx, -1 ) :
        if data[k] < wthresh :
            width = freqs[peak_inx] - freqs[k]
            break
    for k in xrange( peak_inx, index ) :
        if data[k] < wthresh :
            width += freqs[k] - freqs[peak_inx]
            break
    return [ freqs[peak_inx], data[peak_inx], size, width, 0.0 ]


def harmonic_groups( psd_freqs, psd, cfg ) :
    """
    Detect peaks in power spectrum and extract fundamentals of harmonic groups.

    Args:
        psd_freqs (array): frequencies of the power spectrum
        psd (array): power spectrum
        cfg (dict): configuration parameter

    Returns:
        groups (list): all harmonic groups, sorted by fundamental frequency.
                       Each harmonic group contains a 2-d array with frequencies
                       and power of the fundamental and all harmonics.
                       If the power is zero, there was no corresponding peak
                       in the power spectrum.
        fzero_harmonics (list) : The harmonics from
                       which the fundamental frequencies were computed.
        mains (2-d array): frequencies and power of multiples of mains frequency.
        all_freqs (2-d array): peaks in the power spectrum
                  detected with low threshold
                  [frequency, power, size, width, double use count]
        good_freqs (array): frequencies of peaks detected with high threshold
        low_threshold (float): the relative threshold for detecting all peaks in the decibel spectrum
        high_threshold (float): the relative threshold for detecting good peaks in the decibel spectrum
        center (float): the baseline level of the power spectrum
    """

    verbose = cfg['verboseLevel'][0]

    if verbose > 0 :
        print
        print 70*'#'
        print '##### harmonic_groups', 48*'#'

    # decibel power spectrum:
    log_psd = 10.0*np.log10( psd )

    # thresholds:
    low_threshold = cfg['lowThreshold'][0]
    high_threshold = cfg['highThreshold'][0]
    center = np.NaN
    if cfg['lowThreshold'][0] <= 0.0 or cfg['highThreshold'][0] <= 0.0 :
        n = len( log_psd )
        low_threshold, high_threshold, center = threshold_estimate( log_psd[2*n/3:n*9/10],
                                                                    cfg['noiseFactor'][0],
                                                                    cfg['peakFactor'][0] )
        if verbose > 1 :
            print
            print 'low_threshold=', low_threshold, center+low_threshold
            print 'high_threshold=', high_threshold, center+high_threshold
            print 'center=', center

    ## plt.figure()
    ## plt.bar( bins[:-1], hist, width=bins[1]-bins[0] )
    ## plt.axvline( center+low_threshold, color='r' )
    ## plt.axvline( center+high_threshold, color='m' )
    ## plt.show()

    # detect peaks in decibel power spectrum:
    all_freqs = detect_peaks( psd_freqs, log_psd, low_threshold, accept_psd_peaks )

    # select good peaks:
    wthresh = cfg['maxPeakWidthFac'][0]*(psd_freqs[1] - psd_freqs[0])
    if wthresh < cfg['minPeakWidth'][0] :
        wthresh = cfg['minPeakWidth'][0]
    freqs = all_freqs[(all_freqs[:,2]>high_threshold) &
                      (all_freqs[:,0] >= cfg['minimumFrequency'][0]) &
                      (all_freqs[:,0] <= cfg['maximumWorkingFrequency'][0]) &
                      (all_freqs[:,3]<wthresh),:]

    # convert peak sizes back to power:
    freqs[:,1] = 10.0**(0.1*freqs[:,1])
    all_freqs[:,1] = 10.0**(0.1*all_freqs[:,1])

    # detect harmonic groups:
    groups, fzero_harmonics, mains = extract_fundamentals( freqs, all_freqs, psd_freqs[1] - psd_freqs[0], cfg )

    return groups, fzero_harmonics, mains, all_freqs, freqs[:,0], low_threshold, high_threshold, center


def manual_input_wave_or_puls(test_freq, test_power, wave_ls, pulse_ls, known_answer):
    """
    This function helps to assign the fish type (pulse or wave) when the meanslopes of the powerspectrum is
    positiv (indicator for pulsefish) but the fundamental frequency is > 100 Hz. You will again have a lock
    on the powerspectrum of this fish and be foreced to assign it as a pulse- or a wavefish. You also have
    the choise to exclude the fish.

    :param test_freq: list
    :param test_power: list
    :param wave_ls: list
    :param pulse_ls: list

    :return: wave_ls, pulse_ls
    """
    if test_freq[0] in known_answer['freqs']:
        print ''
        print '### using known answers ###'
        print ''
        for i, j in enumerate(known_answer['freqs']):
            if j == test_freq[0]:
                response = known_answer['decision'][i]
    else:
        print ''
        print '### Programm needs input ###'
        print ('The fundamental frequency of this fish is: %.2f Hz.' % test_freq[0])
        print 'Here is the powerspectrum for this fish. Decide!!!'

        fig, ax = plt.subplots()
        ax.plot(test_freq, test_power, 'o')
        # plt.show()
        plt.draw()
        plt.pause(1)

        response = raw_input('Do we have a Wavefish [w] or a Pulsfish [p]? Or exclude the fish [ex]?')
        plt.close()
        if response in ['w', 'p', 'ex']:
            known_answer['freqs'].append(test_freq[0])
            known_answer['decision'].append(response)
        print ''
    if response == "w":
        wave_ls.append(test_freq[0])
    elif response == "p":
        pulse_ls.append(test_freq[0])
    elif response == "ex":
        print 'fish excluded.'
        print ''
    else:
        print '!!! input not valid !!!'
        print 'try again...'
        wave_ls, pulse_ls, known_answer = manual_input_wave_or_puls(test_freq, test_power, wave_ls, pulse_ls, known_answer)
    return wave_ls, pulse_ls, known_answer


def puls_or_wave(fishlist, known_answer, make_plots=False):
    """
    This function gets the array fishlist. (see below)
                    Analysis the data and discriminates between puls and wavefish.
                    returns lists containing the fundamental frequencies for either wave- or pulsfish.

    :param fishlist: dict
    :param make_plots:
    :return:lists: puls_ls, wave_ls
    """

    wave_ls = []
    pulse_ls = []

    for fish_idx in np.arange(len(fishlist)):
        test_freq = []
        test_power = []
        for harmo_idx in np.arange(len(fishlist[fish_idx])):
            test_freq.append(fishlist[fish_idx][harmo_idx][0])
            test_power.append(fishlist[fish_idx][harmo_idx][1])

            # wave_or_puls = []
        slopes = [] #each slope is calculated twice
        for first_idx in np.arange(len(test_power)):
            for second_idx in np.arange(len(test_power)):
                if first_idx > second_idx:
                    slopes.append((test_power[first_idx]-test_power[second_idx])/(test_freq[first_idx]-test_freq[second_idx]))
                if first_idx < second_idx:
                    slopes.append((test_power[second_idx]-test_power[first_idx])/(test_freq[second_idx]-test_freq[first_idx]))
        mean_slopes = np.mean(slopes)

        if mean_slopes > 0:
            if test_freq[0] >= 100:
                wave_ls, pulse_ls, known_answer = manual_input_wave_or_puls(test_freq, test_power, wave_ls, pulse_ls, known_answer)
            else:
                pulse_ls.append(test_freq[0])

        if mean_slopes < -0:
            if test_freq[0] < 100:
                wave_ls, pulse_ls, known_answer = manual_input_wave_or_puls(test_freq, test_power, wave_ls, pulse_ls, known_answer)
            else:
                wave_ls.append(test_freq[0])

        if make_plots:
            fig, ax = plt.subplots()
            ax.plot(test_freq, test_power, 'o')
            plt.show()

    return pulse_ls, wave_ls, known_answer


def when_to_ask_to_start_next_script(filepath):
    if not os.path.exists('num_of_files_processed.npy'):
        help_v = filepath.split('/')
        directory = ''
        filetype = filepath.split('.')[-1]

        for i in np.arange(len(help_v)-1):
            directory = directory + help_v[i] + '/'
        directory = directory + '*.' + filetype

        proc = subprocess.Popen(['ls %s -1 | wc -l' %directory], stdout=subprocess.PIPE, shell=True)
        (out, err) = proc.communicate()

        num_of_files_processed = [int(float(out))]
        np.save('num_of_files_processed.npy', num_of_files_processed)
    else:
        num_of_files_processed = np.load('num_of_files_processed.npy')

    return num_of_files_processed[0]


def num_current_file():
    if not os.path.exists('num_of_current_file.npy'):
        num_of_current_file = [1]
        np.save('num_of_current_file.npy', num_of_current_file)
    else:
        num_of_current_file = np.load('num_of_current_file.npy')
        num_of_current_file[0] += 1
        np.save('num_of_current_file.npy', num_of_current_file)
    return num_of_current_file[0]


class FishTracker :
    def __init__( self, samplingrate ) :
        self.rate = samplingrate
        self.tstart = 0
        self.fish_freqs_dict = {}
        self.pulsfish_freqs_dict = {}
        self.datasize = 200.0  # seconds                               ## DATASIZE ##
        self.step = 0.5                                                ## STEP ##
        self.fresolution = 0.5
        self.twindow = 8.0
        self.fishes = {}
        self.pulsfishes = {}
        self.known_answer = {'freqs':[], 'decision':[]}

    def processdata( self, data, test_longfile=False ): #, rate, fish_freqs_dict, tstart, datasize, step ) :
        """
        gets sound data.
        builds a powerspectrum over 8 sec. these 8 seconds shift through the sound data in steps of 0.5 sec.

        calls function: harmonic_groups witch returns an array (fishlist) containing information for each fish.
                    Example: np.array([fund_freq_fish1, power], [harmonic1_fish1, power], ...),
                             np.array([fund_freq_fish2, power], [harmonic1_fish2, power], ...), ...

        calls function: puls_or_wave. This function gets the array fishlist. (see above)
                    Analysis the data and discriminates between puls and wavefish.
                    returns lists containing the fundamental frequencies for either wave- or pulsfish.

        finally: updates a global dictionary (self.fish_freqs_dict) witch gets the time variable (as key)
                 and the list of WAVE-fishes.

                 updates a global dictionary (self.pulsfish_freqs_dict) witch gets the time variable (as key)
                 and the list of PULS-fishes.
        """
        known_answer = self.known_answer
        nfft = int( np.round( 2**(np.floor(np.log(self.rate/self.fresolution) / np.log(2.0)) + 1.0) ) )
        if nfft < 16 :
            nfft = 16
        tw = int(np.round(self.twindow*self.rate))
        minw = nfft*(cfg['minPSDAverages'][0]+1)/2
        if tw < minw :
            tw = minw
        window = tw/self.rate

        wave_fish_freqs = []
        puls_fish_freqs = []
        fish_time = []

        stepw = int(np.round(self.step*self.rate))

        self.datasize = len( data ) / self.rate

        for t0 in np.arange( 0, len( data ) - tw, stepw ) :
        # running Power spectrum in windows of 8 seconds (tw) in steps of 0,5 seconds (stepw)

            #####################################################
            if test_longfile:  # Only processes 800 seconds of a long file
                if self.tstart+(t0*1.0/self.rate) > 800:
                    break
            #####################################################
            power, freqs = ml.psd( data[t0:t0+tw], NFFT=nfft, noverlap=nfft/2, Fs=self.rate, detrend=ml.detrend_mean )
            fishlist, _, mains, allpeaks, peaks, lowth, highth, center = harmonic_groups( freqs, power, cfg )

            pulse_ls, wave_ls, known_answer = puls_or_wave(fishlist, self.known_answer)
            self.known_answer = known_answer

            cfg['lowThreshold'][0] = lowth
            cfg['highThreshold'][0] = highth

            # wavefish fundamental frequencies:
            for fish in wave_ls :
                if fish not in wave_fish_freqs:
                    wave_fish_freqs.append(fish)
                    fish_time.append(window/2)
                else:
                    fish_time[wave_fish_freqs.index(fish)] += window/2
            temp_dict = {(self.tstart+(t0*1.0/self.rate)): wave_fish_freqs}
            print 'point in time:', self.tstart+(t0*1.0/self.rate), 'seconds.'
            self.fish_freqs_dict.update(temp_dict)
            wave_fish_freqs = []

            # pulsfish fundamental frequencies:
            for pulsfish in pulse_ls:
                if pulsfish not in puls_fish_freqs:
                    puls_fish_freqs.append(pulsfish)
            temp_dict = {(self.tstart+(t0*1.0/self.rate)): puls_fish_freqs}
            self.pulsfish_freqs_dict.update(temp_dict)
            puls_fish_freqs = []


        self.tstart += self.datasize
        self.known_answer = {'freqs':[], 'decision':[]}


    def specto_with_sorted_pulsfish( self ):
        if len(sys.argv) == 2:
            plot_fishes = []
            plot_time = []

            fig, ax = plt.subplots(facecolor= 'white')

            for t in self.pulsfish_freqs_dict.keys():
                ax.scatter([t]*len(self.pulsfish_freqs_dict[t]), self.pulsfish_freqs_dict[t])
            for i in self.pulsfishes.keys():
                print 'Phase III; ', 'Processing pulsfish no.: ',  i
                # print 'III', i
                tnew = np.arange(len(self.pulsfishes[i]))*self.step
                help_tnew = np.arange(len(tnew)//200)

                for k in np.arange(len(tnew)):
                    for l in help_tnew:
                        if tnew[k] > 191.5+l*200:
                            tnew[k] += 8

                for j in np.arange(len(self.pulsfishes[i])):
                    if self.pulsfishes[i][j] is not np.nan:
                        plot_fishes.append(self.pulsfishes[i][j])
                        plot_time.append(tnew[j])
                # print i, np.mean(plot_fishes)

                ax.plot(plot_time, plot_fishes, linewidth= 2)
                plot_fishes = []
                plot_time = []

            ax.set_ylim([0, 2000])
            ax.set_xlabel('time [s]', fontsize ='15')
            ax.set_ylabel('frequencies [hz]', fontsize='15')
            ax.spines["right"].set_visible(False)
            ax.spines["top"].set_visible(False)
            ax.tick_params(axis='both', direction='out')
            ax.get_xaxis().tick_bottom()
            ax.get_yaxis().tick_left()

            plt.xticks(fontsize='15')
            plt.yticks(fontsize='15')
            plt.show()


    def specto_with_sorted_wavefish( self ):
        if len(sys.argv) == 2:
            plot_fishes = []
            plot_time = []

            fig, ax = plt.subplots(facecolor= 'white')

            for t in self.fish_freqs_dict.keys():
                ax.scatter([t]*len(self.fish_freqs_dict[t]), self.fish_freqs_dict[t])
            for i in self.fishes.keys():
                print 'Phase III; ', 'Processing wavefish no.: ',  i
                tnew = np.arange(len(self.fishes[i]))*self.step
                help_tnew = np.arange(len(tnew)//200)

                for k in np.arange(len(tnew)):
                    for l in help_tnew:
                        if tnew[k] > 191.5+l*200:
                            tnew[k] += 8

                for j in np.arange(len(self.fishes[i])):
                    if self.fishes[i][j] is not np.nan:
                        plot_fishes.append(self.fishes[i][j])
                        plot_time.append(tnew[j])
                # print i, np.mean(plot_fishes)

                ax.plot(plot_time, plot_fishes, linewidth= 2)
                plot_fishes = []
                plot_time = []

            ax.set_ylim([0, 2000])
            ax.set_xlabel('time [s]', fontsize ='15')
            ax.set_ylabel('frequencies [hz]', fontsize='15')
            ax.spines["right"].set_visible(False)
            ax.spines["top"].set_visible(False)
            ax.tick_params(axis='both', direction='out')
            ax.get_xaxis().tick_bottom()
            ax.get_yaxis().tick_left()

            plt.xticks(fontsize='15')
            plt.yticks(fontsize='15')
            plt.show()

        # pp = PdfPages('spec_w_fish.pdf')
        # fig.savefig(pp, format='pdf')
        # pp.close()


    def printspecto_pulsfish( self ):
        """
        gets access to the dictionary self.pulsfish_freqs_dict with contains the time as key and the fundamental frequencies
        of the fishes available at this time.

        finaly: builds a scatterplot x-axis: time in sec; y-axis: frequency
        """

        if len(sys.argv) == 2:
            fig, ax = plt.subplots(facecolor= 'white')

            for t in self.pulsfish_freqs_dict.keys():
                ax.scatter([t]*len(self.pulsfish_freqs_dict[t]), self.pulsfish_freqs_dict[t])

            ax.set_ylim([0, 2000])
            ax.set_xlabel('time [s]', fontsize='15')
            ax.set_ylabel('frequencies [hz]', fontsize='15')
            ax.spines["right"].set_visible(False)
            ax.spines["top"].set_visible(False)
            ax.tick_params(axis='both', direction='out')
            ax.get_xaxis().tick_bottom()
            ax.get_yaxis().tick_left()

            plt.xticks(fontsize='15')
            plt.yticks(fontsize='15')
            plt.show()


    def printspecto_wavefish( self ):
        """
        gets access to the dictionary self.fish_freqs_dict with contains the time as key and the fundamental frequencies
        of the fishes available at this time.

        finaly: builds a scatterplot x-axis: time in sec; y-axis: frequency
        """

        if len(sys.argv) == 2:
            fig, ax = plt.subplots(facecolor= 'white')

            for t in self.fish_freqs_dict.keys():
                ax.scatter([t]*len(self.fish_freqs_dict[t]), self.fish_freqs_dict[t])

            ax.set_ylim([0, 2000])
            ax.set_xlabel('time [s]', fontsize='15')
            ax.set_ylabel('frequencies [hz]', fontsize='15')
            ax.spines["right"].set_visible(False)
            ax.spines["top"].set_visible(False)
            ax.tick_params(axis='both', direction='out')
            ax.get_xaxis().tick_bottom()
            ax.get_yaxis().tick_left()

            plt.xticks(fontsize='15')
            plt.yticks(fontsize='15')
            plt.show()


    def sort_my_pulsfish( self ):
        """
        this function works with the global dictionary self.pulsfish_freq_dict and assigns the frequencies to fishes.


        finally: updates a global dict (self.pulsfishes) witch contains the fishes and the frequencies belonging to it.
                    Example: {[1: np.NaN     np.NaN      174.3       175.0       np.NaN ...],
                              [2: np.NaN     180.7       np.NaN      181.9       np.NaN ...], ...
                np.NaN --> fish was not available at this messuing point.
                this dict therefore carries the time variable.

                polt: x-axis: fishno.; y-axis: frequencies
        """

        dict_times = self.pulsfish_freqs_dict.keys()

        for k, t in enumerate(sorted(self.pulsfish_freqs_dict)):

            if t is dict_times[0]:
                for i in np.arange(len(self.pulsfish_freqs_dict[t])):
                    print 'Phase II; ', 'point in time is: ', t, 'secondes.', 'pulsfish no.:', i
                    temp_fish = {len(self.pulsfishes)+1: [self.pulsfish_freqs_dict[t][i]]}
                    self.pulsfishes.update(temp_fish)
            else:
                for i in np.arange(len(self.pulsfish_freqs_dict[t])):
                    print 'Phase II; ', 'point in time is: ', t, 'secondes.', 'pulsfish no.:', i
                    help_v = 0
                    new_freq = self.pulsfish_freqs_dict[t][i]

                    for j in self.pulsfishes.keys():
                        for p in np.arange(len(self.pulsfishes[j]))+1:
                            if self.pulsfishes[j][-p] is not np.nan:
                                index_last_nan = -p
                                break
                        if new_freq > self.pulsfishes[j][index_last_nan] -0.5 and new_freq <= self.pulsfishes[j][index_last_nan] +0.5 and help_v == 0:
                            self.pulsfishes[j].append(new_freq)
                            help_v +=1

                    if help_v is 0:
                        temp_fish = {len(self.pulsfishes)+1: []}
                        for l in np.arange(k):
                            temp_fish[len(self.pulsfishes)+1].append(np.NaN)
                        temp_fish[len(self.pulsfishes)+1].append(self.pulsfish_freqs_dict[t][i])
                        self.pulsfishes.update(temp_fish)
                    elif help_v >= 2:
                        print "added frequency to more than one fish. reduce tolerance!!!"
                        break
                for m in self.pulsfishes.keys():
                    if len(self.pulsfishes[m]) < k+1:
                        self.pulsfishes[m].append(np.nan)

        if len(sys.argv) == 2:
            fig, ax = plt.subplots(facecolor= 'white')
            for n in self.pulsfishes.keys():
                ax.plot([n]*len(self.pulsfishes[n]), self.pulsfishes[n], 'o')
            ax.set_xlim([0, len(self.pulsfishes)+1])
            ax.set_ylim([0, 2000])
            ax.set_xlabel('fish Nr.', fontsize='15')
            ax.set_ylabel('frequency [hz]', fontsize='15')
            ax.spines["right"].set_visible(False)
            ax.spines["top"].set_visible(False)
            ax.tick_params(axis='both', direction='out')
            ax.get_xaxis().tick_bottom()
            ax.get_yaxis().tick_left()
            plt.xticks(fontsize='15')
            plt.yticks(fontsize='15')
            plt.show()


    def sort_my_wavefish( self ):
        """
        this function works with the global dictionary self.fish_freq_dict and assigns the frequencies to fishes.


        finally: updates a global dict (self.fishes) witch contains the fishes and the frequencies belonging to it.
                    Example: {[1: np.NaN     np.NaN      174.3       175.0       np.NaN ...],
                              [2: np.NaN     180.7       np.NaN      181.9       np.NaN ...], ...
                np.NaN --> fish was not available at this mesuing point.
                this dict therefore carries the time variable.

                polt: x-axis: fishno.; y-axis: frequencies
        """

        dict_times = self.fish_freqs_dict.keys()

        for k, t in enumerate(sorted(self.fish_freqs_dict)):

            if t is dict_times[0]:
                for i in np.arange(len(self.fish_freqs_dict[t])):
                    print 'Phase II; ', 'point in time is: ', t, 'secondes.', 'wavefish no.:', i
                    temp_fish = {len(self.fishes)+1: [self.fish_freqs_dict[t][i]]}
                    self.fishes.update(temp_fish)
            else:
                for i in np.arange(len(self.fish_freqs_dict[t])):
                    print 'Phase II; ', 'point in time is: ', t, 'secondes.', 'wavefish no.:', i
                    help_v = 0
                    new_freq = self.fish_freqs_dict[t][i]

                    for j in self.fishes.keys():
                        for p in np.arange(len(self.fishes[j]))+1:
                            if self.fishes[j][-p] is not np.nan:
                                index_last_nan = -p
                                break
                        if new_freq > self.fishes[j][index_last_nan] -0.5 and new_freq <= self.fishes[j][index_last_nan] +0.5 and help_v == 0:
                            self.fishes[j].append(new_freq)
                            help_v +=1

                    if help_v is 0:
                        temp_fish = {len(self.fishes)+1: []}
                        for l in np.arange(k):
                            temp_fish[len(self.fishes)+1].append(np.NaN)
                        temp_fish[len(self.fishes)+1].append(self.fish_freqs_dict[t][i])
                        self.fishes.update(temp_fish)
                    elif help_v >= 2:
                        print "added frequency to more than one fish. reduce tolerance!!!"
                        break
                for m in self.fishes.keys():
                    if len(self.fishes[m]) < k+1:
                        self.fishes[m].append(np.nan)

        if len(sys.argv) == 2:
            fig, ax = plt.subplots(facecolor= 'white')
            for n in self.fishes.keys():
                ax.plot([n]*len(self.fishes[n]), self.fishes[n], 'o')
            ax.set_xlim([0, len(self.fishes)+1])
            ax.set_ylim([0, 2000])
            ax.set_xlabel('fish Nr.', fontsize='15')
            ax.set_ylabel('frequency [hz]', fontsize='15')
            ax.spines["right"].set_visible(False)
            ax.spines["top"].set_visible(False)
            ax.tick_params(axis='both', direction='out')
            ax.get_xaxis().tick_bottom()
            ax.get_yaxis().tick_left()
            plt.xticks(fontsize='15')
            plt.yticks(fontsize='15')
            plt.show()


    def latex_pdf( self ):
        tf = open( 'Brasil.tex', 'w')
        tf.write( '\\documentclass[a4paper,12pt,pdflatex]{article}\n' )
        tf.write( '\\usepackage{graphics}\n' )
        # tf.write( '\\usepackage{siunits}\n' )
        tf.write( '\n' )
        tf.write( '\\begin{document}\n' )
        # tf.write( '\\section*{%s}\n' % filename )
        tf.write( '\\section*{fish of brasil}\n')
        tf.write( '\n' )
        tf.write( '\n' )
        tf.write( '\n' )
        tf.write( '\\includegraphics{sorted_fish.pdf}\n' )
        tf.write( '\\pagebreak\n' )
        tf.write( '\\includegraphics{spec_w_fish.pdf}\n' )
        # tf.write( '\\pagebreak\n' )
        tf.write( '\n' )
        tf.write( '\\begin{tabular}[t]{rr}\n' )
        tf.write( '\\hline\n' )
        tf.write( 'fish no. & freq [Hz] \\\\ \\hline \n' )

        # tf.write( '%s & %d \\\\\n' % (sorted_fish_freqs_2[5], s) )
        for i in self.fishes.keys():
            ffish = []
            for j in np.arange(len(self.fishes[i])):
                if self.fishes[i][j] is not np.nan:
                    ffish.append(self.fishes[i][j])
            if (i) % 35 == 0:
                tf.write( '%s & %s \\\\\n' % (i, np.mean(ffish)) )
                tf.write( '\\hline\n' )
                tf.write( '\\end{tabular}\n' )
                tf.write( '\\begin{tabular}[t]{rr}\n' )
                tf.write( '\\hline\n' )
                tf.write( 'fish no. & freq [Hz] \\\\ \\hline \n' )

            else:
                tf.write( '%s & %s \\\\\n' % (i, np.mean(ffish)) )


        tf.write( '\\hline\n' )
        tf.write( '\\end{tabular}\n' )
        tf.write( '\n' )
        tf.write( '\n' )
        tf.write( '\\end{document}\n' )
        tf.close()
        os.system( 'pdflatex Brasil')
        os.remove( 'Brasil.aux' )
        os.remove( 'Brasil.log' )
        os.remove( 'Brasil.tex' )
        os.remove( 'sorted_fish.pdf' )
        os.remove( 'spec_w_fish.pdf' )


    def puls_main_frequencies(self):
        """
        take the global variable self.pulsfishes.
        builds the mean frequency for each fish.

        :return: list of mean frequenies for each pulsfish.
        """

        mean_fishes = []
        keys = self.pulsfishes.keys()
        build_mean = []

        for fish in keys:
            for time in np.arange(len(self.pulsfishes[fish])):
                if self.pulsfishes[fish][time] is not np.nan:
                    build_mean.append(self.pulsfishes[fish][time])
            mean_fishes.append(np.mean(build_mean))
            build_mean = []
        return mean_fishes


    def wave_main_frequencies(self):
        """
        take the global variable self.fishes.
        builds the mean frequency for each fish.

        :return: list of mean frequenies for each wavefish.
        """

        mean_fishes = []
        keys = self.fishes.keys()
        build_mean = []

        for fish in keys:
            for time in np.arange(len(self.fishes[fish])):
                if self.fishes[fish][time] is not np.nan:
                    build_mean.append(self.fishes[fish][time])
            mean_fishes.append(np.mean(build_mean))
            build_mean = []
        return mean_fishes


    def main_frequency_hist(self):
        mean_fishes = []
        keys = self.fishes.keys()
        build_mean = []

        for i in keys:
            for j in np.arange(len(self.fishes[i])):
                if self.fishes[i][j] is not np.nan:
                    build_mean.append(self.fishes[i][j])
            mean_fishes.append(np.mean(build_mean))
            build_mean = []

        hist, bins = np.histogram(mean_fishes, bins= len(self.fishes)//4)
        width = 0.7 * (bins[1] - bins[0])
        center = (bins[:-1] + bins[1:]) / 2

        fig, ax = plt.subplots(facecolor='white')
        ax.bar(center, hist, align='center', width=width)
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.tick_params(axis='both', direction='out')
        ax.get_xaxis().tick_bottom()
        ax.get_yaxis().tick_left()
        ax.set_ylabel('counts')
        ax.set_xlabel('frequency')
        plt.xticks(np.arange(0, max(mean_fishes)+100, 250))
        plt.title('Histogram')
        plt.show()
        return mean_fishes


    def get_data( self ):
        data = np.zeros( np.ceil( self.rate*self.datasize ), dtype="<i2" )
        return data


    def mean_multi_pulsfish(self):
        """
        if three arguments are given:
        loads/build and npy file witch contains the main frequencies of the wave fishes of one recording.
        saves the file as .npy

        4th arg is a str

        """
        mean_path = ('%s.npy' %sys.argv[3])
        if not os.path.exists(mean_path):
            np.save(mean_path, np.array([]))
        means_frequencies = np.load(mean_path)

        means_frequencies = means_frequencies.tolist()

        keys = self.pulsfishes.keys()
        build_mean = []

        for fish in keys:
            for time in np.arange(len(self.pulsfishes[fish])):
                if self.pulsfishes[fish][time] is not np.nan:
                    build_mean.append(self.pulsfishes[fish][time])
            means_frequencies.append(np.mean(build_mean))
            build_mean = []

        means_frequencies = np.asarray(means_frequencies)
        np.save(mean_path, means_frequencies)

        print ''
        print 'Mean frequencies of the current pulsfishes collected: '
        print means_frequencies
        print ''


    def mean_multi_wavefish(self):
        """
        if three arguments are given:
        loads/build and npy file witch contains the main frequencies of the wave fishes of one recording.
        saves the file as .npy

        3. arg is a str

        """
        mean_path = ('%s.npy' %sys.argv[2])
        if not os.path.exists(mean_path):
            np.save(mean_path, np.array([]))
        means_frequencies = np.load(mean_path)

        means_frequencies = means_frequencies.tolist()

        keys = self.fishes.keys()
        build_mean = []

        for fish in keys:
            for time in np.arange(len(self.fishes[fish])):
                if self.fishes[fish][time] is not np.nan:
                    build_mean.append(self.fishes[fish][time])
            means_frequencies.append(np.mean(build_mean))
            build_mean = []

        means_frequencies = np.asarray(means_frequencies)
        np.save(mean_path, means_frequencies)

        print ''
        print 'Mean frequencies of the current wavefishes collected: '
        print means_frequencies
        print ''


def main():
    datasize = 50.0
    # config file name:
    progs = sys.argv[0].split( '/' )
    cfgfile = progs[-1].split('.')[0] + '.cfg'

    # command line arguments:
    parser = argparse.ArgumentParser(description='Display waveform, spectrogram, and power spectrum of time series data.', epilog='by Jan Benda (2015)')
    parser.add_argument('--version', action='version', version='1.0')
    parser.add_argument('-v', action='count', dest='verbose' )
    parser.add_argument('file', nargs='?', default='', type=str, help='name of the file wih the time series data')
    parser.add_argument('npy_savefile_wave', nargs='?', default='', type=str)
    parser.add_argument('npy_savefile_puls', nargs='?', default='', type=str)
    parser.add_argument('channel', nargs='?', default=0, type=int, help='channel to be displayed')
    args = parser.parse_args()

    # load configuration from the current directory:
    if os.path.isfile( cfgfile ) :
        print 'load configuration', cfgfile
        load_config( cfgfile, cfg )

    # load configuration files from higher directories:
    filepath = args.file
    absfilepath = os.path.abspath( filepath )
    dirs = os.path.dirname( absfilepath ).split( '/' )
    dirs.append( '' )
    maxlevel = len( dirs )-1
    if maxlevel > 3 :
        maxlevel = 3
    for k in xrange( maxlevel, 0, -1 ) :
        path = '/'.join( dirs[:-k] ) + '/' + cfgfile
        if os.path.isfile( path ) :
            print 'load configuration', path
            load_config( path, cfg )

    # set configuration from command line:
    if args.verbose != None :
        cfg['verboseLevel'][0] = args.verbose

    channel = args.channel
    filename = os.path.basename( filepath )
    try:
        import audioread
    except ImportError:
        print 'python module "audioread" is not installed.'
        quit()

    num_of_files_processed = when_to_ask_to_start_next_script(filepath)
    num_of_current_file = num_current_file()


    with audioread.audio_open( filepath ) as af :
        tracen = af.channels
        if channel >= tracen :
            print 'number of traces in file is', tracen
            quit()
        ft = FishTracker( af.samplerate )
        index = 0

        data = ft.get_data()

        for buffer in af:
            fulldata = np.fromstring( buffer, dtype='<i2' ).reshape( -1, af.channels )
            n = fulldata.shape[0]
            if index+n > len( data ) :
                if index == 0 :
                    print "panic!!!! I need a larger buffer!"
                ft.processdata( data[:index]/2.0**15 )
                index = 0
            if n > 0 :
                data[index:index+n] = fulldata[:n, channel]
                index += n
            else :
                break
        if index > 0 :
            ft.processdata( data[:index]/2.0**15 )

        # processes data from wavefishes (dict: self.fish_freqs_dict; dict: self.fishes)
        # processes data from pulsfishes (dict: self.pulsfish_freqs_dict; dict: self.pulsfishes)
        ft.printspecto_wavefish()

        ft.printspecto_pulsfish()

        ft.sort_my_wavefish()

        ft.sort_my_pulsfish()

        ft.specto_with_sorted_wavefish()

        ft.specto_with_sorted_pulsfish()

        if len(sys.argv) == 3:
            ft.mean_multi_wavefish()

        elif len(sys.argv) == 4:
            ft.mean_multi_wavefish()
            ft.mean_multi_pulsfish()

        else:
            wave_main_frequencies = ft.wave_main_frequencies()
            puls_main_frequencies = ft.puls_main_frequencies()
            print 'Mean frequencies of wavefish: ', wave_main_frequencies
            print 'Mean frequencies of pulsfish: ', puls_main_frequencies

        if num_of_current_file == num_of_files_processed:
            os.remove( 'num_of_files_processed.npy' )
            os.remove( 'num_of_current_file.npy' )
            response = raw_input('Do you want to get some figures (df-histo; eod-histo; distribution on fishtype)? [y/n]')
            if response == 'y':
                os.system('python create_plots.py %s %s' % (sys.argv[2], sys.argv[3]))
if __name__ == '__main__':
    main()
