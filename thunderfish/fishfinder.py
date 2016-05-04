#!/usr/bin/python

import sys
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as ml
import matplotlib.colors as mc
from collections import OrderedDict
import dataloader as dl
import peakdetection as pd
import bestwindow as bw
import pyaudio

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
    all_freqs, _ = pd.detect_peaks(log_psd, low_threshold, psd_freqs,
                                   pd.accept_peaks_size_width)

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


###############################################################################
## audio output:

def init_audio() :
    """
    Initializes audio output.

    Returns:
        audio: a handle for subsequent calls of play() and close_audio()
    """
    oldstderr = os.dup( 2 )
    os.close( 2 )
    tmpfile = 'tmpfile.tmp'
    os.open( tmpfile, os.O_WRONLY | os.O_CREAT )
    audio = pyaudio.PyAudio()
    os.close( 2 )
    os.dup( oldstderr )
    os.close( oldstderr )
    os.remove( tmpfile )
    return audio

def close_audio( audio ) :
    """
    Close audio output.

    Args:
        audio: the handle returned by init_audio()
    """
    audio.terminate()           

def play_audio( audio, data, rate ) :
    """
    Play audio data.

    Args:
        audio: the handle returned by init_audio()
        data (array): the data to be played
        rate (float): the sampling rate in Hertz
    """
    # print 'play'
    stream = audio.open( format=pyaudio.paInt16, channels=1, rate=rate, output=True )
    rawdata = data - np.mean( data )
    rawdata /= np.max( rawdata )*2.0
    ## nr = int( np.round( 0.1*rate ) )
    ## if len( rawdata ) > 2*nr :
    ##     for k in xrange( nr ) :
    ##         rawdata[k] *= float(k)/nr
    ##         rawdata[len(rawdata)-k-1] *= float(k)/nr
    # somehow more than twice as many data are needed:
    rawdata = np.hstack( ( rawdata, np.zeros( 11*len( rawdata )/10 ) ) )
    ad = np.array( np.round(2.0**15*rawdata) ).astype( 'i2' )
    stream.write( ad )
    stream.stop_stream()
    stream.close()

def play_tone( audio, frequency, duration, rate ) :
    """
    Play a tone of a given frequency and duration.

    Args:
        audio: the handle returned by init_audio()
        frequency (float): the frequency of the tone in Hertz
        duration (float): the duration of the tone in seconds
        rate (float): the sampling rate in Hertz
    """
    stream = audio.open( format=pyaudio.paInt16, channels=1, rate=rate, output=True )
    time = np.arange( 0.0, duration, 1.0/rate )
    data = np.sin(2.0*np.pi*frequency*time)
    nr = int( np.round( 0.1*rate ) )
    for k in xrange( nr ) :
        data[k] *= float(k)/float(nr)
        data[len(data)-k-1] *= float(k)/float(nr)
    ## somehow more than twice as many data are needed:
    data = np.hstack( ( data, np.zeros( 11*len( data )/10 ) ) )
    ad = np.array( np.round(2.0**14*data) ).astype( 'i2' )
    stream.write( ad )
    stream.stop_stream()
    stream.close()

    
###############################################################################
## plotting etc.
    
class SignalPlot :
    def __init__( self, samplingrate, data, unit, filename, channel ) :
        self.filename = filename
        self.channel = channel
        self.rate = samplingrate
        self.data = data
        self.unit = unit
        self.time = np.arange( 0.0, len( self.data ) )/self.rate
        self.toffset = 0.0
        self.twindow = 8.0
        if self.twindow > self.time[-1] :
            self.twindow = np.round( 2**(np.floor(np.log(self.time[-1]) / np.log(2.0)) + 1.0) )
        self.ymin = -1.0
        self.ymax = +1.0
        self.trace_artist = None
        self.spectrogram_artist = None
        self.fmin = 0.0
        self.fmax = 0.0
        self.decibel = True
        self.fresolution = cfg['initialFrequencyResolution'][0]
        self.deltaf = 1.0
        self.mains_freq = cfg['mainsFreq'][0]
        self.power_label = None
        self.all_peaks_artis = None
        self.good_peaks_artist = None
        self.power_artist = None
        self.power_frequency_label = None
        self.peak_artists = []
        self.legend = True
        self.legendhandle = None
        self.help = cfg['displayHelp'][0]
        self.helptext = []
        self.allpeaks = []
        self.fishlist = []
        self.mains = []
        self.peak_specmarker = []
        self.peak_annotation = []
        self.min_clip = -np.inf
        self.max_clip = np.inf
        self.generate_color_range()

        # audio output:
        self.audio = init_audio()

        # set key bindings:
        plt.rcParams['keymap.fullscreen'] = 'ctrl+f'
        plt.rcParams['keymap.pan'] = 'ctrl+m'
        plt.rcParams['keymap.quit'] = 'ctrl+w, alt+q, q'
        plt.rcParams['keymap.yscale'] = ''
        plt.rcParams['keymap.xscale'] = ''
        plt.rcParams['keymap.grid'] = ''
        plt.rcParams['keymap.all_axes'] = ''
        
        # the figure:
        self.fig = plt.figure( figsize=( 15, 9 ) )
        self.fig.canvas.set_window_title( self.filename + ' channel {0:d}'.format( self.channel ) )
        self.fig.canvas.mpl_connect( 'key_press_event', self.keypress )
        self.fig.canvas.mpl_connect( 'button_press_event', self.buttonpress )
        self.fig.canvas.mpl_connect( 'pick_event', self.onpick )
        self.fig.canvas.mpl_connect( 'resize_event', self.resize )
        # trace plot:
        self.axt = self.fig.add_axes( [ 0.1, 0.7, 0.87, 0.25 ] )
        self.axt.set_ylabel( 'Amplitude [{:s}]'.format( self.unit ) )
        ht = self.axt.text( 0.98, 0.05, '(ctrl+) page and arrow up, down, home, end: scroll', ha='right', transform=self.axt.transAxes )
        self.helptext.append( ht )
        ht = self.axt.text( 0.98, 0.15, '+, -, X, x: zoom in/out', ha='right', transform=self.axt.transAxes )
        self.helptext.append( ht )
        ht = self.axt.text( 0.98, 0.25, 'y,Y,v,V: zoom amplitudes', ha='right', transform=self.axt.transAxes )
        self.helptext.append( ht )
        ht = self.axt.text( 0.98, 0.35, 'p,P: play audio (display,all)', ha='right', transform=self.axt.transAxes )
        self.helptext.append( ht )
        ht = self.axt.text( 0.98, 0.45, 'ctrl-f: full screen', ha='right', transform=self.axt.transAxes )
        self.helptext.append( ht )
        ht = self.axt.text( 0.98, 0.55, 'w: plot waveform into png file', ha='right', transform=self.axt.transAxes )
        self.helptext.append( ht )
        ht = self.axt.text( 0.98, 0.65, 's: save figure', ha='right', transform=self.axt.transAxes )
        self.helptext.append( ht )
        ht = self.axt.text( 0.98, 0.75, 'q: quit', ha='right', transform=self.axt.transAxes )
        self.helptext.append( ht )
        ht = self.axt.text( 0.98, 0.85, 'h: toggle this help', ha='right', transform=self.axt.transAxes )
        self.helptext.append( ht )
        self.axt.set_xticklabels([])
        # spectrogram:
        self.axs = self.fig.add_axes( [ 0.1, 0.45, 0.87, 0.25 ] )
        self.axs.set_xlabel( 'Time [seconds]' )
        self.axs.set_ylabel( 'Frequency [Hz]' )
        # power spectrum:
        self.axp = self.fig.add_axes( [ 0.1, 0.1, 0.87, 0.25 ] )
        ht = self.axp.text( 0.98, 0.9, 'r, R: frequency resolution', ha='right', transform=self.axp.transAxes )
        self.helptext.append( ht )
        ht = self.axp.text( 0.98, 0.8, 'f, F: zoom', ha='right', transform=self.axp.transAxes )
        self.helptext.append( ht )
        ht = self.axp.text( 0.98, 0.7, '(ctrl+) left, right: move', ha='right', transform=self.axp.transAxes )
        self.helptext.append( ht )
        ht = self.axp.text( 0.98, 0.6, 'l: toggle legend', ha='right', transform=self.axp.transAxes )
        self.helptext.append( ht )
        ht = self.axp.text( 0.98, 0.5, 'd: toggle decibel', ha='right', transform=self.axp.transAxes )
        self.helptext.append( ht )
        ht = self.axp.text( 0.98, 0.4, 'm: toggle mains filter', ha='right', transform=self.axp.transAxes )
        self.helptext.append( ht )
        ht = self.axp.text( 0.98, 0.3, 'left mouse: show peak properties', ha='right', transform=self.axp.transAxes )
        self.helptext.append( ht )
        ht = self.axp.text( 0.98, 0.2, 'shift/ctrl + left/right mouse: goto previous/next harmonic', ha='right', transform=self.axp.transAxes )
        self.helptext.append( ht )
        # plot:
        for ht in self.helptext :
            ht.set_visible( self.help )
        self.update_plots( False )
        plt.show()

    def __del( self ) :
        close_audio( self.audio )

    def generate_color_range( self ) :
         # color and marker range:
        self.colorrange = []
        self.markerrange = []
        mr2 = []
        # first color range:
        cc0 = plt.cm.gist_rainbow(np.linspace( 0.0, 1.0, 8.0 ) )
        # shuffle it:
        for k in xrange( (len(cc0)+1)/2 ) :
            self.colorrange.extend( cc0[k::(len(cc0)+1)/2] )
        self.markerrange.extend( len(cc0 )*'o' )
        mr2.extend( len(cc0 )*'v' )
        # second darker color range:
        cc1 = plt.cm.gist_rainbow(np.linspace( 0.33/7.0, 1.0, 7.0 ) )
        cc1 = mc.hsv_to_rgb( mc.rgb_to_hsv( np.array([cc1]))*np.array([1.0, 0.9, 0.7, 0.0 ]) )[0]
        cc1[:,3] = 1.0
        # shuffle it:
        for k in xrange( (len(cc1)+1)/2 ) :
            self.colorrange.extend( cc1[k::(len(cc1)+1)/2] )
        self.markerrange.extend( len(cc1 )*'^' )
        mr2.extend( len(cc1 )*'*' )
        # third lighter color range:
        cc2 = plt.cm.gist_rainbow(np.linspace( 0.67/6.0, 1.0, 6.0 ) )
        cc2 = mc.hsv_to_rgb( mc.rgb_to_hsv( np.array([cc2]))*np.array([1.0, 0.5, 1.0, 0.0 ]) )[0]
        cc2[:,3] = 1.0
        # shuffle it:
        for k in xrange( (len(cc2)+1)/2 ) :
            self.colorrange.extend( cc2[k::(len(cc2)+1)/2] )
        self.markerrange.extend( len(cc2 )*'D' )
        mr2.extend( len(cc2 )*'x' )
        self.markerrange.extend( mr2 )

    def remove_peak_annotation( self ) :
        for fm in self.peak_specmarker :
            fm.remove()
        self.peak_specmarker = []
        for fa in self.peak_annotation :
            fa.remove()
        self.peak_annotation = []

    def annotate_peak( self, peak, harmonics=-1, inx=-1 ) :
        # marker:
        if inx >= 0 :
            m, = self.axs.plot( [self.toffset+0.01*self.twindow], [peak[0]], linestyle='None',
                                color=self.colorrange[inx%len(self.colorrange)],
                                marker=self.markerrange[inx], ms=10.0, mec=None, mew=0.0, zorder=2 )
        else :
            m, = self.axs.plot( [self.toffset+0.01*self.twindow], [peak[0]], linestyle='None',
                                color='k', marker='o', ms=10.0, mec=None, mew=0.0, zorder=2 )
        self.peak_specmarker.append( m )
        # annotation:
        fwidth = self.fmax - self.fmin
        pl = []
        if cfg['labelFrequency'][0] :
            pl.append( r'$f=${:.1f} Hz'.format(peak[0]) )
        if cfg['labelHarmonic'][0] and harmonics >= 0 :
            pl.append( r'$h=${:d}'.format(harmonics) )
        if cfg['labelPower'][0] :
            pl.append( r'$p=${:g}'.format(peak[1]) )
        if cfg['labelWidth'][0] :
            pl.append( r'$\Delta f=${:.2f} Hz'.format(peak[3]) )
        if cfg['labelDoubleUse'][0] :
            pl.append( r'dc={:.0f}'.format(peak[4]) )
        self.peak_annotation.append( self.axp.annotate( '\n'.join(pl), xy=( peak[0], peak[1] ),
                       xytext=( peak[0]+0.03*fwidth, peak[1] ),
                       bbox=dict(boxstyle='round',facecolor='white'),
                       arrowprops=dict(arrowstyle='-') ) )
        
    def annotate_fish( self, fish, inx=-1 ) :
        self.remove_peak_annotation()
        for harmonic, freq in enumerate( fish[:,0] ) :
            peak = self.allpeaks[np.abs(self.allpeaks[:,0]-freq)<0.8*self.deltaf,:]
            if len( peak ) > 0 :
                self.annotate_peak( peak[0,:], harmonic, inx )
        self.fig.canvas.draw()
            
    def update_plots( self, draw=True ) :
        self.remove_peak_annotation()
        # trace:
        self.axt.set_xlim( self.toffset, self.toffset+self.twindow )
        t0 = int(np.round(self.toffset*self.rate))
        t1 = int(np.round((self.toffset+self.twindow)*self.rate))
        if self.trace_artist == None :
            self.trace_artist, = self.axt.plot( self.time[t0:t1], self.data[t0:t1] )
        else :
            self.trace_artist.set_data( self.time[t0:t1], self.data[t0:t1] )
        self.axt.set_ylim( self.ymin, self.ymax )

        # compute power spectrum:
        nfft = int( np.round( 2**(np.floor(np.log(self.rate/self.fresolution) / np.log(2.0)) + 1.0) ) )
        if nfft < 16 :
            nfft = 16
        t00 = t0
        t11 = t1
        w = t11-t00
        minw = nfft*(cfg['minPSDAverages'][0]+1)/2
        if t11-t00 < minw :
            w = minw
            t11 = t00 + w
        if t11 >= len( self.data ) :
            t11 = len( self.data )
            t00 = t11 - w
        if t00 < 0 :
            t00 = 0
            t11 = w           
        power, freqs = ml.psd( self.data[t00:t11], NFFT=nfft, noverlap=nfft/2, Fs=self.rate, detrend=ml.detrend_mean )
        self.deltaf = freqs[1]-freqs[0]
        # detect fish:
        self.fishlist, fzero_harmonics, self.mains, self.allpeaks, peaks, lowth, highth, center = harmonic_groups( freqs, power, cfg )
        highth = center + highth - 0.5*lowth
        lowth = center + 0.5*lowth

        # spectrogram:
        t2 = t1 + nfft
        specpower, freqs, bins = ml.specgram( self.data[t0:t2], NFFT=nfft, Fs=self.rate, noverlap=nfft/2,
                                              detrend=ml.detrend_mean )
        z = 10.*np.log10(specpower)
        z = np.flipud( z )
        extent = self.toffset, self.toffset+np.amax( bins ), freqs[0], freqs[-1]
        self.axs.set_xlim( self.toffset, self.toffset+self.twindow )
        if self.spectrogram_artist == None :
            self.fmax = np.round((freqs[-1]/4.0)/100.0)*100.0
            min = highth
            min = np.percentile( z, 70.0 )
            max = np.percentile( z, 99.9 )+30.0
            #cm = plt.get_cmap( 'hot_r' )
            cm = plt.get_cmap( 'jet' )
            self.spectrogram_artist = self.axs.imshow( z, aspect='auto',
                                                     extent=extent, vmin=min, vmax=max,
                                                     cmap=cm, zorder=1 )
        else :
            self.spectrogram_artist.set_data( z )
            self.spectrogram_artist.set_extent( extent )
        self.axs.set_ylim( self.fmin, self.fmax )

        # power spectrum:
        self.axp.set_xlim( self.fmin, self.fmax )
        if self.deltaf >= 1000.0 :
            dfs = '%.3gkHz' % 0.001*self.deltaf
        else :
            dfs = '%.3gHz' % self.deltaf
        tw = float(w)/self.rate
        if tw < 1.0 :
            tws = '%.3gms' % ( 1000.0*tw )
        else :
            tws = '%.3gs' % ( tw )
        a = 2*w/nfft-1 # number of ffts
        m = ''
        if cfg['mainsFreq'][0] > 0.0 :
            m = ', mains=%.0fHz' % cfg['mainsFreq'][0]
        if self.power_frequency_label == None :
            self.power_frequency_label = self.axp.set_xlabel( r'Frequency [Hz] (nfft={:d}, $\Delta f$={:s}: T={:s}/{:d}{:s})'.format( nfft, dfs, tws, a, m ) )
        else :
            self.power_frequency_label.set_text( r'Frequency [Hz] (nfft={:d}, $\Delta f$={:s}: T={:s}/{:d}{:s})'.format( nfft, dfs, tws, a, m ) )
        self.axp.set_xlim( self.fmin, self.fmax )
        if self.power_label == None :
            self.power_label = self.axp.set_ylabel( 'Power' )
        if self.decibel :
            self.allpeaks[:,1] = 10.0*np.log10( self.allpeaks[:,1] )
            power = 10.0*np.log10( power )
            pmin = np.min( power[freqs<self.fmax] )
            pmin = np.floor(pmin/10.0)*10.0
            pmax = np.max( power[freqs<self.fmax] )
            pmax = np.ceil(pmax/10.0)*10.0
            doty = pmax-5.0
            self.power_label.set_text( 'Power [dB]' )
            self.axp.set_ylim( pmin, pmax )
        else :
            pmax = np.max( power[freqs<self.fmax] )
            doty = pmax
            pmax *= 1.1
            self.power_label.set_text( 'Power' )
            self.axp.set_ylim( 0.0, pmax )
        if self.all_peaks_artis == None :
            self.all_peaks_artis, = self.axp.plot( self.allpeaks[:,0],
                                                   np.zeros( len( self.allpeaks[:,0] ) )+doty,
                                                   'o', color='#ffffff' )
            self.good_peaks_artist, = self.axp.plot( peaks, np.zeros( len( peaks) )+doty,
                                                     'o', color='#888888' )
        else :
            self.all_peaks_artis.set_data( self.allpeaks[:,0],
                                           np.zeros( len( self.allpeaks[:,0] ) )+doty )
            self.good_peaks_artist.set_data( peaks, np.zeros( len( peaks ) )+doty )
        labels = []
        fsizes = [ np.sqrt(np.sum(self.fishlist[k][:,1])) for k in xrange( len(self.fishlist) ) ]
        fmaxsize = np.max( fsizes ) if len( fsizes ) > 0 else 1.0
        self.axp.set_color_cycle( self.colorrange )
        for k in xrange( len( self.peak_artists ) ) :
            self.peak_artists[k].remove()
        self.peak_artists = []
        for k in xrange( len( self.fishlist ) ) :
            if k >= len( self.markerrange ) :
                break
            fpeaks = self.fishlist[k][:,0]
            fpeakinx = [ int(np.round( fp/self.deltaf )) for fp in fpeaks if fp < freqs[-1] ]
            fsize = 7.0+10.0*(fsizes[k]/fmaxsize)**0.5
            fishpoints, = self.axp.plot( fpeaks[:len(fpeakinx)], power[fpeakinx], linestyle='None',
                                         marker=self.markerrange[k], ms=fsize, mec=None, mew=0.0, zorder=1 )
            self.peak_artists.append( fishpoints )
            if self.deltaf < 0.1 :
                labels.append( '%4.2f Hz' % fpeaks[0] )
            elif self.deltaf < 1.0 :
                labels.append( '%4.1f Hz' % fpeaks[0] )
            else :
                labels.append( '%4.0f Hz' % fpeaks[0] )
        if len( self.mains ) > 0 :
            fpeaks = self.mains[:,0]
            fpeakinx = [ np.round( fp/self.deltaf ) for fp in fpeaks if fp < freqs[-1] ]
            fishpoints, = self.axp.plot( fpeaks[:len(fpeakinx)], power[fpeakinx], linestyle='None',
                                         marker='.', color='k', ms=10, mec=None, mew=0.0, zorder=2 )
            self.peak_artists.append( fishpoints )
            labels.append( '%3.0f Hz mains' % cfg['mainsFreq'][0] )
        ncol = len( labels ) / 8 + 1
        self.legendhandle = self.axs.legend( self.peak_artists[:len(labels)], labels, loc='upper right', ncol=ncol )
        self.legenddict = dict()
        for legpoints, (finx, fish) in zip( self.legendhandle.get_lines(), enumerate( self.fishlist ) ) :
            legpoints.set_picker( 8 )
            self.legenddict[legpoints] = [ finx, fish ]
        self.legendhandle.set_visible( self.legend )
        if self.power_artist == None :
            self.power_artist, = self.axp.plot( freqs, power, 'b', zorder=3 )
        else :
            self.power_artist.set_data( freqs, power )
        if draw :
            self.fig.canvas.draw()
                 
    def keypress( self, event ) :
        # print 'pressed', event.key
        if event.key in '+=X' :
            if self.twindow*self.rate > 20 :
                self.twindow *= 0.5
                self.update_plots()
        elif event.key in '-x' :
            if self.twindow < len( self.data )/self.rate :
                self.twindow *= 2.0
                self.update_plots()
        elif event.key == 'pagedown' :
            if self.toffset + 0.5*self.twindow < len( self.data )/self.rate :
                self.toffset += 0.5*self.twindow
                self.update_plots()
        elif event.key == 'pageup' :
            if self.toffset > 0 :
                self.toffset -= 0.5*self.twindow
                if self.toffset < 0.0 :
                    self.toffset = 0.0
                self.update_plots()
        elif event.key == 'a' :
            if np.isinf(self.min_clip) or np.isinf(self.max_clip) :
                # TODO: add these to config parameter:
                clip_win_size = 0.5
                min_clip_fac = 2.0
                self.min_clip, self.max_clip = bw.clip_amplitudes(self.data, int(clip_win_size*self.rate),
                                                                  min_fac=min_clip_fac)
            # TODO: add config parameter:
            idx0, idx1, clipped = bw.best_window_indices(self.data, self.rate, single=True,
                                    win_size=4.0, win_shift=0.1, thresh_ampl_fac=3.0,
                                    min_clip=self.min_clip, max_clip=self.max_clip,
                                    w_cv_ampl=10.0, tolerance=0.5)
            if idx1 > 0 :
                self.toffset = idx0/self.rate
                self.twindow = (idx1 - idx0)/self.rate
                self.update_plots()
        elif event.key == 'ctrl+pagedown' :
            if self.toffset + 5.0*self.twindow < len( self.data )/self.rate :
                self.toffset += 5.0*self.twindow
                self.update_plots()
        elif event.key == 'ctrl+pageup' :
            if self.toffset > 0 :
                self.toffset -= 5.0*self.twindow
                if self.toffset < 0.0 :
                    self.toffset = 0.0
                self.update_plots()
        elif event.key == 'down' :
            if self.toffset + self.twindow < len( self.data )/self.rate :
                self.toffset += 0.05*self.twindow
                self.update_plots()
        elif event.key == 'up' :
            if self.toffset > 0.0 :
                self.toffset -= 0.05*self.twindow
                if self.toffset < 0.0 :
                    self.toffset = 0.0
                self.update_plots()
        elif event.key == 'home':
            if self.toffset > 0.0 :
                self.toffset = 0.0
                self.update_plots()
        elif event.key == 'end':
            toffs = np.floor( len( self.data )/self.rate / self.twindow ) * self.twindow
            if self.toffset < toffs :
                self.toffset = toffs
                self.update_plots()
        elif event.key == 'y':
            h = self.ymax - self.ymin
            c = 0.5*(self.ymax + self.ymin)
            self.ymin = c-h
            self.ymax = c+h
            self.axt.set_ylim( self.ymin, self.ymax )
            self.fig.canvas.draw()
        elif event.key == 'Y':
            h = 0.25*(self.ymax - self.ymin)
            c = 0.5*(self.ymax + self.ymin)
            self.ymin = c-h
            self.ymax = c+h
            self.axt.set_ylim( self.ymin, self.ymax )
            self.fig.canvas.draw()
        elif event.key == 'v':
            t0 = int(np.round(self.toffset*self.rate))
            t1 = int(np.round((self.toffset+self.twindow)*self.rate))
            min = np.min( self.data[t0:t1] )
            max = np.max( self.data[t0:t1] )
            h = 0.5*(max - min)
            c = 0.5*(max + min)
            self.ymin = c-h
            self.ymax = c+h
            self.axt.set_ylim( self.ymin, self.ymax )
            self.fig.canvas.draw()
        elif event.key == 'V':
            self.ymin = -1.0
            self.ymax = +1.0
            self.axt.set_ylim( self.ymin, self.ymax )
            self.fig.canvas.draw()
        elif event.key == 'left' :
            if self.fmin > 0.0 :
                fwidth = self.fmax-self.fmin
                self.fmin -= 0.5*fwidth
                self.fmax -= 0.5*fwidth
                if self.fmin < 0.0 :
                    self.fmin = 0.0
                    self.fmax = fwidth
                self.axs.set_ylim( self.fmin, self.fmax )
                self.axp.set_xlim( self.fmin, self.fmax )
                self.fig.canvas.draw()
        elif event.key == 'right' :
            if self.fmax < 0.5*self.rate :
                fwidth = self.fmax-self.fmin
                self.fmin += 0.5*fwidth
                self.fmax += 0.5*fwidth
                self.axs.set_ylim( self.fmin, self.fmax )
                self.axp.set_xlim( self.fmin, self.fmax )
                self.fig.canvas.draw()
        elif event.key == 'ctrl+left' :
            if self.fmin > 0.0 :
                fwidth = self.fmax-self.fmin
                self.fmin = 0.0
                self.fmax = fwidth
                self.axs.set_ylim( self.fmin, self.fmax )
                self.axp.set_xlim( self.fmin, self.fmax )
                self.fig.canvas.draw()
        elif event.key == 'ctrl+right' :
            if self.fmax < 0.5*self.rate :
                fwidth = self.fmax-self.fmin
                fm = 0.5*self.rate
                self.fmax = np.ceil(fm/fwidth)*fwidth
                self.fmin = self.fmax - fwidth
                if self.fmin < 0.0 :
                    self.fmin = 0.0
                    self.fmax = fwidth
                self.axs.set_ylim( self.fmin, self.fmax )
                self.axp.set_xlim( self.fmin, self.fmax )
                self.fig.canvas.draw()
        elif event.key in 'f' :
            if self.fmax < 0.5*self.rate or self.fmin > 0.0 :
                fwidth = self.fmax-self.fmin
                if self.fmax < 0.5*self.rate :
                    self.fmax = self.fmin + 2.0*fwidth
                elif self.fmin > 0.0 :
                    self.fmin = self.fmax - 2.0*fwidth
                    if self.fmin < 0.0 :
                        self.fmin = 0.0
                        self.fmax = 2.0*fwidth
                self.axs.set_ylim( self.fmin, self.fmax )
                self.axp.set_xlim( self.fmin, self.fmax )
                self.fig.canvas.draw()
        elif event.key in 'F' :
            if self.fmax - self.fmin > 1.0 :
                fwidth = self.fmax-self.fmin
                self.fmax = self.fmin + 0.5*fwidth
                self.axs.set_ylim( self.fmin, self.fmax )
                self.axp.set_xlim( self.fmin, self.fmax )
                self.fig.canvas.draw()
        elif event.key in 'r' :
            if self.fresolution < 1000.0 :
                self.fresolution *= 2.0
                self.update_plots()
        elif event.key in 'R' :
            if 1.0/self.fresolution < self.time[-1] :
                self.fresolution *= 0.5
                self.update_plots()
        elif event.key in 'd' :
            self.decibel = not self.decibel
            self.update_plots()
        elif event.key in 'm' :
            if cfg['mainsFreq'][0] == 0.0 :
                cfg['mainsFreq'][0] = self.mains_freq
            else :
                cfg['mainsFreq'][0] = 0.0
            self.update_plots()
        elif event.key in 't' :
            cfg['peakFactor'][0] -= 0.1
            if cfg['peakFactor'][0] < -5.0 :
                cfg['peakFactor'][0] = -5.0
            print 'peakFactor =', cfg['peakFactor'][0]
            self.update_plots()
        elif event.key in 'T' :
            cfg['peakFactor'][0] += 0.1
            if cfg['peakFactor'][0] > 5.0 :
                cfg['peakFactor'][0] = 5.0
            print 'peakFactor =', cfg['peakFactor'][0]
            self.update_plots()
        elif event.key == 'escape' :
            self.remove_peak_annotation()
            self.fig.canvas.draw()
        elif event.key in 'h' :
            self.help = not self.help
            for ht in self.helptext :
                ht.set_visible( self.help )
            self.fig.canvas.draw()
        elif event.key in 'l' :
            self.legend = not self.legend
            self.legendhandle.set_visible( self.legend )
            self.fig.canvas.draw()
        elif event.key in 'w' :
            self.plot_waveform()
        elif event.key in 'p' :
            self.play_segment()
        elif event.key in 'P' :
            self.play_all()
        elif event.key in '1' :
            self.play_tone( 132.0 ) # C
        elif event.key in '2' :
            self.play_tone( 220.0 ) # A
        elif event.key in '3' :
            self.play_tone( 330.0 ) # E
        elif event.key in '4' :
            self.play_tone( 440.0 ) # A
        elif event.key in '5' :
            self.play_tone( 528.0 ) # C
        elif event.key in '6' :
            self.play_tone( 660.0 ) # E
        elif event.key in '7' :
            self.play_tone( 792.0 ) # G
        elif event.key in '8' :
            self.play_tone( 880.0 ) # A
        elif event.key in '9' :
            self.play_tone( 1056.0 ) # C

    def buttonpress( self, event ) :
        # print 'mouse pressed', event.button, event.key, event.step
        if event.inaxes == self.axp :
            if event.key == 'shift' or event.key == 'control' :
                # show next or previous harmonic:
                if event.key == 'shift' :
                    if event.button == 1 :
                        ftarget = event.xdata/2.0
                    elif event.button == 3 :
                        ftarget = event.xdata*2.0
                else :
                    if event.button == 1 :
                        ftarget = event.xdata/1.5
                    elif event.button == 3 :
                        ftarget = event.xdata*1.5
                foffs = event.xdata - self.fmin
                fwidth = self.fmax - self.fmin
                self.fmin = ftarget - foffs
                self.fmax = self.fmin + fwidth
                self.axs.set_ylim( self.fmin, self.fmax )
                self.axp.set_xlim( self.fmin, self.fmax )
                self.fig.canvas.draw()
            else :
                # put label on peak
                self.remove_peak_annotation()
                # find closest peak:
                fwidth = self.fmax - self.fmin
                peakdist = np.abs(self.allpeaks[:,0]-event.xdata)
                inx = np.argmin( peakdist )
                if peakdist[inx] < 0.005*fwidth :
                    peak = self.allpeaks[inx,:]
                    # find fish:
                    foundfish = False
                    for finx, fish in enumerate( self.fishlist ) :
                        if np.min( np.abs(fish[:,0]-peak[0]) ) < 0.8*self.deltaf :
                            self.annotate_fish( fish, finx )
                            foundfish = True
                            break
                    if not foundfish :
                        self.annotate_peak( peak )
                        self.fig.canvas.draw()
                else :
                    self.fig.canvas.draw()

    def onpick( self, event ) :
        # print 'pick'
        legendpoint = event.artist
        finx, fish = self.legenddict[legendpoint]
        self.annotate_fish( fish, finx )

    def resize( self, event ) :
        # print 'resized', event.width, event.height
        leftpixel = 80.0
        rightpixel = 20.0
        xaxispixel = 50.0
        toppixel = 20.0
        timeaxis = 0.42
        left = leftpixel/event.width
        width = 1.0 - left - rightpixel/event.width
        xaxis = xaxispixel/event.height
        top = toppixel/event.height
        height = (1.0-timeaxis-top)/2.0
        if left < 0.5 and width < 1.0 and xaxis < 0.3 and top < 0.2 :
            self.axt.set_position( [ left, timeaxis+height, width, height ] )
            self.axs.set_position( [ left, timeaxis, width, height ] )
            self.axp.set_position( [ left, xaxis, width, timeaxis-2.0*xaxis ] )

    def plot_waveform( self ) :
        fig = plt.figure()
        ax = fig.add_subplot( 1, 1, 1 )
        name = self.filename.split( '.' )[0]
        if self.channel > 0 :
            ax.set_title( '{filename} channel={channel:d}'.format(
                filename=self.filename, channel=self.channel ) )
            figfile = '{name}-{channel:d}-{time:.4g}s-waveform.png'.format(
                name=name, channel=self.channel, time=self.toffset )
        else :
            ax.set_title( self.filename )
            figfile = '{name}-{time:.4g}s-waveform.png'.format(
                name=name, time=self.toffset )
        t0 = int(np.round(self.toffset*self.rate))
        t1 = int(np.round((self.toffset+self.twindow)*self.rate))
        if self.twindow < 1.0 :
            ax.set_xlabel( 'Time [ms]' )
            ax.set_xlim( 1000.0*self.toffset,
                         1000.0*(self.toffset+self.twindow) )
            ax.plot( 1000.0*self.time[t0:t1], self.data[t0:t1] )
        else :
            ax.set_xlabel( 'Time [s]' )
            ax.set_xlim( self.toffset, self.toffset+self.twindow )
            ax.plot( self.time[t0:t1], self.data[t0:t1] )
        ax.set_ylabel( 'Amplitude [{:s}]'.format( self.unit ) )
        fig.tight_layout()
        fig.savefig( figfile )
        fig.clear()
        plt.close( fig )
        print 'saved waveform figure to', figfile

    def play_segment( self ) :
        t0 = int(np.round(self.toffset*self.rate))
        t1 = int(np.round((self.toffset+self.twindow)*self.rate))
        play_audio( self.audio, self.data[t0:t1], self.rate )
        
    def play_all( self ) :
        play_audio( self.audio, self.data, self.rate )
        
    def play_tone( self, frequency ) :
        play_tone( self.audio, frequency, 1.0, self.rate )
                    

def main():
    # config file name:
    progs = sys.argv[0].split( '/' )
    cfgfile = progs[-1].split('.')[0] + '.cfg'

    # command line arguments:
    parser = argparse.ArgumentParser(description='Display waveform, spectrogram, and power spectrum of time series data.', epilog='by Jan Benda (2015)')
    parser.add_argument('--version', action='version', version='1.0')
    parser.add_argument('-v', action='count', dest='verbose' )
    parser.add_argument('-c', '--save-config', nargs='?', default='', const=cfgfile, type=str, metavar='cfgfile', help='save configuration to file cfgfile (defaults to {0})'.format( cfgfile ))
    parser.add_argument('file', nargs='?', default='', type=str, help='name of the file wih the time series data')
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
    
    # save configuration:
    if len( args.save_config ) > 0 :
        ext = args.save_config.split( '.' )[-1]
        if ext != 'cfg' :
            print 'configuration file name must have .cfg as extension!'
        else :
            print 'write configuration to', args.save_config, '...'
            dump_config( args.save_config, cfg, cfgsec )
        quit()

    # load data:
    channel = args.channel
    filename = os.path.basename(filepath)
    data, freq, unit = dl.load_data(filepath, channel, cfg['verboseLevel'][0])

    # plot:
    sp = SignalPlot(freq, data, unit, filename, channel)

if __name__ == '__main__':
    main()


## data = data/2.0**15
## time = np.arange( 0.0, len( data ) )/freq
## # t=69-69.25: EODf=324 and 344Hz
## #data = data[(time>=69.0) & (time<=69.25)]
## #time = 1000.0*(time[(time>=69.0) & (time<=69.25)]-69.0)
## # 103.23 + 60ms: EODf=324Hz
## data = data[(time>=103.24) & (time<103.28)]
## time = 1000.0*(time[(time>=103.24) & (time<103.28)]-103.24)

#50301L02.WAV t=9 bis 9.15 sec


## 1 fish:
# simple aptero (clipped):
# python fishfinder.py ~/data/fishgrid/Panama2014/MP3_1/20140517_RioCanita/40517L14.WAV
# nice sterno:
# python fishfinder.py ~/data/fishgrid/Panama2014/MP3_1/20140517_RioCanita/40517L31.WAV
# sterno (clipped) with a little bit of background:
# python fishfinder.py ~/data/fishgrid/Panama2014/MP3_1/20140517_RioCanita/40517L26.WAV
# simple brachy (clipped, with a very small one in the background): still difficult, but great with T=4s
# python fishfinder.py ~/data/fishgrid/Panama2014/MP3_1/20140517_RioCanita/40517L30.WAV
# eigenmannia (very nice): EN086.MP3
# single, very nice brachy, with difficult psd:
# python fishfinder.py ~/data/fishgrid/Panama2014/MP3_1/20140517_RioCanita/40517L19.WAV
# python fishfinder.py ~/data/fishgrid/Panama2014/MP3_1/20140517_RioCanita/40517L2[789].WAV

## 2 fish:
# 2 aptero:
# python fishfinder.py ~/data/fishgrid/Panama2014/MP3_1/20140517_RioCanita/40517L10.WAV
# EN098.MP3 and in particular EN099.MP3 nice 2Hz beat!
# 2 brachy beat:
# python fishfinder.py ~/data/fishgrid/Panama2014/MP3_1/20140517_RioCanita/40517L08.WAV
# >= 2 brachys:
# python fishfinder.py ~/data/fishgrid/Panama2014/MP3_1/20140517_RioCanita/40517L2[12789].WAV

## one sterno with weak aptero:
# python fishfinder.py ~/data/fishgrid/Panama2014/MP3_1/20140517_RioCanita/40517L11.WAV
# EN144.MP3

## 2 and 2 fish:
# python fishfinder.py ~/data/fishgrid/Panama2014/MP3_1/20140517_RioCanita/40517L12.WAV

## one aptero with brachy:
# EN148

## lots of fish:
# python fishfinder.py ~/data/fishgrid/Panama2014/MP3_1/20140517_RioCanita/40517L07.WAV
# EN065.MP3 EN066.MP3 EN067.MP3 EN103.MP3 EN104.MP3
# EN109: 1Hz beat!!!!
# EN013: doppel detection of 585 Hz
# EN015,30,31: noise estimate problem

# EN083.MP3 aptero glitch
# EN146 sek 4 sterno frequency glitch

# EN056.MP3 EN080.MP3 difficult low frequencies
# EN072.MP3 unstable low and high freq
# EN122.MP3 background fish detection difficulties at low res

#problems: EN088, EN089, 20140524_RioCanita/EN055 sterno not catched, EN056, EN059
