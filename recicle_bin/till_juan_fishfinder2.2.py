#!/usr/bin/python

# same as fishsorter

__author__ = 'raab'


def description_of_the_whole_code():
    """
    This code works different depending on how many arguments are given.

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
import Auxiliary as aux
from FishRecording import *
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
cfg['minPSDAverages'] = [3, '', 'Minimum number of fft averages for estimating the power spectrum.']
cfg['initialFrequencyResolution'] = [1.0, 'Hz', 'Initial frequency resolution of the power spectrum.']

cfgsec['lowThreshold'] = 'Thresholds for peak detection in power spectra:'
cfg['lowThreshold'] = [0.0, 'dB', 'Threshold for all peaks.\n If 0.0 estimate threshold from histogram.']
cfg['highThreshold'] = [0.0, 'dB', 'Threshold for good peaks. If 0.0 estimate threshold from histogram.']
# cfg['lowThreshold'][0] = 12. # panama
# cfg['highThreshold'][0] = 18. # panama

cfgsec[
    'noiseFactor'] = 'Threshold estimation:\nIf no thresholds are specified they are estimated from the histogram of the decibel power spectrum.'
cfg['noiseFactor'] = [6.0, '', 'Factor for multiplying std of noise floor for lower threshold.']
cfg['peakFactor'] = [0.5, '', 'Fractional position of upper threshold above lower threshold.']

cfgsec['maxPeakWidthFac'] = 'Peak detection in decibel power spectrum:'
cfg['maxPeakWidthFac'] = [3.5, '',
                          'Maximum width of peaks at 0.75 hight in multiples of frequency resolution (might be increased)']
cfg['minPeakWidth'] = [1.0, 'Hz', 'Peaks do not need to be narrower than this.']

cfgsec['mainsFreq'] = 'Harmonic groups:'
cfg['mainsFreq'] = [60.0, 'Hz', 'Mains frequency to be excluded.']
cfg['maxDivisor'] = [4, '', 'Maximum ratio between the frequency of the largest peak and its fundamental']
cfg['freqTolerance'] = [0.7, '',
                        'Harmonics need be within this factor times the frequency resolution of the power spectrum. Needs to be higher than 0.5!']
cfg['maxUpperFill'] = [1, '',
                       'As soon as more than this number of harmonics need to be filled in conescutively stop searching for higher harmonics.']
cfg['maxFillRatio'] = [0.25, '',
                       'Maximum fraction of filled in harmonics allowed (usefull values are smaller than 0.5)']
cfg['maxDoubleUseHarmonics'] = [8, '', 'Maximum harmonics up to which double uses are penalized.']
cfg['maxDoubleUseCount'] = [1, '', 'Maximum overall double use count allowed.']
cfg['powerNHarmonics'] = [10, '', 'Compute total power over the first # harmonics.']

cfgsec['minimumGroupSize'] = 'Acceptance of best harmonic groups:'
cfg['minimumGroupSize'] = [3, '',
                           'Minimum required number of harmonics (inclusively fundamental) that are not filled in and are not used by other groups.']
# cfg['minimumGroupSize'][0] = 2 # panama
cfg['minimumFrequency'] = [20.0, 'Hz', 'Minimum frequency allowed for the fundamental.']
cfg['maximumFrequency'] = [2000.0, 'Hz', 'Maximum frequency allowed for the fundamental.']
cfg['maximumWorkingFrequency'] = [4000.0, 'Hz',
                                  'Maximum frequency to be used to search for harmonic groups and to adjust fundamental frequency.']

cfg['maxHarmonics'] = [0, '', '0: keep all, >0 only keep the first # harmonics.']

cfgsec['displayHelp'] = 'Items to display:'
cfg['displayHelp'] = [False, '', 'Display help on key bindings']
cfg['labelFrequency'] = [True, '', 'Display the frequency of the peak']
cfg['labelHarmonic'] = [True, '', 'Display the harmonic of the peak']
cfg['labelPower'] = [True, '', 'Display the power of the peak']
cfg['labelWidth'] = [True, '', 'Display the width of the peak']
cfg['labelDoubleUse'] = [True, '', 'Display double-use count of the peak']

cfgsec['verboseLevel'] = 'Debugging:'
cfg['verboseLevel'] = [0, '', '0=off upto 4 very detailed']


###############################################################################
## load data:

def load_pickle(filename, trace=0):
    """
    load Joerg's pickle files
    """
    import pickle

    with open(filename, 'rb') as f:
        data = pickle.load(f)
    time = data['time_trace']
    freq = 1000.0 / (time[1] - time[0])
    tracen = data['raw_data'].shape[1]
    if trace >= tracen:
        print 'number of traces in file is', tracen
        quit()
    return freq, data['raw_data'][:, trace], 'mV'


def load_wavfile(filename, trace=0):
    """
    load wav file using scipy io.wavfile
    """
    from scipy.io import wavfile

    freq, data = wavfile.read(filename)
    if len(data.shape) == 1:
        if trace >= 1:
            print 'number of traces in file is', 1
            quit()
        return freq, data / 2.0 ** 15, ''
    else:
        tracen = data.shape[1]
        if trace >= tracen:
            print 'number of traces in file is', tracen
            quit()
        return freq, data[:, trace] / 2.0 ** 15, 'a.u.'


def load_wave(filename, trace=0):
    """
    load wav file using wave module
    """
    try:
        import wave
    except ImportError:
        print 'python module "wave" is not installed.'
        return load_wavfile(filename, trace)

    wf = wave.open(filename, 'r')
    (nchannels, sampwidth, freq, nframes, comptype, compname) = wf.getparams()
    print nchannels, sampwidth, freq, nframes, comptype, compname
    buffer = wf.readframes(nframes)
    format = 'i%d' % sampwidth
    data = np.fromstring(buffer, dtype=format).reshape(-1, nchannels)  # read data
    wf.close()
    print data.shape
    if len(data.shape) == 1:
        if trace >= 1:
            print 'number of traces in file is', 1
            quit()
        return freq, data / 2.0 ** (sampwidth * 8 - 1), ''
    else:
        tracen = data.shape[1]
        if trace >= tracen:
            print 'number of traces in file is', tracen
            quit()
        return freq, data[:, trace] / 2.0 ** (sampwidth * 8 - 1), 'a.u.'


def load_audio(filename, trace=0):
    """
    load wav file using audioread.
    This is not available in python x,y.
    """
    try:
        import audioread
    except ImportError:
        print 'python module "audioread" is not installed.'
        return load_wave(filename, trace)

    data = np.array([])
    with audioread.audio_open(filename) as af:
        tracen = af.channels
        if trace >= tracen:
            print 'number of traces in file is', tracen
            quit()
        data = np.zeros(np.ceil(af.samplerate * af.duration), dtype="<i2")
        index = 0
        for buffer in af:
            fulldata = np.fromstring(buffer, dtype='<i2').reshape(-1, af.channels)
            n = fulldata.shape[0]
            if index + n > len(data):
                n = len(data) - index
            if n > 0:
                data[index:index + n] = fulldata[:n, trace]
                index += n
            else:
                break
    return af.samplerate, data / 2.0 ** 15, 'a.u.'


###############################################################################
## configuration file writing and loading:

def dump_config(filename, cfg, sections=None, header=None, maxline=60):
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

    def write_comment(f, comment, maxline=60, cs='#'):
        # format comment:
        if len(comment) > 0:
            for line in comment.split('\n'):
                f.write(cs + ' ')
                cc = len(cs) + 1  # character count
                for w in line.strip().split(' '):
                    # line too long?
                    if cc + len(w) > maxline:
                        f.write('\n' + cs + ' ')
                        cc = len(cs) + 1
                    f.write(w + ' ')
                    cc += len(w) + 1
                f.write('\n')

    with open(filename, 'w') as f:
        if header != None:
            write_comment(f, header, maxline, '##')
        maxkey = 0
        for key in cfg.keys():
            if maxkey < len(key):
                maxkey = len(key)
        for key, v in cfg.items():
            # possible section entry:
            if sections != None and key in sections:
                f.write('\n\n')
                write_comment(f, sections[key], maxline, '##')

            # get value, unit, and comment from v:
            val = None
            unit = ''
            comment = ''
            if hasattr(v, '__len__') and (not isinstance(v, str)):
                val = v[0]
                if len(v) > 1:
                    unit = ' ' + v[1]
                if len(v) > 2:
                    comment = v[2]
            else:
                val = v

            # next key-value pair:
            f.write('\n')
            write_comment(f, comment, maxline, '#')
            f.write('{key:<{width}s}: {val}{unit:s}\n'.format(key=key, width=maxkey, val=val, unit=unit))


def load_config(filename, cfg):
    """
    Set values of dictionary cfg to values from key-value pairs read in from file.

    Args:
        filename: The name of the file from which to read in the configuration.
        cfg (dict): Configuration keys, values, units, and comments.
    """
    with open(filename, 'r') as f:
        for line in f:
            # do not process empty lines and comments:
            if len(line.strip()) == 0 or line[0] == '#' or not ':' in line:
                continue
            key, val = line.split(':', 1)
            key = key.strip()
            if not key in cfg:
                continue
            cv = cfg[key]
            vals = val.strip().split(' ')
            if hasattr(cv, '__len__') and (not isinstance(cv, str)):
                unit = ''
                if len(vals) > 1:
                    unit = vals[1]
                if unit != cv[1]:
                    print 'unit for', key, 'is', unit, 'but should be', cv[1]
                if type(cv[0]) == bool:
                    cv[0] = (vals[0].lower() == 'true' or vals[0].lower() == 'yes')
                else:
                    cv[0] = type(cv[0])(vals[0])
            else:
                if type(cv[0]) == bool:
                    cfg[key] = (vals[0].lower() == 'true' or vals[0].lower() == 'yes')
                else:
                    cfg[key] = type(cv)(vals[0])


###############################################################################
## peak detection:

def detect_peaks(time, data, threshold, check_func=None, check_conditions=None):
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
                    r = check_func(time, data, event_inx, index, trough_inx, min_inx, threshold, check_conditions)
                    if len(r) > 0:
                        # this really is an event:
                        event_list.append(r)
                else:
                    # this really is an event:
                    event_list.append(time[event_inx])

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

    return np.array(event_list)


###############################################################################
## harmonic group extraction:

def build_harmonic_groups(freqs, more_freqs, deltaf, cfg):
    verbose = cfg['verboseLevel'][0]

    # start at the strongest frequency:
    fmaxinx = np.argmax(freqs[:, 1])
    fmax = freqs[fmaxinx, 0]
    if verbose > 1:
        print
        print 70 * '#'
        print 'freqs:     ', '[', ', '.join(['{:.2f}'.format(f) for f in freqs[:, 0]]), ']'
        print 'more_freqs:', '[', ', '.join(
            ['{:.2f}'.format(f) for f in more_freqs[:, 0] if f < cfg['maximumFrequency'][0]]), ']'
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

    for divisor in xrange(1, cfg['maxDivisor'][0] + 1):

        # define the hypothesized fundamental, which is compared to all higher frequencies:
        fzero = fmax / divisor
        # fzero is not allowed to be smaller than our chosen minimum frequency:
        # if divisor > 1 and fzero < cfg['minimumFrequency'][0]:   # XXX why not also for divisor=1???
        #    break
        fzero_harmonics = 1

        if verbose > 1:
            print '# divisor:', divisor, 'fzero=', fzero

        # ###########################################
        # SEARCH ALL DETECTED FREQUENCIES in freqs
        # this in the end only recomputes fzero!
        newgroup = list()
        npre = -1  # previous harmonics
        ndpre = 0.0  # difference of previous frequency
        connected = True
        for j in xrange(freqs.shape[0]):

            if verbose > 2:
                print 'check freq {:3d} {:8.2f}'.format(j, freqs[j, 0]),

            # IS THE CURRENT FREQUENCY AN INTEGRAL MULTIPLE OF FZERO?
            # divide the frequency-to-be-checked by fzero
            # to get the multiplication factor between freq and fzero
            n = np.round(freqs[j, 0] / fzero)
            if n == 0:
                if verbose > 2:
                    print 'discarded: n == 0'
                continue

            # !! the difference between the current frequency, divided by the derived integer,
            # and fzero should be very very small: 1 resolution step of the fft
            # (freqs[j,0] / n) = should be fzero, plus minus a little tolerance,
            # which is the fft resolution
            nd = np.abs((freqs[j, 0] / n) - fzero)
            # ... compare it to our tolerance
            if nd > freqtol:
                if verbose > 2:
                    print 'discarded: not a harmonic n=%2d d=%5.2fHz tol=%5.2fHz' % (n, nd, freqtol)
                continue

            # two succeeding frequencies should also differ by
            # fzero plus/minus twice! the tolerance:
            if len(newgroup) > 0:
                nn = np.round((freqs[j, 0] - freqs[newgroup[-1], 0]) / fzero)
                if nn == 0:
                    # the current frequency is the same harmonic as the previous one
                    # print divisor, j, freqs[j,0], freqs[newgroup[-1],0]
                    if len(newgroup) > 1:
                        # check whether the current frequency is fzero apart from the previous harmonics:
                        nn = np.round((freqs[j, 0] - freqs[newgroup[-2], 0]) / fzero)
                        nnd = np.abs(((freqs[j, 0] - freqs[newgroup[-2], 0]) / nn) - fzero)
                        if nnd > 2.0 * freqtol:
                            if verbose > 2:
                                print 'discarded: distance to previous harmonics %2d %5.2f %5.2f %8.2f' % (
                                    nn, nnd, freqtol, fzero)
                            continue
                    if ndpre < nd:
                        # the previous frequency is closer to the harmonics, keep it:
                        if verbose > 2:
                            print 'discarded: previous harmonics is closer %2d %5.2f %5.2f %5.2f %8.2f' % (
                                n, nd, ndpre, freqtol, fzero)
                        continue
                    else:
                        # the current frequency is closer to the harmonics, remove the previous one:
                        newgroup.pop()
                else:
                    # check whether the current frequency is fzero apart from the previous harmonics:
                    nnd = np.abs(((freqs[j, 0] - freqs[newgroup[-1], 0]) / nn) - fzero)
                    if nnd > 2.0 * freqtol:
                        if verbose > 2:
                            print 'discarded: distance to previous harmonics %2d %5.2f %5.2f %8.2f' % \
                                  (nn, nnd, freqtol, fzero)
                        continue

            # take frequency:
            newgroup.append(j)  # append index of frequency
            if verbose > 2:
                print 'append n={:.2f} d={:5.2f}Hz tol={:5.2f}Hz'.format(freqs[j, 0] / fzero, nd, freqtol)

            if npre >= 0 and n - npre > 1:
                connected = False
            npre = n
            ndpre = nd

            if connected:
                # adjust fzero as we get more information from the higher frequencies:
                fzero = freqs[j, 0] / n
                fzero_harmonics = int(n)
                if verbose > 2:
                    print 'adjusted fzero to', fzero

        if verbose > 3:
            print 'newgroup:', divisor, fzero, newgroup

        newmoregroup = list()
        fill_ins = 0
        double_use = 0
        ndpre = 0.0  # difference of previous frequency

        # ###########################################
        # SEARCH ALL DETECTED FREQUENCIES in morefreqs
        for j in xrange(more_freqs.shape[0]):

            if verbose > 3:
                print 'check more_freq %3d %8.2f' % (j, more_freqs[j, 0]),

            # IS FREQUENCY A AN INTEGRAL MULTIPLE OF FREQUENCY B?
            # divide the frequency-to-be-checked with fzero:
            # what is the multiplication factor between freq and fzero?
            n = np.round(more_freqs[j, 0] / fzero)
            if n == 0:
                if verbose > 3:
                    print 'discarded: n == 0'
                continue

            # !! the difference between the detection, divided by the derived integer
            # , and fzero should be very very small: 1 resolution step of the fft
            # (more_freqs[j,0] / n) = should be fzero, plus minus a little tolerance,
            # which is the fft resolution
            nd = np.abs((more_freqs[j, 0] / n) - fzero)

            # ... compare it to our tolerance
            if nd > freqtol:
                if verbose > 3:
                    print 'discarded: not a harmonic n=%2d d=%5.2fHz tol=%5.2fHz' % (n, nd, freqtol)
                continue

            # two succeeding frequencies should also differ by fzero plus/minus tolerance:
            if len(newmoregroup) > 0:
                nn = np.round((more_freqs[j, 0] - more_freqs[newmoregroup[-1], 0]) / fzero)
                if nn == 0:
                    # the current frequency is close to the same harmonic as the previous one
                    # print n, newmoregroup[-1], ( more_freqs[j,0] - more_freqs[newmoregroup[-1],0] )/fzero
                    # print divisor, j, n, more_freqs[j,0], more_freqs[newmoregroup[-1],0], more_freqs[newmoregroup[-2],0], newmoregroup[-2]
                    if len(newmoregroup) > 1 and newmoregroup[-2] >= 0:
                        # check whether the current frequency is fzero apart from the previous harmonics:
                        nn = np.round((more_freqs[j, 0] - more_freqs[newmoregroup[-2], 0]) / fzero)
                        nnd = np.abs(((more_freqs[j, 0] - more_freqs[newmoregroup[-2], 0]) / nn) - fzero)
                        if nnd > 2.0 * freqtol:
                            if verbose > 3:
                                print 'discarded: distance to previous harmonics %2d %5.2f %5.2f %8.2f' % (
                                    nn, nnd, freqtol, fzero)
                            continue
                    if ndpre < nd:
                        # the previous frequency is closer to the harmonics, keep it:
                        if verbose > 3:
                            print 'discarded: previous harmonics is closer %2d %5.2f %5.2f %5.2f %8.2f' % (
                                n, nd, ndpre, freqtol, fzero)
                        continue
                    else:
                        # the current frequency is closer to the harmonics, remove the previous one:
                        newmoregroup.pop()
                else:
                    # check whether the current frequency is fzero apart from the previous harmonics:
                    nnd = np.abs(((more_freqs[j, 0] - more_freqs[newmoregroup[-1], 0]) / nn) - fzero)
                    if nnd > 2.0 * freqtol:
                        if verbose > 3:
                            print 'discarded: distance to previous harmonics %2d %5.2f %5.2f %8.2f' % (
                                nn, nnd, freqtol, fzero)
                        continue
            ndpre = nd

            # too many fill-ins upstream of fmax ?
            if more_freqs[j, 0] > fmax and n - 1 - len(newmoregroup) > cfg['maxUpperFill'][0]:
                # finish this group immediately
                if verbose > 3:
                    print 'stopping group: too many upper fill-ins:', n - 1 - len(newmoregroup), '>', \
                        cfg['maxUpperFill'][0]
                break

            # fill in missing harmonics:
            while len(newmoregroup) < n - 1:  # while some harmonics are missing ...
                newmoregroup.append(-1)  # ... add marker for non-existent harmonic
                fill_ins += 1

            # count double usage of frequency:
            if n <= cfg['maxDoubleUseHarmonics'][0]:
                double_use += more_freqs[j, 4]
                if verbose > 3 and more_freqs[j, 4] > 0:
                    print 'double use of %.2fHz' % more_freqs[j, 0],

            # take frequency:
            newmoregroup.append(j)
            if verbose > 3:
                print 'append'

        # double use of points:
        if double_use > cfg['maxDoubleUseCount'][0]:
            if verbose > 1:
                print 'discarded group because of double use:', double_use
            continue

        # ratio of total fill-ins too large:
        if float(fill_ins) / float(len(newmoregroup)) > cfg['maxFillRatio'][0]:
            if verbose > 1:
                print 'dicarded group because of too many fill ins! %d from %d (%g)' % (
                    fill_ins, len(newmoregroup), float(fill_ins) / float(len(newmoregroup))), newmoregroup
            continue

        # REASSEMBLE NEW GROUP BECAUSE FZERO MIGHT HAVE CHANGED AND
        # CALCULATE THE PEAKSUM, GIVEN THE UPPER LIMIT
        # DERIVED FROM morefreqs which can be low because of too many fill ins.
        # newgroup is needed to delete the right frequencies from freqs later on.
        newgroup = []
        fk = 0
        for j in xrange(len(newmoregroup)):
            if newmoregroup[j] >= 0:
                # existing frequency peak:
                f = more_freqs[newmoregroup[j], 0]
                # find this frequency in freqs:
                for k in xrange(fk, freqs.shape[0]):
                    if np.abs(freqs[k, 0] - f) < 1.0e-8:
                        newgroup.append(k)
                        fk = k + 1
                        break
                if fk >= freqs.shape[0]:
                    break

        # fmax might not be in our group, because we adjust fzero:
        if not fmaxinx in newgroup:
            if verbose > 1:
                print "discarded: lost fmax"
            continue

        n = cfg['powerNHarmonics'][0]
        newmoregroup_peaksum = np.sum(more_freqs[newmoregroup[:n], 1])
        fills = np.sum(np.asarray(newmoregroup[:len(best_moregroup)]) < 0)
        best_fills = np.sum(np.asarray(best_moregroup[:len(newmoregroup)]) < 0)
        takes = np.sum(np.asarray(newmoregroup) >= 0)
        best_takes = np.sum(np.asarray(best_moregroup) >= 0)

        if verbose > 1:
            print 'newgroup:      divisor={d}, fzero={fz:.2f}Hz, peaksum={ps}, fills={f}, takes={t}'.format(d=divisor,
                                                                                                            fz=fzero,
                                                                                                            ps=newmoregroup_peaksum,
                                                                                                            f=fills,
                                                                                                            t=takes), newgroup
            print 'newmoregroup:  divisor={d}, fzero={fz:.2f}Hz, peaksum={ps}, fills={f}, takes={t}'.format(d=divisor,
                                                                                                            fz=fzero,
                                                                                                            ps=newmoregroup_peaksum,
                                                                                                            f=fills,
                                                                                                            t=takes), newmoregroup
            if verbose > 2:
                print 'bestgroup:     divisor={d}, fzero={fz:.2f}Hz, peaksum={ps}, fills={f}, takes={t}'.format(
                    d=best_divisor, fz=best_fzero, ps=best_group_peaksum, f=best_fills, t=best_takes), best_group

        # TAKE THE NEW GROUP IF BETTER:
        # sum of peak power must be larger and
        # less fills. But if the new group has more takes,
        # this might compensate for more fills.
        if newmoregroup_peaksum > best_group_peaksum \
                and fills - best_fills <= 0.5 * (takes - best_takes):

            best_group_peaksum = newmoregroup_peaksum
            if len(newgroup) == 1:
                best_group_fill_ins = np.max((2, fill_ins))  # give larger groups a chance XXX we might reduce this!
            else:
                best_group_fill_ins = fill_ins
            best_group = newgroup
            best_moregroup = newmoregroup
            best_divisor = divisor
            best_fzero = fzero
            best_fzero_harmonics = fzero_harmonics

            if verbose > 2:
                print 'new bestgroup:     divisor={d}, fzero={fz:.2f}Hz, peaksum={ps}, fills={f}, takes={t}'.format(
                    d=best_divisor, fz=best_fzero, ps=best_group_peaksum, f=best_fills, t=best_takes), best_group
                print 'new bestmoregroup: divisor={d}, fzero={fz:.2f}Hz, peaksum={ps}, fills={f}, takes={t}'.format(
                    d=best_divisor, fz=best_fzero, ps=best_group_peaksum, f=best_fills, t=best_takes), best_moregroup
            elif verbose > 1:
                print 'took as new best group'

    # ##############################################################

    # no group found:
    if len(best_group) == 0:
        # erase fmax:
        freqs = np.delete(freqs, fmaxinx, axis=0)
        group = np.zeros((0, 5))
        return freqs, more_freqs, group, 1, fmax

    # group found:
    if verbose > 2:
        print
        print '## best groups found for fmax={fm:.2f}Hz: fzero={fz:.2f}Hz, d={d:d}:'.format(fm=fmax, fz=best_fzero,
                                                                                            d=best_divisor)
        print '## bestgroup:     ', best_group, '[', ', '.join(['{:.2f}'.format(f) for f in freqs[best_group, 0]]), ']'
        print '## bestmoregroup: ', best_moregroup, '[', ', '.join(
            ['{:.2f}'.format(f) for f in more_freqs[best_moregroup, 0]]), ']'

    # fill up group:
    group = np.zeros((len(best_moregroup), 5))
    for i, inx in enumerate(best_moregroup):
        # increment double use counter:
        more_freqs[inx, 4] += 1.0
        if inx >= 0:
            group[i, :] = more_freqs[inx, :]
        # take adjusted peak frequencies:
        group[i, 0] = (i + 1) * best_fzero

    if verbose > 1:
        refi = np.nonzero(group[:, 1] > 0.0)[0][0]
        print
        print '# resulting harmonic group for fmax=', fmax
        for i in xrange(group.shape[0]):
            print '{0:8.2f}Hz n={1:5.2f}: p={2:10.3g} p/p0={3:10.3g}'.format(group[i, 0], group[i, 0] / group[0, 0],
                                                                             group[i, 1], group[i, 1] / group[refi, 1])

    # erase from freqs:
    for inx in reversed(best_group):
        freqs = np.delete(freqs, inx, axis=0)

    # freqs: removed all frequencies of bestgroup
    # more_freqs: updated double use count
    # group: the group
    # fmax: fmax
    return freqs, more_freqs, group, best_fzero_harmonics, fmax


def extract_fundamentals(good_freqs, all_freqs, deltaf, cfg):
    """
    Extract fundamental frequencies from power-spectrum peaks.

    Returns:
        group_list (list): list of all harmonic groups found
        fzero_harmonics_list (list): the harmonics from which the fundamental frequencies were computed
        mains_list (2-d array): list of mains peaks found
    """

    verbose = cfg['verboseLevel'][0]
    if verbose > 0:
        print

    # set double use count to zero:
    all_freqs[:, 4] = 0.0

    freqtol = cfg['freqTolerance'][0] * deltaf
    mainsfreq = cfg['mainsFreq'][0]

    # remove power line harmonics from good_freqs:
    # XXX might be improved!!!
    if mainsfreq > 0.0:
        pfreqtol = 1.0  # 1 Hz tolerance
        for inx in reversed(xrange(len(good_freqs))):
            n = np.round(good_freqs[inx, 0] / mainsfreq)
            nd = np.abs(good_freqs[inx, 0] - n * mainsfreq)
            if nd <= pfreqtol:
                if verbose > 1:
                    print 'remove power line frequency', inx, good_freqs[inx, 0], np.abs(
                        good_freqs[inx, 0] - n * mainsfreq)
                good_freqs = np.delete(good_freqs, inx, axis=0)

    group_list = list()
    fzero_harmonics_list = list()
    # as long as there are frequencies left in good_freqs:
    while good_freqs.shape[0] > 0:
        # we check for harmonic groups:
        good_freqs, all_freqs, harm_group, fzero_harmonics, fmax = \
            build_harmonic_groups(good_freqs, all_freqs, deltaf, cfg)

        if verbose > 1:
            print

        # nothing found:
        if harm_group.shape[0] == 0:
            if verbose > 0:
                print 'Nothing found for fmax=%.2fHz' % fmax
            continue

        # count number of harmonics which have been detected, are not fill-ins,
        # and are not doubly used:
        group_size = np.sum((harm_group[:, 1] > 0.0) & (harm_group[:, 4] < 2.0))
        group_size_ok = (group_size >= cfg['minimumGroupSize'][0])

        # check frequency range of fundamental:
        fundamental_ok = (harm_group[0, 0] >= cfg['minimumFrequency'][0] and
                          harm_group[0, 0] <= cfg['maximumFrequency'][0])

        # check power hum (does this really ever happen???):
        mains_ok = ((mainsfreq == 0.0) |
                    (np.abs(harm_group[0, 0] - mainsfreq) > freqtol))

        # check:
        if group_size_ok and fundamental_ok and mains_ok:
            if verbose > 0:
                print 'Accepting harmonic group: {:.2f}Hz p={:10.8f}'.format(
                    harm_group[0, 0], np.sum(harm_group[:, 1]))

            group_list.append(harm_group[:, 0:2])
            fzero_harmonics_list.append(fzero_harmonics)
        else:
            if verbose > 0:
                print 'Discarded harmonic group: {:.2f}Hz p={:10.8f} g={:d} f={:} m={:}'.format(
                    harm_group[0, 0], np.sum(harm_group[:, 1]),
                    group_size, fundamental_ok, mains_ok)

    # sort groups by fundamental frequency:
    ffreqs = [f[0, 0] for f in group_list]
    finx = np.argsort(ffreqs)
    group_list = [group_list[fi] for fi in finx]
    fzero_harmonics_list = [fzero_harmonics_list[fi] for fi in finx]

    # do not save more than n harmonics:
    maxharmonics = cfg['maxHarmonics'][0]
    if maxharmonics > 0:
        for group in group_list:
            if group.shape[0] > maxharmonics:
                if verbose > 0:
                    print 'Discarding some tailing harmonics'
                group = group[:maxharmonics, :]

    if verbose > 0:
        print
        if len(group_list) > 0:
            print '## FUNDAMENTALS FOUND: ##'
            for i in xrange(len(group_list)):
                power = group_list[i][:, 1]
                print '{:8.2f}Hz: {:10.8f} {:3d} {:3d}'.format(group_list[i][0, 0], np.sum(power),
                                                               np.sum(power <= 0.0), fzero_harmonics_list[i])
        else:
            print '## NO FUNDAMENTALS FOUND ##'

    # assemble mains frequencies from all_freqs:
    mains_list = []
    if mainsfreq > 0.0:
        pfreqtol = 1.0
        for inx in xrange(len(all_freqs)):
            n = np.round(all_freqs[inx, 0] / mainsfreq)
            nd = np.abs(all_freqs[inx, 0] - n * mainsfreq)
            if nd <= pfreqtol:
                mains_list.append(all_freqs[inx])
    return group_list, fzero_harmonics_list, np.array(mains_list)


def threshold_estimate(data, noise_factor, peak_factor):
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
    hist, bins = np.histogram(data, 100, density=True)
    inx = hist > np.max(hist) / np.sqrt(np.e)
    lower = bins[inx][0]
    upper = bins[inx][-1]  # needs to return the next bin
    center = 0.5 * (lower + upper)
    noisestd = 0.5 * (upper - lower)

    # low threshold:
    lowthreshold = noise_factor * noisestd

    # high threshold:
    lowerth = center + 0.5 * lowthreshold
    cumhist = np.cumsum(hist) / np.sum(hist)
    upperpthresh = 0.95
    if bins[-2] >= lowerth:
        pthresh = cumhist[bins[:-1] >= lowerth][0]
        upperpthresh = pthresh + 0.95 * (1.0 - pthresh)
    upperbins = bins[cumhist > upperpthresh]
    if len(upperbins) > 0:
        upperth = upperbins[0]
    else:
        upperth = bins[-1]
    highthreshold = lowthreshold + peak_factor * noisestd
    if upperth > lowerth + 0.1 * noisestd:
        highthreshold = lowerth + peak_factor * (upperth - lowerth) + 0.5 * lowthreshold - center

    return lowthreshold, highthreshold, center


def accept_psd_peaks(freqs, data, peak_inx, index, trough_inx, min_inx, threshold, check_conditions):
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
    wthresh = data[trough_inx] + 0.75 * size
    width = 0.0
    for k in xrange(peak_inx, trough_inx, -1):
        if data[k] < wthresh:
            width = freqs[peak_inx] - freqs[k]
            break
    for k in xrange(peak_inx, index):
        if data[k] < wthresh:
            width += freqs[k] - freqs[peak_inx]
            break
    return [freqs[peak_inx], data[peak_inx], size, width, 0.0]


def harmonic_groups(psd_freqs, psd, cfg):
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

    if verbose > 0:
        print
        print 70 * '#'
        print '##### harmonic_groups', 48 * '#'

    # decibel power spectrum:
    log_psd = 10.0 * np.log10(psd)

    # thresholds:
    low_threshold = cfg['lowThreshold'][0]
    high_threshold = cfg['highThreshold'][0]
    center = np.NaN
    if cfg['lowThreshold'][0] <= 0.0 or cfg['highThreshold'][0] <= 0.0:
        n = len(log_psd)
        low_threshold, high_threshold, center = threshold_estimate(log_psd[2 * n / 3:n * 9 / 10],
                                                                   cfg['noiseFactor'][0],
                                                                   cfg['peakFactor'][0])
        if verbose > 1:
            print
            print 'low_threshold=', low_threshold, center + low_threshold
            print 'high_threshold=', high_threshold, center + high_threshold
            print 'center=', center

    # detect peaks in decibel power spectrum:
    all_freqs = detect_peaks(psd_freqs, log_psd, low_threshold, accept_psd_peaks)

    # select good peaks:
    wthresh = cfg['maxPeakWidthFac'][0] * (psd_freqs[1] - psd_freqs[0])
    if wthresh < cfg['minPeakWidth'][0]:
        wthresh = cfg['minPeakWidth'][0]
    freqs = all_freqs[(all_freqs[:, 2] > high_threshold) &
                      (all_freqs[:, 0] >= cfg['minimumFrequency'][0]) &
                      (all_freqs[:, 0] <= cfg['maximumWorkingFrequency'][0]) &
                      (all_freqs[:, 3] < wthresh), :]

    # convert peak sizes back to power:
    freqs[:, 1] = 10.0 ** (0.1 * freqs[:, 1])
    all_freqs[:, 1] = 10.0 ** (0.1 * all_freqs[:, 1])

    # detect harmonic groups:
    groups, fzero_harmonics, mains = extract_fundamentals(freqs, all_freqs, psd_freqs[1] - psd_freqs[0], cfg)

    return groups, fzero_harmonics, mains, all_freqs, freqs[:, 0], low_threshold, high_threshold, center


def filter_fishes(fishlists):
    ###################################################################
    # extract fishes which are consistent for different resolutions
    fundamentals = [[] for i in fishlists]
    for i in np.arange(len(fishlists)):
        for j in np.arange(len(fishlists[i])):
            fundamentals[i].append(fishlists[i][j][0][0])
    consistent_fish_help = [1 for k in fundamentals[0]]

    for i in np.arange(len(fundamentals[0])):
        for j in np.arange(len(fundamentals) - 1):
            for k in np.arange(len(fundamentals[j + 1])):
                if fundamentals[0][i] < fundamentals[j + 1][k] + 2 and fundamentals[0][i] > fundamentals[j + 1][k] - 2:
                    consistent_fish_help[i] += 1
    index_of_valid_fish = []
    for i in np.arange(len(consistent_fish_help)):
        if consistent_fish_help[i] == len(fishlists):
            index_of_valid_fish.append(i)

    fishlist = []
    for i in index_of_valid_fish:
        fishlist.append(fishlists[0][i])

    return fishlist


def wave_or_pulse_psd(power, freqs, data, rate, fresolution, create_dataset=False):
    if create_dataset is True:
        fig, ax = plt.subplots()
        plt.axis([0, 3000, -110, -30])
        ax.plot(freqs, 10.0 * np.log10(power))
    freq_steps = int(125 / fresolution)

    proportions = []
    mean_powers = []
    # embed()
    for i in np.arange((3000 / fresolution) // freq_steps):  # does all of this till the frequency of 3k Hz
        power_db = 10.0 * np.log10(power[i * freq_steps:i * freq_steps + freq_steps])
        power_db_p25_75 = []
        power_db_p99 = np.percentile(power_db, 99)
        power_db_p1 = np.percentile(power_db, 1)
        power_db_p25 = np.percentile(power_db, 25)
        power_db_p75 = np.percentile(power_db, 75)

        ### proportion psd (idea by Jan)
        proportions.append((power_db_p75 - power_db_p25) / (
            power_db_p99 - power_db_p1))  # value between 0 and 1; pulse psds have much bigger values than wave psds
        ###

        ### difference psd (idea by Till)
        for j in np.arange(len(power_db)):
            if power_db[j] > power_db_p25 and power_db[j] < power_db_p75:
                power_db_p25_75.append(power_db[j])
        mean_powers.append(np.mean(power_db_p25_75))

        if create_dataset is True:
            ax.plot(freqs[i * freq_steps + freq_steps / 2], np.mean(power_db_p25_75), 'o', color='r')
    mean_powers_p10 = np.percentile(mean_powers, 10)
    mean_powers_p90 = np.percentile(mean_powers, 90)
    diff = abs(mean_powers_p90 - mean_powers_p10)
    ###

    ### proportion sound trace
    rate = float(rate)
    time = np.arange(len(data)) / rate
    trace_proportions = []
    for i in np.arange(len(data) // (4.0 * rate)):
        temp_trace_data = data[i * rate / 4.0:i * rate / 4.0 + rate / 4.0]
        temp_trace_data_p1 = np.percentile(temp_trace_data, 1)
        temp_trace_data_p25 = np.percentile(temp_trace_data, 25)
        temp_trace_data_p75 = np.percentile(temp_trace_data, 75)
        temp_trace_data_p99 = np.percentile(temp_trace_data, 99)
        trace_proportions.append(
            (temp_trace_data_p75 - temp_trace_data_p25) / (temp_trace_data_p99 - temp_trace_data_p1))
    if np.mean(proportions) < 0.25:
        psd_type = 'wave'
    else:
        psd_type = 'pulse'

    # if diff < 15 and np.mean(proportions) < 0.25:
    #     psd_type = 'wave'
    # elif diff > 15 and np.mean(proportions) > 0.25:
    #     psd_type = 'pulse'
    # else:
    #     psd_type = 'unknown'

    #####################################################################################
    # Creating dataset #

    if create_dataset is True:
        plt.draw()
        plt.pause(1)

        response = raw_input('wave- or pulse psd ? [w/p]')
        plt.close()
        if response == "w":
            file = "wave_diffs.npy"
            file2 = "wave_proportions.npy"
            file3 = "wave_trace_proportions.npy"
        elif response == "p":
            file = "pulse_diffs.npy"
            file2 = "pulse_proportions.npy"
            file3 = "pulse_trace_proportions.npy"
        else:
            quit()

        if not os.path.exists(file):
            np.save(file, np.array([]))
        all_diffs = np.load(file)
        all_diffs = all_diffs.tolist()
        all_diffs.append(diff)
        all_diffs = np.asarray(all_diffs)
        np.save(file, all_diffs)

        if not os.path.exists(file2):
            np.save(file2, np.array([]))
        all_propertions = np.load(file2)
        all_propertions = all_propertions.tolist()
        all_propertions.append(np.mean(proportions))
        all_propertions = np.asarray(all_propertions)
        np.save(file2, all_propertions)

        if not os.path.exists(file3):
            np.save(file3, np.array([]))
        all_trace_proportions = np.load(file3)
        all_trace_proportions = all_trace_proportions.tolist()
        all_trace_proportions.append(np.mean(trace_proportions))
        all_trace_proportions = np.asarray(all_trace_proportions)
        np.save(file3, all_trace_proportions)
    return psd_type


def save_fundamentals(fishlist):
    mean_path = 'fish_wave.npy'
    if not os.path.exists(mean_path):
        np.save(mean_path, np.array([]))
    fundamentals = np.load(mean_path)
    fundamentals = fundamentals.tolist()

    for fish in np.arange(len(fishlist)):
        fundamentals.append(fishlist[fish][0][0])

    fundamentals = np.asarray(fundamentals)
    np.save(mean_path, fundamentals)

    print 'current fundamental frequencies are: ', fundamentals


class FishTracker:
    def __init__(self, samplingrate):
        self.rate = samplingrate
        self.tstart = 0
        self.fish_freqs_dict = {}
        self.datasize = 200.0  # seconds                               ## DATASIZE ##
        self.step = 0.5  ## STEP ##
        self.fresolution = 0.5
        self.twindow = 8.0
        self.fishes = {}

    def processdata(self, data, fish_type, bwin, win_width,
                    test_longfile=False):  # , rate, fish_freqs_dict, tstart, datasize, step ) :
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

        test_resolutions = [self.fresolution, self.fresolution * 2.0, self.fresolution * 4.0]
        all_nfft = []
        for i in np.arange(len(test_resolutions)):
            nfft = int(np.round(2 ** (np.floor(np.log(self.rate / test_resolutions[i]) / np.log(2.0)) + 1.0)))
            if nfft < 16:
                nfft = 16
            all_nfft.append(nfft)

        tw = int(np.round(self.twindow * self.rate))
        minw = all_nfft[0] * (cfg['minPSDAverages'][0] + 1) / 2
        if tw < minw:
            tw = minw

        self.datasize = len(data) / self.rate

        # running Power spectrum in best window (8s)
        fishlists_dif_fresolution = []  # gets fishlists from 3 different psd(powerspectrum)
        for i in np.arange(len(all_nfft)):
            power, freqs = ml.psd(data[(bwin * self.rate):(bwin * 1.0 * self.rate + win_width * 1.0 * self.rate)],
                                  NFFT=all_nfft[i], noverlap=all_nfft[i] / 2, Fs=self.rate,
                                  detrend=ml.detrend_mean)
            if i == 0:
                power_fres1 = power
                freqs_fres1 = freqs

            fishlist, _, mains, allpeaks, peaks, lowth, highth, center = harmonic_groups(freqs, power, cfg)

            # power to dB
            for fish in np.arange(len(fishlist)):
                for harmonic in np.arange(len(fishlist[fish])):
                    fishlist[fish][harmonic][-1] = 10.0 * np.log10(fishlist[fish][harmonic][-1])

            fishlists_dif_fresolution.append(fishlist)

        fishlist = filter_fishes(fishlists_dif_fresolution)
        # embed()
        # filter fishlist so only 3 fishes with max strength are involved

        psd_type = wave_or_pulse_psd(power_fres1, freqs_fres1,
                                     data[(bwin * self.rate):(bwin * self.rate + win_width * self.rate)], self.rate,
                                     self.fresolution)

        wave_ls = [fishlist[fish][0][0] for fish in np.arange(len(fishlist))]

        cfg['lowThreshold'][0] = lowth
        cfg['highThreshold'][0] = highth

        temp_dict = {bwin: wave_ls}

        self.fish_freqs_dict.update(temp_dict)

        # Pulsefish processing
        # if psd_type is 'pulse' and fish_type is 'pulse':
        #     print 'HERE WE HAVE TO BUILD IN THE PULSEFISH ANALYSIS'
        # # embed()
        return power_fres1, freqs_fres1, psd_type, fish_type, fishlist

    def sort_my_wavefish(self):
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
                    temp_fish = {len(self.fishes) + 1: [self.fish_freqs_dict[t][i]]}
                    self.fishes.update(temp_fish)
            else:
                for i in np.arange(len(self.fish_freqs_dict[t])):
                    print 'Phase II; ', 'point in time is: ', t, 'secondes.', 'wavefish no.:', i
                    help_v = 0
                    new_freq = self.fish_freqs_dict[t][i]

                    for j in self.fishes.keys():
                        for p in np.arange(len(self.fishes[j])) + 1:
                            if self.fishes[j][-p] is not np.nan:
                                index_last_nan = -p
                                break
                        if new_freq > self.fishes[j][index_last_nan] - 0.5 and new_freq <= self.fishes[j][
                            index_last_nan] + 0.5 and help_v == 0:
                            self.fishes[j].append(new_freq)
                            help_v += 1

                    if help_v is 0:
                        temp_fish = {len(self.fishes) + 1: []}
                        for l in np.arange(k):
                            temp_fish[len(self.fishes) + 1].append(np.NaN)
                        temp_fish[len(self.fishes) + 1].append(self.fish_freqs_dict[t][i])
                        self.fishes.update(temp_fish)
                    elif help_v >= 2:
                        print "added frequency to more than one fish. reduce tolerance!!!"
                        break
                for m in self.fishes.keys():
                    if len(self.fishes[m]) < k + 1:
                        self.fishes[m].append(np.nan)

    def get_data(self):
        data = np.zeros(np.ceil(self.rate * self.datasize), dtype="<i2")
        return data

    def mean_multi_wavefish(self):
        """
        if three arguments are given:
        loads/build and npy file witch contains the main frequencies of the wave fishes of one recording.
        saves the file as .npy

        3. arg is a str

        """
        mean_path = 'fish_wave.npy'
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

    def exclude_short_files(self, data, index):

        if len(data[:index]) / self.rate <= 10.:
            good_file = False
        else:
            good_file = True
        return good_file

    def bw_psd_and_eod_plot(self, power, freqs, bwin, win_width, data, psd_type, fish_type, fishlist, pulse_data,
                            pulse_freq):
        '''
        create figures showing the best window, its PSD and the the EOD of the fish
        '''

        ### PSD of the best window up to 3kHz #########################################################################
        if len(fishlist) > 4:
            ind = np.argsort([fishlist[fish][0][1] for fish in np.arange(len(fishlist))])[-4:]
            # print ind
        else:
            ind = np.argsort([fishlist[fish][0][1] for fish in np.arange(len(fishlist))])
            # print ind

        # start plot cosmetic with seaborn
        sns.set_context("poster")
        sns.axes_style('white')
        sns.set_style("ticks")

        plot_w = 26.
        plot_h = 15.
        fs = 16  # fontsize of the axis labels
        inch_factor = 2.54

        fig, ax = plt.subplots(figsize=(plot_w / inch_factor, plot_h / inch_factor))
        fig_all = plt.figure(figsize=(plot_w * 2 / inch_factor, plot_h * 2 / inch_factor))
        ax1_all = fig_all.add_subplot(2, 2, (3, 4))

        ax.axis([0, 3000, -110, 0])
        ax1_all.axis([0, 3000, -110, 0])

        ax.plot(freqs, 10.0 * np.log10(power), lw=2, color='dodgerblue', alpha=0.7)
        ax1_all.plot(freqs, 10.0 * np.log10(power), lw=2, color='dodgerblue', alpha=0.7)

        color = ['red', 'blue', 'green', 'cornflowerblue']
        for color_no, fish in enumerate(ind):
            for harmonic in np.arange(len(fishlist[fish])):
                if harmonic == 0:
                    ax.plot(fishlist[fish][harmonic][0], fishlist[fish][harmonic][1], 'o', color=color[color_no],
                            label=('%.2f Hz' % fishlist[fish][0][0]))
                    ax1_all.plot(fishlist[fish][harmonic][0], fishlist[fish][harmonic][1], 'o', color=color[color_no],
                                 label=('%.2f Hz' % fishlist[fish][0][0]))
                else:
                    ax.plot(fishlist[fish][harmonic][0], fishlist[fish][harmonic][1], 'o', color=color[color_no])
                    ax1_all.plot(fishlist[fish][harmonic][0], fishlist[fish][harmonic][1], 'o', color=color[color_no])

        ax.tick_params(axis='both', which='major', labelsize=fs - 2)
        ax1_all.tick_params(axis='both', which='major', labelsize=fs - 2)

        plt.legend(loc='upper right')

        ax.set_xlabel('Frequency [Hz]', fontsize=fs)
        ax1_all.set_xlabel('Frequency [Hz]', fontsize=fs)

        ax.set_ylabel('Power [dB SPL]', fontsize=fs)
        ax1_all.set_ylabel('Power [dB SPL]', fontsize=fs)

        ax.set_title('PSD of best window', fontsize=fs + 2)
        ax1_all.set_title('PSD of best window', fontsize=fs + 2)

        sns.despine(fig=fig, ax=ax, offset=10)
        sns.despine(fig=fig_all, ax=ax1_all, offset=10)

        fig.tight_layout()

        if not os.path.exists('./figures'):
            os.makedirs('./figures')

        fig.savefig('figures/PSD_best_window%.0f.pdf' % (len(glob.glob('figures/PSD_best_window*.pdf')) + 1))
        # variable name for "looping with several sound datas"-case
        plt.close(fig)

        ax2_all = fig_all.add_subplot(2, 2, 2)

        if psd_type is 'wave' or fish_type is 'wave':
            fig3, ax3 = plt.subplots(figsize=(plot_w / inch_factor, plot_h / inch_factor))

            eod_wtimes = np.arange(
                len(data[(bwin * self.rate):(bwin * self.rate + round(self.rate * 1.0 / fishlist[ind[-1]][0][0] * 4))])
                ) * 1.0 / self.rate + bwin
            eod_wampls = data[
                         (bwin * self.rate):(bwin * self.rate + round(self.rate * 1.0 / fishlist[ind[-1]][0][0] * 4))]
            ax3.plot(eod_wtimes, eod_wampls, lw=2, color='dodgerblue', alpha=0.7,
                     label='Dominant frequency: %.2f Hz' % fishlist[ind[-1]][0][0])
            ax3.tick_params(axis='both', which='major', labelsize=fs - 2)
            plt.legend(loc='upper right')
            ax3.set_xlabel('Time [sec]', fontsize=fs)
            ax3.set_ylabel('Amplitude [a.u.]', fontsize=fs)
            ax3.set_title('EOD-Waveform; %s' % sys.argv[1].split('/')[-1], fontsize=fs + 2)
            sns.despine(fig=fig3, ax=ax3, offset=10)
            fig3.tight_layout()
            fig3.savefig('figures/wave-EOD%.0f.pdf' % (len(glob.glob('figures/wave-EOD*.pdf')) + 1))
            plt.close(fig3)

            if fish_type is not 'pulse' or psd_type is not 'pulse':
                # if pulse_data == []:
                ax2_all.plot(eod_wtimes, eod_wampls, lw=2, color='dodgerblue', alpha=0.7)
                ax2_all.tick_params(axis='both', which='major', labelsize=fs - 2)
                # plt.legend(loc= 'upper right')
                ax2_all.set_xlabel('Time [sec]', fontsize=fs)
                ax2_all.set_ylabel('Amplitude [a.u.]', fontsize=fs)
                ax2_all.set_title('EOD-Waveform', fontsize=fs + 2)
                sns.despine(fig=fig_all, ax=ax2_all, offset=10)

        # Soundtrace for a pulsefish-EOD ############################################################################
        # build mean and std over this data !
        if psd_type is 'pulse' or fish_type is 'pulse':
            fig4, ax4 = plt.subplots(figsize=(plot_w / inch_factor, plot_h / inch_factor))

            eod_plot_tw = 0.006
            mean_pulse_data = []
            std_pulse_data = []

            for k in np.arange(len(pulse_data[1])):

                try:
                    tmp_mu = np.mean([pulse_data[pulse][k] for pulse in sorted(pulse_data.keys())])
                    tmp_std = np.std([pulse_data[pulse][k] for pulse in sorted(pulse_data.keys())], ddof=1)
                except IndexError:
                    print('Warning! Something seems odd when calculating sound-trace average. '
                          'Proceeding with fingers crossed...\n')
                    continue

                mean_pulse_data.append(tmp_mu)
                std_pulse_data.append(tmp_std)

            up_std = [mean_pulse_data[i] + std_pulse_data[i] for i in range(len(mean_pulse_data))]
            bottom_std = [mean_pulse_data[i] - std_pulse_data[i] for i in range(len(mean_pulse_data))]

            # get time for plot
            plot_time = ((np.arange(len(mean_pulse_data)) * 1.0 / self.rate) - eod_plot_tw / 2) * 1000  # s to ms

            ax4.plot(plot_time, mean_pulse_data, lw=2, color='dodgerblue', alpha=0.7, label='mean EOD')
            ax4.plot(plot_time, up_std, lw=1, color='red', alpha=0.7, label='std EOD')
            ax4.plot(plot_time, bottom_std, lw=1, color='red', alpha=0.7)
            ax4.tick_params(axis='both', which='major', labelsize=fs - 2)
            ax4.set_xlabel('Time [ms]', fontsize=fs)
            ax4.set_ylabel('Amplitude [a.u.]', fontsize=fs)
            ax4.set_title('Mean pulse-EOD; %s' % sys.argv[1].split('/')[-1], fontsize=fs + 2)
            plt.legend(loc='upper right', fontsize=fs - 4)
            sns.despine(fig=fig4, ax=ax4, offset=10)

            fig4.tight_layout()
            fig4.savefig('figures/pulse-EOD%.0f.pdf' % (len(glob.glob('figures/pulse-EOD*.pdf')) + 1))
            # fig4.savefig('figures/pulse-EOD%.0f.pdf' % (len(glob.glob('figures/pulse-EOD*.pdf'))
            #                                             + len(glob.glob('figures/EOD*.pdf'))+1))
            plt.close(fig4)

            if fish_type is 'pulse' and psd_type is 'pulse':
                ax2_all.plot(plot_time, mean_pulse_data, lw=2, color='dodgerblue', alpha=0.7, label='mean EOD')
                ax2_all.plot(plot_time, up_std, lw=1, color='red', alpha=0.7, label='std EOD')
                ax2_all.plot(plot_time, bottom_std, lw=1, color='red', alpha=0.7)
                ax2_all.tick_params(axis='both', which='major', labelsize=fs - 2)
                ax2_all.set_xlabel('Time [ms]', fontsize=fs)
                ax2_all.set_ylabel('Amplitude [a.u.]', fontsize=fs)
                ax2_all.set_title('Mean pulse-EOD', fontsize=fs + 2)
                sns.despine(fig=fig_all, ax=ax2_all, offset=10)

        text_ax = fig_all.add_subplot(2, 2, 1)
        text_ax.text(-0.1, 0.9, 'Filename:', fontsize=fs)
        text_ax.text(0.5, 0.9, '%s' % sys.argv[1].split('/')[-1], fontsize=fs)

        text_ax.text(-0.1, 0.8, 'File duration:', fontsize=fs)
        text_ax.text(0.5, 0.8, '%.2f s' % (len(data) / self.rate), fontsize=fs)

        text_ax.text(-0.1, 0.7, 'Best window:', fontsize=fs)
        text_ax.text(0.5, 0.7, '%.2f s - %.2f s' % (bwin, bwin + win_width), fontsize=fs)

        text_ax.text(-0.1, 0.6, 'Shown EODf:', fontsize=fs)
        text_ax.text(-0.1, 0.5, 'Fish-Type:', fontsize=fs)

        if fish_type is not 'pulse' or psd_type is not 'pulse':
            # if pulse_data != []:
            text_ax.text(0.5, 0.6, '%.2f Hz' % fishlist[ind[-1]][0][0], fontsize=fs)
            # text.text(0.5, 0.5, 'wave-type', fontsize= fs)
        else:
            text_ax.text(0.5, 0.6, '%.2f Hz' % pulse_freq, fontsize=fs)
            # text.text(0.5, 0.5, 'pulse-type', fontsize= fs)
        text_ax.text(0.5, 0.5, '%s' % fish_type, fontsize=fs)

        text_ax.text(-0.1, 0.4, 'PSD-Type:', fontsize=fs)
        text_ax.text(0.5, 0.4, '%s' % psd_type, fontsize=fs)

        text_ax.text(-0.1, 0.3, 'No. of wave-fishes:', fontsize=fs)
        text_ax.text(0.5, 0.3, '%.0f' % len(fishlist), fontsize=fs)

        sns.despine(fig=fig_all, ax=text_ax, offset=10)
        plt.axis('off')

        fig_all.tight_layout()
        fig_all.savefig('figures/%s.pdf' % sys.argv[1].split('/')[-1][:-4])
        plt.close(fig_all)

    def pulse_sorting(self, bwin, win_width, data):
        # load data and time (0-8s) of bestwindow
        bw_data = data[(bwin * self.rate):(bwin * self.rate + win_width * self.rate)]
        time = np.arange(len(bw_data)) * 1.0 / self.rate

        # get time of data exceeding the threshold
        threshold = max(bw_data) - ((max(bw_data) - np.mean(bw_data)) / 2)

        th_time = []
        for i in np.arange(len(bw_data) - 1):
            if bw_data[i + 1] > threshold and bw_data[i] <= threshold:
                th_time.append(time[i + 1])

        # pulse frequency   PROBLEMATISCH BEI MEHREREN PULSEFISHEN !!!
        pulse_freq = len(th_time) / win_width
        print ''
        print 'Pulse-frequency:', pulse_freq
        print ''

        pulse_frequencies = 'fish_pulse.npy'
        if not os.path.exists(pulse_frequencies):
            np.save(pulse_frequencies, np.array([]))
        pulse_ls = np.load(pulse_frequencies)
        pulse_ls = pulse_ls.tolist()

        pulse_ls.append(pulse_freq)

        pulse_ls = np.asarray(pulse_ls)
        np.save(pulse_frequencies, pulse_ls)

        # for each detected pulse/exceeding-time (count in pulse_data.keys())
        # save a window of the data arround this time
        pulse_data = {}
        eod_plot_tw = 0.006  # seconds shown in plot
        for i in np.arange(len(th_time)):
            plot_data = bw_data[(th_time[i] - eod_plot_tw / 2) * self.rate: (th_time[
                                                                                 i] - eod_plot_tw / 2) * self.rate + eod_plot_tw * self.rate]
            temp_dict = {len(pulse_data) + 1: plot_data}
            pulse_data.update(temp_dict)

        return pulse_data, pulse_freq


def main():
    # config file name:
    progs = sys.argv[0].split('/')
    cfgfile = progs[-1].split('.')[0] + '.cfg'

    # command line arguments:
    parser = argparse.ArgumentParser(
        description='Display waveform, spectrogram, and power spectrum of time series data.',
        epilog='by Jan Benda (2015)')
    parser.add_argument('--version', action='version', version='1.0')
    parser.add_argument('-v', action='count', dest='verbose')
    parser.add_argument('file', nargs='?', default='', type=str, help='name of the file wih the time series data')
    parser.add_argument('npy_savefile_wave', nargs='?', default='', type=str)
    parser.add_argument('npy_savefile_puls', nargs='?', default='', type=str)
    parser.add_argument('channel', nargs='?', default=0, type=int, help='channel to be displayed')
    args = parser.parse_args()

    # load configuration from the current directory:
    if os.path.isfile(cfgfile):
        print('load configuration ' + cfgfile)
        load_config(cfgfile, cfg)

    # load configuration files from higher directories:
    filepath = args.file
    absfilepath = os.path.abspath(filepath)
    dirs = os.path.dirname(absfilepath).split('/')
    dirs.append('')
    maxlevel = len(dirs) - 1
    if maxlevel > 3:
        maxlevel = 3
    for k in xrange(maxlevel, 0, -1):
        path = '/'.join(dirs[:-k]) + '/' + cfgfile
        if os.path.isfile(path):
            print 'load configuration', path
            load_config(path, cfg)

    # set configuration from command line:
    if args.verbose != None:
        cfg['verboseLevel'][0] = args.verbose

    channel = args.channel

    try:
        import audioread
    except ImportError:
        print 'python module "audioread" is not installed.'
        exit(2)

    with audioread.audio_open(filepath) as af:
        tracen = af.channels
        if channel >= tracen:
            print 'number of traces in file is', tracen
            quit()
        ft = FishTracker(af.samplerate)
        index = 0

        data = ft.get_data()

        for buffer in af:
            fulldata = np.fromstring(buffer, dtype='<i2').reshape(-1, af.channels)
            n = fulldata.shape[0]
            if index + n > len(data):
                if index == 0:
                    print "panic!!!! I need a larger buffer!"
                # ft.processdata(data[:index] / 2.0 ** 15)
                index = 0
            if n > 0:
                data[index:index + n] = fulldata[:n, channel]
                index += n
            else:
                break

        # long file analysis
        good_file = ft.exclude_short_files(data, index)
        if good_file == False:
            print "file too short !!!"
            exit()

        # best window algorithm
        mod_file = conv_to_single_ch_audio(filepath)

        Fish = FishRecording(mod_file)
        bwin, win_width = Fish.detect_best_window()

        print '\nbest window is between: %.2f' % bwin, '& %.2f' % (bwin + win_width), 'seconds.\n'

        os.remove(mod_file)

        # fish_type algorithm
        fish_type = Fish.type_detector()
        print('current fish is a ' + fish_type + '-fish')

        # data process: creation of fish-lists containing frequencies, power of fundamentals and harmonics of all fish

        if index > 0:
            power_fres1, freqs_fres1, psd_type, fish_type, fishlist = ft.processdata(data[:index] / 2.0 ** 15,
                                                                                     fish_type, bwin, win_width)

        # Pulse analysis
        pulse_data = []
        pulse_freq = []
        if psd_type == 'pulse' or fish_type == 'pulse':
            print ''
            print 'try to create MEAN PULSE-EOD'
            print ''
            pulse_data, pulse_freq = ft.pulse_sorting(bwin, win_width, data[:index] / 2.0 ** 15)

        # create EOD plots
        # embed()
        ft.bw_psd_and_eod_plot(power_fres1, freqs_fres1, bwin, win_width, data[:index] / 2.0 ** 15, psd_type, fish_type,
                               fishlist, pulse_data, pulse_freq)

        # saves fundamentals of all wave fish !!!
        save_fundamentals(fishlist)


if __name__ == '__main__':
    main()
