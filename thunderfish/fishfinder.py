#!/usr/bin/python

import sys
import os
import warnings
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as ml
import matplotlib.colors as mc
from collections import OrderedDict
import configfile as cf
import dataloader as dl
import harmonicgroups as hg
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
        self.min_clip = cfg['minClipAmplitude'][0]
        self.max_clip = cfg['maxClipAmplitude'][0]
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
        self.fishlist, fzero_harmonics, self.mains, self.allpeaks, peaks, lowth, highth, center = hg.harmonic_groups(freqs, power, cfg)
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
            if self.min_clip == 0.0 or self.max_clip == 0.0 :
                clip_win_inx = int(cfg['clipWindow'][0]*self.rate)
                min_clip_fac = cfg['minClipFactor'][0]
                clip_bins = cfg['clipBins'][0]
                self.min_clip, self.max_clip = bw.clip_amplitudes(
                    self.data, clip_win_inx,
                    min_fac=min_clip_fac, nbins=clip_bins)
            idx0, idx1, clipped = bw.best_window_indices(
                self.data, self.rate, single=cfg['singleBestWindow'][0],
                win_size=cfg['bestWindowSize'][0],
                win_shift=cfg['bestWindowShift'][0],
                thresh_ampl_fac=cfg['bestWindowThresholdFactor'][0],
                min_clip=self.min_clip, max_clip=self.max_clip,
                w_cv_interv=cfg['weightCVInterval'][0],
                w_ampl=cfg['weightAmplitude'][0],
                w_cv_ampl=cfg['weightCVAmplitude'][0],
                tolerance=cfg['bestWindowTolerance'][0])
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

                            
def short_user_warning(message, category, filename, lineno, file=sys.stderr, line=''):
    if category == UserWarning :
        file.write('%s line %d: %s\n' % ('/'.join(filename.split('/')[-2:]), lineno, message))
    else :
        s = warnings.formatwarning(message, category, filename, lineno, line)
        file.write(s)


if __name__ == '__main__':
    warnings.showwarning = short_user_warning

    bw.add_clip_config(cfg, cfgsec)
    bw.add_best_window_config(cfg, cfgsec)
    cfg['bestWindowSize'][0] = 4.0
    
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

    # set verbosity level from command line:
    if args.verbose != None :
        cfg['verboseLevel'][0] = args.verbose
    
    # load configuration from working directory and data directories:
    filepath = args.file
    cf.load_config_files(cfgfile, filepath, cfg, 3, cfg['verboseLevel'][0])

    # set verbosity level from command line (it migh have been overwritten):
    if args.verbose != None :
        cfg['verboseLevel'][0] = args.verbose
    if cfg['verboseLevel'][0] == 0 :
        warnings.filterwarnings("ignore")
    
    # save configuration:
    if len( args.save_config ) > 0 :
        ext = args.save_config.split('.')[-1]
        if ext != 'cfg' :
            print('configuration file name must have .cfg as extension!')
        else :
            print('write configuration to %s ...' % args.save_config)
            cf.dump_config(args.save_config, cfg, cfgsec)
        quit()

    # load data:
    channel = args.channel
    filename = os.path.basename(filepath)
    data, freq, unit = dl.load_data(filepath, channel, cfg['verboseLevel'][0])

    # plot:
    sp = SignalPlot(freq, data, unit, filename, channel)


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
