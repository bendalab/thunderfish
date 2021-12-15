import sys
import os
import warnings
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as ml
from audioio.playaudio import PlayAudio, fade
from audioio.audiowriter import write_audio
from .version import __version__
from .configfile import ConfigFile
from .dataloader import open_data
from .powerspectrum import nfft, decibel, psd, spectrogram
from .powerspectrum import add_multi_psd_config, multi_psd_args
from .harmonics import harmonic_groups, harmonic_groups_args, psd_peak_detection_args
from .harmonics import add_psd_peak_detection_config, add_harmonic_groups_config, colors_markers
from .bestwindow import clip_amplitudes, clip_args, best_window_indices
from .bestwindow import add_clip_config, add_best_window_config, best_window_args
from .thunderfish import configuration, save_configuration
# check: import logging https://docs.python.org/2/howto/logging.html#logging-basic-tutorial


class SignalPlot:
    def __init__(self, data, samplingrate, unit, filename, channel, verbose, cfg):
        self.filename = filename
        self.channel = channel
        self.samplerate = samplingrate
        self.data = data
        self.unit = unit
        self.cfg = cfg
        self.verbose = verbose
        self.tmax = (len(self.data)-1)/self.samplerate
        self.toffset = 0.0
        self.twindow = 8.0
        if self.twindow > self.tmax:
            self.twindow = np.round(2 ** (np.floor(np.log(self.tmax) / np.log(2.0)) + 1.0))
        self.ymin = -1.0
        self.ymax = +1.0
        self.trace_artist = None
        self.spectrogram_artist = None
        self.fmin = 0.0
        self.fmax = 0.0
        self.decibel = True
        self.freq_resolution = self.cfg.value('frequencyResolution')
        self.deltaf = 1.0
        self.mains_freq = self.cfg.value('mainsFreq')
        self.power_label = None
        self.all_peaks_artis = None
        self.good_peaks_artist = None
        self.power_artist = None
        self.power_frequency_label = None
        self.peak_artists = []
        self.legend = True
        self.legendhandle = None
        self.help = False
        self.helptext = []
        self.allpeaks = []
        self.fishlist = []
        self.mains = []
        self.peak_specmarker = []
        self.peak_annotation = []
        self.min_clip = self.cfg.value('minClipAmplitude')
        self.max_clip = self.cfg.value('maxClipAmplitude')
        self.colorrange, self.markerrange = colors_markers()

        # audio output:
        self.audio = PlayAudio()
        
        # set key bindings:
        plt.rcParams['keymap.fullscreen'] = 'ctrl+f'
        plt.rcParams['keymap.pan'] = 'ctrl+m'
        plt.rcParams['keymap.quit'] = 'ctrl+w, alt+q, q'
        plt.rcParams['keymap.yscale'] = ''
        plt.rcParams['keymap.xscale'] = ''
        plt.rcParams['keymap.grid'] = ''
        plt.rcParams['keymap.all_axes'] = ''

        # the figure:
        plt.ioff()
        self.fig = plt.figure(figsize=(15, 9))
        self.fig.canvas.set_window_title(self.filename + ' channel {0:d}'.format(self.channel))
        self.fig.canvas.mpl_connect('key_press_event', self.keypress)
        self.fig.canvas.mpl_connect('button_press_event', self.buttonpress)
        self.fig.canvas.mpl_connect('pick_event', self.onpick)
        self.fig.canvas.mpl_connect('resize_event', self.resize)
        # trace plot:
        self.axt = self.fig.add_axes([0.1, 0.7, 0.87, 0.25])
        self.axt.set_ylabel('Amplitude [{:s}]'.format(self.unit))
        ht = self.axt.text(0.98, 0.05, '(ctrl+) page and arrow up, down, home, end: scroll', ha='right',
                           transform=self.axt.transAxes)
        self.helptext.append(ht)
        ht = self.axt.text(0.98, 0.15, '+, -, X, x: zoom in/out', ha='right', transform=self.axt.transAxes)
        self.helptext.append(ht)
        ht = self.axt.text(0.98, 0.25, 'y,Y,v,V: zoom amplitudes', ha='right', transform=self.axt.transAxes)
        self.helptext.append(ht)
        ht = self.axt.text(0.98, 0.35, 'p,P: play audio (display,all)', ha='right', transform=self.axt.transAxes)
        self.helptext.append(ht)
        ht = self.axt.text(0.98, 0.45, 'ctrl-f: full screen', ha='right', transform=self.axt.transAxes)
        self.helptext.append(ht)
        ht = self.axt.text(0.98, 0.55, 'w: plot waveform into png file', ha='right', transform=self.axt.transAxes)
        self.helptext.append(ht)
        ht = self.axt.text(0.98, 0.65, 's: save figure', ha='right', transform=self.axt.transAxes)
        self.helptext.append(ht)
        ht = self.axt.text(0.98, 0.75, 'S: save audiosegment', ha='right', transform=self.axt.transAxes)
        self.helptext.append(ht)
        ht = self.axt.text(0.98, 0.85, 'q: quit', ha='right', transform=self.axt.transAxes)
        self.helptext.append(ht)
        ht = self.axt.text(0.98, 0.95, 'h: toggle this help', ha='right', transform=self.axt.transAxes)
        self.helptext.append(ht)
        self.axt.set_xticklabels([])
        # spectrogram:
        self.axs = self.fig.add_axes([0.1, 0.45, 0.87, 0.25])
        self.axs.set_xlabel('Time [seconds]')
        self.axs.set_ylabel('Frequency [Hz]')
        # power spectrum:
        self.axp = self.fig.add_axes([0.1, 0.1, 0.87, 0.25])
        ht = self.axp.text(0.98, 0.9, 'r, R: frequency resolution', ha='right', transform=self.axp.transAxes)
        self.helptext.append(ht)
        ht = self.axp.text(0.98, 0.8, 'f, F: zoom', ha='right', transform=self.axp.transAxes)
        self.helptext.append(ht)
        ht = self.axp.text(0.98, 0.7, '(ctrl+) left, right: move', ha='right', transform=self.axp.transAxes)
        self.helptext.append(ht)
        ht = self.axp.text(0.98, 0.6, 'l: toggle legend', ha='right', transform=self.axp.transAxes)
        self.helptext.append(ht)
        ht = self.axp.text(0.98, 0.5, 'd: toggle decibel', ha='right', transform=self.axp.transAxes)
        self.helptext.append(ht)
        ht = self.axp.text(0.98, 0.4, 'm: toggle mains filter', ha='right', transform=self.axp.transAxes)
        self.helptext.append(ht)
        ht = self.axp.text(0.98, 0.3, 'left mouse: show peak properties', ha='right', transform=self.axp.transAxes)
        self.helptext.append(ht)
        ht = self.axp.text(0.98, 0.2, 'shift/ctrl + left/right mouse: goto previous/next harmonic', ha='right',
                           transform=self.axp.transAxes)
        self.helptext.append(ht)
        # plot:
        for ht in self.helptext:
            ht.set_visible(self.help)
        self.update_plots(False)
        plt.show()

    def __del(self):
        self.audio.close()

    def remove_peak_annotation(self):
        for fm in self.peak_specmarker:
            fm.remove()
        self.peak_specmarker = []
        for fa in self.peak_annotation:
            fa.remove()
        self.peak_annotation = []

    def annotate_peak(self, peak, harmonics=-1, inx=-1):
        # marker:
        if inx >= 0:
            m, = self.axs.plot([self.toffset + 0.01 * self.twindow], [peak[0]], linestyle='None',
                               color=self.colorrange[inx % len(self.colorrange)],
                               marker=self.markerrange[inx], ms=10.0, mec=None, mew=0.0, zorder=2)
        else:
            m, = self.axs.plot([self.toffset + 0.01 * self.twindow], [peak[0]], linestyle='None',
                               color='k', marker='o', ms=10.0, mec=None, mew=0.0, zorder=2)
        self.peak_specmarker.append(m)
        # annotation:
        fwidth = self.fmax - self.fmin
        pl = []
        pl.append(r'$f=$%.1f Hz' % peak[0])
        pl.append(r'$h=$%d' % harmonics)
        pl.append(r'$p=$%g' % peak[1])
        pl.append(r'$c=$%.0f' % peak[2])
        self.peak_annotation.append(self.axp.annotate('\n'.join(pl), xy=(peak[0], peak[1]),
                                                      xytext=(peak[0] + 0.03 * fwidth, peak[1]),
                                                      bbox=dict(boxstyle='round', facecolor='white'),
                                                      arrowprops=dict(arrowstyle='-')))

    def annotate_fish(self, fish, inx=-1):
        self.remove_peak_annotation()
        for harmonic, freq in enumerate(fish[:, 0]):
            peak = self.allpeaks[np.abs(self.allpeaks[:, 0] - freq) < 0.8 * self.deltaf, :]
            if len(peak) > 0:
                self.annotate_peak(peak[0, :], harmonic, inx)
        self.fig.canvas.draw()

    def update_plots(self, draw=True):
        self.remove_peak_annotation()
        # trace:
        self.axt.set_xlim(self.toffset, self.toffset + self.twindow)
        t0 = int(np.round(self.toffset * self.samplerate))
        t1 = int(np.round((self.toffset + self.twindow) * self.samplerate))
        if t1>len(self.data):
            t1 = len(self.data)
        time = np.arange(t0, t1) / self.samplerate
        if self.trace_artist == None:
            self.trace_artist, = self.axt.plot(time, self.data[t0:t1])
        else:
            self.trace_artist.set_data(time, self.data[t0:t1])
        self.axt.set_ylim(self.ymin, self.ymax)

        # compute power spectrum:
        n_fft = nfft(self.samplerate, self.freq_resolution)
        t00 = t0
        t11 = t1
        w = t11 - t00
        minw = n_fft * (self.cfg.value('minPSDAverages') + 1) // 2
        if t11 - t00 < minw:
            w = minw
            t11 = t00 + w
        if t11 >= len(self.data):
            t11 = len(self.data)
            t00 = t11 - w
        if t00 < 0:
            t00 = 0
            t11 = w
        freqs, power = psd(self.data[t00:t11], self.samplerate,
                           self.freq_resolution, detrend=ml.detrend_mean)
        self.deltaf = freqs[1] - freqs[0]
        # detect fish:
        h_kwargs = psd_peak_detection_args(self.cfg)
        h_kwargs.update(harmonic_groups_args(self.cfg))
        self.fishlist, fzero_harmonics, self.mains, self.allpeaks, peaks, lowth, highth, center = harmonic_groups(freqs, power, verbose=self.verbose, **h_kwargs)
        highth = center + highth - 0.5 * lowth
        lowth = center + 0.5 * lowth

        # spectrogram:
        t2 = t1 + n_fft
        specpower, freqs, bins = spectrogram(self.data[t0:t2], self.samplerate,
                                             self.freq_resolution,
                                             detrend=ml.detrend_mean)
        z = decibel(specpower)
        z = np.flipud(z)
        extent = self.toffset, self.toffset + np.amax(bins), freqs[0], freqs[-1]
        self.axs.set_xlim(self.toffset, self.toffset + self.twindow)
        if self.spectrogram_artist == None:
            self.fmax = np.round((freqs[-1] / 4.0) / 100.0) * 100.0
            min = highth
            min = np.percentile(z, 70.0)
            max = np.percentile(z, 99.9) + 30.0
            # cm = plt.get_cmap( 'hot_r' )
            cm = plt.get_cmap('jet')
            self.spectrogram_artist = self.axs.imshow(z, aspect='auto',
                                                      extent=extent, vmin=min, vmax=max,
                                                      cmap=cm, zorder=1)
        else:
            self.spectrogram_artist.set_data(z)
            self.spectrogram_artist.set_extent(extent)
        self.axs.set_ylim(self.fmin, self.fmax)

        # power spectrum:
        self.axp.set_xlim(self.fmin, self.fmax)
        if self.deltaf >= 1000.0:
            dfs = '%.3gkHz' % 0.001 * self.deltaf
        else:
            dfs = '%.3gHz' % self.deltaf
        tw = float(w) / self.samplerate
        if tw < 1.0:
            tws = '%.3gms' % (1000.0 * tw)
        else:
            tws = '%.3gs' % (tw)
        a = 2 * w // n_fft - 1  # number of ffts
        m = ''
        if self.cfg.value('mainsFreq') > 0.0:
            m = ', mains=%.0fHz' % self.cfg.value('mainsFreq')
        if self.power_frequency_label == None:
            self.power_frequency_label = self.axp.set_xlabel(
                r'Frequency [Hz] (nfft={:d}, $\Delta f$={:s}: T={:s}/{:d}{:s})'.format(n_fft, dfs, tws, a, m))
        else:
            self.power_frequency_label.set_text(
                r'Frequency [Hz] (nfft={:d}, $\Delta f$={:s}: T={:s}/{:d}{:s})'.format(n_fft, dfs, tws, a, m))
        self.axp.set_xlim(self.fmin, self.fmax)
        if self.power_label == None:
            self.power_label = self.axp.set_ylabel('Power')
        if self.decibel:
            if len(self.allpeaks) > 0:
                self.allpeaks[:, 1] = decibel(self.allpeaks[:, 1])
            power = decibel(power)
            pmin = np.min(power[freqs < self.fmax])
            pmin = np.floor(pmin / 10.0) * 10.0
            pmax = np.max(power[freqs < self.fmax])
            pmax = np.ceil(pmax / 10.0) * 10.0
            doty = pmax - 5.0
            self.power_label.set_text('Power [dB]')
            self.axp.set_ylim(pmin, pmax)
        else:
            pmax = np.max(power[freqs < self.fmax])
            doty = pmax
            pmax *= 1.1
            self.power_label.set_text('Power')
            self.axp.set_ylim(0.0, pmax)
        if self.all_peaks_artis == None:
            self.all_peaks_artis, = self.axp.plot(self.allpeaks[:, 0],
                                                  np.zeros(len(self.allpeaks[:, 0])) + doty,
                                                  'o', color='#ffffff')
            self.good_peaks_artist, = self.axp.plot(peaks, np.zeros(len(peaks)) + doty,
                                                    'o', color='#888888')
        else:
            self.all_peaks_artis.set_data(self.allpeaks[:, 0],
                                          np.zeros(len(self.allpeaks[:, 0])) + doty)
            self.good_peaks_artist.set_data(peaks, np.zeros(len(peaks)) + doty)
        labels = []
        fsizes = [np.sqrt(np.sum(self.fishlist[k][:, 1])) for k in range(len(self.fishlist))]
        fmaxsize = np.max(fsizes) if len(fsizes) > 0 else 1.0
        for k in range(len(self.peak_artists)):
            self.peak_artists[k].remove()
        self.peak_artists = []
        for k in range(len(self.fishlist)):
            if k >= len(self.markerrange):
                break
            fpeaks = self.fishlist[k][:, 0]
            fpeakinx = [int(np.round(fp / self.deltaf)) for fp in fpeaks if fp < freqs[-1]]
            fsize = 7.0 + 10.0 * (fsizes[k] / fmaxsize) ** 0.5
            fishpoints, = self.axp.plot(fpeaks[:len(fpeakinx)], power[fpeakinx], linestyle='None',
                                        color=self.colorrange[k % len(self.colorrange)],
                                        marker=self.markerrange[k], ms=fsize,
                                        mec=None, mew=0.0, zorder=1)
            self.peak_artists.append(fishpoints)
            if self.deltaf < 0.1:
                labels.append('%4.2f Hz' % fpeaks[0])
            elif self.deltaf < 1.0:
                labels.append('%4.1f Hz' % fpeaks[0])
            else:
                labels.append('%4.0f Hz' % fpeaks[0])
        if len(self.mains) > 0:
            fpeaks = self.mains[:, 0]
            fpeakinx = np.array([np.round(fp / self.deltaf) for fp in fpeaks if fp < freqs[-1]], dtype=np.int)
            fishpoints, = self.axp.plot(fpeaks[:len(fpeakinx)],
                                        power[fpeakinx], linestyle='None',
                                        marker='.', color='k', ms=10, mec=None, mew=0.0, zorder=2)
            self.peak_artists.append(fishpoints)
            labels.append('%3.0f Hz mains' % self.cfg.value('mainsFreq'))
        ncol = (len(labels)-1) // 8 + 1
        self.legendhandle = self.axs.legend(self.peak_artists[:len(labels)], labels, loc='upper right', ncol=ncol)
        self.legenddict = dict()
        for legpoints, (finx, fish) in zip(self.legendhandle.get_lines(), enumerate(self.fishlist)):
            legpoints.set_picker(8)
            self.legenddict[legpoints] = [finx, fish]
        self.legendhandle.set_visible(self.legend)
        if self.power_artist == None:
            self.power_artist, = self.axp.plot(freqs, power, 'b', zorder=3)
        else:
            self.power_artist.set_data(freqs, power)
        if draw:
            self.fig.canvas.draw()

    def keypress(self, event):
        # print('pressed', event.key)
        if event.key in '+=X':
            if self.twindow * self.samplerate > 20:
                self.twindow *= 0.5
                self.update_plots()
        elif event.key in '-x':
            if self.twindow < len(self.data) / self.samplerate:
                self.twindow *= 2.0
                self.update_plots()
        elif event.key == 'pagedown':
            if self.toffset + 0.5 * self.twindow < len(self.data) / self.samplerate:
                self.toffset += 0.5 * self.twindow
                self.update_plots()
        elif event.key == 'pageup':
            if self.toffset > 0:
                self.toffset -= 0.5 * self.twindow
                if self.toffset < 0.0:
                    self.toffset = 0.0
                self.update_plots()
        elif event.key == 'a':
            if self.min_clip == 0.0 or self.max_clip == 0.0:
                self.min_clip, self.max_clip = clip_amplitudes(
                    self.data, **clip_args(self.cfg, self.samplerate))
            try:
                if self.cfg.value('bestWindowSize') <= 0.0:
                    self.cfg.set('bestWindowSize', (len(self.data)-1)/self.samplerate)
                idx0, idx1, clipped = best_window_indices(
                    self.data, self.samplerate, min_clip=self.min_clip,
                    max_clip=self.max_clip, **best_window_args(self.cfg))
                if idx1 > 0:
                    self.toffset = idx0 / self.samplerate
                    self.twindow = (idx1 - idx0) / self.samplerate
                    self.twindow *= 2.0/(self.cfg.value('numberPSDWindows')+1.0)
                    self.update_plots()
            except UserWarning as e:
                if self.verbose > 0:
                    print(str(e))
        elif event.key == 'ctrl+pagedown':
            if self.toffset + 5.0 * self.twindow < len(self.data) / self.samplerate:
                self.toffset += 5.0 * self.twindow
                self.update_plots()
        elif event.key == 'ctrl+pageup':
            if self.toffset > 0:
                self.toffset -= 5.0 * self.twindow
                if self.toffset < 0.0:
                    self.toffset = 0.0
                self.update_plots()
        elif event.key == 'down':
            if self.toffset + self.twindow < len(self.data) / self.samplerate:
                self.toffset += 0.05 * self.twindow
                self.update_plots()
        elif event.key == 'up':
            if self.toffset > 0.0:
                self.toffset -= 0.05 * self.twindow
                if self.toffset < 0.0:
                    self.toffset = 0.0
                self.update_plots()
        elif event.key == 'home':
            if self.toffset > 0.0:
                self.toffset = 0.0
                self.update_plots()
        elif event.key == 'end':
            toffs = np.floor(len(self.data) / self.samplerate / self.twindow) * self.twindow
            if self.toffset < toffs:
                self.toffset = toffs
                self.update_plots()
        elif event.key == 'y':
            h = self.ymax - self.ymin
            c = 0.5 * (self.ymax + self.ymin)
            self.ymin = c - h
            self.ymax = c + h
            self.axt.set_ylim(self.ymin, self.ymax)
            self.fig.canvas.draw()
        elif event.key == 'Y':
            h = 0.25 * (self.ymax - self.ymin)
            c = 0.5 * (self.ymax + self.ymin)
            self.ymin = c - h
            self.ymax = c + h
            self.axt.set_ylim(self.ymin, self.ymax)
            self.fig.canvas.draw()
        elif event.key == 'v':
            t0 = int(np.round(self.toffset * self.samplerate))
            t1 = int(np.round((self.toffset + self.twindow) * self.samplerate))
            min = np.min(self.data[t0:t1])
            max = np.max(self.data[t0:t1])
            h = 0.5 * (max - min)
            c = 0.5 * (max + min)
            self.ymin = c - h
            self.ymax = c + h
            self.axt.set_ylim(self.ymin, self.ymax)
            self.fig.canvas.draw()
        elif event.key == 'V':
            self.ymin = -1.0
            self.ymax = +1.0
            self.axt.set_ylim(self.ymin, self.ymax)
            self.fig.canvas.draw()
        elif event.key == 'left':
            if self.fmin > 0.0:
                fwidth = self.fmax - self.fmin
                self.fmin -= 0.5 * fwidth
                self.fmax -= 0.5 * fwidth
                if self.fmin < 0.0:
                    self.fmin = 0.0
                    self.fmax = fwidth
                self.axs.set_ylim(self.fmin, self.fmax)
                self.axp.set_xlim(self.fmin, self.fmax)
                self.fig.canvas.draw()
        elif event.key == 'right':
            if self.fmax < 0.5 * self.samplerate:
                fwidth = self.fmax - self.fmin
                self.fmin += 0.5 * fwidth
                self.fmax += 0.5 * fwidth
                self.axs.set_ylim(self.fmin, self.fmax)
                self.axp.set_xlim(self.fmin, self.fmax)
                self.fig.canvas.draw()
        elif event.key == 'ctrl+left':
            if self.fmin > 0.0:
                fwidth = self.fmax - self.fmin
                self.fmin = 0.0
                self.fmax = fwidth
                self.axs.set_ylim(self.fmin, self.fmax)
                self.axp.set_xlim(self.fmin, self.fmax)
                self.fig.canvas.draw()
        elif event.key == 'ctrl+right':
            if self.fmax < 0.5 * self.samplerate:
                fwidth = self.fmax - self.fmin
                fm = 0.5 * self.samplerate
                self.fmax = np.ceil(fm / fwidth) * fwidth
                self.fmin = self.fmax - fwidth
                if self.fmin < 0.0:
                    self.fmin = 0.0
                    self.fmax = fwidth
                self.axs.set_ylim(self.fmin, self.fmax)
                self.axp.set_xlim(self.fmin, self.fmax)
                self.fig.canvas.draw()
        elif event.key in 'f':
            if self.fmax < 0.5 * self.samplerate or self.fmin > 0.0:
                fwidth = self.fmax - self.fmin
                if self.fmax < 0.5 * self.samplerate:
                    self.fmax = self.fmin + 2.0 * fwidth
                elif self.fmin > 0.0:
                    self.fmin = self.fmax - 2.0 * fwidth
                    if self.fmin < 0.0:
                        self.fmin = 0.0
                        self.fmax = 2.0 * fwidth
                self.axs.set_ylim(self.fmin, self.fmax)
                self.axp.set_xlim(self.fmin, self.fmax)
                self.fig.canvas.draw()
        elif event.key in 'F':
            if self.fmax - self.fmin > 1.0:
                fwidth = self.fmax - self.fmin
                self.fmax = self.fmin + 0.5 * fwidth
                self.axs.set_ylim(self.fmin, self.fmax)
                self.axp.set_xlim(self.fmin, self.fmax)
                self.fig.canvas.draw()
        elif event.key in 'r':
            if self.freq_resolution < 1000.0:
                self.freq_resolution *= 2.0
                self.update_plots()
        elif event.key in 'R':
            if 1.0 / self.freq_resolution < self.tmax:
                self.freq_resolution *= 0.5
                self.update_plots()
        elif event.key in 'd':
            self.decibel = not self.decibel
            self.update_plots()
        elif event.key in 'm':
            if self.cfg.value('mainsFreq') == 0.0:
                self.cfg.set('mainsFreq', self.mains_freq)
            else:
                self.cfg.set('mainsFreq', 0.0)
            self.update_plots()
        elif event.key in 't':
            t_diff = self.cfg.value('highThresholdFactor') - self.cfg.value('lowThresholdFactor')
            self.cfg.set('lowThresholdFactor', self.cfg.value('lowThresholdFactor') - 0.1)
            if self.cfg.value('lowThresholdFactor') < 0.1:
                self.cfg.set('lowThresholdFactor', 0.1)
            self.cfg.set('highThresholdFactor', self.cfg.value('lowThresholdFactor') + t_diff)
            print('lowThresholdFactor =', self.cfg.value('lowThresholdFactor'))
            self.update_plots()
        elif event.key in 'T':
            t_diff = self.cfg.value('highThresholdFactor') - self.cfg.value('lowThresholdFactor')
            self.cfg.set('lowThresholdFactor', self.cfg.value('lowThresholdFactor') + 0.1)
            if self.cfg.value('lowThresholdFactor') > 20.0:
                self.cfg.set('lowThresholdFactor', 20.0)
            self.cfg.set('highThresholdFactor', self.cfg.value('lowThresholdFactor') + t_diff)
            print('lowThresholdFactor =', self.cfg.value('lowThresholdFactor'))
            self.update_plots()
        elif event.key == 'escape':
            self.remove_peak_annotation()
            self.fig.canvas.draw()
        elif event.key in 'h':
            self.help = not self.help
            for ht in self.helptext:
                ht.set_visible(self.help)
            self.fig.canvas.draw()
        elif event.key in 'l':
            self.legend = not self.legend
            self.legendhandle.set_visible(self.legend)
            self.fig.canvas.draw()
        elif event.key in 'w':
            self.plot_waveform()
        elif event.key in 'p':
            self.play_segment()
        elif event.key in 'P':
            self.play_all()
        elif event.key in '1' :
            self.play_tone('c3')
        elif event.key in '2' :
            self.play_tone('a3')
        elif event.key in '3' :
            self.play_tone('e4')
        elif event.key in '4' :
            self.play_tone('a4')
        elif event.key in '5' :
            self.play_tone('c5')
        elif event.key in '6' :
            self.play_tone('e5')
        elif event.key in '7' :
            self.play_tone('g5')
        elif event.key in '8' :
            self.play_tone('a5')
        elif event.key in '9' :
            self.play_tone('c6')
        elif event.key in 'S':
            self.save_segment()

    def buttonpress( self, event ) :
        # print('mouse pressed', event.button, event.key, event.step)
        if event.inaxes == self.axp:
            if event.key == 'shift' or event.key == 'control':
                # show next or previous harmonic:
                if event.key == 'shift':
                    if event.button == 1:
                        ftarget = event.xdata / 2.0
                    elif event.button == 3:
                        ftarget = event.xdata * 2.0
                else:
                    if event.button == 1:
                        ftarget = event.xdata / 1.5
                    elif event.button == 3:
                        ftarget = event.xdata * 1.5
                foffs = event.xdata - self.fmin
                fwidth = self.fmax - self.fmin
                self.fmin = ftarget - foffs
                self.fmax = self.fmin + fwidth
                self.axs.set_ylim(self.fmin, self.fmax)
                self.axp.set_xlim(self.fmin, self.fmax)
                self.fig.canvas.draw()
            else:
                # put label on peak
                self.remove_peak_annotation()
                # find closest peak:
                fwidth = self.fmax - self.fmin
                peakdist = np.abs(self.allpeaks[:, 0] - event.xdata)
                inx = np.argmin(peakdist)
                if peakdist[inx] < 0.005 * fwidth:
                    peak = self.allpeaks[inx, :]
                    # find fish:
                    foundfish = False
                    for finx, fish in enumerate(self.fishlist):
                        if np.min(np.abs(fish[:, 0] - peak[0])) < 0.8 * self.deltaf:
                            self.annotate_fish(fish, finx)
                            foundfish = True
                            break
                    if not foundfish:
                        self.annotate_peak(peak)
                        self.fig.canvas.draw()
                else:
                    self.fig.canvas.draw()

    def onpick(self, event):
        # print('pick')
        legendpoint = event.artist
        finx, fish = self.legenddict[legendpoint]
        self.annotate_fish(fish, finx)

    def resize(self, event):
        # print('resized', event.width, event.height)
        leftpixel = 80.0
        rightpixel = 20.0
        xaxispixel = 50.0
        toppixel = 20.0
        timeaxis = 0.42
        left = leftpixel / event.width
        width = 1.0 - left - rightpixel / event.width
        xaxis = xaxispixel / event.height
        top = toppixel / event.height
        height = (1.0 - timeaxis - top) / 2.0
        if left < 0.5 and width < 1.0 and xaxis < 0.3 and top < 0.2:
            self.axt.set_position([left, timeaxis + height, width, height])
            self.axs.set_position([left, timeaxis, width, height])
            self.axp.set_position([left, xaxis, width, timeaxis - 2.0 * xaxis])

    def plot_waveform(self):
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        name = self.filename.split('.')[0]
        if self.channel > 0:
            ax.set_title('{filename} channel={channel:d}'.format(
                filename=self.filename, channel=self.channel))
            figfile = '{name}-{channel:d}-{time:.4g}s-waveform.png'.format(
                name=name, channel=self.channel, time=self.toffset)
        else:
            ax.set_title(self.filename)
            figfile = '{name}-{time:.4g}s-waveform.png'.format(
                name=name, time=self.toffset)
        t0 = int(np.round(self.toffset * self.samplerate))
        t1 = int(np.round((self.toffset + self.twindow) * self.samplerate))
        if t1>len(self.data):
            t1 = len(self.data)
        time = np.arange(t0, t1) / self.samplerate
        if self.twindow < 1.0:
            ax.set_xlabel('Time [ms]')
            ax.set_xlim(1000.0 * self.toffset,
                        1000.0 * (self.toffset + self.twindow))
            ax.plot(1000.0 * time, self.data[t0:t1])
        else:
            ax.set_xlabel('Time [s]')
            ax.set_xlim(self.toffset, self.toffset + self.twindow)
            ax.plot(time, self.data[t0:t1])
        ax.set_ylabel('Amplitude [{:s}]'.format(self.unit))
        fig.tight_layout()
        fig.savefig(figfile)
        fig.clear()
        plt.close(fig)
        print('saved waveform figure to', figfile)

    def play_segment(self):
        t0 = int(np.round(self.toffset * self.samplerate))
        t1 = int(np.round((self.toffset + self.twindow) * self.samplerate))
        playdata = 1.0 * self.data[t0:t1]
        fade(playdata, self.samplerate, 0.1)
        self.audio.play(playdata, self.samplerate, blocking=False)

    def save_segment(self):
        t0s = int(np.round(self.toffset))
        t1s = int(np.round(self.toffset + self.twindow))
        t0 = int(np.round(self.toffset * self.samplerate))
        t1 = int(np.round((self.toffset + self.twindow) * self.samplerate))
        savedata = 1.0 * self.data[t0:t1]
        filename = self.filename.split('.')[0]
        segmentfilename = '{name}-{time0:.4g}s-{time1:.4g}s.wav'.format(
                name=filename, time0=t0s, time1 = t1s)
        write_audio(segmentfilename, savedata, self.data.samplerate)
        print('saved segment to: ' , segmentfilename)
        
    def play_all(self):
        self.audio.play(self.data[:], self.samplerate, blocking=False)
        
    def play_tone( self, frequency ) :
        self.audio.beep(1.0, frequency)


def short_user_warning(message, category, filename, lineno, file=None, line=''):
    if file is None:
        file = sys.stderr
    if category == UserWarning:
        file.write('%s line %d: %s\n' % ('/'.join(filename.split('/')[-2:]), lineno, message))
    else:
        s = warnings.formatwarning(message, category, filename, lineno, line)
        file.write(s)


def main():
    warnings.showwarning = short_user_warning

    # config file name:
    cfgfile = __package__ + '.cfg'

    # command line arguments:
    parser = argparse.ArgumentParser(
        description='Display waveform, and power spectrum with detected fundamental frequencies of EOD recordings.',
        epilog='by Jan Benda (2015-2017)')
    parser.add_argument('--version', action='version', version=__version__)
    parser.add_argument('-v', action='count', dest='verbose')
    parser.add_argument('-c', '--save-config', nargs='?', default='', const=cfgfile,
                        type=str, metavar='cfgfile',
                        help='save configuration to file cfgfile (defaults to {0})'.format(cfgfile))
    parser.add_argument('file', nargs='?', default='', type=str,
                        help='name of the file with the time series data')
    parser.add_argument('channel', nargs='?', default=0, type=int,
                        help='channel to be displayed')
    args = parser.parse_args()
    filepath = args.file

    # set verbosity level from command line:
    verbose = 0
    if args.verbose != None:
        verbose = args.verbose

    if len(args.save_config):
        # save configuration:
        cfg = configuration()
        cfg.load_files(cfgfile, filepath, 4, verbose)
        save_configuration(cfg, cfgfile)
        return
    elif len(filepath) == 0:
        parser.error('you need to specify a file containing some data')

    # load configuration:
    cfg = configuration()
    cfg.load_files(cfgfile, filepath, 4, verbose-1)

    # load data:
    filename = os.path.basename(filepath)
    channel = args.channel
    # TODO: add blocksize and backsize as configuration parameter!
    with open_data(filepath, channel, 60.0, 10.0, verbose) as data:
        # plot:
        ## if len(data) < 10**8:
        ##     # data[:].copy() makes bestwindow much faster (it's slow in eventdetection):
        ##     SignalPlot(data[:].copy(), data.samplerate, data.unit, filename, channel)
        ## else:
        SignalPlot(data, data.samplerate, data.unit, filename, channel, verbose, cfg)

        
if __name__ == '__main__':
    main()


# 50301L02.WAV t=9 bis 9.15 sec


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

# problems: EN088, EN089, 20140524_RioCanita/EN055 sterno not catched, EN056, EN059
