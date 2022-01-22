import sys
import os
import warnings
import argparse
import numpy as np
import matplotlib.pyplot as plt
from audioio.playaudio import PlayAudio, fade
from audioio.audiowriter import write_audio
from .version import __version__, __year__
from .dataloader import open_data


class SignalPlot:
    def __init__(self, data, samplingrate, unit, filename):
        self.filename = filename
        self.samplerate = samplingrate
        self.data = data
        self.channels = data.channels
        self.unit = unit
        self.tmax = (len(self.data)-1)/self.samplerate
        self.toffset = 0.0
        self.twindow = 10.0
        if self.twindow > self.tmax:
            self.twindow = np.round(2 ** (np.floor(np.log(self.tmax) / np.log(2.0)) + 1.0))
        self.ymin = -1.0
        self.ymax = +1.0
        self.trace_artist = [None] * self.channels
        self.help = False
        self.helptext = []

        # audio output:
        self.audio = PlayAudio()
        
        # set key bindings:
        plt.rcParams['keymap.fullscreen'] = 'f'
        plt.rcParams['keymap.pan'] = 'ctrl+m'
        plt.rcParams['keymap.quit'] = 'ctrl+w, alt+q, q'
        plt.rcParams['keymap.yscale'] = ''
        plt.rcParams['keymap.xscale'] = ''
        plt.rcParams['keymap.grid'] = ''
        plt.rcParams['keymap.all_axes'] = ''

        # the figure:
        plt.ioff()
        self.fig, self.axs = plt.subplots(self.channels, 1, squeeze=False,
                                          figsize=(15, 9))
        self.axs = self.axs.ravel()
        self.fig.canvas.set_window_title(self.filename)
        self.fig.canvas.mpl_connect('key_press_event', self.keypress)
        self.fig.canvas.mpl_connect('resize_event', self.resize)
        # trace plots:
        for k in range(self.channels):
            self.axs[k].set_ylabel(f'C-{k+1} [{self.unit}]')
        for k in range(self.channels-1):
            self.axs[k].xaxis.set_major_formatter(plt.NullFormatter())
        self.axs[self.channels-1].set_xlabel('Time [s]')
        ht = self.axs[0].text(0.98, 0.05, '(ctrl+) page and arrow up, down, home, end: scroll', ha='right',
                           transform=self.axs[0].transAxes)
        self.helptext.append(ht)
        ht = self.axs[0].text(0.98, 0.15, '+, -, X, x: zoom in/out', ha='right', transform=self.axs[0].transAxes)
        self.helptext.append(ht)
        ht = self.axs[0].text(0.98, 0.25, 'y,Y,v,V: zoom amplitudes', ha='right', transform=self.axs[0].transAxes)
        self.helptext.append(ht)
        ht = self.axs[0].text(0.98, 0.35, 'p,P: play audio (display,all)', ha='right', transform=self.axs[0].transAxes)
        self.helptext.append(ht)
        ht = self.axs[0].text(0.98, 0.45, 'f: full screen', ha='right', transform=self.axs[0].transAxes)
        self.helptext.append(ht)
        ht = self.axs[0].text(0.98, 0.55, 'w: plot waveforms into png file', ha='right', transform=self.axs[0].transAxes)
        self.helptext.append(ht)
        ht = self.axs[0].text(0.98, 0.65, 'S: save audiosegment', ha='right', transform=self.axs[0].transAxes)
        self.helptext.append(ht)
        ht = self.axs[0].text(0.98, 0.75, 'q: quit', ha='right', transform=self.axs[0].transAxes)
        self.helptext.append(ht)
        ht = self.axs[0].text(0.98, 0.85, 'h: toggle this help', ha='right', transform=self.axs[0].transAxes)
        self.helptext.append(ht)
        # plot:
        for ht in self.helptext:
            ht.set_visible(self.help)
        self.update_plots(False)
        plt.show()

    def __del(self):
        self.audio.close()

    def update_plots(self, draw=True):
        t0 = int(np.round(self.toffset * self.samplerate))
        t1 = int(np.round((self.toffset + self.twindow) * self.samplerate))
        if t1>len(self.data):
            t1 = len(self.data)
        time = np.arange(t0, t1) / self.samplerate
        for k in range(self.channels):
            self.axs[k].set_xlim(self.toffset, self.toffset + self.twindow)
            if self.trace_artist[k] == None:
                self.trace_artist[k], = self.axs[k].plot(time, self.data[t0:t1,k])
            else:
                self.trace_artist[k].set_data(time, self.data[t0:t1,k])
            self.axs[k].set_ylim(self.ymin, self.ymax)
        self.fig.canvas.draw()

    def resize(self, event):
        # print('resized', event.width, event.height)
        leftpixel = 80.0
        rightpixel = 20.0
        bottompixel = 50.0
        toppixel = 20.0
        x0 = leftpixel / event.width
        x1 = 1.0 - rightpixel / event.width
        y0 = bottompixel / event.height
        y1 = 1.0 - toppixel / event.height
        self.fig.subplots_adjust(left=x0, right=x1, bottom=y0, top=y1,
                                 hspace=0)

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
            for k in range(self.channels):
                self.axs[k].set_ylim(self.ymin, self.ymax)
            self.fig.canvas.draw()
        elif event.key == 'Y':
            h = 0.25 * (self.ymax - self.ymin)
            c = 0.5 * (self.ymax + self.ymin)
            self.ymin = c - h
            self.ymax = c + h
            for k in range(self.channels):
                self.axs[k].set_ylim(self.ymin, self.ymax)
            self.fig.canvas.draw()
        elif event.key == 'v':
            t0 = int(np.round(self.toffset * self.samplerate))
            t1 = int(np.round((self.toffset + self.twindow) * self.samplerate))
            min = np.min(self.data[t0:t1,:])
            max = np.max(self.data[t0:t1,:])
            h = 0.5 * (max - min)
            c = 0.5 * (max + min)
            self.ymin = c - h
            self.ymax = c + h
            for k in range(self.channels):
                self.axs[k].set_ylim(self.ymin, self.ymax)
            self.fig.canvas.draw()
        elif event.key == 'V':
            self.ymin = -1.0
            self.ymax = +1.0
            for k in range(self.channels):
                self.axs[k].set_ylim(self.ymin, self.ymax)
            self.fig.canvas.draw()
        elif event.key in 'h':
            self.help = not self.help
            for ht in self.helptext:
                ht.set_visible(self.help)
            self.fig.canvas.draw()
        elif event.key in 'p':
            self.play_segment()
        elif event.key in 'P':
            self.play_all()
        elif event.key in 'S':
            self.save_segment()
        elif event.key in 'w':
            self.plot_waveform()

    def play_segment(self):
        t0 = int(np.round(self.toffset * self.samplerate))
        t1 = int(np.round((self.toffset + self.twindow) * self.samplerate))
        playdata = 1.0 * np.mean(self.data[t0:t1], 1)
        f = 0.1 if self.twindow > 0.5 else 0.1*self.twindow
        fade(playdata, self.samplerate, f)
        self.audio.play(playdata, self.samplerate, blocking=False)
        
    def play_all(self):
        self.audio.play(np.mean(self.data, 1), self.samplerate, blocking=False)

    def save_segment(self):
        t0s = int(np.round(self.toffset))
        t1s = int(np.round(self.toffset + self.twindow))
        t0 = int(np.round(self.toffset * self.samplerate))
        t1 = int(np.round((self.toffset + self.twindow) * self.samplerate))
        filename = self.filename.split('.')[0]
        segment_filename = f'{filename}-{t0s:.4g}s-{t1s:.4g}s.wav'
        write_audio(segment_filename, self.data[t0:t1,:], self.data.samplerate)
        print('saved segment to: ' , segment_filename)

    def plot_waveform(self):
        fig, axs = plt.subplots(self.channels, 1, squeeze=False)
        axs = axs.ravel()
        fig.subplots_adjust(left=0.12, right=0.98, bottom=0.1, top=0.95,
                            hspace=0)
        name = self.filename.split('.')[0]
        figfile = f'{name}-{self.toffset:.4g}s-traces.png'
        axs[0].set_title(self.filename)
        t0 = int(np.round(self.toffset * self.samplerate))
        t1 = int(np.round((self.toffset + self.twindow) * self.samplerate))
        if t1>len(self.data):
            t1 = len(self.data)
        time = np.arange(t0, t1)/self.samplerate
        if self.twindow < 1.0:
            axs[self.channels-1].set_xlabel('Time [ms]')
            for k in range(self.channels):
                axs[k].set_xlim(1000.0 * self.toffset,
                                1000.0 * (self.toffset + self.twindow))
                axs[k].plot(1000.0 * time, self.data[t0:t1,k])
        else:
            axs[self.channels-1].set_xlabel('Time [s]')
            for k in range(self.channels):
                axs[k].set_xlim(self.toffset, self.toffset + self.twindow)
                axs[k].plot(time, self.data[t0:t1,k])
        for k in range(self.channels):
            axs[k].set_ylim(self.ymin, self.ymax)
            axs[k].set_ylabel(f'C-{k+1} [{self.unit}]')
        for k in range(self.channels-1):
            axs[k].xaxis.set_major_formatter(plt.NullFormatter())
        fig.savefig(figfile)
        plt.close(fig)
        print('saved waveform figure to', figfile)
        

def short_user_warning(message, category, filename, lineno, file=None, line=''):
    if file is None:
        file = sys.stderr
    if category == UserWarning:
        file.write('%s line %d: %s\n' % ('/'.join(filename.split('/')[-2:]), lineno, message))
    else:
        s = warnings.formatwarning(message, category, filename, lineno, line)
        file.write(s)


def main(cargs=None):
    warnings.showwarning = short_user_warning

    # config file name:
    cfgfile = __package__ + '.cfg'

    # command line arguments:
    if cargs is None:
        cargs = sys.argv[1:]
    parser = argparse.ArgumentParser(
        description='Browse mutlichannel EOD recordings.',
        epilog='version %s by Benda-Lab (2022-%s)' % (__version__, __year__))
    parser.add_argument('--version', action='version', version=__version__)
    parser.add_argument('-v', action='count', dest='verbose')
    parser.add_argument('file', nargs=1, default='', type=str,
                        help='name of the file with the time series data')
    args = parser.parse_args(cargs)
    filepath = args.file[0]

    # set verbosity level from command line:
    verbose = 0
    if args.verbose != None:
        verbose = args.verbose

    # load data:
    filename = os.path.basename(filepath)
    with open_data(filepath, -1, 20.0, 5.0, verbose) as data:
        SignalPlot(data, data.samplerate, data.unit, filename)

        
if __name__ == '__main__':
    main()
