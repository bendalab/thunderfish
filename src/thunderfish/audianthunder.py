import os
import sys
import numpy as np
import matplotlib.pyplot as plt

try:
    from audian.audian import audian_cli
    from audian.plugins import Plugins
    from audian.analyzer import Analyzer
except ImportError:
    print()
    print('ERROR: You need to install audian (https://github.com/bendalab/audian):')
    print()
    print('pip install audian')
    print()
    sys.exit(1)

from io import StringIO
from pathlib import Path

from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvas
from matplotlib.backends.backend_qtagg import \
    NavigationToolbar2QT as NavigationToolbar
from matplotlib.backend_bases import MouseButton
from matplotlib.ticker import PercentFormatter

try:
    from PyQt5.QtCore import Signal
except ImportError:
    from PyQt5.QtCore import pyqtSignal as Signal
from PyQt5.QtCore import Qt, QObject, QTime
from PyQt5.QtGui import QFont, QKeySequence
from PyQt5.QtWidgets import QDialog, QShortcut, QVBoxLayout, QHBoxLayout
from PyQt5.QtWidgets import QWidget, QTabWidget, QToolBar, QAction, QStyle
from PyQt5.QtWidgets import QPushButton, QLabel, QScrollArea, QFileDialog

from audioio import fade
from thunderlab.eventdetection import minmax_threshold, detect_peaks
from thunderlab.powerspectrum import decibel, plot_decibel_psd
from thunderlab.tabledata import write_table_args

from .thunderfish import configuration
from .thunderfish import rec_style, spectrum_style
from .thunderfish import wave_eod_styles, pulse_eod_styles, snippet_style
from .thunderfish import wave_spec_styles, pulse_spec_styles
from .bestwindow import clip_args, clip_amplitudes
from .harmonics import colors_markers, plot_harmonic_groups
from .eodanalysis import detect_eods, plot_eod_snippets
from .eodanalysis import plot_eod_recording, zoom_eod_recording, save_analysis
from .pulseanalysis import plot_pulse_eodtimes, plot_pulse_eod
from .pulseanalysis import pulsetrain, plot_pulse_spectrum
from .waveanalysis import plot_wave_eod, plot_wave_spectrum 
from .harmonics import annotate_harmonic_group
from .fakefish import wavefish_eods, musical_intervals, musical_intervals_short


class TimePlot():
    
    def __init__(self, time):
        self.full_time_range = (time[0], time[-1])
        self.zoomed_time_range = None
        self.tfac = 1
        self.canvas = FigureCanvas(Figure(figsize=(10, 5),
                                          layout='constrained'))
        self.navi = NavigationToolbar(self.canvas)
        self.navi.hide()
        self.ax = self.canvas.figure.subplots()
        self.rate = 1/np.mean(np.diff(time))

    def toggle_time_range(self):
        if self.zoomed_time_range is None:
            self.zoomed_time_range = self.ax.get_xlim()
            self.ax.set_xlim(*self.full_time_range)
        else:
            self.ax.set_xlim(*self.zoomed_time_range)
            self.zoomed_time_range = None
        self.canvas.draw()

    def zoom_in(self):
        t0, t1 = self.ax.get_xlim()
        if (t1 - t0)/self.tfac > 0.001:
            t1 = t0 + 0.5*(t1 - t0)
            self.ax.set_xlim(t0, t1)
            self.canvas.draw()

    def zoom_out(self):
        t0, t1 = self.ax.get_xlim()
        if t1 < self.full_time_range[1]:
            t1 = t0 + 2*(t1 - t0)
            if t1 > self.full_time_range[1]:
                t1 = self.full_time_range[1]
            self.ax.set_xlim(t0, t1)
            self.canvas.draw()
            
    def move_backward(self):
        t0, t1 = self.ax.get_xlim()
        if t0 > self.full_time_range[0]:
            dt = 0.5*(t1 - t0)
            if t0 - dt < self.full_time_range[0]:
                dt = t0 - self.full_time_range[0]
            self.ax.set_xlim(t0 - dt, t1 - dt)
            self.canvas.draw()
            
    def move_forward(self):
        t0, t1 = self.ax.get_xlim()
        if t1 < self.full_time_range[1]:
            dt = 0.5*(t1 - t0)
            if t1 + dt > self.full_time_range[1]:
                dt = self.full_time_range[1] - t1
            self.ax.set_xlim(t0 + dt, t1 + dt)
            self.canvas.draw()                
            
    def home(self):
        t0, t1 = self.ax.get_xlim()
        if t0 > self.full_time_range[0]:
            dt = t1 - t0
            t0 = self.full_time_range[0]
            self.ax.set_xlim(t0, t0 + dt)
            self.canvas.draw()
            
    def end(self):
        t0, t1 = self.ax.get_xlim()
        if t1 < self.full_time_range[1]:
            dt = t1 - t0
            t1 = self.full_time_range[1]
            self.ax.set_xlim(t1 - dt, t1)
            self.canvas.draw()                


class TracePlot(TimePlot):
    
    def __init__(self, time, data, unit, eod_props, wave_eodfs,
                 pulse_colors, pulse_markers):
        super().__init__(time)
        twidth = 0.5
        self.tfac = plot_eod_recording(self.ax, data, self.rate, unit,
                                       twidth, time[0], rec_style)
        plot_pulse_eodtimes(self.ax, data, self.rate,
                            twidth, eod_props, time[0],
                            colors=pulse_colors,
                            markers=pulse_markers,
                            frameon=True, loc='upper right')
        zoom_eod_recording(self.ax, eod_props, data, self.rate,
                           twidth, self.tfac, time[0])
        if self.ax.get_legend() is not None:
            self.ax.get_legend().get_frame().set_color('white')


class RatePlot(TimePlot):
    
    def __init__(self, time, eod_props, pulse_colors):
        super().__init__(time)
        prop_cycle = plt.rcParams['axes.prop_cycle']
        colors = prop_cycle.by_key()['color']
        k = 0
        for props in eod_props:
            if props['type'] != 'pulse':
                continue
            if 'times' not in props:
                continue
            times = props['peaktimes'] + self.full_time_range[0]
            rate = 1/np.diff(times)
            mask = np.append(True, np.abs(np.diff(rate))/rate[:-1] < 0.1)
            #color = pulse_colors[k % len(pulse_colors)]
            color = colors[k % len(colors)]
            label = f'{props["EODf"]:6.1f} Hz'
            self.ax.plot(times[:-1][mask], rate[mask], '-o',
                         color=color, label=label)
            self.ax.plot(times[:-1][~mask], rate[~mask], 'o',
                         color=color)
            self.ax.set_xlabel('Time [s]')
            self.ax.set_ylim(bottom=0)
            self.ax.set_ylabel('Rate [Hz]')
            self.ax.legend()
            k += 1
        if self.ax.get_legend() is not None:
            self.ax.get_legend().get_frame().set_color('white')
            
        
class PowerPlot():
    
    def __init__(self, power_freqs, powers, power_thresh,
                 wave_eodfs, wave_indices, wave_colors, wave_markers):
        self.deltaf = np.mean(np.diff(power_freqs))
        self.power_freqs = power_freqs
        self.powers = powers
        self.canvas = FigureCanvas(Figure(figsize=(10, 5),
                                          layout='constrained'))
        self.harmonics_artists = []
        self.harmonics_div = 1
        self.harmonics_freq = 0
        self.annotation = []
        self.pick = QTime.currentTime()
        self.moved = False
        self.canvas.mpl_connect('button_press_event', self.onpress)
        self.canvas.mpl_connect('button_release_event', self.onrelease)
        self.canvas.mpl_connect('motion_notify_event', self.onmove)
        self.canvas.mpl_connect('pick_event', self.onpick)
        QShortcut('ESC', self.canvas).activated.connect(self.clear)
        self.navi = NavigationToolbar(self.canvas)
        self.navi.hide()
        self.ax = self.canvas.figure.subplots()
        if power_thresh is not None:
            self.ax.plot(power_thresh[:, 0],
                         decibel(power_thresh[:, 1]),
                         '#CCCCCC', lw=1)
        self.wave_dict = {}
        if len(wave_eodfs) > 0:
            self.wave_dict = \
                plot_harmonic_groups(self.ax, wave_eodfs,
                                     wave_indices, max_groups=0,
                                     skip_bad=False,
                                     sort_by_freq=True,
                                     label_power=False,
                                     colors=wave_colors,
                                     markers=wave_markers,
                                     legend_rows=10, frameon=False,
                                     bbox_to_anchor=(1, 1),
                                     loc='upper left')
        plot_decibel_psd(self.ax, self.power_freqs, self.powers,
                         log_freq=False, min_freq=0, max_freq=3000,
                         ymarg=5.0, sstyle=spectrum_style)
        self.ax.yaxis.set_major_locator(plt.MaxNLocator(6))
        
    def onpick(self, event):
        self.clear()
        a = event.artist
        if a in self.wave_dict:
            finx, fish = self.wave_dict[a]
            self.annotation = annotate_harmonic_group(self.ax, fish, finx,
                                                      freq_thresh=0.8*self.deltaf)
            self.ax.get_figure().canvas.draw()
            self.pick = QTime.currentTime()
            
    def onpress(self, event):
        self.moved = False
            
    def onmove(self, event):
        self.moved = True
            
    def onrelease(self, event):
        if self.moved:
            return
        if self.pick.msecsTo(QTime.currentTime()) < 100:
            return
        if event.inaxes is not None and event.inaxes == self.ax:
            self.clear(True)
            pmin, pmax = self.ax.get_ylim()
            fmin, fmax = self.ax.get_xlim()
            fwidth = fmax - fmin
            df = 0.01*fwidth
            if abs(self.harmonics_freq - event.xdata) <= df:
                self.harmonics_div += 1
            else:
                self.harmonics_div = 1
                f = event.xdata
                mask = (self.power_freqs >= f - df) & (self.power_freqs <= f + df)
                dbpower = decibel(self.powers[mask])
                thresh = minmax_threshold(dbpower, None, 0.3)
                peaks, troughs = detect_peaks(dbpower, thresh)
                if len(peaks) == 0:
                    i = np.argmax(self.powers[mask])
                elif len(peaks) == 1:
                    i = peaks[0]
                else:
                    i = peaks[np.argmin(np.abs(self.power_freqs[mask][peaks] - f))]
                self.harmonics_freq = self.power_freqs[mask][i]
            f1 = self.harmonics_freq/self.harmonics_div
            for h in range(1, 1000*self.harmonics_div):
                if h*f1 > self.power_freqs[-1]:
                    break
                if h == 1:
                    a = self.ax.text(f1 + 0.01*fwidth, pmax, f'{f1:.1f}Hz',
                                     ha='left', va='top',
                                     bbox=dict(boxstyle='round',
                                               facecolor='white'))
                    self.harmonics_artists.append(a)
                a = self.ax.axvline(h*f1, color='k', lw=1)
                self.harmonics_artists.append(a)
            self.ax.get_figure().canvas.draw()

    def clear(self, keep_div=False):
        if len(self.annotation) > 0:
            for a in self.annotation:
                a.remove()
            self.annotation = []
        if len(self.harmonics_artists) > 0:
            for a in self.harmonics_artists:
                a.remove()
            self.harmonics_artists = []
        if not keep_div:
            self.harmonics_div = 1                
        self.ax.get_figure().canvas.draw()


class FrequenciesPlot(QObject):
    
    sigEODFreq = Signal(float)
    sigEODFreqs = Signal(float, float)
    
    def __init__(self, freqs):
        super().__init__()
        self.canvas = FigureCanvas(Figure(figsize=(10, 5),
                                          layout='constrained'))
        self.canvas.mpl_connect('button_release_event', self.onrelease)
        self.navi = NavigationToolbar(self.canvas)
        self.navi.hide()
        self.axs = self.canvas.figure.subplots(1, 3, sharex=True, sharey=True)
        self.freqs = np.sort(freqs)
        # deltafs:
        ax = self.axs[0]
        ax.set_title('Differences $\\Delta f$')
        deltafs = self.freqs.reshape(-1, 1) - self.freqs.reshape(1, -1)
        vmax = np.max(np.abs(deltafs))
        cma = ax.pcolormesh(deltafs[::-1, :], cmap='seismic',
                            vmin=-vmax, vmax=vmax)
        for r in range(deltafs.shape[0]):
            for c in range(deltafs.shape[1]):
                ax.text(c + 0.5, r + 0.5, f'{deltafs[-1 - r, c]:.1f}',
                        ha='center', va='center',
                        fontsize='large', clip_on=True,
                        bbox=dict(boxstyle='round,pad=0.1', ec='none',
                                  fc='white', alpha=0.8))
        ax.set_aspect('equal')
        ax.xaxis.set_major_locator(plt.FixedLocator(np.arange(len(self.freqs)) + 0.5))
        ax.xaxis.set_major_formatter(plt.FixedFormatter([f'{f:.1f}' for f in self.freqs]))
        ax.yaxis.set_major_locator(plt.FixedLocator(np.arange(len(self.freqs)) + 0.5))
        ax.yaxis.set_major_formatter(plt.FixedFormatter([f'{f:.1f}' for f in reversed(self.freqs)]))
        ax.set_xlabel('EOD$f_i$ [Hz]')
        ax.set_ylabel('EOD$f_j$ [Hz]')
        ax.get_figure().colorbar(cma, ax=ax, label='$\\Delta f$ [Hz]')
        # ratios:
        ax = self.axs[1]
        ax.set_title('Ratios $f_1/f_0$')
        ratios = self.freqs.reshape(-1, 1) / self.freqs.reshape(1, -1)
        cma = ax.pcolormesh(ratios[::-1, :], cmap='seismic', norm='log',
                            vmin=1/5, vmax=5)
        for r in range(ratios.shape[0]):
            for c in range(ratios.shape[1]):
                ax.text(c + 0.5, r + 0.5, f'{ratios[-1 - r, c]:.3f}',
                        ha='center', va='center',
                        fontsize='large', clip_on=True,
                        bbox=dict(boxstyle='round,pad=0.1', ec='none',
                                  fc='white', alpha=0.8))
        ax.set_aspect('equal')
        ax.xaxis.set_major_locator(plt.FixedLocator(np.arange(len(self.freqs)) + 0.5))
        ax.xaxis.set_major_formatter(plt.FixedFormatter([f'{f:.1f}' for f in self.freqs]))
        ax.yaxis.set_major_locator(plt.FixedLocator(np.arange(len(self.freqs)) + 0.5))
        ax.yaxis.set_major_formatter(plt.FixedFormatter([f'{f:.1f}' for f in reversed(self.freqs)]))
        ax.set_xlabel('EOD$f_i$ [Hz]')
        ax.get_figure().colorbar(cma, ax=ax, label='Ratio', ticks=[1/5, 1/2, 1, 2, 5], format='%g')
        # musical intervals:
        ax = self.axs[2]
        ax.set_title('Musical intervals')
        all_intervals = np.array([musical_intervals[k][0] for k in musical_intervals])
        if len(self.freqs) < 6:
            all_names = list(musical_intervals.keys())
        else:
            all_names = [musical_intervals_short[k] for k in musical_intervals]
        intervals = np.zeros(ratios.shape, dtype=int)
        diffs = np.zeros(ratios.shape)
        diff_fracs = np.zeros(ratios.shape)
        for r in range(ratios.shape[0]):
            for c in range(ratios.shape[1]):
                if r != c and 0.5 <= ratios[r, c] < 2.05:
                    if ratios[r, c] >= 0.98:
                        ratio = ratios[r, c]
                    else:
                        ratio = 1/ratios[r, c]
                    intervals[r, c] = np.argmin(np.abs(all_intervals - ratio))
                    diffs[r, c] = ratio - all_intervals[intervals[r, c]]
                    diff_fracs[r, c] = diffs[r, c]/all_intervals[intervals[r, c]]
                else:
                    intervals[r, c] = -1
                    diffs[r, c] = np.nan
                    diff_fracs[r, c] = np.nan
        cma = ax.pcolormesh(100*np.abs(diff_fracs[::-1, :]), cmap='YlOrRd_r',
                            vmin=0, vmax=1)
        for r in range(ratios.shape[0]):
            for c in range(ratios.shape[1]):
                if -1 - r != c and intervals[-1 - r, c] >= 0:
                    idx = intervals[-1 - r, c]
                    if len(self.freqs) < 6:
                        label = f'{all_intervals[idx]:.4f}\n{all_names[idx]}\n$\\Delta$={diffs[-1 - r, c]:.4f}\n{100*diff_fracs[-1 - r, c]:.1f}%'
                    else:
                        label = f'{all_names[idx]}\n{100*diff_fracs[-1 - r, c]:.1f}%'
                    ax.text(c + 0.5, r + 0.5, label,
                            ha='center', va='center',
                            fontsize='large', clip_on=True,
                            bbox=dict(boxstyle='round,pad=0.1', ec='none',
                                      fc='white', alpha=0.8))
        ax.set_aspect('equal')
        ax.xaxis.set_major_locator(plt.FixedLocator(np.arange(len(self.freqs)) + 0.5))
        ax.xaxis.set_major_formatter(plt.FixedFormatter([f'{f:.1f}' for f in self.freqs]))
        ax.yaxis.set_major_locator(plt.FixedLocator(np.arange(len(self.freqs)) + 0.5))
        ax.yaxis.set_major_formatter(plt.FixedFormatter([f'{f:.1f}' for f in reversed(self.freqs)]))
        ax.set_xlabel('EOD$f_i$ [Hz]')
        ax.get_figure().colorbar(cma, ax=ax, extend='max',
                                 format=PercentFormatter(decimals=1),
                                 label='Deviation from musical interval')
        
    def onrelease(self, event):
        if event.inaxes in self.axs:
            r = int(event.ydata)
            c = int(event.xdata)
            eodf1 = self.freqs[-1 - r]
            eodf2 = self.freqs[c]
            if event.button == MouseButton.LEFT:
                self.sigEODFreq.emit(eodf1)
            elif event.button == MouseButton.RIGHT:
                self.sigEODFreq.emit(eodf2)
            elif event.button == MouseButton.MIDDLE:
                self.sigEODFreqs.emit(eodf1, eodf2)
        

class EODPlot():

    def __init__(self, data, rate, mean_eod, spectrum, props, phases, unit):
        n_snippets = 10
        self.canvas = FigureCanvas(Figure(figsize=(10, 5),
                                          layout='constrained'))
        self.navi = NavigationToolbar(self.canvas)
        self.navi.hide()
        gs = self.canvas.figure.add_gridspec(2, 2)
        self.axe = self.canvas.figure.add_subplot(gs[:, 0])
        self.props = props
        self.mean_eod = mean_eod
        self.spectrum = spectrum
        self.duration = len(data)/rate
        if self.props['type'] == 'wave':
            plot_wave_eod(self.axe, self.mean_eod, self.props, phases,
                          unit=unit, **wave_eod_styles)
            self.axa = self.canvas.figure.add_subplot(gs[0, 1])
            self.axp = self.canvas.figure.add_subplot(gs[1, 1], sharex=self.axa)
            plot_wave_spectrum(self.axa, self.axp, self.spectrum, self.props,
                               unit=unit, **wave_spec_styles)
        else:
            plot_pulse_eod(self.axe, self.mean_eod, self.props, phases,
                           unit=unit, **pulse_eod_styles)
            if 'times' in self.props:
                plot_eod_snippets(self.axe, data, rate,
                                  self.mean_eod[0, 0], self.mean_eod[-1, 0],
                                  self.props['times'], n_snippets,
                                  self.props['flipped'],
                                  self.props['aoffs'], snippet_style)
            self.axs = self.canvas.figure.add_subplot(gs[:, 1])
            plot_pulse_spectrum(self.axs, self.spectrum, self.props,
                                **pulse_spec_styles)

    def synthesize(self, rate, duration=2.0):
        if self.props['type'] == 'wave':
            data = wavefish_eods(self.spectrum, self.spectrum[0, 1],
                                 rate=rate, duration=duration)
        else:
            data = pulsetrain(self.props['times'], self.mean_eod,
                              None, self.duration, rate, 0.05)
        return data

    def play(self, audio):
        rate = 44100.0
        data = self.synthesize(rate, 2.0)
        fade(data, rate, 0.1)
        audio.play(data, rate, blocking=False)


class TeeStringIO(StringIO):
    def __init__(self):
        super().__init__()
        self.orig_stdout = sys.stdout
        self.orig_stdout.flush()

    def write(self,*args, **kwargs):
        super().write(*args, **kwargs)
        self.orig_stdout.write(*args, **kwargs)

        
class ThunderfishDialog(QDialog):

    def __init__(self, time, data, unit, ampl_max,
                 power_freqs, power_times, powers,
                 channel, file_path, cfg, audio, parent,
                 *args, **kwargs):
        super().__init__(parent, *args, **kwargs)
        self.time = time
        self.rate = 1/np.mean(np.diff(self.time))
        self.data = data
        self.unit = unit
        self.ampl_max = ampl_max
        self.channel = channel
        self.cfg = cfg
        self.file_path = file_path
        self.navis = []
        self.audio = audio
        self.pulse_colors, self.pulse_markers = colors_markers()
        self.pulse_colors = self.pulse_colors[3:]
        self.pulse_markers = self.pulse_markers[3:]
        self.wave_colors, self.wave_markers = colors_markers()
        
        # collect stdout:
        orig_stdout = sys.stdout
        sys.stdout = TeeStringIO()
        # clipping amplitudes:
        self.min_clip, self.max_clip = \
            clip_amplitudes(self.data, max_ampl=self.ampl_max,
                            **clip_args(self.cfg, self.rate))
        # detect EODs in the data:
        power_freqs, powers, self.wave_eodfs, self.wave_indices, \
        self.eod_props, self.mean_eods, self.spec_data, self.phase_data, \
        self.pulse_data, power_thresh, self.skip_reason = \
          detect_eods(self.data, self.rate, power_freqs, power_times, powers,
                      min_clip=self.min_clip, max_clip=self.max_clip,
                      name=self.file_path, mode='wp',
                      verbose=2, plot_level=0, cfg=self.cfg)
        # add analysis window to EOD properties:
        for props in self.eod_props:
            props['twin'] = time[0]
            props['window'] = time[-1] - time[0]
        self.eodfs = np.array([props['EODf'] for props in self.eod_props])
        self.nwave = 0
        self.npulse = 0
        for i in range(len(self.eod_props)):
            if self.eod_props[i]['type'] == 'pulse':
                self.npulse += 1
            elif self.eod_props[i]['type'] == 'wave':
                self.nwave += 1
        self.neods = self.nwave + self.npulse
        # read out stdout:
        log = sys.stdout.getvalue()
        sys.stdout = orig_stdout
        
        # dialog:
        vbox = QVBoxLayout(self)
        self.tabs = QTabWidget(self)
        self.tabs.setDocumentMode(True)
        self.tabs.setMovable(True)
        self.tabs.setTabsClosable(False)
        self.trace_acts = []
        self.tabs.currentChanged.connect(self.toggle_trace)
        vbox.addWidget(self.tabs)

        # log messages:
        self.log = QLabel(self)
        self.log.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.log.setText(log)
        self.log.setFont(QFont('monospace'))
        self.log.setMinimumSize(self.log.sizeHint())
        self.scroll = QScrollArea(self)
        self.scroll.setWidget(self.log)
        self.tabs.addTab(self.scroll, 'Log')

        # plots:
        plt.rcParams['axes.spines.top'] = False
        plt.rcParams['axes.spines.right'] = False
        
        # tab with recording trace:
        self.trace_plot = TracePlot(self.time, self.data, self.unit,
                                    self.eod_props, self.wave_eodfs,
                                    self.pulse_colors, self.pulse_markers)
        self.navis.append(self.trace_plot.navi)
        self.trace_idx = self.tabs.addTab(self.trace_plot.canvas, 'Trace')
        
        # tab with pulse rates:
        if self.npulse > 0:
            self.rate_plot = RatePlot(self.time, self.eod_props, self.pulse_colors)
            self.rate_plot.ax.set_xlim(*self.trace_plot.ax.get_xlim())
            self.navis.append(self.rate_plot.navi)
            self.rate_idx = self.tabs.addTab(self.rate_plot.canvas, 'Rate')
        else:
            self.rate_plot = None
            self.rate_idx = None

        # tab with power spectrum:
        self.power_plot = PowerPlot(power_freqs, powers, power_thresh,
                                    self.wave_eodfs, self.wave_indices,
                                    self.wave_colors, self.wave_markers)
        self.navis.append(self.power_plot.navi)
        self.spec_idx = self.tabs.addTab(self.power_plot.canvas, 'Spectrum')

        # tab with frequencies:
        if len(self.eodfs) > 1:
            self.freqs_plot = FrequenciesPlot(self.eodfs)
            self.navis.append(self.freqs_plot.navi)
            self.freqs_idx = self.tabs.addTab(self.freqs_plot.canvas,
                                              'Frequencies')
            self.freqs_plot.sigEODFreq.connect(self.raise_and_play)
            self.freqs_plot.sigEODFreqs.connect(self.play_interval)
        else:
            self.freqs_plot = None
            self.freqs_idx = None

        # set current plot:
        if self.nwave > self.npulse:
            self.tabs.setCurrentIndex(self.spec_idx)
        else:
            self.tabs.setCurrentIndex(self.trace_idx)

        self.eod_tabs = None
        if len(self.eod_props) > 0:
            # tabs of EODs:
            self.eod_tabs = QTabWidget(self)
            self.eod_tabs.setDocumentMode(True)
            self.eod_tabs.setMovable(True)
            self.eod_tabs.setTabBarAutoHide(False)
            self.eod_tabs.setTabsClosable(False)
            vbox.addWidget(self.eod_tabs)

            # plot EODs:
            inx = np.argsort(self.eodfs)
            self.eod_plots = []
            for i, k in enumerate(inx):
                eod_plot = EODPlot(self.data, self.rate, self.mean_eods[k],
                                   self.spec_data[k], self.eod_props[k],
                                   self.phase_data[k], self.unit)
                self.eod_plots.append(eod_plot)
                self.navis.append(eod_plot.navi)
                self.eod_tabs.addTab(eod_plot.canvas,
                                     f'{i}: {self.eod_props[k]['EODf']:.1f}Hz')
            # sort EOD frequencies:
            self.eodfs = self.eodfs[inx]

        self.tools = self.setup_toolbar()
        close = QPushButton('&Close', self)
        close.pressed.connect(self.accept)
        QShortcut('q', self).activated.connect(close.animateClick)
        QShortcut('Ctrl+Q', self).activated.connect(close.animateClick)
        hbox = QHBoxLayout()
        hbox.setContentsMargins(0, 0, 0, 0)
        hbox.addWidget(self.tools)
        hbox.addWidget(QLabel())
        hbox.addWidget(close)
        vbox.addLayout(hbox)

    def resizeEvent(self, event):
        if self.eod_tabs is None:
            super().resizeEvent(event)
        else:
            h = (event.size().height() - self.tools.height())//2 - 10
            self.tabs.setMaximumHeight(h)
            self.eod_tabs.setMaximumHeight(h)

    def toggle_maximize(self):
        if self.isMaximized():
            self.showNormal()
        else:
            self.showMaximized()

    def toggle_fullscreen(self):
        if self.isFullScreen():
            self.showNormal()
        else:
            self.showFullScreen()

    def save(self):
        base_name = self.file_path.with_suffix('.zip')
        cstr = f'-c{self.channel}'
        tstr = f'-t{self.time[0]:.0f}s'
        base_name = base_name.with_stem(base_name.stem + cstr + tstr)
        filters = ['All files (*)', 'ZIP files (*.zip)']
        base_name = QFileDialog.getSaveFileName(self, 'Save analysis as',
                                                os.fspath(base_name),
                                                ';;'.join(filters))[0]
        if base_name:
            save_analysis(base_name, True, self.eod_props,
                          self.mean_eods, self.spec_data,
                          self.phase_data, self.pulse_data,
                          self.wave_eodfs, self.wave_indices, self.unit, 0,
                          **write_table_args(self.cfg))

    def play(self):
        if self.audio.active():
            self.audio.stop()
        else:
            playdata = np.array(self.data) - np.mean(self.data)
            fade(playdata, self.rate, 0.1)
            self.audio.play(playdata, self.rate, blocking=False)

    def play_fish(self):
        if self.audio.active():
            self.audio.stop()
        else:
            self.eod_plots[self.eod_tabs.currentIndex()].play(self.audio)

    def home(self):
        for n in self.navis:
            n.home()

    def back(self):
        for n in self.navis:
            n.back()
            
    def forward(self):
        for n in self.navis:
            n.forward()

    def zoom(self):
        for n in self.navis:
            n.zoom()

    def pan(self):
        for n in self.navis:
            n.pan()

    def dispatch_trace(self, func):
        if self.tabs.currentIndex() in [self.trace_idx, self.rate_idx]:
            getattr(self.trace_plot, func)()
            if self.rate_plot is not None:
                getattr(self.rate_plot, func)()

    def toggle_trace(self, index):
        for act in self.trace_acts:
            act.setEnabled(index in [self.trace_idx, self.rate_idx])
        
    def setup_toolbar(self):
        tools = QToolBar(self)

        act = QAction('&Play', self)
        act.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
        act.setToolTip('Play (Space)')
        act.setShortcut(' ')
        act.triggered.connect(self.play)
        tools.addAction(act)

        act = QAction('Play fish', self)
        act.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
        act.setToolTip('Play fish (Ctrl + Space)')
        act.setShortcut('Ctrl+Space')
        act.triggered.connect(self.play_fish)
        tools.addAction(act)
        
        tools.addSeparator()
        
        act = QAction('&Home', self)
        act.setIcon(self.style().standardIcon(QStyle.SP_DirHomeIcon))
        act.setToolTip('Reset zoom (h, Home)')
        act.setShortcuts(['h', 'r'])
        act.triggered.connect(self.home)
        tools.addAction(act)
        
        act = QAction('&Back', self)
        act.setIcon(self.style().standardIcon(QStyle.SP_ArrowBack))
        act.setToolTip('Zoom backward (c)')
        act.setShortcuts(['c', Qt.Key_Backspace])
        act.triggered.connect(self.back)
        tools.addAction(act)

        act = QAction('&Forward', self)
        act.setIcon(self.style().standardIcon(QStyle.SP_ArrowForward))
        act.setToolTip('Zoom forward (v)')
        act.setShortcuts(['v'])
        act.triggered.connect(self.forward)
        tools.addAction(act)

        act = QAction('&Zoom', self)
        #act.setIcon(self.style().standardIcon(QStyle.SP_TitleBarMaxButton))
        act.setToolTip('Rectangular zoom (o)')
        act.setShortcuts(['o'])
        act.triggered.connect(self.zoom)
        tools.addAction(act)
        
        act = QAction('&Pan', self)
        #act.setIcon(self.style().standardIcon(QStyle.SP_DirHomeIcon))
        act.setToolTip('Pan and zoom (p)')
        act.setShortcuts(['p'])
        act.triggered.connect(self.pan)
        tools.addAction(act)
        
        tools.addSeparator()
        
        act = QAction('Trace', self)
        #act.setIcon(self.style().standardIcon(QStyle.SP_DirHomeIcon))
        act.setToolTip('Show all time in trace plot (a)')
        act.setShortcuts(['a'])
        act.triggered.connect(lambda x: self.dispatch_trace('toggle_time_range'))
        tools.addAction(act)
        self.trace_acts.append(act)
        
        act = QAction('Home', self)
        act.setIcon(self.style().standardIcon(QStyle.SP_MediaSkipBackward))
        act.setToolTip('Skip to beginning of trace plot (Home)')
        act.setShortcuts([QKeySequence.MoveToStartOfLine, QKeySequence.MoveToStartOfDocument])
        act.triggered.connect(lambda x: self.dispatch_trace('home'))
        tools.addAction(act)
        self.trace_acts.append(act)
        
        act = QAction('Seek backward', self)
        act.setIcon(self.style().standardIcon(QStyle.SP_MediaSeekBackward))
        act.setToolTip('Seek backward in trace plot (Page up)')
        act.setShortcuts([QKeySequence.MoveToPreviousPage])
        act.triggered.connect(lambda x: self.dispatch_trace('move_backward'))
        tools.addAction(act)
        self.trace_acts.append(act)
        
        act = QAction('Seek forward', self)
        act.setIcon(self.style().standardIcon(QStyle.SP_MediaSeekForward))
        act.setToolTip('Seek forward in trace plot (Page down)')
        act.setShortcuts([QKeySequence.MoveToNextPage])
        act.triggered.connect(lambda x: self.dispatch_trace('move_forward'))
        tools.addAction(act)
        self.trace_acts.append(act)
        
        act = QAction('End', self)
        act.setIcon(self.style().standardIcon(QStyle.SP_MediaSkipForward))
        act.setToolTip('Skip to end of trace plot (End)')
        act.setShortcuts([QKeySequence.MoveToEndOfLine, QKeySequence.MoveToEndOfDocument])
        act.triggered.connect(lambda x: self.dispatch_trace('end'))
        tools.addAction(act)
        self.trace_acts.append(act)
        
        act = QAction('+', self)
        #act.setIcon(self.style().standardIcon(QStyle.SP_DirHomeIcon))
        act.setToolTip('Zoom in to trace plot (+)')
        act.setShortcuts([QKeySequence.ZoomIn, '+', '='])
        act.triggered.connect(lambda x: self.dispatch_trace('zoom_in'))
        tools.addAction(act)
        self.trace_acts.append(act)
        
        act = QAction('-', self)
        #act.setIcon(self.style().standardIcon(QStyle.SP_DirHomeIcon))
        act.setToolTip('Zoom out of trace plot (-)')
        act.setShortcuts([QKeySequence.ZoomOut, '-'])
        act.triggered.connect(lambda x: self.dispatch_trace('zoom_out'))
        tools.addAction(act)
        self.trace_acts.append(act)
        
        tools.addSeparator()

        act = QAction('&Maximize', self)
        act.setIcon(self.style().standardIcon(QStyle.SP_TitleBarMaxButton))
        act.setToolTip('Maximize window (m)')
        act.setShortcuts(['m', 'Ctrl+M', 'Ctrl+Shift+M'])
        act.triggered.connect(self.toggle_maximize)
        tools.addAction(act)

        act = QAction('&Fullscreen', self)
        act.setToolTip('Fullscreen window (f)')
        act.setShortcuts(['f'])
        act.triggered.connect(self.toggle_fullscreen)
        tools.addAction(act)

        tools.addSeparator()

        act = QAction('&Save', self)
        act.setIcon(self.style().standardIcon(QStyle.SP_DialogSaveButton))
        act.setToolTip('Save analysis results to zip file (s)')
        act.setShortcuts(['s', 'CTRL+S'])
        act.triggered.connect(self.save)
        tools.addAction(act)

        return tools

    def raise_and_play(self, eodf):
        inx = np.argmin(np.abs(self.eodfs - eodf))
        self.eod_tabs.setCurrentIndex(inx)
        self.eod_plots[inx].play(self.audio)

    def play_interval(self, eodf1, eodf2):
        rate = 44100.0
        inx1 = np.argmin(np.abs(self.eodfs - eodf1))
        self.eod_tabs.setCurrentIndex(inx1)
        data1 = self.eod_plots[inx1].synthesize(rate, 2.0)
        inx2 = np.argmin(np.abs(self.eodfs - eodf2))
        data2 = self.eod_plots[inx2].synthesize(rate, 2.0)
        n = min(len(data1), len(data2))
        playdata = data1[:n] + data2[:n]
        fade(playdata, rate, 0.1)
        self.audio.play(playdata, rate, blocking=False)
            

class ThunderfishAnalyzer(Analyzer):
    
    def __init__(self, browser):
        super().__init__(browser, 'thunderfish', 'filtered')
        self.dialog = None
        # configure:
        cfgfile = Path(__package__ + '.cfg')
        self.cfg = configuration()
        self.cfg.load_files(cfgfile, browser.data.file_path, 4)
        self.cfg.set('unwrapData', browser.data.data.unwrap)
        
    def analyze(self, t0, t1, channel, traces):
        time, data = traces[self.source_name]
        freqs = None
        spec = None
        if 'spectrogram' in traces:
            times, freqs, spec = traces['spectrogram']
        dialog = ThunderfishDialog(time, data, self.source.unit,
                                   self.source.ampl_max,
                                   freqs, times, spec, channel,
                                   self.browser.data.file_path,
                                   self.cfg, self.browser.audio, self.browser)
        dialog.show()


def audian_analyzer(browser):
    browser.remove_analyzer('plain')
    browser.remove_analyzer('statistics')
    ThunderfishAnalyzer(browser)


def main():
    plugins = Plugins()
    plugins.add_analyzer_factory(audian_analyzer)
    audian_cli(sys.argv[1:], plugins)


if __name__ == '__main__':
    main()
