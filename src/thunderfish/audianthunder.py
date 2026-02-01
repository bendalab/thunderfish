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

from pathlib import Path
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvas
from matplotlib.backends.backend_qtagg import \
    NavigationToolbar2QT as NavigationToolbar
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QDialog, QShortcut, QVBoxLayout
from PyQt5.QtWidgets import QWidget, QTabWidget, QToolBar, QAction, QStyle

from thunderlab.powerspectrum import plot_decibel_psd, multi_psd
from .thunderfish import configuration, detect_eods
from .thunderfish import rec_style, spectrum_style, eod_styles, snippet_style
from .thunderfish import wave_spec_styles, pulse_spec_styles
from .bestwindow import clip_args, clip_amplitudes
from .harmonics import colors_markers, plot_harmonic_groups
from .eodanalysis import plot_eod_recording, plot_pulse_eods
from .eodanalysis import plot_eod_waveform, plot_eod_snippets
from .eodanalysis import plot_wave_spectrum, plot_pulse_spectrum


class ThunderfishDialog(QDialog):

    def __init__(self, time, data, unit, ampl_max, file_path, cfg,
                 parent, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)
        self.time = time
        self.rate = 1/np.mean(np.diff(self.time))
        self.data = data
        self.unit = unit
        self.ampl_max = ampl_max
        self.cfg = cfg
        self.file_path = file_path
        self.navis = []
        self.pulse_colors, self.pulse_markers = colors_markers()
        self.pulse_colors = self.pulse_colors[3:]
        self.pulse_markers = self.pulse_markers[3:]
        self.wave_colors, self.wave_markers = colors_markers()
        # clipping amplitudes:
        self.min_clip, self.max_clip = \
            clip_amplitudes(self.data, max_ampl=self.ampl_max,
                            **clip_args(self.cfg, self.rate))
        # detect EODs in the data:
        psd_data, wave_eodfs, wave_indices, eod_props, \
        mean_eods, spec_data, phase_data, pulse_data, power_thresh, skip_reason, zoom_window = \
          detect_eods(self.data, self.rate,
                      min_clip=self.min_clip, max_clip=self.max_clip,
                      name=self.file_path, mode='wp',
                      verbose=1, plot_level=0, cfg=self.cfg)
        self.nwave = 0
        self.npulse = 0
        for i in range(len(eod_props)):
            if eod_props[i]['type'] == 'pulse':
                self.npulse += 1
            elif eod_props[i]['type'] == 'wave':
                self.nwave += 1
        self.neods = self.nwave + self.npulse
        
        # dialog:
        QShortcut('q', self).activated.connect(self.accept)
        QShortcut('Ctrl+Q', self).activated.connect(self.accept)
        self.tabs = QTabWidget(self)
        self.tabs.setDocumentMode(True)
        self.tabs.setMovable(True)
        self.tabs.setTabsClosable(False)
        
        # tab with recording trace:
        canvas = FigureCanvas(Figure(figsize=(10, 5), layout='constrained'))
        navi = NavigationToolbar(canvas, self)
        navi.hide()
        self.navis.append(navi)
        trace_idx = self.tabs.addTab(canvas, 'Trace')
        ax = canvas.figure.subplots()
        twidth = 0.1
        if len(eod_props) > 0:
            if eod_props[0]['type'] == 'wave':
                twidth = 5.0/eod_props[0]['EODf']
            else:
                if len(wave_eodfs) > 0:
                    twidth = 3.0/eod_props[0]['EODf']
                else:
                    twidth = 10.0/eod_props[0]['EODf']
        twidth = (1+twidth//0.005)*0.005
        plot_eod_recording(ax, self.data, self.rate, self.unit,
                           twidth, time[0], rec_style)
        zoom_window = [1.2, 1.3]
        plot_pulse_eods(ax, self.data, self.rate, zoom_window,
                        twidth, eod_props, time[0],
                        colors=self.pulse_colors,
                        markers=self.pulse_markers,
                        frameon=True, loc='upper right')
        if ax.get_legend() is not None:
            ax.get_legend().get_frame().set_color('white')

        # tab with power spectrum:
        canvas = FigureCanvas(Figure(figsize=(10, 5), layout='constrained'))
        navi = NavigationToolbar(canvas, self)
        navi.hide()
        self.navis.append(navi)
        spec_idx = self.tabs.addTab(canvas, 'Spectrum')
        ax = canvas.figure.subplots()
        if len(wave_eodfs) > 0:
            plot_harmonic_groups(ax, wave_eodfs, wave_indices, max_groups=0,
                                 skip_bad=True,
                                 sort_by_freq=True, label_power=True,
                                 colors=self.wave_colors,
                                 markers=self.wave_markers,
                                 frameon=False, loc='upper right')
        psd_data = multi_psd(self.data, self.rate,
                             1.1*eod_props[0]['dfreq'])[0]
        plot_decibel_psd(ax, psd_data[:, 0], psd_data[:, 1],
                         log_freq=False, min_freq=0, max_freq=3000,
                         ymarg=5.0, sstyle=spectrum_style)
        ax.yaxis.set_major_locator(plt.MaxNLocator(6))
        if self.nwave > self.npulse:
            self.tabs.setCurrentIndex(spec_idx)
        else:
            self.tabs.setCurrentIndex(trace_idx)

        # tabs of EODs:
        self.eod_tabs = QTabWidget(self)
        self.eod_tabs.setDocumentMode(True)
        self.eod_tabs.setMovable(True)
        self.eod_tabs.setTabBarAutoHide(False)
        self.eod_tabs.setTabsClosable(False)
        tools = self.setup_toolbar()
        vbox = QVBoxLayout(self)
        vbox.addWidget(self.tabs)
        vbox.addWidget(self.eod_tabs)
        vbox.addWidget(tools)
        
        # plot EODs:
        for k in range(len(eod_props)):
            props = eod_props[k]
            n_snippets = 10
            canvas = FigureCanvas(Figure(figsize=(10, 5), layout='constrained'))
            navi = NavigationToolbar(canvas, self)
            navi.hide()
            self.navis.append(navi)
            self.eod_tabs.addTab(canvas, f'EODf={eod_props[k]['EODf']:.0f}Hz')
            gs = canvas.figure.add_gridspec(2, 2)
            axe = canvas.figure.add_subplot(gs[:, 0])
            plot_eod_waveform(axe, mean_eods[k], eod_props[k], phase_data[k],
                              unit=self.unit, **eod_styles)
            if props['type'] == 'pulse' and 'times' in props:
                plot_eod_snippets(axe, self.data, self.rate,
                                  mean_eods[k][0, 0], mean_eods[k][-1, 0],
                                  props['times'], n_snippets, props['flipped'],
                                  props['aoffs'], snippet_style)
            if props['type'] == 'wave':
                axa = canvas.figure.add_subplot(gs[0, 1])
                axp = canvas.figure.add_subplot(gs[1, 1], sharex=axa)
                plot_wave_spectrum(axa, axp, spec_data[k], props,
                                   unit=self.unit, **wave_spec_styles)
            else:
                axs = canvas.figure.add_subplot(gs[:, 1])
                plot_pulse_spectrum(axs, spec_data[k], props,
                                    **pulse_spec_styles)

    def resizeEvent(self, event):
        h = event.size().height()//2 - 10
        self.tabs.setMaximumHeight(h)
        self.eod_tabs.setMaximumHeight(h)

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

    def setup_toolbar(self):
        tools = QToolBar(self)
        act = QAction('&Home', self)
        act.setIcon(self.style().standardIcon(QStyle.SP_DirHomeIcon))
        act.setToolTip('Reset zoom (h, Home)')
        act.setShortcuts(['h', 'r', Qt.Key_Home])
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

        return tools
            

class ThunderfishAnalyzer(Analyzer):
    
    def __init__(self, browser, source_name):
        super().__init__(browser, 'thunderfish', source_name)
        self.dialog = None
        # configure:
        cfgfile = Path(__package__ + '.cfg')
        self.cfg = configuration()
        self.cfg.load_files(cfgfile, browser.data.file_path, 4)
        self.cfg.set('unwrapData', browser.data.data.unwrap)
        
    def analyze(self, t0, t1, channel, traces):
        time, data = traces[self.source_name]
        self.dialog = ThunderfishDialog(time, data, self.source.unit,
                                        self.source.ampl_max,
                                        self.browser.data.file_path,
                                        self.cfg, self.browser)
        self.dialog.finished.connect(lambda x: [None for self.dialog in [None]])
        self.dialog.show()


def audian_analyzer(browser):
    browser.remove_analyzer('plain')
    browser.remove_analyzer('statistics')
    ThunderfishAnalyzer(browser, 'data')


def main():
    plt.rcParams['axes.spines.top'] = False
    plt.rcParams['axes.spines.right'] = False
    plugins = Plugins()
    plugins.add_analyzer_factory(audian_analyzer)
    audian_cli(sys.argv[1:], plugins)


if __name__ == '__main__':
    main()
