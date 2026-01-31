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

from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvas
from matplotlib.backends.backend_qtagg import \
    NavigationToolbar2QT as NavigationToolbar
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QDialog, QShortcut, QVBoxLayout, QWidget, QTabWidget

from .thunderfish import configuration, detect_eods
from .eodanalysis import plot_eod_waveform
from .eodanalysis import plot_wave_spectrum, plot_pulse_spectrum


class ThunderfishAnalyzer(Analyzer):
    
    def __init__(self, browser, source_name):
        super().__init__(browser, 'thunderfish', source_name)
        self.dialog = None

        
    def analyze(self, t0, t1, channel, traces):
        time, data = traces[self.source_name]
        # detect EODs in the data:
        rate = 1/np.mean(np.diff(time))
        cfg = configuration()
        psd_data, wave_eodfs, wave_indices, eod_props, \
        mean_eods, spec_data, phase_data, pulse_data, power_thresh, skip_reason, zoom_window = \
          detect_eods(data, rate, min_clip=-1, max_clip=1, name='test', mode='wp',
                      verbose=1, plot_level=0, cfg=cfg)
        # dialog:
        self.dialog = QDialog(self.browser)
        self.dialog.finished.connect(lambda x: [None for self.dialog in [None]])
        QShortcut('q', self.dialog).activated.connect(self.dialog.accept)
        QShortcut('Ctrl+Q', self.dialog).activated.connect(self.dialog.accept)

        vbox = QVBoxLayout(self.dialog)
        tabs = QTabWidget(self.dialog)
        tabs.setDocumentMode(True)
        tabs.setMovable(True)
        tabs.setTabBarAutoHide(False)
        tabs.setTabsClosable(False)
        vbox.addWidget(tabs)
        # plot:
        for k in range(len(eod_props)):
            w = QWidget(self.dialog)
            canvas = FigureCanvas(Figure(figsize=(10, 5), layout='constrained'))
            navi = NavigationToolbar(canvas, w)
            QShortcut('h', w).activated.connect(navi.home)
            QShortcut('r', w).activated.connect(navi.home)
            QShortcut(Qt.Key_Home, w).activated.connect(navi.home)
            QShortcut('c', w).activated.connect(navi.back)
            QShortcut(Qt.Key_Backspace, w).activated.connect(navi.back)
            QShortcut('v', w).activated.connect(navi.forward)
            QShortcut('o', w).activated.connect(navi.zoom)
            QShortcut('p', w).activated.connect(navi.pan)
            QShortcut('s', w).activated.connect(navi.save_figure)
            QShortcut('Ctrl+S', w).activated.connect(navi.save_figure)
            vbox = QVBoxLayout(w)
            vbox.addWidget(canvas)
            vbox.addWidget(navi)
            tabs.addTab(w, f'EODf={eod_props[k]['EODf']:.0f}Hz')
            gs = canvas.figure.add_gridspec(2, 2)
            axe = canvas.figure.add_subplot(gs[:, 0])
            plot_eod_waveform(axe, mean_eods[k], eod_props[k], phase_data[k],
                              unit='')
            if eod_props[k]['type'] == 'wave':
                axa = canvas.figure.add_subplot(gs[0, 1])
                axp = canvas.figure.add_subplot(gs[1, 1], sharex=axa)
                plot_wave_spectrum(axa, axp, spec_data[k], eod_props[k])
            else:
                axs = canvas.figure.add_subplot(gs[:, 1])
                plot_pulse_spectrum(axs, spec_data[k], eod_props[k])
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
