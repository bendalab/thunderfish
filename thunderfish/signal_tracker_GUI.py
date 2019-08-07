import sys
import os
import time
import datetime
import numpy as np
import matplotlib.pyplot as plt

from IPython import embed

from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5 import QtCore
from PyQt5.QtCore import *
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas

# from thunderfish.powerspectrum import spectrogram, next_power_of_two, decibel

from matplotlib.widgets import RectangleSelector, EllipseSelector


def decibel(power, ref_power=1.0, min_power=1e-20):
    """
    Transform power to decibel relative to ref_power.
    ```
    decibel = 10 * log10(power/ref_power)
    ```
    Power values smaller than `min_power` are set to `-np.inf`.

    Parameters
    ----------
    power: float or array
        Power values, for example from a power spectrum or spectrogram.
    ref_power: float or None
        Reference power for computing decibel.
        If set to `None` the maximum power is used.
    min_power: float
        Power values smaller than `min_power` are set to `-inf`.

    Returns
    -------
    decibel_psd: array
        Power values in decibel relative to `ref_power`.
    """
    if isinstance(power, (list, tuple, np.ndarray)):
        tmp_power = power
        decibel_psd = power.copy()
    else:
        tmp_power = np.array([power])
        decibel_psd = np.array([power])
    if ref_power is None:
        ref_power = np.max(decibel_psd)
    decibel_psd[tmp_power <= min_power] = float('-inf')
    decibel_psd[tmp_power > min_power] = 10.0 * np.log10(decibel_psd[tmp_power > min_power]/ref_power)
    if isinstance(power, (list, tuple, np.ndarray)):
        return decibel_psd
    else:
        return decibel_psd[0]


class SubWindow1(QMainWindow):
    def __init__(self):
        super().__init__()

        self.initMe()

    def initMe(self):
        self.setGeometry(300, 150, 200, 200)  # set window proportion
        self.setWindowTitle('E-Fish Tracker')
        self.show()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initMe()

    def initMe(self):
        # implement status Bar
        self.statusBar().showMessage('Welcome to FishLab')

        # MenuBar
        self.init_Actions()

        self.init_MenuBar()

        self.init_ToolBar()

        self.init_figure()

        self.init_var()

        # ToDo: set to auto ?!
        self.setGeometry(200, 50, 1200, 800)  # set window proportion
        self.setWindowTitle('FishLab v1.0')  # set window title

        # ToDo: create icon !!!
        # self.setWindowIcon(QIcon('<path>'))  # set window image (left top)

        central_widget = QWidget(self)

        self.button = QPushButton('Open', central_widget)
        self.button.clicked.connect(self.open)
        self.button2 = QPushButton('Load', central_widget)
        self.button2.clicked.connect(self.load)

        self.gridLayout = QGridLayout()

        self.gridLayout.addWidget(self.canvas, 0, 0, 4, 5)
        self.gridLayout.addWidget(self.button, 4, 1)
        self.gridLayout.addWidget(self.button2, 4, 3)
        # self.setLayout(v)

        central_widget.setLayout(self.gridLayout)
        self.setCentralWidget(central_widget)
        # self.show()  # show the window


        # self.ax.plot(range(10), range(10))
        self.figure.canvas.draw()

    def init_var(self):
        self.spec_img_handle = None
        self.trace_handles = []

    def init_figure(self):
        self.figure = plt.figure()
        self.canvas = FigureCanvas(self.figure)
        self.figure.canvas.mpl_connect('button_press_event', self.buttonpress)
        self.figure.canvas.mpl_connect('button_release_event', self.buttonrelease)

        self.ax = self.figure.add_subplot(111)

    def init_ToolBar(self):
        toolbar = self.addToolBar('TB')  # toolbar needs QMainWindow ?!
        self.addToolBar(Qt.LeftToolBarArea, toolbar)
        toolbar2 = self.addToolBar('TB2')  # toolbar needs QMainWindow ?!
        self.addToolBar(Qt.LeftToolBarArea, toolbar2)

        toolbar3 = self.addToolBar('TB3')  # toolbar needs QMainWindow ?!
        self.addToolBar(Qt.LeftToolBarArea, toolbar3)

        toolbar4 = self.addToolBar('TB4')  # toolbar needs QMainWindow ?!
        self.addToolBar(Qt.LeftToolBarArea, toolbar4)

        # toolbar.addAction(self.Act_interactive_sel)
        toolbar.addAction(self.Act_interactive_sel)
        toolbar.addAction(self.Act_interactive_con)
        toolbar.addAction(self.Act_interactive_GrCon)
        toolbar.addAction(self.Act_interactive_cut)
        toolbar.addAction(self.Act_interactive_del)
        toolbar.addAction(self.Act_interactive_GrDel)

        toolbar2.addAction(self.Act_interactive_zoom)
        toolbar2.addAction(self.Act_interactive_zoom_in)
        toolbar2.addAction(self.Act_interactive_zoom_out)
        toolbar2.addAction(self.Act_interactive_zoom_home)

        toolbar3.addAction(self.Act_interactive_AutoSort)
        toolbar3.addAction(self.Act_interactive_ManualSort)
        toolbar3.addAction(self.Act_undo)

        toolbar4.addAction(self.Act_fine_spec)
        toolbar4.addAction(self.Act_norm_spec)

    def init_MenuBar(self):
        menubar = self.menuBar() # needs QMainWindow ?!
        file = menubar.addMenu('&File') # create file menu ... accessable with alt+F
        file.addActions([self.Act_open, self.Act_load, self.Act_save, self.Act_exit])

        edit = menubar.addMenu('&Edit')
        edit.addActions([self.Act_undo, self.Act_embed])

        settings = menubar.addMenu('&Settings')
        settings.addActions([self.Act_set_psd, self.Act_set_track, self.Act_set_gridLayout])

        tracking = menubar.addMenu('&Tracking')
        tracking.addActions([self.Act_EODtrack, self.Act_Positontrack])

        spectrogram = menubar.addMenu('&Spectrogram')
        spectrogram.addActions([self.Act_compSpec])

    def init_Actions(self):
        #################### Menu ####################
        # --- MenuBar - File --- #
        self.Act_open = QAction('&Open', self)
        self.Act_open.setStatusTip('Open file')
        self.Act_open.triggered.connect(self.open)

        self.Act_load = QAction('&Load', self)
        self.Act_load.setStatusTip('Load traces')
        self.Act_load.setEnabled(False)
        self.Act_load.triggered.connect(self.load)


        self.Act_save = QAction('&Save', self)
        self.Act_save.setEnabled(False)
        self.Act_save.setStatusTip('Save traces')

        # exitMe = QAction(QIcon('<path>'), '&Exit', self)
        self.Act_exit = QAction('&Exit', self)  # trigger with alt+E
        self.Act_exit.setShortcut('ctrl+Q')
        self.Act_exit.setStatusTip('Terminate programm')
        self.Act_exit.triggered.connect(self.close)

        # --- MenuBar - Edit --- #

        self.Act_undo = QAction(QIcon('./gui_sym/undo.png'), '&Undo', self)
        self.Act_undo.setStatusTip('Undo last sorting step')

        self.Act_embed = QAction('&Embed', self)
        self.Act_embed.setStatusTip('Go to shell')
        self.Act_embed.triggered.connect(embed)

        # --- MenuBar - Edit --- #

        self.Act_set_psd = QAction('PSD settings', self)
        self.Act_set_psd.setStatusTip('set PSD settings')

        self.Act_set_track = QAction('Tracking settings', self)
        self.Act_set_track.setStatusTip('set tracking settings')

        self.Act_set_gridLayout = QAction('Grid layout', self)
        self.Act_set_gridLayout.setStatusTip('define grid layout')

        # --- MenuBar - Tracking --- #

        self.Act_EODtrack = QAction('Track EOD traces', self)
        self.Act_EODtrack.setStatusTip('track EOD traces')

        self.Act_Positontrack = QAction('Track position', self)
        self.Act_Positontrack.setStatusTip('track fish locations')

        # --- MenuBar - Spectrogram --- #

        self.Act_compSpec = QAction('Compute full spectrogram', self)
        self.Act_compSpec.setStatusTip('compute full detailed spectrogram')

        ################## ToolBar ###################

        self.Act_interactive_sel = QAction(QIcon('./gui_sym/sel.png'), 'S', self)
        self.Act_interactive_sel.setCheckable(True)

        # self.Act_*.setChecked(False)
        # self.Act_xx.setChecked(True)

        self.Act_interactive_con = QAction(QIcon('./gui_sym/con.png'), 'Connect', self)
        self.Act_interactive_GrCon = QAction(QIcon('./gui_sym/GrCon.png'), 'Group Connect', self)
        self.Act_interactive_del = QAction(QIcon('./gui_sym/del.png'), 'Delete Trace', self)
        self.Act_interactive_GrDel = QAction(QIcon('./gui_sym/GrDel.png'), 'Group Delete', self)
        self.Act_interactive_AutoSort = QAction(QIcon('./gui_sym/auto.png'), 'Auto Connect', self)
        self.Act_interactive_ManualSort = QAction(QIcon('./gui_sym/manuel.png'), 'Manual Connect', self)
        self.Act_interactive_zoom_out = QAction(QIcon('./gui_sym/zoomout.png'), 'Zoom -', self)
        self.Act_interactive_zoom_in = QAction(QIcon('./gui_sym/zoomin.png'), 'zoom +', self)
        self.Act_interactive_zoom_home = QAction(QIcon('./gui_sym/zoom_home.png'), 'zoom Home', self)
        self.Act_interactive_zoom = QAction(QIcon('./gui_sym/zoom.png'), 'Zoom select', self)
        self.Act_interactive_cut = QAction(QIcon('./gui_sym/cut.png'), 'Cut trace', self)

        self.Act_fine_spec = QAction(QIcon('./gui_sym/spec_fine.png'), 'Show fine Spectrogram', self)
        self.Act_norm_spec = QAction(QIcon('./gui_sym/spec_roght.png'), 'Show rough Spectrogram', self)

    def buttonpress(self, e):
        print('press')
        print(e.xdata)
        print(e.ydata)

    def buttonrelease(self, e):
        print('released')
        print(e.xdata)
        print(e.ydata)

    def open(self):
        fd = QFileDialog()
        if os.path.exists('/home/raab/data/'):
            self.filename, ok = fd.getOpenFileName(self, 'Open File', '/home/raab/data/', 'Select Raw-File (*.raw)')
        else:
            self.filename, ok = fd.getOpenFileName(self, 'Open File', '/home/', 'Select Raw-File (*.raw)')

        if ok:
            self.Act_load.setEnabled(True)

    def load(self):
        def get_datetime(folder):
            rec_year, rec_month, rec_day, rec_time = \
                os.path.split(os.path.split(folder)[-1])[-1].split('-')
            rec_year = int(rec_year)
            rec_month = int(rec_month)
            rec_day = int(rec_day)
            rec_time = (int(rec_time.split('_')[0]), int(rec_time.split('_')[1]))

            rec_datetime = datetime.datetime(rec_year, rec_month, rec_day, *rec_time)

            return rec_datetime

        self.folder = os.path.split(self.filename)[0]
        if os.path.exists(os.path.join(self.folder, 'id_tag.npy')):
            self.id_tag = np.load(os.path.join(self.folder, 'id_tag.npy'))

        if os.path.exists(os.path.join(self.folder, 'fund_v.npy')):
            self.fund_v = np.load(os.path.join(self.folder, 'fund_v.npy'))
            self.sign_v = np.load(os.path.join(self.folder, 'sign_v.npy'))
            self.idx_v = np.load(os.path.join(self.folder, 'idx_v.npy'))
            self.ident_v = np.load(os.path.join(self.folder, 'ident_v.npy'))
            self.times = np.load(os.path.join(self.folder, 'times.npy'))
            self.spectra = np.load(os.path.join(self.folder, 'spec.npy'))
            self.start_time, self.end_time = np.load(os.path.join(self.folder, 'meta.npy'))

            self.rec_datetime = get_datetime(self.folder)

            if self.spec_img_handle:
                self.spec_img_handle.remove()
            self.spec_img_handle = self.ax.imshow(decibel(self.spectra)[::-1],
                                                  extent=[self.start_time, self.end_time, 0, 2000],
                                                  aspect='auto', alpha=0.7, cmap='jet', interpolation='gaussian')
            self.ax.set_xlabel('time', fontsize=12)
            self.ax.set_ylabel('frequency [Hz]', fontsize=12)
            self.ax.set_xlim(self.start_time, self.end_time)

            self.plot_traces()


            self.figure.canvas.draw()

            # self.get_clock_time()

        self.Act_save.setEnabled(True)
        self.button.close()
        self.button2.close()

    def plot_traces(self):
        for handle in self.trace_handles:
            handle[0].remove()
        self.trace_handles = []

        possible_identities = np.unique(self.ident_v[~np.isnan(self.ident_v)])

        for i, ident in enumerate(np.array(possible_identities)):
            c = np.random.rand(3)
            h, = self.ax.plot(self.times[self.idx_v[self.ident_v == ident]],
                                   self.fund_v[self.ident_v == ident], marker='.', color=c)
            self.trace_handles.append((h, ident))

    def clock_time(self):
        xlim = self.ax.get_xlim()
        dx = np.diff(xlim)

        if dx <= 20:
            res = 1
        elif dx > 20 and dx <= 120:
            res = 10
        elif dx > 120 and dx <=1200:
            res = 60
        elif dx > 1200 and dx <= 3600:
            res = 600
        elif dx > 3600 and dx <= 7200:
            res = 1800
        else:
            res = 3600


    def stateAsk(self):
        # ToDo: remove !!!
        # print(self.Act_interactive_sel.isChecked())
        print(self.Act_interactive_cut.isChecked())

    def call_win2(self):
        # ToDo: remove !!!
        self.sec_window = SubWindow1()
        self.sec_window.show()


def main():
    app = QApplication(sys.argv)  # create application
    w = MainWindow()  # create window
    w.show()
    sys.exit(app.exec_())  # exit if window is closed


if __name__ == '__main__':
    main()

