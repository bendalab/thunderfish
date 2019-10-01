import sys
import os
import time
import datetime
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing
from functools import partial
from .version import __version__
from .powerspectrum import decibel, next_power_of_two, spectrogram
from .dataloader import open_data, fishgrid_grids, fishgrid_spacings
from .harmonics import harmonic_groups, fundamental_freqs

import multiprocessing
from functools import partial


from IPython import embed

from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5 import QtCore
from PyQt5.QtCore import *
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas


from matplotlib.widgets import RectangleSelector, EllipseSelector

def position_tracking(sign_v, ident_v, elecs, elecs_spacing, n = 6, id = None):
    def get_elec_pos(elecs_y, elecs_x, elecs_y_spacing, elecs_x_spacing ):
        elec_pos = np.zeros((2, elecs_y * elecs_x))
        elec_pos[0] = (np.arange(elecs_y * elecs_x) %  elecs_x) * elecs_x_spacing
        elec_pos[1] = (np.arange(elecs_y * elecs_x) // elecs_x) * elecs_y_spacing

        return elec_pos.T

    elecs_y, elecs_x = elecs
    elecs_y_spacing, elecs_x_spacing = elecs_spacing

    if np.max(sign_v[0]) != 1:
        sqr_sign = np.sqrt(0.1 * 10.**sign_v)
    else:
        sqr_sign = sign_v
        # ToDo: check in goettingen stuff how i calculate distance there !!!

    x_v = np.zeros(len(sign_v))
    y_v = np.zeros(len(sign_v))

    elec_pos = get_elec_pos(elecs_y, elecs_x, elecs_y_spacing, elecs_x_spacing )
    max_power_electrodes = np.argsort(sign_v, axis = 1)[:, -n:][:, ::-1]

    max_power = list(map(lambda x, y: x[y], sign_v, max_power_electrodes))

    x_pos = list(map(lambda x, y: np.sum(elec_pos[x][:, 0] * y) / np.sum(y), max_power_electrodes, max_power ))
    y_pos = list(map(lambda x, y: np.sum(elec_pos[x][:, 1] * y) / np.sum(y), max_power_electrodes, max_power ))


class AnalysisDialog(QMainWindow):
    def __init__(self):
        super().__init__()

        self.initMe()

    def initMe(self):
        # self.setGeometry(300, 150, 200, 200)  # set window proportion

        self.got_changed = False
        self.start_time = 0 * 60
        self.end_time = .1 * 60
        self.data = None
        self.samplerate = None
        self.channels = None
        self.channel_list = []
        self.data_snippet_sec = 15.
        # int(data_snippet_secs * samplerate)
        self.data_snippet_idxs = None

        self.fresolution = 0.5
        self.overlap_frac = 0.95
        self.channel_list = None
        self.nffts_per_psd = 1

        self.fundamentals = []
        self.signatures = []
        self.times = []

        self.setGeometry(300, 150, 600, 600)
        self.setWindowTitle('EODf and signature extraction')

        self.central_widget = QWidget(self)
        self.gridLayout = QGridLayout()

        self.gridLayout.setColumnStretch(0, 1)
        self.gridLayout.setColumnStretch(1, 1)
        self.gridLayout.setColumnStretch(2, 1)
        #self.gridLayout.setColumnStretch(1, 1)
        #self.gridLayout.setColumnStretch(2, 2)
        #self.gridLayout.setColumnStretch(3, 5)

        #self.gridLayout.setRowStretch(0, 1)
        #self.gridLayout.setRowStretch(1, 1)
        #self.gridLayout.setRowStretch(2, 1)
        #self.gridLayout.setRowStretch(3, 2)
        #self.gridLayout.setRowStretch(4, 1)
        # self.gridLayout.setColumnStretch(1, 1)

        #self.layout_plot()ft

        self.init_widgets()

        self.central_widget.setLayout(self.gridLayout)
        self.setCentralWidget(self.central_widget)

    def init_widgets(self):

        self.StartTime = QLineEdit(str(self.start_time), self.central_widget)
        self.gridLayout.addWidget(self.StartTime, 0, 0)
        t0 = QLabel('start time [min]')
        self.gridLayout.addWidget(t0, 0, 1)

        self.EndTime = QLineEdit(str(self.end_time), self.central_widget)
        self.gridLayout.addWidget(self.EndTime, 1, 0)
        t1 = QLabel('end time [min]')
        self.gridLayout.addWidget(t1, 1, 1)

        self.SnippetSize = QLineEdit(str(self.data_snippet_sec), self.central_widget)
        self.gridLayout.addWidget(self.SnippetSize, 2, 0)
        snip_size = QLabel('data snippet size [sec]')
        self.gridLayout.addWidget(snip_size, 2, 1)

        self.FreqResolution = QLineEdit(str(self.fresolution), self.central_widget)
        self.gridLayout.addWidget(self.FreqResolution, 3, 0)
        freqres = QLabel('frequency resolution [Hz]')
        self.gridLayout.addWidget(freqres, 3, 1)

        self.Overlap = QLineEdit(str(self.overlap_frac), self.central_widget)
        self.gridLayout.addWidget(self.Overlap, 4, 0)
        overlap = QLabel('frequency resolution [Hz]')
        self.gridLayout.addWidget(overlap, 4, 1)

        self.NfftPerPsd = QLineEdit(str(self.nffts_per_psd), self.central_widget)
        self.gridLayout.addWidget(self.NfftPerPsd, 5, 0)
        overlap = QLabel('nffts per PSD [n]')
        self.gridLayout.addWidget(overlap, 5, 1)

        Run = QPushButton('&Run', self.central_widget)
        Run.clicked.connect(self.snippet_spectrogram)
        self.gridLayout.addWidget(Run, 6, 0)

        Apply = QPushButton('&Apply', self.central_widget)
        Apply.clicked.connect(self.apply_settings)
        self.gridLayout.addWidget(Apply, 6, 1)

        Cancel = QPushButton('&Cancel', self.central_widget)
        Cancel.clicked.connect(self.close)
        self.gridLayout.addWidget(Cancel, 6, 2)

    def apply_settings(self):
        print('read that input')

    def snippet_spectrogram(self):
        start_idx = int(self.start_time * self.samplerate)
        if self.end_time < 0.0:
            end_time = len(self.data) / self.samplerate
            end_idx = int(len(self.data) - 1)
        else:
            end_idx = int(self.end_time * self.samplerate)
            if end_idx >= int(len(self.data) - 1):
                end_idx = int(len(self.data) - 1)

        last_run = False
        get_spec_plot_matrix = False
        while start_idx <= end_idx:
            print(start_idx, end_idx)
            if start_idx >= end_idx - self.data_snippet_idxs:
                last_run = True

            core_count = multiprocessing.cpu_count()
            pool = multiprocessing.Pool(core_count // 2)
            nfft = next_power_of_two(self.samplerate / self.fresolution)

            func = partial(spectrogram, samplerate=self.samplerate, freq_resolution=self.fresolution, overlap_frac=self.overlap_frac)

            if len(np.shape(self.data)) == 1:
                a = pool.map(func, [self.data[start_idx: start_idx + self.data_snippet_idxs]])  # ret: spec, freq, time
            else:
                a = pool.map(func, [self.data[start_idx: start_idx + self.data_snippet_idxs, channel] for channel in
                                    self.channel_list])  # ret: spec, freq, time

            print('check 1')
            self.spectra = [a[channel][0] for channel in range(len(a))]
            self.spec_freqs = a[0][1]
            self.spec_times = a[0][2]
            pool.terminate()

            self.comb_spectra = np.sum(self.spectra, axis=0)
            self.tmp_times = self.spec_times + (start_idx / self.samplerate)

            ####
            comp_max_freq = 2000
            comp_min_freq = 0
            create_plotable_spectrogram = True

            plot_freqs = self.spec_freqs[self.spec_freqs < comp_max_freq]
            plot_spectra = np.sum(self.spectra, axis=0)[self.spec_freqs < comp_max_freq]

            print('check 2')
            if create_plotable_spectrogram:
                # if not checked_xy_borders:
                if not get_spec_plot_matrix:
                    fig_xspan = 20.
                    fig_yspan = 12.
                    fig_dpi = 80.
                    no_x = fig_xspan * fig_dpi
                    no_y = fig_yspan * fig_dpi

                    min_x = self.start_time
                    max_x = self.end_time

                    min_y = comp_min_freq
                    max_y = comp_max_freq

                    x_borders = np.linspace(min_x, max_x, no_x * 2)
                    y_borders = np.linspace(min_y, max_y, no_y * 2)
                    # checked_xy_borders = False

                    self.tmp_spectra = np.zeros((len(y_borders) - 1, len(x_borders) - 1))

                    recreate_matrix = False
                    if (self.tmp_times[1] - self.tmp_times[0]) > (x_borders[1] - x_borders[0]):
                        x_borders = np.linspace(min_x, max_x, (max_x - min_x) // (self.tmp_times[1] - self.tmp_times[0]) + 1)
                        recreate_matrix = True
                    if (self.spec_freqs[1] - self.spec_freqs[0]) > (y_borders[1] - y_borders[0]):
                        recreate_matrix = True
                        y_borders = np.linspace(min_y, max_y, (max_y - min_y) // (self.spec_freqs[1] - self.spec_freqs[0]) + 1)
                    if recreate_matrix:
                        self.tmp_spectra = np.zeros((len(y_borders) - 1, len(x_borders) - 1))

                    get_spec_plot_matrix = True
                    # checked_xy_borders = True

                for i in range(len(y_borders) - 1):
                    print(i/len(y_borders))
                    for j in range(len(x_borders) - 1):
                        if x_borders[j] > self.tmp_times[-1]:
                            break
                        if x_borders[j + 1] < self.tmp_times[0]:
                            continue

                        t_mask = np.arange(len(self.tmp_times))[(self.tmp_times >= x_borders[j]) & (self.tmp_times < x_borders[j + 1])]
                        f_mask = np.arange(len(plot_spectra))[(plot_freqs >= y_borders[i]) & (plot_freqs < y_borders[i + 1])]

                        if len(t_mask) == 0 or len(f_mask) == 0:
                            continue
                        print('yay')
                        self.tmp_spectra[i, j] = np.max(plot_spectra[f_mask[:, None], t_mask])
            ####
            print('check 3')

            self.power = [np.array([]) for i in range(len(self.spec_times))]

            for t in range(len(self.spec_times)):
                self.power[t] = np.mean(self.comb_spectra[:, t:t + 1], axis=1)
            self.extract_fundamentals_and_signatures()

            non_overlapping_idx = (1 - self.overlap_frac) * nfft
            start_idx += int((len(self.spec_times) - self.nffts_per_psd + 1) * non_overlapping_idx)
            self.times = np.concatenate((self.times, self.tmp_times))

            print('check 4')
            if start_idx >= end_idx or last_run:
                print('done')
                break
        self.got_changed = True
        print('done')
        self.close()

    def extract_fundamentals_and_signatures(self):
        core_count = multiprocessing.cpu_count()
        pool = multiprocessing.Pool(core_count // 2)

        # ToDo the kwarg problem ....

       # if True:
            #psd_freqs
            #psd
            #verbose=0
            #check_freqs=[]
            #low_threshold=0.0
            #high_threshold=0.0
            #thresh_bins=100
            #low_thresh_factor=6.0
            #high_thresh_factor=10.0
            #freq_tol_fac=1.0
            #mains_freq=60.0
            #mains_freq_tol=1.0
            #min_freq=0.0
            #max_freq=2000.0
            #max_divisor=4
            #min_group_size=4
            #max_rel_power_weight=2.0
            #max_rel_power=0.0
            #max_harmonics=0
            #max_groups=0

        func = partial(harmonic_groups, self.spec_freqs)
        a = pool.map(func, self.power)
        log_spectra = decibel(np.array(self.spectra))
        for p in range(len(self.power)):
            tmp_fundamentals = fundamental_freqs(a[p][0])
            self.fundamentals.append(tmp_fundamentals)

            if len(tmp_fundamentals) >= 1:
                f_idx = np.array([np.argmin(np.abs(self.spec_freqs - f)) for f in tmp_fundamentals])
                tmp_signatures = log_spectra[:, np.array(f_idx), p].transpose()
            else:
                tmp_signatures = np.array([])

            self.signatures.append(tmp_signatures)
        pool.terminate()

class GridDialog(QMainWindow):
    def __init__(self):
        super().__init__()

        self.initMe()

    def initMe(self):
        # self.setGeometry(300, 150, 200, 200)  # set window proportion
        self.setGeometry(300, 150, 300, 450)
        self.setWindowTitle('Grid Layout')

        print('init dialig grid')
        self.channels = 1
        self.elecs_x = 1
        self.elecs_x_spacing = 1

        self.elecs_y = 1
        self.elecs_y_spacing = 1

        self.grid_handle = None
        self.e0_handle = None
        self.en_handle = None

        self.elec_TL = 1
        self.elec_BR = 'n'

        self.central_widget = QWidget(self)
        self.gridLayout = QGridLayout()
        self.gridLayout.setColumnStretch(0, 2)
        self.gridLayout.setColumnStretch(1, 1)
        self.gridLayout.setColumnStretch(2, 2)
        self.gridLayout.setColumnStretch(3, 5)

        self.gridLayout.setRowStretch(0, 1)
        self.gridLayout.setRowStretch(1, 1)
        self.gridLayout.setRowStretch(2, 1)
        self.gridLayout.setRowStretch(3, 2)
        self.gridLayout.setRowStretch(4, 1)
        # self.gridLayout.setColumnStretch(1, 1)

        self.layout_plot()

        self.init_widgets()

        self.update_grid()
        self.central_widget.setLayout(self.gridLayout)
        self.setCentralWidget(self.central_widget)
        # self.show()

    def init_widgets(self):
        self.layout_xW = QLineEdit(str(self.elecs_x), self.central_widget)
        self.layout_xW.setValidator(QIntValidator(0, 10))
        self.gridLayout.addWidget(self.layout_xW, 0, 0)

        x = QLabel('x', self.central_widget)
        self.gridLayout.addWidget(x, 0, 1)

        self.layout_yW = QLineEdit(str(self.elecs_y), self.central_widget)
        self.layout_yW.setValidator(QIntValidator(0, 10))
        self.gridLayout.addWidget(self.layout_yW, 0, 2)

        layout_textW = QLabel('grid_layout [x, y]', self.central_widget)
        self.gridLayout.addWidget(layout_textW, 0, 3)

        self.x_spaceW = QLineEdit(str(self.elecs_x_spacing), self.central_widget)
        self.x_spaceW.setValidator(QIntValidator(0, 1000))
        self.gridLayout.addWidget(self.x_spaceW, 1, 0, 1, 3)

        x_space_textW = QLabel('x-spacing [cm]')
        self.gridLayout.addWidget(x_space_textW, 1, 3)

        self.y_spaceW = QLineEdit(str(self.elecs_y_spacing), self.central_widget)
        self.y_spaceW.setValidator(QIntValidator(0, 1000))
        self.gridLayout.addWidget(self.y_spaceW, 2, 0, 1, 3)

        y_space_textW = QLabel('y-spacing [cm]')
        self.gridLayout.addWidget(y_space_textW, 2, 3)

        self.gridLayout.addWidget(self.canvas, 3, 0, 1, 4)

        Apply = QPushButton('&Apply', self.central_widget)
        Apply.clicked.connect(self.apply_settings)
        self.gridLayout.addWidget(Apply, 4, 0, 1, 3)

        Cancel = QPushButton('&Cancel', self.central_widget)
        Cancel.clicked.connect(self.close)
        self.gridLayout.addWidget(Cancel, 4, 3)


    def update_widgets(self):
        self.layout_xW.setText(str(self.elecs_x))
        self.layout_yW.setText(str(self.elecs_y))

        self.x_spaceW.setText(str(self.elecs_x_spacing))
        self.y_spaceW.setText(str(self.elecs_y_spacing))
        self.update_grid()

    def layout_plot(self):
        self.figure = plt.figure()
        self.canvas = FigureCanvas(self.figure)

        self.ax = self.figure.add_axes([0, 0, 1, 1])
        self.ax.axis('off')
        self.ax.invert_yaxis()

    def update_grid(self):
        if self.grid_handle:
            self.grid_handle.remove()
        if self.e0_handle:
            self.e0_handle.remove()
        if self.en_handle:
            self.en_handle.remove()

        X, Y = np.meshgrid(np.arange(int(self.layout_xW.text())), np.arange(int(self.layout_yW.text())))
        x = np.hstack(X) * int(self.x_spaceW.text())
        y = np.hstack(Y) * int(self.y_spaceW.text())

        max_lim = np.max(np.hstack([x, y]))

        self.grid_handle, = self.ax.plot(x[1:-1], y[1:-1], 'o', color='k')
        self.e0_handle = self.ax.text(x[0], y[0], str(self.elec_TL), color='red', va='center', ha='center')
        if len(x) + len(y) > 2:
            self.en_handle = self.ax.text(x[-1], y[-1], str(self.elec_BR), color='red', va='center', ha='center')


        self.ax.set_xlim(-10, max_lim+10)
        self.ax.set_ylim(-10, max_lim+10)
        self.ax.invert_yaxis()
        self.canvas.draw()

    def apply_settings(self):
        self.update_grid()
        print(int(self.layout_xW.text()))

    def keyPressEvent(self, e):
        # print(e.key())
        if e.key() == Qt.Key_Return:
            self.update_grid()


class PlotWidget():
    def __init__(self):
        self.figure = plt.figure()
        self.canvas = FigureCanvas(self.figure)

        self.ax = self.figure.add_subplot(111)

        self.xlim = None
        self.ylim = None

        self.init_xlim = None
        self.init_ylim = None

        self.spec_img_handle = None
        self.trace_handles = []
        self.active_id_handle0 = None
        self.active_id_handle1 = None
        self.active_cut_handle = None
        self.active_group_handle = None

        self.current_task = None
        self.rec_datetime = None
        self.times = None

    def plot_traces(self, ident_v, times, idx_v, fund_v, task = 'init', active_id = None, active_id2 = None, active_ids = None):
        if task == 'init':
            for handle in self.trace_handles:
                handle[0].remove()
            self.trace_handles = []

            possible_identities = np.unique(ident_v[~np.isnan(ident_v)])

            for i, ident in enumerate(np.array(possible_identities)):
                c = np.random.rand(3)
                h, = self.ax.plot(times[idx_v[ident_v == ident]], fund_v[ident_v == ident], marker='.', color=c)
                self.trace_handles.append((h, ident))

            self.xlim = self.ax.get_xlim()
            self.ylim = self.ax.get_ylim()
            self.init_xlim = self.xlim
            self.init_ylim = self.ylim

        elif task == 'post cut':
            handle_idents = np.array([x[1] for x in self.trace_handles])
            refresh_handle = np.array(self.trace_handles)[handle_idents == active_id][0]
            refresh_handle[0].remove()

            c = np.random.rand(3)
            h, = self.ax.plot(times[idx_v[ident_v == active_id]], fund_v[ident_v == active_id], marker='.', color=c)
            self.trace_handles[np.arange(len(self.trace_handles))[handle_idents == active_id][0]] = (h, active_id)

            new_ident = np.max(ident_v[~np.isnan(ident_v)])
            c = np.random.rand(3)
            h, = self.ax.plot(times[idx_v[ident_v == new_ident]], fund_v[ident_v == new_ident], marker='.', color=c)
            self.trace_handles.append((h, new_ident))

        elif task == 'post_connect':
            handle_idents = np.array([x[1] for x in self.trace_handles])

            remove_handle = np.array(self.trace_handles)[handle_idents == active_id2][0]
            remove_handle[0].remove()

            joined_handle = np.array(self.trace_handles)[handle_idents == active_id][0]
            joined_handle[0].remove()

            c = np.random.rand(3)
            # sorter = np.argsort(self.times[self.idx_v[self.ident_v == self.active_ident0]])
            h, = self.ax.plot(times[idx_v[ident_v == active_id]], fund_v[ident_v == active_id], marker='.', color=c)
            self.trace_handles[np.arange(len(self.trace_handles))[handle_idents == active_id][0]] = (h, active_id)
            # self.trace_handles.append((h, self.active_ident0))

            self.trace_handles.pop(np.arange(len(self.trace_handles))[handle_idents == active_id2][0])

        elif task == 'post_delete':
            handle_idents = np.array([x[1] for x in self.trace_handles])
            delete_handle_idx = np.arange(len(self.trace_handles))[handle_idents == active_id][0]
            delete_handle = np.array(self.trace_handles)[handle_idents == active_id][0]
            delete_handle[0].remove()
            self.trace_handles.pop(delete_handle_idx)

        elif task == 'post_group_connect' or task == 'post_group_delete':
            handle_idents = np.array([x[1] for x in self.trace_handles])
            effected_idents = active_ids

            mask = np.array([x in effected_idents for x in handle_idents], dtype=bool)
            delete_handle_idx = np.arange(len(self.trace_handles))[mask]
            delete_handle = np.array(self.trace_handles)[mask]

            delete_afterwards = []
            for dhi, dh in zip(delete_handle_idx, delete_handle):
                dh[0].remove()
                if len(ident_v[ident_v == dh[1]]) >= 1:
                    c = np.random.rand(3)
                    h, = self.ax.plot(times[idx_v[ident_v == dh[1]]], fund_v[ident_v == dh[1]], marker='.', color=c)
                    self.trace_handles[dhi] = (h, dh[1])
                else:
                    delete_afterwards.append(dhi)

            for i in reversed(sorted(delete_afterwards)):
                self.trace_handles.pop(i)

    def highlight_group(self, active_idx, ident_v, times, idx_v, fund_v):
        if self.active_group_handle:
            self.active_group_handle.remove()

        self.active_group_handle, = self.ax.plot(times[idx_v[active_idx]], fund_v[active_idx], 'o', color='orange', markersize=4)

    def highlight_id(self, active_id, ident_v, times, idx_v, fund_v, no):
        if no == 'first':

            if self.active_id_handle0:
                self.active_id_handle0.remove()

            self.active_id_handle0, = self.ax.plot(times[idx_v[ident_v == active_id]], fund_v[ident_v == active_id], color='orange', alpha=0.7, linewidth=4)
        elif no == 'second':
            if self.active_id_handle1:
                self.active_id_handle1.remove()
            self.active_id_handle1, = self.ax.plot(times[idx_v[ident_v == active_id]], fund_v[ident_v == active_id], color='red', alpha=0.7, linewidth=4)

    def highlight_cut(self, active_idx_in_trace, times, idx_v, fund_v):
        if self.active_cut_handle:
            self.active_cut_handle.remove()
        self.active_cut_handle, = self.ax.plot(times[idx_v[active_idx_in_trace]], fund_v[active_idx_in_trace] , 'o', color='red', alpha = 0.7, markersize=5)

    def clock_time(self, rec_datetime, times):
        xlim = self.xlim
        dx = np.diff(xlim)[0]

        label_idx0 = 0
        if dx <= 20:
            res = 1
        elif dx > 20 and dx <= 120:
            res = 10
        elif dx > 120 and dx <=1200:
            res = 60
        elif dx > 1200 and dx <= 3600:
            res = 600  # 10 min
        elif dx > 3600 and dx <= 7200:
            res = 1800  # 30 min
        else:
            res = 3600  # 60 min

        if dx > 1200:
            if rec_datetime.minute % int(res / 60) != 0:
                dmin = int(res / 60) - rec_datetime.minute % int(res / 60)
                label_idx0 = dmin * 60

        xtick = np.arange(label_idx0, times[-1], res)
        datetime_xlabels = list(map(lambda x: rec_datetime + datetime.timedelta(seconds= x), xtick))

        if dx > 120:
            xlabels = list(map(lambda x: ('%2s:%2s' % (str(x.hour), str(x.minute))).replace(' ', '0'), datetime_xlabels))
            rotation = 0
        else:
            xlabels = list(map(lambda x: ('%2s:%2s:%2s' % (str(x.hour), str(x.minute), str(x.second))).replace(' ', '0'), datetime_xlabels))
            rotation = 45
        # ToDo: create mask
        mask = np.arange(len(xtick))[(xtick > self.xlim[0]) & (xtick < self.xlim[1])]
        self.ax.set_xticks(xtick[mask])
        self.ax.set_xticklabels(np.array(xlabels)[mask], rotation = rotation)
        self.ax.set_xlim(*self.xlim)
        self.ax.set_ylim(*self.ylim)
        # embed()
        # quit()

    def zoom(self, x0, x1, y0, y1):
        new_xlim = np.sort([x0, x1])
        new_ylim = np.sort([y0, y1])
        self.ylim = new_ylim
        self.xlim = new_xlim
        self.ax.set_xlim(*new_xlim)
        self.ax.set_ylim(*new_ylim)

    def zoom_in(self):
        xlim = self.xlim
        ylim = self.ylim

        new_xlim = (xlim[0] + np.diff(xlim)[0] * 0.25, xlim[1] - np.diff(xlim)[0] * 0.25)
        new_ylim = (ylim[0] + np.diff(ylim)[0] * 0.25, ylim[1] - np.diff(ylim)[0] * 0.25)
        self.ylim = new_ylim
        self.xlim = new_xlim

        self.ax.set_xlim(*new_xlim)
        self.ax.set_ylim(*new_ylim)
        # self.clock_time()

        # self.figure.canvas.draw()

    def zoom_out(self):
        xlim = self.xlim
        ylim = self.ylim

        new_xlim = (xlim[0] - np.diff(xlim)[0] * 0.25, xlim[1] + np.diff(xlim)[0] * 0.25)
        new_ylim = (ylim[0] - np.diff(ylim)[0] * 0.25, ylim[1] + np.diff(ylim)[0] * 0.25)
        self.ylim = new_ylim
        self.xlim = new_xlim

        self.ax.set_xlim(*new_xlim)
        self.ax.set_ylim(*new_ylim)
        # self.clock_time()

        # self.figure.canvas.draw()

    def zoom_home(self):
        new_xlim = self.init_xlim
        new_ylim = self.init_ylim
        self.ylim = new_ylim
        self.xlim = new_xlim

        self.ax.set_xlim(*new_xlim)
        self.ax.set_ylim(*new_ylim)
        # self.clock_time()

        # self.figure.canvas.draw()

    def move_right(self):
        xlim = self.xlim

        new_xlim = (xlim[0] + np.diff(xlim)[0] * 0.25, xlim[1] + np.diff(xlim)[0] * 0.25)
        self.xlim = new_xlim

        self.ax.set_xlim(*new_xlim)

        self.figure.canvas.draw()

    def move_left(self):
        xlim = self.xlim

        new_xlim = (xlim[0] - np.diff(xlim)[0] * 0.25, xlim[1] - np.diff(xlim)[0] * 0.25)
        self.xlim = new_xlim

        self.ax.set_xlim(*new_xlim)

        self.figure.canvas.draw()

    def move_up(self):
        ylim = self.ylim

        new_ylim = (ylim[0] + np.diff(ylim)[0] * 0.25, ylim[1] + np.diff(ylim)[0] * 0.25)
        self.ylim = new_ylim

        self.ax.set_ylim(*new_ylim)
        self.figure.canvas.draw()

    def move_down(self):
        ylim = self.ylim

        new_ylim = (ylim[0] - np.diff(ylim)[0] * 0.25, ylim[1] - np.diff(ylim)[0] * 0.25)
        self.ylim = new_ylim

        self.ax.set_ylim(*new_ylim)
        self.figure.canvas.draw()


class MainWindow(QMainWindow):
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        self.Plot = PlotWidget()

        self.Plot.figure.canvas.mpl_connect('button_press_event', self.buttonpress)
        self.Plot.figure.canvas.mpl_connect('button_release_event', self.buttonrelease)

        self.Grid = GridDialog()

        self.AnalysisDial = AnalysisDialog()


        self.initMe()

    def initMe(self):
        # implement status Bar
        self.statusBar().showMessage('Welcome to FishLab')
        # MenuBar
        self.init_Actions()

        self.init_MenuBar()

        self.init_ToolBar()

        qApp.installEventFilter(self)

        self.active_idx = None
        self.active_idx2 = None
        self.active_id = None
        self.active_id2 = None
        self.active_idx_in_trace = None
        self.active_ids = None

        # ToDo: set to auto ?!
        self.setGeometry(200, 50, 1200, 800)  # set window proportion
        self.setWindowTitle('FishLab v1.0')  # set window title

        # ToDo: create icon !!!
        # self.setWindowIcon(QIcon('<path>'))  # set window image (left top)

        self.central_widget = QWidget(self)

        self.open_button = QPushButton('Open', self.central_widget)
        self.open_button.clicked.connect(self.open)
        self.load_button = QPushButton('Load', self.central_widget)
        self.load_button.clicked.connect(self.load)
        self.load_button.setEnabled(False)

        self.disp_analysis_button = QPushButton('Display analysis', self.central_widget)
        self.disp_analysis_button.clicked.connect(self.show_updates)

        self.gridLayout = QGridLayout()

        # self.gridLayout.addWidget(self.canvas, 0, 0, 4, 5)
        self.gridLayout.addWidget(self.Plot.canvas, 0, 0, 4, 5)
        self.gridLayout.addWidget(self.open_button, 4, 1)
        self.gridLayout.addWidget(self.disp_analysis_button, 4, 2)
        self.gridLayout.addWidget(self.load_button, 4, 3)
        # self.setLayout(v)

        # self.show()  # show the window
        self.central_widget.setLayout(self.gridLayout)
        # self.installEventFilter(self)

        # self.central_widget.setFocusPolicy(Qt.NoFocus)
        # self.central_widget.installEventFilter(self)

        self.setCentralWidget(self.central_widget)

        # self.figure.canvas.draw()
        self.Plot.canvas.draw()

    def init_ToolBar(self):
        toolbar = self.addToolBar('TB')  # toolbar needs QMainWindow ?!
        self.addToolBar(Qt.LeftToolBarArea, toolbar)
        toolbar2 = self.addToolBar('TB2')  # toolbar needs QMainWindow ?!
        self.addToolBar(Qt.LeftToolBarArea, toolbar2)

        toolbar3 = self.addToolBar('TB3')  # toolbar needs QMainWindow ?!
        self.addToolBar(Qt.LeftToolBarArea, toolbar3)

        toolbar4 = self.addToolBar('TB4')  # toolbar needs QMainWindow ?!
        self.addToolBar(Qt.LeftToolBarArea, toolbar4)

        toolbar.addAction(self.Act_interactive_sel)
        toolbar.addAction(self.Act_interactive_con)
        toolbar.addAction(self.Act_interactive_GrCon)
        toolbar.addAction(self.Act_interactive_cut)
        toolbar.addAction(self.Act_interactive_del)
        toolbar.addAction(self.Act_interactive_GrDel)
        toolbar.addAction(self.Act_interactive_reset)

        toolbar2.addAction(self.Act_interactive_zoom)
        toolbar2.addAction(self.Act_interactive_zoom_in)
        toolbar2.addAction(self.Act_interactive_zoom_out)
        toolbar2.addAction(self.Act_interactive_zoom_home)

        toolbar3.addAction(self.Act_interactive_AutoSort)
        toolbar3.addAction(self.Act_interactive_ManualSort)
        toolbar3.addAction(self.Act_undo)

        toolbar4.addAction(self.Act_fine_spec)
        toolbar4.addAction(self.Act_norm_spec)
        toolbar4.addAction(self.Act_arrowkeys)

    def init_MenuBar(self):
        menubar = self.menuBar() # needs QMainWindow ?!
        file = menubar.addMenu('&File') # create file menu ... accessable with alt+F
        file.addActions([self.Act_open, self.Act_load, self.Act_save, self.Act_exit])

        edit = menubar.addMenu('&Edit')
        edit.addActions([self.Act_undo])

        settings = menubar.addMenu('&Settings')
        settings.addActions([self.Act_set_psd, self.Act_set_track, self.Act_set_gridLayout])

        analysis = menubar.addMenu('&Analysis')
        analysis.addActions([self.Act_PSD_n_harmonic, self.Act_EODtrack, self.Act_Positontrack])

        spectrogram = menubar.addMenu('&Spectrogram')
        spectrogram.addActions([self.Act_compSpec])

        individual = menubar.addMenu('&Individual')
        individual.addActions([self.Act_individual_field])


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

        self.Act_undo = QAction(QIcon('./thunderfish/gui_sym/undo.png'), '&Undo', self)
        self.Act_undo.setStatusTip('Undo last sorting step')

        # --- MenuBar - Edit --- #

        self.Act_set_psd = QAction('PSD settings', self)
        self.Act_set_psd.setStatusTip('set PSD settings')

        self.Act_set_track = QAction('Tracking settings', self)
        self.Act_set_track.setStatusTip('set tracking settings')

        self.Act_set_gridLayout = QAction('Grid layout', self)
        self.Act_set_gridLayout.setStatusTip('define grid layout')
        self.Act_set_gridLayout.triggered.connect(self.grid_layout)

        # --- MenuBar - Tracking --- #
        self.Act_PSD_n_harmonic = QAction('Extract EODf and Signature', self)
        self.Act_PSD_n_harmonic.setStatusTip('Powerspectrum and harmonic groups')
        self.Act_PSD_n_harmonic.triggered.connect(self.ManalysisDial)


        self.Act_EODtrack = QAction('Track EOD traces', self)
        self.Act_EODtrack.setStatusTip('track EOD traces')

        self.Act_Positontrack = QAction('Track position', self)
        self.Act_Positontrack.setStatusTip('track fish locations')
        self.Act_Positontrack.triggered.connect(self.Mposition_tracking)

        # --- MenuBar - Spectrogram --- #

        self.Act_compSpec = QAction('Compute full spectrogram', self)
        self.Act_compSpec.setStatusTip('compute full detailed spectrogram')

        # --- MenuBar - Individual --- #

        self.Act_individual_field = QAction('Show electric field', self)
        self.Act_individual_field.setStatusTip('shows video of electric field')


        ################## ToolBar ###################


        self.Act_interactive_sel = QAction(QIcon('./thunderfish/gui_sym/sel.png'), 'S', self)
        self.Act_interactive_sel.setCheckable(True)
        self.Act_interactive_sel.setEnabled(False)

        # self.Act_*.setChecked(False)
        # self.Act_xx.setChecked(True)

        self.Act_interactive_con = QAction(QIcon('./thunderfish/gui_sym/con.png'), 'Connect', self)
        self.Act_interactive_con.setCheckable(True)
        self.Act_interactive_con.setEnabled(False)

        self.Act_interactive_GrCon = QAction(QIcon('./thunderfish/gui_sym/GrCon.png'), 'Group Connect', self)
        self.Act_interactive_GrCon.setCheckable(True)
        self.Act_interactive_GrCon.setEnabled(False)

        self.Act_interactive_del = QAction(QIcon('./thunderfish/gui_sym/del.png'), 'Delete Trace', self)
        self.Act_interactive_del.setCheckable(True)
        self.Act_interactive_del.setEnabled(False)

        self.Act_interactive_GrDel = QAction(QIcon('./thunderfish/gui_sym/GrDel.png'), 'Group Delete', self)
        self.Act_interactive_GrDel.setCheckable(True)
        self.Act_interactive_GrDel.setEnabled(False)

        self.Act_interactive_reset = QAction(QIcon('./thunderfish/gui_sym/reset.png'), 'Reset Variables', self)
        self.Act_interactive_reset.setEnabled(False)
        self.Act_interactive_reset.triggered.connect(self.reset_variables)



        self.Act_interactive_cut = QAction(QIcon('./thunderfish/gui_sym/cut.png'), 'Cut trace', self)
        self.Act_interactive_cut.setCheckable(True)
        self.Act_interactive_cut.setEnabled(False)

        self.Act_interactive_AutoSort = QAction(QIcon('./thunderfish/gui_sym/auto.png'), 'Auto Connect', self)
        self.Act_interactive_AutoSort.setEnabled(False)
        self.Act_interactive_ManualSort = QAction(QIcon('./thunderfish/gui_sym/manuel.png'), 'Manual Connect', self)
        self.Act_interactive_ManualSort.setEnabled(False)

        self.Act_interactive_zoom_out = QAction(QIcon('./thunderfish/gui_sym/zoomout.png'), 'Zoom -', self)
        self.Act_interactive_zoom_out.triggered.connect(self.Mzoom_out)
        self.Act_interactive_zoom_out.setEnabled(False)

        self.Act_interactive_zoom_in = QAction(QIcon('./thunderfish/gui_sym/zoomin.png'), 'zoom +', self)
        self.Act_interactive_zoom_in.triggered.connect(self.Mzoom_in)
        self.Act_interactive_zoom_in.setEnabled(False)

        self.Act_interactive_zoom_home = QAction(QIcon('./thunderfish/gui_sym/zoom_home.png'), 'zoom Home', self)
        self.Act_interactive_zoom_home.triggered.connect(self.Mzoom_home)
        self.Act_interactive_zoom_home.setEnabled(False)

        self.Act_interactive_zoom = QAction(QIcon('./thunderfish/gui_sym/zoom.png'), 'Zoom select', self)
        self.Act_interactive_zoom.setCheckable(True)
        self.Act_interactive_zoom.setEnabled(False)
        # self.Act_interactive_zoom.toggled.connect(self.Mzoom)


        self.Act_fine_spec = QAction(QIcon('./thunderfish/gui_sym/spec_fine.png'), 'Show fine Spectrogram', self)
        self.Act_fine_spec.setEnabled(False)
        self.Act_norm_spec = QAction(QIcon('./thunderfish/gui_sym/spec_roght.png'), 'Show rough Spectrogram', self)
        self.Act_norm_spec.setEnabled(False)

        self.Act_arrowkeys = QAction(QIcon('./thunderfish/gui_sym/arrowkeys.png'), 'Activate arrorw keys', self)
        self.Act_arrowkeys.setCheckable(True)
        self.Act_arrowkeys.setEnabled(False)

        self.group = QActionGroup(self)
        self.group.addAction(self.Act_interactive_sel)
        self.group.addAction(self.Act_interactive_con)
        self.group.addAction(self.Act_interactive_GrCon)
        self.group.addAction(self.Act_interactive_del)
        self.group.addAction(self.Act_interactive_GrDel)
        self.group.addAction(self.Act_interactive_cut)
        self.group.addAction(self.Act_interactive_zoom)



    def eventFilter(self, source, event):
        # if event.type() == QEvent.KeyPress:
        # print(event.type)
        if event.type() == QEvent.KeyPress:
            if self.Act_arrowkeys.isChecked():
                if event.key() == Qt.Key_Right:
                    self.Plot.move_right()
                    self.Plot.clock_time(self.rec_datetime, self.times)
                    return True
                elif event.key() == Qt.Key_Left:
                    self.Plot.move_left()
                    self.Plot.clock_time(self.rec_datetime, self.times)
                    return True
                elif event.key() == Qt.Key_Up:
                    self.Plot.move_up()
                    self.Plot.clock_time(self.rec_datetime, self.times)
                    return True

                elif event.key() == Qt.Key_Down:
                    self.Plot.move_down()
                    self.Plot.clock_time(self.rec_datetime, self.times)
                    return True

        return super(MainWindow, self).eventFilter(source, event)

    def keyPressEvent(self, e):
        # print(e.key())
        if e.key() == Qt.Key_Return:
            self.execute()
            # print('enter')


    def buttonpress(self, e):
        self.x0 = e.xdata
        self.y0 = e.ydata

    def buttonrelease(self, e):
        self.x1 = e.xdata
        self.y1 = e.ydata
        # if self.current_task == 'Zoom':
        if self.Act_interactive_zoom.isChecked():
            self.Plot.zoom(self.x0, self.x1, self.y0, self.y1)
            self.Plot.clock_time(self.rec_datetime, self.times)
            self.Plot.canvas.draw()

        if self.Act_interactive_cut.isChecked():
            self.get_active_idx_rect()

            if hasattr(self.active_idx, '__len__') and not self.Plot.active_id_handle0:
                if len(self.active_idx) > 0:
                    self.get_active_id(self.active_idx)
                    self.Plot.highlight_id(self.active_id, self.ident_v, self.times, self.idx_v, self.fund_v, 'first')
                    self.Plot.canvas.draw()

            else:
                self.get_active_idx_in_trace()
                self.Plot.highlight_cut(self.active_idx_in_trace, self.times, self.idx_v, self.fund_v)
                self.Plot.canvas.draw()

        if self.Act_interactive_con.isChecked():
            self.get_active_idx_rect()

            if hasattr(self.active_idx, '__len__') and not hasattr(self.active_idx2, '__len__'):
                if len(self.active_idx) > 0:
                    self.get_active_id(self.active_idx)
                    self.Plot.highlight_id(self.active_id, self.ident_v, self.times, self.idx_v, self.fund_v, 'first')
                    self.Plot.canvas.draw()

            elif hasattr(self.active_idx2, '__len__'):
                if len(self.active_idx2) > 0:
                    self.get_active_id(self.active_idx2)
                    self.Plot.highlight_id(self.active_id2, self.ident_v, self.times, self.idx_v, self.fund_v, 'second')
                    self.Plot.canvas.draw()

        if self.Act_interactive_del.isChecked():
            self.get_active_idx_rect()
            if len(self.active_idx) > 0:
                self.get_active_id(self.active_idx)
                self.Plot.highlight_id(self.active_id, self.ident_v, self.times, self.idx_v, self.fund_v, 'first')
                self.Plot.canvas.draw()

        if self.Act_interactive_GrCon.isChecked():
            self.get_active_idx_rect()
            if len(self.active_idx) > 0:
                self.Plot.highlight_group(self.active_idx, self.ident_v, self.times, self.idx_v, self.fund_v)
                self.Plot.canvas.draw()

        if self.Act_interactive_GrDel.isChecked():
            self.get_active_idx_rect()
            if len(self.active_idx) > 0:
                self.Plot.highlight_group(self.active_idx, self.ident_v, self.times, self.idx_v, self.fund_v)
                self.Plot.canvas.draw()


    def open(self):
        fd = QFileDialog()
        if os.path.exists('/home/raab/data/'):
            self.filename, ok = fd.getOpenFileName(self, 'Open File', '/home/raab/data/', 'Select Raw-File (*.raw)')
        else:
            self.filename, ok = fd.getOpenFileName(self, 'Open File', '/home/', 'Select Raw-File (*.raw)')

        if ok:
            self.Act_load.setEnabled(True)
            self.load_button.setEnabled(True)

            self.data = open_data(self.filename, -1, 60.0, 10.0)
            self.AnalysisDial.data = self.data

            self.samplerate= self.data.samplerate
            self.AnalysisDial.samplerate= self.samplerate
            self.AnalysisDial.data_snippet_idxs = int(self.AnalysisDial.data_snippet_sec * self.samplerate)

            self.channels = self.data.channels
            self.Grid.channels = self.data.channels
            self.AnalysisDial.channels = self.data.channels
            self.AnalysisDial.channel_list = np.arange(self.channels)


            if os.path.exists(os.path.join(os.path.split(self.filename)[0], 'fishgrid.cfg')):

                self.elecs_y, self.elecs_x = fishgrid_grids(self.filename)[0]
                self.Grid.elecs_y, self.Grid.elecs_x = fishgrid_grids(self.filename)[0]

                self.elecs_y_spacing, self.elecs_x_spacing = fishgrid_spacings(self.filename)[0]
                self.Grid.elecs_y_spacing, self.Grid.elecs_x_spacing = fishgrid_spacings(self.filename)[0]

                self.Grid.update_widgets()


    def load(self):
        def get_datetime(folder):
            rec_year, rec_month, rec_day, rec_time = \
                os.path.split(os.path.split(folder)[-1])[-1].split('-')
            rec_year = int(rec_year)
            rec_month = int(rec_month)
            rec_day = int(rec_day)
            try:
                rec_time = [int(rec_time.split('_')[0]), int(rec_time.split('_')[1]), 0]
            except:
                rec_time = [int(rec_time.split(':')[0]), int(rec_time.split(':')[1]), 0]

            rec_datetime = datetime.datetime(year=rec_year, month=rec_month, day=rec_day, hour=rec_time[0],
                                             minute=rec_time[1], second=rec_time[2])


            return rec_datetime

        # embed()
        # quit()
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
            # ToDo dirty

            if self.Plot.spec_img_handle:
                self.Plot.spec_img_handle.remove()
            self.Plot.spec_img_handle = self.Plot.ax.imshow(decibel(self.spectra)[::-1],
                                                  extent=[self.start_time, self.end_time, 0, 2000],
                                                  aspect='auto', alpha=0.7, cmap='jet', interpolation='gaussian')
            self.Plot.ax.set_xlabel('time', fontsize=12)
            self.Plot.ax.set_ylabel('frequency [Hz]', fontsize=12)
            self.Plot.ax.set_xlim(self.start_time, self.end_time)

            self.Plot.plot_traces(self.ident_v, self.times, self.idx_v, self.fund_v, task='init')

            self.Plot.clock_time(self.rec_datetime, self.times)

            self.Plot.figure.canvas.draw()

            self.Act_save.setEnabled(True)
            self.Act_interactive_sel.setEnabled(True)
            self.Act_interactive_con.setEnabled(True)
            self.Act_interactive_GrCon.setEnabled(True)
            self.Act_interactive_del.setEnabled(True)
            self.Act_interactive_GrDel.setEnabled(True)
            self.Act_interactive_reset.setEnabled(True)
            self.Act_interactive_cut.setEnabled(True)
            self.Act_interactive_AutoSort.setEnabled(True)
            self.Act_interactive_ManualSort.setEnabled(True)
            self.Act_interactive_zoom_out.setEnabled(True)
            self.Act_interactive_zoom_in.setEnabled(True)
            self.Act_interactive_zoom_home.setEnabled(True)
            self.Act_interactive_zoom.setEnabled(True)
            self.Act_fine_spec.setEnabled(True)
            self.Act_norm_spec.setEnabled(True)
            self.Act_arrowkeys.setEnabled(True)
            self.Act_arrowkeys.setChecked(True)

            self.open_button.close()
            self.load_button.close()

    def Mzoom_in(self):
        self.Plot.zoom_in()
        self.Plot.clock_time(self.rec_datetime, self.times)
        self.Plot.figure.canvas.draw()

    def Mzoom_out(self):
        self.Plot.zoom_out()
        self.Plot.clock_time(self.rec_datetime, self.times)
        self.Plot.figure.canvas.draw()

    def Mposition_tracking(self):
        position_tracking(self.sign_v, self.ident_v, (self.elecs_y, self.elecs_x),
                          (self.elecs_y_spacing, self.elecs_x_spacing))

    def Mzoom_home(self):
        self.Plot.zoom_home()
        self.Plot.clock_time(self.rec_datetime, self.times)
        self.Plot.figure.canvas.draw()

    def execute(self):
        if self.Act_interactive_cut.isChecked():
            if self.active_id and self.active_idx_in_trace:
                self.cut()

        if self.Act_interactive_con.isChecked():
            if self.active_id and self.active_id2:
                self.connect()

        if self.Act_interactive_del.isChecked():
            if self.active_id:
                self.delete()

        if self.Act_interactive_GrCon.isChecked():
            if hasattr(self.active_idx, '__len__'):
                self.get_active_group_ids()
                self.group_connect()

        if self.Act_interactive_GrDel.isChecked():
            if hasattr(self.active_idx, '__len__'):
                self.get_active_group_ids()
                self.group_delete()

    def get_active_idx_rect(self):
        xlim = np.sort([self.x0, self.x1])
        ylim = np.sort([self.y0, self.y1])

        if not hasattr(self.active_idx, '__len__'):
            self.active_idx = np.arange(len(self.fund_v))[(self.fund_v >= np.min(ylim[0])) & (self.fund_v < np.max(ylim[1])) &
                                                          (self.times[self.idx_v] >= np.min(xlim[0])) & (self.times[self.idx_v] < np.max(xlim[1])) &
                                                          (~np.isnan(self.ident_v))]
        else:
            if self.Act_interactive_con.isChecked():
                self.active_idx2 = np.arange(len(self.fund_v))[
                    (self.fund_v >= np.min(ylim[0])) & (self.fund_v < np.max(ylim[1])) &
                    (self.times[self.idx_v] >= np.min(xlim[0])) & (self.times[self.idx_v] < np.max(xlim[1])) &
                    (~np.isnan(self.ident_v))]

    def get_active_id(self, idx):
        if not self.active_id:
            self.active_id = self.ident_v[idx[0]]

        elif self.active_id and not self.active_id2:
            self.active_id2 = self.ident_v[idx[0]]

    def get_active_group_ids(self):
        self.active_ids = np.unique(self.ident_v[self.active_idx])

    def get_active_idx_in_trace(self):
        self.active_idx_in_trace = np.arange(len(self.fund_v))[(self.ident_v == self.active_id) &
                                                               (self.times[self.idx_v] < self.x1)][-1]

    def cut(self):
        next_ident = np.max(self.ident_v[~np.isnan(self.ident_v)]) + 1
        self.ident_v[(self.ident_v == self.active_id) & (self.idx_v <= self.idx_v[self.active_idx_in_trace])] = next_ident

        self.Plot.active_idx_in_trace = None
        self.Plot.plot_traces(self.ident_v, self.times, self.idx_v, self.fund_v, task='post cut', active_id = self.active_id)

        self.reset_variables()

        self.Plot.canvas.draw()

        # if hasattr(self.id_tag, '__len__'):
        #     list_id_tag = list(self.id_tag)
        #     list_id_tag.append([new_ident, 0])
        #     self.id_tag = np.array(list_id_tag)

    def connect(self):
        overlapping_idxs = np.intersect1d(self.idx_v[self.ident_v == self.active_id],
                                          self.idx_v[self.ident_v == self.active_id2])

        # self.ident_v[(self.idx_v == overlapping_idxs) & (self.ident_v == self.active_ident0)] = np.nan
        self.ident_v[(np.in1d(self.idx_v, np.array(overlapping_idxs))) & (self.ident_v == self.active_id)] = np.nan
        self.ident_v[self.ident_v == self.active_id2] = self.active_id

        self.Plot.plot_traces(self.ident_v, self.times, self.idx_v, self.fund_v, task = 'post_connect', active_id = self.active_id, active_id2 = self.active_id2)

        self.reset_variables()

        self.Plot.canvas.draw()


        # if hasattr(self.id_tag, '__len__'):
        #     help_mask = [x in np.array(self.trace_handles)[:, 1] for x in self.id_tag[:, 0]]
        #     mask = np.arange(len(self.id_tag))[help_mask]
        #     self.id_tag = self.id_tag[mask]

    def delete(self):
        self.ident_v[self.ident_v == self.active_id] = np.nan
        self.Plot.plot_traces(self.ident_v, self.times, self.idx_v, self.fund_v, task = 'post_delete', active_id = self.active_id)

        self.reset_variables()

        self.Plot.canvas.draw()

        # self.trace_handles.pop(delete_handle_idx)
        # if hasattr(self.id_tag, '__len__'):
        #     help_mask = [x in np.array(self.trace_handles)[:, 1] for x in self.id_tag[:, 0]]
        #     mask = np.arange(len(self.id_tag))[help_mask]
        #     self.id_tag = self.id_tag[mask]

    def group_connect(self):
        target_ident = self.active_ids[0]

        for ai in self.active_ids:
            if ai == target_ident:
                continue

            overlapping_idxs = np.intersect1d(self.idx_v[self.ident_v == target_ident], self.idx_v[self.ident_v == ai])
            self.ident_v[(np.in1d(self.idx_v, np.array(overlapping_idxs))) & (self.ident_v == ai)] = np.nan
            self.ident_v[self.ident_v == ai] = target_ident

        self.Plot.plot_traces(self.ident_v, self.times, self.idx_v, self.fund_v, task = 'post_group_connect', active_ids=self.active_ids)

        self.reset_variables()
        self.Plot.canvas.draw()

        # if hasattr(self.id_tag, '__len__'):
        #     # embed()
        #     # quit()
        #     help_mask = [x in np.array(self.trace_handles)[:, 1] for x in self.id_tag[:, 0]]
        #     mask = np.arange(len(self.id_tag))[help_mask]
        #     self.id_tag = self.id_tag[mask]

    def group_delete(self):
        self.ident_v[self.active_idx] = np.nan
        self.Plot.plot_traces(self.ident_v, self.times, self.idx_v, self.fund_v, task = 'post_group_delete', active_ids=self.active_ids)

        self.reset_variables()
        self.Plot.canvas.draw()

        # if hasattr(self.id_tag, '__len__'):
        #     # embed()
        #     # quit()
        #     help_mask = [x in np.array(self.trace_handles)[:, 1] for x in self.id_tag[:, 0]]
        #     mask = np.arange(len(self.id_tag))[help_mask]
        #     self.id_tag = self.id_tag[mask]

    def reset_variables(self):
        self.active_idx = None
        self.active_id = None
        if self.Plot.active_id_handle0:
            self.Plot.active_id_handle0.remove()
        self.Plot.active_id_handle0 = None

        self.active_idx2 = None
        self.active_id2 = None
        if self.Plot.active_id_handle1:
            self.Plot.active_id_handle1.remove()
        self.Plot.active_id_handle1 = None

        self.active_idx_in_trace = None
        if self.Plot.active_cut_handle:
            self.Plot.active_cut_handle.remove()
        self.Plot.active_cut_handle = None

        self.active_ids = None
        if self.Plot.active_group_handle:
            self.Plot.active_group_handle.remove()
        self.Plot.active_group_handle = None

        self.Plot.canvas.draw()

    def stateAsk(self):
        # ToDo: remove !!!
        # print(self.Act_interactive_sel.isChecked())
        print(self.Act_interactive_cut.isChecked())

    def grid_layout(self):
        # ToDo: remove !!!
        self.Grid.show()

    def ManalysisDial(self):
        self.AnalysisDial.show()


    def show_updates(self):
        if self.AnalysisDial.got_changed:
            if self.Plot.spec_img_handle:
                self.Plot.spec_img_handle.remove()
            self.Plot.spec_img_handle = self.Plot.ax.imshow(decibel(self.AnalysisDial.tmp_spectra)[::-1],
                                                       extent=[self.AnalysisDial.start_time, self.AnalysisDial.end_time, 0, 2000],
                                                       aspect='auto', alpha=0.7, cmap='jet',
                                                       interpolation='gaussian')

            self.Plot.ax.set_xlim([self.AnalysisDial.start_time, self.AnalysisDial.end_time])
            self.Plot.ax.set_ylim([0, 2000])
            self.Plot.canvas.draw()

            self.AnalysisDial.got_changed = False


def main():
    app = QApplication(sys.argv)  # create application
    w = MainWindow()  # create window
    # p = PlotWidget()
    w.show()
    sys.exit(app.exec_())  # exit if window is closed


if __name__ == '__main__':
    main()

