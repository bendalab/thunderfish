"""
View and explore properties of EOD waveforms.
"""

import os
import glob
import sys
import argparse
import numpy as np
import scipy.signal as sig
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from multiprocessing import Pool, freeze_support, cpu_count
from .version import __version__, __year__
from .configfile import ConfigFile
from .tabledata import TableData, add_write_table_config, write_table_args
from .dataloader import load_data
from .multivariateexplorer import MultivariateExplorer
from .harmonics import add_harmonic_groups_config
from .eodanalysis import add_species_config
from .eodanalysis import wave_quality, wave_quality_args, add_eod_quality_config
from .eodanalysis import pulse_quality, pulse_quality_args
from .powerspectrum import decibel
from .bestwindow import find_best_window, plot_best_data
from .thunderfish import configuration, detect_eods, plot_eods


basename = ''


class EODExplorer(MultivariateExplorer):
    """Simple GUI for viewing and exploring properties of EOD waveforms.

    EODExplorer adapts a MultivariateExplorer to specific needs of EODs.

    Static members
    --------------
    - `groups`: names of groups of data columns that can be selected.
    - `select_EOD_properties()`: select data columns to be explored.
    - `select_color_property()`: select column from data table for colorizing the data.
    """
    
    def __init__(self, data, data_cols, wave_fish, eod_data,
                 add_waveforms, loaded_spec, rawdata_path):
        """
        Parameter
        ---------
        data: TableData
            Full table of EOD properties. Each row is a fish.
        data_cols: list of string or ints
            Names or indices of columns in `data` to be explored.
            You may use the static function `select_EOD_properties()`
            for assisting the selection of columns.
        wave_fish: boolean
            True if data are about wave-type weakly electric fish.
            False if data are about pulse-type weakly electric fish.
        eod_data: list of waveform data
            Either waveform data is only the EOD waveform,
            a ndarray of shape (time, ['time', 'voltage']), or
            it is a list with the first element being the EOD waveform,
            and the second element being a 2D ndarray of spectral properties
            of the EOD waveform with first column being the frequency or harmonics.
        add_waveforms: list of string
            List of what should be shown as waveform. Elements can be
            'first', 'second', 'ampl', 'power', or 'phase'. For 'first' and 'second'
            the first and second derivatives of the supplied EOD waveform a computed and shown.
            'ampl', 'power', and 'phase' select properties of the provided spectral properties.
        loaded_spec: boolean
            Indicates whether eod_data contains spectral properties.
        rawdata_path: string
            Base path to the raw recording, needed to show thunderfish
            when double clicking on a single EOD.
        """
        self.wave_fish = wave_fish
        self.eoddata = data
        self.path = rawdata_path
        MultivariateExplorer.__init__(self, data[:,data_cols],
                                      None, 'EODExplorer')
        tunit = 'ms'
        dunit = '1/ms'
        if wave_fish:
            tunit = '1/EODf'        
            dunit = 'EODf'
        wave_data = eod_data
        xlabels = ['Time [%s]' % tunit]
        ylabels = ['Voltage']
        if 'first' in add_waveforms:
            # first derivative:
            if loaded_spec:
                if hasattr(sig, 'savgol_filter'):
                    derivative = lambda x: (np.column_stack((x[0], \
                        sig.savgol_filter(x[0][:,1], 5, 2, 1, x[0][1,0]-x[0][0,0]))), x[1])
                else:
                    derivative = lambda x: (np.column_stack((x[0][:-1,:], \
                        np.diff(x[0][:,1])/(x[0][1,0]-x[0][0,0]))), x[1])
            else:
                if hasattr(sig, 'savgol_filter'):
                    derivative = lambda x: np.column_stack((x, \
                        sig.savgol_filter(x[:,1], 5, 2, 1, x[1,0]-x[0,0])))
                else:
                    derivative = lambda x: np.column_stack((x[:-1,:], \
                        np.diff(x[:,1])/(x[1,0]-x[0,0])))
            wave_data = list(map(derivative, wave_data))
            ylabels.append('dV/dt [%s]' % dunit)
            if 'second' in add_waveforms:
                # second derivative:
                if loaded_spec:
                    if hasattr(sig, 'savgol_filter'):
                        derivative = lambda x: (np.column_stack((x[0], \
                            sig.savgol_filter(x[0][:,1], 5, 2, 2, x[0][1,0]-x[0][0,0]))), x[1])
                    else:
                        derivative = lambda x: (np.column_stack((x[0][:-1,:], \
                            np.diff(x[0][:,2])/(x[0][1,0]-x[0][0,0]))), x[1])
                else:
                    if hasattr(sig, 'savgol_filter'):
                        derivative = lambda x: np.column_stack((x, \
                            sig.savgol_filter(x[:,1], 5, 2, 2, x[1,0]-x[0,0])))
                    else:
                        derivative = lambda x: np.column_stack((x[:-1,:], \
                            np.diff(x[:,2])/(x[1,0]-x[0,0])))
                wave_data = list(map(derivative, wave_data))
                ylabels.append('d^2V/dt^2 [%s^2]' % dunit)
        if loaded_spec:
            if wave_fish:
                indices = [0]
                phase = False
                xlabels.append('Harmonics')
                if 'ampl' in add_waveforms:
                    indices.append(3)
                    ylabels.append('Ampl [%]')
                if 'power' in add_waveforms:
                    indices.append(4)
                    ylabels.append('Power [dB]')
                if 'phase' in add_waveforms:
                    indices.append(5)
                    ylabels.append('Phase')
                    phase = True
                def get_spectra(x):
                    y = x[1][:,indices]
                    if phase:
                        y[y[:,-1]<0.0,-1] += 2.0*np.pi 
                    return (x[0], y)
                wave_data = list(map(get_spectra, wave_data))
            else:
                xlabels.append('Frequency [Hz]')
                ylabels.append('Power [dB]')
                def get_spectra(x):
                    y = x[1]
                    y[:,1] = decibel(y[:,1], None)
                    return (x[0], y)
                wave_data = list(map(get_spectra, wave_data))
        self.set_wave_data(wave_data, xlabels, ylabels, True)

        
    def fix_scatter_plot(self, ax, data, label, axis):
        """Customize an axes of a scatter plot.

        - Limits for amplitude and time like quantities start at zero.
        - Phases a labeled with multuples of pi.
        - Species labels are rotated.
        """
        if any(l in label for l in ['ampl', 'power', 'width',
                                    'time', 'tau', 'P2-P1-dist',
                                    'var', 'peak', 'trough',
                                    'dist', 'rms', 'noise']):
            if np.all(data[np.isfinite(data)] >= 0.0):
                if axis == 'x':
                    ax.set_xlim(0.0, None)
                elif axis == 'y':
                    ax.set_ylim(0.0, None)
                elif axis == 'c':
                    return 0.0, np.max(data), None
            else:
                if axis == 'x':
                    ax.set_xlim(None, 0.0)
                elif axis == 'y':
                    ax.set_ylim(None, 0.0)
                elif axis == 'c':
                    return np.min(data), 0.0, None
        elif 'phase' in label:
            if axis == 'x':
                ax.set_xlim(-np.pi, np.pi)
                ax.set_xticks(np.arange(-np.pi, 1.5*np.pi, 0.5*np.pi))
                ax.set_xticklabels([u'-\u03c0', u'-\u03c0/2', '0', u'\u03c0/2', u'\u03c0'])
            elif axis == 'y':
                ax.set_ylim(-np.pi, np.pi)
                ax.set_yticks(np.arange(-np.pi, 1.5*np.pi, 0.5*np.pi))
                ax.set_yticklabels([u'-\u03c0', u'-\u03c0/2', '0', u'\u03c0/2', u'\u03c0'])
            elif axis == 'c':
                if ax is not None:
                    ax.set_yticklabels([u'-\u03c0', u'-\u03c0/2', '0', u'\u03c0/2', u'\u03c0'])
                return -np.pi, np.pi, np.arange(-np.pi, 1.5*np.pi, 0.5*np.pi)
        elif 'species' in label:
            if axis == 'x':
                for label in ax.get_xticklabels():
                    label.set_rotation(30)
                ax.set_xlabel('')
                ax.set_xlim(np.min(data)-0.5, np.max(data)+0.5)
            elif axis == 'y':
                ax.set_ylabel('')
                ax.set_ylim(np.min(data)-0.5, np.max(data)+0.5)
            elif axis == 'c':
                if ax is not None:
                    ax.set_ylabel('')
        return np.min(data), np.max(data), None

    
    def fix_waveform_plot(self, axs, indices):
        """ Adapt waveform plots to EOD waveforms, derivatives, and spectra.
        """
        if len(indices) == 0:
            axs[0].text(0.5, 0.5, 'Click to plot EOD waveforms',
                        transform = axs[0].transAxes, ha='center', va='center')
            axs[0].text(0.5, 0.3, 'n = %d' % len(self.raw_data),
                        transform = axs[0].transAxes, ha='center', va='center')
        elif len(indices) == 1:
            file_name = self.eoddata[indices[0],'file'] if 'file' in self.eoddata else basename
            if 'index' in self.eoddata and np.isfinite(self.eoddata[indices[0],'index']) and \
              np.any(self.eoddata[:,'index'] != self.eoddata[0,'index']):
                axs[0].set_title('%s: %d' % (file_name,
                                             self.eoddata[indices[0],'index']))
            else:
                axs[0].set_title(file_name)
            if np.isfinite(self.eoddata[indices[0],'index']):
                axs[0].text(0.05, 0.85, '%.1fHz' % self.eoddata[indices[0],'EODf'],
                            transform = axs[0].transAxes)
        else:
            axs[0].set_title('%d EOD waveforms selected' % len(indices))
        for ax in axs:
            for l in ax.lines:
                l.set_linewidth(3.0)
        for ax, xl in zip(axs, self.wave_ylabels):
            if 'Voltage' in xl:
                ax.set_ylim(top=1.1)
                ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=4))
            if 'dV/dt' in xl:
                ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=4))
            if 'd^2V/dt^2' in xl:
                ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=4))
        if self.wave_fish:
            for ax, xl in zip(axs, self.wave_ylabels):
                if 'Voltage' in xl:
                    ax.set_xlim(-0.7, 0.7)
                if 'Ampl' in xl or 'Power' in xl or 'Phase' in xl:
                    ax.set_xlim(-0.5, 8.5)
                    for l in ax.lines:
                        l.set_marker('.')
                        l.set_markersize(15.0)
                        l.set_markeredgewidth(0.5)
                        l.set_markeredgecolor('k')
                        l.set_markerfacecolor(l.get_color())
                if 'Ampl' in xl:
                    ax.set_ylim(0.0, 100.0)
                    ax.yaxis.set_major_locator(ticker.MultipleLocator(25.0))
                if 'Power' in xl:
                    ax.set_ylim(-60.0, 2.0)
                    ax.yaxis.set_major_locator(ticker.MultipleLocator(20.0))
                if 'Phase' in xl:
                    ax.set_ylim(0.0, 2.0*np.pi)
                    ax.set_yticks(np.arange(0.0, 2.5*np.pi, 0.5*np.pi))
                    ax.set_yticklabels(['0', u'\u03c0/2', u'\u03c0', u'3\u03c0/2', u'2\u03c0'])
        else:
            for ax, xl in zip(axs, self.wave_ylabels):
                if 'Voltage' in xl:
                    ax.set_xlim(-1.0, 1.5)
                if 'Power' in xl:
                    ax.set_xlim(1.0, 2000.0)
                    ax.set_xscale('log')
                    ax.set_ylim(-60.0, 2.0)
                    ax.yaxis.set_major_locator(ticker.MultipleLocator(20.0))
        if len(indices) > 0:
            for ax in axs:
                ax.axhline(c='k', lw=1)

            
    def list_selection(self, indices):
        """ List file names and indices of selection.

        If only a single EOD is selected, list all of its properties.
        """
        if 'index' in self.eoddata and \
           np.any(self.eoddata[:,'index'] != self.eoddata[0,'index']):
            for i in indices:
                file_name = self.eoddata[i,'file'] if 'file' in self.eoddata else basename
                if np.isfinite(self.eoddata[i,'index']):
                    print('%s : %d' % (file_name, self.eoddata[i,'index']))
                else:
                    print(file_name)
        elif 'file' in self.eoddata:
            for i in indices:
                print(self.eoddata[i,'file'])
        if len(indices) == 1:
            # write eoddata line on terminal:
            keylen = 0
            keys = []
            values = []
            for c in range(self.eoddata.columns()):
                k, v = self.eoddata.key_value(indices[0], c)
                keys.append(k)
                values.append(v)
                if keylen < len(k):
                    keylen = len(k)
            for k, v in zip(keys, values):
                fs = '%%-%ds: %%s' % keylen
                print(fs % (k, v.strip()))

                
    def analyze_selection(self, index):
        """ Launch thunderfish on the selected EOD.
        """
        # load data:
        file_base = self.eoddata[index,'file'] if 'file' in self.eoddata else basename
        bp = os.path.join(self.path, file_base)
        fn = glob.glob(bp + '.*')
        if len(fn) == 0:
            print('no recording found for %s' % bp)
            return
        recording = fn[0]
        channel = 0
        try:
            raw_data, samplerate, unit = load_data(recording, channel)
        except IOError as e:
            print('%s: failed to open file: did you provide a path to the raw data (-P option)?' % (recording))
            return
        if len(raw_data) <= 1:
            print('%s: empty data file' % recording)
            return
        # load configuration:
        cfgfile = __package__ + '.cfg'
        cfg = configuration(cfgfile, False, recording)
        cfg.load_files(cfgfile, recording, 4)
        if 'flipped' in self.eoddata:
            fs = 'flip' if self.eoddata[index,'flipped'] else 'none'
            cfg.set('flipWaveEOD', fs)
            cfg.set('flipPulseEOD', fs)
        # best_window:
        data, idx0, idx1, clipped, min_clip, max_clip = find_best_window(raw_data, samplerate,
                                                                         cfg)
        # detect EODs in the data:
        psd_data, fishlist, _, eod_props, mean_eods, \
          spec_data, peak_data, power_thresh, skip_reason, zoom_window = \
          detect_eods(data, samplerate, min_clip, max_clip, recording, 0, 0, cfg)
        # plot EOD:
        idx = int(self.eoddata[index,'index']) if 'index' in self.eoddata else 0
        for k in ['toolbar', 'keymap.back', 'keymap.forward',
                  'keymap.zoom', 'keymap.pan']:
            plt.rcParams[k] = self.plt_params[k]
        fig = plot_eods(file_base, raw_data, samplerate, idx0, idx1, clipped,
                        psd_data[0], fishlist, None, mean_eods, eod_props, peak_data, spec_data,
                        [idx], unit, zoom_window, 10, None, True, False, 'auto',
                        False, 0.0, 3000.0, interactive=True, verbose=0)
        fig.canvas.set_window_title('thunderfish: %s' % file_base)
        plt.show(block=False)


    """ Names of groups of data columns that can be selected by the select_EOD_properties() function.
    """
    groups = ['all', 'allpower', 'noise', 'timing',
              'ampl', 'relampl', 'power', 'relpower', 'phase',
              'time', 'width', 'peaks', 'none']
    
    @staticmethod
    def select_EOD_properties(data, wave_fish, max_n, column_groups, add_columns):
        """ Select data columns to be explored.

        First, groups of columns are selected, then individual
        columns. Columns that are selected twice are removed from the
        selection.

        Parameter
        ---------
        data: TableData
            Table with EOD properties from which columns are selected.
        wave_fish: boolean.
            Indicates if data contains properties of wave- or pulse-type electric fish.
        max_n: int
            Maximum number of harmonics (wae-type fish) or peaks (pulse-type fish)
            to be  selected.
        column_groups: list of string
            List of name denoting groups of columns to be selected. Supported groups are
            listed in `EODExplor.groups`.
        add_columns: list of string or int
            List of further individual columns to be selected.

        Returns
        -------
        data_cols: list of int
            Indices of data columns to be shown by EODExplorer.
        error: string
            In case of an invalid column group, an error string.
        """
        if wave_fish:
            # maximum number of harmonics:
            if max_n == 0:
                max_n = 100
            else:
                max_n += 1
            for k in range(1, max_n):
                if not ('phase%d' % k) in data:
                    max_n = k
                    break
        else:
            # minimum number of peaks:
            min_peaks = -10
            for k in range(1, min_peaks, -1):
                if not ('P%dampl' % k) in data or not np.all(np.isfinite(data[:,'P%dampl' % k])):
                    min_peaks = k+1
                    break
            # maximum number of peaks:
            if max_n == 0:
                max_peaks = 20
            else:
                max_peaks = max_n + 1
            for k in range(1, max_peaks):
                if not ('P%dampl' % k) in data or not np.all(np.isfinite(data[:,'P%dampl' % k])):
                    max_peaks = k
                    break

        # default columns:
        group_cols = ['EODf']
        if 'EODf_adjust' in data:
            group_cols.append('EODf_adjust')
        if len(column_groups) == 0:
            column_groups = ['all']
        for group in column_groups:
            if group == 'none':
                group_cols = []
            elif wave_fish:
                if group == 'noise':
                    group_cols.extend(['noise', 'rmserror', 'power', 'thd',
                                       'dbdiff', 'maxdb', 'p-p-amplitude',
                                       'relampl1', 'relampl2', 'relampl3'])
                elif group == 'timing' or group == 'time':
                    group_cols.extend(['peakwidth', 'troughwidth', 'p-p-distance',
                                       'leftpeak', 'rightpeak', 'lefttrough', 'righttrough'])
                elif group == 'ampl':
                    for k in range(0, max_n):
                        group_cols.append('ampl%d' % k)
                elif group == 'relampl':
                    group_cols.append('thd')
                    group_cols.append('reltroughampl')
                    for k in range(1, max_n):
                        group_cols.append('relampl%d' % k)
                elif group == 'relpower' or group == 'power':
                    for k in range(1, max_n):
                        group_cols.append('relpower%d' % k)
                elif group == 'phase':
                    for k in range(0, max_n):
                        group_cols.append('phase%d' % k)
                elif group == 'all':
                    group_cols.append('thd')
                    group_cols.append('reltroughampl')
                    for k in range(1, max_n):
                        group_cols.append('relampl%d' % k)
                        group_cols.append('phase%d' % k)
                elif group == 'allpower':
                    group_cols.append('thd')
                    for k in range(1, max_n):
                        group_cols.append('relpower%d' % k)
                        group_cols.append('phase%d' % k)
                else:
                    return None, '"%s" is not a valid data group for wavefish' % group
            else:  # pulse fish
                if group == 'noise':
                    group_cols.extend(['noise', 'p-p-amplitude', 'min-ampl', 'max-ampl'])
                elif group == 'timing':
                    group_cols.extend(['tstart', 'tend', 'width', 'tau', 'P2-P1-dist', 'firstpeak', 'lastpeak'])
                elif group == 'power':
                    group_cols.extend(['peakfreq', 'peakpower', 'poweratt5', 'poweratt50', 'lowcutoff'])
                elif group == 'time':
                    for k in range(min_peaks, max_peaks):
                        if k != 1:
                            group_cols.append('P%dtime' % k)
                elif group == 'ampl':
                    for k in range(min_peaks, max_peaks):
                        group_cols.append('P%dampl' % k)
                elif group == 'relampl':
                    for k in range(min_peaks, max_peaks):
                        if k != 1:
                            group_cols.append('P%drelampl' % k)
                elif group == 'width':
                    for k in range(min_peaks, max_peaks):
                        group_cols.append('P%dwidth' % k)
                elif group == 'peaks':
                    group_cols.append('firstpeak')
                    group_cols.append('lastpeak')
                elif group == 'all':
                    group_cols.extend(['firstpeak', 'lastpeak'])
                    for k in range(min_peaks, max_peaks):
                        if k != 1:
                            group_cols.append('P%drelampl' % k)
                            group_cols.append('P%dtime' % k)
                        group_cols.append('P%dwidth' % k)
                    group_cols.extend(['tau', 'P2-P1-dist', 'peakfreq', 'poweratt5'])
                else:
                    return None, '"%s" is not a valid data group for pulsefish' % group
        # additional data columns:
        group_cols.extend(add_columns)
        # translate to indices:
        data_cols = []
        for c in group_cols:
            idx = data.index(c)
            if idx is None:
                print('"%s" is not a valid data column' % c)
            elif idx in data_cols:
                data_cols.remove(idx)
            else:
                data_cols.append(idx)
        return data_cols, None

    
    @staticmethod
    def select_color_property(data, data_cols, color_col):
        """ Select column from data table for colorizing the data.

        Pass the output of this function on to MultivariateExplorer.set_colors().

        Parameter
        ---------
        data: TableData
            Table with all EOD properties from which columns are selected.
        data_cols: list of int
            List of columns selected to be explored.
        color_col: string or int
            Column to be selected for coloring the data.
            If 'row' then use the row index of the data in the table for coloring.

        Returns
        -------
        colors: int or column from data.
            Either index of `data_cols` or additional data from the data table
            to be used for coloring.
        color_label: string
            Label for labeling the color bar.
        color_idx: int or None
            Index of column in `data`.
        error: string
            In case an invalid column is selected, an error string.
        """
        color_idx = data.index(color_col)
        colors = None
        color_label = None
        if color_idx is None and color_col != 'row':
            return None, None, None, '"%s" is not a valid column for color code' % color_col
        if color_idx is None:
            colors = -2
        elif color_idx in data_cols:
            colors = data_cols.index(color_idx)
        else:
            if len(data.unit(color_idx)) > 0 and not data.unit(color_idx) in ['-', '1']:
                color_label = '%s [%s]' % (data.label(color_idx), data.unit(color_idx))
            else:
                color_label = data.label(color_idx)
            colors = data[:,color_idx]
        return colors, color_label, color_idx, None


class PrintHelp(argparse.Action):
    def __call__(self, parser, namespace, values, option_string):
        parser.print_help()
        print('')
        print('mouse:')
        for ma in MultivariateExplorer.mouse_actions:
            print('%-23s %s' % ma)
        print('%-23s %s' % ('double left click', 'run thunderfish on selected EOD waveform'))
        print('')
        print('key shortcuts:')
        for ka in MultivariateExplorer.key_actions:
            print('%-23s %s' % ka)
        parser.exit()      

        
wave_fish = True
load_spec = False
data = None
data_path = None

def load_waveform(idx):
    eodf = data[idx,'EODf']
    if not np.isfinite(eodf):
        if load_spec:
            return None, None
        else:
            return None
    file_name = data[idx,'file'] if 'file' in data else '-'.join(basename.split('-')[:-1])
    file_index = data[idx,'index'] if 'index' in data else 0
    eod_filename = os.path.join(data_path, '%s-eodwaveform-%d.csv' % (file_name, file_index))
    eod_table = TableData(eod_filename)
    eod = eod_table[:,'mean']
    norm = np.max(eod)
    if wave_fish:
        eod = np.column_stack((eod_table[:,'time']*0.001*eodf, eod/norm))
    else:
        eod = np.column_stack((eod_table[:,'time'], eod/norm))
    if not load_spec:
        return eod
    fish_type = 'wave' if wave_fish else 'pulse'
    spec_table = TableData(os.path.join(data_path, '%s-%sspectrum-%d.csv' % (file_name, fish_type, file_index)))
    spec_data = spec_table.array()
    if not wave_fish:
        spec_data = spec_data[spec_data[:,0]<2000.0,:]
        spec_data = spec_data[::5,:]
    return (eod, spec_data)
        

def main():
    global data
    global wave_fish
    global load_spec
    global data_path
    global basename

    # command line arguments:
    parser = argparse.ArgumentParser(add_help=False,
        description='View and explore properties of EOD waveforms.',
        epilog='version %s by Benda-Lab (2019-%s)' % (__version__, __year__))
    parser.add_argument('-h', '--help', nargs=0, action=PrintHelp,
                        help='show this help message and exit')
    parser.add_argument('--version', action='version', version=__version__)
    parser.add_argument('-v', dest='verbose', action='store_true', default=False,
                        help='verbose output')
    parser.add_argument('-l', dest='list_columns', action='store_true',
                        help='list all available data columns and exit')
    parser.add_argument('-j', dest='jobs', nargs='?', type=int, default=None, const=0,
                        help='number of jobs run in parallel for loading waveform data. Without argument use all CPU cores.')
    parser.add_argument('-D', dest='column_groups', default=[], type=str, action='append',
                        choices=EODExplorer.groups,
                        help='default selection of data columns, check them with the -l option')
    parser.add_argument('-d', dest='add_data_cols', action='append', default=[], metavar='COLUMN',
                        help='data columns to be appended or removed (if already listed) for analysis')
    parser.add_argument('-n', dest='max_n', default=0, type=int, metavar='MAX',
                        help='maximum number of harmonics or peaks to be used')
    parser.add_argument('-w', dest='add_waveforms', default=[], type=str, action='append',
                        choices=['first', 'second', 'ampl', 'power', 'phase'],
                        help='add first or second derivative of EOD waveform, or relative amplitude, power, or phase to the plot of selected EODs.')
    parser.add_argument('-s', dest='save_pca', action='store_true',
                        help='save PCA components and exit')
    parser.add_argument('-c', dest='color_col', default='EODf', type=str, metavar='COLUMN',
                        help='data column to be used for color code or "row"')
    parser.add_argument('-m', dest='color_map', default='jet', type=str, metavar='CMAP',
                        help='name of color map')
    parser.add_argument('-p', dest='data_path', default='.', type=str, metavar='PATH',
                        help='path to the analyzed EOD waveform data')
    parser.add_argument('-P', dest='rawdata_path', default='.', type=str, metavar='PATH',
                        help='path to the raw EOD recordings')
    parser.add_argument('-f', dest='format', default='auto', type=str,
                        choices=TableData.formats + ['same'],
                        help='file format used for saving PCA data ("same" uses same format as input file)')
    parser.add_argument('file', default='', type=str,
                        help='a wavefish.* or pulsefish.* summary file as generated by collectfish')
    args = parser.parse_args()
        
    # read in command line arguments:    
    verbose = args.verbose
    list_columns = args.list_columns
    jobs = args.jobs
    file_name = args.file
    column_groups = args.column_groups
    add_data_cols = args.add_data_cols
    max_n = args.max_n
    add_waveforms = args.add_waveforms
    save_pca = args.save_pca
    color_col = args.color_col
    color_map = args.color_map
    data_path = args.data_path
    rawdata_path = args.rawdata_path
    data_format = args.format
    
    # read configuration:
    cfgfile = __package__ + '.cfg'
    cfg = ConfigFile()
    add_eod_quality_config(cfg)
    add_harmonic_groups_config(cfg)
    add_species_config(cfg)
    add_write_table_config(cfg, table_format='csv', unit_style='row',
                           align_columns=True, shrink_width=False)
    cfg.load_files(cfgfile, file_name, 4)
    
    # output format:
    if data_format == 'same':
        ext = os.path.splitext(file_name)[1][1:]
        if ext in TableData.ext_formats:
            data_format = TableData.ext_formats[ext]
        else:
            data_format = 'dat'
    if data_format != 'auto':
        cfg.set('fileFormat', data_format)

    # check color map:
    if not color_map in plt.colormaps():
        parser.error('"%s" is not a valid color map' % color_map)
        
    # load summary data:
    wave_fish = 'wave' in file_name
    data = TableData(file_name)

    # basename:
    basename = os.path.splitext(os.path.basename(file_name))[0]
    
    # check quality:
    skipped = 0
    for r in reversed(range(data.rows())):
        idx = 0
        if 'index' in data:
            idx = data[r,'index']
        skips = ''
        if wave_fish:
            harm_rampl = np.array([0.01*data[r,'relampl%d'%(k+1)] for k in range(3)
                                   if 'relampl%d'%(k+1) in data])
            props = data.row_dict(r)
            if 'clipped' in props:
                props['clipped'] *= 0.01 
            if 'noise' in props:
                props['noise'] *= 0.01 
            if 'rmserror' in props:
                props['rmserror'] *= 0.01
            if 'thd' in props:
                props['thd'] *= 0.01 
            _, skips, msg = wave_quality(props, harm_rampl, **wave_quality_args(cfg))
        else:
            props = data.row_dict(r)
            if 'clipped' in props:
                props['clipped'] *= 0.01 
            if 'noise' in props:
                props['noise'] *= 0.01 
            skips, msg, _ = pulse_quality(props, **pulse_quality_args(cfg))
        if len(skips) > 0:
            if verbose:
                print('skip fish %2d from %s: %s' % (idx, data[r,'file'] if 'file' in data else basename, skips))
            del data[r,:]
            skipped += 1
    if verbose and skipped > 0:
        print('')

    # select columns (EOD properties) to be shown:
    data_cols, error = \
      EODExplorer.select_EOD_properties(data, wave_fish, max_n,
                                        column_groups, add_data_cols)
    if error:
        parser.error(error)

    # select column used for coloring the data:
    colors, color_label, color_idx, error = \
      EODExplorer.select_color_property(data, data_cols, color_col)
    if error:
        parser.error(error)

    # list columns:
    if list_columns:
        for k, c in enumerate(data.keys()):
            s = [' '] * 3
            if k in data_cols:
                s[1] = '*'
            if color_idx is not None and k == color_idx:
                s[0] = 'C'
            print(''.join(s) + c)
        parser.exit()

    # load waveforms:
    load_spec = 'ampl' in add_waveforms or 'power' in add_waveforms or 'phase' in add_waveforms
    if jobs is not None:
        cpus = cpu_count() if jobs == 0 else jobs
        p = Pool(cpus)
        eod_data = p.map(load_waveform, range(data.rows()))
        del p
    else:
        eod_data = list(map(load_waveform, range(data.rows())))

    # explore:
    eod_expl = EODExplorer(data, data_cols, wave_fish, eod_data,
                           add_waveforms, load_spec, rawdata_path)
    # write pca:
    if save_pca:
        eod_expl.compute_pca(False)
        eod_expl.save_pca(basename, False, **write_table_args(cfg))
        eod_expl.compute_pca(True)
        eod_expl.save_pca(basename, True, **write_table_args(cfg))
    else:
        eod_expl.set_colors(colors, color_label, color_map)
        eod_expl.show()


if __name__ == '__main__':
    freeze_support()  # needed by multiprocessing for some weired windows stuff
    main()
