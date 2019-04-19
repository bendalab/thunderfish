"""
View and explore multivariate data.
"""

import numpy as np
from sklearn import decomposition
from sklearn import preprocessing
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.widgets as widgets
import argparse
from .version import __version__, __year__
from .tabledata import TableData


class MultivariateExplorer(object):
    """Simple matplotlib-based GUI for exploring multivariate data.

    Shown are scatter plots of all pairs of variables or PCA axis.
    Scatter plots are colored according to one of the variables.
    Data points can be selected and corresponding waveforms are shown.

    Customize the appearance and information provided by subcallsing MultivariateExplorer and
    reimplementing the functions
    fix_scatter_plot(), fix_waveform_plot(), list_selection(), analyze_selection().
    See the documentation of these functions for details.
    """
    
    def __init__(self, data, labels=None, title=None):
        """ Initialize explorer with scatter-plot data.

        Parameter
        ---------
        data: TableData, 2D array, or list of 1D arrays
            The data to be explored. Each column is a variable.
            For the 2D array the columns are the second dimension,
            for a list of 1D arrays, the list goes over columns,
            i.e. each 1D array is one column.
        labels: list of string
            If data is not a TableData, then this provides labels
            for the data columns.
        title: string
            Title for the window.
        """
        # data and labels:
        if isinstance(data, TableData):
            self.categories = []
            for c, col in enumerate(data):
                if not isinstance(col[0], (int, float)):
                    # categorial data:
                    cats = sorted(set(col))
                    idxcol = [cats.index(x) for x in col]
                    data[:,c] = idxcol
                    self.categories.append(cats)
                else:
                    self.categories.append(None)
            self.raw_data = data.array()
            if labels is None:
                self.raw_labels = []
                for c in range(len(data)):
                    if data.unit(c):
                        self.raw_labels.append('%s [%s]' % (data.label(c), data.unit(c)))
                    else:
                        self.raw_labels.append(data.label(c))
            else:
                self.raw_labels = labels
        else:
            if isinstance(data, np.ndarray):
                self.raw_data = data
            else:
                self.categories = []
                for c, col in enumerate(data):
                    if not isinstance(col[0], (int, float)):
                        # categorial data:
                        cats = sorted(set(col))
                        idxcol = [cats.index(x) for x in col]
                        data[c] = idxcol
                        self.categories.append(cats)
                    else:
                        self.categories.append(None)
                self.raw_data = np.asarray(data).T
            self.raw_labels = labels
        self.title = title if title is not None else 'MultivariateExplorer'
        # no pca data yet:
        self.all_data = [self.raw_data, None, None]
        self.all_labels = [self.raw_labels, None, None]
        self.all_maxcols = [self.raw_data.shape[1], None, None]
        self.all_titles = ['data', 'PCA', 'scaled PCA']
        # pca:
        self.pca_tables = [None, None]
        self._pca_header(data, labels)
        # start showing raw data:
        self.show_mode = 0
        self.data = self.all_data[self.show_mode]
        self.labels = self.all_labels[self.show_mode]
        self.maxcols = self.all_maxcols[self.show_mode]
        if self.maxcols > 6:
            self.maxcols = 6
        # waveform data:
        self.wave_data = []
        self.wave_xlabels = ''
        self.wave_ylabels = []
        self.wave_title = False
        # colors:
        self.color_map = None
        self.extra_colors = None
        self.extra_color_label = None
        self.extra_categories = None
        self.color_values = None
        self.color_set_index = None
        self.color_index = None
        self.color_label = None
        self.color_set_index = 0
        self.color_index = 0
        self.data_colors = None
        self.color_vmin = None
        self.color_vmax = None
        self.color_ticks = None
        self.cbax = None
        # figure variables:
        self.plt_params = {}
        for k in ['toolbar', 'keymap.quit', 'keymap.back', 'keymap.forward',
                  'keymap.zoom', 'keymap.pan', 'keymap.xscale', 'keymap.yscale']:
            self.plt_params[k] = plt.rcParams[k]
            if k != 'toolbar':
                plt.rcParams[k] = ''
        self.xborder = 70.0  # pixel for ylabels
        self.yborder = 50.0  # pixel for xlabels
        self.spacing = 10.0  # pixel between plots
        self.pick_radius = 4.0
        # histogram plots:
        self.hist_ax = []
        self.hist_indices = []
        self.hist_selector = []
        self.hist_nbins = 30
        # scatter plots:
        self.scatter_ax = []
        self.scatter_indices = []
        self.scatter_artists = []
        self.sctter_selector = []
        self.scatter = True
        self.mark_data = []
        self.select_zooms = False
        self.zoom_stack = []
        # magnified scatter plot:
        self.magnified_on = False
        self.magnified_backdrop = None
        self.magnified_size = np.array([0.5, 0.5])
        # waveform plots:
        self.wave_ax = []


    def set_wave_data(self, data, xlabels='', ylabels=[], title=False):
        """ Add waveform data to explorer.

        Parameter
        ---------
        data: (list of) list of 2D arrays
            Waveform data associated with each row `i` of the data.
            Either for a single plot: `wave_data[i][:,x], wave_data[i][:,y]`
            or for several plots `p`:
            `wave_data[p][i][:,x], wave_data[p][i][:,y]`.
            Note that the first (outermost) list is over plots.
            The inner lists are over data points.
        xlabel: string or list of strings
            The xlabels for the waveform plots. If only a string is given, then
            there will be a common xaxis for all the plots, and only the lowest
            one gets a labeled xaxis. If a list of strings is given, each waveform
            plot gets its own labeled x-axis.
        ylabels: list of strings
            The ylabels for each of the waveform plots.
        title: bool or string
            If True or a string, povide space on top of the waveform plots for a title.
            If string, set this as the title for the waveform plots.
        """
        self.wave_data = []
        if data is not None and len(data) > 0:
            if isinstance(data[0], np.ndarray):
                self.wave_data = [data]
            else:
                self.wave_data = data
        self.wave_xlabels = xlabels
        self.wave_ylabels = ylabels
        self.wave_title = title
        self.wave_ax = []

        
    def set_colors(self, colors=0, color_label=None, color_map=None):
        """ Set data column used to color scatter plots.
        
        Parameter
        ---------
        colors: int or 1D array
           Index to colum in data to be used for coloring scatter plots.
           Or data array used to color scaler plots.
        color_label: string
           If colors is an array, this is a label describing the data.
           It is used to label the color bar.
        color_map: string or None
            Name of a matplotlib color map.
            If None 'jet' is used.
        """
        if isinstance(colors, int):
            self.color_set_index = 0
            self.color_index = colors
        else:
            if not isinstance(colors[0], (int, float)):
                # categorial data:
                cats = sorted(set(colors))
                idxcol = [cats.index(x) for x in colors]
                self.extra_colors = idxcol
                self.extra_categories = cats
            else:
                self.extra_colors = colors
            self.extra_color_label = color_label
            self.color_set_index = -1
            self.color_index = 1
        self.color_map = plt.get_cmap(color_map if color_map else 'jet')

        
    def show(self):
        """ Show the interactive scatter plots for exploration.
        """
        plt.rcParams['toolbar'] = 'None'
        plt.rcParams['keymap.quit'] = 'ctrl+w, alt+q, q'
        self.fig = plt.figure(facecolor='white')
        self.fig.canvas.set_window_title(self.title + ': ' + self.all_titles[self.show_mode])
        self.fig.canvas.mpl_connect('key_press_event', self._on_key)
        self.fig.canvas.mpl_connect('resize_event', self._on_resize)
        self.fig.canvas.mpl_connect('pick_event', self._on_pick)
        if self.color_map is None:
            self.color_map = plt.get_cmap('jet')
        self._set_color_column()
        self._init_hist_plots()
        self._init_scatter_plots()
        if len(self.wave_data) > 0:
            self.wave_ax = []
            axx = None
            for k in range(len(self.wave_data)):
                ax = self.fig.add_subplot(1, len(self.wave_data), 1+k, sharex=axx)
                self.wave_ax.append(ax)
                if isinstance(self.wave_xlabels, (list, tuple)):
                    axx=ax
            if not isinstance(self.wave_xlabels, (list, tuple)):
                for ax in self.wave_ax[:-1]:
                    plt.setp(ax.get_xticklabels(), visible=False)
                self.wave_ax[-1].set_xlabel(self.wave_xlabels)
            else:
                for ax, xlabel in zip(self.wave_ax, self.wave_xlabels):
                    ax.set_xlabel(xlabel)
            for ax, ylabel in zip(self.wave_ax, self.wave_ylabels):
                ax.set_ylabel(ylabel)
            if not isinstance(self.wave_title, bool) and self.wave_title:
                self.wave_ax[0].set_title(self.wave_title)
            self.fix_waveform_plot(self.wave_ax, self.mark_data)
        self._plot_magnified_scatter()
        plt.show()


    def _pca_header(self, data, labels):
        if isinstance(data, TableData):
            header = data.table_header()
        else:
            lbs = []
            for l in labels:
                if '[' in l:
                    lbs.append(l.split('[')[0].strip())
                elif '/' in l:
                    lbs.append(l.split('/')[0].strip())
                else:
                    lbs.append(l)
            header = TableData(header=lbs)
        header.set_formats('%.3f')
        header.insert(0, ['PC'] + ['-']*header.nsecs, '', '%d')
        header.insert(1, 'variance', '%', '%.3f')
        for k in range(len(self.pca_tables)):
            self.pca_tables[k] = TableData(header)

                
    def compute_pca(self, scale=False, write=False):
        """ Compute PCA based on the data.

        Parameter
        ---------
        scale: boolean
            If True standardize data before computing PCA, i.e. remove mean
            of each variabel and divide by its standard deviation.
        write: boolean
            If True write PCA components to standard out.
        """
        # pca:
        pca = decomposition.PCA()
        if scale:
            scaler = preprocessing.StandardScaler()
            scaler.fit(self.raw_data)
            pca.fit(scaler.transform(self.raw_data))
            pca_label = 'sPC'
        else:
            pca.fit(self.raw_data)
            pca_label = 'PC'
        for k in range(len(pca.components_)):
            if np.abs(np.min(pca.components_[k])) > np.max(pca.components_[k]):
                pca.components_[k] *= -1.0
        pca_data = pca.transform(self.raw_data)
        pca_labels = [('%s%d (%.1f%%)' if v > 0.01 else '%s%d (%.2f%%)') % (pca_label, k+1, 100.0*v)
                           for k, v in enumerate(pca.explained_variance_ratio_)]
        if np.min(pca.explained_variance_ratio_) >= 0.01:
            pca_maxcols = pca_data.shape[1]
        else:
            pca_maxcols = np.argmax(pca.explained_variance_ratio_ < 0.01)
        if pca_maxcols < 2:
            pca_maxcols = 2
        if pca_maxcols > 6:
            pca_maxcols = 6
        # table with PCA feature weights:
        pca_table = self.pca_tables[1] if scale else self.pca_tables[0]
        pca_table.clear_data()
        pca_table.set_section(pca_label, 0, pca_table.nsecs)
        for k, comp in enumerate(pca.components_):
            pca_table.append_data(k+1, 0)
            pca_table.append_data(100.0*pca.explained_variance_ratio_[k])
            pca_table.append_data(comp)
        if write:
            pca_table.write(table_format='out', unitstyle='none')
        # submit data:
        if scale:
            self.all_data[2] = pca_data
            self.all_labels[2] = pca_labels
            self.all_maxcols[2] = pca_maxcols
        else:
            self.all_data[1] = pca_data
            self.all_labels[1] = pca_labels
            self.all_maxcols[1] = pca_maxcols

            
    def save_pca(self, file_name, scale, **kwargs):
        """ Write PCA data to file.

        Parameter
        ---------
        file_name: string
            Name of ouput file.
        scale: boolean
            If True write PCA components of standardized PCA.
        kwargs: dict
            Additional parameter for TableData.write()
        """
        if scale:
            pca_file = file_name + '-pcacor'
            pca_table = self.pca_tables[1]
        else:
            pca_file = file_name + '-pcacov'
            pca_table = self.pca_tables[0]
        if 'unitstyle' in kwargs:
            del kwargs['unitstyle']
        if 'table_format' in kwargs:
            pca_table.write(pca_file, unitstyle='none', **kwargs)
        else:
            pca_file += '.dat'
            pca_table.write(pca_file, unitstyle='none')

            
    def _set_color_column(self):
        if self.color_set_index == -1:
            if self.color_index == 0:
                self.color_values = np.arange(self.data.shape[0], dtype=np.float)
                self.color_label = 'index'
            elif self.color_index == 1:
                self.color_values = self.extra_colors
                self.color_label = self.extra_color_label
        else:
            self.color_values = self.all_data[self.color_set_index][:,self.color_index]
            self.color_label = self.all_labels[self.color_set_index][self.color_index]
        self.color_vmin, self.color_vmax, self.color_ticks = \
          self.fix_scatter_plot(self.cbax, self.color_values, self.color_label, 'c')
        if self.color_ticks is None:
            if self.color_set_index == 0 and \
               self.categories[self.color_index] is not None:
                self.color_ticks = np.arange(len(self.categories[self.color_index]))
            elif self.color_set_index == -1 and \
                 self.color_index == 1 and \
                 self.extra_categories is not None:
                self.color_ticks = np.arange(len(self.extra_categories))
        self.data_colors = self.color_map((self.color_values - self.color_vmin)/(self.color_vmax - self.color_vmin))

                            
    def _plot_hist(self, ax, magnifiedax, keep_lims):
        ax_xlim = ax.get_xlim()
        ax_ylim = ax.get_ylim()
        try:
            idx = self.hist_ax.index(ax)
            c = self.hist_indices[idx]
            in_hist = True
        except ValueError:
            idx = self.scatter_ax.index(ax)
            c = self.scatter_indices[-1][0]
            in_hist = False
        ax.clear()
        ax.relim()
        ax.autoscale(True)
        ax.hist(self.data[:,c], self.hist_nbins)
        ax.set_xlabel(self.labels[c])
        if self.categories[c] is not None:
            ax.set_xticks(np.arange(len(self.categories[c])))
            ax.set_xticklabels(self.categories[c])
        self.fix_scatter_plot(ax, self.data[:,c], self.labels[c], 'x')
        if magnifiedax:
            ax.set_ylabel('count')
            cax = self.hist_ax[self.scatter_indices[-1][0]]
            ax.set_xlim(cax.get_xlim())
        else:
            if c == 0:
                ax.set_ylabel('count')
            else:
                plt.setp(ax.get_yticklabels(), visible=False)
        if keep_lims:
            ax.set_xlim(*ax_xlim)
            ax.set_ylim(*ax_ylim)
        try:
            selector = widgets.RectangleSelector(ax, self._on_select,
                                                 drawtype='box', useblit=True, button=1,
                                                 state_modifier_keys=dict(move='', clear='', square='', center=''))
        except TypeError:
            selector = widgets.RectangleSelector(ax, self._on_select, drawtype='box',
                                                 useblit=True, button=1)
        if in_hist:
            self.hist_selector[idx] = selector
        else:
            self.sctter_selector[idx] = selector
            self.scatter_artists[idx] = None
        if magnifiedax:
            bbox = ax.get_tightbbox(self.fig.canvas.get_renderer())
            if bbox is not None:
                self.magnified_backdrop = patches.Rectangle((bbox.x0, bbox.y0),
                                                            bbox.width, bbox.height,
                                                            transform=None, clip_on=False,
                                                            facecolor='white', edgecolor='none',
                                                            alpha=0.8, zorder=-5)
                ax.add_patch(self.magnified_backdrop)

                        
    def _init_hist_plots(self):
        n = self.data.shape[1]
        yax = None
        self.hist_ax = []
        for r in range(n):
            ax = self.fig.add_subplot(n, n, (n-1)*n+r+1, sharey=yax)
            self.hist_ax.append(ax)
            self.hist_indices.append(r)
            self.hist_selector.append(None)
            self._plot_hist(ax, False, False)
            yax = ax

                        
    def _plot_scatter(self, ax, magnifiedax, keep_lims, cax=None):
        ax_xlim = ax.get_xlim()
        ax_ylim = ax.get_ylim()
        idx = self.scatter_ax.index(ax)
        c, r = self.scatter_indices[idx]
        if self.scatter:
            ax.clear()
            ax.relim()
            ax.autoscale(True)
            a = ax.scatter(self.data[:,c], self.data[:,r], c=self.color_values,
                           cmap=self.color_map, vmin=self.color_vmin, vmax=self.color_vmax,
                           s=50, edgecolors='none', zorder=10)
            if cax is not None:
                self.fig.colorbar(a, cax=cax, ticks=self.color_ticks)
                cax.set_ylabel(self.color_label)
                self.color_vmin, self.color_vmax, self.color_ticks = \
                  self.fix_scatter_plot(self.cbax, self.color_values, self.color_label, 'c')
                if self.color_ticks is None:
                    if self.color_set_index == 0 and \
                       self.categories[self.color_index] is not None:
                        cax.set_yticklabels(self.categories[self.color_index])
                    elif self.color_set_index == -1 and \
                         self.color_index == 1 and \
                         self.extra_categories is not None:
                        cax.set_yticklabels(self.extra_categories)
        else:
            ax.autoscale(True)
            self.fix_scatter_plot(ax, self.data[:,c], self.labels[c], 'x')
            self.fix_scatter_plot(ax, self.data[:,r], self.labels[r], 'y')
            axrange = [ax.get_xlim(), ax.get_ylim()]
            ax.clear()
            ax.hist2d(self.data[:,c], self.data[:,r], self.hist_nbins, range=axrange,
                      cmap=plt.get_cmap('Greys'))
        a = ax.scatter(self.data[self.mark_data,c], self.data[self.mark_data,r],
                       c=self.data_colors[self.mark_data], s=80, zorder=11)
        self.scatter_artists[idx] = a
        if self.categories[c] is not None:
            ax.set_xticks(np.arange(len(self.categories[c])))
            ax.set_xticklabels(self.categories[c])
        if self.categories[r] is not None:
            ax.set_yticks(np.arange(len(self.categories[r])))
            ax.set_yticklabels(self.categories[r])
        if magnifiedax:
            ax.set_xlabel(self.labels[c])
            ax.set_ylabel(self.labels[r])
            cax = self.scatter_ax[self.scatter_indices[:-1].index(self.scatter_indices[-1])]
            ax.set_xlim(cax.get_xlim())
            ax.set_ylim(cax.get_ylim())
        else:
            if c == 0:
                ax.set_ylabel(self.labels[r])
        self.fix_scatter_plot(ax, self.data[:,c], self.labels[c], 'x')
        self.fix_scatter_plot(ax, self.data[:,r], self.labels[r], 'y')
        if not magnifiedax:
            plt.setp(ax.get_xticklabels(), visible=False)
            if c > 0:
                plt.setp(ax.get_yticklabels(), visible=False)
        if keep_lims:
            ax.set_xlim(*ax_xlim)
            ax.set_ylim(*ax_ylim)
        if magnifiedax:
            bbox = ax.get_tightbbox(self.fig.canvas.get_renderer())
            if bbox is not None:
                self.magnified_backdrop = patches.Rectangle((bbox.x0, bbox.y0),
                                                            bbox.width, bbox.height,
                                                            transform=None, clip_on=False,
                                                            facecolor='white', edgecolor='none',
                                                            alpha=0.8, zorder=-5)
                ax.add_patch(self.magnified_backdrop)
        try:
            selector = widgets.RectangleSelector(ax, self._on_select, drawtype='box',
                                                 useblit=True, button=1,
                                                 state_modifier_keys=dict(move='', clear='', square='', center=''))
        except TypeError:
            selector = widgets.RectangleSelector(ax, self._on_select, drawtype='box',
                                                 useblit=True, button=1)
        self.sctter_selector[idx] = selector

        
    def _init_scatter_plots(self):
        self.cbax = self.fig.add_axes([0.5, 0.5, 0.1, 0.5])
        cbax = self.cbax
        n = self.data.shape[1]
        for r in range(1, n):
            yax = None
            for c in range(r):
                ax = self.fig.add_subplot(n, n, (r-1)*n+c+1, sharex=self.hist_ax[c], sharey=yax)
                self.scatter_ax.append(ax)
                self.scatter_indices.append([c, r])
                self.scatter_artists.append(None)
                self.sctter_selector.append(None)
                self._plot_scatter(ax, False, False, cbax)
                yax = ax
                cbax = None

                
    def _plot_magnified_scatter(self):
        ax = self.fig.add_axes([0.5, 0.9, 0.05, 0.05])
        ax.set_visible(False)
        self.magnified_on = False
        c = 0
        r = 1
        ax.scatter(self.data[:,c], self.data[:,r], c=self.data_colors,
                   s=50, edgecolors='none')
        a = ax.scatter(self.data[self.mark_data,c], self.data[self.mark_data,r],
                       c=self.data_colors[self.mark_data], s=80)
        ax.set_xlabel(self.labels[c])
        ax.set_ylabel(self.labels[r])
        self.fix_scatter_plot(ax, self.data[:,c], self.labels[c], 'x')
        self.fix_scatter_plot(ax, self.data[:,r], self.labels[r], 'y')
        self.scatter_ax.append(ax)
        self.scatter_indices.append([c, r])
        self.scatter_artists.append(a)
        self.sctter_selector.append(None)

        
    def fix_scatter_plot(self, ax, data, label, axis):
        """Customize an axes of a scatter plot.

        This function is called after a scatter plot has been plotted.
        Once for the x axes, once for the y axis and once for the color bar.
        Reimplement this function to set appropriate limits and ticks.

        Return values are only used for the color bar (`axis='c').
        Otherwise they are ignored.
        
        Parameter
        ---------
        ax: matplotlib axes
            Axes of the scatter plot or color bar to be worked on.
        data: 1D array
            Data array of the axes.
        label: string
            Label coresponding to the data array.
        axis: str
            'x', 'y': set properties of x or y axes of ax.
            'c': set properies of color bar axes and return vmin, vmax, and ticks.

        Returns
        -------
            min: float
                minimum value of color bar axis
            max: float
                maximum value of color bar axis
            ticks: list of float
                position of ticks for color bar axis
        """
        return np.min(data), np.max(data), None

    
    def fix_waveform_plot(self, axs, indices):
        """ Customize the waveform plots.

        This function is called once after new data have been plotted
        into the waveform plots.  Reimplement this function to customize
        these plots. In particular to set axis limits and labels, plot
        title, etc.

        For example, you can set the linewidth of all plotted waveforms via:
        ```
        for ax in axs:
            for l in ax.lines:
                l.set_linewidth(3.0)
        ```

        Parameters
        ----------
        axs: list of matplotlib axes
            Axis of the waveform plots to be worked on.
        indices: list of int
            Indices of the waveforms that have been selected and plotted.
        """
        pass

    
    def list_selection(self, indices):
        """ List information about the current selection of data points.

        This function is called when 'l' is pressed.
        Reimplement this function to print some meaningfull information
        about the current selection of data points on to the console.

        Parameter
        ---------
        indices: list of int
            Indices of the data points that have been selected.
        """
        for i in indices:
            print(i)

            
    def analyze_selection(self, index):
        """ Provide further information of single selected data point.

        This function is called when double clicking on a single data item.
        Reimplement this function to prived some further details on this data point.
        This can be an additional figure window. In this case show it non-blocking:
        ```
        plt.show(block=False)
        ```

        Parameter
        ---------
        index: int
            The index of the selected data point.
        """
        pass

    
    def _set_magnified_pos(self, width, height):
        if self.magnified_on:
            xoffs = self.xborder/width
            yoffs = self.yborder/height
            if self.scatter_indices[-1][1] < self.data.shape[1]:
                idx = self.scatter_indices[:-1].index(self.scatter_indices[-1])
                pos = self.scatter_ax[idx].get_position().get_points()
            else:
                pos = self.hist_ax[self.scatter_indices[-1][0]].get_position().get_points()
            pos[0] = np.mean(pos, 0) - 0.5*self.magnified_size
            if pos[0][0] < xoffs: pos[0][0] = xoffs
            if pos[0][1] < yoffs: pos[0][1] = yoffs
            pos[1] = pos[0] + self.magnified_size
            if pos[1][0] > 1.0-self.spacing/width: pos[1][0] = 1.0-self.spacing/width
            if pos[1][1] > 1.0-self.spacing/height: pos[1][1] = 1.0-self.spacing/height
            pos[0] = pos[1] - self.magnified_size
            self.scatter_ax[-1].set_position([pos[0][0], pos[0][1],
                                             self.magnified_size[0], self.magnified_size[1]])

            
    def _make_selection(self, ax, key, x0, x1, y0, y1):
        if not key in ['shift', 'control']:
            self.mark_data = []
        try:
            axi = self.scatter_ax.index(ax)
            # from scatter plots:
            c, r = self.scatter_indices[axi]
            if r < self.data.shape[1]:
                # from scatter:
                for ind, (x, y) in enumerate(zip(self.data[:,c], self.data[:,r])):
                    if x >= x0 and x <= x1 and y >= y0 and y <= y1:
                        if ind in self.mark_data:
                            if key == 'control':
                                self.mark_data.remove(ind)
                        else:
                            self.mark_data.append(ind)
            else:
                # from histogram:
                for ind, x in enumerate(self.data[:,c]):
                    if x >= x0 and x <= x1:
                        if ind in self.mark_data:
                            if key == 'control':
                                self.mark_data.remove(ind)
                        else:
                            self.mark_data.append(ind)
        except ValueError:
            try:
                r = self.hist_ax.index(ax)
                # from histogram:
                for ind, x in enumerate(self.data[:,r]):
                    if x >= x0 and x <= x1:
                        if ind in self.mark_data:
                            if key == 'control':
                                self.mark_data.remove(ind)
                        else:
                            self.mark_data.append(ind)
            except ValueError:
                return

                        
    def _update_selection(self):
        # update scatter plots:
        for artist, (c, r) in zip(self.scatter_artists, self.scatter_indices):
            if artist is not None:
                artist.set_offsets(list(zip(self.data[self.mark_data,c],
                                            self.data[self.mark_data,r])))
                artist.set_facecolors(self.data_colors[self.mark_data])
        # waveform plots:
        for k, ax in enumerate(self.wave_ax):
            ax.clear()
            for idx in self.mark_data:
                if idx < len(self.wave_data[k]):
                    ax.plot(self.wave_data[k][idx][:,0],
                            self.wave_data[k][idx][:,1],
                            c=self.data_colors[idx],
                            picker=self.pick_radius)
        if not isinstance(self.wave_xlabels, (list, tuple)):
            for ax in self.wave_ax[:-1]:
                plt.setp(ax.get_xticklabels(), visible=False)
            self.wave_ax[-1].set_xlabel(self.wave_xlabels)
        else:
            for ax, xlabel in zip(self.wave_ax, self.wave_xlabels):
                ax.set_xlabel(xlabel)
        for ax, ylabel in zip(self.wave_ax, self.wave_ylabels):
            ax.set_ylabel(ylabel)
        if not isinstance(self.wave_title, bool) and self.wave_title:
            self.wave_ax[0].set_title(self.wave_title)
        if len(self.wave_ax) > 0:
            self.fix_waveform_plot(self.wave_ax, self.mark_data)
        self.fig.canvas.draw()

        
    def _on_key(self, event):
        #print('pressed', event.key)
        plot_zoom = True
        if event.key in ['left', 'right', 'up', 'down']:
            if self.magnified_on:
                if event.key == 'left':
                    if self.scatter_indices[-1][0] > 0:
                        self.scatter_indices[-1][0] -= 1
                    else:
                        plot_zoom = False
                elif event.key == 'right':
                    if self.scatter_indices[-1][0] < self.scatter_indices[-1][1]-1 and \
                       self.scatter_indices[-1][0] < self.maxcols-1:
                        self.scatter_indices[-1][0] += 1
                    else:
                        plot_zoom = False
                elif event.key == 'up':
                    if self.scatter_indices[-1][1] > 1:
                        if self.scatter_indices[-1][1] >= self.data.shape[1]:
                            self.scatter_indices[-1][1] = self.maxcols-1
                        else:
                            self.scatter_indices[-1][1] -= 1
                        if self.scatter_indices[-1][0] >= self.scatter_indices[-1][1]:
                            self.scatter_indices[-1][0] = self.scatter_indices[-1][1]-1
                    else:
                        plot_zoom = False
                elif event.key == 'down':
                    if self.scatter_indices[-1][1] < self.maxcols:
                        self.scatter_indices[-1][1] += 1
                        if self.scatter_indices[-1][1] >= self.maxcols:
                            self.scatter_indices[-1][1] = self.data.shape[1]
                    else:
                        plot_zoom = False
        else:
            plot_zoom = False
            if event.key == 'escape':
                self.scatter_ax[-1].set_position([0.5, 0.9, 0.05, 0.05])
                self.magnified_on = False
                self.scatter_ax[-1].set_visible(False)
                self.fig.canvas.draw()
            elif event.key in 'oz':
                self.select_zooms = not self.select_zooms
            elif event.key == 'backspace':
                if len(self.zoom_stack) > 0:
                    ax, xmin, xmax, ymin, ymax = self.zoom_stack.pop()
                    ax.set_xlim(xmin, xmax)
                    ax.set_ylim(ymin, ymax)
                    if ax in self.scatter_ax[:-1]:
                        axidx = self.scatter_ax[:-1].index(ax)
                        if self.scatter_indices[axidx][0] == self.scatter_indices[-1][0]:
                            self.scatter_ax[-1].set_xlim(xmin, xmax)
                        if self.scatter_indices[axidx][1] == self.scatter_indices[-1][1]:
                            self.scatter_ax[-1].set_ylim(ymin, ymax)
                    elif ax in self.hist_ax:
                        if self.scatter_indices[-1][1] == self.data.shape[1] and \
                           self.scatter_indices[-1][0] == self.hist_ax.index(ax):
                            self.scatter_ax[-1].set_xlim(xmin, xmax)
                            self.scatter_ax[-1].set_ylim(ymin, ymax)
                    self.fig.canvas.draw()
            elif event.key in '+=':
                self.pick_radius *= 1.5
            elif event.key in '-':
                if self.pick_radius > 5.0:
                    self.pick_radius /= 1.5
            elif event.key in '0':
                self.pick_radius = 4.0
            elif event.key in ['pageup', 'pagedown', '<', '>']:
                if event.key in ['pageup', '<'] and self.maxcols > 2:
                    self.maxcols -= 1
                elif event.key in ['pagedown', '>'] and self.maxcols < self.raw_data.shape[1]:
                    self.maxcols += 1
                self._update_layout()
            elif event.key in 'cC':
                if event.key in 'c':
                    self.color_index -= 1
                    if self.color_index < 0:
                        self.color_set_index -= 1
                        if self.color_set_index < -1:
                            self.color_set_index = len(self.all_data)-1
                        if self.color_set_index >= 0:
                            if self.all_data[self.color_set_index] is None:
                                self.compute_pca(self.color_set_index>1, True)
                            self.color_index = self.all_data[self.color_set_index].shape[1]-1
                        else:
                            self.color_index = 0 if self.extra_colors is None else 1
                else:
                    self.color_index += 1
                    if (self.color_set_index >= 0 and \
                        self.color_index >= self.all_data[self.color_set_index].shape[1]) or \
                        (self.color_set_index < 0 and \
                         self.color_index >= (1 if self.extra_colors is None else 2)):
                        self.color_index = 0
                        self.color_set_index += 1
                        if self.color_set_index >= len(self.all_data):
                            self.color_set_index = -1
                        elif self.all_data[self.color_set_index] is None:
                            self.compute_pca(self.color_set_index>1, True)
                self._set_color_column()
                for ax in self.scatter_ax:
                    if len(ax.collections) > 0:
                        ax.collections[0].set_facecolors(self.data_colors)
                for a in self.scatter_artists:
                    if a is not None:
                        a.set_facecolors(self.data_colors[self.mark_data])
                for ax in self.wave_ax:
                    for l, c in zip(ax.lines, self.data_colors[self.mark_data]):
                        l.set_color(c)
                self._plot_scatter(self.scatter_ax[0], False, True, self.cbax)
                self.fix_scatter_plot(self.cbax, self.color_values,
                                      self.color_label, 'c')
                self.fig.canvas.draw()
            elif event.key in 'nN':
                if event.key in 'N':
                    self.hist_nbins = (self.hist_nbins*3)//2
                elif self.hist_nbins >= 15:
                    self.hist_nbins = (self.hist_nbins*2)//3
                for ax in self.hist_ax:
                    self._plot_hist(ax, False, True)
                if self.scatter_indices[-1][1] >= self.data.shape[1]:
                    self._plot_hist(self.scatter_ax[-1], True, True)
                elif not self.scatter:
                    self._plot_scatter(self.scatter_ax[-1], True, True)
                if not self.scatter:
                    for ax in self.scatter_ax[:-1]:
                        self._plot_scatter(ax, False, True)
                self.fig.canvas.draw()
            elif event.key in 'h':
                self.scatter = not self.scatter
                for ax in self.scatter_ax[:-1]:
                    self._plot_scatter(ax, False, True)
                if self.scatter_indices[-1][1] < self.data.shape[1]:
                    self._plot_scatter(self.scatter_ax[-1], True, True)
                self.fig.canvas.draw()
            elif event.key in 'pP':
                self.all_maxcols[self.show_mode] = self.maxcols
                if event.key == 'p':
                    self.show_mode += 1
                    if self.show_mode >= len(self.all_data):
                        self.show_mode = 0
                else:
                    self.show_mode -= 1
                    if self.show_mode < 0:
                        self.show_mode = len(self.all_data)-1
                if self.show_mode == 1:
                    print('PCA components')
                elif self.show_mode == 2:
                    print('scaled PCA components')
                else:
                    print('data')
                if self.all_data[self.show_mode] is None:
                    self.compute_pca(self.show_mode>1, True)
                self.data = self.all_data[self.show_mode]
                self.labels = self.all_labels[self.show_mode]
                self.maxcols = self.all_maxcols[self.show_mode]
                self.zoom_stack = []
                self.fig.canvas.set_window_title(self.title + ': ' + self.all_titles[self.show_mode])
                for ax in self.hist_ax:
                    self._plot_hist(ax, False, False)
                for ax in self.scatter_ax[:-1]:
                    self._plot_scatter(ax, False, False)
                self._update_layout()
            elif event.key in 'l':
                if len(self.mark_data) > 0:
                    print('')
                    print('selected:')
                    self.list_selection(self.mark_data)
        if plot_zoom:
            for k in reversed(range(len(self.zoom_stack))):
                if self.zoom_stack[k][0] == self.scatter_ax[-1]:
                    del self.zoom_stack[k]
            self.scatter_ax[-1].clear()
            self.scatter_ax[-1].set_visible(True)
            self.magnified_on = True
            self._set_magnified_pos(self.fig.get_window_extent().width,
                                    self.fig.get_window_extent().height)
            if self.scatter_indices[-1][1] < self.data.shape[1]:
                self._plot_scatter(self.scatter_ax[-1], True, False)
            else:
                self._plot_hist(self.scatter_ax[-1], True, False)
            self.fig.canvas.draw()

            
    def _on_select(self, eclick, erelease):
        if eclick.dblclick:
            if len(self.mark_data) > 0:
                self.analyze_selection(self.mark_data[-1])
            return
        x0 = min(eclick.xdata, erelease.xdata)
        x1 = max(eclick.xdata, erelease.xdata)
        y0 = min(eclick.ydata, erelease.ydata)
        y1 = max(eclick.ydata, erelease.ydata)
        ax = erelease.inaxes
        if ax is None:
            ax = eclick.inaxes
        xmin, xmax = ax.get_xlim()
        ymin, ymax = ax.get_ylim()
        dx = 0.02*(xmax-xmin)
        dy = 0.02*(ymax-ymin)
        if x1 - x0 < dx and y1 - y0 < dy:
            bbox = ax.get_window_extent().transformed(self.fig.dpi_scale_trans.inverted())
            width, height = bbox.width, bbox.height
            width *= self.fig.dpi
            height *= self.fig.dpi
            dx = self.pick_radius*(xmax-xmin)/width
            dy = self.pick_radius*(ymax-ymin)/height
            x0 = erelease.xdata - dx
            x1 = erelease.xdata + dx
            y0 = erelease.ydata - dy
            y1 = erelease.ydata + dy
        elif self.select_zooms:
            self.zoom_stack.append((ax, xmin, xmax, ymin, ymax))
            ax.set_xlim(x0, x1)
            ax.set_ylim(y0, y1)
        self._make_selection(ax, erelease.key, x0, x1, y0, y1)
        self._update_selection()

        
    def _on_pick(self, event):
        for ax in self.wave_ax:
            for k, l in enumerate(ax.lines):
                if l is event.artist:
                    self.mark_data = [self.mark_data[k]]
        self._update_selection()
        if event.mouseevent.dblclick:
            if len(self.mark_data) > 0:
                self.analyze_selection(self.mark_data[-1])

                    
    def _set_layout(self, width, height):
        xoffs = self.xborder/width
        yoffs = self.yborder/height
        dx = (1.0-xoffs)/self.maxcols
        dy = (1.0-yoffs)/self.maxcols
        xs = self.spacing/width
        ys = self.spacing/height
        xw = dx - xs
        yw = dy - ys
        for c, ax in enumerate(self.hist_ax):
            if c < self.maxcols:
                ax.set_position([xoffs+c*dx, yoffs, xw, yw])
                ax.set_visible(True)
            else:
                ax.set_visible(False)
                ax.set_position([0.99, 0.01, 0.01, 0.01])
        for ax, (c, r) in zip(self.scatter_ax[:-1], self.scatter_indices[:-1]):
            if r < self.maxcols:
                ax.set_position([xoffs+c*dx, yoffs+(self.maxcols-r)*dy, xw, yw])
                ax.set_visible(True)
            else:
                ax.set_visible(False)
                ax.set_position([0.99, 0.01, 0.01, 0.01])
        self.cbax.set_position([xoffs+dx, yoffs+(self.maxcols-1)*dy, 0.3*xoffs, yw])
        self._set_magnified_pos(width, height)
        if self.magnified_backdrop is not None:  # XXX Why is it sometimes None????
            bbox = self.scatter_ax[-1].get_tightbbox(self.fig.canvas.get_renderer())
            if bbox is not None:
                self.magnified_backdrop.set_bounds(bbox.x0, bbox.y0, bbox.width, bbox.height)
        x0 = xoffs+((self.maxcols+1)//2)*dx
        y0 = yoffs+((self.maxcols+1)//2)*dy
        if self.maxcols%2 == 0:
            x0 += xoffs
            y0 += yoffs
        if len(self.wave_ax) > 0:
            dy = 1.0-y0
            if self.wave_title and not isinstance(self.wave_xlabels, (list, tuple)):
                dy -= 2*ys
            dy /= len(self.wave_ax)
            if isinstance(self.wave_xlabels, (list, tuple)):
                ys += 2.5*ys 
            for k, ax in enumerate(self.wave_ax):
                ax.set_position([x0, y0+(len(self.wave_ax)-1-k)*dy, 1.0-x0-xs, dy-ys])

            
    def _update_layout(self):
        if self.scatter_indices[-1][1] < self.data.shape[1]:
            if self.scatter_indices[-1][1] >= self.maxcols:
                self.scatter_indices[-1][1] = self.maxcols-1
            if self.scatter_indices[-1][0] >= self.scatter_indices[-1][1]:
                self.scatter_indices[-1][0] = self.scatter_indices[-1][1]-1
            self._plot_scatter(self.scatter_ax[-1], True, False)
        else:
            if self.scatter_indices[-1][0] >= self.maxcols:
                self.scatter_indices[-1][0] = self.maxcols-1
                self._plot_hist(self.scatter_ax[-1], True, False)
        self._set_layout(self.fig.get_window_extent().width,
                        self.fig.get_window_extent().height)
        self.fig.canvas.draw()

        
    def _on_resize(self, event):
        self._set_layout(event.width, event.height)


def main():
    # parse command line:
    parser = argparse.ArgumentParser(add_help=True,
        description='View and explore multivariate data.',
        epilog='version %s by Benda-Lab (2019-%s)' % (__version__, __year__))
    parser.add_argument('file', nargs='?', default='', type=str,
                        help='a file containing a table of data (csv file or similar)')
    args = parser.parse_args()
    if args.file:
        # load data:
        data = TableData(args.file)
        # initialize explorer:
        expl = MultivariateExplorer(data)
    else:
        # generate data:
        n = 100
        data = []
        data.append(np.random.randn(n) + 2.0)
        data.append(1.0+0.1*data[0] + 1.5*np.random.randn(n))
        data.append(-3.0*data[0] - 2.0*data[1] + 1.8*np.random.randn(n))
        idx = np.random.randint(0, 3, n)
        names = ['aaa', 'bbb', 'ccc']
        data.append([names[i] for i in idx])
        # generate waveforms:
        waveforms1 = []
        waveforms2 = []
        time = np.arange(0.0, 10.0, 0.01)
        for r in range(len(data[0])):
            x = data[0][r]*np.sin(2.0*np.pi*data[1][r]*time + data[2][r])
            waveforms1.append(np.column_stack((time, x)))
            y = data[0][r]*np.exp(-0.5*((time-data[1][r])/(0.2*data[2][r]))**2.0)
            waveforms2.append(np.column_stack((time, y)))
        # initialize explorer:
        expl = MultivariateExplorer(data,
                                    map(chr, np.arange(len(data))+ord('A')),
                                    'Explorer')
        expl.set_wave_data([waveforms1, waveforms2], 'Time', ['Sine', 'Gauss'])
    # explore data:
    expl.set_colors()
    expl.show()


if __name__ == '__main__':
    main()