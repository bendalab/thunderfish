"""
# Explore EODs of many fish.
"""

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.widgets as widgets
from multiprocessing import Pool, freeze_support, cpu_count
from .version import __version__, __year__
from .tabledata import TableData


def color_range(data, color_col=None):
    if color_col is None or color_col < 0:
        colors = np.arange(len(data))/float(len(data))
    else:
        cmin = np.min(data[:,color_col])
        cmax = np.max(data[:,color_col])
        colors = (data[:,color_col] - cmin)/(cmax - cmin)
    return colors


class Explorer(object):
    
    def __init__(self, data, labels, colors, color_map, detailed_data):
        if isinstance(data, TableData):
            self.data = data.array()
            if labels is None:
                self.labels = []
                for c in range(len(data)):
                    self.labels.append('%s [%s]' % (data.label(c), data.unit(c)))
            else:
                self.labels = labels
        else:
            self.data = data
            self.labels = labels
        self.press = False
        self.move = False
        self.event_axes = None
        self.event_xdata = None
        self.event_ydata = None
        plt.rcParams['keymap.quit'] = 'ctrl+w, alt+q, q'
        plt.rcParams['keymap.back'] = 'backspace'
        plt.rcParams['keymap.forward'] = 'v'        
        self.fig = plt.figure(facecolor='white')
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)
        self.fig.canvas.mpl_connect('button_press_event', self.on_press)
        self.fig.canvas.mpl_connect('button_release_event', self.on_release)
        self.fig.canvas.mpl_connect('motion_notify_event', self.on_motion)
        self.fig.canvas.mpl_connect('resize_event', self.on_resize)
        self.xborder = 60.0  # pixel for ylabels
        self.yborder = 50.0  # pixel for xlabels
        self.spacing = 10.0  # pixel between plots
        self.pick_radius = 0.05
        self.histax = []
        self.color_map = plt.get_cmap(color_map)
        self.data_colors = self.color_map(colors)
        self.default_colors = colors
        self.color_index = -1
        self.detailed_data = detailed_data
        self.histax = []
        self.plot_histograms()
        self.corrax = []
        self.corrindices = []
        self.corrartists = []
        self.corrselect = []
        self.mark_data = []
        self.plot_correlations()
        self.zoomon = False
        self.zoom_size = np.array([0.4, 0.4])
        self.zoom_back = None
        self.plot_zoomed_correlations()
        self.dax = self.fig.add_subplot(2, 3, 3)
        self.dax.plot(self.detailed_data[0][:,0], self.detailed_data[0][:,1],
                      c=self.data_colors[0], lw=3)
        self.fix_detailed_plot(self.dax, [0])
        plt.show()
        
    def plot_histograms(self):
        n = self.data.shape[1]
        self.histax = []
        for r in range(n):
            ax = self.fig.add_subplot(n, n, (n-1)*n+r+1)
            ax.hist(self.data[:,r], 30)
            ax.set_xlabel(self.labels[r])
            if r == 0:
                ax.set_ylabel('count')
            else:
                ax.set_yticklabels([])
            self.fix_scatter_plot(ax, self.data[:,r], self.labels[r], 'x')
            self.histax.append(ax)

    def plot_correlations(self):
        self.mark_data = [0]
        n = self.data.shape[1]
        for r in range(1, n):
            yax = None
            for c in range(r):
                ax = self.fig.add_subplot(n, n, (r-1)*n+c+1, sharex=self.histax[c], sharey=yax)
                ax.scatter(self.data[:,c], self.data[:,r], c=self.data_colors,
                           s=50, edgecolors='none')
                a = ax.scatter(self.data[self.mark_data[0],c], self.data[self.mark_data[0],r],
                               c=self.data_colors[0], s=80)
                plt.setp(ax.get_xticklabels(), visible=False)
                if c == 0:
                    ax.set_ylabel(self.labels[r])
                else:
                    plt.setp(ax.get_yticklabels(), visible=False)
                self.fix_scatter_plot(ax, self.data[:,c], self.labels[c], 'x')
                self.fix_scatter_plot(ax, self.data[:,r], self.labels[r], 'y')
                try:
                    selector = widgets.RectangleSelector(ax, self.on_select, drawtype='box',
                                                         minspanx=5, minspany=5,
                                                         spancoords='pixels',
                                                         button=1, state_modifier_keys=dict(move='', clear='', square='', center=''))
                except TypeError:
                    selector = widgets.RectangleSelector(ax, self.on_select, drawtype='box',
                                                         minspanx=5, minspany=5,
                                                         spancoords='pixels',
                                                         button=1)
                yax = ax
                self.corrax.append(ax)
                self.corrindices.append([c, r])
                self.corrartists.append(a)
                self.corrselect.append(selector)

    def plot_zoomed_correlations(self):
        ax = self.fig.add_axes([0.5, 0.9, 0.05, 0.05])
        ax.set_visible(False)
        self.zoomon = False
        c = 0
        r = 1
        ax.scatter(self.data[:,c], self.data[:,r], c=self.data_colors,
                   s=50, edgecolors='none')
        a = ax.scatter(self.data[0,c], self.data[0,r], c=self.data_colors[0], s=80)
        ax.set_xlabel(self.labels[c])
        ax.set_ylabel(self.labels[r])
        self.fix_scatter_plot(ax, self.data[:,c], self.labels[c], 'x')
        self.fix_scatter_plot(ax, self.data[:,r], self.labels[r], 'y')
        self.corrax.append(ax)
        self.corrindices.append([c, r])
        self.corrartists.append(a)
                
    def fix_scatter_plot(self, ax, data, label, axis):
        pass

    def fix_detailed_plot(self, ax, indices):
        pass

    def set_zoom_pos(self, width, height):
        if self.zoomon:
            xoffs = self.xborder/width
            yoffs = self.yborder/height
            if self.corrindices[-1][1] < self.data.shape[1]:
                idx = self.corrindices[:-1].index(self.corrindices[-1])
                pos = self.corrax[idx].get_position().get_points()
            else:
                pos = self.histax[self.corrindices[-1][0]].get_position().get_points()
            pos[0] = np.mean(pos, 0) - 0.5*self.zoom_size
            if pos[0][0] < xoffs: pos[0][0] = xoffs
            if pos[0][1] < yoffs: pos[0][1] = yoffs
            pos[1] = pos[0] + self.zoom_size
            if pos[1][0] > 1.0-self.spacing/width: pos[1][0] = 1.0-self.spacing/width
            if pos[1][1] > 1.0-self.spacing/height: pos[1][1] = 1.0-self.spacing/height
            pos[0] = pos[1] - self.zoom_size
            self.corrax[-1].set_position([pos[0][0], pos[0][1],
                                          self.zoom_size[0], self.zoom_size[1]])

    def update_selection(self, ax, key, xmin, xmax, ymin, ymax):
        if not key in ['shift', 'control']:
            self.mark_data = []
        try:
            axi = self.corrax.index(ax)
            # from scatter plots:
            c, r = self.corrindices[axi]
            for ind, (x, y) in enumerate(zip(self.data[:,c], self.data[:,r])):
                if x >= xmin and x <= xmax and y >= ymin and y <= ymax:
                    self.mark_data.append(ind)
        except ValueError:
            try:
                r = self.histax.index(ax)
                # from histogram:
                for ind, x in enumerate(self.data[:,r]):
                    if x >= xmin and x <= xmax:
                        self.mark_data.append(ind)
            except ValueError:
                return
        # update scatter plots:
        for artist, (c, r) in zip(self.corrartists, self.corrindices):
            artist.set_offsets(list(zip(self.data[self.mark_data,c],
                                        self.data[self.mark_data,r])))
            artist.set_facecolors(self.data_colors[self.mark_data])
        # detailed plot:
        self.dax.clear()
        for idx in self.mark_data:
            if idx < len(self.detailed_data):
                self.dax.plot(self.detailed_data[idx][:,0], self.detailed_data[idx][:,1],
                            c=self.data_colors[idx], lw=3)
        if len(self.mark_data) == 0:
            self.dax.text(0.5, 0.5, 'Click to plot details', transform = self.dax.transAxes,
                          ha='center', va='center')
        self.fix_detailed_plot(self.dax, self.mark_data)
        self.fig.canvas.draw()
        
    def on_key(self, event):
        #print('pressed', event.key)
        plot_zoom = True
        if event.key in ['left', 'right', 'up', 'down']:
            if self.zoomon:
                if event.key == 'left':
                    if self.corrindices[-1][0] > 0:
                        self.corrindices[-1][0] -= 1
                    else:
                        plot_zoom = False
                elif event.key == 'right':
                    if self.corrindices[-1][0] < self.corrindices[-1][1]-1:
                        self.corrindices[-1][0] += 1
                    else:
                        plot_zoom = False
                elif event.key == 'up':
                    if self.corrindices[-1][1] > 1:
                        self.corrindices[-1][1] -= 1
                        if self.corrindices[-1][0] >= self.corrindices[-1][1]:
                            self.corrindices[-1][0] = self.corrindices[-1][1]-1
                    else:
                        plot_zoom = False
                elif event.key == 'down':
                    if self.corrindices[-1][1] < self.data.shape[1]:
                        self.corrindices[-1][1] += 1
                    else:
                        plot_zoom = False
        else:
            plot_zoom = False
            if event.key == 'escape':
                self.corrax[-1].set_position([0.5, 0.9, 0.05, 0.05])
                self.zoomon = False
                self.corrax[-1].set_visible(False)
                self.fig.canvas.draw()
            elif event.key in '+=':
                self.pick_radius *= 1.5
            elif event.key in '-':
                if self.pick_radius > 0.001:
                    self.pick_radius /= 1.5
            elif event.key in '0':
                self.pick_radius = 0.05
            elif event.key in 'cC':
                if event.key in 'c':
                    self.color_index -= 1
                    if self.color_index < -1:
                        self.color_index = self.data.shape[1] - 1
                else:
                    self.color_index += 1
                    if self.color_index >= self.data.shape[1]:
                        self.color_index = -1
                if self.color_index == -1:
                    self.data_colors = self.color_map(self.default_colors)
                else:
                    colors = color_range(self.data, self.color_index)
                    self.data_colors = self.color_map(colors)
                for ax in self.corrax:
                    ax.collections[0].set_facecolors(self.data_colors)
                for a in self.corrartists:
                    a.set_facecolors(self.data_colors[self.mark_data])
                for l, c in zip(self.dax.lines, self.data_colors[self.mark_data]):
                    l.set_color(c)
                self.fig.canvas.draw()
        if plot_zoom:
            self.corrax[-1].clear()
            self.corrax[-1].set_visible(True)
            self.zoomon = True
            self.set_zoom_pos(self.fig.get_window_extent().width,
                              self.fig.get_window_extent().height)
            if self.corrindices[-1][1] < self.data.shape[1]:
                self.corrax[-1].scatter(self.data[:,self.corrindices[-1][0]],
                                        self.data[:,self.corrindices[-1][1]],
                                        c=self.data_colors, s=50, edgecolors='none', zorder=10)
                self.corrartists[-1] = self.corrax[-1].scatter(
                                        self.data[self.mark_data,self.corrindices[-1][0]],
                                        self.data[self.mark_data,self.corrindices[-1][1]],
                                        c=self.data_colors[self.mark_data], s=80, zorder=11)
                self.corrax[-1].set_xlabel(self.labels[self.corrindices[-1][0]])
                self.corrax[-1].set_ylabel(self.labels[self.corrindices[-1][1]])
                self.fix_scatter_plot(self.corrax[-1], self.data[:,self.corrindices[-1][0]],
                                      self.labels[self.corrindices[-1][0]], 'x')
                self.fix_scatter_plot(self.corrax[-1], self.data[:,self.corrindices[-1][1]],
                                      self.labels[self.corrindices[-1][1]], 'y')
            else:
                self.corrax[-1].hist(self.data[:,self.corrindices[-1][0]], 30)
                self.corrax[-1].set_xlabel(self.labels[self.corrindices[-1][0]])
                self.corrax[-1].set_ylabel('count')
                self.fix_scatter_plot(self.corrax[-1], self.data[:,self.corrindices[-1][0]],
                                      self.labels[self.corrindices[-1][0]], 'x')
            bbox = self.corrax[-1].get_tightbbox(self.fig.canvas.get_renderer())
            self.zoom_back = patches.Rectangle((bbox.x0, bbox.y0), bbox.width, bbox.height,
                                               transform=None, clip_on=False,
                                               facecolor='white', edgecolor='none',
                                               alpha=0.8, zorder=-5)
            self.corrax[-1].add_patch(self.zoom_back)
            self.fig.canvas.draw()

    def on_press(self, event):
        self.press = True
        self.move = False
        self.event_xdata = event.xdata
        self.event_ydata = event.ydata
        self.event_axes = event.inaxes

    def on_motion(self, event):
        if self.press and not self.move:
            self.move = True
            self.press = False
    
    def on_release(self, event):
        do_plot = False
        if event.inaxes:
            self.event_axes = event.inaxes
        if self.event_axes is None:
            return
        if self.press and not self.move:
            xmin, xmax = self.event_axes.get_xlim()
            ymin, ymax = self.event_axes.get_ylim()
            dx = self.pick_radius*(xmax-xmin)
            dy = self.pick_radius*(ymax-ymin)
            xmin = event.xdata - dx
            ymin = event.ydata - dy
            xmax = event.xdata + dx
            ymax = event.ydata + dy
            do_plot = True
        elif not self.press and self.move:
            xmin = min(self.event_xdata, event.xdata)
            xmax = max(self.event_xdata, event.xdata)
            ymin = min(self.event_ydata, event.ydata)
            ymax = max(self.event_ydata, event.ydata)
            #do_plot = True
        if do_plot:
            self.update_selection(self.event_axes, event.key, xmin, xmax, ymin, ymax)
        self.move = False
        self.press = False

    def on_select(self, eclick, erelease):
        xmin = min(eclick.xdata, erelease.xdata)
        xmax = max(eclick.xdata, erelease.xdata)
        ymin = min(eclick.ydata, erelease.ydata)
        ymax = max(eclick.ydata, erelease.ydata)
        ax = erelease.inaxes
        if ax is None:
            ax = eclick.inaxes
        self.update_selection(ax, erelease.key, xmin, xmax, ymin, ymax)

    def on_resize(self, event):
        n = self.data.shape[1]
        xoffs = self.xborder/event.width
        yoffs = self.yborder/event.height
        dx = (1.0-xoffs)/n
        dy = (1.0-yoffs)/n
        xs = self.spacing/event.width
        ys = self.spacing/event.height
        xw = dx - xs
        yw = dy - ys
        for c, ax in enumerate(self.histax):
            ax.set_position([xoffs+c*dx, yoffs, xw, yw])
        for ax, (c, r) in zip(self.corrax[:-1], self.corrindices[:-1]):
            ax.set_position([xoffs+c*dx, yoffs+(n-r)*dy, xw, yw])
        self.set_zoom_pos(event.width, event.height)
        if self.zoom_back is not None:  # XXX Why is it sometimes None????
            bbox = self.corrax[-1].get_tightbbox(self.fig.canvas.get_renderer())
            if bbox is not None:
                self.zoom_back.set_bounds(bbox.x0, bbox.y0, bbox.width, bbox.height)
        x0 = xoffs+(n//2+1)*dx
        y0 = yoffs+(n//2+1)*dy
        self.dax.set_position([x0, y0, 1.0-x0-xs, 1.0-y0-3*ys])

            
class EODExplorer(Explorer):
    
    def __init__(self, data, labels, colors, color_map, wave_fish, eod_data, eod_metadata):
        self.wave_fish = wave_fish
        self.eod_metadata = eod_metadata
        Explorer.__init__(self, data, labels, colors, color_map, eod_data)

    def fix_scatter_plot(self, ax, data, label, axis):
        if any(l in label for l in ['ampl', 'width', 'tau']):
            if np.all(data >= 0.0):
                if axis == 'x':
                    ax.set_xlim(left=0.0)
                else:
                    ax.set_ylim(bottom=0.0)
            else:
                if axis == 'x':
                    ax.set_xlim(right=0.0)
                else:
                    ax.set_ylim(top=0.0)
        elif 'phase' in label:
            if axis == 'x':
                ax.set_xlim(-np.pi, np.pi)
            else:
                ax.set_ylim(-np.pi, np.pi)

    def fix_detailed_plot(self, ax, indices):
        if len(indices) > 0 and indices[-1] < len(self.eod_metadata):
            ax.set_title(self.eod_metadata[indices[-1]]['file'])
            if len(indices) > 1:
                ax.text(-0.6, 0.8, 'n=%d' % len(indices))
            else:
                ax.text(-0.6, 0.8, '%.1fHz' % self.eod_metadata[indices[-1]]['EODf'])
        if self.wave_fish:
            ax.set_xlim(-0.7, 0.7)
            ax.set_xlabel('Time [1/EODf]')
            ax.set_ylim(-1.0, 1.0)
        else:
            ax.set_xlim(-0.5, 1.5)
            ax.set_xlabel('Time [ms]')
            ax.set_ylim(-1.5, 1.0)
        ax.set_ylabel('Amplitude')


wave_fish = True
data = None
data_path = None

def load_waveform(idx):
    eodf = data[idx,'EODf']
    file_name = data[idx,'file']
    file_index = data[idx,'index'] if 'index' in data else 0
    eod_table = TableData(os.path.join(data_path, '%s-eodwaveform-%d.csv' % (file_name, file_index)))
    eod = eod_table[:,'mean']
    if wave_fish:
        norm = max(np.max(eod), np.abs(np.min(eod)))
        return np.vstack((eod_table[:,'time']*0.001*eodf, eod/norm)).T
    else:
        norm = np.max(eod)
        return np.vstack((eod_table[:,'time'], eod/norm)).T

def main():
    global data
    global wave_fish
    global data_path

    # command line arguments:
    parser = argparse.ArgumentParser(prefix_chars='-+', add_help=False,
        description='Explore EOD data generated by thunderfish.',
        epilog='version %s by Benda-Lab (2019-%s)' % (__version__, __year__))
    parser.add_argument('-h', '--help', action='store_true',
                        help='show this help message and exit')
    parser.add_argument('--version', action='version', version=__version__)
    parser.add_argument('-l', dest='list_columns', action='store_true',
                        help='list all available data columns and exit')
    parser.add_argument('-j', dest='jobs', nargs='?', type=int, default=None, const=0,
                        help='number of jobs run in parallel. Without argument use all CPU cores.')
    parser.add_argument('-d', dest='data_cols', action='append', default=[], metavar='COLUMN',
                        help='data columns to be analyzed')
    parser.add_argument('+d', dest='add_data_cols', action='append', default=[], metavar='COLUMN',
                        help='data columns to be appended for analysis')
    parser.add_argument('-c', dest='color_col', default='EODf', type=str, metavar='COLUMN',
                        help='data column to be used for color code or "row"')
    parser.add_argument('-m', dest='color_map', default='jet', type=str, metavar='CMAP',
                        help='name of color map')
    parser.add_argument('-p', dest='data_path', default='.', type=str, metavar='PATH',
                        help='path to the EOD waveform data')
    parser.add_argument('file', default='', type=str,
                        help='a wavefish.* or pulsefish.* summary file as generated by collectfish')
    args = parser.parse_args()
    
    # help:
    if args.help:
        parser.print_help()
        print('')
        print('mouse:')
        print('left click             select data points')
        print('shift + left click     add data points to selection')
        print('')
        print('key shortcuts:')
        print('c, C                   cycle color map trough data columns')
        print('+, -                   increase, decrease pick radius')
        print('0                      reset pick radius')
        print('left, right, up, down  move enlarged scatter plot')
        print('escape                 close enlarged scatter plot')
        parser.exit()
        
    # read in command line arguments:    
    list_columns = args.list_columns
    jobs = args.jobs
    file_name = args.file
    data_cols = args.data_cols
    add_data_cols = args.add_data_cols
    color_col = args.color_col
    color_map = args.color_map
    data_path = args.data_path

    # check color map:
    if not color_map in plt.colormaps():
        parser.error('"%s" is not a valid color map' % color_map)
        
    # load summary data:
    wave_fish = 'wave' in file_name
    data = TableData(file_name)

    if list_columns:
        for c in data.keys():
            print(c)
        parser.exit()

    # add cluster column (experimental):
    if wave_fish:
        # wavefish cluster:
        cluster = np.zeros(data.rows())
        cluster[(data[:,'phase1'] < 0) & (data[:,'EODf'] < 300.0)] = 1
        cluster[(data[:,'phase1'] < 0) & (data[:,'EODf'] > 300.0)] = 2
        cluster[(data[:,'phase1'] < 0) & (data[:,'EODf'] > 300.0) & (data[:,'relampl2'] > 20.0)] = 3
        cluster[data[:,'phase1'] > 0] = 3
        data.append('cluster', '', '%d', cluster)

    # data columns:
    if len(data_cols) == 0:
        if wave_fish:
            data_cols = ['EODf', 'relampl1', 'phase1', 'relampl2', 'phase2']
        else:
            data_cols = ['EODf', 'P1width', 'P2relampl', 'P2time', 'P2width', 'tau', 'peakfreq', 'poweratt5']
    else:
        for c in data_cols:
            if data.index(c) is None:
                parser.error('"%s" is not a valid data column' % c)

    # additional data columns:
    for c in add_data_cols:
        if data.index(c) is None:
            parser.error('"%s" is not a valid data column' % c)
        else:
            data_cols.append(c)

    # color code:
    color_idx = data.index(color_col)
    if color_idx is None and not color_col in ['row', 'cluster']:
        parser.error('"%s" is not a valid column for color code' % color_col)
    colors = color_range(data, color_idx)

    # load metadata:
    eod_metadata = []
    for idx in range(data.rows()):
        eodf = data[idx,'EODf']
        file_name = data[idx,'file']
        file_index = data[idx,'index'] if 'index' in data else 0
        eod_metadata.append({'EODf': eodf, 'file': file_name, 'index': file_index})

    # load waveforms:
    if jobs is not None:
        cpus = cpu_count() if jobs == 0 else jobs
        p = Pool(cpus)
        eod_data = p.map(load_waveform, range(data.rows()))
        del p
    else:
        eod_data = list(map(load_waveform, range(data.rows())))

    data = data[:,data_cols]

    # explore:
    eodexp = EODExplorer(data, None, colors, color_map, wave_fish, eod_data, eod_metadata)


if __name__ == '__main__':
    freeze_support()  # needed by multiprocessing for some weired windows stuff
    main()
