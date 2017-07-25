import sys
import numpy as np
import argparse
import matplotlib.pyplot as plt
from IPython import embed

class Human_tracker():
    def __init__(self, data, time, rises=None):
        self.data = data
        self.time = time
        self.rises = rises

        self.trace_handlers = []
        self.rise_peak_handlers = []
        self.rise_end_handlers = []

        self.active_plot_handle0 = None
        self.active_plot_handle1 = None
        self.active_fish0 = None
        self.active_fish1 = None

        self.active_index0 = None
        self.active_index1 = None
        self.tmp_plot_handle0 = None
        self.tmp_plot_handle1 = None

        self.active_rise0 = None

        self.current_task = None

        plt.rcParams['keymap.fullscreen'] = 'ctrl+f'
        plt.rcParams['keymap.back'] = ''  # was c
        plt.rcParams['keymap.home'] = ''  # was was r


        self.fig = plt.figure()
        self.fig.canvas.mpl_connect('key_press_event', self.keypress)
        self.fig.canvas.mpl_connect('button_press_event', self.buttonpress)

        self.ax = self.fig.add_axes([0.1, 0.1, 0.8, 0.8])
        self.plot_data()

        plt.show()

    def plot_data(self):
        for i in range(len(self.data)):
            color = np.random.rand(3, 1)
            rph = None
            reh = None
            h, =self.ax.plot(self.time[~np.isnan(self.data[i])], self.data[i][~np.isnan(self.data[i])], color=color, marker='.')
            if self.rises != None:
                # clean out to small rises
                self.rises[i] = [rise for rise in self.rises[i] if (rise[1][0] - rise[1][1]) > 1]

                peak_idx = [rise[0][0] for rise in self.rises[i]]
                floor_idx = [rise[0][1] for rise in self.rises[i]]

                rph, = self.ax.plot(self.time[peak_idx], self.data[i][peak_idx], 'o', color=color, markersize=7)
                reh, = self.ax.plot(self.time[floor_idx], self.data[i][floor_idx], 's', color=color, markersize=7)

            self.trace_handlers.append(h)
            self.rise_peak_handlers.append(rph if rph else None)
            self.rise_end_handlers.append(reh if reh else None)

    def keypress(self, event):
        # zoom plot
        if event.key in '-':
            y_lims = self.ax.get_ylim()
            y_range = np.abs(y_lims[1] - y_lims[0])
            self.ax.set_ylim([y_lims[0] - 1./3 * y_range, y_lims[1] + 1./3 * y_range])

            x_lims = self.ax.get_xlim()
            x_range = np.abs(x_lims[1] - x_lims[0])
            self.ax.set_xlim([x_lims[0] - 1. / 3 * x_range, x_lims[1] + 1. / 3 * x_range])

        if event.key in '+':
            y_lims = self.ax.get_ylim()
            y_range = np.abs(y_lims[1] - y_lims[0])
            self.ax.set_ylim([y_lims[0] + 1. / 3 * y_range, y_lims[1] - 1. / 3 * y_range])

            x_lims = self.ax.get_xlim()
            x_range = np.abs(x_lims[1] - x_lims[0])
            self.ax.set_xlim([x_lims[0] + 1. / 3 * x_range, x_lims[1] - 1. / 3 * x_range])

        # move plot
        if event.key == 'up':
            y_lims = self.ax.get_ylim()
            y_range = np.abs(y_lims[1] - y_lims[0])
            self.ax.set_ylim([y_lims[0] + y_range / 3., y_lims[1] + y_range / 3.])

        if event.key == 'down':
            y_lims = self.ax.get_ylim()
            y_range = np.abs(y_lims[1] - y_lims[0])
            self.ax.set_ylim([y_lims[0] - y_range / 3., y_lims[1] - y_range / 3.])

        if event.key == 'right':
            x_lims = self.ax.get_xlim()
            x_range = np.abs(x_lims[1] - x_lims[0])
            self.ax.set_xlim([x_lims[0] + x_range / 3., x_lims[1] + x_range / 3.])

        if event.key == 'left':
            x_lims = self.ax.get_xlim()
            x_range = np.abs(x_lims[1] - x_lims[0])
            self.ax.set_xlim([x_lims[0] - x_range / 3., x_lims[1] - x_range / 3.])

        # set tasks
        if event.key in 'j':
            self.current_task = 'join'
            print ('\ntask: (trace) join')

        if event.key in 'c':
            self.current_task = 'cut'
            print ('\ntask: (trace) cut')

        if event.key in 'r':
            if self.rises == None:
                print('\nno rise file loaded')
            else:
                rise_tasks = ['modify rise peak', 'modify rise end', 'add rise', 'remove rise']
                if self.current_task not in rise_tasks:
                    self.current_task = rise_tasks[0]
                elif self.current_task == rise_tasks[0]:
                    self.current_task = rise_tasks[1]
                elif self.current_task == rise_tasks[1]:
                    self.current_task = rise_tasks[2]
                elif self.current_task == rise_tasks[2]:
                    self.current_task = rise_tasks[3]
                elif self.current_task == rise_tasks[3]:
                    self.current_task = rise_tasks[0]
                else:
                    print('something really strange happend o_O')

                print('\ntask: %s' % self.current_task)

        # terminate plot
        if event.key in 'q':
            quit()

        if event.key in 'e':
            embed()
            quit()

        if event.key == 'enter':
            if self.current_task == 'join':
                if self.active_fish0 == None or self.active_fish1 == None:
                    print('need 2 active fish... ')
                else:
                    self.join_traces()

            if self.current_task == 'cut':
                if self.active_fish0 == None or self.active_index0 == None:
                    print('active trace or cutting index missing')
                else:
                    self.cut_trace()

            if self.current_task == 'modify rise peak':
                if self.active_fish0 == None or self.active_rise0 == None or self.active_index0 == None or self.active_index1 == None:
                    print('need active trace, active rise and two active indices.')
                else:
                    self.modify_rise_peak()

            if self.current_task == 'modify rise end':
                if self.active_fish0 == None or self.active_rise0 == None or self.active_index0 == None or self.active_index1 == None:
                    print('need active trace, active rise and two active indices.')
                else:
                    self.modify_rise_end()

        self.fig.canvas.draw()

    def buttonpress(self, event):
        if event.inaxes == self.ax:
            if event.button == 2:
                self.all_reset()

            if self.current_task == 'join':
                if event.button == 1:
                    x = event.xdata
                    y = event.ydata

                    min_distance = []  # gewichtung von time und frequency
                    for f in self.data:
                        df = np.min(np.abs(f[~np.isnan(f)] - y))
                        dt = np.min(np.abs(self.time[~np.isnan(f)] - x))
                        min_distance.append(np.sqrt(df**2 + dt**2))

                    if np.argmin(min_distance) != self.active_fish1:
                        if not self.active_plot_handle0 == None:
                            self.active_plot_handle0.set_linewidth(self.active_plot_handle0.get_linewidth() - 3.)

                        self.active_fish0 = np.argmin(min_distance)
                        self.active_plot_handle0 = self.trace_handlers[self.active_fish0]
                        self.active_plot_handle0.set_linewidth(self.active_plot_handle0.get_linewidth() + 3.)

                if event.button == 3:
                    x = event.xdata
                    y = event.ydata

                    min_distance = []  # gewichtung von time und frequency
                    for f in self.data:
                        df = np.min(np.abs(f[~np.isnan(f)] - y))
                        dt = np.min(np.abs(self.time[~np.isnan(f)] - x))
                        min_distance.append(np.sqrt(df**2 + dt**2))

                    if np.argmin(min_distance) != self.active_fish0:
                        if not self.active_plot_handle1 == None:
                            self.active_plot_handle1.set_linewidth(self.active_plot_handle1.get_linewidth() - 3.)

                        self.active_fish1 = np.argmin(min_distance)
                        self.active_plot_handle1 = self.trace_handlers[self.active_fish1]
                        self.active_plot_handle1.set_linewidth(self.active_plot_handle1.get_linewidth() + 3.)

            if self.current_task == 'cut':
                if event.button == 1:
                    x = event.xdata
                    y = event.ydata

                    min_distance = []  # gewichtung von time und frequency
                    for f in self.data:
                        df = np.min(np.abs(f[~np.isnan(f)] - y))
                        dt = np.min(np.abs(self.time[~np.isnan(f)] - x))
                        min_distance.append(np.sqrt(df**2 + dt**2))

                    if np.argmin(min_distance) != self.active_fish1:
                        if not self.active_plot_handle0 == None:
                            self.active_plot_handle0.set_linewidth(self.active_plot_handle0.get_linewidth() - 3.)

                        self.active_fish0 = np.argmin(min_distance)
                        self.active_plot_handle0 = self.trace_handlers[self.active_fish0]
                        self.active_plot_handle0.set_linewidth(self.active_plot_handle0.get_linewidth() + 3.)

                if event.button == 3:
                    if self.active_fish0 == None:
                        print('need active trace')
                    else:
                        x = event.xdata
                        next_t = self.time[~np.isnan(self.data[self.active_fish0])][self.time[~np.isnan(self.data[self.active_fish0])] > x][0]

                        self.active_index0 = np.where(self.time == next_t)[0][0]

                        if not self.tmp_plot_handle0 == None:
                            self.tmp_plot_handle0.remove()

                        self.tmp_plot_handle0, = self.ax.plot(self.time[self.active_index0],
                                                              self.data[self.active_fish0][self.active_index0], 'o',
                                                              color = 'red', markersize = 10, alpha=0.5)

            if self.current_task == 'modify rise peak':
                if event.button == 1:
                    x = event.xdata
                    y = event.ydata

                    min_distance = []
                    for trace in range(len(self.rises)):
                        for rise in range(len(self.rises[trace])):
                            min_distance.append([trace, rise, np.sqrt((x - self.time[self.rises[trace][rise][0][0]])**2 + (y-self.rises[trace][rise][1][0])**2)])

                    ind_rise = np.argmin([min_distance[i][2] for i in range(len(min_distance))])

                    self.active_fish0 = min_distance[ind_rise][0]
                    self.active_rise0 = min_distance[ind_rise][1]
                    self.active_index0 = self.rises[self.active_fish0][self.active_rise0][0][0]

                    if not self.tmp_plot_handle0 == None:
                            self.tmp_plot_handle0.remove()

                    self.tmp_plot_handle0, = self.ax.plot(self.time[self.active_index0],
                                                              self.data[self.active_fish0][self.active_index0], 'o',
                                                              color = 'red', markersize = 10, alpha=0.5)
                if event.button == 3:
                    if self.active_fish0 == None:
                        print('need active rise')
                    else:
                        x = event.xdata
                        next_t = self.time[~np.isnan(self.data[self.active_fish0])][self.time[~np.isnan(self.data[self.active_fish0])] > x][0]
                        self.active_index1 = np.where(self.time == next_t)[0][0]

                        if not self.tmp_plot_handle1 == None:
                            self.tmp_plot_handle1.remove()

                        self.tmp_plot_handle1, = self.ax.plot(self.time[self.active_index1],
                                                              self.data[self.active_fish0][self.active_index1], 'o',
                                                              color = 'green', markersize = 10, alpha=0.5)

            if self.current_task == 'modify rise end':
                if event.button == 1:
                    x = event.xdata
                    y = event.ydata

                    min_distance = []
                    for trace in range(len(self.rises)):
                        for rise in range(len(self.rises[trace])):
                            min_distance.append([trace, rise, np.sqrt((x - self.time[self.rises[trace][rise][0][1]])**2 + (y-self.rises[trace][rise][1][1])**2)])

                    ind_rise = np.argmin([min_distance[i][2] for i in range(len(min_distance))])

                    self.active_fish0 = min_distance[ind_rise][0]
                    self.active_rise0 = min_distance[ind_rise][1]
                    self.active_index0 = self.rises[self.active_fish0][self.active_rise0][0][1]

                    if not self.tmp_plot_handle0 == None:
                            self.tmp_plot_handle0.remove()

                    self.tmp_plot_handle0, = self.ax.plot(self.time[self.active_index0],
                                                              self.data[self.active_fish0][self.active_index0], 'o',
                                                              color = 'red', markersize = 10, alpha=0.5)
                if event.button == 3:
                    if self.active_fish0 == None:
                        print('need active rise')
                    else:
                        x = event.xdata
                        next_t = self.time[~np.isnan(self.data[self.active_fish0])][self.time[~np.isnan(self.data[self.active_fish0])] > x][0]
                        self.active_index1 = np.where(self.time == next_t)[0][0]

                        if not self.tmp_plot_handle1 == None:
                            self.tmp_plot_handle1.remove()

                        self.tmp_plot_handle1, = self.ax.plot(self.time[self.active_index1],
                                                              self.data[self.active_fish0][self.active_index1], 'o',
                                                              color = 'green', markersize = 10, alpha=0.5)

        self.fig.canvas.draw()

    def join_traces(self):
        color = np.random.rand(3, 1)

        # transefere data
        self.data[self.active_fish0][np.isnan(self.data[self.active_fish0])] = self.data[self.active_fish1][np.isnan(self.data[self.active_fish0])]

        # remove old plot handles and plot new one
        self.trace_handlers[self.active_fish0].remove()
        self.trace_handlers[self.active_fish1].remove()
        # self.trace_handlers[self.active_fish0], = self.ax.plot(self.time[~np.isnan(self.data[self.active_fish0])],
        #                                                        self.data[self.active_fish0][~np.isnan(self.data[self.active_fish0])], marker='.')

        self.trace_handlers[self.active_fish0], = self.ax.plot(self.time[~np.isnan(self.data[self.active_fish0])],
                                                               self.data[self.active_fish0][~np.isnan(self.data[self.active_fish0])], marker='.', color= color)

        # remove old rise plot handles and plot new ones
        if self.rises != None:
            self.rises[self.active_fish0] += self.rises[self.active_fish1]  # transfer rises

            self.rise_peak_handlers[self.active_fish0].remove()
            self.rise_peak_handlers[self.active_fish1].remove()
            self.rise_end_handlers[self.active_fish0].remove()
            self.rise_end_handlers[self.active_fish1].remove()

            peak_idx = [rise[0][0] for rise in self.rises[self.active_fish0]]
            floor_idx = [rise[0][1] for rise in self.rises[self.active_fish0]]

            # self.rise_peak_handlers[self.active_fish0], = self.ax.plot(self.time[peak_idx], self.data[self.active_fish0][peak_idx], 'o', color='red', markersize=7, markerfacecolor='None')
            self.rise_peak_handlers[self.active_fish0], = self.ax.plot(self.time[peak_idx], self.data[self.active_fish0][peak_idx], 'o', color=color, markersize=7)
            # self.rise_end_handlers[self.active_fish0], = self.ax.plot(self.time[floor_idx], self.data[self.active_fish0][floor_idx], 's', color='green', markersize=7, markerfacecolor='None')
            self.rise_end_handlers[self.active_fish0], = self.ax.plot(self.time[floor_idx], self.data[self.active_fish0][floor_idx], 's', color=color, markersize=7)

            self.rises.pop(self.active_fish1)

        # remove transfered data at old index and remove unused plot handles
        self.data.pop(self.active_fish1)
        self.trace_handlers.pop(self.active_fish1)
        self.rise_peak_handlers.pop(self.active_fish1)
        self.rise_end_handlers.pop(self.active_fish1)

        # set active plot and fish to None
        self.current_task = None
        self.active_fish0 = None
        self.active_fish1 = None
        self.active_plot_handle0 = None
        self.active_plot_handle1 = None

    def cut_trace(self):
        # trace plot handle removal
        self.trace_handlers[self.active_fish0].remove()
        self.tmp_plot_handle0.remove()

        c0 = np.random.rand(3, 1)
        c1 = np.random.rand(3, 1)

        # transfere trace data and re-plot it
        self.data.append(np.full(len(self.data[0]), np.nan))
        self.data[-1][self.active_index0:] = self.data[self.active_fish0][self.active_index0:]
        self.data[self.active_fish0][self.active_index0:] = np.full(len(self.data[self.active_fish0][self.active_index0:]), np.nan)

        self.trace_handlers[self.active_fish0], = self.ax.plot(self.time[~np.isnan(self.data[self.active_fish0])],
                                                               self.data[self.active_fish0][~np.isnan(self.data[self.active_fish0])], marker='.', color=c0)

        h, = self.ax.plot(self.time[~np.isnan(self.data[-1])], self.data[-1][~np.isnan(self.data[-1])], marker='.', color=c1)
        self.trace_handlers.append(h)

        # remove rises plot handels
        if self.rises != None:
            self.rise_peak_handlers[self.active_fish0].remove()
            self.rise_end_handlers[self.active_fish0].remove()

            i1 = np.arange(len(self.rises[self.active_fish0]))[np.array([self.rises[self.active_fish0][i][0][0] for i in range(len(self.rises[self.active_fish0]))]) >= self.active_index0]

            self.rises.append([])
            for i in reversed(i1):
                self.rises[-1].append(self.rises[self.active_fish0][i])
                self.rises[self.active_fish0].pop(i)

            # pre cut fish rises
            peak_idx0 = [rise[0][0] for rise in self.rises[self.active_fish0]]
            floor_idx0 = [rise[0][1] for rise in self.rises[self.active_fish0]]

            self.rise_peak_handlers[self.active_fish0], = self.ax.plot(self.time[peak_idx0], self.data[self.active_fish0][peak_idx0], 'o', color=c0, markersize=7)
            self.rise_end_handlers[self.active_fish0], = self.ax.plot(self.time[floor_idx0], self.data[self.active_fish0][floor_idx0], 's', color=c0, markersize=7)

            peak_idx1 = [rise[0][0] for rise in self.rises[-1]]
            floor_idx1 = [rise[0][1] for rise in self.rises[-1]]
            h, = self.ax.plot(self.time[peak_idx1], self.data[-1][peak_idx1], 'o', color=c1, markersize=7)
            self.rise_peak_handlers.append(h)
            h, = self.ax.plot(self.time[floor_idx1], self.data[-1][floor_idx1], 's', color=c1, markersize=7)
            self.rise_end_handlers.append(h)
        else:
            self.rise_peak_handlers.append(None)
            self.rise_end_handlers.append(None)

        self.tmp_plot_handle0 = None
        self.current_task = None
        self.active_fish0 = None
        self.active_index0 = None

        self.fig.canvas.draw()

    def modify_rise_peak(self):
        self.rise_peak_handlers[self.active_fish0].remove()
        self.tmp_plot_handle0.remove()
        self.tmp_plot_handle0 = None
        self.tmp_plot_handle1.remove()
        self.tmp_plot_handle1 = None

        self.rises[self.active_fish0][self.active_rise0][0][0] = self.active_index1
        self.rises[self.active_fish0][self.active_rise0][1][0] = self.data[self.active_fish0][self.active_index1]

        c0 = self.trace_handlers[self.active_fish0].get_color()

        peak_idx = [rise[0][0] for rise in self.rises[self.active_fish0]]
        self.rise_peak_handlers[self.active_fish0], = self.ax.plot(self.time[peak_idx], self.data[self.active_fish0][peak_idx], 'o', color=c0, markersize=7)

        self.all_reset()
        self.fig.canvas.draw()

    def modify_rise_end(self):
        self.rise_end_handlers[self.active_fish0].remove()
        self.tmp_plot_handle0.remove()
        self.tmp_plot_handle0 = None
        self.tmp_plot_handle1.remove()
        self.tmp_plot_handle1 = None

        self.rises[self.active_fish0][self.active_rise0][0][1] = self.active_index1
        self.rises[self.active_fish0][self.active_rise0][1][1] = self.data[self.active_fish0][self.active_index1]

        c1 = self.trace_handlers[self.active_fish0].get_color()

        floor_idx = [rise[0][1] for rise in self.rises[self.active_fish0]]
        self.rise_end_handlers[self.active_fish0], = self.ax.plot(self.time[floor_idx], self.data[self.active_fish0][floor_idx], 's', color=c1, markersize=7)

        self.all_reset()
        self.fig.canvas.draw()

    def all_reset(self):
        self.active_fish0 = None
        self.active_fish1 = None
        self.active_index0 = None
        self.active_index1 = None
        self.active_rise0 = None

        self.current_task = None

        if self.active_plot_handle0 != None:
            self.active_plot_handle0.set_linewidth(self.active_plot_handle0.get_linewidth() - 3.)
            self.active_plot_handle0 = None

        if self.active_plot_handle1 != None:
            self.active_plot_handle1.set_linewidth(self.active_plot_handle1.get_linewidth() - 3.)
            self.active_plot_handle1 = None

        if self.tmp_plot_handle0 != None:
            self.tmp_plot_handle0.remove()
            self.tmp_plot_handle0 = None

        if self.tmp_plot_handle1 != None:
            self.tmp_plot_handle1.remove()
            self.tmp_plot_handle1 = None


def main():
    parser = argparse.ArgumentParser(
        description='Pre analysis GUI to modify tracked EOD frequency traces of single- or multi electrode EOD recordings of weakly electric fish.',
        epilog='by bendalab (2015-2017)')
    parser.add_argument('-v', action='count', dest='verbose', help='verbosity level')
    parser.add_argument('file', nargs=1, default='', type=str, help='numpy file containing tracked EOD frequency traces from tracker_2.py.')
    parser.add_argument('rises', nargs='?', default='', type=str, help='rises detected for every EOD frequency trace.')
    args = parser.parse_args()

    a = np.load(args.file[0], mmap_mode='r+')
    data = list(a.copy())
    time = np.load(args.file[0].replace('fishes.', 'times.'))

    if args.rises != '':
        rises = list(np.load(args.rises))
    else:
        rises = None

    Human_tracker(data, time, rises)

if __name__ == '__main__':
    main()