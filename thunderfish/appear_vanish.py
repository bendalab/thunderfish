import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import glob
from IPython import embed
import scipy.stats as scp
from tqdm import tqdm
import matplotlib.mlab as mlab

def decibel(power, ref_power=1.0, min_power=1e-20):
    """
    Transform power to decibel relative to ref_power.
    ```
    decibel = 10 * log10(power/ref_power)
    ```

    Parameters
    ----------
    power: array
        Power values of the power spectrum or spectrogram.
    ref_power: float
        Reference power for computing decibel. If set to `None` the maximum power is used.
    min_power: float
        Power values smaller than `min_power` are set to `np.nan`.

    Returns
    -------
    decibel_psd: array
        Power values in decibel.
    """
    if ref_power is None:
        ref_power = np.max(power)
    decibel_psd = power.copy()
    decibel_psd[power < min_power] = np.nan
    decibel_psd[power >= min_power] = 10.0 * np.log10(decibel_psd[power >= min_power]/ref_power)
    return decibel_psd

def get_relevant_electrodes_power(x_pos, y_pos, sign_v):
    x_pos[x_pos >= 7] = 6.999999999999999999999999999999999
    y_pos[y_pos >= 7] = 6.999999999999999999999999999999999
    x_pos[x_pos < 0] = 0.000000000000000000000000000000001
    y_pos[y_pos < 0] = 0.000000000000000000000000000000001

    # ToDo: continue here

    x1_pos = x_pos + .2
    x1_pos[x1_pos >= 7] -= .4

    y1_pos = y_pos + .2
    y1_pos[y1_pos >= 7] -= .4

    impacts = []

    for Cxpos, Cypos in zip([x_pos, x1_pos, x_pos], [y_pos, y_pos, y1_pos]):
        x0 = np.floor(Cxpos)
        x1 = np.ceil(Cxpos)

        y0 = np.floor(Cypos)
        y1 = np.ceil(Cypos)

        x_sep = Cxpos % 1
        y_sep = Cypos % 1


        elecs = []
        for x in [x0, x1]:
            for y in [y0, y1]:
                elecs.append(x*8 + y)
        elecs = np.array(elecs, dtype = int)

        imp = (1 - x_sep) * ((1-y_sep) * sign_v[np.arange(len(sign_v)), elecs[0, :]] + y_sep * sign_v[np.arange(len(sign_v)), elecs[2, :]]) + \
              x_sep  * ((1-y_sep) * sign_v[np.arange(len(sign_v)), elecs[1, :]] + y_sep * sign_v[np.arange(len(sign_v)), elecs[3, :]])

        impacts.append(imp)

    x_impact = np.abs(impacts[1] - impacts[0]) / 20
    y_impact = np.abs(impacts[2] - impacts[0]) / 20

    impact = np.sqrt(x_impact**2 + y_impact**2)

    return impact


def load_data(folder0, only_sign = False):
    if only_sign:
        sign_v0 = np.load(os.path.join(folder0, 'sign_v.npy'))
        return sign_v0
    else:
        fund_v0 = np.load(os.path.join(folder0, 'fund_v.npy'))
        sign_v0 = np.load(os.path.join(folder0, 'sign_v.npy'))
        idx_v0 = np.load(os.path.join(folder0, 'idx_v.npy'))
        id_v0 = np.load(os.path.join(folder0, 'ident_v.npy'))
        times0 = np.load(os.path.join(folder0, 'times.npy'))
        spec0 =  np.load(os.path.join(folder0, 'spec.npy'))
        id_tag =  np.load(os.path.join(folder0, 'id_tag.npy'))
        st0_str = folder0[-6:].replace('_', '').replace(':', '').replace('/', '').replace('-', '')
        m0 = int(st0_str[:2]) * 60 +  int(st0_str[2:])

        elec_xy = np.array([np.arange(64) // 8, np.arange(64) % 8]).transpose()
        power_e = np.argsort(sign_v0, axis=1)[:, -6:]

        help_sign_v = (sign_v0.transpose() - np.min(sign_v0, axis=1)).transpose()
        normed_sign_v = (help_sign_v.transpose() / np.max(help_sign_v, axis=1)).transpose()
        normed_sign_of_e = np.take(normed_sign_v, power_e)

        x = np.sum(elec_xy[:, 0][power_e] * normed_sign_of_e, axis=1) / np.sum(normed_sign_of_e, axis=1)
        y = np.sum(elec_xy[:, 1][power_e] * normed_sign_of_e, axis=1) / np.sum(normed_sign_of_e, axis=1)

        # embed()
        # quit()

        plot = False
        if plot:
            print('im in')
            from matplotlib.animation import ArtistAnimation
            # from tqdm import tqdm

            fig, ax = plt.subplots(figsize=(20/2.54, 12/2.54))

            # anim = ArtistAnimation(fig, update, frames=np.arange(0, 30), interval=100)
            ax.set_xticks([0, 2, 4, 6])
            ax.set_xticklabels([0, 1, 2, 3])
            ax.set_xlabel('x-pos [m]', fontsize=14)
            ax.set_yticks([0, 2, 4, 6])
            ax.set_yticklabels([0, 1, 2, 3])
            ax.set_ylabel('y-pos [m]', fontsize=14)
            ax.tick_params(labelsize=12)
            id0 = 3.
            id1 = 2.

            i0 = np.arange(len(id_v0))[id_v0 == id0]
            i1 = np.arange(len(id_v0))[id_v0 == id1]
            frame = 0

            imgs = []

            grid, = ax.plot(elec_xy[:, 0], elec_xy[:, 1], 'x', color='white', markersize=3)
            for enu, i in enumerate(i0):
                print(enu)
                if enu >= 400:
                    break

                # if enu == 0:
                img = ax.imshow(sign_v0[i].reshape((8, 8)).T, interpolation='gaussian', cmap='hot')
                elec, = ax.plot(elec_xy[:, 0][power_e[i]], elec_xy[:, 1][power_e[i]], 'o', color='red')
                pos, = ax.plot(x[i], y[i], 'o', color='k')
                imgs.append([grid, img, elec, pos])

                # img.remove()
                # elec.remove()
                # pos.remove()
                # else:
                #
                #     img.set_data(sign_v0[i].reshape((8, 8)).T)
                #     elec.set_data(elec_xy[:, 0][power_e[i]], elec_xy[:, 1][power_e[i]])
                #     pos.set_data(x[i], y[i])
                #     imgs.append([grid, img, elec, pos])


                    # plt.draw()
                # fig.canvas.draw()
                # plt.pause(.1)

            ani = ArtistAnimation(fig, imgs, interval=150, blit=True, repeat_delay=0)
            ani.save('test.gif', dpi=80, writer='imagemagick')



            fig, ax = plt.subplots(figsize=(20/2.54, 12/2.54))

            # anim = ArtistAnimation(fig, update, frames=np.arange(0, 30), interval=100)
            ax.set_xticks([0, 2, 4, 6])
            ax.set_xticklabels([0, 1, 2, 3])
            ax.set_xlabel('x-pos [m]', fontsize=14)
            ax.set_yticks([0, 2, 4, 6])
            ax.set_yticklabels([0, 1, 2, 3])
            ax.set_ylabel('y-pos [m]', fontsize=14)
            ax.tick_params(labelsize=12)
            id0 = 3.
            id1 = 2.

            i0 = np.arange(len(id_v0))[id_v0 == id0]
            i = i0[40]

            grid, = ax.plot(elec_xy[:, 0], elec_xy[:, 1], 'x', color='white', markersize=3)
            img = ax.imshow(sign_v0[i].reshape((8, 8)).T, interpolation='gaussian', cmap='hot')
            elec, = ax.plot(elec_xy[:, 0][power_e[i]], elec_xy[:, 1][power_e[i]], 'o', color='red', alpha = 0.2)
            pos, = ax.plot(x[i], y[i], 'o', color='k')

            ax.scatter([3, 4, 3, 4], [0, 0, 1, 1], s=80, facecolor='none', edgecolor='k')
            #
            ax.plot([3, 3], [0, 1], lw = 3, color='k')
            ax.plot([4, 4], [0, 1], lw = 3, color='k')
            ax.plot([3, 4], [.6, .6], 'x', markersize=10, color='firebrick')
            #
            ax.plot([3, 4], [.6, .6], lw=3, color='k')
            #
            ax.plot(3.4, .6, 'o', color='blue')
            ax.plot(3.4, .6, 'x', markersize=10, color='firebrick')

            # print('done')
            embed()
            quit()
        return fund_v0, idx_v0, id_v0, times0, spec0, m0, id_tag, x, y


class Traces():
    def __init__(self, folders, shift):
        self.folders = folders
        self.shift = shift
        self.appear = []
        self.vanish = []
        self.stay_m = []

        self.current_task = None

        self.fig, self.ax = plt.subplots(figsize=(55. / 2.54, 30. / 2.54), facecolor='white')
        self.fig.canvas.mpl_connect('key_press_event', self.keypress)
        self.fig.canvas.mpl_connect('button_press_event', self.buttonpress)
        self.fig.canvas.mpl_connect('button_release_event', self.buttonrelease)

        # keymap.fullscreen : f, ctrl+f       # toggling
        # keymap.home : h, r, home            # home or reset mnemonic
        # keymap.back : left, c, backspace    # forward / backward keys to enable
        # keymap.forward : right, v           #   left handed quick navigation
        # keymap.pan : p                      # pan mnemonic
        # keymap.zoom : o                     # zoom mnemonic
        # keymap.save : s                     # saving current figure
        # keymap.quit : ctrl+w, cmd+w         # close the current figure
        # keymap.grid : g                     # switching on/off a grid in current axes
        # keymap.yscale : l                   # toggle scaling of y-axes ('log'/'linear')
        # keymap.xscale : L, k                # toggle scaling of x-axes ('log'/'linear')
        # keymap.all_axes : a                 # enable all axes

        plt.rcParams['keymap.save'] = ''  # was s
        plt.rcParams['keymap.back'] = ''  # was c
        plt.rcParams['keymap.forward'] = ''
        plt.rcParams['keymap.yscale'] = ''
        plt.rcParams['keymap.pan'] = ''
        plt.rcParams['keymap.home'] = ''
        plt.rcParams['keymap.fullscreen'] = ''

        self.fund_v = []
        self.idx_v = []
        self.sign_v = []
        self.id_v = []
        self.times = []
        self.m0 = []
        self.id_tag = []
        self.x_pos = []
        self.y_pos = []

        self.x = (None, None)
        self.y = (None, None)

        self.id_connect = [] # [rec, id, in_prev, in_next, color]
        self.id_handle = []

        self.connections = [] # [(rec_no, id), (rec_no, id)]
        self.connections_handle = []

        self.active_rec_id0 = (None, None)
        self.active_rec_id1 = (None, None)
        self.last_xy_lims = None
        self.task_text = None

        # embed()
        # quit()
        if os.path.exists('/home/raab/analysis/connections.npy') and os.path.exists('/home/raab/analysis/id_connect.npy'):
            self.id_connect = list(np.load('/home/raab/analysis/id_connect.npy'))
            self.connections = list(np.load('/home/raab/analysis/connections.npy'))

        for i in tqdm(range(len(self.folders))):
            Cfund_v, Cidx_v, Cid_v, Ctimes, Cspec, Cm0, Cid_tag, Cx, Cy = load_data(self.folders[i])
            self.fund_v.append(Cfund_v)
            self.idx_v.append(Cidx_v)
            self.id_v.append(Cid_v)
            self.times.append(Ctimes)
            self.m0.append(Cm0)
            self.id_tag.append(Cid_tag)
            self.x_pos.append(Cx)
            self.y_pos.append(Cy)
            # self.sign_v.append(Csign_v)

        # embed()
        # quit()

        for i in tqdm(range(len(self.folders))):
            # fund_v, idx_v, id_v, times, spec, m0, id_tag = load_data(self.folders[i])
            ids = self.id_tag[i][:, 0][self.id_tag[i][:, 1] == 1]

            # non taged plotting #
            mask = [id not in ids for id in self.id_v[i]]
            non_taged_idx = np.arange(len(self.id_v[i]))[mask]
            non_taged_idx = non_taged_idx[~np.isnan(self.id_v[i][non_taged_idx])]

            self.ax.plot((self.times[i][self.idx_v[i][non_taged_idx]][::90] + self.shift[i] * 60) / 3600, self.fund_v[i][non_taged_idx][::90], '.', color='white', markersize=1)

            # taged plotting #
            for id in ids:
                got_entry = False
                c = np.random.rand(3)
                if len(np.shape(self.id_connect)) > 1:
                    id_idx_of_interest = np.arange(len(self.id_connect))[(np.array(self.id_connect)[:, 0] == i) & (np.array(self.id_connect)[:, 1] == id)]
                    if len(id_idx_of_interest) == 0:
                        c = np.random.rand(3)
                    else:
                        got_entry = True
                        c = self.id_connect[id_idx_of_interest[0]][4]
                h, = self.ax.plot((self.times[i][self.idx_v[i][self.id_v[i] == id]][::90] + self.shift[i] * 60) / 3600,
                                  self.fund_v[i][self.id_v[i] == id][::90], color=c, marker='.', markersize=1)

                if got_entry:
                    self.id_handle.append(h)

                else:
                    self.id_connect.append(np.array([i, id, None, None, c], dtype = object))
                    self.id_handle.append(h)

        for i in range(len(self.connections)):
            id_connect_idx0 = np.arange(len(self.id_connect))[(np.array(self.id_connect)[:, 0] == self.connections[i][0][0]) & (np.array(self.id_connect)[:, 1] == self.connections[i][0][1])][0]
            id_connect_idx1 = np.arange(len(self.id_connect))[(np.array(self.id_connect)[:, 0] == self.connections[i][1][0]) & (np.array(self.id_connect)[:, 1] == self.connections[i][1][1])][0]

            c = self.id_connect[id_connect_idx0][4]

            h, = self.ax.plot([(self.times[self.id_connect[id_connect_idx0][0]][-1] + self.shift[self.id_connect[id_connect_idx0][0]] * 60) / 3600,
                               (self.times[self.id_connect[id_connect_idx1][0]][0] + self.shift[self.id_connect[id_connect_idx1][0]] * 60) / 3600],
                              [self.fund_v[self.id_connect[id_connect_idx0][0]][self.id_v[self.id_connect[id_connect_idx0][0]] == self.id_connect[id_connect_idx0][1]][-1],
                               self.fund_v[self.id_connect[id_connect_idx1][0]][self.id_v[self.id_connect[id_connect_idx1][0]] == self.id_connect[id_connect_idx1][1]][0]],
                              color=c)

            self.connections_handle.append(h)

        night_end = np.arange((455 + 24 * 60) / 60, (self.times[-1][-1] + self.shift[-1] * 60) / 3600, 24)

        self.ax.fill_between([0, 455 / 60], [400, 400], [950, 950], color='#666666')
        self.ax.fill_between([455 / 60, 455 / 60 + 12], [400, 400], [950, 950], color='#dddddd')
        for ne in night_end:
            self.ax.fill_between([ne - 12, ne], [400, 400], [950, 950], color='#666666', edgecolor=None)
            self.ax.fill_between([ne, ne + 12], [400, 400], [950, 950], color='#dddddd', edgecolor=None)

        self.ax.set_ylim([400, 950])
        self.ax.set_xlim([0, night_end[-1]+12])
        # plt.legend(loc=1)
        self.ax.set_ylabel('EOD frequency [Hz]')
        self.ax.set_xlabel('date')


        x_ticks = ['10.04.', '11.04.', '12.04.', '13.04.', '14.04.', '15.04.', '16.04.', '17.04.', '18.04.']
        self.ax.set_xticks(np.arange((1440 - self.m0[0] + 12 *  60) / 60, (self.times[-1][-1] / 60 + self.shift[-1]) / 60 + 24, 24))
        self.ax.set_xticklabels(x_ticks, rotation = 50)


        # embed()
        # quit()

        plt.show()

    def keypress(self, event):
        if event.key == 'a':
            self.current_task = 'analysis'

        if event.key == 'z':
            self.current_task = 'zoom'
            if self.task_text:
                self.task_text.remove()
            self.task_text = self.fig.text(0.1, 0.9, self.current_task)
        if event.key == 'c':
            self.current_task = 'connect'
            if self.task_text:
                self.task_text.remove()
            self.task_text = self.fig.text(0.1, 0.9, self.current_task)
        if event.key == 'e':
            embed()

        if event.key == 'backspace':
            if hasattr(self.last_xy_lims, '__len__'):
                self.ax.set_xlim(self.last_xy_lims[0][0], self.last_xy_lims[0][1])
                self.ax.set_ylim(self.last_xy_lims[1][0], self.last_xy_lims[1][1])

        if event.key == 'h':
            self.ax.set_ylim([400, 950])
            self.ax.set_xlim([-0.25, (self.times[-1][-1] + self.shift[-1] * 60) / 3600. + 0.25])

        if event.key == 'ctrl+q':
            plt.close(self.fig)
            # self.main_fig.close()
            return

        if event.key == 's':
            self.current_task = 'save_plot'

        if event.key == 'ctrl+s':
            if self.task_text:
                self.task_text.remove()
            self.task_text = self.fig.text(0.1, 0.9, 'saving ...')

            np.save('/home/raab/analysis/id_connect.npy', self.id_connect)
            np.save('/home/raab/analysis/connections.npy', self.connections)

            if self.task_text:
                self.task_text.remove()
            self.task_text = None
            self.fig.canvas.draw()

        if event.key == 'up':
            ylims = self.ax.get_ylim()
            self.ax.set_ylim(ylims[0] + 0.5 * (ylims[1] - ylims[0]), ylims[1] + 0.5 * (ylims[1] - ylims[0]))

        if event.key == 'down':
            ylims = self.ax.get_ylim()
            self.ax.set_ylim(ylims[0] - 0.5 * (ylims[1] - ylims[0]), ylims[1] - 0.5 * (ylims[1] - ylims[0]))

        if event.key == 'right':
            xlims = self.ax.get_xlim()[:]
            self.ax.set_xlim(xlims[0] + 0.5 * (xlims[1] - xlims[0]), xlims[1] + 0.5 * (xlims[1] - xlims[0]))

        if event.key == 'left':
            xlims = self.ax.get_xlim()[:]
            self.ax.set_xlim(xlims[0] - 0.5 * (xlims[1] - xlims[0]), xlims[1] - 0.5 * (xlims[1] - xlims[0]))

        if event.key == 'enter':
            if self.current_task == 'analysis':
                self.duration_analysis()

            if self.current_task == 'save_plot':
                print('saving plot...')
                self.fig.set_size_inches(20. / 2.54, 12. / 2.54)

                plot_nr = len(glob.glob('/home/raab/Desktop/plot*'))
                plt.tight_layout()
                self.fig.savefig('/home/raab/Desktop/plot%.0f.pdf' % plot_nr)

                print('... done!')
                self.fig.set_size_inches(55. / 2.54, 30. / 2.54)

            if self.current_task == 'connect':
                if self.active_rec_id0[1] != None and self.active_rec_id1[1] != None:
                    self.id_connect = np.array(self.id_connect)
                    # embed()
                    # quit()
                    id_connect_idx0 = np.arange(len(self.id_connect))[(self.id_connect[:, 0] == self.active_rec_id0[0]) & (self.id_connect[:, 1] == self.active_rec_id0[1])][0]
                    id_connect_idx1 = np.arange(len(self.id_connect))[(self.id_connect[:, 0] == self.active_rec_id1[0]) & (self.id_connect[:, 1] == self.active_rec_id1[1])][0]

                    c = self.id_connect[id_connect_idx0][4]
                    h, =self.ax.plot([(self.times[self.active_rec_id0[0]][-1] + self.shift[self.active_rec_id0[0]] * 60) / 3600,
                                      (self.times[self.active_rec_id1[0]][0] + self.shift[self.active_rec_id1[0]] * 60) / 3600],
                                     [self.fund_v[self.active_rec_id0[0]][self.id_v[self.active_rec_id0[0]] == self.active_rec_id0[1]][-1],
                                      self.fund_v[self.active_rec_id1[0]][self.id_v[self.active_rec_id1[0]] == self.active_rec_id1[1]][0]],
                                     color = c)
                    # self.id_connect[id_connect_idx1][5].set_color(self.id_connect[id_connect_idx0][4])
                    self.id_handle[id_connect_idx1].set_color(self.id_connect[id_connect_idx0][4])
                    self.id_connect[id_connect_idx1][4] = self.id_connect[id_connect_idx0][4]

                    self.id_connect[id_connect_idx0][3] = self.id_connect[id_connect_idx1][1]
                    self.id_connect[id_connect_idx1][2] = self.id_connect[id_connect_idx0][1]

                    self.connections.append([self.active_rec_id0, self.active_rec_id1])
                    self.connections_handle.append(h)

                    self.active_rec_id0 = (None, None)
                    self.active_rec_id1 = (None, None)

        self.fig.canvas.draw()

    def buttonpress(self, event):
        if event.button == 2:
            if self.task_text:
                self.task_text.remove()
            self.current_task = None
            self.current_task = None
            self.active_rec_id0 = (None, None)
            self.active_rec_id1 = (None, None)
        else:
            self.x = (event.xdata, 0)
            self.y = (event.ydata, 0)

    def buttonrelease(self, event):
        self.x = (self.x[0], event.xdata)
        self.y = (self.y[0], event.ydata)

        if self.current_task == 'zoom':
            self.last_xy_lims = [self.ax.get_xlim(), self.ax.get_ylim()]

            self.ax.set_xlim([np.min(self.x), np.max(self.x)])
            self.ax.set_ylim([np.min(self.y), np.max(self.y)])

        if self.current_task == 'connect':
            rec_nr = None
            for i in range(len(self.times)):
                if self.x[0] < (self.times[i][-1] + self.shift[i] * 60) / 3600 and self.x[0] > (self.times[i][0] + self.shift[i] * 60) / 3600:
                    rec_nr = i
                    break
                elif self.x[1] < (self.times[i][-1] + self.shift[i] * 60) / 3600 and self.x[1] > (self.times[i][0] + self.shift[i] * 60) / 3600:
                    rec_nr = i
                    break
                else:
                    continue

            active_idxs = np.arange(len(self.fund_v[rec_nr]))[(self.fund_v[rec_nr] > np.min(self.y)) &
                                                              (self.fund_v[rec_nr] < np.max(self.y)) &
                                                              ((self.times[rec_nr][self.idx_v[rec_nr]] + self.shift[rec_nr] * 60) / 3600. > np.min(self.x)) &
                                                              ((self.times[rec_nr][self.idx_v[rec_nr]] + self.shift[rec_nr] * 60) / 3600. < np.max(self.x))]
            possible_ids = np.array(np.array(self.id_connect)[:, 1][np.array(self.id_connect)[:, 0] == rec_nr], dtype=float)
            mask = [self.id_v[rec_nr][x] in possible_ids for x in active_idxs]
            active_idxs = active_idxs[mask]
            if event.button == 1:
                self.active_rec_id0 = (rec_nr ,self.id_v[rec_nr][active_idxs][0])
            elif event.button == 3:
                self.active_rec_id1 = (rec_nr, self.id_v[rec_nr][active_idxs][0])

        self.fig.canvas.draw()


    def reshape_data(self):
        valid_dur = []

        all_id_x_pos = []
        all_id_idxs = []
        all_id_y_pos = []
        all_id_freq = []
        all_id_pos_time = []
        # all_id_sign = []

        for i in tqdm(range(len(self.id_connect))):
            rec_no = self.id_connect[i][0]
            id_no = self.id_connect[i][1]

            id_x_pos = []
            id_y_pos = []
            id_freq = []

            id_idx = []
            for x in range(len(self.folders)):
                id_idx.append([])

            id_pos_time = []

            if self.id_connect[i][2] != None:  # id has another start id ...
                continue

            if self.id_connect[i][3] == None:  # got start and end id in same recording
                t0 = self.times[rec_no][self.idx_v[rec_no][self.id_v[rec_no] == id_no]][0] + self.shift[rec_no] * 60
                tn = self.times[rec_no][self.idx_v[rec_no][self.id_v[rec_no] == id_no]][-1] + self.shift[
                    rec_no] * 60

                # if t0 >= 300 or rec_no == 0:
                if t0 >= self.times[rec_no][0] + self.shift[rec_no] * 60 + 300 and tn < self.times[rec_no][-1] + \
                        self.shift[rec_no] * 60 - 300:
                    valid_dur.append(True)
                else:
                    valid_dur.append(False)
                self.appear.append((self.m0[0] + t0 / 60) % 1440)
                self.stay_m.append((tn - t0) / 60)
                # if tn <= self.times[rec_no][-1] - 300:
                self.vanish.append((self.m0[0] + tn / 60) % 1440)

                id_x_pos.extend(self.x_pos[rec_no][self.id_v[rec_no] == id_no])
                id_y_pos.extend(self.y_pos[rec_no][self.id_v[rec_no] == id_no])
                id_freq.extend(self.fund_v[rec_no][self.idx_v[rec_no][self.id_v[rec_no] == id_no]]) # ToDo fehlerhaft

                id_idx[rec_no].extend(np.arange(len(self.fund_v[rec_no]))[self.id_v[rec_no] == id_no])

                id_pos_time.extend(
                    self.times[rec_no][self.idx_v[rec_no][self.id_v[rec_no] == id_no]] + self.shift[rec_no] * 60)

            else:  # look for end in next recording
                t0 = self.times[rec_no][self.idx_v[rec_no][self.id_v[rec_no] == id_no]][0] + self.shift[rec_no] * 60

                id_x_pos.extend(self.x_pos[rec_no][self.id_v[rec_no] == id_no])
                id_y_pos.extend(self.y_pos[rec_no][self.id_v[rec_no] == id_no])

                id_pos_time.extend(
                    self.times[rec_no][self.idx_v[rec_no][self.id_v[rec_no] == id_no]] + self.shift[rec_no] * 60)
                id_freq.extend(self.fund_v[rec_no][self.idx_v[rec_no][self.id_v[rec_no] == id_no]])
                id_idx[rec_no].extend(np.arange(len(self.fund_v[rec_no]))[self.id_v[rec_no] == id_no])
                # id_sign.extend(self.sign_v[rec_no][self.idx_v[rec_no][self.id_v[rec_no] == id_no]])
                rec_no2 = rec_no + 1
                id_to_find = self.id_connect[i][3]

                while True:
                    a = np.array(self.id_connect)[(np.array(self.id_connect)[:, 0] == rec_no2) & (
                                np.array(self.id_connect)[:, 1] == id_to_find)][0]
                    id_x_pos.extend(self.x_pos[rec_no2][self.id_v[rec_no2] == id_to_find])
                    id_y_pos.extend(self.y_pos[rec_no2][self.id_v[rec_no2] == id_to_find])
                    id_freq.extend(self.fund_v[rec_no2][self.idx_v[rec_no2][self.id_v[rec_no2] == id_to_find]])

                    id_idx[rec_no2].extend(np.arange(len(self.fund_v[rec_no2]))[self.id_v[rec_no2] == id_to_find])
                    # id_sign.extend(self.sign_v[rec_no2][self.idx_v[rec_no2][self.id_v[rec_no2] == id_to_find]])
                    id_pos_time.extend(
                        self.times[rec_no2][self.idx_v[rec_no2][self.id_v[rec_no2] == id_to_find]] + self.shift[
                            rec_no2] * 60)
                    # id_y_pos.extend(self.y[rec_no][self.id_v[rec_no] == id_no])
                    if a[3] == None:
                        tn = self.times[rec_no2][self.idx_v[rec_no2][self.id_v[rec_no2] == id_to_find]][-1] + \
                             self.shift[rec_no2] * 60
                        break
                    else:
                        rec_no2 += 1
                        id_to_find = a[3]

                # if t0 >= 300 or rec_no == 0:
                if t0 >= self.times[rec_no][0] + self.shift[rec_no] * 60 + 300 and tn < self.times[rec_no2][-1] + \
                        self.shift[rec_no2] * 60 - 300:
                    valid_dur.append(True)
                else:
                    valid_dur.append(False)
                self.appear.append((self.m0[0] + t0 / 60) % 1440)
                self.stay_m.append((tn - t0) / 60)
                # if tn <= self.times[rec_no][-1] - 300:
                self.vanish.append((self.m0[0] + tn / 60) % 1440)
            all_id_x_pos.append(id_x_pos)
            all_id_y_pos.append(id_y_pos)
            all_id_freq.append(id_freq)

            all_id_idxs.append(id_idx)
            # all_id_sign.append(id_sign)
            all_id_pos_time.append(id_pos_time)
        return all_id_x_pos, all_id_idxs, all_id_y_pos, all_id_freq, all_id_pos_time, valid_dur

    def upper_crust(self, rec_0, rec_n, all_id_idxs, i, sign_v, plot=False):
        elec_xy = np.array([np.arange(64) // 8, np.arange(64) % 8]).transpose()
        dS = []
        p_at_d = []

        for rec in np.arange(rec_0, rec_n + 1):
            if rec == 2:
                continue
            idxs = np.array(all_id_idxs[i][rec])

            for x, y, p in zip(self.x_pos[rec][idxs], self.y_pos[rec][idxs], sign_v[rec][idxs]):
                a = np.sqrt((x - elec_xy[:, 0]) ** 2 + (y - elec_xy[:, 1]) ** 2)
                dS.extend(a)
                p_at_d.extend(p)

        dS = np.array(dS)
        p_at_d = np.array(p_at_d)
        p_at_d = p_at_d / 20.

        bins = np.logspace(np.log10(0.1), np.log10(8), num=50)
        bc = []
        b_data = []
        p975 = []
        pctls = []
        for l_bin, u_bin in zip(bins[:-1], bins[1:]):
            bc.append(np.mean([l_bin, u_bin]))
            cData = p_at_d[(dS >= l_bin) & (dS < u_bin)]

            b_data.append(cData)
            if len(cData) == 0:
                p975.append(None)
                pctls.append(np.array([None, None, None, None, None]))
            else:
                p975.append(np.percentile(cData, 97.5))
                pctls.append(np.percentile(cData, (2.5, 25, 50, 75, 97.5)))

        p975 = np.array(p975)
        bc = np.array(bc) / 2
        pctls = np.array(pctls)

        slope, q, _, _, _ = scp.linregress(np.log10(bc[(bc < 1) & (bc >= 0.2) & (p975 != None)]),
                                                   np.array(pctls[:, 4][(bc < 1) & (bc >= 0.2) & (p975 != None)],
                                                            dtype=float))
        if plot:
            fig2, ax2 = plt.subplots()

            ax2.fill_between(bc[p975 != None], np.array(pctls[:, 1][p975 != None], dtype=float),
                             np.array(pctls[:, 3][p975 != None], dtype=float), color='cornflowerblue', alpha=0.5)
            ax2.plot(bc[p975 != None], np.array(pctls[:, 0][p975 != None], dtype=float), lw=2, color='cornflowerblue')
            ax2.plot(bc[p975 != None], np.array(pctls[:, 4][p975 != None], dtype=float), lw=2, color='cornflowerblue')
            ax2.plot(bc[p975 != None], np.array(pctls[:, 2][p975 != None], dtype=float), lw=2, color='orange')

            ax2.set_xscale('log')

            ax2.plot(bc[(bc < 1) & (bc >= 0.2) & (p975 != None)], q + slope * np.log10(bc[(bc < 1) & (bc >= 0.2) & (p975 != None)]), color='orange', lw=2)

        return slope, q

    def dependent_interpolation(self, time_of_interest, t_fish0, t_fish1, m0, m1, sign_v, all_id_idxs, recording, i, j,
                                kernel = np.ones(100) / 100):

        freq_fish0 = self.fund_v[recording][all_id_idxs[i][recording]][m0]
        interp_freq_fish0 = np.interp(time_of_interest, t_fish0, freq_fish0)

        xpos_fish0 = self.x_pos[recording][all_id_idxs[i][recording]][m0]
        interp_xpos_fish0 = np.interp(time_of_interest, t_fish0, xpos_fish0)
        # interp_xpos_fish0 = np.convolve(interp_xpos_fish0, kernel, 'same')

        ypos_fish0 = self.y_pos[recording][all_id_idxs[i][recording]][m0]
        interp_ypos_fish0 = np.interp(time_of_interest, t_fish0, ypos_fish0)
        # interp_ypos_fish0 = np.convolve(interp_ypos_fish0, kernel, 'same')

        sign_fish0 = sign_v[recording][all_id_idxs[i][recording]][m0]
        interp_sign_fish0 = np.zeros((len(time_of_interest), np.shape(sign_v[recording])[1]))
        for e in np.arange(np.shape(sign_fish0)[1]):
            interp_sign_fish0[:, e] = np.interp(time_of_interest, t_fish0, sign_fish0[:, e])

        # interp values fish 1
        freq_fish1 = self.fund_v[recording][all_id_idxs[j][recording]][m1]
        interp_freq_fish1 = np.interp(time_of_interest, t_fish1, freq_fish1)

        xpos_fish1 = self.x_pos[recording][all_id_idxs[j][recording]][m1]
        interp_xpos_fish1 = np.interp(time_of_interest, t_fish1, xpos_fish1)
        # interp_xpos_fish1 = np.convolve(interp_xpos_fish1, kernel, 'same')

        ypos_fish1 = self.y_pos[recording][all_id_idxs[j][recording]][m1]
        interp_ypos_fish1 = np.interp(time_of_interest, t_fish1, ypos_fish1)
        # interp_ypos_fish1 = np.convolve(interp_ypos_fish1, kernel, 'same')

        sign_fish1 = sign_v[recording][all_id_idxs[j][recording]][m1]
        interp_sign_fish1 = np.zeros((len(time_of_interest), np.shape(sign_v[recording])[1]))
        for e in np.arange(np.shape(sign_fish1)[1]):
            interp_sign_fish1[:, e] = np.interp(time_of_interest, t_fish1, sign_fish1[:, e])

        return interp_xpos_fish0, interp_ypos_fish0, interp_sign_fish0, interp_freq_fish0, interp_xpos_fish1, \
               interp_ypos_fish1, interp_sign_fish1, interp_freq_fish1


    def detect_envelope_of_interest(self, hs_total_spectrum, hs_total_spec_time, spec_freqs, norm):
        total_over_th = []
        itt_o_i = []
        for itt in range(np.shape(hs_total_spectrum)[1]):
            th_l_u = spec_freqs[:-1][(hs_total_spectrum[:, itt][:-1] < norm[2][:-1]) & (hs_total_spectrum[:, itt][1:] > norm[2][1:])]
            th_u_l = spec_freqs[:-1][(hs_total_spectrum[:, itt][:-1] > norm[2][:-1]) & (hs_total_spectrum[:, itt][1:] < norm[2][1:])]

            if len(th_u_l) > 0 and len(th_l_u) > 0:
                if th_u_l[0] < th_l_u[0]:
                    th_u_l = th_u_l[1:]
                if len(th_u_l) != len(th_l_u):
                    th_l_u = th_l_u[:len(th_u_l)]
                if len(th_l_u) == 0:
                    continue

                if np.max(np.abs(th_u_l - th_l_u)) > 0.6:
                    ioi = np.arange(len(th_u_l))[(np.abs(th_u_l - th_l_u) > 0.6) & (np.abs(th_u_l - th_l_u) < 1)]
                    for Cioi in ioi:

                        h = np.full(len(spec_freqs), np.nan)
                        h[(spec_freqs > th_l_u[Cioi]) & (spec_freqs <= th_u_l[Cioi])] = \
                            hs_total_spectrum[:, itt][(spec_freqs > th_l_u[Cioi]) & (spec_freqs <= th_u_l[Cioi])]

                        total_over_th.append(h)
                        itt_o_i.append(itt)


        return total_over_th, itt_o_i

    def find_close_fish(self, all_id_idxs, all_id_x_pos,  all_id_y_pos, all_id_pos_time):

        sr = 3.0515715594055357
        nfft = 2**8
        noverlap = 2**7
        dt_nfft = nfft /sr / 2

        pairs = []
        min_dS = []
        dt = []
        for i in tqdm(range(len(all_id_idxs))):
            t0 = all_id_pos_time[i][0]
            tn = all_id_pos_time[i][-1]

            rec_0 = 0
            rec_n = 0
            got_start = False

            while True:
                if got_start == False and t0 > self.times[rec_0][-1] + self.shift[rec_0] * 60:
                    rec_0 += 1
                    rec_n += 1
                else:
                    got_start = True

                if got_start and tn > self.times[rec_n][-1] + self.shift[rec_n] * 60:
                    rec_n += 1
                else:
                    break

            sign_v = [None for folder in self.folders]

            for rec in np.arange(rec_0, rec_n+1):
                if rec == 2:
                    continue

                sign_v[rec] = load_data(self.folders[rec], only_sign=True)  # LOAD FUNCTION

            for j in tqdm(range(i + 1, len(all_id_idxs))):
                t0_2 = all_id_pos_time[j][0]
                tn_2 = all_id_pos_time[j][-1]

                if t0 > tn_2 or t0_2 > tn_2:
                    continue


                total_time_of_interest = []
                total_distance = []

                for recording in np.arange(rec_0, rec_n + 1):
                    if recording == 2:
                        continue

                    time_of_interest = self.times[recording][(self.times[recording] + self.shift[recording] * 60 >= all_id_pos_time[i][0]) &
                                                             (self.times[recording] + self.shift[recording] * 60 <= all_id_pos_time[i][-1]) &
                                                             (self.times[recording] + self.shift[recording] * 60 >= all_id_pos_time[j][0]) &
                                                             (self.times[recording] + self.shift[recording] * 60 <= all_id_pos_time[j][-1])] + self.shift[recording] * 60
                    if len(time_of_interest) == 0:
                        continue

                    m0 = np.array((self.times[recording][self.idx_v[recording][all_id_idxs[i][recording]]] + self.shift[recording] * 60 >= time_of_interest[0]) &
                                  (self.times[recording][self.idx_v[recording][all_id_idxs[i][recording]]] + self.shift[recording] * 60 <= time_of_interest[-1]))

                    m1 = np.array((self.times[recording][self.idx_v[recording][all_id_idxs[j][recording]]] + self.shift[recording] * 60 >= time_of_interest[0]) &
                                  (self.times[recording][self.idx_v[recording][all_id_idxs[j][recording]]] + self.shift[recording] * 60 <= time_of_interest[-1]))

                    t_fish0 = self.times[recording][self.idx_v[recording][all_id_idxs[i][recording]]][m0] + self.shift[recording] * 60
                    t_fish1 = self.times[recording][self.idx_v[recording][all_id_idxs[j][recording]]][m1] + self.shift[recording] * 60

                    if len(t_fish0) == 0 or len(t_fish1) == 0:
                        continue

                    # INTERPOLATION FUNKTION
                    interp_xpos_fish0, interp_ypos_fish0, interp_sign_fish0, interp_freq_fish0, interp_xpos_fish1, interp_ypos_fish1, interp_sign_fish1, interp_freq_fish1 = \
                        self.dependent_interpolation(time_of_interest, t_fish0, t_fish1, m0, m1, sign_v, all_id_idxs,
                                                     recording, i, j)

                    distance_f0f1 = np.sqrt((interp_xpos_fish0 - interp_xpos_fish1) ** 2 + (interp_ypos_fish0 - interp_ypos_fish1) ** 2)

                    total_time_of_interest.extend(time_of_interest)
                    total_distance.extend(distance_f0f1)

                if len(total_time_of_interest) == 0:
                    continue

                total_distance = np.array(total_distance) / 2

                time_bin_center = np.arange(nfft / sr / 2, total_time_of_interest[-1], nfft/sr/2)

                binned_distance = []
                for tidx in tqdm(range(len(time_bin_center))):
                    Cdistances = np.array(total_distance)[(total_time_of_interest >= time_bin_center[tidx] - dt_nfft) &
                                                          (total_time_of_interest < time_bin_center[tidx] + dt_nfft)]
                    binned_distance.append(np.mean(Cdistances))
                if np.any(np.array(binned_distance) < 0.5):
                    print(np.nanmin(binned_distance), (total_time_of_interest[-1] - total_time_of_interest[0]) / 3600, i, j)
                    min_dS.append(np.nanmin(binned_distance))
                    dt.append(total_time_of_interest[-1] - total_time_of_interest[0])
                    pairs.append((i, j))
            del sign_v

        return pairs, min_dS, dt



    def duration_analysis(self):
        plt.close()

        dates = ['09.04', '10.04.', '10.04.', '11.04.', '11.04.', '12.04.', '12.04.', '13.04.', '13.04.', '14.04.',
                 '14.04.', '15.04.', '15.04.', '16.04.', '16.04.', '17.04.', '17.04.']
        all_id_x_pos, all_id_idxs, all_id_y_pos, all_id_freq, all_id_pos_time, valid_dur = self.reshape_data()

        if not os.path.exists('../../../analysis/pairs.npy'):
            pairs, min_dS, dt = self.find_close_fish(all_id_idxs, all_id_x_pos,  all_id_y_pos, all_id_pos_time)
            pairs = np.array(pairs)
            min_dS = np.array(min_dS)
            dt = np.array(dt)
        else:
            pairs = np.load('../../../analysis/pairs.npy')
            min_dS = np.load('../../../analysis/pairs_min_ds.npy')
            dt = np.load('../../../analysis/pairs_dt.npy')

        # pairs longer 12h and minimal distance
        # embed()
        # quit()
        s_pairs = pairs[dt/ 3600 > 12][np.argsort(min_dS[dt/3600 > 12])]
        sorted_pairs = []
        sorted_pairs.extend(s_pairs[s_pairs[:, 1] == s_pairs[0, 0]][::-1, ::-1])
        sorted_pairs.extend(s_pairs[s_pairs[:, 0] == s_pairs[0, 0]])
        # sorted_pairs = sorted_pairs[sorted_pairs[:, 0] == sorted_pairs[0, 0]]

        # i = sorted_pairs[0][0]
        # j = sorted_pairs[0][1]

        glob_speed_slopes = []
        glob_speed_bins = []
        glob_spec_time_speed = []
        glob_spec_time_distance = []
        glob_spec_time_impact = []
        glob_slopes = []
        glob_intercepts = []
        glob_hs_total_spectrum = []
        glob_hs_total_spec_time = []
        glob_day_spec_mask = []
        glob_night_spec_mask = []
        glob_spec_freqs = []
        glob_total_time_of_interest = []
        glob_total_distance = []
        glob_total_impact = []
        glob_total_spectrum = []
        glob_total_spec_time = []
        glob_dS_bins = []
        glob_dS_bin_impact = []
        glob_dS_bin_slope = []
        glob_dS_bin_intercept = []

        for i, j in sorted_pairs:

            impact_power = []

            # ToDo: Initiate first for-loop
            # for i in range(len(all_id_idxs)):
            # i = 2

            # Find relevant recordings and load sign_v's
            t0 = all_id_pos_time[i][0]
            tn = all_id_pos_time[i][-1]

            rec_0 = 0
            rec_n = 0
            got_start = False

            while True:
                if got_start == False and t0 > self.times[rec_0][-1] + self.shift[rec_0] * 60:
                    rec_0 += 1
                    rec_n += 1
                else:
                    got_start = True

                if got_start and tn > self.times[rec_n][-1] + self.shift[rec_n] * 60:
                    rec_n += 1
                else:
                    break

            sign_v = [None for folder in self.folders]

            for rec in np.arange(rec_0, rec_n+1):
                if rec == 2:
                    continue

                sign_v[rec] = load_data(self.folders[rec], only_sign=True)  # LOAD FUNCTION

            slope, q = self.upper_crust(rec_0, rec_n, all_id_idxs, i, sign_v, plot=True)

            ############################################# impact on center ####################################
            # embed()
            # quit()
            total_distance_to_center = [] # to electrode 27
            total_impact_on_center = [] # on electrode 27
            total_time_to_center = []
            ttoi = []

            for rec in np.arange(rec_0, rec_n+1):
                if rec == 2:
                    continue
                Celec_x = 4
                Celec_y = 4
                toi = self.times[rec][(self.times[rec] + self.shift[rec] * 60 >= all_id_pos_time[i][0]) &
                                      (self.times[rec] + self.shift[rec] * 60 <= all_id_pos_time[i][-1])] + self.shift[rec] * 60

                impact_on_center = (sign_v[rec][all_id_idxs[i][rec]][:, 36] / 20)
                fish_t = self.times[rec][self.idx_v[rec][all_id_idxs[i][rec]]] + self.shift[rec] * 60
                fish_x_pos = self.x_pos[rec][all_id_idxs[i][rec]]
                fish_y_pos = self.y_pos[rec][all_id_idxs[i][rec]]

                distance_to_center = np.sqrt((fish_x_pos - Celec_x) ** 2 + (fish_y_pos - Celec_y)**2) / 2

                total_distance_to_center.extend(distance_to_center)
                total_impact_on_center.extend(impact_on_center)
                total_time_to_center.extend(fish_t)
                ttoi.extend(toi)

            total_distance_to_center = np.interp(ttoi, total_time_to_center, total_distance_to_center)
            total_impact_on_center = np.interp(ttoi, total_time_to_center, total_impact_on_center)
            total_speed_to_center = np.abs(np.diff(total_distance_to_center)) / np.diff(ttoi) # m/s
            kernel = np.ones(100) / 100
            total_speed_to_center = np.convolve(total_speed_to_center, kernel, 'same')

            if True:
                print('some plotting ... excluded!')
                # fig, ax = plt.subplots(2, 1, figsize=(20/2.54, 12/2.54), facecolor='white', sharex=True)
                # ax[1].plot(ttoi[:-1], total_speed_to_center)
                # ax[0].plot(ttoi, total_distance_to_center)
                # plt.show()
                # # total_impact_on_center = 10**(np.array(total_impact_on_center))
                # # fig, ax = plt.subplots()
                #
                # fig = plt.figure(figsize=(20/2.54, 12/2.54), facecolor='white')
                # ax = fig.add_axes(  [0.1, 0.1, 0.7, 0.6])
                # ax_u = fig.add_axes([0.1, 0.7, 0.7, 0.2], sharex=ax)
                # ax_r = fig.add_axes([0.8, 0.1, 0.1, 0.6], sharey=ax)
                # ax_u.axis('off')
                # ax_r.axis('off')
                #
                # ax.plot(total_distance_to_center, total_impact_on_center, '.', markersize = 1, alpha = 0.1)
                # ax.set_xlabel('distance to center (e3/3) [m]')
                # ax.set_ylabel('amplitude [mV/cm]')
                # # H, xedges, yedges = np.histogram2d(np.array(total_distance_to_center), np.array(total_impact_on_center), bins = 100)
                # # ax.imshow(H.T[::-1], extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], interpolation='gaussian', aspect='auto', cmap='jet')
                #
                # # ylim = ax.get_ylim()
                # # xlim = ax.get_xlim()
                # n, bins = np.histogram(total_distance_to_center, bins = 100)
                # n = n / np.sum(n) / (bins[1] - bins[0])
                # ax_u.bar(bins[:-1] + (bins[1] -  bins[0]) / 2, n, width = (bins[1] -  bins[0]) * 0.8)
                #
                # n, bins = np.histogram(total_impact_on_center, bins=50)
                # n = n / np.sum(n) / (bins[1] - bins[0])
                # ax_r.barh(bins[:-1] + (bins[1] -  bins[0]) / 2, n, height = (bins[1] -  bins[0]) * 0.8)
                # plt.show()


            ###################################################################################################

            # ToDo: Initiate second for-loop
            # for j in range(len(all_id_idxs)):
            # j = 3
            c = np.random.rand(3)
            # if i == j: # ToDo: necessary when looped
            #     continue

            t0_2 = all_id_pos_time[j][0]
            tn_2 = all_id_pos_time[j][-1]

            if t0 > tn_2 or t0_2 > tn_2:
                # continue # ToDo: necessary when looped
                print('impossible')

            total_time_of_interest = []
            total_impact = []
            total_freq1 = []
            total_psd_power = []
            total_psd_freq = []
            total_distance = []

            total_spectrum = []
            total_spec_time = []

            total_speed_spec = []
            total_speed_spec_time = []

            total_speed = []

            for recording in np.arange(rec_0, rec_n+1):
                if recording == 2:
                    continue

                time_of_interest = self.times[recording][(self.times[recording] + self.shift[recording] * 60 >= all_id_pos_time[i][0]) &
                                                         (self.times[recording] + self.shift[recording] * 60 <= all_id_pos_time[i][-1]) &
                                                         (self.times[recording] + self.shift[recording] * 60 >= all_id_pos_time[j][0]) &
                                                         (self.times[recording] + self.shift[recording] * 60 <= all_id_pos_time[j][-1])] + self.shift[recording] * 60
                if len(time_of_interest) == 0:
                    continue

                m0 = np.array((self.times[recording][self.idx_v[recording][all_id_idxs[i][recording]]] + self.shift[recording] * 60 >= time_of_interest[0]) &
                              (self.times[recording][self.idx_v[recording][all_id_idxs[i][recording]]] + self.shift[recording] * 60 <= time_of_interest[-1]))

                m1 = np.array( (self.times[recording][self.idx_v[recording][all_id_idxs[j][recording]]] + self.shift[recording] * 60 >= time_of_interest[0]) &
                               (self.times[recording][self.idx_v[recording][all_id_idxs[j][recording]]] + self.shift[recording] * 60 <= time_of_interest[-1]) )

                t_fish0 = self.times[recording][self.idx_v[recording][all_id_idxs[i][recording]]][m0] + self.shift[recording] * 60
                t_fish1 = self.times[recording][self.idx_v[recording][all_id_idxs[j][recording]]][m1] + self.shift[recording] * 60

                if len(t_fish0) == 0 or len(t_fish1) == 0:
                    continue

                # INTERPOLATION FUNKTION
                interp_xpos_fish0, interp_ypos_fish0, interp_sign_fish0, interp_freq_fish0, interp_xpos_fish1, interp_ypos_fish1, interp_sign_fish1, interp_freq_fish1 = \
                    self.dependent_interpolation(time_of_interest, t_fish0, t_fish1, m0, m1, sign_v, all_id_idxs, recording, i, j)

                distance_f0f1 = np.sqrt((interp_xpos_fish0 - interp_xpos_fish1)**2 + (interp_ypos_fish0 - interp_ypos_fish1)**2)

                impact = get_relevant_electrodes_power(interp_xpos_fish0, interp_ypos_fish0, interp_sign_fish1)

                total_time_of_interest.extend(time_of_interest)
                total_impact.extend(impact)
                total_freq1.extend(interp_freq_fish1)
                total_distance.extend(distance_f0f1)
                dS_speed = np.abs(np.diff(distance_f0f1)) / np.diff(time_of_interest)
                kernel = np.ones(100) / 100
                conv_dS_speed = np.convolve(dS_speed, kernel, 'same')

                sr = 1/(time_of_interest[1]-time_of_interest[0])
                # nfft = 2**12 = 23min; 2**10 = 5min (340s); 2**8 = 85s; 2**6 = 21s
                nfft = 2**10
                noverlap = 2**9

                psd_power, psd_freq = mlab.psd(10 **(impact/ 20), NFFT = nfft, Fs =sr, detrend=mlab.detrend_none, noverlap=noverlap)
                # psd_power, psd_freq = mlab.psd(10 **(impact/ 20), NFFT = nfft, Fs =sr, detrend=mlab.detrend_none)
                # print(len(impact))
                total_psd_freq.append(psd_freq)
                total_psd_power.append(psd_power)

                spectrum, spec_freqs, spec_time = mlab.specgram(10 **(impact/ 20), NFFT=nfft, Fs=sr, detrend=mlab.detrend_mean, window=mlab.window_hanning,
                                                                noverlap=noverlap, pad_to=None, sides='default',
                                                                scale_by_freq=None)

                speed_spec, speed_spec_freq, speed_spec_time = mlab.specgram(conv_dS_speed, NFFT=nfft, Fs=sr, detrend=mlab.detrend_mean, window=mlab.window_hanning,
                                                                             noverlap=noverlap, pad_to=None, sides='default',
                                                                             scale_by_freq=None)


                total_speed_spec.append(decibel(speed_spec))
                total_spectrum.append(decibel(spectrum))
                total_spec_time.append(spec_time + self.shift[recording] * 60)
                total_speed_spec_time.append(speed_spec_time + self.shift[recording] * 60)

                total_speed.extend(conv_dS_speed / 2)


            # embed()
            # quit()

            ################################### 4 col plots #######################################
            total_distance = np.array(total_distance) / 2
            hs_total_spectrum = np.hstack(total_spectrum)
            hs_total_spec_time = np.hstack(total_spec_time)

            spec_time_day_min = ((hs_total_spec_time / 60) + self.m0[0]) % 1440
            night_spec_mask = np.arange(len(spec_time_day_min))[(spec_time_day_min < 360) | (spec_time_day_min > 1080)]
            day_spec_mask = np.arange(len(spec_time_day_min))[(spec_time_day_min > 360) & (spec_time_day_min < 1080)]

            # ToDo: necessary here ?

            slopes = []
            intercepts = []
            for itt in tqdm(range(np.shape(hs_total_spectrum)[1])):
                itt0 = itt - 5 if itt >= 5 else 0
                itt1 = itt + 5

                SCslope, SCintercept, _, _, _ = scp.linregress(np.log10(spec_freqs[(spec_freqs >= 0.05) & (spec_freqs <= 1)]), np.mean(hs_total_spectrum[:, itt0:itt1],axis=1)[(spec_freqs >= 0.05) & (spec_freqs <= 1)] / 10)
                Cslope, Cintercept, _, _, _ = scp.linregress(np.log10(spec_freqs[(spec_freqs >= 0.05) & (spec_freqs <= 1)]), hs_total_spectrum[:, itt][(spec_freqs >= 0.05) & (spec_freqs <= 1)] / 10)
                # f, a = plt.subplots()
                # a.plot(spec_freqs, np.mean(hs_total_spectrum[:, itt0:itt1], axis= 1) / 10, color='k', lw=2)
                # a.plot(spec_freqs[(spec_freqs >= 0.05) & (spec_freqs <= 1)], SCintercept + SCslope * np.log10(spec_freqs[(spec_freqs >= 0.05) & (spec_freqs <= 1)]), color='orange', lw = 2)
                # a.set_xscale('log')
                # if itt >= 20:
                #     break
                slopes.append(Cslope)
                intercepts.append(Cintercept)



            dt_nfft = hs_total_spec_time[1] - hs_total_spec_time[0]

            spec_time_distance = []
            spec_time_distance_m = []
            spec_time_impact = []
            spec_time_speed = []

            for cidx in tqdm(range(len(hs_total_spec_time))):
                Cdistances = np.array(total_distance)[(total_time_of_interest >= hs_total_spec_time[cidx] - dt_nfft) & (total_time_of_interest < hs_total_spec_time[cidx] + dt_nfft)]
                Cimpact = np.array(total_impact)[(total_time_of_interest >= hs_total_spec_time[cidx] - dt_nfft) & (total_time_of_interest < hs_total_spec_time[cidx] + dt_nfft)]
                # ToDo:quick and dirty solution
                Cspeed = np.array(total_speed)[(np.array(total_time_of_interest)[:len(total_speed)] >= hs_total_spec_time[cidx] - dt_nfft) & np.array((total_time_of_interest)[:len(total_speed)] < hs_total_spec_time[cidx] + dt_nfft)]
                if len(Cdistances) > 0:
                    spec_time_distance.append(np.mean(Cdistances))
                    spec_time_distance_m.append(Cdistances)

                    spec_time_impact.append(np.mean(Cimpact))

                    n, bins = np.histogram(Cspeed)
                    bc = bins[:-1] + (bins[1]-bins[0]) / 2

                    spec_time_speed.append(bc[n == np.max(n)][0])
                else:
                    spec_time_distance.append(np.nan)
                    spec_time_distance_m.append(np.nan)

                    spec_time_impact.append(np.nan)
                    spec_time_speed.append(np.nan)

            # ToDo PCA ?!
            # length = []
            # for m in spec_time_distance_m:
            #     length.append(len(m))
            # ml = np.min(length)
            # for i in range(len(spec_time_distance_m)):
            #     spec_time_distance_m[i] = spec_time_distance_m[i][:ml]

            # spec_time_distance_m_norm = (np.array(spec_time_distance_m).transpose() - np.mean(spec_time_distance_m, axis= 1)).transpose()
            # from sklearn import decomposition
            # n_comp = 2
            # pca = decomposition.PCA(n_components=n_comp)
            # pca.fit(np.array(spec_time_distance_m_norm))  # data ist die Matrix mit den snippets
            # pcadata = pca.transform(spec_time_distance_m_norm)
            # # plot the projection onto the first 2 PCA vectors:
            # fig, ax = plt.subplots()
            # ax.scatter(pcadata[:,0], pcadata[:,1])
            # plt.show()

            speed_bins = np.arange(0, np.nanmax(np.array(spec_time_speed)), .05)
            speed_slopes = []
            for lb, ub in zip(speed_bins[:-1], speed_bins[1:]):
                speed_slopes.append(np.array(slopes)[(np.array(spec_time_speed) >= lb) & (np.array(spec_time_speed) > ub)])

            dS_bins = np.arange(0, 4.2, .2)
            dS_bin_impact = []
            dS_bin_slope = []
            dS_bin_intercept = []
            for lb, ub in zip(dS_bins[:-1], dS_bins[1:]):
                dS_bin_impact.append(np.array(total_impact)[(total_distance >= lb) & (total_distance < ub)])
                dS_bin_slope.append(np.array(slopes)[(spec_time_distance >= lb) & (spec_time_distance < ub)])
                dS_bin_intercept.append(np.array(intercepts)[(spec_time_distance >= lb) & (spec_time_distance < ub)])

            del sign_v
            glob_speed_slopes.append(speed_slopes)
            glob_speed_bins.append(speed_bins)
            glob_spec_time_speed.append(spec_time_speed)
            glob_spec_time_distance.append(spec_time_distance)
            glob_spec_time_impact.append(spec_time_impact)
            glob_slopes.append(slopes)
            glob_intercepts.append(intercepts)
            glob_hs_total_spectrum.append(hs_total_spectrum)
            glob_hs_total_spec_time.append(hs_total_spec_time)
            glob_day_spec_mask.append(day_spec_mask)
            glob_night_spec_mask.append(night_spec_mask)
            glob_spec_freqs.append(spec_freqs)
            glob_total_time_of_interest.append(total_time_of_interest)
            glob_total_distance.append(total_distance)
            glob_total_impact.append(total_impact)
            glob_total_spectrum.append(total_spectrum)
            glob_total_spec_time.append(total_spec_time)
            glob_dS_bins.append(dS_bins)
            glob_dS_bin_impact.append(dS_bin_impact)
            glob_dS_bin_slope.append(dS_bin_slope)
            glob_dS_bin_intercept.append(dS_bin_intercept)

        print('do the plotting manually...')
        embed()
        quit()

        # ToDo: frequency time plot of all candidates !!!
        # ToDo: maybe distance plot beneath !!!

        max_bin_count = 0
        for enu in range(len(glob_speed_bins)):
            if len(glob_speed_bins[enu]) - 1 > max_bin_count:
                max_bin_count = len(glob_speed_bins[enu]) - 1
                max_bins = glob_speed_bins[enu][:-1]

        collective_bin_slopes = []
        for enu in range(max_bin_count):
            collective_bin_slopes.append([])
            for f in range(len(glob_speed_slopes)):
                if enu >= len(glob_speed_slopes[f]):
                    continue
                else:
                    collective_bin_slopes[-1].extend(glob_speed_slopes[f][enu])

        f, a = plt.subplots(figsize=(20/2.54, 12/2.54), facecolor='white')
        a.set_xlabel('speed [m/s]')
        a.set_ylabel('slope / power law')
        a.boxplot(collective_bin_slopes, positions = max_bins + (max_bins[1] - max_bins[0]) / 2, widths = (max_bins[1] - max_bins[0])* 0.8 )
        # a.plot(glob_spec_time_speed[enu], glob_slopes[enu], '.', color='grey', markersize=1, alpha =0.5)


        for enu in range(len(glob_speed_bins)):
            f, a = plt.subplots(figsize=(20/2.54, 12/2.54), facecolor='white')
            a.set_xlabel('speed [m/s]')
            a.set_ylabel('slope / power law')
            a.boxplot(glob_speed_slopes[enu], positions = glob_speed_bins[enu][:-1] + (glob_speed_bins[enu][1] - glob_speed_bins[enu][0]) / 2, widths = (glob_speed_bins[enu][1] - glob_speed_bins[enu][0])* 0.8 )
            a.plot(glob_spec_time_speed[enu], glob_slopes[enu], '.', color='grey', markersize=1, alpha =0.5)

            ss, ii, r, p, _ = scp.linregress(np.array(glob_spec_time_speed[enu])[glob_spec_time_speed[enu] > glob_speed_bins[enu][1]], np.array(glob_slopes[enu])[glob_spec_time_speed[enu] > glob_speed_bins[enu][1]])
            a.plot([glob_speed_bins[enu][1], glob_speed_bins[enu][-1]], np.array([glob_speed_bins[enu][1], glob_speed_bins[enu][-1]])* ss + ii, label='p=%.3f\nr=%.2f' % (p, r))
            a.set_xlim([np.nanmin(glob_spec_time_speed[enu]) - 0.05, np.nanmax(glob_spec_time_speed[enu]) + 0.05])
            a.legend()

        ############################################
        # FOR POSTER
        slope_fig = plt.figure(figsize=(20./2.54, 10/2.54), facecolor='white')
        slope_ax = slope_fig.add_axes([.1, .125, .675, .825])
        slope_ax2 = slope_fig.add_axes([.8, .125, .15, .825])

        interc_fig = plt.figure(figsize=(20./2.54, 10/2.54), facecolor='white')
        interc_ax = interc_fig.add_axes([.1, .125, .675, .825])
        interc_ax2 = interc_fig.add_axes([.8, .125, .15, .825])

        ds_fig = plt.figure(figsize=(20./2.54, 10/2.54), facecolor='white')
        ds_ax = ds_fig.add_axes([.1, .125, .675, .825])
        ds_ax2 = ds_fig.add_axes([.8, .125, .15, .825])

        imp_fig = plt.figure(figsize=(20./2.54, 10/2.54), facecolor='white')
        imp_ax = imp_fig.add_axes([.1, .125, .675, .825])
        imp_ax2 = imp_fig.add_axes([.8, .125, .15, .825])

        speed_fig = plt.figure(figsize=(20./2.54, 10/2.54), facecolor='white')
        speed_ax = speed_fig.add_axes([.1, .125, .675, .825])
        speed_ax2 = speed_fig.add_axes([.8, .125, .15, .825])

        slope_ax.set_ylabel(r'exponent $\alpha$', fontsize=12)
        interc_ax.set_ylabel(r'factor c', fontsize=12)
        ds_ax.set_ylabel('distance [m]', fontsize=12)
        imp_ax.set_ylabel('electric field [mV/cm]', fontsize=12)
        speed_ax.set_ylabel('speed [m/s]', fontsize=12)

        d_slopes = []
        n_slopes = []

        d_interc = []
        n_interc = []

        d_ds = []
        n_ds = []

        d_imp = []
        n_imp = []

        d_speed = []
        n_speed = []

        for enu in np.arange(len(glob_slopes), dtype=int):
            d_slopes.append(np.array(glob_slopes[enu])[glob_day_spec_mask[enu]])
            n_slopes.append(np.array(glob_slopes[enu])[glob_night_spec_mask[enu]])

            d_interc.append(np.array(glob_intercepts[enu])[glob_day_spec_mask[enu]])
            n_interc.append(np.array(glob_intercepts[enu])[glob_night_spec_mask[enu]])

            d_ds.append(np.array(glob_spec_time_distance[enu])[glob_day_spec_mask[enu]][~np.isnan(np.array(glob_spec_time_distance[enu])[glob_day_spec_mask[enu]])])
            n_ds.append(np.array(glob_spec_time_distance[enu])[glob_night_spec_mask[enu]][~np.isnan(np.array(glob_spec_time_distance[enu])[glob_night_spec_mask[enu]])])

            d_imp.append(np.array(glob_spec_time_impact[enu])[glob_day_spec_mask[enu]][~np.isnan(np.array(glob_spec_time_impact[enu])[glob_day_spec_mask[enu]])])
            n_imp.append(np.array(glob_spec_time_impact[enu])[glob_night_spec_mask[enu]][~np.isnan(np.array(glob_spec_time_impact[enu])[glob_night_spec_mask[enu]])])

            d_speed.append(np.array(glob_spec_time_speed[enu])[glob_day_spec_mask[enu]][~np.isnan(glob_spec_time_speed[enu])[glob_day_spec_mask[enu]]])
            n_speed.append(np.array(glob_spec_time_speed[enu])[glob_night_spec_mask[enu]][~np.isnan(glob_spec_time_speed[enu])[glob_night_spec_mask[enu]]])

        bps = []

        bp = slope_ax.boxplot(d_slopes, positions=np.arange(len(d_slopes))+1, sym = '')
        bps.append(bp)
        bp = slope_ax.boxplot(n_slopes, positions=np.arange(len(n_slopes))+2 + len(d_slopes), sym = '')
        bps.append(bp)
        bp = slope_ax2.boxplot([np.hstack(d_slopes), np.hstack(n_slopes)], widths= .3, sym='')
        bps.append(bp)
        slope_ax2.set_yticks([])
        slope_ax2.set_ylim(slope_ax.get_ylim())

        stat, p = scp.mannwhitneyu(np.hstack(d_slopes), np.hstack(n_slopes))
        print('\n'+r'expotnet $\alpha$ day night')
        print('Mann-Whitney-U: U = %.3f; p = %.3f' % (stat, p))

        bp = interc_ax.boxplot(d_interc, positions=np.arange(len(d_slopes))+1, sym = '')
        bps.append(bp)
        bp = interc_ax.boxplot(n_interc, positions=np.arange(len(n_slopes))+2 + len(d_slopes), sym = '')
        bps.append(bp)
        bp = interc_ax2.boxplot([np.hstack(d_interc), np.hstack(n_interc)], widths= .3, sym='')
        bps.append(bp)
        interc_ax2.set_yticks([])
        interc_ax2.set_ylim(interc_ax.get_ylim())
        stat, p = scp.mannwhitneyu(np.hstack(d_interc), np.hstack(n_interc))
        print('\nfactor c day night')
        print('Mann-Whitney-U: U = %.3f; p = %.3f' % (stat, p))

        bp = ds_ax.boxplot(d_ds, positions=np.arange(len(d_slopes))+1, sym = '')
        bps.append(bp)
        bp = ds_ax.boxplot(n_ds, positions=np.arange(len(n_slopes))+2 + len(d_slopes), sym = '')
        bps.append(bp)
        bp = ds_ax2.boxplot([np.hstack(d_ds), np.hstack(n_ds)], widths= .3, sym='')
        bps.append(bp)
        ds_ax2.set_yticks([])
        ds_ax.set_ylim([0, 3.5])
        ds_ax2.set_ylim(ds_ax.get_ylim())
        stat, p = scp.mannwhitneyu(np.hstack(d_ds), np.hstack(n_ds))
        print('\ndistance day night')
        print('Mann-Whitney-U: U = %.3f; p = %.3f' % (stat, p))

        bp = imp_ax.boxplot(d_imp, positions=np.arange(len(d_slopes))+1, sym = '')
        bps.append(bp)
        bp = imp_ax.boxplot(n_imp, positions=np.arange(len(n_slopes))+2 + len(d_slopes), sym = '')
        bps.append(bp)
        bp = imp_ax2.boxplot([np.hstack(d_imp), np.hstack(n_imp)], widths= .3, sym='')
        bps.append(bp)
        imp_ax2.set_yticks([])
        imp_ax2.set_ylim(imp_ax.get_ylim())
        stat, p = scp.mannwhitneyu(np.hstack(d_imp), np.hstack(n_imp))
        print('\namplitude day night')
        print('Mann-Whitney-U: U = %.3f; p = %.3f' % (stat, p))

        bp = speed_ax.boxplot(d_speed, positions=np.arange(len(d_slopes))+1, sym = '')
        bps.append(bp)
        bp = speed_ax.boxplot(n_speed, positions=np.arange(len(n_slopes))+2 + len(d_slopes), sym = '')
        bps.append(bp)
        bp = speed_ax2.boxplot([np.hstack(d_speed), np.hstack(n_speed)], widths= .3, sym='')
        bps.append(bp)
        speed_ax2.set_yticks([])
        speed_ax2.set_ylim(speed_ax.get_ylim())
        stat, p = scp.mannwhitneyu(np.hstack(d_speed), np.hstack(n_speed))
        print('\nspeed day night')
        print('Mann-Whitney-U: U = %.3f; p = %.3f' % (stat, p))

        axx = [slope_ax, interc_ax, ds_ax, imp_ax, speed_ax]

        for c_ax in axx:
            c_ax.set_xlabel('Fish ID', fontsize=12)
            c_ax.set_xlim([.5, 21.5])
            ylim = c_ax.get_ylim()
            c_ax.fill_between([11, 21.5], [ylim[0], ylim[0]], [ylim[1], ylim[1]], color='#bbbbbb')
            c_ax.set_ylim(ylim)

            c_ax.set_xticks(np.concatenate((np.arange(10)+1, np.arange(10) + 12)))
            c_ax.set_xticklabels(np.arange(20) % 10 + 1)
            c_ax.tick_params(labelsize=10)

        axx2 = [slope_ax2, interc_ax2, ds_ax2, imp_ax2, speed_ax2]

        for c_ax in axx2:
            ylim = c_ax.get_ylim()
            c_ax.set_xlim(.5, 2.5)
            c_ax.fill_between([1.5, 2.5], [ylim[0], ylim[0]], [ylim[1], ylim[1]], color='#bbbbbb')
            c_ax.set_ylim(ylim)
            c_ax.set_xticks([1, 2])
            c_ax.set_xticklabels(['day', 'night'], rotation=70)
            c_ax.tick_params(labelsize=10)

        for bp in bps:
            for element in ['whiskers', 'fliers', 'means', 'medians', 'caps']:
                plt.setp(bp[element], color='k')


        plt.show()


        ############################
        fig, ax = plt.subplots(3, 2, figsize=(20./2.54, 20/2.54), facecolor='white')
        for enu in range(len(glob_slopes)):
        # for enu in np.arange(2):
            # ax[0].set_title('power law')
            ax[0, 0].set_ylabel('slope')
            ax[0, 1].set_ylabel('intercept')
            ax[1, 0].set_ylabel('distance')
            ax[1, 1].set_ylabel('impact')
            ax[2, 0].set_ylabel('speed [m/s]')

            ax[0, 0].boxplot([np.array(glob_slopes[enu])[glob_day_spec_mask[enu]], np.array(glob_slopes[enu])[glob_night_spec_mask[enu]]], positions = [enu+1, enu+1+len(glob_slopes)], sym='')
            ax[0, 1].boxplot([np.array(glob_intercepts[enu])[glob_day_spec_mask[enu]], np.array(glob_intercepts[enu])[glob_night_spec_mask[enu]]], positions = [enu+1, enu+1+len(glob_slopes)], sym='')
            ax[1, 0].boxplot([np.array(glob_spec_time_distance[enu])[glob_day_spec_mask[enu]][~np.isnan(np.array(glob_spec_time_distance[enu])[glob_day_spec_mask[enu]])],
                              np.array(glob_spec_time_distance[enu])[glob_night_spec_mask[enu]][~np.isnan(np.array(glob_spec_time_distance[enu])[glob_night_spec_mask[enu]])]],
                             positions = [enu+1, enu+1+len(glob_slopes)], sym='')
            ax[1, 1].boxplot([np.array(glob_spec_time_impact[enu])[glob_day_spec_mask[enu]][~np.isnan(np.array(glob_spec_time_impact[enu])[glob_day_spec_mask[enu]])],
                              np.array(glob_spec_time_impact[enu])[glob_night_spec_mask[enu]][~np.isnan(np.array(glob_spec_time_impact[enu])[glob_night_spec_mask[enu]])]],
                             positions = [enu+1, enu+1+len(glob_slopes)], sym='')

            ax[2, 0].boxplot([np.array(glob_spec_time_speed[enu])[glob_day_spec_mask[enu]][~np.isnan(glob_spec_time_speed[enu])[glob_day_spec_mask[enu]]],
                              np.array(glob_spec_time_speed[enu])[glob_night_spec_mask[enu]][~np.isnan(glob_spec_time_speed[enu])[glob_night_spec_mask[enu]]]],
                             positions = [enu+1, enu+1+len(glob_slopes)], sym='')

        axx = np.hstack(ax)
        for c_ax in axx[:-1]:
            c_ax.set_xlim([.5, 14.5])
            ylim = c_ax.get_ylim()
            c_ax.fill_between([7.5, 14.5], [ylim[0], ylim[0]], [ylim[1], ylim[1]], color='#bbbbbb')
            c_ax.set_ylim(ylim)

        axx[0].set_xticks([])
        axx[1].set_xticks([])
        axx[2].set_xticks([])

        axx[3].set_xticks(np.arange(14)+1)
        axx[3].set_xticklabels(np.arange(14) % 7 + 1)

        axx[4].set_xticks(np.arange(14)+1)
        axx[4].set_xticklabels(np.arange(14) % 7 + 1)

        axx[5].set_visible(False)

        plt.tight_layout()

        ###########################################################################
        # FOR POSTER

        # for i in range(len(glob_slopes)):
        fig33, ax33 = plt.subplots(3, 3, figsize=(18/2.54, 18/2.54), facecolor='white', sharex='col', sharey ='row') # row, col

        fig33.text(0.05, .765, 'amplitude [mV/cm]', va='center', ha='center', rotation = 90, fontsize=12)
        fig33.text(0.05, .33 / 2 + .33, r'exponent $\alpha$', va='center', ha='center', rotation = 90, fontsize=12)
        fig33.text(0.05, .225, 'factor c', va='center', ha='center', rotation = 90, fontsize=12)

        ax33[2, 0].set_xlabel('distance [m]', fontsize=12)
        ax33[2, 1].set_xlabel('amplitude [mV/cm]', fontsize=12)
        ax33[2, 2].set_xlabel(r'exponent $\alpha$', fontsize=12)
        # ax33[0, 0].set_ylabel('amplitude [mV/cm]', fontsize=12)
        # ax33[1, 0].set_ylabel(r'exponent $\alpha$', fontsize=12)
        # ax33[2, 0].set_ylabel('factor c', fontsize=12)

        ax33[0, 1].set_visible(False)
        ax33[0, 2].set_visible(False)
        ax33[1, 2].set_visible(False)

        # ToDo: check imshow matrix proportion !!!

        # H, xedges, yedges = np.histogram2d(np.concatenate(glob_total_distance), np.concatenate(glob_total_impact), bins= 100)
        H, xedges, yedges = np.histogram2d(np.concatenate(glob_total_distance), np.concatenate(glob_total_impact), bins= [np.linspace(0, 3.5, 100), np.linspace(0, 0.2, 100)])
        ax33[0, 0].imshow(H.T[::-1], extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], interpolation='gaussian', aspect='auto', cmap='jet')
        stats, p = scp.pearsonr(np.concatenate(glob_total_distance), np.concatenate(glob_total_impact))
        ax33[0, 0].text(2.4, .17, 'r=%.2f\np=%.3f' % (stats, p), color='white', fontsize=8)
        # ax33[0, 1].set_ylim([0, 0.6])

        # H, xedges, yedges = np.histogram2d(np.concatenate(glob_spec_time_distance)[~np.isnan(np.concatenate(glob_spec_time_distance))],
        #                                    np.concatenate(glob_slopes)[~np.isnan(np.concatenate(glob_spec_time_distance))], bins = 100)
        H, xedges, yedges = np.histogram2d(np.concatenate(glob_spec_time_distance)[~np.isnan(np.concatenate(glob_spec_time_distance))],
                                           np.concatenate(glob_slopes)[~np.isnan(np.concatenate(glob_spec_time_distance))], bins = [np.linspace(0, 3.5, 100), np.linspace(-3.5, 0, 100)])
        ax33[1, 0].imshow(H.T[::-1], extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], interpolation='gaussian', aspect='auto', cmap='jet')
        stats, p = scp.pearsonr(np.concatenate(glob_spec_time_distance)[~np.isnan(np.concatenate(glob_spec_time_distance))],
                                           np.concatenate(glob_slopes)[~np.isnan(np.concatenate(glob_spec_time_distance))])
        ax33[1, 0].text(2.4, -0.55, 'r=%.2f\np=%.3f' % (stats, p), color='white', fontsize=8)

        # H, xedges, yedges = np.histogram2d(np.concatenate(glob_spec_time_distance)[~np.isnan(np.concatenate(glob_spec_time_distance))],
        #                                    np.concatenate(glob_intercepts)[~np.isnan(np.concatenate(glob_spec_time_distance))], bins = 100)
        H, xedges, yedges = np.histogram2d(np.concatenate(glob_spec_time_distance)[~np.isnan(np.concatenate(glob_spec_time_distance))],
                                           np.concatenate(glob_intercepts)[~np.isnan(np.concatenate(glob_spec_time_distance))], bins = [np.linspace(0, 3.5, 100), np.linspace(-8, -2, 100)])
        ax33[2, 0].imshow(H.T[::-1], extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], interpolation='gaussian', aspect='auto', cmap='jet')
        stats, p = scp.pearsonr(np.concatenate(glob_spec_time_distance)[~np.isnan(np.concatenate(glob_spec_time_distance))],
                                           np.concatenate(glob_intercepts)[~np.isnan(np.concatenate(glob_spec_time_distance))])
        ax33[2, 0].text(2.4, -3, 'r=%.2f\np=%.3f' % (stats, p), color='white', fontsize=8)

        # H, xedges, yedges = np.histogram2d(np.concatenate(glob_spec_time_impact)[~np.isnan(np.concatenate(glob_spec_time_impact))],
        #                                    np.concatenate(glob_slopes)[~np.isnan(np.concatenate(glob_spec_time_impact))], bins = 100)
        H, xedges, yedges = np.histogram2d(np.concatenate(glob_spec_time_impact)[~np.isnan(np.concatenate(glob_spec_time_impact))],
                                           np.concatenate(glob_slopes)[~np.isnan(np.concatenate(glob_spec_time_impact))], bins = [np.linspace(0, 0.5, 100), np.linspace(-3.5, 0, 100)])
        ax33[1, 1].imshow(H.T[::-1], extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], interpolation='gaussian', aspect='auto', cmap='jet')
        stats, p = scp.pearsonr(np.concatenate(glob_spec_time_impact)[~np.isnan(np.concatenate(glob_spec_time_impact))],
                                           np.concatenate(glob_slopes)[~np.isnan(np.concatenate(glob_spec_time_impact))])
        ax33[1, 1].text(.35, -0.55, 'r=%.2f\np=%.3f' % (stats, p), color='white', fontsize=8)

        # H, xedges, yedges = np.histogram2d(np.concatenate(glob_spec_time_impact)[~np.isnan(np.concatenate(glob_spec_time_impact))],
        #                                    np.concatenate(glob_intercepts)[~np.isnan(np.concatenate(glob_spec_time_impact))], bins = 100)
        H, xedges, yedges = np.histogram2d(np.concatenate(glob_spec_time_impact)[~np.isnan(np.concatenate(glob_spec_time_impact))],
                                           np.concatenate(glob_intercepts)[~np.isnan(np.concatenate(glob_spec_time_impact))], bins = [np.linspace(0, 0.5, 100), np.linspace(-8, -2, 100)])
        ax33[2, 1].imshow(H.T[::-1], extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], interpolation='gaussian', aspect='auto', cmap='jet')
        stats, p = scp.pearsonr(np.concatenate(glob_spec_time_distance)[~np.isnan(np.concatenate(glob_spec_time_distance))],
                                           np.concatenate(glob_intercepts)[~np.isnan(np.concatenate(glob_spec_time_distance))])
        ax33[2, 1].text(.35, -3, 'r=%.2f\np=%.3f' % (stats, p), color='white', fontsize=8)

        # H, xedges, yedges = np.histogram2d(np.concatenate(glob_slopes), np.concatenate(glob_intercepts), bins = 100)
        H, xedges, yedges = np.histogram2d(np.concatenate(glob_slopes), np.concatenate(glob_intercepts), bins = [np.linspace(-3, 0, 100), np.linspace(-8, -2, 100)])
        ax33[2, 2].imshow(H.T[::-1], extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], interpolation='gaussian', aspect='auto', cmap='jet')
        stats, p = scp.pearsonr(np.concatenate(glob_slopes), np.concatenate(glob_intercepts))
        ax33[2, 2].text(-.95, -3, 'r=%.2f\np=%.3f' % (stats, p), color='white', fontsize=8)

        for ax in np.hstack(ax33):
            ax.tick_params(labelsize=10)


        ######################################

        # Todo: extract one powerspectrum here !!! for day and night

        for speed_slopes, speed_bins, spec_time_speed, spec_time_distance, spec_time_impact, slopes, intercepts, \
            hs_total_spectrum, hs_total_spec_time, day_spec_mask, night_spec_mask, spec_freqs, total_time_of_interest, \
            total_distance, total_impact, total_spectrum, total_spec_time, dS_bins, dS_bin_impact, dS_bin_slope, \
            dS_bin_intercept in zip(glob_speed_slopes, glob_speed_bins, glob_spec_time_speed, glob_spec_time_distance,
                                    glob_spec_time_impact, glob_slopes, glob_intercepts, glob_hs_total_spectrum,
                                    glob_hs_total_spec_time, glob_day_spec_mask, glob_night_spec_mask, glob_spec_freqs,
                                    glob_total_time_of_interest, glob_total_distance, glob_total_impact, glob_total_spectrum,
                                    glob_total_spec_time, glob_dS_bins, glob_dS_bin_impact, glob_dS_bin_slope, glob_dS_bin_intercept):

            # f, a = plt.subplots(figsize=(20/2.54, 12/2.54), facecolor='white')
            # a.set_xlabel('speed [m/s]')
            # a.set_ylabel('slope / power law')
            # a.boxplot(speed_slopes, positions = speed_bins[:-1] + (speed_bins[1] - speed_bins[0]) / 2, widths = (speed_bins[1] - speed_bins[0])* 0.8 )
            # a.plot(spec_time_speed, slopes, '.', color='grey', markersize=1, alpha =0.5)
            #
            # ss, ii, r, p, _ = scp.linregress(np.array(spec_time_speed)[spec_time_speed > speed_bins[1]], np.array(slopes)[spec_time_speed > speed_bins[1]])
            # a.plot([speed_bins[1], speed_bins[-1]], np.array([speed_bins[1], speed_bins[-1]])* ss + ii, label='p=%.3f\nr=%.2f' % (p, r))
            # a.set_xlim([np.nanmin(spec_time_speed) - 0.05, np.nanmax(spec_time_speed) + 0.05])
            # a.legend()


            f, a = plt.subplots(1, 2, figsize=(20./2.54, 12/2.54), facecolor='white', sharey=True)
            a[0].set_ylabel('power [dB]', fontsize=12)
            a[0].set_xlabel('frequency [Hz]', fontsize=12)
            a[1].set_xlabel('frequency [Hz]', fontsize=12)

            a[0].set_title('day mean envelope PSD', fontsize=12)
            a[0].plot(spec_freqs, np.mean(hs_total_spectrum[:, day_spec_mask], axis=1) / 10, color='k', lw = 1)
            a[0].fill_between(spec_freqs,
                              np.mean(hs_total_spectrum[:, day_spec_mask], axis=1) / 10 - np.std(hs_total_spectrum[:, day_spec_mask], axis=1) / 10 / np.sqrt(len(day_spec_mask)),
                              np.mean(hs_total_spectrum[:, day_spec_mask], axis=1) / 10 + np.std(hs_total_spectrum[:, day_spec_mask], axis=1) / 10 / np.sqrt(len(day_spec_mask)), color='grey', alpha = 0.5)

            night_slope, night_intercept, _, _, _ = scp.linregress(
                np.log10(spec_freqs[(spec_freqs >= 0.05) & (spec_freqs <= 1)]),
                np.mean(hs_total_spectrum[:, night_spec_mask], axis=1)[(spec_freqs >= 0.05) & (spec_freqs <= 1)] / 10)
            day_slope, day_intercept, _, _, _ = scp.linregress(
                np.log10(spec_freqs[(spec_freqs >= 0.05) & (spec_freqs <= 1)]),
                np.mean(hs_total_spectrum[:, day_spec_mask], axis=1)[(spec_freqs >= 0.05) & (spec_freqs <= 1)] / 10)

            a[0].plot(spec_freqs[(spec_freqs >= 0.05) & (spec_freqs <= 1)], day_intercept + day_slope * np.log10(spec_freqs[(spec_freqs >= 0.05) & (spec_freqs <= 1)]), color='orange', lw = 2, label='q:%.3f\nc:%.3f' % (day_slope, day_intercept))
            a[0].set_xscale('log')
            a[0].legend(loc=3)

            a[1].set_title('night mean envelope PSD', fontsize=12)
            a[1].plot(spec_freqs, np.mean(hs_total_spectrum[:, night_spec_mask], axis=1) / 10, color='k', lw = 1)
            a[1].fill_between(spec_freqs,
                              np.mean(hs_total_spectrum[:, night_spec_mask], axis=1) / 10 - np.std(hs_total_spectrum[:, night_spec_mask] / np.sqrt(len(night_spec_mask)), axis=1) / 10,
                              np.mean(hs_total_spectrum[:, night_spec_mask], axis=1) / 10 + np.std(hs_total_spectrum[:, night_spec_mask] / np.sqrt(len(night_spec_mask)), axis=1) / 10, color='grey', alpha = 0.5)
            a[1].plot(spec_freqs[(spec_freqs >= 0.05) & (spec_freqs <= 1)], night_intercept + night_slope * np.log10(spec_freqs[(spec_freqs >= 0.05) & (spec_freqs <= 1)]), color='orange', lw = 2, label='q:%.3f\nc:%.3f' % (night_slope, night_intercept))
            a[1].set_xscale('log')
            a[1].legend(loc=3, fontsize=10)
            a[0].tick_params(labelsize=10)
            a[1].tick_params(labelsize=10)
            plt.tight_layout()



            # fig, ax = plt.subplots(3, 2, figsize=(20./2.54, 20/2.54), facecolor='white', sharex='col')
            # # ax[0].set_title('power law')
            # ax[0, 0].set_ylabel('slope')
            # ax[0, 1].set_ylabel('intercept')
            # ax[1, 0].set_ylabel('distance')
            # ax[1, 1].set_ylabel('impact')
            # ax[2, 0].set_ylabel('speed [m/s]')
            #
            # ax[0, 0].boxplot([np.array(slopes)[day_spec_mask], np.array(slopes)[night_spec_mask]])
            # ax[0, 1].boxplot([np.array(intercepts)[day_spec_mask], np.array(intercepts)[night_spec_mask]])
            # ax[1, 0].boxplot([np.array(spec_time_distance)[day_spec_mask][~np.isnan(np.array(spec_time_distance)[day_spec_mask])],
            #                   np.array(spec_time_distance)[night_spec_mask][~np.isnan(np.array(spec_time_distance)[night_spec_mask])]])
            # ax[1, 1].boxplot([np.array(spec_time_impact)[day_spec_mask][~np.isnan(np.array(spec_time_impact)[day_spec_mask])],
            #                   np.array(spec_time_impact)[night_spec_mask][~np.isnan(np.array(spec_time_impact)[night_spec_mask])]])
            #
            # ax[2, 0].boxplot([np.array(spec_time_speed)[day_spec_mask][~np.isnan(spec_time_speed)[day_spec_mask]],
            #                   np.array(spec_time_speed)[night_spec_mask][~np.isnan(spec_time_speed)[night_spec_mask]]])
            #
            #
            #
            # ax[2, 0].set_xticks([1, 2])
            # ax[1, 1].set_xticks([1, 2])
            # ax[2, 0].set_xticklabels(['day', 'night'], rotation = 90)
            # ax[1, 1].set_xticklabels(['day', 'night'], rotation = 90)
            # ax[2, 1].set_visible(False)
            # plt.tight_layout()

            ###

            fig4c, ax4c = plt.subplots(4, 1, figsize=(20/2.54, 24/2.54), facecolor='white', sharex=True)
            ax4c[0].set_ylabel('distance [m]')
            ax4c[1].set_ylabel('impact [mV/cm * a]')
            ax4c[2].set_ylabel('q / power law')
            ax4c[3].set_ylabel('c / intercept')
            ax4c[3].set_xlabel('time [s]')

            ax4c[0].plot(total_time_of_interest, np.array(total_distance))
            ax4c[1].plot(total_time_of_interest, total_impact)

            ax4c[2].plot(hs_total_spec_time, slopes)
            ax4c[3].plot(hs_total_spec_time, intercepts)


            ax4c[0].plot(hs_total_spec_time[~np.isnan(spec_time_distance)], np.array(spec_time_distance)[~np.isnan(np.array(spec_time_distance))], color='green', marker = '.')

            fig4c, ax4c = plt.subplots(4, 1, figsize=(20 / 2.54, 24 / 2.54), facecolor='white', sharex=True)
            ax4c[0].set_ylabel('n')
            ax4c[1].set_ylabel('impact [mV/cm]')
            ax4c[2].set_ylabel('q / power law')
            ax4c[3].set_ylabel('c / intercept')
            ax4c[3].set_xlabel('distance [m]')

            n, bins = np.histogram(total_distance, bins = dS_bins)
            ax4c[0].bar( dS_bins[:-1] + (dS_bins[1] - dS_bins[0]) / 2, n, width = (dS_bins[1] - dS_bins[0]) * 0.9, align='center')
            ax4c[1].boxplot(dS_bin_impact, sym='', positions = dS_bins[:-1] + (dS_bins[1] - dS_bins[0]) / 2, widths = (dS_bins[1] - dS_bins[0]) / 2)
            ax4c[2].boxplot(dS_bin_slope, sym='', positions = dS_bins[:-1] + (dS_bins[1] - dS_bins[0]) / 2, widths = (dS_bins[1] - dS_bins[0]) / 2)
            ax4c[3].boxplot(dS_bin_intercept, sym='', positions = dS_bins[:-1] + (dS_bins[1] - dS_bins[0]) / 2, widths = (dS_bins[1] - dS_bins[0]) / 2)

            ################################### 4 col plots - END #######################################
            ################################### 3 X 3 plots #######################################

            # fig33, ax33 = plt.subplots(3, 3, figsize=(20/2.54, 20/2.54), facecolor='white', sharex='col', sharey ='row') # row, col
            # ax33[2, 0].set_xlabel('distance [m]')
            # ax33[2, 1].set_xlabel('amp [mV/cm * a]')
            # ax33[2, 2].set_xlabel('q / power law')
            # ax33[0, 0].set_ylabel('amp [mV/m * a]')
            # ax33[1, 0].set_ylabel('q / power law')
            # ax33[2, 0].set_ylabel('c / intercept')
            #
            # ax33[0, 1].set_visible(False)
            # ax33[0, 2].set_visible(False)
            # ax33[1, 2].set_visible(False)
            #
            # # ToDo: check imshow matrix proportion !!!
            #
            # H, xedges, yedges = np.histogram2d(np.array(total_distance), np.array(total_impact), bins= 100)
            # ax33[0, 0].imshow(H.T[::-1], extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], interpolation='gaussian', aspect='auto', cmap='jet')
            #
            # H, xedges, yedges = np.histogram2d(np.array(spec_time_distance)[~np.isnan(np.array(spec_time_distance))], np.array(slopes)[~np.isnan(np.array(spec_time_distance))], bins = 100)
            # ax33[1, 0].imshow(H.T[::-1], extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], interpolation='gaussian', aspect='auto', cmap='jet')
            #
            # H, xedges, yedges = np.histogram2d(np.array(spec_time_distance)[~np.isnan(np.array(spec_time_distance))], np.array(intercepts)[~np.isnan(np.array(spec_time_distance))], bins = 100)
            # ax33[2, 0].imshow(H.T[::-1], extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], interpolation='gaussian', aspect='auto', cmap='jet')
            #
            # H, xedges, yedges = np.histogram2d(np.array(spec_time_impact)[~np.isnan(np.array(spec_time_impact))], np.array(slopes)[~np.isnan(np.array(spec_time_impact))], bins = 100)
            # ax33[1, 1].imshow(H.T[::-1], extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], interpolation='gaussian', aspect='auto', cmap='jet')
            #
            # H, xedges, yedges = np.histogram2d(np.array(spec_time_impact)[~np.isnan(np.array(spec_time_impact))], np.array(intercepts)[~np.isnan(np.array(spec_time_impact))], bins = 100)
            # ax33[2, 1].imshow(H.T[::-1], extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], interpolation='gaussian', aspect='auto', cmap='jet')
            #
            # H, xedges, yedges = np.histogram2d(np.array(slopes), np.array(intercepts), bins = 100)
            # ax33[2, 2].imshow(H.T[::-1], extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], interpolation='gaussian', aspect='auto', cmap='jet')

            ################################### 3 X 3 plots - END #######################################

            # figx, axx = plt.subplots(2, 2, sharex='col')
            # axx[0, 0].plot(total_time_of_interest - total_time_of_interest[0], np.array(total_distance), lw = 2)
            # # ToDo: get white spaces between... extend closes gaps between !!!
            # for spec_idx in range(len(total_spectrum)):
            #     axx[1, 0].imshow(total_spectrum[spec_idx][::-1], extent=[total_spec_time[spec_idx][0], total_spec_time[spec_idx][-1], spec_freqs[0], spec_freqs[-1]], aspect='auto', cmap='jet')
            #
            # norm = np.percentile(np.hstack(total_spectrum), (10, 50, 90), axis= 1)
            #
            # axx[0, 1].plot(spec_freqs, norm[1], color='darkgreen', lw = 2, zorder = 3)
            # axx[0, 1].plot(spec_freqs, norm[0], color='k', lw = 2, zorder = 3)
            # axx[0, 1].plot(spec_freqs, norm[2], color='k', lw = 2, zorder = 3)

            # total_over_th = np.zeros(len(freqs))

            # EVENT DETECTION FUNCTION
            ################### FUNCTION OPTIMIZE ########################################

            # total_over_th, itt_o_i = self.detect_envelope_of_interest(hs_total_spectrum, hs_total_spec_time, spec_freqs, norm)

            # total_over_th = []
            # norm_total_over_th = []
            # itt_o_i = []
            #
            # for itt in tqdm(range(np.shape(hs_total_spectrum)[1])):
            #     if hs_total_spec_time[itt] < 900:
            #         continue
            #     else:
            #         mask = np.arange(len(hs_total_spec_time))[(hs_total_spec_time > hs_total_spec_time[itt] - 900) & (hs_total_spec_time < hs_total_spec_time[itt])]
            #
            #     # print(len(mask))
            #     if len(mask) < 80:
            #         continue
            #
            #     norm = np.percentile(hs_total_spectrum[:, mask], (5, 50, 95), axis=1)
            #     iqr = norm[2] - norm[0]
            #
            #     # ToDo: rethink this shit !!!
            #     th_l_u = spec_freqs[:-1][(hs_total_spectrum[:, itt][:-1] < norm[2][:-1]) & (hs_total_spectrum[:, itt][1:] > norm[2][1:])]
            #     th_u_l = spec_freqs[:-1][(hs_total_spectrum[:, itt][:-1] > norm[2][:-1]) & (hs_total_spectrum[:, itt][1:] < norm[2][1:])]
            #
            #     if hs_total_spectrum[0, itt] > norm[2][0]:
            #         th_l_u = np.concatenate((np.array([spec_freqs[0]]), th_l_u))
            #     if hs_total_spectrum[-1, itt] > norm[2][-1]:
            #         th_u_l = np.concatenate((np.array([spec_freqs[0]]), th_u_l))
            #
            #     # fig, ax = plt.subplots()
            #     # got_it = False
            #     if len(th_u_l) > 0 and len(th_l_u) > 0:
            #         if th_u_l[0] < th_l_u[0]:
            #             th_u_l = th_u_l[1:]
            #         if len(th_u_l) != len(th_l_u):
            #             th_l_u = th_l_u[:len(th_u_l)]
            #         if len(th_l_u) == 0:
            #             continue
            #
            #         if np.max(np.abs(th_u_l - th_l_u)) > 0.6:
            #             ioi = np.arange(len(th_u_l))[(np.abs(th_u_l - th_l_u) > 0.6)]
            #             # ioi = np.arange(len(th_u_l))[(np.abs(th_u_l - th_l_u) > 0.6) & (np.abs(th_u_l - th_l_u) < 1)]
            #             for Cioi in ioi:
            #                 h = np.full(len(spec_freqs), np.nan)
            #                 h[(spec_freqs > th_l_u[Cioi]) & (spec_freqs <= th_u_l[Cioi])] = hs_total_spectrum[:, itt][(spec_freqs > th_l_u[Cioi]) & (spec_freqs <= th_u_l[Cioi])]
            #
            #                 over_th_to_iqr = (h - norm[2]) / iqr
            #                 if len(over_th_to_iqr[~np.isnan(over_th_to_iqr)]) == 0:
            #                     continue
            #                 # if np.any(over_th_to_iqr > .25):
            #                 if len(over_th_to_iqr[~np.isnan(over_th_to_iqr)][over_th_to_iqr[~np.isnan(over_th_to_iqr)] >= .25]) / len(over_th_to_iqr[~np.isnan(over_th_to_iqr)]) >= .25:
            #                     if np.nanmax(h - norm[2]) > 12:
            #                         total_over_th.append(h)
            #                         norm_total_over_th.append(h - norm[2])
            #                         itt_o_i.append(itt)

            ################### FUNCTION OPTIMIZE ##############################

            # if total_over_th != []:
            #     # for enu, itt in enumerate(itt_o_i):
            #     for enu in np.arange(len(itt_o_i))[np.argsort(np.nanmax(norm_total_over_th, axis = 1))]:
            #         c = np.random.rand(3)
            #         alpha = enu / (len(itt_o_i) - 1)
            #         axx[0, 1].plot(spec_freqs, total_over_th[enu], color=c, lw = 2, zorder = 2, alpha = alpha**2)
            #         axx[1, 1].plot(spec_freqs, norm_total_over_th[enu], color=c, lw = 2, zorder = 2, alpha = alpha**2)
            #         axx[1, 0].plot(hs_total_spec_time[itt_o_i[enu]], spec_freqs[-1] + 0.1, '.', color = c)
            #     axx[1, 1].plot(spec_freqs, np.nanmean(norm_total_over_th, axis=0), '--', color='k', lw=2)
            #
            # axx[0, 1].set_xscale('log')
            # axx[0, 1].set_xlim([0.05, 2])
            # axx[1, 1].set_xlim(axx[0, 1].get_xlim())
            # axx[1, 1].set_xscale('log')
            # axx[0, 0].set_xlim(axx[1, 0].get_xlim())
            #
            # plt.show()

            # fig, ax = plt.subplots(2, 1, figsize=(20/2.54, 14/2.54), facecolor='white', sharex=True)
            # fig3, ax3 = plt.subplots(figsize=(20/2.54, 12/2.54), facecolor='white', sharex=True)

            # if len(total_impact) > 0:
            #     p = 10.**(np.array(total_impact))
            #
            #     if (total_time_of_interest[-1] - total_time_of_interest[0]) / 3600 > 12:
            #
            #         # ax[1].plot(total_time_of_interest, p, lw = 2, color=c)
            #         # ax[0].plot(total_time_of_interest, total_distance, lw = 2, color=c)
            #         # ax[1].set_xlabel('time [s]')
            #         # ax[0].set_ylabel('distance [m]')
            #         # ax[1].set_ylabel(r'impact [mV / cm $\alpha$]')
            #
            #         # ax3.loglog(total_psd_freq[0][total_psd_freq[0] > 0.01],
            #         #            np.sum(np.array(total_psd_power), axis=0)[total_psd_freq[0] > 0.01] /
            #         #            np.sum(np.array(total_psd_power), axis=0)[total_psd_freq[0] > 0.01][0], color=c, marker = '.')
            #         impact_power.append(np.sum(np.array(total_psd_power), axis=0)[total_psd_freq[0] > 0.01] / np.sum(np.array(total_psd_power), axis=0)[total_psd_freq[0] > 0.01][0])




            ################### WIP ######################
            # embed()
            # quit()

            ### AREA ANALYSIS ####
        last_min = (self.times[-1][-1] + self.shift[len(self.times)-1] * 60) / 60
        first_day_start_m = 1440 - self.m0[0] + 6 * 60
        day_night_switch_m = np.concatenate((np.array([0]), np.arange(first_day_start_m, last_min + 720, 720)))

        #############################################################################
        # FOR POSTER

        colors = ['k', '#BA2D22', '#53379B', '#F47F17', '#3673A4', '#AAB71B', '#DC143C', '#1E90FF', 'firebrick', 'cornflowerblue', 'forestgreen']
        sorted_pairs = np.array(sorted_pairs)

        for enu, i in enumerate(np.hstack([sorted_pairs[0, 0], sorted_pairs[:, 1]])):
            print(enu)
            day_area = []
            night_area = []

            # for i in tqdm(range(len(all_id_x_pos))):
            start_phase = 0
            end_phase = 0
            for j in range(len(day_night_switch_m)):
                if all_id_pos_time[i][0] / 60 >= day_night_switch_m[j]:
                    start_phase = j
                if all_id_pos_time[i][-1] / 60 >= day_night_switch_m[j]:
                    end_phase = j + 1

            id_pos_time_min = np.array(all_id_pos_time[i]) / 60

            subplot_count = end_phase - start_phase
            subplot_count = subplot_count if subplot_count % 2 == 0 else subplot_count + 1
            if start_phase % 2 != 0 and end_phase % 2 != 0:
                subplot_count += 2

            sp = 0 if start_phase % 2 == 0 else 1

            fig = plt.figure(figsize=(subplot_count / 2 * 6 / 2.54, 12/2.54), facecolor='white')
            if enu == 0:
                g_fig = plt.figure(figsize=(subplot_count / 2 * 6 / 2.54, 12/2.54), facecolor='white')
                g_axs = []
            day_title = False
            night_title = False

            if sp == 1:
                ax = fig.add_subplot(2, int(subplot_count / 2), 1)
                ax.set_frame_on(False)
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_ylabel(r'$\bf{night}$' + '\ny [m]', fontsize=12)
                if start_phase == 0:
                    ax.set_title(r'$\bf{%s}$' % dates[0], fontsize=12)
                else:
                    ax.set_title(r'$\bf{%s}$' % dates[start_phase], fontsize=12)
                night_title = True
                if enu == 0:
                    g_ax = g_fig.add_subplot(2, int(subplot_count / 2), 1)
                    g_ax.set_frame_on(False)
                    g_ax.set_xticks([])
                    g_ax.set_yticks([])
                    g_ax.set_ylabel(r'$\bf{night}$' + '\ny [m]', fontsize=12)
                    if start_phase == 0:
                        g_ax.set_title(r'$\bf{%s}$' % dates[0], fontsize=12)
                    else:
                        g_ax.set_title(r'$\bf{%s}$' % dates[start_phase], fontsize=12)

            id_night_area = []
            id_day_area = []
            for j in np.arange(start_phase, end_phase):

                c_sp = (sp % 2) * subplot_count / 2 + sp // 2 +1

                ax = fig.add_subplot(2, int(subplot_count / 2), c_sp)
                if enu == 0:
                    g_ax = g_fig.add_subplot(2, int(subplot_count / 2), c_sp)
                    g_axs.append(g_ax)

                snip_x_pos = np.array(all_id_x_pos[i])[(id_pos_time_min >= day_night_switch_m[j]) & (id_pos_time_min < day_night_switch_m[j+1])]
                snip_y_pos = np.array(all_id_y_pos[i])[(id_pos_time_min >= day_night_switch_m[j]) & (id_pos_time_min < day_night_switch_m[j+1])]

                H, xedges, yedges = np.histogram2d(snip_x_pos, snip_y_pos, bins=(np.arange(9) - 0.5, np.arange(9) - 0.5))
                # H, xedges, yedges = np.histogram2d(snip_x_pos, snip_y_pos, bins=(np.arange(-0.5, 7.51, 0.5), np.arange(-0.5, 7.51, 0.5)))
                H_turned = H.T
                # H_turned /= np.max(H_turned)
                ax.imshow(H_turned[::-1] / np.max(H_turned), interpolation='gaussian', extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], vmin=0, vmax=1)
                ax.set_xlim([0, 7])
                ax.set_ylim([0, 7])
                #

                X, Y = np.meshgrid(xedges[:-1] +  (xedges[1] - xedges[0]) / 2, yedges[:-1] +  (yedges[1] - yedges[0]) / 2)
                # ax.contour(X, Y, H_turned / np.max(H_turned), levels=[.05, .25], colors = ['orange', 'firebrick'], alpha=0.7)
                ax.contour(X, Y, H_turned / np.max(H_turned), levels=[.25, .5], colors = ['orange', 'firebrick'], alpha=0.7)
                g_axs[j].contour(X, Y, H_turned / np.max(H_turned), levels=[.5], colors = colors[enu], alpha=0.7)

                ax.plot(snip_x_pos, snip_y_pos, color='white', alpha = 0.1, lw= 1)

                # g_axs[j].contour(X, Y, H_turned / np.max(H_turned), levels=[.05, .25], colors = [colors[enu], colors[enu]], alpha=0.5)

                ax.set_xticks([0, 2, 4, 6])
                ax.set_xticklabels([0, 1, 2, 3])
                p = len(np.hstack(H_turned)[np.hstack(H_turned >= np.max(H_turned) * 0.1)]) * ((xedges[1] - xedges[0]) / 2)**2
                ax.text(0, 0, r'%.2f$m^2$' % p, color='orange')
                ax.set_yticks([0, 2, 4, 6])
                ax.set_yticklabels([0, 1, 2, 3])

                if sp % 2 == 0:
                    try:
                        ax.set_title(r'$\bf{%s}$' % (dates[j]), fontsize=12)
                        if enu == 0:
                            g_axs[j].set_title(r'$\bf{%s}$' % (dates[j]), fontsize=12)
                    except:
                        print('got that error!')
                        embed()
                        quit()

                    id_night_area.append(p)
                    if night_title == False:
                        if enu == 0:
                            g_axs[j].set_ylabel(r'$\bf{night}$'+'\ny [m]', fontsize=12)
                        ax.set_ylabel(r'$\bf{night}$'+'\ny [m]', fontsize=12)
                        night_title = True
                elif sp % 2 == 1:
                    ax.set_xlabel('x [m]', fontsize=12)
                    if enu == 0:
                        g_axs[j].set_xlabel('x [m]', fontsize=12)
                    id_day_area.append(p)
                    if day_title == False:
                        if enu == 0:
                            g_axs[j].set_ylabel(r'$\bf{day}$'+'\ny [m]', fontsize=12)
                        ax.set_ylabel(r'$\bf{day}$'+'\ny [m]', fontsize=12)
                        day_title = True
                sp += 1
                # loop end

                night_area.append(id_night_area)
                day_area.append(id_day_area)

                # fig.suptitle('A. leptohynchus; EOD frequency: %.1fHz' % np.mean(all_id_freq[i]))
                plt.tight_layout()

        g_fig.tight_layout()
        plt.show()
        ####################################################################################
        # if sp <= 4:
        #     plt.close()
        # else:
        #     print(np.mean(all_id_freq[i]))


        quit()
        ### DURATION ANALYSIS ###

        fig, ax = plt.subplots(1, 2, sharey = True)
        # ax[0].plot(np.ones(len(np.hstack(night_area)))*0.875 + np.random.rand(len(np.hstack(night_area))) / 4, np.hstack(night_area), '.', markersize=1)
        # ax[0].plot(np.ones(len(np.hstack(day_area)))*1.875 + np.random.rand(len(np.hstack(day_area))) / 4, np.hstack(day_area), '.', markersize=1)
        ax[0].boxplot([np.hstack(night_area), np.hstack(day_area)], sym = '')
        ax[0].set_ylabel('occupied area in $m^2$')
        ax[0].set_xticklabels(['night', 'day'])
        ax[0].set_title('all individuals')
        d = []
        n = []
        for i in range(len(night_area)):
            if len(night_area[i]) == 0 or len(day_area[i]) == 0:
                continue
            d.append(np.mean(day_area[i]))
            n.append(np.mean(night_area[i]))

        ax[1].boxplot([n, d], sym = '')
        ax[1].set_xticklabels(['night', 'day'])
        ax[1].set_title('individuals > 12h')
        plt.show()
        #
        # id_pos_time_min = (np.array(all_id_pos_time[i]) / 60 + self.m0[0]) % 1440
        #
        # id_x_pos_night = np.array(all_id_x_pos[i])[(id_pos_time_min < 360) | (id_pos_time_min >= 1080)]
        # id_y_pos_night = np.array(all_id_y_pos[i])[(id_pos_time_min < 360) | (id_pos_time_min >= 1080)]
        #
        # id_x_pos_day = np.array(all_id_x_pos[i])[(id_pos_time_min >= 360) & (id_pos_time_min < 1080)]
        # id_y_pos_day = np.array(all_id_y_pos[i])[(id_pos_time_min >= 360) & (id_pos_time_min < 1080)]
        #
        # # print(len(id_x_pos_night), len(id_x_pos_day))
        #
        # if len(id_x_pos_night) >= 0 and len(id_x_pos_day) >= 0:
        #     pass
        # else:
        #     continue
        #
        #
        # fig, ax = plt.subplots(1, 2, facecolor='white', figsize=(20./2.54, 12/2.54))
        # H, xedges, yedges = np.histogram2d(id_x_pos_night, id_y_pos_night, bins=(np.arange(9) - 0.5, np.arange(9) - 0.5))
        # H_turned = H.T
        # ax[0].imshow(H_turned[::-1], interpolation='gaussian', extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], vmin=0, vmax=np.max(H_turned))
        # ax[0].set_title('night')
        #
        # H2, xedges2, yedges2 = np.histogram2d(id_x_pos_day, id_y_pos_day, bins=(np.arange(9) - 0.5, np.arange(9) - 0.5))
        # H_turned2 = H2.T
        # ax[1].imshow(H_turned2[::-1], interpolation='gaussian', extent=[xedges2[0], xedges2[-1], yedges2[0], yedges2[-1]], vmin=0, vmax=np.max(H_turned2))
        # ax[1].set_title('day')
        #
        # plt.show()

        # H, xedges, yedges = np.histogram2d(all_id_x_pos[i], all_id_y_pos[i], bins=(np.arange(9) - 0.5, np.arange(9) - 0.5))
        # H_turned = H.T
        # fig, ax = plt.subplots()
        # ax.imshow(H_turned[::-1], interpolation='gaussian', extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])
        # ax.set_title('%.0f' % len(all_id_y_pos[i]))
        # plt.show()

        embed()
        quit()

        ################# plotting ###########################
        # FOR POSTER
        th = 60
        from matplotlib.ticker import StrMethodFormatter
        # Jan ###
        fig = plt.figure(facecolor='white', figsize= (20 /2.54, 12/2.54))
        ax = fig.add_axes([0.1, 0.15, 0.8, 0.7])
        ax3 = fig.add_axes([0.1, 0.85, 0.8, .1])
        # fig, ax = plt.subplots(facecolor='white', figsize= (20 /2.54, 12/2.54))
        ax2 = ax.twinx()
        # ax.get_xaxis().set_major_formatter(StrMethodFormatter('{x:.0f}'))  Todo: no idea what this was ?!
        mask = np.argsort(self.stay_m)
        sorted_stay_m = np.array(self.stay_m)[mask]
        log_bins = np.logspace(np.log10(np.min(self.stay_m)), np.log10(np.max(self.stay_m)), 50)
        n, bin_edges = np.histogram(sorted_stay_m, bins=log_bins)
        n_sure, bin_edges = np.histogram(sorted_stay_m[np.array(valid_dur)[mask]], bins=log_bins)
        bw = np.diff(bin_edges)
        bc = bin_edges[:-1] + bw/2
        ax2.semilogx(sorted_stay_m, np.arange(len(sorted_stay_m)), '.', markersize=1, color='firebrick', zorder=1)

        ax2.set_xticks([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 40, 50, 60,
                        2*60, 3*60, 4*60, 5*60, 6*60, 7*60, 8*60, 9*60, 10*60, 11*60, 12*60, 18*60, 24*60, 30*60, 36*60, 42*60, 48*60])
        ax2.set_xticklabels(['1 min', '', '', '', '', '', '', '', '', '10 min', '', '', '', '', '1 h',
                             '', '', '', '', '', '', '', '', '', '', '12 h', '', '', '', '', '', '48 h'])

        ax2.set_xlabel('time [min]', fontsize=12)
        ax2.set_ylabel('# fish', fontsize=12, color='firebrick')

        ax.bar(bc, n_sure / bw / len(self.stay_m), width = bw * 0.9, color='cornflowerblue', zorder = 2, alpha= 0.8)
        ax.bar(bc, (n - n_sure) / bw / len(self.stay_m), bottom = n_sure / bw / len(self.stay_m) ,width = bw * 0.9,
               color='cornflowerblue', zorder = 2, alpha= 0.4)

        ax.set_ylabel('probability', fontsize=12)
        ax.set_yscale('log')
        ax.minorticks_off()
        for tick in ax.get_xticklabels():
            tick.set_rotation(90)

        ax3.boxplot(self.stay_m, sym='', vert=False, widths=.8)
        ax3.set_xscale('log')
        ax3.set_ylim([0, 2])

        ax3.set_xlim(ax.get_xlim())
        ax3.set_xticks([])
        ax3.set_yticks([])
        ax3.minorticks_off()
        ax3.axis('off')

        ax.spines['top'].set_visible(False)
        ax2.spines['top'].set_visible(False)

        ax.xaxis.set_ticks_position('bottom')
        ax2.xaxis.set_ticks_position('bottom')

        ax2.spines['right'].set_color('firebrick')
        ax2.tick_params(colors='firebrick', labelsize=10)

        plt.show()

        #####################################################################
        # FOR POSTER
        fig, ax = plt.subplots(figsize=(20. / 2.54, 12. / 2.54), facecolor='white')

        for i in tqdm(range(len(self.folders))):
            ids = self.id_tag[i][:, 0][self.id_tag[i][:, 1] == 1]

            # non taged plotting #
            # mask = [id not in ids for id in self.id_v[i]]
            mask = []
            for id in self.id_v[i]:
                mask.append(id not in ids)

            non_taged_idx = np.arange(len(self.id_v[i]))[mask]
            non_taged_idx = non_taged_idx[~np.isnan(self.id_v[i][non_taged_idx])]

            ax.plot((self.times[i][self.idx_v[i][non_taged_idx]][::90] + self.shift[i] * 60) / 3600, self.fund_v[i][non_taged_idx][::90], '.', color='white', markersize=1)

            # taged plotting #
            for id in ids:
                got_entry = False
                c = np.random.rand(3)
                if len(np.shape(self.id_connect)) > 1:
                    id_idx_of_interest = np.arange(len(self.id_connect))[(np.array(self.id_connect)[:, 0] == i) & (np.array(self.id_connect)[:, 1] == id)]
                    if len(id_idx_of_interest) == 0:
                        c = np.random.rand(3)
                    else:
                        got_entry = True
                        c = self.id_connect[id_idx_of_interest[0]][4]
                ax.plot((self.times[i][self.idx_v[i][self.id_v[i] == id]][::90] + self.shift[i] * 60) / 3600,
                        self.fund_v[i][self.id_v[i] == id][::90], color=c, marker='.', markersize=1)

        for i in range(len(self.connections)):
            id_connect_idx0 = np.arange(len(self.id_connect))[(np.array(self.id_connect)[:, 0] == self.connections[i][0][0]) & (np.array(self.id_connect)[:, 1] == self.connections[i][0][1])][0]
            id_connect_idx1 = np.arange(len(self.id_connect))[(np.array(self.id_connect)[:, 0] == self.connections[i][1][0]) & (np.array(self.id_connect)[:, 1] == self.connections[i][1][1])][0]

            c = self.id_connect[id_connect_idx0][4]

            self.ax.plot([(self.times[self.id_connect[id_connect_idx0][0]][-1] + self.shift[self.id_connect[id_connect_idx0][0]] * 60) / 3600,
                          (self.times[self.id_connect[id_connect_idx1][0]][0] + self.shift[self.id_connect[id_connect_idx1][0]] * 60) / 3600],
                         [self.fund_v[self.id_connect[id_connect_idx0][0]][self.id_v[self.id_connect[id_connect_idx0][0]] == self.id_connect[id_connect_idx0][1]][-1],
                          self.fund_v[self.id_connect[id_connect_idx1][0]][self.id_v[self.id_connect[id_connect_idx1][0]] == self.id_connect[id_connect_idx1][1]][0]],
                          color=c)

        night_end = np.arange((455 + 24 * 60) / 60, (self.times[-1][-1] + self.shift[-1] * 60) / 3600, 24)

        ax.fill_between([0, 455 / 60], [400, 400], [950, 950], color='#666666')
        ax.fill_between([455 / 60, 455 / 60 + 12], [400, 400], [950, 950], color='#dddddd')
        for ne in night_end:
            ax.fill_between([ne - 12, ne], [400, 400], [950, 950], color='#666666', edgecolor=None)
            ax.fill_between([ne, ne + 12], [400, 400], [950, 950], color='#dddddd', edgecolor=None)

        ax.set_ylim([400, 950])
        ax.set_xlim([0, night_end[-1]+12])
        # plt.legend(loc=1)
        ax.set_ylabel('EOD frequency [Hz]', fontsize=12)
        # ax.set_xlabel('date', fontsize=12)


        x_ticks = ['10.04.', '11.04.', '12.04.', '13.04.', '14.04.', '15.04.', '16.04.', '17.04.', '18.04.']
        ax.set_xticks(np.arange((1440 - self.m0[0] + 12 *  60) / 60, (self.times[-1][-1] / 60 + self.shift[-1]) / 60 + 24, 24))
        ax.set_xticklabels(x_ticks, rotation = 50)
        plt.tight_layout()

        #########################################
        # ToDo: test
        relevant = np.hstack([sorted_pairs[0, 0], sorted_pairs[:, 1]])

        fig, ax = plt.subplots(figsize=(20/2.54, 12/2.54), facecolor='white')

        for i in tqdm(range(len(self.folders))):
            ids = self.id_tag[i][:, 0][self.id_tag[i][:, 1] == 1]

            # non taged plotting #
            # mask = [id not in ids for id in self.id_v[i]]
            mask = []
            for id in self.id_v[i]:
                mask.append(id not in ids)

            non_taged_idx = np.arange(len(self.id_v[i]))[mask]
            non_taged_idx = non_taged_idx[~np.isnan(self.id_v[i][non_taged_idx])]

            ax.plot((self.times[i][self.idx_v[i][non_taged_idx]][::1000] + self.shift[i] * 60) / 3600, self.fund_v[i][non_taged_idx][::1000], '.', color='white', markersize=1, rasterized=True)

        for i in tqdm(range(len(all_id_idxs))):
            if i in relevant:
                c = colors[np.arange(len(relevant))[relevant == i][0]]
            else:
                c = np.random.rand(3)
            for rec in range(len(all_id_idxs[i])):
            # for rec in range(2):
                ax.plot((self.times[rec][self.idx_v[rec][all_id_idxs[i][rec]]][::1000] + self.shift[rec] * 60) / 3600., self.fund_v[rec][all_id_idxs[i][rec]][::1000], color=c, marker='.', markersize=1, rasterized=True)

        night_end = np.arange((455 + 24 * 60) / 60, (self.times[-1][-1] + self.shift[-1] * 60) / 3600, 24)

        ax.fill_between([0, 455 / 60], [400, 400], [950, 950], color='#888888', rasterized=True)
        ax.fill_between([455 / 60, 455 / 60 + 12], [400, 400], [950, 950], color='#dddddd', rasterized=True)
        for ne in night_end:
            ax.fill_between([ne - 12, ne], [400, 400], [950, 950], color='#888888', edgecolor=None, rasterized=True)
            ax.fill_between([ne, ne + 12], [400, 400], [950, 950], color='#dddddd', edgecolor=None, rasterized=True)

        ax.set_ylim([400, 950])
        # plt.legend(loc=1)
        ax.set_ylabel('EOD frequency [Hz]', fontsize=12)
        # ax.set_xlabel('date', fontsize=12)

        x_ticks = ['10.04.', '11.04.', '12.04.', '13.04.', '14.04.', '15.04.', '16.04.', '17.04.', '18.04.']
        ax.set_xticks(np.arange((1440 - self.m0[0] + 12 *  60) / 60, (self.times[-1][-1] / 60 + self.shift[-1]) / 60 + 24, 24))
        ax.set_xticklabels(x_ticks, rotation = 50)
        # ax.set_xlim([0, lt])
        ax.tick_params(labelsize=10)
        ax.set_ylim([400, 950])
        ax.set_xlim([0, night_end[-1]+12])
        plt.tight_layout()
        # plt.show()

        ############################################

        fig, ax = plt.subplots(figsize=(20/2.54, 12/2.54), facecolor='white')
        lt = 0
        for enu, i in enumerate(np.hstack([sorted_pairs[0, 0], sorted_pairs[:, 1]])):
            c = colors[enu]
            for rec in range(len(all_id_idxs[i])):
                ax.plot((self.times[rec][self.idx_v[rec][all_id_idxs[i][rec]]][::200] + self.shift[rec] * 60) / 3600., self.fund_v[rec][all_id_idxs[i][rec]][::200], color=c, marker='.', markersize=1, rasterized=True)
                if len(self.fund_v[rec][all_id_idxs[i][rec]]) > 0:
                    if (self.times[rec][self.idx_v[rec][all_id_idxs[i][rec]]][-1] + self.shift[rec] * 60) / 3600. > lt:
                        lt = (self.times[rec][self.idx_v[rec][all_id_idxs[i][rec]]][-1] + self.shift[rec] * 60) / 3600.


        night_end = np.arange((455 + 24 * 60) / 60, (self.times[-1][-1] + self.shift[-1] * 60) / 3600, 24)

        ax.fill_between([0, 455 / 60], [400, 400], [950, 950], color='#888888', rasterized=True)
        ax.fill_between([455 / 60, 455 / 60 + 12], [400, 400], [950, 950], color='#dddddd', rasterized=True)
        for ne in night_end:
            ax.fill_between([ne - 12, ne], [400, 400], [950, 950], color='#888888', edgecolor=None, rasterized=True)
            ax.fill_between([ne, ne + 12], [400, 400], [950, 950], color='#dddddd', edgecolor=None, rasterized=True)

        ax.set_ylim([450, 950])
        ax.set_xlim([0, night_end[-1]+12])
        # plt.legend(loc=1)
        ax.set_ylabel('EOD frequency [Hz]', fontsize=12)
        # ax.set_xlabel('date', fontsize=12)

        x_ticks = ['10.04.', '11.04.', '12.04.', '13.04.', '14.04.', '15.04.', '16.04.', '17.04.', '18.04.']
        ax.set_xticks(np.arange((1440 - self.m0[0] + 12 *  60) / 60, (self.times[-1][-1] / 60 + self.shift[-1]) / 60 + 24, 24))
        ax.set_xticklabels(x_ticks, rotation = 50)
        ax.set_xlim([0, lt])
        ax.tick_params(labelsize=10)
        plt.tight_layout()
        plt.show()
            # for rec in range(len(all_id_idxs[i])):
            #     pass




def main():

    folders = ['/home/raab/data/colombia/2016-04-09-22_25',
               '/home/raab/data/colombia/2016-04-10-11_12',
               '/home/raab/data/colombia/2016-04-10-22:14/error-22:14',
               '/home/raab/data/colombia/2016-04-11-09:56',
               '/home/raab/data/colombia/2016-04-11-22:11',
               '/home/raab/data/colombia/2016-04-12-19_05',
               '/home/raab/data/colombia/2016-04-13-07_08',
               '/home/raab/data/colombia/2016-04-13-20_08',
               '/home/raab/data/colombia/2016-04-14-19_12',
               '/home/raab/data/colombia/2016-04-15-07_58',
               '/home/raab/data/colombia/2016-04-15-20_02',
               '/home/raab/data/colombia/2016-04-16-08_19',
               '/home/raab/data/colombia/2016-04-16-18_45',
               '/home/raab/data/colombia/2016-04-17-08_06',
               '/home/raab/data/colombia/2016-04-17-19_04']

    # folders = ['/home/raab/data/colombia/2016-04-09-22_25',
    #            '/home/raab/data/colombia/2016-04-10-11_12',
    #            '/home/raab/data/colombia/2016-04-11-09:56',
    #            '/home/raab/data/colombia/2016-04-11-22:11',
    #            '/home/raab/data/colombia/2016-04-12-19_05',
    #            '/home/raab/data/colombia/2016-04-13-07_08',
    #            '/home/raab/data/colombia/2016-04-13-20_08',
    #            '/home/raab/data/colombia/2016-04-14-19_12',
    #            '/home/raab/data/colombia/2016-04-15-07_58',
    #            '/home/raab/data/colombia/2016-04-15-20_02',
    #            '/home/raab/data/colombia/2016-04-16-08_19',
    #            '/home/raab/data/colombia/2016-04-16-18_45',
    #            '/home/raab/data/colombia/2016-04-17-08_06',
    #            '/home/raab/data/colombia/2016-04-17-19_04']



    shift = [0, 767, 1429, 2131, 2866, 4120, 4843, 5623, 7007, 7773, 8497, 9234, 9860, 10661, 11319]
    #
    # shift = [0, 767, 2131, 2866, 4120, 4843, 5623, 7007, 7773, 8497, 9234, 9860, 10661, 11319]


    if len(sys.argv) >= 2:
        folders = folders[:3]
        shift = shift[:3]

    Traces(folders, shift)


if __name__ == '__main__':
    main()