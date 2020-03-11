import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from tqdm import tqdm
from IPython import embed
import matplotlib.animation as manimation
import glob

def load_data(folder):
    fund_v = np.load(os.path.join(folder, 'fund_v.npy'))
    sign_v = np.load(os.path.join(folder, 'sign_v.npy'))
    # sign_v = np.sqrt(np.exp(sign_v))
    idx_v = np.load(os.path.join(folder, 'idx_v.npy'))
    ident_v = np.load(os.path.join(folder, 'ident_v.npy'))
    times = np.load(os.path.join(folder, 'times.npy'))
    tmp_spectra = np.load(os.path.join(folder, 'spec.npy'))
    start_time, end_time = np.load(os.path.join(folder, 'meta.npy'))
    return fund_v, sign_v, idx_v, ident_v, times, tmp_spectra, start_time, end_time

def grid_props():
    xy = []
    neighbours = []
    n_tolerance_e = 1

    for i in range(64):
        xy.append((i // 8, i % 8))
    xy = np.array(xy)

    for x, y in xy:
        neighbor_coords = []
        for i in np.arange(-n_tolerance_e, n_tolerance_e + 1):
            for j in np.arange(-n_tolerance_e, n_tolerance_e + 1):
                if i == 0 and j == 0:
                    continue
                else:
                    neighbor_coords.append([x + i, y + j])

        for k in reversed(range(len(neighbor_coords))):
            if all((i >= 0) & (i <= 7) for i in neighbor_coords[k]):
                continue
            else:
                neighbor_coords.pop(k)
        neighbours.append(np.array([n[0] * 8 + n[1] for n in neighbor_coords]))

    return xy, neighbours

def get_pos(sign_v, xy, neighbours):
    x_v = np.zeros(len(sign_v))
    y_v = np.zeros(len(sign_v))

    for i in tqdm(range(len(sign_v)), desc='get coords'):
    # for i in tqdm(range(10000), desc='get coords'):
        max_p_electrode = np.argmax(sign_v[i])
        eoi = np.append(neighbours[max_p_electrode], max_p_electrode) # electrodes of interest
        sp_sign = np.sqrt(0.1 * 10.**sign_v[i])

        rel_power = sp_sign[eoi] / np.sum(sp_sign[eoi])


        # embed()
        # quit()
        #
        # base_zero_sig = sign_v[i] - np.min(sign_v[i])
        # rel_power = base_zero_sig[eoi] / abs(np.sum(base_zero_sig[eoi]))
        # rel_power = sign_v[i][eoi] / np.sum(sign_v[eoi])

        x_v[i] = np.sum(xy[eoi][:, 0] * rel_power)
        y_v[i] = np.sum(xy[eoi][:, 1] * rel_power)

    return x_v, y_v

def fill_n_smooth_positions(fund_v, ident_v, idx_v, times, x_v, y_v):
    idents = np.unique(ident_v[~np.isnan(ident_v)])

    x_ident = np.full((len(idents), len(times)), np.nan)
    y_ident = np.full((len(idents), len(times)), np.nan)

    for enu in tqdm(range(len(idents)), desc = 'reshape position data'):
        # embed()
        x_ident[enu][idx_v[ident_v == idents[enu]]] = x_v[ident_v == idents[enu]]
        y_ident[enu][idx_v[ident_v == idents[enu]]] = y_v[ident_v == idents[enu]]

        x = idx_v[ident_v == idents[enu]]
        if len(x) <= 10:
            continue

        kernel = np.ones(3) / 3.
        x_ident[enu][~np.isnan(x_ident[enu])] = np.convolve(x_ident[enu][~np.isnan(x_ident[enu])], kernel, mode='same')
        y_ident[enu][~np.isnan(y_ident[enu])] = np.convolve(y_ident[enu][~np.isnan(y_ident[enu])], kernel, mode='same')

        # x = np.arange(len(x_ident[enu]))[~np.isnan(x_ident[enu])]

        y = x_ident[enu][~np.isnan(x_ident[enu])]
        x_ident[enu][x[0]: x[-1]+1] = np.interp(np.arange(x[0], x[-1] + 1), x, y)

        y = y_ident[enu][~np.isnan(y_ident[enu])]
        y_ident[enu][x[0]: x[-1]+1] = np.interp(np.arange(x[0], x[-1] + 1), x, y)


    return idents, x_ident, y_ident

def create_movie(ident_v, idents, idx_v, times, x_ident, y_ident, fund_v, start_time, end_time, video_start, video_end, video_name, folder, test = False):
    # embed()
    # quit()

    time_str = os.path.split(folder)[0][-5:].replace('_', '').replace(':', '')
    h = int(time_str[0:2])
    m = int(time_str[2:])
    start_m = m + 60 * h

    if video_end < 0:
        video_end = times[-1]
    l0 = None
    l1 = None
    l2 = None

    FFMpegWriter = manimation.writers['ffmpeg']
    metadata = dict(title='Movie Test', artist='Matplotlib', comment='Movie support!')
    writer = FFMpegWriter(fps=9, metadata=metadata)

    fig = plt.figure(figsize=(40/2.54, 24/2.54), facecolor='white')
    ax1 = fig.add_axes([.05, .66, .4, .29])
    ax2 = fig.add_axes([.05, .36, .4, .29])
    ax2_1 = fig.add_axes([.05, .05, .4, .29])
    ax1.set_xticks([])
    ax2.set_xticks([])
    ax3 = fig.add_axes([.5, .15, .45, .7])

    ax1.set_ylabel('frequency [Hz]')
    ax2.set_ylabel('frequency [Hz]')
    ax2_1.set_ylabel('frequency [Hz]')
    ax2_1.set_xlabel('time [s]')
    ax3.set_xlabel('x-position [cm]')
    ax3.set_ylabel('y-position [cm]')
    ax3.set_xticks([0, 1, 2, 3, 4, 5, 6, 7])
    ax3.set_xticklabels([0, 50, 100, 150, 200, 250, 300, 350])

    ax3.set_yticks([0, 1, 2, 3, 4, 5, 6, 7])
    ax3.set_yticklabels([0, 50, 100, 150, 200, 250, 300, 350])

    # plt.show()
    # embed()
    # quit()
    ax3.set_xlim([-0.1, 7.1])
    ax3.set_ylim([-0.1, 7.1])

    ax1.set_ylim([725, 1055])
    ax2.set_ylim([645, 735])
    ax2_1.set_ylim([400, 655])
    if test:
        plt.show(block=False)
        # clock = fig.text(0.695, 0.91, '10:11', fontsize=24)
        # plt.show()

    meta = []
    for ident in idents:
        meta.append([ident, None, [None, None, None, None, None, None], None, None, None, None]) # [identity, color, marker_handles, trace_handle, text_handle, path, axis]

    frames = np.arange(len(times))[(times >= video_start) & (times <= video_end)]
    # frames = len(times[times < 60 * 60])
    # embed()
    # quit()
    if test:
        mov_name = 'test.mp4'
    else:
        mov_name = video_name + '.mp4'
    with writer.saving(fig, "/home/raab/Desktop/" + mov_name, 300):

        for i in tqdm(frames, desc='create images'):

            c_time = np.floor(start_m + times[i] / 60)
            c_sec = times[i] % (times[i] // 60 * 60)

            # print(c_sec)
            c_time_str = '%2.f:%2.f:%2.f' % ((c_time // 60 % 24), c_time % 60, c_sec)
            # embed()
            # quit()
            # c_time_str = '%2.f:%2.f' % ((c_time // 60 % 24), c_time % 60)
            c_time_str = c_time_str.replace(' ', '0')
            if i == frames[0]:
                clock = fig.text(0.695, 0.91, c_time_str, fontsize=24)
            else:
                clock.set_text(c_time_str)

            for j in range(len(idents)):

                # embed()
                # quit()
                if not meta[j][2][0] == None: # marker
                    meta[j][2][0].remove()
                meta[j][2][0] == None

                meta[j][2] = np.roll(meta[j][2], -1)

                if not meta[j][2][1] == None:
                    meta[j][2][1].set_alpha(0.1)

                if not meta[j][2][2] == None:
                    meta[j][2][2].set_alpha(0.3)

                if not meta[j][2][3] == None:
                    meta[j][2][3].set_alpha(0.5)

                if not meta[j][2][4] == None:
                    meta[j][2][4].set_alpha(0.7)

                if not meta[j][2][5] == None:
                    meta[j][2][5].set_alpha(0.9)

                f = fund_v[ident_v == idents[j]]
                t = times[idx_v[ident_v == idents[j]]]

                if meta[j][1] == None: # color
                    meta[j][1] = np.random.rand(3)

                #############################################################################
                if ~np.isnan(x_ident[j][i]): # if no position
                    if len(fund_v[(ident_v == idents[j]) & (idx_v == i)]) >= 1: # if got frequency
                        if not meta[j][4] == None: # if text present
                            meta[j][4].remove()
                            meta[j][4] = None
                        meta[j][4] = ax3.text(x_ident[j][i], y_ident[j][i], '%.1f' %(fund_v[(ident_v == idents[j]) & (idx_v == i)][0]))
                    else: # if no frequency
                        if meta[j][4] == None: # if no text
                            txt = '%.1f' % fund_v[(ident_v == idents[j]) & (idx_v <= i)][0]
                        else: # if text
                            txt = meta[j][4].get_text()

                        # if not meta[j][4] == None: # if text available
                            meta[j][4].remove()
                            meta[j][4] = None

                        meta[j][4] = ax3.text(x_ident[j][i], y_ident[j][i], txt)

                # if np.all(meta[j][2] == None):
                if np.all(x is None for x in meta[j][2]):
                    meta[j][4].remove()
                    meta[j][4] = None
                #############################################################################

                if len(fund_v[(ident_v == idents[j]) & (idx_v <= i)]) >= 1 and len(f[(t > times[i] - 60) & (t < times[i] + 60)]) >=1:
                    f_to_base = fund_v[(ident_v == idents[j]) & (idx_v <= i)][-1] - np.min(f[(t > times[i] - 60) & (t < times[i] + 60)])
                    marker_enlarge = (f_to_base / 20.) * 22 if f_to_base < 20 else 22

                else:
                    marker_enlarge = 0

                if meta[j][6] == None:
                    if f[0] >= 730:
                        meta[j][2][-1], = ax3.plot(x_ident[j][i], y_ident[j][i], marker='s', color=meta[j][1], markersize=8 + marker_enlarge) # marker

                        if meta[j][3] == None: # trace
                            meta[j][3], = ax1.plot(t[(t > times[i] -60) & (t < times[i] +60)], f[(t > times[i] -60) & (t < times[i] +60)], color=meta[j][1], lw=2, marker='.')
                        else:
                            meta[j][3].set_data(t[(t > times[i] -60) & (t < times[i] +60)], f[(t > times[i] -60) & (t < times[i] +60)])

                        meta[j][6] = 0 # subplot Nr.

                    elif f[0] < 730 and f[0] > 650:
                        if meta[j][3] == None: # marker
                            meta[j][3], = ax2.plot(t[(t > times[i] - 60) & (t < times[i] + 60)], f[(t > times[i] - 60) & (t < times[i] + 60)], color=meta[j][1], lw=2, marker='.')
                        else:
                            meta[j][3].set_data( t[(t > times[i] - 60) & (t < times[i] + 60)], f[(t > times[i] - 60) & (t < times[i] + 60)] )

                        meta[j][2][-1], = ax3.plot(x_ident[j][i], y_ident[j][i], marker='o', color=meta[j][1], markersize=8+ marker_enlarge)
                        meta[j][6] = 1 # subplot Nr.

                    else:
                        if meta[j][3] == None:
                            meta[j][3], = ax2_1.plot(t[(t > times[i] -60) & (t < times[i] +60)], f[(t > times[i] -60) & (t < times[i] +60)], color=meta[j][1], lw=2, marker='.')
                        else:
                            meta[j][3].set_data(t[(t > times[i] -60) & (t < times[i] +60)], f[(t > times[i] -60) & (t < times[i] +60)])
                        meta[j][2][-1], = ax3.plot(x_ident[j][i], y_ident[j][i], marker='o', color=meta[j][1], markersize=8+ marker_enlarge)
                        meta[j][6] = 2
                else:
                    if meta[j][6] == 0:
                        meta[j][2][-1], = ax3.plot(x_ident[j][i], y_ident[j][i], marker='s', color=meta[j][1],
                                                   markersize=8+ marker_enlarge)
                        if meta[j][3] == None:
                            meta[j][3], = ax1.plot(t[(t > times[i] - 60) & (t < times[i] + 60)],
                                                   f[(t > times[i] - 60) & (t < times[i] + 60)], color=meta[j][1], lw=2,
                                                   marker='.')
                        else:
                            meta[j][3].set_data(t[(t > times[i] - 60) & (t < times[i] + 60)],
                                                f[(t > times[i] - 60) & (t < times[i] + 60)])

                    elif meta[j][6] == 1:
                        if meta[j][3] == None:
                            meta[j][3], = ax2.plot(t[(t > times[i] - 60) & (t < times[i] + 60)],
                                                   f[(t > times[i] - 60) & (t < times[i] + 60)], color=meta[j][1], lw=2,
                                                   marker='.')
                        else:
                            meta[j][3].set_data(t[(t > times[i] - 60) & (t < times[i] + 60)],
                                                f[(t > times[i] - 60) & (t < times[i] + 60)])

                        meta[j][2][-1], = ax3.plot(x_ident[j][i], y_ident[j][i], marker='o', color=meta[j][1],
                                                   markersize=8+ marker_enlarge)

                    else:
                        if meta[j][3] == None:
                            meta[j][3], = ax2_1.plot(t[(t > times[i] - 60) & (t < times[i] + 60)],
                                                     f[(t > times[i] - 60) & (t < times[i] + 60)], color=meta[j][1],
                                                     lw=2, marker='.')
                        else:
                            meta[j][3].set_data(t[(t > times[i] - 60) & (t < times[i] + 60)],
                                                f[(t > times[i] - 60) & (t < times[i] + 60)])
                        meta[j][2][-1], = ax3.plot(x_ident[j][i], y_ident[j][i], marker='o', color=meta[j][1],
                                                   markersize=8+ marker_enlarge)

                datas = np.array([h.get_data() for h in meta[j][2] if not h == None])

                if not meta[j][5] == None: # path line
                    meta[j][5].remove()
                meta[j][5], = ax3.plot(np.reshape(datas[:, 0], len(datas[:, 0])), np.reshape(datas[:, 1], len(datas[:, 1])), lw = 2, alpha= 0.5, color =meta[j][1])


            if i % 10 == 0 or i == frames[0]:
                # print('yay')
                for enu, axx in enumerate([ax1, ax2, ax2_1]):
                    current_funds = []
                    for j in range(len(meta)):
                        if meta[j][6] == enu:
                            current_funds.append(meta[j][3].get_data()[1])
                    current_funds = np.hstack(current_funds)

                    axx.set_ylim([np.min(current_funds)-5, np.max(current_funds)+5])


                # current_indices = np.arange(len(times))[(times >= times[i] - 30) & (times <= times[i] + 30)]
                # current_fund = fund_v[current_indices][(~np.isnan(fund_v[current_indices])) & (~np.isnan(ident_v[current_indices]))]
                #
                # f1 = current_fund[(current_fund >= 730) & (current_fund < 1050)]
                # f2 = current_fund[(current_fund < 730) & (current_fund > 650)]
                # f2_1 = current_fund[(current_fund <= 650) & (current_fund > 400)]
                #
                # ax1.set_ylim([np.min(f1)-1, np.max(f1)+1])
                # ax2.set_ylim([np.min(f2)-1, np.max(f2)+1])
                # ax2_1.set_ylim([np.min(f2_1)-1, np.max(f2_1)+1])
                #
                # ax1.set_ylim([np.min(f1) - 5, np.max(f1) + 5])
                # ax2.set_ylim([np.min(f2) - 5, np.max(f2) + 5])
                # ax2_1.set_ylim([np.min(f2_1) - 5, np.max(f2_1) + 5])


            ax1.set_xlim([times[i]-30, times[i]+30])
            if not l0 == None:
                l0.remove()
            l0, = ax1.plot([times[i], times[i]], ax1.get_ylim(), color='red')
            ax2.set_xlim([times[i]-30, times[i]+30])
            if not l1 == None:
                l1.remove()
            l1, = ax2.plot([times[i], times[i]], ax2.get_ylim(), color='red')

            ax2_1.set_xlim([times[i] - 30, times[i] + 30])
            if not l2 == None:
                l2.remove()
            l2, = ax2_1.plot([times[i], times[i]], ax2_1.get_ylim(), color='red')

            fig.canvas.draw()
            if not test:
                writer.grab_frame()

            if i - frames[0] >= len(frames):
                break


def main():
    folder = sys.argv[1]
    video_start = 0
    video_end = -1
    video_name = 'random'

    if len(sys.argv) >= 3:
        video_start = float(sys.argv[2]) * 60
    if len(sys.argv) >= 4:
        video_end = float(sys.argv[3]) * 60
    if len(sys.argv) >= 5:
        video_name = sys.argv[4]


    fund_v, sign_v, idx_v, ident_v, times, tmp_spectra, start_time, end_time = load_data(folder)

    xy, neighbours = grid_props()

    x_v, y_v = get_pos(sign_v, xy, neighbours)

    idents, x_ident, y_ident = fill_n_smooth_positions(fund_v, ident_v, idx_v, times, x_v, y_v)

    create_movie(ident_v, idents, idx_v, times, x_ident, y_ident, fund_v, start_time, end_time, video_start, video_end, video_name, folder)



if __name__ == '__main__':
    main()