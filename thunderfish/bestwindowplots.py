import bestwindow as bw


def plot_clipping(data, winx0, winx1, bins,
                  h, min_clip, max_clip, min_ampl, max_ampl):
    plt.subplot(2, 1, 1)
    plt.plot(data[winx0:winx1], 'b')
    plt.axhline(min_clip, color='r')
    plt.axhline(max_clip, color='r')
    plt.ylim(-1.0, 1.0)
    plt.subplot(2, 1, 2)
    plt.bar(bins[:-1], h, width=np.mean(np.diff(bins)))
    plt.axvline(min_clip, color='r')
    plt.axvline(max_clip, color='r')
    plt.xlim(-1.0, 1.0)
    plt.show()


def plot_best_window(data, rate, peak_idx, trough_idx, idx0, idx1,
                     win_times, cv_interv, mean_ampl, cv_ampl, clipped_frac,
                     cost, thresh, win_idx0, win_idx1, ax, fs=10):
    # raw data:
    time = np.arange(0.0, len(data)) / rate
    ax[0].plot(time, data, color='royalblue', lw=3)
    if np.mean(clipped_frac[win_idx0:win_idx1]) > 0.01:
        ax[0].plot(time[idx0:idx1], data[idx0:idx1], color='magenta', lw=3)
    else:
        ax[0].plot(time[idx0:idx1], data[idx0:idx1], color='red', lw=3)
    ax[0].plot(time[peak_idx], data[peak_idx], 'o', mfc='crimson', mec='crimson', mew=2., ms=6)
    ax[0].plot(time[trough_idx], data[trough_idx], 'o', mfc='lime', mec='lime', mew=2., ms=6)
    up_lim = np.max(data) * 1.05
    down_lim = np.min(data) * .95
    ax[0].set_ylim((down_lim, up_lim))
    ax[0].set_ylabel('Amplitude [a.u]', fontsize=fs)

    # cv of inter-peak intervals:
    ax[1].plot(win_times[cv_interv < 1000.0], cv_interv[cv_interv < 1000.0], 'o', ms=10, color='grey', mew=2.,
               mec='black', alpha=0.6)
    ax[1].plot(win_times[win_idx0:win_idx1], cv_interv[win_idx0:win_idx1], 'o', ms=10, color='red', mew=2., mec='black',
               alpha=0.6)
    ax[1].set_ylabel('CV intervals', fontsize=fs)
    ax[1].set_ylim(bottom=0.0)

    # mean amplitude:
    ax[2].plot(win_times[mean_ampl > 0.0], mean_ampl[mean_ampl > 0.0], 'o', ms=10, color='grey', mew=2., mec='black',
               alpha=0.6)
    ax[2].plot(win_times[win_idx0:win_idx1], mean_ampl[win_idx0:win_idx1], 'o', ms=10, color='red', mew=2., mec='black',
               alpha=0.6)
    ax[2].set_ylabel('Mean amplitude [a.u]', fontsize=fs)
    ax[2].set_ylim(bottom=0.0)

    # cv:
    ax[3].plot(win_times[cv_ampl < 1000.0], cv_ampl[cv_ampl < 1000.0], 'o', ms=10, color='grey', mew=2., mec='black',
               alpha=0.6)
    ax[3].plot(win_times[win_idx0:win_idx1], cv_ampl[win_idx0:win_idx1], 'o', ms=10, color='red', mew=2., mec='black',
               alpha=0.6)
    ax[3].set_ylabel('CV amplitude', fontsize=fs)
    ax[3].set_ylim(bottom=0.0)

    # cost:
    ax[4].plot(win_times[cost < thresh + 10], cost[cost < thresh + 10], 'o', ms=10, color='grey', mew=2., mec='black',
               alpha=0.6)
    ax[4].plot(win_times[win_idx0:win_idx1], cost[win_idx0:win_idx1], 'o', ms=10, color='red', mew=2., mec='black',
               alpha=0.6)
    ax[4].axhline(thresh, color='k')
    ax[4].set_ylabel('Cost', fontsize=fs)
    ax[4].set_xlabel('Time [sec]', fontsize=fs)

    ##     windows = np.arange(len(peak_rate)) * win_shift
    ##     up_th = np.ones(len(windows)) * pk_mode[0][0] + tot_pks*rate_th
    ##     down_th = np.ones(len(windows)) * pk_mode[0][0] - tot_pks*rate_th
    ##     axs[1].fill_between(windows, y1=down_th, y2=up_th, color='forestgreen', alpha=0.4, edgecolor='k', lw=1)

    ##     cvs_th_array = np.ones(len(windows)) * cv_th
    ##     axs[2].fill_between(windows, y1=np.zeros(len(windows)), y2=cvs_th_array, color='forestgreen',
    ##                         alpha=0.4, edgecolor='k', lw=1)

    ##     clipping_lim = np.ones(len(windows)) * axs[3].get_ylim()[-1]
    ##     clipping_th = np.ones(len(windows))*ampls_th
    ##     axs[3].fill_between(windows, y1=clipping_th, y2=clipping_lim,
    ##                         color='tomato', alpha=0.6, edgecolor='k', lw=1)
    ##     axs[3].plot(windows[best_window], max_ampl_window[best_window], 'o', ms=25, mec='black', mew=3,
    ##                 color='purple', alpha=0.8)


if __name__ == "__main__":
    print("Checking bestwindowplot module ...")
    import sys
    import numpy as np
    import matplotlib.pyplot as plt

    title = "bestwindow"
    if len(sys.argv) < 2:
        # generate data:
        print("generate waveform...")
        rate = 100000.0
        time = np.arange(0.0, 1.0, 1.0 / rate)
        f = 600.0

        if True:  # ToDo: If True what??
            snippets = []
            amf = 20.0
            for ampl in [0.2, 0.5, 0.8]:
                for am_ampl in [0.0, 0.3, 0.9]:
                    data = ampl * np.sin(2.0 * np.pi * f * time) * (1.0 + am_ampl * np.sin(2.0 * np.pi * amf * time))
                    data[data > 1.3] = 1.3
                    data[data < -1.3] = -1.3
                    snippets.extend(data)
            data = np.asarray(snippets)
            title += " test sines"
        else:
            data = 0.1 * np.sin(2.0 * np.pi * f * time)
            title += " sine"
        data += 0.01 * np.random.randn(len(data))
    else:
        import dataloader as dl

        print("load %s ..." % sys.argv[1])
        data, rate, unit = dl.load_data(sys.argv[1], 0)
        title += " " + sys.argv[1]

    # determine clipping amplitudes:
    clip_win_size = 0.5
    min_clip_fac = 2.0
    min_clip, max_clip = bw.clip_amplitudes(data, int(clip_win_size * rate),
                                            min_fac=min_clip_fac)
    # min_clip, max_clip = bw.clip_amplitudes(data, int(clip_win_size*rate),
    #                                        min_fac=min_clip_fac,
    #                                        plot_hist_func=plot_clipping)

    # setup plots:
    fig, ax = plt.subplots(5, sharex=True, figsize=(20, 12))
    fig.canvas.set_window_title(title)

    # compute best window:
    print("call bestwindow() function...")
    bw.best_window_indices(data, rate, single=False,
                           win_size=1.0, win_shift=0.1,
                           min_clip=min_clip, max_clip=max_clip,
                           w_cv_ampl=10.0, tolerance=0.5,
                           plot_data_func=plot_best_window, ax=ax, fs=12)

    plt.tight_layout()
    plt.show()
