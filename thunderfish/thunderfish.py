import numpy as np
import argparse
import config_tools as ct
import dataloader as dl
import bestwindow as bw
import checkpulse as chp
import powerspectrum as ps
import harmonicgroups as hg
import consistentfishes as cf
import eodanalysis as ea
import matplotlib.pyplot as plt


def output_plot(axs, audio_file, pulse_fish_width, pulse_fish_psd, EOD_count, mean_IPI, inter_eod_intervals, std_IPI,
                raw_data, samplerate, idx0, idx1, filtered_fishlist, period):
    """
    Creates an output plot for the Thunderfish program.

    This output contains the raw trace where the analysis window is marked, the power-spectrum of this analysis window
    where the detected fish are marked, a mean EOD plot, a histogram of the inter EOD interval and further information
    about the recording that is analysed.

    :param axs: (list) list of axis fo plots.
    :param audio_file: (string) path to and name of audiofile.
    :param pulse_fish_width: (bool) True if a pulsefish has been detected by analysis of the EODs.
    :param pulse_fish_psd: (bool) True if a pulsefish has been detected by analysis of the PSD.
    :param EOD_count: (int) number of detected EODs.
    :param mean_IPI: (float) mean inter EOD interval.
    :param inter_eod_intervals: (array) time difference from one to another detected EOD.
    :param std_IPI: (float) standard deviation of the inter EOD interval.
    :param raw_data: (array) dataset.
    :param samplerate: (float) samplerate of the dataset.
    :param idx0: (float) index of the beginning of the analysis window in the dataset.
    :param idx1: (float) index of the end of the analysis window in the dataset.
    :param filtered_fishlist: (array) frequency and power of fundamental frequency/harmonics of several fish.
    :param period: (float) mean EOD time difference.
    """
    # ToDo: Transfer everything needed for every plot into this function !!!
    # plot inter EOD interval histogram
    n, edges = np.histogram(inter_eod_intervals, bins=100)

    axs[3].bar(edges[:-1], n, edges[1]-edges[0])
    axs[3].plot([mean_IPI, mean_IPI], [0, max(n)], '--', color= 'red', lw=2, label='mean')
    axs[3].plot([mean_IPI - std_IPI, mean_IPI - std_IPI], [0, max(n)], '--', color= 'green', lw=2, label='std')
    axs[3].plot([mean_IPI + std_IPI, mean_IPI + std_IPI], [0, max(n)], '--', color= 'green', lw=2)
    axs[3].set_xlabel('inter EOD interval [ms]')
    axs[3].set_ylabel('n')
    axs[3].legend(loc= 'upper right', frameon=False)

    # plot trace
    time = np.arange(len(raw_data)) / samplerate

    axs[2].plot(time[:idx0], raw_data[:idx0], color='blue')
    axs[2].plot(time[idx1:], raw_data[idx1:], color='blue')
    axs[2].plot(time[idx0:idx1], raw_data[idx0:idx1], color='red', label='analysis window')
    axs[2].set_xlabel('Time [sec]')
    axs[2].set_ylabel('Amplitude [a.u.]')
    axs[2].legend(loc='upper right', frameon=False)

    try:
        dom_freq = filtered_fishlist[np.argsort([filtered_fishlist[fish][0][1] for fish in range(len(filtered_fishlist))])[-1]][0][0]
        fish_count = len(filtered_fishlist)
    except IndexError:
        dom_freq = 1./ period
        fish_count = 1

    filename = audio_file.split('/')[-1]
    ax5 = plt.subplot2grid((5, 6), (0, 0), colspan=6)
    ax5.text(0.5, 0.5, 'Thunderfish: %s' % filename, fontsize= 30, horizontalalignment='center')
    ax5.set_frame_on(False)
    ax5.get_xaxis().set_visible(False)
    ax5.get_yaxis().set_visible(False)
    for ax in axs:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.get_xaxis().tick_bottom()
        ax.get_yaxis().tick_left()

    ax6 = plt.subplot2grid((5, 6), (3, 2), colspan=2, rowspan=2)
    ax6.set_frame_on(False)
    ax6.get_xaxis().set_visible(False)
    ax6.get_yaxis().set_visible(False)

    fishtype = 'pulse' if pulse_fish_width and pulse_fish_psd else 'wave'
    ax6.text(0.1, 0.9, 'fishtype:', fontsize=14)
    ax6.text(0.6, 0.9, '%s' %fishtype, fontsize=14)

    ax6.text(0.1, 0.7, '# detected fish:', fontsize=14)
    ax6.text(0.6, 0.7, '%.0f' % fish_count, fontsize=14)

    if fishtype is 'wave':
        ax6.text(0.1, 0.5, 'dominant frequency:', fontsize=14)
        ax6.text(0.6, 0.5, '%.1f Hz' % dom_freq, fontsize=14)
    else:
        ax6.text(0.1, 0.5, 'Mean pulse frequency:', fontsize=14)
        ax6.text(0.6, 0.5, '%.1f Hz' % dom_freq, fontsize=14)

    ax6.text(0.1, 0.3, '# detected EODs:', fontsize=14)
    ax6.text(0.6, 0.3, '%.0f' %EOD_count, fontsize=14)

    ax6.text(0.1, 0.1, 'mean EOD interval:', fontsize=14)
    ax6.text(0.6, 0.1, '%.2f ms' %mean_IPI, fontsize=14)

    plt.tight_layout()

    # TODO: plot result in pdf!
    plt.show()


def main(audio_file, channel=0, output_folder='', beat_plot=False, verbose=0):
    fig = plt.figure(facecolor='white', figsize=(18., 10.))
    ax1 = plt.subplot2grid((5, 6), (1, 3), colspan=3, rowspan=2)
    ax2 = plt.subplot2grid((5, 6), (3, 0), colspan=2, rowspan=2)
    ax3 = plt.subplot2grid((5, 6), (1, 0), colspan=3, rowspan=2)
    ax4 = plt.subplot2grid((5, 6), (3, 4), colspan=2, rowspan=2)

    # get config dictionary
    cfg = ct.get_config_dict()

    # load data:
    raw_data, samplerate, unit = dl.load_data(audio_file, channel)
    if len(raw_data) == 0:
        return

    # calculate best_window:
    clip_win_size = 0.5
    min_clip, max_clip = bw.clip_amplitudes(raw_data, int(clip_win_size * samplerate))
    idx0, idx1, clipped = bw.best_window_indices(raw_data, samplerate, single=True, win_size=4.0, min_clip=min_clip,
                                                 max_clip=max_clip)
    data = raw_data[idx0:idx1]

    # pulse-type fish?
    pulse_fish_width, pta_value = chp.check_pulse_width(data, samplerate)

    # calculate powerspectrums with different frequency resolutions
    psd_data = ps.multi_resolution_psd(data, samplerate, fresolution=[0.5, 2 * 0.5, 4 * 0.5])
    ps.plot_decibel_psd(psd_data[0][0], psd_data[0][1], ax1, fs=12)

    # find the fishes in the different powerspectrums:
    fishlists = []
    for i in range(len(psd_data)):
        fishlist = hg.harmonic_groups(psd_data[i][1], psd_data[i][0], cfg)[0]
        fishlists.append(fishlist)

    # find the psd_type
    pulse_fish_psd, proportion = chp.check_pulse_psd(psd_data[0][0], psd_data[0][1])

    # filter the different fishlists to get a fishlist with consistent fishes:
    if not pulse_fish_width and not pulse_fish_psd:
        filtered_fishlist = cf.consistent_fishes(fishlists)
        cf.consistent_fishes_psd_plot(filtered_fishlist, ax=ax1)
    else:
        filtered_fishlist = []

    # analyse eod waveform:
    mean_eod, std_eod, time, eod_times = ea.eod_waveform(data, samplerate)
    period = np.mean(np.diff(eod_times))

    # inter-peal interval
    inter_peak_intervals = np.diff(eod_times)* 1000. # in ms
    mean_IPI = np.mean(inter_peak_intervals)
    std_IPI = np.std(inter_peak_intervals, ddof=1)

    ### PLOTTING ###
    # plot waveform:
    ea.eod_waveform_plot(time, mean_eod, std_eod, ax2, unit=unit)
    ax2.set_xlim([-500*period, 500*period])  # half a period in milliseconds
    # TODO: make xlim dependent on fish type!

    output_plot([ax1, ax2, ax3, ax4], audio_file, pulse_fish_width, pulse_fish_psd, len(eod_times), mean_IPI,
                inter_peak_intervals, std_IPI, raw_data, samplerate, idx0, idx1, filtered_fishlist, period)


if __name__ == '__main__':
    # command line arguments:
    parser = argparse.ArgumentParser(
        description='Analyse short EOD recordings of weakly electric fish.',
        epilog='by bendalab (2015-2016)')
    parser.add_argument('--version', action='version', version='1.0')
    parser.add_argument('-v', action='count', dest='verbose')
    parser.add_argument('file', nargs='?', default='', type=str, help='name of the file wih the time series data')
    parser.add_argument('channel', nargs='?', default=0, type=int, help='channel to be displayed')
    parser.add_argument('output_folder', nargs='?', default=".", type=str, help="location to store results, figures")
    args = parser.parse_args()

    main(args.file, args.channel, args.output_folder, verbose=args.verbose)
