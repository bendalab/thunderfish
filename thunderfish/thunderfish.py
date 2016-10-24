"""
thunderfish

some words of documentation...

Run it from the thunderfish development directory as:
python3 -m thunderfish.thunderfish audiofile.wav   or
python -m thunderfish.thunderfish audiofile.wav
"""

import numpy as np
import argparse
import os
import matplotlib.pyplot as plt
from .configfile import ConfigFile
from .harmonicgroups import add_psd_peak_detection_config, add_harmonic_groups_config
from .bestwindow import add_clip_config, add_best_window_config
from .dataloader import load_data
from .bestwindow import clip_amplitudes, best_window_indices
from .checkpulse import check_pulse_width, check_pulse_psd
from .powerspectrum import plot_decibel_psd, multi_resolution_psd
from .harmonicgroups import harmonic_groups, harmonic_groups_args, psd_peak_detection_args, fundamental_freqs_and_db
from .consistentfishes import consistent_fishes_psd_plot, consistent_fishes
from .eodanalysis import eod_waveform_plot, eod_waveform
from .csvmaker import write_csv


def output_plot(base_name, pulse_fish_width, pulse_fish_psd, EOD_count, median_IPI, inter_eod_intervals,
                raw_data, samplerate, idx0, idx1, filtered_fishlist, period, time_eod, mean_eod, std_eod, unit,
                psd_data, output_folder, save_plot=False, show_plot=False):
    """
    Creates an output plot for the Thunderfish program.

    This output contains the raw trace where the analysis window is marked, the power-spectrum of this analysis window
    where the detected fish are marked, a mean EOD plot, a histogram of the inter EOD interval and further information
    about the recording that is analysed.


    Parameters
    ----------
    base_name: string
        Basename of audio_file.
    pulse_fish_width: bool
        True if a pulse-fish has been detected by analysis of the EODs.
    pulse_fish_psd: bool
        True if a pulse-fish has been detected by analysis of the PSD.
    EOD_count: int
        Number of detected EODs.
    median_IPI: float
        Mean inter EOD interval.
    inter_eod_intervals: array
        Time difference from one to another detected EOD.
    raw_data: array
        Dataset.
    samplerate: float
        Sampling-rate of the dataset.
    idx0: float
        Index of the beginning of the analysis window in the dataset.
    idx1: float
        Index of the end of the analysis window in the dataset.
    filtered_fishlist: array
        Frequency and power of fundamental frequency/harmonics of several fish.
    period: float
        Mean EOD time difference.
    time_eod: array
        Time for the mean EOD plot.
    mean_eod: array
        Mean trace for the mean EOD plot.
    std_eod: array
        Standard deviation for the mean EOD plot.
    unit: string
        Unit of the trace and the mean EOD.
    psd_data: array
        Power spectrum of the analysed data for different frequency resolutions.
    output_folder: string
        Path indicating where output-files will be saved.
    save_plot: bool
        If True, the plot will be saved in output_folder.
    show_plot: bool
        If True (and saveplot=False) it will show the plot without saving.
    """

    fig = plt.figure(facecolor='white', figsize=(14., 10.))
    ax1 = fig.add_axes([0.05, 0.9, 0.9, 0.1])  # title
    ax2 = fig.add_axes([0.075, 0.05, 0.8, 0.1])  # trace
    ax3 = fig.add_axes([0.075, 0.6, 0.4, 0.3])  # psd
    ax4 = fig.add_axes([0.075, 0.2, 0.4, 0.3])  # mean eod
    ax5 = fig.add_axes([0.575, 0.6, 0.4, 0.3])  # meta data
    ax6 = fig.add_axes([0.575, 0.2, 0.4, 0.3])  # inter EOD histogram

    # plot title
    if not pulse_fish_width and not pulse_fish_psd:
        ax1.text(-0.05, .75, '%s --- Recoding of a wavefish.' % base_name, fontsize=20, color='grey')
    elif pulse_fish_width and pulse_fish_psd:
        ax1.text(-0.02, .65, '%s --- Recoding of a pulsefish.' % base_name, fontsize=20, color='grey')
    else:
        ax1.text(-0.05, .75, '%s --- Recoding of wave- and pulsefish.' % base_name, fontsize=20, color='grey')
    ax1.text(0.83, .8, 'Thunderfish by Bendalab', fontsize=16, color='grey')
    ax1.set_frame_on(False)
    ax1.get_xaxis().set_visible(False)
    ax1.get_yaxis().set_visible(False)
    ############

    # plot trace
    time = np.arange(len(raw_data)) / samplerate
    ax2.plot(time[:idx0], raw_data[:idx0], color='blue')
    ax2.plot(time[idx1:], raw_data[idx1:], color='blue')
    ax2.plot(time[idx0:idx1], raw_data[idx0:idx1], color='red', label='analysis\nwindow')
    ax2.set_xlabel('Time [sec]')
    ax2.set_ylabel('Amplitude [a.u.]')
    ax2.legend(bbox_to_anchor=(1.15, 1), frameon=False)
    ############

    # plot psd
    try:
        dom_freq = \
        filtered_fishlist[np.argsort([filtered_fishlist[fish][0][1] for fish in range(len(filtered_fishlist))])[-1]][0][
            0]
        fish_count = len(filtered_fishlist)
    except IndexError:
        dom_freq = 1. / period
        fish_count = 1

    plot_decibel_psd(psd_data[0][0], psd_data[0][1], ax3, fs=12)
    if not pulse_fish_width and not pulse_fish_psd:
        consistent_fishes_psd_plot(filtered_fishlist, ax=ax3)
    ax3.set_title('Powerspectrum (%.0f detected fish)' % fish_count)

    ##########

    # plot mean EOD
    eod_waveform_plot(time_eod, mean_eod, std_eod, ax4, unit=unit)
    if pulse_fish_width and pulse_fish_psd:
        ax4.set_title('Mean EOD (%.0f EODs; Pulse frequency: ~%.1f Hz)' % (EOD_count, dom_freq), fontsize=14)
        # No xlim needed for pulse-fish, because of defined start and end in eod_waveform function of eodanalysis.py
    else:
        ax4.set_title('Mean EOD (%.0f EODs; Dominant frequency: %.1f Hz)' % (EOD_count, dom_freq), fontsize=14)
        ax4.set_xlim([-600 * period, 600 * period])
    ###############

    # plot meta data
    ax5.set_frame_on(False)
    ax5.get_xaxis().set_visible(False)
    ax5.get_yaxis().set_visible(False)

    # fishtype = 'pulse' if pulse_fish_width and pulse_fish_psd else 'wave'
    # ax5.text(0.1, 0.9, 'wave-/pulsefish detected:', fontsize=14)
    # ax5.text(0.6, 0.9, '%s / %s' %('-' if pulse_fish_psd else '+', '+' if pulse_fish_width or pulse_fish_psd else '-'),
    #          fontsize=14)

    # ax5.text(0.1, 0.9, 'fishtype:', fontsize=14)
    # ax5.text(0.6, 0.9, '%s' %fishtype, fontsize=14)

    # ax5.text(0.1, 0.7, '# detected fish:', fontsize=14)
    # ax5.text(0.6, 0.7, '%.0f' % fish_count, fontsize=14)
    #
    # if fishtype is 'wave':
    #     ax5.text(0.1, 0.5, 'dominant frequency:', fontsize=14)
    #     ax5.text(0.6, 0.5, '%.1f Hz' % dom_freq, fontsize=14)
    # else:
    #     ax5.text(0.1, 0.5, 'Mean pulse frequency:', fontsize=14)
    #     ax5.text(0.6, 0.5, '%.1f Hz' % dom_freq, fontsize=14)
    #
    # ax5.text(0.1, 0.3, '# detected EODs:', fontsize=14)
    # ax5.text(0.6, 0.3, '%.0f' %EOD_count, fontsize=14)
    #
    # ax5.text(0.1, 0.1, 'median EOD interval:', fontsize=14)
    # ax5.text(0.6, 0.1, '%.2f ms' % (median_IPI* 1000), fontsize=14)

    ################

    # plot inter EOD interval histogram
    tmp_period = 1000. / dom_freq
    tmp_period = tmp_period - tmp_period % 0.05
    inter_eod_intervals *= 1000.  # transform sec in msec
    median_IPI *= 1000.  # tranform sec in msec
    n, edges = np.histogram(inter_eod_intervals, bins=np.arange(tmp_period - 5., tmp_period + 5., 0.05))

    ax6.bar(edges[:-1], n, edges[1] - edges[0] - 0.001)
    ax6.plot([median_IPI, median_IPI], [0, max(n)], '--', color='red', lw=2, label='median %.2f ms' % median_IPI)
    ax6.set_xlabel('inter EOD interval [ms]')
    ax6.set_ylabel('n')
    ax6.set_title('Inter EOD interval histogram', fontsize=14)

    if max(inter_eod_intervals) - min(inter_eod_intervals) < 1.:
        ax6.set_xlim([median_IPI - 0.5, median_IPI + 0.5])
    # ax6.set_xlim([0, 20])
    ax6.legend(loc='upper right', frameon=False)

    # cosmetics
    for ax in [ax2, ax3, ax4, ax6]:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.get_xaxis().tick_bottom()
        ax.get_yaxis().tick_left()

    # save figure as pdf
    if save_plot:
        plt.savefig(os.path.join(output_folder, base_name + '.pdf'))
        plt.close()
    elif show_plot:
        plt.show()
        plt.close()
    else:
        plt.close()


def thunderfish(filename, channel=0, save_csvs=False, save_plot=False, output_folder='.', verbosearg=0):
    # check if output_folder ends with a '/'
    if output_folder[-1] != '/':
        output_folder += '/'

    # create output folder
    if save_csvs or save_plot:
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

    # configuration options:
    cfg = ConfigFile()
    cfg.add_section('Debugging:')
    cfg.add('verboseLevel', 0, '', '0=off upto 4 very detailed')

    add_psd_peak_detection_config(cfg)
    add_harmonic_groups_config(cfg)
    add_clip_config(cfg)
    add_best_window_config(cfg, w_cv_ampl=10.0)
    verbose = cfg.value('verboseLevel')
    if verbosearg is not None:
        verbose = verbosearg
    outfilename = os.path.splitext(os.path.basename(filename))[0]

    # load data:
    raw_data, samplerate, unit = load_data(filename, channel)
    if len(raw_data) == 0:
        return

    # calculate best_window:
    clip_win_size = 0.5
    min_clip, max_clip = clip_amplitudes(raw_data, int(clip_win_size * samplerate))
    try:
        idx0, idx1, clipped = best_window_indices(raw_data, samplerate, single=True, win_size=8.0, min_clip=min_clip,
                                                  max_clip=max_clip, w_cv_ampl=10.0, th_factor=0.8)
    except UserWarning as e:
        print(str(e))
        return
    data = raw_data[idx0:idx1]

    # pulse-type fish?
    pulse_fish_width, pta_value = check_pulse_width(data, samplerate)

    # calculate powerspectra with different frequency resolutions:
    psd_data = multi_resolution_psd(data, samplerate, fresolution=[0.5, 2 * 0.5, 4 * 0.5])

    # find the fishes in the different powerspectra:
    fishlists = []
    for i in range(len(psd_data)):
        h_kwargs = psd_peak_detection_args(cfg)
        h_kwargs.update(harmonic_groups_args(cfg))
        fishlist = harmonic_groups(psd_data[i][1], psd_data[i][0], verbose, **h_kwargs)[0]
        fishlists.append(fishlist)

    # find the psd_type:
    pulse_fish_psd, proportion = check_pulse_psd(psd_data[0][0], psd_data[0][1])

    # filter the different fishlists to get a fishlist with consistent fishes:
    if not pulse_fish_width and not pulse_fish_psd:
        filtered_fishlist = consistent_fishes(fishlists)
        # analyse eod waveform:
        mean_eod, std_eod, time, eod_times = eod_waveform(data, samplerate, th_factor=0.6)
        if save_csvs:
            # write csv file with main EODF and corresponding power in dB of detected fishes:
            csv_matrix = fundamental_freqs_and_db(filtered_fishlist)
            csv_name = output_folder + outfilename + '-wavefish_eodfs.csv'
            header = ['fundamental frequency (Hz)', 'power (dB)']
            write_csv(csv_name, header, csv_matrix)
    else:
        filtered_fishlist = []
        # analyse eod waveform:
        mean_eod_window = 0.002
        mean_eod, std_eod, time, eod_times = eod_waveform(data, samplerate, th_factor=0.6, start=-mean_eod_window,
                                                          stop=mean_eod_window)

    # write mean EOD
    if save_csvs:
        header = ['time (ms)', 'mean', 'std']
        write_csv(output_folder + outfilename + '-mean_waveform.csv', header,
                  np.column_stack((1000.0 * time, mean_eod, std_eod)))

    period = np.mean(np.diff(eod_times))

    # analyze inter-peak intervals:
    inter_peak_intervals = np.diff(eod_times)  # in sec
    lower_perc, upper_perc = np.percentile(inter_peak_intervals, [1, 100 - 1])
    inter_eod_intervals = inter_peak_intervals[(inter_peak_intervals > lower_perc) &
                                               (inter_peak_intervals < upper_perc)]
    median_IPI = np.median(inter_eod_intervals)
    std_IPI = np.std(inter_eod_intervals, ddof=1)

    if save_plot or not save_csvs:
        output_plot(outfilename, pulse_fish_width, pulse_fish_psd, len(eod_times), median_IPI,
                    inter_eod_intervals, raw_data, samplerate, idx0, idx1, filtered_fishlist,
                    period, time, mean_eod, std_eod, unit, psd_data, output_folder,
                    save_plot=save_plot, show_plot=not save_csvs)


def main():
    # command line arguments:
    parser = argparse.ArgumentParser(
        description='Analyse short EOD recordings of weakly electric fish.',
        epilog='by bendalab (2015-2016)')
    parser.add_argument('--version', action='version', version='1.0')
    parser.add_argument('-v', action='count', dest='verbose', help='verbosity level')
    parser.add_argument('file', nargs='?', default='', type=str, help='name of the file wih the time series data')
    parser.add_argument('channel', nargs='?', default=0, type=int, help='channel to be displayed')
    parser.add_argument('-p', dest='save_plot', action='store_true', help='use this argument to save output plot')
    parser.add_argument('-s', dest='save_csvs', action='store_true',
                        help='use this argument to save csv-analysis-files')
    parser.add_argument('-o', dest='output_folder', default=".", type=str,
                        help="path where to store results and figures")
    args = parser.parse_args()

    thunderfish(args.file, args.channel, args.save_csvs, args.save_plot, args.output_folder, verbosearg=args.verbose)


if __name__ == '__main__':
    main()
    print('\nThanks for using thunderfish! Your files have been analyzed!')
