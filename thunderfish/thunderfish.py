"""
thunderfish

Run it from the thunderfish development directory as:
python3 -m thunderfish.thunderfish audiofile.wav   or
python -m thunderfish.thunderfish audiofile.wav
"""

import sys
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from .version import __version__
from .configfile import ConfigFile
from .harmonicgroups import add_psd_peak_detection_config, add_harmonic_groups_config
from .bestwindow import add_clip_config, add_best_window_config, clip_args, best_window_args
from .dataloader import load_data
from .bestwindow import clip_amplitudes, best_window_indices
from .checkpulse import check_pulse_width, add_check_pulse_width_config, check_pulse_width_args
from .powerspectrum import plot_decibel_psd, multi_resolution_psd
from .harmonicgroups import harmonic_groups, harmonic_groups_args, psd_peak_detection_args, fundamental_freqs, fundamental_freqs_and_power, colors_markers, plot_harmonic_groups
from .consistentfishes import consistent_fishes
from .eodanalysis import eod_waveform_plot, eod_waveform
from .csvmaker import write_csv


def output_plot(base_name, pulse_fish, pulse_period, eod_count, median_IPI, inter_eod_intervals,
                raw_data, samplerate, idx0, idx1, fishlist, wave_period, time_eod, mean_eod, std_eod, unit,
                psd_data, power_n_harmonics, label_power, max_freq=3000.0,
                output_folder='', save_plot=False, show_plot=False):
    """
    Creates an output plot for the Thunderfish program.

    This output contains the raw trace where the analysis window is marked, the power-spectrum of this analysis window
    where the detected fish are marked, a mean EOD plot, a histogram of the inter EOD interval and further information
    about the recording that is analysed.


    Parameters
    ----------
    base_name: string
        Basename of audio_file.
    pulse_fish: bool
        True if a pulse-fish has been detected by analysis of the EODs.
    pulse_period: float
        The mena interval between pulse-fish pulses.
    eod_count: int
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
    fishlist: array
        Frequency and power of fundamental frequency/harmonics of several fish.
    wave_period: float
        Mean EOD time difference of dominant wavefish.
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
    power_n_harmonics: int
        Maximum number of harmonics over which the total power of the signal is computed.
    label_power: boolean
        If `True` put the power in decibel in addition to the frequency into the legend.
    output_folder: string
        Path indicating where output-files will be saved.
    save_plot: bool
        If True, the plot will be saved in output_folder.
    show_plot: bool
        If True (and saveplot=False) it will show the plot without saving.
    """

    fig = plt.figure(facecolor='white', figsize=(14., 10.))
    ax1 = fig.add_axes([0.02, 0.9, 0.96, 0.1])  # title
    ax2 = fig.add_axes([0.075, 0.05, 0.9, 0.1]) # trace
    ax3 = fig.add_axes([0.075, 0.6, 0.7, 0.3])  # psd
    ax4 = fig.add_axes([0.075, 0.2, 0.4, 0.3])  # mean eod
    ax5 = fig.add_axes([0.575, 0.6, 0.4, 0.3])  # meta data
    ax6 = fig.add_axes([0.575, 0.2, 0.4, 0.3])  # inter EOD histogram
    
    # plot title
    wavetitle = ""
    if len(fishlist) == 1:
        wavetitle = "a wavefish"
    elif len(fishlist) > 1:
        wavetitle = "%d wavefish" % len(fishlist)
    pulsetitle = ""
    if pulse_fish:
        pulsetitle = "a pulsefish"
    if len(wavetitle)==0 and len(pulsetitle)==0:
        ax1.text(0.0, .72, '%s     Recording - no fish.' % base_name, fontsize=22)
    elif len(wavetitle)>0 and len(pulsetitle)>0:
        ax1.text(0.0, .72, '%s     Recording of %s and %s.' % (base_name, pulsetitle, wavetitle),
                 fontsize=22)
    else:
        ax1.text(0.0, .72, '%s     Recording of %s.' % (base_name, pulsetitle+wavetitle),
                 fontsize=22)
        
    ax1.text(1.0, .77, 'thunderfish by Benda-Lab', fontsize=16, ha='right')
    ax1.text(1.0, .5, 'Version %s' % __version__, fontsize=16, ha='right')
    ax1.set_frame_on(False)
    ax1.get_xaxis().set_visible(False)
    ax1.get_yaxis().set_visible(False)
    ############

    # plot trace
    time = np.arange(len(raw_data)) / samplerate
    ax2.plot(time[:idx0], raw_data[:idx0], color='blue')
    ax2.plot(time[idx1:], raw_data[idx1:], color='blue')
    ax2.plot(time[idx0:idx1], raw_data[idx0:idx1], color='red', label='analysis\nwindow')
    ax2.text(time[(idx0+idx1)//2], 0.0, 'analysis\nwindow', ha='center', va='center')
    ax2.set_xlim(time[0], time[-1])
    ax2.set_xlabel('Time [sec]')
    ax2.set_ylabel('Amplitude [a.u.]')
    #ax2.legend(bbox_to_anchor=(1.15, 1), frameon=False)
    ############

    # plot psd
    if len(fishlist) > 0:
        colors, markers = colors_markers()
        plot_harmonic_groups(ax3, fishlist, max_freq=max_freq, max_groups=12, sort_by_freq=True,
                             power_n_harmonics=power_n_harmonics, label_power=label_power,
                             colors=colors, markers=markers, legend_rows=12,
                             frameon=False, bbox_to_anchor=(1.0, 1.1), loc='upper left',
                             title='EOD frequencies')
    plot_decibel_psd(ax3, psd_data[0][1], psd_data[0][0], max_freq=max_freq, color='blue')
    ax3.set_title('Powerspectrum (%d detected wave-fish)' % len(fishlist), y=1.05)

    ##########

    # plot mean EOD
    if len(mean_eod) > 0:
        eod_waveform_plot(time_eod, mean_eod, std_eod, ax4, unit=unit)
        if pulse_fish:
            ax4.set_title('Average EOD of pulsefish (n=%d EODs, frequency: %.1f Hz)' % (eod_count, 1.0/pulse_period), fontsize=14, y=1.05)
            # No xlim needed for pulse-fish, because of defined start and end in eod_waveform function of eodanalysis.py
        else:
            ax4.set_title('Average EOD of dominant wavefish (n=%d EODs, frequency: %.1f Hz)' % (eod_count, 1.0/wave_period), fontsize=14, y=1.05)
            ax4.set_xlim([-750 * wave_period, 750 * wave_period])
            
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
    # ax5.text(0.6, 0.7, '%.0f' % len(fishlist), fontsize=14)
    #
    # if fishtype is 'wave':
    #     ax5.text(0.1, 0.5, 'dominant frequency:', fontsize=14)
    #     ax5.text(0.6, 0.5, '%.1f Hz' % dom_freq, fontsize=14)
    # else:
    #     ax5.text(0.1, 0.5, 'Mean pulse frequency:', fontsize=14)
    #     ax5.text(0.6, 0.5, '%.1f Hz' % dom_freq, fontsize=14)
    #
    # ax5.text(0.1, 0.3, '# detected EODs:', fontsize=14)
    # ax5.text(0.6, 0.3, '%.0f' %eod_count, fontsize=14)
    #
    # ax5.text(0.1, 0.1, 'median EOD interval:', fontsize=14)
    # ax5.text(0.6, 0.1, '%.2f ms' % (median_IPI* 1000), fontsize=14)

    ################

    # plot inter EOD interval histogram
    if len(inter_eod_intervals)>2 and median_IPI>0.0:
        tmp_period = 1000. * np.mean(inter_eod_intervals)
        tmp_period = tmp_period - tmp_period % 0.05
        inter_eod_intervals *= 1000.  # transform sec in msec
        median_IPI *= 1000.  # tranform sec in msec
        n, edges = np.histogram(inter_eod_intervals, bins=np.arange(tmp_period - 5., tmp_period + 5., 0.05))

        ax6.bar(edges[:-1], n, edges[1] - edges[0] - 0.001)
        ax6.plot([median_IPI, median_IPI], [0, max(n)], '--', color='red', lw=2, label='median %.2f ms' % median_IPI)
        ax6.set_xlabel('inter EOD interval [ms]')
        ax6.set_ylabel('n')
        ax6.set_title('Inter EOD interval histogram', fontsize=14, y=1.05)

        max_IPI = np.ceil(np.max(inter_eod_intervals)+0.5)
        if max_IPI/median_IPI < 1.2:
            max_IPI = np.ceil(1.2*median_IPI)
        min_IPI = np.floor(np.min(inter_eod_intervals)-0.5)
        if min_IPI/median_IPI > 0.8:
            min_IPI = np.floor(0.8*median_IPI)
        ax6.set_xlim(min_IPI, max_IPI)
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
    elif show_plot:
        plt.show()
    plt.close()


def thunderfish(filename, channel=0, save_csvs=False, save_plot=False,
                output_folder='.', cfgfile='', save_config='', verbose=0):
    # configuration options:
    cfg = ConfigFile()
    cfg.add_section('Power spectrum estimation:')
    cfg.add('frequencyResolution', 0.5, 'Hz', 'Frequency resolution of the power spectrum.')
    cfg.add('numberPSDWindows', 2, '', 'Number of windows on which power spectra are computed.')
    cfg.add('numberPSDResolutions', 1, '', 'Number of power spectra computed within each window with decreasing resolution.')
    cfg.add('frequencyThreshold', 1.0, 'Hz', 'The fundamental frequency of each fish needs to be detected in each power spectrum within this threshold.')
    # TODO: make this threshold dependent on frequency resolution!
    add_psd_peak_detection_config(cfg)
    add_harmonic_groups_config(cfg)
    add_clip_config(cfg)
    add_best_window_config(cfg, win_size=8.0, w_cv_ampl=10.0)
    add_check_pulse_width_config(cfg)

    # load configuration from working directory and data directories:
    cfg.load_files(cfgfile, filename, 3, verbose)

    # save configuration:
    if len(save_config) > 0:
        ext = os.path.splitext(save_config)[1]
        if ext != os.extsep + 'cfg':
            print('configuration file name must have .cfg as extension!')
        else:
            print('write configuration to %s ...' % save_config)
            cfg.dump(save_config)
        return None

    # check data file:
    if len(filename) == 0:
        return 'you need to specify a file containing some data'

    # create output folder
    if save_csvs or save_plot:
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
    outfilename = os.path.splitext(os.path.basename(filename))[0]

    # check channel:
    if channel < 0:
        return 'invalid channel %d' % channel

    # load data:
    raw_data, samplerate, unit = load_data(filename, channel, verbose=verbose)
    if len(raw_data) == 0:
        return 'empty data file %s' % filename

    # calculate best_window:
    found_bestwindow = True
    min_clip = cfg.value('minClipAmplitude')
    max_clip = cfg.value('maxClipAmplitude')
    if min_clip == 0.0 or max_clip == 0.0:
        min_clip, max_clip = clip_amplitudes(raw_data, **clip_args(cfg, samplerate))
    try:
        idx0, idx1, clipped = best_window_indices(raw_data, samplerate,
                                                  min_clip=min_clip, max_clip=max_clip,
                                                  **best_window_args(cfg))
    except UserWarning as e:
        print('best_window: ' + str(e))
        found_bestwindow = False
        idx0 = 0
        idx1 = len(raw_data)
    data = raw_data[idx0:idx1]

    # pulse-type fish?
    pulse_fish, pta_value, pulse_period = \
        check_pulse_width(data, samplerate, verbose=verbose,
                          **check_pulse_width_args(cfg))

    # calculate powerspectra within sequential windows and different frequency resolutions:
    numpsdwindows = cfg.value('numberPSDWindows')
    numpsdresolutions = cfg.value('numberPSDResolutions')
    minfres = cfg.value('frequencyResolution')
    fresolution=[minfres]
    for i in range(1, numpsdresolutions):
        fresolution.append(2*fresolution[-1])
    n_incr = len(data)//(numpsdwindows+1) # half overlapping
    psd_data = []
    deltaf = minfres
    for k in range(numpsdwindows):
        mr_psd_data = multi_resolution_psd(data[k*n_incr:(k+2)*n_incr], samplerate,
                                           fresolution=fresolution)
        deltaf = mr_psd_data[0][1][1] - mr_psd_data[0][1][0]
        psd_data.extend(mr_psd_data)
    
    # find the fishes in the different powerspectra:
    h_kwargs = psd_peak_detection_args(cfg)
    h_kwargs.update(harmonic_groups_args(cfg))
    fishlists = []
    for i, psd in enumerate(psd_data):
        fishlist = harmonic_groups(psd[1], psd[0], verbose-1, **h_kwargs)[0]
        if verbose > 0:
            print('fundamental frequencies detected in power spectrum of window %d at resolution %d:'
                  % (i//numpsdresolutions, i%numpsdresolutions))
            if len(fishlist) > 0:
                print('  ' + ' '.join(['%.1f' % freq[0, 0] for freq in fishlist]))
            else:
                print('  none')
        fishlists.append(fishlist)
    # filter the different fishlists to get a fishlist with consistent fishes:
    filtered_fishlist = consistent_fishes(fishlists, df_th=cfg.value('frequencyThreshold'))
    if verbose > 0:
        if len(filtered_fishlist) > 0:
            print('fundamental frequencies consistent in all power spectra:')
            print('  ' + ' '.join(['%.1f' % freq[0, 0] for freq in filtered_fishlist]))
        else:
            print('no fundamental frequencies are consistent in all power spectra')

    # remove multiples of pulse fish frequency from fishlist:
    if pulse_fish:
        fishlist = []
        for fish in filtered_fishlist:
            eodf = fish[0,0]
            n = np.round(eodf*pulse_period)
            pulse_eodf = n/pulse_period
            if verbose > 1:
                print('check wavefish at %.1f Hz versus %d-th harmonic of pulse frequency %.1f Hz at resolution of %.1f Hz'
                      % (eodf, n, pulse_eodf, n*deltaf))
            # TODO: make the threshold a parameter!
            if np.abs(eodf-pulse_eodf) > n*deltaf:
                fishlist.append(fish)
            else:
                if verbose > 0:
                    print('removed frequency %.1f Hz, because it is multiple of pulsefish %.1f' % (eodf, pulse_eodf))
    else:
        fishlist = filtered_fishlist

    # period of dominant fish:
    wave_period = -1.0
    if len(fishlist)>0:
        wave_period = 1.0/fishlist[np.argsort([fish[0,1] for fish in fishlist])[-1]][0,0]

    # save fish frequencies and amplitudes:
    if save_csvs and found_bestwindow:
        if len(fishlist) > 0:
            # write csv file with main EODf and corresponding power in dB of detected fishes:
            csv_matrix = fundamental_freqs_and_power(fishlist, cfg.value('powerNHarmonics'))
            csv_name = os.path.join(output_folder, outfilename + '-wavefish_eodfs.csv')
            header = ['fundamental frequency (Hz)', 'power (dB)']
            write_csv(csv_name, header, csv_matrix)
    #if pulse_fish:
        # TODO: write frequency amd parameter of pulse-fish to -pulsefish_eodfs.csv
        
    # analyse eod waveform:
    mean_eod = []
    std_eod = []
    eod_times = []
    time = []
    if pulse_fish:
        mean_eod_window = 0.002
        mean_eod, std_eod, time, eod_times = \
            eod_waveform(data, samplerate,
                         percentile=cfg.value('pulseWidthPercentile'),
                         th_factor=cfg.value('pulseWidthThresholdFactor'),
                         start=-mean_eod_window, stop=mean_eod_window)
    elif len(fishlist) > 0:
        mean_eod, std_eod, time, eod_times = \
            eod_waveform(data, samplerate,
                         percentile=cfg.value('pulseWidthPercentile'),
                         th_factor=cfg.value('pulseWidthThresholdFactor'),
                         period=wave_period)

    inter_eod_intervals = []
    median_IPI = -1.0
    std_IPI = -1.0
    if len(mean_eod)>0:
        # write mean EOD
        if save_csvs and found_bestwindow:
            header = ['time (ms)', 'mean', 'std']
            write_csv(os.path.join(output_folder, outfilename + '-mean_waveform.csv'), header,
                      np.column_stack((1000.0 * time, mean_eod, std_eod)))

        # analyze inter-peak intervals:
        inter_peak_intervals = np.diff(eod_times)  # in sec
        lower_perc, upper_perc = np.percentile(inter_peak_intervals, [1, 100 - 1])
        inter_eod_intervals = inter_peak_intervals[(inter_peak_intervals > lower_perc) &
                                                   (inter_peak_intervals < upper_perc)]
        if len(inter_eod_intervals) > 2:
            median_IPI = np.median(inter_eod_intervals)
            std_IPI = np.std(inter_eod_intervals, ddof=1)

    if save_plot or not save_csvs:
        if not found_bestwindow:
            pulsefish = False
            inter_eod_intervals = []
            fishlist = []
            mean_eod = []
        output_plot(outfilename, pulse_fish, pulse_period, len(eod_times), median_IPI,
                    inter_eod_intervals, raw_data, samplerate, idx0, idx1, fishlist,
                    wave_period, time, mean_eod, std_eod, unit,
                    psd_data, cfg.value('powerNHarmonics'), True, 3000.0, output_folder,
                    save_plot=save_plot, show_plot=not save_csvs)


def main():
    # config file name:
    cfgfile = __package__ + '.cfg'

    # command line arguments:
    parser = argparse.ArgumentParser(
        description='Analyze EOD waveforms of weakly electric fish.',
        epilog='by bendalab (2015-2017)')
    parser.add_argument('--version', action='version', version=__version__)
    parser.add_argument('-v', action='count', dest='verbose', help='verbosity level')
    parser.add_argument('-c', '--save-config', nargs='?', default='', const=cfgfile,
                        type=str, metavar='cfgfile',
                        help='save configuration to file cfgfile (defaults to {0})'.format(cfgfile))
    parser.add_argument('file', nargs='?', default='', type=str, help='name of the file with the time series data')
    parser.add_argument('channel', nargs='?', default=0, type=int, help='channel to be analyzed')
    parser.add_argument('-p', dest='save_plot', action='store_true', help='save output plot as pdf file')
    parser.add_argument('-s', dest='save_csvs', action='store_true',
                        help='save analysis results as csv-files')
    parser.add_argument('-o', dest='output_folder', default=".", type=str,
                        help="path where to store results and figures")
    args = parser.parse_args()

    # set verbosity level from command line:
    verbose = 0
    if args.verbose != None:
        verbose = args.verbose

    msg = thunderfish(args.file, args.channel, args.save_csvs, args.save_plot, args.output_folder,
                cfgfile, args.save_config, verbose=verbose)
    if msg is not None:
        parser.error(msg)


if __name__ == '__main__':
    main()
