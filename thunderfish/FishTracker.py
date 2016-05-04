import os
import glob
import matplotlib.pyplot as plt
#import seaborn as sns
import matplotlib.mlab as ml
from harmonic_tools import *
import sorting_tools as st
from IPython import embed


class FishTracker:
    def __init__(self, filename, samplingrate):
        self.rate = samplingrate
        self.tstart = 0
        self.fish_freqs_dict = {}
        self.datasize = 200.0  # seconds
        self.step = 0.5
        self.fresolution = 0.5
        self.twindow = 8.0
        self.fishes = {}
        self.filename = filename

    def processdata(self, data, fish_type, bwin, win_width, config_dict,
                    test_longfile=False):  # , rate, fish_freqs_dict, tstart, datasize, step ) :
        """
        gets sound data.
        builds a powerspectrum over 8 sec. these 8 seconds shift through the sound data in steps of 0.5 sec.

        calls function: harmonic_groups witch returns an array (fishlist) containing information for each fish.
                    Example: np.array([fund_freq_fish1, power], [harmonic1_fish1, power], ...),
                             np.array([fund_freq_fish2, power], [harmonic1_fish2, power], ...), ...

        calls function: puls_or_wave. This function gets the array fishlist. (see above)
                    Analysis the data and discriminates between puls and wavefish.
                    returns lists containing the fundamental frequencies for either wave- or pulsfish.

        finally: updates a global dictionary (self.fish_freqs_dict) witch gets the time variable (as key)
                 and the list of WAVE-fishes.

                 updates a global dictionary (self.pulsfish_freqs_dict) witch gets the time variable (as key)
                 and the list of PULS-fishes.
        """

        test_resolutions = [self.fresolution, self.fresolution * 2.0, self.fresolution * 4.0]
        all_nfft = []
        for i in np.arange(len(test_resolutions)):
            nfft = int(np.round(2 ** (np.floor(np.log(self.rate / test_resolutions[i]) / np.log(2.0)) + 1.0)))
            if nfft < 16:
                nfft = 16
            all_nfft.append(nfft)

        tw = int(np.round(self.twindow * self.rate))
        minw = all_nfft[0] * (config_dict['minPSDAverages'][0] + 1) / 2
        if tw < minw:
            tw = minw

        self.datasize = len(data) / self.rate

        # running Power spectrum in best window (8s)
        fishlists_dif_fresolution = []  # gets fishlists from 3 different psd(powerspectrum)
        for i in np.arange(len(all_nfft)):
            power, freqs = ml.psd(data[(bwin * self.rate):(bwin * 1.0 * self.rate + win_width * 1.0 * self.rate)],
                                  NFFT=all_nfft[i], noverlap=all_nfft[i] / 2, Fs=self.rate,
                                  detrend=ml.detrend_mean)
            if i == 0:
                power_fres1 = power
                freqs_fres1 = freqs

            fishlist, _, mains, allpeaks, peaks, lowth, highth, center = harmonic_groups(freqs, power, config_dict)

            # power to dB
            for fish in np.arange(len(fishlist)):
                for harmonic in np.arange(len(fishlist[fish])):
                    fishlist[fish][harmonic][-1] = 10.0 * np.log10(fishlist[fish][harmonic][-1])

            fishlists_dif_fresolution.append(fishlist)

        fishlist = st.filter_fishes(fishlists_dif_fresolution)

        if fish_type is 'pulse':
            for i in np.arange(len(fishlist))[::-1]:
                if fishlist[i][0][0] < 150:
                    fishlist.pop(i)

        psd_type, mean_proportions = st.wave_or_pulse_psd(power_fres1, freqs_fres1,
                                        data[(bwin * self.rate):(bwin * self.rate + win_width * self.rate)], self.rate,
                                        self.fresolution)
        
        wave_ls = [fishlist[fish][0][0] for fish in np.arange(len(fishlist))]

        config_dict['lowThreshold'][0] = lowth
        config_dict['highThreshold'][0] = highth

        temp_dict = {bwin: wave_ls}

        self.fish_freqs_dict.update(temp_dict)

        # Pulsefish processing
        # if psd_type is 'pulse' and fish_type is 'pulse':
        #     print 'HERE WE HAVE TO BUILD IN THE PULSEFISH ANALYSIS'
        # # embed()
        return power_fres1, freqs_fres1, psd_type, fish_type, fishlist, mean_proportions

    def sort_my_wavefish(self):
        """
        this function works with the global dictionary self.fish_freq_dict and assigns the frequencies to fishes.


        finally: updates a global dict (self.fishes) witch contains the fishes and the frequencies belonging to it.
                    Example: {[1: np.NaN     np.NaN      174.3       175.0       np.NaN ...],
                              [2: np.NaN     180.7       np.NaN      181.9       np.NaN ...], ...
                np.NaN --> fish was not available at this mesuing point.
                this dict therefore carries the time variable.

                polt: x-axis: fishno.; y-axis: frequencies
        """

        dict_times = self.fish_freqs_dict.keys()

        for k, t in enumerate(sorted(self.fish_freqs_dict)):

            if t is dict_times[0]:
                for i in np.arange(len(self.fish_freqs_dict[t])):
                    print 'Phase II; ', 'point in time is: ', t, 'secondes.', 'wavefish no.:', i
                    temp_fish = {len(self.fishes) + 1: [self.fish_freqs_dict[t][i]]}
                    self.fishes.update(temp_fish)
            else:
                for i in np.arange(len(self.fish_freqs_dict[t])):
                    print 'Phase II; ', 'point in time is: ', t, 'secondes.', 'wavefish no.:', i
                    help_v = 0
                    new_freq = self.fish_freqs_dict[t][i]

                    for j in self.fishes.keys():
                        for p in np.arange(len(self.fishes[j])) + 1:
                            if self.fishes[j][-p] is not np.nan:
                                index_last_nan = -p
                                break
                        if self.fishes[j][index_last_nan] - 0.5 < new_freq <= self.fishes[j][index_last_nan] + 0.5 and \
                                        help_v == 0:
                            self.fishes[j].append(new_freq)
                            help_v += 1

                    if help_v is 0:
                        temp_fish = {len(self.fishes) + 1: []}
                        for l in np.arange(k):
                            temp_fish[len(self.fishes) + 1].append(np.NaN)
                        temp_fish[len(self.fishes) + 1].append(self.fish_freqs_dict[t][i])
                        self.fishes.update(temp_fish)
                    elif help_v >= 2:
                        print "added frequency to more than one fish. reduce tolerance!!!"
                        break
                for m in self.fishes.keys():
                    if len(self.fishes[m]) < k + 1:
                        self.fishes[m].append(np.nan)

    def get_data(self):
        data = np.zeros(np.ceil(self.rate * self.datasize), dtype="<i2")
        return data

    def mean_multi_wavefish(self):
        """
        if three arguments are given:
        loads/build and npy file witch contains the main frequencies of the wave fishes of one recording.
        saves the file as .npy

        3. arg is a str

        """
        mean_path = 'fish_wave.npy'
        if not os.path.exists(mean_path):
            np.save(mean_path, np.array([]))
        means_frequencies = np.load(mean_path)

        means_frequencies = means_frequencies.tolist()

        keys = self.fishes.keys()
        build_mean = []

        for fish in keys:
            for time in np.arange(len(self.fishes[fish])):
                if self.fishes[fish][time] is not np.nan:
                    build_mean.append(self.fishes[fish][time])
            means_frequencies.append(np.mean(build_mean))
            build_mean = []

        means_frequencies = np.asarray(means_frequencies)
        np.save(mean_path, means_frequencies)

        print ''
        print 'Mean frequencies of the current wavefishes collected: '
        print means_frequencies
        print ''

    def exclude_short_files(self, data, index):

        if len(data[:index]) / self.rate <= 10.:
            good_file = False
        else:
            good_file = True
        return good_file

    def bw_psd_and_eod_plot(self, power, freqs, bwin, win_width, data, psd_type, fish_type, fishlist, pulse_data,
                            pulse_freq, output_path, mean_proportions, r_value, wave_data, save_data_for_plot=False, only_psd_plot=False):
        """
        Create figures showing the best window, its PSD and the the EOD of the fish
        """

        # PSD of the best window up to 3kHz #
        if len(fishlist) > 4:
            ind = np.argsort([fishlist[fish][0][1] for fish in np.arange(len(fishlist))])[-4:]
            # print ind
        else:
            ind = np.argsort([fishlist[fish][0][1] for fish in np.arange(len(fishlist))])
            # print ind

        plot_path = output_path + 'figures/'
        if not os.path.exists(plot_path):
            os.makedirs(plot_path)
        # start plot cosmetic with seaborn
        sns.set_context("poster")
        sns.axes_style('white')
        sns.set_style("ticks")

        plot_w = 26.
        plot_h = 15.
        fs = 16  # fontsize of the axis labels
        inch_factor = 2.54

        fig_all = plt.figure(figsize=(plot_w * 2 / inch_factor, plot_h * 2 / inch_factor))
        ax1_all = fig_all.add_subplot(2, 6, (7, 12))

        power_dB = 10.0 * np.log10(power)
        ax1_all.axis([0, 3000, min(power_dB[:len(freqs[freqs<3000])]), max(power_dB)+10])
        ax1_all.plot(freqs, power_dB, lw=2, color='royalblue', alpha=0.7)

        if save_data_for_plot:
            np.save('wave_psd_data.npy', np.array([freqs, power_dB]))

        if fish_type is 'wave' or psd_type is 'wave':
            color = ['red', 'blue', 'green', 'cornflowerblue']
            for color_no, fish in enumerate(ind):
                for harmonic in np.arange(len(fishlist[fish])):
                    if harmonic == 0:
                        ax1_all.plot(fishlist[fish][harmonic][0], fishlist[fish][harmonic][1], 'o', color=color[color_no],
                                     label=('%.2f Hz' % fishlist[fish][0][0]))
                    else:
                        ax1_all.plot(fishlist[fish][harmonic][0], fishlist[fish][harmonic][1], 'o', color=color[color_no])

        ax1_all.tick_params(axis='both', which='major', labelsize=fs - 2)
        ax1_all.legend(loc='upper right')
        ax1_all.set_xlabel('Frequency [Hz]', fontsize=fs)
        ax1_all.set_ylabel('Power [dB SPL]', fontsize=fs)
        ax1_all.set_title('PSD of best window', fontsize=fs + 2)
        sns.despine(fig=fig_all, ax=ax1_all, offset=10)

        # ax2_all = fig_all.add_subplot(2, 6, (4, 6))
        #####################################################################################################
        if only_psd_plot:
            fs=18
            figx, axx = plt.subplots(figsize=(40./ inch_factor, 20. / inch_factor))

            # power_dB = 10.0 * np.log10(power)
            axx.axis([0, 2500, min(power_dB[:len(freqs[freqs<3000])]), max(power_dB)+10])
            axx.plot(freqs, power_dB, lw=2, color='royalblue', alpha=0.7)

            if fish_type is 'wave' or psd_type is 'wave':
                color = ['red', 'blue', 'green', 'cornflowerblue']
                for color_no, fish in enumerate(ind):
                    for harmonic in np.arange(len(fishlist[fish])):
                        if harmonic == 0:
                            axx.plot(fishlist[fish][harmonic][0], fishlist[fish][harmonic][1], 'o', color=color[color_no],
                                         label=('%.2f Hz' % fishlist[fish][0][0]))
                        else:
                            axx.plot(fishlist[fish][harmonic][0], fishlist[fish][harmonic][1], 'o', color=color[color_no])

            axx.tick_params(axis='both', which='major', labelsize=fs - 2)
            axx.legend(loc='upper right')
            axx.set_xlabel('Frequency [Hz]', fontsize=fs)
            axx.set_ylabel('Power [dB SPL]', fontsize=fs)
            axx.set_title('PSD of best window', fontsize=fs + 2)
            sns.despine(fig=figx, ax=axx, offset=10)
            figx.tight_layout()

            plt.show(figx)
            quit()


        #####################################################################################################
        if fish_type is 'wave' and psd_type is 'wave':
            ax2_all = fig_all.add_subplot(2, 6, (4, 6))
            if len(fishlist) == 1:
                eod_plot_tw = 0.030
                mean_wave_data = []
                std_wave_data = []

                datapoints_p_wave = np.median([len(wave_data[i]) for i in wave_data.keys()])

                for i in wave_data.keys():
                    if len(wave_data[i]) != int(datapoints_p_wave):
                        # print i
                        del wave_data[i]

                for k in np.arange(int(datapoints_p_wave)):

                    try:
                        tmp_mu = np.mean([wave_data[wave][k] for wave in sorted(wave_data.keys())])
                        tmp_std = np.std([wave_data[wave][k] for wave in sorted(wave_data.keys())], ddof=1)
                    except IndexError:
                        print('Warning! Something seems odd when calculating sound-trace average. '
                              'Proceeding with fingers crossed...\n')
                        continue

                    mean_wave_data.append(tmp_mu)
                    std_wave_data.append(tmp_std)

                up_std = [mean_wave_data[i] + std_wave_data[i] for i in range(len(mean_wave_data))]
                bottom_std = [mean_wave_data[i] - std_wave_data[i] for i in range(len(mean_wave_data))]
                plot_time = ((np.arange(len(mean_wave_data)) * 1.0 / self.rate) - eod_plot_tw / 2) * 1000  # s to ms

                ax2_all.set_xlim([-1.5/fishlist[0][0][0] * 1000, 1.5/fishlist[0][0][0] * 1000])
                ax2_all = fig_all.add_subplot(2, 6, (4, 6))
                ax2_all.plot(plot_time, mean_wave_data, lw=2, color='firebrick', alpha=0.7, label='mean EOD')
                ax2_all.fill_between(plot_time, up_std, bottom_std, color='grey', alpha=0.3)
                ax2_all.tick_params(axis='both', which='major', labelsize=fs - 2)
                ax2_all.set_xlabel('Time [ms]', fontsize=fs)
                ax2_all.set_ylabel('Amplitude [a.u.]', fontsize=fs)
                ax2_all.set_title('Mean wave-EOD', fontsize=fs + 2)
                sns.despine(fig=fig_all, ax=ax2_all, offset=10)

            else:
                # ax2_all = fig_all.add_subplot(2, 6, (4, 6))
                # fig3, ax3 = plt.subplots(figsize=(plot_w / inch_factor, plot_h / inch_factor))
                # ToDo: Brasil data doesnt work properly: Line 260 Index out of bounds
                eod_wtimes = (np.arange(
                    len(data[(bwin * self.rate):(bwin * self.rate + round(self.rate * 1.0 / fishlist[ind[-1]][0][0] * 4))])
                    ) * 1.0 / self.rate)*1000.0
                eod_wampls = data[
                             (bwin * self.rate):(bwin * self.rate + round(self.rate * 1.0 / fishlist[ind[-1]][0][0] * 4))]

                # WHEN SINGLEWAVEFISH (len(fishlist) =1) gemittelter plot; anderenfall continue
                ax2_all.set_xlim([eod_wtimes[0], eod_wtimes[-1]])
                ax2_all.plot(eod_wtimes, eod_wampls, lw=2, color='firebrick', alpha=0.7)
                ax2_all.tick_params(axis='both', which='major', labelsize=fs - 2)
                ax2_all.set_xlabel('Time [ms]', fontsize=fs)
                ax2_all.set_ylabel('Amplitude [a.u.]', fontsize=fs)
                ax2_all.set_title('EOD-Waveform', fontsize=fs + 2)
                sns.despine(fig=fig_all, ax=ax2_all, offset=10)

        if fish_type is 'pulse' or psd_type is 'pulse':
            eod_plot_tw = 0.006
            mean_pulse_data = []
            std_pulse_data = []

            datapoints_p_pulse = np.median([len(pulse_data[i]) for i in pulse_data.keys()])

            for i in pulse_data.keys():
                if len(pulse_data[i]) != int(datapoints_p_pulse):
                    # print i
                    del pulse_data[i]

            for k in np.arange(int(datapoints_p_pulse)):

                try:
                    tmp_mu = np.mean([pulse_data[pulse][k] for pulse in sorted(pulse_data.keys())])
                    tmp_std = np.std([pulse_data[pulse][k] for pulse in sorted(pulse_data.keys())], ddof=1)
                except IndexError:
                    print('Warning! Something seems odd when calculating sound-trace average. '
                          'Proceeding with fingers crossed...\n')
                    continue

                mean_pulse_data.append(tmp_mu)
                std_pulse_data.append(tmp_std)

            up_std = [mean_pulse_data[i] + std_pulse_data[i] for i in range(len(mean_pulse_data))]
            bottom_std = [mean_pulse_data[i] - std_pulse_data[i] for i in range(len(mean_pulse_data))]

            # get time for plot
            plot_time = ((np.arange(len(mean_pulse_data)) * 1.0 / self.rate) - eod_plot_tw / 2) * 1000  # s to ms
            if fish_type is 'pulse' and psd_type is 'pulse':
                ax2_all = fig_all.add_subplot(2, 6, (4, 6))
                ax2_all.plot(plot_time, mean_pulse_data, lw=2, color='firebrick', alpha=0.7, label='mean EOD')
                ax2_all.fill_between(plot_time, up_std, bottom_std, color='grey', alpha=0.3)
                # ax2_all.plot(plot_time, up_std, lw=1, color='red', alpha=0.7, label='std EOD')
                # ax2_all.plot(plot_time, bottom_std, lw=1, color='red', alpha=0.7)
                ax2_all.tick_params(axis='both', which='major', labelsize=fs - 2)
                ax2_all.set_xlabel('Time [ms]', fontsize=fs)
                ax2_all.set_ylabel('Amplitude [a.u.]', fontsize=fs)
                ax2_all.set_title('Mean pulse-EOD', fontsize=fs + 2)
                sns.despine(fig=fig_all, ax=ax2_all, offset=10)
            else:
                ax3 = fig_all.add_subplot(2, 6, (3, 4))
                ax3.plot(plot_time, mean_pulse_data, lw=2, color='firebrick', alpha=0.7, label='mean EOD')
                ax3.fill_between(plot_time, up_std, bottom_std, color='grey', alpha=0.3)
                # ax3.plot(plot_time, up_std, lw=1, color='red', alpha=0.7, label='std EOD')
                # ax3.plot(plot_time, bottom_std, lw=1, color='red', alpha=0.7)
                ax3.tick_params(axis='both', which='major', labelsize=fs - 2)
                # plt.legend(loc='upper right')
                ax3.set_xlabel('Time [ms]', fontsize=fs)
                ax3.set_ylabel('Amplitude [a.u.]', fontsize=fs)
                ax3.set_title('EOD-Waveform pulse', fontsize=fs + 2)
                sns.despine(fig=fig_all, ax=ax3, offset=10)
                # fig3.tight_layout()
                # fig3.savefig(plot_path + 'pulse-EOD%.0f.pdf' % (len(glob.glob(plot_path + 'pulse-EOD*.pdf')) + 1))
                # plt.close(fig3)
                ax5 = fig_all.add_subplot(2, 6, (5, 6))
                eod_wtimes = (np.arange(
                            len(data[(bwin * self.rate):(bwin * self.rate + round(self.rate * 1.0 / fishlist[ind[-1]][0][0] * 4))])
                            ) * 1.0 / self.rate)*1000.0
                eod_wampls = data[
                             (bwin * self.rate):(bwin * self.rate + round(self.rate * 1.0 / fishlist[ind[-1]][0][0] * 4))]

                # WHEN SINGLEWAVEFISH (len(fishlist) =1) gemittelter plot; anderenfall continue
                ax5.set_xlim([eod_wtimes[0], eod_wtimes[-1]])
                ax5.plot(eod_wtimes, eod_wampls, lw=2, color='firebrick', alpha=0.7)
                ax5.tick_params(axis='both', which='major', labelsize=fs - 2)
                ax5.set_xlabel('Time [ms]', fontsize=fs)
                ax5.set_ylabel('Amplitude [a.u.]', fontsize=fs)
                ax5.set_title('EOD-Waveform wave', fontsize=fs + 2)
                sns.despine(fig=fig_all, ax=ax5, offset=10)


        if psd_type is fish_type:
            text_ax = fig_all.add_subplot(2, 6, (1, 3))
        else:
            text_ax = fig_all.add_subplot(2, 6, (1, 2))
        text_ax.text(-0.1, 0.9, 'Filename:', fontsize=fs)
        text_ax.text(0.5, 0.9, '%s' % self.filename, fontsize=fs)

        text_ax.text(-0.1, 0.8, 'File duration:', fontsize=fs)
        text_ax.text(0.5, 0.8, '%.2f s' % (len(data) / self.rate), fontsize=fs)

        text_ax.text(-0.1, 0.7, 'Best window:', fontsize=fs)
        text_ax.text(0.5, 0.7, '%.2f s - %.2f s' % (bwin, bwin + win_width), fontsize=fs)

        text_ax.text(-0.1, 0.6, 'Shown EODf:', fontsize=fs)
        text_ax.text(-0.1, 0.5, 'Fish-Type:', fontsize=fs)

        if fish_type is not 'pulse' or psd_type is not 'pulse':
            text_ax.text(0.5, 0.6, '%.2f Hz' % fishlist[ind[-1]][0][0], fontsize=fs)
        else:
            text_ax.text(0.5, 0.6, '%.2f Hz' % pulse_freq, fontsize=fs)
        text_ax.text(0.5, 0.5, '%s' % fish_type + ' (%.2f ; th = 0.1)' % r_value, fontsize=fs)

        text_ax.text(-0.1, 0.4, 'PSD-Type:', fontsize=fs)
        text_ax.text(0.5, 0.4, '%s' % psd_type + ' (%.2f ; th = 0.27)' % mean_proportions, fontsize=fs)

        # text_ax.text(-0.1, 0.3, 'No. of wave-fishes:', fontsize=fs)
        # text_ax.text(0.5, 0.3, '%.0f' % len(fishlist), fontsize=fs)

        sns.despine(fig=fig_all, ax=text_ax, offset=10)
        plt.axis('off')

        fig_all.tight_layout()
        fig_all.savefig(plot_path + '%s.pdf' % self.filename[:-4])
        plt.close(fig_all)

    def pulse_sorting(self, bwin, win_width, data, fish_type, psd_type):
        # load data and time (0-8s) of bestwindow
        bw_data = data[(bwin * self.rate):(bwin * self.rate + win_width * self.rate)]
        time = np.arange(len(bw_data)) * 1.0 / self.rate

        # get time of data exceeding the threshold

        if fish_type is 'wave' and psd_type is 'pulse':
            threshold = np.percentile(bw_data, 90)
        else:
            threshold = max(bw_data) - ((max(bw_data) - np.mean(bw_data)) / 2)

        th_time = []
        for i in np.arange(len(bw_data) - 1):
            if bw_data[i + 1] > threshold and bw_data[i] <= threshold:
                th_time.append(time[i + 1])

        # pulse frequency   PROBLEMATISCH BEI MEHREREN PULSEFISHEN !!!
        pulse_freq = len(th_time) / win_width
        print('')
        print('Pulse-frequency:', pulse_freq)
        print('')

        pulse_frequencies = 'fish_pulse.npy'
        if not os.path.exists(pulse_frequencies):
            np.save(pulse_frequencies, np.array([]))
        pulse_ls = np.load(pulse_frequencies)
        pulse_ls = pulse_ls.tolist()

        pulse_ls.append(pulse_freq)

        pulse_ls = np.asarray(pulse_ls)
        np.save(pulse_frequencies, pulse_ls)

        # for each detected pulse/exceeding-time (count in pulse_data.keys())
        # save a window of the data arround this time
        pulse_data = {}
        eod_plot_tw = 0.006  # seconds shown in plot
        for i in np.arange(len(th_time)):
            plot_data = bw_data[(th_time[i] - eod_plot_tw / 2) * self.rate: (th_time[
                                                                                 i] - eod_plot_tw / 2) * self.rate + eod_plot_tw * self.rate]
            temp_dict = {len(pulse_data) + 1: plot_data}
            pulse_data.update(temp_dict)

        return pulse_data, pulse_freq

    def wave_sorting(self, bwin, win_width, data):
        bw_data = data[(bwin * self.rate):(bwin * self.rate + win_width * self.rate)]
        time = np.arange(len(bw_data)) * 1.0 / self.rate

        # threshold = max(bw_data) - ((max(bw_data) - np.mean(bw_data)) / 2)
        threshold = np.percentile(bw_data, 75)
        th_time = []
        for i in np.arange(len(bw_data) - 1):
            if bw_data[i + 1] > threshold and bw_data[i] <= threshold:
                th_time.append(time[i + 1])

        wave_data = {}
        # eod_plot_tw = 0.006  # seconds shown in plot
        eod_plot_tw = 0.030
        for i in np.arange(len(th_time)):
            plot_data = bw_data[(th_time[i] - eod_plot_tw / 2) * self.rate: (th_time[
                                                                                 i] - eod_plot_tw / 2) * self.rate + eod_plot_tw * self.rate]
            temp_dict = {len(wave_data) + 1: plot_data}
            wave_data.update(temp_dict)
        return wave_data
