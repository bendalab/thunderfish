__author__ = 'raab'

### Class functions (Class:FishTracker)
def mean_multi_pulsfish(self):
        """
        if three arguments are given:
        loads/build and npy file witch contains the main frequencies of the wave fishes of one recording.
        saves the file as .npy

        4th arg is a str

        """
        mean_path = 'fish_pulse.npy'
        if not os.path.exists(mean_path):
            np.save(mean_path, np.array([]))
        means_frequencies = np.load(mean_path)

        means_frequencies = means_frequencies.tolist()

        keys = self.pulsfishes.keys()
        build_mean = []

        for fish in keys:
            for time in np.arange(len(self.pulsfishes[fish])):
                if self.pulsfishes[fish][time] is not np.nan:
                    build_mean.append(self.pulsfishes[fish][time])
            means_frequencies.append(np.mean(build_mean))
            build_mean = []

        means_frequencies = np.asarray(means_frequencies)
        np.save(mean_path, means_frequencies)

        print ''
        print 'Mean frequencies of the current pulsfishes collected: '
        print means_frequencies
        print ''

def puls_main_frequencies(self):
        """
        take the global variable self.pulsfishes.
        builds the mean frequency for each fish.

        :return: list of mean frequenies for each pulsfish.
        """

        mean_fishes = []
        keys = self.pulsfishes.keys()
        build_mean = []

        for fish in keys:
            for time in np.arange(len(self.pulsfishes[fish])):
                if self.pulsfishes[fish][time] is not np.nan:
                    build_mean.append(self.pulsfishes[fish][time])
            mean_fishes.append(np.mean(build_mean))
            build_mean = []
        return mean_fishes

def latex_pdf(self):
        tf = open('Brasil.tex', 'w')
        tf.write('\\documentclass[a4paper,12pt,pdflatex]{article}\n')
        tf.write('\\usepackage{graphics}\n')
        # tf.write( '\\usepackage{siunits}\n' )
        tf.write('\n')
        tf.write('\\begin{document}\n')
        # tf.write( '\\section*{%s}\n' % filename )
        tf.write('\\section*{fish of brasil}\n')
        tf.write('\n')
        tf.write('\n')
        tf.write('\n')
        tf.write('\\includegraphics{sorted_fish.pdf}\n')
        tf.write('\\pagebreak\n')
        tf.write('\\includegraphics{spec_w_fish.pdf}\n')
        # tf.write( '\\pagebreak\n' )
        tf.write('\n')
        tf.write('\\begin{tabular}[t]{rr}\n')
        tf.write('\\hline\n')
        tf.write('fish no. & freq [Hz] \\\\ \\hline \n')

        # tf.write( '%s & %d \\\\\n' % (sorted_fish_freqs_2[5], s) )
        for i in self.fishes.keys():
            ffish = []
            for j in np.arange(len(self.fishes[i])):
                if self.fishes[i][j] is not np.nan:
                    ffish.append(self.fishes[i][j])
            if (i) % 35 == 0:
                tf.write('%s & %s \\\\\n' % (i, np.mean(ffish)))
                tf.write('\\hline\n')
                tf.write('\\end{tabular}\n')
                tf.write('\\begin{tabular}[t]{rr}\n')
                tf.write('\\hline\n')
                tf.write('fish no. & freq [Hz] \\\\ \\hline \n')

            else:
                tf.write('%s & %s \\\\\n' % (i, np.mean(ffish)))

        tf.write('\\hline\n')
        tf.write('\\end{tabular}\n')
        tf.write('\n')
        tf.write('\n')
        tf.write('\\end{document}\n')
        tf.close()
        os.system('pdflatex Brasil')
        os.remove('Brasil.aux')
        os.remove('Brasil.log')
        os.remove('Brasil.tex')
        os.remove('sorted_fish.pdf')
        os.remove('spec_w_fish.pdf')

def sort_my_pulsfish(self):
        """
        this function works with the global dictionary self.pulsfish_freq_dict and assigns the frequencies to fishes.


        finally: updates a global dict (self.pulsfishes) witch contains the fishes and the frequencies belonging to it.
                    Example: {[1: np.NaN     np.NaN      174.3       175.0       np.NaN ...],
                              [2: np.NaN     180.7       np.NaN      181.9       np.NaN ...], ...
                np.NaN --> fish was not available at this messuing point.
                this dict therefore carries the time variable.

                polt: x-axis: fishno.; y-axis: frequencies
        """

        dict_times = self.pulsfish_freqs_dict.keys()

        for k, t in enumerate(sorted(self.pulsfish_freqs_dict)):

            if t is dict_times[0]:
                for i in np.arange(len(self.pulsfish_freqs_dict[t])):
                    print 'Phase II; ', 'point in time is: ', t, 'secondes.', 'pulsfish no.:', i
                    temp_fish = {len(self.pulsfishes) + 1: [self.pulsfish_freqs_dict[t][i]]}
                    self.pulsfishes.update(temp_fish)
            else:
                for i in np.arange(len(self.pulsfish_freqs_dict[t])):
                    print 'Phase II; ', 'point in time is: ', t, 'secondes.', 'pulsfish no.:', i
                    help_v = 0
                    new_freq = self.pulsfish_freqs_dict[t][i]

                    for j in self.pulsfishes.keys():
                        for p in np.arange(len(self.pulsfishes[j])) + 1:
                            if self.pulsfishes[j][-p] is not np.nan:
                                index_last_nan = -p
                                break
                        if new_freq > self.pulsfishes[j][index_last_nan] - 0.5 and new_freq <= self.pulsfishes[j][
                            index_last_nan] + 0.5 and help_v == 0:
                            self.pulsfishes[j].append(new_freq)
                            help_v += 1

                    if help_v is 0:
                        temp_fish = {len(self.pulsfishes) + 1: []}
                        for l in np.arange(k):
                            temp_fish[len(self.pulsfishes) + 1].append(np.NaN)
                        temp_fish[len(self.pulsfishes) + 1].append(self.pulsfish_freqs_dict[t][i])
                        self.pulsfishes.update(temp_fish)
                    elif help_v >= 2:
                        print "added frequency to more than one fish. reduce tolerance!!!"
                        break
                for m in self.pulsfishes.keys():
                    if len(self.pulsfishes[m]) < k + 1:
                        self.pulsfishes[m].append(np.nan)

                        # if len(sys.argv) == 2:
                        #     fig, ax = plt.subplots(facecolor= 'white')
                        #     for n in self.pulsfishes.keys():
                        #         ax.plot([n]*len(self.pulsfishes[n]), self.pulsfishes[n], 'o')
                        #     ax.set_xlim([0, len(self.pulsfishes)+1])
                        #     ax.set_ylim([0, 2000])
                        #     ax.set_xlabel('fish Nr.', fontsize='15')
                        #     ax.set_ylabel('frequency [hz]', fontsize='15')
                        #     ax.spines["right"].set_visible(False)
                        #     ax.spines["top"].set_visible(False)
                        #     ax.tick_params(axis='both', direction='out')
                        #     ax.get_xaxis().tick_bottom()
                        #     ax.get_yaxis().tick_left()
                        #     plt.xticks(fontsize='15')
                        #     plt.yticks(fontsize='15')
                        #     plt.show()

def printspecto_pulsfish(self):
        """
        gets access to the dictionary self.pulsfish_freqs_dict with contains the time as key and the fundamental frequencies
        of the fishes available at this time.

        finaly: builds a scatterplot x-axis: time in sec; y-axis: frequency
        """

        if len(sys.argv) == 2:
            fig, ax = plt.subplots(facecolor='white')

            for t in self.pulsfish_freqs_dict.keys():
                ax.scatter([t] * len(self.pulsfish_freqs_dict[t]), self.pulsfish_freqs_dict[t])

            ax.set_ylim([0, 2000])
            ax.set_xlabel('time [s]', fontsize='15')
            ax.set_ylabel('frequencies [hz]', fontsize='15')
            ax.spines["right"].set_visible(False)
            ax.spines["top"].set_visible(False)
            ax.tick_params(axis='both', direction='out')
            ax.get_xaxis().tick_bottom()
            ax.get_yaxis().tick_left()

            plt.xticks(fontsize='15')
            plt.yticks(fontsize='15')
            plt.show()

def specto_with_sorted_pulsfish(self):
        if len(sys.argv) == 2:
            plot_fishes = []
            plot_time = []

            fig, ax = plt.subplots(facecolor='white')

            for t in self.pulsfish_freqs_dict.keys():
                ax.scatter([t] * len(self.pulsfish_freqs_dict[t]), self.pulsfish_freqs_dict[t])
            for i in self.pulsfishes.keys():
                print 'Phase III; ', 'Processing pulsfish no.: ', i
                # print 'III', i
                tnew = np.arange(len(self.pulsfishes[i])) * self.step
                help_tnew = np.arange(len(tnew) // 200)

                for k in np.arange(len(tnew)):
                    for l in help_tnew:
                        if tnew[k] > 191.5 + l * 200:
                            tnew[k] += 8

                for j in np.arange(len(self.pulsfishes[i])):
                    if self.pulsfishes[i][j] is not np.nan:
                        plot_fishes.append(self.pulsfishes[i][j])
                        plot_time.append(tnew[j])
                # print i, np.mean(plot_fishes)

                ax.plot(plot_time, plot_fishes, linewidth=2)
                plot_fishes = []
                plot_time = []

            ax.set_ylim([0, 2000])
            ax.set_xlabel('time [s]', fontsize='15')
            ax.set_ylabel('frequencies [hz]', fontsize='15')
            ax.spines["right"].set_visible(False)
            ax.spines["top"].set_visible(False)
            ax.tick_params(axis='both', direction='out')
            ax.get_xaxis().tick_bottom()
            ax.get_yaxis().tick_left()

            plt.xticks(fontsize='15')
            plt.yticks(fontsize='15')
            plt.show()

def main_frequency_hist(self):
        mean_fishes = []
        keys = self.fishes.keys()
        build_mean = []

        for i in keys:
            for j in np.arange(len(self.fishes[i])):
                if self.fishes[i][j] is not np.nan:
                    build_mean.append(self.fishes[i][j])
            mean_fishes.append(np.mean(build_mean))
            build_mean = []

        hist, bins = np.histogram(mean_fishes, bins=len(self.fishes) // 4)
        width = 0.7 * (bins[1] - bins[0])
        center = (bins[:-1] + bins[1:]) / 2

        fig, ax = plt.subplots(facecolor='white')
        ax.bar(center, hist, align='center', width=width)
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.tick_params(axis='both', direction='out')
        ax.get_xaxis().tick_bottom()
        ax.get_yaxis().tick_left()
        ax.set_ylabel('counts')
        ax.set_xlabel('frequency')
        plt.xticks(np.arange(0, max(mean_fishes) + 100, 250))
        plt.title('Histogram')
        plt.show()
        return mean_fishes

def wave_main_frequencies(self):
        """
        take the global variable self.fishes.
        builds the mean frequency for each fish.

        :return: list of mean frequenies for each wavefish.
        """

        mean_fishes = []
        keys = self.fishes.keys()
        build_mean = []

        for fish in keys:
            for time in np.arange(len(self.fishes[fish])):
                if self.fishes[fish][time] is not np.nan:
                    build_mean.append(self.fishes[fish][time])
            mean_fishes.append(np.mean(build_mean))
            build_mean = []
        return mean_fishes

def specto_with_sorted_wavefish(self):
        if len(sys.argv) == 2:
            plot_fishes = []
            plot_time = []

            fig, ax = plt.subplots(facecolor='white')

            for t in self.fish_freqs_dict.keys():
                ax.scatter([t] * len(self.fish_freqs_dict[t]), self.fish_freqs_dict[t])
            for i in self.fishes.keys():
                print 'Phase III; ', 'Processing wavefish no.: ', i
                tnew = np.arange(len(self.fishes[i])) * self.step
                help_tnew = np.arange(len(tnew) // 200)

                for k in np.arange(len(tnew)):
                    for l in help_tnew:
                        if tnew[k] > 191.5 + l * 200:
                            tnew[k] += 8

                for j in np.arange(len(self.fishes[i])):
                    if self.fishes[i][j] is not np.nan:
                        plot_fishes.append(self.fishes[i][j])
                        plot_time.append(tnew[j])
                # print i, np.mean(plot_fishes)

                ax.plot(plot_time, plot_fishes, linewidth=2)
                plot_fishes = []
                plot_time = []

            ax.set_ylim([0, 2000])
            ax.set_xlabel('time [s]', fontsize='15')
            ax.set_ylabel('frequencies [hz]', fontsize='15')
            ax.spines["right"].set_visible(False)
            ax.spines["top"].set_visible(False)
            ax.tick_params(axis='both', direction='out')
            ax.get_xaxis().tick_bottom()
            ax.get_yaxis().tick_left()

            plt.xticks(fontsize='15')
            plt.yticks(fontsize='15')
            plt.show()

            # pp = PdfPages('spec_w_fish.pdf')
            # fig.savefig(pp, format='pdf')
            # pp.close()

def printspecto_wavefish(self):
        """
        gets access to the dictionary self.fish_freqs_dict with contains the time as key and the fundamental frequencies
        of the fishes available at this time.

        finaly: builds a scatterplot x-axis: time in sec; y-axis: frequency
        """

        if len(sys.argv) == 2:
            fig, ax = plt.subplots(facecolor='white')

            for t in self.fish_freqs_dict.keys():
                ax.scatter([t] * len(self.fish_freqs_dict[t]), self.fish_freqs_dict[t])

            ax.set_ylim([0, 2000])
            ax.set_xlabel('time [s]', fontsize='15')
            ax.set_ylabel('frequencies [hz]', fontsize='15')
            ax.spines["right"].set_visible(False)
            ax.spines["top"].set_visible(False)
            ax.tick_params(axis='both', direction='out')
            ax.get_xaxis().tick_bottom()
            ax.get_yaxis().tick_left()

            plt.xticks(fontsize='15')
            plt.yticks(fontsize='15')
            plt.show()


### non-Class functions
def when_to_ask_to_start_next_script(filepath):
    if not os.path.exists('num_of_files_processed.npy'):
        help_v = filepath.split('/')
        directory = ''
        filetype = filepath.split('.')[-1]

        for i in np.arange(len(help_v) - 1):
            directory = directory + help_v[i] + '/'
        directory = directory + '*.' + filetype

        proc = subprocess.Popen(['ls %s -1 | wc -l' % directory], stdout=subprocess.PIPE, shell=True)
        (out, err) = proc.communicate()

        num_of_files_processed = [int(float(out))]
        np.save('num_of_files_processed.npy', num_of_files_processed)
    else:
        num_of_files_processed = np.load('num_of_files_processed.npy')

    return num_of_files_processed[0]

def num_current_file():
    if not os.path.exists('num_of_current_file.npy'):
        num_of_current_file = [1]
        np.save('num_of_current_file.npy', num_of_current_file)
    else:
        num_of_current_file = np.load('num_of_current_file.npy')
        num_of_current_file[0] += 1
        np.save('num_of_current_file.npy', num_of_current_file)
    return num_of_current_file[0]

def manual_input_wave_or_puls(test_freq, test_power, wave_ls, pulse_ls, known_answer):
    """
    This function helps to assign the fish type (pulse or wave) when the meanslopes of the powerspectrum is
    positiv (indicator for pulsefish) but the fundamental frequency is > 100 Hz. You will again have a lock
    on the powerspectrum of this fish and be foreced to assign it as a pulse- or a wavefish. You also have
    the choise to exclude the fish.

    :param test_freq: list
    :param test_power: list
    :param wave_ls: list
    :param pulse_ls: list

    :return: wave_ls, pulse_ls
    """
    if test_freq[0] in known_answer['freqs']:
        print ''
        print '### using known answers ###'
        print ''
        for i, j in enumerate(known_answer['freqs']):
            if j == test_freq[0]:
                response = known_answer['decision'][i]
    else:
        print ''
        print '### Programm needs input ###'
        print ('The fundamental frequency of this fish is: %.2f Hz.' % test_freq[0])
        print 'Here is the powerspectrum for this fish. Decide!!!'

        fig, ax = plt.subplots()
        ax.plot(test_freq, test_power, 'o')
        # plt.show()
        plt.draw()
        plt.pause(1)

        response = raw_input('Do we have a Wavefish [w] or a Pulsfish [p]? Or exclude the fish [ex]?')
        plt.close()
        if response in ['w', 'p', 'ex']:
            known_answer['freqs'].append(test_freq[0])
            known_answer['decision'].append(response)
        print ''
    if response == "w":
        wave_ls.append(test_freq[0])
    elif response == "p":
        pulse_ls.append(test_freq[0])
    elif response == "ex":
        print 'fish excluded.'
        print ''
    else:
        print '!!! input not valid !!!'
        print 'try again...'
        wave_ls, pulse_ls, known_answer = manual_input_wave_or_puls(test_freq, test_power, wave_ls, pulse_ls,
                                                                    known_answer)
    return wave_ls, pulse_ls, known_answer

def puls_or_wave(fishlist, known_answer, make_plots=False):
    """
    This function gets the array fishlist. (see below)
                    Analyses the data and discriminates between pulse and wavefish.
                    returns lists containing the fundamental frequencies for either wave- or pulse-fish.

    :param fishlist: dict
    :param make_plots:
    :return:lists: puls_ls, wave_ls
    """

    wave_ls = []
    pulse_ls = []

    for fish_idx in np.arange(len(fishlist)):
        test_freq = []
        test_power = []
        for harmo_idx in np.arange(len(fishlist[fish_idx])):
            test_freq.append(fishlist[fish_idx][harmo_idx][0])
            test_power.append(fishlist[fish_idx][harmo_idx][1])

            # wave_or_puls = []
        slopes = []  # each slope is calculated twice
        for first_idx in np.arange(len(test_power)):
            for second_idx in np.arange(len(test_power)):
                if first_idx > second_idx:
                    slopes.append((test_power[first_idx] - test_power[second_idx]) / (
                    test_freq[first_idx] - test_freq[second_idx]))
                if first_idx < second_idx:
                    slopes.append((test_power[second_idx] - test_power[first_idx]) / (
                    test_freq[second_idx] - test_freq[first_idx]))
        mean_slopes = np.mean(slopes)

        if mean_slopes > 0:
            if test_freq[0] >= 100:
                wave_ls, pulse_ls, known_answer = manual_input_wave_or_puls(test_freq, test_power, wave_ls, pulse_ls,
                                                                            known_answer)
            else:
                pulse_ls.append(test_freq[0])

        if mean_slopes < -0:
            if test_freq[0] < 100:
                wave_ls, pulse_ls, known_answer = manual_input_wave_or_puls(test_freq, test_power, wave_ls, pulse_ls,
                                                                            known_answer)
            else:
                wave_ls.append(test_freq[0])

        if make_plots:
            fig, ax = plt.subplots()
            ax.plot(test_freq, test_power, 'o')
            plt.show()

    return pulse_ls, wave_ls, known_answer