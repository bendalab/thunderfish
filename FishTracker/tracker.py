import sys
sys.path.append('/home/raab/raab_code/thunderfish/thunderfish')
import numpy as np
import matplotlib.pyplot as plt
import dataloader as dl
import audioio as ai
import powerspectrum as ps
import harmonicgroups as hg
import config_tools as ct
from IPython import embed
import matplotlib.mlab as mlab
import pickle
import glob
import thunderfish.chirp as cp

def pickle_save(fishes, all_times, all_chirp_freq, all_chirp_time, audio_file):
    file_name = audio_file.split('/')[-1].split('.')[-2]
    no = str(len(glob.glob('*.p')))

    file = open('%s.p' % (file_name+'_fishes_'+no), "wb")
    pickle.dump(fishes, file)

    file2 = open('%s.p' % (file_name+'_all_times_'+no), "wb")
    pickle.dump(all_times, file2)

    file3 = open('%s.p' % (file_name + '_all_chirp_freq_' + no), "wb")
    pickle.dump(all_chirp_freq, file3)

    file4 = open('%s.p' % (file_name + '_all_chirp_time_' + no), "wb")
    pickle.dump(all_chirp_time, file4)
    # embed()

def sort_fishes(all_fundamentals):
    fishes = [[ 0. ]]
    last_fish_fundamentals = [ 0. ]

    # loop throught lists of fundamentals detected in sequenced PSDs.
    for list in range(len(all_fundamentals)):
        # if this list is empty add np.nan to every deteted fish list.
        if len(all_fundamentals[list]) == 0:
                for fish in range(len(fishes)):
                    fishes[fish].append(np.nan)
        # when list is not empty...
        else:
            # ...loop trought every fundamental for each list
            for idx in range(len(all_fundamentals[list])):
                # calculate the difference to all last detected fish fundamentals
                diff = abs(np.asarray(last_fish_fundamentals) - all_fundamentals[list][idx])
                # find the fish where the frequency fits bests (th = 1Hz)
                if diff[np.argsort(diff)[0]] < 1. and diff[np.argsort(diff)[0]] > -1.:
                    # add the frequency to the fish array and update the last_fish_fundamentals list
                    fishes[np.argsort(diff)[0]].append(all_fundamentals[list][idx])
                    last_fish_fundamentals[np.argsort(diff)[0]] = all_fundamentals[list][idx]
                # if its a new fish create a list of nans with the frequency in the end (list has same length as other
                # lists.) and add frequency to last_fish_fundamentals
                else:
                    fishes.append([np.nan for i in range(list)])
                    fishes[-1].append(all_fundamentals[list][idx])
                    last_fish_fundamentals.append(all_fundamentals[list][idx])

        # wirte an np.nan for every fish that is not detected in this window!
        for fish in range(len(fishes)):
            if len(fishes[fish]) != list +1:
                fishes[fish].append(np.nan)

    # reshape everything to arrays
    for fish in range(len(fishes)):
        fishes[fish] = np.asarray(fishes[fish])
    # remove first fish because it has beeen used for the first comparison !
    fishes.pop(0)

    return fishes

def spectogram(data, samplerate, fresolution=0.5, detrend=mlab.detrend_none, window=mlab.window_hanning, overlap=0.5,
               pad_to=None, sides='default', scale_by_freq=None):
    nfft = int(np.round(2 ** (np.floor(np.log(samplerate / fresolution) / np.log(2.0)) + 1.0)))
    if nfft < 16:
        nfft = 16
    noverlap = nfft*overlap
    spectrum, freqs, time = mlab.specgram(data, NFFT=nfft, Fs=samplerate, detrend=detrend, window=window,
                                          noverlap=noverlap, pad_to=pad_to, sides=sides, scale_by_freq=scale_by_freq)

    return spectrum, freqs, time

def main(audio_file, data_snippet_secs = 60., nffts_per_psd = 4, start_time= 0, end_time = None):
    cfg = ct.get_config_dict()

    data = dl.open_data(audio_file, 0, 60.0, 10.0)
    samplrate = data.samplerate
    # fig, ax = plt.subplots()
    all_fundamentals = []
    all_times = np.array([])
    all_chirp_time = np.array([])
    all_chirp_freq = np.array([])
    while start_time < int((len(data)-data_snippet_secs*samplrate) / samplrate):
        # all_fundamentals = []
        tmp_data = data[start_time*samplrate : (start_time+data_snippet_secs)*samplrate] # gaps between snippets !!!!
        # print('data loaded ...')

        spectrum, freqs, time = ps.spectrogram(tmp_data, samplrate, fresolution=2., overlap_frac=0.95)

        all_times = np.concatenate((all_times, time[::4]+start_time))
        # print('spectogramm calculated ...')

        for t in range(len(time)-(nffts_per_psd-1))[::4]:
            power = np.mean(spectrum[:, t:t+nffts_per_psd], axis=1)

            fishlist = hg.harmonic_groups(freqs, power, cfg)[0]

            if not fishlist == []:
                fundamentals = hg.extract_fundamental_freqs(fishlist)
                all_fundamentals.append(fundamentals)
            else:
                all_fundamentals.append(np.array([]))
#####################################################################################
        # NOW: all_fundamentals gets updated...
        # need: all_clean_fundamentals [ ... ... ...]


        all_fundamentals_array = np.concatenate((all_fundamentals))
        hist_data = np.histogram(all_fundamentals_array, bins= np.arange(min(all_fundamentals_array), max(all_fundamentals_array)+2., 2.))
        th_n = sum(hist_data[0]) * 0.01

        test_fundamentals = hist_data[1][hist_data[0] > th_n]
        test_fundamentals = test_fundamentals[ np.concatenate((np.array([True]), np.diff(test_fundamentals) > 5.)) ]

        clean_fundamentals = [np.array([]) for i in all_fundamentals]
        for i in range(len(test_fundamentals)):
            tmp_fundamentals = [all_fundamentals[j][(all_fundamentals[j] > test_fundamentals[i] - 10.) &
                                                    (all_fundamentals[j] < test_fundamentals[i] + 10.)]
                                for j in range(len(all_fundamentals))]
            clean_fundamentals = [np.concatenate((clean_fundamentals[k], tmp_fundamentals[k])) for k in range(len(clean_fundamentals))]
        clean_fundamentals = [np.unique(clean_fundamentals[i]) for i in range(len(clean_fundamentals))]

        fig, ax = plt.subplots()
        for i in range(len(all_fundamentals)):
            ax.plot(all_times[i] * np.ones(len(all_fundamentals[i])), all_fundamentals[i], '.', color='k')
            ax.plot(all_times[i] * np.ones(len(clean_fundamentals[i])), clean_fundamentals[i], '.', color='red')
        plt.show()
######################################################################################
        embed()
        quit()
        power = np.mean(spectrum, axis=1)
        fishlist = hg.harmonic_groups(freqs, power, cfg)[0]

        chirp_time, chirp_freq = cp.chirp_detection(spectrum, freqs, time, fishlist, freq_tolerance=4.)
        if len(chirp_time) > 0:
            print('--- CHIRP ---')
            print('time [sec]')
            print(chirp_time + start_time)
            print('frequency [Hz]')
            print(chirp_freq)

            all_chirp_time = np.concatenate((all_chirp_time, chirp_time + start_time))
            all_chirp_freq = np.concatenate((all_chirp_freq, chirp_freq))

        if (int(start_time) % int(data_snippet_secs * 30)) > -1 and (int(start_time) % int(data_snippet_secs * 30)) < 1:
            print start_time / 60
        if (int(start_time) % int(data_snippet_secs * 60)) > -1 and (int(start_time) % int(data_snippet_secs * 60)) < 1 and int(start_time) != 0:
            fishes = sort_fishes(all_fundamentals)
            # embed()
            # quit()
            print 'Minute'
            print start_time / 60
            print 'saving data...'

            pickle_save(fishes, all_times, all_chirp_freq, all_chirp_time, audio_file)
            all_fundamentals = []
            all_times = np.array([])
            all_chirp_time = np.array([])
            all_chirp_freq = np.array([])
        start_time += data_snippet_secs
        # print (start_time / 60)
        if end_time:
            if start_time >= end_time:
                break
    # plt.show()
    fishes = sort_fishes(all_fundamentals)
    print('Whole file processed...\nsaving data...')
    pickle_save(fishes, all_times, all_chirp_freq, all_chirp_time, audio_file)
    print('Done!')

if __name__ == '__main__':
    audio_file = sys.argv[1]
    main(audio_file)