import sys
import numpy as np
import copy
from scipy.stats import gmean
from scipy import signal
from scipy import optimize
import matplotlib.pyplot as plt
import matplotlib.colors as mplcolors
from thunderfish.dataloader import open_data
from thunderfish.peakdetection import detect_peaks
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
from collections import deque
import nixio as nix
import time
import os
import pickle

deltat = 60.0  # seconds of buffer size
thresh = 0.05
mind = 0.1  # minimum distance between peaks
peakwidththresh = 30 # maximum distance between max(peak) and min(trough) of a peak, in datapoints
new = 0

def main():  ############################################################# Get arguments eodsfilepath, plot, (opt)save, (opt)new  

    filepath = sys.argv[1]
    sys.argv = sys.argv[1:]

    plot = 0
    save = 0
    print(sys.argv)
    if len(sys.argv)==2:
        plot = int(sys.argv[1])
        print(plot)
    if len(sys.argv)==3:
        plot = int(sys.argv[1])
        save = int(sys.argv[2])
        print('saving results: ', save)
    import ntpath
    if len(sys.argv)==4:
        plot = int(sys.argv[1])
        save = int(sys.argv[2])
        new = int(sys.argv[3])
        print('saving results: ', save)
    ntpath.basename("a/b/c")
    def path_leaf(path):
        head, tail = ntpath.split(path)
        return tail or ntpath.basename(head)
    filename = path_leaf(filepath) 
    prefixlen = filename.find('_')+1
    starttime = "2000"
    home = os.path.expanduser('~')
    path =  filename[prefixlen:-4]+"/"
    os.chdir(home+'/'+path)                                     # operating in directory home/audiofilename/ 
    
   # if os.path.exists(filename[prefixlen:-4]+'_AmpFreq4.pdf'):
   #     new = 0

    with open_data(filename[prefixlen:-4]+".WAV", 0, 60, 0.0, 0.0) as data:
        samplerate = data.samplerate
        datalen = len(data)

    ############################################################# Fileimport and analyze; or skip, if analyzed data already exists
    if new == 1 or not os.path.exists('classes/'+ filename[prefixlen:-4]+"_classes.npz"):
         print('new analyse')
         eods = np.load(filename, mmap_mode='c')

    #     time1 = 40000
    #     time2 = 45000
    #     time1x = time1 * samplerate
    #     time2x = time2 * samplerate
    #     startpeak = np.where(((eods[0]>time1x)&(eods[0]<time2x))== True)
    #     endpeak = startpeak[0][-1]
    #     startpeak= startpeak[0][0]
    #     print(startpeak, endpeak)
    #     eods = eods[:,startpeak:endpeak] #  in case one does not want to analyze the whole file
#         print(eods[0][-1])
         #samplerate = 32000  #
         classlist = eods[3]

         # seperate Eods by class, apply low-level discard options
         fishes = []
         discardcondition1 = lambda fishclass: (len(fishclass[0]) < 11) # condition_1 : Take only classes with length over 10 peaks
         classamount = len(np.unique(classlist))
         #print('classamount: ', classamount)
         for i , num in enumerate(np.unique(classlist)):
             if classamount >= 100 and i % (classamount//100) == 0:
                 print(i)
             fishclass = eods[:,:][: , classlist == num]
             fish = []
             if len(fishclass[0]) < 12:
                continue
             for i , feature in enumerate(fishclass):
                 if i != 3:
                     fish.append(feature)
#             print('fish - printing to check structure', fish)
             temp_classisi = np.diff(fishclass[0])
             #print(temp_classisi)
             #print('plot smooth vs orig', len(temp_classisi))
             binlen=10
        #     temp_classisi_medians = temp_classisi#bin_median(temp_classisi, 1)
        #     smoothed = savgol_filter(temp_classisi_medians,11,1)
        #     diff = np.square(smoothed-temp_classisi_medians)
        #     data = np.array(diff)
        #     result = np.median(data[:(data.size // binlen) * binlen].reshape(-1, binlen),axis=1)
        #     result2 = bin_percentilediff(temp_classisi, 20)
        #     if len(result) > 7 and len(result2) > 7:
        #         smoothedresult = savgol_filter(result, 7, 1)
        #         smoothedresult2 = savgol_filter(result2, 7, 1)
        #     else:
        #         smoothedresult = result
        #         smoothedresult2 = result2
        #     #plt.plot(np.arange(0,len(result)*binlen, binlen),result)
        #     #plt.plot(smoothed)
        #     #plt.plot(np.arange(0,len(result2)*20, 20), smoothedresult2)
        #     #plt.plot(np.arange(0,len(result2)*20, 20), result2)
        #     #  plt.plot(temp_classisi_medians)
        #     #plt.plot(np.arange(0, len(smoothedresult)*binlen, binlen),smoothedresult)
        #     noiseindice = np.where(smoothedresult > 100000)
        #     #print(noiseindice)
        #     noiseindice = np.multiply(noiseindice, binlen)
        #     #print(noiseindice)
        #     noiseindice = [x for i in noiseindice[0] for x in range(i, i+10)]
        #     print(np.diff(noiseindice))
        #     noiseindice = np.split(noiseindice, np.where((np.diff(noiseindice) != 1 ) & (np.diff(noiseindice) != 2) & (np.diff(noiseindice) != 3))[0]+1 )
        #     #print(noiseindice)
        #     noiseindice = [x for arr in noiseindice if len(arr) > 20 for x in arr[50:-51]]
        #     noiseindice= np.array(noiseindice)
        #     #print(noiseindice)
        #     fish = np.array(fish)  
        #     # Noise delete applial 
        #   #  if len(noiseindice) >0 :
        #   #      fish[:,noiseindice] = np.nan #np.setdiff1d(np.arange(0, len(fish[0]),1),(noiseindice))] = np.nan
        #     fish = list(fish)  
        #     #plt.plot(temp_classisi) 
            #    plt.show()
             binlen = 60
             #print(len(fish[0]))
             if discardcondition1(fish) == False:   # condition length < 10
            # if False:
                 mean, std, d2, d8 = bin_array_mean(temp_classisi,binlen) 
                 #                 print('mean, std, d2, d8', mean, std, d2, d8)
                 count = ((mean * 4 >= d8) * (d2 >=  mean * 0.25)) .sum()  # condition_2 : if 0.2, and 0.8 deciles of the ISI of ONE SECOND/binlen are in the area of the median by a factor of 2, then the class seems to have not too much variability.
                 # Problem: Case, Frequency changes rapidly during one second/binlen , then the 0.8 or 0.2 will be out of the area...
                 # But then there is one wrong estimation, not too much of a problem
                 #print('fish')
              #   if count >= 0.5*(len(temp_classisi)//binlen +1):
                 if True:
                     fishes.append(fish)
         #print('len fishes after append', len(fishes))
         #print('printing fishes to check structure', fishes[0][0])  
        # ontimes = np.load('ontime'+filename[prefixlen:-4]+'.npz')
        # ontime = []
        # #         for c, items in enumerate(ontimes.items()):
        # #                 ontime.append(items[1])
        # ontime.append(ontimes['on'])
        # ontime.append(ontimes['near'])
        # ontime.append(ontimes['far'])
        #
        # if plot == 1:
        #     plot_ontimes(ontime)
                 
         #print(eods[0][-1]//samplerate, len(ontime[0]))
         if fishes is not None:

            #for fish in fishes:
            #    fish[0]
            
            # improving the fishpeak-data by adding peaks at places where theses peaks are hidden behind other (stronger)peaks
            #fishes = fill_hidden_3(fishes, eods, filename) # cl-dict : x y z -dict
            # filling holes or removing unexpected peaks from the class which are most likely caused by false classification
            #fishes, weirdparts = fill_holes(fishes)
            #fishes, weirdparts = fill_holes(fishes)

            if fishes is not None:
             if len(fishes) > 0:
                for cl, fish in enumerate(fishes):
                    ### Filter to only get ontimes close and nearby
                    for i, x in enumerate(fish[0]):
                        print(x)
                        #if x//samplerate < len(ontime[0]):
#                       #     print(ontime[1][x//samplerate], ontime[0][x//samplerate])
                        #    if ontime[0][x//samplerate] != 1 and ontime[1][x//samplerate] != 1 and ontime[2][x//samplerate] != 1:
                        #        for feat_i, feature in enumerate(fish):
                        #            fishes[cl][feat_i][i]  = np.nan
                        #        print(x//samplerate, ' ignored')
                    isi = [isi for isi in np.diff(fishes[cl][0])]
                    isi.append(isi[-1])
                    fishes[cl].append(isi)
            #fishes[i]        # the structure of the array fishes
            #       0 x
            #       1 y
            #       2 h
            #       3 isi 
                npFishes = fishes

                
      #   fishfeaturecount = len(fishes[cl])
      #   for cl in range(len(np.unique(classlist))-1):
      #       
      #       fishlen = len(fishes[cl][0])
      #       npFishes[cl]= np.memmap(filename[prefixlen:-4]+"_Fish%d"%cl+ ".npmmp", dtype='float32', mode='w+', shape=(fishfeaturecount, fishlen), order = 'F')
      #       np.zeros([fishfeaturecount, len(fishes[cl]['x'])])
      #       for i, feature in enumerate(['x', 'y', 'h', 'isi']): #enumerate(fishes[cl]):
      #           if feature == 'isi':
      #               fishes[cl][feature].append(fishes[cl][feature][-1])
      #           npFishes[cl][i] = np.array(fishes[cl][feature])
      #   
         
#         np.set_printoptions(threshold=np.nan)
                #
                if save == 1 and not os.path.exists('classes/'):
                    os.makedirs('classes/')

                #np.save('classes/'+ filename[prefixlen:-4]+"_class%d"%i, fish)
                #print('this', len(npFishes))
                if save == 1:
                    with open('classes/'+ filename[prefixlen:-4]+"_classes.lst", "wb") as fp:   #Pickling  
                        pickle.dump(npFishes, fp) 
                    #np.savez('classes/'+ filename[prefixlen:-4]+"_classes", npFishes)
    else:
        npFishes = []
        try:
            with open('classes/'+ filename[prefixlen:-4]+"_classes.lst", "rb") as fp:   #Pickling
                 npFishes = pickle.load(fp)
             #    npFishload=np.load('classes/'+ filename[prefixlen:-4]+"_classes.npz")
            print('loaded classes')
        except:
            print('no classes found')
    #    for fishes in npFishload.files:
    #        print('loaded ', fishes)
    #        for fish in npFishload[fishes]:
    #            fishtemp = np.zeros([4,len(fish[0])])
    #            for i, fishfeature in enumerate(fish):
    #                fishtemp[i] = fishfeature
    #            npFishes.append(fishtemp)
    #print('npFishes to check structure', npFishes[0][0][0]) 
#    if not os.path.exists('classes/'):
#        os.makedirs('classes/')
#    if not os.path.exists('classes/'+ filename[prefixlen:-4]+"_classes_red"):
#np.save('classes/'+ filename[prefixlen:-4]+"_class%d"%i, fish)
    if new == 1 or not os.path.exists('classes/'+ filename[prefixlen:-4]+"_classes_red.lst"):
#        reducednpFishes = npFishes
        reducednpFishes = reduce_classes(npFishes)# reducing classes by putting not overlapping classes together
        #print('reduced')
        if save == 1:
            with open('classes/'+ filename[prefixlen:-4]+"_classes_red.lst", "wb") as fp:   #Pickling
                pickle.dump(reducednpFishes, fp) 
        #np.savez('classes/'+ filename[prefixlen:-4]+"_classes_red.npz", reducednpFishes)
    else:
        with open('classes/'+ filename[prefixlen:-4]+"_classes_red.lst", "rb") as fp:   #Pickling
            reducednpFishes = pickle.load(fp)
        #print('len reduced ', len(reducednpFishes))
    if len(reducednpFishes) == 0:
        print('no on-/ or nearbytimeclass with sufficient length or good enough data. quitting')
        quit()
#        reducednpFishload=np.load('classes/'+ filename[prefixlen:-4]+"_classes_red.npz")
#
#        for fishes in reducednpFishload.files:
#            print('loaded reduced classes')
#            for fish in reducednpFishload[fishes]:
#                fishtemp = np.zeros([4,len(fish[0])])
#                for i, fishfeature in enumerate(fish):
#                    fishtemp[i] = fishfeature
#                reducednpFishes.append(fishtemp)
#
#    for i, rfish in enumerate(reducednpFishes):
#        if not os.path.exists('classes/'):
#            os.makedirs('classes/')
#        np.save('classes/'+ filename[prefixlen:-4]+"_class%d_reduced"%i, rfish)
    #print('reducednpFishes to check structure', reducednpFishes[0][3])



    window_freq = 1
    freqavgsecpath = filename[prefixlen:-4]+"_freqs2.npy"
    if new == 1 or not os.path.exists(freqavgsecpath):    
        print('new freq calcing')
        avg_freq = np.zeros([len(reducednpFishes),datalen//(samplerate*window_freq)+1])
        avg_isi = np.zeros([len(reducednpFishes),datalen//(samplerate*window_freq)+1])
        for i, fish in enumerate(reducednpFishes):
            fish = np.array(fish)
            avg_freqs_temp = []
            avg_isi_temp = []
            peak_ind = 0
            sec = 0
            for secx in np.arange(fish[0][0],fish[0][-1], samplerate*window_freq):
                #count_peaks_in_second = ((secx < fish[0]) & (fish[0] < secx+samplerate*window_freq)).sum()
               # isimean_peaks_in_second = fish[3][(secx < fish[0]) & (fish[0] < secx+samplerate*window_freq)].mean()                      #  #  #  #  #  #  #  #  # Using median instead of mean. Thus, hopefully overgoing outlier-isis, which are due to Peaks hidden beneath stronger Peaks of another fish.
                #freq_in_bin = samplerate/isimean_peaks_in_second
                sec_peaks =  fish[3][(secx <= fish[0]) & (fish[0] < secx+samplerate*window_freq)]
                #sec_freq = np.divide(samplerate,sec_peaks)
                print(sec_peaks)
                if len(sec_peaks) > 0:
                    #perctop, percbot = np.percentile(sec_peaks, [45, 55])
                    #peakisi_in_bin = sec_peaks[(perctop>=sec_peaks)&(sec_peaks>=percbot)].mean()
                    #print(perctop, percbot, peaks_in_bin)
                    #isimean_peaks_in_bin = sec_peaks[(perctop >=sec_peaks)&(sec_peaks>=percbot)].mean()
                    isimean_peaks_in_bin = np.median(sec_peaks)
                    freq_in_bin = samplerate/isimean_peaks_in_bin
                else: freq_in_bin = np.nan
               ################################################################################################################################### TODO 
                #isimean_peaks_in_bin = np.median(fish[3][(secx < fish[0]) & (fish[0] < secx+samplerate*window_freq)])
                print(freq_in_bin)
                #freq_in_bin = count_peaks_in_second
                if 5 < freq_in_bin < 140:
                    avg_freqs_temp.append(freq_in_bin)
                else:
                    avg_freqs_temp.append(np.nan)
                sec+=1
                #print(sec, freq_in_bin)
           # avg_freqs_temp, noiseindice = noisedelete_smoothing(avg_freqs_temp, 3, 2, 100000, 1000)
            #avg_freqs_temp, noiseindice = noisedelete_lowpass(avg_freqs_temp, binlen= 10)
            avg_freq[i, fish[0][0]//(samplerate*window_freq) : fish[0][0]//(samplerate*window_freq)+sec] = np.array(avg_freqs_temp)
        #plt.show()





        if save == 1:
            np.save(freqavgsecpath, avg_freq)
    else:
        avg_freq = np.load(freqavgsecpath)
        print('loaded freqs')
    #for i in avg_isi_fish:
    #    print('avg_freqs_byisi')    
    #    plt.plot(i)
    #plt.xlabel('seconds')
    #plt.ylabel('isi of peaks')
    #plt.show()
   # cmap = plt.get_cmap('jet')
   # colors =cmap(np.linspace(0, 1.0, 3000)) #len(np.unique(classlist))))
   # np.random.seed(22)
   # np.random.shuffle(colors)
   # colors = [colors[cl] for cl in range(len(avg_freq_fish))]
   # for i, col in zip(avg_freq_fish, colors):
   #     print('avg_freqs', 'len:' ,len(avg_freq_fish))    
   #     plt.plot(i, color = col)
   # plt.xlabel('seconds')
   # plt.ylabel('frequency of peaks')
   # plt.show()
   ## #print(avg_freqs[0])
    
     
    window_avg = 1
    ampavgsecpath = filename[prefixlen:-4]+'_amps2.npy'
        #freqtime = np.arange(0, len(data), samplerate)
    if new == 1 or not os.path.exists(ampavgsecpath):
        avg_amps_temp = []
        peak_ind = 0
        
        avg_amp = np.zeros([len(reducednpFishes),datalen//(samplerate*window_avg)+1])
        #avg_amp_fish = np.memmap(ampavgsecpath, dtype='float32', mode='w+', shape=(len(reducednpFishes),datalen//samplerate+1))
       
        for i, fish in enumerate(reducednpFishes):
          if len(fish[0]) >= 20:
            #print('amp, ', i, '/', len(reducednpFishes))
            step = 0
            avg_amps_temp = []
            for secx in np.arange(fish[0][0],fish[0][-1], samplerate*window_avg):
                 amp_in_second = fish[2][(secx < fish[0]) & (fish[0] < secx+samplerate*window_avg)].mean()
                # print(i, peak_ind, amp_in_second)
                 avg_amps_temp.append(amp_in_second)
                 step+=1
        #print('avg_amps_temp', avg_amps_temp)
        #avg_amps = np.memmap(ampavgsecpath, dtype='float32', mode='w+', shape=(len(avg_amps_temp), ))
        #avg_amps[:] = avg_amps_temp
        
            avg_amps_temp = np.array(avg_amps_temp)
            avg_amps_temp[np.where(np.isnan(avg_amps_temp))] = 0.0
            avg_amp[i, fish[0][0]//(samplerate*window_avg) : fish[0][0]//(samplerate*window_avg)+step] = avg_amps_temp
           
        if save == 1:
            np.save(ampavgsecpath, avg_amp)
#        np.save(ampavgsecpath, avg_amp_fish)
        # print('avg_amps ',avg_amps)
        #avg_freqs.append(np.mean(eods_freq[i:i+samplerate]))
    else:
        #avg_amps = np.memmap(ampavgsecpath, dtype='float32', mode='r', shape=(data//samplerate))
        avg_amp = np.load(ampavgsecpath)
        print('loaded amp')

    if new == 1 or plot == 1 :
        # Plotting #######################################################################################################################
        ##################################################################################################################################

        import matplotlib.gridspec as gridspec
        gs = gridspec.GridSpec(2, 2, height_ratios=(1, 1), width_ratios=(1, 0.02))

        # Tuning colors
        maxfreq = 140
        coloroffset = 5
        # Colorbar Choice
        cmap = plt.get_cmap('magma')#'gist_rainbow')
        cmap_amp = plt.get_cmap('Blues')#'gist_rainbow')
        # Colorbar Workaround
        Z = [[0,0],[0,0]]
        min, max = (0, maxfreq)
        step = 1
        levels = np.arange(min,max+step,step)
        CS3 = plt.contourf(Z, levels, cmap=cmap)
        plt.clf()
        plt.close()
        #####################
        # Colorbar Workaround
        Z = [[0,0],[0,0]]
        min, max = (0, 1)
        step = 1/100
        levels = np.arange(min,max+step,step)
        CSa = plt.contourf(Z, levels, cmap=cmap_amp)
        plt.clf()
        plt.close()
        #####################
        # mapping colormap onto fixed array of frequencyrange
        step = 1/maxfreq
        collist = cmap(np.arange(0, 1+step, step))
        ampstep = 1/200
        collist_amp = cmap_amp(np.arange(0, 1+ampstep, ampstep))
        collist_amp = collist_amp[100:]#[::-1]
        print(collist[0], collist[-1], collist[140])

        plt.rcParams['figure.figsize'] = 20,4.45
        ampax = plt.subplot(gs[1,:-1])
        #freqax = ampax.twinx()
        freqax = plt.subplot(gs[0,:-1], sharex=ampax)
        barax = plt.subplot(gs[1,-1])
        ampbarax = plt.subplot(gs[0,-1])
        avg_freq[ avg_freq == 0 ] = np.nan
        avg_amp[ avg_amp == 0 ] = np.nan
      #  colorlist = np.zeros([len(avg_freq)])
      #  valuecount = 0

        # remove amp where freq is np.nan
        # might actually not belong in the plotting section..
        #for f, a in zip(avg_freq, avg_amp):
        #   a[np.isnan(f)] = np.nan

        for f, a in zip(avg_freq, avg_amp):
            myred='#d62728'
            myorange='#ff7f0e'
            mygreen='#2ca02c'
            mylightgreen="#bcbd22"
            mygray="#7f7f7f"
            myblue='#1f77b4'
            mylightblue="#17becf"
            newlightblue = "#e1f7fd"
            # getting the right color for each scatterpoint
            fc = f[~np.isnan(f)]
            #collist = np.append(np.array([collist[0,:]]*30),(collist[30:]), axis = 0)
            fc[fc > maxfreq] = maxfreq
            #fc[fc < coloroffset] = 0
            #collist = np.append(np.array([collist[0,:]]*coloroffset),(collist[coloroffset:]), axis = 0)
            #col = [collist[v-coloroffset] if c >= coloroffset else collist[0] for v in fc if coloroffset <= v <= maxfreq]
            col = [collist[int(v)] for v in fc]
            ampcol = [collist_amp[int(v*100/2)] for v in a[~np.isnan(a)]]
            # plotting
            l1 = ampax.scatter(np.arange(0, len(a)*window_avg, window_avg) ,a,  s = 1,label = 'amplitude', color = col)#colors[col], ls = ':')
            l2 = freqax.scatter(np.arange(0,len(f)*window_freq,window_freq),f, s = 1, label = 'frequency', color = ampcol)#colors[col])
       # ls = l1+l2
        #labels = [l.get_label() for l in ls]
       # ampax.legend(ls, labels, loc=0)
        ampax.set_xlabel('Time [s]')
        ampax.set_ylabel('amplitude of peaks')
        freqax.set_ylabel('frequency of peaks')
        freqbar =plt.colorbar(CS3, cax = barax)
        ampbar = plt.colorbar(CSa, cax = ampbarax )
        freqbar.set_ticks([0,20,40,60,80,100,120])
        ampbar.set_ticks([0,0.2,0.4,0.6,0.8,1.0,1.2,1.4,1.8])
        ampbar.set_clim(-1,1)
        freqax.set_xlim(0,len(a)*window_avg)
        freqax.set_ylim(0,maxfreq)
        ampax.set_xlim(0, len(a)*window_avg)
        ampax.set_ylim(0,2)
        plt.setp(freqax.get_xticklabels(), visible=False)
        # remove last tick label for the second subplot
        yticks = ampax.yaxis.get_major_ticks()
        yticks[-1].label1.set_visible(False)
        plt.subplots_adjust(hspace=.0)
        print('plot', plot)
        if plot == 1:
            print('show plot')
            plt.show()
        if save == 1:
           plt.savefig(filename[prefixlen:-4]+'_AmpFreq5.pdf')
    else:
        print('already saved figure, if you want to see the result start with plot == 1')


def bin_percentilediff(data, binlen): 
    data = np.array(data)
    return np.percentile(data[:(data.size // binlen) * binlen].reshape(-1, binlen),95, axis=1) - np.percentile(data[:(data.size // binlen) * binlen].reshape(-1, binlen), 5 , axis=1)

def bin_mean(data, binlen): 
    return  np.mean(data[:(data.size // binlen) * binlen].reshape(-1, binlen),axis=1)
   # window_bigavg = 300
   # big_bin = []
   # for i in np.arange(0,len(avg_freq[0]),window_bigavg): #     print('iiii?', i)
   #     collector = []
   #     for f, a, col  in zip(avg_freq, avg_amp, colorlist):
   #             for data in f[i//window_freq:(i+window_bigavg)//window_freq]:
   #                 if data != 0 and not np.isnan(data):
   #                     collector.append(data)
   #     print(collector)
   #     if len(collector) >100:
   #         big_bin.append(collector)
   # for part in big_bin:
   #     print('i')
   #     plt.hist(part, bins = 250,  range = (0,250))
   #     plt.show()
       

def bin_ratio_std_mean(array, binlen):
    #print( bin_array_std(array, binlen)/bin_array_mean(array,binlen) )
     mean, std, d2, d8 = bin_array_mean(array,binlen) 
     #print('mean, std, d2, d8', mean, std, d2, d8)
     return   mean * 2 > d8 > mean > d2 > mean * 0.5

                         
def bin_array_std(array, binlen):
    bins = len(array)//binlen
    stds = np.zeros((bins+1))
    #print(array[0: binlen])
    for i in range(len(stds)):
        stds[i] = np.std(array[i*binlen: (i+1)*binlen])
    #print('stds0', stds[0], len(array))
    return stds


def bin_array_mean(array, binlen):
    bins = len(array)//binlen +1  if len(array) % binlen != 0 else len(array)//binlen
    means = np.zeros((bins))
    #print(array[0: binlen])
    stds = np.zeros((bins))
    d2 =  np.zeros((bins))
    d8 =  np.zeros((bins))
    for i in range(bins):
        stds[i] = np.std(array[i*binlen: (i+1)*binlen])
        means[i] = np.median(array[i*binlen: (i+1)*binlen])
        d2[i] =   np.percentile(array[i*binlen: (i+1)*binlen], 20)
        d8[i] =  np.percentile(array[i*binlen: (i+1)*binlen], 80)
       
            #  means[i] = np.mean(array[i*binlen: (i+1)*binlen])
    #print('mean0',means[0], len(array))
    return means, stds, d2, d8            

    
        
        
def bin_ndarray(ndarray, new_shape, operation='sum'):
    """
    Bins an ndarray in all axes based on the target shape, by summing or
        averaging.

    Number of output dimensions must match number of input dimensions and 
        new axes must divide old ones.

    Example
    -------
    >>> m = np.arange(0,100,1).reshape((10,10))
    >>> n = bin_ndarray(m, new_shape=(5,5), operation='sum')
    >>> print(n)

    [[ 22  30  38  46  54]
     [102 110 118 126 134]
     [182 190 198 206 214]
     [262 270 278 286 294]
     [342 350 358 366 374]]

    """
    operation = operation.lower()
    if not operation in ['sum', 'mean', 'std']:
        raise ValueError("Operation not supported.")
    if ndarray.ndim != len(new_shape):
        raise ValueError("Shape mismatch: {} -> {}".format(ndarray.shape,
                                                           new_shape))
    compression_pairs = [(d, c//d) for d,c in zip(new_shape,
                                                  ndarray.shape)]

    #print(len(new_shape))
    flattened = [l for p in compression_pairs for l in p]
 
    ndarray = ndarray.reshape(len(flattened))
    for i in range(len(new_shape)):
        op = getattr(ndarray, operation)
        ndarray = op(-1*(i+1))
    return ndarray       
   


        


def fill_hidden_3(fishes, eods, filename):
    fishes = fishes
    #print('hidden_calcing...')
    nohidefishes = []
    for cl, fish in enumerate(fishes):
        #print('Step1: Fish ', cl, ' ', cl, ' / ', len(fishes))
        #f = np.memmap(filename[prefixlen:-4]+"_Fish%d"%cl+ "X.npmmp", dtype='float32', mode='w+', shape=(3,len(fish[0])*2), order = 'F')
        f = np.zeros([3, len(fish[0])*2])
        fishisi = np.diff(fish[0])
        isi = fishisi[0]
        lst_offst =0 
        for i, newisi in enumerate(fishisi):
       #     print(cl, '              ..currently peak ', i, ' / ' , len(fishisi))
            newi = i+lst_offst
            if newi > len(f[0])-1: #   Errör
                  #  print('Oh shit, nparray to small. doubling size')
                    f_new = np.empty([3,len(f[0])*2])
                    f_new[:,:len(f[0])]=f
                    f = f_new
            f[0][newi]=fish[0][i]
            f[1][newi]=fish[1][i]
            f[2][newi]=fish[2][i]
            
#            print(i, newi)


            #  print(cl, fish[0][i], isi, newisi)
            if newisi > 2.8*isi:
                guessx = fish[0][i] + isi
                while guessx < fish[0][i] + newisi-0.8*isi:
                    peakx = peakaround3(guessx, isi*0.1, eods)
                    if peakx is not None:
                        newi = i+lst_offst
                        f[0][newi+1]=peakx
                        f[1][newi+1]=fish[1][i]
                        f[2][newi+1]=fish[2][i]
                        #print('estimated hidden peak: ', f[0][newi+1], f[2][newi+1])
                        guessx = peakx + isi + (peakx-guessx)
                        lst_offst +=1
                        #print('offset+1 at' ,i , peakx)
                        continue
                    break
            isi = newisi
       
        
        
        nohidefishes.append(np.array([f[0,0:newi+1],f[1,0:newi+1],f[2,0:newi+1]]))

    
       #print(x[0], x[200])
    return nohidefishes


def fill_hidden_Not(fishes, eods, filename):
    fishes = fishes
    #print('hidden_calcing...')
    nohidefishes = []
    #for cl, fish in enumerate(fishes):
        #print('Step1: Fish ', cl, ' ', cl, ' / ', len(fishes))
        #f = np.memmap(filename[prefixlen:-4]+"_Fish%d"%cl+ "X.npmmp", dtype='float32', mode='w+', shape=(3,len(fish[0])*2), order = 'F')
    return nohidefishes

def noisedelete_smoothing(array, binlen, method, thr1, thr2):
     if len(array) <= 2:
         if np.mean(array) > 140:
             for a in array:
                 a = np.nan
         return array, np.arange(0, len(array), 1)
     temp_classisi = array
     if len(array) > 11:
         smoothed = savgol_filter(temp_classisi, 11, 1)
     else: smoothed = savgol_filter(temp_classisi, 3, 1)
     diff = np.square(smoothed-temp_classisi)
     data = np.array(diff)
     #plt.plot(diff, color = 'green')
     result = np.median(data[:(data.size // binlen) * binlen].reshape(-1, binlen),axis=1)
     result2 = bin_percentilediff(temp_classisi, binlen)
     if method == 1:
         result = result
     elif method == 2:
         result = result2
     if len(result) > 7:
         smoothedresult = savgol_filter(result, 7, 1)
     else:
         smoothedresult = result
     #plt.plot(np.arange(0,len(result)*binlen, binlen),result)
     #plt.plot(smoothed)
     #plt.plot(np.arange(0,len(result2)*20, 20), smoothedresult2)
     #plt.plot(np.arange(0,len(result2)*20, 20), result2)
    # plt.plot(temp_classisi, color = 'black')
    # plt.plot(np.arange(0, len(result)*binlen, binlen),smoothedresult, 'red')
     if method ==1 :
         noiseindice = np.where(smoothedresult > thr1)
     elif method == 2:
         noiseindice = np.where(result > thr2)[0]
     elif method == 3:
         noiseindice = np.where(data > 1000)
     print(noiseindice)
     noiseindice = np.multiply(noiseindice, binlen)
     print(noiseindice)
     noiseindice = [x for i in noiseindice for x in range(i, i+binlen)]
     print(np.diff(noiseindice))
     noiseindice = np.split(noiseindice, np.where((np.diff(noiseindice) != 1 ) & (np.diff(noiseindice) != 2) & (np.diff(noiseindice) != 3))[0]+1 )
     #print(noiseindice)
     noiseindice = [x for arr in noiseindice if len(arr) > 1 for x in arr]
     noiseindice= np.array(noiseindice)
     #print(noiseindice)
     array = np.array(array)  
     # Noise delete applial 
     if np.median(array) > 150:
         noiseindice = np.arange(0, len(array), 1)
     if len(noiseindice) > 0:
         array[noiseindice] = np.nan
     return array, noiseindice

def noisedelete_lowpass(array,binlen): 
    origarray = array
    if len(array) <= 5:
         if np.mean(array) > 140 or np.mean(array) < 15:
             for a in array:
                 a = np.nan
         return array, [] #np.arange(0, len(array), 1)
    array = np.array(array) 
    from scipy.signal import butter, lfilter
    indice = []
    alldata = np.empty_like(array)
    if len(array[np.isnan(array)]) > 0:
        arrays = np.split(array, np.where(np.abs(np.diff(np.isnan(array))) == 1)[0]+1)
        indice = np.where(np.abs(np.diff(np.isnan(array))) == 1)[0]+1
        indice = np.append(np.array([0]),indice)
    else:
        arrays = [array]
        indice = [0]
    for array,index in zip(arrays, indice):
        if len(array) <2 or len(array[np.isnan(array)]) > 0:
            alldata[index:index + len(array)] = array[:]
            continue
        print(array, 'array')
        fs = 100
        cutoff =  25
        binlen = binlen
        data = np.array(array, dtype = 'float64')
        overlap = len(data)%binlen
        if overlap > 0:
           data = np.append(data, np.array([data[-1]]*(binlen-overlap)), axis = 0)
        dataext = np.empty([data.shape[0]+20])
        dataext[:10]= data[0]
        dataext[-10:] = data[-1]
        dataext[10:-10]=data
        B, A = butter(1, cutoff/ (fs / 2), btype = 'low')
        #lpf_array = np.empty_like(dataext)
        lpf_array= lfilter(B, A, dataext, axis = 0)
        lpf_array = lfilter(B, A, lpf_array[::-1])[::-1]
        lpf_binned_array = lpf_array[:(data.size // binlen) * binlen].reshape(-1, binlen)
        lpf_array = lpf_array[10:-10]
        if overlap > 0:
            lpf_array[-(binlen-overlap):] = np.nan
            data[-(binlen-overlap):] = np.nan
        binned_array = data[:(data.size // binlen) * binlen].reshape(-1, binlen)
        lpf_binned_array = lpf_array[:(data.size // binlen) * binlen].reshape(-1, binlen)
        filterdiffs = np.empty([binned_array.shape[0]])
        #a = signal.firwin(1, cutoff = 0.3, window = "hamming")
        for i, (bin_content, bin_filtered)  in enumerate(zip(binned_array, lpf_binned_array)):
            if i == binned_array.shape[0] - 1:
                bin_content = bin_content[:-(binlen-overlap)]
                bin_filtered = bin_filtered[:-(binlen-overlap)]
            filterdiffs[i] = np.mean(np.square(np.subtract(bin_filtered[~np.isnan(bin_filtered)], bin_content[~np.isnan(bin_content)])))
           # filterdiff = filterdiff / len(bin_content)
        print(filterdiffs)
        binned_array[filterdiffs > 1, :] = np.nan
        if overlap > 0:
            data = binned_array.flatten()[:-(binlen-overlap)]
        else:
            data = binned_array.flatten()
        print(data,    'data')
        alldata[index:index + len(data)] = data
        # twin[np.isnan(data)] = np.nan
   # plt.plot(alldata, color = 'red')
   # plt.plot(np.add(origarray, 2), color = 'blue')
   # plt.ylim(0, 150)
   # plt.show()
    return alldata, []

    # noiseindice = np.multiply(noiseindice, binlen)
    # print(noiseindice)
    # noiseindice = [x for i in noiseindice for x in range(i, i+binlen)]
    # print(np.diff(noiseindice))
    # noiseindice = np.split(noiseindice, np.where((np.diff(noiseindice) != 1 ) & (np.diff(noiseindice) != 2) & (np.diff(noiseindice) != 3))[0]+1 )
     
    # #print(noiseindice)
    # noiseindice = [x for arr in noiseindice if len(arr) > 1 for x in arr]
    # noiseindice= np.array(noiseindice)
    # #print(noiseindice)
    # array = np.array(array)  
    # # Noise delete applial 
    # if np.median(array) > 150:
    #     noiseindice = np.arange(0, len(array), 1)
    # if len(noiseindice) > 0:
    #     array[noiseindice] = np.nan
    #    return array, noiseindice


def peakaround3(guessx, interval, eods): 
    pksinintv = eods[0][ ((guessx-interval < eods[0]) & (eods[0] < guessx+interval))]
    if len(pksinintv)>0:
        return(pksinintv[0])
    elif len(pksinintv) >1:
        pksinintv = pksinintv[np.argmin(abs(pksinintv - guessx))]
        return(pksinintv)  ## might be bad, not tested
       # for px in fish[0]:
       #     distold = interval
       #     if px < guessx-interval:
       #         continue
       #    # print('in area', guessx-interval)
       #     if guessx-interval < px < guessx+interval:
       #         found = True
       #         dist = px-guessx
       #         if abs(dist) < abs(distold):
       #             distold = dist
       #     if px > guessx+interval:
       #         
       #         if found == True:
       #             print(guessx, dist)
       #             time.sleep(5)
       #             return guessx + dist
       #             
       #         else:
       #             
       #             break
    return None



def fill_holes(fishes):   #returns peakx, peaky, peakheight            # Fills holes that seem to be missed peaks in peakarray with fake (X/Y/height)-Peaks
 retur = []
 lost = []

 #print('fill_holes fishes', fishes)

 for cl, fish in enumerate(fishes):
    #print('Step2: Fish', cl)
    fishisi = np.diff(fish[0])
    mark = np.zeros_like(fishisi)
    isi = 0
    #print('mark', mark)
  #  print('fishisi' , fishisi)
    #find zigzag:
    c=0
    c0= 0
    n=0
    for i, newisi in enumerate(fishisi):
     #   print(newisi, isi)                    
        if abs(newisi - isi)>0.15*isi:                              ##  ZigZag-Detection : actually peaks of two classes in one class - leads to overlapping frequencys which shows in a zigzag pattern
            if (newisi > isi) != (fishisi[i-1] > isi):
                c+=1
            # print(abs(newisi - isi), 'x = ', fish[i].x)
            c0+=1
        elif c > 0:
            n += 1
        if n == 6:
            if c > 6:
               # print ('zigzag x = ', fish['x'][i-6-c0], fish['x'][i-6])   
                mark[i-6-c0:i-6]= -5
            c = 0
            c0=0
            n = 0
            
        #if c > 0:
            # print(i, c)
       # if c == 6:
           # print('zigzag!')
        isi = newisi
    isi = 0
    for i, newisi in enumerate(fishisi):                            ##  fill holes of up to 3 Peaks            # Changed to: Only up to 1 Peak  because : Holes might be intended for communicational reasons
        #print('mark: ' , mark)
        if mark[i] == -5: continue
        if i+2 >= len(fishisi):
            continue
        if  (2.2*isi > newisi > 1.8*isi) and (1.5*isi>fishisi[i+1] > 0.5*isi) :
            mark[i] = 1
            isi = newisi
           # print('found 1!' , i)
        elif (2.2*isi > newisi > 1.8*isi) and (2.2*isi> fishisi[i+1] > 1.8*isi) and (1.5*isi > fishisi[i+2] > 0.5*isi):
            mark[i] = 1
            isi = isi
        #elif  3.4*isi > newisi > 2.6*isi and 1.5*isi > fishisi[i+1] > 0.5*isi:
        #    mark[i] = 2 
            
        elif (0.6* isi > newisi > 0):
           # print('-1 found', i )
            if mark[i] ==0 and mark[i+1] ==0 and mark[i-1]==0 :
            #    isi  newisi
            #    continue
               # print('was not already set')
                if fishisi[i-2] > isi < fishisi[i+1]:
                    mark[i] = -1
                   # print('-1')
                elif isi > fishisi[i+1] < fishisi[i+2]:
                    mark[i+1] = -1
                  #  print('-1')
        isi = newisi
    x  = []
    y = []
    h = []
    x_lost=[]
    y_lost=[]
    h_lost=[]
  #  print('filledmarks: ', mark)
    for i, m in enumerate(mark):
        if m == -1 :
           # print('-1 at x = ', fish['x'][i])
            continue
        if m == -5:
            x_lost.append(fish[0][i])
            y_lost.append(fish[1][i])
            h_lost.append(fish[2][i])
            x.append(fish[0][i])
            y.append(fish[1][i])
            h.append(fish[2][i])
            continue
        x.append(fish[0][i])
        y.append(fish[1][i])
        h.append(fish[2][i])
        if m == 1:
           # print('hofly added peak at x = ' , fish['x'][i])
            x.append(fish[0][i] + fishisi[i-1])
            y.append( 0.5*(fish[1][i]+fish[1][i+1]))
            h.append(0.5*(fish[2][i]+fish[2][i+1]))
        elif m== 2:
            x.append(fish[0][i] + fishisi[i])
            y.append( 0.5*(fish[1][i]+fish[1][i+1]))
            h.append(0.5*(fish[2][i]+fish[2][i+2]))
            x.append(fish[0][i] + 2*fishisi[i-1])
            y.append( 0.5*(fish[1][i]+fish[1][i+2]))
            h.append(0.5*(fish[2][i]+fish[2][i+2]))
           # print('added at x = ', fish[0][i] + fishisi[i])
    x = np.array(x)
    y= np.array(y)
    h = np.array(h)
    x_lost = np.array(x_lost)
    y_lost = np.array(y_lost)
    h_lost = np.array(h_lost)
    #print('retur', x, y, h)
    retur.append([x,y,h])
    lost.append([x_lost,y_lost,h_lost])
   # filledpeaks =np.array(filledpeaks)
   # print(filledpeaks.shape)
   # filledpeaks.
   
 return retur, lost
        

#    eods[-len(thisblock_eods[:,]):] = thisblock_eods
#    eods = np.memmap("eods_"+filename[:-3]+"npy", dtype='float32', mode='r+', shape=(4,eods_len))
    #fp = np.memmap(filepath[:-len(filename)]+"eods_"+filename[:-3]+"npy", dtype='float32', mode='r+', shape=(4,len(thisblock_eods[:,])))
    #nix   print( b.data_arrays)
    #     for cl in np.unique(cllist):
    #  currentfish_x = x[:][cllist == cl]
    #  currentfish_y = y[:][cllist == cl]
    #  currentfish_h d= x[:][cllist == cl]
    #nix       try:
    #nix           xpositions[cl] = b.create_data_array("f%d_eods" %cl, "spiketimes", data = currentfish_x)
    #nix           xpositions[cl].append_set_dimension()
    #nix     #      thisfish_eods = b.create_multi_tag("f%d_eods_x"%cl, "eods.position", xpositions[cl])
    #nix     #      thisfish_eods.references.append(nixdata)
    #nix       except nix.pycore.exceptions.exceptions.DuplicateName:
    #nix     
    #nix           xpositions[cl].append(currentfish_x)
    
    
    #thisfish_eods.create_feature(y, nix.LinkType.Indexed)
    #b.create_multi_tag("f%d_eods_y"%cl, "eods.y", positions = y)
    #b.create_multi_tag("f%d_eods_h"%cl, "eods.amplitude", positions = h)
    #thisfish_eods.create_feature 
    
    #nix file.close()
    # Save Data
    # Needed:
    # Meta: Starttime, Startdate, Length
    # x, y, h, cl, difftonextinclass -> freq ? , 
    
    # Later: Find "Nofish"
    #        Find "Twofish"
    #        Find "BadData"
    #        Find "Freqpeak"
    #        ? Find "Amppeak"
    #        
    
    #  bigblock = np.array(bigblock)
    #  x=xarray(bigblock)
    #  y=yarray(bigblock)
    #  cl=clarray(bigblock)
    
    
    #nix file  = nix.File.open(file_name, nix.FileMode.ReadWrite)
    #nix b = file.blocks[0]
    #nix nixdata = b.data_arrays[0]
    #nix cldata = []
    #nix print(classes)
    #nix print(b.data_arrays)
    #nix for i in range(len(np.unique(classes))):
    #nix     cldata.append(b.data_arrays[i+1])
    
    
    # for cl in 
    
    # for cl in 
    #     x = thisfish_eods
    
    
    #nix file.close()
    
                 

def reduce_classes(npFishes):
    offtimeclasses = []
    for i, fish in enumerate(npFishes):
        fish = np.array(fish)
        #print(fish[0])
      #  print('nüFishes before and after command')
      #  print('bef', npFishes[i][0][0])
      #  print(fish[:,:][:,np.where(~np.isnan(fish[0]))].reshape(4,-1))
        npFishes[i] = fish[:,:][:,np.where(~np.isnan(fish[0]))][:,0]
      #  print('after', npFishes[i][0][0])
        if len(npFishes[i][0]) <= 100:
            offtimeclasses.append(i)
            #print('delete class ', i)
    #print('Len offtime vs len Fishes', len(offtimeclasses), len(npFishes))
    for index in sorted(offtimeclasses, reverse=True):
        del npFishes[index]
    #print('npFishes to check features', npFishes[0][3])  
    srt_beg = sort_beginning(npFishes)
   # print(len(npFishes[0]))
   # print(len(srt_beg))
    #srt_end = sort_ending(npFishes)
    if len(srt_beg) >= 1:
      reduced = []
      reduced.append(srt_beg[0])
      #for i, fish in enumerate(srt_beg):
      #print(len(srt_beg))
      #print('reducing classes')
      for i in range(1, len(srt_beg)):
        #print('.', end = '')
        cl = 0
        reducedlen_beg = len(reduced)
        while cl < reducedlen_beg:
            cond1 = reduced[cl][0][-1] < srt_beg[i][0][0]
            cond2 = False
            nxt=i+1
            while nxt < len(srt_beg) and srt_beg[i][0][-1] > srt_beg[nxt][0][0]:   #part ends after another part started (possibly in the other part.
                if len(srt_beg[nxt][0]) > len(srt_beg[i][0]):#            -> lencheck to pick longer part)
                    reduced.append(srt_beg[i])
                #    print('case1')
                    break
                nxt+=1
            else:
                cond2 = True
          #  print('lenreduced', len(reduced), len(srt_beg))
            #print(i, cl, cond1, cond2 )
            if cond1 and cond2:
                #print(reduced[cl].shape, srt_beg[i].shape)
                reduced[cl] =  np.concatenate((reduced[cl],srt_beg[i]), axis=1)
                #print(len(reduced[cl][0]), len(srt_beg[i][0]))
                cl+=1
                break
            if cond2 == False:
                break
            cl+=1
        else:
            reduced.append(srt_beg[i])

      #print('len red',  len(reduced))
      #print(len(npFishes[0]))
      return reduced
    else:
      return []  
    
def sort_beginning(npFishes):
    srted = npFishes
    srted.sort(key=lambda x: x[0][0])
    #for i in srted[0][0]:
    #    print(i)

    return srted

def sort_ending(npFishes):
    srted = npFishes[:]
    srted.sort(key=lambda x: x[0][-1])
    return srted

def noisedscrd(fishes):
    for fish in fishes:
        print(np.std(fish[2]))


def plot_ontimes(ontime):
    plt.fill_between(range(len(ontime[0])), ontime[0],  color = '#1e2c3c', label = 'close') #'#324A64'
    plt.fill_between(range(len(ontime[1])), ontime[1],  color = '#324A64', label = 'nearby')
    plt.fill_between(range(len(ontime[2])), ontime[2],  color = '#8ea0b4', label = 'far')
    plt.xlabel('seconds')
    plt.ylabel('position')
    plt.legend(loc = 1)
    plt.ylim(0,1.5)
  #  plt.xlim(0,len())
    plt.show()


        
if __name__ == '__main__':
    main()
