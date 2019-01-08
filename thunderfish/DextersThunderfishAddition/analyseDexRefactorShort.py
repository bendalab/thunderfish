#    Script to detect and classify EODs in recordings of weakly electric pulse
#    fish, Dexter Früh, 2018
# #    it is suggested to save the recording in
#       workingdirectory/recording/recording.WAV

#    results will be saved in workingdirectory/recording/
#
#    input:
#      -  [Recorded Timeseries] recording.WAV
#    outputs(optional):
#      -  [Detected and Classified EODs]
#            (Numpy Array with Shape (Number of EODs, 4 (Attributes of EODs)),
#            with the EOD-Attributes
#               -   x-location of the EOD
#                       (time/x-coordinate/datapoint in recording)
#               -   y-location of the EOD
#                       (Amplitude of the positive peak of the pulse-EOD)
#               -   height of the EOD(largest distance between peak and through in the EOD)
#               -   class of the EOD
#           eods_recording.npy
#      -   [plots of the results of each analyse step for each
#               analysepart (timeinterval of length = deltat) of the recording]
#
#    required command line arguments at function call
#        - save  : if True, save the results to a numpy file (possibly
#                                                          overwrite existing)
#        - plot  : if True, plot results in each analysestep
#        - new   : if True, do a new analysis of the recording, even if there
#                       is an existing analyzed .npy file with the right name.
#
#    call with:
#    python3 scriptname.py save plot new (starttime endtime[sec] for only
#                                                       partial analysis)
#
#   other parameters are behind imports and some hardcoded at the relevant
#       codestep
import sys
import numpy as np
import copy
from scipy.stats import gmean
from scipy import stats
from scipy import signal
from scipy import optimize
import matplotlib
from fish import ProgressFish
import matplotlib.pyplot as plt
from thunderfish.dataloader import open_data
from thunderfish.peakdetection import detect_peaks
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
from collections import deque
import ntpath
import nixio as nix
import time
import os
from shutil import copy2

from ownDataStructures import Peak, Tr, Peaklist
import DextersThunderfishAddition as dta

from IPython import embed
# parameters for the analysis

deltat = 30.0  # seconds of buffer size
thresh = 0.04 # minimal threshold for peakdetection
peakwidth = 20 # width of a peak and minimal distance between two EODs
# basic parameters for thunderfish.dataloader.open_data
verbose = 0
channel = 0
# timeinterval to analyze other than the whole recording
#starttime = 0
#endtime = 0
#timegiven = False

def main():         #  analyse_dex.py filename save plot new  (optional starttime endtime [sec])
    home = os.path.expanduser('~')
    os.chdir(home)
    # defaults for optional arguments
    timegiven = False
    plot_steps = False
    # parse command line arguments - filepath, save, plot, new (, starttime,
    filepath = sys.argv[1]
    save = int(sys.argv[2])
    plot_steps = int(sys.argv[3])
    new = int(sys.argv[4])
    if len(sys.argv[:])>5:
        timegiven = True
        starttime = int(sys.argv[5])
        endtime = int(sys.argv[6])
        #print(starttime, endtime)
    peaks = np.array([])
    troughs = np.array([])
    cutsize = 20
    maxwidth = 50 #10
    ultimate_threshold = thresh+0.01
    filename = path_leaf(filepath)
    proceed = input('Currently operates in home directory. If given a pulsefish recording filename.WAV, then a folder filename/ will be created in the home directory and all relevant files will be stored there. continue? [y/n]').lower()
    if proceed == 'n':
     quit()
    elif proceed == 'y':
        pass
    #do something
    elif proceed != 'y':
         quit()
    datasavepath = filename[:-4]
    print(datasavepath)
    eods_len = 0
    ### ## starting analysis
    if new == 1 or not os.path.exists(filename[:-4]+"/eods5_"+filename[:-3]+"npy"):
        ### ##  import data
        with open_data(filepath, channel, deltat, 0.0, verbose) as data:
           if save == 1 or save == 0:
               if not os.path.exists(datasavepath):
                   os.makedirs(datasavepath)
                   copy2(filepath, datasavepath)
           samplerate = data.samplerate
           ### ## split datalength into smaller blocks
           nblock = int(deltat*data.samplerate)
           if timegiven == True:
               parttime1 = starttime*samplerate
               parttime2 = endtime*samplerate
               data = data[parttime1:parttime2]
           if len(data)%nblock != 0:
               blockamount = len(data)//nblock + 1
           else:
               blockamount = len(data)//nblock
           bigblock = []
           ### ## output first (0%) progress bar
           print('blockamount: ' , blockamount)
           progress = 0
           print(progress, '%' , end = " ", flush = True)
           fish = ProgressFish(total = blockamount)
           olddatalen = 0
           startblock = 0
           ## iterating through the blocks, detecting peaks in each block
           for idx in range(startblock, blockamount):
               ### ## print progress
               if progress < (idx*100 //blockamount):
                   progress = (idx*100)//blockamount
               progressstr = 'Partstatus: '+ str(0) + ' '*2 + ' % (' + '0' + ' '*4+ '/' + '?'+' '*4+ '), Filestatus:'
               fish.animate(amount = idx, dexextra = progressstr)
               progressstr = 'Partstatus: '+ 'Part ' + '0'+ '/''5'+' Filestatus:'
               fish.animate(amount = idx, dexextra = progressstr)
               datx = data[idx*nblock:(idx+1)*nblock]
               # ---------- analysis --------------------------------------------------------------------------
               # step1: detect peaks in timeseries
               pk, tr = detect_peaks(datx, thresh)
               troughs = tr
               # continue with analysis only if multiple peaks are detected
               if len(pk) > 2:
                   peaks = dta.makeeventlist(pk,tr,datx,peakwidth)
                   #dta.plot_events_on_data(peaks, datx)
                   peakindices, peakx, peakh = dta.discardnearbyevents(peaks[0],peaks[1],peakwidth)
                   peaks = peaks[:,peakindices]
                   progressstr = 'Partstatus: '+ 'Part ' + '1'+ '/''5'+' Filestatus:'
                   fish.animate(amount = idx, dexextra = progressstr)
                   if len(peaks) > 0:
                       ### ## connects the current part with the one that came before, to allow for a continuous analysis
                       if idx >= startblock+1:
                           peaklist = connect_blocks(peaklist)
                       else:
                           peaklist = Peaklist([])
                       snips, aligned_snips = dta.cut_snippets(datx,peaks[0], 15, int_met = "cubic", int_fact = 10,max_offset = 1.5)
                       progressstr = 'Partstatus: '+ 'Part ' + '2'+ '/''5'+' Filestatus:'
                       fish.animate(amount = idx, dexextra = progressstr)
                       # calculates principal components
                       pcs = dta.pc(aligned_snips)#pc_refactor(aligned_snips)
                       #print('dbscan')
                       # clusters the features(principal components) using dbscan algorithm. clusterclasses are saved into the peak-object as Peak.pccl
                       order = 5
                       minpeaks = 3 if deltat < 2 else 10
                       peaks = dta.cluster_events(pcs, peaks, order, 0.4, minpeaks, False, olddatalen, method = 'DBSCAN')
                       
                       #dta.plot_events_on_data(peaks, datx)
                       olddatalen = len(datx)
                       num = 1
                       progressstr = 'Partstatus: '+ 'Part ' + '3'+ '/''5'+' Filestatus:'
                       fish.animate(amount = idx, dexextra = progressstr)
                       # classifies the peaks using the data from the clustered classes and a simple amplitude-walk which classifies peaks as different classes if their amplitude is too far from any other classes' last three peaks
                       peaks, peaklist = dta.ampwalkclassify3_refactor(peaks, peaklist, thresh) # classification by amplitude
                       #join_count=0
                     #  while True and joincc(peaklist, peaks) == True and join_count < 200:
                     #        join_count += 1
                     #        continue
                       # discards all classes that contain less than mincl EODs
                       minlen = 6   # >=1
                       peaks = dta.discard_short_classes(peaks, minlen)
                       if len(peaks[0]) > 0:
                           peaks = dta.discard_wave_pulses(peaks, datx)
                       # plots the data part and its detected and classified peaks
                       if plot_steps == True:
                           dta.plot_events_on_data(peaks, datx)
                           pass
                   # map the analyzed EODs of the buffer part to the whole
                   # recording
                   worldpeaks = np.copy(peaks)
                   # change peaks location in the buffered part to the location relative to the
                   idx = 1
                   # peaklocations relative to whole recording 
                   worldpeaks[0] = worldpeaks[0] + (idx*nblock)
                   peaklist.len = idx*nblock
                   thisblock_eods = np.delete(peaks,3,0)
                   thisblockeods_len = len(thisblock_eods[0])
                   progressstr = 'Partstatus: '+ 'Part ' + '4'+ '/''5'+' Filestatus:'
                   fish.animate(amount = idx, dexextra = progressstr)
                   # save the peaks of the current buffered part to a numpy-memmap on the disk
                   if thisblockeods_len> 0 and save == 1 or save == 0:
                       if idx == 0:
                               eods = np.memmap(datasavepath+"/eods_"+filename[:-3]+"npmmp", dtype='float64', mode='w+', shape=(4,thisblockeods_len), order = 'F')
                       dtypesize = 8#4 #float32 is 32bit = >4< bytes long  ---changed to float64 -> 8bit
                       eods = np.memmap(datasavepath+"/eods_"+filename[:-3]+"npmmp", dtype='float64', mode='r+', offset = dtypesize*eods_len*4, shape=(4,thisblockeods_len), order = 'F')
                       eods[:] = thisblock_eods
                       eods_len += thisblockeods_len
                   # to clean the plt buffer...
                   plt.close()
                   # get and print the measured times of the algorithm parts for the
                   # current buffer
                   progressstr = 'Partstatus: '+ 'Part ' + '5'+ '/''5'+' Filestatus:'
                   fish.animate(amount = idx, dexextra = progressstr)
                 #  plt.show()
        # after the last buffered part has finished, save the memory mapped
        # numpy file of the detected and classified EODs to a .npy file to the
        # disk
        eods = np.memmap(datasavepath+"/eods_"+filename[:-3]+"npmmp", dtype='float64', mode='r+', shape=(4,eods_len), order = 'F')
        print('before final saving: print unique eodcl: ' , np.unique(eods[3]))
        if save == 1:
           # #print('eods', eods[3])
           path = filename[:-4]+"/"
           if not os.path.exists(path):
               os.makedirs(path)
           if eods_len > 0:
               print('Saved!')
               np.save(filename[:-4]+"/eods8_"+filename[:-3]+"npy", eods)
           else:
               #np.save(filename[:-4]+"/eods5_"+filename[:-3]+"npy", thisblock_eods)
               print('not saved')

    else: # if there already has been a certain existing result file and 'new' was set to False
        print('already analyzed')


       # not used data implementation using NIX
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
      #nix #print(classes)
      #nix #print(b.data_arrays)
      #nix for i in range(len(np.unique(classes))):
      #nix     cldata.append(b.data_arrays[i+1])
       
      
       # for cl in 

      # for cl in 
      #     x = thisfish_eods
           

      #nix file.close()

def path_leaf(path):
    ntpath.basename("a/b/c")
    head, tail = ntpath.split(path)
    return tail or ntpath.basename(head)

def fill_hidden(fishclasses):
    
    fishes = fishclasses
    
    nohidefishes = {}
    for cl in fishes:
        x =[]
        y = []
        h = []
        fish = fishes[cl]
      #  #print('fish', fish)
        fishisi = calcisi(fish)
        isi = fishisi[0]
        for i, newisi in enumerate(fishisi):
            leftpeak = fish[i]
            x.append(leftpeak.x)
            y.append(leftpeak.y)
            h.append(leftpeak.height)
            if newisi > 2.8*isi:
                guessx = leftpeak.x + isi
               
                while guessx < leftpeak.x + newisi-0.8*isi:
                
                    peakx = peakaround(guessx, isi*0.1, fishes)
                    if peakx is not None:
                        x.append(peakx)
                        y.append(leftpeak.y)
                        h.append(leftpeak.height)
                        guessx = peakx+ isi + (peakx-guessx)
               
                        continue
                    break
            isi = newisi
        nohidefishes[cl]= {'x':x,'y':y,'h':h}
    return nohidefishes

def plotheights(peaklist):
    heights = heightarray(peaklist)
    x_locations = xarray(peaklist)
    plt.scatter(x_locations, heights)
    plt.show()

def ploteods(eods, data):
    plt.plot(range(len(data)),data, color = 'black')
    classlist = eods[3]
    cmap = plt.get_cmap('jet')
    colors =cmap(np.linspace(0, 1.0, 3000)) #len(np.unique(classlist))))
    np.random.seed(22)
    np.random.shuffle(colors)
    colors = [colors[cl] for cl in np.unique(classlist)]
    # plt.plot(xarray(peaksofclass), yarray(peaksofclass), '.c', ms=20)
    x=0
    if len(classlist)>0:
       # #print(classlist)
       # #print('classes: ' , np.unique(classlist))
        from collections import Counter
        count = Counter(classlist)
       # #print('longest class: ',  count.most_common()[0])
    for num, color in zip(np.unique(classlist), colors):
        peaksofclass = eods[:,:][:, classlist == num]
        #xpred = linreg_pattern(peaksofclass[0:3])
        #for p in peaksofclass[0:3]:
        #            #print(p.x)
        ##print(xpred, peaksofclass[3].x)            
                
        #if len(peaksofclass) > 1000:
        #    plt.plot(xarray(peaksofclass), yarray(peaksofclass), '.', color = 'red',   ms =20)
        #else:
        plt.plot(peaksofclass[0], peaksofclass[1], '.', color = color,   ms =20)
    plt.show()

def fill_hidden_3(fishes):
    
    fishes = fishes
    
    nohidefishes = {}
    for cl, fish in fishes.items():
        x =[]
        y = []
        h = []
       # fish = fishes[cl] passt net, fishes is np.array mit (cl, (xyh))
        fishisi = np.diff(fish[0])
        isi = fishisi[0]
        for i, newisi in enumerate(fishisi):
            leftpeak = i
            x.append(fish[0][i])
            y.append(fish[1][i])
            h.append(fish[2][i])
          #  #print(cl, fish[0][i], isi, newisi)
            if newisi > 2.8*isi:
                guessx = fish[0][i] + isi
            
                while guessx < fish[0][i] + newisi-0.8*isi:
               
                    peakx = peakaround3(guessx, isi*0.1, fishes)
                    if peakx is not None:
                       # #print(jup)
                        x.append(peakx)
                        y.append(fish[1][i])
                        h.append(fish[2][i])
                        guessx = peakx+ isi + (peakx-guessx)
               
                        continue
                    break
            isi = newisi
        nohidefishes[cl]= {'x':x,'y':y,'h':h}
    
    return nohidefishes

def peakaround2(guessx, interval, fishes):
    found = False
    for cl, fish in fishes.items():
        for px in fish['x']:
            distold = interval
            if px < guessx-interval:
                continue
           # #print('in area', guessx-interval)
            if guessx-interval < px < guessx+interval:
                found = True
                dist = px-guessx
                if abs(dist) < abs(distold):
                    distold = dist
            if px > guessx+interval:
                if found == True:
              #      #print(guessx, dist)
                    return guessx + dist
                else: break
    return None

def peakaround3(guessx, interval, fishes):
    found = False
    for cl, fish in fishes.items():
        for px in fish[0]:
            distold = interval
            if px < guessx-interval:
                continue
           # #print('in area', guessx-interval)
            if guessx-interval < px < guessx+interval:
                found = True
                dist = px-guessx
                if abs(dist) < abs(distold):
                    distold = dist
            if px > guessx+interval:
                if found == True:
              #      #print(guessx, dist)
                    return guessx + dist
                else: break
    return None

def peakaround(guessx, interval, fishes):
    found = False
    for cl, fish in fishes.items():
        for peak in fish:
        
            distold = interval
            if peak.x < guessx-interval:
                continue
           # #print('in area')
            if guessx-interval < peak.x < guessx+interval:
                found = True
                dist = peak.x-guessx
                if abs(dist) < abs(distold):
                    distold = dist
            if peak.x > guessx+interval:
                if found == True:
                   # #print(guessx, dist)
                    return guessx + dist
                else: break
    return None

def fill_holes(fishes):   #returns peakx, peaky, peakheight            # Fills holes that seem to be missed peaks in peakarray with fake (X/Y/height)-Peaks
     retur = {}
     lost = {}
     for cl, fish in fishes.items():
        fishisi = np.diff(fish['x'])
        mark = np.zeros_like(fishisi)
        isi = 0
        ##print('mark', mark)
      #  #print('fishisi' , fishisi)
        #find zigzag:
        c=0
        c0= 0
        n=0
        for i, newisi in enumerate(fishisi):
            if abs(newisi - isi)>0.15*isi:
                if (newisi > isi) != (fishisi[i-1] > isi):
                    c+=1
                # #print(abs(newisi - isi), 'x = ', fish[i].x)
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
                # #print(i, c)
           # if c == 6:
               # #print('zigzag!')
            isi = newisi
        isi = 0
        for i, newisi in enumerate(fishisi):
            ##print('mark: ' , mark)
            if mark[i] == -5: continue
            if i+2 >= len(fishisi):
                continue
            if  (2.2*isi > newisi > 1.8*isi) and (1.5*isi>fishisi[i+1] > 0.5*isi) :
                mark[i] = 1
                isi = newisi
               # #print('found 1!' , i)
            elif (2.2*isi > newisi > 1.8*isi) and (2.2*isi> fishisi[i+1] > 1.8*isi) and (1.5*isi > fishisi[i+2] > 0.5*isi):
                mark[i] = 1
                isi = isi
            elif  3.4*isi > newisi > 2.6*isi and 1.5*isi > fishisi[i+1] > 0.5*isi:
                mark[i] = 2 
                
            elif (0.6* isi > newisi > 0):
               # #print('-1 found', i )
                if mark[i] ==0 and mark[i+1] ==0 and mark[i-1]==0 :
                #    isi = newisi
                #    continue
                   # #print('was not already set')
                    if fishisi[i-2] > isi < fishisi[i+1]:
                        mark[i] = -1
                       # #print('-1')
                    elif isi > fishisi[i+1] < fishisi[i+2]:
                        mark[i+1] = -1
                      #  #print('-1')
            isi = newisi
        filldpeaks = []
        x  = []
        y = []
        h = []
        x_lost=[]
        y_lost=[]
        h_lost=[]
      #  #print('filledmarks: ', mark)
        for i, m in enumerate(mark):
            if m == -1 :
               # #print('-1 at x = ', fish['x'][i])
                continue
            if m == -5:
                x_lost.append(fish['x'][i])
                y_lost.append(fish['y'][i])
                h_lost.append(fish['h'][i])
                x.append(fish['x'][i])
                y.append(fish['y'][i])
                h.append(fish['h'][i])
                continue
            x.append(fish['x'][i])
            y.append(fish['y'][i])
            h.append(fish['h'][i])
            if m == 1:
               # #print('hofly added peak at x = ' , fish['x'][i])
                x.append(fish['x'][i] + fishisi[i-1])
                y.append( 0.5*(fish['y'][i]+fish['y'][i+1]))
                h.append(0.5*(fish['h'][i]+fish['h'][i+1]))
            elif m== 2:
                x.append(fish['x'][i] + fishisi[i])
                y.append( 0.5*(fish['y'][i]+fish['y'][i+1]))
                h.append(0.5*(fish['h'][i]+fish['h'][i+2]))
                x.append(fish['x'][i] + 2*fishisi[i-1])
                y.append( 0.5*(fish['y'][i]+fish['y'][i+2]))
                h.append(0.5*(fish['h'][i]+fish['h'][i+2]))
               # #print('added at x = ', fish['x'][i] + fishisi[i])
        retur[cl] = {'x':x,'y':y,'h':h}
        lost[cl] = {'xlost':x_lost,'ylost':y_lost,'hlost':h_lost}
       # filledpeaks =np.array(filledpeaks)
       # #print(filledpeaks.shape)
       # filledpeaks.
     return retur, lost

def calc_tsh_noise(peaks, data):
    heights = np.vectorize(lambda peak: peak.height)(peaks)
          #  peakx = xarray(peaks)
            #    peakxlist = peakx.tolist()
          #  #print('datenstdanfang: ', np.std(data))
          #  datatsh = np.mean(np.abs(data))#
          #  datatsh = 2* np.std(data)
          #  peakareas = [i for x in peakx for i in range(x-10, x+10) if (i < len(data))]
            #    peakareas = np.arange(peakx-10, peakx+10, 1)
          #  relevantdata = []
            #peakareas = np.unique(peakareas)
            # #print(len(peakareas), len(data), ' len peakarea and data' , datatsh)
            #relevantdata is the data without the areas around the peaks, to calculate the standard deviation of the noise
            #c = 0
    tsh = 0.1*np.std(heights) 
    
    #for i, dat in enumerate(data):
    #    if peakareas[c] == i and c<len(peakareas)-1:
    #        c+=1
    #        relevantdata.append(dat)       #####BULLSHIT!
    #    
    
    ##print(peakareas[0:20], data[0:20],'peakareas and data')
    # relevantdata = relevantdata[:][relevantdata < 4 * np.std(relevantdata)]
   
    #    tsh = np.std(relevantdata)
    # tsh = 5* np.std(data)
    ##print('threshold: ', tsh)
    return tsh

def dbscan(pcs, peaks, order, eps, min_samples, takekm, olddatalen):
    from sklearn.cluster import DBSCAN
    from sklearn import metrics
    from mpl_toolkits.mplot3d import Axes3D
    from sklearn.cluster import AgglomerativeClustering
    peaklist = peaks.list
    try:
        X = pcs[0:order].T
    except:
        X = pcs[order].T
    # #############################################################################
    # Compute DBSCAN
    db = DBSCAN(eps, min_samples).fit(X)
    from sklearn.cluster import KMeans
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_
    plot_steps = False
    #if not takekm:
        #for i, p in enumerate(peaklist): 
        #    print('label ', labels[i]) 
    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    #print('Estimated number of clusters: %d' % n_clusters_)
    # #############################################################################
    # Plot result
    # Black removed and is used for noise instead.
    unique_labels = set(labels)
    colors = [plt.cm.Spectral(each)
              for each in np.linspace(0, 1, len(unique_labels))]
    fig = plt.figure()
    if plot_steps:
        ax = fig.add_subplot(111, projection = '3d')
    classmeans = []
    for k, col in zip(unique_labels, colors):
        if k == -1:
            # Black used for noise.
            col = [0, 0, 0, 1]
        class_member_mask = (labels == k)
        xy = X[class_member_mask & core_samples_mask]
        # #print(xy)
        classmeans.append(xy)
        if plot_steps:
            ax.plot(xy[:, 0], xy[:, 1],xy[:,2], 'o', markerfacecolor=tuple(col),
                 markeredgecolor='k', markersize=14)
            xy = X[class_member_mask & ~core_samples_mask]
            ax.plot(xy[:, 0], xy[:, 1],  xy[:, 2], 'o', markerfacecolor=tuple(col),
                 markeredgecolor='k', markersize=6)
    if plot_steps:
        ax.set_title('Estimated number of clusters: %d' % n_clusters_)
        plt.show()
    classmeans = [np.mean(meanlist, axis = 0) for meanlist in classmeans]
    valid = False
    dist = 0
    for i,  clustermeans1 in enumerate(classmeans):
        for clustermeans2 in classmeans:
            tdist = np.sqrt(np.sum(np.square(clustermeans1-clustermeans2)))
            #if tdist > dist:
    #            dist = tdist
            #print('dist', dist)
        if dist>=0:
            valid = True
            if olddatalen > 0:
                alignlabels(labels, peaks, olddatalen)
            for i, p in enumerate(peaklist):
                pcclasses[peaknum] = labels[i]
            return valid
    if takekm:
        km = KMeans(n_clusters=3, n_init = 3, init = 'random', tol=1e-5, random_state=170, verbose = True).fit(X)
        core_samples_mask = np.zeros_like(km.labels_, dtype=bool)
        labels = km.labels_
        if takekm:
            for i, p in enumerate(peaklist): 
    #            print('label ', labels[i]) 
                pcclasses[peaknum] = p.pccl
        # Number of clusters in labels, ignoring noise if present.
        n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
        #print('Estimated number of clusters: %d' % n_clusters_)
        # #############################################################################
        # Plot result
        # Black removed and is used for noise instead.
        unique_labels = set(labels)
        colors = [plt.cm.Spectral(each)
                  for each in np.linspace(0, 1, len(unique_labels))]
        fig = plt.figure()
        ax = fig.add_subplot(111, projection = '3d')
        for k, col in zip(unique_labels, colors):
            if k == -1:
                # Black used for noise.
                col = [0, 0, 0, 1]
            class_member_mask = (labels == k)
            xy = X[class_member_mask]
    #        print(col)
            ax.plot(xy[:, 0], xy[:, 1],xy[:,2], 'o', markerfacecolor=tuple(col),
                 markeredgecolor='k', markersize=14)
        ax.set_title('Estimated number of clusters: %d' % n_clusters_)
        #plt.show()


        from sklearn.neighbors import kneighbors_graph
        knn_graph = kneighbors_graph(X, 15, include_self=False)
        ac = AgglomerativeClustering(linkage = 'complete', n_clusters = 3, connectivity = knn_graph).fit(X) 
        core_samples_mask = np.zeros_like(ac.labels_, dtype=bool)
        labels = ac.labels_
        if takekm:
            for i, p in enumerate(peaklist): 
                print('label ', labels[i]) 
                pcclasses[peaknum] = labels[i] 
        # Number of clusters in labels, ignoring noise if present.
        n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
        #print('Estimated number of clusters: %d' % n_clusters_)
        # #############################################################################
        # Plot result
        # Black removed and is used for noise instead.
        unique_labels = set(labels)
        colors = [plt.cm.Spectral(each)
                  for each in np.linspace(0, 1, len(unique_labels))]
        fig = plt.figure()
        ax = fig.add_subplot(111, projection = '3d')
        for k, col in zip(unique_labels, colors):
            if k == -1:
                # Black used for noise.
                col = [0, 0, 0, 1]
            class_member_mask = (labels == k)
            xy = X[class_member_mask]
            print(col)
            ax.plot(xy[:, 0], xy[:, 1],xy[:,2], 'o', markerfacecolor=tuple(col),
                 markeredgecolor='k', markersize=14)
        ax.set_title('Estimated number of clusters: %d' % n_clusters_)
        #plt.show()

def ampwalkclassify3_refactor(peaks,peaklist):                          # final classificator
    classamount = peaklist.classamount
    # for i in range(start, len(peaks)-start):
    lastofclass = peaklist.lastofclass        # dict of a lists of the last few heightvalues of a class, f.E ((1,[0.7,0.68,0.71]), (5, [0.2, 0.21, 0.21]))
    lastofclassx = peaklist.lastofclassx      # dict of a list of the last few x-values of a class
    a=0
    elem = 0
    thresholder = []
    comperr = 1
    classesnearby = peaklist.classesnearby     # list of the classes of the last n peaks (currently 12)  f.E:[1,2,1,2,1,3,2,1,...]
    classesnearbyx = peaklist.classesnearbyx   # list of the x-values of the last n peaks,               f.E:[13300, 13460, 13587, 13690, 13701, ...]
    classesnearbypccl = peaklist.classesnearbypccl # list of the pc-classified classes of the last n peaks
    classes = np.zeros((len(peaks[0])))
    pcclasses = peaks[3]
    positions = peaks[0]
    heights = peaks[1]

   # #print('nearbyclasses at start:' ,classesnearby, classesnearbyx)
   # for peak in peaks:
   #          peak.cl = peak.pccl+2 
   # peaklist.classlist =  np.vectorize(lambda peak: peak.cl, otypes=[object])(peaklist.list)
   # return peaks
    cl = 0
    maxdistance = 30000    #    Max distance to possibly belong to the same class
    factor = 1.6     # factor by which a peak fits into a class, f.E: classheight = 1 , factor = 2 => peaks accepted in range (0.5,2)
    c=0
    peakamount = len(peaks.T)
    #fish = ProgressFish(total = peakamount)
    for peaknum, p in enumerate(peaks.T):
        perc = str((peaknum*100)//peakamount)
    #    fish.animate(amount = "", dexextra = 'Partstatus: '+ ' '*(3-len(perc)) +perc + ' % (' + ' '*(4-len(str(peaknum)))+str(peaknum) + '/' + ' ' *(4-len(str(peakamount)))+str(peakamount) + '), Filestatus:')
        awc_btime = []
        if len(lastofclass) == 0:    # Dict with all classes, containing the heights of the last few peaks
            lastofclass[1] = deque()
            lastofclassx[1]= deque()
            lastofclass[1].append(heights[peaknum])
            lastofclassx[1].append(positions[peaknum])
            classesnearby.append(1)
            classesnearbyx.append(-1)
            classesnearbypccl.append(pcclasses[peaknum])
            classes[peaknum] = 1
            classamount += 1
            continue
        time1 = time.time()
        for i, cl in enumerate(classesnearby):
            if  (positions[peaknum]-classesnearbyx[i]) > maxdistance:
                classesnearby.pop(i)
                classesnearbyx.pop(i)
                classesnearbypccl.pop(i)
        lastofclassisis = []
        for i in classesnearby:
          #  print(i, classesnearby)
            lastofclassisis.append(np.median(np.diff(lastofclassx[i])))
        meanisi = np.mean(lastofclassisis)
        if 32000 > 20*meanisi> 6000:
            maxdistance = 20*meanisi
            #print(meanisi, maxdistance , 'maxdistance ----------------------------------------------------------------------------------------------')

        time2 = time.time()
        awc_btime.append(time2-time1) #0
        cl = 0  # 'No class'
        comperr = 1
        ##print('classesnearby at a peak', classesnearby)
        clnrby = np.unique(classesnearby)
        time1 = time.time()
#        classmean = 0
       #     if pcclasses[peaknum] == -1:
       #         factor = 1.2
       #     else:
       #         factor = 1.6

        for i in clnrby:
            #print('cl: ', i)
          #  if classesnearbypccl[classesnearby.index(i)] == -1:
          #      factor = 2.2
          #  else: factor = 1.6
            classmean = np.mean(lastofclass[i])
            logerror = np.abs(np.log2(heights[peaknum])-np.log2(classmean)) 
            abserror = np.abs(heights[peaknum]-classmean) 
            logthresh = np.log2(factor) 
            #ä#print(np.std(lastofclass[i])) absthresh = 0.5*classmean #  #print('test log', np.abs(np.log2(np.array([0.4,0.5,1,1.5,2,2.4]))-np.log2(np.array([1,1,1,1,1,1]))) ) #   abs(classmean*0.5)
            #relerror = error 
            relerror = logerror
            relabserror = abserror/thresh
           # if 1140 < p.num < 1150:
           #     print(p.num)
           #     print('for classes at one peak: classmean, height, abserror, thresh',
           #       classmean,heights[peaknum], logerror, logthresh)
            #print(len(classesnearbypccl), len(classesnearby))
            #print(classmean, heights[peaknum], logerror, logthresh, pcclasses[peaknum], classesnearbypccl[classesnearby.index(i)])
            if classesnearbypccl[classesnearby.index(i)] == pcclasses[peaknum] or pcclasses[peaknum] == -1:# or  
              if logerror < logthresh:     ## SameClass-Condition
                if relerror < comperr and (positions[peaknum]-classesnearbyx[classesnearby.index(i)])<maxdistance:
                    holdlastcl = cl
                    cl = i
#                    if comperr/error < np.log2(1.02):
#                        factor = 1.5
                    holdlastcomperr = comperr
                    comperr = relerror
        #    else:
                #print('classfitsnot-------------------------------------------------------------------------------------------------')
        #else:
        #    if cl == 0:
                #print('noclassfits!')
#        if cl!=0 and holdlastcl != 0 and holdlastcomperr - relerror < 0.05 * holdlastcomperr:
#            if np.std(np.diff(lastofclassx[holdlastcl])) < 0.05 * np.mean(np.diff(lastofclassx[holdlastcl])) and np.std(np.diff(lastofclassx[cl])) < 0.05 * np.mean(np.diff(lastofclassx[cl])):
#
#                isicompaeecl =  np.mean(np.diff(lastofclassx[holdlastcl]))
#                isithiscl = np.mean(np.diff(lastofclassx[cl]))
#                isithis =  positions[peaknum] - lastofclassx[cl][-1]
#                isicompare = positions[peaknum]  -lastofclassx[holdlastcl][-1]
#                thisisierror = (isithis-isithiscl)/isithiscl
#                compareisierror = (isicompare-isicomparecl)/isicomparecl
#                #print('isidecision!', isithis, isithiscl, isicompare, isicomparecl)
#                if thisisierror > 2*compareisierror:
#                    cl = holdlastcl

        time2 = time.time()
        awc_btime.append(time2-time1) #1
        time1 = time.time()
        if pcclasses[peaknum] != -1:
            if cl != 0 :
                #print(cl)
                if len(lastofclass[cl]) >= 3:
                    lastofclass[cl].popleft()
                if len(lastofclassx[cl]) >= 3:
                    lastofclassx[cl].popleft()
                lastofclass[cl].append(heights[peaknum])
                lastofclassx[cl].append(positions[peaknum])
                classes[peaknum] = cl
            else:                                    # Add new class
                cl = classamount+1
                #print('existingclasses: ', classamount) 
                classamount = cl
                
                #print('newclass: ----------------------------------------------------------------', cl)
                lastofclass[cl] = deque()
                lastofclassx[cl] = deque()
                lastofclass[cl].append(heights[peaknum])
                lastofclassx[cl].append(positions[peaknum])
                classes[peaknum] = cl
                classesnearby.append(cl)
                classesnearbyx.append(positions[peaknum])
                classesnearbypccl.append(pcclasses[peaknum])
            ##print('tatsaechlich: ', cl)         
            if len(classesnearby) >= 12: #kacke implementiert? 
                minind = classesnearbyx.index(min(classesnearbyx))
                del lastofclass[classesnearby[minind]]
                del lastofclassx[classesnearby[minind]]
                #print(classesnearby[minind], 'del')
                classesnearby.pop(minind)
                classesnearbyx.pop(minind)
                classesnearbypccl.pop(minind)
    #        for ind, clnrby in enumerate(reversed(classesnearby)):
    #            classesnearbyx
    #            del lastofclass[classesnearby[ind]]
    #           # del lastofclassx[classesnearby[minind]]
    #            classesnearby.pop(minind)
    #            classesnearbyx.pop(minind)
            try:
                ind=classesnearby.index(cl)
                classesnearbyx[ind] = positions[peaknum]
            #    #print(ind ,' --------------------------------------here -----------------------------')
            except ValueError:
                classesnearby.append(cl)
                classesnearbyx.append(positions[peaknum])
                classesnearbypccl.append(pcclasses[peaknum])
        else:
            if cl != 0:
                classes[peaknum] = cl
            else:
                cl = classamount+1
                #print('existingclasses: ', classamount) 
                classamount = cl
                #print('newclass: ', cl)
                lastofclass[cl] = deque()
                lastofclassx[cl] = deque()
                lastofclass[cl].append(heights[peaknum])
                lastofclassx[cl].append(positions[peaknum])
                classes[peaknum] = cl
                classesnearby.append(cl)
                classesnearbyx.append(positions[peaknum])
                classesnearbypccl.append(pcclasses[peaknum])
            if len(classesnearby) >= 12: #kacke implementiert? 
                minind = classesnearbyx.index(min(classesnearbyx))
                del lastofclass[classesnearby[minind]]
                del lastofclassx[classesnearby[minind]]
                #print(classesnearby[minind], 'del')
                classesnearby.pop(minind)
                classesnearbyx.pop(minind)
                classesnearbypccl.pop(minind)
    #        for ind, clnrby in enumerate(reversed(classesnearby)):
    #            classesnearbyx
    #            del lastofclass[classesnearby[ind]]
    #           # del lastofclassx[classesnearby[minind]]
    #            classesnearby.pop(minind)
    #            classesnearbyx.pop(minind)
            try:
                ind=classesnearby.index(cl)
                classesnearbyx[ind] = positions[peaknum]
            #    #print(ind ,' --------------------------------------here -----------------------------')
            except ValueError:
                classesnearby.append(cl)
                classesnearbyx.append(positions[peaknum])
                classesnearbypccl.append(pcclasses[peaknum])
           #     #print('classesnearby after a peak', classesnearby)
  #     for clnum, cls in enumerate(classesnearby): ## deleting almost identical classes (< % difference in amplitude)
  #         if cls == False:
  #             continue
  #         if True:
  #             continue
  #         compare = np.mean(lastofclass[cls])
  #         for i in classesnearby[clnum:-1]:
  #             if i== False:
  #                 continue
  #             if i != cls and abs(compare - np.mean(lastofclass[i])) < compare*0.01:   ## 
  #              #   #print(compare)
  #              #   #print( np.mean(np.vectorize(lambda peak: peak.height)(lastofclass[i])))
  #                 clindex = classesnearby.index(cls)
  #                 classesnearby[clindex] = False
  #                 classesnearbyx[clindex] = False
  #                 del lastofclass[cls]
  #                 del lastofclassx[cls]
  #                # cl = holdlastcl
  #                # if cl == cls:
  #                     
  #                     
  #                 #print('combinedsomeclasses that were similar', cl, cls)
        time2 = time.time()
  #      awc_btime.append(time2-time1) #2
  #      classesnearby = [cls for cls in classesnearby if cls != False] 
  #      classesnearbyx = [clx for clx in classesnearbyx if clx != False]
  # 
  # 
        #print('awc_btime ', awc_btime , ' newpeak-------------------------------------------------------- :')
    peaklist.lastofclass = lastofclass
    peaklist.lastofclassx = lastofclassx
    peaklist.classesnearby = classesnearby
    peaklist.classesnearbyx = classesnearbyx
    peaklist.classlist =  classes # np.vectorize(lambda peak: peak.cl, otypes=[object])(peaklist.list)
    peaklist.classamount = classamount
    peaks = np.append(peaks,classes[None,:], axis = 0)
    return peaks, peaklist

def joincc(peaklist,peaks):
    # connects classes that appear after each other... 
   # peaklist = peaks.list
    joinedsome = False
    classlist = peaks[4]
    peaksofclass = {}
    last = []
    connect = {}                              #connect classes in connect+
    classcount = dict.fromkeys(classlist, 0)
   ##print(classcount)
    #classcount = [0]*len(np.unique(classlist))
   # #print(np.unique(classlist))
    for cl in np.unique(classlist):
        peaksofclass[cl]= peaks[:,classlist == cl]
    for i in range(len(peaks[0])):    # i is the increasing index of the peaks
        p = peaks[:,i]
        poc = peaksofclass[p[4]]
        classcount[p[4]]+=1
        countclass = p[4]                   #the current class before it might be changed to the connected class
        if p[4] in connect:
            p[4] = connect[p[4]]                 #peakclass is changed to connected class
            #    #print('changed ', countclass, 'to', p.cl)
            joinedsome = True

        if len(poc) == classcount[countclass]:                     #the current peak is last peak of its class
            last = poc[-len(poc) if len(poc) <= 5 else 5:]  #the last peaks of the class
          #  #print('last: ', last)
            #mean_last = np.mean(np.vectorize(lambda peak: peak[2])(last))
            mean_last = np.mean(last[2,:])
            nextfirst = {}                  # the first peaks of the next coming class(es)
            #      #print('class: ', countclass, 'at x = ', p.x, 'mean_last: ', mean_last)
            for nexti in range(20):                           # the next 10 peaks are considered if they belong to the same classe
                if i + nexti >= len(peaks[0]): break
                inextp = peaks[:,i+nexti]
                if classcount[inextp[4]] == 0:                #current peak is first peak of its class
                 #   #print('found a new begin! its class:' , inextp.cl)
                    ponc = peaksofclass[inextp[4]]            #
                    nextfirst[inextp[4]] = ponc[0:len(ponc) if len(ponc) <= 5 else 5]
                 #   #print(np.mean(np.vectorize(lambda peak: peak.height)(nextfirst[inextp.cl])))
                    # #print(nextfirst)
            compare = 1
            c = 0
            nextclass = -1
            for nextcl, first in nextfirst.items():
                mean_nextfirst = np.mean(first[2,:])#np.mean(np.vectorize(lambda peak: peak.height)(first))
            #    #print(mean_nextfirst)
                error = abs(mean_nextfirst - mean_last)/(mean_nextfirst)
                if error < 1:
                    if compare < error:
                        continue
                    compare = error
                    if nextcl in connect:                          #if the peak that ist considered belongs to a class, that is already supposed to be connected to the current class
                        pocc = peaksofclass[connect[nextcl]]       #peaks of the currently supposed connected class 
                        if (   abs(mean_nextfirst - np.mean(pocc[-len(pocc) if -len(pocc) <= 5 else 5:][2]))
                             < abs(mean_nextfirst - mean_last) ):
                            continue
                    nextclass = nextcl
            if nextclass != -1:
                connect[nextclass] = p[4]
                #    #print('connect ', p.cl , ' and ', nextcl)
    for cl in peaklist.classesnearby:
        if cl in connect:
          #  #print('cl, connect', cl, connect[cl])
            peaklist.classesnearby[peaklist.classesnearby.index(cl)] = connect[cl]
            peaklist.lastofclass[connect[cl]]=peaklist.lastofclass[cl]
            peaklist.lastofclassx[connect[cl]]= peaklist.lastofclassx[cl] 
    peaklist.classlist = peaks[4]
    return joinedsome
   # for poc in peaksofclass:  
   #     if len(poc) >= 3:
   #         newlast = poc[-3:]
   #         first = poc[:3]
   #     else:
   #         newlast = poc[-len(poc):]
   #         first = poc[:len(poc)] 
   #     if last != []:
   #          if abs(np.mean(first) - np.mean(last)) <  0:
   #              #print('oh')

def discardwaves_refactor(peaks, data):
     deleteclasses = []
     for cl in np.unique(peaks[3]):
         peaksofclass = peaks[:,peaks[3] == cl]
         isi = np.diff(peaksofclass[0])
         isi_mean = np.mean(isi)
        # #print('isismean',isi_mean)
         widepeaks = 0
        # #print('width',peaksofclass[2].width)
         isi_tenth_area = lambda x, isi:np.arange(np.floor(x-0.1*isi),np.ceil(x+0.1*isi),1, dtype = np.int)
         for p in peaksofclass.T:
             data = np.array(data)
             try:
                 for dp_around in data[isi_tenth_area(p[0],isi_mean)]:#np.floor(p[0]-0.1*isi_mean), np.ceil(p[0]+0.1*isi_mean),1)]:#
                    if dp_around <= p[1]-p[2]:
                       break
             except IndexError:
                 pass
         else:
             widepeaks+=1
            ## p.isreal_pleateaupeaks()
         if widepeaks > len(peaksofclass)*0.5:
             deleteclasses.append(cl)
     for cl in deleteclasses:
            peaks = peaks[:,peaks[3]!=cl]
     return peaks

def smallclassdiscard(peaks, mincl):
    classlist = peaks[3]
    smallclasses = [cl for cl in np.unique(classlist) if len(classlist[classlist
                                                                     == cl]) <
                 mincl]
    delete = np.zeros(len(classlist))
    for cl in smallclasses:
        delete[classlist == cl] == 1
        peaks = peaks[:,delete != 1]
    return peaks

def makepeak(data_x,cutsize, maxwidth, peakx, ltr, data_ltr, rtr, data_rtr, num, minhlr):
        #if len(data) > peakx + cutsize/2:
        return Peak(peakx, data_x, maketr(data_ltr, ltr), maketr(data_rtr, rtr), maxwidth, num, minhlr)#data[peakx-cutsize/2:peakx+cutsize/2], num)
        #else:
         #   return Peak(peakx, data[peakx],
          #              maketr(data, ltr),
           #             maketr(data, rtr),
            #            maxwidth,
             #           #data[peakx-cutsize/2:-1],
              #          num)

def maketr(data_x, x):
        if x is not None:
            return Tr(x,data_x)
        else:
            return None

def makepeaklist(pkfirst, data, pk, tr, cutsize, maxwidth): 
        peaklist = np.empty([len(pk)], dtype = Peak)
        trtopk = pkfirst
        pktotr = 1-pkfirst
        trlen = len(tr)
        pklen = len(pk)
        minhlr = lambda i, mwl, mwr : min(
                                             abs( data[pk[i]] - min( data[pk[i]-mwl:pk[i]] ) if len(data[pk[i]-mwl:pk[i]]) > 0 else 0 )
                                                 ,
                                                            abs(  data[pk[i]]-   min(
                                                                                       data[pk[i]:pk[i]+mwr]    )  if len(data[pk[i]:pk[i]+mwr]) > 0 else 0   )
                                                                                                              )
        #print(min( data[pk[0]-0:pk[2]]) )

        if pktotr == 0:
            peaklist[0] = makepeak(data[0], cutsize, maxwidth, pk[0], None, None,  tr[pktotr], data[pktotr],  0, minhlr(0, 0, maxwidth))
        else:
            peaklist[0] = makepeak(data[0], cutsize, maxwidth, pk[0], 
                                   tr[-trtopk],
                                   data[-trtopk], tr[pktotr], data[pktotr], 
                                   0, minhlr(0, min(maxwidth, 
                                                    pk[0]-tr[-trtopk]) , maxwidth))
        for i in range(1,pklen-1):
                                   peaklist[i] = makepeak(data[pk[i]], cutsize, maxwidth, pk[i], tr[i-trtopk], data[tr[i-trtopk]], tr[i+pktotr],data[tr[i+pktotr]], i, minhlr(i, maxwidth, maxwidth))
        if pktotr == 0 and pklen <= trlen:
            peaklist[pklen-1] = makepeak(data[pk[pklen-1]],cutsize, maxwidth, pk[pklen-1], tr[pklen-trtopk-1], data[pklen-trtopk-1], tr[pklen+pktotr-1], data[pklen+pktotr-1],  i, minhlr(pklen-1, maxwidth, min(maxwidth, tr[pklen+pktotr-1]-pk[pklen-1])))
        else:
            peaklist[pklen-1] = makepeak(data[pk[pklen-1]],cutsize, maxwidth, pk[pklen-1], tr[pklen-trtopk-1],data[pklen-trtopk-1], None, None,  pklen-1, minhlr(pklen-1, maxwidth, 0))
        return peaklist

#def doublepeaks(peaks, peakwidth):
#    dif2 = peaks[1].x-peaks[0].x
#    if dif2 > 5* peakwidth:
#            peaks[0].real = False
#    for i in range(1,len(peaks)-1):
#        dif1 = dif2
#        dif2 = peaks[i+1].x-peaks[i].x
#        if dif1 > 5* peakwidth and dif2 > 5* peakwidth:
#            peaks[i].real = False
#    if dif2 > 5* peakwidth:
#        peaks[len(peaks)-1] = False
#    return peaks

def discardunrealpeaks(peaklist):
    peaks = peaklist[:][np.vectorize(lambda peak: peak.real, otypes=[object])(peaklist) == True]
    for i, p in enumerate(peaks):
       pass
  # p.num = i
    return peaks

def discardnearbypeaks(peaks, peakwidth):
        peaksx = xarray(peaks)
        pkdiff = np.diff(peaksx)
        # peakwidth = avg_peakwidth(pknum,tr)
        pknumdel= np.empty(len(peaksx))
        pknumdel.fill(False)
#        peaksy = yarray(peaks)
        peaksh = heightarray(peaks)
        for i,diff in enumerate(pkdiff):
         #  #print(peaks[i].height)
           if diff < peakwidth: #* peaks[i].height:   ### Trial Error
               if peaksh[i+1] > 1.01 *peaksh[i] :
                   pknumdel[i] = True
               else:
    #               print(peaksh[i],peaksh[i+1])
                   pknumdel[i+1] = True
        peaks = peaks[pknumdel!=True]
        for i, p in enumerate(peaks):
            p.num = i
        return peaks

def interpol(data, kind):
    #kind = 'linear' , 'cubic'
    width = len(data)
    x = np.linspace(0, width-1, num = width, endpoint = True)
    return interp1d(x, data[0:width], kind , assume_sorted=True)

def cutcenter(peak):
    p = peak
    cut = p.cut
    pl=p.distancetoltr
    pr=p.distancetortr
    if pl is None:
        pl = 10
        tx = p.x-10
    else: tx = p.ltr.x
    if pr is None:
        pr = 10
    if pl < p.maxwidth and pr > 1:

        width=len(cut)
       # #print('distancetoltr',pl)
        peakshape = cut
        interpolfreq = 1
        xnew = np.linspace(0,len(peakshape)-1, len(peakshape)*interpolfreq, endpoint= True)
        curvyf = interpol(peakshape)
        curvy= curvyf(xnew)
        #px = p.cutsize/2 * 4
        #left = px - (5*4)
        #plt.plot(xnew, curvy)
        #x_0 = optimize.fsolve(curvyf, 1.0)
        #  f = interp1d(x, y)
       # f2 = interp1d(range(width), data[x:x+width], kind='cubic')
        ##xnew = np.linspace(0, width-1, num = width*4, endpoint = True)
        ##print(xnew)
       # plt.plot(xnew,f2(xnew))
        ##print("show")
        #plt.show
        trx = (p.cutsize/2 - (p.x - tx) )
        if trx >0 :
            xstart = trx
        else:
            xstart = 0
       # #print('pkx: ', p.x, 'ltrx: ', p.ltr.x)
       # #print('trx in intpol', x)
        x = xstart
        if curvyf(x) < 0:
            left = 0
            right= 0
            while(x < width-1 and curvyf(x) < 0) :
                left = x
          #      #print(curvyf(x))
                x+=0.25
                right = x
            #    #print('x: ', x , 'left, right: ', curvyf(left), curvyf(right))
            x = left+(1-curvyf(right)/(curvyf(right)-curvyf(left)))*1/interpolfreq
         #   #print(x)
        else:
            x = 0
       # #print(x_int)
       # plt.scatter(xstart, curvyf(xstart), marker = 'x', s=150, zorder=2, linewidth=2, color='red')
       # plt.scatter(x, curvyf(x), marker='x', s=150, zorder=2, linewidth=2, color='black')
       # plt.show
       # #print(x_int)
        #p.relcutcenter = (p.ltr.x + x_int)-p.x
        ##print('cent',p.relcutcenter)
        #return (p.ltr.x + x_int)-p.x
        
    #    while(data[x]>0)
    else:
        x= 0

    return x

def relcutarray(peaks):
    return np.vectorize(lambda peak: peak.relcutcenter)(peaks)

def xarray(peaks):
    if len(peaks)>0:
        peakx = np.vectorize(lambda peak: peak.x)(peaks)
        return peakx
    else: return []

def yarray(peaks):
    if len(peaks)>0:
        return np.vectorize(lambda peak: peak.y)(peaks)
    else: return []
    
def heightarray(peaks):
    if len(peaks)>0:
        return np.vectorize(lambda peak: peak.height)(peaks)
    else: return []

def clarray(peaks):
    if len(peaks)>0:
        return np.vectorize(lambda peak: peak.cl)(peaks)
    else: return []
def pcclarray(peaks):
    if len(peaks)>0:
        return np.vectorize(lambda peak: peak.pccl)(peaks)
    else: return []

def peakxarray( ):
        peakx = np.empty([len])
        peakx = np.vectorize(lambda peak: peak.x)(peaks)
        return peakx

def peakyarray( ):
        peaky= np.empty([len])
        return np.vectorize(lambda peak: peak.y)(peaks)


def classify( ):
        #template = peaks[0]
        meanfit =  np.mean(np.vectorize(fit, otypes=[object])(template,peaks))
        for p in peaks:
            if fit(template,p) < meanfit:
               # #print('classified ', fit(template,p) , ' meanfit: ' , meanfit)
                p.currentclass = 1

def classifyhiker(template, peaks):
        meanfit =  np.mean(np.vectorize(fitinterpol2, otypes=[object])(template,peaks))
        #toclassify = peaks.tolist()
        firstnot = 0
        for c in range(1,5):
            first = True
            template = peaks[firstnot]
            for i, p in enumerate(peaks[firstnot:]):
                if p.currentclass == 0:
                    if fitinterpol2(template,p) < meanfit:
                      #  #print('peak number ' , i, 'classified  as ', c,  fit(template,p) , ' meanfit: ' , meanfit)
                        p.currentclass = c
                        template = p
                    elif first == True:
                       # #print('peak number ' , i, 'classified  as First! ', c,  fit(template,p) , ' meanfit: ' , meanfit)
                        firstnot = i
                        first = False
                    else:
                        None
                        ##print('peak number ' , i, 'classified  as not classified!',  fit(template,p) , ' meanfit: ' , meanfit)
        return peaks


  #  def Templatefitnext( , number, templnum):
   #     for p in peaks:
    #        if fit(peaks[templnum], p) < fitparameter:        

def cut_snippets(data, peaklist, rnge):
    snippets = []
    positions = xarray(peaklist) 
    heights = heightarray(peaklist) 
    for pos in positions:
        snippets.append(data[(pos+rnge[0]):(pos+rnge[1])])
    scaledsnips = np.empty_like(snippets)
    for i, snip in enumerate(snippets):
        top = -rnge[0]
       # plt.plot(snip)
        scaledsnips[i] = snip * 1/heights[i] 
        #plt.plot(scaledsnips[i])
        #  print('plted')
#    plt.show()
    #print('1')
    alignedsnips = np.empty((len(snippets), (rnge[1]-rnge[0])*10-30-10))
    standardized = np.empty((len(snippets), (rnge[1]-rnge[0])*10-10))
    intfact = 10
    for i, snip in enumerate(scaledsnips):
       if len(snip) < ((rnge[1]-rnge[0])): 
            if i == 0:
                snip =np.concatenate([np.zeros([((rnge[1]-rnge[0]) - len(snip))]),np.array(snip)])
            if i == len(scaledsnips):
                snip = np.concatenate([snip, np.zeros([((rnge[1]-rnge[0])-len(snip))])])
            else:
    #            print('this')
                snip = np.zeros([(rnge[1]-rnge[0])]) 
       interpoled_snip = interpol(snip)(np.arange(0, len(snip)-1, 1/intfact)) if len(snip) > 0 else np.zeros([(rnge[1]-rnge[0]-1)*intfact ]) #interpolfactor 10
       
       intsnipheight   = np.max(interpoled_snip) - np.min(interpoled_snip)
       if intsnipheight == 0:
           intsnipheight = 1
       interpoled_snip = (interpoled_snip - max(interpoled_snip))* 1/intsnipheight
       standardized[i] = interpoled_snip
    #print('2')
    mean = np.mean(standardized, axis = 0)
    #plt.plot(mean)
#    plt.show()
    #plt.plot(mean[10*-rnge[0]-10*5:-10*rnge[1]+21])
#    plt.show()
    meantop = np.argmax(mean)
    for i, snip in enumerate(standardized):
        #plt.show()
        interpoled_snip = snip #standardized[i]
        cc = crosscorrelation(interpoled_snip[15:-15], mean)
        #cc = crosscorrelation(interpoled_snip[15 + 10*-rnge[0]-10*7:-15+ -10*rnge[1]+ 31], mean[10*-rnge[0]-10*7:-10*rnge[1]+31])
        #plt.plot(interpoled_snip[15 + 10*-rnge[0]-10*7:-15+ -10*rnge[1]+ 31]) 
        #top = np.argmax(interpoled_snip)
        #offset = meantop - top
        #if not(-15 <= offset <= 15): offset = 0
        offset = -15 + np.argmax(cc)
        interpoled_snip = interpoled_snip[15-offset:-15-offset] if offset != -15 else interpoled_snip[30:]
        #print(offset)
        #plt.plot(interpoled_snip)
        if len(interpoled_snip[~np.isnan(interpoled_snip)])>0:
            alignedsnips[i] = interpoled_snip
    #plt.show()
   # print('3')
    return snippets, alignedsnips



def fit(templ, peak):
        fit = np.sum(np.square(templ.cut - peak.cut))
        return fit
    
def fitinterpol2(templ,peak):
    t = templ
    p = peak
    if p.real and t.real:
        fit = np.sum(np.square(t.cutaligned-p.cutaligned))
    else:
        fit = 0
    return fit

    

def fitinterpol( templ, peak):
    t = templ
    p = peak
    if p.real:
        centerp = cutcenter(p)
        centert = cutcenter(t)
        shiftp = centerp-p.cutsize/2
        shiftt = centert-t.cutsize/2
        
        if shiftp > -5:
            shiftp  = min(5, 5+centerp-p.cutsize/2)
        else: shiftp = 0
    
        if shiftt > -5:
            shiftt  = min(5, 5+centert-t.cutsize/2)
        else: shiftt = 0
        
        xnew = np.linspace(0,p.cutsize-11, (p.cutsize-1) * 4,endpoint = True)
        #peak_interpoled = interpol(p.cut)(xnew)
        #plt.plot(xnew, interpol(p.cut)(xnew+shift))
      #  #print(interpol(templ.cut)(xnew+shiftt)-interpol(p.cut)(xnew+shiftp))
        fit = np.sum(np.square(interpol(templ.cut)(xnew+shiftt)-interpol(p.cut)(xnew+shiftp)))
    else:
        fit = 0
    return fit 
    

def plotdata(peaks, data):
        x = xarray(peaks)
        y = yarray(peaks)
        plt.plot(range(len(data)),data)
        plt.plot(x, y, '.r', ms=20)
        #for p in peaks:
       #     #print(p.height, p.x, p.y, p.distancetoltr, p.distancetortr, p.nexttrdistance)
     #   plt.plot(tr, data[tr], '.g', ms=20)
        plt.show()


def plotdatabyx(peaksx, data):
        x = peaksx
        y = data[peaksx]
        plt.plot(range(len(data)),data)
        plt.plot(x, y, '.r', ms=20)
        plt.show()
        #for p in peaks:
       #     #print(p.height, p.x, p.y, p.distancetoltr, p.distancetortr, p.nexttrdistance)
     #   plt.plot(tr, data[tr], '.g', ms=20)

def plotpeak(peaks):
        #plt.plot(peaks), cutpeaks) #bei betrachtung aller blocks zu groß!
        for p in peaks:
            plt.plot(range(p.cutsize),p.cut)
        #plt.plot(pk, x[pk] , '.r', ms=20)
        plt.show()   


def periodicinclass(peaks, cl):
    noiselist = []
    classlist = np.vectorize(lambda peak: peak.cl, otypes=[object])(peaks)
    peaks = xarray(peaks)
    peaks = peaks[:][classlist == cl]
    periodic = []
    periodiccollector = []
    error2 = []
    isperiodic = True
    b=1
    c=2
    ctofar = False
    compdif = 0
    dif = 0
    count = 1
    foundtriple = False
    next = 0
    for i in range(len(peaks)-1):
        if i != next: continue
      #  #print(i, 'foundtriple', foundtriple)
        error2 = []
        b=1
        c=0
        A = peaks[i]
        B = peaks[i+b]
        compdif = dif
        while foundtriple == True and count <= 3 and i+1 < len(peaks)-1:
            while B-A < compdif*1.5 and i+b+1 < len(peaks)-1:
              #  #print('newdif: ', B-A, 'olddif:' , dif)
                if abs((B-A) - compdif) < compdif*0.4:
                    error2.append(abs((B-A) - dif))
                b+=1
                B = peaks[i+b]
            if len(error2) > 0:
                bestB = error2.index(min(error2))
                B = peaks[i+1 + bestB]
                periodic.append(B)
                dif = 0.5*(dif + (B-A))
              #  #print('match found')
                b = 1+bestB
                break
            else:
                count+=1 
                compdif = dif*count
        else:
            if foundtriple == True:
              #  #print('no further match found, ')
                isperiodic = False
                
            
            
                
        while foundtriple == False and i+c< len(peaks)-1:
            while i+c < len(peaks)-1:
                A = peaks[i]
                B = peaks[i+b]
                C = peaks[i+c]
                dif1 = B - A
                dif2 = C - B
                if (C-B > (B-A)*1.5):
                   break
                if abs(dif1 - dif2) < dif1*0.4:
                    error2.append(abs(dif1-dif2))
                c +=1
                #C = peaks[i+c]                         # C weiterlaufenlassen, bis zu weit
            else:
                if len(error2) == 0:
              #      #print('no triple found')
                    isperiodic = False
            if len(error2) > 0:
                bestC = error2.index(min(error2))
                C = peaks[i+2 + bestC]
                c = 2+ bestC
                periodic.extend((A,B,C))
                dif1 = B - A
                dif2 = C - B
              #  #print('dif1: ', dif1, 'dif2: ', dif2)
                dif = 0.5*(dif2+dif1)
                foundtriple = True
             #   #print('triple found', i+c, 'dif : ', dif)
            else:
                error2 = []                             # B weiterlaufen lassen, C reset auf B+1
                b +=1
                c = b+1
        
        if isperiodic == False:
            if len(periodic) > 3:
                periodiccollector.append(periodic)
                isperiodic = True
                periodic = []
        if c!=0:
            next = i+c
        else:
            next = i+b 
    if len(periodiccollector) > 0:
      #  for i in range(len(periodiccollector)):
      #      #print('collector ', i, periodiccollector[i])
        return periodiccollector
    else:
        #print('no periodicity found')
        return []    



def noisediscard(peaklist, tsh_n, ultimate_threshold):
    detected_noise = False
   ##print('noisetsh: ', tsh_n)
    for p in peaklist.list:
        
        if p.height < tsh_n or p.height < ultimate_threshold:
            p.noise = True
            detected_noise = True
    peaklist.list = peaklist.list[:][np.vectorize(lambda peak: peak.noise, otypes=[object])(peaklist.list) == False]
       # #print(peaks)
        # for cl in classlist:
       #     diff = np.vectorize(lambda peak: peak.x, otypes=[object])(peaks[:][classlist == cl])
       #     meandiff = np.mean(diff)
       #     msecompare = np.mean(np.square(diff-(diff*0.8)))
       #     mse = np.mean(np.square(diff-meandiff))
       #     if mse > msecompare:
       #         noiselist.append(cl)
       # for p in peaks:
            #if p.cl in noiselist:
      #      if p.height < 0.1:
      #          p.noise = True
      #  peaks = peaks[:][np.vectorize(lambda peak: peak.noise, otypes=[object])(peaks) == False]
      #  return peaks
    return detected_noise


def plotPCclasses_ref(peaks, data):
    plt.plot(range(len(data)),data, color = 'black')
    print(peaks)
    classlist = np.array(peaks[3],dtype = 'int')
    cmap = plt.get_cmap('jet')
    colors =cmap(np.linspace(0, 1.0, 3000)) #len(np.unique(classlist))))
    np.random.seed(22)
    np.random.shuffle(colors)
    colors = [colors[cl] for cl in np.unique(classlist)]
    print('classlist', np.unique(classlist)) 
    # plt.plot(xarray(peaksofclass), yarray(peaksofclass), '.c', ms=20)
    #  x=0
#    if len(classlist)>0:
       # #print(classlist)
       # #print('classes: ' , np.unique(classlist))
        #from collections import Counter
        #count = Counter(classlist)
      #  #print('longest class: ',  count.most_common()[0])
    for num, color in zip(np.unique(classlist), colors):
        if num == -1 :
            color = 'black'
        peaksofclass = peaks[:,classlist == num]
        print(num)
        plt.plot(peaksofclass[0], peaksofclass[1], '.', color = color,   ms =20)
        #plt.scatter(peaks[0], peaks[2])
   # for p in peaks:
   #     plt.text(p.x, p.y, p.num)
    #plt.show()

    print('show pcclasses')
    plt.show()
    plt.close()
    
def plotampwalkclasses_refactored(peaks, data):
    plt.plot(range(len(data)),data, color = 'black')
    classlist = np.array(peaks[3],dtype=np.int)
    cmap = plt.get_cmap('jet')
    colors =cmap(np.linspace(0, 1.0, 3000)) #len(np.unique(classlist))))
    np.random.seed(22)
    np.random.shuffle(colors)
    colors = [colors[cl] for cl in np.unique(classlist)]
    # plt.plot(xarray(peaksofclass), yarray(peaksofclass), '.c', ms=20)
    #  x=0
#    if len(classlist)>0:
       # #print(classlist)
       # #print('classes: ' , np.unique(classlist))
        #from collections import Counter
        #count = Counter(classlist)
      #  #print('longest class: ',  count.most_common()[0])
    for cl, color in zip(np.unique(classlist), colors):
        peaksofclass = peaks[:,classlist == cl]
        #xpred = linreg_pattern(peaksofclass[0:3])
        #for p in peaksofclass[0:3]:
        #            #print(p.x)
        ##print(xpred, peaksofclass[3].x)            
                
        #if len(peaksofclass) > 1000:
        #    plt.plot(xarray(peaksofclass), yarray(peaksofclass), '.', color = 'red',   ms =20)
        #else:
        
        plt.plot(peaksofclass[0],peaksofclass[1], '.', color = color,   ms =20)
        plt.scatter(peaksofclass[0], peaksofclass[2])
   # for p in peaks:
   #     plt.text(p.x, p.y, p.num)
    plt.show()

   #  plt.show()
    plt.close()
    

def crosscorrelation(sig, data):
    autocorr = signal.fftconvolve(data, sig[::-1],  mode='valid')
    return autocorr

def plottemplatefits(data, peaks, tr, templnum):
      # 
        plotdata(peaks, data, tr)
        plt.plot(range(len(data)),data)
        classes = np.vectorize(lambda peak: peak.currentclass, otypes=[object])(peaks)
        class1 = peaks[:][classes == 1 ]
        if len(class1) > 0:
            plt.plot(xarray(class1), yarray(class1), '.r', ms=20)
        class2 = peaks[:][classes == 2 ]
        if len(class2) > 0:
            plt.plot(xarray(class2), yarray(class2), '.g', ms=20)
        class3 = peaks[:][classes == 3 ]
        if len(class3) > 0:
            plt.plot(xarray(class3), yarray(class3), '.c', ms=20)
        class4 = peaks[:][classes == 4 ]
        if len(class4) > 0:
            plt.plot(xarray(class4), yarray(class4), '.y', ms=20)
        
       # for p in peaks:                                       # <--
       #     plt.text(p.x , p.y, p.num)

        # plt.plot(tr, data[tr], '.g', ms=20)
        plt.show()

def linreg_pattern(peaks):
    from sklearn import datasets, linear_model
    from sklearn.metrics import mean_squared_error, r2_score

    peaksx = xarray(peaks)
    peaksx = peaksx.reshape(-1,1)
    #peaksh = heightarray(peaks)
    #peakx = peak.x
    # Create linear regression object
    regr = linear_model.LinearRegression()
    numbers = np.arange(len(peaks)).reshape(-1,1)
    # Train the model using the training sets
    regr.fit(numbers, peaksx)
    
    # Make predictions using the testing set
    peakx_pred = regr.predict(len(peaks))
    # # The coefficients
    # #print('Coefficients: \n', regr.coef_)
    # # The mean squared error
    # #print("Mean squared error: %.2f"
    #       % mean_squared_error(diabetes_y_test, diabetes_y_pred))
    # # Explained variance score: 1 is perfect prediction
    # #print('Variance score: %.2f' % r2_score(diabetes_y_test, diabetes_y_pred)

        
    # Plot outputs
    #plt.scatter(peaksx, peaksh,  color='black')
    #plt.scatter(peakx, peakh_pred, color='blue')

    #plt.xticks(())
    #plt.yticks(())

   # plt.show()

    return peakx_pred

def linreg(peaks, peak):
    from sklearn import datasets, linear_model
    from sklearn.metrics import mean_squared_error, r2_score

    peaksx = xarray(peaks)
    peaksx = peaksx.reshape(-1,1)
    peaksh = heightarray(peaks)
    peakx = peak.x
    # Create linear regression object
    regr = linear_model.LinearRegression()

    # Train the model using the training sets
    regr.fit(peaksx, peaksh)

    # Make predictions using the testing set
    peakh_pred = regr.predict(peakx)
 
    # # The coefficients
    # #print('Coefficients: \n', regr.coef_)
    # # The mean squared error
    # #print("Mean squared error: %.2f"
    #       % mean_squared_error(diabetes_y_test, diabetes_y_pred))
    # # Explained variance score: 1 is perfect prediction
    # #print('Variance score: %.2f' % r2_score(diabetes_y_test, diabetes_y_pred)

        
    # Plot outputs
    #plt.scatter(peaksx, peaksh,  color='black')
    #plt.scatter(peakx, peakh_pred, color='blue')

    #plt.xticks(())
    #plt.yticks(())

   # plt.show()



    return peakh_pred

def wp_transform(x):
    import pywt
    wp = pywt.WaveletPacket(data=x, wavelet='haar', mode='symmetric')
    print('maxlevel: ', wp[''].maxlevel)
    return (np.array([node.data for node in wp.get_level(wp[''].maxlevel, 'freq')])).flatten()

def wpfeats(snips):
    size = len(wp_transform(snips[0]))
    wp = np.empty([len(snips), size])
    for i, snip in enumerate(snips):
        print(wp_transform(snip))
        wp[i] = (wp_transform(snip))
    #wp = wp.T
    print(wp[0])
    wpcoef = wp.T
    print(wp[0])
    from sklearn.preprocessing import StandardScaler
    wpcoef = StandardScaler().fit_transform(wpcoef) 
    coeffvalues = []
    for coeff in wpcoef:
        stat, crit, sig = stats.anderson(coeff, dist = 'norm')
       # coeffvalues.append(stat)
        coeffvalues.append(np.sum(np.abs(coeff))) 
    coeffvalues = np.array(coeffvalues)
    coeffs = np.argsort(coeffvalues)[::-1][:10]
    print(coeffvalues[coeffs])
    return wp.T[coeffs]




def pc(cutsnippets, peaklist):
    # (observations, features) matrix
     M = np.empty([len(cutsnippets), len(cutsnippets[0])])
     for i, snip in enumerate(cutsnippets):
         M[i] = snip[:]
     from sklearn.preprocessing import StandardScaler
     StandardScaler().fit_transform(M)
    # #print(M.shape, ' Mshape')
    # singular value decomposition factorises your data matrix such that:
    # 
    #   M = U*S*V.T     (where '*' is matrix multiplication)
    # 
    # * U and V are the singular matrices, containing orthogonal vectors of
    #   unit length in their rows and columns respectively.
    #
    # * S is a diagonal matrix containing the singular values of M - these 
    #   values squared divided by the number of observations will give the 
    #   variance explained by each PC.
    #
    # * if M is considered to be an (observations, features) matrix, the PCs
    #   themselves would correspond to the rows of S^(1/2)*V.T. if M is 
    #   (features, observations) then the PCs would be the columns of
    #   U*S^(1/2).
    #
    # * since U and V both contain orthonormal vectors, U*V.T is equivalent 
    #   to a whitened version of M.
    
     U, s, Vt = np.linalg.svd(M, full_matrices=False)
     V = Vt.T
    
    # PCs are already sorted by descending order 
    # of the singular values (i.e. by the
    # proportion of total variance they explain)
     S = np.diag(s)
    # PC = (s*V)
    # PCs:
     #print(U.shape)
     #print(S.shape)
     #print(V.shape)
     #print(s[0], U[0,:])
     
     #PC1 = (s[0] * U[:,0])
     #PC2 = (s[1] * U[:,1])
     #for i, p in enumerate(peaklist):
     #    p.pc1 = PC1[i]
     #    p.pc2 = PC2[i]
     
     #mu = peaks.mean(axis=0)
     #fig, ax = plt.subplots()
     #ax.scatter(xData, yData)
     #for axis in U:
     #    start, end = mu, mu + sigma * axis
     #    ax.annotate(
     #        '', xy=end, xycoords='data',
     #        xytext=start, textcoords='data',
     #        arrowprops=dict(facecolor='red', width=2.0))
     #ax.set_aspect('equal')
     #plt.show()
    
     
    # if plot_steps:
    #     plt.scatter(PC1, PC2)
    #     plt.show()
    
    # PCData1 = (U[:,0]*M)
    # PCData2 = (U[:,1]*M)
    # plt.scatter(PCData1, PCData2)
    # plt.show()
     
     #plt.scatter(U[:,0],U[:,1])
     #plt.show()
     #print('done')
     #return PC
    
     # if we use all of the PCs we can reconstruct the noisy signal perfectly
     #Mhat = np.dot(U, np.dot(S, V.T))
     #print('Using all PCs, MSE = %.6G' %(np.mean((M - Mhat)**2)))
      
     #plt.show()
     return S@U.T

def gettime(x, samplerate, starttime):
        startm = int(starttime[-2:])
        starth = int(starttime[:-2])
        seconds = x/samplerate
        m, s = divmod(seconds, 60)
        m = m + startm
        h, m = divmod(m, 60)
        h = h+starth
        return "%d:%02d:%02d" % (h, m, s)

def connect_blocks(oldblock):
    newblock = Peaklist([])
    newblock.lastofclass    = oldblock.lastofclass
    newblock.lastofclassx    = oldblock.lastofclassx
    newblock.classesnearby  = oldblock.classesnearby
    newblock.classesnearbypccl  = oldblock.classesnearbypccl
    newblock.classesnearbyx = [clnearbyx - oldblock.len for clnearbyx in oldblock.classesnearbyx]
    newblock.classamount = oldblock.classamount
    return newblock
   ##print('classesnearbyx! old, new ' , oldblock_len,oldblock.classesnearbyx , newblock.classesnearbyx)

if __name__ == '__main__':
    main()



# deleted Code, but unsure if really want to delete:

             #nix   #print( b.data_arrays)
                
           #     for cl in np.unique(cllist):

                  #  currentfish_x = x[:][cllist == cl]
                  #  currentfish_y = y[:][cllist == cl]
                  #  currentfish_h = x[:][cllist == cl]

                    
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
               



# in analyseEods    
# in analyseEods    classlist = eods[3] #np.vectorize(lambda peak: peak.cl, otypes=[object])(worldpeaks.list)
# in analyseEods    fishclass = {}
# in analyseEods    #print('classlist: ', classlist)
# in analyseEods    # #print('Classes at end: ', np.unique(classlist))
# in analyseEods   
# in analyseEods    
# in analyseEods    fishes = {}
# in analyseEods    for num in np.unique(classlist):
# in analyseEods        fishes[num] = eods[:,:][: , classlist == num]
# in analyseEods        
# in analyseEods    
# in analyseEods    
# in analyseEods    
# in analyseEods    fishes = fill_hidden_3(fishes) # cl-dict : x y z -dict
# in analyseEods    #maxlencl = max(fishes, key=lambda k: fishes[k]['x'][-1]-fishes[k]['x'][0])
# in analyseEods    
# in analyseEods    fishes, weirdparts = fill_holes(fishes)
# in analyseEods    fishes, weirdparts = fill_holes(fishes)
# in analyseEods
# in analyseEods    for cl in np.unique(classlist):
# in analyseEods        isi = [isi for isi in np.diff(fishes[cl]['x'])]
# in analyseEods        fishes[cl][3]= isi
# in analyseEods


#npFish       
#npFish   npFishes = {}
#npFish   fishfeaturecount = len(fishes[cl])
#npFish   for cl in np.unique(classlist):        
#npFish       npFishes[cl]= np.zeros([fishfeaturecount, len(fishes[cl]['x'])])
#npFish       for i, feature in enumerate(['x', 'y', 'h', 'isi']): #enumerate(fishes[cl]):
#npFish           if feature == 'isi':
#npFish               fishes[cl][feature].append(fishes[cl][feature][-1])
#npFish        #   #print(feature, cl)    
#npFish           npFishes[cl][i] = np.array(fishes[cl][feature])
#npFish  # #print(npFishes[classlist[0]][0])
#npFish  # #print(npFishes[classlist[0]][2])
#npFish  # #print(npFishes[classlist[0]][3])
#npFish   #np.savetxt('worldpeaks_x_y_cl_2', (x,y,cl, isi), fmt="%s")
#npFish
#npFish   np.set_printoptions(threshold=np.nan)
#npFish   
#npFish   for i, cl in enumerate(np.unique(classlist)):                             #Neue Klassennamen!
#npFish       x = npFishes[cl][0]  
#npFish       y = npFishes[cl][1]
#npFish       h = npFishes[cl][2]
#npFish       isi =  npFishes[cl][3]
#npFish       
#npFish       np.savetxt(filename[:-4]+'Fish_xyhisi_cl%d' % i, npFishes[cl], fmt="%s")
#npFish       
#npFish
#npFish
    




    #         /   TODO: Peakclassifikator bei weit wegliegenden klassen? Done
    #         /   TODO: Class2 implementation auf class linreg übertragen Done - Doof
    #             TODO: Klassen zusammenfuegen/ Noise zusammenfuegen
    #                        - Wenn last 3 und first 3 zueinander passen in 1. Amplitude und 2. Periode (falls peaks) oder 2. randomzeugs? - Noiseerkennung und 2. Amplitude
    #             TODO: Klassen filtern auf Patternausreißer
    #            
