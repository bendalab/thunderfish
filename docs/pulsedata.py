"""
## Load and save all data nessecary for visualizing the pulses.py alorithm
"""

# This script only works if you have the right input data (pulseeods.) in the right filepath.
path = '../../pulseeods/'

# the following recordings in pulseeods/ are used:
recordings = ['test/012_Leticia_2018-01-21_Jorge_Molina_2_pulsefish_eel.wav',
              'test/013_Leticia_2018-01-21_Jorge_Molina_1_pulsefish_eel.wav',
              'validation/008_Leticia_2018-01-21_Jorge_Molina_2_pulsefish.wav',
              'validation/061_Leticia_2019-10-14_Jaqcui_Goebl_3_pulsefish.wav',
              'validation/094_Lake-Nabugabo-Uganda_2019-0x-0x_Stefan_Mucha_1_pulsefish.wav',
              'validation/106_Panama_2014-05-17_Ruediger_Krahe_1_pulsefish_1_wavefish.wav',
              'validation/112_Panama_2014-05-18_Ruediger_Krahe_1_pulsefish_1_wavefish.wav',
              'validation/119_Sanmartin_2019-10-11_Jaqcui_Goebl_1_pulsefish.wav']

import sys
import re
import glob
import numpy as np

sys.path.append("../thunderfish/")
from thunderfish.pulses import extract_pulsefish
from thunderfish.dataloader import load_data
from thunderfish.bestwindow import find_best_window, plot_best_data
from thunderfish.thunderfish import configuration

import warnings
def warn(*args,**kwargs):
    """
    Ignore all warnings.
    """
    pass
warnings.warn=warn


def get_peakdata(filename, return_data=[]):
	""" Extract pulsefish data from a recording file.

	Parameters
	----------
	filename: string
		Path to the recording file.
	return_data: list of strings (optional)
		Specify which data to log when analysing the recording.
		Options are: 'all_eod_times', 'peak_detection', 'all_cluster_steps', 
		'snippet_clusters','masks','BGM', 'eod_deletion', 'moving_fish'.
		For implementation see thunderfish.pulses.extract_pulsefish().
		Defaults to [].

	Returns
	-------
	file_data : tuple (4)
		Tuple with:
			data : 1-D array
        		The data array of the best window
        	samplerate : float or int
        		Original sample rate of the recording data.
		    idx0 : int
		        The start index of the best window in the original data.
		    idx1 : int
		        The end index of the best window in the original data.
	pulse_data : tuple (5)
		Tuple with:
			mean_eods: list of 2D arrays
		        The average EOD for each detected fish. First column is time in seconds,
	    	    second column the mean eod, third column the standard error.
		    eod_times: list of 1D arrays
		        For each detected fish the times of EOD peaks or troughs in seconds.
		        Use these timepoints for EOD averaging.
		    eod_peaktimes: list of 1D arrays
		        For each detected fish the times of EOD peaks in seconds.
		    zoom_window: tuple of floats
		        Start and endtime of suggested window for plotting EOD timepoints.
		   	plot_dict: dictionary
		   		Dictionary with logging variables defined by return_data.
	"""
	cfgfile = 'thunderfish.cfg'
	cfg = configuration(cfgfile, False, filename)

	# load data:
	try: 	
		raw_data, samplerate, unit = load_data(filename, 0)
	except:
		raise Exception('Data not found. Please clone pulseeods (https://github.com/bendalab/pulseeods) into %s, or clone in a different folder and change the path'%filename)
	# best_window:
	data, idx0, idx1, clipped, min_clip, max_clip = find_best_window(raw_data, samplerate, cfg)

	# return data, window and pulsefish data
	return (data, samplerate, idx0, idx1), extract_pulsefish(data, samplerate, filename, return_data=return_data)


def get_nth_item(filename, item='pulse', n=0, c=0, wf=4, return_peaks=False):
	""" Extract the nth item from the pulsefish data.

	Extracts a pulse EOD, wave EOD or an artefact (specified by 'item') from a recording.
	If 'peak_detection' is in return_data, all peak detection steps on the extracted are returned as well.

	Parameters
	----------
	filename: string
		Path to the recording file.
	item: string (optional)
		Item to extract from the data. Options are: 'pulse', 
		'artefact' and 'wave'. Defaults to 'pulse'.
	n: int (optional)
		The occurence of the item to return. 
		Defaults to 0.
	c: int (optional)
		The cluster number to return. As pulseclusters are sorted by amplitude, 
		defining c=0 would return the largest EOD.
		Defaults to 0.
	wf: int or float (optional)
		The width of the recording snippet to return as a factor of the width 
		between the peak-trough pair. Defaults to 4.
	return_peaks: boolean (optional)
		Set to True to return all peak detection steps on the extracted snippet.
		Defaults to False.

	Returns
	-------
	return_dict : dictionary
		Dictionary with keys: 'data', and if return_peaks==True, 'peak_data'.
			return_dict['data'] : numpy array (2, snippet length)
				X and Y data of the extracted snippet.
			return_dict['p1'] : numpy array (4,n)
				The peaks (x,y)	and the troughs (x,y) for peak detection step 1.
			return_dict['p2'] : numpy array (4,n)
				The peaks (x,y)	and the troughs (x,y) for peak detection step 2.
			return_dict['p3'] : numpy array (4,n)
				The peaks (x,y)	and the troughs (x,y) for peak detection step 3.
			return_dict['p4'] : numpy array (4,n)
				The peaks (x,y)	and the troughs (x,y) for peak detection step 4.
	"""

	return_dict = {}
	
	# I need all eod times (peak and trough) to find the right snippet width.
	return_data = ['all_eod_times']
	
	if return_peaks:
		return_data.append('peak_detection')

	if item == 'wave' or item == 'artefact':
		return_data.append('masks')
	elif item != 'pulse':
		raise Exception('Item must be pulse, wave or artefact')
		return 0

	# detect pulse-EODs in the data:
	(data, samplerate, idx0, idx1), (_, eod_times, eod_peaktimes, zoom_window, plot_dict) = get_peakdata(filename, return_data)

	# extract the nth item from the cth cluster
	if item=='pulse':
		eod_ptime = eod_peaktimes[c][n]
		eod_ttime = plot_dict['eod_troughtimes'][c][n]
	elif item=='artefact':
		masks = plot_dict['masks']
		all_times = plot_dict['all_times']
		eod_ptime = all_times[0][(masks[0]==True)][n]
		eod_ttime = all_times[1][(masks[0]==True)][n]
	elif item=='wave':
		
		masks = plot_dict['masks']
		all_times = plot_dict['all_times']

		# for some wavefish, the peak and trough that are detected 
		# are far away from eachother which results in the wrong snippet width.
		# Therefore, it returns the smallest wave snippet.

		eod_ptime = 0
		eod_ttime = 9999999999

		for n in range(len(all_times[0][(masks[0]==False)&((masks[1]==True)|(masks[2]==True))])):
			ceod_ptime = all_times[0][(masks[0]==False)&((masks[1]==True)|(masks[2]==True))][n]
			ceod_ttime = all_times[1][(masks[0]==False)&((masks[1]==True)|(masks[2]==True))][n]
			if np.abs(ceod_ttime - ceod_ptime) < np.abs(eod_ptime-eod_ttime):
				eod_ttime = ceod_ptime
				eod_ptime = ceod_ttime

	if return_peaks:
		# data is interpolated, so use this new samplerate to extract the snippet
		samplerate = plot_dict['interp_f']*samplerate
		T=(idx1-idx0)/samplerate
		t = np.linspace(0,T,int(samplerate*T))
		w = np.abs(eod_ptime - eod_ttime)*wf
		t_mid = np.where(t-eod_ptime>=0)[0][0]

		y = plot_dict['data'][t_mid-int(w*samplerate):t_mid+int(w*samplerate)]
		x = 1000*np.linspace(t_mid/samplerate-w,t_mid/samplerate+w,len(y))
		
		# extract the peakdata for the current snippet.
		p1 = 1000/samplerate*plot_dict['peaks_1'][(plot_dict['peaks_1']>t_mid-int(w*samplerate))&(plot_dict['peaks_1']<t_mid+int(w*samplerate))]
		t1 = 1000/samplerate*plot_dict['troughs_1'][(plot_dict['troughs_1']>t_mid-int(w*samplerate))&(plot_dict['troughs_1']<t_mid+int(w*samplerate))]
		p2 = 1000/samplerate*plot_dict['peaks_2'][(plot_dict['peaks_2']>t_mid-int(w*samplerate))&(plot_dict['peaks_2']<t_mid+int(w*samplerate))]
		t2 = 1000/samplerate*plot_dict['troughs_2'][(plot_dict['troughs_2']>t_mid-int(w*samplerate))&(plot_dict['troughs_2']<t_mid+int(w*samplerate))]
		p3 = 1000/samplerate*plot_dict['peaks_3'][(plot_dict['peaks_3']>t_mid-int(w*samplerate))&(plot_dict['peaks_3']<t_mid+int(w*samplerate))]
		t3 = 1000/samplerate*plot_dict['troughs_3'][(plot_dict['troughs_3']>t_mid-int(w*samplerate))&(plot_dict['troughs_3']<t_mid+int(w*samplerate))]
		p4 = 1000/samplerate*plot_dict['peaks_4'][(plot_dict['peaks_4']>t_mid-int(w*samplerate))&(plot_dict['peaks_4']<t_mid+int(w*samplerate))]
		t4 = 1000/samplerate*plot_dict['troughs_4'][(plot_dict['troughs_4']>t_mid-int(w*samplerate))&(plot_dict['troughs_4']<t_mid+int(w*samplerate))]
		
		p1y = plot_dict['data'][(samplerate/1000*p1).astype('int')]
		t1y = plot_dict['data'][(samplerate/1000*t1).astype('int')]
		p2y = plot_dict['data'][(samplerate/1000*p2).astype('int')]
		t2y = plot_dict['data'][(samplerate/1000*t2).astype('int')]
		p3y = plot_dict['data'][(samplerate/1000*p3).astype('int')]
		t3y = plot_dict['data'][(samplerate/1000*t3).astype('int')]
		p4y = plot_dict['data'][(samplerate/1000*p4).astype('int')]
		t4y = plot_dict['data'][(samplerate/1000*t4).astype('int')]

		return_dict['data'] = np.vstack((x,y))
		try:
			return_dict['p1'] = np.vstack((p1,p1y,t1,t1y))
			return_dict['p2'] = np.vstack((p2,p2y,t2,t2y))
			return_dict['p3'] = np.vstack((p3,p3y,t3,t3y))
			return_dict['p4'] = np.vstack((p4,p4y,t4,t4y))

			return return_dict

		except:
			raise Exteption('This example range does not allow for all peak-trough pair to be shown, try choosing a different value for wf')
			return 0

	else:
		# return snippet of raw data
		T= (idx1-idx0)/samplerate
		t = np.linspace(0,T,int(samplerate*T))
		w = np.abs(eod_ptime - eod_ttime)*wf
		y = data[np.where(t-eod_ptime>=0)[0][0]-int(w*samplerate):np.where(t-eod_ptime>=0)[0][0]+int(w*samplerate)]
		x = 1000*np.linspace(0,len(y)/samplerate,len(y))

		return_dict['data'] = np.vstack((x,y))

		return return_dict

def get_eods():
	""" Get three pulse-type EODs, one artefact and one wavetype-EOD from raw data files.
		Data is saved as .npy files in data/.
	"""
	data = get_nth_item(glob.glob(path + 'test/012*')[0],c=1)
	np.save('data/pulse_eod_1',data['data'])

	data = get_nth_item(glob.glob(path + 'validation/119*')[0])
	np.save('data/pulse_eod_2',data['data'])

	data = get_nth_item(glob.glob(path + 'test/012*')[0],n=1)
	np.save('data/pulse_eod_3',data['data'])

	data = get_nth_item(glob.glob(path + 'test/013*')[0],'artefact')
	np.save('data/artefact_1',data['data'])

	data = get_nth_item(glob.glob(path + 'validation/112*')[0],'wave',wf=5)
	np.save('data/wave_eod_1',data['data'])

def get_peak_detection():
	""" Get two pulse-type EODs snippets and all peak detection steps in this snippet.
		Data is saved as .npz file in data/.
	"""
	data = get_nth_item(glob.glob(path + 'validation/008*')[0], wf=8, return_peaks=True)
	np.savez('data/peakdata_1',**data)
	
	data = get_nth_item(glob.glob(path + 'validation/106*')[0], wf=18, return_peaks=True)
	np.savez('data/peakdata_2',**data)

def get_all_clustering_steps():
	""" Get data from all clustering steps for one recording file.
		Data is saved as .npz file in data/.
	"""
	_, (_, _, _, _, all_cluster_values) = get_peakdata(glob.glob(path + 'test/012*')[0], return_data='all_cluster_steps')
	np.savez('data/clustering',**all_cluster_values)

def get_bgm():
	""" Get data from two BGM clustering steps (one on EOD width and one on EOD height) for one recording file.
		Data is saved as .npz file in data/.
	"""
	bgms = ['BGM_width','BGM_height']
	_, (_, _, _, _, bgm_values) = get_peakdata(glob.glob(path + 'test/012*')[0], return_data=bgms)
	
	np.savez('data/BGM_width',**bgm_values['BGM_width'])
	np.savez('data/BGM_height',**bgm_values['BGM_height_0'])

def get_snippet_clusters():
	""" Get data from one clustering step on EOD shape for one recording file.
		Data is saved as .npz file in data/.
	"""
	_, (_, _, _, _, f_values) = get_peakdata(glob.glob(path + 'validation/061*')[0],return_data=['snippet_clusters'])
	np.savez('data/feature_extraction',**f_values['snippet_clusters_0_2_trough'])

def get_moving_fish():
	""" Get data from a moving fish detection step for two recording files.
		Data is saved as .npz file in data/.
	"""
	_, (_, _, _, _, f_values) = get_peakdata(glob.glob(path + 'test/012*')[0],return_data=['moving_fish'])
	np.savez('data/moving_fish2',**f_values['moving_fish'])

	_, (_, _, _, _, f_values) = get_peakdata(glob.glob(path + 'validation/094*')[0],return_data=['moving_fish'])
	np.savez('data/moving_fish',**f_values['moving_fish'])

def get_assesment_params():
	""" Get data from the pulse-type EOD detection step for two recording files. 
		Pulse detection is saved for one good EOD pulse, one artefact, one wavefish and one EOD sidepeak.
		Data is saved as .npz file in data/.
	"""

	(_, samplerate, _, _), (_, _, _, _, f_values) = get_peakdata(glob.glob(path + 'validation/119*')[0],return_data=['eod_deletion'])
	
	# go through all clusters
	for k in f_values.keys():
		# evaluate masks (m)
		if 'mask_' in str(k):

			label = re.findall(r'\d+', k)[0]
			if any(f_values[k]) == False:
				# if all masks are false, it is a good EOD
				good_eod = label
			elif any(f_values[k][:-1]) == False:
				# if all masks but the last are False, it is a sidepeak
				sidepeak = label
			elif f_values[k][0] == True:
				# if the first mask is True it is an artefact.
				artefact = label

	# save data
	np.savez('data/good_eod_ad',**{'samplerates':[f_values['samplerate'],samplerate],'values':f_values['vals_'+good_eod]})
	np.savez('data/artefact_ad',**{'samplerates':[f_values['samplerate'],samplerate],'values':f_values['vals_'+artefact]})
	np.savez('data/sidepeak_ad',**{'samplerates':[f_values['samplerate'],samplerate],'values':f_values['vals_'+sidepeak]})

	# load recording with wavefish
	(_, samplerate, _, _), (_, _, _, _, f_values) = get_peakdata(glob.glob(path + 'validation/112*')[0],return_data=['eod_deletion'])

	wlen = 999999999999
	# look for wave.
	for k in f_values.keys():
		if 'mask_' in str(k):
			label = re.findall(r'\d+', k)[0]
			if (f_values[k][0]==False)&(f_values[k][2] == True):
				# take only the most narrow one for display (ideally just display one period of the wave).
				if len(f_values['vals_'+label][0]) < wlen:
					wave_eod = label
					wlen = len(f_values['vals_'+label][0])

	np.savez('data/wave_eod_ad',**{'samplerates':[f_values['samplerate'],samplerate],'values':f_values['vals_'+wave_eod]})

get_bgm()
get_eods()
get_peak_detection()
get_all_clustering_steps()
get_snippet_clusters()
get_moving_fish()
get_assesment_params()
