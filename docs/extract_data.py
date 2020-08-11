# This script only works if you have the right input data (pulseeods.)
path = '../../pulseeods/'

import sys, glob
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
import matplotlib
sys.path.append("../thunderfish/")

from thunderfish.pulses import extract_pulsefish
from thunderfish.dataloader import load_data
from thunderfish.bestwindow import find_best_window, plot_best_data
from thunderfish.thunderfish import configuration
import plottools.colors as cols

def get_nth_item(filename,item='pulse',n=0,c=0,wf=4, return_data=[]):
	cfgfile = 'thunderfish.cfg'
	cfg = configuration(cfgfile, False, filename)

	# load data:
	raw_data, samplerate, unit = load_data(filename, 0)
	
	# best_window:
	data, idx0, idx1, clipped, min_clip, max_clip = find_best_window(raw_data, samplerate, cfg)

	# detect pulse-EODs in the data:
	_, eod_times, eod_peaktimes, zoom_window, plot_dict = extract_pulsefish(data, samplerate, filename, return_data=return_data)

	if 'eod_deletion' in return_data:
		plot_dict['og_samplerate'] = samplerate
		return plot_dict 
	elif 'all_cluster_steps' in return_data or 'moving_fish' in return_data or any('BGM' in s for s in return_data) or any('feature_extraction' in s for s in return_data):
		return plot_dict

	if item=='pulse':
		eod_ptime = eod_peaktimes[c][n]
		eod_ttime = plot_dict['eod_troughtimes'][c][n]

	elif item=='artefact':
		masks = plot_dict['masks']
		all_times = plot_dict['all_times']
		# take the nth artefact.
		eod_ptime = all_times[0][(masks[0]==True)][n]
		eod_ttime = all_times[1][(masks[0]==True)][n]

	elif item=='wave':
		masks = plot_dict['masks']
		all_times = plot_dict['all_times']

		# take the thinnest wave
		eod_ptime = 0
		eod_ttime = 9999999999

		for n in range(len(all_times[0][(masks[0]==False)&((masks[1]==True)|(masks[2]==True))])):
			ceod_ptime = all_times[0][(masks[0]==False)&((masks[1]==True)|(masks[2]==True))][n]
			ceod_ttime = all_times[1][(masks[0]==False)&((masks[1]==True)|(masks[2]==True))][n]
			if np.abs(ceod_ttime - ceod_ptime) < np.abs(eod_ptime-eod_ttime):
				eod_ttime = ceod_ptime
				eod_ptime = ceod_ttime

	if 'peak_detection' in return_data:
		samplerate = plot_dict['interp_f']*samplerate
		T=(idx1-idx0)/samplerate
		t = np.linspace(0,T,int(samplerate*T))
		w = np.abs(eod_ptime - eod_ttime)*wf
		t_mid = np.where(t-eod_ptime>=0)[0][0]

		y = plot_dict['data'][t_mid-int(w*samplerate):t_mid+int(w*samplerate)]
		x = 1000*np.linspace(t_mid/samplerate-w,t_mid/samplerate+w,len(y))
		
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

		try:
			return np.vstack((x,y)), np.vstack((p1,p1y,t1,t1y)), np.vstack((p2,p2y,t2,t2y)), np.vstack((p3,p3y,t3,t3y)), np.vstack((p4,p4y,t4,t4y))
		except:
			raise Exteption('This example range does not allow for all peak-trough pair to be shown, try choosing a different value for wf')
			return 0
	else:
		T=(idx1-idx0)/samplerate
		t = np.linspace(0,T,int(samplerate*T))
		w = np.abs(eod_ptime - eod_ttime)*wf
		y = data[np.where(t-eod_ptime>=0)[0][0]-int(w*samplerate):np.where(t-eod_ptime>=0)[0][0]+int(w*samplerate)]
		x = 1000*np.linspace(0,len(y)/samplerate,len(y))

	return x,y

def get_eods():
	x,y = get_nth_item(glob.glob(path + 'test/012*')[0],c=1,return_data=['masks','all_eod_times'])
	np.save('data/pulse_eod_1',np.vstack((x,y)))

	x,y = get_nth_item(glob.glob(path + 'validation/119*')[0],return_data=['masks','all_eod_times'])
	np.save('data/pulse_eod_2',np.vstack((x,y)))

	x,y = get_nth_item(glob.glob(path + 'test/012*')[0],n=1,return_data=['masks','all_eod_times'])
	np.save('data/pulse_eod_3',np.vstack((x,y)))

	x,y = get_nth_item(glob.glob(path + 'test/013*')[0],'artefact',return_data=['masks','all_eod_times'])
	np.save('data/artefact_1',np.vstack((x,y)))

	x,y = get_nth_item(glob.glob(path + 'validation/112*')[0],'wave',n=0,return_data=['masks','all_eod_times'],wf=6)
	np.save('data/wave_eod_1',np.vstack((x,y)))

def get_peak_detection():
	data,p1,p2,p3,p4 = get_nth_item(glob.glob(path + 'validation/008*')[0], wf=8, return_data=['all_eod_times','peak_detection'])
	np.savez('data/peakdata_1',data,p1,p2,p3,p4)
	
	data,p1,p2,p3,p4 = get_nth_item(glob.glob(path + 'validation/106*')[0], wf=18, return_data=['all_eod_times','peak_detection'])
	np.savez('data/peakdata_2',data,p1,p2,p3,p4)

def get_all_clustering_steps():
	all_cluster_values = get_nth_item(glob.glob(path + 'test/012*')[0], return_data='all_cluster_steps')
	np.savez('data/clustering',**all_cluster_values)

def get_bgm():
	bgms = ['BGM_width','BGM_height_0']
	bgm_values = get_nth_item(glob.glob(path + 'test/012*')[0], return_data=bgms)
	for bgm in bgms:
		np.savez('data/%s'%bgm,**bgm_values[bgm])

def get_feature_extraction():
	f_values = get_nth_item(glob.glob(path + 'validation/061*')[0],return_data=['feature_extraction_0_2_trough'])
	np.savez('data/feature_extraction',**f_values['feature_extraction_0_2_trough'])

def get_moving_fish():
	f_values = get_nth_item(glob.glob(path + 'test/012*')[0],return_data=['moving_fish'])
	np.savez('data/moving_fish2',**f_values['moving_fish'])

	f_values = get_nth_item(glob.glob(path + 'validation/094*')[0],return_data=['moving_fish'])
	np.savez('data/moving_fish',**f_values['moving_fish'])

def get_assesment_params():
	# good eod + sidepeak.
	f_values = get_nth_item(glob.glob(path + 'validation/119*')[0],return_data=['eod_deletion'])

	for k in f_values.keys():
		if str(k)[0]=='m':
			if any(f_values[k]) == False:
				good_eod = int(k[1])
			elif any(f_values[k][:-1]) == False:
				sidepeak = int(k[1])
			elif f_values[k][0] == True:
				artefact = int(k[1])

	np.savez('data/good_eod_ad',*[[f_values['samplerate'],f_values['og_samplerate']],f_values[good_eod]])
	np.savez('data/artefact_ad',*[[f_values['samplerate'],f_values['og_samplerate']],f_values[artefact]])
	np.savez('data/sidepeak_ad',*[[f_values['samplerate'],f_values['og_samplerate']],f_values[sidepeak]])

	f_values = get_nth_item(glob.glob(path + 'validation/112*')[0],return_data=['eod_deletion'])

	wlen = 999999999999
	# look for wave.
	for k in f_values.keys():
		if str(k)[0]=='m':
			# take only the most narrow one.
			if (f_values[k][0]==False)&(f_values[k][2] == True):
				if len(f_values[int(k[1])][0]) < wlen:
					wave_eod = int(k[1])
					wlen = len(f_values[int(k[1])][0])

	np.savez('data/wave_eod_ad',*[[f_values['samplerate'],f_values['og_samplerate']],f_values[wave_eod]])

get_bgm()
get_eods()
get_peak_detection()
get_all_clustering_steps()
get_feature_extraction()
get_moving_fish()
get_assesment_params()