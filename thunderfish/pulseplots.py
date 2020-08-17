"""
Plot and save key steps in pulses.py for visualizing the alorithm.

"""

import pickle, glob
import plottools.scalebars as sb
from matplotlib import rcParams, gridspec
import matplotlib.pyplot as plt
import numpy as np
from plottools import colors as cols
import matplotlib
from scipy import stats
from matplotlib.patches import ConnectionPatch, Rectangle
from matplotlib.ticker import NullFormatter

# plotting parameters and colors
rcParams['font.family'] = 'monospace'
cmap = plt.get_cmap("Dark2")
c_g = cmap(0)
c_o = cmap(1)
c_grey = cmap(7)
cmap_pts = [cmap(2),cmap(3)]

def arrowed_spines(ax, ms=10):
	""" Create an arrowed spine on the y-axis of a plot.

		Parameters
		----------
		ax : matplotlib figure axis
			Axis on which the arrow should be plot. 
	"""
	xmin, xmax = ax.get_xlim() 
	ymin, ymax = ax.get_ylim()
	ax.scatter([xmin],[ymax],s=ms,marker='^', clip_on=False,color='k')
	ax.set_xlim([xmin,xmax])
	ax.set_ylim([ymin,ymax])

def loghist(ax,x,bmin,bmax,n,c,orientation='vertical',label=''):
	""" Plot histogram with logarithmic scale.

		Parameters
		----------
		ax : matplotlib axis
			Axis to plot the histogram on.
		x : numpy array
			Input data for histogram.
		bmin : float
			Minimum value for the histogram bins.
		bmax : float
			Maximum value for the histogram bins. 
		n : int
			Number of bins.
		c : matplotlib color
			Color of histogram.
		orientation : string (optional)
			Histogram orientation.
			Defaults to 'vertical'.
		label : string (optional)
			Label for x. 
			Defaults to '' (no label).

		Returns
		-------
		n : array
			The values of the histogram bins.
		bins : array
			The edges of the bins.
		patches : BarContainer
			Container of individual artists used to create the histogram.

	"""
	return ax.hist(x,bins=np.exp(np.linspace(np.log(bmin),np.log(bmax),n)),color=c,orientation=orientation,label=label)

def plot_all(data, eod_p_times, eod_tr_times, fs, mean_eods):
    """Quick way to view the output of extract_pulsefish in a single plot.

    Parameters
    ----------
    data: array
        Recording data.
    eod_p_times: array of ints
        EOD peak indices.
    eod_tr_times: array of ints
        EOD trough indices.
    fs: float
        Samplerate.
    mean_eods: list of numpy arrays
        Mean EODs of each pulsefish found in the recording.
    """

    fig = plt.figure(figsize=(10,5))

    if len(eod_p_times) > 0:
        gs = gridspec.GridSpec(2, len(eod_p_times))
        ax = fig.add_subplot(gs[0,:])
        ax.plot(np.arange(len(data))/fs,data,c='k',alpha=0.3)
        
        for i,(pt,tt) in enumerate(zip(eod_p_times,eod_tr_times)):
            ax.plot(pt,data[(pt*fs).astype('int')],'o',label=i+1,ms=10,c=cmap(i))
            ax.plot(tt,data[(tt*fs).astype('int')],'o',label=i+1,ms=10,c=cmap(i))
            
        ax.set_xlabel('time [s]')
        ax.set_ylabel('amplitude [V]')

        for i, m in enumerate(mean_eods):
            ax = fig.add_subplot(gs[1,i])
            ax.plot(1000*m[0], 1000*m[1], c='k')

            ax.fill_between(1000*m[0],1000*(m[1]-m[2]),1000*(m[1]+m[2]),color=cmap(i))
            ax.set_xlabel('time [ms]')
            ax.set_ylabel('amplitude [mV]') 
    else:
        plt.plot(np.arange(len(data))/fs,data,c='k',alpha=0.3)

    plt.tight_layout()


def plot_clustering(samplerate, eod_widths, eod_hights, eod_shapes, disc_masks, merge_masks):
	"""Plot all clustering steps.
	Plot clustering steps on width, height and shape. Then plot the remaining EODs after 
	the EOD assessment step and the EODs after the merge step.

	Parameters
	----------
	samplerate : float
		Samplerate of EOD snippets.
	eod_widths : list of three 1D numpy arrays
		The first list entry gives the unique labels of all width clusters as a list of ints.
		The second list entry gives the width values for each EOD in samples as a 1D numpy array of ints.
		The third list entry gives the width labels for each EOD as a 1D numpy array of ints.
	eod_hights : nested lists (2 layers) of three 1D numpy arrays
		The first list entry gives the unique labels of all height clusters as a list of ints for each width cluster.
		The second list entry gives the height values for each EOD as a 1D numpy array of floats for each width cluster.
		The third list entry gives the height labels for each EOD as a 1D numpy array of ints for each width cluster.
	eod_shapes : nested lists (3 layers) of three 1D numpy arrays
		The first list entry gives the raw EOD snippets as a 2D numpy array for each height cluster in a width cluster.
		The second list entry gives the snippet PCA values for each EOD as a 2D numpy array of floats for each height cluster in a width cluster.
		The third list entry gives the shape labels for each EOD as a 1D numpy array of ints for each height cluster in a width cluster.
	disc_masks : Nested lists (two layers) of 1D numpy arrays
		The masks of EODs that are discarded by the discarding step of the algorithm. The masks are 1D boolean arrays where 
		instances that are set to True are discarded by the algorithm. Discarding masks are saved in nested lists that represent the width and height clusters.
	merge_masks : Nested lists (two layers) of 2D numpy arrays
		The masks of EODs that are discarded by the merging step of the algorithm. The masks are 2D boolean arrays where 
		for each sample point `i` either `merge_mask[i,0]` or `merge_mask[i,1]` is set to True. Here, merge_mask[:,0] represents the 
		peak-centered clusters and `merge_mask[:,1]` represents the trough-centered clusters. Merge masks are saved in nested lists 
		that represent the width and height clusters.
	"""

	# create figure + transparant figure.
	fig = plt.figure(figsize=(12,7))
	transFigure = fig.transFigure.inverted()

	# set up the figure layout
	outer = gridspec.GridSpec(1,5,width_ratios=[1,1,2,1,2],left=0.05,right=0.95)

	# set titles for each clustering step
	titles = ['1. Widths','2. Heights','3. Shape','4. Pulse EODs','5. Merge']
	for i, title in enumerate(titles):
		title_ax = gridspec.GridSpecFromSubplotSpec(1,1,subplot_spec = outer[i])
		ax = fig.add_subplot(title_ax[0])
		ax.text(0,110,title,ha='center',va='bottom',clip_on=False)
		ax.set_xlim([-100,100])
		ax.set_ylim([-100,100])
		ax.axis('off')

	# compute sizes for each axis
	w_size = 1
	h_size = len(eod_hights[1])

	shape_size = np.sum([len(sl) for sl in eod_shapes[0]])
	
	# count required axes sized for the last two plot columns.
	disc_size = 0
	merge_size= 0
	for shapelabel, dmasks, mmasks in zip(eod_shapes[2],disc_masks,merge_masks):
		for sl, dm, mm in zip(shapelabel,dmasks,mmasks):
			uld1 = np.unique((sl[0]+1)*np.invert(dm[0]))
			uld2 = np.unique((sl[1]+1)*np.invert(dm[1]))
			disc_size = disc_size+len(uld1[uld1>0])+len(uld2[uld2>0])
			
			uld1 = np.unique((sl[0]+1)*mm[0])
			uld2 = np.unique((sl[1]+1)*mm[1])
			merge_size = merge_size+len(uld1[uld1>0])+len(uld2[uld2>0])

	# set counters to keep track of the plot axes
	disc_block = 0
	merge_block = 0
	shape_count = 0

	# create all axes
	width_hist_ax = gridspec.GridSpecFromSubplotSpec(w_size,1,subplot_spec = outer[0])
	hight_hist_ax = gridspec.GridSpecFromSubplotSpec(h_size,1,subplot_spec = outer[1])
	shape_ax = gridspec.GridSpecFromSubplotSpec(shape_size,1, subplot_spec = outer[2])
	shape_windows = [gridspec.GridSpecFromSubplotSpec(2,2, hspace=0.0, wspace=0.0, subplot_spec = shape_ax[i]) for i in range(shape_size)]
	
	EOD_delete_ax = gridspec.GridSpecFromSubplotSpec(disc_size,1,subplot_spec=outer[3])
	EOD_merge_ax = gridspec.GridSpecFromSubplotSpec(merge_size,1,subplot_spec=outer[4])

    # plot width labels histogram
	ax1 = fig.add_subplot(width_hist_ax[0])
	# set axes features.
	ax1.set_xscale('log')
	ax1.spines['top'].set_visible(False)
	ax1.spines['right'].set_visible(False)
	ax1.spines['bottom'].set_visible(False)
	ax1.axes.xaxis.set_visible(False)
	ax1.set_yticklabels([])

	# indices for plot colors (dark to light)
	colidxsw = -np.linspace(-1.25, -0.5, h_size)

	for i, (wl, colw, uhl, eod_h, eod_h_labs, w_snip, w_feat, w_lab, w_dm, w_mm) in enumerate(zip(eod_widths[0], colidxsw, eod_hights[0], eod_hights[1], eod_hights[2], eod_shapes[0], eod_shapes[1], eod_shapes[2], disc_masks, merge_masks)):

		# plot width hist
		hw, _, _ = ax1.hist(eod_widths[1][eod_widths[2]==wl], bins=np.linspace(np.min(eod_widths[1]),np.max(eod_widths[1]),100),color=cols.lighter(c_o,colw),orientation='horizontal')
		
		# set arrow when the last hist is plot so the size of the axes are known.
		if i == h_size-1:
			arrowed_spines(ax1,ms=20)

		# determine total size of the hight historgams now.
		my,b = np.histogram(eod_h,bins=np.exp(np.linspace(np.min(np.log(eod_h)),np.max(np.log(eod_h)),100)))
		maxy = np.max(my)

		# set axes features for hight hist.
		ax2 = fig.add_subplot(hight_hist_ax[h_size-i-1])
		ax2.set_xscale('log')
		ax2.spines['top'].set_visible(False)
		ax2.spines['right'].set_visible(False)
		ax2.spines['bottom'].set_visible(False)
		ax2.set_xlim([0.9,maxy])
		ax2.axes.xaxis.set_visible(False)
		ax2.set_yscale('log')
		ax2.yaxis.set_major_formatter(NullFormatter())
		ax2.yaxis.set_minor_formatter(NullFormatter())

		# define colors for plots
		colidxsh = -np.linspace(-1.25,-0.5,len(uhl))

		for n, (hl, hcol, snippets, features, labels, dmasks, mmasks) in enumerate(zip(uhl, colidxsh, w_snip, w_feat, w_lab, w_dm, w_mm)):


			hh,_,_=loghist(ax2,eod_h[eod_h_labs==hl],np.min(eod_h),np.max(eod_h),100,cols.lighter(c_g,hcol),orientation='horizontal')

			# set arrow spines only on last plot
			if n==len(uhl)-1:
				arrowed_spines(ax2,ms=10)

			# plot line from the width histogram to the height histogram.
			if n==0:
				coord1 = transFigure.transform(ax1.transData.transform([np.median(hw[hw!=0]),np.median(eod_widths[1][eod_widths[2]==wl])]))
				coord2 = transFigure.transform(ax2.transData.transform([0.9,np.mean(eod_h)]))
				line = matplotlib.lines.Line2D((coord1[0],coord2[0]),(coord1[1],coord2[1]),
				                               transform=fig.transFigure,color='grey',linewidth=0.5)
				fig.lines.append(line)

			# compute sizes of the eod_discarding and merge steps
			s1 = np.unique((labels[0]+1)*(~dmasks[0]))
			s2 = np.unique((labels[1]+1)*(~dmasks[1]))
			disc_block = disc_block + len(s1[s1>0]) + len(s2[s2>0])
			
			s1 = np.unique((labels[0]+1)*(mmasks[0]))
			s2 = np.unique((labels[1]+1)*(mmasks[1]))
			merge_block = merge_block + len(s1[s1>0]) + len(s2[s2>0])

			axs = []
			disc_count = 0
			merge_count = 0

			# now plot the clusters for peak and trough centerings
			for pt, cmap_pt in zip([0,1],cmap_pts):
				
				ax3 = fig.add_subplot(shape_windows[shape_size-1-shape_count][pt,0])
				ax4 = fig.add_subplot(shape_windows[shape_size-1-shape_count][pt,1])

				# remove axes
				ax3.axes.xaxis.set_visible(False)
				ax4.axes.yaxis.set_visible(False)
				ax3.axes.yaxis.set_visible(False)
				ax4.axes.xaxis.set_visible(False)

				# set color indices
				colidxss = -np.linspace(-1.25,-0.5,len(np.unique(labels[pt][labels[pt]>=0])))
				j=0
				for c in np.unique(labels[pt]):
					
					if c<0:
						# plot noise features + snippets
						ax3.plot(features[pt][labels[pt]==c,0],features[pt][labels[pt]==c,1],'.',color='lightgrey',label='-1',rasterized=True)
						ax4.plot(snippets[pt][labels[pt]==c].T,linewidth=0.1,color='lightgrey',label='-1',rasterized=True)
					else:
						# plot cluster features and snippets
						ax3.plot(features[pt][labels[pt]==c,0],features[pt][labels[pt]==c,1],'.',color=cols.lighter(cmap_pt,colidxss[j]),label=c,rasterized=True)
						ax4.plot(snippets[pt][labels[pt]==c].T,linewidth=0.1,color=cols.lighter(cmap_pt,colidxss[j]),label=c,rasterized=True)
						
						# check if the current cluster is an EOD, if yes, plot it.
						if np.sum(dmasks[pt][labels[pt]==c]) == 0:

							ax = fig.add_subplot(EOD_delete_ax[disc_size-disc_block+disc_count])
							ax.axis('off')

							# plot mean EOD snippet
							ax.plot(np.mean(snippets[pt][labels[pt]==c],axis=0),color=cols.lighter(cmap_pt,colidxss[j]))
							disc_count = disc_count + 1

							# match colors and draw line..							
							coord1 = transFigure.transform(ax4.transData.transform([ax4.get_xlim()[1], ax4.get_ylim()[0] + 0.5*(ax4.get_ylim()[1]-ax4.get_ylim()[0])]))
							coord2 = transFigure.transform(ax.transData.transform([ax.get_xlim()[0],ax.get_ylim()[0] + 0.5*(ax.get_ylim()[1]-ax.get_ylim()[0])]))
							line = matplotlib.lines.Line2D((coord1[0],coord2[0]),(coord1[1],coord2[1]),
					                               transform=fig.transFigure,color='grey',linewidth=0.5)
							fig.lines.append(line)	
							axs.append(ax)

							# check if the current EOD survives the merge step
							# if so, plot it.
							if np.sum(mmasks[pt,labels[pt]==c])>0:

								ax = fig.add_subplot(EOD_merge_ax[merge_size-merge_block+merge_count])
								ax.axis('off')
								
								ax.plot(np.mean(snippets[pt][labels[pt]==c],axis=0),color=cols.lighter(cmap_pt,colidxss[j]))
								merge_count = merge_count + 1

						j=j+1

				if pt==0:
					# draw line from hight cluster to EOD shape clusters.
					coord1 = transFigure.transform(ax2.transData.transform([np.median(hh[hh!=0]),np.median(eod_h[eod_h_labs==hl])]))
					coord2 = transFigure.transform(ax3.transData.transform([ax3.get_xlim()[0],ax3.get_ylim()[0]]))
					line = matplotlib.lines.Line2D((coord1[0],coord2[0]),(coord1[1],coord2[1]),
					                               transform=fig.transFigure,color='grey',linewidth=0.5)
					fig.lines.append(line)

			shape_count = shape_count + 1
			
			if len(axs)>0:
				# plot lines that indicate the merged clusters.
				coord1 = transFigure.transform(axs[0].transData.transform([axs[0].get_xlim()[1]+0.1*(axs[0].get_xlim()[1]-axs[0].get_xlim()[0]), axs[0].get_ylim()[1]-0.25*(axs[0].get_ylim()[1]-axs[0].get_ylim()[0])]))
				coord2 = transFigure.transform(axs[-1].transData.transform([axs[-1].get_xlim()[1]+0.1*(axs[-1].get_xlim()[1]-axs[-1].get_xlim()[0]), axs[-1].get_ylim()[0]+0.25*(axs[-1].get_ylim()[1]-axs[-1].get_ylim()[0])]))
				line = matplotlib.lines.Line2D((coord1[0],coord2[0]),(coord1[1],coord2[1]),
                               transform=fig.transFigure,color='grey',linewidth=1)
				fig.lines.append(line)

def plot_bgm(x, means, variances, weights, use_log, labels, labels_am, xlab):
	"""Plot a BGM clustering step either on EOD width or height.

	Parameters
	----------
	x : 1D numpy array of floats
		BGM input values.
	means : list of floats
		BGM Gaussian means
	variances : list of floats
		BGM Gaussian variances.
	weights : list of floats
		BGM Gaussian weights.
	use_log : boolean
		True if the z-scored logarithm of the data was used as BGM input.
	labels : 1D numpy array of ints
		Labels defined by BGM model (before merging based on merge factor).
	labels_am : 1D numpy array of ints
		Labels defined by BGM model (after merging based on merge factor).
	xlab : string
		Label for plot (defines the units of the BGM data).
	"""

	if 'width' in xlab:
		ccol = c_o
	elif 'height' in xlab:
		ccol = c_g
	else:
		ccol = 'b'

	# get the transform that was used as BGM input
	if use_log:
		x_transform = stats.zscore(np.log(x))
		xplot = np.exp(np.linspace(np.log(np.min(x)),np.log(np.max(x)),1000))
	else:
		x_transform = stats.zscore(x)
		xplot = np.linspace(np.min(x),np.max(x),1000)

	# compute the x values and gaussians
	x2 = np.linspace(np.min(x_transform),np.max(x_transform),1000)
	gaussians = []
	gmax = 0
	for i, (w,m,std) in enumerate(zip(weights, means, variances)):
		gaus = np.sqrt(w*stats.norm.pdf(x2,m,np.sqrt(std)))
		gaussians.append(gaus)
		gmax = max(np.max(gaus),gmax)
	
	# compute classes defined by gaussian intersections
	classes = np.argmax(np.vstack(gaussians),axis=0)
	
	# find the minimum of any gaussian that is within its class
	gmin = 100
	for i,c in enumerate(np.unique(classes)):
		gmin=min(gmin,np.min(gaussians[c][classes==c]))

	# set up the figure
	fig, ax1 = plt.subplots(figsize=(8,4.8))
	fig_ysize = 4
	ax2 = ax1.twinx()
	ax1.spines['top'].set_visible(False)
	ax2.spines['top'].set_visible(False)
	ax1.set_xlabel('x [a.u.]')
	ax1.set_ylabel('#')
	ax2.set_ylabel('Likelihood')
	ax2.set_yscale('log')
	ax1.set_yscale('log')
	if use_log:
		ax1.set_xscale('log')
	ax1.set_xlabel(xlab)

	# define colors for plotting gaussians
	colidxs = -np.linspace(-1.25,-0.5,len(np.unique(classes)))

	# plot the gaussians
	for i,c in enumerate(np.unique(classes)):
		ax2.plot(xplot,gaussians[c],c=cols.lighter(c_grey,colidxs[i]),linewidth=2,label=r'$N(\mu_%i,\sigma_%i)$'%(c,c))
	
	# plot intersection lines
	ax2.vlines(xplot[1:][np.diff(classes)!=0],0,gmax/gmin,color='k',linewidth=2,linestyle='--')
	ax2.set_ylim([gmin,np.max(np.vstack(gaussians))*1.1])

	# plot data distributions and classes
	colidxs = -np.linspace(-1.25,-0.5,len(np.unique(labels)))
	for i,l in enumerate(np.unique(labels)):
		if use_log:
			h,binn,_=loghist(ax1,x[labels==l],np.min(x),np.max(x),100,cols.lighter(ccol,colidxs[i]),label=r'$x_%i$'%l)
		else:
			h,binn,_=ax1.hist(x[labels==l],bins=np.linspace(np.min(x),np.max(x),100),color=cols.lighter(ccol,colidxs[i]),label=r'$x_%i$'%l)

	# annotate merged clusters
	for l in np.unique(labels_am):
		maps = np.unique(labels[labels_am==l])
		if len(maps) > 1:
			x1 = x[labels==maps[0]]
			x2 = x[labels==maps[1]]

			print(np.median(x1))
			print(np.median(x2))
			print(gmax)
			ax2.plot([np.median(x1),np.median(x2)],[1.2*gmax,1.2*gmax],c='k',clip_on=False)
			ax2.plot([np.median(x1),np.median(x1)],[1.1*gmax,1.2*gmax],c='k',clip_on=False)
			ax2.plot([np.median(x2),np.median(x2)],[1.1*gmax,1.2*gmax],c='k',clip_on=False)
			ax2.annotate(r'$\frac{|{\tilde{x}_%i-\tilde{x}_%i}|}{max(\tilde{x}_%i,\tilde{x}_%i)} < \epsilon$'%(maps[0],maps[1],maps[0],maps[1]),[np.median(x1)*1.1,gmax*1.2], xytext=(10, 10), textcoords='offset points',fontsize=12,annotation_clip=False,ha='center')

	# add legends and plot.
	ax2.legend(loc='lower left',frameon=False,bbox_to_anchor=(-0.05,1.3),ncol=len(np.unique(classes)))
	ax1.legend(loc='upper left',frameon=False,bbox_to_anchor=(-0.05,1.3),ncol=len(np.unique(labels)))
	plt.tight_layout()
	plt.show()

def plot_feature_extraction(raw_snippets, normalized_snippets, features, labels, dt, pt):
	"""Plot clustering step on EOD shape.
		
	Parameters
	----------
	raw_snippets : 2D numpy array
		Raw EOD snippets.
	normalized_snippets : 2D numpy array
		Normalized EOD snippets.
	features : 2D numpy array
		PCA values for each normalized EOD snippet.
	labels : 1D numpy array of ints
		Cluster labels.
	dt : float
		Sample interval of snippets.
	pt : int
		Set to 0 for peak-centered EODs and set to 1 for trough-centered EODs.
	"""

	ccol = cmap_pts[pt]

	# set up the figure layout
	fig = plt.figure(figsize=(((2+0.2)*4.8),4.8))
	outer = gridspec.GridSpec(1,2,wspace=0.2,hspace=0)

	x = np.arange(-dt*1000*raw_snippets.shape[1]/2,dt*1000*raw_snippets.shape[1]/2,dt*1000)

	snip_ax = gridspec.GridSpecFromSubplotSpec(2,1,subplot_spec = outer[0],hspace=0.35)
	pc_ax = gridspec.GridSpecFromSubplotSpec(features.shape[1]-1,features.shape[1]-1,subplot_spec = outer[1],hspace=0,wspace=0)
	
	# 3 plots: raw snippets, normalized, pcs.
	ax_raw_snip = fig.add_subplot(snip_ax[0])
	ax_normalized_snip = fig.add_subplot(snip_ax[1])

	colidxs = -np.linspace(-1.25,-0.5,len(np.unique(labels[labels>=0])))
	j=0

	for c in np.unique(labels):
		if c<0:
			color='lightgrey'
		else:
			color = cols.lighter(ccol,colidxs[j])
			j=j+1

		ax_raw_snip.plot(x,raw_snippets[labels==c].T,color=color,label='-1',rasterized=True,alpha=0.25)
		ax_normalized_snip.plot(x,normalized_snippets[labels==c].T,color=color,alpha=0.25)
		ax_raw_snip.spines['top'].set_visible(False)
		ax_raw_snip.spines['right'].set_visible(False)
		ax_raw_snip.get_xaxis().set_ticklabels([])
		ax_raw_snip.set_title('Raw snippets')
		ax_raw_snip.set_ylabel('Amplitude [a.u.]')
		ax_normalized_snip.spines['top'].set_visible(False)
		ax_normalized_snip.spines['right'].set_visible(False)
		ax_normalized_snip.set_title('Normalized snippets')
		ax_normalized_snip.set_ylabel('Amplitude [a.u.]')
		ax_normalized_snip.set_xlabel('Time [ms]')

		ax_raw_snip.axis('off')
		ax_normalized_snip.axis('off')

		ax_overlay = fig.add_subplot(pc_ax[:,:])
		ax_overlay.set_title('Features')
		ax_overlay.axis('off')

		for n in range(features.shape[1]):
			for m in range(n):
				ax = fig.add_subplot(pc_ax[n-1,m])
				ax.scatter(features[labels==c,m],features[labels==c,n],marker='.',color=color,alpha=0.25)				
				ax.set_xlim([np.min(features),np.max(features)])
				ax.set_ylim([np.min(features),np.max(features)])
				ax.get_xaxis().set_ticklabels([])
				ax.get_yaxis().set_ticklabels([])
				ax.get_xaxis().set_ticks([])
				ax.get_yaxis().set_ticks([])

				if m==0:
					ax.set_ylabel('PC %i'%(n+1))

				if n==features.shape[1]-1:
					ax.set_xlabel('PC %i'%(m+1))

		ax = fig.add_subplot(pc_ax[0,features.shape[1]-2])
		ax.set_xlim([np.min(features),np.max(features)])
		ax.set_ylim([np.min(features),np.max(features)])

		size = max(1,int(np.ceil(-np.log10(np.max(features)-np.min(features)))))
		wbar = np.floor((np.max(features)-np.min(features))*10**size)/10**size

		# should be smaller than the actual thing! so like x% of it?
		sb.xscalebar(ax,0,0,wbar,wformat='%%.%if'%size)
		sb.yscalebar(ax,0,0,wbar,hformat='%%.%if'%size)
		ax.axis('off')

def plot_moving_fish(ws, dts, clusterss, ts, fishcounts, T, ignore_stepss):
	"""Plot moving fish detection step.

	Parameters
	----------
	ws : list of floats
		Median width for each width cluster that the moving fish algorithm is computed on (in seconds).
	dts : list of floats
		Sliding window size (in seconds) for each width cluster.
	clusterss : list of 1D numpy int arrays
		Cluster labels for each EOD cluster in a width cluster.
	ts : list of 1D numpy float arrays
		EOD emission times for each EOD in a width cluster.
	fishcounts : list of lists
		Sliding window timepoints and fishcounts for each width cluster.
	T : float
		Lenght of analyzed recording in seconds.
	ignore_stepss : list of 1D int arrays
		Mask for fishcounts that were ignored (ignored if True) in the moving_fish analysis.
	"""
	fig = plt.figure()

	# create gridspec
	outer = gridspec.GridSpec(len(ws),1)

	for i, (w, dt, clusters, t, fishcount, ignore_steps) in enumerate(zip(ws, dts, clusterss, ts, fishcounts, ignore_stepss)):
		
		gs = gridspec.GridSpecFromSubplotSpec(2,1,subplot_spec = outer[i])
		
		# axis for clusters
		ax1 = fig.add_subplot(gs[0])
		# axis for fishcount
		ax2 = fig.add_subplot(gs[1])

		# plot clusters as eventplot
		for cnum,c in enumerate(np.unique(clusters[clusters>=0])):
			ax1.eventplot(t[clusters==c],lineoffsets=cnum,linelengths=0.5,color=cmap(i))
			cnum = cnum + 1

		# Plot the sliding window
		rect=Rectangle((0,-0.5),dt,cnum,linewidth=1,linestyle='--',edgecolor='k',facecolor='none',clip_on=False)
		ax1.add_patch(rect)
		ax1.arrow(dt+0.1,-0.5, 0.5,0,head_width=0.1, head_length=0.1,facecolor='k',edgecolor='k')
		
		# plot parameters
		ax1.set_title(r'$\tilde{w}_%i = %.3f ms$'%(i,1000*w))
		ax1.set_ylabel('cluster #')
		ax1.set_yticks(range(0,cnum))
		ax1.set_xlabel('time')
		ax1.set_xlim([0,T])
		ax1.axes.xaxis.set_visible(False)
		ax1.spines['bottom'].set_visible(False)
		ax1.spines['top'].set_visible(False)
		ax1.spines['right'].set_visible(False)
		ax1.spines['left'].set_visible(False)

		# plot for fishcount
		x = fishcount[0]
		y = fishcount[1]

		ax2 = fig.add_subplot(gs[1])
		ax2.spines['top'].set_visible(False)
		ax2.spines['right'].set_visible(False)
		ax2.spines['bottom'].set_visible(False)
		ax2.axes.xaxis.set_visible(False)

		yplot = np.copy(y)
		ax2.plot(x+dt/2,yplot,linestyle='-',marker='.',c=cmap(i),alpha=0.25)
		yplot[ignore_steps.astype(bool)] = np.NaN
		ax2.plot(x+dt/2,yplot,linestyle='-',marker='.',c=cmap(i))
		ax2.set_ylabel('Fish count')
		ax2.set_yticks(range(int(np.min(y)),1+int(np.max(y))))
		ax2.set_xlim([0,T])

		if i < len(ws)-1:
		    ax2.axes.xaxis.set_visible(False)
		else:
			ax2.axes.xaxis.set_visible(False)
			sb.xscalebar(ax2,1,0,1,wunit='s',ha='right')

		con = ConnectionPatch([0,-0.5], [dt/2,y[0]], "data", "data",
		    axesA=ax1, axesB=ax2,color='k')
		ax2.add_artist(con)
		con = ConnectionPatch([dt,-0.5], [dt/2,y[0]], "data", "data",
		    axesA=ax1, axesB=ax2,color='k')
		ax2.add_artist(con)

		plt.xlim([0,T])