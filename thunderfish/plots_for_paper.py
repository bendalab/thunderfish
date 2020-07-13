import matplotlib.pyplot as plt
from matplotlib import gridspec
import numpy as np
from scipy import stats

from matplotlib.patches import ConnectionPatch
import matplotlib
from matplotlib.lines import Line2D

# upgrade numpy functions for backwards compatibility:
if not hasattr(np, 'isin'):
    np.isin = np.in1d

def unique_counts(ar):
    """
    Find the unique elements of an array and their counts, ignoring shape.

    The following code is condensed from numpy version 1.17.0.
    """
    try:
        return np.unique(ar, return_counts=True)
    except TypeError:
        ar = np.asanyarray(ar).flatten()
        ar.sort()
        mask = np.empty(ar.shape, dtype=np.bool_)
        mask[:1] = True
        mask[1:] = ar[1:] != ar[:-1]
        idx = np.concatenate(np.nonzero(mask) + ([mask.size],))
        return ar[mask], np.diff(idx)  

cmap = plt.get_cmap("Dark2")
cmap1 = plt.get_cmap('Reds_r')
cmap2 = plt.get_cmap('Blues_r')
cmap3 = plt.get_cmap('YlOrBr_r')

# add layers for artefact detection, wave, sidepeak, and finally merge :)

def loghist(ax,x,bmin,bmax,n,c,orientation='vertical',label=''):
	return ax.hist(x,bins=np.exp(np.linspace(np.log(bmin),np.log(bmax),n)),color=c,orientation=orientation,label=label)

def plot_clustering(all_clusters,all_clusters_aduf, all_clusters_adwas,mask,uwl,all_uhl,width_labels,all_hightlabels,all_shapelabels,all_shapelabels_ad,eod_widths,eod_hights,all_snippets,all_features):

	newplot=0
	mh = 0
	msum = 0
	for sl in all_shapelabels:
		mh = max(mh,len(sl)*2)
		msum = msum+len(sl)

	fig = plt.figure(figsize=(5,5))
	transFigure = fig.transFigure.inverted()

	gl = len(all_hightlabels)*mh

	# set up the figure layout
	outer = gridspec.GridSpec(1,3,width_ratios=[1,1,2])

	width_hist_ax = gridspec.GridSpecFromSubplotSpec(1,1,subplot_spec = outer[0])
	hight_hist_ax = gridspec.GridSpecFromSubplotSpec(len(uwl),1,subplot_spec = outer[1])

	shape_ax = gridspec.GridSpecFromSubplotSpec(msum,1, subplot_spec = outer[2])
	shape_windows = [gridspec.GridSpecFromSubplotSpec(2,2, hspace=0.0, wspace=0.0, subplot_spec = shape_ax[i]) for i in range(msum)]
    
    # plot width labels histogram
	ax1 = fig.add_subplot(width_hist_ax[0])

	for i, (wl, uhl, ahl, asl, asl_ad, cfeat, csnip) in enumerate(zip(uwl,all_uhl,all_hightlabels,all_shapelabels,all_shapelabels_ad,all_features,all_snippets)):
		colidxs = np.linspace(0,155,len(uwl)).astype('int')
		#loghist(ax1,eod_widths[width_labels==wl],np.min(eod_widths),np.max(eod_widths),100,cmap3(colidxs[i]))
		hw,_,_ = ax1.hist(eod_widths[width_labels==wl],bins=np.linspace(np.min(eod_widths),np.max(eod_widths),100),color=cmap3(colidxs[i]),orientation='horizontal')
		
		ax1.set_xscale('log')
		ax1.spines['top'].set_visible(False)
		ax1.spines['right'].set_visible(False)
		ax1.spines['bottom'].set_visible(False)
		ax1.axes.xaxis.set_visible(False)
		ax1.axes.yaxis.set_visible(False)

		my,b = np.histogram(eod_hights,bins=np.exp(np.linspace(np.min(np.log(eod_hights)),np.max(np.log(eod_hights)),100)))
		maxy = np.max(my)
		
		ax2 = fig.add_subplot(hight_hist_ax[len(uwl)-i-1])

		ax2.set_yscale('log')
		ax2.set_xscale('log')
		ax2.spines['top'].set_visible(False)
		ax2.spines['right'].set_visible(False)
		ax2.spines['bottom'].set_visible(False)
		ax2.axes.xaxis.set_visible(False)
		ax2.axes.yaxis.set_visible(False)
		#ax2.spines['left'].set_linewidth(1)


		#ax1.scatter([10],[np.median(eod_widths[width_labels==wl])])
		#ax2.scatter([1],np.mean(eod_hights))
		

		#transFigure = fig.transFigure.inverted()
		#coord1 = transFigure.transform(ax1.transData.transform([np.median(eod_widths[width_labels==wl]),0]))
		#coord2 = transFigure.transform(ax2.transData.transform([np.mean(eod_hights),maxy]))
		#line = matplotlib.lines.Line2D((coord1[0],coord2[0]),(coord1[1],coord2[1]),
		#                               transform=fig.transFigure,clip_on=False)
		#fig.lines.append(line)
		
		#if  i>0:
		#	ax2.spines['left'].set_visible(False)
			#ax2.get_yaxis().set_ticks([])

		ceodh = eod_hights[width_labels==wl]

		cc_aduf = all_clusters_aduf[width_labels==wl]
		cc_adwas = all_clusters_adwas[width_labels==wl]
		cc_ac = all_clusters[width_labels==wl]
		cc_mask = mask[width_labels==wl]


		for n, (hl, asll) in enumerate(zip(uhl,asl)):

			cf = [cfeat[0][ahl==hl],cfeat[1][ahl==hl]]
			cs = [csnip[0][ahl==hl],csnip[1][ahl==hl]]
			c_asl_ad = [asl_ad[0][ahl==hl], asl_ad[1][ahl==hl]]

			c_aduf = cc_aduf[ahl==hl]
			c_adwas = cc_adwas[ahl==hl]
			c_ac = cc_ac[ahl==hl]
			c_mask = cc_mask[ahl==hl]

			colidxs = np.linspace(0,155,len(uhl)).astype('int')
			#ax2.hist(ceodh[ahl==hl],bins=np.exp(np.linspace(np.min(np.log(eod_hights)),np.max(np.log(eod_hights)),100)))
			hh,_,_=loghist(ax2,ceodh[ahl==hl],np.min(eod_hights),np.max(eod_hights),100,cmap2(colidxs[n]),orientation='horizontal')

			ax2.set_xlim([0.9,maxy])

			if n==0:
				coord1 = transFigure.transform(ax1.transData.transform([np.median(hw[hw!=0]),np.median(eod_widths[width_labels==wl])]))
				coord2 = transFigure.transform(ax2.transData.transform([0.9,np.mean(eod_hights)]))
				line = matplotlib.lines.Line2D((coord1[0],coord2[0]),(coord1[1],coord2[1]),
				                               transform=fig.transFigure,color='grey',linewidth=0.5)
				fig.lines.append(line)

			

			for pt in [0,1]:
				ax3 = fig.add_subplot(shape_windows[msum-1-newplot][pt,0])
				ax4 = fig.add_subplot(shape_windows[msum-1-newplot][pt,1])
				
				#ax5 = fig.add_subplot(gs[4,i*mh+n*2+pt])
				#ax6 = fig.add_subplot(gs[5,i*mh+n*2+pt])
				#ax7 = fig.add_subplot(gs[4,i*mh+n*2+pt])
				#ax8 = fig.add_subplot(gs[5,i*mh+n*2+pt])
				#ax3.spines['top'].set_visible(False)
				#ax3.spines['right'].set_visible(False)

				#if  i>0 or n>0 or pt>0:
				#	ax3.spines['left'].set_visible(False)
				#	ax3.get_yaxis().set_ticks([])

				#ax3.spines['right'].set_visible(False)
				#ax4.spines['left'].set_visible(False)
				#ax4.spines['right'].set_visible(False)

				#if pt == 1:
				#	ax3.spines['bottom'].set_visible(False)
				#	ax4.spines['bottom'].set_visible(False)
				#else:
				#	ax3.spines['top'].set_visible(False)
				#	ax4.spines['top'].set_visible(False)


				#if  i>0 or n>0:
				#	ax4.spines['left'].set_visible(False)
				#	ax4.get_yaxis().set_ticks([])
				ax3.axes.xaxis.set_visible(False)
				ax4.axes.yaxis.set_visible(False)
				ax3.axes.yaxis.set_visible(False)
				ax4.axes.xaxis.set_visible(False)
				#ax5.axis('off')
				#ax6.axis('off')
				#ax7.axis('off')
				#ax8.axis('off')

				colidxs = np.linspace(0,155,len(np.unique(asll[pt][asll[pt]>=0]))).astype('int')
				j=0

				
				for c in np.unique(asll[pt]):
					if c<0:
						ax3.plot(cf[pt][asll[pt]==c,0],cf[pt][asll[pt]==c,1],'.',color='lightgrey',label='-1',rasterized=True)
						ax4.plot(cs[pt][asll[pt]==c].T,color='lightgrey',label='-1',rasterized=True)
					else:
						ax3.plot(cf[pt][asll[pt]==c,0],cf[pt][asll[pt]==c,1],'.',color=cmap1(colidxs[j]),label=c,rasterized=True)
						ax4.plot(cs[pt][asll[pt]==c].T,color=cmap1(colidxs[j]),label=c,rasterized=True)
						j=j+1

				if pt==0:

					coord1 = transFigure.transform(ax2.transData.transform([np.median(hh[hh!=0]),np.median(ceodh[ahl==hl])]))
					coord2 = transFigure.transform(ax3.transData.transform([ax3.get_xlim()[0],ax3.get_ylim()[0]]))
					line = matplotlib.lines.Line2D((coord1[0],coord2[0]),(coord1[1],coord2[1]),
					                               transform=fig.transFigure,color='grey',linewidth=0.5)
					fig.lines.append(line)	
				

				'''
				j=0
				for c in np.unique(c_asl_ad[pt]):
					if c<0 and np.sum(asll[pt][c_asl_ad[pt]])>0:
						ax5.plot(cs[pt][c_asl_ad[pt]==c].T,color='lightgrey',label='-1',rasterized=True)
					else:
						ax5.plot(cs[pt][c_asl_ad[pt]==c].T,color=cmap1(colidxs[j]),label=c,rasterized=True)
						j=j+1
				j=0
				for c in np.unique(c_aduf[:,pt]):
					if c<0 and np.sum(c_asl_ad[pt][c_aduf[:,pt]==c])>0:
						pass
						#ax6.plot(cs[pt][c_aduf[:,pt]==c].T,color='lightgrey',label='-1',rasterized=True)
					else:
						ax6.plot(cs[pt][c_aduf[:,pt]==c].T,color=cmap1(colidxs[j]),label=c,rasterized=True)
						j=j+1
				

				j=0
				for c in np.unique(c_adwas[:,pt]):
					if c<0 and np.sum(asll[pt][c_adwas[:,pt]==c])>0:
						ax7.plot(cs[pt][c_adwas[:,pt]==c].T,color='lightgrey',label='-1',rasterized=True)
					elif c<0:
						pass
					else:
						ax7.plot(cs[pt][c_adwas[:,pt]==c].T,color=cmap1(colidxs[j]),label=c,rasterized=True)
						
						if np.sum(c_mask[:,pt][c_adwas[:,pt]==c])>0:
							ax8.plot(cs[pt][c_adwas[:,pt]==c].T,color=cmap1(colidxs[j]),label=c,rasterized=True)
						j=j+1
				'''
			newplot = newplot+1
	#plt.tight_layout()
	
	
	
	
	#con = ConnectionPatch(xyA=[10,np.median(eod_widths[width_labels==wl])], xyB=[10,10], coordsA="data", coordsB="data",
    #          axesA=ax2, axesB=ax1, color="red")
	#ax2.add_artist(con)
	plt.show()

def plot_bgm(model,x,x_transform,labels,labels_before_merge, xlab1,xlab2,use_log):	
	
	fig, ax1 = plt.subplots()
	ax2 = ax1.twinx()
	ax1.spines['top'].set_visible(False)
	ax2.spines['top'].set_visible(False)

	x2 = np.linspace(np.min(x_transform),np.max(x_transform),1000)
	if use_log:
		xplot = np.exp(np.linspace(np.log(np.min(x)),np.log(np.max(x)),1000))
	else:
		xplot = np.linspace(np.min(x),np.max(x),1000)
		bins2 = np.linspace(np.min(x),np.max(x),100)

	gaussians = []
	gmax = 0

	ah, _, _ = loghist(ax1,x,np.min(x),np.max(x),100,'k')
	ax1.clear()

	#sort model attributes by model_means_
	means = [m[0] for m in model.means_]
	weights = [w for w in model.weights_]
	variances = [v[0][0] for v in model.covariances_]
	weights = [w for _,w in sorted(zip(means,weights))]
	variances =  [v for _,v in sorted(zip(means,variances))]
	means =  sorted(means)

	for i, (w,m,std) in enumerate(zip(weights, means, variances)):
		gaus = np.sqrt(w*stats.norm.pdf(x2,m,np.sqrt(std)))
		gaussians.append(gaus)
		gmax = max(gmax,np.max(gaus))
		# plot where argmax changes.
	
	classes = np.argmax(np.vstack(gaussians),axis=0)

	# find the minimum of any gaussian that is within its class and set to 1
	# but I only care about the ones that are labeled..

	gmin = 100
	for i,c in enumerate(np.unique(classes)):
		gmin=min(gmin,np.min(gaussians[c][classes==c]))
	
	colidxs = np.linspace(0,155,len(np.unique(classes))).astype('int')

	for i,c in enumerate(np.unique(classes)):
		ax2.plot(xplot,gaussians[c],c=cmap1(colidxs[i]),linewidth=2,label=r'$N(\mu_%i,\sigma_%i)$'%(c,c))
	
	ax2.vlines(xplot[1:][np.diff(classes)!=0],0,gmax/gmin,color='k',linewidth=2,linestyle='--')
	ax2.set_ylim([gmin,gmax*1.1])

	ax1.set_xlabel('x [a.u.]')
	ax1.set_ylabel('#')
	ax2.set_ylabel('Likelihood')
	ax2.set_yscale('log')

	colidxs = np.linspace(0,155,len(np.unique(labels_before_merge))).astype('int')

	hs = []
	bins = []

	for i,l in enumerate(np.unique(labels_before_merge)):
		if use_log:
			h,binn,_=loghist(ax1,x[labels_before_merge==l],np.min(x),np.max(x),100,cmap2(colidxs[i]),label=r'$x_%i$'%l)
			hs.append(h)
			bins.append(binn)
		else:
			h,binn,_=ax1.hist(x[labels_before_merge==l],bins=bins2,color=cmap2(colidxs[i]),label=r'$x_%i$'%l)
			hs.append(h)
			bins.append(binn)

	legend_loc = 1.2
	
	# if one label maps to two classes, blur out the line and gaussians for these
	# maybe draw a line and write sth like median..=x~.. |mu1-mu2|/(mu1+mu2) < eta?
	for l in np.unique(labels):
		maps = np.unique(labels_before_merge[labels==l])
		if len(maps) > 1:
			x1 = x[labels_before_merge==maps[0]]
			x2 = x[labels_before_merge==maps[1]]

			#line = matplotlib.lines.Line2D([np.median(x1),np.median(x2)],[gmax,gmax],color='grey')
			#line = matplotlib.lines.Line2D([np.median(x1),np.median(x2)],[gmax,gmax],color='grey')
			ax2.plot([np.median(x1),np.median(x2)],[gmax*1.2,gmax*1.2],c='k',clip_on=False)
			ax2.plot([np.median(x1),np.median(x1)],[gmax*1.2,gmax*1.1],c='k',clip_on=False)
			ax2.plot([np.median(x2),np.median(x2)],[gmax*1.2,gmax*1.1],c='k',clip_on=False)

			#ax2.text(np.median(x1)*1.1,gmax*1.35,r'$\frac{|{\tilde{x}_%i-\tilde{x}_%i}|}{max(\tilde{x}_%i,\tilde{x}_%i)} < \epsilon$'%(maps[0],maps[1],maps[0],maps[1]),fontsize=12)
			ax2.annotate(r'$\frac{|{\tilde{x}_%i-\tilde{x}_%i}|}{max(\tilde{x}_%i,\tilde{x}_%i)} < \epsilon$'%(maps[0],maps[1],maps[0],maps[1]),[np.median(x1)*1.1,gmax*1.2], xytext=(0, 10), textcoords='offset points',fontsize=12,annotation_clip=False)
			#line.set_clip_on(False)

			#ax2.lines.append(line)

			legend_loc = 1.4
		
			# line between these should be blurred.. so save it?

	ax1.set_yscale('log')
	ax2.legend(loc='lower left',frameon=False,bbox_to_anchor=(0,legend_loc),ncol=len(np.unique(classes)))
	ax1.legend(loc='upper left',frameon=False,bbox_to_anchor=(0,legend_loc),ncol=len(np.unique(labels_before_merge)))
	if use_log:
		ax1.set_xscale('log')
	ax1.set_xlabel(xlab2)
	plt.tight_layout()
	plt.show()


def plot_merge(c1,c2):
	plt.figure()
	colidxs1 = np.linspace(0,155,len(np.unique(c1[c1!=-1]))).astype('int')
	colidxs2 = np.linspace(0,155,len(np.unique(c2[c2!=-1]))).astype('int')

	plt.subplot(121)
	uv, uc = unique_counts(c1[c1>=0])
	plt.bar(range(len(uv)),uc,color=cmap1(colidxs1))

	plt.subplot(122)
	uv, uc = unique_counts(c2[c2>=0])
	plt.bar(range(len(uv)),uc,color=cmap2(colidxs2))
	plt.show()

	plt.figure()
	
	for i,c in enumerate(np.unique(c1[c1!=-1])):
		plt.plot(np.arange(len(c1))[c1==c],np.zeros(len(c1[c1==c])),'o',c=cmap1(colidxs1[i]),label='N = %i'%len(c1[c1==c]))
	for i,c in enumerate(np.unique(c2[c2!=-1])):
		plt.plot(np.arange(len(c2))[c2==c],np.ones(len(c2[c2==c])),'o',c=cmap2(colidxs2[i]),label='N = %i'%len(c2[c2==c]))

	plt.legend()
	plt.show()

def plot_peak_detection(data,samplerate,interp_f,orig_x_peaks,orig_x_troughs,peaks,troughs,x_peaks,x_troughs,apeaks, atroughs):
    plt.figure()
    dt = 1/(samplerate*interp_f)
    x = np.arange(0,len(data)*dt,dt)

    xmin = 0.1
    xmax = 0.2

    plt.subplot(411)
    plt.title('Detect all peaks and troughs')
    plt.plot(x,data,c='b')
    plt.plot(orig_x_peaks*dt,data[orig_x_peaks],'o', c='red', label='a. peaks')
    plt.plot(orig_x_troughs*dt,data[orig_x_troughs],'o',c='k',label='a. troughs')
    #plt.legend()
    plt.xlim([xmin,xmax])
    plt.ylim([1.1*np.min(data[(x<xmax) & (x>xmin)]),1.1*np.max(data[(x<xmax) & (x>xmin)])])

    plt.subplot(412)
    plt.title('Choose one trough for every peak')
    plt.plot(x,data,c='b')
    plt.plot(apeaks*dt,data[apeaks.astype('int')],'o', c='r', label='b. peaks')
    plt.plot(atroughs*dt,data[atroughs.astype('int')],'o', c='k', label= 'b. troughs')
    #plt.legend()
    plt.xlim([xmin,xmax])
    plt.ylim([1.1*np.min(data[(x<xmax) & (x>xmin)]),1.1*np.max(data[(x<xmax) & (x>xmin)])])


    plt.subplot(413)
    plt.title('Delete p-t combinations that are too narrow or too wide')
    plt.plot(x,data,c='b')
    plt.plot(peaks*dt,data[peaks.astype('int')],'o', c='r', label='b. peaks')
    plt.plot(troughs*dt,data[troughs.astype('int')],'o', c='k', label= 'b. troughs')
    #plt.legend()
    plt.xlim([xmin,xmax])
    plt.ylim([1.1*np.min(data[(x<xmax) & (x>xmin)]),1.1*np.max(data[(x<xmax) & (x>xmin)])])


    plt.subplot(414)
    plt.title('Delete connecting points (e.g. peaks with same trough)')
    plt.plot(x,data,c='b')
    plt.plot(x_peaks*dt,data[x_peaks.astype('int')],'o', c='r', label= 'c. peaks')
    plt.plot(x_troughs*dt,data[x_troughs.astype('int')],'o', c='k', label='c. troughs')        
    plt.xlim([xmin,xmax])
    plt.ylim([1.1*np.min(data[(x<xmax) & (x>xmin)]),1.1*np.max(data[(x<xmax) & (x>xmin)])])
    #plt.legend()
    plt.tight_layout()
    plt.show()


def plot_feature_extraction(raw_snippets, normalized_snippets, features, labels, dt):
	# set up the figure layout
	fig = plt.figure(figsize=(8,4))
	outer = gridspec.GridSpec(1,2)

	x = np.arange(-dt*1000*raw_snippets.shape[1]/2,dt*1000*raw_snippets.shape[1]/2,dt*1000)

	snip_ax = gridspec.GridSpecFromSubplotSpec(2,1,subplot_spec = outer[0],hspace=0.35)
	pc_ax = gridspec.GridSpecFromSubplotSpec(features.shape[1]-1,features.shape[1]-1,subplot_spec = outer[1],hspace=0,wspace=0)
	
	# 3 plots: raw snippets, normalized, pcs.
	ax_raw_snip = fig.add_subplot(snip_ax[0])
	ax_normalized_snip = fig.add_subplot(snip_ax[1])

	colidxs = np.arange(len(np.unique(labels[labels>=0]))).astype('int')#np.linspace(0,155,len(np.unique(labels))).astype('int')
	j=0

	legend_objects = []
	legend_labels = []

	for c in np.unique(labels):
		if c<0:
			color='lightgrey'
		else:
			color=cmap(colidxs[j])
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

		ax_overlay = fig.add_subplot(pc_ax[:,:])
		ax_overlay.set_title('Features')
		ax_overlay.axis('off')

		for n in range(features.shape[1]):
			for m in range(n):
				ax = fig.add_subplot(pc_ax[n-1,m])

				ax.scatter(features[labels==c,n],features[labels==c,m],marker='.',color=color,alpha=0.25)
				
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
		ax.get_xaxis().set_ticklabels([])
		ax.get_yaxis().set_ticklabels([])
		ax.get_xaxis().set_ticks([])
		ax.get_yaxis().set_ticks([])
		ax.spines['top'].set_visible(False)
		ax.spines['right'].set_visible(False)
		ax.spines['bottom'].set_linewidth(2)
		ax.spines['left'].set_linewidth(2)
		ax.set_xlabel('%.3f'%(np.max(features)-np.min(features)))
		ax.set_ylabel('%.3f'%(np.max(features)-np.min(features)))
	
	plt.tight_layout()

	legend_elements = []
	for i,ci in enumerate(colidxs):
		legend_elements.append(Line2D([0], [0], marker='.', color='w', label=i+1,
                  markerfacecolor=cmap(ci), markersize=15))
	legend_elements.append(Line2D([0], [0], marker='.', color='w', label='noise',
                  markerfacecolor='lightgrey', markersize=15))
	'''
	fig.legend(handles=legend_elements,   # The labels for each line
           loc="lower right",   # Position of legend
           ncol= len(legend_elements),
           frameon=False,
           title='Clusters:'
           )
	plt.subplots_adjust(bottom=0.25)
	'''
	fig.legend(handles=legend_elements,   # The labels for each line
           loc="center right",   # Position of legend
           ncol= 1,
           frameon=False,
           title='Cluster #'
           )
	plt.subplots_adjust(right=0.85)
	plt.show()

def plot_EOD_discarding(mean_eods, ffts, N):
	fig = plt.figure(figsize=(5,5))
	gs = gridspec.GridSpec(N,4,figure=fig) 

    # plot
	pnum = 0
	for i,(mw,fw) in enumerate(zip(mean_eods,ffts)):
		for j, (m,f) in enumerate(zip(mw,fw)):
			ax = fig.add_subplot(gs[pnum,0])
			ax.plot(m)
			ax = fig.add_subplot(gs[pnum,1])
			ax.plot(f[0][:2*f[2]],f[1][:2*f[2]])
			ax.vlines(f[0][f[2]],0,np.max(f[1]))
			ax.fill_between(f[0][:f[2]*2],f[1][:f[2]*2])
			ax.fill_between(f[0][:f[2]],f[1][:f[2]])
			pnum = pnum+1
	return fig,gs


def plot_moving_fish(xs,running_sums,widths):
	ax = plt.subplot(111)
	for i,(x,r,w) in enumerate(zip(xs,running_sums,widths)):
		ax.plot(x,r,c=cmap(i),label='width = '%w)
	plt.legend()
	plt.show()