import matplotlib.pyplot as plt
from matplotlib import gridspec
import numpy as np
from scipy import stats

from matplotlib.patches import ConnectionPatch, Rectangle
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

def plot_clustering(all_clusters,all_clusters_ad,clusters_merge,uwl,all_uhl,width_labels,all_hightlabels,all_shapelabels,eod_widths,eod_hights,all_snippets,all_features):

	ad_count = 0
	merge_count = 0
	n_block = 0

	newplot=0
	mh = 0
	msum = 0
	for sl in all_shapelabels:
		mh = max(mh,len(sl)*2)
		msum = msum+len(sl)

	fig = plt.figure(figsize=(8,5))
	transFigure = fig.transFigure.inverted()

	gl = len(all_hightlabels)*mh

	# set up the figure layout
	outer = gridspec.GridSpec(1,5,width_ratios=[1,1,2,1,2])

	width_hist_ax = gridspec.GridSpecFromSubplotSpec(1,1,subplot_spec = outer[0])
	hight_hist_ax = gridspec.GridSpecFromSubplotSpec(len(uwl),1,subplot_spec = outer[1])

	shape_ax = gridspec.GridSpecFromSubplotSpec(msum,1, subplot_spec = outer[2])
	shape_windows = [gridspec.GridSpecFromSubplotSpec(2,2, hspace=0.0, wspace=0.0, subplot_spec = shape_ax[i]) for i in range(msum)]
    
	del_ax_sum = len(np.unique(all_clusters_ad[0][all_clusters_ad[0]>=0])) + len(np.unique(all_clusters_ad[1][all_clusters_ad[1]>=0]))
	
	merge_ax_sum = len(np.unique(clusters_merge[clusters_merge>=0]))

	EOD_delete_ax = gridspec.GridSpecFromSubplotSpec(del_ax_sum,1,subplot_spec=outer[3])
	EOD_merge_ax = gridspec.GridSpecFromSubplotSpec(merge_ax_sum,1,subplot_spec=outer[4])

    # plot width labels histogram
	ax1 = fig.add_subplot(width_hist_ax[0])

	for i, (wl, uhl, ahl, asl, cfeat, csnip) in enumerate(zip(uwl,all_uhl,all_hightlabels,all_shapelabels,all_features,all_snippets)):
		colidxs = np.linspace(0,155,len(uwl)).astype('int')
		hw,_,_ = ax1.hist(eod_widths[width_labels==wl],bins=np.linspace(np.min(eod_widths),np.max(eod_widths),100),color=cmap3(colidxs[i]),orientation='horizontal')
		
		ax1.set_xscale('log')
		ax1.spines['top'].set_visible(False)
		ax1.spines['right'].set_visible(False)
		ax1.spines['bottom'].set_visible(False)
		ax1.axes.xaxis.set_visible(False)
		ax1.axes.yaxis.set_visible(False)
		ax1.set_title('1.')

		my,b = np.histogram(eod_hights,bins=np.exp(np.linspace(np.min(np.log(eod_hights)),np.max(np.log(eod_hights)),100)))
		maxy = np.max(my)
		
		ax2 = fig.add_subplot(hight_hist_ax[len(uwl)-i-1])
		if len(uwl)-i-1 == 0:
			ax2.set_title('2.')

		ax2.set_yscale('log')
		ax2.set_xscale('log')
		ax2.spines['top'].set_visible(False)
		ax2.spines['right'].set_visible(False)
		ax2.spines['bottom'].set_visible(False)
		ax2.axes.xaxis.set_visible(False)
		ax2.axes.yaxis.set_visible(False)
		i

		ceodh = eod_hights[width_labels==wl]
		cc_ad = [all_clusters_ad[0][width_labels==wl],all_clusters_ad[1][width_labels==wl]]
		cc_merge = clusters_merge[:,width_labels==wl]

		for n, (hl, asll) in enumerate(zip(uhl,asl)):

			axs = []
			axs_m = []

			cf = [cfeat[0][ahl==hl],cfeat[1][ahl==hl]]
			cs = [csnip[0][ahl==hl],csnip[1][ahl==hl]]

			c_ad = [cc_ad[0][ahl==hl],cc_ad[1][ahl==hl]]
			c_merge = cc_merge[:,ahl==hl]

			colidxs = np.linspace(0,155,len(uhl)).astype('int')
			hh,_,_=loghist(ax2,ceodh[ahl==hl],np.min(eod_hights),np.max(eod_hights),100,cmap2(colidxs[n]),orientation='horizontal')

			ax2.set_xlim([0.9,maxy])

			if n==0:
				coord1 = transFigure.transform(ax1.transData.transform([np.median(hw[hw!=0]),np.median(eod_widths[width_labels==wl])]))
				coord2 = transFigure.transform(ax2.transData.transform([0.9,np.mean(eod_hights)]))
				line = matplotlib.lines.Line2D((coord1[0],coord2[0]),(coord1[1],coord2[1]),
				                               transform=fig.transFigure,color='grey',linewidth=0.5)
				fig.lines.append(line)

			
			# begin bij tellen bij N - # unique clusters voor peak en trough
			n_block = n_block + len(np.unique(c_ad[0][c_ad[0]>=0])) + len(np.unique(c_ad[1][c_ad[1]>=0]))
			ad_count = 0
			for pt in [0,1]:
				
				ax3 = fig.add_subplot(shape_windows[msum-1-newplot][pt,0])
				if msum-1-newplot ==0:
					ax3.set_title('3.')
				ax4 = fig.add_subplot(shape_windows[msum-1-newplot][pt,1])
				
				ax3.axes.xaxis.set_visible(False)
				ax4.axes.yaxis.set_visible(False)
				ax3.axes.yaxis.set_visible(False)
				ax4.axes.xaxis.set_visible(False)

				colidxs = np.linspace(0,155,len(np.unique(asll[pt][asll[pt]>=0]))).astype('int')
				j=0

				for c in np.unique(asll[pt]):
					if c<0:
						ax3.plot(cf[pt][asll[pt]==c,0],cf[pt][asll[pt]==c,1],'.',color='lightgrey',label='-1',rasterized=True)
						ax4.plot(cs[pt][asll[pt]==c].T,color='lightgrey',label='-1',rasterized=True)
					else:
						ax3.plot(cf[pt][asll[pt]==c,0],cf[pt][asll[pt]==c,1],'.',color=cmap1(colidxs[j]),label=c,rasterized=True)
						ax4.plot(cs[pt][asll[pt]==c].T,color=cmap1(colidxs[j]),label=c,rasterized=True)
						
						if np.sum(c_ad[pt][asll[pt]==c]) >=0:
							#npa = len(np.unique(c_ad[1][c_ad[1]>=0]))

							ax = fig.add_subplot(EOD_delete_ax[del_ax_sum - n_block + ad_count])
							if del_ax_sum - n_block + ad_count == 0:
								ax.set_title('4.')
							
							ax.axis('off')
							ax.plot(np.mean(cs[pt][asll[pt]==c],axis=0),color=cmap1(colidxs[j]))
							ad_count = ad_count + 1

							# match colors and draw line..							
							coord1 = transFigure.transform(ax4.transData.transform([ax4.get_xlim()[1], ax4.get_ylim()[0] + 0.5*(ax4.get_ylim()[1]-ax4.get_ylim()[0])]))
							coord2 = transFigure.transform(ax.transData.transform([ax.get_xlim()[0],ax.get_ylim()[0] + 0.5*(ax.get_ylim()[1]-ax.get_ylim()[0])]))
							line = matplotlib.lines.Line2D((coord1[0],coord2[0]),(coord1[1],coord2[1]),
					                               transform=fig.transFigure,color='grey',linewidth=0.5)
							fig.lines.append(line)	
							axs.append(ax)

							if np.sum(c_merge[pt,asll[pt]==c])>0:
								ax = fig.add_subplot(EOD_merge_ax[merge_ax_sum-1-merge_count])
								if merge_ax_sum-1-merge_count ==0:
									ax.set_title('5.')
								ax.axis('off')
								ax.plot(np.mean(cs[pt][asll[pt]==c],axis=0),color=cmap1(colidxs[j]))
								merge_count = merge_count + 1
								# match colors and draw line..
								axs_m.append(ax)
						j=j+1


				if pt==0:
					coord1 = transFigure.transform(ax2.transData.transform([np.median(hh[hh!=0]),np.median(ceodh[ahl==hl])]))
					coord2 = transFigure.transform(ax3.transData.transform([ax3.get_xlim()[0],ax3.get_ylim()[0]]))
					line = matplotlib.lines.Line2D((coord1[0],coord2[0]),(coord1[1],coord2[1]),
					                               transform=fig.transFigure,color='grey',linewidth=0.5)
					fig.lines.append(line)	

			for axp in axs:
				for ax in axs_m:
					coord1 = transFigure.transform(axp.transData.transform([axp.get_xlim()[1], axp.get_ylim()[0] + 0.5*(axp.get_ylim()[1]-axp.get_ylim()[0])]))
					coord2 = transFigure.transform(ax.transData.transform([ax.get_xlim()[0],ax.get_ylim()[0] + 0.5*(ax.get_ylim()[1]-ax.get_ylim()[0])]))
					line = matplotlib.lines.Line2D((coord1[0],coord2[0]),(coord1[1],coord2[1]),
                               transform=fig.transFigure,color='grey',linewidth=0.5)
					fig.lines.append(line)
				
			newplot = newplot+1

def plot_bgm(model,x,x_transform,labels,labels_before_merge, xlab,use_log):	
	
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
	ax1.set_xlabel(xlab)
	plt.tight_layout()

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
	
	fig.legend(handles=legend_elements,   # The labels for each line
           loc="center right",   # Position of legend
           ncol= 1,
           frameon=False,
           title='Cluster #'
           )
	plt.subplots_adjust(right=0.85)

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


def plot_moving_fish(fig,gs,iw,wclusters,weod_t,ev_num,dt,weod_widths):
	ax1 = fig.add_subplot(gs[iw*2])
	cnum = 0
	for c in np.unique(wclusters[wclusters>=0]):
	    plt.eventplot(weod_t[wclusters==c],lineoffsets=ev_num,linelengths=0.5,color=cmap(iw))
	    ev_num = ev_num+1
	    cnum = cnum + 1

	rect=Rectangle((0,ev_num-cnum-0.5),dt,cnum,linewidth=1,linestyle='--',edgecolor='k',facecolor='none',clip_on=False)
	# Add the patch to the Axes
	ax1.add_patch(rect)
	ax1.arrow(dt+0.1,ev_num-cnum-0.5, 0.5,0,head_width=0.1, head_length=0.1,facecolor='k',edgecolor='k')
	ax1.set_title(r'$\tilde{w}_%i = %.3f ms$'%(iw,1000*np.median(weod_widths)))
	return ax1, cnum

def plot_fishcount(fig,ax1,ev_num,cnum,T,gs,iw,wc_num,x,y,dt,ignore_steps):
    ax1.set_ylabel('cluster #')
    ax1.set_yticks(range(ev_num-cnum,ev_num))
    ax1.set_xlabel('time')
    
    #plt.legend(frameon=False)
    ax1.set_xlim([0,T])
    ax1.axes.xaxis.set_visible(False)
    ax1.spines['bottom'].set_visible(False)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.spines['left'].set_visible(False)

    ax2 = fig.add_subplot(gs[2*iw+1])
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['left'].set_visible(False)
    ax2.spines['bottom'].set_visible(False)
    
    if iw < wc_num-1:
        ax2.axes.xaxis.set_visible(False)
    else:
        ax2.set_xlabel('Time [ms]')
    
    yplot = np.copy(y)
    ax2.plot(x+dt/2,yplot,linestyle='-',marker='.',c=cmap(iw),alpha=0.25)
    yplot[ignore_steps.astype(bool)] = np.NaN
    ax2.plot(x+dt/2,yplot,linestyle='-',marker='.',c=cmap(iw))
    ax2.set_ylabel('Fish count')
    ax2.set_yticks(range(int(np.min(y)),1+int(np.max(y))))
    ax2.set_xlim([0,T])

    con = ConnectionPatch([0,ev_num-cnum-0.5], [dt/2,y[0]], "data", "data",
        axesA=ax1, axesB=ax2,color='k')
    ax2.add_artist(con)
    con = ConnectionPatch([dt,ev_num-cnum-0.5], [dt/2,y[0]], "data", "data",
        axesA=ax1, axesB=ax2,color='k')
    ax2.add_artist(con)

    plt.xlim([0,T])

def plot_all(data, eod_p_times, eod_tr_times, fs, mean_eods):
    '''
    Quick way to view the output and intermediate steps of extract_pulsefish in a plot.

    Parameters:
    -----------
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
    '''

    try:
        cmap = plt.get_cmap("tab10")
    except ValueError:
        cmap = plt.get_cmap("jet")

    try:
        fig = plt.figure(constrained_layout=True,figsize=(10,5))
    except TypeError:
        fig = plt.figure(figsize=(10,5))
    if len(eod_p_times) > 0:
        gs = gridspec.GridSpec(2, len(eod_p_times))
        ax = fig.add_subplot(gs[0,:])
        ax.plot(np.arange(len(data))/fs,data,c='k',alpha=0.3)
        
        for i,(pt,tt) in enumerate(zip(eod_p_times,eod_tr_times)):
            ax.plot(pt,data[(pt*fs).astype('int')],'o',label=i+1,ms=10,c=cmap(i))
            ax.plot(tt,data[(tt*fs).astype('int')],'o',label=i+1,ms=10,c=cmap(i))
            
        #for i,t in enumerate(eod_p_times):
        #    ax.plot(t,data[(t*fs).astype('int')],'o',label=i+1,c=cmap(i))
        ax.set_xlabel('time [s]')
        ax.set_ylabel('amplitude [V]')
        #ax.axis('off')

        for i, m in enumerate(mean_eods):
            ax = fig.add_subplot(gs[1,i])
            ax.plot(1000*m[0], 1000*m[1], c='k')

            ax.fill_between(1000*m[0],1000*(m[1]-m[2]),1000*(m[1]+m[2]),color=cmap(i))
            ax.set_xlabel('time [ms]')
            ax.set_ylabel('amplitude [mV]') 
    else:
        plt.plot(np.arange(len(data))/fs,data,c='k',alpha=0.3)
