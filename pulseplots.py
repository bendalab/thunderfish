"""
## Plot and save key steps in pulses.py for visualizing the alorithm.

"""

import pickle, glob
import numpy as np
from scipy import stats
from matplotlib import rcParams, gridspec, ticker
import matplotlib.pyplot as plt
try:
    from matplotlib.colors import colorConverter as cc
except ImportError:
    import matplotlib.colors as cc
try:
    from matplotlib.colors import to_hex
except ImportError:
    from matplotlib.colors import rgb2hex as to_hex
from matplotlib.patches import ConnectionPatch, Rectangle
from matplotlib.lines import Line2D
import warnings
def warn(*args,**kwargs):
    """
    Ignore all warnings.
    """
    pass
warnings.warn=warn

# plotting parameters and colors
rcParams['font.family'] = 'monospace'
cmap = plt.get_cmap("Dark2")
c_g = cmap(0)
c_o = cmap(1)
c_grey = cmap(7)
cmap_pts = [cmap(2),cmap(3)]

def darker(color, saturation):
    """Make a color darker.

    From bendalab/plottools package.

    Parameters
    ----------
    color: dict or matplotlib color spec
        A matplotlib color (hex string, name color string, rgb tuple)
        or a dictionary with an 'color' or 'facecolor' key.
    saturation: float
        The smaller the saturation, the darker the returned color.
        A saturation of 0 returns black.
        A saturation of 1 leaves the color untouched.
        A saturation of 2 returns white.
    Returns
    -------
    color: string or dictionary
        The darker color as a hexadecimal RGB string (e.g. '#rrggbb').
        If `color` is a dictionary, a copy of the dictionary is returned
        with the value of 'color' or 'facecolor' set to the darker color.
    """
    try:
        c = color['color']
        cd = dict(**color)
        cd['color'] = darker(c, saturation)
        return cd
    except (KeyError, TypeError):
        try:
            c = color['facecolor']
            cd = dict(**color)
            cd['facecolor'] = darker(c, saturation)
            return cd
        except (KeyError, TypeError):
            if saturation > 2:
                sauration = 2
            if saturation > 1:
                return lighter(color, 2.0-saturation)
            if saturation < 0:
                saturation = 0
            r, g, b = cc.to_rgb(color)
            rd = r*saturation
            gd = g*saturation
            bd = b*saturation
            return to_hex((rd, gd, bd)).upper()

def lighter(color, lightness):
    """Make a color lighter.

    From bendalab/plottools package.

    Parameters
    ----------
    color: dict or matplotlib color spec
        A matplotlib color (hex string, name color string, rgb tuple)
        or a dictionary with an 'color' or 'facecolor' key.
    lightness: float
        The smaller the lightness, the lighter the returned color.
        A lightness of 0 returns white.
        A lightness of 1 leaves the color untouched.
        A lightness of 2 returns black.

    Returns
    -------
    color: string or dict
        The lighter color as a hexadecimal RGB string (e.g. '#rrggbb').
        If `color` is a dictionary, a copy of the dictionary is returned
        with the value of 'color' or 'facecolor' set to the lighter color.
    """
    try:
        c = color['color']
        cd = dict(**color)
        cd['color'] = lighter(c, lightness)
        return cd
    except (KeyError, TypeError):
        try:
            c = color['facecolor']
            cd = dict(**color)
            cd['facecolor'] = lighter(c, lightness)
            return cd
        except (KeyError, TypeError):
            if lightness > 2:
                lightness = 2
            if lightness > 1:
                return darker(color, 2.0-lightness)
            if lightness < 0:
                lightness = 0
            r, g, b = cc.to_rgb(color)
            rl = r + (1.0-lightness)*(1.0 - r)
            gl = g + (1.0-lightness)*(1.0 - g)
            bl = b + (1.0-lightness)*(1.0 - b)
            return to_hex((rl, gl, bl)).upper()


def xscalebar(ax, x, y, width, wunit=None, wformat=None, ha='left', va='bottom',
              lw=None, color=None, capsize=None, clw=None, **kwargs):
    """Horizontal scale bar with label.

    From bendalab/plottools package.

    Parameters
    ----------
    ax: matplotlib axes
        Axes where to draw the scale bar.
    x: float
        x-coordinate where to draw the scale bar in relative units of the axes.
    y: float
        y-coordinate where to draw the scale bar in relative units of the axes.
    width: float
        Length of the scale bar in units of the data's x-values.
    wunit: string or None
        Optional unit of the data's x-values.
    wformat: string or None
        Optional format string for formatting the label of the scale bar
        or simply a string used for labeling the scale bar.
    ha: 'left', 'right', or 'center'
        Scale bar aligned left, right, or centered to (x, y)
    va: 'top' or 'bottom'
        Label of the scale bar either above or below the scale bar.
    lw: int, float, None
        Line width of the scale bar.
    color: matplotlib color
        Color of the scalebar.
    capsize: float or None
        If larger then zero draw cap lines at the ends of the bar.
        The length of the lines is given in points (same unit as linewidth).
    clw: int, float, None
        Line width of the cap lines.
    kwargs: key-word arguments
        Passed on to `ax.text()` used to print the scale bar label.
    """
    ax.autoscale(False)
    # ax dimensions:
    pixelx = np.abs(np.diff(ax.get_window_extent().get_points()[:,0]))[0]
    pixely = np.abs(np.diff(ax.get_window_extent().get_points()[:,1]))[0]
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    unitx = xmax - xmin
    unity = ymax - ymin
    dxu = np.abs(unitx)/pixelx
    dyu = np.abs(unity)/pixely
    # transform x, y from relative units to axis units:
    x = xmin + x*unitx
    y = ymin + y*unity
    # bar length:
    if wformat is None:
        wformat = '%.0f'
        if width < 1.0:
            wformat = '%.1f'
    try:
        ls = wformat % width
        width = float(ls)
    except TypeError:
        ls = wformat
    # bar:
    if ha == 'left':
        x0 = x
        x1 = x+width
    elif ha == 'right':
        x0 = x-width
        x1 = x
    else:
        x0 = x-0.5*width
        x1 = x+0.5*width
    # line width:
    if lw is None:
        lw = 2
    # color:
    if color is None:
        color = 'k'
    # scalebar:
    lh = ax.plot([x0, x1], [y, y], '-', color=color, lw=lw,
                 solid_capstyle='butt', clip_on=False)
    # get y position of line in figure pixel coordinates:
    ly = np.array(lh[0].get_window_extent(ax.get_figure().canvas.get_renderer()))[0,1]
    # caps:
    if capsize is None:
        capsize = 0
    if clw is None:
        clw = 0.5
    if capsize > 0.0:
        dy = capsize*dyu
        ax.plot([x0, x0], [y-dy, y+dy], '-', color=color, lw=clw,
                solid_capstyle='butt', clip_on=False)
        ax.plot([x1, x1], [y-dy, y+dy], '-', color=color, lw=clw,
                solid_capstyle='butt', clip_on=False)
    # label:
    if wunit:
        ls += u'\u2009%s' % wunit
    if va == 'top':
        th = ax.text(0.5*(x0+x1), y, ls, clip_on=False,
                     ha='center', va='bottom', **kwargs)
        # get y coordinate of text bottom in figure pixel coordinates:
        ty = np.array(th.get_window_extent(ax.get_figure().canvas.get_renderer()))[0,1]
        dty = ly+0.5*lw + 2.0 - ty
    else:
        th = ax.text(0.5*(x0+x1), y, ls, clip_on=False,
                     ha='center', va='top', **kwargs)
        # get y coordinate of text bottom in figure pixel coordinates:
        ty = np.array(th.get_window_extent(ax.get_figure().canvas.get_renderer()))[1,1]
        dty = ly-0.5*lw - 2.0 - ty
    th.set_position((0.5*(x0+x1), y+dyu*dty))
    return x0, x1, y

        
def yscalebar(ax, x, y, height, hunit=None, hformat=None, ha='left', va='bottom',
              lw=None, color=None, capsize=None, clw=None, **kwargs):
    """Vertical scale bar with label.

    From bendalab/plottools package.

    Parameters
    ----------
    ax: matplotlib axes
        Axes where to draw the scale bar.
    x: float
        x-coordinate where to draw the scale bar in relative units of the axes.
    y: float
        y-coordinate where to draw the scale bar in relative units of the axes.
    height: float
        Length of the scale bar in units of the data's y-values.
    hunit: string
        Unit of the data's y-values.
    hformat: string or None
        Optional format string for formatting the label of the scale bar
        or simply a string used for labeling the scale bar.
    ha: 'left' or 'right'
        Label of the scale bar either to the left or to the right
        of the scale bar.
    va: 'top', 'bottom', or 'center'
        Scale bar aligned above, below, or centered on (x, y).
    lw: int, float, None
        Line width of the scale bar.
    color: matplotlib color
        Color of the scalebar.
    capsize: float or None
        If larger then zero draw cap lines at the ends of the bar.
        The length of the lines is given in points (same unit as linewidth).
    clw: int, float
        Line width of the cap lines.
    kwargs: key-word arguments
        Passed on to `ax.text()` used to print the scale bar label.
    """
    ax.autoscale(False)
    # ax dimensions:
    pixelx = np.abs(np.diff(ax.get_window_extent().get_points()[:,0]))[0]
    pixely = np.abs(np.diff(ax.get_window_extent().get_points()[:,1]))[0]
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    unitx = xmax - xmin
    unity = ymax - ymin
    dxu = np.abs(unitx)/pixelx
    dyu = np.abs(unity)/pixely
    # transform x, y from relative units to axis units:
    x = xmin + x*unitx
    y = ymin + y*unity
    # bar length:
    if hformat is None:
        hformat = '%.0f'
        if height < 1.0:
            hformat = '%.1f'
    try:
        ls = hformat % height
        width = float(ls)
    except TypeError:
        ls = hformat
    # bar:
    if va == 'bottom':
        y0 = y
        y1 = y+height
    elif va == 'top':
        y0 = y-height
        y1 = y
    else:
        y0 = y-0.5*height
        y1 = y+0.5*height
    # line width:
    if lw is None:
        lw = 2
    # color:
    if color is None:
        color = 'k'
    # scalebar:
    lh = ax.plot([x, x], [y0, y1], '-', color=color, lw=lw,
                 solid_capstyle='butt', clip_on=False)
    # get x position of line in figure pixel coordinates:
    lx = np.array(lh[0].get_window_extent(ax.get_figure().canvas.get_renderer()))[0,0]
    # caps:
    if capsize is None:
        capsize = 0
    if clw is None:
        clw = 0.5
    if capsize > 0.0:
        dx = capsize*dxu
        ax.plot([x-dx, x+dx], [y0, y0], '-', color=color, lw=clw, solid_capstyle='butt',
                clip_on=False)
        ax.plot([x-dx, x+dx], [y1, y1], '-', color=color, lw=clw, solid_capstyle='butt',
                clip_on=False)
    # label:
    if hunit:
        ls += u'\u2009%s' % hunit
    if ha == 'right':
        th = ax.text(x, 0.5*(y0+y1), ls, clip_on=False, rotation=90.0,
                     ha='left', va='center', **kwargs)
        # get x coordinate of text bottom in figure pixel coordinates:
        tx = np.array(th.get_window_extent(ax.get_figure().canvas.get_renderer()))[0,0]
        dtx = lx+0.5*lw + 2.0 - tx
    else:
        th = ax.text(x, 0.5*(y0+y1), ls, clip_on=False, rotation=90.0,
                     ha='right', va='center', **kwargs)
        # get x coordinate of text bottom in figure pixel coordinates:
        tx = np.array(th.get_window_extent(ax.get_figure().canvas.get_renderer()))[1,0]
        dtx = lx-0.5*lw - 1.0 - tx
    th.set_position((x+dxu*dtx, 0.5*(y0+y1)))
    return x, y0, y1


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

def plot_eod_properties():
	""" Plot 3 pulse-type EODs, one artefact and one wave snippet to visualize raw data.
		Plot is saved in img/eod_properties.png.
	"""

	# create figure and grid
	fig=plt.figure(figsize=(6,3))
	gs = gridspec.GridSpec(2,4)

	# define data files
	np_files = ['data/pulse_eod_1','data/pulse_eod_2','data/pulse_eod_3','data/artefact_1','data/wave_eod_1']

	# go through each data file and plot each snippet on a new axis.
	for i, np_file in enumerate(np_files):

		# plot the first snippet on a bigger axis
		if i==0:
			ax=fig.add_subplot(gs[:,:2])
		elif i<3:
			ax=fig.add_subplot(gs[i-1,2])
		else:
			ax=fig.add_subplot(gs[i-3,3])

		[x,y] = np.load(np_file+'.npy')
		ax.plot(x,y,c=cmap(i))
		
		# plot the lines and annotation to show EOD features on the first snippet.
		if i==0:
			ax.plot([x[np.argmin(y)],x[np.argmax(y)]],[np.min(y),np.max(y)],linestyle='--',marker='o',c='k')
			ax.plot([x[np.argmin(y)],x[np.argmax(y)]],[np.min(y),np.min(y)],linestyle='--',c='k')
			ax.plot([x[np.argmax(y)],x[np.argmax(y)]],[np.min(y),np.max(y)],linestyle='--',c='k')
			ax.annotate('w',[0.5*(x[np.argmin(y)]+x[np.argmax(y)]),np.min(y)],xytext=(0, -12), textcoords='offset points',fontsize=14,ha='center')
			ax.annotate('h',[x[np.argmax(y)],0.5*(np.min(y)+np.max(y))],xytext=(3, 0), textcoords='offset points',fontsize=14,ha='left')
			ax.annotate('s',[0.5*(x[np.argmin(y)]+x[np.argmax(y)]),0.5*(np.min(y)+np.max(y))],xytext=(-12, 0), textcoords='offset points',fontsize=14)
		
		h = np.max(y)-np.min(y)
		w = np.abs(x[np.argmax(y)]-x[np.argmin(y)])
		s = h/w


		# annotate the EOD height, width and slope for each snippet.
		if i==0:
			ax.text(0.1, -0.04,u"h = $%.2f$" "\n" u"w = $%.2f\u2009ms$" "\n" u"s = $%.2f\u2009ms^{-1}$"%(h,w,s),
		     horizontalalignment='left',
		     verticalalignment='top',
		     transform = ax.transAxes,
		     bbox={'linewidth':0,'facecolor':'None'})
		else:
			ax.text(0.1, -0.1,r"h = $%.2f$" "\n" u"w = $%.2f\u2009ms$" "\n" u"s = $%.2f\u2009ms^{-1}$"%(h,w,s),
		     horizontalalignment='left',
		     verticalalignment='top',
		     transform = ax.transAxes,
		     bbox={'linewidth':0,'facecolor':'None'})

		ax.axis('off')

	plt.tight_layout()
	plt.savefig('img/EOD_properties.png')
	plt.show()

def plot_peak_detection():
	""" Plot 2 pulse-type EOD snippet and all consecutive peak detection steps.
		Plot is saved in img/peak_detection.png.
	"""

	# load variables to plot
	np_files = ['data/peakdata_1','data/peakdata_2']

	# create figure and grid
	fig = plt.figure(figsize=(6,3))
	gs = gridspec.GridSpec(2,4)

	# go through each data file
	for i,np_file in enumerate(np_files):

		# load all peak data from the zipped numpy files
		with np.load(np_file+'.npz') as npd:
			data = npd['data']
			p1 = npd['p1']
			p2 = npd['p2']
			p3 = npd['p3']
			p4 = npd['p4']

		# plot peak detection step nr 1
		ax = fig.add_subplot(gs[i,0])
		ax.axis('off')
		if i==0:
			ax.set_title('1.')
		ax.plot(data[0],data[1],c=cmap(7),alpha=0.5)
		ax.plot(p1[0],p1[1],'o', c=cmap(0), alpha=0.75)
		ax.plot(p1[2],p1[3],'o', c=cmap(1), alpha=0.75)

		# plot peak detection step nr 2
		ax = fig.add_subplot(gs[i,1])
		ax.axis('off')
		if i==0:
			ax.set_title('2.')
		ax.plot(data[0],data[1],c=cmap(7),alpha=0.5)
		ax.plot(p2[0],p2[1],'o', c=cmap(0), alpha=0.75)
		ax.plot(p2[2],p2[3],'o', c=cmap(1), alpha=0.75)
		ax.plot(np.vstack((p2[0],p2[2])),np.vstack((p2[1],p2[3])),linestyle='--',c='k')

		# plot peak detection step nr 3
		ax = fig.add_subplot(gs[i,2])
		ax.axis('off')
		if i==0:
			ax.set_title('3.')
		ax.plot(data[0],data[1],c=cmap(7),alpha=0.5)
		ax.plot(p3[0],p3[1],'o', c=cmap(0), alpha=0.75)
		ax.plot(p3[2],p3[3],'o', c=cmap(1), alpha=0.75)
		ax.plot(np.vstack((p3[0],p3[2])),np.vstack((p3[1],p3[3])),linestyle='--',c='k')

		# plot peak detection step nr 4
		ax = fig.add_subplot(gs[i,3])
		ax.axis('off')
		if i==0:
			ax.set_title('4.')
		ax.plot(data[0],data[1],c=cmap(7),alpha=0.5)
		ax.plot(p4[0],p4[1],'o', c=cmap(0), alpha=0.75)
		ax.plot(p4[2],p4[3],'o', c=cmap(1), alpha=0.75)
		ax.plot(np.vstack((p4[0],p4[2])),np.vstack((p4[1],p4[3])),linestyle='--',c='k')

	plt.tight_layout()
	plt.savefig('img/peak_detection.png')
	plt.show()


def plot_clustering():
	""" Plot all clustering steps for one wild recording.
		Plot is saved in img/clustering.png.
	"""
	with np.load('data/clustering.npz',allow_pickle=True) as pd:
		# extract all variables from the dictionary:
		samplerate = pd['samplerate']
		eod_widths = pd['EOD_widths']
		eod_hights = pd['EOD_heights']
		eod_shapes = pd['EOD_shapes']    #shapes, features, labels
		disc_masks = pd['discarding_masks']
		merge_masks = pd['merge_masks']

	# create figure + transparant figure.
	fig = plt.figure(figsize=(8,5))
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
	h_size = len(eod_hights)
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
		hw, _, _ = ax1.hist(eod_widths[1][eod_widths[2]==wl], bins=np.linspace(np.min(eod_widths[1]),np.max(eod_widths[1]),100),color=lighter(c_o,colw),orientation='horizontal')
		
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
		ax2.yaxis.set_major_formatter(ticker.NullFormatter())
		ax2.yaxis.set_minor_formatter(ticker.NullFormatter())

		# define colors for plots
		colidxsh = -np.linspace(-1.25,-0.5,len(uhl))

		for n, (hl, hcol, snippets, features, labels, dmasks, mmasks) in enumerate(zip(uhl, colidxsh, w_snip, w_feat, w_lab, w_dm, w_mm)):

			hh,_,_=loghist(ax2,eod_h[eod_h_labs==hl],np.min(eod_h),np.max(eod_h),100,lighter(c_g,hcol),orientation='horizontal')

			# set arrow spines only on last plot
			if n==len(uhl)-1:
				arrowed_spines(ax2,ms=10)

			# plot line from the width histogram to the height histogram.
			if n==0:
				coord1 = transFigure.transform(ax1.transData.transform([np.median(hw[hw!=0]),np.median(eod_widths[1][eod_widths[2]==wl])]))
				coord2 = transFigure.transform(ax2.transData.transform([0.9,np.mean(eod_h)]))
				line = Line2D((coord1[0],coord2[0]),(coord1[1],coord2[1]),
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
						ax3.plot(features[pt][labels[pt]==c,0],features[pt][labels[pt]==c,1],'.',color=lighter(cmap_pt,colidxss[j]),label=c,rasterized=True)
						ax4.plot(snippets[pt][labels[pt]==c].T,linewidth=0.1,color=lighter(cmap_pt,colidxss[j]),label=c,rasterized=True)
						
						# check if the current cluster is an EOD, if yes, plot it.
						if np.sum(dmasks[pt][labels[pt]==c]) == 0:

							ax = fig.add_subplot(EOD_delete_ax[disc_size-disc_block+disc_count])
							ax.axis('off')

							# plot mean EOD snippet
							ax.plot(np.mean(snippets[pt][labels[pt]==c],axis=0),color=lighter(cmap_pt,colidxss[j]))
							disc_count = disc_count + 1

							# match colors and draw line..							
							coord1 = transFigure.transform(ax4.transData.transform([ax4.get_xlim()[1], ax4.get_ylim()[0] + 0.5*(ax4.get_ylim()[1]-ax4.get_ylim()[0])]))
							coord2 = transFigure.transform(ax.transData.transform([ax.get_xlim()[0],ax.get_ylim()[0] + 0.5*(ax.get_ylim()[1]-ax.get_ylim()[0])]))
							line = Line2D((coord1[0],coord2[0]),(coord1[1],coord2[1]),
                                          transform=fig.transFigure,color='grey',linewidth=0.5)
							fig.lines.append(line)	
							axs.append(ax)

							# check if the current EOD survives the merge step
							# if so, plot it.
							if np.sum(mmasks[pt,labels[pt]==c])>0:

								ax = fig.add_subplot(EOD_merge_ax[merge_size-merge_block+merge_count])
								ax.axis('off')
								
								ax.plot(np.mean(snippets[pt][labels[pt]==c],axis=0),color=lighter(cmap_pt,colidxss[j]))
								merge_count = merge_count + 1

						j=j+1

				if pt==0:
					# draw line from hight cluster to EOD shape clusters.
					coord1 = transFigure.transform(ax2.transData.transform([np.median(hh[hh!=0]),np.median(eod_h[eod_h_labs==hl])]))
					coord2 = transFigure.transform(ax3.transData.transform([ax3.get_xlim()[0],ax3.get_ylim()[0]]))
					line = Line2D((coord1[0],coord2[0]),(coord1[1],coord2[1]),
                                  transform=fig.transFigure,color='grey',linewidth=0.5)
					fig.lines.append(line)

			shape_count = shape_count + 1
			
			if len(axs)>0:
				# plot lines that indicate the merged clusters.
				coord1 = transFigure.transform(axs[0].transData.transform([axs[0].get_xlim()[1]+0.1*(axs[0].get_xlim()[1]-axs[0].get_xlim()[0]), axs[0].get_ylim()[1]-0.25*(axs[0].get_ylim()[1]-axs[0].get_ylim()[0])]))
				coord2 = transFigure.transform(axs[-1].transData.transform([axs[-1].get_xlim()[1]+0.1*(axs[-1].get_xlim()[1]-axs[-1].get_xlim()[0]), axs[-1].get_ylim()[0]+0.25*(axs[-1].get_ylim()[1]-axs[-1].get_ylim()[0])]))
				line = Line2D((coord1[0],coord2[0]),(coord1[1],coord2[1]),
                              transform=fig.transFigure,color='grey',linewidth=1)
				fig.lines.append(line)
	plt.savefig('img/clustering.png')
	plt.show()

def plot_bgm(mode,ccol):
	""" Plot a BGM clustering step either on EOD width or height.
		Plot is saved in img/*mode*_clusters.png.

		Parameters
		----------
		mode : string
			Which cluster step to plot. Options are 'width' and 'height'. 
		ccol : matplotlib color
			Color to use for histogram.
	"""
	if mode == 'width':
		with np.load('data/BGM_width.npz',allow_pickle=True) as pd:
			# extract all variables from the dictionary:
			x = pd['x']
			means = pd['BGM'][1]
			variances = pd['BGM'][2]
			weights = pd['BGM'][0]    #shapes, features, labels
			use_log = pd['use_log']
			labels = pd['labels']
			xlab = pd['xlab']
	elif mode == 'height':
		with np.load('data/BGM_height.npz',allow_pickle=True) as pd:
			# extract all variables from the dictionary:
			x = pd['x']
			means = pd['BGM'][1]
			variances = pd['BGM'][2]
			weights = pd['BGM'][0]    #shapes, features, labels
			use_log = pd['use_log']
			labels = pd['labels']
			xlab = pd['xlab']
	else:
		print('define a mode (width or height)')
		return 0


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
	fig, ax1 = plt.subplots(figsize=(5,3.5))
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
		ax2.plot(xplot,gaussians[c],c=lighter(c_grey,colidxs[i]),linewidth=2,label=r'$N(\mu_%i,\sigma_%i)$'%(c,c))
	
	# plot intersection lines
	ax2.vlines(xplot[1:][np.diff(classes)!=0],0,gmax/gmin,color='k',linewidth=2,linestyle='--')
	ax2.set_ylim([gmin,np.max(np.vstack(gaussians))*1.1])

	# plot data distributions and classes
	colidxs = -np.linspace(-1.25,-0.5,len(np.unique(labels)))
	for i,l in enumerate(np.unique(labels)):
		if use_log:
			h,binn,_=loghist(ax1,x[labels==l],np.min(x),np.max(x),100,lighter(ccol,colidxs[i]),label=r'$x_%i$'%l)
		else:
			h,binn,_=ax1.hist(x[labels==l],bins=np.linspace(np.min(x),np.max(x),100),color=lighter(ccol,colidxs[i]),label=r'$x_%i$'%l)

	# add legends and plot.
	ax2.legend(loc='lower left',frameon=False,bbox_to_anchor=(-0.05,1.2),ncol=len(np.unique(classes)))
	ax1.legend(loc='upper left',frameon=False,bbox_to_anchor=(-0.05,1.2),ncol=len(np.unique(labels)))
	plt.subplots_adjust(top=0.7)
	plt.tight_layout()
	plt.savefig('img/%s_clusters.png'%mode)
	plt.show()

def plot_feature_extraction():
	""" Plot clustering step on EOD shape.
		Plot is saved in img/shape_clusters.
	"""

	with np.load('data/feature_extraction.npz',allow_pickle=True) as pd:
		# extract all variables from the dictionary:
		raw_snippets = pd['raw_snippets']
		normalized_snippets = pd['snippets']
		features = pd['features']
		labels = pd['clusters']
		dt = 1/pd['samplerate']

	ccol = cmap_pts[1]

	# set up the figure layout
	fig = plt.figure(figsize=(((2+0.2)*3),3))
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
			color = lighter(ccol,colidxs[j])
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
		xscalebar(ax,0,0,wbar,wformat='%%.%if'%size)
		yscalebar(ax,0,0,wbar,hformat='%%.%if'%size)
		ax.axis('off')
	plt.savefig('img/shape_clusters.png')
	plt.show()

def plot_moving_fish():
	""" Plot moving fish detection step.
		Plot is saved in img/moving_fish.png.
	"""
	fig = plt.figure(figsize=(7,3))
	with np.load('data/moving_fish.npz',allow_pickle=True) as pd:
		# extract all variables from the dictionary:
		ws = pd['w']
		dts = pd['dt']
		clusterss = pd['clusters']
		ts = pd['t']
		fishcounts = pd['fishcount']
		T = pd['T']
		ignore_stepss = pd['ignore_steps']

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
			xscalebar(ax2,1,0,1,wunit='s',ha='right')

		con = ConnectionPatch([0,-0.5], [dt/2,y[0]], "data", "data",
		    axesA=ax1, axesB=ax2,color='k')
		ax2.add_artist(con)
		con = ConnectionPatch([dt,-0.5], [dt/2,y[0]], "data", "data",
		    axesA=ax1, axesB=ax2,color='k')
		ax2.add_artist(con)

		plt.xlim([0,T])
	plt.savefig('img/moving_fish.png')
	plt.show()

def plot_eod_deletion():
	""" Plot one pulse-type EOD,  one artefact, one wave-type EOD, one sidepeak, 
		and the feature extraction steps for filtering out pulse-type EODs.
		Plot is saved in img/eod_assessment.
	"""

	fig = plt.figure()
	gs = gridspec.GridSpec(5,4,figure=fig,height_ratios=[0.25,1,1,1,1]) 

	ax = fig.add_subplot(gs[0,0])
	ax.text(0,0,r'$\overline{EOD}$',ha='center',va='center')
	ax.set_xlim([-100,100])
	ax.set_ylim([-100,100])
	ax.axis('off')
	ax = fig.add_subplot(gs[0,1])
	ax.text(0,0,r'$\mathcal{F}\{\overline{EOD}\}$',ha='center',va='center')
	ax.set_xlim([-100,100])
	ax.set_ylim([-100,100])
	ax.axis('off')
	ax = fig.add_subplot(gs[0,2])
	ax.text(0,0,r'$\dfrac{EOD_{width}}{ISI}$',ha='center',va='center')
	ax.set_xlim([-100,100])
	ax.set_ylim([-100,100])
	ax.axis('off')
	ax = fig.add_subplot(gs[0,3])
	ax.text(0,0,r'$\overline{EOD}$ & peaks',ha='center',va='center')
	ax.set_xlim([-100,100])
	ax.set_ylim([-100,100])
	ax.axis('off')

	fnames = ['good_eod_ad','artefact_ad','wave_eod_ad','sidepeak_ad']

	for i, fname in enumerate(fnames):

		with np.load('data/%s.npz'%fname,allow_pickle=True) as pd:
			# extract all variables from the dictionary:
			samplerates = pd['samplerates']
			values = pd['values']
		
		mean_eod = values[0]
		fft = values[1]
		isis = values[2]
		ex_mean_eod = values[3][0]
		pk = values[3][1][0]
		tr = values[3][1][1]
		hpk = values[3][2]

		samplerate = samplerates[0]
		samplerate_og = samplerates[1]

		# plot original snippet
		ax = fig.add_subplot(gs[i+1,0])
		ax.plot(1000*np.arange(0,len(mean_eod)/samplerate,1/samplerate),mean_eod,c=cmap(7))
		xscalebar(ax,1,0,max(1,int(250*len(mean_eod)/samplerate)),ha='right',wunit='ms')
		ax.axis('off')

		# plot fft
		ax = fig.add_subplot(gs[i+1,1])
		x = np.linspace(0,samplerate_og/2,len(fft))
		ax.plot(x,np.abs(fft),c=cmap(0))
		ax.fill_between(x,np.abs(fft),color=cmap(0),alpha=0.25)
		ax.fill_between(x[:int(len(fft)/2)],np.abs(fft[:int(len(fft)/2)]),color=cmap(0),alpha=0.5)
		ax.get_yaxis().set_visible(False)
		ax.spines['top'].set_visible(False)
		ax.spines['right'].set_visible(False)
		
		ax.set_xticks([0,samplerate_og/4,samplerate_og/2])
		if i==3:
			ax.set_xticklabels([0,r'$\dfrac{f_{Nyq}}{2}$',r'$f_{Nyq}$'])
		else:
			ax.set_xticklabels(["","",""])

		# plot ISI distribution
		ax = fig.add_subplot(gs[i+1,2])
		vals,_,_=ax.hist(isis,bins=np.linspace(0,1,100),color=cmap(1))
		ax.plot(np.median(isis),np.max(vals)*1.2,marker='v',color=cmap(1))
		
		ax.set_xlim([-0.02,1.02])
		ax.set_xticks([0,1])
		if i<3:
			ax.set_xticklabels(["",""])
		ax.axes.get_yaxis().set_visible(False)
		ax.spines['top'].set_visible(False)
		ax.spines['right'].set_visible(False)
		ax.spines['left'].set_visible(False)

		# plot extended snippet
		ax = fig.add_subplot(gs[i+1,3])
		ax.plot(1000*np.arange(0,len(ex_mean_eod)/samplerate,1/samplerate),ex_mean_eod,c=cmap(7))
		xscalebar(ax,1,0,max(1,int(250*len(ex_mean_eod)/samplerate)),ha='right',wunit='ms')
		ax.axis('off')

		# plot midline
		ax.vlines(1000*len(ex_mean_eod)/samplerate/2,np.min(ex_mean_eod),np.max(ex_mean_eod),linestyle='--',color='k')
		
	    # plot detected peaks and troughs in snippet
		ax.plot(1000*pk/samplerate, ex_mean_eod[pk],'o',c='k')
		ax.plot(1000*tr/samplerate, ex_mean_eod[tr],'o',c='k')

		# mark the bigges pk-trough combi.
		ax.plot((1000*np.array([hpk])).T/samplerate,ex_mean_eod[hpk],linestyle='--',linewidth=2, marker='o',c=cmap(2),clip_on=False)

		ax.axis('off')
	plt.tight_layout()
	plt.savefig('img/EOD_assessment.png')
	plt.show()

plot_eod_properties()
plot_peak_detection()
plot_clustering()
plot_bgm('width',c_o)
plot_bgm('height',c_g)
plot_feature_extraction()
plot_moving_fish()
plot_eod_deletion()
