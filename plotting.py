#plotting.py:
##functions for plotting data analyses

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import ArrowStyle
import matplotlib.gridspec as gridspec
import parse_logs as pl
import parse_ephys as pe
import parse_timestamps as pt
import parse_trials as ptr
import h5py
import scipy.stats
import scipy as sp

"""
A function to plot the significance of the regression. Same as the above function but
for the sub-functions that work on bin sizes
coefficients for a all units over epochs defined elsewhere, for all sessions* (see file_lists.py)
Exaclt the same as plot_session_regression but it includes data from multple sessions
Inputs:
	-coeffs: array of coefficients
	-sig_vals:
	-mse
	-epoch_idx
	-trial_len: length of the whole thing in seconds
"""
def plot_regressions(coeffs,sig_vals,MSE,epoch_idx,trial_len):
	##assume that the epochs are the same from the ml_regress functions
	epoch_labels = ['Pre-action','Action','Delay','Outcome']
	epochs = ['choice','action','delay','outcome']
	colors = ['g','r','k','c']
	##assume the following regressors; in this order
	regressors = ['Choice','Reward', 'CxR',
					'Q upper','Q lower',
					'Q chosen']
	##some metadata
	num_units = coeffs.shape[0]
	num_windows = coeffs.shape[2]
	##the x-axis 
	x_coords = np.linspace(0,trial_len,num_windows)
	##setup the figure to plot the significant values
	##sig_vals = sig_vals.sum(axis=0)/float(num_units)
	sig_vals = sig_vals/float(num_units)
	fig = plt.figure()
	gs = gridspec.GridSpec(len(regressors),num_windows)
	min_y = sig_vals.min()
	max_y = sig_vals.max()
	for r in range(len(regressors)):
		for e in range(len(epochs)):
			epoch = epochs[e]
			idx = epoch_idx[epoch]
			x = x_coords[idx]
			ax = plt.subplot(gs[r,idx[0]:idx[-1]+1])
			#ax.axhspan(0,0.05,facecolor='b',alpha=0.5) ##significance threshold
			if x.size == 1:
				x_center = x[0]/2.0
			else:
				x_center = (x[-1]-x[0])/2
			color = colors[e]
			ydata = sig_vals[r,idx]
			ax.plot(x,ydata,color=color,linewidth=2,marker='o',label=epoch)
			# for xpt,ypt in zip(x,ydata):
			# 	if ypt <=0.05:
			# 		ax.text(xpt,ypt+0.1,"*",fontsize=16) ##TODO: figure out how to get significance here
			ax.set_ylim(min_y,max_y)
			##conditional axes labels
			if r+1 == len(regressors):
				ax.set_xticks(np.round(x,1))
			else:
				ax.set_xticklabels([])
			if e+1 == len(epochs):
				ax.yaxis.tick_right()
				ax.set_yticks([0,max_y])
			else:
				ax.set_yticklabels([])
			if r == 0:
				ax.set_title(epoch_labels[e],fontsize=14,weight='bold')
			if e == 0:
				ax.set_ylabel(regressors[r],fontsize=12,weight='bold')
			if r+1 == len(regressors) and e == 2:
				ax.set_xlabel("Time in trial, s",fontsize=14,weight='bold')
	fig.suptitle("Proportion of "+str(num_units)+" units",fontsize=14)
	#setup the figure to plot the coefficients
	coeffs = coeffs.mean(axis=0)
	fig = plt.figure()
	gs = gridspec.GridSpec(len(regressors),num_windows)
	min_y = coeffs.min()
	max_y = coeffs.max()
	for r in range(len(regressors)):
		for e in range(len(epochs)):
			epoch = epochs[e]
			idx = epoch_idx[epoch]
			x = x_coords[idx]
			ax = plt.subplot(gs[r,idx[0]:idx[-1]+1])
			#ax.axhspan(0,0.05,facecolor='b',alpha=0.5) ##significance threshold
			if x.size == 1:
				x_center = x[0]/2.0
			else:
				x_center = (x[-1]-x[0])/2
			color = colors[e]
			ydata = coeffs[r,idx]
			ax.plot(x,ydata,color=color,linewidth=2,marker='o',label=epoch)
			# for xpt,ypt in zip(x,ydata):
			# 	if ypt <=0.05:
			# 		ax.text(xpt,ypt+0.1,"*",fontsize=16) ##TODO: figure out how to get significance here
			ax.set_ylim(min_y,max_y)
			##conditional axes labels
			if r+1 == len(regressors):
				ax.set_xticks(np.round(x,1))
			else:
				ax.set_xticklabels([])
			if e+1 == len(epochs):
				ax.yaxis.tick_right()
				#ax.set_yticks([-.5,.5])
			else:
				ax.set_yticklabels([])
			if r == 0:
				ax.set_title(epoch_labels[e],fontsize=14,weight='bold')
			if e == 0:
				ax.set_ylabel(regressors[r],fontsize=12,weight='bold')
			if r+1 == len(regressors) and e == 2:
				ax.set_xlabel("Time in trial, s",fontsize=14,weight='bold')
	fig.suptitle("Mean regression coeffs.",fontsize=14)
	##now plot the predictability values
	fig = plt.figure()
	gs = gridspec.GridSpec(1,num_windows)
	min_y = MSE.min()
	max_y = MSE.max()
	for e in range(len(epochs)):
		epoch = epochs[e]
		idx = epoch_idx[epoch]
		x = x_coords[idx]
		ax = plt.subplot(gs[0,idx[0]:idx[-1]+1])
		if x.size == 1:
			x_center = x[0]/2.0
		else:
			x_center = (x[-1]-x[0])/2
		color = colors[e]
		ydata = MSE[idx]
		ax.plot(x,ydata,color=color,linewidth=2,marker='o',label=epoch)
		ax.set_ylim(min_y,max_y)
		ax.set_xticks(np.round(x,1))
		if e+1 == len(epochs):
			ax.yaxis.tick_right()
		else:
			ax.set_yticklabels([])
		ax.set_title(epoch_labels[e],fontsize=14,weight='bold')
		if e == 0:
			ax.set_ylabel("Mean sq. error",fontsize=12,weight='bold')
		if e == 2:
			ax.set_xlabel("Time in trial, s",fontsize=14,weight='bold')
	fig.suptitle("Prediction error",fontsize=14)

"""
Caclulates and plots the interval between action and outcome
 for a single session.
Inputs:
	-dictionary of results produced by parse_timestamps
Outputs:
	plot
"""
def ao_duration_analysis(f_in):
	##get the relevant data
	results_dict = pt.sort_by_trial(f_in)
	##start with the upper lever rewarded sessions
	try:
		upper_trials = results_dict['upper_rewarded']
		upper_trial_durs = ptr.get_ao_interval(upper_trials)
	except KeyError:
		upper_trial_durs = None
	##move on to the lower lever if applicable
	try:
		lower_trials = results_dict['lower_rewarded']
		lower_trial_durs = ptr.get_ao_interval(lower_trials)
	except KeyError:
		lower_trial_durs = None
	##get some basic stats
	fig,(ax,ax2) = plt.subplots(2,1,sharex=True)
	fig.patch.set_facecolor('white')
	fig.set_size_inches(10,4)
	if upper_trial_durs is not None:
		upper_dur_mean = abs(upper_trial_durs).mean()
		upper_dur_std = abs(upper_trial_durs).std()
		##outliers more than 2 std dev
		upper_outliers = abs(upper_trial_durs[np.where(abs(upper_trial_durs)>3*upper_dur_std)])
		##get just the successful trials
		r_idx = np.where(upper_trial_durs>0)
		r_upper_durs = upper_trial_durs[r_idx]
		r_upper_times = upper_trials[r_idx,0]
		##get just the unsuccessful trials
		u_idx = np.where(upper_trial_durs<0)
		u_upper_durs = upper_trial_durs[u_idx]
		u_upper_times = upper_trials[u_idx,0]
		##plot this stuff
		ax.scatter(r_upper_times,abs(r_upper_durs),edgecolor='green',marker='o',s=30,
			linewidth=2,facecolors=('green',),alpha=0.7,label='rewarded upper lever')
		ax.scatter(u_upper_times,abs(u_upper_durs),color='green',marker='x',s=30,
			linewidth=2,label='unrewarded upper lever')
		ax2.scatter(r_upper_times,abs(r_upper_durs),edgecolor='green',marker='o',s=30,
			linewidth=2,facecolors=('green',),alpha=0.7,label='rewarded upper lever')
		ax2.scatter(u_upper_times,abs(u_upper_durs),color='green',marker='x',s=30,
			linewidth=2,label='unrewarded upper lever')		
	if lower_trial_durs is not None:
		lower_dur_mean = abs(lower_trial_durs).mean()
		lower_dur_std = abs(lower_trial_durs).std()
		##outliers
		lower_outliers = abs(lower_trial_durs[np.where(abs(lower_trial_durs)>3*lower_dur_std)])
		##get just the successful trials
		r_idx = np.where(lower_trial_durs>0)
		r_lower_durs = lower_trial_durs[r_idx]
		r_lower_times = lower_trials[r_idx,0]
		##get just the unsuccessful trials
		u_idx = np.where(lower_trial_durs<0)
		u_lower_durs = lower_trial_durs[u_idx]
		u_lower_times = lower_trials[u_idx,0]
		##plot this stuff
		ax.scatter(r_lower_times,abs(r_lower_durs),edgecolor='red',marker='o',s=30,
			linewidth=2,facecolors=('red',),alpha=0.7,label='rewarded lower lever')
		ax.scatter(u_lower_times,abs(u_lower_durs),color='red',marker='x',s=30,
			linewidth=2,label='unrewarded lower lever')
		ax2.scatter(r_lower_times,abs(r_lower_durs),edgecolor='red',marker='o',s=30,
			linewidth=2,facecolors=('red',),alpha=0.7,label='rewarded lower lever')
		ax2.scatter(u_lower_times,abs(u_lower_durs),color='red',marker='x',s=30,
			linewidth=2,label='unrewarded lower lever')
	for label in ax2.xaxis.get_ticklabels()[1::2]:
		label.set_visible(False)
	for label in ax.xaxis.get_ticklabels()[1::2]:
		label.set_visible(False)
	for label in ax2.xaxis.get_ticklabels()[::2]:
		label.set_fontsize(14)
	for label in ax.xaxis.get_ticklabels()[::2]:
		label.set_fontsize(14)
	for label in ax2.yaxis.get_ticklabels()[1::2]:
		label.set_visible(False)
	for label in ax.yaxis.get_ticklabels()[1::2]:
		label.set_visible(False)
	for label in ax2.yaxis.get_ticklabels()[::2]:
		label.set_fontsize(14)
	for label in ax.yaxis.get_ticklabels()[::2]:
		label.set_fontsize(14)
	##if there are outliers, break the axis
	try:
		outliers = np.hstack((upper_outliers,lower_outliers))
	except NameError: ##if we only have one kind of trial
		if upper_trial_durs is not None:
			outliers = upper_outliers
		else:
			outliers = lower_outliers
	if outliers.size > 0:
		ax2.set_ylim(-1,max(2*lower_dur_std,2*upper_dur_std))
		ax.set_ylim(outliers.min()-5,outliers.max()+10)
		# hide the spines between ax and ax2
		ax.spines['bottom'].set_visible(False)
		ax2.spines['top'].set_visible(False)
		ax.xaxis.tick_top()
		ax.tick_params(labeltop='off')  # don't put tick labels at the top
		ax2.xaxis.tick_bottom()

		# This looks pretty good, and was fairly painless, but you can get that
		# cut-out diagonal lines look with just a bit more work. The important
		# thing to know here is that in axes coordinates, which are always
		# between 0-1, spine endpoints are at these locations (0,0), (0,1),
		# (1,0), and (1,1).  Thus, we just need to put the diagonals in the
		# appropriate corners of each of our axes, and so long as we use the
		# right transform and disable clipping.

		d = .015  # how big to make the diagonal lines in axes coordinates
		# arguments to pass plot, just so we don't keep repeating them
		kwargs = dict(transform=ax.transAxes, color='k', clip_on=False)
		ax.plot((-d, +d), (-d, +d), **kwargs)        # top-left diagonal
		ax.plot((1 - d, 1 + d), (-d, +d), **kwargs)  # top-right diagonal

		kwargs.update(transform=ax2.transAxes)  # switch to the bottom axes
		ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs)  # bottom-left diagonal
		ax2.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)  # bottom-right diagonal
	else:
		fig.delaxes(ax)
		fig.draw()
	if outliers.size > 0:
		legend=ax.legend(frameon=False)
		try:
			plt.text(0.2, 0.9,'upper lever mean = '+str(upper_dur_mean),ha='center',
				va='center',transform=ax.transAxes,fontsize=14)
		except NameError:
			pass
		try:
			plt.text(0.2, 0.8,'lower lever mean = '+str(lower_dur_mean),ha='center',
				va='center',transform=ax.transAxes,fontsize=14)
		except NameError:
			pass
	else:
		legend=ax2.legend(frameon=False)
	for label in legend.get_texts():
		label.set_fontsize('large')
	legend.get_frame().set_facecolor('none')
	ax2.set_xlabel("Time in session, s",fontsize=14)
	ax2.set_ylabel("Trial duration, s",fontsize=14)
	fig.suptitle("Duration of trials",fontsize=14)


"""
takes in a data dictionary produced by parse_log
plots the lever presses and the switch points for levers
"""
def plot_presses(f_in, sigma = 20):
	##extract relevant data
	data_dict = h5py.File(f_in,'r')
	top = data_dict['top_lever']
	bottom = data_dict['bottom_lever']
	duration = int(np.ceil(data_dict['session_length']))
	top_rewarded = np.asarray(data_dict['top_rewarded'])/60.0
	bottom_rewarded = np.asarray(data_dict['bottom_rewarded'])/60.0
	##convert timestamps to histogram structures
	top, edges = np.histogram(top, bins = duration)
	bottom, edges = np.histogram(bottom, bins = duration)
	##smooth with a gaussian window
	top = pe.gauss_convolve(top, sigma)
	bottom = pe.gauss_convolve(bottom, sigma)
	##get plotting stuff
	data_dict.close()
	x = np.linspace(0,np.ceil(duration/60.0), top.size)
	mx = max(top.max(), bottom.max())
	mn = min(top.min(), bottom.min())
	fig = plt.figure()
	gs = gridspec.GridSpec(2,2)
	ax = fig.add_subplot(gs[0,:])
	ax2 = fig.add_subplot(gs[1,0])
	ax3 = fig.add_subplot(gs[1,1], sharey=ax2)
	##the switch points
	ax.vlines(top_rewarded, mn, mx, colors = 'r', linestyles = 'dashed', 
		linewidth = '2', alpha = 0.5, label = "top rewarded")
	ax.vlines(bottom_rewarded, mn, mx, colors = 'b', linestyles = 'dashed', 
		linewidth = '2', alpha = 0.5, label = "bottom rewarded")
	ax2.vlines(top_rewarded, mn, mx, colors = 'r', linestyles = 'dashed', 
		linewidth = '2', alpha = 0.5, label = "top rewarded")
	ax2.vlines(bottom_rewarded, mn, mx, colors = 'b', linestyles = 'dashed', 
		linewidth = '2', alpha = 0.5, label = "bottom rewarded")
	ax3.vlines(top_rewarded, mn, mx, colors = 'r', linestyles = 'dashed', 
		linewidth = '2', alpha = 0.5, label = "top rewarded")
	ax3.vlines(bottom_rewarded, mn, mx, colors = 'b', linestyles = 'dashed', 
		linewidth = '2', alpha = 0.5, label = "bottom rewarded")
	ax.plot(x, top, color = 'r', linewidth = 2, label = "top lever")
	ax.plot(x, bottom, color = 'b', linewidth = 2, label = "bottom_lever")
	ax.legend()
	ax.set_ylabel("press rate", fontsize = 14)
	fig.suptitle("Lever press performance", fontsize = 18)
	ax.set_xlim(-1, x[-1]+1)
	for tick in ax.xaxis.get_major_ticks():
		tick.label.set_fontsize(14)
	for tick in ax.yaxis.get_major_ticks():
		tick.label.set_fontsize(14)
	#plot them separately
	##figure out the order of lever setting to create color spans
	# if top_rewarded.min() < bottom_rewarded.min():
	# 	for i in range(top_rewarded.size):
	# 		try:
	# 			ax2.axvspan(top_rewarded[i], bottom_rewarded[i], facecolor = 'r', alpha = 0.2)
	# 		except IndexError:
	# 			ax2.axvspan(top_rewarded[i], duration, facecolor = 'r', alpha = 0.2)
	# else:
	# 	for i in range(bottom_rewarded.size):
	# 		try:
	# 			ax3.axvspan(bottom_rewarded[i], top_rewarded[i], facecolor = 'b', alpha = 0.2)
	# 		except IndexError:
	# 			ax3.axvspan(bottom_rewarded[i], duration, facecolor = 'b', alpha = 0.2)
	ax2.plot(x, top, color = 'r', linewidth = 2, label = "top lever")
	ax3.plot(x, bottom, color = 'b', linewidth = 2, label = "bottom_lever")
	ax2.set_ylabel("press rate", fontsize = 14)
	ax2.set_xlabel("Time in session, mins", fontsize = 14)
	ax3.set_xlabel("Time in session, mins", fontsize = 14)
	fig.suptitle("Lever press performance", fontsize = 18)
	ax2.set_xlim(-1, x[-1]+1)
	ax3.set_xlim(-1, x[-1]+1)
	for tick in ax2.xaxis.get_major_ticks():
		tick.label.set_fontsize(14)
	for tick in ax2.yaxis.get_major_ticks():
		tick.label.set_fontsize(14)
	for tick in ax3.xaxis.get_major_ticks():
		tick.label.set_fontsize(14)
	for tick in ax3.yaxis.get_major_ticks():
		tick.label.set_fontsize(14)
	ax2.set_title("top only", fontsize = 14)
	ax3.set_title("bottom only", fontsize = 14)

"""
Plots a sample of the raw data matrix, X.
Inputs:
	-X; raw data matrix calculated by pca.get_datamatrix()
	-trial_len: length of one trial in bins
"""
def plot_X_sample(X,trial_len):
	fig, ax = plt.subplots(1)
	cax=ax.imshow(X[:,:trial_len*10],origin='lower',interpolation='none',aspect='auto')
	x = np.arange(0,trial_len*10,trial_len)
	ax.vlines(x,0,X.shape[0], linestyle='dashed',color='white')
	ax.set_ylabel("Neuron #",fontsize=18)
	ax.set_xlabel("Bin #", fontsize=18)
	cb = fig.colorbar(cax,label="spikes")
	fig.set_size_inches(10,6)
	fig.suptitle("Showing first 10 trials",fontsize=20)

	"""
A function to plot the covariance matrix
Inputs:
	-C: covariance matrix
"""
def plot_cov(C):
	fig, ax = plt.subplots(1)
	cax = ax.imshow(C,interpolation='none')
	cb = plt.colorbar(cax,label='correlation',aspect='auto')
	ax.set_xlabel("Neuron #",fontsize=14)
	ax.set_ylabel("Neuron #",fontsize=14)
	fig.set_size_inches(8,8)
	fig.suptitle("Covariance of data",fontsize=16)

"""
A function to plot eigenvals and a null distribution
Inputs:
	-C: covariance matrix
	-X: raw data matrix
"""
def plot_eigen(C,X,w,v):
	##compute the Marcenko-pastur distribution
	q = 1.0*X.shape[1]/X.shape[0]
	lmax = (1+np.sqrt(1/q))**2
	lmin = (1-np.sqrt(1/q))**2
	x = np.linspace(lmin,lmax,1000)
	f = q/2*np.pi*(np.sqrt(lmax-x)*(x-lmin))/x
	##now plot
	fig, ax = plt.subplots(1)
	ax.plot(x,f,linewidth=2,color='r',label='random distribution')
	##now plot a histogram of the eigenvalues
	ax.hist(w,8,orientation='vertical',facecolor='k',alpha=0.5)
	ax.set_ylabel("Counts")
	ax.set_xlabel("eigenvalues")
	ax.legend()
	##compute the Tracy-Widiom distrubution value corresponding to 
	##the highest eigenvalue (finite-size correction):
	tw_max = lmax + X.shape[1]**-2/3
	##plot the eigenvectors
	##and the associated eigenvalues
	idx = np.argsort(w)[::-1]
	evals = w[idx]
	evecs = v[:,idx]
	fig, (ax1,ax2) = plt.subplots(2,sharex=True)
	ax1.plot(evals,'-o')
	ax1.plot(np.arange(evals.shape[0]),np.ones(evals.shape[0])*lmax,
	        '--',color='k',label='lmax')
	ax1.plot(np.arange(evals.shape[0]),np.ones(evals.shape[0])*tw_max,
	       '--',color='r',label='tw_max')
	ax1.set_xlabel("eigenvector #",fontsize=14)
	ax1.set_ylabel('eigenvalue',fontsize=14)
	ax1.legend()
	cax = ax2.imshow(evecs.T,interpolation='none',aspect='auto')
	cbaxes = fig.add_axes([0.93, 0.12, 0.03, 0.3]) 
	cb = plt.colorbar(cax, cax=cbaxes) 
	ax2.set_xlabel("Principal component #",fontsize=14)
	ax2.set_ylabel("Neuron #",fontsize=14)
	fig.set_size_inches(8,8)
	##plot the first PC
	PC1 = evecs[0,:]
	fig, (ax1,ax2) = plt.subplots(2,sharex=True)
	ax1.stem(PC1)
	op = np.outer(PC1,PC1)
	cax = ax2.imshow(op,interpolation='none',aspect='auto')
	cbaxes = fig.add_axes([0.93, 0.12, 0.03, 0.3]) 
	cb = plt.colorbar(cax, cax=cbaxes) 
	ax2.set_xlabel("Neuron #",fontsize=14)
	ax2.set_ylabel("Neuron #",fontsize=14)
	ax1.set_title("PC1",fontsize=16)
	fig.set_size_inches(6,12)
	fig.subplots_adjust(hspace=0.01)


"""
a function to plot the three first 3 PCs in
3D space.
Inputs:
	Xpca: X-matrix projected onto the PCs
	trial_bins: # of bins per trial in X
	n_trial: number of trials to plot
"""
def plot_pc_trials(Xpca,trial_bins,n_trials=10):
	fig = plt.figure()
	ax = fig.add_subplot(111,projection='3d')
	for i in range(n_trials):
		ax.plot(Xpca[0,i*trial_bins:(i+1)*trial_bins],
				Xpca[1,i*trial_bins:(i+1)*trial_bins],
				Xpca[2,i*trial_bins:(i+1)*trial_bins])
	plt.title("First three PCs")


###this function plots the performance ACROSS DAYS of a given animal.
##It takes in a directiry where the raw (.txt) log files are stored.
##NOTE THAT THIS FUNCTION OPERATES ON RAW .txt FILES!!!!
##this allows for an accurate plotting based in the date recorded
def plot_epoch(directory,plot=True):
	##grab a list of all the logs in the given directory
	fnames = pl.get_log_file_names(directory)
	##x-values are the julian date of the session
	dates = [pl.get_cdate(f) for f in fnames]
	##y-values are the success rates (or percent correct?) for each session
	scores = []
	for session in fnames:
		# print "working on session "+ session
		result = pl.parse_log(session)
		# print "score is "+str(get_success_rate(result))
		scores.append(pl.get_success_rate(result))
	##convert lists to arrays for the next steps
	dates = np.asarray(dates)
	scores = np.asarray(scores)
	##files may not have been opened in order of ascending date, so sort them
	sorted_idx = np.argsort(dates)
	dates = dates[sorted_idx]
	##adjust dates so they start at 0
	dates = dates-(dates[0]-1)
	scores = scores[sorted_idx]
	##we want to not draw lines when there are non-consecutive training days:
	##our x-axis will then be a contiuous range of days
	x = range(1,dates[-1]+1)
	##insert None values in the score list when a date was skipped
	skipped = []
	for idx, date in enumerate(x):
		if date not in dates:
			scores = np.insert(scores,idx,np.nan)
	if plot:
		fig = plt.figure()
		ax = fig.add_subplot(1,1,1)
		ax.plot(x, scores, 'o', color = "c")
		ax.set_xlabel("Training day")
		ax.set_ylabel("Correct trials per min")
	return x, dates, scores

##this function takes in a list of directories where RAW(!) .txt logs are stored
def plot_epochs_multi(directories):
	
	##assume the folder name is the animal name, and that is is two chars
	fig = plt.figure()
	ax = fig.add_subplot(111)
	for d in directories:
		name = d[-11:-9]
		x, dates, scores = plot_epoch(d, plot = False)
		##get a random color to plot this data with
		c = np.random.rand(3,)
		ax.plot(x, scores, 's', markersize = 10, color = c)
		ax.plot(x, scores, linewidth = 2, color = c, label = name)
	ax.legend(loc=2)
	##add horizontal lines showing surgery and recording days
	x_surg = [16,17,18,19]
	y_surg = [-.1,-.1,-.1,-.1]
	x_rec = range(25,44)
	y_rec = np.ones(len(x_rec))*-0.1
	x_pre = range(0,15)
	y_pre = np.ones(len(x_pre))*-0.1
	ax.plot(x_surg, y_surg, linewidth = 4, color = 'k')
	ax.plot(x_pre, y_pre, linewidth = 4, color = 'c')
	ax.plot(x_rec, y_rec, linewidth =4, color = 'r')
	ax.text(15,-0.32, "Surgeries", fontsize = 16)
	ax.text(32,-0.32, "Recording", fontsize = 16)
	ax.text(4,-0.32, "Pre-training", fontsize = 16)		
	ax.set_xlabel("Training day", fontsize=16)
	ax.set_ylabel("Correct trials per min", fontsize=16)
	fig.suptitle("Performance across days", fontsize=16)
	for tick in ax.xaxis.get_major_ticks():
		tick.label.set_fontsize(14)
	for tick in ax.yaxis.get_major_ticks():
		tick.label.set_fontsize(14)

"""
A quick and dirty method for plotting projections from a results
dictionary.
Inputs:
	-results_dict: a dictionary of results as returned from PCA.value_projections
	-epoch: str, just for the title to keep in mind what epoch the data was taken from
"""
def plot_projections(results_dict,epoch='choice'):
	for key in results_dict.keys():
		##parse the data in this dict list
		data = results_dict[key]
		data1_x = data[0]
		data1_y = data[1]
		data2_x = data[2]
		data2_y = data[3]
		fig,ax = plt.subplots(1)
		ax.plot(data1_x,data1_y,linewidth=2,color='b',label='data1')
		ax.plot(data2_x,data2_y,linewidth=2,color='r',label='data2')
		ax.plot(data1_x[0],data1_y[0],marker='o',color='g')
		ax.plot(data2_x[0],data2_y[0],marker='o',color='g')
		ax.set_xlabel("axis 1")
		ax.set_ylabel("axis 2")
		ax.set_title(key)

"""
A function to plot the results of logistic regression
Inputs:
	X: unit data matrix in shape trials x units x bins/time
	y: binary event matrix (1-d)
	sig_idx: index of units with significant predicability
"""
def plot_log_units(X,y,sig_idx):
	##get the indexes of non-sig units
	nonsig_idx = np.asarray([t for t in range(X.shape[1]) if t not in sig_idx])
	n_total = X.shape[1]
	n_sig = sig_idx.size
	n_nonsig = n_total - n_sig
	##determine the number of subplots to use for each catagory
	if n_sig % 5 == 0:
		sig_rows = n_sig/5
	else:
		sig_rows = (n_sig/5)+1
	if n_nonsig% 5 == 0:
		nonsig_rows = n_nonsig/5
	else:
		nonsig_rows = (n_nonsig/5)+1
	##make the figure objects
	sigfig = plt.figure()
	nonsigfig = plt.figure()
	##get the indexes of the two different conditions
	idx_c1 = np.where(y==0)[0]
	idx_c2 = np.where(y==1)[0]
	##start by plotting the significant figures
	for n in range(n_sig):
		idx = sig_idx[n]
		##the data for this unit
		c1_mean, c1_h1, c1_h2 = mean_confidence_interval(X[idx_c1,idx,:])
		c2_mean, c2_h1, c2_h2 = mean_confidence_interval(X[idx_c2,idx,:])
		##the subplot axis for this unit's plot
		ax = sigfig.add_subplot(sig_rows,5,n+1)
		ax.plot(c1_mean,color='b',linewidth=2,label='condition 1')
		ax.plot(c2_mean,color='r',linewidth=2,label='condition 2')
		ax.fill_between(np.arange(c1_mean.size),c1_h1,c1_h2,
			color='b',alpha=0.5)
		ax.fill_between(np.arange(c2_mean.size),c2_h1,c2_h2,
			color='r',alpha=0.5)
		if n >= n_sig-5:
			ax.set_xlabel("Bins",fontsize=12)
		else:
			ax.set_xticklabels([])
		if n % 5 == 0:
			ax.set_ylabel("FR, z-score",fontsize=12)
		else:
			ax.set_yticklabels([])
		ax.set_title("Unit "+str(sig_idx[n]),fontsize=12)
		if n == 0:
			ax.legend(bbox_to_anchor=(1.2, 1.2))
	##now do the non-significant figures
	for n in range(n_nonsig):
		idx = nonsig_idx[n]
		##the data for this unit
		c1_mean, c1_h1, c1_h2 = mean_confidence_interval(X[idx_c1,idx,:])
		c2_mean, c2_h1, c2_h2 = mean_confidence_interval(X[idx_c2,idx,:])
		##the subplot axis for this unit's plot
		ax = nonsigfig.add_subplot(nonsig_rows,5,n+1)
		ax.plot(c1_mean,color='b',linewidth=2,label='condition 1')
		ax.plot(c2_mean,color='r',linewidth=2,label='condition 2')
		ax.fill_between(np.arange(c1_mean.size),c1_h1,c1_h2,
			color='b',alpha=0.5)
		ax.fill_between(np.arange(c2_mean.size),c2_h1,c2_h2,
			color='r',alpha=0.5)
		if n >= n_sig-5:
			ax.set_xlabel("Bins",fontsize=12)
		else: 
			ax.set_xticklabels([])
		if n % 5 == 0:
			ax.set_ylabel("FR, z-score",fontsize=12)
		else: 
			ax.set_yticklabels([])
		ax.set_title("Unit "+str(nonsig_idx[n]),fontsize=12)
		if n == 0:
			ax.legend(bbox_to_anchor=(1.2, 1.2))
	##finish up
	sigfig.suptitle("Units with significant (P<.05) predictability",fontsize=14)
	nonsigfig.suptitle("Units with non-significant predicability",fontsize=14)
	#sigfig.tight_layout()
	#nonsigfig.tight_layout()


"""
A helper function to calculate the mean and 95% CI of some data
Imputs:
	Data: array of values
	confidence: value for CI
"""
def mean_confidence_interval(a, confidence=0.95):
    n = a.shape[0]
    m, se = np.mean(a,axis=0), scipy.stats.sem(a,axis=0)
    h = np.zeros(m.shape)
    for i in range(m.shape[0]):
    	h[i] = se[i] * sp.stats.t._ppf((1+confidence)/2., n-1)
    return m, m-h, m+h