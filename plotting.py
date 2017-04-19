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
import full_analyses as fa
import h5py
import scipy.stats
import scipy as sp
import session_analysis as sa
import matplotlib as ml
import dpca
from scipy import stats
from matplotlib import cm
import matplotlib
import collections

"""
A function to plot examplar units from logistic regression, from one session at a time.
Inputs:
	f_data: file path to a hdf5 file with logistic regression data
	sig_level: p-value threshold for significance
	test_type: string; there should be two statistucs; a log-liklihood ratio p-val
		from the fit from statsmodels ('llrp_pvals'), and a p-val from doing permutation
		testing with shuffled data ('pvals'). This flag lets you select which one to use.
	accuracy_thresh: the threshold to use when considering candidate units
"""
def plot_example_log_units(f_data,sig_level=0.05,test_type='pvals',accuracy_thresh=0.8,sample_ms=40):
	global color_lut
	##get the data in dictionary form
	data = sa.get_log_regression_samples(f_data,sig_level=sig_level,test_type=test_type,
		accuracy_thresh=accuracy_thresh)
	##get the different flavors of encoding
	encoding_types = list(data)
	for t in encoding_types:
		##get some info about the data here
		event_types = list(data[t])
		n_units = len(data[t][event_types[0]])
		##proceed if there are units to plot
		if n_units > 0:
			n_samples = data[t][event_types[0]][0].shape[1]
			##make a figure for this unit
			for u in range(n_units):
				fig,axes = plt.subplots(1,3)
				fig.suptitle(t,fontsize=14,weight='bold')
				for i,epoch in enumerate(list(sa.event_pairs)):
					ax = axes[i]
					events = sa.event_pairs[epoch]
					if epoch == 'outcome':
						x = np.arange(-sample_ms,(n_samples-1)*sample_ms,sample_ms)
					else:
						x = np.arange(-sample_ms*(n_samples-2),sample_ms*2,sample_ms)
					for j,e in enumerate(events):
						mean, upper, lower = mean_and_sem(data[t][e][u])
						ax.plot(x,mean,linewidth=2,color=color_lut[epoch][j],label=e)
						ax.fill_between(x,lower,upper,color=color_lut[epoch][j],alpha=0.5)
					ax.vlines(0,ax.get_ylim()[0],ax.get_ylim()[1],color='k',linewidth=2,linestyle='dashed')
					if i == 0:
						ax.set_ylabel("Firing rate, z-score",fontsize=14,weight='bold')
					ax.set_xlabel("Samples",fontsize=14,weight='bold')
					for tick in ax.xaxis.get_major_ticks():
						tick.label.set_fontsize(14)
					for tick in ax.yaxis.get_major_ticks():
						tick.label.set_fontsize(14)
					ax.set_title(epoch,fontsize=14,weight='bold')
					ax.legend()





"""
A function to plot logistic regression data for all sessions, looking specifically
at whether units encode more than one parameter.
"""
def plot_log_units_all2(dir_list=None,session_range=None,sig_level=0.05,test_type='llr_pval',cmap='brg'):
	##coding this in just to save time
	if dir_list == None:
		dir_list = [
		"/Volumes/Untitled/Ryan/DS_animals/results/LogisticRegression/40ms_bins_0.05/S1",
		"/Volumes/Untitled/Ryan/DS_animals/results/LogisticRegression/40ms_bins_0.05/S2",
		"/Volumes/Untitled/Ryan/DS_animals/results/LogisticRegression/40ms_bins_0.05/S3"
		]
	##get the data
	results = fa.analyze_log_regressions(dir_list,session_range=session_range,sig_level=sig_level,
		test_type=test_type)
	epochs = list(results.columns)
	##we'll sort everything by the first epoch
	n_units = results.index.max()
	##parse the data according to which units encode what param
	data = np.zeros(n_units)
	for i in range(n_units):
		line = results.loc[i]
		if not np.isnan(line['action']) and (np.isnan(line['context']) and np.isnan(line['outcome'])):
			##case where it's action-only
			data[i] = 0
		elif (not np.isnan(line['action']) and not np.isnan(line['context'])) and np.isnan(line['outcome']):
			##case wher it's action and context
			data[i] = 1
		elif not np.isnan(line['context']) and (np.isnan(line['action']) and np.isnan(line['outcome'])):
			##case where it's context-only
			data[i] = 2
		elif (not np.isnan(line['context']) and not np.isnan(line['outcome'])) and np.isnan(line['action']):
			##case where it's context and outcome
			data[i] = 3
		elif not np.isnan(line['outcome']) and (np.isnan(line['action']) and np.isnan(line['context'])):
			##case where it's outcome-only
			data[i] = 4
		elif (not np.isnan(line['outcome']) and not np.isnan(line['action'])) and np.isnan(line['context']):
			##case where it's action and outcome
			data[i] = 5
		elif (not np.isnan(line['outcome']) and not np.isnan(line['action']) and not np.isnan(line['context'])):
			##case where it encodes all 3
			data[i] = 6
		else:
			print("Warning: no catagory found for unit "+str(i))
	##now sort them all in order
	sort_idx = np.argsort(data)
	data = data[sort_idx]
	##determine the size of the image matrix based on the number of units
	side_len = np.ceil(np.sqrt(n_units)).astype(int)
	##set up the figure with GridSpec
	fig = plt.figure()
	vmax = 6 ##store the global max an min prediction strengths so we can scale all plots equally
	vmin = 0
	##produce an image matrix of NaNs (should be white); these will later be filled with 
	##sig unit values when appropriate
	img_mat = np.empty((side_len,side_len))
	img_mat[:] = np.nan
	##fill the image matrix with values from the sig units
	for s in range(n_units):
		##fill the image matrix spot for this unit
		c,r = rc_idx(s,side_len)
		img_mat[r,c] = data[s]
		##now plot the image matrix
	ax = fig.add_subplot(111)
	ax.set_title("Encoding for all units",fontsize=14,weight='bold')
	ax.set_ylabel("Units",fontsize=14,weight='bold')
	ax.set_xlabel("Units",fontsize=14,weight='bold')
	##turn off the x and y labels
	ax.set_xticklabels([])
	ax.set_yticklabels([])
	##plot the actual data
	ax.imshow(img_mat,interpolation='none',cmap=cmap)
	for j in range(n_units):
		r,c = rc_idx(j,side_len)
		ax.text(r,c,str(j+1),fontsize=4)
	##add a legend, using the colormap values. There's probably a more elegant way to do this without 
	##making fake lines, but whatev
	labeldict = {
	'Action\nonly':0,'Action +\ncontext':1,'Context\nonly':2,'Context +\noutcome':3,
	'Outcome\nonly':4,'Outcome +\naction':5,'Action+ \noutcome +\ncontext':6}
	lines = []
	for label in list(labeldict):
		lines.append(matplotlib.lines.Line2D([],[],color=cm.brg(labeldict[label]/6.0),markersize=100,label=label))
	labels = [h.get_label() for h in lines] 
	ax.legend(handles=lines,labels=labels,bbox_to_anchor=(0,1))
	##let's also make a bar graph out of this
	bar_dict = {}
	for key in list(labeldict):
		bar_dict[key] = (data==labeldict[key]).sum()/float(n_units)
	x = np.arange(len(bar_dict))
	fig2 = plt.figure()
	ax2 = fig2.add_subplot(111)
	bars = ax2.bar(x,list(bar_dict.values()),align='center',width=0.5)
	for i in x:
		label = labels[i]
		bars[i].set_color(cm.brg(labeldict[label]/6.0))
	ax2.set_xticks(x)
	ax2.set_xticklabels(list(bar_dict),rotation=45,weight='bold',fontsize=10)
	ax2.set_ylabel('Percent of significant units',fontsize=14,weight='bold')
	ax2.set_title("Proportion of units encoding task params",fontsize=14,weight='bold')



"""
A function to plot logistic regression data for all sessions.
"""
def plot_log_units_all(dir_list=None,session_range=None,sig_level=0.05,test_type='llr_pval'):
	##coding this in just to save time
	if dir_list == None:
		dir_list = [
		"/Volumes/Untitled/Ryan/DS_animals/results/LogisticRegression/50ms_bins_0.05/S1",
		"/Volumes/Untitled/Ryan/DS_animals/results/LogisticRegression/50ms_bins_0.05/S2",
		"/Volumes/Untitled/Ryan/DS_animals/results/LogisticRegression/50ms_bins_0.05/S3"
		]
	##get the data
	results = fa.analyze_log_regressions(dir_list,session_range=session_range,sig_level=sig_level,
		test_type=test_type)
	##the different epochs
	epochs = list(results.columns)
	##we'll sort everything by the first epoch
	n_units = results.index.max()
	##we have to do a little trick, because NaN's are ignored normally (we want to sort them)
	to_sort = np.nan_to_num(list(np.asarray(results[epochs[0]])))
	sort_idx = np.argsort(to_sort)[::-1][:n_units]
	##determine the size of the image matrix based on the number of units
	side_len = np.ceil(np.sqrt(n_units)).astype(int)
	##set up the figure with GridSpec
	fig = plt.figure()
	gs = gridspec.GridSpec(1,len(epochs),wspace=0,hspace=0)
	##a list to store all the image plots
	cplots = []
	vmax = 0 ##store the global max an min prediction strengths so we can scale all plots equally
	vmin = 0.5
	##residual from copied code
	m = 0
	##outer loop is for plotting epochs (columns)
	for n, epoch in enumerate(epochs):
		##produce an image matrix of NaNs (should be white); these will later be filled with 
		##sig unit values when appropriate
		img_mat = np.empty((side_len,side_len))
		img_mat[:] = np.nan
		##now we need to get the data for this set.
		data = np.asarray(results[epoch])[sort_idx]
		##fill the image matrix with accuracy values from the sig units
		for s in range(n_units):
			k = data[s] ##the strength of the prediction for this unit
			if k > vmax:
				vmax = k
			if k < vmin:
				vmin = k
			##fill the image matrix spot for this unit
			c,r = rc_idx(s,side_len)
			img_mat[r,c] = k
		##now plot the image matrix
		ax = plt.subplot(gs[0,n])
		ax.set_title(epoch,fontsize=14,weight='bold')
		##if it's the first column, add a y-axis label
		if n == 0:
			ax.set_ylabel("Units",fontsize=14,weight='bold')
		ax.set_xlabel("Units",fontsize=14,weight='bold')
		##turn off the x and y labels
		ax.set_xticklabels([])
		ax.set_yticklabels([])
		##plot the actual data
		cplots.append(ax.imshow(img_mat,interpolation='none',cmap='summer'))
		for j in range(n_units):
			r,c = rc_idx(j,side_len)
			ax.text(r,c,str(j+1),fontsize=4)
	##rescale all plots to the same range
	for cp in cplots:
		cp.set_clim(vmin=vmin,vmax=vmax)
	##add a colorbar 
	cbaxes = fig.add_axes([0.05, 0.15, 0.9, 0.025])
	cb = fig.colorbar(cax=cbaxes,mappable=cplots[0],orientation='horizontal')
	cbaxes.set_xlabel("Prediction accuracy",fontsize=14)


"""
A function to plot the reversal curves early VS late.
"""
def plot_rev_curves(early_range=[0,5],late_range=[-6,-1],window=[80,80],save=False):
	##get the early data
	early = fa.get_switch_prob(session_range=early_range,window=window)
	##the late data
	late = fa.get_switch_prob(session_range=late_range,window=window)
	##smooth things out a bit
	early = pe.gauss_convolve(early,1)
	late = pe.gauss_convolve(late,1)
	##the x-axis in trials
	x = np.arange(-window[0],window[1])
	##plot it
	fig,ax = plt.subplots(1)
	ax.plot(x,early,color='b',marker='o',label='first 4 days',linewidth=2)
	ax.plot(x,late,color='cyan',marker='o',label='last 4 days',linewidth=2)
	ax.vlines(0,0,1,color='r',linestyle='dashed',linewidth=2)
	ax.set_ylabel("Probability of pressing lever 2",fontsize=14)
	ax.set_xlabel("Trials after switch",fontsize=14)
	for tick in ax.xaxis.get_major_ticks():
		tick.label.set_fontsize(14)
	for tick in ax.yaxis.get_major_ticks():
		tick.label.set_fontsize(14)
	ax.legend()
	ax.set_title("Performance around block switches",fontsize=14)
	if save:
		fig.savefig("/Volumes/Untitled/Ryan/DS_animals/plots/switch_curves.png")
		fig.savefig("/Volumes/Untitled/Ryan/DS_animals/plots/switch_curves.svg")




"""
A function to plot performance (correct press/all presses)
for all animals across sessions.
"""
def plot_performance_all(save=False):
	##Start by getting the data
	data = fa.get_p_correct()
	##add a break in the data to acocunt for surgeries
	surgery_break = np.zeros((data.shape[0],4))
	surgery_break[:] = np.nan
	pre_surgery = data[:,0:8]
	post_surgery = data[:,8:]
	data = np.concatenate((pre_surgery,surgery_break,post_surgery),axis=1)
	##get the mean and std error
	mean = np.nanmean(data,axis=0)
	err = np.nanstd(data,axis=0)/np.sqrt(data.shape[0])
	##the x-axis
	x = np.arange(1,data.shape[1]+1)
	##set up the plot
	fig,(ax,ax2) = plt.subplots(nrows=1,ncols=2,sharey=True)
	ax.errorbar(x,mean,linewidth=2,color='k',yerr=err,capthick=2)
	##also plot animals individually
	for i in range(data.shape[0]):
		ax.plot(x,data[i,:],color='b',linewidth=2,alpha=0.5)
	for tick in ax.xaxis.get_major_ticks():
		tick.label.set_fontsize(14)
	for tick in ax.yaxis.get_major_ticks():
		tick.label.set_fontsize(14)
	ax.set_xlabel("Training day",fontsize=14)
	ax.set_ylabel("Percent correct",fontsize=14)
	ax.set_title("Performance over days",fontsize=14)
	ax.text(8,ax.get_ylim()[0]+0.05,"Implant\nsurgery",fontsize=14,color='r')
	ax.plot(np.arange(8,13),np.ones(5)*ax.get_ylim()[0]+0.02,linewidth=4,color='r')
	##now compare first 3 and last 3 days
	first3 = data[:,0:4].mean(axis=1)
	last3 = data[:,-7:-3].mean(axis=1) ##I think one animal didn't do as many days
	means = np.array([first3.mean(),last3.mean()])
	sems = np.array([stats.sem(first3),stats.sem(last3)])
	##
	x = np.array([1,2])
	xerr = np.ones(2)*0.1
	for i in range(first3.size):
		ax2.plot(x[0],first3[i],color='cyan',linewidth=2,marker='o',alpha=0.5)
		ax2.plot(x[1],last3[i],color='blue',linewidth=2,marker='o',alpha=0.5)
	ax2.errorbar(x,means,yerr=sems,xerr=xerr,fmt='none',ecolor='k',capthick=2,elinewidth=2)
	plt.xticks(x,['first 4','last 4'])
	for ticklabel in ax2.get_xticklabels():
		ticklabel.set_fontsize(14)
	for ticklabel in ax2.get_yticklabels():
		ticklabel.set_fontsize(14)
	# ax2.set_ylabel("Percent correct",fontsize=14)
	ax2.set_xlabel("Training day",fontsize=14)
	ax2.set_title("Comparison eary-late",fontsize=14)
	tval,pval = stats.ttest_rel(first3,last3)
	ax2.text(1.5,0.82,"p={0:.4f}".format(pval))
	print("mean first 4="+str(first3.mean()))
	print("mean last 4="+str(last3.mean()))
	print("pval="+str(pval))
	print("tval="+str(tval))
	plt.tight_layout()
	if save:
		fig.savefig("/Volumes/Untitled/Ryan/DS_animals/plots/performance.png")
		fig.savefig("/Volumes/Untitled/Ryan/DS_animals/plots/performance.svg")





"""
A function to plot performance after switches (correct press/all presses),
as defined by the number of trials to reach criterion, for all animals across sessions.
Inputs:
	crit_trials: the number of trials to average over to determine performance
	crit_level: the criterion performance level to use
	exclude_first: whether to exclude the first block, but only if there is more than one block.
"""
def plot_switch_performance_all(crit_trials=5,crit_level=0.8,exclude_first=False,
	save=False):
	##Start by getting the data
	data = fa.get_criterion(crit_trials=crit_trials,
				crit_level=crit_level,exclude_first=exclude_first)
	##add a break in the data to acocunt for surgeries
	surgery_break = np.zeros((data.shape[0],4))
	surgery_break[:] = np.nan
	pre_surgery = data[:,0:8]
	post_surgery = data[:,8:]
	data = np.concatenate((pre_surgery,surgery_break,post_surgery),axis=1)
	##get the mean and std error
	mean = np.nanmean(data,axis=0)
	err = np.nanstd(data,axis=0)/np.sqrt(data.shape[0])
	##the x-axis
	x = np.arange(1,data.shape[1]+1)
	##set up the plot
	fig,(ax,ax2) = plt.subplots(nrows=1,ncols=2,sharey=True)
	ax.errorbar(x,mean,linewidth=2,color='k',yerr=err,capthick=2)
	##also plot animals individually
	for i in range(data.shape[0]):
		ax.plot(x,data[i,:],color='g',linewidth=2,alpha=0.5)
	for tick in ax.xaxis.get_major_ticks():
		tick.label.set_fontsize(14)
	for tick in ax.yaxis.get_major_ticks():
		tick.label.set_fontsize(14)
	ax.set_xlabel("Training day",fontsize=14)
	ax.set_ylabel("Trials",fontsize=14)
	ax.set_title("Trials to reach criteria ("+str(crit_level)+")\n after switch",fontsize=14)
	ax.text(8,ax.get_ylim()[0]+2,"Implant\nsurgery",fontsize=14,color='r')
	ax.plot(np.arange(8,13),np.ones(5)*ax.get_ylim()[0]+0.02,linewidth=4,color='r')
	##now compare first 3 and last 3 days
	first3 = data[:,0:4].mean(axis=1)
	last3 = data[:,-7:-3].mean(axis=1) ##I think one animal didn't do as many days
	means = np.array([first3.mean(),last3.mean()])
	sems = np.array([stats.sem(first3),stats.sem(last3)])
	##
	x = np.array([1,2])
	xerr = np.ones(2)*0.1
	for i in range(first3.size):
		ax2.plot(x[0],first3[i],color='lightgreen',linewidth=2,marker='o',alpha=0.5)
		ax2.plot(x[1],last3[i],color='green',linewidth=2,marker='o',alpha=0.5)
	ax2.errorbar(x,means,yerr=sems,xerr=xerr,fmt='none',ecolor='k',capthick=2,elinewidth=2)
	plt.xticks(x,['first 4','last 4'])
	for ticklabel in ax2.get_xticklabels():
		ticklabel.set_fontsize(14)
	for ticklabel in ax2.get_yticklabels():
		ticklabel.set_fontsize(14)
	# ax2.set_ylabel("Number of trials",fontsize=14)
	ax2.set_xlabel("Training day",fontsize=14)
	ax2.set_title("Comparison eary-late",fontsize=14)
	tval,pval = stats.ttest_rel(first3,last3)
	ax2.text(1.5,42,"p={0:.4f}".format(pval))
	print("mean first 4="+str(first3.mean()))
	print("mean last 4="+str(last3.mean()))
	print("pval="+str(pval))
	print("tval="+str(tval))
	plt.tight_layout()
	if save:
		fig.savefig("/Volumes/Untitled/Ryan/DS_animals/plots/switch_performance.png")
		fig.savefig("/Volumes/Untitled/Ryan/DS_animals/plots/switch_performance.svg")

"""
A function to plot volatility
for all animals across sessions.
"""
def plot_volatility_all(save=False):
	##Start by getting the data
	data = fa.get_volatility()
	##add a break in the data to acocunt for surgeries
	surgery_break = np.zeros((data.shape[0],4))
	surgery_break[:] = np.nan
	pre_surgery = data[:,0:8]
	post_surgery = data[:,8:]
	data = np.concatenate((pre_surgery,surgery_break,post_surgery),axis=1)
	##get the mean and std error
	mean = np.nanmean(data,axis=0)
	err = np.nanstd(data,axis=0)/np.sqrt(data.shape[0])
	##the x-axis
	x = np.arange(1,data.shape[1]+1)
	##set up the plot
	fig,(ax,ax2) = plt.subplots(nrows=1,ncols=2,sharey=True)
	ax.errorbar(x,mean,linewidth=2,color='k',yerr=err,capthick=2)
	##also plot animals individually
	for i in range(data.shape[0]):
		ax.plot(x,data[i,:],color='purple',linewidth=2,alpha=0.5)
	for tick in ax.xaxis.get_major_ticks():
		tick.label.set_fontsize(14)
	for tick in ax.yaxis.get_major_ticks():
		tick.label.set_fontsize(14)
	ax.set_xlabel("Training day",fontsize=14)
	ax.set_ylabel("Percent of switches",fontsize=14)
	ax.set_title("Volatility after unrewarded trials",fontsize=14)
	ax.text(8,ax.get_ylim()[0]+0.05,"Implant\nsurgery",fontsize=14,color='r')
	ax.plot(np.arange(8,13),np.ones(5)*ax.get_ylim()[0]+0.02,linewidth=4,color='r')
	##now compare first 3 and last 3 days
	first3 = data[:,0:4].mean(axis=1)
	last3 = data[:,-7:-3].mean(axis=1) ##I think one animal didn't do as many days
	means = np.array([first3.mean(),last3.mean()])
	sems = np.array([stats.sem(first3),stats.sem(last3)])
	#####
	x = np.array([1,2])
	xerr = np.ones(2)*0.1
	for i in range(first3.size):
		ax2.plot(x[0],first3[i],color='plum',linewidth=2,marker='o',alpha=0.5)
		ax2.plot(x[1],last3[i],color='purple',linewidth=2,marker='o',alpha=0.5)
	ax2.errorbar(x,means,yerr=sems,xerr=xerr,fmt='none',ecolor='k',capthick=2,elinewidth=2)
	plt.xticks(x,['first 4','last 4'])
	for ticklabel in ax2.get_xticklabels():
		ticklabel.set_fontsize(14)
	for ticklabel in ax2.get_yticklabels():
		ticklabel.set_fontsize(14)
	ax2.set_xlabel("Training day",fontsize=14)
	ax2.set_title("Comparison eary-late",fontsize=14)
	tval,pval = stats.ttest_rel(first3,last3)
	ax2.text(1.5,0.42,"p={0:.4f}".format(pval))
	print("mean first 4="+str(first3.mean()))
	print("mean last 4="+str(last3.mean()))
	print("pval="+str(pval))
	print("tval="+str(tval))
	plt.tight_layout()
	if save:
		fig.savefig("/Volumes/Untitled/Ryan/DS_animals/plots/volatility.png")
		fig.savefig("/Volumes/Untitled/Ryan/DS_animals/plots/volatility.svg")

"""
A function to plot persistence
for all animals across sessions.
"""
def plot_persistence_all(save=False):
	##Start by getting the data
	data = fa.get_persistence()
	##add a break in the data to acocunt for surgeries
	surgery_break = np.zeros((data.shape[0],4))
	surgery_break[:] = np.nan
	pre_surgery = data[:,0:8]
	post_surgery = data[:,8:]
	data = np.concatenate((pre_surgery,surgery_break,post_surgery),axis=1)
	##get the mean and std error
	mean = np.nanmean(data,axis=0)
	err = np.nanstd(data,axis=0)/np.sqrt(data.shape[0])
	##the x-axis
	x = np.arange(1,data.shape[1]+1)
	##set up the plot
	fig,(ax,ax2) = plt.subplots(nrows=1,ncols=2,sharey=True)
	ax.errorbar(x,mean,linewidth=2,color='k',yerr=err,capthick=2)
	##also plot animals individually
	for i in range(data.shape[0]):
		ax.plot(x,data[i,:],color='orange',linewidth=2,alpha=0.5)
	for tick in ax.xaxis.get_major_ticks():
		tick.label.set_fontsize(14)
	for tick in ax.yaxis.get_major_ticks():
		tick.label.set_fontsize(14)
	ax.set_xlabel("Training day",fontsize=14)
	ax.set_ylabel("Persistence score",fontsize=14)
	ax.set_title("Persistence after rewarded trials",fontsize=14)
	ax.text(8,ax.get_ylim()[0]+0.05,"Implant\nsurgery",fontsize=14,color='r')
	ax.plot(np.arange(8,13),np.ones(5)*ax.get_ylim()[0]+0.02,linewidth=4,color='r')
	##now compare first 3 and last 3 days
	first3 = data[:,0:4].mean(axis=1)
	last3 = data[:,-7:-3].mean(axis=1) ##I think one animal didn't do as many days
	means = np.array([first3.mean(),last3.mean()])
	sems = np.array([stats.sem(first3),stats.sem(last3)])
	##
	x = np.array([1,2])
	xerr = np.ones(2)*0.1
	for i in range(first3.size):
		ax2.plot(x[0],first3[i],color='orange',linewidth=2,marker='o',alpha=0.5)
		ax2.plot(x[1],last3[i],color='darkorange',linewidth=2,marker='o',alpha=0.7)
	ax2.errorbar(x,means,yerr=sems,xerr=xerr,fmt='none',ecolor='k',capthick=2,elinewidth=2)
	plt.xticks(x,['first 4','last 4'])
	for ticklabel in ax2.get_xticklabels():
		ticklabel.set_fontsize(14)
	for ticklabel in ax2.get_yticklabels():
		ticklabel.set_fontsize(14)
	ax2.set_xlabel("Training day",fontsize=14)
	ax2.set_title("Comparison eary-late",fontsize=14)
	tval,pval = stats.ttest_rel(first3,last3)
	ax2.text(1.5,4,"p={0:.4f}".format(pval))
	print("mean first 4="+str(first3.mean()))
	print("mean last 4="+str(last3.mean()))
	print("pval="+str(pval))
	print("tval="+str(tval))
	plt.tight_layout()
	if save:
		fig.savefig("/Volumes/Untitled/Ryan/DS_animals/plots/persistence.png")
		fig.savefig("/Volumes/Untitled/Ryan/DS_animals/plots/persistence.svg")


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
	fig,ax = plt.subplots(1)
	fig.patch.set_facecolor('white')
	fig.set_size_inches(10,4)
	if upper_trial_durs is not None:
		upper_dur_mean = abs(upper_trial_durs).mean()
		upper_dur_std = abs(upper_trial_durs).std()
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
		# ax2.scatter(r_upper_times,abs(r_upper_durs),edgecolor='green',marker='o',s=30,
		# 	linewidth=2,facecolors=('green',),alpha=0.7,label='rewarded upper lever')
		# ax2.scatter(u_upper_times,abs(u_upper_durs),color='green',marker='x',s=30,
		# 	linewidth=2,label='unrewarded upper lever')		
	if lower_trial_durs is not None:
		lower_dur_mean = abs(lower_trial_durs).mean()
		lower_dur_std = abs(lower_trial_durs).std()
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
		# ax2.scatter(r_lower_times,abs(r_lower_durs),edgecolor='red',marker='o',s=30,
		# 	linewidth=2,facecolors=('red',),alpha=0.7,label='rewarded lower lever')
		# ax2.scatter(u_lower_times,abs(u_lower_durs),color='red',marker='x',s=30,
		# 	linewidth=2,label='unrewarded lower lever')
	# for label in ax2.xaxis.get_ticklabels()[1::2]:
	# 	label.set_visible(False)
	for label in ax.xaxis.get_ticklabels()[1::2]:
		label.set_visible(False)
	# for label in ax2.xaxis.get_ticklabels()[::2]:
	# 	label.set_fontsize(14)
	for label in ax.xaxis.get_ticklabels()[::2]:
		label.set_fontsize(14)
	# for label in ax2.yaxis.get_ticklabels()[1::2]:
	# 	label.set_visible(False)
	for label in ax.yaxis.get_ticklabels()[1::2]:
		label.set_visible(False)
	# for label in ax2.yaxis.get_ticklabels()[::2]:
	# 	label.set_fontsize(14)
	for label in ax.yaxis.get_ticklabels()[::2]:
		label.set_fontsize(14)
	# ##if there are outliers, break the axis
	# try:
	# 	outliers = np.hstack((upper_outliers,lower_outliers))
	# except NameError: ##if we only have one kind of trial
	# 	if upper_trial_durs is not None:
	# 		outliers = upper_outliers
	# 	else:
	# 		outliers = lower_outliers
	# if outliers.size > 0:
	# 	ax2.set_ylim(-1,max(2*lower_dur_std,2*upper_dur_std))
	# 	ax.set_ylim(outliers.min()-5,outliers.max()+10)
	# 	# hide the spines between ax and ax2
	# 	ax.spines['bottom'].set_visible(False)
	# 	ax2.spines['top'].set_visible(False)
	# 	ax.xaxis.tick_top()
	# 	ax.tick_params(labeltop='off')  # don't put tick labels at the top
	# 	ax2.xaxis.tick_bottom()

	# 	# This looks pretty good, and was fairly painless, but you can get that
	# 	# cut-out diagonal lines look with just a bit more work. The important
	# 	# thing to know here is that in axes coordinates, which are always
	# 	# between 0-1, spine endpoints are at these locations (0,0), (0,1),
	# 	# (1,0), and (1,1).  Thus, we just need to put the diagonals in the
	# 	# appropriate corners of each of our axes, and so long as we use the
	# 	# right transform and disable clipping.

	# 	d = .015  # how big to make the diagonal lines in axes coordinates
	# 	# arguments to pass plot, just so we don't keep repeating them
	# 	kwargs = dict(transform=ax.transAxes, color='k', clip_on=False)
	# 	ax.plot((-d, +d), (-d, +d), **kwargs)        # top-left diagonal
	# 	ax.plot((1 - d, 1 + d), (-d, +d), **kwargs)  # top-right diagonal

	# 	kwargs.update(transform=ax2.transAxes)  # switch to the bottom axes
	# 	ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs)  # bottom-left diagonal
	# 	ax2.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)  # bottom-right diagonal
	# else:
	# 	fig.delaxes(ax)
	# 	fig.draw()
	# if outliers.size > 0:
	# 	legend=ax.legend(frameon=False)
	# 	try:
	# 		plt.text(0.2, 0.9,'upper lever mean = '+str(upper_dur_mean),ha='center',
	# 			va='center',transform=ax.transAxes,fontsize=14)
	# 	except NameError:
	# 		pass
	# 	try:
	# 		plt.text(0.2, 0.8,'lower lever mean = '+str(lower_dur_mean),ha='center',
	# 			va='center',transform=ax.transAxes,fontsize=14)
	# 	except NameError:
	# 		pass
	# else:
	legend=ax.legend(frameon=False)
	for label in legend.get_texts():
		label.set_fontsize('large')
	legend.get_frame().set_facecolor('none')
	ax.set_xlabel("Time in session, s",fontsize=14)
	ax.set_ylabel("Trial duration, s",fontsize=14)
	fig.suptitle("Duration of trials",fontsize=14)


"""
takes in a data dictionary produced by parse_log
plots the lever presses and the switch points for levers
"""
def plot_presses(f_in, sigma = 5):
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
A function to plot the lever presses in a slightly different way, 
with a little more detail about presses and correct/reward values.
This plotting function uses a results dictionary from the function
pt.get_event_data, which will parse a behavioral timestamps file into 
the appropriate format.
Inputs:
	f_behavior: a hdf5 file path to where the raw behavior data is stored.
"""
def plot_trial(f_behavior):
	results = pt.get_event_data(f_behavior)
	##we will split this data a little further, into
	##different catagories.
	scalex = 1000*60 ##conversion factor from ms to min
	correct_upper = np.intersect1d(results['upper_lever'],results['correct_lever'])
	correct_lower = np.intersect1d(results['lower_lever'],results['correct_lever'])
	incorrect_upper = np.intersect1d(results['upper_lever'],results['incorrect_lever'])
	incorrect_lower = np.intersect1d(results['lower_lever'],results['incorrect_lever'])
	##now grab the correct, but unrewarded levers for each lever type
	corr_unrew_upper = np.intersect1d(correct_upper,results['unrewarded_lever'])
	corr_unrew_lower = np.intersect1d(correct_lower,results['unrewarded_lever'])
	##finally, remove these indices from the correct arrays
	correct_upper = np.setdiff1d(np.union1d(correct_upper, corr_unrew_upper), 
		np.intersect1d(correct_upper, corr_unrew_upper))
	correct_lower = np.setdiff1d(np.union1d(correct_lower, corr_unrew_lower), 
		np.intersect1d(correct_lower, corr_unrew_lower))
	correct_lower = correct_lower/scalex
	correct_upper = correct_upper/scalex
	incorrect_upper = incorrect_upper/scalex
	incorrect_lower = incorrect_lower/scalex
	corr_unrew_upper = corr_unrew_upper/scalex
	corr_unrew_lower = corr_unrew_lower/scalex
	##create the plot
	fig, ax = plt.subplots(1)
	ax.set_xlim(-1,results['session_length']/scalex)
	##mark the rule changes
	ax.vlines(results['upper_rewarded']/scalex, 0.5, 2.5, colors = 'r', linestyles = 'dashed', 
		linewidth = '2', alpha = 0.5)
	ax.vlines(results['lower_rewarded']/scalex, 0.5, 2.5, colors = 'b', linestyles = 'dashed', 
		linewidth = '2', alpha =0.5)
	##now plot the presses
	ax.plot(correct_upper,np.ones(correct_upper.size)*2+np.random.uniform(-0.2,0.2,size=correct_upper.size),
		linestyle='none',marker='o',color='r',label='correct upper')
	ax.plot(correct_lower,np.ones(correct_lower.size)+np.random.uniform(-0.2,0.2,size=correct_lower.size),
		linestyle='none',marker='o',color='b',label='correct lower')
	ax.plot(incorrect_upper,np.ones(incorrect_upper.size)*2+np.random.uniform(-0.2,0.2,size=incorrect_upper.size),
		linestyle='none',marker='x',color='r',label='incorrect upper')
	ax.plot(incorrect_lower,np.ones(incorrect_lower.size)+np.random.uniform(-0.2,0.2,size=incorrect_lower.size),
		linestyle='none',marker='x',color='b',label='incorrect lower')
	ax.plot(corr_unrew_upper,np.ones(corr_unrew_upper.size)*1.7+np.random.uniform(-0.2,0.2,size=corr_unrew_upper.size),
		linestyle='none',marker='o',color='r',markerfacecolor='none',label='correct unrewarded upper')
	ax.plot(corr_unrew_lower,np.ones(corr_unrew_lower.size)*1.3+np.random.uniform(-0.2,0.2,size=corr_unrew_lower.size),
		linestyle='none',marker='o',color='b',markerfacecolor='none',label='correct unrewarded lower')
	ax.set_xlabel("Time in session, mins",fontsize=14)
	ax.set_yticks([1,2])
	ax.set_yticklabels(['Lower\npress','Upper\npress'])
	for tick in ax.xaxis.get_major_ticks():
		tick.label.set_fontsize(14)
	for tick in ax.yaxis.get_major_ticks():
		tick.label.set_fontsize(14)
	ax.set_title("Session "+f_behavior[-11:-5],fontsize=14)
	ax.legend(bbox_to_anchor=(1,1))




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
	colors = ['r','b','g','k','purple','orange']
	##assume the folder name is the animal name, and that is is two chars
	fig = plt.figure()
	ax = fig.add_subplot(111)
	for n,d in enumerate(directories):
		name = d[-11:-9]
		x, dates, scores = plot_epoch(d, plot = False)
		##get a random color to plot this data with
		c = colors[n]
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
def plot_log_units(results,condition):
	data = results[condition]
	sig_idx = data['idx']
	X = data['X']
	##get the indexes of non-sig units
	nonsig_idx = np.asarray([t for t in range(X.shape[0]) if t not in sig_idx])
	n_total = X.shape[0]
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
	events = [x for x in list(data) if 'lever' in x or 'poke' in x]
	idx_c1 = data[events[0]]
	idx_c2 = data[events[1]]
	##start by plotting the significant figures
	for n in range(n_sig):
		idx = sig_idx[n]
		##the data for this unit
		c1_mean, c1_h1, c1_h2 = mean_and_sem(X[idx,idx_c1,:])
		c2_mean, c2_h1, c2_h2 = mean_and_sem(X[idx,idx_c2,:])
		##the subplot axis for this unit's plot
		ax = sigfig.add_subplot(sig_rows,5,n+1)
		ax.plot(c1_mean,color='b',linewidth=2,label=events[0])
		ax.plot(c2_mean,color='r',linewidth=2,label=events[1])
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
			ax.legend(bbox_to_anchor=(0.1, 0.1))
	##now do the non-significant figures
	for n in range(n_nonsig):
		idx = nonsig_idx[n]
		##the data for this unit
		c1_mean, c1_h1, c1_h2 = mean_and_sem(X[idx,idx_c1,:])
		c2_mean, c2_h1, c2_h2 = mean_and_sem(X[idx,idx_c2,:])
		##the subplot axis for this unit's plot
		ax = nonsigfig.add_subplot(nonsig_rows,5,n+1)
		ax.plot(c1_mean,color='b',linewidth=2,label=events[0])
		ax.plot(c2_mean,color='r',linewidth=2,label=events[1])
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
Another function to plot individual unit results from logistic regression (for one session).
This one plots only the significant units, but looks at activity over all epochs
Inputs:
	-f_in: the data file for logistic regression results
"""
def plot_log_units2(f_in):
	##open the file
	f = h5py.File(f_in,'r')
	current_file = f_in[-11:-5]
	## a LUT for different legend labels
	LUT = {
	'block_type':['Upper rewarded','Lower rewarded'],
	'choice':['Upper lever',"Lower lever"],
	'reward':['Rewarded','Unrewarded']
	}
	##determine some metadata about this file
	epochs = ['choice','delay','outcome'] ##this isn't everything but all I want to use for now
	conditions = ['block_type','choice','reward']
	##we already have a function that will give us all of the units with any significant
	##predictability, so let' use it
	cond_idx,cond_ps,multis,n_sig,n_total = sa.parse_log_regression(f_in,epochs) ##n_sig is the array of all sig units
	##make a separate plot for each unit
	for t,sig_idx in enumerate(n_sig):
		fig = plt.figure()
		gs = gridspec.GridSpec(len(epochs),len(conditions))
		##outer loop is for plotting epochs (columns)
		for n, epoch in enumerate(epochs):
			##get the data matrix for this epoch
			X = np.asarray(f[epoch]['X'])
			##inner loop is for task variables
			for m, condition in enumerate(conditions):
				##now plot the data
				ax = plt.subplot(gs[m,n])
				##get the y-data for the choices
				y = np.asarray(f[epoch][condition]['y'])
				##index of trials for the different conditions
				idx_c1 = np.where(y==0)[0]
				idx_c2 = np.where(y==1)[0]
				m_c1 = np.mean(X[idx_c1,sig_idx,:],axis=0) 
				se_c1 = scipy.stats.sem(X[idx_c1,sig_idx,:],axis=0)
				m_c2 = np.mean(X[idx_c2,sig_idx,:],axis=0) 
				se_c2 = scipy.stats.sem(X[idx_c2,sig_idx,:],axis=0)
				##if it's the first row, add a title
				if m == 0:
					ax.set_title(epoch,fontsize=14,weight='bold')
				##if it's the first column, add a y-axis label
				if n == 0:
					ax.set_ylabel("zscore FR for\n"+condition,fontsize=14,weight='bold')
				else:
					ax.set_yticklabels([])
				if m == len(conditions)-1:
					ax.set_xlabel("Bins",fontsize=14)
				else:
					ax.set_xticklabels([])
				ax.plot(m_c1,color='b',linewidth=2,label=LUT[condition][0])
				ax.plot(m_c2,color='r',linewidth=2,label=LUT[condition][1])
				ax.fill_between(np.arange(m_c1.size),m_c1-se_c1,m_c1+se_c1,
					color='b',alpha=0.5)
				ax.fill_between(np.arange(m_c2.size),m_c2-se_c2,m_c2+se_c2,
					color='r',alpha=0.5)
				##add text to signify the prediction strength
				if sig_idx in cond_idx[condition]:
					ax.text(0.1,0.1,"*",fontsize=16,weight='bold',transform=ax.transAxes)
				if n == 0:
					ax.set_ylabel("zscore FR for\n"+condition,fontsize=14,weight='bold')
					ax.legend()
				else:
					ax.set_yticklabels([])
		fig.suptitle("Unit "+str(sig_idx),fontsize=16)
	f.close()		
	##TODO identify for which epochs the unit is significant,and add text about the encoding strength


"""
A function to plot the results of logistic regression
Inputs:
	-f_in: file path to hdf5 file where results are stored
"""
def plot_log_regression(f_in):
	##open the file
	f = h5py.File(f_in,'r')
	current_file = f_in[-11:-5]
	##determine some metadata about this file
	epochs = ['choice','delay','outcome'] ##this isn't everything but all I want to use for now
	conditions = ['block_type','choice','reward']
	##how many units total were recorded?
	n_units = f[epochs[0]]['X'].shape[1]
	##determine the size of the image matrix based on the number of units
	side_len = np.ceil(np.sqrt(n_units)).astype(int)
	##set up the figure with GridSpec
	fig = plt.figure()
	gs = gridspec.GridSpec(len(epochs),len(conditions),wspace=0,hspace=0)
	##a list to store all the image plots
	cplots = []
	vmax = 0 ##store the global max an min prediction strengths so we can scale all plots equally
	vmin = 0
	##outer loop is for plotting epochs (columns)
	for n, epoch in enumerate(epochs):
		##inner loop is for task variables
		for m, condition in enumerate(conditions):
			##produce an image matrix of NaNs (should be white); these will later be filled with 
			##sig unit values when appropriate
			img_mat = np.empty((side_len,side_len))
			img_mat[:] = np.nan
			##now we need to get the data for this set.
			group = f[epoch][condition]
			sig_idx = np.asarray(group['sig_idx']) ##the indices of units that were significant 
			kappas = np.asarray(group['pred_strength']) ##the prediction strength values for *all* units
			##fill the image matrix with values from the sig units only
			for s in range(sig_idx.size):
				idx = sig_idx[s] ##the index of the significant unit out of all units
				k = kappas[idx] ##the strength of the prediction for this unit
				if k > vmax:
					vmax = k
				if k < vmin:
					vmin = k
				##fill the image matrix spot for this unit
				c,r = rc_idx(idx,side_len)
				img_mat[r,c] = k
			##now plot the image matrix
			ax = plt.subplot(gs[m,n])
			##if it's the first row, add a title
			if m == 0:
				ax.set_title(epoch,fontsize=14,weight='bold')
			##if it's the first column, add a y-axis label
			if n == 0:
				ax.set_ylabel(condition,fontsize=14,weight='bold')
			if m == len(conditions)-1:
				ax.set_xlabel("Units")
			##turn off the x and y labels
			ax.set_xticklabels([])
			ax.set_yticklabels([])
			##plot the actual data
			cplots.append(ax.imshow(img_mat,interpolation='none',cmap='summer'))
			for j in range(n_units):
				r,c = rc_idx(j,side_len)
				ax.text(r,c,str(j+1))
	##rescale all plots to the same range
	for cp in cplots:
		cp.set_clim(vmin=0,vmax=vmax)
	##add a colorbar 
	cbaxes = fig.add_axes([0.05, 0.05, 0.9, 0.025])
	cb = fig.colorbar(cax=cbaxes,mappable=cplots[0],orientation='horizontal')
	cbaxes.set_xlabel("Prediction strength",fontsize=14)
	#cb.set_ticks(np.arange(vmin,vmax))
	# cbytick_obj = plt.getp(cb.ax.axes, 'xticklabels')
	# plt.setp(cbytick_obj, fontsize='x-small')
	f.close()
	fig.suptitle(current_file,fontsize=16)

"""
A function to plot the results from analyzing all log regression files
Inputs:
	results: results dictionary returned by fa.analyze_log_regressions
"""
def plot_all_log_regressions(results):
	##make 5 separate figures
	colors = ['r','b','g','k','purple','orange']
	##this will be used as a normalizing factor
	n_totals = results['num_total']
	n_animals = n_totals.shape[0]
	##first one is the total percentage of significant units over training
	fig,ax = plt.subplots(1)
	##get the data for this plot
	data = (results['num_sig']/n_totals)*100
	mean = np.nanmean(data,axis=0)
	sem = np.nanstd(data,axis=0)/np.sqrt(data.shape[0])
	ax.errorbar(np.arange(0,mean.shape[0]),mean,linewidth=4,color='k',yerr=sem)
	for i in range(n_animals):
		ax.plot(data[i,:],color=colors[i],linewidth=2,
			alpha=0.5,label='animal '+str(i))
		ax.set_xlabel("Training day",fontsize=14)
		ax.set_ylabel("Percentage of units",fontsize=14)
		ax.legend(bbox_to_anchor=(1.1, 0.3))
		ax.set_title("Percent significant of all recorded",fontsize=14)
	for tick in ax.xaxis.get_major_ticks():
		tick.label.set_fontsize(14)
	for tick in ax.yaxis.get_major_ticks():
		tick.label.set_fontsize(14)
	##next one is the block type
	fig,ax = plt.subplots(1)
	##get the data for this plot
	data = (results['num_block_type']/n_totals[i,:])*100
	mean = np.nanmean(data,axis=0)
	sem = np.nanstd(data,axis=0)/np.sqrt(data.shape[0])
	ax.errorbar(np.arange(0,mean.shape[0]),mean,linewidth=4,color='k',yerr=sem)
	for i in range(n_animals):
		ax.plot(data[i,:],color=colors[i],linewidth=2,
			alpha=0.5,label='animal '+str(i))
		ax.set_xlabel("Training day",fontsize=14)
		ax.set_ylabel("Percentage of units",fontsize=14)
		ax.legend(bbox_to_anchor=(1.1, 0.3))
		ax.set_title("Percent of significant units encoding block type",fontsize=14)
	for tick in ax.xaxis.get_major_ticks():
		tick.label.set_fontsize(14)
	for tick in ax.yaxis.get_major_ticks():
		tick.label.set_fontsize(14)
	##next one is the choice
	fig,ax = plt.subplots(1)
	##get the data for this plot
	data = (results['num_choice']/n_totals)*100
	mean = np.nanmean(data,axis=0)
	sem = np.nanstd(data,axis=0)/np.sqrt(data.shape[0])
	ax.errorbar(np.arange(0,mean.shape[0]),mean,linewidth=4,color='k',yerr=sem)
	for i in range(n_animals):
		ax.plot(data[i,:],color=colors[i],linewidth=2,
			alpha=0.5,label='animal '+str(i))
		ax.set_xlabel("Training day",fontsize=14)
		ax.set_ylabel("Percentage of units",fontsize=14)
		ax.legend(bbox_to_anchor=(1.1, 0.3))
		ax.set_title("Percent of significant units encoding choice",fontsize=14)
	for tick in ax.xaxis.get_major_ticks():
		tick.label.set_fontsize(14)
	for tick in ax.yaxis.get_major_ticks():
		tick.label.set_fontsize(14)
	##next one is the reward
	fig,ax = plt.subplots(1)
	##get the data for this plot
	data = (results['num_reward']/n_totals)*100
	mean = np.nanmean(data,axis=0)
	sem = np.nanstd(data,axis=0)/np.sqrt(data.shape[0])
	ax.errorbar(np.arange(0,mean.shape[0]),mean,linewidth=4,color='k',yerr=sem)
	for i in range(n_animals):
		ax.plot(data[i,:],color=colors[i],linewidth=2,
			alpha=0.5,label='animal '+str(i))
		ax.set_xlabel("Training day",fontsize=14)
		ax.set_ylabel("Percentage of units",fontsize=14)
		ax.legend(bbox_to_anchor=(1.1, 0.3))
		ax.set_title("Percent of significant units encoding reward",fontsize=14)
	for tick in ax.xaxis.get_major_ticks():
		tick.label.set_fontsize(14)
	for tick in ax.yaxis.get_major_ticks():
		tick.label.set_fontsize(14)
	##next one is multi-units
	fig,ax = plt.subplots(1)
	##get the data for this plot
	data = (results['multi_units']/n_totals)*100
	mean = np.nanmean(data,axis=0)
	sem = np.nanstd(data,axis=0)/np.sqrt(data.shape[0])
	ax.errorbar(np.arange(0,mean.shape[0]),mean,linewidth=4,color='k',yerr=sem)
	for i in range(n_animals):
		ax.plot(data[i,:],color=colors[i],linewidth=2,
			alpha=0.5,label='animal '+str(i))
		ax.set_xlabel("Training day",fontsize=14)
		ax.set_ylabel("Percentage of units",fontsize=14)
		ax.legend(bbox_to_anchor=(1.1, 0.3))
		ax.set_title("Percent of significant units encoding multiple params",fontsize=14)
	for tick in ax.xaxis.get_major_ticks():
		tick.label.set_fontsize(14)
	for tick in ax.yaxis.get_major_ticks():
		tick.label.set_fontsize(14)

"""
A function to plot a spike raster. 
Inputs:
	spikes: an n-trials x n-neuron x n-timebins array of data
	color: color of hash marks
	n_trials: number of trials to plot (too many crowds the plot)
"""
def plot_raster(spike_trains,color='k',n_trials=10):
	trial_len = spike_trains.shape[2] ##length of one trial
	for t in range(n_trials):
			plt.vlines(t*trial_len,0,spike_trains.shape[1],color='r',linestyle='dashed',linewidth=2)
			for train in range(spike_trains.shape[1]):
				plt.vlines(ml.mlab.find(spike_trains[t,train,:]>0)+t*trial_len,
					0.5+train,1.5+train,color=color)
	plt.title('Raster plot',fontsize=14)
	plt.xlabel('bins',fontsize=14)
	plt.ylabel('trial',fontsize=14)
	plt.show()


"""
Plots a sample of the raw data matrix, X.
Inputs:
	-X; raw data matrix; trials x neurons x bins
	-n_trials: number of trials to plot
"""
def plot_X_sample(X,n_trials=10):
	fig, ax = plt.subplots(1)
	trial_len = X.shape[2]
	##collapse the data matrix along the time axis
	X_cat = np.concatenate(X,axis=1)
	cax=ax.imshow(X_cat[:,:trial_len*n_trials],origin='lower',interpolation='none',aspect='auto')
	x = np.arange(0,trial_len*n_trials,trial_len)
	ax.vlines(x,0,X_cat.shape[0], linestyle='dashed',color='white')
	ax.set_ylabel("Neuron #",fontsize=16)
	ax.set_xlabel("Bin #", fontsize=16)
	cb = fig.colorbar(cax,label="spikes rate, zscore")
	fig.set_size_inches(10,6)
	fig.suptitle("Showing first "+str(n_trials)+" trials",fontsize=16)

"""
A function to plot the results of dpca. 
Inputs:
	Z: transformed spike matrix (output from dpca.session_dpca)
	time: time axis (output from dpca.session_dpca)
"""
def plot_dpca_results(Z,time,sig_masks,var_explained,events,n_components=3):
	##get the condition LUT dictionary from the dpca module
	LUT = dpca.condition_LUT
	##set up the figure
	fig = plt.figure()
	n_conditions = len(list(LUT))
	##we'll plot the first n dPC's for each condition
	for c in range(n_conditions):
		condition = list(LUT)[c]
		title = LUT[condition]
		data = Z[condition]
		for p in range(n_components):
			plotnum = (c*n_components+p)+1
			ax = fig.add_subplot(n_conditions,n_components,plotnum)
			ax.plot(time,data[p,0,0,:],color='r',linewidth=2,label='Upper lever, correct')
			ax.plot(time,data[p,1,1,:],color='b',linewidth=2,label='Lower lever, correct')
			ax.plot(time,data[p,0,1,:],color='b',linewidth=2,label='Lower lever, incorrect',
				linestyle='dashed')
			ax.plot(time,data[p,1,0,:],color='r',linewidth=2,label='Upper lever, incorrect',
				linestyle='dashed')
			##now get the significance masks (unless there isn't one)
			try:
				mask = sig_masks[condition][p]
				sigx = np.where(mask==True)[0]
				sigy = np.ones(sigx.size)*ax.get_ylim()[0]
				ax.plot(sigx,sigy,color='k',linewidth=2)
			except KeyError:
				pass
			##now plot the lines corresponding to the events
			plt.vlines(events,ax.get_ylim()[0],ax.get_ylim()[1],linestyle='dashed',
				color='k',alpha=0.5)	
			if c+1 == n_conditions:
				ax.set_xlabel('Time in trial, ms',fontsize=14)
				for tick in ax.xaxis.get_major_ticks():
					tick.label.set_fontsize(14)
				ax.locator_params(axis='x',tight=True,nbins=4)
			else: 
				ax.set_xticklabels([])
			if p == 0:
				ax.set_ylabel(title,fontsize=14)
			for tick in ax.yaxis.get_major_ticks():
				tick.label.set_fontsize(14)
			ax.locator_params(axis='y',tight=True,nbins=4)
			if c == 0:
				ax.set_title("Component "+str(p),fontsize=14)
	##now do a second plot that shows the variance explained by the combination
	##of all components
	fig,ax = plt.subplots(1)
	all_var = []
	for v in var_explained.keys():
		all_var.append(var_explained[v])
	all_var = np.cumsum(np.asarray(all_var).sum(axis=0)*100)
	ax.plot(np.arange(1,all_var.size+1),all_var,linewidth=2,marker='o')
	ax.set_ylabel("percent variance explained",fontsize=14)
	ax.set_ylabel("Number of components",fontsize=14)
	ax.set_title("Variance explained by dPCA",fontsize=14)
	for tick in ax.xaxis.get_major_ticks():
		tick.label.set_fontsize(14)
	for tick in ax.yaxis.get_major_ticks():
		tick.label.set_fontsize(14)
	ax.set_xlim(0,all_var.size+2)
	ax.set_ylim(0,100)
	plt.show()

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

"""
A helper function to calculate the mean and S.E.M. for a set of data
"""
def mean_and_sem(a):
	n = a.shape[0]
	m, se = np.mean(a,axis=0), scipy.stats.sem(a,axis=0)
	return m, m-se, m+se

"""
A helper function to get a column and row value for a square matrix
given an index referring to the total array size
Inputs:
	-idx: index to start with
	-side_len: length of the side of the square matrix
Returns:
	row, col transformation of the given index
"""
def rc_idx(idx,side_len):
	ids = int(idx)
	side_len = int(side_len)
	row = idx/side_len
	col = idx%side_len
	return int(row),int(col)

color_lut = {
	'action':['red','blue'],
	'context':['maroon','cyan'],
	'outcome':['gold','green']
}

