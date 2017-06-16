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
import tensortools as tt
from tensor_analysis import align_factors, _validate_factors
import model_fitting as mf
import file_lists

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
def plot_log_units_all(dir_list=None,session_range=None,sig_level=0.05,test_type='llr_pvals'):
	##coding this in just to save time
	if dir_list == None:
		dir_list = [
		r"D:\Ryan\DS_animals\results\LogisticRegression\80gauss_40ms_bins\S1",
		r"D:\Ryan\DS_animals\results\LogisticRegression\80gauss_40ms_bins\S2",
		r"D:\Ryan\DS_animals\results\LogisticRegression\80gauss_40ms_bins\S3"
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
		cplots.append(ax.imshow(img_mat,interpolation='none',cmap='YlOrRd'))
		# for j in range(n_units):
		# 	r,c = rc_idx(j,side_len)
		# 	ax.text(r,c,str(j+1),fontsize=4)
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
def plot_trials(f_behavior,save=False):
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
Plots model fits over days, comparing HMM and RL using log-liklihood
"""
def plot_model_fits(save=False):
	RL_fits, HMM_fits = fa.model_fits3()
	##add a break in the data to acocunt for surgeries
	surgery_break = np.zeros((RL_fits.shape[0],4))
	surgery_break[:] = np.nan
	pre_surgery_RL = RL_fits[:,0:8]
	post_surgery_RL = RL_fits[:,8:]
	pre_surgery_HMM = HMM_fits[:,0:8]
	post_surgery_HMM = HMM_fits[:,8:]
	RL = np.concatenate((pre_surgery_RL,surgery_break,post_surgery_RL),axis=1)
	HMM = np.concatenate((pre_surgery_HMM,surgery_break,post_surgery_HMM),axis=1)
	"""
	Start with the plot for RL
	"""
	data = RL
	##get the mean and std error
	mean = np.nanmean(RL,axis=0)
	err = np.nanstd(RL,axis=0)/np.sqrt(RL.shape[0])
	##the x-axis
	x = np.arange(1,data.shape[1]+1)
	##set up the plot
	fig,(ax,ax2) = plt.subplots(nrows=1,ncols=2,sharey=True)
	ax.errorbar(x,mean,linewidth=2,color='b',yerr=err,capthick=2)
	##also plot animals individually
	for i in range(data.shape[0]):
		ax.plot(x,data[i,:],color='b',linewidth=2,alpha=0.3)
	for tick in ax.xaxis.get_major_ticks():
		tick.label.set_fontsize(14)
	for tick in ax.yaxis.get_major_ticks():
		tick.label.set_fontsize(14)
	ax.set_xlabel("Training day",fontsize=14)
	ax.set_ylabel("Log-liklihood",fontsize=14)
	ax.set_title("Goodness-of-fit for\nRL models",fontsize=14)
	ax.text(8,ax.get_ylim()[0]+0.05,"Implant\nsurgery",fontsize=14,color='r')
	ax.plot(np.arange(8,13),np.ones(5)*ax.get_ylim()[0]+0.02,linewidth=4,color='r')
	##now compare first 3 and last 3 days
	first3 = data[:,0:4].mean(axis=1)
	last3 = data[:,-6:-2].mean(axis=1) ##I think one animal didn't do as many days
	means = np.array([first3.mean(),last3.mean()])
	sems = np.array([stats.sem(first3),stats.sem(last3)])
	#####
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
		fig.savefig(r"D:\Ryan\DS_animals\plots\RL_fits.png")
		fig.savefig(r"D:\Ryan\DS_animals\plots\RL_fits.svg")
	"""
	Now do the plot for HMM
	"""
	data = HMM
	##get the mean and std error
	mean = np.nanmean(HMM,axis=0)
	err = np.nanstd(HMM,axis=0)/np.sqrt(HMM.shape[0])
	##the x-axis
	x = np.arange(1,data.shape[1]+1)
	##set up the plot
	fig,(ax,ax2) = plt.subplots(nrows=1,ncols=2,sharey=True)
	ax.errorbar(x,mean,linewidth=2,color='g',yerr=err,capthick=2)
	##also plot animals individually
	for i in range(data.shape[0]):
		ax.plot(x,data[i,:],color='g',linewidth=2,alpha=0.3)
	for tick in ax.xaxis.get_major_ticks():
		tick.label.set_fontsize(14)
	for tick in ax.yaxis.get_major_ticks():
		tick.label.set_fontsize(14)
	ax.set_xlabel("Training day",fontsize=14)
	ax.set_ylabel("Log-liklihood",fontsize=14)
	ax.set_title("Goodness-of-fit for\nHidden Markov Models",fontsize=14)
	ax.text(8,ax.get_ylim()[0]+0.05,"Implant\nsurgery",fontsize=14,color='r')
	ax.plot(np.arange(8,13),np.ones(5)*ax.get_ylim()[0]+0.02,linewidth=4,color='r')
	##now compare first 3 and last 3 days
	first3 = data[:,0:4].mean(axis=1)
	last3 = data[:,-6:-2].mean(axis=1) ##I think one animal didn't do as many days
	means = np.array([first3.mean(),last3.mean()])
	sems = np.array([stats.sem(first3),stats.sem(last3)])
	#####
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
		fig.savefig(r"D:\Ryan\DS_animals\plots\HMM_fits.png")
		fig.savefig(r"D:\Ryan\DS_animals\plots\HMM_fits.svg")

"""
Plots model fits over days, comparing HMM and RL using the accuracy of 
action predictions
"""
def plot_model_fits2(win=[150,25],chunk=10):
	RL_fits,HMM_fits = fa.model_fits2(win=win)
	##not an equal number of trials for all animals; get rid of extra
	RL_fits = RL_fits[:,:-15]
	HMM_fits = HMM_fits[:,:-15]
	# lengths = []
	# for i in range(RL_fits.shape[0]):
	# 	dataseg = RL_fits[i,:]
	# 	lengths.append((dataseg[~np.isnan(dataseg)]).shape[0])
	# length = min(lengths)
	# RL_fits = RL_fits[:,:length]
	# HMM_fits = HMM_fits[:,:length]
	##first, plot the model fits over time
	mean_RL = np.nanmean(RL_fits,axis=0)
	sem_RL = np.nanstd(RL_fits,axis=0)/np.sqrt(RL_fits.shape[0])
	mean_HMM = np.nanmean(HMM_fits,axis=0)
	sem_HMM = np.nanstd(HMM_fits,axis=0)/np.sqrt(HMM_fits.shape[0])
	fig = plt.figure()
	ax1 = fig.add_subplot(121)
	x = np.arange(0,mean_RL.size*win[1],win[1])
	ax1.plot(x,mean_HMM,linewidth=2,color='k',label='HMM accuracy')
	ax1.plot(x,mean_RL,linewidth=2,color='b',label='RL_accuracy')
	ax1.fill_between(x,mean_HMM-sem_HMM,mean_HMM+sem_HMM,color='k',alpha=0.5)
	ax1.fill_between(x,mean_RL-sem_RL,mean_RL+sem_RL,color='b',alpha=0.5)
	ax1.set_xlabel("Trials",fontsize=14)
	ax1.set_ylabel("Model accuracy",fontsize=14)
	for tick in ax1.xaxis.get_major_ticks():
		tick.label.set_fontsize(14)
	for tick in ax1.yaxis.get_major_ticks():
		tick.label.set_fontsize(14)
	ax1.legend()
	##now plot the early-late comparisons
	ax2 = fig.add_subplot(122)
	lengths = []
	for i in range(RL_fits.shape[0]):
		dataseg = RL_fits[i,:]
		lengths.append((dataseg[~np.isnan(dataseg)]).shape[0])
	length = min(lengths)
	RL_fits = RL_fits[:,:length]
	HMM_fits = HMM_fits[:,:length]
	RL_early = RL_fits[:,0:chunk].mean(axis=1)
	RL_late = RL_fits[:,-chunk:].mean(axis=1)
	HMM_early = HMM_fits[:,0:chunk].mean(axis=1)
	HMM_late = HMM_fits[:,-chunk:].mean(axis=1)
	HMM_means = np.array([HMM_early.mean(),HMM_late.mean()])
	HMM_sems = np.array([stats.sem(HMM_early),stats.sem(HMM_late)])
	RL_means = np.array([RL_early.mean(),RL_late.mean()])
	RL_sems = np.array([stats.sem(RL_early),stats.sem(RL_late)])
	#####
	x = np.array([1,2])
	xerr = np.ones(2)*0.1
	for i in range(RL_early.size):
		ax2.plot(x,np.array([RL_early[i],RL_late[i]]),color='b',linewidth=2,marker='o',alpha=0.5)
	for i in range(HMM_early.size):
		ax2.plot(x,np.array([HMM_early[i],HMM_late[i]]),color='k',linewidth=2,marker='o',alpha=0.5)
	ax2.errorbar(x,HMM_means,yerr=HMM_sems,xerr=xerr,fmt='none',ecolor='k',capthick=2,elinewidth=2)
	ax2.errorbar(x,RL_means,yerr=RL_sems,xerr=xerr,fmt='none',ecolor='b',capthick=2,elinewidth=2)
	plt.xticks(x,['Early','Late'])
	for ticklabel in ax2.get_xticklabels():
		ticklabel.set_fontsize(14)
	for ticklabel in ax2.get_yticklabels():
		ticklabel.set_fontsize(14)
	ax2.set_xlabel("Epoch",fontsize=14)
	tval_RL,pval_RL = stats.ttest_rel(RL_early,RL_late)
	tval_HMM,pval_HMM = stats.ttest_rel(HMM_early,HMM_late)
	tval_both_early,pval_both_early = stats.ttest_ind(HMM_early,RL_early)
	tval_both_late,pval_both_late = stats.ttest_ind(HMM_late,RL_late)
	# ax2.text(1.5,0.42,"p={0:.4f}".format(pval))
	print("mean RL_early="+str(RL_early.mean()))
	print("mean RL_late="+str(RL_late.mean()))
	print("pval="+str(pval_RL))
	print("tval="+str(tval_RL))
	print("mean HMM_early="+str(HMM_early.mean()))
	print("mean HMM_late="+str(HMM_late.mean()))
	print("pval="+str(pval_HMM))
	print("tval="+str(tval_HMM))
	print("pval both early={}".format(pval_both_early))
	print("tval both early={}".format(tval_both_early))
	print("pval both late={}".format(pval_both_late))
	print("tval both late={}".format(tval_both_late))




"""
plots trial data from a model
"""
def plot_single_model_fit(results=None,f_behavior=None):
	if results is None:
		##get the model fits
		results = mf.fit_models(f_behavior)
	actions = results['actions']
	outcomes = results['outcomes']
	RL_actions = results['RL_actions']
	HMM_actions = results['HMM_actions']
	first_block = results['first_block']
	##different block start times
	block_starts = np.concatenate([np.array([0]),results['switch_times']])
	switch_times = results['switch_times']
	if results['first_block'] == 'lower_rewarded':
		colors = []
		for i in range(len(block_starts)):
			if i%2 == 0:
				colors.append('b')
			else:
				colors.append('r')
	elif results['first_block'] == 'upper_rewarded':
		colors = []
		for i in range(len(block_starts)):
			if i%2 == 0:
				colors.append('r')
			else:
				colors.append('b')
	##a sub-function to figure out which actions were rewarded
	def parse_actions(actions,outcomes,switch_times,first_block):
		conditions = {
		'upper_rewarded':[],
		'lower_rewarded':[]
		}
		current_block = first_block
		trial_idx = 0
		for t in range(len(switch_times-1)):
			##the trial numbers for this block
			trialnums = list(range(trial_idx,switch_times[t]))
			conditions[current_block]+=(trialnums)
			trial_idx = switch_times[t]
			current_block = [x for x in list(conditions) if not x == current_block][0]
		##the last block
		conditions[current_block]+=(list(range(trial_idx,len(actions))))
		##now parse the actions
		upper_rewarded = [] #correct
		upper_unrewarded = [] #correct
		upper_incorrect = [] #incorrect
		lower_rewarded = []
		lower_unrewarded = []
		lower_incorrect = []
		##start with lower lever actions
		for trial in np.where(actions==1)[0]:
			if trial in conditions['upper_rewarded']:
				lower_incorrect.append(trial)
			elif trial in conditions['lower_rewarded'] and outcomes[trial] == 1:
				lower_rewarded.append(trial)
			elif trial in conditions['lower_rewarded'] and outcomes[trial] == 0:
				lower_unrewarded.append(trial)
		for trial in np.where(actions==2)[0]:
			if trial in conditions['lower_rewarded']:
				upper_incorrect.append(trial)
			elif trial in conditions['upper_rewarded'] and outcomes[trial] == 1:
				upper_rewarded.append(trial)
			elif trial in conditions['upper_rewarded'] and outcomes[trial] == 0:
				upper_unrewarded.append(trial)
		return upper_rewarded,upper_unrewarded,upper_incorrect,lower_rewarded,lower_unrewarded,lower_incorrect
	##create the plot for HMM
	fig = plt.figure()
	ax = fig.add_subplot(211)
	ax2 = ax.twinx()
	##plot the block switches
	ax.vlines(block_starts,-0.2,1.2,linestyle='dashed',linewidth=2,colors=colors)
	##the belief state for lower lever
	ax2.plot(results['state_vals'][0,:],linewidth=2,color='k')
	##the actual actions
	upper_rewarded,upper_unrewarded,upper_incorrect,lower_rewarded,lower_unrewarded,lower_incorrect = parse_actions(actions,outcomes,switch_times,first_block)
	ax.plot(upper_rewarded,np.ones(len(upper_rewarded))*1.25+np.random.uniform(-0.04,0.04,size=len(upper_rewarded)),marker='o',color='r',linestyle='none')
	ax.plot(upper_unrewarded,np.ones(len(upper_unrewarded))*1.25+np.random.uniform(-0.04,0.04,size=len(upper_unrewarded)),marker='o',markerfacecolor='none',color='r',linestyle='none')
	ax.plot(upper_incorrect,np.ones(len(upper_incorrect))*1.25+np.random.uniform(-0.04,0.04,size=len(upper_incorrect)),linestyle='none',marker='+',color='r')
	ax.plot(lower_rewarded,np.zeros(len(lower_rewarded))+np.random.uniform(-0.04,0.04,size=len(lower_rewarded)),marker='o',color='b',linestyle='none')
	ax.plot(lower_unrewarded,np.zeros(len(lower_unrewarded))+np.random.uniform(-0.04,0.04,size=len(lower_unrewarded)),marker='o',markerfacecolor='none',color='b',linestyle='none')
	ax.plot(lower_incorrect,np.zeros(len(lower_incorrect))+np.random.uniform(-0.04,0.04,size=len(lower_incorrect)),linestyle='none',marker='+',color='b')
	##the model actions
	upper_rewarded,upper_unrewarded,upper_incorrect,lower_rewarded,lower_unrewarded,lower_incorrect = parse_actions(HMM_actions,outcomes,switch_times,first_block)
	ax.plot(upper_rewarded,np.ones(len(upper_rewarded))+np.random.uniform(-0.04,0.04,size=len(upper_rewarded)),marker='o',color='r',linestyle='none')
	ax.plot(upper_unrewarded,np.ones(len(upper_unrewarded))+np.random.uniform(-0.04,0.04,size=len(upper_unrewarded)),marker='o',markerfacecolor='none',color='r',linestyle='none')
	ax.plot(upper_incorrect,np.ones(len(upper_incorrect))+np.random.uniform(-0.04,0.04,size=len(upper_incorrect)),linestyle='none',marker='+',color='r')
	ax.plot(lower_rewarded,np.zeros(len(lower_rewarded))-0.25+np.random.uniform(-0.04,0.04,size=len(lower_rewarded)),marker='o',color='b',linestyle='none')
	ax.plot(lower_unrewarded,np.zeros(len(lower_unrewarded))-0.25+np.random.uniform(-0.04,0.04,size=len(lower_unrewarded)),marker='o',markerfacecolor='none',color='b',linestyle='none')
	ax.plot(lower_incorrect,np.zeros(len(lower_incorrect))-0.25+np.random.uniform(-0.04,0.04,size=len(lower_incorrect)),linestyle='none',marker='+',color='b')
	ax.set_yticks([-0.25,0,1,1.25])
	ax.set_yticklabels(['Predicted','Actual','Predicted','Actual'])
	ax2.set_ylabel("Belief = lower_rewarded")
	ax.set_title("Hidden Markov model fit",fontsize=14)
	ax.set_xticks([])
	ax.text(10,0.35,"Log-liklihood:{0:.3f}".format(results['ll_HMM']))
	##create the plot for RL
	ax = fig.add_subplot(212)
	ax2 = ax.twinx()
	##plot the block switches
	ax.vlines(block_starts,-0.2,1.2,linestyle='dashed',linewidth=2,colors=colors)
	##the belief state for lower lever
	ax2.plot(results['Qvals'][0,:],linewidth=2,color='b')
	ax2.plot(results['Qvals'][1,:],linewidth=2,color='r')
	##the actual actions
	upper_rewarded,upper_unrewarded,upper_incorrect,lower_rewarded,lower_unrewarded,lower_incorrect = parse_actions(actions,outcomes,switch_times,first_block)
	ax.plot(upper_rewarded,np.ones(len(upper_rewarded))*1.25+np.random.uniform(-0.04,0.04,size=len(upper_rewarded)),marker='o',color='r',linestyle='none')
	ax.plot(upper_unrewarded,np.ones(len(upper_unrewarded))*1.25+np.random.uniform(-0.04,0.04,size=len(upper_unrewarded)),marker='o',markerfacecolor='none',color='r',linestyle='none')
	ax.plot(upper_incorrect,np.ones(len(upper_incorrect))*1.25+np.random.uniform(-0.04,0.04,size=len(upper_incorrect)),linestyle='none',marker='+',color='r')
	ax.plot(lower_rewarded,np.zeros(len(lower_rewarded))+np.random.uniform(-0.04,0.04,size=len(lower_rewarded)),marker='o',color='b',linestyle='none')
	ax.plot(lower_unrewarded,np.zeros(len(lower_unrewarded))+np.random.uniform(-0.04,0.04,size=len(lower_unrewarded)),marker='o',markerfacecolor='none',color='b',linestyle='none')
	ax.plot(lower_incorrect,np.zeros(len(lower_incorrect))+np.random.uniform(-0.04,0.04,size=len(lower_incorrect)),linestyle='none',marker='+',color='b')
	##the model actions
	upper_rewarded,upper_unrewarded,upper_incorrect,lower_rewarded,lower_unrewarded,lower_incorrect = parse_actions(RL_actions,outcomes,switch_times,first_block)
	ax.plot(upper_rewarded,np.ones(len(upper_rewarded))+np.random.uniform(-0.04,0.04,size=len(upper_rewarded)),marker='o',color='r',linestyle='none')
	ax.plot(upper_unrewarded,np.ones(len(upper_unrewarded))+np.random.uniform(-0.04,0.04,size=len(upper_unrewarded)),marker='o',markerfacecolor='none',color='r',linestyle='none')
	ax.plot(upper_incorrect,np.ones(len(upper_incorrect))+np.random.uniform(-0.04,0.04,size=len(upper_incorrect)),linestyle='none',marker='+',color='r')
	ax.plot(lower_rewarded,np.zeros(len(lower_rewarded))-0.25+np.random.uniform(-0.04,0.04,size=len(lower_rewarded)),marker='o',color='b',linestyle='none')
	ax.plot(lower_unrewarded,np.zeros(len(lower_unrewarded))-0.25+np.random.uniform(-0.04,0.04,size=len(lower_unrewarded)),marker='o',markerfacecolor='none',color='b',linestyle='none')
	ax.plot(lower_incorrect,np.zeros(len(lower_incorrect))-0.25+np.random.uniform(-0.04,0.04,size=len(lower_incorrect)),linestyle='none',marker='+',color='b')
	ax.set_yticks([-0.25,0,1,1.25])
	ax.set_yticklabels(['Predicted','Actual','Predicted','Actual'])
	ax.set_xlabel("Trials",fontsize=14)
	ax2.set_ylabel("Action values")
	ax2.set_ylim(-2,2)
	ax.set_title("Q-learning model fit",fontsize=14)
	ax.text(10,0.35,"Log-liklihood:{0:.3f}".format(results['ll_RL']))


"""
Plots the distributions of trial durations early to late, as well as overall
"""
def plot_trial_durations(early_range=[0,5],late_range=[13,20],max_duration=30*1000):
	##start by getting the data
	early_durations = fa.get_trial_durations(max_duration=max_duration,session_range=early_range)/1000.0
	late_durations = fa.get_trial_durations(max_duration=max_duration,session_range=late_range)/1000.0
	all_durations = fa.get_trial_durations(max_duration=max_duration,session_range=None)/1000.0
	fig = plt.figure()
	gs = gridspec.GridSpec(2,2)
	ax = fig.add_subplot(gs[0,:])
	ax2 = fig.add_subplot(gs[1,0])
	ax3 = fig.add_subplot(gs[1,1], sharey=ax2)
	##now plot the histograms
	n,bins,patches=ax.hist(all_durations,50,normed=1,facecolor='green',alpha=0.75,edgecolor='green',linewidth=3)
	##add a 'best_fit' line
	y = matplotlib.mlab.normpdf(bins,all_durations.mean(),all_durations.std())
	l = ax.plot(bins,y,'k--',linewidth=1)
	ax.set_title("All sessions",fontsize=14,weight='bold')
	ax.set_xlabel("Trial duration (s)",fontsize=14)
	ax.set_ylabel("Probability",fontsize=14)
	for tick in ax.xaxis.get_major_ticks():
		tick.label.set_fontsize(14)
	for tick in ax.yaxis.get_major_ticks():
		tick.label.set_fontsize(14)
	ax.text(5,0.3,"mean={0:.2f}".format(all_durations.mean()))
	##for early sessions
	n,bins2,patches=ax2.hist(early_durations,bins=bins,normed=1,facecolor='cyan',alpha=0.75,edgecolor='cyan',linewidth=3)
	##add a 'best_fit' line
	y = matplotlib.mlab.normpdf(bins,early_durations.mean(),early_durations.std())
	l = ax2.plot(bins,y,'k--',linewidth=1)
	ax2.set_title("Early sessions",fontsize=14,weight='bold')
	ax2.set_xlabel("Trial duration (s)",fontsize=14)
	ax2.set_ylabel("Probability",fontsize=14)
	for tick in ax2.xaxis.get_major_ticks():
		tick.label.set_fontsize(14)
	for tick in ax2.yaxis.get_major_ticks():
		tick.label.set_fontsize(14)
	ax2.text(5,0.3,"mean={0:.2f}".format(early_durations.mean()))
	##for late sessions
	n,bins3,patches=ax3.hist(late_durations,bins=bins,normed=1,facecolor='blue',alpha=0.75,edgecolor='blue',linewidth=3)
	##add a 'best_fit' line
	y = matplotlib.mlab.normpdf(bins,late_durations.mean(),late_durations.std())
	l = ax3.plot(bins,y,'k--',linewidth=1)
	ax3.set_title("Late sessions",fontsize=14,weight='bold')
	ax3.set_xlabel("Trial duration (s)",fontsize=14)
	ax3.set_ylabel("Probability",fontsize=14)
	for tick in ax3.xaxis.get_major_ticks():
		tick.label.set_fontsize(14)
	for tick in ax3.yaxis.get_major_ticks():
		tick.label.set_fontsize(14)
	ax3.text(5,0.3,"mean={0:.2f}".format(late_durations.mean()))
	fig.suptitle("Distibution of trial durations",fontsize=14)
	plt.tight_layout()


"""
A function to plot decision variable stuff
Inputs:
	-window [pre_event, post_event] window, in ms
	-smooth_method: type of smoothing to use; choose 'bins', 'gauss', 'both', or 'none'
	-smooth_width: size of the bins or gaussian kernel in ms. If 'both', input should be a list
		with index 0 being the gaussian width and index 1 being the bin width
	-max_duration: maximum allowable trial duration, in ms
	-min_rate: minimum acceptable spike rate, in Hz
	-z_score: bool
	-trial_duration: if you want to scale all trials to the same length
	-state_thresh: percentage for state estimation to split trials into weak or strong belief
		states. 0.1 = top 10% of weak or strong trials are put into this catagory
"""
def plot_decision_vars(pad=[2000,120],smooth_method='both',smooth_width=[100,40],
	max_duration=4000,min_rate=0.1,z_score=True,trial_duration=None,state_thresh=0.1):
	results = {}
	upper_odds = []
	lower_odds = []
	upper_strong_odds = []
	lower_strong_odds = []
	upper_weak_odds = []
	lower_weak_odds = []
	for animal in file_lists.animals:
		data = fa.decision_vars(animal,pad=pad,smooth_method=smooth_method,smooth_width=smooth_width,
			max_duration=max_duration,min_rate=min_rate,z_score=z_score,trial_duration=trial_duration,
			state_thresh=state_thresh)
		upper_odds.append(data['upper_odds'])
		lower_odds.append(data['lower_odds'])
		upper_strong_odds.append(data['upper_strong_odds'])
		lower_strong_odds.append(data['lower_strong_odds'])
		upper_weak_odds.append(data['upper_weak_odds'])
		lower_weak_odds.append(data['lower_weak_odds'])
	results['upper_odds'] = np.concatenate(upper_odds,axis=0)
	results['upper_strong_odds'] = np.concatenate(upper_strong_odds,axis=0)
	results['upper_weak_odds'] = np.concatenate(upper_weak_odds,axis=0)
	results['lower_odds'] = np.concatenate(lower_odds,axis=0)
	results['lower_strong_odds'] = np.concatenate(lower_strong_odds,axis=0)
	results['lower_weak_odds'] = np.concatenate(lower_weak_odds,axis=0)
	##create a figure
	fig,ax = plt.subplots(1)
	fig2,ax2 = plt.subplots(1)
	##now work on each thing that we want to plot, one at a time
	##starting with lower
	lower_mean = results['lower_odds'].mean(axis=0)
	lower_sem = stats.sem(results['lower_odds'],axis=0)
	lower_strong_mean = results['lower_strong_odds'].mean(axis=0)
	lower_strong_sem = stats.sem(results['lower_strong_odds'],axis=0)
	lower_weak_mean = results['lower_weak_odds'].mean(axis=0)
	lower_weak_sem = stats.sem(results['lower_weak_odds'],axis=0)
	##now repeat for upper
	upper_mean = results['upper_odds'].mean(axis=0)
	upper_sem = stats.sem(results['upper_odds'],axis=0)
	upper_strong_mean = results['upper_strong_odds'].mean(axis=0)
	upper_strong_sem = stats.sem(results['upper_strong_odds'],axis=0)
	upper_weak_mean = results['upper_weak_odds'].mean(axis=0)-0.8
	upper_weak_sem = stats.sem(results['upper_weak_odds'],axis=0)-0.8
	##now plot
	x = np.linspace(-pad[0]/1000,0,lower_mean.shape[0])
	ax2.plot(x,lower_mean,color='b',linewidth=2,linestyle='--',label='all lower')
	ax2.fill_between(x,lower_mean-lower_sem,lower_mean+lower_sem,color='b',alpha=0.5)
	ax.plot(x,lower_strong_mean,color='b',linewidth=2,label='strong lower')
	ax.fill_between(x,lower_strong_mean-lower_strong_sem,lower_strong_mean+lower_strong_sem,
		color='b',alpha=0.5)
	ax.plot(x,lower_weak_mean,color='b',linewidth=2,linestyle=':',label='weak_lower')
	ax.fill_between(x,lower_weak_mean-lower_weak_sem,lower_weak_mean+lower_weak_sem,
		color='b',alpha=0.5)
	##upper
	ax2.plot(x,upper_mean,color='r',linewidth=2,linestyle='--',label='all upper')
	ax2.fill_between(x,upper_mean-upper_sem,upper_mean+upper_sem,color='r',alpha=0.5)
	ax.plot(x,upper_strong_mean,color='r',linewidth=2,label='strong upper')
	ax.fill_between(x,upper_strong_mean-upper_strong_sem,upper_strong_mean+upper_strong_sem,
		color='r',alpha=0.5)
	ax.plot(x,upper_weak_mean,color='r',linewidth=2,linestyle=':',label='weak_upper')
	ax.fill_between(x,upper_weak_mean-upper_weak_sem,upper_weak_mean+upper_weak_sem,
		color='r',alpha=0.5)
	ax.set_xlabel('Time to lever press (s)',fontsize=14)
	ax.set_ylabel('Log odds',fontsize=14)
	ax.legend()
	ax2.set_xlabel('Time to lever press (s)',fontsize=14)
	ax2.set_ylabel('Log odds',fontsize=14)
	ax2.legend()
	print("n upper strong = {}".format(results['upper_strong_odds'].shape[0]))
	print("n upper weak = {}".format(results['upper_weak_odds'].shape[0]))
	print("n lower strong = {}".format(results['lower_strong_odds'].shape[0]))
	print("n lower weak = {}".format(results['lower_weak_odds'].shape[0]))
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
def plot_dpca_results(Z,var_explained,conditions,bin_size,pad=None,n_components=3):	
	##set up the figure
	fig = plt.figure()
	##add time and interaction as a conditions
	conditions = ['time']+conditions+['interaction']
	n_conditions = len(conditions)
	c_idx = ['t',conditions[1][0]+'t',conditions[2][0]+'t',conditions[1][0]+conditions[2][0]+'t'] ##the letters used to index the conditions in Z
	##get scaled time axis
	time = np.linspace(0,bin_size*Z['t'].shape[-1],Z['t'].shape[-1])
	##we'll plot the first n dPC's for each condition
	axes = []
	for c in range(len(conditions)):
		condition = conditions[c]
		data = Z[c_idx[c]]
		for p in range(n_components):
			plotnum = (c*n_components+p)+1
			ax = fig.add_subplot(n_conditions,n_components,plotnum)
			##plot both kinds of trials for each condition
			ax.plot(time,data[p,0,0,:],color='r',linewidth=2,
				label=dpca.condition_pairs[conditions[1]][0]+",\n"+dpca.condition_pairs[conditions[2]][0])
			ax.plot(time,data[p,0,1,:],color='r',linewidth=2,
				label=dpca.condition_pairs[conditions[1]][0]+",\n"+dpca.condition_pairs[conditions[2]][1],
				linestyle='dashed')
			##now for the second condition
			ax.plot(time,data[p,1,0,:],color='b',linewidth=2,
				label=dpca.condition_pairs[conditions[1]][1]+",\n"+dpca.condition_pairs[conditions[2]][0])
			ax.plot(time,data[p,1,1,:],color='b',linewidth=2,
				label=dpca.condition_pairs[conditions[1]][1]+",\n"+dpca.condition_pairs[conditions[2]][1],
			linestyle='dashed')
			##now get the significance masks (unless there isn't one, like for time)
			# try:
			# 	mask = sig_masks[c_idx[c]][p]
			# 	sigx = np.where(mask==True)[0]
			# 	sigy = np.ones(sigx.size)*ax.get_ylim()[0]
			# 	ax.plot(sigx,sigy,color='k',linewidth=2)
			# except KeyError:
			# 	pass
			##now plot the lines corresponding to the events, if requested
			if pad is not None:
				plt.vlines(np.array([pad[0],time.max()-pad[1]]),ax.get_ylim()[0],ax.get_ylim()[1],
					linestyle='dashed',color='k',alpha=0.5)
			if c+1 == n_conditions:
				ax.set_xlabel('Time in trial, ms',fontsize=14)
				for tick in ax.xaxis.get_major_ticks():
					tick.label.set_fontsize(14)
				ax.locator_params(axis='x',tight=True,nbins=4)
			else: 
				ax.set_xticklabels([])
			if p == 0:
				ax.set_ylabel(condition,fontsize=14)
				for tick in ax.yaxis.get_major_ticks():
					tick.label.set_fontsize(14)
			else:
				ax.set_yticklabels([])
			ax.locator_params(axis='y',tight=True,nbins=4)
			if c == 0:
				ax.set_title("Component "+str(p),fontsize=14)
			axes.append(ax)
			ax.set_ylim(-5.5,7.5)
	axes[n_components-1].legend(bbox_to_anchor=(1,1))
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
	##do a third plot that shows the breakdown of variance explained in 
	##a pie chart
	sizes = np.zeros(len(c_idx))
	for n,i in enumerate(c_idx):
		sizes[n] = sum(var_explained[i])
	sizes = sizes/sizes.sum()
	fig,ax = plt.subplots()
	ax.pie(sizes,labels=conditions,autopct='%1.1f%%',startangle=90)
	ax.axis('equal')
	plt.show()

	
	"""Plots a KTensor.

	Each parameter can be passed as a list if different formatting is
	desired for each set of factors. For example, if `X` is a 3rd-order
	tensor (i.e. `X.ndim == 3`) then `X.plot(color=['r','k','b'])` plots
	all factors for the first mode in red, the second in black, and the
	third in blue. On the other hand, `X.plot(color='r')` produces red
	plots for each mode.

	Parameters
	----------
	plots : str or list
		One of {'bar','line'} to specify the type of plot for each factor.
		The default is 'line'.
	color : matplotlib color or list
		Color for plots associated with each set of factors
	lw : int or list
		Specifies line width on plots. Default is 2
	ylim : str, y-axis limits or list
		Specifies how to set the y-axis limits for each mode of the
		decomposition. For a third-order, rank-2 model, setting
		ylim=['link', (0,1), ((0,1), (-1,1))] specifies that the
		first pair of factors have the same y-axis limits (chosen
		automatically), the second pair of factors both have y-limits
		(0,1), and the third pair of factors have y-limits (0,1) and
		(-1,1).
	"""
def plot_tensors(factors,trial_data,n_factors=4,plots=['bar','line','scatter'],ylim='link',
	yticks=True,width_ratios=None,scatter_kw=dict(),line_kw=dict(),bar_kw=dict(),
	titles=['Estimated\nneuron factors', 'Estimated\ntime factors', 'Estimated\ntrial factors']):
	
	trial_info = ptr.parse_trial_data(trial_data)
	factors, ndim, rank = _validate_factors(factors)
	rank = n_factors
	figsize = (8, rank)

	# helper function for parsing plot options
	def _broadcast_arg(arg, argtype, name):
		"""Broadcasts plotting option `arg` to all factors
		"""
		if arg is None or isinstance(arg, argtype):
			return [arg for _ in range(ndim)]
		elif isinstance(arg, list):
			return arg
		else:
			raise ValueError('Parameter %s must be a %s or a list'
							 'of %s' % (name, argtype, argtype))

	# parse optional inputs
	plots = _broadcast_arg(plots, str, 'plots')
	ylim = _broadcast_arg(ylim, (tuple, str), 'ylim')
	bar_kw = _broadcast_arg(bar_kw, dict, 'bar_kw')
	line_kw = _broadcast_arg(line_kw, dict, 'line_kw')
	scatter_kw = _broadcast_arg(scatter_kw, dict, 'scatter_kw')

	# parse plot widths, defaults to equal widths
	if width_ratios is None:
		width_ratios = [1 for _ in range(ndim)]

	# default scatterplot options
	for sckw in scatter_kw:
		if not "edgecolor" in sckw.keys():
			sckw["edgecolor"] = "none"
		if not "s" in sckw.keys():
			sckw["s"] = 10

	#setup figure
	fig, axes = plt.subplots(rank, ndim,
						   figsize=figsize,
						   gridspec_kw=dict(width_ratios=width_ratios))
	if rank == 1: axes = axes[None, :]

	# main loop, plot each factor
	plot_obj = np.empty((rank, ndim), dtype=object)
	for r in range(rank):
		for i, f in enumerate(factors):

			# determine type of plot
			if plots[i] == 'bar':
				plot_obj[r,i] = axes[r,i].bar(np.arange(1, f.shape[0]+1), f[:,r], **bar_kw[i])
				axes[r,i].set_xlim(0, f.shape[0]+1)
			elif plots[i] == 'scatter':
				plot_obj[r,i] = axes[r,i]
				for trial_type in [x for x in list(trial_info) if not (x =='n_blocks' or x=='block_lengths')]:
					x = trial_info[trial_type]
					y = f[x,r]
					marker,color,facecolor = get_line_props(trial_type)
					plot_obj[r,i].scatter(x,y,color=color,marker=marker,facecolor=facecolor,label=trial_type)
				axes[r,i].set_xlim(0, f.shape[0])
			elif plots[i] == 'line':
				plot_obj[r,i] = axes[r,i].plot(f[:,r], '-', **line_kw[i])
				axes[r,i].set_xlim(0, f.shape[0])
			else:
				raise ValueError('invalid plot type')

			# format axes
			axes[r,i].locator_params(nbins=4)
			axes[r,i].spines['top'].set_visible(False)
			axes[r,i].spines['right'].set_visible(False)
			axes[r,i].xaxis.set_tick_params(direction='out')
			axes[r,i].yaxis.set_tick_params(direction='out')
			axes[r,i].yaxis.set_ticks_position('left')
			axes[r,i].xaxis.set_ticks_position('bottom')
			##set the title 
			if r == 0:
				axes[r,i].set_title(titles[i])

			# remove xticks on all but bottom row
			if r != rank-1:
				plt.setp(axes[r,i].get_xticklabels(), visible=False)
	axes[0,2].legend(bbox_to_anchor=(1,1))
	# link y-axes within columns
	for i, yl in enumerate(ylim):
		if yl is None:
			continue
		elif yl == 'link':
			yl = [a.get_ylim() for a in axes[:,i]]
			y0, y1 = min([y[0] for y in yl]), max([y[1] for y in yl])
			[a.set_ylim((y0, y1)) for a in axes[:,i]]
		elif yl == 'tight':
			[a.set_ylim(np.min(factors[i][:,r])*0.75, np.max(factors[i][:,r])*1.1)  for r, a in enumerate(axes[:,i])]
		elif isinstance(yl[0], (int, float)) and len(yl) == 2:
			[a.set_ylim(yl) for a in axes[:,i]]
		elif isinstance(yl[0], (tuple, list)) and len(yl) == rank:
			[a.set_ylim(lims) for a, lims in zip(axes[:,i], yl)]
		else:
			raise ValueError('ylimits not properly specified')

	# format y-ticks
	for r in range(rank):
		for i in range(ndim):
			if not yticks:
				axes[r,i].set_yticks([])
			else:
				# only two labels
				ymin, ymax = np.round(axes[r,i].get_ylim(), 2)
				axes[r,i].set_ylim((ymin, ymax))

				# remove decimals from labels
				if ymin.is_integer():
					ymin = int(ymin)
				if ymax.is_integer():
					ymax = int(ymax)

				# update plot
				axes[r,i].set_yticks([ymin, ymax])
				axes[r,i].set_yticklabels([str(ymin), str(ymax)])

	plt.tight_layout()

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

"""
A function to convert a saved dPCA file to a dictionary for plotting etc
"""
def convert_dpca_file(f_in):
	f = h5py.File(f_in,'r')
	Z = {}
	for key in list(f['Z']):
		Z[key] = np.asarray(f['Z'][key])
	var_explained = {}
	for key in list(f['var_explained']):
		var_explained[key] = np.asarray(f['var_explained'][key])
		sig_masks = {}
	for key in list(f['sig_masks']):
		sig_masks[key] = np.asarray(f['sig_masks'][key])
	f.close()
	return Z, var_explained, sig_masks

color_lut = {
	'action':['red','blue'],
	'context':['maroon','cyan'],
	'outcome':['gold','green']
}

##helper function for tensor plots
def get_line_props(trial_type):
	if trial_type == 'upper_correct_rewarded':
		marker='o'; facecolor='r'; color='r'
	elif trial_type == 'upper_correct_unrewarded':
		marker='o'; facecolor='none'; color='r'
	elif trial_type == 'upper_incorrect':
		marker = '+'; facecolor = 'r'; color = 'r'
	elif trial_type == 'lower_correct_rewarded':
		marker='o'; facecolor='b'; color='b'
	elif trial_type == 'lower_correct_unrewarded':
		marker='o'; facecolor='none'; color='b'
	elif trial_type == 'lower_incorrect':
		marker='+'; facecolor='b'; color='b'
	else:
		print("Unknown trial type: {}".format(trial_type))
	return marker,color,facecolor

"""
sometimes we 'equalize' array lengths by adding nans to 
uneven length arrs. This can be a problem if we want to 
run stats on the last x-values. So this function takes care of
that, by only considering the last x non-nan values 
along the second axis.
Inputs:
	arr: equalized array, with axis 0 the subject axis
	n_vals: the last n non-nan vals to take
Returns:
	result: an p-subject by n_vals array
"""
def last_vals(arr,n_vals):
	results = np.zeros((arr.shape[0],n_vals))
	for p in range(arr.shape[0]):
		dataseg = arr[p,:]
		dataseg = dataseg[~np.isnan(dataseg)]
		results[p,:] = dataseg[-n_vals:]
	return results