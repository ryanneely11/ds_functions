###dpca.py
##functions for implementing dPCA, using the library from Kobak et al, 2016

import numpy as np
from dPCA import dPCA
import session_analysis as sa

##some global variables that are specific to my current project
condition_pairs = [
('upper_rewarded','lower_rewarded'),
('upper_lever','lower_lever')]

conditions = [
'block_type',
'choice']

condition_LUT = collections.ordereddict()
condition_LUT['t'] = "Condition-independent component"
condition_LUT['bt'] = "Blocktype-dependent component",
condition_LUT['ct'] = "Choice-dependent component"
condition_LUT['cbt'] = "Interaction component"

###TODO: get datasets for everything, combined


"""
A function to get a dataset in the correct format to perform dPCA on it.
Right now this function is designed to just work on one session at a time;
based on the eLife paper though we may be able to expand to looking at many
datasets over many animals.
Inputs:
	f_behavior: path to the behavior timestamps
	f_ephys: path to the spike data
	smooth_method: type of smoothing to use; choose 'bins', 'gauss', 'both', or 'none'
	smooth_width: size of the bins or gaussian kernel in ms. If 'both', input should be a list
		with index 0 being the gaussian width and index 1 being the bin width
	pad: a window for pre- and post-trial padding, in ms. In other words, an x-ms period of time 
		before lever press to consider the start of the trial, and an x-ms period of time after
		reward to consider the end of the trial
	z_score: if True, z-scores the array
	timestretch: if True, uses the time stretching function (below) to equate the lengths of all trials.
Returns:
	X_mean: data list of shape n-neurons x condition-1 x condition-2, ... x n-timebins
	X_trials: data from individual trials: n-trials x n-neurons x condition-1, condition-2, ... x n-timebins.
		If data is unbalanced (ie different #'s of trials per condition), max dimensions are used and empty
		spaces are filled with NaN
	conditions: list of the names of each condition, in order according to the matrix.
"""

def get_dataset(f_behavior,f_ephys,smooth_method='both',smooth_width=[40,50],
	pad=[200,200],z_score=True,remove_unrew=True):
	##use global parameters here
	global condition_pairs
	###most of the work is done by our other function. Get the data separated from all
	##trials, and the indices of trials from each condition:
	X_all,ts_idx = sa.split_trials(f_behavior,f_ephys,smooth_method=smooth_method,
		smooth_width=smooth_width,pad=pad,z_score=z_score,
		timestretch=True,remove_unrew=remove_unrew)
	##now the data is in dims trials x neurons x time
	##get some meta about this data:
	n_neurons = X_all.shape[1]
	n_bins = X_all.shape[2]
	##the max number of trials in any condition pair
	n_trials = 0
	for i in range(len(ts_idx.keys())):
		if ts_idx.values()[i].shape[0] > n_trials:
			n_trials = ts_idx.values()[i].shape[0]
	##container for the whole thing
	condition_sizes = []
	for i in range(len(condition_pairs)):
		condition_sizes.append(len(condition_pairs[i]))
	datashape = [n_trials,n_neurons]+condition_sizes+[n_bins]
	X_trials = np.zeros(datashape)
	##construct the data arrays for each condition independently
	for i in range(len(condition_pairs)):
		##make a container for this dataset. It will be shape:
		##trials x neurons x groups in this condition x timebins
		dset = np.empty((len(condition_pairs[i]),n_trials,n_neurons,n_bins))
		dset[:] = np.nan
		for p in range(len(condition_pairs[i])):
			key = condition_pairs[i][p]
			idx = ts_idx[key]
			data = X_all[idx,:,:]
			dset[p,0:data.shape[0],:,:] = data
		##now add this to the master container. Things get complicated, because
		##we need to iterate over the middle axes for an arbitrary number of axes...
		##****THE LINE BELOW MUST BE CHANGED FOR A DIFF NUM OF CONDITIONS****
		###Don't know how to iterate the None,None, axes in the dset...
		np.rollaxis(X_trials,i+2)[:,:,:,:,:] += dset[:,:,:,None,:] 
	##Now to get X_mean (average over trials) we just take the average...
	X_mean = np.nanmean(X_trials,axis=0)
	return X_mean,X_trials

"""
This function actually runs dpca, relying on some globals for 
the details. ***NOTE: there are some hard-to-avoid elements here that are
			hardcoded specificallyfor this dataset.*****
Inputs:
	X_mean: array of data averaged over trials
	X_trials: array of data including individual trial data
"""
def session_dpca(f_behavior,f_ephys,smooth_method='both',smooth_width=[40,50],
	pad=[200,200],z_score=True,n_components=10,remove_unrew=True):
###########
##HARDCODED ELEMENTS-CHANGE FOR DIFFERENT EXPERIMENT PARAMS
##########
	labels = 'cbt'
	join = {'ct':['c','ct'],'bt':['b','bt'],'cbt':['cb','cbt']}
##########
##END HARDCODED
#########
	##initialize the dpca object
	dpca = dPCA.dPCA(labels=labels,join=join,n_components=n_components,
		regularizer='auto')
	dpca.protect = ['t']
	##get the data
	X_mean,X_trials = get_dataset(f_behavior,f_ephys,smooth_method=smooth_method,
		smooth_width=smooth_width,pad=pad,z_score=z_score,remove_unrew=remove_unrew)
	##if we didn't z-score the data, we need to at least center it
	if not z_score:
			###hardcoded bit here too
			unit_mean = X_mean.reshape((X_mean.shape[0],-1)).mean(axis=1)[:,None,None,None]
			X_mean -= unit_mean
	Z = dpca.fit_transform(X_mean,X_trials)
	##get the time axis
	if smooth_method == 'both':
		time = np.linspace(-1*pad[0],smooth_width[1]*X_mean.shape[-1]-pad[0],X_mean.shape[-1])
	elif smooth_method == 'bins':
		time = np.linspace(-1*pad[0],smooth_width*X_mean.shape[-1]-pad[0],X_mean.shape[-1])	
	else:
		time = np.arange(0,X_mean.shape[-1])
	##Next, get the variance explained:
	var_explained = dpca.explained_variance_ratio_
	##finally, get the significance masks (places where the demixed components are significant)
	sig_masks = dpca.significance_analysis(X_mean,X_trials,axis='t',
		n_shuffles=100,n_splits=10,n_consecutive=2)
	return Z,time,var_explained,sig_masks	



