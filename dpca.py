###dpca.py
##functions for implementing dPCA, using the library from Kobak et al, 2016

import numpy as np
from dPCA import dPCA
import parse_timestamps as pt
import parse_ephys as pe
import collections
from scipy import interpolate
from scipy.stats import zscore
import parse_trials as ptr
import model_fitting as mf
from itertools import chain, combinations

###TODO: get datasets for everything, combined
condition_pairs = {
	'context':['upper_rewarded','lower_rewarded'],
	'action':['upper_lever','lower_lever'],
	'outcome':['rewarded_poke','unrewarded_poke'],
	'u_level':['low','med','high']
}

"""
A function to get a dataset in the correct format to perform dPCA on it.
Right now this function is designed to just work on one session at a time;
based on the eLife paper though we may be able to expand to looking at many
datasets over many animals.
Inputs:
	f_behavior: file path to behavior data
	f_ephys: file path to ephys data
	conditions: should be two of the following: 'context', action', or 'outcome'. Can't be all three because
		there is no such thing as a rewarded trial with upper lever contex and lower lever action.
	smooth_method: type of smoothing to use; choose 'bins', 'gauss', 'both', or 'none'
	smooth_width: size of the bins or gaussian kernel in ms. If 'both', input should be a list
		with index 0 being the gaussian width and index 1 being the bin width
	pad: a window for pre- and post-trial padding, in ms. In other words, an x-ms period of time 
		before lever press to consider the start of the trial, and an x-ms period of time after
		reward to consider the end of the trial. For best results, should be a multiple of the bin size
	z_score: if True, z-scores the array
	trial_duration: specifies the trial length (in ms) to squeeze trials into. If None, the function uses
		the median trial length over the trials in the file
	min_rate: the min spike rate, in Hz, to accept. Units below this value will be removed.
	max_duration: maximum allowable trial duration (ms)
Returns:
	X_c: data from individual trials: n-trials x n-neurons x condition-1, condition-2, ... x n-timebins.
"""

def get_dataset(f_behavior,f_ephys,conditions,smooth_method='both',smooth_width=[80,40],pad=[400,400],
	z_score=True,trial_duration=None,max_duration=5000,min_rate=0.1,balance=True):
	global condition_pairs
	##get the spike dataset, and the trial info
	X,trial_data = ptr.get_trial_spikes(f_behavior=f_behavior,f_ephys=f_ephys,smooth_method=smooth_method,
		smooth_width=smooth_width,pad=pad,z_score=z_score,trial_duration=trial_duration,
		max_duration=max_duration,min_rate=min_rate)
	##get some metadata about this session
	n_units = X.shape[1]
	n_bins = X.shape[2]
	if balance:
		trial_index,n_trials = balance_trials(trial_data,conditions)
	else:
		trial_index,n_trials = unbalance_trials(trial_data,conditions)
	trial_types = list(trial_index)
	##allocate space for the dataset
	if n_trials > 0:
		X_c = np.empty((n_trials,n_units,len(condition_pairs[conditions[0]]),
			len(condition_pairs[conditions[1]]),n_bins))
		X_c[:] = np.nan
		for t in trial_index.keys():
			##based on the key, figure out where these trials should be placed in the dataset
			##I **think** that we should always expect the context[0] trial type to be the first part of the string
			c1_type = t[:t.index('+')]
			c2_type = t[t.index('+')+1:]
			c1_idx = condition_pairs[conditions[0]].index(c1_type)
			c2_idx = condition_pairs[conditions[1]].index(c2_type)
			##now add the data to the dataset using these indices
			for i,j in enumerate(trial_index[t]):
				X_c[i,:,c1_idx,c2_idx,:] = X[j,:,:]
	else:
		X_c = None
	return np.nan_to_num(X_c)


"""
A function to get 2 dPCA datasets: X_c, data from all trials in a session;
and X_b; trials that fit some belief criteria based on a hidden markov model.
Inputs:
	f_behavior: file path to behavior data
	f_ephys: file path to ephys data
	conditions: should be two of the following: 'context', action', or 'outcome'. Can't be all three because
		there is no such thing as a rewarded trial with upper lever contex and lower lever action.
	smooth_method: type of smoothing to use; choose 'bins', 'gauss', 'both', or 'none'
	smooth_width: size of the bins or gaussian kernel in ms. If 'both', input should be a list
		with index 0 being the gaussian width and index 1 being the bin width
	pad: a window for pre- and post-trial padding, in ms. In other words, an x-ms period of time 
		before lever press to consider the start of the trial, and an x-ms period of time after
		reward to consider the end of the trial. For best results, should be a multiple of the bin size
	z_score: if True, z-scores the array
	trial_duration: specifies the trial length (in ms) to squeeze trials into. If None, the function uses
		the median trial length over the trials in the file
	min_rate: the min spike rate, in Hz, to accept. Units below this value will be removed.
	max_duration: maximum allowable trial duration (ms)
Returns:
	X_c: data from individual trials: n-trials x n-neurons x condition-1, condition-2, ... x n-timebins.
"""

def get_dataset_hmm(f_behavior,f_ephys,conditions,smooth_method='both',smooth_width=[80,40],pad=[400,400],
	trial_duration=None,max_duration=5000,min_rate=0.1,belief_range=(0,0.1)):
	global condition_pairs
	##the first step is to fit a hidden markov model to the data
	fit_results = mf.fit_models(f_behavior)
	##now get the info about belief states
	b_a = fit_results['e_HMM'][0,:] ##belief in state corresponding to lower lever
	b_b = fit_results['e_HMM'][1,:] ##belief in state corresponding to upper lever
	##the belief strength is besically the magnitude of the difference between
	##the belief in the two possible states
	belief = abs(b_a-b_b)
	belief_idx = np.where(np.logical_and(belief>=belief_range[0],
		belief<=belief_range[1]))[0]
	##get the spike dataset, and the trial info
	X,trial_data = ptr.get_trial_spikes(f_behavior=f_behavior,f_ephys=f_ephys,smooth_method=smooth_method,
		smooth_width=smooth_width,pad=pad,z_score=True,trial_duration=trial_duration,
		max_duration=max_duration,min_rate=min_rate)
	##get some metadata about this session
	n_units = X.shape[1]
	n_bins = X.shape[2]
	# if balance: 
	# 	trial_index,n_trials = balance_trials(trial_data,conditions)
	# else:
	trial_index,n_trials = unbalance_trials(trial_data,conditions) ##because of limited trial number we will unbalance
	trial_types = list(trial_index)
	##allocate space for the dataset
	if n_trials > 0:
		X_c = np.empty((n_trials,n_units,len(condition_pairs[conditions[0]]),
			len(condition_pairs[conditions[1]]),n_bins))
		X_c[:] = np.nan
		for t in trial_index.keys():
			##based on the key, figure out where these trials should be placed in the dataset
			##I **think** that we should always expect the context[0] trial type to be the first part of the string
			c1_type = t[:t.index('+')]
			c2_type = t[t.index('+')+1:]
			c1_idx = condition_pairs[conditions[0]].index(c1_type)
			c2_idx = condition_pairs[conditions[1]].index(c2_type)
			##now add the data to the dataset using these indices
			for i,j in enumerate(trial_index[t]):
				X_c[i,:,c1_idx,c2_idx,:] = X[j,:,:]
		X_c = np.nan_to_num(X_c)
	else:
		X_c = np.empty([])
	##now repeat for the HMM-defined trials
	X = X[belief_idx]
	trial_data = trial_data.loc[belief_idx].reset_index(drop=True)
	##get some metadata about this session
	n_units = X.shape[1]
	n_bins = X.shape[2]
	# if balance: 
	# 	trial_index,n_trials = balance_trials(trial_data,conditions)
	# else:
	trial_index,n_trials = unbalance_trials(trial_data,conditions) ##because of limited trial number we will unbalance
	trial_types = list(trial_index)
	##allocate space for the dataset
	if n_trials > 0:
		X_b = np.empty((n_trials,n_units,len(condition_pairs[conditions[0]]),
			len(condition_pairs[conditions[1]]),n_bins))
		X_b[:] = np.nan
		for t in trial_index.keys():
			##based on the key, figure out where these trials should be placed in the dataset
			##I **think** that we should always expect the context[0] trial type to be the first part of the string
			c1_type = t[:t.index('+')]
			c2_type = t[t.index('+')+1:]
			c1_idx = condition_pairs[conditions[0]].index(c1_type)
			c2_idx = condition_pairs[conditions[1]].index(c2_type)
			##now add the data to the dataset using these indices
			for i,j in enumerate(trial_index[t]):
				X_b[i,:,c1_idx,c2_idx,:] = X[j,:,:]
		X_b = np.nan_to_num(X_b)
	else:
		X_b = np.empty([])
	return X_c,X_b
"""
A multiprocessing implementation of get_dataset()
"""
def get_dataset_mp(args):
	##parse args
	f_behavior = args[0]
	f_ephys = args[1]
	conditions = args[2]
	smooth_method = args[3]
	smooth_width = args[4]
	pad = args[5]
	z_score = args[6]
	trial_duration = args[7]
	max_duration = args[8]
	min_rate = args[9]
	balance = args[10]
	current_file = f_behavior[-11:-5]
	print("Adding data from file "+current_file)
	X_c = get_dataset(f_behavior,f_ephys,conditions,smooth_method=smooth_method,smooth_width=smooth_width,
		pad=pad,z_score=z_score,trial_duration=trial_duration,max_duration=max_duration,min_rate=min_rate,
		balance=balance)
	return X_c

"""
A multiprocessing implementation of get_dataset_hmm()
"""
def get_hmm_mp(args):
	##parse args
	f_behavior = args[0]
	f_ephys = args[1]
	conditions = args[2]
	smooth_method = args[3]
	smooth_width = args[4]
	pad = args[5]
	trial_duration = args[6]
	max_duration = args[7]
	min_rate = args[8]
	belief_range = args[9]
	current_file = f_behavior[-11:-5]
	print("Adding data from file "+current_file)
	X_c,X_b = get_dataset_hmm(f_behavior,f_ephys,conditions,smooth_method=smooth_method,smooth_width=smooth_width,
		pad=pad,trial_duration=trial_duration,max_duration=max_duration,min_rate=min_rate,belief_range=belief_range)
	return X_c,X_b
"""
This function actually runs dpca, relying on some globals for 
the details. ***NOTE: there are some hard-to-avoid elements here that are
			hardcoded specificallyfor this dataset.*****
Inputs:
	X_trials: array of data including individual trial data
	n_components: the number of components to fit
	conditions: list of conditions used to generate the dataset
"""
def run_dpca(X_trials,n_components,conditions,get_sig=True,optimize='auto'):
	##create labels from the first letter of the conditions
	##***this might not work if the implementation changes***
	labels = ''
	for c in range(len(conditions)):
		labels+=conditions[c][0]
	labels+='t'
	#create a dictionary argument that joins the time- and condition-dependent components
	join = join_labels(labels)
	##initialize the dpca object
	dpca = dPCA.dPCA(labels=labels,join=join,n_components=n_components,
		regularizer=optimize)
	dpca.protect = ['t']
	Z = dpca.fit_transform(np.nanmean(X_trials,axis=0),trialX=X_trials)
	##Next, get the variance explained:
	var_explained = dpca.explained_variance_ratio_
	#finally, get the significance masks (places where the demixed components are significant)
	if get_sig:
		sig_masks = dpca.significance_analysis(np.nanmean(X_trials,axis=0),X_trials,axis='t',
			n_shuffles=25,n_splits=3,n_consecutive=1)
	else:
		sig_masks = None
	return Z,var_explained,sig_masks



"""
A function to fit dpca using one dataset, and then use that
fit to transform a second dataset. 
Inputs:
	-X_trials1: dataset to use for initial fit
	-X_trials2: dataset to transform
	-n_components: number of components to use
	-conditions: conditions list
Returns:
	Z1, Z2: transformed marginalizations
	var_explained
"""
def fit_dpca_two(X_trials1,X_trials2,n_components,conditions):
	##create labels from the first letter of the conditions
	##***this might not work if the implementation changes***
	labels = conditions[0][0]+conditions[1][0]+'t'
	#create a dictionary argument that joins the time- and condition-dependent components
	join = join_labels(labels)
	##initialize the dpca object
	dpca = dPCA.dPCA(labels=labels,join=join,n_components=n_components,
		regularizer='auto')
	dpca.protect = ['t']
	##now fit dPCA with the first dataset
	dpca.fit(np.nanmean(X_trials1,axis=0),trialX=X_trials1)
	##now transform both datasets using this fit
	Z1 = dpca.transform(np.nanmean(X_trials1,axis=0))
	var_explained1 = dpca.explained_variance_ratio_
	Z2 = dpca.transform(np.nanmean(X_trials2,axis=0))
	var_explained2 = dpca.explained_variance_ratio_
	return Z1,Z2,var_explained1,var_explained2




"""
Multiprocess implementation of run_dpca
"""
def run_dpca_mp(args):
	##parse args
	X_trials = args[0]
	n_components = args[1]
	conditions = args[2]
	Z,var_explained,sig_masks = run_dpca(X_trials,n_components,conditions)
	return Z,var_explained,sig_masks	


"""
A function to equate the lengths of trials by using a piecewise
linear stretching procedure, outlined in 

"Kobak D, Brendel W, Constantinidis C, et al. Demixed principal component analysis of neural population data. 
van Rossum MC, ed. eLife. 2016;5:e10989. doi:10.7554/eLife.10989."

The function makes some important assumptions about the data:
1) that there is even pre- and post- padding before an action timestamp and outcome timestamp, respectively
	##actually this supports more epochs than just one, but I have no use for it yet
2) following this, that the epoch that needs to be stretched is the time between action and outcome.
3) the requested duruation passed as an argument applies to this middle epoch

Inputs:
	X_trials: a list containing ephys data from a variety of trials. This function assumes
		that trials are all aligned to the first event. data for each trial should be cells x timebins
	ts: array of timestamps used to generate the ephys trial data. 
		***These timestamps should be relative to the start of each trial. 
			for example, if a lever press happens 210 ms into the start of trial 11,
			the timestamp for that event should be 210.***
	median_dur: the duration to stretch the middle epoch to, such that all trials are the same length

"""
def stretch_trials(X_trials,ts,median_dur):
	##determine how many events we have
	n_events = ts.shape[1]
	##and how many trials
	n_trials = len(X_trials)
	##and how many neurons
	n_neurons = X_trials[0].shape[0]
	pieces = [] ##this will be a list of each streched epoch piece
	##check to see if the first event is aligned to the start of the data,
	##or if there is some pre-event data included.
	if not np.all(ts[:,1]==0):
		##make sure each trial is padded the same amount
		if np.all(ts[:,1]==ts[0,1]):
			pad1 = ts[0,1] ##this should be the pre-event window for all trials
			##add this first piece to the collection
			data = np.zeros((n_trials,n_neurons,pad1))
			for t in range(n_trials):
				data[t,:,:] = X_trials[t][:,0:pad1]
			pieces.append(data)
		else:
			print("First event is not aligned for all trials")
	##do the timestretching for each epoch individually
	for e in np.arange(1,n_events-2):
		##get just the interval for this particular epoch
		epoch_ts = ts[:,e:e+2]
		##now interpolate this to be a uniform size
		data_new = interp_trials(X_trials,epoch_ts,median_dur)
		pieces.append(data_new)
	##finally, see if the ephys data has any padding after the final event
	##collect the differences between the last trial outcome and the trial end
	t_diff = np.zeros(n_trials)
	for i in range(n_trials):
		t_diff[i] = ts[i,-1]-ts[i,-2]
	if not np.all(t_diff==0):
		##make sure padding is equal for all trials
		if np.all(t_diff==t_diff[0]):
			pad2 = t_diff[0].astype(int)
			data = np.zeros((n_trials,n_neurons,pad2))
			for t in range(n_trials):
				data[t,0:X_trials[t].shape[0],:] = X_trials[t][:,-pad2:]
			pieces.append(data)
		else:
			print("Last event has uneven padding")
			pad2 = np.floor(t_diff[0]).astype(int)
			data = np.zeros((n_trials,n_neurons,pad2))
			for t in range(n_trials):
				data[t,:,:] = X_trials[t][:,-pad2:]
			pieces.append(data)
	##finally, concatenate everything together!
	X = np.concatenate(pieces,axis=2)
	return X

"""
A helper function that does interpolation on one trial.

Inputs:
	data: list of trial data to work with; each trial should be neurons x bins
	epoch_ts: timestamps for each trial of the epoch to interpolate over, in bins
	new_dur: the requested size of the trial after interpolation
Returns:
	data_new: a numpy array with the data stretched to fit
"""
def interp_trials(data,epoch_ts,new_dur):
	##run a check to make sure the dataset has all of the same number of neurons
	n_neurons = np.zeros(len(data))
	for i in range(len(data)):
		n_neurons[i] = data[i].shape[0]
	if not np.all(n_neurons==n_neurons[0]):
		raise ValueError("Trials have different numbers of neurons")
	xnew = np.arange(new_dur) ##this will be the timebase of the interpolated trials
	data_new = np.zeros((len(data),int(n_neurons[0]),xnew.shape[0]))
	##now operate on each trial, over this particular epoch
	for t in range(len(data)):
		##get the actual data for this trial
		trial_data = data[t]
		##now, trial_data is in the shape units x bins.
		##we need to interpolate data from each unit individually:
		for n in range(trial_data.shape[0]):
			##get the data for neuron n in trial t and epoch e
			y = trial_data[n,epoch_ts[t,0]:epoch_ts[t,1]]
			x = np.arange(y.shape[0])
			##create an interpolation object for these data
			f = interpolate.interp1d(x,y,bounds_error=False,fill_value='extrapolate',
				kind='nearest')
			##now use this function to interpolate the data into the 
			##correct size
			ynew = f(xnew)
			##now put the data into its place
			data_new[t,n,:] = ynew
	return data_new

"""
A helper function to get timestamps in a trial-relative format
(assuming they are relative to absulute session time to begin with).
Inputs:
	ts: the session-relative timestamps, in ms
	pad: the padding used on the data 
	smooth_method: the smoothing method used on the matched data
	smooth_width: the smooth width used
returns:
	ts_rel: timestamps relative to the start of each trial, and scaled according to bin size
"""
def get_relative_ts(ts,pad,smooth_method,smooth_width):
	##first convert to bins from ms, and offset according to the padding
	if smooth_method == 'bins':
		ts_rel = ts/smooth_width
		offset = pad[0]/float(smooth_width)
	elif smooth_method == 'both':
		ts_rel = ts/smooth_width[1]
		offset = pad[0]/float(smooth_width[1])
	else: 
		ts_rel = ts
		offset = pad[0]
	##now get the timstamps in relation to the start of each trial
	ts_rel[:,1] = ts_rel[:,1]-ts_rel[:,0]
	##now account for padding
	ts_rel[:,0] = offset
	ts_rel[:,1] = ts_rel[:,1]+offset
	##get as integer
	ts_rel = np.ceil(ts_rel).astype(int)
	return ts_rel

"""
A helper function to do z-scoring across trials by doing some array manupulation
Inputs:
	X_trials: a array or list of trial data, where each of this first dim
		is data from one trial in neurons x bins dimensions
Returns:
	X_result: the same shape as X_trials, but the data has been z-scored for each neuron
		using data across all trials
"""
def zscore_across_trials(X_trials):
	##ideally, we want to get the zscore value for each neuron across all trials. To do this,
	##first we need to concatenate all trial data for each neuron
	##make a data array to remember the length of each trial
	trial_lens = np.zeros(len(X_trials))
	for i in range(len(X_trials)):
		trial_lens[i] = X_trials[i].shape[1]
	trial_lens = trial_lens.astype(int)
	##now concatenate all trials
	X_trials = np.concatenate(X_trials,axis=1)
	##now zscore each neuron's activity across all trials
	for n in range(X_trials.shape[0]):
		X_trials[n,:] = zscore(X_trials[n,:])
	##finally, we want to put everything back in it's place
	if np.all(trial_lens==trial_lens[0]): ##case where all trials are same length (probably timestretch==True)
		X_result = np.zeros((trial_lens.shape[0],X_trials.shape[0],trial_lens[0]))
		for i in range(trial_lens.shape[0]):
			X_result[i,:,:] = X_trials[:,i*trial_lens[i]:(i+1)*trial_lens[i]]
		X_trials = X_result
	else:
		print("different trial lengths detected; parsing as list")
		X_result = []
		c = 0
		for i in range(trial_lens.shape[0]):
			X_result.append(X_trials[:,c:c+trial_lens[i]])
			c+=trial_lens[i]
	return X_result

"""
dPCA requires balanced conditions, meaning that there needs to be even numbers of each trial
type. In my task, animals didn't necessarily do the same number of trials across all combinations
of conditions. So, to accout for that, we will only take n trials of each type, where n is the
lowest number of trials for any given condition. This function takes in a trial data dataframe, 
a list of the two conditions to consider, and returns balanced indices of trials across all conditions.
Inputs:
	trial_data: pandas dataframe with trial data, like that returned by ptr.get_full_trials
	conditions: should be two of the following: 'context', action', or 'outcome'. Can't be all three because
			there is no such thing as a rewarded trial with upper lever contex and lower lever action.
Returns:
	trial_index: dictionary with trial indices for all conditions
	n_trials: number of trials 
"""
def balance_trials(trial_data,conditions):
	##begin by getting the data for both of the conditions
	cond_1 = trial_data[conditions[0]]
	cond_2 = trial_data[conditions[1]]
	##figure out how many different trial types we have
	n_types = np.unique(cond_1).size * np.unique(cond_2).size
	##make a dictionary to store the indices of each trial type
	trial_index = {}
	for c1 in np.unique(cond_1):
		for c2 in np.unique(cond_2):
			trial_index[c1+"+"+c2] = []
	for i in range(len(trial_data.index)):
		trial_type = trial_data[conditions[0]][i]+"+"+trial_data[conditions[1]][i]
		##now just add the index to the dictionary
		trial_index[trial_type].append(i)
	##now, what is the minimum number of trials across all trial types?
	min_trials = min([len(x) for x in trial_index.values()])
	##from here, just randomly sample trials from conditions that have more than the min
	##number of trials so everything is balanced
	for k in trial_index.keys():
		if len(trial_index[k]) > min_trials:
			sub_idx = np.random.randint(0,len(trial_index[k]),min_trials)
			trial_index[k] = np.asarray(trial_index[k])[sub_idx]
	return trial_index,min_trials

"""
This function is a counterpart to balance_trials. It also picks out the indices of the various trial
types, but doesn't remove any trials from any condition, meaning that the index lists can be different 
lengths.
Inputs:
	trial_data: pandas dataframe with trial data, like that returned by ptr.get_full_trials
	conditions: should be two of the following: 'context', action', or 'outcome'. Can't be all three because
			there is no such thing as a rewarded trial with upper lever contex and lower lever action.
Returns:
	trial_index: dictionary with trial indices for all conditions
	max_trials: maximum number of trials for any trial type
"""
def unbalance_trials(trial_data,conditions):
	##begin by getting the data for both of the conditions
	cond_1 = trial_data[conditions[0]]
	cond_2 = trial_data[conditions[1]]
	##figure out how many different trial types we have
	n_types = np.unique(cond_1).size * np.unique(cond_2).size
	##make a dictionary to store the indices of each trial type
	trial_index = {}
	for c1 in np.unique(cond_1):
		for c2 in np.unique(cond_2):
			trial_index[c1+"+"+c2] = []
	for i in range(len(trial_data.index)):
		trial_type = trial_data[conditions[0]][i]+"+"+trial_data[conditions[1]][i]
		##now just add the index to the dictionary
		trial_index[trial_type].append(i)
	##run a check to see if any of the trial types have 0 occurances
	##we will discard this session if this is the case
	discard = False
	for key, value in zip(trial_index.keys(),trial_index.values()):
		if len(value) == 0:
			discard = True
			print("This session is missing trials of type {0}; discarding...".format(key))
	##now, what is the minimum number of trials across all trial types?
	if not discard:
		max_trials = max([len(x) for x in trial_index.values()])
		for k in trial_index.keys():
			if len(trial_index[k]) < max_trials:
				trial_index[k] = np.random.choice(trial_index[k],max_trials)
	else:
		max_trials = 0
	return trial_index,max_trials

"""
A function to get the indices of trials that occur after a context switch.
Inputs:
	trial_data: pandas data array of trial timestmaps and labels
	n_after: number of trials after a switch to include
Returns: 
	switch_trials: indices of trials that occur after a conext switch.
"""
def get_switch_index(trial_data,n_after):
	##find the indices where there is a context switch
	new_block_starts = []
	max_trials = max(trial_data.index)
	for i in range(len(trial_data.index)-1):
		current_context = trial_data['context'][i]
		next_context = trial_data['context'][i+1]
		if not current_context == next_context:
			new_block_starts.append(i)
	switch_trials = []
	for t in range(len(new_block_starts)):
		for n in range(n_after):
			if new_block_starts[t]+n <= max_trials:
				switch_trials.append(new_block_starts[t]+n)
	return switch_trials


"""
##############################################
###############################################
##############################################
          Functions to produce datasets for MATLAB
"""

def get_dataset_mlab(f_behavior,f_ephys,conditions,smooth_method='both',smooth_width=[80,40],pad=[400,400],
	z_score=True,trial_duration=None,max_duration=5000,min_rate=0.1):
	global condition_pairs
	##get the spike dataset, and the trial info
	X,trial_data = ptr.get_trial_spikes(f_behavior=f_behavior,f_ephys=f_ephys,smooth_method=smooth_method,
		smooth_width=smooth_width,pad=pad,z_score=z_score,trial_duration=trial_duration,
		max_duration=max_duration,min_rate=min_rate)
	##if uncertainty level is one of the requested marginalizations, add this data to the trial_data set
	if 'u_level' in conditions:
		trial_data = mf.append_uncertainty(trial_data,condition_pairs['u_level'])
	##get some metadata about this session
	n_units = X.shape[1]
	n_bins = X.shape[2]
	#sort out the different trial types
	trial_index,n_trials = split_trials(trial_data,conditions)
	trial_types = list(trial_index)
	##allocate space for the dataset
	if n_trials > 0:
		X_c = np.empty((n_trials,n_units,len(condition_pairs[conditions[0]]),
			len(condition_pairs[conditions[1]]),n_bins))
		X_c[:] = np.nan
		##generate an array that provides info about how many trial numbers we have for each trial type
		trialNum = np.empty((n_units,len(condition_pairs[conditions[0]]),
				len(condition_pairs[conditions[1]])))
		trialNum[:] = np.nan
		for t in trial_index.keys():
			##based on the key, figure out where these trials should be placed in the dataset
			##I **think** that we should always expect the context[0] trial type to be the first part of the string
			c1_type = t[:t.index('+')]
			c2_type = t[t.index('+')+1:]
			c1_idx = condition_pairs[conditions[0]].index(c1_type)
			c2_idx = condition_pairs[conditions[1]].index(c2_type)
			##now add the data to the dataset using these indices
			for i,j in enumerate(trial_index[t]):
				X_c[i,:,c1_idx,c2_idx,:] = X[j,:,:]
			##record how many trials of this type we have
			trialNum[:,c1_idx,c2_idx] = len(trial_index[t])
	else:
		print("One marginalization has no trials.")
		X_c = None
		trialNum = None
	return X_c,trialNum


"""
A multiprocessing implementation of get_dataset()
"""
def get_dataset_mp_mlab(args):
	##parse args
	f_behavior = args[0]
	f_ephys = args[1]
	conditions = args[2]
	smooth_method = args[3]
	smooth_width = args[4]
	pad = args[5]
	z_score = args[6]
	trial_duration = args[7]
	max_duration = args[8]
	min_rate = args[9]
	current_file = f_behavior[-11:-5]
	print("Adding data from file "+current_file)
	X_c,trialNum = get_dataset_mlab(f_behavior,f_ephys,conditions,smooth_method=smooth_method,smooth_width=smooth_width,
		pad=pad,z_score=z_score,trial_duration=trial_duration,max_duration=max_duration,min_rate=min_rate)
	return X_c,trialNum


def split_trials(trial_data,conditions):
	##begin by getting the data for both of the conditions
	cond_1 = trial_data[conditions[0]]
	cond_2 = trial_data[conditions[1]]
	##figure out how many different trial types we have
	n_types = np.unique(cond_1).size * np.unique(cond_2).size
	##make a dictionary to store the indices of each trial type
	trial_index = {}
	for c1 in np.unique(cond_1):
		for c2 in np.unique(cond_2):
			trial_index[c1+"+"+c2] = []
	for i in range(len(trial_data.index)):
		trial_type = trial_data[conditions[0]][i]+"+"+trial_data[conditions[1]][i]
		##now just add the index to the dictionary
		trial_index[trial_type].append(i)
	##now, what is the maximum number of trials across all trial types?
	max_trials = max([len(x) for x in trial_index.values()])
	min_trials = min([len(x) for x in trial_index.values()])
	##this is a check to be sure we have at least 1 trial per condition
	if min_trials > 4:
		n_trials = max_trials
	else:
		n_trials = 0
	return trial_index,n_trials

"""
A helper function to join labels with time axes
"""
def join_labels(labels):
	##assume the last index is 't' (time)
	s = list(labels[:-1])
	perms = chain.from_iterable(combinations(s, r) for r in range(len(s)+1))
	perms = list(map(''.join,perms))[1:]
	join = {}
	for l in perms:
		join[l+'t'] = [l,l+'t']
	return join

