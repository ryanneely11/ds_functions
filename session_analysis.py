##session_analysis.py
## a function to run various analyses on full sessions 

import numpy as np
import parse_timestamps as pt
import parse_trials as ptr
import parse_ephys as pe
import regression as re
import file_lists
import log_regression2 as lr2
import log_regression3 as lr3
import os
import h5py
from functools import reduce
import dpca
import os
import pandas as pd
import model_fitting as mf
from sklearn import linear_model
import linear_regression as linr
import linear_regression2 as lin2
from scipy.stats import pearsonr
import tensor_analysis as ta
save_root = os.path.join(file_lists.save_loc,"LogisticRegression/80gauss_40ms_bins")

"""
A function to do logistic regression prediction of action using population
activity.
Inputs:
	f_behavior: file path to behavior data
	f_ephys: file path to ephys data
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
	n_iter: number of cross-validation iterations
Returns:
"""
def log_pop_action(f_behavior,f_ephys,smooth_method='both',smooth_width=[100,50],
	pad=[2000,100],z_score=True,trial_duration=None,max_duration=5000,min_rate=0.1,
	n_iter=10):
	
	##start by getting the spike data and trial data
	spike_data,trial_data = ptr.get_trial_spikes(f_behavior,f_ephys,
		smooth_width=smooth_width,smooth_method=smooth_method,pad=pad,z_score=z_score,
		trial_duration=trial_duration,max_duration=max_duration,min_rate=min_rate)
	##convert the action data to binary
	actions = (np.asarray(trial_data['action'])=='upper_lever').astype(int)
	##get the action part of the spike data
	if smooth_method=='both':
		bin_size=smooth_width[1]
	elif smooth_method=='bins':
		bin_size=smooth_width
	elif smooth_method=='gauss':
		bin_size=1
	action_bins = int(pad[0]/bin_size)
	X = spike_data[:,:,:action_bins]
	##now run the logistic regression
	accuracy = lr3.pop_logit(X,actions,add_constant=True,n_iter=n_iter)
	return accuracy
"""
A function to run linear regression on data from one session.
Inputs: 
	f_behavior: file path to behavior data
	f_ephys: file path to ephys data
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
	n_iter: number of iterations to use for permutation testing
	perc: if True, return the percentage of units rather than the counts
Returns:
	f_counts: number of sig units at each time point using f-test
	p_counts: "" using permutation test
"""
def linear_regression(f_behavior,f_ephys,smooth_method='both',smooth_width=[100,50],
	pad=[800,800],z_score=True,trial_duration=None,min_rate=0.1,max_duration=5000,
	n_iter=1000,perc=False):
	##first get the spike data and the trial data
	spike_data,regressors = lin2.get_datasets(f_behavior,f_ephys,smooth_method=smooth_method,
		smooth_width=smooth_width,pad=pad,z_score=z_score,trial_duration=trial_duration,
		min_rate=min_rate,max_duration=max_duration)
	regressors = np.asarray(regressors)
	##Run the regression on these data
	f_counts,p_counts = lin2.regress_spike_matrix(regressors,spike_data,add_constant=True,n_iter=n_iter)
	if perc:
		f_counts = f_counts/float(spike_data.shape[1])
		p_counts = p_counts/float(spike_data.shape[1])
	return f_counts,p_counts

"""
A function to run linear regression on data from one session. Returns the number of
paramters encoded by each unit.
Inputs: 
	f_behavior: file path to behavior data
	f_ephys: file path to ephys data
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
	n_iter: number of iterations to use for permutation testing
	perc: if True, return the percentage of units rather than the counts
Returns:
	f_counts: number of sig units at each time point using f-test
	p_counts: "" using permutation test
"""
def linear_regression2(f_behavior,f_ephys,smooth_method='both',smooth_width=[100,50],
	pad=[1000,1000],z_score=True,trial_duration=None,min_rate=0.1,max_duration=5000,
	n_iter=0,n_consecutive=4):
	##first get the spike data and the trial data
	spike_data,regressors = lin2.get_datasets(f_behavior,f_ephys,smooth_method=smooth_method,
		smooth_width=smooth_width,pad=pad,z_score=z_score,trial_duration=trial_duration,
		min_rate=min_rate,max_duration=max_duration)
	regressors = np.asarray(regressors)
	##Run the regression on these data
	n_encoded = lin2.regress_spike_matrix2(regressors,spike_data,add_constant=True,n_iter=n_iter)
	return n_encoded

"""
A function to parse the results of a logistic regression. 
Inputs: 
	f_data: file path to a hdf5 file with logistic regression data
	sig_level: p-value threshold for significance
	test_type: string; there should be two statistucs; a log-liklihood ratio p-val
		from the fit from statsmodels ('llrp_pvals'), and a p-val from doing permutation
		testing with shuffled data ('pvals'). This flag lets you select which one to use.
Returns:
	results: a dictionary with the results
"""
def parse_log_regression(f_data,sig_level=0.05,test_type='llr_pvals'):
	global event_pairs
	##open the file
	f = h5py.File(f_data,'r')
	results = {}
	##now we want to just get data from units that encode something significantly
	conditions = list(f)
	##look at data from each condition
	for c in conditions:
		results[c] = {}
		##find the indices of units that are significant here
		sig_idx = np.where(np.asarray(f[c][test_type])<sig_level)[0]
		##only continue if there are any sig units
		if sig_idx.size > 0:
			results[c]['idx'] = sig_idx
			results[c]['pvals'] = np.asarray(f[c][test_type])[sig_idx]
			results[c]['accuracy'] = np.asarray(f[c]['accuracies'])[sig_idx]
			results[c]['chance'] = np.asarray(f[c]['chance_rates'])[sig_idx]
			##get the trials of each type for this context
			##first we need to figure out the labels
			labels = [x for x in list(f[c]) if x.startswith('is_')]
			labels[0] = labels[0][3:] ##the first one is the 1's in the y-array
			labels.append([y for y in event_pairs[c] if not y == labels[0]][0])
			##now reverse so it matches (see below)
			labels.reverse()
			##the second label corresponds the the 0's in the y-array
			##now we can get the traces for each event type in this condition
			##(just subsets of our X-array)
			for i, label in enumerate(labels):
				event_idx = np.where(np.asarray(f[c]['is_'+labels[-1]])==i)[0]
				results[c][label] = event_idx ## the indices for this event
			##finally, just add the X-data
			results[c]['X'] = np.asarray(f[c]['X'])
	f.close()
	return results

"""
A function to parse the results of a logistic regression, and extract some
good examples of units with different encoding properties. 
Inputs: 
	f_data: file path to a hdf5 file with logistic regression data
	sig_level: p-value threshold for significance
	test_type: string; there should be two statistucs; a log-liklihood ratio p-val
		from the fit from statsmodels ('llrp_pvals'), and a p-val from doing permutation
		testing with shuffled data ('pvals'). This flag lets you select which one to use.
	accuracy_thresh: the threshold to use when considering candidate units
Returns:
	results: a dictionary with the results
"""
def get_log_regression_samples(f_data,sig_level=0.05,test_type='llr_pvals',accuracy_thresh=0.8):
	global event_pairs
	##open the file
	f = h5py.File(f_data,'r')
	results = {}
	##now we want to just get data from units that encode something significantly
	conditions = list(f)
	##look at data from each condition
	for c in conditions:
		results[c] = {}
		##find the indices of units that are significant here
		sig_idx = np.where(np.asarray(f[c][test_type])<sig_level)[0]
		##only continue if there are any sig units
		if sig_idx.size > 0:
			results[c]['idx'] = sig_idx
			results[c]['pvals'] = np.asarray(f[c][test_type])[sig_idx]
			results[c]['accuracy'] = np.asarray(f[c]['accuracies'])[sig_idx]
			results[c]['chance'] = np.asarray(f[c]['chance_rates'])[sig_idx]
			##get the trials of each type for this context
			##first we need to figure out the labels
			labels = [x for x in list(f[c]) if x.startswith('is_')]
			labels[0] = labels[0][3:] ##the first one is the 1's in the y-array
			labels.append([y for y in event_pairs[c] if not y == labels[0]][0])
			##now reverse so it matches (see below)
			labels.reverse()
			##the second label corresponds the the 0's in the y-array
			##now we can get the traces for each event type in this condition
			##(just subsets of our X-array)
			for i, label in enumerate(labels):
				event_idx = np.where(np.asarray(f[c]['is_'+labels[-1]])==i)[0]
				results[c][label] = event_idx ## the indices for this event
			##finally, just add the X-data
			results[c]['X'] = np.asarray(f[c]['X'])
	f.close()
	##now work out which units encode what
	sig_idx = []
	for epoch in conditions:
		try:
			sig_idx.append(results[epoch]['idx'])
		except KeyError: ##case where there were no sig units in this epoch
			pass
	sig_idx = np.unique(np.concatenate(sig_idx)) ##unique gets rid of any duplicates
	##create a dataframe to store info about which units encode what
	encode_data = pd.DataFrame(columns=conditions,index=sig_idx)
	cursor = 0
	for epoch in conditions:
		try:
			epoch_sig = results[epoch]['idx'] ##the indices of significant units in this epoch
			for i,unitnum in enumerate(epoch_sig):
				##we know all units meet significance criteria, so just save the accuracy
				encode_data[epoch][unitnum] = results[epoch]['accuracy'][i]
		except KeyError: ##no sig units in this epoch
			pass
	##now let's set up a dictionary where we can store examplary unit IDs
	example_units = {
	'action_only':[],
	'action+context':[],
	'context_only':[],
	'context+outcome':[],
	'outcome_only':[],
	'outcome+action':[],
	'outcome+action+context':[]
	}
	for i in sig_idx:
		line = encode_data.loc[i]
		if not np.isnan(line['action']) and (np.isnan(line['context']) and np.isnan(line['outcome'])):
			##case where it's action-only
			if line['action'] >= accuracy_thresh:
				example_units['action_only'].append(i)
		elif (not np.isnan(line['action']) and not np.isnan(line['context'])) and np.isnan(line['outcome']):
			##case wher it's action and context
			if line['action'] >= accuracy_thresh and line['context'] >= accuracy_thresh:
				example_units['action+context'].append(i)
		elif not np.isnan(line['context']) and (np.isnan(line['action']) and np.isnan(line['outcome'])):
			##case where it's context-only
			if line['context'] >= accuracy_thresh:
				example_units['context_only'].append(i)
		elif (not np.isnan(line['context']) and not np.isnan(line['outcome'])) and np.isnan(line['action']):
			##case where it's context and outcome
			if line['outcome'] >= accuracy_thresh and line['context'] >= accuracy_thresh:
				example_units['context+outcome'].append(i)
		elif not np.isnan(line['outcome']) and (np.isnan(line['action']) and np.isnan(line['context'])):
			if line['outcome'] >= accuracy_thresh:
				example_units['outcome_only'].append(i)
		elif (not np.isnan(line['outcome']) and not np.isnan(line['action'])) and np.isnan(line['context']):
			##case where it's action and outcome
			if line['action'] >= accuracy_thresh and line['outcome'] >= accuracy_thresh:
				example_units['outcome+action'].append(i)
		elif (not np.isnan(line['outcome']) and not np.isnan(line['action']) and not np.isnan(line['context'])):
			##case where it encodes all 3
			if line['action'] >= accuracy_thresh and line['context'] >= accuracy_thresh and line['outcome'] >= accuracy_thresh:
				example_units['outcome+action+context'].append(i)
		else:
			print("Warning: no catagory found for unit "+str(i))
	##now we know what units we can get data from, so let's save the actual spike data for each possible 
	##condition for each of these units.
	##another dictionary to store the data:
	data_dict = {
	'action_only':{},
	'action+context':{},
	'context_only':{},
	'context+outcome':{},
	'outcome_only':{},
	'outcome+action':{},
	'outcome+action+context':{}
	}
	##now get the indices of all the different trial types
	trial_idx = {}
	for epoch in event_pairs:
		events = event_pairs[epoch]
		for event in events:
			trial_idx[event] = results[epoch][event]
	for t in list(data_dict):
		for e in list(trial_idx):
			data_dict[t][e] = []
	##now get the data
	for unit_type in list(example_units):
		unit_nums = example_units[unit_type]
		if len(unit_nums)>0:
			for unit in unit_nums:
				##get all the trails for each trial type for this unit
				for epoch in event_pairs:
					X_epoch = results[epoch]['X']
					for event in event_pairs[epoch]:
						event_idx = list(trial_idx[event])
						X_event = X_epoch[unit,event_idx,:]
						data_dict[unit_type][event].append(X_event)
	return data_dict



"""
A session to run logistic regression on pairs of task variables. Output is to 
a file, so data is saved. 
Inputs:
	-f_behavior: path to behavior data
	-f_ephys: path to ephys data
	-window: time window to use for analysis, in ms
	-smooth_method: type of smoothing to use; choose 'bins', 'gauss', or 'none'
	-smooth_width: size of the bins or gaussian kernel in ms
	-z_score: if True, z-scores the array
	
Returns:
	all data saved to file.
"""
def log_regress_session(f_behavior,f_ephys,win=500,smooth_method='gauss',
	smooth_width=30,z_score=True,min_rate=0):
	global event_pairs
	global save_root
	print("Computing regressions for "+f_behavior[-11:-5]+":")
	##open a file to save the data
	save_path = os.path.join(save_root,f_behavior[-11:-9],f_behavior[-11:-5]+".hdf5")
	##create the file
	try:
		f_out = h5py.File(save_path,'x')
		f_out.close()
		for event in list(event_pairs):
			print("Computing regression on "+event+" trials...")
			##get all of the event pairs	
			##start by getting the list of event pairs
			ts_ids = event_pairs[event]
			##create a custom window depending on the epoch we are interested in
			if event == 'context' or event == 'action':
				window = [win,50] ##we'll pad with 50 ms just in case
			elif event == 'outcome':
				window = [50,win]
			##now get the data arrays for each of the event types
			X_all = []
			y_all = []
			y_strings_all = []
			for i,name in enumerate(ts_ids):
				X_data = ptr.get_event_spikes(f_behavior,f_ephys,name,window=window,
					smooth_method=smooth_method,smooth_width=smooth_width,z_score=z_score,
					min_rate=min_rate)
				##now create label data for this set
				y_data = np.ones(X_data.shape[0])*i
				y_strings = np.empty(X_data.shape[0],dtype='<U19')
				y_strings[:] = name
				X_all.append(X_data)
				y_all.append(y_data)
			##concatenate data
			X_all = ptr.remove_nan_units(X_all)
			X_all = np.concatenate(X_all,axis=0)
			y_all = np.concatenate(y_all,axis=0)
			##now re-arrange the X_data so it's units x trials x bins
			X_all = np.transpose(X_all,(1,0,2))
			##now we can run the regression
			accuracies,chance_rates,pvals,llr_pvals = lr3.permutation_test_multi(X_all,y_all)
			##finally, we can save these data
			print("Saving...")
			f_out = h5py.File(save_path,'a')
			group = f_out.create_group(event)
			group.create_dataset("accuracies",data=accuracies)
			group.create_dataset("chance_rates",data=chance_rates)
			group.create_dataset("pvals",data=pvals)
			group.create_dataset("X",data=X_all)
			group.create_dataset("is_"+ts_ids[1],data=y_all)
			group.create_dataset('llr_pvals',data=llr_pvals)
			f_out.close()
			print("Done")
		print("Session complete")
	except IOError:
		print("This file exists! Skipping...")

"""
A function to return an array of trial durations. Here I'm definig that as the time between
action and outcome.
Inputs:
	f_behavior: data file path
	max_duration: maximum allowable duration of trials (anything longer is deleted)
Returns:
	trial_durs: array of trial durations
"""
def session_trial_durations(f_behavior,max_duration=5000):
	data = ptr.get_full_trials(f_behavior,max_duration=max_duration)
	return np.asarray(data['outcome_ts']-data['action_ts']).astype(int)

"""
A function to run dPCA analysis on data from one session

"""
def session_dpca(f_behavior,f_ephys,smooth_method='both',smooth_width=[40,50],
	pad=[200,200],z_score=True,n_components=10,remove_unrew=True):
	##get the data
	X_mean,X_trials = dpca.get_dataset(f_behavior,f_ephys,smooth_method=smooth_method,
		smooth_width=smooth_width,pad=pad,z_score=z_score,remove_unrew=remove_unrew)
	##if we didn't z-score the data, we need to at least center it
	if not z_score:
		###hardcoded bit here too
		unit_mean = X_mean.reshape((X_mean.shape[0],-1)).mean(axis=1)[:,None,None,None]
		X_mean -= unit_mean
	##get the time axis
	if smooth_method == 'both':
		time = np.linspace(-1*pad[0],smooth_width[1]*X_mean.shape[-1]-pad[0],X_mean.shape[-1])
	elif smooth_method == 'bins':
		time = np.linspace(-1*pad[0],smooth_width*X_mean.shape[-1]-pad[0],X_mean.shape[-1])	
	else:
		time = np.arange(0,X_mean.shape[-1])
	Z,var_explained,sig_masks = dpca.run_dpca(X_mean,X_trials,n_components)
	events = np.array([0,max(time)-pad[1]])
	return Z,time,var_explained,sig_masks,events

"""
This analysis was inspired by a talk from Newsome. The idea is to use logistic
regression to get the log odds of making a left or right choice on each trial, across
time within trials. An important assumtption is that the beta weights are more or less
constant across time within one epoch.
Inputs:
	-f_behavior: path to behavior data file
	-f_ephys: path to ephys data file
	-window [pre_event, post_event] window, in ms
	-smooth_method: type of smoothing to use; choose 'bins', 'gauss', 'both', or 'none'
	-smooth_width: size of the bins or gaussian kernel in ms. If 'both', input should be a list
		with index 0 being the gaussian width and index 1 being the bin width
	-min_rate: minimum acceptable spike rate, in Hz
"""
def decision_variables(f_behavior,f_ephys,pad,smooth_method='both',smooth_width=[80,40],
	min_rate=0.1,z_score=True,trial_duration=None,max_duration=4000):
	##get the raw data 
	X,trial_data = ptr.get_trial_spikes(f_behavior,f_ephys,smooth_method=smooth_method,
		smooth_width=smooth_width,pad=pad,z_score=z_score,trial_duration=trial_duration,
		max_duration=max_duration,min_rate=min_rate)
	##only take the data from the pre-action epoch
	epoch = 'action'
	if epoch == 'action':
		if smooth_method == 'bins':
			X = X[:,:,:int(pad[0]/smooth_width)]
		elif smooth_method == 'both':
			X = X[:,:,:int(pad[0]/smooth_width[1])]
		else:
			X = X[:,:,:int(pad[0])]
	if epoch == 'outcome':
		if smooth_method == 'bins':
			X = X[:,:,int(-pad[1]/smooth_width):]
		elif smooth_method == 'both':
			X = X[:,:,int(-pad[1]/smooth_width[1]):]
		else:
			X = X[:,:,int(-pad[1]):]
	##add a constant term
	intercept = np.ones((X.shape[0],1,X.shape[2]))
	X = np.concatenate([X,intercept],axis=1)
	##construct the labels
	labels = (np.asarray(trial_data['action'])=='upper_lever').astype(int)
	##compute the beta weights across the whole trial interval
	betas = lr2.get_betas(X,labels)
	##now take the mean across the whole interval (this assumes they are relatively constant)
	betas = np.mean(betas[:,-4:],axis=1)
	##OK, now we can compute the log odds (?) for all of the trials
	odds = np.zeros((X.shape[0],X.shape[2]))
	for t in range(X.shape[2]):
		odds[:,t] = np.dot(X[:,:,t],betas)
	##get the duration of this session to return 
	session_duration = pe.get_session_duration(f_ephys)
	return odds,trial_data,session_duration


"""
MP implementation
"""
def mp_decision_vars(args):
	##parse args
	f_behavior = args[0]
	f_ephys = args[1]
	pad = args[2]
	smooth_method = args[3]
	smooth_width = args[4]
	min_rate = args[5]
	z_score = args[6]
	trial_duration = args[7]
	max_duration = args[8]
	print("Processing session {}".format(f_behavior[-11:-5]))
	odds,trial_data,session_duration = decision_variables(f_behavior,f_ephys,pad,smooth_method=smooth_method,
		smooth_width=smooth_width,min_rate=min_rate,z_score=z_score,trial_duration=trial_duration,
		max_duration=max_duration)
	return odds,trial_data,session_duration

"""
A function similar to the decision variables function that instead returns
the linear regression estimate of confidence (strength of belief state)
over some time window prior to action. 
Inputs:
	-f_behavior: path to behavior data file
	-f_ephys: path to ephys data file
	-window [pre_event, post_event] window, in ms
	-smooth_method: type of smoothing to use; choose 'bins', 'gauss', 'both', or 'none'
	-smooth_width: size of the bins or gaussian kernel in ms. If 'both', input should be a list
		with index 0 being the gaussian width and index 1 being the bin width
	-min_rate: minimum acceptable spike rate, in Hz
Returns:
	predicted: predicted belief strength for each trial over the action window
	trial_data: the trial_data matrix for this session
"""
def lin_regress_belief(f_behavior,f_ephys,pad,smooth_method='both',smooth_width=[100,50],
	min_rate=0.1,z_score=True,trial_duration=None,max_duration=4000):
	##get the raw data 
	X,trial_data = ptr.get_trial_spikes(f_behavior,f_ephys,smooth_method=smooth_method,
		smooth_width=smooth_width,pad=pad,z_score=z_score,trial_duration=trial_duration,
		max_duration=max_duration,min_rate=min_rate)
	##only take the data from the pre-action epoch
	epoch = 'action'
	if epoch == 'action':
		if smooth_method == 'bins':
			X = X[:,:,:int(pad[0]/smooth_width)]
		elif smooth_method == 'both':
			X = X[:,:,:int(pad[0]/smooth_width[1])]
		else:
			X = X[:,:,:int(pad[0])]
	if epoch == 'outcome':
		if smooth_method == 'bins':
			X = X[:,:,int(-pad[1]/smooth_width):]
		elif smooth_method == 'both':
			X = X[:,:,int(-pad[1]/smooth_width[1]):]
		else:
			X = X[:,:,int(-pad[1]):]
	##now get the belief strength by using the HMM model
	model_data = mf.fit_models_from_trial_data(trial_data)
	confidence = np.abs(model_data['state_vals'][0]-model_data['state_vals'][1])
	##now we can regress the confidence using the spike data
	predicted,r2,r2_adj,mse = linr.fit_timecourse(X,confidence,add_constant=True,n_iter=10)
	return predicted,r2,mse,trial_data,confidence


"""
A function to compare the "belief strength" of a given trial based on hidden markov
model data to whether or not the last trial was rewarded.
Inputs:
	f_behavior: data file
	max_duration: the ceiling of trial durations to consider valid
Returns:
"""
def uncertainty_vs_last_rewarded(f_behavior,max_duration=5000):
	##start by getting the trial data for this session
	trial_data = ptr.get_full_trials(f_behavior,max_duration=max_duration)
	##now determine which trials were rewarded or unrewarded
	last_rew_idx,last_unrew_idx = ptr.split_by_last_outcome(trial_data)
	##compute the hidden markov model 
	model_data = mf.fit_models_from_trial_data(trial_data)
	##return the confidence, belief states
	uncertainty = mf.uncertainty_from_trial_data(trial_data)
	##split into last rewarded and last unrewarded belief states
	last_rew_uncertainty = uncertainty[last_rew_idx]
	last_unrew_uncertainty = uncertainty[last_unrew_idx]
	return last_rew_uncertainty,last_unrew_uncertainty

"""
A function to compare trial factors from tensor analysis and belief estimations from HMM
Inputs:
	f_behavior: file path to behavior data
	f_ephys: file path to ephys data

"""
def tensors_v_uncertainty(f_behavior,f_ephys,smooth_method='both',smooth_width=[80,40],
	pad=[1200,1200],trial_duration=None,min_rate=0,max_duration=3000,
	n_components=4,epoch='outcome',verbose=True):
	##start by computing the tensors
	model,info,trial_data = ta.run_tensor(f_behavior,f_ephys,smooth_method=smooth_method,
		smooth_width=smooth_width,pad=pad,z_score=True,trial_duration=trial_duration,
		min_rate=min_rate,max_duration=max_duration,n_components=n_components,epoch=epoch)
	##now compute the uncertainty trial-to-trial
	uncertainty = mf.uncertainty_from_trial_data(trial_data)
	##finally compute the strongest correlation for the trial factors
	trial_factors = model[2]
	ccs = np.zeros(n_components)
	pvals = np.zeros(n_components)
	for i in range(n_components):
		ccs[i],pvals[i] = pearsonr(uncertainty,trial_factors[:,i])
	##find the trial factor with the most significant correlation to uncertainty
	best_idx = np.argmin(pvals)
	cc = ccs[best_idx]
	pval = pvals[best_idx]
	trial_factor = trial_factors[:,best_idx]
	if verbose:
		print(f_behavior[-11:-5])
		print("CC = {}".format(cc))
		print("P = {}".format(pval))
	return cc,pval,trial_factor,uncertainty

"""multiprocessing impementation"""
def mp_tensors_v_uncertainty(args):
	##parse args
	f_behavior = args[0]
	f_ephys = args[1]
	smooth_method = args[2]
	smooth_width = args[3]
	pad = args[4]
	trial_duration = args[5]
	min_rate = args[6]
	max_duration = args[7]
	n_components = args[8]
	epoch = args[9]
	cc,pval,trial_factor,uncertainty = tensors_v_uncertainty(f_behavior,f_ephys,
		smooth_method=smooth_method,smooth_width=smooth_width,pad=pad,
		trial_duration=trial_duration,min_rate=min_rate,
		max_duration=max_duration,n_components=n_components,epoch=epoch,verbose=False)
	print(f_behavior[-11:-5])
	print("CC = {}".format(cc))
	print("P = {}".format(pval))
	return cc,pval,trial_factor,uncertainty

"""
This function is designed to "standardize" a given trial. The rationale is that
many sessions have a similar structure, and if we standardize all sessions, we can
concatenate them and then run analyses on data from all sessions and animals,
concatenated into one big matrix.
Inputs:
	f_behavior: data file with behavioral data
	f_ephys: data file with ephys data
	session_template: a "standard session" matching the ptr.get_full_trials dataset format
		(but without timestamps)
Returns:
	X: data matrix with ephys data for individual trials that match the session template
"""
def standardize_session(f_behavior,f_ephys,session_template):
	pass



"""
A helper function to align event timestamps with session-relative values 
to be relative to individual trials.
Inputs:
	ts: an array of timestamps in the shape trials x (ts_1,...ts_n)
Returns:
	ts_rel an array of timestamps relative to the start of each trial
"""
def align_ts(ts):
	##alocate memory for the output
	ts_rel = np.zeros(ts.shape)
	for i in range(1,ts.shape[1]):
		ts_rel[:,i] = ts[:,i]-ts[:,0]
	return ts_rel

"""
A helper function to get some metadata about each session
Inputs:
	f_in: behavior data file
	max_duration: trial length threshold for exclusion
Returns:
	metadata: dictionary with some basic info
"""
def get_session_meta(f_behavior,max_duration=5000):
	##start by parsing the data
	data = ptr.get_full_trials(f_behavior,max_duration=max_duration)
	metadata = {
	'unrewarded':np.where(data['outcome']=='unrewarded_poke')[0],
	'rewarded':np.where(data['outcome']=='rewarded_poke')[0],
	'upper_lever':np.where(data['action']=='upper_lever')[0],
	'lower_lever':np.where(data['action']=='lower_lever')[0],
	'upper_context':np.where(data['context']=='upper_rewarded')[0],
	'lower_context':np.where(data['context']=='lower_rewarded')[0],
	}
	trial_info = ptr.parse_trial_data(data)
	metadata['n_blocks'] = trial_info['n_blocks']
	metadata['block_lengths'] = trial_info['block_lengths']
	metadata['first_block'] = data['context'][0]
	metadata['mean_block_len'] = np.mean(trial_info['block_lengths'])
	# metadata['reward_rate'] = np.mean([len(trial_info['upper_correct_rewarded'])/(
	# 	len(trial_info['upper_correct_rewarded'])+len(trial_info['upper_correct_unrewarded'])),
	# 	len(trial_info['lower_correct_rewarded'])/(len(trial_info['lower_correct_rewarded'])+len(
	# 		trial_info['lower_correct_unrewarded']))])
	return metadata

"""
Simple function to return the number of recorded units in a session
Inputs:
	f_ephys: data file to look in
Returns n_units (int)
"""
def get_n_units(f_ephys):
	f = h5py.File(f_ephys,'r')
	n_units = len([x for x in list(f) if x.startswith('sig')])
	f.close()
	return n_units

def get_session_duration(f_ephys):
	f = h5py.File(f_ephys,'r')
	AD_chans = [x for x in list(f) if x.endswith('_ts')]
	ad = AD_chans[0]
	duration = np.ceil(np.asarray(f[ad]).max()*1000).astype(int)
	f.close()
	return duration
"""
A dictionary of event pairs to use in analyses.
"""
event_pairs = {
	'context':['upper_context_lever','lower_context_lever'],
	'action':['upper_lever','lower_lever'],
	'outcome':['rewarded_poke','unrewarded_poke']
}



