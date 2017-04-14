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

save_root = os.path.join(file_lists.save_loc,"LogisticRegression/50ms_bins_0.05")

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
a function to run regression analyses over 
all epochs in a single session. 
Inputs:
	-f_behavior: hdf5 file with behavioral data
	-f_ephys: hdf5 file with ephys data
	-epoch_durations: the duration, in seconds, of the epochs (see the list in the function)
	-smooth_method: type of smoothing to use; choose 'bins', 'gauss', or 'none'
	-smooth_width: size of the bins or gaussian kernel in ms
	-z_score: if True, z-scores the array
			$$NOTE$$: this implementatin does not allow binning AND gaussian smoothing.
Returns:
	-C: matrix of coefficients, shape units x regressors x bins (all epochs are concatenated)
	-num_sig: matrix with the counts of units showing significant regression
		values at for each regressor at each bin (regressors x bins)
	-mse: mean squared error of the model fit at each bin 
		based on x-validation (size = bins)
	-epoch_idx: the indices of bins corresponding to the different epochs
"""
def session_regression(f_behavior,f_ephys,epoch_durations=[1,0.5,1,1],
	smooth_method='bins',smooth_width=50,z_score=False):
	##specify a bin size based on input arguments
	if smooth_method == 'bins':
		bin_size = smooth_width
	else:
		bin_size = 1 ##'gauss' and 'none' options use a bin width of 1 ms
	##a list of the different epochs
	epoch_list = ['choice','action','delay','outcome']
	##get the behavior and regressor values
	ts,ts_idx = pt.get_trial_data(f_behavior) ##trial timestamps and indices
	R = re.regressors_model1(ts_idx) ##the regressors matrix
	##get the ephys data
	X = pe.get_spike_data(f_ephys,smooth_method=smooth_method,
		smooth_width=smooth_width,z_score=z_score) ##the spike data for the full trial
	##storage for results
	C = []
	num_sig = []
	mse = []
	epoch_idx = {}
	cursor = 0 ##keep track of where we are in the trial
	##go through each epoch and get the data
	for epoch, dur in zip(epoch_list,epoch_durations):
		##get the by-trial windows for this epoch
		ts_e = ptr.get_epoch_windows(ts,epoch,dur)
		##convert these ts to bins
		ts_e = ptr.ts_to_bins(ts_e,bin_size)
		##get the windowed data for this epoch for all trials
		Xe = pe.X_windows(X,ts_e)
		##run the regression and get the results
		c,ns,m = re.regress_matrix(Xe,R)
		##put the data in it's place
		C.append(c)
		num_sig.append(ns)
		mse.append(m)
		epoch_idx[epoch] = np.arange(cursor,c.shape[2]+cursor) ##number of bins in this epoch+where the last epoch ended
		cursor += c.shape[2]
	##concatenate all the arrays
	C = np.concatenate(C,axis=2)
	num_sig = np.concatenate(num_sig,axis=2)
	mse = np.concatenate(mse)
	return C,num_sig,mse,epoch_idx

"""
A function to get the condition-averaged responses
for each unit in a session. In other words, we are
getting the average responses for each unit in a session
over all trials in a given condition. There are overlaps
between conditions- ie 'upper_rewarded' trials can belong to
'upper_lever or 'lower_lever' choice conditions. Also at this point
I'm not sure whether to include unrewarded (incorrect) trials or not.
Inputs:
	-f_behavior: file path to behavioral data
	=f_ephys: file path to ephys data
	-eopch: (str), name of epoch to use for trial data
	-epoch_duration: duration of epoch to look at (in sec)
	-smooth_method: smooth method to use; see ephys functions
	-smooth_width: smooth width, lol
	-use_unrewarded: bool, whether to include or exclude unrewarded trials.
		if included, these become their own condition.
Returns: 
	X. data matrix of size conditions x units x bins
	order: list of strings defining the order of the conditions in the matrix
"""
def condition_averaged_responses(f_behavior,f_ephys,epoch='choice',epoch_duration=1,
	smooth_method='bins',smooth_width=50,use_unrewarded=True):
		##specify a bin size based on input arguments
	if smooth_method == 'bins':
		bin_size = smooth_width
	else:
		bin_size = 1 ##'gauss' and 'none' options use a bin width of 1 ms
	ts,cond_idx = pt.get_trial_data(f_behavior) ##trial timestamps and  condition indices
	if not use_unrewarded: ##get rid of unrewarded trials if requested
		cond_idx = ptr.remove_unrewarded(cond_idx)
	##the list of conditions
	conditions = list(cond_idx)
	##get the ephys data. 
	X = pe.get_spike_data(f_ephys,smooth_method=smooth_method,
		smooth_width=smooth_width,z_score=False) ##the spike data for the full trial
	##the output data
	Xc = []
	##get the data for each unit and each condition
	for c in conditions:
		ts_c = ts[cond_idx[c],:] ##the timestamps for only this condition
		##get the by-trial windows for this set of trials for the given epoch
		wins = ptr.get_epoch_windows(ts_c,epoch,epoch_duration)
		##convert these ts to bins
		wins = ptr.ts_to_bins(wins,bin_size)
		##get the windowed data for this epoch for all trials
		x = pe.X_windows(X,wins)
		##add the mean taken across trials
		Xc.append(x.mean(axis=0))
	return np.asarray(Xc),conditions


""" 
A function to do logistic regression on unit data from a session,
incuding multiple intervals and behavioral parameters.
Behavioral conditions to be tested:
	1) top lever (1) or bottom lever (0)
	2) top_rewarded block (1) VS bottom rewarded block (0)
	3) rewarded action (1) or unrewarded action (0)
Trial intervals to be tested:
	1) pre_action
	2) peri-action
	3)"delay" period between action and outcome
	4) post-outcome
Inputs:
	-f_behavior: hdf5 file with behavioral data
	-f_ephys: hdf5 file with ephys data
	-epoch_durations: the duration, in seconds, of the epochs (see the list in the function)
	-smooth_method: type of smoothing to use; choose 'bins', 'gauss', or 'none'
	-smooth_width: size of the bins or gaussian kernel in ms
	-z_score: if True, z-scores the array
	-save: if True, saves data to the results folder
Returns:
	results: dictionary containing significant index values for each epoch and condition
"""
# def log_regress_session(f_behavior,f_ephys,epoch_durations=[1,0.4,1,1],smooth_method='bins',
# 	smooth_width=200,z_score=True,save=True):
# 	##define some parameters
# 	##specify a bin size based on input arguments
# 	if smooth_method == 'bins':
# 		bin_size = smooth_width
# 	else:
# 		bin_size = 1 ##'gauss' and 'none' options use a bin width of 1 ms
# 	epoch_list = ['choice','action','delay','outcome'] ##the different trial epochs
# 	condition_list = ['choice','block_type','reward'] ##the different conditions to predict
# 	results = {} ##the return dictionary
# 	##create the file to save if requested
# 	if save:
# 		current_file = f_behavior[-11:-5]
# 		out_path = os.path.join(file_lists.save_loc,current_file+".hdf5")
# 		f_out = h5py.File(out_path,'w-')
# 	##start by getting the behavioral data,and put it into an array.
# 	##we will use the old regression functions to get the values we need.
# 	ts,ts_idx = pt.get_trial_data(f_behavior) ##trial timestamps and indices
# 	R = re.regressors_model1(ts_idx) ##the regressors matrix,using the old function for linreg
# 	Y = np.zeros((3,R.shape[0])) ##the new matrix with the conditions that we care about
# 	Y[0,:] = lr.binary_y(R[:,0]) ##the upper or lower lever press choice
# 	Y[1,:] = R[:,1] ##the outcome (rewarded or unrewarded)
# 	Y[2,:] = lr.binary_y(R[:,3]) ##the block type (upper_rewarded = 1, lower_rewarded = 0)
# 	##now get the epys data for the full session
# 	X = pe.get_spike_data(f_ephys,smooth_method=smooth_method,
# 		smooth_width=smooth_width,z_score=z_score)
# 	##run through each epoch
# 	for epoch,duration in zip(epoch_list,epoch_durations):
# 		ts_e = ptr.get_epoch_windows(ts,epoch,duration) ##the timestamp windows for this epoch
# 		ts_e = ptr.ts_to_bins(ts_e,bin_size)
# 		##now the spike data for this epoch
# 		Xe = pe.X_windows(X,ts_e)
# 		##now create a sub-dictionary for this epoch
# 		results[epoch] = {}
# 		##also create a group in the datafile
# 		if save:
# 			f_out.create_group(epoch)
# 			##save the raw data here 
# 			f_out[epoch].create_dataset("X",data=Xe)
# 		##now predict different conditions for this epoch
# 		for n, condition in enumerate(condition_list):
# 			y = Y[n,:]
# 			sig_idx = lr.regress_array(Xe,y)
# 			##save this data in the results dictionary
# 			results[epoch][condition] = sig_idx
# 			##also get the strength of a the unit's prediction for this epoch and cond
# 			pred_strength = lr.matrix_pred_strength(Xe,y)
# 			##save in the file
# 			if save:
# 				group = f_out[epoch].create_group(condition)
# 				group.create_dataset("sig_idx",data=sig_idx)
# 				group.create_dataset("y",data=y)
# 				group.create_dataset("pred_strength",data=pred_strength)
# 	if save:
# 		f_out.close()
# 	return results


"""
A function to parse log regression results. For now, it will do two things:
1) determine the proportion of units distinguishing between each of the three
	behavioral parameters (choice, block type, and reward), pooling all behavioral epochs
2) determine the number of units that have overlapping representations, again pooling all epochs
Inputs:
	f_in: file name of log regression data file
	epochs: optional list; if specified, only takes data from a given epoch or epochs. 
		Otherwise data from all epochs is pooled.  
REturns:
	cond_idx: the indices of units that differentiate between different parameters
	cond_kappas: the prediction quality values of significant units
	multi_units: list of indices of units that encode multiple parameters
	all_sig: the full list of significant unit indices
	total_units: (int), the total number of units for this session
"""
def parse_log_regression(f_in,epochs=None):
	##open the data file
	f = h5py.File(f_in,'r')
	##let's start by pooling data from all epochs
	if epochs is None:
		epochs = list(f)
	conditions = [x for x in list(f[epochs[0]]) if x != 'X'] ##the data matrix is also stored here
	##let's make a dictionary of indices for each condition, pooling across epochs
	##for the hell of it we'll also keep info about the prediction strengths
	cond_idx = {}
	cond_kappas = {}
	##keep track all the unique significant units
	all_sig = []
	n_total = 0 ## the total number of units
	for c in conditions:
		cond_idx[c] = []
		cond_kappas[c] = []
	for e in epochs:
		for c in conditions:
			sig_idx = np.asarray(f[e][c]['sig_idx'])
			ps = np.asarray(f[e][c]['pred_strength'])[sig_idx] ##make sure to only take the sig values!
			n_total = np.asarray(f[e][c]['pred_strength']).size
			##add this data to the dictionary
			for p in ps:
				cond_kappas[c].append(p)
			for i in sig_idx:
				if i not in cond_idx[c]: ##don't add if it's already listed
					cond_idx[c].append(i)
				if i not in all_sig:
					all_sig.append(i)
	##now convert the lists to arrays
	for c in conditions:
		cond_idx[c] = np.asarray(cond_idx[c])
		cond_kappas[c] = np.asarray(cond_kappas[c])
	##now let's determine which units represent more than one task variable
	#********NOTE: this last step assumes 3 conditions, will need to be changed if there are more or less!!!****
	assert len(conditions) == 3
	multi_units = reduce(np.intersect1d,(cond_idx[conditions[0]],cond_idx[conditions[1]],
		cond_idx[conditions[2]]))
	return cond_idx,cond_kappas,multi_units,np.asarray(all_sig),n_total




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
A dictionary of event pairs to use in analyses.
"""
event_pairs = {
	'context':['upper_context_lever','lower_context_lever'],
	'action':['upper_lever','lower_lever'],
	'outcome':['rewarded_poke','unrewarded_poke']
}



