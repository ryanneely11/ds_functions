##session_analysis.py
## a function to run various analyses on full sessions 

import numpy as np
import parse_timestamps as pt
import parse_trials as ptr
import parse_ephys as pe
import regression as re
import file_lists
import log_regression as lr
import os
import h5py

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
	conditions = cond_idx.keys()
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
def log_regress_session(f_behavior,f_ephys,epoch_durations=[1,0.4,1,1],smooth_method='bins',
	smooth_width=200,z_score=True,save=True):
	##define some parameters
	##specify a bin size based on input arguments
	if smooth_method == 'bins':
		bin_size = smooth_width
	else:
		bin_size = 1 ##'gauss' and 'none' options use a bin width of 1 ms
	epoch_list = ['choice','action','delay','outcome'] ##the different trial epochs
	condition_list = ['choice','bock_type','reward'] ##the different conditions to predict
	results = {} ##the return dictionary
	##create the file to save if requested
	if save:
		current_file = f_behavior[-11:-5]
		out_path = os.path.join(file_lists.save_loc,current_file+".hdf5")
		f_out = h5py.File(out_path,'w-')
	##start by getting the behavioral data,and put it into an array.
	##we will use the old regression functions to get the values we need.
	ts,ts_idx = pt.get_trial_data(f_behavior) ##trial timestamps and indices
	R = re.regressors_model1(ts_idx) ##the regressors matrix,using the old function for linreg
	Y = np.zeros((3,R.shape[0])) ##the new matrix with the conditions that we care about
	Y[0,:] = lr.binary_y(R[:,0]) ##the upper or lower lever press choice
	Y[1,:] = R[:,1] ##the outcome (rewarded or unrewarded)
	Y[2,:] = lr.binary_y(R[:,3]) ##the block type (upper_rewarded = 1, lower_rewarded = 0)
	##now get the epys data for the full session
	X = pe.get_spike_data(f_ephys,smooth_method=smooth_method,
		smooth_width=smooth_width,z_score=z_score)
	##run through each epoch
	for epoch,duration in zip(epoch_list,epoch_durations):
		ts_e = ptr.get_epoch_windows(ts,epoch,duration) ##the timestamp windows for this epoch
		ts_e = ptr.ts_to_bins(ts_e,bin_size)
		##now the spike data for this epoch
		Xe = pe.X_windows(X,ts_e)
		##now create a sub-dictionary for this epoch
		results[epoch] = {}
		##also create a group in the datafile
		if save:
			f_out.create_group(epoch)
		##now predict different conditions for this epoch
		for n, condition in enumerate(condition_list):
			y = Y[n,:]
			sig_idx = lr.regress_array(Xe,y)
			##save this data in the results dictionary
			results[epoch][condition] = sig_idx
			##save in the file
			if save:
				f_out[epoch].create_dataset(condition,data=sig_idx)
	if save:
		f_out.close()
	return results