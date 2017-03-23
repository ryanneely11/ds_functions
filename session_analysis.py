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
from functools import reduce

"""
A function to get a data array of trials for a given session.
Resulting data array is in the order n_trials x n_neurons x n_timpoints.
Also returned is a ductionary with information about trial types.
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
	X: data list of shape n-trials x n-neurons x n-timebins
	ts_idx: dictionary indicating which subsets of trials belong to various trial types
"""
def split_trials(f_behavior,f_ephys,smooth_method='bins',smooth_width=50,
	pad=[200,200],z_score=False,timestretch=False):
	##get the raw data matrix first 
	X_raw = pe.get_spike_data(f_ephys,smooth_method='none',smooth_width=None,
		z_score=False) ##don't zscore or smooth anything yet
	##now get the window data for the trials in this session
	ts,ts_idx = pt.get_trial_data(f_behavior) #ts is shape trials x ts, and in seconds
	##now, convert to ms and add padding 
	ts = ts*1000.0
	trial_wins = ts ##save the raw ts for later, and add the padding to get the full trial windows
	trial_wins[:,0] = trial_wins[:,0] - pad[0]
	trial_wins[:,1] = trial_wins[:,1] + pad[0]
	##now get the windowed data around the trial times. Return value is a list of the trials
	X_trials = pe.X_windows(X_raw,ts)
	##now do the smoothing, if requested
	if smooth_method != 'none':
		for t in range(len(X_trials)):
			X_trials[t] = pe.smooth_spikes(X_trials[t],smooth_method,smooth_width)
	##now, do timestretching, if requested
	if timestretch:
		pass
"""
#####################################################################
###########TODO##############
##need to decide when and how to do smoothing. Interpolation will probably work better
on smoothed data, but if I smooth the full session data including binning, then the trial
timestamps lose some accuracy and taking the data windows will result in some jitter.

"""
	##now z-score, if requested
	if z_score:
		for a in range(len(X_trials)):
			X_trials[a] = zscore(X_trials[a])
	return X_trials




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
	condition_list = ['choice','block_type','reward'] ##the different conditions to predict
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
			##save the raw data here 
			f_out[epoch].create_dataset("X",data=Xe)
		##now predict different conditions for this epoch
		for n, condition in enumerate(condition_list):
			y = Y[n,:]
			sig_idx = lr.regress_array(Xe,y)
			##save this data in the results dictionary
			results[epoch][condition] = sig_idx
			##also get the strength of a the unit's prediction for this epoch and cond
			pred_strength = lr.matrix_pred_strength(Xe,y)
			##save in the file
			if save:
				group = f_out[epoch].create_group(condition)
				group.create_dataset("sig_idx",data=sig_idx)
				group.create_dataset("y",data=y)
				group.create_dataset("pred_strength",data=pred_strength)
	if save:
		f_out.close()
	return results


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
		epochs = f.keys()
	conditions = [x for x in f[epochs[0]].keys() if x != 'X'] ##the data matrix is also stored here
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
A function to equate the lengths of trials by using a piecewise
linear stretching procedure, outlined in 

"Kobak D, Brendel W, Constantinidis C, et al. Demixed principal component analysis of neural population data. 
van Rossum MC, ed. eLife. 2016;5:e10989. doi:10.7554/eLife.10989."

Inputs:
	X_trials: a list containing ephys data from a variety of trials. This function assumes
		that trials are all aligned to the first event. data for each trial should be cells x timebins
	ts: array of timestamps used to generate the ephys trial data. 
		***These timestamps should be relative to the start of each trial. 
			for example, if a lever press happens 210 ms into the start of trial 11,
			the timestamp for that event should be 210.***

"""
def stretch_trials(X_trials,ts):
	##determine how many events we have
	n_events = ts.shape[1]
	##and how many trials
	n_trials = len(X_trials)
	##and how many neurons
	n_neurons = X_trials[0].shape[0]
	pieces = [] ##this will be a list of each streched epoch piece
	##check to see if the first event is aligned to the start of the data,
	##or if there is some pre-event data included.
	if not np.all(ts[:,0]==0):
		##make sure each trial is padded the same amount
		if np.all(ts[:,0]==ts[0,0]):
			pad1 = ts[0,0] ##this should be the pre-event window for all trials
			##add this first piece to the collection
			data = np.zeros((n_trials,n_neurons,pad1))
			for t in range(n_trials):
				data[t,:,:] = X_trials[t][:,0:pad1]
			pieces.append(data)
		else:
			print "First event is not aligned for all trials"
	##do the timestretching for each event epoch individually
	for e in range(1,n_events):
		##get just the interval for this particular epoch
		epoch_ts = ts[:,e-1:e+1]
		##now get the median duration of this epoch for all trials as an integer. 
		median_dur = int(np.ceil(np.median(epoch_ts[:,1]-epoch_ts[:,0])))
		xnew = np.arange(median_dur) ##this will be the timebase of the interpolated trials
		data = np.zeros((n_trials,n_neurons,xnew.shape[0]))
		##now operate on each trial, over this particular epoch
		for t in range(n_trials):
			##get the actual data for this trial
			trial_data = X_trials[t]
			##now, trial_data is in the shape units x bins.
			##we need to interpolate data from each unit individually:
			for n in range(trial_data.shape[0]):
				##get the data for neuron n in trial t and epoch e
				y = trial_data[n,epoch_ts[t,0]:epoch_ts[t,1]]
				x = np.arange(y.shape[0])
				##create an interpolation object for these data
				f = interpolate.interp1d(x,y,bounds_error=False,fill_value='extrapolate')
				##now use this function to interpolate the data into the 
				##correct size
				ynew = f(xnew)
				##now put the data into its place
				data[t,n,:] = ynew
		pieces.append(data)
	##finally, see if the ephys data has any padding after the final event
	##collect the differences between the last timestamp of each trial and the trial length
	t_diff = np.zeros(n_trials)
	for i in range(n_trials):
		t_diff[i] = X_trials[i].shape[1]-ts[i,-1]
	if not np.all(t_diff<=1):
		##make sure padding is equal for all trials
		if np.all(t_diff==t_diff[0]):
			pad2 = tdiff[0]
			data = np.zeros((n_trials,n_neurons,pad2))
			for t in range(n_trials):
				data[t,:,:] = X_trials[t][:,-pad2:]
			pieces.append(data)
		else:
			print "Last event has uneven padding"
			pad2 = np.floor(tdiff[0]).astype(int)
			data = np.zeros((n_trials,n_neurons,pad2))
			for t in range(n_trials):
				data[t,:,:] = X_trials[t][:,-pad2:]
				pieces.append(data)
	##finally, concatenate everything together!
	X = np.concatenate(pieces,axis=2)
	return X


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