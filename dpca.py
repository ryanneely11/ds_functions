###dpca.py
##functions for implementing dPCA, using the library from Kobak et al, 2016

import numpy as np
# from dPCA import dPCA
import parse_timestamps as pt
import parse_ephys as pe
import collections
from scipy import interpolate
from scipy.stats import zscore

##some global variables that are specific to my current project
condition_pairs = [
('upper_rewarded','lower_rewarded'),
('upper_lever','lower_lever')]

conditions = [
'block_type',
'choice']

condition_LUT = collections.OrderedDict()
condition_LUT['t'] = "Independent"
condition_LUT['bt'] = "Blocktype"
condition_LUT['ct'] = "Choice"
condition_LUT['cbt'] = "Interaction"

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
	remove_unrew: if True, excludes trials that were correct but unrewarded.
Returns:
	X_mean: data list of shape n-neurons x condition-1 x condition-2, ... x n-timebins
	X_trials: data from individual trials: n-trials x n-neurons x condition-1, condition-2, ... x n-timebins.
		If data is unbalanced (ie different #'s of trials per condition), max dimensions are used and empty
		spaces are filled with NaN
"""

def get_dataset(f_behavior,f_ephys,smooth_method='both',smooth_width=[40,50],
	pad=[200,200],z_score=True,remove_unrew=True):
	##use global parameters here
	global condition_pairs
	###most of the work is done by our other function. Get the data separated from all
	##trials, and the indices of trials from each condition:
	X_all,ts_idx = split_trials(f_behavior,f_ephys,smooth_method=smooth_method,
		smooth_width=smooth_width,pad=pad,z_score=z_score,
		timestretch=True,remove_unrew=remove_unrew)
	##now the data is in dims trials x neurons x time
	##get some meta about this data:
	n_neurons = X_all.shape[1]
	n_bins = X_all.shape[2]
	##the max number of trials in any condition pair
	n_trials = 0
	for val in ts_idx.values():
		if val.shape[0] > n_trials:
			n_trials = val.shape[0]
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
	pad=[200,200],z_score=False,timestretch=False,remove_unrew=False):
	##get the raw data matrix first 
	X_raw = pe.get_spike_data(f_ephys,smooth_method='none',smooth_width=None,
		z_score=False) ##don't zscore or smooth anything yet
	##now get the window data for the trials in this session
	ts,ts_idx = pt.get_trial_data(f_behavior,remove_unrew=remove_unrew) #ts is shape trials x ts, and in seconds
	##now, convert to ms and add padding 
	ts = ts*1000.0
	trial_wins = ts.astype(int) ##save the raw ts for later, and add the padding to get the full trial windows
	trial_wins[:,0] = trial_wins[:,0] - pad[0]
	trial_wins[:,1] = trial_wins[:,1] + pad[1]
	##convert this array to an integer so we can use it as indices
	trial_wins = trial_wins.astype(int)
	##now get the windowed data around the trial times. Return value is a list of the trials
	X_trials = pe.X_windows(X_raw,trial_wins)
	##now do the smoothing, if requested
	if smooth_method != 'none':
		for t in range(len(X_trials)):
			X_trials[t] = pe.smooth_spikes(X_trials[t],smooth_method,smooth_width)
	##now, do timestretching, if requested
	if timestretch:
		##first step is to get the proper timestamps.
		##we want to use the original ts, because they contain actual event times and not trial start/stop
		##if data is binned, then we need the timestamps in terms of bins
		ts_rel = get_relative_ts(ts,pad,smooth_method,smooth_width)
		##now run the data through the function
		X_trials = stretch_trials(X_trials,ts_rel)
	##now z-score, if requested
	if z_score:
		X_trials = zscore_across_trials(X_trials)
	return X_trials, ts_idx


"""
This function actually runs dpca, relying on some globals for 
the details. ***NOTE: there are some hard-to-avoid elements here that are
			hardcoded specificallyfor this dataset.*****
Inputs:
	X_mean: array of data averaged over trials
	X_trials: array of data including individual trial data
"""
def run_dpca(X_mean,X_trials,n_components):
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
	Z = dpca.fit_transform(X_mean,X_trials)
	##Next, get the variance explained:
	var_explained = dpca.explained_variance_ratio_
	##finally, get the significance masks (places where the demixed components are significant)
	sig_masks = dpca.significance_analysis(X_mean,X_trials,axis='t',
		n_shuffles=100,n_splits=10,n_consecutive=2)
	return Z,var_explained,sig_masks	

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
def stretch_trials(X_trials,ts,median_dur=None):
	##determine how many events we have
	n_events = ts.shape[1]
	##and how many trials
	n_trials = len(X_trials)
	##and how many neurons
	n_neurons = 0
	for i in range(len(X_trials)):
		if X_trials[i].shape[0] > n_neurons:
			n_neurons = X_trials[i].shape[0]
	pieces = [] ##this will be a list of each streched epoch piece
	##check to see if the first event is aligned to the start of the data,
	##or if there is some pre-event data included.
	if not np.all(ts[:,0]==0):
		##make sure each trial is padded the same amount
		if np.all(ts[:,0]==ts[0,0]):
			pad1 = ts[0,0] ##this should be the pre-event window for all trials
			##add this first piece to the collection
			data = np.empty((n_trials,n_neurons,pad1))
			data[:] = np.nan
			for t in range(n_trials):
				data[t,0:X_trials[t].shape[0],:] = X_trials[t][:,0:pad1]
			pieces.append(data)
		else:
			print("First event is not aligned for all trials")
	##do the timestretching for each event epoch individually
	for e in range(1,n_events):
		##get just the interval for this particular epoch
		epoch_ts = ts[:,e-1:e+1]
		##now get the median duration of this epoch for all trials as an integer (if not specified) 
		if median_dur is None:
			median_dur = int(np.ceil(np.median(epoch_ts[:,1]-epoch_ts[:,0])))
		data_new = interp_trials(X_trials,epoch_ts,median_dur)
		pieces.append(data_new)
	##finally, see if the ephys data has any padding after the final event
	##collect the differences between the last timestamp of each trial and the trial length
	t_diff = np.zeros(n_trials)
	for i in range(n_trials):
		t_diff[i] = X_trials[i].shape[1]-ts[i,-1]
	if not np.all(t_diff<=1):
		##make sure padding is equal for all trials
		if np.all(t_diff==t_diff[0]):
			pad2 = t_diff[0]
			data = np.zeros((n_trials,n_neurons,pad2))
			for t in range(n_trials):
				data[t,0:X_trials[t].shape[0],:] = X_trials[t][:,-pad2:]
			pieces.append(data)
		else:
			print("Last event has uneven padding")
			pad2 = np.floor(t_diff[0]).astype(int)
			data = np.zeros((n_trials,n_neurons,pad2))
			for t in range(n_trials):
				data[t,0:X_trials[t].shape[0],:] = X_trials[t][:,-pad2:]
			pieces.append(data)
	##finally, concatenate everything together!
	X = np.concatenate(pieces,axis=2)
	return X

"""
A helper function that does interpolation on one trial.
Inputs:
	data: list of trial data to work with; each trial should be neurons x bins
	epoch_ts: timestamps for each trial of the epoch to interpolate over
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
	data_new = np.zeros((len(data),n_neurons[0],xnew.shape[0]))
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


