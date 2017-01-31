##parse_trials.py
##functions to parse organized timestamp arrays,
##of the type returned by parse_timestamps.py functions

import numpy as np

"""
This function takes in a full array of trial timestamps,
and returns an array (in the same order) containing 
the start,stop values for windows around a particular epoch
in each trial.
Inputs:
	ts: array of timestamps, organized trials x (action,outcome)
	epoch_type: one of 'choice','action','delay','outcome'
	duration: duration, in seconds, of the epoch window
Returns: 
	-windows: an array of trials x (epoch start, epoch end)
"""
def get_epoch_windows(ts,epoch_type,duration):
	##data array to return
	windows = np.zeros((ts.shape))
	##we will need to take different windows depending
	##on which epoch we are interested in:
	if epoch_type == 'choice':
		windows[:,0] = ts[:,0]-duration
		windows[:,1] = ts[:,0]
	elif epoch_type == 'action':
		windows[:,0] = ts[:,0]-(duration/2.0)
		windows[:,1] = ts[:,0]+(duration/2.0)
	elif epoch_type == 'delay':
		windows[:,0] = ts[:,1]-duration
		windows[:,1] = ts[:,1]
	elif epoch_type == 'outcome':
		windows[:,0] = ts[:,1]
		windows[:,1] = ts[:,1]+duration
	else: ##case where epoch_type is unrecognized
		print "Unrecognized epoch type!"
		windows = None
	return windows

"""
A function to clean a timestamp index dictionary of all unrewarded trials
Inputs:
	ts_idx: timestamps dictionary
Returns:
	ts_rewarded: same sor of dictionary but with no unrewarded trials
"""
def remove_unrewarded(ts_idx):
	ts_rewarded = {} ##dict to return
	##get the labels of conditions not related to outcome
	conditions = [x for x in ts_idx.keys() if x != 'rewarded' and x != 'unrewarded']
	for c in conditions:
		new_idx = np.asarray([i for i in ts_idx[c] if i in ts_idx['rewarded']])
		ts_rewarded[c] = new_idx
	return ts_rewarded



"""
A function to convert timestamps in sec 
to timestamps in bins of x ms
Inputs:
	ts: an array of timestamps, in sec
	bin_size: size of bins, in ms
Returns:
	ts_bins: array of timestamps in terms of bins
"""
def ts_to_bins(ts,bin_size):
	##convert the timestamps (in sec) to ms:
	ts_bins = ts*1000.0 ##now in terms of ms
	##now divide by bins
	ts_bins = np.ceil(ts_bins/bin_size).astype(int)
	##make sure all the windows are the same size
	win_lens = ts_bins[:,1]-ts_bins[:,0]
	mean_len = np.round(win_lens.mean()).astype(int) ##this should be the window length in bins
	##if one of the windows is a different length, add or subtract a bin to make it equal
	i = 0
	while i < win_lens.shape[0]:
		win_lens = ts_bins[:,1]-ts_bins[:,0]
		if win_lens[i] != mean_len:
			diff = mean_len-win_lens[i]
			if diff > 0: ##case where the window is too short
				ts_bins[i,1]+diff
				print "Equalizing window by "+str(diff)+" bins"
			if diff < 0: ##case where the window is too long
				ts_bins[i,1] - diff
				print "Equalizing window by "+str(diff)+" bins"
		else:
			i+=1
	return ts_bins

##get the time between the action and the outcome (checking the nosepoke) 
def get_ao_interval(data):
	##diff between the lever and the nosepoke
	result = np.zeros(data.shape[0])
	for i in range(result.shape[0]):
		result[i]=abs(data[i,2])-abs(data[i,1])
		##encode the outcome
		if data[i,2] < 0:
			result[i] = -1.0*result[i]
	return result

