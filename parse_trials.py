##parse_trials.py
##functions to parse organized timestamp arrays,
##of the type returned by parse_timestamps.py functions

import numpy as np
import parse_timestamps as pt

"""
A function to look at the behavior around reversals.
Inputs:
	f_behavior: data file to get timestamps from
	f_behavior_last: last session datafile, optional. Will consider
		a switch from last session block to this session block
	window: window in trials to look at. [n_trials_pre,n_trials_post]
Returns:
	U-L: array of event ids from upper to lower switches, size n_pre+n_post,
		centered arround the switch
	L-U: array of event ids from lower to upper switches
"""
def get_reversals(f_behavior,f_behavior_last=None,window=[30,30]):
	##parse the data 
	data = pt.get_event_data(f_behavior)
	##parse data form the last session if requested
	last_block_type = None
	if f_behavior_last is not None:
		data_last = pt.get_event_data(f_behavior_last)
		##figure out which block was last
		if data_last['upper_rewarded'].max()>data_last['lower_rewarded'].max():
			last_block_type = 'upper_rewarded'
			last_block_ts = data['upper_rewarded'].max()
		elif data_last['lower_rewarded'].max()>data_last['upper_rewarded'].max():
			last_block_type = 'lower_rewarded'
			last_block_ts = data['lower_rewarded'].max()
		else:
			print("Unrecognized block type")
	##get all the timestamps of the reversals for the current session
	reversals = np.concatenate((data['upper_rewarded'],
		data['lower_rewarded'])) ##first one is the start, not a "reversal"
	##make an array to keep track of the block IDs
	block_ids = np.empty(reversals.size,dtype='<U14')
	block_ids[0:data['upper_rewarded'].size]=['upper_rewared']
	block_ids[data['upper_rewarded'].size:] = ['lower_rewarded']
	##now arrange temporally
	idx = np.argsort(reversals)
	reversals = reversals[idx]
	block_ids = block_ids[idx]
	##now get the data for correct and incorrect presses
	trials = np.concatenate((data['correct_lever'],data['incorrect_lever']))
	ids = np.empty(trials.size,dtype='<U15')
	ids[0:data['correct_lever'].size]=['correct_lever']
	ids[data['correct_lever'].size:] = ['incorrect_lever']	
	idx = np.argsort(trials)
	trials = trials[idx]
	ids = ids[idx]
	##check and see if we will use data from the previous session
	if last_block_type != None and last_block_type != block_ids[0]:
		##a switch happened from last session to this session
		##now we compute the first reversal from last block to this block
		##start by getting the last n-trials from the last session
		last_trials = np.concatenate((data_last['correct_lever'],data_last['incorrect_lever']))
		last_ids = np.empty(last_trials.size,dtype='<U15')
		last_ids[0:data_last['correct_lever'].size]=['correct_lever']
		last_ids[data_last['correct_lever'].size:] = ['incorrect_lever']
		##now arrange temporally
		idx = np.argsort(last_trials)
		last_trials = last_trials[idx]
		last_ids = last_ids[idx]
		##get the timestamps in terms of the start of the current session
		last_trials = last_trials-data_last['session_length'][0]
		last_block_ts = np.array([last_block_ts-data_last['session_length'][0]])
		##finally, add all of these data to the data from the current session
		reversals = np.concatenate((last_block_ts,reversals))
		trials = np.concatenate((last_trials,trials))
		ids = np.concatenate((last_ids,ids))
		##pad the ids and trials in case our window exceeds their bounds
		pad = np.empty(window[0])
		pad[:] = np.nan
		trials = np.concatenate((pad,trials,pad))
		ids = np.concatenate((pad,ids,pad))
	##now we have everything we need to start parsing reversals
	##a container to store the data
	reversal_data = np.zeros((reversals.size-1,window[0]+window[1]))
	for r in range(1,reversals.size): ##ignore the first one, which is the start point
		rev_ts = reversals[r] ##the timestamp of the reversal
		##figure out where this occurs in terms of trial indices
		rev_idx = np.nanargmax(trials>rev_ts)
		##whatever the correct lever is in the pre-switch period, this is our lever 1
		for i in range(window[0]):
			if ids[rev_idx-(window[0]-i)] == 'correct_lever':
				reversal_data[r-1,i] = 1
			elif ids[rev_idx-(window[0]-i)] == 'incorrect_lever':
				reversal_data[r-1,i] = 2
			else:
				print("unrecognized lever type")
		##now in the post-switch period, lever 1 is the incorrect lever
		for i in range(window[0],window[0]+window[1]):
			if ids[rev_idx+(i-window[0])] == 'correct_lever':
				reversal_data[r-1,i] = 2
			elif ids[rev_idx-(window[0]-i)] == 'incorrect_lever':
				reversal_data[r-1,i] = 1
			else:
				print("unrecognized lever type")
	return reversal_data








"""
A function to compute "persistence". Here, I'm defining that
as the liklihood that an animal switches levers after getting a 
rewarded trial on one lever.
Inputs:
	f_behavior: path to a behavior data file.
"""
def get_persistence(f_behavior):
	##parse the data
	data = pt.get_event_data(f_behavior)
	##concatenate all of the upper and lower trials
	ts = np.concatenate((data['upper_lever'],data['lower_lever']))
	##get an array of ids for upper and lower trials
	ids = np.empty(ts.size,dtype='object')
	ids[0:data['upper_lever'].size] = ['upper_lever']
	ids[data['upper_lever'].size:] = ['lower_lever']
	##now sort
	idx = np.argsort(ts)
	ts = ts[idx]
	ids = ids[idx]
	##now get all of the rewarded lever timestamps
	rew_ts = data['rewarded_lever']
	##now, each of these corresponds to an upper or lower lever ts.
	##figure out which ones:
	rew_idx = np.in1d(ts,rew_ts)
	rew_idx = np.where(rew_idx==True)[0]
	##now we know every place in the ids where a press was rewarded.
	##for each of these, ask if the next press was the same or different:
	n_switches = 0
	for trial in range(rew_idx.size-1):
		rew_action = ids[rew_idx[trial]]
		next_action = ids[rew_idx[trial+1]]
		if rew_action != next_action:
			n_switches += 1
	##get the percentage of time that he switched after n rewarded trial
	persistence = (n_switches/float(rew_idx.size))
	return persistence

"""
A function to compute "volatility". Here, I'm defining that
as the liklihood that an animal switches levers after getting an 
unrewarded trial on one lever.
Inputs:
	f_behavior: path to a behavior data file.
"""
def get_volatility(f_behavior):
	##parse the data
	data = pt.get_event_data(f_behavior)
	##concatenate all of the upper and lower trials
	ts = np.concatenate((data['upper_lever'],data['lower_lever']))
	##get an array of ids for upper and lower trials
	ids = np.empty(ts.size,dtype='object')
	ids[0:data['upper_lever'].size] = ['upper_lever']
	ids[data['upper_lever'].size:] = ['lower_lever']
	##now sort
	idx = np.argsort(ts)
	ts = ts[idx]
	ids = ids[idx]
	##now get all of the unrewarded lever timestamps
	unrew_ts = data['unrewarded_lever']
	##now, each of these corresponds to an upper or lower lever ts.
	##figure out which ones:
	unrew_idx = np.in1d(ts,unrew_ts)
	unrew_idx = np.where(unrew_idx==True)[0]
	##now we know every place in the ids where a press was unrewarded.
	##for each of these, ask if the next press was the same or different:
	n_switches = 0
	for trial in range(unrew_idx.size-1):
		unrew_action = ids[unrew_idx[trial]]
		next_action = ids[unrew_idx[trial+1]]
		if unrew_action != next_action:
			n_switches += 1
	##get the percentage of time that he switched after an unrewarded trial
	volatility = n_switches/float(unrew_idx.size)
	return volatility


"""
This function calculates the mean number of trials to reach criterion after
a behavioral switch. If the session has more than one switch, it's the average
of the two. 
Inputs:
	f_behavior: path to the data file to use for analysis
	crit_trials: the number of trials to average over to determine performance
	crit_level: the criterion performance level to use
	exclude_first: whether to exclude the first block, but only if there is more than one block.
Returns:
	mean_trials: the number of trials to reach criterion after a context switch
"""
def mean_trials_to_crit(f_behavior,crit_trials,crit_level,exclude_first=False):
	##parse data first
	data = pt.get_event_data(f_behavior)
	##concatenate all the block switches together
	switches = np.concatenate((data['upper_rewarded'],data['lower_rewarded']))
	if switches.size>1 and exclude_first == True:
		switches = switches[1:]
	##get the trials to criteria for all switches in the block
	n_trials = np.zeros(switches.size)
	for i in range(switches.size):
		##get the trials to criterion for this block switch
		n_trials[i] = trials_to_crit(switches[i],data['correct_lever'],
			data['incorrect_lever'],crit_trials,crit_level)
	return np.nanmean(n_trials)


"""
This is a simple function to return the percent
correct actions over all actions for a given behavioral session.
Inputs:
	-f_behavior: file path to the hdf5 file with the behavior data
Returns:
	-p_correct: the percent correct over the whole session
"""
def calc_p_correct(f_behavior):
	##first parse the data
	data = pt.get_event_data(f_behavior)
	p_correct = data['correct_lever'].shape[0]/(data['correct_lever'].shape[
		0]+data['incorrect_lever'].shape[0])
	return p_correct


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
		print("Unrecognized epoch type!")
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
	conditions = [x for x in list(ts_idx) if x != 'rewarded' and x != 'unrewarded']
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
		diff = 0
		win_lens = ts_bins[:,1]-ts_bins[:,0]
		if win_lens[i] != mean_len:
			diff = mean_len-win_lens[i]
			if diff > 0: ##case where the window is too short
				ts_bins[i,1] = ts_bins[i,1] + abs(diff)
				print("Equalizing window by "+str(diff)+" bins")
			if diff < 0: ##case where the window is too long
				ts_bins[i,1] = ts_bins[i,1] - abs(diff)
				print("Equalizing window by "+str(diff)+" bins")
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

"""
A function which determines how many trials it takes before
an animal reaches some criterion performance level after a block switch.
Inputs:
	block_start: the timestamp marking the start of the block
	correct_lever: the array of correct lever timestamps
	incorrect_lever: the array of incorrect lever timestamps
	crit_trials: number of trials to average over when calculating criterion performance
	crit_level: criterion performance level as a fraction (correct/total trials)
Returns:
	n_trials: number of trials it takes before an animal reaches criterion performance
"""
def trials_to_crit(block_start,correct_lever,incorrect_lever,
	crit_trials=5,crit_level=0.7):
	##get the correct lever and incorrect lever trials that happen
	##after the start of this block
	correct_lever = correct_lever[np.where(correct_lever>=block_start)[0]]
	incorrect_lever = incorrect_lever[np.where(incorrect_lever>=block_start)[0]]
	ids = np.empty((correct_lever.size+incorrect_lever.size),dtype='object')
	ids[0:correct_lever.size] = 'correct'
	ids[correct_lever.size:] = 'incorrect'
	##concatenate the timestamps and sort
	ts = np.concatenate((correct_lever,incorrect_lever))
	idx = np.argsort(ts)
	ids = ids[idx]
	##now, go through and get the % correct at every trial step
	n_trials = 0
	for i in range(ids.size-crit_trials):
		trial_block = ids[i:i+crit_trials]
		n_correct = (trial_block=='correct').astype(float).sum()
		n_incorrect = (trial_block=='incorrect').astype(float).sum()
		p_correct = n_correct/(n_correct+n_incorrect)
		if p_correct >= crit_level:
			# print("criterion reached")
			n_trials = i+crit_trials
			break
	if n_trials == 0: ##case where we never reached criterion
		n_trials = np.nan
		print("Criterion never reached after "+str(ids.size)+" trials")
	return n_trials



