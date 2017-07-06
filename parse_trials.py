##parse_trials.py
##functions to parse organized timestamp arrays,
##of the type returned by parse_timestamps.py functions

import numpy as np
import parse_timestamps as pt
import dpca
import parse_ephys as pe
import h5py
import pandas as pd
import file_lists
import os


"""
A function to get full trial data.
Inputs:
	f_behavior: file path to an hdf5 file with behavioral data
	pad: time (ms) around action and outcome timestamps to consider the start and end of the trial
	max_duration: upper bound on allowable trial durations, defined as period between action and outcome
Returns:
	data: a pandas dataframe with timestamps and event ID info for each trial
"""
def get_full_trials(f_behavior,pad=[400,400],max_duration=5000):
	f = h5py.File(f_behavior,'r')
	##first, mix all of the data we care about together into two arrays, 
	##one with timstamps and the other with timestamp ids.
	##for now, we don't care about the following timestamps:
	exclude = ['session_length','reward_primed','reward_idle',
	'bottom_rewarded','top_rewarded']
	event_ids = [x for x in list(f) if x not in exclude]
	##let's also store info about trial epochs
	upper_rewarded = np.asarray(f['top_rewarded'])
	lower_rewarded = np.asarray(f['bottom_rewarded'])
	session_length = np.floor(np.asarray(f['session_length'])*1000.0).astype(int)
	f.close()
	ts,ts_ids = pt.mixed_events(f_behavior,event_ids)
	##now we can do a little cleanup of the events. 
	##to remove duplicate nose pokes:
	ts,ts_ids = pt.remove_duplicate_pokes(ts,ts_ids)
	##to remove accidental lever presses
	ts,ts_ids = pt.remove_lever_accidents(ts,ts_ids)
	##finally, make sure the last trial completed before the end of the session
	ts,ts_ids = pt.check_last_trial(ts,ts_ids)
	##now get the ranges of the different block times
	##****This section is super confusing, but it works so...***
	block_times = pt.get_block_times(lower_rewarded,upper_rewarded,session_length)
	try:
		upper_rewarded = np.floor(np.asarray(block_times['upper'])[:,0]*1000.0).astype(int)
	except KeyError:
		upper_rewarded = np.array([])
	try:
		lower_rewarded = np.floor(np.asarray(block_times['lower'])[:,0]*1000.0).astype(int)
	except KeyError:
		lower_rewarded = np.array([])
	block_times = pt.get_block_times(lower_rewarded,upper_rewarded,session_length) ##now it's in ms
	##get a list of when the trials "started;" ie when the reward lever became active (we won't use this
	##timestamp in the end but it's useful for parsing the trial)
	trial_starts = np.where(ts_ids=='trial_start')[0]
	n_trials = trial_starts.size
	##now construct a pandas dataframe to store all of this information
	columns = ['start_ts','action_ts','outcome_ts','end_ts','context','action','outcome']
	data = pd.DataFrame(index=np.arange(n_trials),columns=columns)
	data.index.name = 'trial_number'
	##run through each of the trials and add data to the final dataframe
	for t in range(n_trials):
		##the data for this trial is everything in between this trial_start and the next one
		try:
			trial_events = ts_ids[trial_starts[t]:trial_starts[t+1]]
			trial_ts = ts[trial_starts[t]:]
		except IndexError: ##case where it's the last trial
			trial_events = ts_ids[trial_starts[t]:]
			trial_ts = ts[trial_starts[t]:]
		##determine which of the events in this set are the first action and first outcome
		##(if he presses more than once it's only the first one that counts; likewise for pokes)
		action_idx, outcome_idx = get_first_idx(trial_events)
		#add these data to the dataframe
		data['action_ts'][t] = trial_ts[action_idx]
		##annoying, but we need to switch the name here for consistency with other functions
		if trial_events[action_idx] == 'top_lever':
			action = 'upper_lever'
		elif trial_events[action_idx] == 'bottom_lever':
			action = 'lower_lever'
		else:
			print("Warning: detected action was not top or bottom lever")
		data['action'][t] = action
		outcome = trial_events[outcome_idx]
		data['outcome_ts'][t] = trial_ts[outcome_idx]
		data['outcome'][t] = outcome
		##add info about trial start and end based on padding parameters
		data['start_ts'][t] = trial_ts[action_idx]-pad[0]
		data['end_ts'][t] = trial_ts[outcome_idx]+pad[1]
		##now we need to know what the context was for this trial
		data['context'][t] = pt.which_block(block_times,trial_ts[0])
	data = check_trial_len(data,max_length=max_duration)
	return data


"""
A function to get windowed data around a particular type of event
Inputs:
	-f_behavior: path to behavior data file
	-f_ephys: path to ephys data file
	-event_name: string corresponding to the event type to time lock to
	-window [pre_event, post_event] window, in ms
	-smooth_method: type of smoothing to use; choose 'bins', 'gauss', 'both', or 'none'
	-smooth_width: size of the bins or gaussian kernel in ms. If 'both', input should be a list
		with index 0 being the gaussian width and index 1 being the bin width
	-if nan==True, any low-rate units that are marked for removal are simply filled with np.nan
Returns:
	-X data array in shape n_units x n_trials x b_bins
"""
def get_event_spikes(f_behavior,f_ephys,event_name,window=[400,0],
	smooth_method='gauss',smooth_width=30,z_score=True,min_rate=0.1,nan=True):
	##start by getting the parsing the behavior timestamps
	ts = pt.get_event_data(f_behavior)[event_name]
	##get the spike data for the full session
	X_raw = pe.get_spike_data(f_ephys,smooth_method='none',z_score=False) ##dont' bin or smooth yet
	##generate the data windows 
	windows = np.zeros((ts.size,2))
	windows[:,0] = ts-window[0]
	windows[:,1] = ts+window[1]
	windows = windows.astype(int)
	##now get the data windows
	X_trials = pe.X_windows(X_raw,windows) ##this is a LIST, not an array
	##get rid of really low rate units
	X_trials = remove_low_rate_units(X_trials,min_rate,nan=nan)
	##NOW we can do smoothing and z-scoring
	if smooth_method != 'none':
		for t in range(len(X_trials)):
			X_trials[t] = pe.smooth_spikes(X_trials[t],smooth_method,smooth_width)
	X_trials = np.asarray(X_trials)
	if z_score:
		X_trials = dpca.zscore_across_trials(X_trials)
	return X_trials

"""
A function to get windowed data around a particular type of event
Inputs:
	-f_behavior: path to behavior data file
	-f_ephys: path to ephys data file
	-event_name: string corresponding to the event type to time lock to
	-window [pre_event, post_event] window, in ms
Returns:
	-X data array in shape n_units x n_trials x b_bins
"""
def get_event_lfp(f_behavior,f_ephys,event_name,window=[400,0]):
	##start by getting the parsing the behavior timestamps
	ts = pt.get_event_data(f_behavior)[event_name]
	##get the spike data for the full session
	X_raw = pe.get_lfp_data(f_ephys) ##dont' bin or smooth yet
	##generate the data windows 
	windows = np.zeros((ts.size,2))
	windows[:,0] = ts-window[0]
	windows[:,1] = ts+window[1]
	windows = windows.astype(int)
	##now get the data windows
	X_trials = pe.X_windows(X_raw,windows) ##this is a LIST, not an array
	return X_trials

"""
A function to get spike data for all the units/trials in a session. 
Because trials can be a different length, this function stretches or
squeezes trials into a uniform length using interpolation.
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
Returns:
	X: processed spike data in the shape trials x units x bins
	trial_data: pandas datafame with info about the individual trials
"""
def get_trial_spikes(f_behavior,f_ephys,smooth_method='both',smooth_width=[80,40],pad=[400,400],
	z_score=True,trial_duration=None,max_duration=5000,min_rate=0.1):
	##check if padding is a multiple of bin size; warn if not
	if smooth_method == 'bins':
		if (pad[0]%smooth_width !=0) or (pad[1]%smooth_width != 0):
			print("Warning: pading should be a multiple of smooth width for best results")
	elif smooth_method == 'both':
		if (pad[0]%smooth_width[1] !=0) or (pad[1]%smooth_width[1] != 0):
			print("Warning: pading should be a multiple of smooth width for best results")
	##parse the raw data
	trial_data = get_full_trials(f_behavior,pad=pad,max_duration=max_duration)
	X_raw = pe.get_spike_data(f_ephys,smooth_method='none',z_score=False)
	##generate the data windows
	windows = np.zeros((len(trial_data.index),2))
	windows[:,0] = trial_data['start_ts']
	windows[:,1] = trial_data['end_ts']
	windows.astype(int)
	##get the spike data windows. The output of this is a LIST
	X_trials = pe.X_windows(X_raw,windows)
	##get rid of low rate units
	X_trials = remove_low_rate_units(X_trials,min_rate,nan=False) ##should still be a list
	if smooth_method != 'none':
		for t in range(len(X_trials)):
			X_trials[t] = pe.smooth_spikes(X_trials[t],smooth_method,smooth_width)
	##now we can work on equating all the trial lengths
	##if no specific trial length is requested, just use the median trial dur
	if trial_duration is None:
		trial_duration = np.median(windows[:,1]-windows[:,0])
	##convert this to bins, if data has been binned
	if smooth_method == 'bins':
		trial_duration = trial_duration/float(smooth_width)
	elif smooth_method == 'both':
		trial_duration = trial_duration/float(smooth_width[1])
	##now get this number as an integer
	##keep in mind, this is meant to be the duration of the 
	##period between action and outcome, which is NOT the full trial duration
	##if pad != [0,0]. Data in the pre-and post-intervals will not be interpolated
	trial_duration = np.ceil(trial_duration).astype(int)
	##get the timestamps relative to the start of each trial
	if smooth_method == 'bins':
		ts_rel = get_ts_rel(trial_data,bin_width=smooth_width)
	elif smooth_method == 'both':
		ts_rel = get_ts_rel(trial_data,bin_width=smooth_width[1])
	else: 
		ts_rel = get_ts_rel(trial_data)
	##now we can stretch/squeeze all of the trials to be the same length
	X_trials = dpca.stretch_trials(X_trials,ts_rel,trial_duration)
	if z_score:
		X_trials = dpca.zscore_across_trials(X_trials)
	return X_trials,trial_data

"""
A function to get lfp segments occurring during individual trials
Inputs:	
	f_behavior: file path to behavior data
	f_ephys: file path to ephys data
	pad: a window for pre- and post-trial padding, in ms. In other words, an x-ms period of time 
		before lever press to consider the start of the trial, and an x-ms period of time after
		reward to consider the end of the trial. For best results, should be a multiple of the bin size
	trial_duration: specifies the trial length (in ms) to squeeze trials into. If None, the function uses
		the median trial length over the trials in the file
	max_duration: maximum allowable trial duration (ms)
Returns:
	X: processed lfp data in the shape trials x units x ms
	trial_data: pandas datafame with info about the individual trials
"""
def get_trial_lfp(f_behavior,f_ephys,pad,trial_duration,max_duration):
	##get the trial timestamps
	trial_data = get_full_trials(f_behavior,pad=pad,max_duration=max_duration)
	##get the raw LFP data
	X_raw = pe.get_lfp_data(f_ephys)
	##generate the data windows
	windows = np.zeros((len(trial_data.index),2))
	windows[:,0] = trial_data['start_ts']
	windows[:,1] = trial_data['end_ts']
	windows.astype(int)
	##get the LFP data windows. The output of this is a LIST
	X_trials = pe.X_windows(X_raw,windows)
	##now we can work on equating all the trial lengths
	##if no specific trial length is requested, just use the median trial dur
	if trial_duration is None:
		trial_duration = np.median(windows[:,1]-windows[:,0])
	##now get this number as an integer
	##keep in mind, this is meant to be the duration of the 
	##period between action and outcome, which is NOT the full trial duration
	##if pad != [0,0]. Data in the pre-and post-intervals will not be interpolated
	trial_duration = np.ceil(trial_duration).astype(int)
	ts_rel = get_ts_rel(trial_data)
	##now we can stretch/squeeze all of the trials to be the same length
	X_trials = dpca.stretch_trials(X_trials,ts_rel,trial_duration)
	return X_trials,trial_data

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
		try:
			if data_last['upper_rewarded'].max()>data_last['lower_rewarded'].max():
				last_block_type = 'upper_rewarded'
				last_block_ts = data['upper_rewarded'].max()
			elif data_last['lower_rewarded'].max()>data_last['upper_rewarded'].max():
				last_block_type = 'lower_rewarded'
				last_block_ts = data['lower_rewarded'].max()
			else:
				print("Unrecognized block type")
		except ValueError:
			print("Only one block type detected")
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
			# else:
			# 	print("unrecognized lever type")
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


"""
A helper function to remove data from units with low spike rates. 
Inputs:
	X_trials: a list or array of binary array of spike data, in shape 
		trials x (units x bins). Assumes 1-ms binning!
	min_rate: minimum spike rate, in Hz
	nan: if True, replaces low spiking unit data with np.nan, but doesn't remove
Returns:
	X_trials: same array with data removed from units that don't meet the min
		spike rate requirements
"""
def remove_low_rate_units(X_trials,min_rate=0.1,nan=False):
	##copy data into a new list, so we retain compatibility with arrays
	X_clean = []
	for i in range(len(X_trials)):
		X_clean.append(X_trials[i])
	##get some data about the data
	n_trials = len(X_clean)
	n_units = X_clean[0].shape[0]
	n_bins = X_clean[1].shape[1]
	##create a matrix that tracks spike rates over trials
	unit_rates = np.zeros(n_units)
	##now add together the mean rate for each unit over all trials
	for t in range(n_trials):
		unit_rates[:] += (X_clean[t].sum(axis=1)/X_clean[t].shape[1])*1000
	##take the mean rate over all trials
	unit_rates = unit_rates/len(X_trials)
	##find unit indices where the rate is below the min allowed 
	remove = np.where(unit_rates<min_rate)[0]
	##now get rid of the units, if any
	if remove.size > 0:
		for t in range(n_trials):
			if nan:
				X_clean[t][remove,:] = np.nan
			else:
				X_clean[t] = np.delete(X_clean[t],remove,axis=0)
	return X_clean

"""
A helper function to remove nan-ed unit data from two or more sets of
trial data.
Inputs:
	X_all: a list of trials data
returns:
	X_clean: a list of trials data with nan units removed from all sets
"""
def remove_nan_units(X_all):
	##the indices of units to remove
	to_remove = []
	for i in range(len(X_all)):
		to_remove.append(np.where(np.isnan(X_all[i].mean(axis=0).mean(axis=1)))[0])
	to_remove = np.concatenate(to_remove)
	X_clean = []
	for i in range(len(X_all)):
		X_clean.append(np.delete(X_all[i],to_remove,axis=1))
	return X_clean

"""
A helper function to return the index of the first action and first outcome
given an array of event ids from one single trial
Inputs:
	trial_events: string array of events (in order) from one trial
Returns:
	action_idx: index of first action
	outcome_idx: index of first outcome
"""
def get_first_idx(trial_events):
	actions = ['top_lever','bottom_lever']
	outcomes = ['rewarded_poke','unrewarded_poke']
	action_idx = None
	outcome_idx = None
	for i,event in enumerate(trial_events):
		if event in actions:
			action_idx = i
			break
	for j,event in enumerate(trial_events):
		if event in outcomes:
			outcome_idx = j
			break
	if action_idx == None or outcome_idx == None:
		raise TypeError("This trial is incomplete (no action or outcome)")
	return action_idx, outcome_idx


"""
A helper function to determine if a trial duration (between action and outcome)
exceeds a certain maximum time limit.
Inputs:
		data: a pandas dataframe with trial data, of the type returned by get_full_trials()
		max_length: the maximum allowable threshold length, in ms
Returns: 
	dataframe: a copy of the dataframe with trials removed that exceeded the time limit
"""
def check_trial_len(data,max_length=5000):
	to_remove = []
	for i in range(len(data.index)):
		duration = data['outcome_ts'][i]-data['action_ts'][i]
		if duration > max_length:
			to_remove.append(i)
		if duration < 0 :
			print("Error: negative trial length detected (removed)")
			to_remove.append(i)
	if len(to_remove)>0:
		pass
		#print("Dropping "+str(len(to_remove))+" trials over "+str(max_length/1000)+" sec")
	data = data.drop(to_remove) 
	##now replace the index to exclude the dropped trials
	data.index = np.arange(len(data.index))
	return data

"""
A helper function to get an array of trial timestamps relative to the start of
each trial, given a dataframe of trial data. Assumes dataframe timestamps are in ms, 
but will automatically convert them to bins if the bin_width param is specified.
Inputs:
	data: pandas dataframe with trial data
	bin_width: bin width to use. If None, assumes 1-ms bins (no conversion)
Returns:
	ts_rel: array of timestamps for each trial, relative to the start of each trial
"""
def get_ts_rel(data,bin_width=None):
	###we are going to make some assumptions. First, that the first and last
	##chunks are all of even padding. Check that now:
	pre_pad = data['action_ts'][0] - data['start_ts'][0]
	post_pad = data['end_ts'][0] - data['outcome_ts'][0]
	assert np.all(data['action_ts']-data['start_ts']==pre_pad), "Uneven pre-padding"
	assert np.all(data['end_ts']-data['outcome_ts']==post_pad), "Uneven post-padding"
	##things will not work properly if the padding isn't evenly divisible by bin width
	if bin_width is not None:
		assert pre_pad%bin_width == 0
		assert post_pad%bin_width == 0
	##now set up the return array
	events = ['start_ts','action_ts','outcome_ts','end_ts']
	ts_rel = np.zeros((len(data.index),len(events)))
	##the first index will always be zero, so skip that.
	##second index (action) is the pre-pad width
	if bin_width is not None:
		ts_rel[:,1] = pre_pad/bin_width
		##third index (outcome) is the difference between action and outcome, + pre-pad
		ts_rel[:,2] = (data['outcome_ts']-data['action_ts']+pre_pad)/float(bin_width)
		##last index is just the third index plus the post-padding
		ts_rel[:,3] = ts_rel[:,2]+post_pad/bin_width
	else:
		ts_rel[:,1] = pre_pad
		##third index (outcome) is the difference between action and outcome, + pre-pad
		ts_rel[:,2] = (data['outcome_ts']-data['action_ts']+pre_pad)
		##last index is just the third index plus the post-padding
		ts_rel[:,3] = ts_rel[:,2]+post_pad
	return ts_rel.astype(int)


"""
A function to parse a trial_data pandas array in order to split trials
into different trial types.
Inputs:
	trial_data: a pandas array of the type returned by get_full_trials
Returns: 
	trial_info: a dictionary, basically a further parsing of the data into 
		different classes, indicating which trials are in which class
"""
def parse_trial_data(trial_data):
	##set up our output dictionary
	trial_info = {
	'upper_correct_rewarded':[],
	'upper_correct_unrewarded':[],
	'upper_incorrect':[],
	'lower_correct_rewarded':[],
	'lower_correct_unrewarded':[],
	'lower_incorrect':[],
	'n_blocks':1,
	'block_lengths':[]
	}
	for t in range(len(trial_data.index)):
		trial = trial_data.loc[t]
		if trial['action'] == 'upper_lever' and trial['context'] == 'upper_rewarded' and trial['outcome'] == 'rewarded_poke':
			trial_info['upper_correct_rewarded'].append(t)
		elif trial['action'] == 'upper_lever' and trial['context'] == 'upper_rewarded' and trial['outcome'] == 'unrewarded_poke':
			trial_info['upper_correct_unrewarded'].append(t)
		elif trial['action'] == 'upper_lever' and trial['context'] == 'lower_rewarded':
			trial_info['upper_incorrect'].append(t)
		elif trial['action'] == 'lower_lever' and trial['context'] == 'lower_rewarded' and trial['outcome'] == 'rewarded_poke':
			trial_info['lower_correct_rewarded'].append(t)
		elif trial['action'] == 'lower_lever' and trial['context'] == 'lower_rewarded' and trial['outcome'] == 'unrewarded_poke':
			trial_info['lower_correct_unrewarded'].append(t)
		elif trial['action'] == 'lower_lever' and trial['context'] == 'upper_rewarded':
			trial_info['lower_incorrect'].append(t)
		else:
			print("Unknown trial type for trial {}".format(t))
	##now get the block lengths data
	current_block = trial_data.loc[0]['context']
	block_length = 0
	for t in range(len(trial_data.index)):
		if current_block == trial_data.loc[t]['context']: ##case where the block does not change
			block_length+=1
		else: ##case where block DOES change
			# print("block length is {}".format(block_length))
			trial_info['block_lengths'].append(block_length)
			trial_info['n_blocks'] +=1
			block_length = 0
			current_block = trial_data.loc[t]['context']
	##add the length of th last block
	trial_info['block_lengths'].append(block_length)
	return trial_info


""" 
A helper function to return the actual session number
(files are named in reference to before or after implant)
Input:
	fname: string file name for a behavior file
Returns:
	session_number
"""
def get_session_number(fname):
	##first figure out what animal this session is from
	animal = fname[-11:-9]
	##get all the behavior files from this animal
	ani_files = [os.path.normpath(x) for x in file_lists.behavior_files if x[-11:-9] == animal]
	##now get the position of this file in the full list
	return ani_files.index(fname)

"""
A function to split trials up according to what happened on the previous trial;
ie if the previous trial was rewarded or unrewarded. 
Inputs:
	trial_data DataFrame
Returns:
	prev_rewarded: index of trials where the previous trial was rewarded
	prev_unrewarded: intex of trials where the previous trial was unrewarded
"""
def split_by_last_outcome(trial_data):
	prev_unrewarded_idx = np.where(trial_data['outcome']=='unrewarded_poke')[0] + 1
	prev_rewarded_idx = np.where(trial_data['outcome']=='rewarded_poke')[0] + 1
	##make sure we aren't including a trial extra
	if prev_rewarded_idx.max() == len(trial_data):
		prev_rewarded_idx = prev_rewarded_idx[:-1]
	else:
		prev_unrewarded_idx = prev_unrewarded_idx[:-1]
	return prev_rewarded_idx,prev_unrewarded_idx



