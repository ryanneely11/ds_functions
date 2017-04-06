##parse_timestamps.py
##functions to parse timestamp data
import h5py
import numpy as np

"""
This is a function to look at session data a different way.
Probably rats don't understand the trial structure, so this
splits things up by event types. It's not hugely different than the
other sorting methods, but it includes a bit more data this way and
doesnt't try to shoehorn things into a trial structure.
Inputs:
	f_behavior: an hdf5 file with the raw behavioral data
Returns:
	results: a dictionary with timestamps split into catagories, 
		and converted into ms.
"""
def get_event_data(f_behavior):
	##first, mix all of the data we care about together into two arrays, 
	##one with timstamps and the other with timestamp ids.
	##for now, we don't care about the following timestamps:
	f = h5py.File(f_behavior,'r')
	exclude = ['session_length','reward_primed','reward_idle',
	'trial_start','bottom_rewarded','top_rewarded']
	event_ids = [x for x in list(f) if x not in exclude]
	##let's also store info about trial epochs
	upper_rewarded = np.asarray(f['top_rewarded'])
	lower_rewarded = np.asarray(f['bottom_rewarded'])
	session_length = np.floor(np.asarray(f['session_length'])*1000.0).astype(int)
	f.close()
	ts,ts_ids = mixed_events(f_behavior,event_ids)
	##now we can do a little cleanup of the events. 
	##to remove duplicate nose pokes:
	ts,ts_ids = remove_duplicate_pokes(ts,ts_ids)
	##to remove accidental lever presses
	ts,ts_ids = remove_lever_accidents(ts,ts_ids)
	##now get the ranges of the different block times
	##****This section is super confusing, but it works so...***
	block_times = get_block_times(lower_rewarded,upper_rewarded,session_length)
	try:
		upper_rewarded = np.floor(np.asarray(block_times['upper'])[:,0]*1000.0).astype(int)
	except KeyError:
		upper_rewarded = np.array([])
	try:
		lower_rewarded = np.floor(np.asarray(block_times['lower'])[:,0]*1000.0).astype(int)
	except KeyError:
		lower_rewarded = np.array([])
	block_times = get_block_times(lower_rewarded,upper_rewarded,session_length) ##now it's in ms
	##finally, we can parse these into different catagories of events
	upper_lever = [] ##all upper lever presses
	lower_lever = [] ##all lower presses
	rewarded_lever = [] ##all presses followed by a rewarded poke
	unrewarded_lever = [] ##all presses followed by an unrewarded poke
	rewarded_poke = []
	unrewarded_poke = []
	correct_lever = [] ##any press that was correct for the context
	incorrect_lever = []
	correct_poke = [] ##pokes that happened after correct levers
	incorrect_poke = [] ##pokes that happened after incorrect levers
	##run through the events and parse into the correct place
	actions = ['top_lever','bottom_lever']
	outcomes = ['rewarded_poke','unrewarded_poke']
	##start us off...
	last_event = ts_ids[0]
	for i in range(ts_ids.size-1):
		current_event = ts_ids[i+1]
		##need to consider all cases
		if last_event in actions: ##case where the preceeding event was a lever press
			##first consider what type of press
			if last_event == 'top_lever':
				upper_lever.append(ts[i])
			elif last_event == 'bottom_lever':
				lower_lever.append(ts[i])
			else:
				print("Error: unknown action type")
				break
			##now we need to consider if this was a correct press or not
			if (last_event=='top_lever' and which_block(block_times,
				ts[i])=='upper_rewarded') or (last_event=='bottom_lever' and 
				which_block(block_times,ts[i])=='lower_rewarded'):
				correct_lever.append(ts[i])
			elif (last_event=='top_lever' and which_block(block_times,
				ts[i])=='lower_rewarded') or (last_event=='bottom_lever' and 
				which_block(block_times,ts[i])=='upper_rewarded'):
				incorrect_lever.append(ts[i])
			else:
				print("Error: lever press neither correct or incorrect")
				break
			##Now we need to figure out if this was a rewarded action or not
			if current_event == 'rewarded_poke':
				rewarded_lever.append(ts[i])
			elif current_event == 'unrewarded_poke':
				unrewarded_lever.append(ts[i])
			##it's possible the next event was a lever press so we won't catch an exception
			last_event = current_event
		elif last_event in outcomes:
			if last_event == 'rewarded_poke':
				rewarded_poke.append(ts[i])
			elif last_event == 'unrewarded_poke':
				unrewarded_poke.append(ts[i])
			else:
				print("Error: unknown poke type")
				break
			##now decide if this was a correct poke, event if unrewarded
			if (ts_ids[i-1]=='top_lever' and which_block(block_times,
				ts[i-1])=='upper_rewarded') or (ts_ids[i-1]=='bottom_lever' and 
				which_block(block_times,ts[i-1])=='lower_rewarded'):
				correct_poke.append(ts[i])
			elif (ts_ids[i-1]=='top_lever' and which_block(block_times,
				ts[i-1])=='lower_rewarded') or (ts_ids[i-1]=='bottom_lever' and 
				which_block(block_times,ts[i-1])=='upper_rewarded'):
				incorrect_poke.append(ts[i])
			last_event = current_event
		else:
			print("Error: unknown event type")
			break
	##create a dictionary of this data
	results = {
	'upper_lever':np.asarray(upper_lever),
	'lower_lever':np.asarray(lower_lever),
	'rewarded_lever':np.asarray(rewarded_lever),
	'unrewarded_lever':np.asarray(unrewarded_lever),
	'correct_lever':np.asarray(correct_lever),
	'incorrect_lever':np.asarray(incorrect_lever),
	'rewarded_poke':np.asarray(rewarded_poke),
	'unrewarded_poke':np.asarray(unrewarded_poke),
	'correct_poke':np.asarray(correct_poke),
	'incorrect_poke':np.asarray(incorrect_poke),
	'upper_rewarded':upper_rewarded,
	'lower_rewarded':lower_rewarded,
	'session_length':session_length
	}
	return results



"""
A function that takes in a behavior timestamps file (HDF5)
and returns an array of timestamps, as well as an index
of what catagory each trial falls into (ie rewarded, unrewarded, 
upper lever, etc)
Inputs:
	-f_in: file path to where the data is
	-remove_unrew: if True, excludes trials where the response was correct,
		but the trial was still unrewarded
Returns:
	-ts: trial x event array. Axis 0 is trials, axis 1 is action, outcome
	-ts_idx: dictionary of array containing the indices of different 
		trial types.
"""
def get_trial_data(f_in,remove_unrew=False):
	ts_idx = {}
	##sort_by_trial does most of the work:
	block_data = sort_by_trial(f_in,remove_unrew=remove_unrew)
	##now just parse the TS a little further
	block_types = list(block_data) ##keep the block order (upper_rewarded, etc consistent)
	##make sure we only have 2 block types (excpected)
	assert len(block_types) == 2
	##concatenate the data from both block types
	ts = np.vstack((block_data[block_types[0]],block_data[block_types[1]]))
	##add the indices of the different block types to the dict
	block1_len = block_data[block_types[0]].shape[0]
	block2_len = block_data[block_types[1]].shape[0]
	ts_idx[block_types[0]] = np.arange(block1_len)
	ts_idx[block_types[1]] = np.arange(block1_len,block1_len+block2_len)
	##get rid of dim 0 of the TS (containins arbitrary trial start timestamps)
	ts = ts[:,1:]
	##any negative action ts (now first index in dim 0) indicate a trial where the lower lever
	##was pushed:
	ts_idx['lower_lever'] = np.where(ts[:,0]<0)[0]
	##any positive action ts mean the upper lever was pushed:
	ts_idx['upper_lever'] = np.where(ts[:,0]>0)[0]
	##any negative outcome ts mean that trial was unrewarded:
	ts_idx['unrewarded'] = np.where(ts[:,1]<0)[0]
	##and finally any positive outcome ts mean the trial was rewarded:
	ts_idx['rewarded'] = np.where(ts[:,1]>0)[0]
	##now we can get rid of the negative ts values:
	ts = abs(ts)
	##let's perform a check to see if any of the timestamps don't make sense
	if np.any(ts[:,1]-ts[:,0]<0):
		print("Warning: detected at least 1 timestamp out of order. Check source file...")
	##return the results:
	return ts,ts_idx


""" a function to split behavior timestamp data into individual trials.
	Inputs:
		f_in: the file path pointing to an hdf5 file containing
		the behavior event time stamps
		-remove_unrew: if True, excludes trials where the response was correct,
			but the trial was still unrewarded
	Returns: 
		Two n-trial x i behavioral events-sized arrays.
		One contains all trials where the lower lever is rewarded;
		The other contains all the trials where the upper lever is rewarded. 
		Behavioral events are indexed temporally:
		0-trial start; 1-action (U or L); 2-poke(R or U)
"""
def sort_by_trial(f_in,remove_unrew=False):
	#load data file
	f = h5py.File(f_in,'r')
	#get the arrays of timestamps into a dictionary in memory
	data_dict = {
		'lower_lever':np.asarray(f['bottom_lever']),
		'upper_lever':np.asarray(f['top_lever']),
		'reward_idle':np.asarray(f['reward_idle']),
		'reward_primed':np.asarray(f['reward_primed']),
		'rewarded_poke':np.asarray(f['rewarded_poke']),
		'unrewarded_poke':np.asarray(f['unrewarded_poke']),
		'trial_start':np.asarray(f['trial_start']),
		'session_end':np.asarray(f['session_length']),
		'lower_rewarded':np.asarray(f['bottom_rewarded']),
		'upper_rewarded':np.asarray(f['top_rewarded']),
	}
	f.close()
	##create the output dictionary
	result = {}
	##get the dictionary containing the block information
	block_times = get_block_times(data_dict['lower_rewarded'],data_dict['upper_rewarded'],
		data_dict['session_end'])
		##start with all of the lower lever blocks
	try:	
		for lb in range(len(block_times['lower'])):
			block_data = get_block_data(block_times['lower'][lb],data_dict)
			trial_times = sort_block(block_data)
			##remove correct unrewarded, if requested
			if remove_unrew:
				trial_times = remove_correct_unrewarded(trial_times,'lower_rewarded')
			##case where there is a dictionary entry
			try:
				result['lower_rewarded'] = np.vstack((result['lower_rewarded'],trial_times))
			except KeyError:
				result['lower_rewarded'] = trial_times
	except KeyError:
		pass
	##repeat for upper lever blocks
	try:
		for ub in range(len(block_times['upper'])):
			block_data = get_block_data(block_times['upper'][ub],data_dict)
			trial_times = sort_block(block_data)
			##remove correct unrewarded, if requested
			if remove_unrew:
				trial_times = remove_correct_unrewarded(trial_times,'upper_rewarded')
			##case where there is a dictionary entry
			try:
				result['upper_rewarded'] = np.vstack((result['upper_rewarded'],trial_times))
			except KeyError:
				result['upper_rewarded'] = trial_times
	except KeyError:
		pass
	return result



"""
A helper function for sort_by_trial; sorts out trials for one block.
Inputs:
	-block_data: a dictionary containing all the timestamps
	for one block, which is a period of time in which the lever-
	reward contingency is constant.
	-trial_max: maximum trial length to tolerate
Returns:
	an n-trial by i behavioral events-sized array

	****													****
	****Important: in order to simplify the output arrays, 	****
	****I'm going to use a positive/negative code in the following way:
		For each trial:
			-Timestamp 0 = the start of the trial
			-Timestamp 1 = action; negative number = lower lever; positive = upper lever
			-Timestamp 2 = outcome; negative number = unrewarded; positive = rewarded
"""
def sort_block(block_data,trial_max=5):
	##let's define the order of events that we want:
	ordered_events = ['start','action','outcome']
	##allocate memory for the result array
	result = np.zeros((block_data['trial_start'].size,3))
	##fill the results array for each trial
	for i in range(result.shape[0]):
		trial_start = block_data['trial_start'][i] ##the start of this trial
		try: 
			trial_end = block_data['trial_start'][i+1]
		except IndexError:
			trial_end = max(block_data['rewarded_poke'].max(),
							block_data['unrewarded_poke'].max())
		##***ACTIONS***

		##now find the first action
		#idx of any upper presses in the interval
		upper_idx = np.nonzero(np.logical_and(block_data['upper_lever']>trial_start,
			block_data['upper_lever']<trial_end))[0]
		lower_idx = np.nonzero(np.logical_and(block_data['lower_lever']>trial_start,
			block_data['lower_lever']<trial_end))[0]
		##case 1: both upper and lower lever presses happened
		if upper_idx.size>0 and lower_idx.size>0:
			##find which action happened first
			upper_presses = block_data['upper_lever'][upper_idx] ##the actual timestamps
			lower_presses = block_data['lower_lever'][lower_idx]
			##if the first upper press happened first:
			if upper_presses.min()<lower_presses.min():
				action = upper_presses.min()
			elif lower_presses.min()<upper_presses.min():
				action = -1*lower_presses.min()
			else:
				##error case
				print("something wrong in upper/lower comparison")
				break
		#case 2: only upper lever was pressed
		elif upper_idx.size>0 and lower_idx.size==0:
			action = block_data['upper_lever'][upper_idx].min()
		##case 3: only lower lever was pressed
		elif upper_idx.size==0 and lower_idx.size>0:
			action = -1*block_data['lower_lever'][lower_idx].min()
		##case 4: something is wrong!
		else:
			print("Error- no action for this trial??")
			break
		
		##***OUTCOMES***
		
		##ts of any rewarded pokes
		reward_idx = np.nonzero(np.logical_and(block_data['rewarded_poke']>trial_start,
			block_data['rewarded_poke']<=trial_end))[0]
		##case where this was a rewarded trial
		if reward_idx.size == 1:
			outcome = block_data['rewarded_poke'][reward_idx]
		##case where this was not a rewarded trial
		elif reward_idx.size == 0:
			##let's get the unrewarded ts
			unreward_idx = np.nonzero(np.logical_and(block_data['unrewarded_poke']>trial_start,
				block_data['unrewarded_poke']<=trial_end))[0]
			if unreward_idx.size > 0:
				unrewarded_pokes = block_data['unrewarded_poke'][unreward_idx]
				outcome = -1*unrewarded_pokes.min()
			else:
				print("Error: no pokes for this trial")
				break
		else:
			print("error: too many rewarded pokes for this trial")
			break

		##now add the data to the results
		result[i,0] = trial_start
		result[i,1] = action
		result[i,2] = outcome
	##now make sure none of the trials violate the max trial duration
	t = 0
	while t < result.shape[0]:
		if (abs(result[t,2])-abs(result[t,1])) >= trial_max:
			print("removing trial of length "+str(abs(result[t,2])-abs(result[t,1])))
			result = np.delete(result,t,axis=0)
		else:
			t+=1
	return result


"""
A helper function for sort_by_trial; determines how many blocks
are in a file, and where the boundaries are. 
Inputs:
	-Arrays containing the lower/upper rewarded times, 
	as well as the session_end timestamp.
	-min_length is the cutoff length in secs; any blocks shorter 
		than this will be excluded
Outputs:
	A dictionary for each type of block, where the item is a list of 
	arrays with start/end times for that block.
"""
def get_block_times(lower_rewarded, upper_rewarded,session_end,min_length=5*60):
	##get a list of all the block times, and a corresponding list
	##of what type of block we are talking about
	block_starts = []
	block_id = []
	for i in range(lower_rewarded.size):
		block_starts.append(lower_rewarded[i])
		block_id.append('lower')
	for j in range(upper_rewarded.size):
		block_starts.append(upper_rewarded[j])
		block_id.append('upper')
	##sort the start times and ids
	idx = np.argsort(block_starts)
	block_starts = np.asarray(block_starts)[idx]
	block_id = np.asarray(block_id)[idx]
	result = {}
	##fill the dictionary
	spurious = 0
	for b in range(block_starts.size):
		##range is from the start of the current block
		##to the start of the second block, or the end of the session.
		start = block_starts[b]
		try:
			stop = block_starts[b+1]
		except IndexError:
			stop = session_end
		##check to make sure this block meets the length requirements
		if stop-start > min_length:
			rng = np.array([start,stop])
			##create the entry if needed
			try:
				result[block_id[b]].append(rng)
			except KeyError:
				result[block_id[b]] = [rng]
		else:
			spurious +=1
	if spurious > 0:
		print("Cleaned "+str(spurious)+" spurious block switches")
	return result


"""
another helper function. This one takes in a block start, stop
time and returns only the subset of timestamps that are within
that range.
Inputs:
	block_edges: an array [start,stop]
	data_dict: the dictionary of all the different timestamps
Outputs:
	a modified data dictionary with only the relevant data
"""
def get_block_data(block_edges,data_dict):
	result = {} #dict to return
	keys = list(data_dict)
	for key in keys:
		data = data_dict[key]
		idx = np.nonzero(np.logical_and(data>block_edges[0],data<=block_edges[1]))[0]
		result[key] = data[idx]
	##figure out if the last trial was completed; if not get rid of it
	last_trial = result['trial_start'].max()
	##need some error catching here in case there were no upper or lower levers in this block
	try:
		last_upper = result['upper_lever'].max()
	except ValueError: ##case of empty array
		last_upper = np.array([0])
	try:
		last_lower = result['lower_lever'].max()
	except ValueError:
		last_lower = np.array([0])
	last_action = max(last_upper,last_lower)
	last_poke = max(result['rewarded_poke'].max(),result['unrewarded_poke'].max())
	if (last_trial < last_action) and (last_trial < last_poke):
		pass
	else:
		result['trial_start'] = np.delete(result['trial_start'],-1)
	return result



"""
A helper function to remove trials that are correct, but unrewarded.
Inputs:
	block_ts: array of timestamps for one block, in shape trials x ts 
		(see sort_block output)
	block_type: either 'upper_rewarded' or 'lower_rewarded'
Returns:
	clean_ts: same block of timestamps, but with correct, unrewarded trials removed
"""
def remove_correct_unrewarded(block_ts,block_type):
	##block_ts[:,0] is trial start time, so we can ignore it
	to_remove = [] ##index of trials to remove
	##we have two cases:
	if block_type == 'upper_rewarded':
		for trial in range(block_ts.shape[0]):
			##if this trial was correct but unrewarded, mark for removal
			if (block_ts[trial,1]>0) and (block_ts[trial,2]<0):
				to_remove.append(trial)
	elif block_type == 'lower_rewarded':
		for trial in range(block_ts.shape[0]):
			##if this trial was correct but unrewarded, mark for removal
			if (block_ts[trial,1]<0) and (block_ts[trial,2]<0):
				to_remove.append(trial)
	else:
		raise KeyError("Unrecognized block type")
	##get the indices of all trials we want to keep
	clean_idx = [x for x in np.arange(block_ts.shape[0]) if not x in to_remove]
	##get a new array with only these trials
	clean_ts = block_ts[clean_idx,:]
	return clean_ts

"""
A helper function that gets removes spurius unrewarded pokes from a pair of data arrays
(matched timestamps and timestamp IDs). Rats can trigger multiple unrewarded pokes
one after the other by licking in the water port. We want to only consider a series of
unrewarded pokes as a single unrewarded poke. 
Inputs:
	ts: 1-d numpy array of timestamps
	ts_ids: 1-d numpy array of matched timestamp ids
Returns:
	Duplicate arrays with timestamp and timestamp IDs of consecutive unrewarded
	pokes removed. 
"""
def remove_duplicate_pokes(ts,ts_ids):
	##define the kind of events that we are interested in
	poke_types = ['rewarded_poke','unrewarded_poke']
	##keep a running list of indices where we have spurious pokes
	to_remove = []
	##run through the list and remove duplicate pokes
	i = 0
	last_event = 'none'
	while i < ts.size:
		this_event = ts_ids[i]
		if this_event in poke_types: ##we only care if this event is a poke
			if last_event in poke_types: ##case where we have two back to back pokes
				to_remove.append(i)
				last_event = this_event
				i+=1
			else: ##case where last event was not a poke
				last_event = this_event
				i+=1
		else: ##move onto the next event
			last_event = this_event
			i+=1
	keep = [x for x in range(ts.size) if x not in to_remove]
	return ts[keep],ts_ids[keep]

"""
A helper function to remove accidental lever presses. Sometimes, 
the top lever will be pressed by accident by the headstage cables when the animal
is trying to press the bottom lever. We can assume this happens when a top and a bottom
lever press happen within a short time window of each other, and this would only
happen accidentally for the top lever, so we will get rid of that timestamp.
Inputs:
	ts: 1-d numpy array of timestamps (in ms)
	ts_ids: 1-d numpy array of matched timestamp ids
	tolerance: the time gap that is allowable betweeen presses to be
		considered intentional.
Returns:
	Duplicate arrays with timestamp and timestamp IDs of putative accidental
	top lever presses. 
"""
def remove_lever_accidents(ts,ts_ids,tolerance=100):
	##define the ids we are interested in
	lever_types = ['top_lever','bottom_lever']
	##keep a list of ids to remove
	to_remove = []
	last_event = ts_ids[0]
	last_timestamp = ts[0]
	for i in range(1,ts_ids.size):
		this_event = ts_ids[i]
		this_timestamp = ts[i]
		##we only care if this event AND the last event are lever presses
		if (this_event in lever_types) and (last_event in lever_types):
			##also check that the lever press types are different
			if this_event != last_event:
				##see if this sequence violates our interval tolerance
				if this_timestamp-last_timestamp <= tolerance:
					##if all these conditions are met, remove which ever ts is the upper lever
					if this_event == 'top_lever':
						to_remove.append(i)
					elif last_event == 'top_lever':
						to_remove.append(i-1)
		last_event = this_event
	keep = [x for x in range(ts.size) if x not in to_remove]
	return ts[keep],ts_ids[keep]



"""
A helper function to get a sorted and mixed list of events and event ids
from a behavioral events file.
Inputs: 
	f_in: datafile to draw from 
	event_list: a list of event ids to include in the output arrays
Returns:
	ts: a list of timestamps, sorted in ascending order
	ts_ids: a matcehd list of timestamp ids
"""
def mixed_events(f_in,event_list):
	ts = []
	ts_ids = []
	##open the file
	f = h5py.File(f_in)
	##add everything to the master lists
	for event in event_list:
		data = np.floor((np.asarray(f[event])*1000.0)).astype(int)
		for i in range(data.size):
			ts.append(data[i])
			ts_ids.append(event)
	f.close()
	##convert to arrays
	ts = np.asarray(ts)
	ts_ids = np.asarray(ts_ids)
	##now sort in ascending order
	idx = np.argsort(ts)
	ts = ts[idx]
	ts_ids = ts_ids[idx]
	return ts, ts_ids


"""
Another little helper function to check which block type (upper or lower rewarded)
a give timestamp falls into.
Inputs:
	-result: the output from get_block_times
	-ts: the timestamp to check for membership
	***MAKE SURE THEY ARE BOTH ON THE SAME TIMESCALE!***
returns:
	-upper_times: an array of timestamps (in ms) where upper lever was rewarded
	-lower_times: ditto but for lower lever rewarded
"""
def which_block(results,ts):
	block = None
	for b in list(results):
		##start with the lower epochs
		for epoch in results[b]:
			if ts >= epoch[0] and ts <= epoch[1]:
				block = b+'_rewarded'
				break
	return block

"""
A helper function to clean an array of block times.
Sometimes the raspberry pi had a glitch where it would
record multiple block switches in a row. Since we know that no block
was ever less than 5 mins, we can exclude get rid of these spurious block switches.
Inputs:
	upper_rewarded: an array of putative upper rewarded block switch times
	lower_rewarded: an array of putative lower rewarded block switch times
	threshold: shortest allowable block, in s
Returns: 
	clean_upper: a copy of the input array with false block switch times removed
	clean_lower: ditto
"""
# def clean_block_times(upper_rewarded,lower_rewarded,threshold=5*60):
# 	##first clean each set of times separately
# 	keep_upper = np.concatenate((np.array([0]),np.where(np.diff(upper_rewarded>threshold))[0]+1))
# 	keep_lower = np.concatenate((np.array([0]),np.where(np.diff(lower_rewarded>threshold))[0]+1))
# 	upper_rewarded = upper_rewarded[keep_upper]
# 	lower_rewarded = lower_rewarded[keep_lower]
# 	##add all the block switch times together, and create an index array
# 	times = np.concatenate((upper_rewarded,lower_rewarded))
# 	upper_ids = np.empty(upper_rewarded.size,dtype='object')
# 	upper_ids[:] = 'upper_rewarded'
# 	lower_ids = np.empty(lower_rewarded.size,dtype='object')
# 	lower_ids[:] = 'lower_rewarded'
# 	ids = np.concatenate((upper_ids,lower_ids))
# 	##now sort the times and indices
# 	idx = np.argsort(times)
# 	times = times[idx]
# 	ids = ids[idx]
# 	##indices to keep
# 	valid_idx = np.where(np.diff(times)>threshold)[0]
# 	invalid_idx = np.where(np.diff(times)<threshold)[0]
# 	if invalid_idx.size>0:
# 		print("Cleaning "+str(len(invalid_idx))+" spurious block switches")
# 	clean_times = times[valid_idx]
# 	clean_ids = ids[valid_idx]
# 	upper_idx = np.where(clean_ids=='upper_rewarded')[0]
# 	lower_idx = np.where(clean_ids=='lower_rewarded')[0]
# 	return clean_times[upper_idx],clean_times[lower_idx]

