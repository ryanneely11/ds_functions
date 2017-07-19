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
	##finally, get the context-dependent presses
	upper_context_levers = get_context_levers(upper_rewarded,lower_rewarded,
		np.asarray(upper_lever),np.asarray(lower_lever),session_length,'upper')
	lower_context_levers = get_context_levers(upper_rewarded,lower_rewarded,
		np.asarray(upper_lever),np.asarray(lower_lever),session_length,'lower')
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
	'upper_context_lever':upper_context_levers,
	'lower_context_lever':lower_context_levers,
	'upper_rewarded':upper_rewarded,
	'lower_rewarded':lower_rewarded,
	'session_length':session_length
	}
	return results


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
A helper function to get presses of all kinds occurring during a particular
context
Inputs:
	-upper_rewarded: array of switches to upper lever rewarded
	-lower_rewarded: ditto for lower lever
	-session_length: 
	-context, the context to get lever presses for
Returns:
	-context_presses: timestamps of all presses that occur in the given context
"""
def get_context_levers(upper_rewarded,lower_rewarded,upper_lever,lower_lever,
	session_length,context):
	##start by getting the block boundaries
	block_times = get_block_times(lower_rewarded,upper_rewarded,session_length)
	##now just get the periods for the context that we care about
	try:
		block_windows = block_times[context]
		##now get the timestamps of all presses in each block
		presses = []
		for block in block_windows:
			##upper and lower lever timestamp arrays contain all the presses
			idx_upper = np.nonzero(np.logical_and(upper_lever>=block[0],
				upper_lever<=block[1]))[0]
			idx_lower = np.nonzero(np.logical_and(lower_lever>=block[0],
				lower_lever<=block[1]))[0]
			presses.append(upper_lever[idx_upper])
			presses.append(lower_lever[idx_lower])
		presses = np.concatenate(presses)
	except KeyError:
		presses = np.array([])
	return presses

"""
A helper function to remove the last trial, in a set of data, if the session ended
before the conclustion of the trial.
Inputs:
	ts, ts_ids: cleaned output arrays from get_mixed events that have been passed through
		remove_lever_accidents and remove_duplicate pokes
Returns:
	ts, ts_ids: duplicate arrays to the input, but if the last trial did not finish it is removed.
"""
def check_last_trial(ts,ts_ids):
	##start by getting the index of the last trial start
	last_trial_start = np.where(ts_ids=='trial_start')[0][-1]
	##now get a slice containing the event ids following the start of the last trial
	last_ids = ts_ids[last_trial_start:]
	##check to see if there was a poke in this slice, meaning the trial concluded
	if not ('rewarded_poke' in last_ids or 'unrewarded_poke' in last_ids):
		ts, ts_ids = ts[:last_trial_start], ts_ids[:last_trial_start]
	return ts, ts_ids


