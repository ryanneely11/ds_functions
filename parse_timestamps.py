##parse_timestamps.py
##functions to parse timestamp data
import h5py
import numpy as np

"""
A function that takes in a behavior timestamps file (HDF5)
and returns an array of timestamps, as well as an index
of what catagory each trial falls into (ie rewarded, unrewarded, 
upper lever, etc)
Inputs:
	-f_in: file path to where the data is
Returns:
	-ts: trial x event array. Axis 0 is trials, axis 1 is action, outcome
	-ts_idx: dictionary of array containing the indices of different 
		trial types.
"""
def get_trial_data(f_in):
	ts_idx = {}
	##sort_by_trial does most of the work:
	block_data = sort_by_trial(f_in)
	##now just parse the TS a little further
	block_types = block_data.keys() ##keep the block order (upper_rewarded, etc consistent)
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
	##return the results:
	return ts,ts_idx


""" a function to split behavior timestamp data into individual trials.
	Inputs:
		f_in: the file path pointing to an hdf5 file containing
		the behavior event time stamps
	Returns: 
		Two n-trial x i behavioral events-sized arrays.
		One contains all trials where the lower lever is rewarded;
		The other contains all the trials where the upper lever is rewarded. 
		Behavioral events are indexed temporally:
		0-trial start; 1-action (U or L); 2-poke(R or U)
"""
def sort_by_trial(f_in):
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
				print "something wrong in upper/lower comparison"
				break
		#case 2: only upper lever was pressed
		elif upper_idx.size>0 and lower_idx.size==0:
			action = block_data['upper_lever'][upper_idx].min()
		##case 3: only lower lever was pressed
		elif upper_idx.size==0 and lower_idx.size>0:
			action = -1*block_data['lower_lever'][lower_idx].min()
		##case 4: something is wrong!
		else:
			print "Error- no action for this trial??"
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
				print "Error: no pokes for this trial"
				break
		else:
			print "error: too many rewarded pokes for this trial"
			break

		##now add the data to the results
		result[i,0] = trial_start
		result[i,1] = action
		result[i,2] = outcome
	##now make sure none of the trials violate the max trial duration
	t = 0
	while t < result.shape[0]:
		if (abs(result[t,2])-abs(result[t,1])) >= trial_max:
			print "removing trial of length "+str(abs(result[t,2])-abs(result[t,1]))
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
			print "Block length only "+str(stop-start)+" secs; excluding"
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
	keys = data_dict.keys()
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