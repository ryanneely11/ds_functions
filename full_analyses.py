##full_analyses
##function to analyze data from all sessions and all animals

import numpy as np
import session_analysis as sa
import parse_ephys as pe
import parse_timestamps as pt
import parse_trials as ptr
import file_lists
import os
import h5py
import PCA
import glob
import dpca

"""
A function to compute a probability of switching over trials at a reversal
boundary. 
Inputs:
	session_range: list of [start,stop] for sessions to consider
	window: number of trials [pre,post] to look at around a switch point
Returns:
	a pre_trials+post_trials length array with the probability of picking a lever 2
		at any given trial. Lever 1 is whatever lever was rewarded before the switch.
"""
def get_switch_prob(session_range=[0,4],window=[30,30]):
	##load metadata and file info from the repository file
	animals = file_lists.animals ##list of animal names
	file_dict = file_lists.split_behavior_by_animal() ##dict of file paths by animal
	master_list = [] ##to store data from all animals
	##get data for each animal
	for a in animals:
		files = file_dict[a][session_range[0]:session_range[1]]
		for i in range(len(files)):
			print("Working on file "+files[i])
			if i > 0:
				master_list.append(ptr.get_reversals(files[i],
					f_behavior_last=files[i-1],window=window))
			else:
				master_list.append(ptr.get_reversals(files[i],window=window))
	##now look at the reversal probability across all of these sessions
	master_list = np.concatenate(master_list,axis=0)
	l2_prob = get_l2_prob(master_list)
	return l2_prob

"""
A function to compute "volatility". Here, I'm defining that
as the liklihood that an animal switches levers after getting an 
unrewarded trial on one lever.
"""
def get_volatility():
	##load metadata and file info from the repository file
	animals = file_lists.animals ##list of animal names
	file_dict = file_lists.split_behavior_by_animal() ##dict of file paths by animal
	master_list = [] ##to store data from all animals
	##get data for each animal across sessions
	for a in animals:
		animal_data = [] ##to store data from each session from this animal
		files = file_dict[a]
		for f in files:
			print("Working on file "+f)
			animal_data.append(ptr.get_volatility(f))
		master_list.append(np.asarray(animal_data))
	##convert to an even-sized np array
	master_list = equalize_arrs(master_list)
	return master_list	

"""
A function to compute "persistence". Here, I'm defining that
as the liklihood that an animal switches levers after getting a
rewarded trial on one lever.
"""
def get_persistence():
	##load metadata and file info from the repository file
	animals = file_lists.animals ##list of animal names
	file_dict = file_lists.split_behavior_by_animal() ##dict of file paths by animal
	master_list = [] ##to store data from all animals
	##get data for each animal across sessions
	for a in animals:
		animal_data = [] ##to store data from each session from this animal
		files = file_dict[a]
		for f in files:
			print("Working on file "+f)
			animal_data.append(ptr.get_persistence(f))
		master_list.append(np.asarray(animal_data))
	##convert to an even-sized np array
	master_list = equalize_arrs(master_list)
	return master_list	

"""
A function that calculates how many trials it takes for an animal to 
reach a criterion level of peformance after block switches. Function returns the
mean trials after all block switches in a session, for all sessions and animals.
Inputs:
	crit_trials: the number of trials to average over to determine performance
	crit_level: the criterion performance level to use
	exclude_first: whether to exclude the first block, but only if there is more than one block.
Returns:
	n_trials: an n-animal x s-session array with the trials to criterion for
	all sessions for all animals. Uneven session lengths are masked with NaN.	
"""
def get_criterion(crit_trials=5,crit_level=0.7,exclude_first=False):
		##load metadata and file info from the repository file
	animals = file_lists.animals ##list of animal names
	file_dict = file_lists.split_behavior_by_animal() ##dict of file paths by animal
	master_list = [] ##to store data from all animals
	##get data for each animal across sessions
	for a in animals:
		animal_data = [] ##to store data from each session from this animal
		files = file_dict[a]
		for f in files:
			print("Working on file "+f)
			animal_data.append(ptr.mean_trials_to_crit(f,crit_trials=crit_trials,
				crit_level=crit_level,exclude_first=exclude_first))
		master_list.append(np.asarray(animal_data))
	##convert to an even-sized np array
	master_list = equalize_arrs(master_list)
	return master_list

"""
A function to calulate the percent correct across sessions for all animals.
Inputs:
	
Returns:
	p_correct: an n-animal x s-session array with the percent correct across
	all trials for all animals. Uneven session lengths are masked with NaN.
"""
def get_p_correct():
	##load metadata and file info from the repository file
	animals = file_lists.animals ##list of animal names
	file_dict = file_lists.split_behavior_by_animal() ##dict of file paths by animal
	master_list = [] ##to store data from all animals
	##get data for each animal across sessions
	for a in animals:
		animal_data = [] ##to store data from each session from this animal
		files = file_dict[a]
		for f in files:
			print("Working on file "+f)
			animal_data.append(ptr.calc_p_correct(f))
		master_list.append(np.asarray(animal_data))
	##convert to an even-sized np array
	master_list = equalize_arrs(master_list)
	return master_list

"""
A function to get a dataset that includes data from all animals and sessions,
specifically formatted to use in a dPCA analysis.
Inputs:
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
def get_dpca_dataset(smooth_method='both',smooth_width=[40,50],
	pad=[200,200],z_score=True,remove_unrew=True):
	##a container to store all of the X_trials data
	X_all = []
	ts_all = []
	ts_idx_all = None
	##some metrics to help us keep track of things
	max_trials = 0
	n_neurons = 0
	for f_behavior,f_ephys in zip(file_lists.behavior_files,file_lists.ephys_files):
		current_file = f_behavior[-11:-5]
		print("Adding data from file "+current_file)
		##get the raw data matrix first 
		X_raw = pe.get_spike_data(f_ephys,smooth_method='none',smooth_width=None,
			z_score=False) ##don't zscore or smooth anything yet
		##save the number of neurons for this session
		n_neurons += X_raw.shape[0]
		##now get the window data for the trials in this session
		ts,ts_idx = pt.get_trial_data(f_behavior,remove_unrew=remove_unrew) #ts is shape trials x ts, and in seconds
		##store the info about number of trials here
		if ts.shape[0] > max_trials:
			max_trials = ts.shape[0]
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
		##add the data to the master lists
		X_all.append(X_trials)
		ts_all.append(ts)
		if ts_idx_all != None:
			ts_idx_all = concat_ts_idx(ts_idx_all,ts_idx)
		else:
			ts_idx_all = ts_idx
	return X_all,ts_all,ts_idx_all
	##now we need to get the median trial duration over ALL trials.
	# median_dur = get_median_dur(ts_all)
	# ##now, for each session, we want to interpolate all the trials to match this
	# ##master trial length.
	# for s in range(len(X_all)):
	# 	session_data = X_all[s]
	# 	session_ts = ts_all[s]
	# 	##get the timestamps relative to the start of trials
	# 	ts_rel = dpca.get_relative_ts(session_ts,pad,smooth_method,smooth_width)
	# 	##now run the data through the stretch trials function
	# 	stretched = dpca.stretch_trials(session_data,ts_rel,median_dur=median_dur)
	# 	##finally, replace it in the master list
	# 	X_all[s] = stretched
	# ##OK, now all sessions should be the same length.
	# ##let's construct the data container that will hold all the data
	# X_full = np.empty((max_trials,n_neurons,X_all[0][0].shape[1]))
	# ##keep track of where we are putting neurons 
	# n_idx = 0
	# ##now fill up the array
	# for s in range(len(X_all)):
	# 	session_data = np.asarray(X_all[s])
	# 	X_full[0:session_data.shape[0],n_idx:n_idx+session_data.shape[1],:] = session_data
	# return X_full, ts_idx_all
"""
A function to run and compile regression data for ALL sessions
	-epoch_durations: the duration, in seconds, of the epochs (see the list in the function)
	-smooth_method: type of smoothing to use; choose 'bins', 'gauss', or 'none'
	-smooth_width: size of the bins or gaussian kernel in ms
	-z_score: if True, z-scores the array
			$$NOTE$$: this implementatin does not allow binning AND gaussian smoothing.
	-save: if True, saves data at the end
Returns:
	-C: matrix of coefficients, shape units x regressors x bins (all epochs are concatenated)
	-num_sig: matrix with the counts of units showing significant regression
		values at for each regressor at each bin (regressors x bins)
	-mse: mean squared error of the model fit at each bin 
		based on x-validation (size = bins)
	-epoch_idx: the indices of bins corresponding to the different epochs
"""
def full_regression(epoch_durations=[1,0.5,1,1],smooth_method='bins',
	smooth_width=50,z_score=False,save=True):
	##run session regression on all files in the lists
	results_files = []
	for f_behavior,f_ephys in zip(file_lists.behavior_files,file_lists.ephys_files):
		current_file = f_behavior[-11:-5]
		print("Starting on file "+current_file)
		out_path = os.path.join(file_lists.save_loc,current_file+".hdf5")
		try: 
			f_out = h5py.File(out_path,'w-')
			##if data does not exist, calculate it and save it
			c,ns,mse,epoch_idx = sa.session_regression(f_behavior,f_ephys,
				epoch_durations=epoch_durations,smooth_method=smooth_method,
				smooth_width=smooth_width,z_score=z_score)
			f_out.create_dataset("coeffs",data=c)
			f_out.create_dataset("num_sig",data=ns)
			f_out.create_dataset("mse",data=mse)
			for key in epoch_idx.keys():
				f_out.create_dataset(key,data=epoch_idx[key])
			f_out.close()
			results_files.append(out_path)
		except IOError:
			results_files.append(out_path)
			print(current_file+" exists, moving on...")
	##if all data is saved, you can go back to the files and save the data that you want.
	##data to return
	coeffs = []
	num_sig = []
	mse = []
	num_total_units = 0
	for results_file in results_files:
		epoch_idx = get_epoch_idx_dict(results_file)
		f_in = h5py.File(results_file,'r')
		c = np.asarray(f_in['coeffs'])
		ns =np.asarray(f_in['num_sig'])
		m = np.asarray(f_in['mse'])
		num_units = c.shape[0]
		coeffs.append(c)
		num_sig.append(ns)
		mse.append(m)
		num_total_units+=num_units
	##concatenate all arrays
	coeffs = np.concatenate(coeffs,axis=0)
	num_sig = np.asarray(num_sig).sum(axis=0)
	mse = np.asarray(mse).mean(axis=0)
	if save:
		out_path = os.path.join(file_lists.save_loc,"all_files_regression.hdf5")
		f_out = h5py.File(out_path,'w-')
		f_out.create_dataset("coeffs",data=coeffs)
		f_out.create_dataset("num_sig",data=num_sig)
		f_out.create_dataset("mse",data=mse)
		f_out.create_dataset("num_units",data=np.array([num_total_units]))
		for key in epoch_idx.keys():
			f_out.create_dataset(key,data=epoch_idx[key])
		f_out.close()
	return coeffs,num_sig,mse,epoch_idx,num_total_units

"""
A function to get the condition-averaged responses for all sessions in one giant 
data matrix. The conditions are not concatenated, but just use the function
in parse_ephys to get the matrix for PCA.
Inputs:
	-epoch: (str), name of epoch to use for trial data
	-epoch_duration: duration of epoch to look at (in sec)
	-smooth_method: smooth method to use; see ephys functions
	-smooth_width: smooth width, lol
	-use_unrewarded: bool, whether to include or exclude unrewarded trials.
		if included, these become their own condition.
Returns: 
	X. data matrix of size conditions x units x bins
	order: list of strings defining the order of the conditions in the matrix
"""
def cond_avg_matrix(epoch,epoch_duration,smooth_method='bins',smooth_width=50,
	use_unrewarded='True'):
	##go through all files
	Xc = []
	conditions = []
	for f_behavior,f_ephys in zip(file_lists.behavior_files,file_lists.ephys_files):
		current_file = f_behavior[-11:-5]
		print("Starting on file "+current_file)
		x,cond = sa.condition_averaged_responses(f_behavior,f_ephys,epoch_duration=epoch_duration,
			smooth_method=smooth_method,smooth_width=smooth_width,use_unrewarded=use_unrewarded)
		Xc.append(x)
		conditions.append(cond)
	##make sure the conditions are all in the same order
	for c in range(len(conditions)):
		assert conditions[c] == conditions[0]
	return np.concatenate(Xc,axis=1),conditions[0]


"""
A function to run the logistic regression on all files.
No outputs are created, but data is saved to individual HDf5 files.
Inputs: 
	-epoch_durations: the duration, in seconds, of the epochs (see the list in the function)
	-smooth_method: type of smoothing to use; choose 'bins', 'gauss', or 'none'
	-smooth_width: size of the bins or gaussian kernel in ms
	-z_score: if True, z-scores the array
REturns: 
	None,but data is saved.
"""
def full_log_regression(epoch_durations=[1,0.5,1,1],smooth_method='bins',smooth_width=200,
	z_score=True,save=True):
	for f_behavior,f_ephys in zip(file_lists.behavior_files,file_lists.ephys_files):
		current_file = f_behavior[-11:-5]
		print("Starting on file "+current_file)
		try:
			results = sa.log_regress_session(f_behavior,f_ephys,epoch_durations=epoch_durations,
				smooth_method=smooth_method,smooth_width=smooth_width,z_score=z_score,save=save)
		except IOError:
			print(current_file+" exists, skipping...")
	print("Done!")
	return None


"""
A function to run analyses on logistic regression data. Input is a list of
directories, so it will analyze one or more animals. Function looks for hdf5 files,
so any hdf5 files in the directories should only be regression results files.
Inputs:
	dir_list: list of directories where data is stored.
	epochs: list, optional. If you only want to take data from certain epochs.
Returns:
	results: dictionary of results for each directory (animal)
"""
def analyze_log_regressions(dir_list,epochs=None):
	##assume the folder name is the animal name, and that it is two characters
	##also assuming that we are looking at the following conditions:
	##set up the results dictionary
	results = {
		'num_block_type':[],
		'num_choice':[],
		'num_reward':[],
		'multi_units':[],
		'num_sig':[],
		'num_total':[]}
	for d in dir_list:
		name = d[-11:-9]
		##get the list of files in this directory
		flist = get_file_names(d)
		flist.sort() ##get them in order of training session
		n_bt = np.zeros(len(flist)) ##these are the arrays to go into the results lists
		n_c = np.zeros(len(flist))
		n_r = np.zeros(len(flist))
		n_mu = np.zeros(len(flist))
		n_s = np.zeros(len(flist))
		n_t = np.zeros(len(flist))
		for n,f in enumerate(flist):
			##parse the results for this file
			cond_idx,cond_ps,multis,n_sig,n_total = sa.parse_log_regression(f,epochs)
			##add the data to the arrays
			n_bt[n] = cond_idx['block_type'].size
			n_c[n] = cond_idx['choice'].size
			n_r[n] = cond_idx['reward'].size
			n_mu[n] = multis.size
			n_s[n] = n_sig.size
			n_t[n] = n_total
		##now add the data to the master lists
		results['num_block_type'].append(n_bt)
		results['num_choice'].append(n_c)
		results['num_reward'].append(n_r)
		results['multi_units'].append(n_mu)
		results['num_sig'].append(n_s)
		results['num_total'].append(n_t)	
	##now get these all into a pretty array
	results['num_block_type'] = equalize_arrs(results['num_block_type'])
	results['num_choice'] = equalize_arrs(results['num_choice'])
	results['num_reward'] = equalize_arrs(results['num_reward'])
	results['multi_units'] = equalize_arrs(results['multi_units'])
	results['num_sig'] = equalize_arrs(results['num_sig'])
	results['num_total'] = equalize_arrs(results['num_total'])
	return results


"""
A function to project neural data from various conditions onto various
axes defined by de-noised regression vectors.
Inputs:
	-f_regression: HDF5 file where the regression data is stored
	-f_data: HDF5 file where the data matrix is stored
	=epoch: str of the epoch name used to compile data for the spike matrix
Returns:
	-results: a dictionary of all the various projections onto different axes
"""
def condition_projections(f_regression,f_data,epoch='choice',n_pcs=12):
	##retrieve the regression data
	f = h5py.File(f_regression,'r')
	R = np.asarray(f['coeffs'])
	idx = np.asarray(f[epoch])
	R = R[:,:,idx]
	f.close()
	##now retrieve the data matrix
	f = h5py.File(f_data,'r')
	Xc = np.asarray(f['data'])
	conditions = list(np.asarray(f['conditions']))
	f.close()
	result = PCA.value_projections(Xc,R,conditions,n_pcs)
	return result

"""
A helper function to concatenate dictionaries of timestamp indices. 
Useful when you are combining trials from many sessions but want to 
keep track of which trials where which types.
Inputs:
	master: master ts_idx dictionary to add to.
	to_add: ts_idx dictionary whose indices need to be added to the master.
Returns:
	master: master ts_idx with new indices added and offset appropriately
"""
def concat_ts_idx(master,to_add):
	##we'll copy master just to make sure we don't mess anything up
	new_master = master.copy()
	##first we need to figure out how many trials are already in the master
	offset = 0
	for key in new_master.keys():
		if max(new_master[key])>offset:
			offset = max(new_master[key])
	##now add the indices to the master, accounting for the offset
	for key in new_master.keys():
		new_master[key] = np.concatenate((new_master[key],to_add[key]+(offset+1)))
	return new_master


"""
a helper function to get epoch_idx dictionary from a file of regression data
Inputs:
	-f_in: file path of HDF5 file with regression data
Returns:
	-epoch_idx dictionary with the indices corresponding to the various epochs (dict keys)
"""
def get_epoch_idx_dict(f_in):
	epoch_idx = {
	"choice":None,
	"action":None,
	"delay":None,
	"outcome":None
	}
	f = h5py.File(f_in,'r')
	for key in epoch_idx.keys():
		epoch_idx[key] = np.asarray(f[key])
	f.close()
	return epoch_idx

##returns a list of file paths for all hdf5 files in a directory
def get_file_names(directory):
	##get the current dir so you can return to it
	cd = os.getcwd()
	filepaths = []
	os.chdir(directory)
	for f in glob.glob("*.hdf5"):
		filepaths.append(os.path.join(directory,f))
	os.chdir(cd)
	return filepaths

"""
a function to equalize the length of different-length arrays
by adding np.nans
Inputs:
	-list of arrays (1-d) of different shapes
Returns:
	2-d array of even size
"""
def equalize_arrs(arrlist):
	longest = 0
	for i in range(len(arrlist)):
		if arrlist[i].shape[0] > longest:
			longest = arrlist[i].shape[0]
	result = np.zeros((len(arrlist),longest))
	result[:] = np.nan
	for i in range(len(arrlist)):
		result[i,0:arrlist[i].shape[0]] = arrlist[i]
	return result

"""
A helper function to get the median trial duration for a list of timestamp 
arrays.
Inputs:
	ts_list: a list of timetamp arrays from a number of sessions
Returns:
	median_dur: the median trial duration, in whatever units the list was
"""
def get_median_dur(ts_list):
	##store a master list of durations
	durations = np.array([])
	for s in range(len(ts_list)):
		durations = np.concatenate((durations,(ts_list[s][:,1]-ts_list[s][:,0])))
	return int(np.ceil(np.median(durations)))

"""
A helper function to be used with reversal data. Computes the probability
of a lever 2 press as a function of trial number.
Inputs:
	an array of stacked reversal data, in the format n_reversals x n_trials
Returns:
	an n-trial array with the probability of a lever 2 press at each trial number
"""
def get_l2_prob(reversal_data):
	result = np.zeros(reversal_data.shape[1])
	for i in range(reversal_data.shape[1]):
		n_l1 = (reversal_data[:,i] == 1).sum()
		n_l2 = (reversal_data[:,i] == 2).sum()
		result[i] = float(n_l2)/float(n_l1+n_l2)
	return result
