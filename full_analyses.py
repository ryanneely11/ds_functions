##full_analyses
##function to analyze data from all sessions and all animals

import numpy as np
import session_analysis as sa
import parse_ephys as pe
import parse_timestamps as pt
import parse_trials as ptr
import plotting as ptt
import file_lists
import os
import h5py
import PCA
import glob
import dpca
import pandas as pd




"""
A function to run logistic regression using the new functions in lr3.
	-window: time window to use for analysis, in ms
	-smooth_method: type of smoothing to use; choose 'bins', 'gauss', or 'none'
	-smooth_width: size of the bins or gaussian kernel in ms
	-z_score: if True, z-scores the array
	
Returns:
	all data saved to file.
"""
def log_regression2(window=500,smooth_method='gauss',smooth_width=30,z_score=True,
	min_rate=0.1):
	for f_behavior,f_ephys in zip(file_lists.e_behavior,file_lists.ephys_files):
		sa.log_regress_session(f_behavior,f_ephys,win=window,
			smooth_method=smooth_method,smooth_width=smooth_width,z_score=z_score,
			min_rate=min_rate)
	print("All regressions complete")



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
This function returns an array of every trial duration for every animal.
Inputs:
	max_duration: maximum allowable duration of trials (anything longer is deleted)
	session_range: endpoints of a range of session numbers [start,stop] to restrict analysis to (optional)
Returns:
	trial_durs: an of trial durations
"""
def get_trial_durations(max_duration=5000,session_range=None):
	all_durations = []
	if session_range is not None:
		files = [x for x in file_lists.behavior_files if int(x[-7:-5]) in np.arange(session_range[0],session_range[1])]
	else:
		files = file_lists.behavior_files
	for f_behavior in files:
		all_durations.append(sa.session_trial_durations(f_behavior,max_duration=max_duration))
	return np.concatenate(all_durations)

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
	balance_trials: if True, equates the number of trials across all conditions by removal
	min_trials: minimum number of trials required for each trial type. If a session doesn't meet 
		this criteria, it will be excluded.
Returns:
	X_c: data from individual trials;
		 shape n-trials x n-neurons x condition-1 x condition-2, ... x n-timebins
"""
def get_dpca_dataset(conditions,smooth_method='both',smooth_width=[80,40],pad=[400,400],
	z_score=True,max_duration=5000,min_rate=0.1,balance=True,min_trials=15):
	##a container to store all of the X_trials data
	X_all = []
	##the first step is to determine the median trial length for all sessions
	med_duration = np.median(get_trial_durations(max_duration=max_duration,session_range=None)).astype(int)
	for f_behavior,f_ephys in zip(file_lists.e_behavior,file_lists.ephys_files):
		current_file = f_behavior[-11:-5]
		print("Adding data from file "+current_file)
		##append the dataset from this session
		dataset = dpca.get_dataset(f_behavior,f_ephys,conditions,smooth_method=smooth_method,
			smooth_width=smooth_width,pad=pad,z_score=z_score,trial_duration=med_duration,
			max_duration=max_duration,min_rate=min_rate,balance=balance)
		if dataset is not None:
			X_all.append(dataset)
	##now, get an idea of how many trials we have per dataset
	n_trials = [] ##keep track of how many trials are in each session
	include = [] ##keep track of which sessions have more then the min number of trials
	for i in range(len(X_all)):
		n = X_all[i].shape[0]
		if n >= min_trials:
			include.append(i)
			n_trials.append(n)
	##from this set, what is the minimum number of trials?
	print("Including {0!s} sessions out of {1!s}".format(len(include),len(X_all)))
	min_trials = min(n_trials)
	##now all we have to do is concatenate everything together!
	X_c = []
	for s in include:
		X_c.append(X_all[s][0:min_trials,:,:,:,:])
	return np.concatenate(X_c,axis=1)
"""
Same as "get dpca dataset" but just goes the final step of running dpca, saving the results,
and plotting the data.
"""
def save_dpca(conditions,smooth_method='both',smooth_width=[80,40],pad=[400,400],
	z_score=True,max_duration=5000,min_rate=0.1,balance=True,min_trials=15,plot=True):
	save_file = "/Volumes/Untitled/Ryan/DS_animals/results/dPCA/{}+{}.hdf5".format(conditions[0],conditions[1])
	X_c = get_dpca_dataset(conditions,smooth_method=smooth_method,smooth_width=smooth_width,
		pad=pad,z_score=z_score,max_duration=max_duration,min_rate=min_rate,balance=balance,min_trials=min_trials)
	##fit the data
	Z,var_explained,sig_masks = dpca.run_dpca(X_c,10,conditions)
	##now save:
	f = h5py.File(save_file,'w')
	f.create_dataset("X_c",data=X_c)
	f.create_group("Z")
	for key,value in zip(Z.keys(),Z.values()):
		f['Z'].create_dataset(key,data=value)
	f.create_group("var_explained")
	for key,value in zip(var_explained.keys(),var_explained.values()):
		f['var_explained'].create_dataset(key,data=value)
	f.create_group("sig_masks")
	for key,value in zip(sig_masks.keys(),sig_masks.values()):
		f['sig_masks'].create_dataset(key,data=value)
	f.close()
	if plot:
		ptt.plot_dpca_results(Z,var_explained,sig_masks,conditions,smooth_width[1],
			pad=pad,n_components=3)




"""
A function to run analyses on logistic regression data. Input is a list of
directories, so it will analyze one or more animals. Function looks for hdf5 files,
so any hdf5 files in the directories should only be regression results files.
Inputs:
	dir_list: list of directories where data is stored.
Returns:
	results: dictionary of results for each directory (animal)
"""
def analyze_log_regressions(dir_list,session_range=None,sig_level=0.05,test_type='llr_pvals'):
	##assume the folder name is the animal name, and that it is two characters
	##set up a results dictionary
	epochs = ['action','context','outcome'] ##this could change in future implementations
	##create a dataframe object to store all of the data across sessions/animals
	all_data = pd.DataFrame(columns=epochs)
	cursor = 0 #keep track of how many units we're adding to the dataframe
	for d in dir_list:
		##get the list of files in this directory
		flist = get_file_names(d)
		flist.sort() ##get them in order of training session
		##take only the requested sessions, if desired
		if session_range is not None:
			flist = flist[session_range[0]:session_range[1]]
		for f in flist:
			##parse the results for this file
			results = sa.parse_log_regression(f,sig_level=sig_level,test_type=test_type)
			##get an array of EVERY significant unit across all epochs
			sig_idx = []
			for epoch in epochs:
				try:
					sig_idx.append(results[epoch]['idx'])
				except KeyError: ##case where there were no sig units in this epoch
					pass
			sig_idx = np.unique(np.concatenate(sig_idx)) ##unique gets rid of any duplicates
			##construct a pandas dataframe object to store the data form this session
			global_idx = np.arange(cursor,cursor+sig_idx.size) ##the index relative to all the other data so far
			data = pd.DataFrame(columns=epochs,index=global_idx)
			##now run through each epoch, and add data to the local dataframe
			for epoch in epochs:
				try:
					epoch_sig = results[epoch]['idx'] ##the indices of significant units in this epoch
					for i,unitnum in enumerate(epoch_sig):
						#get the index of this unitnumber in reference to the local dataframe
						unit_idx = np.argwhere(sig_idx==unitnum)[0][0] ##first in the master sig list
						unit_idx = global_idx[unit_idx]
						##we know all units meet significance criteria, so just save the accuracy
						data[epoch][unit_idx] = results[epoch]['accuracy'][i]
				except KeyError: ##no sig units in this epoch
					pass
			all_data = all_data.append(data)
			cursor += sig_idx.size
	return all_data

"""
A function to return some metadata about all files recorded
inputs:
	max_duration: trial limit threshold (ms)
"""
def get_metadata(max_duration=5000):
	metadata = {
	'training_sessions':0,
	'sessions_with_ephys':0,
	'rewarded_trials':0,
	'unrewarded_trials':0,
	'upper_context_trials':0,
	'lower_context_trials':0,
	'upper_lever_trials':0,
	'lower_lever_trials':0,
	'units_recorded':0
	}
	for f_behavior in file_lists.behavior_files:
		meta = sa.get_session_meta(f_behavior,max_duration)
		metadata['training_sessions']+=1
		metadata['rewarded_trials']+=meta['rewarded'].size
		metadata['unrewarded_trials']+=meta['unrewarded'].size
		metadata['upper_context_trials']+=meta['upper_context'].size
		metadata['lower_context_trials']+=meta['lower_context'].size
		metadata['lower_lever_trials']+=meta['lower_lever'].size
		metadata['upper_lever_trials']+=meta['upper_lever'].size
	for f_ephys in file_lists.ephys_files:
		metadata['sessions_with_ephys']+=1
		metadata['units_recorded']+=sa.get_n_units(f_ephys)
	return metadata



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

