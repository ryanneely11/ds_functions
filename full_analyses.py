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
import multiprocessing as mp
import log_regression3 as lr3
import log_regression2 as lr2
import model_fitting as mf
import file_lists_unsorted as flu

"""
A function to compute decision variables
Inputs:
	-window [pre_event, post_event] window, in ms
	-smooth_method: type of smoothing to use; choose 'bins', 'gauss', 'both', or 'none'
	-smooth_width: size of the bins or gaussian kernel in ms. If 'both', input should be a list
		with index 0 being the gaussian width and index 1 being the bin width
	-min_rate: minimum acceptable spike rate, in Hz
"""
def decision_vars(window=[800,100],smooth_method='bins',smooth_width=100,min_rate=0.5):
	upper_odds = []
	lower_odds = []
	for f_behavior, f_ephys in zip(file_lists.e_behavior,file_lists.ephys_files):
		try:
			u,l = sa.decision_variables(f_behavior,f_ephys,window=window,smooth_method=smooth_method,
				smooth_width=smooth_width,min_rate=min_rate)
			upper_odds.append(u)
			lower_odds.append(l)
		except:
			print("warning- skipping one file")
	return np.concatenate(upper_odds,axis=0),np.concatenate(lower_odds,axis=0)


"""
Model fitting: this function fits an HMM model and an RL model to each behavioral
training session, and plots the goodness-of-fit for each model type over time for 
all animals.
Inputs:
	None
Returns:
	-RL_fits: log-liklihood across sessions for each animal; animals x sessions 
	-HMM_fits: ll across sessions for each animal 
"""
def model_fits():
	files_by_animal = file_lists.split_behavior_by_animal()
	n_sessions = max([len(x) for x in files_by_animal.values()])-6 ##first 6 sessions only have 1 lever
	n_animals = len(list(files_by_animal))
	HMM_fits = np.zeros((n_animals,n_sessions))
	RL_fits = np.zeros((n_animals,n_sessions))
	HMM_fits[:] = np.nan
	RL_fits[:] = np.nan
	for n,animal in enumerate(file_lists.animals):
		sessions = files_by_animal[animal][6:] ##first 6 sessions only have 1 lever
		for m, f_behavior in enumerate(sessions):
			results = mf.fit_models(f_behavior)
			HMM_fits[n,m] = results['ll_HMM']
			RL_fits[n,m] = results['ll_RL']
	return RL_fits,HMM_fits

"""
A possibly better way to look at model fits by using prediction accuracy
as the metric, and fitting models using data concatenated across training 
sessions for individual animals.
Inputs:
	win: sliding window in form [window, winstep]
Returns:
	RL_fits: accuracy of RL model over time for each animal
	HMM_fits: accuracy of HMM model over time for each animal
"""
def model_fits2(win=[100,25]):
	animals = file_lists.animals
	RL_fits = []
	HMM_fits = []
	for animal in animals:
		results = mf.fit_models_all(animal)
		RL_actions = results['RL_actions']
		HMM_actions = results['HMM_actions']
		actions = results['actions']
		RL_fits.append(mf.sliding_accuracy(actions,RL_actions,win))
		HMM_fits.append(mf.sliding_accuracy(actions,HMM_actions,win))
	##add NANs to equalize the length of the arrays
	RL_fits = equalize_arrs(RL_fits)
	HMM_fits = equalize_arrs(HMM_fits)
	return RL_fits,HMM_fits


"""
A function to get parameterized behavior data arrays for all
sessions, in order to build a logistic regression model. 
"""
def behavior_regression(n_back=3,max_duration=5000):
	##create a container for all the dataframes
	X_all = []
	y_all = []
	for f_behavior in file_lists.e_behavior:
		y,X = lr3.get_behavior_data(f_behavior,n_back=n_back,max_duration=max_duration)
		X_all.append(X)
		y_all.append(y)
	##now concatenate all of them
	X_all = pd.concat(X_all,ignore_index=True)
	y_all = pd.concat(y_all,ignore_index=True)
	##the pandas dataset format may come in handy in the future, but
	##for now I'll just convert it to numpy arrays for easy interfacing with the lr2/3 functions
	##it also appears that the session number and trial numbers aren't 
	##really important so I'll leave those out
	return np.asarray(y_all['value']).astype(float),np.asarray(X_all).astype(float)


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
A function to concatenate the spike data from all sessions for a given animal.
This function is meant to operate on the assumtion that the number of spike
channels (and roughly the identity of neurons on those channels) stays constant.
An error will be returned if there are variations in spike channel number across days.
Inputs:
	animal_id: string identifier of animal to get data for
	smooth_method: type of smoothing to use; choose 'bins', 'gauss', 'both', or 'none'
	smooth_width: size of the bins or gaussian kernel in ms. If both, input should be a list
		with index 0 being the gaussian width and index 1 being the bin width
Returns:
	X: spike data matrix of size units x bins. Data is z-scored WITHIN each DAY
		in an attempt to make things at least a little bit consistent
"""
def concatenate_spikes(animal_id,smooth_method='both',smooth_width=[80,40]):
	##get the file names from the multiunit hash
	ephys_files = flu.split_ephys_by_animal()[animal_id]
	data = [] ##container for individual data pieces
	for f_ephys in ephys_files:
		data.append(pe.get_spike_data(f_ephys,smooth_method=smooth_method,
			smooth_width=smooth_width,z_score=True))
	X = np.concatenate(data)



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
	arglist = [[f_behavior,f_ephys,conditions,smooth_method,smooth_width,pad,z_score,med_duration,
				max_duration,min_rate,balance] for f_behavior,f_ephys in zip(file_lists.e_behavior,
					file_lists.ephys_files)]
	##assign data collection to multiple processes
	pool = mp.Pool(processes=8)
	async_result = pool.map_async(dpca.get_dataset_mp,arglist)
	pool.close()
	pool.join()
	results = async_result.get()
	for i in range(len(results)):
		if results[i].size > 1:
			X_all.append(results[i])
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
	#now all we have to do is concatenate everything together!
	X_c = []
	for s in include:
		X_c.append(X_all[s][0:min_trials,:,:,:,:])
	return np.concatenate(X_c,axis=1)
	return X_c

"""
A function to get a dpca dataset, but only include trials that fit a certain belief
state criteria as defined by a hidden markov model fitted to the behavioral data.
Inputs:
	smooth_method: type of smoothing to use; choose 'bins', 'gauss', 'both', or 'none'
	smooth_width: size of the bins or gaussian kernel in ms. If 'both', input should be a list
		with index 0 being the gaussian width and index 1 being the bin width
	pad: a window for pre- and post-trial padding, in ms. In other words, an x-ms period of time 
		before lever press to consider the start of the trial, and an x-ms period of time after
		reward to consider the end of the trial
	balance_trials: if True, equates the number of trials across all conditions by removal
	min_trials: minimum number of trials required for each trial type. If a session doesn't meet 
		this criteria, it will be excluded.
	belief_range: the range of belief strengths to accept
Returns:
	X_c: data from individual trials;
		 shape n-trials x n-neurons x condition-1 x condition-2, ... x n-timebins
"""
def get_dpca_dataset_hmm(conditions,smooth_method='both',smooth_width=[80,40],pad=[400,400],
	max_duration=5000,min_rate=0.1,balance=True,min_trials=15,belief_range=(0,0.1)):
	##a container to store all of the X_trials data
	X_all_c = []
	X_all_b = []
	##the first step is to determine the median trial length for all sessions
	med_duration = np.median(get_trial_durations(max_duration=max_duration,session_range=None)).astype(int)
	arglist = [[f_behavior,f_ephys,conditions,smooth_method,smooth_width,pad,med_duration,
				max_duration,min_rate,belief_range] for f_behavior,f_ephys in zip(file_lists.e_behavior,
					file_lists.ephys_files)]
	##assign data collection to multiple processes
	pool = mp.Pool(processes=8)
	async_result = pool.map_async(dpca.get_hmm_mp,arglist)
	pool.close()
	pool.join()
	results = async_result.get()
	for i in range(len(results)):
		if results[i][0].size > 1 and results[i][1].size > 1:
			X_all_c.append(results[i][0])
			X_all_b.append(results[i][1])
	assert len(X_all_c) == len(X_all_b)
	##now, get an idea of how many trials we have per dataset
	n_trials_c = [] ##keep track of how many trials are in each session
	n_trials_b = []
	include = [] ##keep track of which sessions have more then the min number of trials
	for i in range(len(X_all_c)):
		nc = X_all_c[i].shape[0]
		nb = X_all_b[i].shape[0]
		if nc >= min_trials and nb >= min_trials:
			include.append(i)
			n_trials_c.append(nc)
			n_trials_b.append(nb)
	##from this set, what is the minimum number of trials?
	print("Including {0!s} sessions out of {1!s}".format(len(include),len(X_all_c)))
	min_trials_c = min(n_trials_c)
	min_trials_b = min(n_trials_b)
	#now all we have to do is concatenate everything together!
	X_c = []
	X_b = []
	for s in include:
		X_c.append(X_all_c[s][0:min_trials_c,:,:,:,:])
		X_b.append(X_all_b[s][0:min_trials_b,:,:,:,:])
	X_c = np.concatenate(X_c,axis=1)
	X_b = np.concatenate(X_b,axis=1)
	return X_c, X_b

"""
A very similar function to get_dpca_dataset, but just returns data from all the sessions
in a list, without balancing the number of trials across sessions.
Useful for running dpca on each dataset individually.
Inputs:
	smooth_method: type of smoothing to use; choose 'bins', 'gauss', 'both', or 'none'
	smooth_width: size of the bins or gaussian kernel in ms. If 'both', input should be a list
		with index 0 being the gaussian width and index 1 being the bin width
	pad: a window for pre- and post-trial padding, in ms. In other words, an x-ms period of time 
		before lever press to consider the start of the trial, and an x-ms period of time after
		reward to consider the end of the trial
	z_score: if True, z-scores the array
	balance_trials: if True, equates the number of trials across all conditions by removal
		but only for each session independently
Returns:
	X_c: data from individual sessions; a list with each dataset in the format
		 shape n-trials x n-neurons x condition-1 x condition-2, ... x n-timebins
"""
def get_dpca_datasets_all(conditions,smooth_method='both',smooth_width=[80,40],pad=[400,400],
	z_score=True,max_duration=5000,min_rate=0.1,balance=True):
	##a container to store all of the X_trials data
	X_all = []
	session_names = []
	##the first step is to determine the median trial length for all sessions
	med_duration = np.median(get_trial_durations(max_duration=max_duration,session_range=None)).astype(int)
	arglist = [[f_behavior,f_ephys,conditions,smooth_method,smooth_width,pad,z_score,med_duration,
				max_duration,min_rate,balance] for f_behavior,f_ephys in zip(file_lists.e_behavior,
					file_lists.ephys_files)]
	##assign data collection to multiple processes
	pool = mp.Pool(processes=8)
	async_result = pool.map_async(dpca.get_dataset_mp,arglist)
	pool.close()
	pool.join()
	results = async_result.get()
	for i in range(len(results)):
		if results[i][0].size > 1:
			X_all.append(results[i][0])
			session_names.append(results[i][1])
	return X_all, session_names

"""
A function to return dpca-transformed datasets from all sessions
Inputs:
	smooth_method: type of smoothing to use; choose 'bins', 'gauss', 'both', or 'none'
	smooth_width: size of the bins or gaussian kernel in ms. If 'both', input should be a list
		with index 0 being the gaussian width and index 1 being the bin width
	pad: a window for pre- and post-trial padding, in ms. In other words, an x-ms period of time 
		before lever press to consider the start of the trial, and an x-ms period of time after
		reward to consider the end of the trial
	z_score: if True, z-scores the array
	balance_trials: if True, equates the number of trials across all conditions by removal
		but only for each session independently
	n_components: number of components to fit
Returns:
	X_d: list of dpca-transformed datasets (each dataset is a dictionary with marginalizations)
"""
def run_dpca_all(conditions,smooth_method='both',smooth_width=[80,40],pad=[400,400],
	z_score=True,max_duration=5000,min_rate=0.1,balance=True,n_components=12):
	##first get the datasets
	X_all = get_dpca_datasets_all(conditions,smooth_method=smooth_method,smooth_width=smooth_width,
		pad=pad,z_score=z_score,max_duration=max_duration,min_rate=min_rate,balance=balance)
	##generate the argument list for the processor pool
	arglist = [[x,n_components,conditions] for x in X_all]
	pool = mp.Pool(processes=8)
	async_result = pool.map_async(dpca.run_dpca_mp,arglist)
	pool.close()
	pool.join()
	results = async_result.get()
	Z_all = []
	var_all = []
	sig_all = []
	for i in range(len(results)):
		Z_all.append(results[i][0])
		var_all.append(results[i][1])
		sig_all.append(results[i][2])
	return Z_all,var_all,sig_all

"""
A function to concatenate the spike data from all sessions for a given animal.
This function is meant to operate on the assumtion that the number of spike
channels (and roughly the identity of neurons on those channels) stays constant.
An error will be returned if there are variations in spike channel number across days.
Inputs:
	animal_id: string identifier of animal to get data for
	smooth_method: type of smoothing to use; choose 'bins', 'gauss', 'both', or 'none'
	smooth_width: size of the bins or gaussian kernel in ms. If both, input should be a list
		with index 0 being the gaussian width and index 1 being the bin width
Returns:
	X: spike data matrix of size units x bins. Data is z-scored WITHIN each DAY
		in an attempt to make things at least a little bit consistent
"""
def concatenate_spikes(animal_id,smooth_method='both',smooth_width=[80,40]):
	##get the file names from the multiunit hash
	ephys_files = flu.split_ephys_by_animal()[animal_id]
	data = [] ##container for individual data pieces
	for f_ephys in ephys_files:
		data.append(pe.get_spike_data(f_ephys,smooth_method=smooth_method,
			smooth_width=smooth_width,z_score=True))
	X = np.concatenate(data)

"""
This function is designed to concatenate behavioral data across sessions,
for each animal individually. The initial purpose is to create a dataset
to run model fitting on, but it could probably be used for something else.
Inputs:
	animal_id: ID of animal to use 
Returns:
	actions: array of actions, where 1 = lower lever, and 2 = upper lever
	outcomes: array of outcomes (1=rewarded, 0=unrewarded)
	switch_times: array of trial values at which point the rewarded lever switched
	first_block: rule identitiy of the first block
"""
def concatenate_behavior(animal_id):
	actions = []
	outcomes = []
	switch_times = []
	first_block = None
	session_list = file_lists.split_behavior_by_animal()[animal_id][6:] ##first 6 days have only one lever
	block_types = ['upper_rewarded','lower_rewarded']
	n_trials = 0
	##populate the master lists with the first file
	a,o,st,first_block = mf.get_session_data(session_list[0])
	actions.append(a)
	outcomes.append(o)
	switch_times.append(st)
	n_trials += a.size ##to keep track of how many trials have been added
	##record the identity of the last block
	def get_last_block(first_block,switch_times):
		block_types = ['upper_rewarded','lower_rewarded']
		if len(switch_times)%2 > 0: ##if we have an odd number of blocks, then
		##the last block in the session is NOT the same as the starting block
			last_block = [x for x in block_types if x != first_block][0]
		elif len(switch_times)%2 == 0:
			last_block = first_block
		return last_block
	last_block = get_last_block(first_block,st)
	##now run through the remaining sessions
	for i in range(1,len(session_list)):
		f_behavior = session_list[i]
		a,o,st,fb = mf.get_session_data(f_behavior)
		##append new data
		actions.append(a)
		outcomes.append(o)
		##need to compute the last block for this session before we mess with 
		##the block switches
		this_last = get_last_block(fb,st)
		##figure out if the blocks switched from last session end to new session start
		if last_block != fb:
			st = np.concatenate((np.array([0]),st))
		##make sure we offset the trial count
		switch_times.append(st + n_trials)
		last_block = this_last
		n_trials+=a.size
	return np.concatenate(actions),np.concatenate(outcomes),np.concatenate(switch_times),first_block

"""
This function is designed to concatenate behavioral data across sessions,
for each animal individually. The initial purpose is to create a dataset
to run model fitting on, but it could probably be used for something else.
Inputs:
	animal_id: ID of animal to use
	smooth_method: type of smoothing to use; choose 'bins', 'gauss', 'both', or 'none'
	smooth_width: size of the bins or gaussian kernel in ms. If both, input should be a list
		with index 0 being the gaussian width and index 1 being the bin width
Returns:
	actions: array of actions, where 1 = lower lever, and 2 = upper lever
	outcomes: array of outcomes (1=rewarded, 0=unrewarded)
	switch_times: array of trial values at which point the rewarded lever switched
	first_block: rule identitiy of the first block
	X: spike data concatenated across all sessions
"""
def concatenate_behavior_and_ephys(animal_id):
	actions = []
	outcomes = []
	switch_times = []
	first_block = None
	behavior_file = flu.split_behavior_by_animal(match_ephys=True)[animal_id] ##first 6 days have only one lever
	ephys_files = flu.split_ephys_by_animal()[animal_id]
	##make sure everything matches
	b_days = [x[-8:-5] for x in behavior_files]
	e_days = [x[-10:-7] for x in ephys_files]
	assert b_days == e_days
	spike_data = []
	block_types = ['upper_rewarded','lower_rewarded']
	n_trials = 0
	##populate the master lists with the first file
	a,o,st,first_block = mf.get_session_data(session_list[0])
	actions.append(a)
	outcomes.append(o)
	switch_times.append(st)
	n_trials += a.size ##to keep track of how many trials have been added
	##record the identity of the last block
	def get_last_block(first_block,switch_times):
		block_types = ['upper_rewarded','lower_rewarded']
		if len(switch_times)%2 > 0: ##if we have an odd number of blocks, then
		##the last block in the session is NOT the same as the starting block
			last_block = [x for x in block_types if x != first_block][0]
		elif len(switch_times)%2 == 0:
			last_block = first_block
		return last_block
	last_block = get_last_block(first_block,st)
	##now run through the remaining sessions
	for i in range(1,len(session_list)):
		f_behavior = session_list[i]
		a,o,st,fb = mf.get_session_data(f_behavior)
		##append new data
		actions.append(a)
		outcomes.append(o)
		##need to compute the last block for this session before we mess with 
		##the block switches
		this_last = get_last_block(fb,st)
		##figure out if the blocks switched from last session end to new session start
		if last_block != fb:
			st = np.concatenate((np.array([0]),st))
		##make sure we offset the trial count
		switch_times.append(st + n_trials)
		last_block = this_last
		n_trials+=a.size
	return np.concatenate(actions),np.concatenate(outcomes),np.concatenate(switch_times),first_block

"""
A function to get a dataset that includes data from all animals and sessions, for all trials AND switch trials.
specifically formatted to use in a dPCA analysis, but only including trials right after a context swith.
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
	n_after: numbe of trials after a switch to take data from
Returns:
	X_c: data from individual trials;
		 shape n-trials x n-neurons x condition-1 x condition-2, ... x n-timebins
"""
def get_switch_data(conditions,smooth_method='both',smooth_width=[80,40],pad=[400,400],
	z_score=True,max_duration=5000,min_rate=0.1,balance=True,min_trials_c=15,min_trials_s=5,
	n_after=10):
	##a container to store all of the X_trials data
	X_all_c = []
	X_all_s = []
	##the first step is to determine the median trial length for all sessions
	med_duration = np.median(get_trial_durations(max_duration=max_duration,session_range=None)).astype(int)
	##generate input arguments for multiprocessing
	arglist = [[f_behavior,f_ephys,conditions,smooth_method,smooth_width,pad,z_score,med_duration,
				max_duration,min_rate,balance,n_after] for f_behavior,f_ephys in zip(file_lists.e_behavior,
					file_lists.ephys_files)]
	##assign data collection to multiple processes
	pool = mp.Pool(processes=8)
	async_result = pool.map_async(dpca.get_switch_and_data_mp,arglist)
	pool.close()
	pool.join()
	results = async_result.get()
	for i in range(len(results)):
		if results[i][1] != None:
			X_all_c.append(results[i][0])
			X_all_s.append(results[i][1])
	##now, get an idea of how many trials we have per dataset
	n_trials_c = [] ##keep track of how many trials are in each session
	include_c = [] ##keep track of which sessions have more then the min number of trials
	for i in range(len(X_all_c)):
		n = X_all_c[i].shape[0]
		if n >= min_trials_c:
			include_c.append(i)
			n_trials_c.append(n)
	##from this set, what is the minimum number of trials?
	print("Including {0!s} sessions out of {1!s}".format(len(include_c),len(X_all_c)))
	min_trials_c = min(n_trials_c)
	##now all we have to do is concatenate everything together!
	X_c = []
	for s in include_c:
		X_c.append(X_all_c[s][0:min_trials_c,:,:,:,:])
	##now, get an idea of how many trials we have per dataset
	n_trials_s = [] ##keep track of how many trials are in each session
	include_s = [] ##keep track of which sessions have more then the min number of trials
	for i in range(len(X_all_s)):
		n = X_all_s[i].shape[0]
		if n >= min_trials_s:
			include_s.append(i)
			n_trials_s.append(n)
	##from this set, what is the minimum number of trials?
	print("Including {0!s} sessions out of {1!s}".format(len(include_s),len(X_all_s)))
	min_trials_s = min(n_trials_s)
	##now all we have to do is concatenate everything together!
	X_s = []
	for s in include_s:
		X_s.append(X_all_s[s][0:min_trials_s,:,:,:,:])	
	return np.concatenate(X_c,axis=1), np.concatenate(X_s,axis=1)
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
	'units_recorded':0,
	'mean_block_length':0,
	'mean_blocks_per_session':0,
	'mean_reward_rate':0
	}
	for f_behavior in file_lists.e_behavior:
		meta = sa.get_session_meta(f_behavior,max_duration)
		metadata['training_sessions']+=1
		metadata['rewarded_trials']+=meta['rewarded'].size
		metadata['unrewarded_trials']+=meta['unrewarded'].size
		metadata['upper_context_trials']+=meta['upper_context'].size
		metadata['lower_context_trials']+=meta['lower_context'].size
		metadata['lower_lever_trials']+=meta['lower_lever'].size
		metadata['upper_lever_trials']+=meta['upper_lever'].size
		metadata['mean_block_length']+=meta['mean_block_len']
		metadata['mean_blocks_per_session']+=meta['n_blocks']
		metadata['mean_reward_rate']+=meta['reward_rate']
	metadata['mean_block_length'] = metadata['mean_block_length']/len(file_lists.e_behavior)
	metadata['mean_blocks_per_session'] = metadata['mean_blocks_per_session']/len(file_lists.e_behavior)
	metadata['mean_reward_rate'] = metadata['mean_reward_rate']/len(file_lists.e_behavior)
	for f_ephys in file_lists.ephys_files:
		metadata['sessions_with_ephys']+=1
		metadata['units_recorded']+=sa.get_n_units(f_ephys)
	return metadata

"""
A function to get a "template" trial_data dataset
that can be used to standardize all sessions for tensor analysis
"""
def build_template_session(max_duration=5000,n_back=3,epoch='early'):
	contexts = ['lower_rewarded','upper_rewarded']
	##get some meta about how the sessions are structured, on average
	metadata = get_metadata(max_duration=max_duration)
	##build a model that can predict behavior
	y,X = behavior_regression(max_duration=max_duration,n_back=n_back)
	model = lr2.get_model(X,y)
	##create the template dataframe
	n_blocks = np.round(metadata['mean_blocks_per_session']).astype(int)
	trials_per_block = np.round(metadata['mean_block_length']).astype(int)
	reward_rate = metadata['mean_reward_rate']
	columns = ['context','action','outcome']
	trial_data = pd.DataFrame(index=np.arange(n_blocks*trials_per_block),columns=columns)
	##go trial-by-trial and create the data
	##create a pandas dataframe because it's easier for me to keep track
	features = ['training_day','trial_number']
	for i in range(n_back):
		features.append('action-'+str(i+1))
		features.append('outcome-'+str(i+1))
		features.append('interaction-'+str(i+1))
	for n,ctx in enumerate(contexts):
		for t in range(trials_per_block):
			##generate the trial info for this trial to use in the model
			x = pd.DataFrame(index=[0],columns=features)
			trial_num = (n*trials_per_block)+t
			x['trial_number'][0] = trial_num
			if epoch=='early':
				session_num = np.random.randint(15,20)
			if epoch == 'late':
				session_num = np.random.randint(40,48)
			x['training_day'][0] = session_num
			for i in range(n_back):	
				if trial_num > n_back:
					x['action-'+str(i+1)][0] = lr3.trial_lut[trial_data['action'][trial_num-(i+1)]]
					x['outcome-'+str(i+1)][0] = lr3.trial_lut[trial_data['outcome'][trial_num-(i+1)]]
					x['interaction-'+str(i+1)][0] = lr3.trial_lut[trial_data['action'][trial_num-(
						i+1)]]*lr3.trial_lut[trial_data['outcome'][trial_num-(i+1)]]
				else:
					x['action-'+str(i+1)][0] = 0
					x['outcome-'+str(i+1)][0] = 0
					x['interaction-'+str(i+1)][0] = 0
			##we already know the context for this trial
			trial_data['context'][trial_num] = ctx
			##now we can predict the action for this trial
			action = model.predict(x)
			if action[0] == 2:
				trial_data['action'][trial_num] = 'upper_lever' #***CAREFUL IF THIS KEY/VALUE PAIR CHANGES
			elif action[0] == 1:
				trial_data['action'][trial_num] = 'lower_lever'
			else:
				print("Warning: unknown action value: {}".format(action[0]))
				break
			##next, determine what the outcome of this trial will be, given the current context and the
			##determined reward rate
			if trial_data['action'][trial_num] == 'upper_lever' and ctx == 'lower_rewarded':
				outcome = 'unrewarded_poke'
			elif trial_data['action'][trial_num] == 'lower_lever' and ctx == 'upper_rewarded':
				outcome = 'unrewarded_poke'
			elif trial_data['action'][trial_num] == 'upper_lever' and ctx == 'upper_rewarded':
				if np.random.random() < reward_rate:
					outcome = 'rewarded_poke'
				else:
					outcome = 'unrewarded_poke'
			elif trial_data['action'][trial_num] == 'lower_lever' and ctx == 'lower_rewarded':
				if np.random.random() < reward_rate:
					outcome = 'rewarded_poke'
				else:
					outcome = 'unrewarded_poke'
			else:
				print("Unkown action type: {}".format(trial_data['action'][trial_num]))
				break
			trial_data['outcome'][trial_num] = outcome
	return trial_data



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

