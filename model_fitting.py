##model_fitting.py
##functions to do model fitting

import numpy as np
import session_analysis as sa
import SMC as smc
import HMM_model as hmm
import RL_model as rl
import file_lists
import parse_trials as ptr
from sklearn.cluster import KMeans

"""
A function to fit RL and HMM models to behavior data from
one session, and compute the goodness-of-fit using log liklihood
Inputs:
	f_behavior: behavior data file
Returns:
	Results: ictionary with the following fields
		actions: actions performed by the subject.
			1= lower_lever, 2 = upper_lever
		RL_actions: actions performed by the RL model
		HMM_actions: actions performed by the HMM model
		e_RL: resulting particles from RL model
		e_HMM: resulting particles form HMM model
		ll_RL: log-liklihood from RL model
		ll_HMM: log-liklihood for HMM model
"""
def fit_models(f_behavior):
	##first parse the data from this session
	actions,outcomes,switch_times,first_block = get_session_data(f_behavior)
	##compute model fits. for RL model:
	initp = rl.initp(10000)
	sd_jitter = [0.01,0.01,0.001,0.001]
	e_RL,v_RL = smc.SMC(actions,outcomes,initp,sd_jitter,rl.rescorlawagner,rl.boltzmann)
	##now for HMM
	initp = hmm.initp(10000)
	sd_jitter = [0.01,0.01,0.001,0.001,0.001]
	e_HMM,v_HMM = smc.SMC(actions,outcomes,initp,sd_jitter,hmm.compute_belief,hmm.action_weights)
	##now compute the actions that would be taken by each model
	RL_actions,RL_Pa = rl.compute_actions(e_RL[0,:],e_RL[1,:],e_RL[3,:])
	HMM_actions,HMM_Pa = hmm.compute_actions(e_HMM[0,:])
	##finally, compute the log-liklihood for each model
	ll_RL = log_liklihood(actions,RL_Pa)
	ll_HMM = log_liklihood(actions,HMM_Pa)
	##compile all of the data into a results dictionary
	results = {
	'actions':actions,
	'outcomes':outcomes,
	'RL_actions':RL_actions,
	'HMM_actions':HMM_actions,
	'e_RL':e_RL,
	'e_HMM':e_HMM,
	'll_RL':ll_RL,
	'll_HMM':ll_HMM,
	'switch_times':switch_times,
	'first_block':first_block
	}
	return results

"""
Same as above function, but fits models to all sessions for one animal
concatenated together.
Inputs:
	animal_id: id of the animal to get sessions from
Returns
	Results: ictionary with the following fields
		actions: actions performed by the subject.
			1= lower_lever, 2 = upper_lever
		RL_actions: actions performed by the RL model
		HMM_actions: actions performed by the HMM model
		e_RL: resulting particles from RL model
		e_HMM: resulting particles form HMM model
		ll_RL: log-liklihood from RL model
		ll_HMM: log-liklihood for HMM model
"""
def fit_models_all(animal_id):
	actions, outcomes,switch_times,first_block = concatenate_behavior(animal_id)
	##compute model fits. for RL model:
	initp = rl.initp(10000)
	sd_jitter = [0.01,0.01,0.001,0.001]
	e_RL,v_RL = smc.SMC(actions,outcomes,initp,sd_jitter,rl.rescorlawagner,rl.boltzmann)
	##now for HMM
	initp = hmm.initp(10000)
	sd_jitter = [0.01,0.01,0.001,0.001,0.001]
	e_HMM,v_HMM = smc.SMC(actions,outcomes,initp,sd_jitter,hmm.compute_belief,hmm.action_weights)
	##now compute the actions that would be taken by each model
	RL_actions,RL_Pa = rl.compute_actions(e_RL[0,:],e_RL[1,:],e_RL[3,:])
	HMM_actions,HMM_Pa = hmm.compute_actions(e_HMM[0,:])
	##finally, compute the log-liklihood for each model
	ll_RL = log_liklihood(actions,RL_Pa)
	ll_HMM = log_liklihood(actions,HMM_Pa)
	##compile all of the data into a results dictionary
	results = {
	'actions':actions,
	'outcomes':outcomes,
	'RL_actions':RL_actions,
	'HMM_actions':HMM_actions,
	'e_RL':e_RL,
	'e_HMM':e_HMM,
	'll_RL':ll_RL,
	'll_HMM':ll_HMM,
	'switch_times':switch_times,
	'first_block':first_block
	}
	return results

"""
Same as above function, but computes actions,etc from a trial_data DataFrame
Inputs:
	animal_id: id of the animal to get sessions from
Returns
	Results: ictionary with the following fields
		actions: actions performed by the subject.
			1= lower_lever, 2 = upper_lever
		RL_actions: actions performed by the RL model
		HMM_actions: actions performed by the HMM model
		e_RL: resulting particles from RL model
		e_HMM: resulting particles form HMM model
		ll_RL: log-liklihood from RL model
		ll_HMM: log-liklihood for HMM model
"""
def fit_models_from_trial_data(trial_data):
	actions,outcomes,switch_times,first_block = get_session_data_from_trial_data(trial_data)
	##compute model fits. for RL model:
	initp = rl.initp(10000)
	sd_jitter = [0.01,0.01,0.001,0.001]
	e_RL,v_RL = smc.SMC(actions,outcomes,initp,sd_jitter,rl.rescorlawagner,rl.boltzmann)
	##now for HMM
	initp = hmm.initp(10000)
	sd_jitter = [0.01,0.01,0.001,0.001,0.001]
	e_HMM,v_HMM = smc.SMC(actions,outcomes,initp,sd_jitter,hmm.compute_belief,hmm.action_weights)
	##compile all of the data into a fit_results dictionary
	fit_results = {
	'actions':actions,
	'outcomes':outcomes,
	'e_RL':e_RL,
	'e_HMM':e_HMM,
	}
	##now use the fitted parameter values to estimate the model actions
	RL_actions,RL_p_lower,Qvals = rl.run_model(fit_results)
	HMM_actions,HMM_p_lower,state_vals = hmm.run_model(fit_results)
	##finally, compute the log-liklihood for each model
	ll_RL = log_liklihood(actions,RL_p_lower)
	ll_HMM = log_liklihood(actions,HMM_p_lower)
	##compile the results into a ductionary
	results = {
	'actions':actions,
	'outcomes':outcomes,
	'RL_actions':RL_actions,
	'HMM_actions':HMM_actions,
	'RL_p_lower':RL_p_lower,
	'HMM_p_lower':HMM_p_lower,
	'Qvals':Qvals,
	'state_vals':state_vals,
	'll_RL':ll_RL,
	'll_HMM':ll_HMM,
	'switch_times':switch_times,
	'first_block':first_block
	}
	return results

"""
A function to compute the uncertainty, which is 1/(p(state1)-p(state2))
"""
def uncertainty_from_trial_data(trial_data):
	results = fit_models_from_trial_data(trial_data)
	certainty = abs(results['state_vals'][0]-results['state_vals'][1])
	uncertainty = np.log(1/certainty)
	##the first value is infinite, so replace/approximte it with the second
	uncertainty[0] = uncertainty[1]
	return uncertainty

"""
A function to append the uncertainty estimation to an existing trial_data
dataset (computes the uncertainty from said trial data set). Includes both 
raw uncertainty value, as well as a grouping index (high, medium, low) based
on K-means clustering.
"""
def append_uncertainty(trial_data,levels=['low','med','high']):
	##first, compute the uncertainty
	uncertainty = uncertainty_from_trial_data(trial_data)
	##now cluster the uncertainty into three groups
	labels = KMeans(n_clusters=len(levels)).fit_predict(uncertainty.reshape(uncertainty.shape[0],1))
	##now we need to figure out which groupings correspond to high-med-low
	group_indices = []
	group_means = []
	for group_label in np.unique(labels):
		idx = np.where(labels==group_label)[0]
		group_indices.append(idx)
		vals = uncertainty[idx]
		group_means.append(vals.mean())
	sorted_low_to_high = np.argsort(group_means)
	##now we can do some work to add the data to the dataset.
	trial_data['uncertainty'] = uncertainty
	blank = np.empty(trial_data.shape[0])
	trial_data['u_level'] = blank
	for i,n in enumerate(sorted_low_to_high):
		trial_data.loc[group_indices[n],'u_level'] = levels[i]
	return trial_data



	##next we want to know the ordering of the uncertainty levels in each of these groups




"""
A function to get the action or outcome
data from one session.
Inputs:
	-f_behavior: data file
	-model_type: if RL, actions are reported;
		if HMM, switches are reported
Returns:
	-actions: int array sequence of actions
	-outcomes: int array sequence of outcomes
	-switch_times: occurances of a block switch
	-first_rewarded: the first block type
"""
def get_session_data(f_behavior):
	meta = sa.get_session_meta(f_behavior)
	n_trials = (meta['unrewarded']).size+(meta['rewarded']).size
	actions = np.zeros(n_trials)
	outcomes = np.zeros(n_trials)
	switch_times = []
	start = 0
	for l in meta['block_lengths']:
		switch_times.append(start+l)
		start+=l
	##last index is just the end of the trial, so we can ignore this
	switch_times = np.asarray(switch_times)[:-1]
	first_block = meta['first_block']
	actions[meta['lower_lever']] = 1
	actions[meta['upper_lever']] = 2
	outcomes[meta['rewarded']] = 1
	return actions,outcomes,switch_times,first_block

"""
A different way of getting session data using a trial_data
DataFrame. 
Inputs:
	trial_data: a DataFrame with trial data; works even if
		it is a concatenation of many sessions
Returns:
	-actions: int array sequence of actions
	-outcomes: int array sequence of outcomes
	-switch_times: occurances of a block switch
	-first_rewarded: the first block type
"""
def get_session_data(f_behavior,max_duration=5000):
	trial_data = ptr.get_full_trials(f_behavior,pad=[400,400],max_duration=max_duration)
	n_trials = trial_data.index.size
	actions = np.zeros(n_trials)
	outcomes = np.zeros(n_trials)
	first_block = trial_data['context'][0]
	upper_levers = np.where(trial_data['action']=='upper_lever')[0]
	lower_levers = np.where(trial_data['action']=='lower_lever')[0]
	rewarded = np.where(trial_data['outcome']=='rewarded_poke')[0]
	unrewarded = np.where(trial_data['outcome']=='unrewarded_poke')[0]
	actions[upper_levers]=2
	actions[lower_levers]=1
	outcomes[rewarded]=1
	outcomes[unrewarded]=0
	ctx = np.asarray(trial_data['context']=='upper_rewarded').astype(int)
	switch_times = np.where(np.diff(ctx)!=0)[0]
	return actions,outcomes,switch_times,first_block

"""
A different way of getting session data using a trial_data
DataFrame. 
Inputs:
	trial_data: a DataFrame with trial data; works even if
		it is a concatenation of many sessions
Returns:
	-actions: int array sequence of actions
	-outcomes: int array sequence of outcomes
	-switch_times: occurances of a block switch
	-first_rewarded: the first block type
"""
def get_session_data_from_trial_data(trial_data):
	n_trials = trial_data.index.size
	actions = np.zeros(n_trials)
	outcomes = np.zeros(n_trials)
	first_block = trial_data['context'][0]
	upper_levers = np.where(trial_data['action']=='upper_lever')[0]
	lower_levers = np.where(trial_data['action']=='lower_lever')[0]
	rewarded = np.where(trial_data['outcome']=='rewarded_poke')[0]
	unrewarded = np.where(trial_data['outcome']=='unrewarded_poke')[0]
	actions[upper_levers]=2
	actions[lower_levers]=1
	outcomes[rewarded]=1
	outcomes[unrewarded]=0
	ctx = np.asarray(trial_data['context']=='upper_rewarded').astype(int)
	switch_times = np.where(np.diff(ctx)!=0)[0]
	return actions,outcomes,switch_times,first_block


"""
A function to compute the log liklihood given:
Inputs:
	-subject_actions: the actual actions performed by the subject
	-Pa: the probability of action a on each trial computed by the model
	-Pb: the probability of action b on each trial computed by the model
Returns:
	log_liklihood of model fit
"""
def log_liklihood(subject_actions,Pa):
	Pb = 1-Pa
	s_a = (subject_actions == 1).astype(int)
	s_b = (subject_actions == 2).astype(int)
	##the equation to compute log L
	logL = ((s_a*np.log(Pa)).sum()/s_a.sum())+(
		(s_b*np.log(Pb)).sum()/s_b.sum())
	return logL

"""
A function to compute prediction accuracy; in other words,
when what percentage of the subject's behavior was correctly
predicted by the model?
Inputs:
	subject_actions: array of actions taken by the subject
	model_actions: array of ations taken by the model
Returns: 
	accuracy: percentage correct prediction by the model
"""
def accuracy(subject_actions,model_actions):
	return (subject_actions==model_actions).sum()/subject_actions.size

"""
A function to compute accuracy over time in a sliding window
Inputs:
	subject_actions: array of actions taken by the subject
	model_actions: array of ations taken by the model
	win: sliding window as [window,win_step]
Returns: 
	accuracy: percentage correct prediction by the model over time
"""
def sliding_accuracy(subject_actions,model_actions,win=[50,10]):
	N = subject_actions.size
	Nwin = int(win[0])
	Nstep = int(win[1])
	winstart = np.arange(0,N-Nwin,Nstep)
	nw = winstart.shape[0]
	acc = np.zeros(nw)
	for n in range(nw):
		idx = np.arange(winstart[n],winstart[n]+Nwin)
		data_s = subject_actions[idx]
		data_m = model_actions[idx]
		acc[n] = accuracy(data_s,data_m)
	return acc

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
	a,o,st,first_block = get_session_data(session_list[0])
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
		a,o,st,fb = get_session_data(f_behavior)
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