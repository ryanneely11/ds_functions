##model_fitting.py
##functions to do model fitting

import numpy as np
import session_analysis as sa
import SMC as smc
import HMM_model as hmm
import RL_model as rl

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
	initp = rl.initp(1000)
	sd_jitter = [0.01,0.01,0.001,0.001]
	e_RL,v_RL = smc.SMC(actions,outcomes,initp,sd_jitter,rl.rescorlawagner,rl.boltzmann)
	##now for HMM
	initp = hmm.initp(1000)
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

