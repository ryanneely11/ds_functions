##model_fitting.py
##functions to do model fitting

import numpy as np
from RL_model import RL_agent
from HMM_model import HMM_agent
from task import bandit
import session_analysis as sa
import parse_trials as ptr
import multiprocessing as mp

"""
a function to return the negative log-liklihood of
an RL model given a list of params, X, and a behavior
session file to compare to.
Inputs:
	x: the parameter list: [alpha,beta,eta]
Returns:
	negative log-liklihood of model
"""
def score_RL_model(x,f_behavior):
	actions,p_rewarded,switch_after = get_bandit_info(f_behavior)
	b_a,b_b = get_action_sequence(actions,f_behavior)
	n_trials = 20
	arglist =[x,actions,p_rewarded,switch_after,b_a,b_b]
	results = []
	for i in range(n_trials):
		results.append(mp_RL_fit(arglist))
	# arglist = [
	# [x,f_behavior,actions,p_rewarded,switch_after,b_a,b_b] for i in range(n_trials)]
	# pool = mp.Pool(processes=n_trials)
	# async_result = pool.map_async(mp_RL_fit,arglist)
	# pool.close()
	# pool.join()
	# results = async_result.get()
	return np.mean(results)

"""
A function to do multiprocessed model fitting
"""
def mp_RL_fit(args):
	##parse the input tuple
	x = args[0]
	actions=args[1]
	p_rewarded = args[2]
	switch_after=args[3]
	b_a = args[4]
	b_b = args[5]
	alpha = x[0]
	beta = x[1]
	eta = x[2]
	##init the task using the given session
	task = bandit(actions,p_rewarded,switch_after)
	##now, we can produce the outputs of the model given our params:
	model = RL_agent(task,actions,alpha,beta,eta)
	for i in range(len(b_a)-1):
		model.run()
	p_a = model.log['p_a']
	p_b = model.log['p_b']
	##now let's compute the log-liklihood:
	logL = log_liklihood(b_a,b_b,p_a,p_b)
	##return the negative for our optimizer
	return -logL

"""
a function to return the negative log-liklihood of
an HHM model given a list of params, X, and a behavior
session file to compare to.
Inputs:
	x: the parameter list: [alpha,beta,delta,correct_mean,incorrect_mean]
Returns:
	negative log-liklihood of model
"""
def score_HMM_model(x,f_behavior):
	actions,p_rewarded,switch_after = get_bandit_info(f_behavior)
	b_a,b_b = get_action_sequence(actions,f_behavior)
	n_trials = 20
	arglist = [x,actions,p_rewarded,switch_after,b_a,b_b]
	results = []
	for i in range(n_trials):
		results.append(mp_HMM_fit(arglist))
	# arglist = [
	# [x,actions,p_rewarded,switch_after,b_a,b_b] for i in range(n_trials)]
	# pool = mp.Pool(processes=n_trials)
	# async_result = pool.map_async(mp_HMM_fit,arglist)
	# pool.close()
	# pool.join()
	# results = async_result.get()
	return np.mean(results)

"""
A function to do multiprocessed model fitting
"""
def mp_HMM_fit(args):
	##parse the input tuple
	x = args[0]
	actions=args[1]
	p_rewarded = args[2]
	switch_after=args[3]
	b_a = args[4]
	b_b = args[5]
	alpha = x[0]
	beta = x[1]
	delta = x[2]
	correct_mean = x[3]
	incorrect_mean = x[4]
	##init the task using the given session
	task = bandit(actions,p_rewarded,switch_after)
	##now, we can produce the outputs of the model given our params:
	model = HMM_agent(task,actions,alpha,beta,delta,correct_mean,incorrect_mean)
	for i in range(len(b_a)-1):
		model.run()
	p_a = model.log['p_a']
	p_b = model.log['p_b']

	##now let's compute the log-liklihood:
	logL = log_liklihood(b_a,b_b,p_a,p_b)
	##return the negative for our optimizer
	return -logL

"""
A function to get a sequence of switches
(binary; switch or no switch) given an array
of real bahavior data trial actions.
Inputs:
	f_Behavior: path to behavior data file
Returns
	switch_sequence: the sequence specifying if the
		subject switched (1) or did not switch (0)
		on at that trial
"""
def get_switch_sequence(f_behavior):
	##parse the trials
	actions = ptr.get_full_trials(f_behavior)['action']
	switch_sequence = []
	last_action = actions[0]
	for i in range(len(actions)):
		if actions[i] == last_action:
			switch_sequence.append(0)
		elif actions[i] != last_action:
			switch_sequence.append(1)
		last_action = actions[i]
	return switch_sequence

"""
A function to get the action sequence for comparison to the models.
Inputs:
	actions: the order actions are used in the model to compare to 
	f_behavior: the data file
Returns:
	b_a: 1's where action a occurred
	b_b: 1's where action b occurred
"""
def get_action_sequence(actions,f_behavior):
	action_seq = ptr.get_full_trials(f_behavior)['action']
	b_a = (action_seq==actions[0]).astype(int)
	b_b = (action_seq==actions[1]).astype(int)
	return b_a,b_b

"""
A function to get bandit info from a behavior file.
Inputs: 
	-f_behavior: dat ile to use
REturns:
	-all parameterns necessary to create a bandit to approximate this session
"""
def get_bandit_info(f_behavior):
	##get the metadata info for this session
	meta = sa.get_session_meta(f_behavior)
	block_lengths = meta['block_lengths']
	##bandit will start with the first aciton in the
	##actions argument list
	if meta['first_block'] == 'lower_rewarded':
		actions = ['lower_lever','upper_lever']
	elif meta['first_block'] == 'upper_rewarded':
		actions = ['upper_lever','lower_lever']
	##get the probability of a reward for the correct action
	p_rewarded = meta['reward_rate']
	##finally, initialize a bandit with these params
	return actions,p_rewarded,block_lengths[:-1]

"""
A function to compute the log liklihood given:
Inputs:
	-b_switch: a list of actual behavior, whether or
		not the animal switched (1) or stayed (0) on each trial
	-p_switch: the model's computed probability of switching or staying
		on each trial
Returns:
	log_liklihood of model fit
"""
def log_liklihood(b_a,b_b,p_a,p_b):
	b_a = np.asarray(b_a)
	p_a = np.asarray(p_a)
	b_b = np.asarray(b_b)
	p_b = np.asarray(p_b)
	##the equation to compute log L
	logL = ((b_a*np.log(p_a)).sum()/b_a.sum())+(
		(b_b*np.log(p_b)).sum()/b_b.sum())
	return logL



	