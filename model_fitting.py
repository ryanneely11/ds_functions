##model_fitting.py
##functions to do model fitting

import numpy as np
from RL_model import RL_agent
from HMM_model import HMM_agent
from task import bandit
import session_analysis as sa
import parse_trials as ptr
from scipy.optimize import minimize,brute,fmin_slsqp
import file_lists
import os
import pandas as pd
import itertools

"""
A function to perform optimization on RL_models.
Inputs:
	-session_range: a range of session numbers to use for model fitting
Returns:
	-resutls: dictionary of results from scipy.optimize.minimize
"""
def fit_RL_model(session_range):
	##define some reasonable bounds for parameter values
	alpha_bounds = (0,10)
	beta_bounds = (0,50)
	eta_bounds = (0,1)
	bounds = [alpha_bounds,beta_bounds,eta_bounds]
	##get behavior data to fit the model to
	actions,p_rewarded,switch_after,b_a,b_b = get_fit_info(session_range)
	##initial guess
	x0 = [0.2,0.8,0.1]
	##set up the minimizer
	results = brute(score_RL_model,bounds,args=(actions,p_rewarded,
		switch_after,b_a,b_b),Ns=20,disp=True)
	return results

"""
A function to perform optimization on HMM models.
Inputs:
	-session_range: a range of session numbers to use for model fitting
Returns:
	-resutls: dictionary of results from scipy.optimize.minimize
"""
def fit_HMM_model(session_range):
	##define some reasonable bounds for parameter values
	alpha_bounds = (0,10)
	beta_bounds = (0,100)
	delta_bounds = (0,1)
	correct_mean_bounds = (0,1)
	incorrect_mean_bounds = (0,1)
	bounds = [alpha_bounds,beta_bounds,delta_bounds,
	correct_mean_bounds,incorrect_mean_bounds]
	##get behavior data to fit the model to
	actions,p_rewarded,switch_after,b_a,b_b = get_fit_info(session_range)
	##initial guess
	x0 = [0.5,20,0.1,0.5,0.5]
	##set up the minimizer
	results = brute(score_HMM_model,bounds,args=(actions,p_rewarded,
		switch_after,b_a,b_b),Ns=20,disp=True)
	return results


	
"""
A function to do brute-force optimization.
"""
def brute_optimize(func,ranges,gridspace=100):
	n_params = len(ranges)
	##create the gridpoints by equal spacing in the given ranges
	gridpoints = []
	for i in range(n_params):
		r = ranges[0]
		gridpoints.append(np.linspace(r[0],r[1],gridspace))
	##create a generator to return all combinations of possible param values
	for combination in itertools.product(*gridpoints):
		pass
	


"""
a function to return the negative log-liklihood of
an RL model given a list of params, X, and a behavior
session file to compare to.
Inputs:
	x: the parameter list: [alpha,beta,eta]
	actions: list of possible actions
	p_rewarded = probability of a reward given a correct response
	switch_after: block switches
	b_a,b_b: actual behavior responses for action a and b (bool)
Returns:
	mean negative log-liklihood of model score
"""
def score_RL_model(x,actions,p_rewarded,switch_after,b_a,b_b):
	##see if the input parameters are in bounds.
	alpha_bounds = (0,10)
	beta_bounds = (0,100)
	delta_bounds = (0,1)
	correct_mean_bounds = (0,1)
	incorrect_mean_bounds = (0,1)
	bounds = [alpha_bounds,beta_bounds,delta_bounds,
	correct_mean_bounds,incorrect_mean_bounds]
	if check_bounds(x,bounds):
		n_trials = 20
		results = []
		alpha = x[0]
		beta = x[1]
		eta = x[2]
		for i in range(n_trials):
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
			results.append(-logL)
		results = np.nanmean(results)
	##if the params are not in bounds, return a large number
	else:
		results = 10000.0
	return results

"""
a function to return the negative log-liklihood of
an HHM model given a list of params, X, and a behavior
session file to compare to.
Inputs:
	x: the parameter list: [alpha,beta,delta,correct_mean,incorrect_mean]
	actions: list of possible actions
	p_rewarded = probability of a reward given a correct response
	switch_after: block switches
	b_a,b_b: actual behavior responses for action a and b (bool)
Returns:
	mean negative log-liklihood of model scores
"""
def score_HMM_model(x,actions,p_rewarded,switch_after,b_a,b_b):
	n_trials = 20
	results = []
	alpha = x[0]
	beta = x[1]
	delta = x[2]
	correct_mean = x[3]
	incorrect_mean = x[4]
	for i in range(n_trials):
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
		results.append(-logL)
	return np.nanmean(results)

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
A function to construct bandit parameters using data from
many sessions, as well as actual action data.
Inputs:
	-session_range: range of sessions to use data from
Returns:
	-parameters needed to initialize a task.bandit
	b_a,b_b: behavioral choice at each trial (action a or action b, bool)
"""
def get_fit_info(session_range):
	##set up some lists to concatenate everything
	action_seq = []
	context = []
	outcomes = []
	for f_behavior in file_lists.behavior_files:
		##check to see if this file is in the specified range
		if ptr.get_session_number(os.path.normpath(f_behavior)) in range(
			session_range[0],session_range[1]):
			##parse the data
			trial_data = ptr.get_full_trials(f_behavior)
			##add the data to the master lists
			action_seq.append(np.asarray(trial_data['action']))
			context.append(np.asarray(trial_data['context']))
			outcomes.append(np.asarray(trial_data['outcome']))
	action_seq = np.concatenate(action_seq)
	context = np.concatenate(context)
	outcomes = np.concatenate(outcomes)
	##was a given action correct given the context
	def is_correct(context,action):
		if context == 'upper_rewarded':
			if action == 'upper_lever':
				result = True
			elif action == 'lower_lever':
				result = False
		elif context == 'lower_rewarded':
			if action == 'upper_lever':
				result = False
			elif action == 'lower_lever':
				result = True
		else:
			print('Unknown context')
			result = False
		return result
	##determine the emperical correct rewarded percentage
	correct_rewarded = 0
	correct_total = 0
	for i in range(outcomes.size):
		if is_correct(context[i],action_seq[i]):
			correct_total +=1
			if outcomes[i] == 'rewarded_poke':
				correct_rewarded += 1
	p_correct = float(correct_rewarded)/correct_total
	##determine the order of actions
	if context[0] == 'lower_rewarded':
		actions = ['lower_lever','upper_lever']
	elif context[0] == 'upper_rewarded':
		actions = ['upper_lever','lower_lever']
	b_a = (action_seq==actions[0]).astype(int)
	b_b = (action_seq==actions[1]).astype(int)
	##determine when the blocks switch
	switch_after = []
	last_context = context[0]
	for i in range(context.size):
		current_context = context[i]
		if current_context != last_context:
			switch_after.append(i)
		last_context = current_context
	return actions, p_correct, switch_after, b_a, b_b



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

"""
A function to check if a set of parameters is in bounds.
Inputs:
	x: parameter set
	bounds: list of tuples with [min,max] bounds
Returns:
	is_valid: if True, parameters are in bounds
"""
def check_bounds(x,bounds):
	is_valid = True
	for param,bound in zip(x,bounds):
		if param <= bound[0] or param >= bound[1]:
			is_valid = False
	return is_valid


"""
A function to parse model results in a similar
way to how we would parse behavior data results.
Inputs:
	model: the model object
	bandit: the bandit object
Returns: 
	event_data: event data from the model's behavior
"""
def parse_model_results(model,bandit):
	##get some metadata about these things
	n_trials = len(model.log['action'])
	n_blocks = len(bandit.switch_after)
	block_switch = bandit.switch_after
	actions = bandit.actions
	if actions[0] == 'upper_lever':
		contexts = ['upper_rewarded','lower_rewarded']
	elif actions[0] == 'lower_lever':
		contexts = ['lower_rewarded','upper_rewarded']
	columns = ['context','action','outcome']
	##construct a pandas dataset for this data
	trial_data = pd.DataFrame(index=range(n_trials),columns=columns)
	##fill out the context info
	trial_data['context'][0:block_switch[0]] = contexts[0]
	for i in range(n_blocks-1):
		trial_data['context'][block_switch[i]:block_switch[i+1]] = contexts[(i+1)%2]
	prev_block = trial_data['context'][block_switch[-1]-1]
	last_block = [x for x in contexts if not x == prev_block]
	trial_data['context'][block_switch[-1]:] = last_block[0]
	##the action data is easy...
	trial_data['action'][:] =np.asarray(model.log['action'])
	##outcome isn't that hard either
	rewarded = np.where(np.asarray(model.log['outcome'])==1)[0]
	unrewarded = np.where(np.asarray(model.log['outcome'])==0)[0]
	trial_data['outcome'][rewarded] = 'rewarded_poke'
	trial_data['outcome'][unrewarded] = 'unrewarded_poke'
	correct_upper = []
	correct_lower = []
	incorrect_upper = []
	incorrect_lower = []
	correct_unrew_upper = []
	correct_unrew_lower =[]
	for i in range(n_trials):
		if trial_data['action'][i] == 'upper_lever': ##case where it was an upper press
			if trial_data['context'][i] == 'upper_rewarded': ##case where it was correct
				if trial_data['outcome'][i] == 'rewarded_poke':
					correct_upper.append(i)
				elif trial_data['outcome'][i] == 'unrewarded_poke':
					correct_unrew_upper.append(i)
			elif trial_data['context'][i] == 'lower_rewarded':
				incorrect_upper.append(i)
		elif trial_data['action'][i] == 'lower_lever':
			if trial_data['context'][i] == 'lower_rewarded':
				if trial_data['outcome'][i] == 'rewarded_poke':
					correct_lower.append(i)
				elif trial_data['outcome'][i] == 'unrewarded_poke':
					correct_unrew_lower.append(i)
			elif trial_data['context'][i] == 'upper_rewarded':
				incorrect_lower.append(i)
		else:
			print("Action does not fit any catagories: {}".format(trial_data['action'][i]))
		event_data = {
		'correct_upper':np.asarray(correct_upper),
		'correct_lower':np.asarray(correct_lower),
		'correct_unrew_upper':np.asarray(correct_unrew_upper),
		'correct_unrew_lower':np.asarray(correct_unrew_lower),
		'incorrect_upper':np.asarray(incorrect_upper),
		'incorrect_lower':np.asarray(incorrect_lower),
		'block_switches':np.asarray(block_switch)
		}
	return trial_data,event_data


