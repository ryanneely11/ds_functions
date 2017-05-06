##model_fitting.py
##functions to do model fitting

import numpy as np
from RL_model import RL_agent
from HMM_model import HMM_agent
from task import bandit
import session_analysis as sa
import parse_trials as ptr

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
	##init the task using the given session
	bandit = get_bandit(f_behavior)
	##get the subject's actual behavior 
	b_switch = get_switch_sequence(f_behavior)
	##x is the parameter list, which we can parse
	##for clarity:
	alpha = x[0]
	beta = x[1]
	eta = x[2]
	actions = ['upper_lever','lower_lever']
	##now, we can produce the outputs of the model given our params:
	model = RL_agent(bandit,actions,alpha,beta,eta)
	for i in range(len(b_switch)-1):
		model.run()
	p_switch = model.log['p_switch']
	##now let's compute the log-liklihood:
	logL = log_liklihood(b_switch,p_switch)
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
A function to simulate the task specifically from one
trial. Uses the log data to figure out when
the context switch[es] occurred, what the
eperical reward probability was, and initializes
a bandit.
Inputs:
	f_behavior: behavior data file
Returns:
	bandit: a task.bandit object
"""
def get_bandit(f_behavior):
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
	return bandit(actions,p_rewarded,block_lengths)

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
def log_liklihood(b_switch,p_switch):
	b_switch = np.asarray(b_switch)
	p_switch = np.asarray(p_switch)
	b_stay = np.logical_not(b_switch).astype(int)
	p_stay = 1-p_switch
	##the equation to compute log L
	logL = ((b_switch*np.log(p_switch)).sum()/b_switch.sum())+(
		(b_stay*np.log(p_stay)).sum()/b_stay.sum())
	return logL



	