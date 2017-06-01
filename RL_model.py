##RL_model: a reinforecement learning model
##implemented using the Rescorla Wagner rule

import numpy as np

"""
A function to initialize random parameters for the RL model
"""
def initp(n_particles):
	p = np.random.randn(4,n_particles)+0.5
	p[2,:] = np.random.rand(n_particles)+np.log(0.1)#eta
	p[3,:] = np.random.rand(n_particles) #beta
	return p

"""
A function to convert an array of action strings,
ie 'upper_lever' into ints. In this case, upper = 2,
lower = 1.
Input:
	action_names: list or array of action strings
Returns:
	actions: array where strings are converted to int codes
"""
def convert_actions(action_names):
	actions = np.zeros(len(action_names))
	upper = np.where(action_names=='upper_lever')[0]
	lower = np.where(action_names=='lower_lever')[0]
	actions[upper] = 2
	actions[lower] = 1
	return actions

"""
An update function based on the rescorla wagner rule. Simply 
updates the action value of an action by the difference between
the expected and actual reward, multiplied by alpha(the learning rate).
Inputs:
	-action: the action taken in this trial
	-outcome: the outcome for the given action
	-particles: the particle samples representing the PDF of
		the hidden variables, where
			-index[0,:] = action values for choice a
			-index[1,:] = action values for choice b
			-index[2,:] = eta parameter (learning rate)
			-index[3,:] = beta parameter (inverse temperature)
Returns:
	particle_next: updated particles array
"""
def rescorlawagner(action,reward,particles):
	eta = np.exp(particles[2,:]) ##the particles representing the alpha var
	particles[int(action-1),:] = particles[int(action-1),:]+eta*(
		reward-particles[int(action-1),:])
	return particles

"""
An alternate form of action selection, that doesn't
require an alpha equivalence point parameter.
Inputs:
	-action: the index of the action taken; 1 or 2
	-particles: the probability distribution
Returns:
	Pa: probability of action a
"""
def boltzmann(action,particles):
	beta = np.exp(particles[3,:])
	return 1.0/(1+np.exp(-2.0*(action-1.5)*(beta*np.diff(particles[0:2,:],axis=0).squeeze())))


"""
A function to take arrays of action values and fitted beta parameters
at each times step and compute the array of corresponding actions.
Inputs:
	Qa: action values for action a
	Qb: action values for action b
	Beta: beta parameters
Returns:
	Pa: the computed probability of action a
	actions: an int array of the actual actions taken
"""
def compute_actions(Qa,Qb,Beta):
	Beta = np.exp(Beta)
	choices = np.array([1,2])
	actions = np.zeros(Qa.shape)
	Pa = np.zeros(Qa.shape)
	for i in range(Qa.size):
		qa = Qa[i]
		qb = Qb[i]
		beta = Beta[i]
		pa = 1.0/(1+np.exp(beta*(qb-qa)))
		probs = np.array([pa,1-pa])
		actions[i] = np.random.choice(choices,p=probs)
		Pa[i] = pa
	return actions,Pa

"""
A function to compute the reward prediction error.
Inputs are data from a fitted model/behavior pair.
Inputs:
	Results a results dictionary from model_fitting.fit_models.
Returns:
	RPE: the reward prediction error at every trial
"""
def RPE(results):
	##first get the value of upper and lower lever presses on each trial
	Q_upper = results['e_RL'][1,:]
	Q_lower = results['e_RL'][0,:]
	##actions and outcomes (actual)
	actions = results['actions']
	outcomes = results['outcomes']
	##allocate memory
	RPE = np.zeros(actions.shape)
	##compute for each trial
	for t in range(actions.size):
		if actions[t] == 2: ##case upper lever
			Q = Q_upper[t]
		elif actions[t] == 1: #case lower lever
			Q = Q_lower[t]
		RPE[t] = Q-outcomes[t]
	return RPE