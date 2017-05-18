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
			-index[2,:] = alpha parameter (indecision point)
			-index[3,:] = beta parameter (inverse temperature)
			-index[4,:] = eta parameter (learning rate)
Returns:
	particle_next:
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
	pass