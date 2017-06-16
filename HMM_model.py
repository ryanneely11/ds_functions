
##HMM model
##Functions/class to implement an actor that 
##uses a HMM model to make action choices

import numpy as np

"""
A function to initialize random parameters for the RL model
These particles are arraned as follows:
Particles[0,:] = b1; belief probability of being in state corresponding to action 1
particles[1,:] = b2; belief probabiliy of being in state corresponding to action 2
particles[2,:] = c; probability of a reward given state is correct
particles[3,:] = d; probability of no reward given state is incorrect
particles[4,:] = gamma: transition probability between states
"""
def initp(n_particles):
	p = np.random.rand(5,n_particles)
	return p

"""
A function that computes the belief that the state corresponds to action
1 at time t.
Inputs:
	action: observed action at time t-1
	reward: obeserved reward at time t-1
	particles: particles array
Returns:
	particles: updated particles array
"""
def compute_belief(action,reward,particles):
	##probability of switching to a new state
	Pswitch = transition_prob(particles)
	##probability of stayng in the current state
	Pstay = 1-transition_prob(particles)
	##probability of observing this action,reward pair given being in state 1
	Po1 = emission_prob(action,reward,1,particles)
	##probability of observing this action,reward pair given being in state 2
	Po2 = emission_prob(action,reward,2,particles)
	##previous belief probability of being in state 1
	b1 = particles[0,:]
	##previous belief probability of being in state 2
	b2 = particles[1,:]
	##new belief about being in state 1
	B1 = Pstay*((Po1*b1)/(Po2*b2+Po1*b1))+Pswitch*((Po2*b2)/(Po1*b1+Po2*b2))
	B2 = Pstay*((Po2*b2)/(Po1*b1+Po2*b2))+Pswitch*((Po1*b1)/(Po2*b2+Po1*b1))
	particles[0,:] = B1
	particles[1,:] = B2
	return particles


"""
A function to return the transition probability given
a current state
Inputs:
	particles: particles array
returns:
	Pswitch: transition probability; ie the probability that the next 
		state will be different than the current one 
"""
def transition_prob(particles):
	gamma = particles[4,:]
	Pswitch = 1-gamma
	return Pswitch

"""
A function to compute the emission probabilities; ie
the probability of observing a particular outcome given
a particular state.
Input:
	action: the action taken
	reward: the value of the outcome observed
	state: the int corresponding to the possible state
	particles: particles array
Returns:
	Po: the probability of observing the given outcome if you were in the given state
"""
def emission_prob(action,reward,state,particles):
	##the probability with which a reward tells us we are in state a
	c = particles[2,:]
	##the probability with which a punishment tells us we are not in state a
	d = particles[3,:]
	if state == action:
		if reward == 1:
			Po = c
		elif reward == 0:
			Po = -d
	elif state != action:
		if reward == 1:
			Po = -c
		elif reward ==0:
			Po = d
	return 0.5+0.5*Po

"""
A function to compute the weighting values across
belief states, given the probability of performing
action A, given the belief states, b, and the actual
action taken by the subject
Inputs: 
	-Particles: n-parameters by m-particles array
	-action: the action that was actually taken by the subject
Returns:
	weights: weighted values
"""
def action_weights(action,particles):
	##we only need the belief states from the particles
	b = particles[0,:]
	return 1.0/(1+np.exp(40*(action-1.5)*(b-0.5)))


"""
A function to take arrays of action values and fitted beta parameters
at each times step and compute the array of corresponding actions.
Inputs:
	Ba: belief states for action a
Returns:
	Pa: the computed probability of action a
	actions: an int array of the actual actions taken
"""
def compute_actions(Ba):
	choices = np.array([1,2])
	actions = np.zeros(Ba.shape)
	Pa = np.zeros(Ba.shape)
	for i in range(Ba.size):
		ba = Ba[i]
		pa = 1.0/(1+np.exp(-20*(ba-0.5)))
		probs = np.array([pa,1-pa])
		actions[i] = np.random.choice(choices,p=probs)
		Pa[i] = pa
	return actions,Pa

"""
A function to compute the state prediction error, given 
results from a model that has already been fit.
Inputs:
	-results: dictionary output from model_fitting.fit_models
Returns:
	-SPE: state prediction error on every trial
"""
def SPE(results):
	##first get the relevant data from the results dict
	actions = results['actions']
	outcomes = results['outcomes']
	s_upper = results['e_HMM'][1,:] ##upper lever state prob
	s_lower = results['e_HMM'][0,:] ##Lower lever prob
	SPE = np.zeros(actions.shape)
	##compute spe for each trial
	for t in range(actions.size):
		if actions[t] == 2: ##case upper lever
			s = s_upper[t] ##probability we are in the upper state
		elif actions[t] == 1: ##case lower lever
			s = s_lower[t] ##p(lower_state)
		SPE[t] = s-outcomes[t]
	return SPE

"""
A function that runs a session using an RL model
with definable (fixed) parameters
Inputs:
	fit_results: model fitting results
Returns:
	results: performance of model using the given parameters
"""
def run_model(fit_results):
	##parse some of the fitting results to construct the model
	actions = fit_results['actions']
	outcomes = fit_results['outcomes']
	n_trials = actions.size
	state_vals = np.zeros((2,n_trials)) ##this will be [0,:] Qlower and [1,:] Qupper
	params = np.zeros((5,1))
	params[0,:] = 0.5 ##p(state = lower)
	params[1,:] = 0.5 ##p(state = upper)
	params[2,:] = fit_results['e_HMM'][2,-1] ##p(reward | correct)
	params[3,:] = fit_results['e_HMM'][3,-1]##p(reward | incorrect)
	params[4,:] = fit_results['e_HMM'][4,-1]##gamma (transition probability)
	##now run through each trial and compute the values at each step
	for t in range(1,n_trials):
		params = compute_belief(actions[t-1],outcomes[t-1],params)
		state_vals[:,t] = params[0:2,0]
	##now compute the model's actions and action selection probability
	HMM_actions,p_lower = compute_actions(state_vals[0,:])
	return HMM_actions,p_lower,state_vals