import numpy as np


"""
Inputs:
	action_seq: sequence of actions represented by integers
	reward_seq: sequence of outcomes (binary) array
	updatef: function to use for updating
	observef: function for observations
	init_p: initial particle array; should be some distrubution
		around the best guess. Size n-params x m-particles. This represents
		the probability distribution of the hidden variables
	sd_jitter: standard deviation to use for jitter
Returns:
	e_val: expected value of hidden parameters
	v_val: variances of hidden parameters
"""
def SMC(action_seq,reward_seq,init_p,sd_jitter):
	##TODO: add these to the input params
	updatef = rescorlawagner
	observef = action_prob
	##initialize some params
	assert action_seq.size == reward_seq.size
	n_trials = action_seq.size
	n_params = init_p.shape[0]
	n_particles = init_p.shape[1]
	y_part = np.copy(init_p)
	##the output data arrays
	e_val = np.zeros((n_params,n_trials))
	v_val = np.zeros((n_params,n_trials))
	##step through the routine
	for t in range(1,n_trials):
		##update step
		y_part_next = updatef(action_seq[t-1],reward_seq[t-1],y_part)
		e_val[:,t] = np.mean(y_part_next,axis=1)
		v_val[:,t] = np.var(y_part_next,axis=1)
		##weighting step
		w = observef(action_seq[t],y_part_next)
		w = w/np.sum(w) ##normalization
		##resampling 
		idx = residual_resampling(w,n_particles)
		y_part = y_part_next[:,idx]
		##add jitter
		y_part = y_part + np.random.randn(n_params,n_particles
			)*np.outer(sd_jitter,np.ones(n_particles))
	return e_val,v_val

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
Compute the probability of choosing 
action a (Pa), then make a choice based on this probability.
Inputs:
	action: action taken last time
	particles, where 
		-index[0,:] = action values for choice a
		-index[1,:] = action values for choice b
		-index[2,:] = eta parameter (learning rate)
		-index[3,:] = beta parameter (inverse temperature)
		-index[4,:] = alpha parameter (indecision point)
Returns:
	Pa, probability of action a
	action: chosen action
	p_switch: probability of switching actions
"""
def action_prob(action,particles):
	##probability of choosing action a
	a_scalar = -2*(action-1.5) ##flip the sign for action a or b
	Qa = particles[0,:]
	Qb = particles[1,:]
	beta = np.exp(particles[3,:])
	alpha = np.exp(particles[4,:])
	Pa = luce_choice((a_scalar*beta)*((Qb-Qa)-alpha))
	return Pa

"""
A helper function to imlement the 
Luce Choice Rule.
Inputs:
	z: value to use in computation
returns:
	Pa: action probability
"""
def luce_choice(z):
	return 1.0/(1+np.exp(-z))

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
A function to resample particles
Inputs:
	weights: action probability weights
	n_particles: number of particles to resample
Returns:
	idx: index to use for resampling the particles
"""
def residual_resampling(weights,n_particles):
	k = np.floor(weights*n_particles).astype(int)
	n_residuals = int(n_particles-np.sum(k))
	idx = []
	for i in range(int(np.max(k))):
		idx += list(np.where(k>=i+1)[0])
	weights = (weights*n_particles)-k
	weights = weights/np.sum(weights)
	resid_idx = pdf2rand(weights,n_residuals)
	idx = idx + resid_idx
	return idx

"""
A function to approximate multinomial distrubution
Inputs: 
	pdf= density vector
	n_samp: number of samples
Returns:
	sample: approximation
"""
def pdf2rand(pdf,n_samp):
	c = np.max(pdf)
	r = pdf/c
	sample = []
	N = len(pdf)
	accept = np.zeros(n_samp).astype(bool)
	firstrnd = np.zeros(n_samp)
	randsample = np.zeros((2,n_samp))
	nn = n_samp
	while nn > 0:
		randsample = np.random.rand(2,nn)
		firstrnd = (np.floor(randsample[0,:]*N)).astype(int)
		accept = r[firstrnd] > randsample[1,:]
		sample += list(firstrnd[accept])
		nn = (accept==0).sum()
	return sample

"""
A function to initialize parameters for the RL model
"""
def init_p_RL(n_particles):
	p = np.random.randn(5,n_particles)+0.5
	p[2,:] = np.random.rand(n_particles)+np.log(0.1)#eta
	p[3,:] = np.random.rand(n_particles) #beta
	p[4,:] = np.random.rand(n_particles) #alpha
	return p

def simple_resampling(pdf,n_samp):
	return pdf2rand(pdf,n_samp)


####################################################
####################################################
#############      HMM    ###########################

def HMM_update(action,reward,particles):
	Xt = particles[0,:] ##the state value
	delta = particles[3,:] ##the transition probability
	corr_mean = particles[4,:] ##the probability of a reward when correct
	incorr_mean = particles[5,:]


"""
A function to compute the Bayesian prior based on the upcoming choice, the 
posterior from the previous trial, and the transition probability.
Input:
	choice; 'switch' or 'stay'
	Xt: posterior probability of the state distribution
Returns: 
	prior_Xt: estimate that the current state is correct
"""
def compute_prior(choice,Xt,delta):
	##get the transition probabilities for the different states
	P_trans = self.compute_p_transition(choice,delta) ##this is a list with 2 values
	return (P_trans[0]*posterior_Xt)+(P_trans[1]*(1-posterior_Xt))

"""
Returns the probability that the state is correct given the choice to stay or 
switch, computed for (index[0]), the case where the previous state was correct
and (index[1]) the case where the previous state was incorrect.
Inputs:
	choice; either 'stay' or 'switch'
	delta: transition probability
Retuns:
	P_transition: list of of probabilities that the current state estimate is correct
"""
def compute_p_transition(choice,delta):
	P_transition = [0,0]
	if choice == 'switch':
		P_transition[0] = self.delta ##case where the state was correct but we switched
		P_transition[1] = 1-self.delta ##case where the state was incorrect and we switched
	elif choice == 'stay':
		P_transition[0] = 1-self.delta
		P_transition[1] = self.delta
	return P_transition

"""
A function to compute the posterior given the outcome of the trial, 
as well as the prior, and some info about the reward probabilities.
Inputs:
	outcome: the outcome of the current trial
	prior_Xt: the prior estimate of the current trial
Returns:
	posterior_Xt
"""
def compute_posterior(outcome,prior_Xt):
	##compute the probability of this outcome given different state possibilities
	P_outcomes = self.compute_p_outcome(outcome)
	posterior = (P_outcomes[0]*prior_Xt)/((P_outcomes[0]*prior_Xt)+(
		P_outcomes[1]*(1-prior_Xt)))
	#diagnostics:
	if posterior < 0 or posterior > 1:
		print("posterior error: prior={}, p_outcome[0]={}, p_outcome[1]={}".format(
			prior_Xt,P_outcomes[0],P_outcomes[1]))
	return posterior

"""
A function to compute the probability of recieving a reward given
being in one of the states. 
Inputs:
	reward value, either 1 or 0
Returns:
	P_outcomes: the probability of receiving the given outcome
		assuming you were in [state1(correct), state2(incorrect)]
"""
def compute_p_outcome(outcome):
	P_outcomes = [0,0]
	if outcome == 1:
		P_outcomes[0] = np.random.binomial(10,self.correct_mean)/10
		P_outcomes[1] = 1-P_outcomes[0]
	elif outcome == 0:
		P_outcomes[0] = np.random.binomial(10,self.incorrect_mean)/10
		P_outcomes[1] = 1-P_outcomes[0]
	##make sure we don't have any zeros
	for i in range(len(P_outcomes)):
		if P_outcomes[i] == 0:
			P_outcomes[i] = 0.02
	return P_outcomes