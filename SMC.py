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
	y_part = init_p
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
		w = observef(y_part_next)
		w = w/sum(w) ##normalization
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
	eta = np.exp(particles[4:]) ##the particles representing the alpha var
	particles[action-1,:] = particles[action-1,:]+eta*(
		reward-particles[action-1,:])
	return particles


"""
Compute the probability of choosing 
action a (Pa), then make a choice based on this probability.
Inputs:
	action: action taken last time
	particles, where 
		-index[0,:] = action values for choice a
		-index[1,:] = action values for choice b
		-index[2,:] = alpha parameter (indecision point)
		-index[3,:] = beta parameter (inverse temperature)
		-index[4,:] = eta parameter (learning rate)
Returns:
	Pa, probability of action a
	action: chosen action
	p_switch: probability of switching actions
"""
def action_prob(particles):
	##probability of choosing action a
	Qa = particles[0,:]
	Qb = particles[1,:]
	beta = particles[3,:]
	alpha = particles[2,:]
	Pa = luce_choice(beta*((Qa-Qb)-alpha))
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
A function to resample particles
Inputs:
	weights: action probability weights
	n_particles: number of particles to resample
Returns:
	idx: index to use for resampling the particles
"""
def residual_resampling(weights,n_particles):
	k = np.floor(weights*n_particles).astype(bool).astype(int)
	n_residuals = int(n_particles-sum(k))
	idx = []
	for i in range(1,int(np.max(k))+1):
		idx += list(np.where(k>=i)[0])
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
	p[2,:] = np.random.rand(n_particles) #alpha
	p[3,:] = np.random.rand(n_particles) #beta
	p[4,:] =np.random.rand(n_particles)+np.log(0.1)#eta
	return p