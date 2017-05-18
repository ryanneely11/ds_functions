
##HMM model
##Functions/class to implement an actor that 
##uses a HMM model to make action choices

import numpy as np

"""
A function to initialize random parameters for the RL model
"""
def initp_HMM(n_particles):
	p = np.random.rand(5,n_particles) ##Xt
	p[1,:] = np.random.rand(n_particles)*np.log(25)#beta
	p[3,:] = np.ones(n_particles)*0.8+np.random.randn()/10
	p[4,:] = np.zeros(n_particles)+np.random.randn()/10
	return p
