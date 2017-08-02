##linear_regression2.py

##new functions to run linear regression

import numpy as np
import parse_trials as ptr
import model_fitting as mf
import pandas as pd
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
import multiprocessing as mp
from sklearn.metrics import mean_squared_error

to_regress = ['action','outcome','state','uncertainty',
'action x\noutcome', 'action x\nstate','action x\n uncertainty','outcome x\nstate',
'outcome x\nuncertainty','state x\nuncertainty']

"""
A function to pull regression data, including regressors and regressands
out of behavioral and session data. Regressors include:
-Action choice
-Trial outcome
-Task state
-Uncertainty (from HMM)
Inputs:
	f_behavior: file path to behavior data
	f_ephys: file path to ephys data
	smooth_method: type of smoothing to use; choose 'bins', 'gauss', 'both', or 'none'
	smooth_width: size of the bins or gaussian kernel in ms. If 'both', input should be a list
		with index 0 being the gaussian width and index 1 being the bin width
	pad: a window for pre- and post-trial padding, in ms. In other words, an x-ms period of time 
		before lever press to consider the start of the trial, and an x-ms period of time after
		reward to consider the end of the trial. For best results, should be a multiple of the bin size
	z_score: if True, z-scores the array
	trial_duration: specifies the trial length (in ms) to squeeze trials into. If None, the function uses
		the median trial length over the trials in the file
	min_rate: the min spike rate, in Hz, to accept. Units below this value will be removed.
	max_duration: maximum allowable trial duration (ms)
Returns:

"""
def get_datasets(f_behavior,f_ephys,smooth_method='both',smooth_width=[80,40],
	 pad=[1200,800],z_score=True,trial_duration=None,min_rate=0.1,max_duration=5000):
	global to_regress
	##start by getting the spike data and trial data
	spike_data,trial_data = ptr.get_trial_spikes(f_behavior,f_ephys,
		smooth_width=smooth_width,smooth_method=smooth_method,pad=pad,z_score=z_score,
		trial_duration=trial_duration,max_duration=max_duration,min_rate=min_rate)
	n_trials = spike_data.shape[0]
	##get the uncertainty from the trial data
	uncertainty = mf.uncertainty_from_trial_data(trial_data)
	##now, compute the regressors from the HMM and trial data
	##let's keep these as a pandas dataset just for clarity
	regressors = pd.DataFrame(columns=to_regress,index=np.arange(n_trials))
	##now fill out the regressors
	regressors['action'] = np.asarray(trial_data['action']=='upper_lever').astype(int)+1
	regressors['outcome'] = np.asarray(trial_data['outcome']=='rewarded_poke').astype(int)
	regressors['state'] = np.asarray(trial_data['context']=='upper_rewarded').astype(int)+1
	regressors['uncertainty'] = uncertainty
	##now do the interactions
	regressors['action x\noutcome'] = regressors['action']*regressors['outcome']
	regressors['action x\nstate'] = regressors['action']*regressors['state']
	regressors['action x\n uncertainty'] =regressors['action']*regressors['uncertainty']
	regressors['outcome x\nstate'] = regressors['outcome']*regressors['state']
	regressors['outcome x\nuncertainty']  = regressors['outcome']*regressors['uncertainty']
	regressors['state x\nuncertainty'] = regressors['state']*regressors['uncertainty']
	##Now just return the data arrays
	return spike_data,regressors

"""
A function to fit an OLS linear regression model, and return the significance of the 
coefficients.
inputs:
	X: the regressor data, should be n-observations x k-regressors
	y: the spike rate for each trial in a given bin. In shape (trials,)
Returns:
	p-values: the significance of each of the regressor coefficients
"""
def lin_ftest(X,y,add_constant=True):
	##get X in the correct shape for sklearn function
	if len(X.shape) == 1:
		X = X.reshape(-1,1)
	if add_constant:
		X = sm.add_constant(X)
	##now get the p-value info for this model
	model = sm.OLS(y,X,hasconst=True)
	results = model.fit(method='pinv')
	return results.pvalues[1:]

"""
A function to perform a permutation test for significance
by shuffling the training data.
Inputs:
	args: a tuple of arguments, in the following order:
		X: the independent data; trials x features
		y: the class data, in shape (trials,)
		n_iter_p: number of times to run the permutation test
returns:
	p_val: proportion of times that the shuffled accuracy outperformed
		the actual data (significance test)
"""
def permutation_test(X,y,n_iter=1000,add_constant=True):
	##parse the arguments tuple
	if add_constant:
		X = sm.add_constant(X)
	##get the coefficients of the real data, to use as the comparison value
	model = sm.OLS(y,X,has_constant=True)
	results = model.fit(method='pinv')
	coeffs = results.params[1:]##first index is the constant
	#now run the permutation test, keeping track of how many times the shuffled
	##accuracy outperforms the actual
	times_exceeded = np.zeros(X.shape[1]-1)
	for i in range(n_iter):
		y_shuff = np.random.permutation(y)
		model = sm.OLS(y_shuff,X,has_constant=True)
		results = model.fit(method='pinv')
		coeffs_shuff = results.params[1:]
		times_exceeded += (np.abs(coeffs_shuff)>np.abs(coeffs)).astype(int)
	return times_exceeded/n_iter

"""
A function to analyze the significance of a regressor/regressand pair over
the course of several timesteps. It is expected that the regressor (X) values
are constant while the y-values are changing. Ie, the spike rates (y) over the
course of trials (X) with fixed regressor values.
Inputs: (in list format)
	X: regressor array, size n-observations x k-features
	Y: regressand array, size n-observations x t bins/timesteps
	add_constant: if True, adds a constant
	n_iter: number of iterations to run the permutation test. if 0, no 
		permutation test is performed.
Returns:
	f_pvals: pvalues of each coefficient at each time step using f-test statistic (coeffs x bins)
	p_pvals: pvals using permutation test
"""
def regress_timecourse(args):
	##parse args
	X = args[0]
	y = args[1]
	add_constant = args[2]
	n_iter = args[3]
	if add_constant:
		n_coeffs = X.shape[1]
	else:
		n_coeffs = X.shape[1]-1
	n_bins = y.shape[1] ##number of time bins
	##setup output
	f_pvals = np.zeros((n_coeffs,n_bins))
	p_pvals = np.zeros((n_coeffs,n_bins))
	##run through analysis at each time step
	for b in range(n_bins):
		f_pvals[:,b] = lin_ftest(X,y[:,b],add_constant=add_constant)
		if n_iter > 0:
			p_pvals[:,b] = permutation_test(X,y[:,b],add_constant=add_constant,n_iter=n_iter)
		else:
			p_pvals[:,b] = np.nan
	return f_pvals,p_pvals

"""
a function to do regression on a spike matrix consisting of
binned spike data from many neurons across time. Result is a matrix
counting the number of significant neurons for each regressor in
each time bin.
Inputs:
	X: regressor array, size n-observations x k-features
	Y: regressand array, size n-observations x p neurons x t bins/timesteps
	add_constant: if True, adds a constant
	n_iter: number of iterations to run the permutation test. if 0, no 
		permutation test is performed.
Returns:
	f_counts: number of significant neurons at each time point according to f-test
	p_counts: same thing but using a permutation test
"""
def regress_spike_matrix(X,Y,add_constant=True,n_iter=1000):
	n_neurons = Y.shape[1]
	if add_constant:
		n_coeffs = X.shape[1]
	else:
		n_coeffs = X.shape[1]-1
	n_bins = Y.shape[2] ##number of time bins
	##setup output data
	f_pvals = np.zeros((n_neurons,n_coeffs,n_bins))
	p_pvals = np.zeros((n_neurons,n_coeffs,n_bins))
	##basically just perform regress_timecourse for each neuron
	##use multiprocessing to speed up the permutation testing.
	arglist = [[X,Y[:,n,:],add_constant,n_iter] for n in range(n_neurons)]
	pool = mp.Pool(processes=n_neurons)
	async_result = pool.map_async(regress_timecourse,arglist)
	pool.close()
	pool.join()
	results = async_result.get()
	for n in range(len(results)):
		f_pvals[n,:,:] = results[n][0]
		p_pvals[n,:,:] = results[n][1]
	##now we want to count up the significant neurons
	p_thresh = 0.05
	f_counts = (f_pvals <= p_thresh).sum(axis=0)
	p_counts = (p_pvals <= p_thresh).sum(axis=0)
	return f_counts,p_counts

"""
a function to do regression on a spike matrix consisting of
binned spike data from many neurons across time. unlike the above function,
this one looks at each neuron and asks if single neurons represent
one or more variables
Inputs:
	X: regressor array, size n-observations x k-features
	Y: regressand array, size n-observations x p neurons x t bins/timesteps
	add_constant: if True, adds a constant
	n_iter: number of iterations to run the permutation test. if 0, no 
		permutation test is performed.
	n_consecutive: requires that encoding be significant for n consecutive time bins
Returns:
	n_encoded: number of parameters with some significant encoded by each neuron
		at some point during the trial

	###TODO: generate a permutation test-compatible version of this#####
"""
def regress_spike_matrix2(X,Y,add_constant=True,n_iter=0,n_consecutive=4):
	n_neurons = Y.shape[1]
	if add_constant:
		n_coeffs = X.shape[1]
	else:
		n_coeffs = X.shape[1]-1
	n_bins = Y.shape[2] ##number of time bins
	##setup output data
	f_pvals = np.zeros((n_neurons,n_coeffs,n_bins))
	p_pvals = np.zeros((n_neurons,n_coeffs,n_bins))
	##basically just perform regress_timecourse for each neuron
	##use multiprocessing to speed up the permutation testing.
	arglist = [[X,Y[:,n,:],add_constant,n_iter] for n in range(n_neurons)]
	pool = mp.Pool(processes=n_neurons)
	async_result = pool.map_async(regress_timecourse,arglist)
	pool.close()
	pool.join()
	results = async_result.get()
	for n in range(len(results)):
		f_pvals[n,:,:] = results[n][0]
		p_pvals[n,:,:] = results[n][1]
	##now we want to count up the significant neurons
	p_thresh = 0.05
	##this array will store, for each neuron, the number of 
	##variables that it encodes significantly at any point
	##during the trial
	n_neurons = f_pvals.shape[0]
	n_regressors = f_pvals.shape[1]
	n_encoded = np.zeros((n_neurons))
	for n in range(n_neurons):
		unit_data = f_pvals[n,:,:]
		##does this neuron encode any of the parameters at any point?
		encoded = np.zeros(n_regressors)
		for i in range(n_regressors):
			encoded[i] = bool(contiguous_regions(unit_data[i,:]<=0.05,n_consecutive))
		##count up how many params are encoded and add to the data
		n_encoded[n] = encoded.sum()
	return n_encoded
##helper
def contiguous_regions(condition,n_consecutive):
	"""Finds contiguous True regions of the boolean array "condition". Returns
	a 2D array where the first column is the start index of the region and the
	second column is the end index."""
	# Find the indicies of changes in "condition"
	d = np.diff(condition)
	idx, = d.nonzero() 
	# We need to start things after the change in "condition". Therefore, 
	# we'll shift the index by 1 to the right.
	idx += 1
	if condition[0]:
		# If the start of condition is True prepend a 0
		idx = np.r_[0, idx]
	if condition[-1]:
		# If the end of condition is True, append the length of the array
		idx = np.r_[idx, condition.size] # Edit
	# Reshape the result into two columns
	idx.shape = (-1,2)
	##find the length of contiguous sections
	n_contiguous = np.diff(idx,axis=-1)
	##count how many meet or exceed the min consecutive criteria
	n_meeting = (n_contiguous>=n_consecutive).sum()
	return n_meeting