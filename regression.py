##regression.py
##contains functions for running regression analyses

import numpy as np
from sklearn import linear_model
from sklearn.model_selection import cross_val_score
import multiprocessing as mp

"""
A function to run regression on a data matrix.
Inputs:
	-X: data matrix in shape trials x units x bins
	-R: matrix of regressors in shape trials x regressors
Returns:
	-C: matrix of coefficients, shape units x regressors x bins
	-num_sig: matrix with the counts of units showing significant regression
		values at for each regressor at each bin (regressors x bins)
	-mse: mean squared error of the model fit at each bin 
		based on x-validation (size = bins)
"""
def regress_matrix(X,R):
	##allocate the output arrays
	C = np.zeros((X.shape[1],R.shape[1],X.shape[2]))
	num_sig = np.zeros((X.shape[1],R.shape[1],X.shape[2]))
	MSE = np.zeros(X.shape[2])
	##split data into individual packets that can be handled
	##by multipe cores:
	arglist = [(X[:,:,i],R) for i in range(X.shape[2])] ##each tuple contains 1 bin of X data
	pool = mp.Pool(processes=mp.cpu_count())
	async_result = pool.map_async(run_regression,arglist)
	pool.close()
	pool.join() ##wait for processes to finish	
	data = async_result.get() ##a list of results from each bin
	##parse into the output arrays
	for b in range(X.shape[2]):
		C[:,:,b] = data[b][0]
		num_sig[:,:,b] = data[b][1]
		MSE[b] = data[b][2]
	return C,num_sig,MSE

"""
A function to return regressor values given a dictionary
of trial indices that have been categorized into 
different trial categories (see the ts_idx return from
pt.get_trial_data). This function returns the followin
regressors:
	-C(t): choice (upper (C=1) or lower (C=-1) lever)
	-R(t): outcome (rewarded (R=1) or unrewarded (R=0)
	-X(t): choice-reward outcome (C x R)
	-Qu(t): Action value of upper lever (0 or 0.85)
	-Ql(t): Action value of lower lever (0 or 0.85)
	-Qc(t): Action value of chosen lever (0 or 0.85)
Inputs:
	ts_idx: dictionary of trial indices
Outputs:
	R: matrix of regression values (in the same order as above)
"""
def regressors_model1(ts_idx):
	num_regressors = 6 ##this model happens to have 6 regressors
	##how many trials do we have? This will be the highest index
	#value, which we can find using only the upper and lower rewarded
	#blocks (they contain all trials between them)
	num_trials = max(ts_idx['upper_rewarded'].max(),ts_idx['lower_rewarded'].max())+1
	##allocate the output data array
	R = np.zeros((num_trials,num_regressors))
	#now we can figure out the regression values for all trials by
	##just checking which catagories a given trial index belongs to
	for t in range(num_trials):
		##the value of the upper and lower levers for this trial
		if t in ts_idx['upper_rewarded']:
			Qu = 0.85
			Ql = 0.05
		elif t in ts_idx['lower_rewarded']:
			Qu = 0.05
			Ql = 0.85
		else:
			raise IndexError("Trial with unknown block type")
		##the choice of lever for this trial (1 = upper, -1 = lower)
		if t in ts_idx['upper_lever']:
			C = 1
		elif t in ts_idx['lower_lever']:
			C = -1
		else:
			raise IndexError("Trial with unknown action")
		##the outcome of this trial (1 = rewarded, 0 = unrewarded)
		if t in ts_idx['rewarded']:
			r = 1
		elif t in ts_idx['unrewarded']:
			r = 0
		else:
			raise IndexError("Trial with unknown outcome")
		##the choice-reward interaction
		X = C*r
		##the value of the action that was chosen
		if C == 1: ##if the upper lever was chosen, Qc == Qu
			Qc = Qu
		elif C == -1: #if the lower was chosen, Qc == Ql
			Qc = Ql
		##now put all the values into the return array
		trial_vals = np.array([C,r,X,Qu,Ql,Qc])
		R[t,:] = trial_vals
	return R

"""
A function to run a multiple linear regression and return the coefficient
values. The error term/intercept is accounted for by the function.
Inputs:
	args, a tuple containing:
		-y: a vector of regressands, which is the spike rate over 
			an interval of interest for n trials. Can be multi-dimensional (include 
			data for multiple units). Should take the form n_samples x m_units
		-X: an n-trials by m-regressors array
Returns:
	-coeff: fitted coefficient values, size n_units x m_coeffs
	-num_sig: number of units with a significant value for each coefficient, size n-coeffs
	-MSE: mean squarred error of the regression fit, calculated by cross-validation
"""
def run_regression(args):
	##parse input tuple
	y = args[0] ##spike data
	X = args[1] ##regressors
	##initialize the regression
	regr = linear_model.MultiTaskElasticNetCV(fit_intercept=True)
	##fit the model
	regr.fit(X,y)
	##get the coefficients
	coeff = regr.coef_
	##get the accuracy of the prediction
	score = cross_val_score(regr,X,y)
	##determine the number of significant units at this timepoint
	num_sig = np.zeros(coeff.shape)
	for u in range(coeff.shape[0]): ##the number of units
		#F,p = t_test_coeffs(y[:,u],X) ##uncomment to use t-test (parametric)
		p = permutation_test(coeff[u,:],y[:,u],X) ##uncomment for permutation test
		sig_idx = np.where(p<=0.05)[0]
		num_sig[u,sig_idx] = 1
	return coeff,num_sig,abs(score).mean()

"""
A function to test the significance of regression coefficients
using a permutation test. A parametric t-test could also be used, but
if action values are included in the regression it makes more sense to 
use permutation tests (see Kim et al, 2009)
Inputs:
	-coeffs: the values of the coefficients to test for significance
	-y: regressand data used to generate the coefficients
	-X: regressor data (trials x features)
	-repeat: number of trials to conduct
Returns:
	p: p-value for each coefficient, defined as the frequency with which
		the shuffled result exceeded the actual result
"""
def permutation_test(coeffs,y,X,repeat=1000):
	regr = linear_model.RidgeCV(fit_intercept=True)
	##the # of times the shuffled val exceeded the experimental val
	c_exceeded = np.zeros(coeffs.size)
	for i in range(repeat):
		y_shuff = np.random.permutation(y)
		regr.fit(X,y_shuff)
		c_shuff = regr.coef_
		for i in range(c_shuff.size):
			if abs(c_shuff[i]) > abs(coeffs[i]):
				c_exceeded[i]+=1
	p = c_exceeded/float(repeat)
	return p

"""
A function to run a parametric t-test on the regression coefficients.
Inputs:
	y: regressand data
	X: regressor array (n-trials by m-regressors)
Returns:
	F: F-values for each regressor
	p: p-value for each regression coefficient
"""
def t_test_coeffs(y,X):
	F,p = f_regression(X,y)
	return F,p


"""
a helper function to find the maximum norm of an array
along the LAST AXIS. IE, if you want to get the max coeff. value
over a given interval
Inputs:
	A: array; must be 3-D with dims coeffs x bins/time x units
Returns:
	Amax: maximum absolute value of A along the last axis
"""
def max_norm(A):
	Amax = np.zeros((A.shape[0],A.shape[2]))
	for i in range(A.shape[0]): ##the coefficients axis
		for j in range(A.shape[2]): ##the units axis
			idx = np.argmax(abs(A[i,:,j]))
			Amax[i,j] = A[i,idx,j]
	return Amax


"""
A function to get the indices of units that have significant regression 
coefficients for at least 50% of an epoch interval
Inputs:
	-num_sig: binary array that has 1s indicating that a unit
		had a significant regression coefficient at the timepoint.
		(see output from run_regression)
Returns:
	-sig_units: a list; each list entry corresponds to one
		regression coefficient (same order as input args).
		The value at that index is an array of indices pointing
		to units that had significant regression coefficients for 
		at least 1/2 of the time interval
"""
def find_sig_units(num_sig):
	sig_units = []
	interval = num_sig.shape[2] ##the length of the time interval
	for c in range(num_sig.shape[1]): ##the coefficient dimension
		idx = []
		for u in range(num_sig.shape[0]):
			if (num_sig[u,c,:].sum() >= interval/2.0):
				idx.append(u)
		sig_units.append(np.asarray(idx))
	return sig_units
