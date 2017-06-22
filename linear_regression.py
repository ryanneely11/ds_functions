##linear_regression.py
#functions to run linear regression on task variables

import pandas as pd
import numpy as np
import model_fitting as mf
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
import multiprocessing as mp
from sklearn.metrics import mean_squared_error


"""
Utilizes the trial_data dataframe to return
an array of regressors for each trial. Regressors will be:
-action choice
-outcome
-Q_lower-Q_upper
-S_upper_rewarded-S_lower_rewarded
Inputs:
	Trial_data dataframe
Returns:

"""
def get_regressors(trial_data):
	##the column names for each regressor
	columns = ['action','outcome','q_diff','belief']
	n_trials = trial_data.shape[0]
	##using the trial data, get the Q-learning model results and the HMM results
	fits = mf.fit_models_from_trial_data(trial_data)
	##let's create a pandas dataframe for all the regressors
	regressors = pd.DataFrame(columns=columns,index=np.arange(n_trials))
	##now add all of the relevant data to the DataFrame
	regressors['action'] = fits['actions']
	regressors['outcome'] = fits['outcomes']
	regressors['state'] = (trial_data['context']=='upper_rewarded').astype(int)
	regressors['q_diff'] = np.abs(fits['Qvals'][1,:]-fits['Qvals'][0,:])
	regressors['belief'] = np.abs(fits['state_vals'][0,:]-fits['state_vals'][1,:])
	return regressors

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
Instead of using regression to predict the firing rates of one
neuron across trials, we can do the reverse and instead use population activity to
predict the activity of a continuous task variable (like logistic regression
but the without binary variables). In this case, we want to return something
that tells us how predictive the population is for that task parameter.
Inputs:
	X: neural data; t-trials by n_neurons (single bin)
	y: continuous task variable across trials
	add_constant: True if you want to add a constant to the X data
	n_iter: number of times to run the cross-validation
Returns:
	betas: beta parameters
	r2: variance explained
	r2_adj: adjusted r2 for multiple regressors
	mse: mean squared error of fit; cross-validated
"""
def lin_fit(X,y,add_constant=True,n_iter=10):
	if add_constant:
		X = sm.add_constant(X,has_constant='add')
	mse = np.zeros(n_iter)
	##use cross validation to compute the mean squared error
	for i in range(n_iter):
		##split the data into train and test sets
		X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.33,
			random_state=42)
		##now fit to the test data
		model = sm.OLS(y_train,X_train,hasconst=True)
		results = model.fit(method='pinv')
		predictions = results.predict(X_test)
		mse[i] = mean_squared_error(y_test,predictions)
	##now get the % variance explained and adjusted R2 stats
	##using the full model
	model = sm.OLS(y,X,hasconst=True)
	results = model.fit(method='pinv')
	r2 = results.rsquared
	r2_adj = results.rsquared_adj
	betas = results.params
	return betas, r2, r2_adj, mse.mean()
	
"""
A function that takes in a timecourse of neural activity, 
and returns the estimate of the belief state, the r-squared, and 
mean squared error of the estimate over that timecourse.
Inputs:
	X: neural data, in shape trials x neurons x timesteps
	y: dependent data
	add_constant: True if no constant is present in X data
	n_iter: number of iterations for cross validation
Returns: 
	predictions: a n-trials x t-timepoints array showing the predicted
		values for each trial over time
	mse: mean squared error of the predictions over time
	r2: ""
	r2_adj: ""
"""
def fit_timecourse(X,y,add_constant=True,n_iter=10):
	n_trials = X.shape[0]
	# print("n_trials={}".format(n_trials))
	n_bins = X.shape[2]
	# print("n_bins={}".format(n_bins))
	if add_constant:
		n_neurons = X.shape[1]+1
	else:
		n_neurons = X.shape[1]
	# print("n_neurons={}".format(n_neurons))
	predictions = np.zeros((n_trials,n_bins))
	r2 = np.zeros(n_bins)
	r2_adj = np.zeros(n_bins)
	mse = np.zeros(n_bins)
	betas = np.zeros((n_neurons,n_bins))
	# print("shape of betas={}".format(betas.shape))
	for b in range(n_bins):
		# print("bin#{}".format(b))
		betas[:,b],r2[b],r2_adj[b],mse[b] = lin_fit(X[:,:,b],y,
		add_constant=add_constant,n_iter=n_iter)
	##now compute the predictions using a fixed beta
	# betas = betas[:,-10:].mean(axis=1)
	# betas = np.mean(betas,axis=1)
	betas = np.argmax(betas,axis=1)
	for b in range(n_bins):
		if add_constant:
			x = sm.add_constant(X[:,:,b],has_constant='add')
		else:
			x = X[:,:,b]
		predictions[:,b] = np.dot(x,betas)
	return predictions,r2,r2_adj,mse
