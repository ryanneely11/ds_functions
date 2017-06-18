##log_regression2:

#Functions for logistic regression. New implementation
#uses statsmodels to do model fitting and accuracy testing

import numpy as np
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
import pandas as pd
import multiprocessing as mp
import parse_trials as ptr

"""
A function to fit a cross-validated logistic regressions, and return the 
model accuracy using three-fold cross-validation.
inputs:
	X: the independent data; could be spike rates over period.
		in shape samples x features (ie, trials x spike rates)
	y: the class data; should be binary. In shape (trials,)
	n_iter: the number of times to repeat the x-validation (mean is returned)
Returns:
	accuracy: mean proportion of test data correctly predicted by the model.
	llr_p: The chi-squared probability of getting a log-likelihood ratio statistic greater than llr.
		 llr has a chi-squared distribution with degrees of freedom df_model
"""
def log_fit(X,y,add_constant=False,n_iter=10):
	##get X in the correct shape for sklearn function
	if len(X.shape) == 1:
		X = X.reshape(-1,1)
	accuracy = np.zeros(n_iter)
	if add_constant:
		X = sm.add_constant(X)	
	for i in range(n_iter):
		##split the data into train and test sets
		X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.33,random_state=42)
		##make sure you have both classes of values in your training and test sets
		if np.unique(y_train).size<2 or np.unique(y_test).size<2:
			print("Re-splitting cross val data; only one class type in current set")
			X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.5,random_state=42)
		##now fit to the test data
		logit = sm.Logit(y_train,X_train)
		results = logit.fit(method='cg',disp=False,skip_hession=True,
			warn_convergence=False)
		##get the optimal cutoff point for the training data
		thresh = find_optimal_cutoff(y_train,results.predict(X_train))
		##now try to predict the test data
		y_pred = (results.predict(X_test)>thresh).astype(float)
		##lastly, compare the accuracy of the prediction
		accuracy[i] = (y_pred==y_test).sum()/float(y_test.size)
	##now get the p-value info for this model
	logit = sm.Logit(y,X)
	results = logit.fit(method='newton',disp=False,skip_hession=True,warn_convergence=False)
	llr_p = results.llr_pvalue
	return accuracy.mean(),llr_p

"""
A function to do population spike activity model fitting across some
time window/binned spikes. 
inputs:
	X: the independent data; could be spike rates over period.
		in shape samples x features x time bins (ie, trials x spike rates)
	y: the class data; should be binary. In shape (trials,)
	n_iter: the number of times to repeat the x-validation (mean is returned)
	add_constant: if the function should add a constant
Returns:
	accuracy: mean proportion of test data correctly predicted by the model
		at each time bin
"""
def pop_logit(X,y,add_constant=False,n_iter=10):
	n_bins = X.shape[2]
	accuracy = np.zeros(n_bins)
	for b in range(n_bins):
		a,pval = log_fit(X[:,:,b],y,add_constant=add_constant,n_iter=n_iter)
		accuracy[b] = a
	return accuracy

"""
A function to perform a permutation test for significance
by shuffling the training data. Uses the cross-validation strategy above.
Inputs:
	args: a tuple of arguments, in the following order:
		X: the independent data; trials x features
		y: the class data, in shape (trials,)
		n_iter_cv: number of times to run cv on each interation of the test
		n_iter_p: number of times to run the permutation test
returns:
	accuracy: the computed accuracy of the model fit
	p_val: proportion of times that the shuffled accuracy outperformed
		the actual data (significance test)
"""
def permutation_test(args):
	##parse the arguments tuple
	X = args[0]
	y = args[1]
	n_iter_cv = args[2]
	n_iter_p = args[3]
	##get the accuracy of the real data, to use as the comparison value
	a_actual,llr_p = log_fit(X,y,n_iter=n_iter_cv)
	#now run the permutation test, keeping track of how many times the shuffled
	##accuracy outperforms the actual
	times_exceeded = 0
	chance_rates = [] 
	for i in range(n_iter_p):
		y_shuff = np.random.permutation(y)
		a_shuff,llr_p_shuff = log_fit(X,y_shuff,n_iter=n_iter_cv)
		if a_shuff > a_actual:
			times_exceeded += 1
		chance_rates.append(a_shuff)
	return a_actual, np.asarray(chance_rates).mean(), float(times_exceeded)/n_iter_p, llr_p
	

"""
a function to run permutation testing on multiple
INDEPENDENT datasets that all correspond to one class dataset; ie
many X's that correspond to the same y. For example, data from several
recorded simultaneously, and the outcomes of a number of trials. We will be
using python's multiprocessing function to speed things up.
Inputs:
	X: dependent data in the form n_datasets x n_trials x n_samples
		ie, n_units x n_trials x n_bins
	y: the binary class data that applies to each X
	n_iter_cv: the number of cross-validation iterations to run
	n_iter_p: the number of permutation iterations to run
Returns:
	accuracies: an array of the prediction accuracies for each dataset
	chance_rates: the chance accuracy rates
	p_vals: an array of the significance values for each dataset
"""
def permutation_test_multi(X,y,n_iter_cv=5,n_iter_p=500):
	##make sure that the array is in binary form
	if (y.min() != 0) or (y.max() != 1):
		print("Converting to binary y values")
		y = binary_y(y)
	##setup multiprocessing to do the permutation testing
	arglist = [(X[n,:,:],y,n_iter_cv,n_iter_p) for n in range(X.shape[0])]
	pool = mp.Pool(processes=mp.cpu_count())
	async_result = pool.map_async(permutation_test,arglist)
	pool.close()
	pool.join()
	results = async_result.get()
	##parse the results
	accuracies = np.zeros(X.shape[0])
	chance_rates = np.zeros(X.shape[0])
	p_vals = np.zeros(X.shape[0])
	llr_pvals = np.zeros(X.shape[0])
	for i in range(len(results)):
		accuracies[i] = results[i][0]
		p_vals[i] = results[i][2]
		chance_rates[i] = results[i][1]
		llr_pvals[i] = results[i][3]
	return accuracies,chance_rates,p_vals,llr_pvals

"""
This function returns the beta parameters across several time
steps for binned spike counts from a number of neurons. 
Inputs:
	X: binned spike data, shape n-trials x m neurons x t bins
	y: labels, in binary format, for each trial
Returns:
	betas: beta values at each time step
"""
def get_betas(X,y,add_intercept=False):
	##allocate space
	betas = np.zeros((X.shape[1],X.shape[2])) ##add one because we will add a constant
	##compute for each time step
	for i in range(X.shape[2]): ##the bin number
		if add_intercept:
			x = sm.tools.tools.add_constant(X[:,:,i]) ##should be trials (obs) x units
			betas = np.zeros((X.shape[1]+1,X.shape[2]))
		else:
			x = X[:,:,i]
		logit = sm.Logit(y,x)
		results = logit.fit(method='newton',disp=False,skip_hessian=True,warn_convergence=False)
		betas[:,i] = results.params
	return betas

"""
A helper function to make a non-binary array
that consists of only two values into a binary array of 1's 
and 0's to use in regression
Inputs:
	y: a non-binary array that ONLY CONSISTS OF DATA THAT TAKES TWO VALUES
Returns:
	y_b: binary version of y, where y.min() has been changed to 0 and y.max()
		is represented by 1's
"""
def binary_y(y):
	ymin = y.min()
	ymax = y.max()
	y_b = np.zeros(y.shape)
	y_b[y==ymax]=1
	return y_b

""" Find the optimal probability cutoff point for a classification model related to event rate
Parameters
----------
target : Matrix with dependent or target data, where rows are observations

probs : Matrix with predicted data, where rows are observations

Returns
-------     
list type, with optimal cutoff value

"""
def find_optimal_cutoff(target, probs):
	fpr, tpr, threshold = roc_curve(target, probs)
	i = np.arange(len(tpr)) 
	roc = pd.DataFrame({'tf' : pd.Series(tpr-(1-fpr), index=i), 'threshold' : pd.Series(threshold, index=i)})
	roc_t = roc.ix[(roc.tf-0).abs().argsort()[:1]]
	return list(roc_t['threshold'])[0]

"""
A function to get regression coefficients for a pandas array of
trial data from one session. This model takes into account:
-action history
-reward history
-action-reward interaction history
-the training day
-the trial number in the session
Inputs:
	f_behavior: the behavior data file
	n_back: the number of previous trials to include in the history
	max_duration: the maximum allowable trial duration (ms)
Returns:
	y: outcome array (1 for upper, 0 for lower lever)
	X: predictor array
"""
def get_behavior_data(f_behavior,n_back=3,max_duration=5000):
	global trial_lut
	##start by parsing the trials
	trial_data = ptr.get_full_trials(f_behavior,max_duration=max_duration)
	##get the session number for this session
	session_num = ptr.get_session_number(f_behavior)
	##pre-allocation
	n_trials = len(trial_data.index)
	##how many features we have depends on the length of our
	##action/reward history
	features = ['training_day','trial_number']
	for i in range(n_back):
		features.append('action-'+str(i+1))
		features.append('outcome-'+str(i+1))
		features.append('interaction-'+str(i+1))
	##create the data arrays
	y = pd.DataFrame(index=np.arange(n_trials),columns=['value','action'])
	X = pd.DataFrame(index=np.arange(n_trials),columns=features)
	"""
	Now parse each trial using the following values:
	reward/no reward: 1 or 0
	upper lever/lower lever: 2 or 1
	"""
	for t in range(n_trials):
		##get the trial data for this trial
		trial = trial_data.loc[t]
		##fill out the outcomes array first
		y['value'][t] = trial_lut[trial['action']]
		y['action'][t] = trial['action']
		X['training_day'][t] = session_num
		X['trial_number'][t] = t
		for i in range(n_back):
			if t > n_back:
				X['action-'+str(i+1)][t] = trial_lut[trial_data['action'][t-(i+1)]]
				X['outcome-'+str(i+1)][t] = trial_lut[trial_data['outcome'][t-(i+1)]]
				X['interaction-'+str(i+1)][t] = trial_lut[trial_data['action'][t-(
					i+1)]]*trial_lut[trial_data['outcome'][t-(i+1)]]
			else:
				X['action-'+str(i+1)][t] = 0
				X['outcome-'+str(i+1)][t] = 0
				X['interaction-'+str(i+1)][t] = 0
	return y,X

trial_lut = {
'upper_lever':2,
'lower_lever':1,
'rewarded_poke':1,
'unrewarded_poke':0
}