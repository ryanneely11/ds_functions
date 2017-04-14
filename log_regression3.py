##log_regression2:

#Functions for logistic regression. New implementation
#uses statsmodels to do model fitting and accuracy testing

import numpy as np
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
import pandas as pd
import multiprocessing as mp

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
"""
def log_fit(X,y,n_iter=5):
	##get X in the correct shape for sklearn function
	if len(X.shape) == 1:
		X = X.reshape(-1,1)
	accuracy = np.zeros(n_iter)
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
	return accuracy.mean()

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
	a_actual = log_fit(X,y,n_iter=n_iter_cv)
	#now run the permutation test, keeping track of how many times the shuffled
	##accuracy outperforms the actual
	times_exceeded = 0
	chance_rates = [] 
	for i in range(n_iter_p):
		y_shuff = np.random.permutation(y)
		a_shuff = log_fit(X,y_shuff,n_iter=n_iter_cv)
		if a_shuff > a_actual:
			times_exceeded += 1
		chance_rates.append(a_shuff)
	return a_actual, np.asarray(chance_rates).mean(), float(times_exceeded)/n_iter_p

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
	for i in range(len(results)):
		accuracies[i] = results[i][0]
		p_vals[i] = results[i][2]
		chance_rates[i] = results[i][1]
	return accuracies,chance_rates,p_vals

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




