##log_regression.py
## functions to do logistic regression on neural data

import numpy as np
from sklearn import linear_model
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import make_scorer, cohen_kappa_score
import plotting as ptt
import multiprocessing as mp

"""
A function to perform the regression using the regression class
from scikit-learn. Optimized here for use with a single
independent and dependent variable. Returns a score of how well the 
data predicts the outcome.
Inputs:
	-X: independent variable, here this will probably be spike rates
		over some window
	-y: dependent, catagorical variable. Here this is trial outcome, or
		lever choice, etc.
Returns:
	- 
"""
def run_cv(X,y):
	##get X im the correct shape
	if len(X.shape) == 1:
		X = X.reshape(-1,1)
	lr = linear_model.LogisticRegressionCV(penalty='l2',fit_intercept=True,
		solver='liblinear',max_iter=1000,n_jobs=1)
	##make a scorer object using matthews correlation
	scorer = make_scorer(cohen_kappa_score)
	##make a cross validation object to use for x-validation
	kf = KFold(n_splits=3,shuffle=True)
	score = cross_val_score(lr,X,y,n_jobs=1,cv=kf,scoring=scorer) ##3-fold x-validation using kappa score
	return score.mean()


""" 
a function to run a permutation test on cross-validation data.
The idea is to test how well an independent variable (ie spike rates)
can predict an outcome, ie lever choice. To test this, we will compute the
cross validation score for the model fit with the actual data, then
shuffle the arrays and see how well the data is fit when shuffled. 
We will do this many times and see how frequently the shuffled data predicts better
than the actual data. The idea is that if the actual data is in fact predictive,
it should almost always outperform the shuffled data.
Inputs:
	args: tuple of arguments in the following order:
	 -X: independent variable, here this will probably be spike rates
		over some window
	-y: dependent, catagorical variable. Here this is trial outcome, or
		lever choice, etc.
Returns:
	p_val: percentage of the time that the shuffled data outperformed the
		actual data
"""
def permutation_test(args):
	X = args[0]
	y = args[1]
	repeat = 1000
	if len(X.shape) == 1:
		X = X.reshape(-1,1) ##only needed in the 1-D X case
	lr = linear_model.LogisticRegressionCV(penalty='l2',fit_intercept=True,
		solver='liblinear',max_iter=1000,n_jobs=1) ##set up the model
	##make a scorer object using matthews correlation
	scorer = make_scorer(cohen_kappa_score)
	##make a cross validation object to use for x-validation
	kf = KFold(n_splits=3,shuffle=True) ##VERY important that shuffle == True (not default in sklearn)
	##get the accuary score for the actual data
	f1_actual = cross_val_score(lr,X,y,n_jobs=1,scoring=scorer,cv=kf).mean() ##3-fold x-validation using f1 score
	##now repeat with shuffled data
	times_exceeded = 0 ##numer of times the suffled data predicted better then the actual
	for i in range(repeat):
		y_shuff = np.random.permutation(y)
		f1_test = cross_val_score(lr,X,y_shuff,n_jobs=1,scoring=scorer,cv=kf).mean()
		if f1_test > f1_actual:
			times_exceeded += 1
	return float(times_exceeded)/repeat




"""
A function to run logistic regression on all units in a given Ddata array, for a particular
behavioral outcome (ie lever choice, reward, etc).
It then returns the index of units that show significant predictability for that variable.
Inputs:	
	X: array of unit data over some epoch, dimensions trials x units x bins/time
		y: binary array (1's and zeros) of behavioral outcomes to use in supervised learning
		plot: if True, plots the data for each condition for all units
Returns: 
	-idx: index of units with significant (<.05) predictability for the variable of interest
"""
def regress_array(X,y):
	##make sure that the array is in binary form
	if (y.min() != 0) or (y.max() != 1):
		print "Converting to binary y values"
		y = binary_y(y)
	##setup multiprocessing to do the permutation testing
	arglist = [(X[:,n,:],y) for n in range(X.shape[1])]
	pool = mp.Pool(processes=mp.cpu_count())
	async_result = pool.map_async(permutation_test,arglist)
	pool.close()
	pool.join()
	sig_results = np.asarray(async_result.get())
	##determine the index of units with significant predictablility
	sig_idx = np.where(sig_results<=0.05)
	return sig_idx[0]

"""
A function that determines the strength of the prediction of units in an array using 
a functional metric.
Inputs:	
	X: array of unit data over some epoch, dimensions trials x units x bins/time
	y: binary array (1's and zeros) of behavioral outcomes to use in supervised learning
		plot: if True, plots the data for each condition for all units
Returns: 
	-xvals: strength of fit for each unit, computed using three-fold cross validation
"""
def matrix_pred_strength(X,y):
	xvals = np.zeros((X.shape[1]))
	for u in range(X.shape[1]): ##compute the value for each unit separately
		x = X[:,u,:]
		xvals[u] = run_cv(x,y)
	return xvals


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