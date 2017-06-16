##linear_regression.py
#functions to run linear regression on task variables

import pandas as pd
import numpy as np
import model_fitting as mf
import statsmodels.api as sm
from sklearn.model_selection import train_test_split


"""
Utilizes the trial_data dataframe to return
an array of regressors for each trial. Regressors will be:
-action choice
-outcome
-Q_lower
-Q_upper
-S_upper_rewarded
-S_lower_rewarded 
Inputs:
	Trial_data dataframe
Returns:

"""
def get_regressors(trial_data):
	##the column names for each regressor
	columns = ['action','outcome','q_lower','q_upper','s_upper','s_lower']
	n_trials = trial_data.shape[0]
	##using the trial data, get the Q-learning model results and the HMM results
	fits = mf.fit_models_from_trial_data(trial_data)
	##let's create a pandas dataframe for all the regressors
	regressors = pd.DataFrame(columns=columns,index=np.arange(n_trials))
	##now add all of the relevant data to the DataFrame
	regressors['action'] = fits['actions']
	regressors['outcome'] = fits['outcomes']
	regressors['q_lower'] = fits['Qvals'][0,:]
	regressors['q_upper'] = fits['Qvals'][1,:]
	regressors['s_lower'] = fits['state_vals'][0,:]
	regressors['s_upper'] = fits['state_vals'][1,:]
	return regressors

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
def lin_fit(X,y,n_iter=5,add_constant=True):
	##get X in the correct shape for sklearn function
	if len(X.shape) == 1:
		X = X.reshape(-1,1)
	if add_constant:
		y = sm.add_constant(y)
	accuracy = np.zeros(n_iter)
	for i in range(n_iter):
		##split the data into train and test sets
		X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.33,random_state=42)
		##now fit to the test data
		model = sm.OLS(y_train,X_train,hasconst=True)
		results = model.fit(method='pinv')
		"""
		Some accuracy testing here
		"""
	##now get the p-value info for this model
	model = sm.OLS(y,X,hasconst=True)
	results = model.fit(method='pinv')
	return results

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
	