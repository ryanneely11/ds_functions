##full_analyses
##function to analyze data from all sessions and all animals

import numpy as np
import session_analysis as sa
import file_lists
import os
import h5py
import PCA
import glob

"""
A function to run and compile regression data for ALL sessions
	-epoch_durations: the duration, in seconds, of the epochs (see the list in the function)
	-smooth_method: type of smoothing to use; choose 'bins', 'gauss', or 'none'
	-smooth_width: size of the bins or gaussian kernel in ms
	-z_score: if True, z-scores the array
			$$NOTE$$: this implementatin does not allow binning AND gaussian smoothing.
	-save: if True, saves data at the end
Returns:
	-C: matrix of coefficients, shape units x regressors x bins (all epochs are concatenated)
	-num_sig: matrix with the counts of units showing significant regression
		values at for each regressor at each bin (regressors x bins)
	-mse: mean squared error of the model fit at each bin 
		based on x-validation (size = bins)
	-epoch_idx: the indices of bins corresponding to the different epochs
"""
def full_regression(epoch_durations=[1,0.5,1,1],smooth_method='bins',
	smooth_width=50,z_score=False,save=True):
	##run session regression on all files in the lists
	results_files = []
	for f_behavior,f_ephys in zip(file_lists.behavior_files,file_lists.ephys_files):
		current_file = f_behavior[-11:-5]
		print "Starting on file "+current_file
		out_path = os.path.join(file_lists.save_loc,current_file+".hdf5")
		try: 
			f_out = h5py.File(out_path,'w-')
			##if data does not exist, calculate it and save it
			c,ns,mse,epoch_idx = sa.session_regression(f_behavior,f_ephys,
				epoch_durations=epoch_durations,smooth_method=smooth_method,
				smooth_width=smooth_width,z_score=z_score)
			f_out.create_dataset("coeffs",data=c)
			f_out.create_dataset("num_sig",data=ns)
			f_out.create_dataset("mse",data=mse)
			for key in epoch_idx.keys():
				f_out.create_dataset(key,data=epoch_idx[key])
			f_out.close()
			results_files.append(out_path)
		except IOError:
			results_files.append(out_path)
			print current_file+" exists, moving on..."
	##if all data is saved, you can go back to the files and save the data that you want.
	##data to return
	coeffs = []
	num_sig = []
	mse = []
	num_total_units = 0
	for results_file in results_files:
		epoch_idx = get_epoch_idx_dict(results_file)
		f_in = h5py.File(results_file,'r')
		c = np.asarray(f_in['coeffs'])
		ns =np.asarray(f_in['num_sig'])
		m = np.asarray(f_in['mse'])
		num_units = c.shape[0]
		coeffs.append(c)
		num_sig.append(ns)
		mse.append(m)
		num_total_units+=num_units
	##concatenate all arrays
	coeffs = np.concatenate(coeffs,axis=0)
	num_sig = np.asarray(num_sig).sum(axis=0)
	mse = np.asarray(mse).mean(axis=0)
	if save:
		out_path = os.path.join(file_lists.save_loc,"all_files_regression.hdf5")
		f_out = h5py.File(out_path,'w-')
		f_out.create_dataset("coeffs",data=coeffs)
		f_out.create_dataset("num_sig",data=num_sig)
		f_out.create_dataset("mse",data=mse)
		f_out.create_dataset("num_units",data=np.array([num_total_units]))
		for key in epoch_idx.keys():
			f_out.create_dataset(key,data=epoch_idx[key])
		f_out.close()
	return coeffs,num_sig,mse,epoch_idx,num_total_units

"""
A function to get the condition-averaged responses for all sessions in one giant 
data matrix. The conditions are not concatenated, but just use the function
in parse_ephys to get the matrix for PCA.
Inputs:
	-epoch: (str), name of epoch to use for trial data
	-epoch_duration: duration of epoch to look at (in sec)
	-smooth_method: smooth method to use; see ephys functions
	-smooth_width: smooth width, lol
	-use_unrewarded: bool, whether to include or exclude unrewarded trials.
		if included, these become their own condition.
Returns: 
	X. data matrix of size conditions x units x bins
	order: list of strings defining the order of the conditions in the matrix
"""
def cond_avg_matrix(epoch,epoch_duration,smooth_method='bins',smooth_width=50,
	use_unrewarded='True'):
	##go through all files
	Xc = []
	conditions = []
	for f_behavior,f_ephys in zip(file_lists.behavior_files,file_lists.ephys_files):
		current_file = f_behavior[-11:-5]
		print "Starting on file "+current_file
		x,cond = sa.condition_averaged_responses(f_behavior,f_ephys,epoch_duration=epoch_duration,
			smooth_method=smooth_method,smooth_width=smooth_width,use_unrewarded=use_unrewarded)
		Xc.append(x)
		conditions.append(cond)
	##make sure the conditions are all in the same order
	for c in range(len(conditions)):
		assert conditions[c] == conditions[0]
	return np.concatenate(Xc,axis=1),conditions[0]


"""
A function to run the logistic regression on all files.
No outputs are created, but data is saved to individual HDf5 files.
Inputs: 
	-epoch_durations: the duration, in seconds, of the epochs (see the list in the function)
	-smooth_method: type of smoothing to use; choose 'bins', 'gauss', or 'none'
	-smooth_width: size of the bins or gaussian kernel in ms
	-z_score: if True, z-scores the array
REturns: 
	None,but data is saved.
"""
def full_log_regression(epoch_durations=[1,0.5,1,1],smooth_method='bins',smooth_width=200,
	z_score=True,save=True):
	for f_behavior,f_ephys in zip(file_lists.behavior_files,file_lists.ephys_files):
		current_file = f_behavior[-11:-5]
		print "Starting on file "+current_file
		try:
			results = sa.log_regress_session(f_behavior,f_ephys,epoch_durations=epoch_durations,
				smooth_method=smooth_method,smooth_width=smooth_width,z_score=z_score,save=save)
		except IOError:
			print current_file+" exists, skipping..."
	print "Done!"
	return None


"""
A function to run analyses on logistic regression data. Input is a list of
directories, so it will analyze one or more animals. Function looks for hdf5 files,
so any hdf5 files in the directories should only be regression results files.
Inputs:
	dir_list: list of directories where data is stored.
Returns:
	results: dictionary of results for each directory (animal)
"""
def analyze_log_regressions(dir_list):
	##assume the folder name is the animal name, and that it is two characters
	##also assuming that we are looking at the following conditions:
	conditions = ['block_type','choice','reward']
	for d in dir_list:
		name = d[-11:-9]
		##get the list of files in this directory
		flist = get_file_names(d)
		flist.sort() ##get them in order of training session
		##we are looking at 3 things:
		unit_counts = {} ##number of units encoding each parameter
		mean_pred = {} ##mean prediction strength of each parameter for significant units
		for c in conditions:
			unit_counts[c] = np.zeros(len(flist))
			mean_pred[c] = np.zeros(len(flist))
		multi_units = np.zeros(len(flist)) ##the proportion of significant units with multiple representations
		for n,f in enumerate(flist):
			##parse the results for this file
			cond_idx,cond_ps,multis,n_sig = parse_log_regression(f)
			n_sig = n_sig.size ##just care about how many for now
			##add the data to the master arrays
			for c in conditions:
				unit_counts[c][n] = float(cond_idx[c].size)/n_sig ##here we are computing the proportion, not the total number
				mean_pred[c][n] = cond_ps[c].mean() ##also take the mean here
			multi_units[n] = float(multis.size)/n_sig ##also a proportion
	return unit_counts,mean_pred,multi_units


"""
A function to project neural data from various conditions onto various
axes defined by de-noised regression vectors.
Inputs:
	-f_regression: HDF5 file where the regression data is stored
	-f_data: HDF5 file where the data matrix is stored
	=epoch: str of the epoch name used to compile data for the spike matrix
Returns:
	-results: a dictionary of all the various projections onto different axes
"""
def condition_projections(f_regression,f_data,epoch='choice',n_pcs=12):
	##retrieve the regression data
	f = h5py.File(f_regression,'r')
	R = np.asarray(f['coeffs'])
	idx = np.asarray(f[epoch])
	R = R[:,:,idx]
	f.close()
	##now retrieve the data matrix
	f = h5py.File(f_data,'r')
	Xc = np.asarray(f['data'])
	conditions = list(np.asarray(f['conditions']))
	f.close()
	result = PCA.value_projections(Xc,R,conditions,n_pcs)
	return result

"""
a helper function to get epoch_idx dictionary from a file of regression data
Inputs:
	-f_in: file path of HDF5 file with regression data
Returns:
	-epoch_idx dictionary with the indices corresponding to the various epochs (dict keys)
"""
def get_epoch_idx_dict(f_in):
	epoch_idx = {
	"choice":None,
	"action":None,
	"delay":None,
	"outcome":None
	}
	f = h5py.File(f_in,'r')
	for key in epoch_idx.keys():
		epoch_idx[key] = np.asarray(f[key])
	f.close()
	return epoch_idx

##returns a list of file paths for all hdf5 files in a directory
def get_file_names(directory):
	##get the current dir so you can return to it
	cd = os.getcwd()
	filepaths = []
	os.chdir(directory)
	for f in glob.glob("*.hdf5"):
		filepaths.append(os.path.join(directory,f))
	os.chdir(cd)
	return filepaths