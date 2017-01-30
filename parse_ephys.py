##parse_ephys.py
##functions to parse ephys data from hdf5 files

import h5py
import numpy as np
from scipy.stats import zscore
from scipy.ndimage.filters import gaussian_filter

"""
A function to return a spike data matrix from
a single recording session, of dims units x bins
Inputs:
	f_in: data file to get the spikes from
	smooth_method: type of smoothing to use; choose 'bins', 'gauss', or 'none'
	smooth_width: size of the bins or gaussian kernel in ms
	z_score: if True, z-scores the array
	$$NOTE$$: this implementatin does not allow binning AND gaussian smoothing.
Returns:
	X: spike data matrix of size units x bins
"""
def get_spike_data(f_in,smooth_method='bins',smooth_width=50,z_score=False):
	##get the duration of this session in ms
	duration = get_session_duration(f_in)
	numBins = int(np.ceil(float(duration)/100)*100)
	##open the file and get the name of the sorted units
	f = h5py.File(f_in,'r')
	##get the names of all the sorted units contained in this file
	units_list = [x for x in f.keys() if x.startswith("sig")]
	##sort this list just in case I want to go back and look at the data unit-by-unit
	units_list.sort()
	#the data to return
	X = np.zeros((len(units_list),numBins))
	##add the data to the list of arrays 
	for n,u in enumerate(units_list):
		##do the binary transform (ie 1 ms bins)
		X[n,:] = pt_times_to_binary(np.asarray(f[u]),duration)
	f.close()
	##now smooth, if requested
	if smooth_method == 'bins':
		Xbins = []
		for a in range(X.shape[0]):
			Xbins.append(bin_spikes(X[a,:],smooth_width))
		X = np.asarray(Xbins)
	elif smooth_method == 'gauss':
		for a in range(X.shape[0]):
			X[a,:] = gauss_convolve(X[a,:],smooth_width)
	elif smooth_method == 'none':
		pass
	else:
		raise KeyError("Unrecognized bin method")
		X = None
	if z_score:
		for a in range(X.shape[0]):
			X[a,:] = zscore(X[a,:])
	return X

"""
A function to parse a data raw data array (X) into windows of time.
Inputs: 
	X, data array of size units x bins/ms
	windows: array of windows to use to parse the data array; 
		##NEEDS TO BE IN THE SAME UNITS OF TIME AS X### (shape trials x (start,stop))
Returns:
	Xw: data array of trials x units x bins/time
"""
def X_windows(X,windows):
	##add some padding onto the end of the X array in case some of the windows ovverrun
	##the session. TODO: might need to also add padding to the start of the array (but that changes the ts)
	pad = np.zeros((X.shape[0],1000))
	X = np.hstack((X,pad))
	##allocate memory for the return array
	Xw = np.zeros((windows.shape[0],X.shape[0],(windows[0,1]-windows[0,0])))
	for t in range(windows.shape[0]): ##go through each window
		idx = np.arange(windows[t,0],windows[t,1]) ##the indices of the data for this window
		Xw[t,:,:] = X[:,idx]
	return Xw

"""
a helper function to convert spike times to a binary array
ie, an array where each bin is a ms, and a 1 indicates a spike 
occurred and a 0 indicates no spike
Inputs:
	-signal: an array of spike times in s(!)
	-duration: length of the recording in ms(!)
Outputs:
	-A duration-length 1-d array as described above
"""
def pt_times_to_binary(signal,duration):
	##convert the spike times to ms
	signal = signal*1000.0
	##get recodring length
	duration = float(duration)
	##set the number of bins as the next multiple of 100 of the recoding duration;
	#this value will be equivalent to the number of milliseconds in 
	#the recording (plus a bit more)
	numBins = int(np.ceil(duration/100)*100)
	##do a little song and dance to ge the spike train times into a binary format
	bTrain = np.histogram(signal,bins=numBins,range=(0,numBins))
	bTrain = bTrain[0].astype(bool).astype(int)
	return bTrain

"""
A helper function to get the duration of a session.
Operates on the principal that the session duration is
equal to the length of the LFP (slow channel, A/D) recordings 
Inputs:
	-file path of an hdf5 file with the ephys data
Outputs:
	-duration of the session in ms(!), as an integer rounded up
"""
def get_session_duration(f_in):
	f = h5py.File(f_in, 'r')
	##get a list of the LFP channel timestamp arrays
	##(more accurate than the len of the value arrs in cases where
	##the recording was paused)
	AD_ts = [x for x in f.keys() if x.endswith('_ts')]
	##They should all be the same, so just get the first one
	sig = AD_ts[0]
	duration = np.ceil(f[sig][-1]*1000.0).astype(int)
	f.close()
	return duration

"""
A function to convolve data with a gaussian kernel of width sigma.
Inputs:
	array: the data array to convolve. Will work for multi-D arrays;
		shape of data should be samples x trials
	sigma: the width of the kernel, in samples
"""
def gauss_convolve(array, sigma):
	##remove singleton dimesions and make sure values are floats
	array = array.squeeze().astype(float)
	##allocate memory for result
	result = np.zeros(array.shape)
	##if the array is 2-D, handle each trial separately
	try:
		for trial in range(array.shape[1]):
			result[:,trial] = gaussian_filter(array[:,trial],sigma=sigma,order=0,
				mode="constant",cval = 0.0)
	##if it's 1-D:
	except IndexError:
		if array.shape[0] == array.size:
			result = gaussian_filter(array,sigma=sigma,order=0,mode="constant",cval = 0.0)
		else:
			print "Check your array input to gaussian filter"
	return result

"""
A helper function to bin arrays already in binary format
Inputs:
	data:1-d binary spike train
	bin_width: with of bins to use
Returns:
	1-d binary spike train with spike counts in each bin
"""
def bin_spikes(data,bin_width):
	bin_vals = []
	idx = 0
	while idx < data.size:
		bin_vals.append(data[idx:idx+bin_width].sum())
		idx += bin_width
	return np.asarray(bin_vals)

"""
A helper function that takes a data array of form
trials x units x bins/time, and concatenates all the trials,
so the result is in the form units x (trialsxbins)
Inputs: 
	X, data array in shape trials x units x bins/time
Returns: 
	Xc, data array in shape units x (trials x bins)
"""
def concatenate_trials(X):
	return np.concatenate(X,axis=1)
