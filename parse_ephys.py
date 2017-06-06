##parse_ephys.py
##functions to parse ephys data from hdf5 files

import h5py
import numpy as np
from scipy.stats import zscore
from scipy.ndimage.filters import gaussian_filter
import os
import glob
try:
	import plxread
except:
	print("Warning: plxread not imported")

"""
A function to return a spike data matrix from
a single recording session, of dims units x bins
Inputs:
	f_in: data file to get the spikes from
	smooth_method: type of smoothing to use; choose 'bins', 'gauss', 'both', or 'none'
	smooth_width: size of the bins or gaussian kernel in ms. If both, input should be a list
		with index 0 being the gaussian width and index 1 being the bin width
	z_score: if True, z-scores the array
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
	units_list = [x for x in list(f) if x.startswith("sig")]
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
	X = smooth_spikes(X,smooth_method,smooth_width)
	if z_score:
		for a in range(X.shape[0]):
			X[a,:] = zscore(X[a,:])
	return X

"""
A function to return an lfp data matrix from
a single recording session, of dims channels x bins
Inputs:
	f_in: data file to get lfp from
Returns:
	X: lfp data matrix in size channles x ms 
"""
def get_lfp_data(f_in):
	f = h5py.File(f_in,'r')
	#get the list of lfp channels in the file
	ad_data = [x for x in f.keys() if x.startswith('AD') and not x.endswith('_ts')]
	ad_ts = [x for x in f.keys() if x.endswith('_ts')]
	##make sure all the AD channels have the same duration
	durs = np.asarray([f[x].size for x in ad_data])
	assert np.all(durs==durs[0])
	L = np.zeros((len(ad_data),durs[0]))
	##add the data to the list of arrays 
	for i in range(len(ad_ts)):
		ts = np.asarray(f[ad_ts[i]])
		raw_ad = np.asarray(f[ad_data[i]]) 
		#convert the ad ts to samples, and integers for indexing
		ts = np.ceil((ts*1000)).astype(int)
		ts = ts-ts[0]
		##account for any gaps caused by pausing the plexon session ****IMPORTANT STEP****	
		## The LFP signal may have fewer points than "duration" if 
		##the session was paused, so we need to account for this
		full_ad = np.zeros(durs[0])
		full_ad[ts] = raw_ad
		L[i,:] = full_ad
	f.close()
	return L


"""
A function to parse a data raw data array (X) into windows of time.
Inputs: 
	X, data array of size units x bins/ms
	windows: array of windows to use to parse the data array; 
		##NEEDS TO BE IN THE SAME UNITS OF TIME AS X### (shape trials x (start,stop))
Returns:
	Xw: data LIST of trials x (units x bins/time). Not converting to an array in order
		to support trials of different lengths
"""
def X_windows(X,windows):
	##add some padding onto the ends of the X array in case some of the windows ovverrun
	##the session. 
	pad = np.zeros((X.shape[0],1000))
	X = np.hstack((pad,X,pad))
	##now add to the timestamps to account for the offset
	windows = windows+1000
	##allocate a list for the return array
	Xw = []
	for t in range(windows.shape[0]): ##go through each window
		idx = np.arange(windows[t,0],windows[t,1],dtype='i') ##the indices of the data for this window
		Xw.append(X[:,idx])
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
	AD_ts = [x for x in list(f) if x.endswith('_ts')]
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
			result = gaussian_filter(array,sigma=sigma,order=0,mode="mirror")
		else:
			print("Check your array input to gaussian filter")
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
	n_bins = int(data.size/bin_width)
	bin_vals = np.zeros(n_bins)
	for i in range(n_bins):
		bin_vals[i] = data[i*bin_width:(i+1)*bin_width].sum()
	return bin_vals

"""
A helper function to do spike smoothing. 
Inputs: 
	X: 2-d data array in the shape units x timebins (assuming 1 ms bins here),
		and value of each bin is a spike count)
	smooth_method: type of smoothing to use; choose 'bins', 'gauss', 'both', or 'none'
	smooth_width: size of the bins or gaussian kernel in ms. If both, input should be a list
		with index 0 being the gaussian width and index 1 being the bin width
	Returns: 
		X: smoothed version of X
"""
def smooth_spikes(X,smooth_method,smooth_width):
	if smooth_method == 'bins':
		Xbins = []
		for a in range(X.shape[0]):
			Xbins.append(bin_spikes(X[a,:],smooth_width))
		X = np.asarray(Xbins)
	elif smooth_method == 'gauss':
		for a in range(X.shape[0]):
			X[a,:] = gauss_convolve(X[a,:],smooth_width)
	elif smooth_method == 'both':
		##first smooth with a kernel 
		for a in range(X.shape[0]):
			X[a,:] = gauss_convolve(X[a,:],smooth_width[0])
		##now bin the data
		Xbins = []
		for a in range(X.shape[0]):
			Xbins.append(bin_spikes(X[a,:],smooth_width[1]))
		X = np.asarray(Xbins)
	elif smooth_method == 'none':
		pass
	else:
		raise KeyError("Unrecognized bin method")
		X = None
	return X

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


"""
this script looks in a directory, takes the plx files and saves a copy as an HDF5 file.

"""
def batch_plx_to_hdf5(directory):
	##first, get a list of the plx files in the directory:
	cd = os.getcwd() ##to return to the cd later
	os.chdir(directory)
	for f in glob.glob("*.plx"):
		cur_file = os.path.join(directory,f)
		print("Saving "+cur_file)
		##create the output file in the same dir
		try:
			out_file = h5py.File(cur_file.strip('plx')+'hdf5','w-')
			##parse the plx file
			data = plxread.import_file(cur_file,AD_channels=range(1,256),import_unsorted=False,
				verbose=False,import_wf=True)
			##save the data
			for k in data.keys():
				out_file.create_dataset(k,data=data[k])
			out_file.close()
		except IOError:
			print(cur_file.strip('plx')+'hdf5 exists; skipping')
	os.chdir(cd)
	print("Done!")
	return None

"""
this script looks in a directory, takes the plx files and saves a copy as an 
HDF5 file.
It is similar to the above function, except it bundles together all of the data
on one electrode, including unsorted timestamps. Therefore it returns 
"neuron hash" on each channel, rather than the sorted single units.
Inputs:
	directory: path to plx file directory (all files here will be converted)
Returns:
	None, but data will be saved as HDF5 files in the same directory.
		all spike data for each channel will be saved as the dataset "sigxxx",
		similar to other formats
**********Note: plxread is crashing in python 3 when you try to import AD_channels,
		so for now I am not importing any*******************
"""
def batch_plx_to_hdf5_hash(directory):
	##first, get a list of the plx files in the directory:
	cd = os.getcwd() ##to return to the cd later
	os.chdir(directory)
	for f in glob.glob("*.plx"):
		cur_file = os.path.join(directory,f)
		print("Saving "+cur_file)
		##create the output file in the same dir
		try:
			out_file = h5py.File(cur_file.strip('.plx')+'_h.hdf5','w-')
			##parse the plx file
			data = plxread.import_file(cur_file,import_unsorted=True,
				verbose=False,import_wf=True,AD_channels=range(1,96))
			##save the non-spike data
			for k in [x for x in data.keys() if not x.startswith('sig')]:
				out_file.create_dataset(k,data=data[k])
			##now concatenate all of the sorted and unsorted timestamps for each channel
			spk_chans = [x for x in data.keys() if x.startswith('sig')]
			while len(spk_chans)>0:
				##pick the first channel in the list and get all of the
				##spikes that were recorded on this channel
				cnum = spk_chans[0][-4:-1]
				chan_list = [y for y in spk_chans if y[-4:-1]==cnum]
				##now concatenate all of these timestamps together
				nhash = np.concatenate([data[x] for x in chan_list])
				##add this data to the hdf5 file
				out_file.create_dataset('sig'+cnum+'i',data=nhash)
				##finally, remove these channels from the spk_chans list
				spk_chans = [x for x in spk_chans if not x in chan_list]
			out_file.close()
		except IOError:
			print(cur_file.strip('plx')+'hdf5 exists; skipping')
	os.chdir(cd)
	print("Done!")
	return None