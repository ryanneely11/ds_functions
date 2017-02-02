##PCA.py
##functions to do PCA-type analyses

import numpy as np
import plotting as ptt
from scipy.stats import zscore
from sklearn.decomposition import PCA
from scipy.stats import zscore
import regression as re
import parse_ephys as pe
from numpy import linalg as la
"""
A function to z-score a data matrix
Inputs:
	X, data matrix in the form n_units x (m-conditions x bins)
Returns:
	Xz: z-scored version of X
"""
def zscore_matrix(X):
	Xz = np.zeros(X.shape)
	for i in range(X.shape[0]):
		Xz[i,:] = zscore(X[i,:])
	##get rid of any nan values
	Xz = np.nan_to_num(Xz)
	return Xz

"""
A function that uses probabalistic PCA to decompose a
data matrix, X.
Inputs:
	X: data matrix of shape n-units by t-samples
Outputs:
	X_pca: data matrix of n-components by n_units
	var_explained: array containing the ratio of variance explained for each component 
"""
def ppca(X):
	##init the PCA object
	pca = PCA(svd_solver='full')
	pca.fit(X.T)
	X_pca = pca.components_
	var_explained = pca.explained_variance_ratio_
	return X_pca, var_explained

"""
A function to compute the covariance matrix of a raw data matrix, X (see above)
Inputs:
	-X: raw data matrix; N-units by (t*b)
Returns:
	-C: covariance matrix of the data
"""
def cov_matrix(X,plot=True):
	C = np.dot(X,X.T)/X.shape[1]
	if plot:
		ptt.plot_cov(C)
	return C

"""
A function to compute the "denoising matrix" based on the computed
PCs.
Inputs: 
	-v; array of principal components (eigenvectors) arranged 
	in order of variance explained, and i of shape [n vectors x n values]
	-num_PCs: how many PCs to use in the de-noising matrix
Returns: D; de-noising matrix of size N-units by N-units
"""
def get_denoising_matrix(v,num_PCs):
	D = np.dot(v[0:num_PCs,:].T,v[0:num_PCs])
	return D


"""
a population to compute the "denoised" population
response given a denoising matrix and some raw data
Inputs: 
	-X; raw data matrix
	-D: denoising matrix
Returns:
	-Xpca: denoised version of X
"""
def denoise_X(X,D):
	return np.dot(D,X)

"""
A function to orthoganalize a matrix
Inputs:
	B: matrix to orthoganalize
Returns:
	Q: orthogonalized matrix
"""
def qr_decomp(B):
	q,r = np.linalg.qr(B,mode='full')
	return q

"""
A function to arrange a regression matrix, R,
of the kind returned by fa.full_regression so that
it can be projected onto a denoising matrix.
Inputs:
	R: regression matrix, shape units x regressors x bins
Returns:
	Rb: same matrix reorganized to regressors x bins x units
"""
def permute_regressors(R):
	return np.transpose(R,[1,2,0])

"""
Helper function to separate a concatenated matrix, Xz,
back into a condition-separated matrix, Xc.
Inputs:
	Xz: matrix of units x (conditionsxbins)
	T: length in bins of one condition-averaged response
Returns:
	Xzc: matrix of num_conditions x num_units x bins
"""
def separate_Xz(Xz,T):
    num_conditions = Xz.shape[1]/T 
    Xzc = np.zeros((num_conditions,Xz.shape[0],T))
    for i in range(num_conditions):
        Xzc[i,:,:] = Xz[:,i*T:(i+1)*T]
    return Xzc

"""
A function that uses spectral decomposition to find the
eigenvalues and eigenvectors of a covariance matrix
Inputs:
	-C: covariance matrix of the data
Returns:
	-w: eigenvalues
	-v: eigenvectors (PC's)
"""
def spectral_decomp(C,X=None,plot=True):
	w,v = la.eig(C)
	##organize them in order of most to least variance captured
	idx = np.argsort(w)[::-1]
	w = w[idx]
	v = v[:,idx].T ##rearrange so the first axis indexes the PC vectors
	if plot:
		ptt.plot_eigen(C,X,w,v)
	return w,v

"""
A function to calculate the de-noised neural trajectories
projected onto the regression value axes
Inputs:
	-Xc: condition-averaged reponse vectors for all units, 
		shape conditions x units x bins
	-R: matrix of regression values, shape units x coefficients x time/bins
	-conditions: a list of conditions in Xc, in the same order as in the matrix
	-n_pcs: the number of PCs to keep after dimensionality reduction
Returns:
	-
"""
def value_projections(Xc,R,conditions,n_pcs=12):
	##coefficients; these are assumed based on the regression model used
	coeffs = ['choice','outcome','C x O','Qu','Ql','Qc']
	#the length of a response
	T = Xc.shape[2]
	##get a concatenated version of X
	Xz = pe.concatenate_trials(Xc)
	##zscore this matrix
	Xz = zscore_matrix(Xz)
	##do dimensionality reduction on this data
	pcs,var_explained = ppca(Xz)
	##compute the denoising matrix
	D = get_denoising_matrix(pcs,n_pcs)
	##denoise the regression coeffs
	Rpca = denoise_X(D,permute_regressors(R))
	##get the abs max of the regression vectors across time
	Rmax = re.max_norm(Rpca)
	##now orthoganalize the Rmax matrix with QR decomp
	Q = qr_decomp(Rmax.T)
	##split Xz back into multiple conditions
	Xzc = separate_Xz(Xz,T)
	##do the projections
	results = {} ##dictionary to return
	##define the different condition pairs to use together
	c_pairs = [('upper_lever','lower_lever'),
				('rewarded','unrewarded'),
				('upper_rewarded','lower_rewarded')]
	##define the various axes pairs to use
	ax_pairs = [('choice','outcome'),
				('choice','Qc'),
				('choice','Qu'),
				('choice','Ql'),
				('outcome','Qc'),
				('outcome','Qu'),
				('outcome','Ql'),
				('Qc','Ql'),
				('Qc','Qu')]
	for c_pair in c_pairs:
		##get the spike data for the current two conditions
		data1 = Xzc[conditions.index(c_pair[0]),:,:]
		data2 = Xzc[conditions.index(c_pair[1]),:,:]
		for ax_pair in ax_pairs:
			x_axis = Q[:,coeffs.index(ax_pair[0])]
			y_axis = Q[:,coeffs.index(ax_pair[1])]
			##now do the projection for this set
			data1_x = np.dot(x_axis,data1)
			data1_y = np.dot(y_axis,data1)
			data2_x = np.dot(x_axis,data2)
			data2_y = np.dot(y_axis,data2)
			##now add to the return dictionary
			tag = c_pair[0]+", "+c_pair[1]+" on "+ax_pair[0]+"(x), "+ax_pair[1]+"(y)"
			results[tag] = [data1_x,data1_y,data2_x,data2_y]
	return results

