import numpy as np
from scipy.io import loadmat
import h5py
import sklearn
from sklearn import linear_model
import scipy
from scipy import stats
import nibabel as nib
#from brainiak.searchlight.searchlight import Searchlight
from sklearn.model_selection import StratifiedShuffleSplit, GridSearchCV, cross_val_score
from sklearn.svm import SVC
from scipy.spatial.distance import euclidean

def fitRidge(Xtrain, Xtest, ytrain, alpha=1.0): # 1.0 if unspecified
	rr = sklearn.linear_model.Ridge(alpha=alpha)
	rr.fit(Xtrain, ytrain)
	return rr.predict(Xtest) # returns predicted ytest

def cvGetFits(X, y, nfold, algo="ridge", alpha=1.0):
# returns predictions, and average correlation
	if algo =="ridge":
		m = sklearn.linear_model.Ridge(alpha=alpha)
	#allow other options?
	else:
		m = sklearn.linear_model.LinearRegression()
	preds = sklearn.model_selection.cross_val_predict(m, X, y, cv=nfold) # can be list or mat
	if len(np.shape(preds)) == 1: # scalar pred
		r = scipy.stats.pearsonr(preds, y)[0]
	else:
		# max across dimensions
		r = np.max([scipy.stats.pearsonr(preds[:,i], y[:,i])[0] for i in range(preds.shape[1])])

        # mean across TRs
        # r = np.mean([scipy.stats.pearsonr(preds[i,:], y[i,:])[0] for i in range(preds.shape[0])])
	return preds, r

def searchlight(sub='P01'):
	pass

def cvLSTMDecoding_full(opt, nfold=24, algo="ridge", alpha=1.0, sub='P01', layer=0, sl_result=None):
# returns predictions, and average correlation
# for single subject; use whole-brain, grey matter mask, or some mask
	# all subs = 'P01', 'M02', 'M04', 'M07', 'M08', 'M09', 'M14', 'M15'
	#y
	y0=np.load('/gpfs/milgram/project/chun/hf246/Language/Pereira/expt2/lstm_embeddings/layer0embeddings.npy')
	y1=np.load('/gpfs/milgram/project/chun/hf246/Language/Pereira/expt2/lstm_embeddings/layer1embeddings.npy')
	#x
	# opt includes wb, gm, sl
	if opt=='gm':
		f = h5py.File('/gpfs/milgram/scratch/chun/hf246/pereira_fmri/' + sub + '/examples_mask.mat', 'r')
		g = np.array(f['examplesGordon'])
		g = np.transpose(g)
		x=g
	elif opt=='wb':
		f = h5py.File('/gpfs/milgram/scratch/chun/hf246/pereira_fmri/' + sub + '/examples_mask.mat', 'r')
		g = np.array(f['examplesGordon'])
		g = np.transpose(g)
		x=g
	elif opt=='sl': # contaminated searchlight
		f = h5py.File('/gpfs/milgram/scratch/chun/hf246/pereira_fmri/' + sub + '/examples_mask.mat', 'r')
		v = np.array(f['examplesVolume'])
		v = np.transpose(v, (0,3,2,1))
		v = v.reshape(v.shape[0], v.shape[1] * v.shape[2] * v.shape[3])
		x=v[:,sl_result]
	# lstm layer 0 or 1
	if layer==0:
		return cvGetFits(x, y0, nfold)
	else:
		return cvGetFits(x, y1, nfold)

def cvLSTMDecoding_nested(opt, nfold=23, algo="ridge", alpha=1.0, sub='P01'):
# feed in within training set: for searchlight or param optim purposes? may not be required if built into searchlight
# for single subject; use whole-brain, grey matter mask, or some mask
	# all subs = 'P01', 'M02', 'M04', 'M07', 'M08', 'M09', 'M14', 'M15'
	pass



