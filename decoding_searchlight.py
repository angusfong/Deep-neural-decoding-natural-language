import numpy as np
import sys
sys.path.insert(0, '../google_lm/')
import h5py
import lstm_decoding_utils
import sklearn
from sklearn.utils import shuffle
import scipy.io
import os
import sys

os.chdir('../Wehbe')

nvoxels = 1000 # for searchlight

sub = sys.argv[1]
print('Subject '+sub)

path_to_fmri = '/gpfs/milgram/scratch/chun/hf246/wehbe_fmri/'

contexts = ['0s_', '1s_', '2s_', '4s_', '16s_', '1600s_']
n = 1295
layer = sys.argv[2]
print(layer)

if layer == 'layer1':
	path_to_embeddings = '/gpfs/milgram/project/chun/hf246/Language/Wehbe/'
	nnodes = 1024
else:
	path_to_embeddings = '/gpfs/milgram/project/chun/kxt3/official/glove/'
	nnodes = 300

# load subject fMRI data
f = np.load('subject_' + sub + '.npz')
v = f['v']
v2D = v.reshape((v.shape[0] * v.shape[1] * v.shape[2], v.shape[3])) # checked that reshaping is ok (nvoxel, nTR)

for context_ind, context in enumerate(contexts):
	print('Context: ' + context)	
	embeddings = np.load(path_to_embeddings + 'TR_' + layer + '_' + context + 'embeddings.npy')
	sub_preds = np.zeros((n, nnodes))
	for fold in range(1,5): # 4 folds (runs) total
		print('Fold ' + str(fold))
		# define train and test data
		'''
		#old procedure, by fold
		excl_inds = range((fold - 1) * 65, fold * 65)
		excl_inds = [ind for ind in excl_inds if ind < n] #last fold has 5 less data pts
		incl_inds = [i for i in range(n) if i < min(excl_inds) - 10 or i > max(excl_inds) + 10] #10TR moat
		'''
		#new procedure, by run
		excl_inds = np.where(f['nonempty_time'][:,1] == int(fold))[0]
		incl_inds = [i for i in range(n) if i not in excl_inds] 

		# voxel selection (run before)
		sl_file = sub + '_' + layer + '_'  + context + 'run' + str(fold) + '_SL_result.npy'
		sl = np.load(sl_file)
		sl_reshaped = sl.reshape(sl.shape[0] * sl.shape[1] * sl.shape[2]) # as long as flattened ok, order doesn't matter
		selected = sl_reshaped.argsort()[-nvoxels:][::-1]

		# ridge regression
		x=v2D[selected,].transpose()
		xtrain = x[incl_inds,:]
		xtest = x[excl_inds,:]
		ytrain = embeddings[incl_inds,:]
		ytest = embeddings[excl_inds,:]
		fold_preds = lstm_decoding_utils.fitRidge(xtrain, xtest, ytrain)
		sub_preds[excl_inds,:] = fold_preds
	np.save('subject' + sub + '_sl' + str(nvoxels) + '_' + layer + '_' + context + 'decoded',sub_preds)
