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

sub = sys.argv[1]
print('Subject'+sub)

path_to_fmri = '/gpfs/milgram/scratch/chun/hf246/wehbe_fmri/'

contexts = ['0s_', '1s_', '2s_', '4s_', '16s_', '1600s_']
n = 1295
nnodes = 1024
layer = sys.argv[2]

if layer == 'layer1':
	path_to_embeddings = '/gpfs/milgram/project/chun/hf246/Language/Wehbe/'
else:
	path_to_embeddings = '/gpfs/milgram/project/chun/kxt3/official/glove/'

# Just have to do the following the first time, binds into one npy file per context
# for context in ['0s_', '1s_', '2s_', '4s_', '16s_', '1600s_']:
#	tmp = np.zeros((n, nnodes))
#	for s in range(1295):
#		tmp[s,:] = np.load('TR' + str(s+1) + "_layer1_" + context + "embeddings.npy")
#	np.save('TR_layer1_' + context + 'embeddings', tmp)

#whole brain decoding
# shuffle
rand_ind = shuffle(range(n),random_state=456345)
#rs = np.zeros((len(contexts),len(subs)))

sub_results = {}
#f = h5py.File('/gpfs/milgram/scratch/chun/hf246/wehbe_fmri/subject_' + sub + '.mat', 'r')
#f = scipy.io.loadmat(path_to_fmri + 'subject_' + sub + '.mat')
f = np.load('subject_'+sub+'.npz')
wb = np.array(f['nonempty_data'])
#wb = wb[rand_ind,:]
for context_ind in range(len(contexts)):
	context = contexts[context_ind]
	print('Context:' + context)
	lstm = np.load(path_to_embeddings + 'TR_' + layer + '_' + context + 'embeddings.npy')
	lstm = lstm[rand_ind,:]
	preds, r = lstm_decoding_utils.cvGetFits(wb, lstm, 20)		
	#now, need to unshuffle preds
	#unshuffled_preds = np.zeros((n,nnodes))
	#for i in range(n):
	#	unshuffled_preds[i,:]=preds[rand_ind.index(i),:]
	np.save('subject' + sub + '_wb_' + layer + '_' + context + 'decoded',preds)
	#rs[context_ind,s] = r

#np.save('wb_rs_context', rs)

'''
#searchlight decoding
nvoxels = 5000 # for searchlight
rand_ind = shuffle(range(n),random_state=456345)
subs = ['1','2','3','4','5','6','7','8']
rs = np.zeros((len(contexts),len(subs)))
for s in range(len(subs)):
	sub = subs[s]
	sub_preds = np.zeros((n, nnodes))
	f = scipy.io.loadmat(path_to_fmri + 'subject_' + sub + '.mat')
	v = np.array(f['v'])
	v = v[:,:,:,rand_ind]
	v2D = v.reshape((v.shape[0]*v.shape[1]*v.shape[2],v.shape[3]), order='F')
	for context_ind in range(len(contexts)):	
		context = contexts[context_ind]	
		lstm = np.load('TR_layer1_' + context + 'embeddings.npy')
		lstm = lstm[rand_ind,:]
		for fold in range(1,21):
			excl_inds = range((fold - 1) * 65, fold * 65)
			excl_inds = [ind for ind in excl_inds if ind < 1295] #last fold has 5 less data pts
			incl_inds = [element for i, element in enumerate(range(n)) if i not in excl_inds]
			f = sub + '_' + layer + '_'  + context + 'fold' + str(fold) + '_SL_result.npy'
			sl = np.load(f)
			sl_reshaped = sl.reshape(sl.shape[0] * sl.shape[1] * sl.shape[2], order='F')
			selected = sl_reshaped.argsort()[-nvoxels:][::-1]
			x=v2D[selected,]
			x=x.transpose()
			xtrain = x[incl_inds,:]
			xtest = x[excl_inds,:]
			ytrain = lstm[incl_inds,:]
			ytest = lstm[excl_inds,:]
			fold_preds = lstm_decoding_utils.fitRidge(xtrain, xtest, ytrain)
			sub_preds[excl_inds,:] = fold_preds
		r = np.max([scipy.stats.pearsonr(fold_preds[:,i], ytest[:,i])[0] for i in range(np.shape(fold_preds)[1])])
		print('Context: ' + context + ': r = ' + str(r))
		rs[context_ind,s] = r
		unshuffled_preds = np.zeros((n,nnodes))
		for i in range(n):
			unshuffled_preds[i,:]=sub_preds[rand_ind.index(i),:]
		np.save('subject' + sub + '_sl' + str(nvoxels) + '_' + context + 'decoded',unshuffled_preds)

np.save('wb_sl5000_context', rs)
'''
