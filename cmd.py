#python matrices

import numpy as np
import os
from scipy.io import loadmat
import scipy
from scipy import stats
from sklearn import linear_model
import pickle
import lstm_decoding_utils
import h5py
import time
from sklearn.utils import shuffle

pred_wb_0, r_wb_0 = lstm_decoding_utils.cvLSTMDecoding_full('wb', nfold=24, algo="ridge", alpha=1.0, sub='M02', layer=0)
pred_gm_0, r_gm_0 = lstm_decoding_utils.cvLSTMDecoding_full('gm', nfold=24, algo="ridge", alpha=1.0, sub='M02', layer=0)
pred_wb_1, r_wb_1 = lstm_decoding_utils.cvLSTMDecoding_full('wb', nfold=24, algo="ridge", alpha=1.0, sub='M02', layer=1)
pred_gm_1, r_gm_1 = lstm_decoding_utils.cvLSTMDecoding_full('gm', nfold=24, algo="ridge", alpha=1.0, sub='M02', layer=1)

# to write, do this; also for future searchlight
with open('expt2_pred_embeddings.pckl', 'wb') as f:  # Python 3: open(..., 'wb')
    pickle.dump([pred_wb_0, r_wb_0, pred_gm_0, r_gm_0, pred_wb_1, r_wb_1, pred_gm_1, r_gm_1], f)

# to retrieve, do this
with open('expt2_pred_embeddings.pckl', 'rb') as f:
	pred_wb_0, r_wb_0, pred_gm_0, r_gm_0, pred_wb_1, r_wb_1, pred_gm_1, r_gm_1 = pickle.load(f, encoding='latin1')


# computing the rank accuracy

# load the true embeddings
y0=np.load('/gpfs/milgram/project/chun/hf246/Language/Pereira/expt2/lstm_embeddings/layer0embeddings_tokenized.npy')
y1=np.load('/gpfs/milgram/project/chun/hf246/Language/Pereira/expt2/lstm_embeddings/layer1embeddings_tokenized.npy')

rand_ind = shuffle(range(384),random_state=1)
y0=y0[rand_ind,:]
y1=y1[rand_ind,:]
# for j in num decoded:
	# corrs = [pearsonr(it, i) for i in true embeddings]
	# rank = argsort(argsort(corrs))[j]

for pred in [pred_wb_0, pred_wb_1, pred_gm_0, pred_gm_1]:
	n = np.shape(pred)[0]
	ranks = []
	for dec in range(n):
		corrs = [scipy.stats.pearsonr(pred[dec], y0[i,])[0] for i in range(n)]
		rank = np.argsort(np.argsort(corrs))[dec]
		rank = n - rank
		ranks.append(rank)
	score = 1 - ( (np.mean(ranks) - 1) / (n - 1) )
	score

pred_wb_0_score = '0.6773417101827677'
pred_wb_1_score = '0.4955124020887729'
pred_gm_0_score = '0.6907841057441253' # pretty good
pred_gm_1_score = '0.49237108355091386'

# ran searchlight on FULL SUB DATA (not partitioned), this is to subset the voxels
sub = 'M07'
m = np.load(sub + '_layer0SL_result.npy')
m_reshaped = m.reshape(m.shape[0] * m.shape[1] * m.shape[2])
selected = m_reshaped.argsort()[-5000:][::-1]
# the following is train-test contamination. just to see the upper bound of performance after searchlight.
# please do not publish this absolutely ridiculous cheating bullshit, which probably sucks anyway.
pred_sl_0, r_sl_0 = lstm_decoding_utils.cvLSTMDecoding_full('sl', nfold=24, algo="ridge", alpha=1.0, 
	sub=sub, layer=0, sl_result=selected)

pred = pred_sl_0
n = np.shape(pred)[0]
ranks = []
for dec in range(n):
	corrs = [scipy.stats.pearsonr(pred[dec], y0[i,])[0] for i in range(n)]
	rank = np.argsort(np.argsort(corrs))[dec]
	rank = n - rank
	ranks.append(rank)

score = 1 - ( (np.mean(ranks) - 1) / (n - 1) )
score

m02score = 0.72461107484769371
m04score = 0.65300669060052219
m07score = 0.69931054177545693

# ok, so this is better than gm, but still worse than the universal linguistic decoder. 
# continue with a reserved optimism. 
# do searchlight for folds.
# learn lambda for ridge.
# external validate to expt3.

# lambda
subs = ['P01', 'M02', 'M04', 'M07', 'M08', 'M09', 'M14', 'M15']
n = 384
nnodes = 1024
ntrain = 368
nvoxels = 5000 # for searchlight

results = {}
for s in range(len(subs)):
	sub_results = {}
	sub = subs[s]
	#x
	# searchlight method
	f = h5py.File('/gpfs/milgram/scratch/chun/hf246/pereira_fmri/' + sub + '/examples_mask.mat', 'r')
	v = np.array(f['examplesVolume'])
	v = v.transpose((3,2,1,0))
	v2D = v.reshape(v.shape[0] * v.shape[1] * v.shape[2], v.shape[3], order='F').transpose()
	for layer in 0,1:
		if layer == 0:
			y=np.load('/gpfs/milgram/project/chun/hf246/Language/Pereira/expt2/lstm_embeddings/layer0embeddings_tokenized.npy')
		elif layer == 1:
			y=np.load('/gpfs/milgram/project/chun/hf246/Language/Pereira/expt2/lstm_embeddings/layer1embeddings_tokenized.npy')
		# shuffle
		rand_ind = shuffle(range(384),random_state=1)
		v2D = v2D[rand_ind,:]
		y=y[rand_ind,:]		
		for use_sl in True, False:
			key = 'lstm' + str(layer)
			if use_sl:
				key = key + '_sl'
			else:
				key = key + '_gm'
			print(key)
			if not use_sl:
				g = np.array(f['examplesGordon']) # for now only
				g = g[:,rand_ind]
				x = np.transpose(np.array(g)) # for now only
			sub_preds = np.zeros((n,nnodes))
			for fold in range(1,25): # fold is 1 up
				if use_sl:
					sl = np.load('%s_fold%d_layer%dSL_result.npy' % (sub, fold, layer))
					sl_reshaped = sl.reshape(sl.shape[0] * sl.shape[1] * sl.shape[2], order='F')
					selected = sl_reshaped.argsort()[-nvoxels:][::-1]
					x=v2D[:,selected]
				excl_inds = range((fold - 1) * 16, fold * 16)
				incl_inds = [element for i, element in enumerate(range(n)) if i not in excl_inds]
				xtrain = x[incl_inds,:]
				xtest = x[excl_inds,:]
				ytrain = y[incl_inds,:]
				ytest = y[excl_inds,:]
				# cv
				alphas = (0.1,1,10,100)
				m = linear_model.RidgeCV(alphas, cv=None,store_cv_values=True)
				begin_time = time.time()
				m.fit(x,y)
				end_time = time.time()
				selected_alpha = m.alpha_
				print('fold ' + str(fold) + ' alpha = ' + str(selected_alpha))
				# verify alpha selection
				#alphas_mses = m.cv_values_
				#[np.mean(alphas_mses[:,:,i]) for i in range(len(alphas))]
				#alphas_preds = m.cv_values_
				#alphas_results = []
				#for a in range(len(alphas)):
				#	a_preds = alphas_preds[:,:,a]
				#	a_rs = [scipy.stats.pearsonr(a_preds[i,], y[i,])[0] for i in range(ntrain)]
				#	alphas_results.append(np.mean(a_rs))
				#
				#alphas_results
				#alphas[alphas_results.index(max(alphas_results))]
				fold_preds = lstm_decoding_utils.fitRidge(xtrain, xtest, ytrain, alpha=selected_alpha)
				sub_preds[excl_inds,:] = fold_preds
				
			# score metric	
			ranks = []
			for dec in range(n):
				corrs = [scipy.stats.pearsonr(sub_preds[dec], y[i,])[0] for i in range(n)]
				rank = np.argsort(np.argsort(corrs))[dec]
				rank = n - rank
				ranks.append(rank)
			score = 1 - ( (np.mean(ranks) - 1) / (n - 1) )
			score
			sub_results[key] = score
	results[sub] = sub_results
	print(sub_results)
	print(results)

np.save('expt2_lstm_decoding_results', results) #0gm,0sl,1gm,1sl

# sanity check for 3D indexing, selected is the indices for the 44276 in gordon
# v[gordonindices] should == gordonexamples
#g3D = np.array(f['gordonin3D'])
#gv = g3D.reshape(88*85*128)
#gv = np.array([i for i, x in enumerate(gv) if x])
#np.shape(np.intersect1d(gv,selected))

#g1 = v2D[:,gv]
#g2 = np.transpose(np.array(f['examplesGordon']))

#for x in range(88):
#	for y in range(128):
#		for z in range(85):
#			if v[1,x,y,z]!= 0:
#				print('%d,%d,%d' % (x,y,z))

# null distribution of corr values
