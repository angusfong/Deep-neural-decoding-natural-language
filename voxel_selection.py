import os
import lstm_decoding_utils
import sys
import time
import h5py
import numpy as np
from scipy.io import loadmat
import sklearn
from sklearn import linear_model
import scipy
from scipy import stats
import nibabel as nib
from brainiak.searchlight.searchlight import Searchlight
from sklearn.model_selection import StratifiedShuffleSplit, GridSearchCV, cross_val_score
from sklearn.svm import SVC
from sklearn.utils import shuffle
import rpy2.robjects as robjects
import pandas as pd
import scipy.io

os.chdir('../Wehbe')

n = 1295
sub = sys.argv[1]
layer = sys.argv[2] #layer0, layer1, glove
context = sys.argv[3]
fold = int(sys.argv[4]) #remember to shuffle!

print("Subject %s, Layer %s, Context %s, Fold %d" % (sub,layer,context,fold))

# read fmri data
f = np.load('subject_' + sub + '.npz')
v = f['v']
v = np.array(v)
#v = np.transpose(v, (3,2,1,0))
mask = f['mask']
vdim = np.array(np.shape(v))[[0,1,2]]

# embeddings
if layer == 'layer1':
	path_to_embeddings = '/gpfs/milgram/project/chun/hf246/Language/Wehbe/'
else:
	path_to_embeddings = '/gpfs/milgram/project/chun/kxt3/official/glove/'

y = np.load(path_to_embeddings + 'TR_' + layer + '_' + context + 'embeddings.npy')

'''
#old procedure, by fold
excl_inds = range((fold - 1) * 65, fold * 65)
excl_inds = [ind for ind in excl_inds if ind < n] #last fold has 5 less data pts
incl_inds = [i for i in range(n) if i < min(excl_inds) - 10 or i > max(excl_inds) + 10] #10TR moat
'''

#new procedure, by run
excl_inds = np.where(f['nonempty_time'][:,1] == fold)[0]
incl_inds = [i for i in range(n) if i not in excl_inds] 

v = v[:,:,:,incl_inds]
y = y[incl_inds,:]

# params here
data = v
mask = mask #opened in file above
bcvar = range(1, n+1)
sl_rad=1
max_blk_edge = 5
pool_size = 1

# Start the clock to time searchlight
begin_time = time.time()

# Create the searchlight object
sl = Searchlight(sl_rad=sl_rad,max_blk_edge=max_blk_edge)
#print("Setup searchlight inputs")
#print("Input data shape: " + str(data.shape))
#print("Input mask shape: " + str(mask.shape) + "\n")

# Distribute the information to the searchlights (preparing it to run)
sl.distribute([data], mask)

# Data that is needed for all searchlights is sent to all cores via the sl.broadcast function. In this example, we are sending the labels for classification to all searchlights.
sl.broadcast(bcvar)

#set up kernel
def kernel_fn(data, sl_mask, myrad, bcvar, nfold=20):
	# Pull out the data
	v = data[0]
	v2D = v.reshape(v.shape[0] * v.shape[1] * v.shape[2], v.shape[3], order='F').transpose()
	# Check if the number of voxels is what you expect.
	#print("Searchlight data shape: " + str(v.shape))
	#print("Searchlight data shape after reshaping: " + str(v2D.shape))
	#print("Searchlight mask shape:" + str(sl_mask.shape) + "\n")
	#print("Searchlight mask (note that the center equals 1):\n" + str(sl_mask) + "\n")
	t1 = time.time()
	preds, r = lstm_decoding_utils.cvGetFits(v2D, y, nfold=nfold)
	#print('r:',r)
	t2 = time.time()
	#print('Kernel duration: ' + str(t2 - t1) + "\n\n")
	return r #up to 0.45
	#return r

# execute the searchlight
#print("Begin SearchLight\n")
sl_result = sl.run_searchlight(kernel_fn, pool_size=pool_size)
#print("End SearchLight\n")

end_time = time.time()

# Print outputs
print("Summarize searchlight results for %s_%d_%s" % (sub, fold, layer))
print("Number of searchlights run: " + str(len(sl_result[mask==1])))
print("Performance for each kernel function: " + str(sl_result[mask==1]))
print('Total searchlight duration (including start up time): ' + str(end_time - begin_time))

# Save the results to a .nii file
output_name = ('%s_%s_%srun%d_SL_result.npy' % (sub, layer, context, fold))
sl_result = sl_result.astype('double')  # Convert the output into a precision format that can be used by other applications
sl_result[np.isnan(sl_result)] = 0  # Exchange nans with zero to ensure compatibility with other applications
np.save(output_name, sl_result)

