import lstm_decoding_utils
import os
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

# how many stimuli
n = 384

# load data for sub
sub = sys.argv[1]
path_to_fmri = '/gpfs/milgram/scratch/chun/hf246/pereira_fmri/'
path_to_sub = path_to_fmri + sub

# LSTM layer (1 or 2)
layer = int(sys.argv[2])

# fold (if given), to partition data, otherwise full
partition = len(sys.argv) == 4
if partition:
	fold = int(sys.argv[3])

print("Subject %s, Layer %d, Fold %d" % (sub,layer,fold))

# read fmri data
f = h5py.File(path_to_sub + '/examples_mask.mat', 'r')

v = f['examplesVolume']
v = np.array(v)
v = np.transpose(v, (3,2,1,0))

g3D = f['gordonin3D']
g3D = np.array(g3D)
g3D = np.transpose(g3D)

f.close()

vdim = np.array(np.shape(v))[[0,1,2]]

# lstm embeddings
if layer == 0:
	y=np.load('/gpfs/milgram/project/chun/hf246/Language/Pereira/expt2/lstm_embeddings/layer0embeddings_tokenized.npy')
elif layer == 1:
	y=np.load('/gpfs/milgram/project/chun/hf246/Language/Pereira/expt2/lstm_embeddings/layer1embeddings_tokenized.npy')

# shuffle
rand_ind = shuffle(range(384),random_state=1)
y = y[rand_ind,:]
v = v[:,:,:,rand_ind]

# partition x and y, as required
if partition:
	excl_inds = range((fold - 1) * 16, fold * 16)
	incl_inds = [element for i, element in enumerate(range(n)) if i not in excl_inds]
	v = v[:,:,:,incl_inds]
	y = y[incl_inds,:]

# arbritrary mask of one voxel
# small_mask = np.zeros(vdim)
# small_mask[80, 54, 9] = 1

# params here
data = v
mask = g3D
bcvar = range(1, 385)
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
def kernel_fn(data, sl_mask, myrad, bcvar, nfold=23):
	# Pull out the data
	v = data[0]
	v2D = v.reshape(v.shape[0] * v.shape[1] * v.shape[2], v.shape[3], order='F').transpose()
	# Check if the number of voxels is what you expect.
	# print("Searchlight data shape: " + str(data[0].shape))
	# print("Searchlight data shape after reshaping: " + str(data2D.shape))
	# print("Searchlight mask shape:" + str(sl_mask.shape) + "\n")
	# print("Searchlight mask (note that the center equals 1):\n" + str(sl_mask) + "\n")
	t1 = time.time()
	preds, rs = lstm_decoding_utils.cvGetFits(v2D, y, nfold=nfold)
	t2 = time.time()
	# print('Kernel duration: ' + str(t2 - t1) + "\n\n")
	return np.mean(rs)

# execute the searchlight
#print("Begin SearchLight\n")
sl_result = sl.run_searchlight(kernel_fn, pool_size=pool_size)
#print("End SearchLight\n")

end_time = time.time()

# Print outputs
print("Summarize searchlight results for %s_%d_%d" % (sub, fold, layer))
print("Number of searchlights run: " + str(len(sl_result[mask==1])))
print("Performance for each kernel function: " + str(sl_result[mask==1]))
print('Total searchlight duration (including start up time): ' + str(end_time - begin_time))

# Save the results to a .nii file
output_name = ('%s_fold%d_layer%dSL_result.npy' % (sub, fold, layer))
sl_result = sl_result.astype('double')  # Convert the output into a precision format that can be used by other applications
sl_result[np.isnan(sl_result)] = 0  # Exchange nans with zero to ensure compatibility with other applications
np.save(output_name, sl_result)

