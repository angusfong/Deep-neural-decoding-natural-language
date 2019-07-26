import os
import sys
import time
import numpy as np
from scipy.io import loadmat
import scipy
from scipy import stats
import nibabel as nib
from brainiak.searchlight.searchlight import Searchlight
import rpy2.robjects as robjects

os.chdir('../Wehbe')

#sub='1';context='0s_';fold=1; partition=True

# how many stimuli
n = 1295

# load data for sub
sub = sys.argv[1]
path_to_fmri = '/gpfs/milgram/scratch/chun/hf246/wehbe_fmri/'

# LSTM layer
#layer = sys.argv[2] #layer0, layer1, glove
layer = 'glove'

# context
context = sys.argv[2]

print("Subject %s, Layer %s, Context %s" % (sub,layer,context))

# read fmri data
f = np.load('subject_' + sub + '.npz')
v = np.array(f['v'])
mask = np.array(f['mask'])
vdim = np.array(np.shape(v))[[0,1,2]]

# lstm embeddings and model rdm
if layer == 'layer0' or layer == 'layer1':
  path_to_embeddings = '/gpfs/milgram/project/chun/hf246/Language/Wehbe/'
elif layer == 'glove':
  path_to_embeddings = '/gpfs/milgram/project/chun/kxt3/official/glove/'
robjects.r['load'](path_to_embeddings+'TR_'+layer+'_context_embeddings.RData')
y=robjects.r[layer + '_' + context]
y=np.array(y)
model_rdm = np.corrcoef(y)[np.triu_indices(len(y),1)]

# arbritrary mask of one voxel
#small_mask = np.zeros(vdim)
#small_mask[5, 20, 9] = 1

# params here
data = v
mask = mask
#mask = small_mask #opened in file above
bcvar = None
sl_rad=2
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

#set up kernel
def kernel_fn(data, sl_mask, myrad, bcvar):
    # Pull out the data
    v = data[0]
    v2D = v.reshape(v.shape[0] * v.shape[1] * v.shape[2], v.shape[3]).T
    # Check if the number of voxels is what you expect.
    #print("Searchlight data shape: " + str(v.shape))
    #print("Flattened shape: " + str(v2D.shape))
    #print("Searchlight data shape after reshaping: " + str(v2D.shape))
    #print("Searchlight mask shape:" + str(sl_mask.shape) + "\n")
    #print("Searchlight mask (note that the center equals 1):\n" + str(sl_mask) + "\n")
    #t1 = time.time()
    cube_rdm = np.corrcoef(v2D)[np.triu_indices(len(v2D),1)]
    #print('voxel rdm shape: ' + str(cube_rdm.shape))
    #print('model rdm shape:' + str(model_rdm.shape))
    r = scipy.stats.pearsonr(cube_rdm, model_rdm)[0]
    print(r)
    #t2 = time.time()
    #print('Kernel duration: ' + str(t2 - t1) + "\n\n")
    return r
    #return r

# execute the searchlight
#print("Begin SearchLight\n")
sl_result = sl.run_searchlight(kernel_fn, pool_size=pool_size)
#print("End SearchLight\n")

end_time = time.time()

# Print outputs
print("Summarize searchlight results for %s_%s" % (sub, layer))
print("Number of searchlights run: " + str(len(sl_result[mask==1])))
print("Performance for each kernel function: " + str(sl_result[mask==1]))
print('Total searchlight duration (including start up time): ' + str(end_time - begin_time))

# Save the results to a .nii file
output_name = ('%s_%s_%srdm_SL_result.npy' % (sub, layer, context))
sl_result = sl_result.astype('double')  # Convert the output into a precision format that can be used by other applications
sl_result[np.isnan(sl_result)] = 0  # Exchange nans with zero to ensure compatibility with other applications
np.save(output_name, sl_result)

# note: outfile is 1_layer1_0s_rdm_SL_result.npy
