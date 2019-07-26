import numpy as np
import scipy.io
import os
import scipy.stats

os.chdir('../Wehbe')

path_to_fmri = '/gpfs/milgram/scratch/chun/hf246/wehbe_fmri/'

for sub in ['1','2','3','4','5','6','7','8']:
	print('Subject ' + sub)
	f = scipy.io.loadmat(path_to_fmri + 'subject_' + sub + '.mat')

	# nonempty_data
	time = f['time']

	empty_inds = []

	for run in range(1,5):
		run_inds = np.where(time[:,1]==run)[0]
		empty_inds.extend(run_inds[0:12]) # 2 TR's accounting for hemodynamic lag
		empty_inds.extend(run_inds[-2:])


	nonempty_data = np.array([f['data'][i] for i in range(len(f['data'])) if i not in empty_inds])
	nonempty_time = np.array([time[i,:] for i in range(time.shape[0]) if i not in empty_inds])

	# normalization
	data_zscored = np.zeros(nonempty_data.shape)
	for run in range(1,5):
		data_zscored[nonempty_time[:,1]==run,:] = scipy.stats.zscore(nonempty_data[nonempty_time[:,1]==run,:], axis=0)

	# remove nas
	nonempty_data = np.nan_to_num(data_zscored)

	# v
	coltocoord = f['meta'][0][0][6] - 1 # 6 is coltocoord
	v = np.zeros((f['meta'][0][0][7].shape) + (1295,)) # 7 is coordtocol

	#populate v
	for t in range(1295):
		for vind in range(coltocoord.shape[0]):
			v[coltocoord[vind][0],coltocoord[vind][1],coltocoord[vind][2],t] = nonempty_data[t,vind]

	# mask
	mask = np.zeros((f['meta'][0][0][7].shape)) # 7 is coordtocol

	#populate mask
	for vind in range(coltocoord.shape[0]):
		mask[coltocoord[vind][0],coltocoord[vind][1],coltocoord[vind][2]] = 1

	# save everything
	outfile = 'subject_%s.npz' % (sub)
	np.savez(outfile, nonempty_data=nonempty_data, v=v, mask=mask, nonempty_time=nonempty_time)