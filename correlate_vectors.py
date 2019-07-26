import numpy as np

os.chdir('../Wehbe')

sub = '1'

corrs = []

for context in ['0s_','1s_','2s_','4s_','16s_','1600s_']:
    x = np.load('subject'+sub+'_wb_'+context+'decoded.npy')
    print(context)
    corr = np.corrcoef(x)
    print(corr.shape)
    print(np.mean(corr))
    corrs.append(corr)

import matplotlib.pyplot as plt
plt.plot(corrs)
