import os
import numpy as np
import tensorflow as tf

import sys
sys.path.insert(0, '/gpfs/milgram/project/chun/hf246/Language/google_lm/lm_1b')

from data_utils import *
import lm_1b_eval as lmf

path_to_pereira = '/gpfs/milgram/project/chun/hf246/Language/Pereira'
path_to_expt2 = path_to_pereira + '/expt2'
path_to_expt3 = path_to_pereira + '/expt3'

#expt 2
os.chdir(path_to_expt2)

f = open('stimuli_384sentences_dereferencedpronouns.txt?dl=1', 'r')
data = f.readlines() # 384 values in a list

os.chdir(path_to_expt2 + '/lstm_embeddings')

for c in range(96): #96 concepts, ordered alphabetically
	# additionally, it would be good to map the concepts into broader topics
	for i in range(4): #4 sentences per concept

		print 'concept' + str(c+1) + ', sentence' + str(i+1)
		cdir = str(c+1) + '_' + str(i+1)

		os.makedirs(cdir)
		
		s = data[c*4 + i]

		print(s)
		
		print("Layer 0...")
		lmf._DumpSentenceEmbedding(s, cdir, layer=0)

		print("Layer 1...")
		lmf._DumpSentenceEmbedding(s, cdir, layer=1)
		

		




		



