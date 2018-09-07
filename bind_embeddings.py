import os
import sys

import numpy as np
from six.moves import xrange
import tensorflow as tf

# if _DumpSentenceEmbedding still outputs a file... that has changed
#path_to_embeddings = '/gpfs/milgram/project/chun/hf246/Language/Pereira/expt2/lstm_embeddings'
#
#os.chdir(path_to_embeddings)
#
#sentences = []
#
#for c in range(96):
#	for i in range(4):
#		sentences.append(str(c+1) + '_' + str(i+1) + '/')
#
#layer0_embeddings = np.zeros((384, 1024))
#layer1_embeddings = np.zeros((384, 1024))
#
#for s in range(len(sentences)):
#	layer0 = np.load(sentences[s] + 'lstm0_emb_step_' 
#		+ str(int(len(os.listdir(sentences[s]))/2 - 1)) + '.npy')
#	layer1 = np.load(sentences[s] + 'lstm1_emb_step_' 
#		+ str(int(len(os.listdir(sentences[s]))/2 - 1)) + '.npy')
#	layer0_embeddings[s,:]=layer0
#	layer1_embeddings[s,:]=layer1
#
#np.save('layer0embeddings', layer0_embeddings)
#np.save('layer1embeddings', layer1_embeddings)


# _DumpSentenceEmbedding now returns embeddings (NOW TOKENIZED INPUTS TO LM_1B)

input = 'Wehbe' # or 'Wehbe' or 'Pereira'

os.chdir('/gpfs/milgram/project/chun/hf246/Language/google_lm/lm_1b')
import lm_1b_eval

if input=='Pereira':
	file_sentences = '/gpfs/milgram/project/chun/hf246/Language/Pereira/expt2/stimuli_384sentences_dereferencedpronouns.txt?dl=1'
elif input == 'Wehbe':
	file_sentences = '/gpfs/milgram/project/chun/hf246/Language/Wehbe/HP_sentences.txt'

f = open(file_sentences, 'r')
sentences = f.read().splitlines()
f.close()

layer0_embeddings = np.zeros((len(sentences), 1024))
layer1_embeddings = np.zeros((len(sentences), 1024))
for i in range(len(sentences)):
	layer0_embeddings[i,:]=lm_1b_eval._DumpSentenceEmbedding(sentences[i],layer=0)
	layer1_embeddings[i,:]=lm_1b_eval._DumpSentenceEmbedding(sentences[i],layer=1)

if input=='Pereira':
	np.save('/gpfs/milgram/project/chun/hf246/Language/Pereira/expt2/lstm_embeddings/layer0embeddings_tokenized', layer0_embeddings)
	np.save('/gpfs/milgram/project/chun/hf246/Language/Pereira/expt2/lstm_embeddings/layer1embeddings_tokenized', layer1_embeddings)
elif input=='Wehbe':
	np.save('/gpfs/milgram/project/chun/hf246/Language/Wehbe/sentence_layer0embeddings', layer0_embeddings)
	np.save('/gpfs/milgram/project/chun/hf246/Language/Wehbe/sentence_layer1embeddings', layer1_embeddings)	

