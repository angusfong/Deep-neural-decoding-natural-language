import os
import sys
import numpy as np
from six.moves import xrange
import tensorflow as tf

os.chdir('../Wehbe')

sentences = False # either sentences, or TR

input = 'Wehbe' # or 'Wehbe' or 'Pereira'
# this script now gets embeddings one sentence at a time (parallel across jobs)
s = int(sys.argv[1])
print('Sentence ' + str(s))
tmp = np.zeros([1024,])
# context settings
# '' for none, 
# 'docwise_' for everything up to, including, the sentence
# 'paragraph_' for paragraph up to, and including
# 'ns_' for n sentences up to
# 'nw_' for n words up to
context = sys.argv[2]

if (len(sys.argv) == 4):
	n_context = int(sys.argv[3])

print('Context: ' + context)
print('n_context: ' + str(n_context))

os.chdir('/gpfs/milgram/project/chun/hf246/Language/google_lm/lm_1b')
import lm_1b_eval

if input=='Pereira':
	file_sentences = '/gpfs/milgram/project/chun/hf246/Language/Pereira/expt2/stimuli_384sentences_dereferencedpronouns.txt?dl=1'
elif input == 'Wehbe':
	if sentences:
		type = 'sentence'
		filenam = '/gpfs/milgram/project/chun/hf246/Language/Wehbe/HP_sentences_lstm.txt'
	else:
		type = 'TR'
		filenam = '/gpfs/milgram/project/chun/hf246/Language/Wehbe/wordsTR.txt'

f = open(filenam, 'r')
sentences = f.read().splitlines()
f.close()

layer0_embeddings = np.zeros((1024,))
layer1_embeddings = np.zeros((1024,))

if context=='':
	print('LSTM input: ' + sentences[s-1])
	print('Sentence ' + str(s) + ': Layer 0')
	layer0_embeddings=lm_1b_eval._DumpSentenceEmbedding(sentences[s-1],layer=0)
	print('Sentence ' + str(s) + ': Layer 1')
	layer1_embeddings=lm_1b_eval._DumpSentenceEmbedding(sentences[s-1],layer=1)
elif context=='docwise_':
	seq=''.join([sentences[i] for i in range(s)])
	print('Sentence ' + str(s) + ': Layer 0')
	layer0_embeddings=lm_1b_eval._DumpSentenceEmbedding(seq,layer=0)
	print('Sentence ' + str(s) + ': Layer 1')
	layer1_embeddings=lm_1b_eval._DumpSentenceEmbedding(seq,layer=1)
elif context=='paragraph_':
	file_sentences = '/gpfs/milgram/project/chun/hf246/Language/Wehbe/HP_sentences.txt'
	f = open(file_sentences, 'r')
	sentences = f.read().splitlines()
	f.close()
	#...
elif context=='w_':
	all_seq = ''.join([sentences[i] for i in range(s-1)])
	all_seq_arr = all_seq.split(' ')
	context_seq = ' '.join([all_seq_arr[i] for i in range(-np.min([n_context,len(all_seq_arr)]),0,1)])
	seq = ' '.join([context_seq, sentences[s-1]])

	print('LSTM input: ' + seq)

	print('Sentence ' + str(s) + ': Layer 0')
	layer0_embeddings=lm_1b_eval._DumpSentenceEmbedding(seq,layer=0)
	print('Sentence ' + str(s) + ': Layer 1')
	layer1_embeddings=lm_1b_eval._DumpSentenceEmbedding(seq,layer=1)

	context = str(n_context) + context
elif context=='s_':
	all_seq = [sentences[i] for i in range(s-1)]
	context_seq = ' '.join([all_seq[i] for i in range(-np.min([n_context,len(all_seq)]),0,1)])
	seq = ' '.join([context_seq, sentences[s-1]])

	print('LSTM input: ' + seq)

	print('Sentence ' + str(s) + ': Layer 0')
	layer0_embeddings=lm_1b_eval._DumpSentenceEmbedding(seq,layer=0)
	print('Sentence ' + str(s) + ': Layer 1')
	layer1_embeddings=lm_1b_eval._DumpSentenceEmbedding(seq,layer=1)

	context = str(n_context) + context

if input=='Pereira':
	np.save('/gpfs/milgram/project/chun/hf246/Language/Pereira/expt2/lstm_embeddings/layer0embeddings_tokenized', layer0_embeddings)
	np.save('/gpfs/milgram/project/chun/hf246/Language/Pereira/expt2/lstm_embeddings/layer1embeddings_tokenized', layer1_embeddings)
elif input=='Wehbe':
	np.save('/gpfs/milgram/project/chun/hf246/Language/Wehbe/' + type + str(s) + '_layer0_' + context + 'embeddings', layer0_embeddings)
	np.save('/gpfs/milgram/project/chun/hf246/Language/Wehbe/' + type + str(s) + '_layer1_' + context + 'embeddings', layer1_embeddings)

print('Saving to: /gpfs/milgram/project/chun/hf246/Language/Wehbe/' + type + str(s) + '_layer1_' + context + 'embeddings.npy')	

