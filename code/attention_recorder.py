from collections import Counter
from collections import defaultdict
import numpy as np
import pickle

class AttentionRecorder(object):

	def __init__(self):

		self.counter = defaultdict(list)
		self.word_transitions = defaultdict(list)
	
	def _get_lang(self, inp, ind_max):

		i = ind_max
		lang = None
		pos = 0

		while True:

			i = i - 1
			pos += 1
			if i <= 0: break

			if inp[i] == ":":

				lang = inp[i - 1]
				break

		return lang, pos 
		
			

			
			
	def collect(self, attention_weights, inp, out, y, pos_in_out):

		inp = list(inp)
		#print(inp)
		#print(attention_weights)
		#print(len(inp), len(attention_weights) )
		assert len(inp) == len(attention_weights)
		ind_max = np.argmax(attention_weights)
		max_att_char = inp[ind_max]
		lang, pos = self._get_lang(inp, ind_max)
		
		self.word_transitions[y].append(lang)

		if lang is not None:

			self.counter[(pos_in_out, out)].append((lang, pos, max_att_char))

	def checkout(self): 


		pickle_out = open("att.pickle5","wb")
		pickle.dump(self.counter, pickle_out)
		pickle_out.close()
		pickle_out = open("att.transitions.pickle","wb")
		pickle.dump(self.word_transitions, pickle_out)
		pickle_out.close()
		self.counter = defaultdict(list)


		
		
