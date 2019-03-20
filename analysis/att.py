#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pickle
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
import random
import copy
from sklearn.metrics import pairwise_distances

def get_mappings(atts):

	latin2daughter = Counter()
	latinpos2lang = Counter()
	latinchar2lang = Counter()
	pos2pos = Counter()
	latinchar2daughterpos = Counter()
	
	for k,values in atts.items():
	
		pos, latin_char = k
		pos -= 1
		
		for (daughter_lang, daughter_pos, daughter_char) in values:
		
			latin2daughter[(latin_char, daughter_char)] += 1
			latinchar2lang[(latin_char, daughter_lang)] += 1
			latinchar2daughterpos[(latin_char, daughter_pos)] += 1
			
			if pos < 20:
				latinpos2lang[(pos, daughter_lang)] += 1
				pos2pos[(pos, daughter_pos)] += 1
				
	return latin2daughter, latinpos2lang, latinchar2lang, pos2pos, latinchar2daughterpos

def plot(counter):

	def get_keys_and_vals():
	
		keys, vals = set(), set()
		
		for (k,v), count in counter.items():
		
			keys.add(k)
			vals.add(v)
		
		return list(keys), list(vals)
	
	keys, vals = get_keys_and_vals()
	heatmap = np.zeros((len(keys), len(vals)))
	for (k,v), count in counter.items():
	
		heatmap[keys.index(k), vals.index(v)] = count
	
	fig, ax = plt.subplots()
	ax.set_xticks(np.arange(len(vals)))
	ax.set_yticks(np.arange(len(keys)))
	ax.set_xticklabels(vals)
	ax.set_yticklabels(keys)

	plt.imshow(heatmap, cmap='hot', interpolation='nearest')
	plt.colorbar()
	plt.show()

	print(heatmap)
	
	
infile = open("att.pickle",'rb')
atts = pickle.load(infile)
infile.close()

latin2daughter, latinpos2lang, latinchar2lang, pos2pos, latinchar2daughterpos = get_mappings(atts)
plot(latinchar2daughterpos)

		
