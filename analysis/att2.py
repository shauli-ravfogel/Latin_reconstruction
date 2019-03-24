#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pickle
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
import random
import copy
from sklearn.metrics import pairwise_distances
from sklearn.manifold import TSNE
from collections import defaultdict
import itertools

def get_heatmap(include_pos = True):

	langs = ["i","f","r","p","s"]
	langs_pairs = itertools.product(*langs)

	infile = open("att.transitions.pickle",'rb')
	atts = pickle.load(infile)
	infile.close()
	
	max_word_size = 20
	
	heatmap = np.zeros((len(atts), 25)) if not include_pos else  np.zeros((len(atts), 5*5*max_word_size))
	keys = []
	
	for ind, (k,v) in enumerate(atts.items()):
		keys.append(k)
		
		transitions = np.zeros((5,5)) if not include_pos else np.zeros((5,5,max_word_size))
	
		for ind2, (lang, next_lang) in enumerate(zip(v, v[1:])):
	
			i,j = langs.index(lang), langs.index(next_lang)
			if not include_pos:
				transitions[i, j] += 1
			else:
				transitions[i, j, ind2] += 1
	
		heatmap[ind] = transitions.reshape(-1)
		
	return heatmap, keys

def plot(heatmap):
	ax = plt.subplot()
	plt.imshow(heatmap, cmap='hot', interpolation='nearest', aspect = "0.01")
	plt.colorbar()
	plt.show()
	
def do_tsne(heatmap, words):

		proj = TSNE(n_components=2).fit_transform(heatmap)

		fig, ax = plt.subplots()
		
		xs, ys = proj[:,0], proj[:,1]
		xs, ys = list(xs), list(ys)
		
		ax.scatter(xs, ys)
		

		for i, w in enumerate(words):
		
			if i % 15 == 0 and True:
				try:
					#print w
					ax.annotate(w, (xs[i],ys[i]), size = 15)
				except: 
					print(w)
					pass
		plt.show()

def words_by_most_attended_lang():

	infile = open("att.transitions.pickle",'rb')
	atts = pickle.load(infile)
	infile.close()
	words_by_lang = defaultdict(list)
	
	for w, langs in atts.items():
	
		counter = Counter()
		for lang in langs:
		
			counter[lang] += 1
		
		max_lang, count = max(counter.items(), key = lambda tup: tup[1])
		words_by_lang[max_lang].append(w)
	
	return words_by_lang

with open("all-orto-no-diac.txt", "r") as f:

	lines = f.readlines()

w2inp = {}
for line in lines:
	
	langs = line.strip().split("\t")
	w2inp["<" + langs[-1] + ">"] = "\t".join(langs[:-1])
	
#heatmap, words = get_heatmap()
#plot(heatmap)
#do_tsne(heatmap, words)
words_by_most_attended_lang = words_by_most_attended_lang()

for lang,vals in words_by_most_attended_lang.items():

	print("Language: {}".format(lang) + "\n")
	for v in vals:
	
		print (w2inp[v] + "\t" + v)
	print ("===================================================")
