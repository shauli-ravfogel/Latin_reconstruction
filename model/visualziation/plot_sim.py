#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
import random
import copy
from sklearn.metrics import pairwise_distances

import matplotlib
matplotlib.rc('font', family='Doulos')


def normalize(vecs):
	vecs = copy.deepcopy(vecs)
	
	for i, row in enumerate(vecs):
	
		vecs[i] = row / np.linalg.norm(row)
	
	return vecs
	
def load(fname):

	with open(fname, "r") as f:
	
		lines = f.readlines()	
	
	chars, vecs = [], []
	
	for line in lines:
		
		char, vec = line.strip().split("\t")
		chars.append(char)
		vec = np.array([float(v) for v in vec.split(" ")])
		vecs.append(vec)
	
	return chars, normalize(np.array(vecs))
		
lang1_chars, lang1_vecs = load("embeddings.i.txt")
lang2_chars, lang2_vecs = load("embeddings.l.txt")
			
#distances = pairwise_distances(lang1_chars, metric='euclidean')
distances = lang1_vecs.dot(lang2_vecs.T)
print(len(lang1_chars), len(lang2_chars), distances.shape)

fig, ax = plt.subplots()
ax.set_xticks(np.arange(len(lang2_chars)))
ax.set_yticks(np.arange(len(lang1_chars)))
ax.set_xticklabels(lang2_chars)
ax.set_yticklabels(lang1_chars)

plt.imshow(distances, cmap='hot', interpolation='nearest')
plt.colorbar()
plt.show()



