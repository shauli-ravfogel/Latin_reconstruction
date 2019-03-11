#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
from scipy.cluster.hierarchy import dendrogram, linkage
import scipy
import matplotlib.pyplot as plt


voc = []
voc_lang = []
vecs = []

phonemes = ["o", "u", "e", "i", "a"]

for lang in ["sp", "it", "fr", "ro", "pt", "la"]:
#for lang in ["i", "f", "s"]:
	embedding_filename = "embeddings."+ lang[0] +".txt"
	
	with open(embedding_filename, "r", encoding = "utf-8") as f:

		lines = f.readlines()
			
		# collect the embedding of different verbs
	
		embeddings, words, labels = [], [], []
		
		for line in lines:
		
			letter, vector_string = line.strip().split("\t")
			if letter in phonemes:
				vec = [float(v) for v in vector_string.split(" ")]
				vecs.append((vec, lang+"-"+letter))

vecs, labels = zip(*vecs)
#labels = [l.encode('utf-8') for l in labels]
print (len(vecs), len(labels))
print (labels)
Z = linkage(vecs, optimal_ordering  = True, method = "average")
order = scipy.cluster.hierarchy.leaves_list(Z)
print (order)
sorted_labels = sorted(labels, key = lambda l: order[labels.index(l)])
print (sorted_labels)
plt.figure()
plt.title("-".join(phonemes))
#plt.figure()
dendrogram(Z, labels = labels, leaf_font_size = 15, orientation = "right", color_threshold = 0)  
plt.show()
