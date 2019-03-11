#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
from scipy.cluster.hierarchy import dendrogram, linkage
import scipy
import matplotlib.pyplot as plt

lang = sys.argv[1]

embedding_filename = "embeddings."+ lang +".txt"
voc_filename = "voc.txt"
language_voc_filename = "voc" + lang + ".txt"


voc = []
voc_lang = []
vecs = []

with open(voc_filename, "r" , encoding = "utf-8") as f:

	voc = f.readlines()
	voc = [x.strip() for x in voc]

with open(language_voc_filename, "r", encoding = "utf-8") as f:

	voc_lang = f.readlines()
	voc_lang = [x.strip() for x in voc_lang]

print (embedding_filename)
with open(embedding_filename, "r", encoding = "utf-8") as f:

		lines = f.readlines()
			
		# collect the embedding of different verbs
	
		embeddings, words, labels = [], [], []
		
		for line in lines:
		
			letter, vector_string = line.strip().split("\t")
			vec = [float(v) for v in vector_string.split(" ")]
			print (letter, letter in voc_lang)
			#if letter in voc_lang:
			
			vecs.append((vec, letter))

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
dendrogram(Z, labels = labels, leaf_font_size = 17, orientation = "right", color_threshold = 0)  
plt.show()
