
import sklearn
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
import random
import sys


def is_ascii(s):
    return all(ord(c) < 128 for c in s)



def visaulize(embedding_filename, tokens = False):
	
		"""
		plot the TSNE projection of the embedded vectors of the verbs in the training set.

		embedding: the trained embedding

		word2key: word, key dictionary

		verbs: a set of tuples (present_tense_verb, the verb_pos)

		"""

		with open(embedding_filename, "r") as f:
			lines = f.readlines()
			
		# collect the embedding of different verbs
	
		embeddings, words, labels = [], [], []
		
		for line in lines[:1500]:
			line = line.strip().split("\t")
			if len(line) < 2 and tokens:
				continue
				
			if tokens:
				word, vector_string = line
				words.append(word)
			else:
				word, vector_string = line
				words.append(word)
			vec = vector_string.split(" ") 
			if vec[0] == "0.000": continue
			vec = [float(v) for v in vec]
			embeddings.append(vec)
			
			
		embeddings = np.array(embeddings)


		# calculate TSNE projection & plot

		print ("calculating projection...")
		
		proj = TSNE(n_components=2).fit_transform(embeddings)

		fig, ax = plt.subplots()
		
		xs, ys = proj[:,0], proj[:,1]
		xs, ys = list(xs), list(ys)
		
		ax.scatter(xs, ys)
		
		

		for i, w in enumerate(words):
			if i % 15 == 0 and True:
				try:
					#print w
					ax.annotate(w, (xs[i],ys[i]), size = 14)
				except: 
					print(w)
					pass
		# plt.title("t-SNE projection of learned embeddings for suffixed words", fontsize=25)
		#plt.legend(plots, label_colors, scatterpoints=1, title="Suffix", fontsize=22)
		plt.show()


if __name__ == '__main__':
		visaulize("semantic_rep-cyclic.txt")
		#visaulize("states2.txt", tokens = False)

