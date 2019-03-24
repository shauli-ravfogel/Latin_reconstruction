#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, os.path
import errno

def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc: # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else: raise

def safe_open_w(path):
    ''' Open "path" for writing, creating any parent directories as needed.
    '''
    mkdir_p(os.path.dirname(path))
    return open(path, 'w')
    
class Collector(object):

	def __init__(self, encoders, voc_file, embedding_filename,  features):
		self.encoders = encoders
		self.voc_file = voc_file
		self.embedding_filename = embedding_filename
		self.features = features
		
	def collect(self, size = 15000):
	   
		print ("collecting embedding...")


		
		for k, lang in enumerate(["s", "i", "f", "p", "r", "l"]):
		
			with open("voc/voc.txt", "r", encoding = "utf-8") as f:
				lines = f.readlines()
			
			f3 = open("voc/voc"+lang+".txt", "r")
			relevant_letters = f3.readlines()
			relevant_letters = [c.strip() for c in relevant_letters]
			f3.close()
			
			vecs = []
		
			for i, line in enumerate(lines[:size]):
				#print i, len(lines)
				#if i % 500 == 0:
					#print "collecting embedding, line {}/{}".format(i, size)
				word = line.strip()
				if (not self.features) or lang == "l":
					vec = self.encoders[0].encode(word, lang).value()
				else:

					vec = self.encoders[0].encode(word, lang)[0].value()
				
				if word in relevant_letters:
					vecs.append((word, vec))
		
			f = safe_open_w("../analysis/embeddings/"+self.embedding_filename+"."+lang+".txt")
			#print "len: ", len(vecs)
			for (w,v) in vecs:
				as_string = " ".join([str(round(float(val),5)) for val in v])
				f.write(w+"\t"+as_string+"\n")
			f.close()
		
		with safe_open_w("../analysis/embeddings/langs_embeddings.txt") as f:
			
			for i, lang in enumerate(self.encoders[0].langs):
			
				as_string = " ".join([str(round(float(val),5)) for val in self.encoders[0].E_lang[i].value()])
				f.write(lang + "\t" + as_string + "\n")
		
		if self.features:
		
			
			with safe_open_w("../analysis/embeddings/plus_embeddings.txt") as f:
			
				for i in range(22):
				
					v = self.encoders[0].E_plus[i].value()
					as_string = " ".join([str(round(float(val),5)) for val in v])
					f.write(self.encoders[0].ft.names[i] + "\t" + as_string + "\n")
			
			with safe_open_w("../analysis/embeddings/minus_embeddings.txt") as f:
			
				for i in range(22):
				
					v = self.encoders[0].E_minus[i].value()
					as_string = " ".join([str(round(float(val),5)) for val in v])
					f.write(self.encoders[0].ft.names[i] + "\t" + as_string + "\n")

			with safe_open_w("../analysis/embeddings/irrelevant_embeddings.txt") as f:
			
				for i in range(22):
				
					v = self.encoders[0].E_not_relevant[i].value()
					as_string = " ".join([str(round(float(val),5)) for val in v])
					f.write(self.encoders[0].ft.names[i] + "\t" + as_string + "\n")				
