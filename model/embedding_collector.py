#!/usr/bin/env python3
# -*- coding: utf-8 -*-

class Collector(object):

	def __init__(self, encoders, voc_file, embedding_filename):
		self.encoders = encoders
		self.voc_file = voc_file
		self.embedding_filename = embedding_filename
		
	def collect(self, size = 15000):
	   
		print ("collecting embedding...")


		
		for k, lang in enumerate(["s", "i", "f", "p", "r", "l"]):
		
			with open("voc.txt", "r", encoding = "utf-8") as f:
				lines = f.readlines()
			
			f3 = open("voc"+lang+".txt", "r")
			relevant_letters = f3.readlines()
			relevant_letters = [c.strip() for c in relevant_letters]
			f3.close()
			
			vecs = []
		
			for i, line in enumerate(lines[:size]):
				#print i, len(lines)
				#if i % 500 == 0:
					#print "collecting embedding, line {}/{}".format(i, size)
				word = line.strip()
				vec = self.encoders[k].encode(word).value()
				if word in relevant_letters:
					vecs.append((word, vec))
		
			f = open(self.embedding_filename+"."+lang+".txt", "w")
			#print "len: ", len(vecs)
			for (w,v) in vecs:
				as_string = " ".join([str(round(float(val),5)) for val in v])
				f.write(w+"\t"+as_string+"\n")
			f.close()
				
