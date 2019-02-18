#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# https://invokeit.wordpress.com/frequency-word-lists/

import random
random.seed(0)
from wiktionaryparser import WiktionaryParser
parser = WiktionaryParser()
parser.set_default_language('french')

words = []

lang_id = "it"

with open(lang_id + "-lat-all", "r") as f:

	lines = f.readlines()

existing_words = set()
for l in lines:

	word, lat = l.strip().split("\t")
	existing_words.add(word)
	
	
with open("clean_" + lang_id +  ".txt", "r", encoding = "ISO-8859-1") as f:
#with open("italiano.txt", "r", encoding = "ISO-8859-1") as f:
	lines = f.readlines()

#random.shuffle(lines)
words = [l.strip() for l in lines]
results = []


for k, w in enumerate(words):
	#print(w)
	if k % 1000 == 0:
	
		print ("{}/{}, {}%".format(k, len(words), (1.*k)/len(words)*100))
	try:
		
		if w in existing_words: continue
		
		result = parser.fetch(w)
		#print(result)
		if result:

			ety = result[0]['etymology']
			#print(ety)
		
			if "from latin" in ety.lower() or "based on latin" in ety.lower():
				splitted = ety.lower().split()
				latin_ind = splitted.index("latin")
				if latin_ind + 1 == len(splitted): continue
				latin_word = splitted[latin_ind + 1]
				if latin_word.endswith(",") or latin_word.endswith(".") or latin_word.endswith(";"): latin_word = latin_word[:-1]
				print ((w, latin_word))
				results.append((w, latin_word))
				with open(lang_id + "-lat-all", "a+") as f:
					f.write(w + "\t" + latin_word + "\n")
	except:
		pass
"""
with open("sp-lat", "w") as f:

	for w,l in results:
	
		f.write(w + "\t" + l + "\n")
"""
