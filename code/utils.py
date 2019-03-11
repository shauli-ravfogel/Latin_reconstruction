#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from collections import defaultdict
import random

def create_data_dicts(fname, ablation_mask = [1,1,1,1,1,1]):

	with open(fname, "r", encoding = 'utf-8') as f:
		lines = f.readlines()
		
	
	langs = ["rm", "fr", "it", "sp", "pt", "lt"]
	data = []
	
	for i, line in enumerate(lines):
	
		#print (i, line)
		
		data_dict = {}
		if i == 0: continue
		
		words = line.strip().split("\t")
		for j, w in enumerate(words):
			w = w.split(" ")[0].split(",")[0]
			data_dict[langs[j]] = w.lower() if ablation_mask[j]==1 else "-"
		
		data.append(data_dict)

	return data
	
	
	
def read_letters(fname):

	
	with open(fname, "r", encoding = 'utf-8') as f:
		lines = f.readlines()
		
	letters = set()
	langwletters = defaultdict(set)
	
	for line in lines:
	
		words = line.split(None)
		words_by_lang = line.strip().split("\t")
		# (line)
		
		for i,c in enumerate(["r","f","i","s","p","l"]):
			langwletters[c] = langwletters[c] | (set(words_by_lang[i]))
			
		
		for w in words:
			for char in w.lower():
				letters.add(char)
				
	return letters, langwletters
			
			
def create_dataset(data_dicts):

	dataset = []

	for k, data_dict in enumerate(data_dicts):
		#if k > 100: break
		
		if len(data_dict) == 0: continue
		if not "lt" in data_dict: 
			continue
			print("ERROR", data_dict)
		y = "<" + data_dict['lt'] + ">"
		x = ""
		x = "<*f:"+data_dict['fr']+"*"+"i:"+data_dict['it']+"*"+"s:"+data_dict['sp']+"*"+"p:"+data_dict['pt']+"*"+"r:" + data_dict['rm']+"*"+">"
		dataset.append((x,y))
		
	return dataset

def get_datasets(id, ablation_mask = [1,1,1,1,1,1]):

	
	train_dict, dev_dict, test_dict = create_data_dicts("train{}.txt".format(id), ablation_mask), create_data_dicts("dev{}.txt".format(id), ablation_mask), create_data_dicts("test{}.txt".format(id), ablation_mask)	
	train = create_dataset(train_dict)
	dev = create_dataset(dev_dict)
	test = create_dataset(test_dict)

	return train, dev, test

def create_voc(id):
	letters, langwletters = read_letters("train{}.txt".format(id))
	letters.add(":")
	letters.add("*")
	letters.add(">")
	letters.add("<unk>")
	letters.add("<")

	C2I = {c:i for i,c in enumerate(sorted(letters))}
	I2C = {i:c for i,c in enumerate(sorted(letters))}

	with open("voc/voc.txt", "w") as f:
	
		for c in letters:
		
			f.write(c + "\n")
	for i,l in enumerate(["f","i","s","p","r","l"]):
	
		with open("voc/voc"+l+".txt", "w") as f:
		
			for char in langwletters[l]:
				
				f.write(char + "\n")
					
	return letters, C2I, I2C


