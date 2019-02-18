#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import random 
import sys

"orthography_dataset_no_diac.txt"
"orthography_dataset_no_diac.txt"
def read_dataset(fname = "all-clean-no-diac.txt"):

	with open(fname, "r") as f:

		lines = f.readlines()
	lines = [l.strip() for l in lines]
	return lines

def split_train_dev(lines):

	random.shuffle(lines)
	l = len(lines)
	t = int(l * 0.92)
	train = lines[:t]
	rest = lines[t:]

	l = int(len(rest)/2.5)
	dev, test = rest[:l], rest[l:]

	return train, dev, test

def write_to_file(data, fname):

	with open(fname, "w") as f:

		for d in data:

			f.write(d + "\n")

id = sys.argv[1]
random.seed(id)
train, dev, test = split_train_dev(read_dataset())
write_to_file(train, "train{}.txt".format(id))
write_to_file(dev, "dev{}.txt".format(id))
write_to_file(test, "test{}.txt".format(id))
