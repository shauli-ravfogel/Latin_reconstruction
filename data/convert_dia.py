#!/usr/bin/env python3
# -*- coding: utf-8 -*-

with open("all-original.txt", "r") as f:

	lines = f.readlines()
	
for i, line in enumerate(lines):

	line = line.replace("ī", "i")
	line = line.replace("ā", "a")
	line =line.replace("ē", "e")
	line = line.replace("ō", "o")
	line = line.replace("ū", "u")
	lines[i] = line
	

with open("all-original-no-diac.txt", "w") as f:

	for line in lines:
	
		f.write(line)
