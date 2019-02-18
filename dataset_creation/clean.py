#!/usr/bin/python3
# -*- coding: utf-8 -*-


#RO	FR	IT	SP	PT	LT
langs = ["romanian", "french", "italian", "spanish", "portuguese"]

with open("all.txt", "r") as f:

	lines = f.readlines()

lines = [line.lower().replace("?", "").replace("german\t", "\t").replace("english\t", "\t") for line in lines]
problems = []

with open("all-clean.txt", "w", encoding="utf-8") as f:
	f.write("\t".join(langs).upper() + "\t" + "LATIN-ORIGINAL" + "\t" + "LATIN-CORRECT" + "\n")
	
	for line in lines:

		descs, latin_original, form = line.strip().split("\t")
		form = form.split(" ")[0].split(",")[0].split(" ")[0]
		if form == "-": 
			problems.append(line)
			continue
		descs = list(filter(lambda d: ":" in d, descs.split("*")))
		descs = {tup.split(":")[0]:tup.split(":")[1] for tup in descs}
		d = {}
		at_least_one = False
		
		for lang in langs:
		
			if lang in descs:
				at_least_one = True
				d[lang] = descs[lang].split(" ")[0].split(",")[0].split(" ")[0]
			else:
				d[lang] = "-"
		items = d.items()
		items = sorted(items, key = lambda tup: langs.index(tup[0]))
		if at_least_one:
			f.write("\t".join([item[1] for item in items]) + "\t" + form + "\n")

with open("problems.txt", "w") as f:

	for p in problems:
	
		f.write(p)
