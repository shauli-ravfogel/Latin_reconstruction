#!/usr/bin/python3
# -*- coding: utf-8 -*-

from bs4 import BeautifulSoup
import urllib.request
import re
import os
import time

try:
	os.remove("all.txt")
except:
	pass


html_page = urllib.request.urlopen("https://en.wiktionary.org/wiki/jaculabilis")
html_page = urllib.request.urlopen("https://en.wiktionary.org/wiki/manduco")
soup = BeautifulSoup(html_page, 'html.parser')
noun_class = "prettytable inflection-table"
nouns_class2 = "prettytable inflection-table inflection-table-la"
verb_class = "inflection-table vsSwitcher vsToggleCategory-inflection"
verb_class2 = "inflection-table vsSwitcher vsToggleCategory-inflection-la"
#inflections = soup.find_all('table', class_ = 'prettytable inflection-table')
inflections = soup.find('table', {'class':[noun_class, verb_class, nouns_class2]})
#table_head = inflections[0]

def filter_translations(descs):

	ind = 0
	chosen = -1
	
	for i, d in enumerate(descs):
	
		prev = d.find_previous()

		if prev:
			prev_prev = d.find_previous().find_previous()
			if (prev.has_attr("class") and "".join(prev["class"]) == "derivedterms") or (prev.has_attr("id") and prev["id"] == "Descendants"):
				return i
			if prev_prev:
	
				if (prev_prev.has_attr("class") and "".join(prev_prev["class"]) == "derivedterms") or (prev_prev.has_attr("id") and prev_prev["id"] == "Descendants"):
		
					return i
	
	return -1
			
	

def get_words(fname="all_latin_words.txt"):
	with open(fname, "r") as f:
	
		lines = f.readlines()
	
	words = [line.strip() for line in lines]
	return words

def collect(words):
   	
	for i, w in enumerate(words):
		#w = "abbatissa"
		print ("w is {} and index is {}".format(w, i))
		w = w.replace(" ", "_")
		try:

			html_page = urllib.request.urlopen("http://en.wiktionary.org/w/index.php?title="+w+"&printable=yes")

		except:

			continue
		soup = BeautifulSoup(html_page, 'html.parser')

		table = soup.find('table', {'class':[noun_class, verb_class, nouns_class2, verb_class2]})

		if not table: continue

		infi, acc, children = None, None, None
		cls = " ".join(table["class"])
		if (cls == noun_class) or cls == nouns_class2:
			accusative_title = table.find(title="accusative case")
			if accusative_title: 
					accusative_line = accusative_title.find_next("td")
					#print(accusative_line)
					if accusative_line.findChild(class_="Latn"):
						#acc = accusative_line.find("a")
						#if acc:
						acc = accusative_line.text.strip()
						print("acc is ", acc)
		else:

			infi = table.findChild(lambda tag: tag.name == "th" and tag.text.strip() == "infinitives")
			if infi and infi.nextSibling:
				infi = infi.nextSibling
				if infi.nextSibling:
					infi = infi.nextSibling.text.strip()
					print("infi is ", infi)
	
		descs = soup.find_all(lambda tag: tag.name == "li" and ":" in tag.text and len(tag.text) < 40)

		start_ind = filter_translations(descs)
		if start_ind != -1:
			descs = descs[start_ind:]
		
			descs = [c.text.lower().replace(": ", ":").replace("\n","").replace("→ ", "") for c in descs if c.findChild(class_="Latn")]
			descs = [d for d in descs if ":" in d and not "hyphenation" in d]
			children = descs
		
		if children:
			
			with open("all.txt", "a+", encoding="utf-8") as f:

				form = acc if acc else infi if infi else "-"
				children_str = "*".join(children) if children else ""
				f.write(children_str + "\t" + w + "\t" + form + "\n")
		#exit()
collect(get_words())	
