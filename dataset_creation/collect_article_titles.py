#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
    get_category_items.py
	
    MediaWiki Action API Code Samples
    Demo of `Categorymembers` module: List twenty items in a category.
    MIT license
"""

import requests

S = requests.Session()

URL = "https://en.wiktionary.org/w/api.php"

PARAMS = {
    'action': "query",
    'list': "categorymembers",
    'cmtitle': "Category:Latin lemmas",
    'cmlimit': 200000,
    'format': "json"
}

R = S.get(url=URL, params=PARAMS)
DATA = R.json()
print(DATA["continue"])
all_words = []
continue_param = DATA["continue"]["cmcontinue"]

for dictionary in DATA["query"]["categorymembers"]:

	all_words.append(dictionary["title"])

while "continue" in DATA:
	print ("continue")
	continue_param = DATA["continue"]["cmcontinue"]
	PARAMS['cmcontinue'] = continue_param
	R = S.get(url=URL, params=PARAMS)
	DATA = R.json()
	for dictionary in DATA["query"]["categorymembers"]:

		all_words.append(dictionary["title"])
	print(len(all_words))

with open("all_latin_words.txt", "w", encoding = "utf-8") as f:

	for w in all_words:
		if not "Category" in w:
				f.write(w + "\n")
