import pywiktionary
import requests
import codecs
from pywiktionary import Wiktionary


def read_word_list(fname="en-eu-all"):

	with open(fname, "r") as f:
	
		lines = f.readlines()
	words = [l.strip().split("\t") for l in lines]
	
	return words
	
	
def main(words):

	wikt = Wiktionary(XSAMPA=True)
	IPAs = []
	
	for i, w in enumerate(words):
	
		result = wikt.lookup(w)
		found_ipa = False		
		print (i,"/",len(words))
		if isinstance(result, dict):
		
			if "English" in result:
				
				
				dicts = result["English"]
				if isinstance(dicts, list):
				
					word_dict = dicts[0]
					IPAs.append(word_dict["IPA"])
					found_ipa = True
		if not found_ipa:
		
			IPAs.append("-")
			
	return IPAs

def write_IPAs(IPAs, words):
	
	assert len(IPAs) == len(words)
	
	with open("en-eu-ipa", "w") as f:
	
		for (e,pie), ipa in zip(words, IPAs):
		
			f.write(e + ";" + ipa + "\t" + pie + "\n")
			



words = read_word_list()
en_words, pie_words = zip(*words)
ipas = main(en_words)
write_IPAs(ipas, words)
