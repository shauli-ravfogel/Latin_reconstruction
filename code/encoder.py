#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import dynet as dy
import panphon

EMBEDDING_SIZE = 100


class Encoder(object):

    def __init__(self, model, C2I):

        self.model = model
        self.C2I = C2I
        self.E = model.add_lookup_parameters((len(C2I), EMBEDDING_SIZE))
        self.E_lang = model.add_lookup_parameters((7, EMBEDDING_SIZE))
        self.langs = ["s", "i", "r", "f", "p", "l", "sep"]
        self.W_combine = model.add_parameters((EMBEDDING_SIZE, 2 * EMBEDDING_SIZE))

    def encode(self, c, lang):
    
        W_combine = dy.parameter(self.W_combine)
        
        char_ind = self.C2I[c] if c in self.C2I else self.C2I["<unk>"]
        char_encoded = dy.lookup(self.E, char_ind)
        lang_encoded = self.E_lang[self.langs.index(lang)]
        
        return W_combine * dy.concatenate([char_encoded, lang_encoded])
        
class FeaturesEncoder(object):

	def __init__(self, model, C2I):

		self.model = model
		self.C2I = C2I
		self.E_plus = model.add_lookup_parameters((22, EMBEDDING_SIZE))
		self.E_minus = model.add_lookup_parameters((22, EMBEDDING_SIZE))
		self.E_not_relevant = model.add_lookup_parameters((22, EMBEDDING_SIZE))
		self.E = model.add_lookup_parameters((len(C2I), EMBEDDING_SIZE))
		self.ft = panphon.FeatureTable()
		self.E_lang = model.add_lookup_parameters((7, EMBEDDING_SIZE))
		self.langs = ["s", "i", "r", "f", "p", "l", "sep"]
		self.W_combine = model.add_parameters((EMBEDDING_SIZE, 2 * EMBEDDING_SIZE))
		
	def encode(self, word, lang):
		
		if lang != "l" and lang != "sep":
			encoding = []
			list_of_features = self.ft.word_to_vector_list(word, numeric=False)
			W_combine = dy.parameter(self.W_combine)
			lang_encoded = self.E_lang[self.langs.index(lang)]
		
			for char_repr in list_of_features:
				char_repr = list(char_repr)
				
				v = dy.vecInput(EMBEDDING_SIZE)

				for i, feature in enumerate(char_repr):
			
					if feature == "-":
				
						v += self.E_minus[i]
					
					elif feature == "+":
				
						v += self.E_plus[i] 
					
					elif feature == "0":
					
						v += self.E_not_relevant[i]
				v = W_combine * dy.concatenate([v, lang_encoded])
				#print("v is ", v)
				encoding.append(v)
			if encoding == []: encoding = [dy.vecInput(EMBEDDING_SIZE)]
			return encoding
		else:
			W_combine = dy.parameter(self.W_combine)
			char_ind = self.C2I[word] if word in self.C2I else self.C2I["<unk>"]
			char_encoded = dy.lookup(self.E, char_ind)
			lang_encoded = self.E_lang[self.langs.index(lang)]
			return W_combine * dy.concatenate([char_encoded, lang_encoded])			
