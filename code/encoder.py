#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import dynet as dy

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
