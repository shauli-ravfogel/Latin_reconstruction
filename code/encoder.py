#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import dynet as dy

EMBEDDING_SIZE = 100


class Encoder(object):

    def __init__(self, model, C2I):

        self.model = model
        self.C2I = C2I
        self.E = model.add_lookup_parameters((len(C2I), EMBEDDING_SIZE))

    def encode(self, c):
        #print (c, c in self.C2I)
        char_ind = self.C2I[c] if c in self.C2I else self.C2I["<unk>"]

        char_encoded = dy.lookup(self.E, char_ind)
        return char_encoded
