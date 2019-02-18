import utils
import dynet as dy
import numpy as np
import random
random.seed(0)
np.random.seed(0)

class TransformerEncoder(object):

    def __init__(self, model, c2i, voc, d_model = 100, N = 3, h = 10):
    
        self.d_model = d_model
        self.d_v = d_model / h
        self.d_k = d_model / h
        self.N = N
        self.h = h
        self.c2i = c2i
        self.model = model
        self.voc = voc
        self.initialize()
        
    def initialize(self):
    
        self.E_pos = self.model.add_lookup_parameters((512, self.d_model))
        self.E_tok = self.model.add_lookup_parameters((len(self.voc), self.d_model))
        self.W_q = [[self.model.add_parameters((self.d_model, self.d_k)) for h in range(self.h)] for i in range(self.N)]
        self.W_k = [[self.model.add_parameters((self.d_model, self.d_k)) for h in range(self.h)] for i in range(self.N)]
        self.W_v = [[self.model.add_parameters((self.d_model, self.d_v)) for h in range(self.h)] for i in range(self.N)]
        self.W_o = [self.model.add_parameters((self.h * self.d_v, self.d_model)) for i in range(self.N)]
        self.W_ff = [self.model.add_parameters((self.d_model, self.d_model)) for i in range(self.N)]
        self.W_pred = self.model.add_parameters((2, self.d_model))
        self.trainer = dy.AdamTrainer(self.model)
        self.W_combine = self.model.add_parameters((self.d_model, self.d_model * 2))
        self.E_sep = self.model.add_lookup_parameters((len(self.voc), self.d_model))
        self.E_lang = self.model.add_lookup_parameters((6, self.d_model))
        self.langs = ["f", "i", "s", "r", "p", "l"]
        
    def save_position_embeddings(self):
    
        e = self.E_pos.as_array()
        with open("embeddings.txt", "w") as f:
        
            for row in e:
            
                elem = ['%.3f' % number for number in row]
                f.write(" ".join(elem) + "\n")
        
    def attention(self, Q, K, V):
        weights = dy.softmax(dy.cdiv(Q * dy.transpose(K), dy.scalarInput(np.sqrt(self.d_k))), d = 1)
        #v = weights.value()
        #np.save("att.txt", weights.value())
        return weights * V
    
    def multi_head(self, Q, K, V, W_Qs, W_Ks, W_Vs, W_o):
        
        heads = [self.attention(Q * dy.parameter(W_Qs[i]), K * dy.parameter(W_Ks[i]), V * dy.parameter(W_Vs[i])) for i in range(self.h)]
        return dy.concatenate_cols(heads) * dy.parameter(W_o)
    
    def _get_chars_encodings(self, x):
    
    
            x_splitted = x.split("*")
            x_splitted = [lang_and_word.split(":") for lang_and_word in x_splitted]
            start = self.E_sep[self.c2i[x_splitted[0][0]]]
            end = self.E_sep[self.c2i[x_splitted[-1][0]]]
            rest = x_splitted[1:-1]

            encoding_chars = [[dy.parameter(self.W_combine)*dy.concatenate([self.E_lang[self.langs.index(lang)], self.E_tok[self.c2i[c] if c in self.c2i else self.c2i["-"]]]) for c in word] for (lang, word) in rest]
            encoding_seps = [self.E_sep[self.c2i[lang]] for (lang, word) in rest]

            #encoding_chars  = [item for sublist in encoding_chars for item in sublist]
            encoded_x = [start]
            for i in range(len(encoding_chars)):
                encoded_x.append(encoding_seps[i])
                for c in encoding_chars[i]:
                    encoded_x.append(c)
                    
            encoded_x.append(end)
            return encoded_x
            
    def encode(self, seq):

        langs_and_chars = seq.split("*")
        h = dy.transpose(dy.concatenate_cols(self._get_chars_encodings(seq)))
        #h = dy.transpose(dy.concatenate_cols([self.E_pos[i] + self.E_tok[self.c2i[w]] for (i,w) in enumerate(seq)]))
        
        for i in range(self.N):

            W_Qs, W_Ks, W_Vs, W_o, W_ff = self.W_q[i], self.W_k[i], self.W_v[i], self.W_o[i], self.W_ff[i]
            Q = h
            V = h
            K = h
            h_new = self.multi_head(Q, K, V, W_Qs, W_Ks, W_Vs, W_o)
            h = h + h_new
            h = dy.rectify(h * dy.parameter(W_ff)) + h
            #h = dy.layer_norm(h) # didn't manage to make it work. use regular normalization instead
            #h = self.normalize(h)

        return h
    
    def normalize(self, activations):
    
        means = dy.mean_dim(activations, d = [1], b = 0)
        stds = dy.std_dim(activations, d = [1], b = 0)
        activations = dy.cdiv((activations - means), stds)
        return activations
            
    
