import dynet as dy
import numpy as np

class LatinEmbeddings(object):

    def __init__(self, fname = "../data/model.txt"):
        with open(fname, "r") as f:
        
            lines = f.readlines()
        
        self.w2vec = {}
        for i, line in enumerate(lines):
        
            if i == 0: continue
                
            word, vec = line.strip().split(" ", 1)
            vec = np.array([float(x) for x in vec.split(" ")])
            self.w2vec[word] = vec
    
    def get_word_embedding(self, latin_word):
    
            latin_word = latin_word.replace("<", "").replace(">", "")
            
            if latin_word in self.w2vec:
                vec = self.w2vec[latin_word]
                vec /= np.linalg.norm(vec)
            else:
            
                vec = np.zeros(100)
                
            v = dy.vecInput(100)
            v.set(vec)
            return v
    
