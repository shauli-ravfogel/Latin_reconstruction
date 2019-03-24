#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import dynet as dy
import numpy as np
import random


EMBEDDING_SIZE = 100
LSTM_SIZE = 100
NUM_LAYERS  = 1
REVERSE = False
TRAIN = True
DROPOUT = 0.1


    
def edit_distance(s1, s2):
    s1 = s1.replace("<", "").replace(">", "")
    s2 = s2.replace("<", "").replace(">", "")
    
    if len(s1) > len(s2):
        s1, s2 = s2, s1

    distances = range(len(s1) + 1)
    for i2, c2 in enumerate(s2):
        distances_ = [i2+1]
        for i1, c1 in enumerate(s1):
            if c1 == c2:
                distances_.append(distances[i1])
            else:
                distances_.append(1 + min((distances[i1], distances[i1 + 1], distances_[-1])))
        distances = distances_
    return distances[-1]
    
class Network(object):

    def __init__(self, C2I, I2C, model, encoders, embedding_collector, att_collector, id, dropout, lstm_size, optimizer, model_type, embs_wrapper, include_embeddings, features):
    
        self.C2I = C2I
        self.I2C = I2C
        self.id = id
        self.model = model
        self.add_parameters(dropout, lstm_size, optimizer, model_type, include_embeddings)
        self.encoders = encoders
        self.l2e = {"s": self.encoders[0], "i": self.encoders[1], "f": self.encoders[2], "p": self.encoders[3],
        "r": self.encoders[4], "l": self.encoders[5], "sep": self.encoders[6]}
        self.langs = ["s", "i", "r", "f", "p", "l"]
        self.embedding_collector = embedding_collector
        self.embedding_provider = embs_wrapper
        self.include_embeddings = include_embeddings
        self.att_collector = att_collector
        self.best_acc = -1
        self.features = features
        self.accs = [-1, -1]
        
    def add_parameters(self, dropout, lstm_size, optimizer, model_type, include_embeddings, gru = True):
    
        if model_type == "gru":
            self.encoder_rnn = dy.GRUBuilder(NUM_LAYERS, EMBEDDING_SIZE, lstm_size, self.model)
            self.encoder_rnn.set_dropout(dropout)
            self.encoder_rnn2 = dy.GRUBuilder(NUM_LAYERS, EMBEDDING_SIZE, lstm_size, self.model)
            self.encoder_rnn2.set_dropout(dropout)
            self.decoder_rnn = dy.GRUBuilder(NUM_LAYERS, EMBEDDING_SIZE+lstm_size, lstm_size, self.model)
            self.decoder_rnn.set_dropout(dropout)
        else:
        
            self.encoder_rnn = dy.CoupledLSTMBuilder(NUM_LAYERS, EMBEDDING_SIZE, lstm_size, self.model)
            self.encoder_rnn.set_dropout(dropout)
            self.encoder_rnn2 = dy.CoupledLSTMBuilder(NUM_LAYERS, EMBEDDING_SIZE, lstm_size, self.model)
            self.encoder_rnn2.set_dropout(dropout)
            self.decoder_rnn = dy.CoupledLSTMBuilder(NUM_LAYERS, EMBEDDING_SIZE+lstm_size, lstm_size, self.model)
            self.decoder_rnn.set_dropout(dropout)    
        
        global DROPOUT
        DROPOUT = dropout
        
        self.W1 =  self.model.add_parameters((200,  lstm_size))
        self.b1 = self.model.add_parameters((200, 1))
        self.W2 =  self.model.add_parameters((100,  200))
        self.b2 = self.model.add_parameters((100, 1))
        self.W3 =  self.model.add_parameters((len(self.C2I),  100))
        self.b3 = self.model.add_parameters((len(self.C2I), 1))
        self.W_query = self.model.add_parameters((lstm_size, lstm_size))
        self.W_key = self.model.add_parameters((lstm_size, lstm_size))
        self.W_val = self.model.add_parameters((lstm_size, lstm_size))
        self.W_att = self.model.add_parameters((1, EMBEDDING_SIZE))
        self.W_c_s = self.model.add_parameters((lstm_size, EMBEDDING_SIZE))
        self.W_direct = self.model.add_parameters((len(self.C2I),  lstm_size))
        self.b_att = self.model.add_parameters((lstm_size, 1))
        self.b_direct = self.model.add_parameters((len(self.C2I), 1))
        self.E_lang = self.model.add_lookup_parameters((7, EMBEDDING_SIZE))
        #self.W_pred = self.model.add_lookup_parameters((lstm_size + lstm_size, lstm_size))
        self.latin_semantic_rep = {}
        
        #self.W_latin_embeddings = self.model.add_lookup_parameters((EMBEDDING_SIZE, 100))
        #self.W_latin_embeddings2 = self.model.add_lookup_parameters((EMBEDDING_SIZE, EMBEDDING_SIZE))
        
        if optimizer == "sgd":   
             self.trainer = dy.SimpleSGDTrainer(self.model)
        elif optimizer == "rms":
             self.trainer = dy.RMSPropTrainer(self.model)
        if optimizer == "cyclic":
             self.trainer = dy.CyclicalSGDTrainer(self.model)
        elif optimizer == "adam":
            self.trainer = dy.AdamTrainer(self.model)
        else:
              self.trainer = dy.AdagradTrainer(self.model)
              
        
        
    def _get_semantic_rep(self, y):
                  w_emb = dy.parameter(self.W_latin_embeddings)
                  w_emb2 = dy.parameter(self.W_latin_embeddings2)
                  semantic_rep = w_emb2 * dy.rectify(w_emb * self.embedding_provider.get_word_embedding(y))
                  self.latin_semantic_rep[y] = semantic_rep.npvalue()
                  return semantic_rep    
    
    def encode0(self, x, y, bilstm = False, train = True):
            
            char_dropout = 0.0 if train else 0
            
            x_splitted = x.split("*")
            x_splitted = [lang_and_word.split(":") for lang_and_word in x_splitted]
            start = self.l2e["sep"].encode(x_splitted[0][0], "sep")
            end = self.l2e["sep"].encode(x_splitted[-1][0], "sep")
            rest = x_splitted[1:-1]
            
            encoding_chars = [[self.l2e[lang].encode(c, lang) for c in word] for (lang, word) in rest]
            encoding_seps = [self.l2e["sep"].encode(lang , "sep") for (lang, word) in rest]

            #encoding_chars  = [item for sublist in encoding_chars for item in sublist]
            encoded_x = [start]
            if self.include_embeddings:
                  encoded_x.append(self._get_semantic_rep(y))

                  
            for i in range(len(encoding_chars)):
                encoded_x.append(encoding_seps[i])
                for c in encoding_chars[i]:
                    encoded_x.append(c)
                    
            encoded_x.append(end)
            if self.include_embeddings:
                  encoded_x.append(self._get_semantic_rep(y))
            
 
            s = self.encoder_rnn.initial_state()
            states = s.transduce(encoded_x)
            last_state = states[-1]
        
            if bilstm:
        
                s2 = self.encoder_rnn2.initial_state()
                states2 = s2.transduce(encoded_x[::-1])
                last_state2 = states2[-1]
            
                states = [dy.esum([v1, v2]) for (v1, v2) in zip(states, states2[::-1])]
            
            #assert len(encoded_x) == len(states)
            
            #states = [s+c for (s,c) in zip(states, encoded_x)]
            return states, encoded_x

    def encode1(self, x, bilstm = False, train = True):
    
        encoded_x = [self.encoders[0].encode(c, "i") for c in x]
        s = self.encoder_rnn.initial_state()
        states = s.transduce(encoded_x)
        last_state = states[-1]   
        
        return states, encoded_x
    
    def encode(self, x,  bilstm = False, train = True):  

        word_encodings = [self.encoders[0].encode(word, lang[0]) for lang, word in x.items()]
                  
        all_vecs = [self.encoders[0].encode("<", "sep")] + [item for sublist in word_encodings for item in sublist] + [self.encoders[0].encode(">", "sep")]
        s = self.encoder_rnn.initial_state()
        states = s.transduce(all_vecs)
        
        return states, all_vecs, ""
                
    def encode2(self, x,  bilstm = False, train = True):  
    
        langs = x.keys()
        words_encodings = []
        for lang in langs:
                word_encoding = []
                
                if not self.features:
                   for c in x[lang]:
                
                        word_encoding.append(self.encoders[0].encode(c, lang[0]))
                else:
                       word_encoding = self.encoders[0].encode(x[lang], lang[0])
                       #print(word_encoding)
                words_encodings.append(word_encoding)
        
        all_strings = ["<"]
        all_vecs = [self.encoders[0].encode("<", "sep")]
        
        assert len(langs) == len(words_encodings)
        
        for lang, word_vecs in zip(langs, words_encodings):
        
                all_strings.extend(["*", lang[0], ":"])
                all_vecs.append(self.encoders[0].encode("*", "sep"))
                all_vecs.append(self.encoders[0].encode(lang[0], "sep"))
                all_vecs.append(self.encoders[0].encode(":", "sep"))
                
                #assert len(x[lang]) == len(word_vecs)
                
                for c,vec in zip(x[lang], word_vecs):
                
                        all_strings.append(c)
                        all_vecs.append(vec)
                
        all_strings.append(">")
        all_vecs.append(self.encoders[0].encode(">", "sep"))
        
        #assert len(all_strings) == len(all_vecs)

        s = self.encoder_rnn.initial_state()
        states = s.transduce(all_vecs)
        
        if bilstm:
                s2 = self.encoder_rnn2.initial_state()
                states2 = s2.transduce(all_vecs[::-1])
                states = [dy.esum([v1, v2]) for (v1, v2) in zip(states, states2[::-1])]      
                
        return states, all_vecs, "".join(all_strings)
        
        
        
                
    def predict_letter(self, state, y, linear = False):
    
        W1 = dy.parameter(self.W1)
        W2 = dy.parameter(self.W2)
        W3 = dy.parameter(self.W3)
        b1 = dy.parameter(self.b1)
        b2 = dy.parameter(self.b2)
        b3 = dy.parameter(self.b3)
        W_direct = dy.parameter(self.W_direct)

        if not linear:
            h = dy.rectify(W1 * state + b1)
            scores = W3 * dy.rectify(W2 * h + b2)
        else:
            scores = W_direct * state + dy.parameter(self.b_direct)
        
        return scores
        
    def attend(self, query, states, encoded_input):
    
        #return states[-1]
        query = dy.parameter(self.W_query) * query
        
        W_att = dy.parameter(self.W_att)
        b_att = dy.parameter(self.b_att)

        #scores = [(W_att * dy.concatenate([query, state]) + b_att) for state in states]
        #scores = [(dy.dot_product(query, dy.parameter(self.W_key) * state) + b_att) for state in states]
        scores = [dy.dot_product(query, state) for state in states]
        weights = dy.softmax(dy.concatenate(scores))
        self.att_weights = weights.npvalue()
        #weighted_states =  dy.esum([dy.cmult(w,dy.parameter(self.W_val) * s) for (w,s) in zip(weights, states)])
        #weighted_states =  dy.esum([dy.cmult(w, s) for (w,s) in zip(weights, states)])
        weighted_states =  dy.esum([dy.cmult(w, dy.parameter(self.W_c_s)*c + dy.parameter(self.W_key) * s) for (w,s,c) in zip(weights, states, encoded_input)])
        
        return weighted_states
        
    def decode(self, states, y, encoded_input, train = False):
    
        def sample(probs):
            return np.argmax(probs)
        
        s = self.decoder_rnn.initial_state()

        start_encoded = self.l2e["sep"].encode("<s>", "sep")
        out=[]
        loss = dy.scalarInput(0.)
        #s =  s.add_input(states[-1]) #s.add_input(dy.concatenate([start_encoded, states[-1]]))
        s = s.add_input(dy.concatenate([start_encoded, states[-1]]))


        generated_string = []
        weighted_states = states[-1]        

        for char in y:
            true_char_encoded = self.l2e["l"].encode(char, "l")

            scores = self.predict_letter(s.output()+weighted_states, y)

            generated_string.append(scores)
                
            weighted_states = self.attend(s.output(), states, encoded_input)
            #s = s.add_input(weigatt_weightshted_states) #s.add_input(dy.concatenate([true_char_encoded, weighted_states]))
            s = s.add_input(dy.concatenate([true_char_encoded, weighted_states]))
            if char in self.C2I:
                loss += dy.pickneglogsoftmax(scores, self.C2I[char])
            
        return loss, generated_string
            
    def generate(self, states, encoded_x, x, y):
        
        i = 0
        s = self.decoder_rnn.initial_state()
        start_encoded = self.l2e["sep"].encode("<s>", "sep")
            
        #s = s.add_input(states[-1]) #s.add_input(dy.concatenate([start_encoded, states[-1]]))
        s = s.add_input(dy.concatenate([start_encoded, states[-1]]))
        #s = s.add_input(dy.concatenate([start_encoded, self.attend(s.output(), states)]))
        generated_string = ""
        weighted_states = states[-1]
                 
        while i < 30:
            i+=1
            scores = self.predict_letter(s.output()+weighted_states, y)
            letter = self.I2C[np.argmax(scores.npvalue())]

            #if i > 1: self.att_collector.collect(self.att_weights, x, letter, y, i)
            generated_string += letter
            char_encoded =  self.l2e["l"].encode(letter, "l")

            weighted_states = self.attend(s.output(), states, encoded_x)
            #s = s.add_input(weighted_states) #s.add_input(dy.concatenate([char_encoded, weighted_states]))
            s = s.add_input(dy.concatenate([char_encoded, weighted_states]))

            if letter == ">":
                break
               
        return generated_string
            
  
            
    def train(self, train_data, dev_data, num_epochs = 150, batch_size = 10):

        for I in range(num_epochs):

                print ("EPOCH NUMBER {}".format(I))
                self.embedding_collector.collect()
                avg_loss = 0.
                random.shuffle(train_data)
                good, bad = 0., 0.
                avg_edit_distance = 0.
                q = 0.
                losses = []
                
                preds = []
                 
                for i, (x,y) in enumerate(train_data):

                    if i % batch_size == 0 and i > 0:
                        
                        loss_sum = dy.esum(losses)
                        loss_sum.forward()
                        loss_sum.backward()
                        self.trainer.update()
                        losses = []
                        
                        # evaluate trainset accuracy
                
                        for (word_probs, y_true) in preds:
                
                           generated_string = ""
                           for char_probs in word_probs:
                    
                              generated_string += self.I2C[np.argmax(char_probs.npvalue())]
                    
                           if generated_string == y_true:
                    
                               good += 1
                           else:
                                bad += 1
                        
                        preds = []                        
                        dy.renew_cg()
                            
                    
                    encoded_state, encoded_x, x_str = self.encode(x, y, train = True)
   
                    loss, probs = self.decode(encoded_state, y, encoded_x, train = True)
                    preds.append((probs,y))
                    
                    
                    losses.append(loss)
                    

                    
                    
        
                    if i%2000 == 0 and i > 0:
                        print (i)
                        #print (avg_loss)
                        avg_loss = 0.
                        #self.test(dev_data)
                
                #print ('DROPOUT = 0.5')
                #self.embedding_collector.collect()
                print ("training accuracy: {}".format(good / (good + bad)))
                acc, edit_dis = self.evaluate(dev_data)
                self.accs.append(acc)
                
                patience = 20
                
                if I > 8 and abs(min(self.accs[-patience:]) - max(self.accs[-patience:])) < 0.01:
                
                      return 0
                
                if acc > self.best_acc:
                    self.best_acc = acc
                    self.model.save("model11.m")

                
        
        return 0
                
    def evaluate(self, dev_data):
        good, bad = 0., 0.
        avg_edit_distance = 0
        self.encoder_rnn.set_dropout(0.0)
        self.decoder_rnn.set_dropout(0.0)
        to_write = []
        
        #with open("preds"+str(self.id)+".txt", "w") as f:
        for i, (x,y) in enumerate(dev_data):
            
                dy.renew_cg()
                encoded_state, encoded_x, x_str = self.encode(x, y, train = False)

                reconstruction = self.generate(encoded_state, encoded_x, x_str, y)
                dis =  edit_distance(reconstruction, y)
                avg_edit_distance += dis
                
                #words = x_str.split(":")
                #words = "\t".join([w[:-2] for w in words[1:]]).replace("*", "")
                words = "\t".join(x.values())

                to_write.append((words + "\t" + reconstruction + "\t" + y +"\t" +  str(dis) + "\t" + str("%0.3f" % ((dis/(1 * len(y))))) + "\n", dis/(1.*len(y))))
                
                if reconstruction == y:
                    good += 1
                else:
                    bad += 1
                    
        self.encoder_rnn.set_dropout(DROPOUT)
        self.decoder_rnn.set_dropout(DROPOUT)
        
        with open("preds11.txt", "w") as f:
        #with open("preds-ipa.txt", "w") as f:
        
           to_write = sorted(to_write, key = lambda tup: tup[1])
           for string, _ in to_write:
           
              f.write(string)
      
        with open("semantic_rep-cyclic.txt", "w") as f:
        
             for w,vec in self.latin_semantic_rep.items():
             
                as_str = " ".join(["%0.3f" % x for x in vec])
                f.write(w + "\t" + as_str + "\n")
        
        print ("accuuracy: {}; edit distance: {}".format(good / (good + bad), avg_edit_distance/len(dev_data)))
        self.att_collector.checkout()
        return good / (good + bad), avg_edit_distance/len(dev_data)

        
        
        
        
