#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import dynet as dy
import numpy as np
import random

EMBEDDING_SIZE = 100
LSTM_SIZE = 100
NUM_LAYERS  = 1
DROPOUT = 0.1
REVERSE = False
TRAIN = True

def edit_distance(s1, s2):
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

    def __init__(self, C2I, I2C, model, encoders, embedding_collector, id):
    
        self.C2I = C2I
        self.I2C = I2C
        self.id = id
        self.model = model
        self.add_parameters()
        self.encoders = encoders
        self.l2e = {"s": self.encoders[0], "i": self.encoders[1], "f": self.encoders[2], "p": self.encoders[3],
        "r": self.encoders[4], "l": self.encoders[5], "sep": self.encoders[6]}
        
        self.embedding_collector = embedding_collector
        self.best_acc = -1
        
    def add_parameters(self, gru = False):
    
        if gru:
            self.encoder_rnn = dy.GRUBuilder(NUM_LAYERS, EMBEDDING_SIZE, LSTM_SIZE, self.model)
            self.encoder_rnn.set_dropout(DROPOUT)
            self.encoder_rnn2 = dy.GRUBuilder(NUM_LAYERS, EMBEDDING_SIZE, LSTM_SIZE, self.model)
            self.encoder_rnn2.set_dropout(DROPOUT)
            self.decoder_rnn = dy.GRUBuilder(NUM_LAYERS, EMBEDDING_SIZE+LSTM_SIZE, LSTM_SIZE, self.model)
            self.decoder_rnn.set_dropout(DROPOUT)
        else:
        
            self.encoder_rnn = dy.LSTMBuilder(NUM_LAYERS, EMBEDDING_SIZE, LSTM_SIZE, self.model)
            self.encoder_rnn.set_dropout(DROPOUT)
            self.encoder_rnn2 = dy.LSTMBuilder(NUM_LAYERS, EMBEDDING_SIZE, LSTM_SIZE, self.model)
            self.encoder_rnn2.set_dropout(DROPOUT)
            self.decoder_rnn = dy.LSTMBuilder(NUM_LAYERS, EMBEDDING_SIZE+LSTM_SIZE, LSTM_SIZE, self.model)
            self.decoder_rnn.set_dropout(DROPOUT)
        
        self.W1 =  self.model.add_parameters((100,  LSTM_SIZE))
        self.b1 = self.model.add_parameters((100, 1))
        self.W2 =  self.model.add_parameters((len(self.C2I),  100))
        self.b2 = self.model.add_parameters((len(self.C2I), 1))
        self.W1_0 = self.model.add_parameters((100,  100))
        self.W_query = self.model.add_parameters((LSTM_SIZE, LSTM_SIZE))
        self.W_query2 = self.model.add_parameters((LSTM_SIZE, LSTM_SIZE))
        self.W_query3 = self.model.add_parameters((LSTM_SIZE, LSTM_SIZE))
        
        self.W_key = self.model.add_parameters((LSTM_SIZE, LSTM_SIZE))
        self.W_val = self.model.add_parameters((LSTM_SIZE, LSTM_SIZE))
        self.W_att = self.model.add_parameters((LSTM_SIZE, LSTM_SIZE + EMBEDDING_SIZE))
        self.W_att2 = self.model.add_parameters((LSTM_SIZE, LSTM_SIZE))
        self.W_direct = self.model.add_parameters((len(self.C2I),  LSTM_SIZE))
        self.b_att = self.model.add_parameters((1, 1))
        self.W_lang_id =  self.model.add_parameters((EMBEDDING_SIZE,  2*EMBEDDING_SIZE))
        self.W_lang_id2 = self.model.add_parameters((EMBEDDING_SIZE,  EMBEDDING_SIZE))
        self.E_lang = self.model.add_lookup_parameters((6, EMBEDDING_SIZE))
        self.E_pos = self.model.add_lookup_parameters((50, EMBEDDING_SIZE))
        
        self.lang2ind = {"s": 0, "i": 1, "f": 2, "r": 3, "p": 4, "l": 5}
        self.trainer = dy.AdamTrainer(self.model)
        self.lang_matrices = [self.model.add_parameters((EMBEDDING_SIZE, EMBEDDING_SIZE)) for i in range(6)]
        #self.trainer = dy.CyclicalSGDTrainer(self.model)
        self.accs = []
        
        
    def encode(self, x, bilstm = False, language_dropout = 0.0, char_dropout = 0.00, train = True):
            
            x_splitted = x.split("*")
            x_splitted = [lang_and_word.split(":") for lang_and_word in x_splitted]
            start = self.l2e["sep"].encode(x_splitted[0][0])
            end = self.l2e["sep"].encode(x_splitted[-1][0])
            rest = x_splitted[1:-1]

           
           #encoding_chars = [[dy.parameter(self.W_lang_id2) * dy.rectify(dy.parameter(self.W_lang_id)*dy.concatenate([self.l2e[lang].encode(c),self.E_lang[self.lang2ind[lang]]])) for c in word] for (lang, word) in rest]
            encoding_chars = [ [self.l2e[lang].encode(c)+self.E_lang[self.lang2ind[lang]] + self.E_pos[i] for i,c in enumerate(word)] for (lang, word) in rest]
            #encoding_chars = [[dy.parameter(self.lang_matrices[self.lang2ind[lang]]) * self.l2e[lang].encode(c) for c in word] for (lang, word) in rest]
            encoding_seps = [self.l2e["sep"].encode(lang) for (lang, word) in rest]

            #encoding_chars  = [item for sublist in encoding_chars for item in sublist]
            encoded_x = [start]
            for i in range(len(encoding_chars)):
                encoded_x.append(encoding_seps[i])
                
                if random.random() < language_dropout:
                       encoded_x.append(self.l2e["sep"].encode("-"))
                else:
                       for c in encoding_chars[i]:
                            
                            if train and random.random() < char_dropout:
                                encoded_x.append(self.l2e["sep"].encode("?"))
                            else:
                                encoded_x.append(c)            
                
                if i == 1: break
                    
            encoded_x.append(end)
            #encoded_x = [self.encoder.encode(c) for c in x]

            s = self.encoder_rnn.initial_state()
            states = s.transduce(encoded_x)
            #states = [s+c for (s,c) in zip(states, encoded_x)]
            last_state = states[-1]
        
            if bilstm:
        
                s2 = self.encoder_rnn2.initial_state()
                states2 = s2.transduce(encoded_x[::-1])
                last_state2 = states2[-1]
            
                states = [dy.esum([v1, v2]) for (v1, v2) in zip(states, states2[::-1])]
             

            return states, encoded_x

    def encode0(self, x, bilstm = False, language_dropout = 0.0, char_dropout = 0.00, train = True):
    
        encoded_x = [self.encoders[0].encode(c) for c in x]
        s = self.encoder_rnn.initial_state()
        states = s.transduce(encoded_x)
        #states = [s+c for (s,c) in zip(states, encoded_x)]
        last_state = states[-1]   
        
        return states, encoded_x
        
    def predict_letter(self, state, linear = False):
    
        W1 = dy.parameter(self.W1)
        W1_0 = dy.parameter(self.W1_0)
        W2 = dy.parameter(self.W2)
        b1 = dy.parameter(self.b1)
        b2 = dy.parameter(self.b2)
        W_direct = dy.parameter(self.W_direct)
        
        if not linear:
            #h = dy.rectify(W1 * state + b1)
            h = dy.rectify(W1_0 * dy.rectify(W1 * state + b1))
            scores = W2 * h + b2
        else:
            scores = W_direct * state + b2
        
        return scores
        
    def attend(self, query, states, encoded_input):
        
        #return states[-1]
        query = dy.parameter(self.W_query) * query
        W_att = dy.parameter(self.W_att)
        b_att = dy.parameter(self.b_att)
        

        #scores = [(dy.dot_product(query, dy.parameter(self.W_key) * state) + b_att) for state,c in zip(states, encoded_input)]
        scores = [dy.dot_product(query, state) for state,c in zip(states, encoded_input)]

        weights = dy.softmax(dy.concatenate(scores))

        #weighted_states =  dy.esum([dy.cmult(w,dy.parameter(self.W_val) * c) for (w,s,c) in zip(weights, states,encoded_input)])
        weighted_states =  dy.esum([dy.cmult(w, W_att * dy.concatenate([s,c])) for (w,s,c) in zip(weights, states, encoded_input)])
        #weighted_states =  dy.esum([dy.cmult(w,s+c) for (w,s,c) in zip(weights, states, encoded_input)])

        return weighted_states # + dy.parameter(self.W_att2) * states[-1]
        
    def decode(self, states, y, encoded_input, train = False):
    
        def sample(probs):
            return np.argmax(probs)
        
        s = self.decoder_rnn.initial_state()

        start_encoded = self.l2e["sep"].encode("<s>")
        out=[]
        loss = dy.scalarInput(0.)
        #s =  s.add_input(states[-1]) #s.add_input(dy.concatenate([start_encoded, states[-1]]))
        s = s.add_input(dy.concatenate([start_encoded, states[-1]]))


        generated_string = ""
                

        for char in y:
            true_char_encoded = self.l2e["l"].encode(char)

            scores = self.predict_letter(s.output())

            if not train or train:
                letter = self.I2C[np.argmax(scores.npvalue())]
                generated_string += letter
                
            weighted_states = self.attend(s.output(), states, encoded_input)
            #s = s.add_input(weighted_states) #s.add_input(dy.concatenate([true_char_encoded, weighted_states]))
            s = s.add_input(dy.concatenate([true_char_encoded, weighted_states]))
            if char in self.C2I:
                loss += dy.pickneglogsoftmax(scores, self.C2I[char])
            
        return loss, generated_string
            
    def generate(self, states, encoded_input):
        
        i = 0
        s = self.decoder_rnn.initial_state()
        start_encoded = self.l2e["l"].encode("<s>")
            
        #s = s.add_input(states[-1]) #s.add_input(dy.concatenate([start_encoded, states[-1]]))
        s = s.add_input(dy.concatenate([start_encoded, states[-1]]))
        #s = s.add_input(dy.concatenate([start_encoded, self.attend(s.output(), states)]))
        generated_string = ""
                
        while i < 25:
            i+=1
            scores = self.predict_letter(s.output())
            letter = self.I2C[np.argmax(scores.npvalue())]
            #print (letter)
            generated_string += letter
            char_encoded =  self.l2e["l"].encode(letter)
            
            weighted_states = self.attend(s.output(), states, encoded_input)
            #s = s.add_input(weighted_states) #s.add_input(dy.concatenate([char_encoded, weighted_states]))
            s = s.add_input(dy.concatenate([char_encoded, weighted_states]))

            if letter == ">":
                break
               
        return generated_string
            
        
            
            
    def train(self, train_data, dev_data, num_epochs = 120, batch_size = 25):

        for I in range(num_epochs):
        
                print ("EPOCH NUMBER {}".format(I))
                
                avg_loss = 0.
                random.shuffle(train_data)
                good, bad = 0., 0.
                avg_edit_distance = 0.
                q = 0.
                   
                losses = []
                   
                for i, (x,y) in enumerate(train_data):

                    if i % batch_size == 0 and i > 0:
                        
                        loss_sum = dy.esum(losses)
                        loss_sum.forward()
                        loss_sum.backward()
                        self.trainer.update()
                        losses = []
                        dy.renew_cg()
                            
                    
                    encoded_states, encoded_input  = self.encode(x, train = True)
                    
                    loss, latin_rec = self.decode(encoded_states, y, encoded_input, train = True)
                    
                    
                    if latin_rec == y:
                        good += 1
                    else:
                        bad += 1
                        
                    if i%10 == 0:
                        avg_edit_distance += edit_distance(latin_rec, y)
                        q+=1
                    avg_loss += loss.value()


                    
                    losses.append(loss)
                    

                    
           
        
                    if i%2000 == 0 and i > 0:
                        print (i)
                        #print (avg_loss)
                        avg_loss = 0.
                        #self.test(dev_data)
                
                print ("train accuracy: {}".format(good/(good+bad)))
                print ("evaluating accuracy on dev set.")
                #print ('DROPOUT = 0.5')
                #self.embedding_collector.collect()
                acc = self.evaluate(dev_data)
                self.accs.append((acc, str(I)))
                if acc > self.best_acc:
                  self.best_acc = acc
                  self.model.save("model.m")
                self.embedding_collector.collect(self.lang2ind, self.E_lang)
        
        best_acc, best_index = max(self.accs, key = lambda acc_and_index: acc_and_index[0])
        print("best model is ", best_index)
        return best_index
                
    def evaluate(self, dev_data):
        good, bad = 0., 0.
        avg_edit_distance = 0
        self.encoder_rnn.set_dropout(0.0)
        self.decoder_rnn.set_dropout(0.0)
            
        with open("preds"+str(self.id)+".txt", "w") as f:
            for i, (x,y) in enumerate(dev_data):
            
                dy.renew_cg()
                encoded_states, encoded_input = self.encode(x, train = False)
                reconstruction = self.generate(encoded_states, encoded_input)
                dis =  edit_distance(reconstruction, y)
                avg_edit_distance += dis
                
                f.write(x + "\t" + reconstruction + "\t" + y +"\t" +  str(dis) + "\n")
                
                if reconstruction == y:
                    good += 1
                else:
                    bad += 1
        self.encoder_rnn.set_dropout(DROPOUT)
        self.decoder_rnn.set_dropout(DROPOUT)
        print ("accuuracy: {}; edit distance: {}".format(good / (good + bad), avg_edit_distance/len(dev_data)))
        return good / (good + bad)

    def test(self, dev_data):
            TRAIN = False
            f = open("no_italian.txt", "wb")
            f2 = open("states2.txt", "w")
            
            good, bad = 0., 0.
            avg_edit_distance = 0.
            self.encoder_rnn.set_dropout(0.0)
            self.decoder_rnn.set_dropout(0.0)
            outputs = []
            if REVERSE:
            
                f.write(("LATIN-INPUT    LANGS-TRUE    LANGS-PREDICTION    EDIT-DISTNACE\n").encode('utf-8'))
            else:
                f.write(("INPUT    LATIN-TRUE    LATIN-RECONSTRUCTION    EDIT-DISTANCE\n").encode('utf-8'))
                
            for i, (x,y) in enumerate(dev_data):
            
                dy.renew_cg()
                encoded_state = self.encode(x)
                states_str = [str(round(float(v), 5)) for v in encoded_state[-1].npvalue()]
                state_str = " ".join(states_str)
                f2.write(y + "\t" + state_str+"\n")

                reconstruction = self.generate(encoded_state)
                if REVERSE:
                    f.write(("<"+x + "\t"  + y + "\t" + reconstruction + "\t" + str(edit_distance(reconstruction, y)) + "\n").encode('utf-8'))
                else:
                    f.write((x + "\t" + "<" + y + "\t" + "<" + reconstruction + "\t" + str(edit_distance(reconstruction, y)) + "\n").encode('utf-8'))
                
                if reconstruction == y:
                    good += 1
                else:
                    bad += 1
                    
                
                avg_edit_distance += edit_distance(reconstruction, y)
            
            print ("dev accuracy: {}; dev average edit distance: {}".format(good/(good+bad), avg_edit_distance/len(dev_data)))
            self.encoder_rnn.set_dropout(DROPOUT)
            self.decoder_rnn.set_dropout(DROPOUT)
            f.close()
            f2.close()
            TRAIN = True
            print ("WITH diac.")
            return good/(good+bad)
            

        
    
        
        
        
        
    
        
        
        
        
