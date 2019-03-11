#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import dynet as dy
import utils
from network import *
from encoder import *
from embedding_collector import *
from transformer_encoder import *
import argparse
import sys

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--running_id', type=int,
                    help='the id of the model (for averaging)')
    parser.add_argument('--model_size', type=int,
                    help='lstm hidden layer size')
    parser.add_argument('--optimizer', type=str,
                    help='type of the optimizer')
    parser.add_argument('--batch_size', type=int,
                    help='batch_size')   
    parser.add_argument('--dropout', type=float,
                    help='dropout value')       
    parser.add_argument('--network', type=str, required = True, help='lstm/gru')  
    parser.add_argument('--dynet-autobatch', type=int,
                    help='')                                     
    args = parser.parse_args()
    id = args.running_id
    model = dy.Model()
    ablation_mask = [1,1,1,1,1,1]
    train, dev, test = utils.get_datasets(1)
    letters, C2I, I2C = utils.create_voc(1)

    encoder = Encoder(model, C2I)
    encoders = []
    for i in range(6): # 6 languages encoders + separator encoder
        #encoder = Encoder(model, C2I)
        encoders.append(encoder)
    encoders.append(Encoder(model, C2I))
    
    embedding_collector = Collector(encoders, "voc/voc.txt", "embeddings/embeddings")
    network = Network(C2I, I2C, model, encoders, embedding_collector, id, dropout = args.dropout, lstm_size = args.model_size, optimizer = args.optimizer, model_type = args.network)
    #model.populate("model-lstm.m.1")
    best_index = network.train(train, dev, batch_size = args.batch_size)
    #model.populate("model-lstm.m.1")
    network.evaluate(test)
