#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import dynet as dy
import utils
from network import *
from encoder import *
from embedding_collector import *
from transformer_encoder import *
from attention_recorder import *
from embs_wrapper import *
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
    parser.add_argument('--dynet-mem', type=int,
                    help='')
    parser.add_argument('--include_embeddings', type=int,
                    help='')
    parser.add_argument('--features', type=int,
                    help='whether or not to represent input as phonolgocial features')
                                                                               
    args = parser.parse_args()
    id = args.running_id
    model = dy.Model()
    ablation_mask = [1,1,1,1,1,1] # ["rm", "fr", "it", "sp", "pt", "lt"]
    train, dev, test, test_missing = utils.get_datasets(id, ablation_mask)
    letters, C2I, I2C = utils.create_voc(id)

    latin_embeddings  = LatinEmbeddings()

    
    encoder = Encoder(model, C2I) if not args.features else FeaturesEncoder(model, C2I)
    encoders = []
    for i in range(6): # 6 languages encoders + separator encoder
        #encoder = Encoder(model, C2I)
        encoders.append(encoder)
    encoders.append(Encoder(model, C2I))
    attention_recorder = AttentionRecorder()
    embedding_collector = Collector(encoders, "voc/voc.txt", "embeddings/embeddings", args.features)
    network = Network(C2I, I2C, model, encoders, embedding_collector, attention_recorder, id, dropout = args.dropout, lstm_size = args.model_size, optimizer = args.optimizer, model_type = args.network, embs_wrapper = latin_embeddings, include_embeddings = args.include_embeddings, features = args.features)
    #model.populate("model11.m")
    best_index = network.train(train, dev, batch_size = args.batch_size)
    #model.populate("model-lstm.m.1")
    network.evaluate(test)
