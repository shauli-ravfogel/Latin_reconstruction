#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import dynet as dy
import utils
from network import *
from encoder import *
from embedding_collector import *
from transformer_encoder import *
import sys

if __name__ == '__main__':
    
    id = sys.argv[1]
    model = dy.Model()
    ablation_mask = [1,1,1,1,1,1]
    train, dev, test = utils.get_datasets(id)
    letters, C2I, I2C = utils.create_voc(id)
    print(C2I)
    encoder = Encoder(model, C2I)
    encoders = []
    for i in range(6): # 6 languages encoders + separator encoder
        encoder = Encoder(model, C2I)
        encoders.append(encoder)
    encoders.append(Encoder(model, C2I))
    
    embedding_collector = Collector(encoders, "voc.txt", "embeddings")
    network = Network(C2I, I2C, model, encoders, embedding_collector, id)
    #network = TransformerEncoder(model, C2I, letters)
    #network = TransformerNetwork(C2I, I2C, model, encoders, embedding_collector, id, transformer_encoder)
    #model.populate("model.m")
    best_index = network.train(train, dev)
    model.populate("model.m"+best_index)
    network.embedding_collector.collect()
    network.evaluate(test)
