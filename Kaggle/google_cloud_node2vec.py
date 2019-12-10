# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 22:16:31 2019

@author: fvice
"""

import random
import numpy as np
import pandas as pd
import csv
from tqdm import tqdm
import os
import networkx as nx
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import lightgbm

def main():
    
    with open(r"training.txt", "r") as f:
        reader = csv.reader(f)
        training  = list(reader)
    # in order of training examples
    training = [element[0].split(" ") for element in training]
    training = pd.DataFrame(training, columns=['Node1', 'Node2', 'Link'])
    print("Training examples shape: {}".format(training.shape))
    
    with open(r"testing.txt", "r") as f:
        reader = csv.reader(f)
        testing  = list(reader)
    # in order of testing examples
    testing = [element[0].split(" ") for element in testing]
    testing = pd.DataFrame(testing, columns=['Node1', 'Node2'])
    print("Testing examples shape: {}".format(testing.shape))
    
    train_graph_split_path = 'pickles/train_graph_split.PICKLE'
    
    if os.path.exists(train_graph_split_path):
        with open(train_graph_split_path, 'rb') as f:
            keep_indices = pickle.load(f)
        f.close()
    else:
        keep_indices = random.sample(range(len(training)), k=int(len(training) * 0.05))
        with open(train_graph_split_path, '+wb') as f:
            pickle.dump(keep_indices, f)
        f.close()
    
    data_train_val = training.iloc[keep_indices]
    data_train = training.loc[~training.index.isin(keep_indices)]
    
    linked_nodes = data_train.loc[data_train['Link']=='1']
    linked_nodes = linked_nodes[['Node1', 'Node2']]
    linked_nodes.to_csv('linked_nodes.txt', sep=' ', index=False, header=False)
    graph=nx.read_edgelist('linked_nodes.txt', create_using=nx.Graph(), nodetype = str)
    
    from node2vec import Node2Vec
    from node2vec.edges import HadamardEmbedder
    dicti = {}
    p_d, p_wl, p_nw, p_w, p_mc, p_bw = 64, 30, 200, 10, 1, 4
    
    node2vec = Node2Vec(graph, dimensions=p_d, walk_length=p_wl, num_walks=p_nw, workers=15)
    model = node2vec.fit(window=p_w, min_count=p_mc, batch_words=p_bw)  # Any keywords acceptable by gensim.Word2Vec can be passed
    # Embed edges using Hadamard method
    edges_embs = HadamardEmbedder(keyed_vectors=model.wv)
    
    embed_size = len(edges_embs[('0', '1')])
    df_train = pd.DataFrame(0, index=np.arange(len(data_train_val)), columns=range(embed_size))
    j = []
    for j, i in tqdm(enumerate(data_train_val.index), position=0, leave=True, total = len(data_train_val)):
        try:
            df_train.loc[j] = edges_embs[(data_train_val.loc[i]['Node1'], data_train_val.loc[i]['Node2'])]
        except:
            df_train.loc[j] = np.zeros(embed_size)
            
    X = df_train
    y = data_train_val['Link']
    y = list(map(lambda i: int(i), y))
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=1)
    
    lgbm = lightgbm.LGBMClassifier()
    model_lgbm = lgbm.fit(X_train, y_train)
    predictions = model_lgbm.predict(X_test)
    dicti['params:{}:{}:{}:{}:{}:{}'.format(p_d, p_wl, p_nw, p_w, p_mc, p_bw)] = f1_score(y_test, predictions)
    print(dicti['params:{}:{}:{}:{}:{}:{}'.format(p_d, p_wl, p_nw, p_w, p_mc, p_bw)])


if __name__ == "__main__":
    model = main()