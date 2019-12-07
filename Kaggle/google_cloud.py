# -*- coding: utf-8 -*-
"""
Created on Sat Dec  7 21:42:49 2019

@author: fvice
"""

import networkx as nx
import pandas as pd
import os
import random
import stellargraph as sg
from stellargraph.data import UnsupervisedSampler
from stellargraph.mapper import Attri2VecLinkGenerator
from stellargraph.layer import Attri2Vec, link_classification
from tensorflow import keras
import csv
from tqdm import tqdm
import pickle

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
        
    '''
    uncomment lines for reduced corpus with stopword removal. In future integrate stemmer here, multi-language
    '''
    NODE_INFO_DIRECTORY = r"node_information/text/"
    
    corpus_path = r"pickles/simple_corpus.PICKLE" 
    ids_path = r"pickles/ids.PICKLE"
    if os.path.exists(corpus_path):
        with open(corpus_path, 'rb') as f:
            corpus = pickle.load(f)
        f.close()
        with open(ids_path, 'rb') as f:
            ids = pickle.load(f)
        f.close()
    else:
        corpus = []
        ids = []
        for filename in tqdm(os.listdir(NODE_INFO_DIRECTORY), position=0, leave=True):
            with open(NODE_INFO_DIRECTORY + filename, 'r', encoding='UTF-8', errors='ignore') as f:
                doc_string = []
                for line in f:
                    [doc_string.append(token.strip()) for token in line.lower().strip().split(" ") if token != ""]
                corpus.append(' '.join(doc_string))
                ids.append(filename[:-4])
        with open(corpus_path, '+wb') as f:
            pickle.dump(corpus, f)
        f.close()
        with open(ids_path, '+wb') as f:
            pickle.dump(ids, f)
        f.close() 
        
    stemmed_corpus_path = r"pickles/stemmed_corpus.PICKLE" 
    if os.path.exists(stemmed_corpus_path):
        with open(stemmed_corpus_path, 'rb') as f:
            stemmed_corpus = pickle.load(f)
        f.close()
    else:
        print('Stemmed corpus unavailable')
    
    # in order of alphabetical text information i.e. 0, 1, 10, 100
    node_info = pd.DataFrame({'id': ids, 'corpus': corpus, 'stemmed': stemmed_corpus})
    print("Training node info shape: {}".format(node_info.shape))

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
        
    linked_nodes = training.loc[training['Link']=='1']
    linked_nodes = linked_nodes[['Node1', 'Node2']]
    edgelist = linked_nodes.rename(columns={"Node1": "source", "Node2": "target"})
               
    
    lda_path = r"pickles/stemmed_lda_matrix.PICKLE"
    if os.path.exists(lda_path):
        with open(lda_path, 'rb') as f:
            lda = pickle.load(f)
        f.close()

    lda.shape
    
    
    feature_names = node_column_names = ["w_{}".format(ii) for ii in range(10)]
    node_data = pd.DataFrame(lda, columns=node_column_names)
    node_data.index = [str(i) for i in node_data.index]
    
    
    G_all_nx = nx.from_pandas_edgelist(edgelist)
    
        
    all_node_features = node_data[feature_names]
    
    
    G_all = sg.StellarGraph(G_all_nx, node_features=all_node_features)
    
    print(G_all.info())
    
    G_all.get_feature_for_nodes(['0'])
    
    ## Get DBLP Subgraph 
    ### with papers published before a threshold year
    
    
    sub_linked_nodes = data_train_val.loc[data_train_val['Link']=='1']
    sub_linked_nodes = sub_linked_nodes[['Node1', 'Node2']]
    subgraph_edgelist = sub_linked_nodes.rename(columns={"Node1": "source", "Node2": "target"})
    
    
    G_sub_nx = nx.from_pandas_edgelist(subgraph_edgelist)
    
    
    
    subgraph_node_ids = sorted(list(G_sub_nx.nodes))
    
    
    subgraph_node_features = node_data[feature_names].reindex(subgraph_node_ids)
    
    
    G_sub = sg.StellarGraph(G_sub_nx, node_features=subgraph_node_features)
    
    print(G_sub.info())
    
    ## Train attri2vec on the DBLP Subgraph
    
    
    nodes = list(G_sub.nodes())
    number_of_walks = int(input('Number of Walks: '))
    length = int(input('Walk length: '))
    
    
    unsupervised_samples = UnsupervisedSampler(G_sub, nodes=nodes, length=length, number_of_walks=number_of_walks)
    
    
    batch_size = 50
    epochs = int(input('Enter number of epochs: '))
    
    
    generator = Attri2VecLinkGenerator(G_sub, batch_size)
    
        
    layer_sizes = [128]
    attri2vec = Attri2Vec(layer_sizes=layer_sizes, generator=generator.flow(unsupervised_samples), bias=False, normalize=None)
    
    # Build the model and expose input and output sockets of attri2vec, for node pair inputs:
    x_inp, x_out = attri2vec.build()
    
    
    prediction = link_classification(
        output_dim=1, output_act="sigmoid", edge_embedding_method='ip'
    )(x_out)
    
    
    model = keras.Model(inputs=x_inp, outputs=prediction)
    
    model.compile(
        optimizer=keras.optimizers.Adam(lr=1e-2),
        loss=keras.losses.binary_crossentropy,
        metrics=[keras.metrics.binary_accuracy],
    )
    
    history = model.fit_generator(
        generator.flow(unsupervised_samples),
        epochs=epochs,
        verbose=1,
        use_multiprocessing=bool(int(input('Multiprocessing? 1/0: '))),
        workers=int(input('Number of workers: ')),
        shuffle=True,
    )
    print(history)
    model.save('model_walks{}len{}e{}.h5'.format(number_of_walks, length, epochs))
    return model

if __name__ == "__main__":
    model = main()
    