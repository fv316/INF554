# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 22:16:31 2019

@author: fvice
"""

import numpy as np
import pandas as pd
from tqdm import tqdm
import seaborn as sns
import os
import networkx as nx
import pdb
import pickle
from collections import Counter
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from sklearn.feature_extraction import text as fe
from sklearn.decomposition import NMF, LatentDirichletAllocation

def main():
    
    corpus_matrix_path = r"pickles/stemmed_corpus_word_matrix.PICKLE"
    if os.path.exists(corpus_matrix_path):
        with open(corpus_matrix_path, 'rb') as f:
            corpus_word_matrix = pickle.load(f)
        f.close()
    else:
        vectorizer1 = fe.CountVectorizer()
        corpus_word_matrix = vectorizer1.fit_transform(tqdm(stemmed_corpus, position=0, leave=True))
        with open(corpus_matrix_path, '+wb') as f:
            pickle.dump(corpus_word_matrix, f)
        f.close()
    
    corpus_matrix_path = r"pickles/stemmed_corpus_tfidf_matrix.PICKLE"
    if os.path.exists(corpus_matrix_path):
        with open(corpus_matrix_path, 'rb') as f:
            corpus_tfidf_matrix = pickle.load(f)
        f.close()
    else:
        vectorizer3 = fe.TfidfVectorizer()
        corpus_tfidf_matrix = vectorizer3.fit_transform(tqdm(stemmed_corpus, position=0, leave=True))
        with open(corpus_matrix_path, '+wb') as f:
            pickle.dump(corpus_tfidf_matrix, f)
        f.close()
    
    lda_path = r"pickles/stemmed_lda_128_matrix.PICKLE"
    if os.path.exists(lda_path):
        with open(lda_path, 'rb') as f:
            lda = pickle.load(f)
        f.close()
    else:

        lda = LatentDirichletAllocation(n_components=128, max_iter=5,
                                        learning_method='online',
                                        learning_offset=50.,
                                        random_state=0).fit_transform(corpus_word_matrix)
        
        with open(lda_path, '+wb') as f:
            pickle.dump(lda, f)
        f.close()
    
    lda_path = r"pickles/stemmed_lda_64_matrix.PICKLE"
    if os.path.exists(lda_path):
        with open(lda_path, 'rb') as f:
            lda = pickle.load(f)
        f.close()
    else:

        lda = LatentDirichletAllocation(n_components=64, max_iter=5,
                                        learning_method='online',
                                        learning_offset=50.,
                                        random_state=0).fit_transform(corpus_word_matrix)
        
        with open(lda_path, '+wb') as f:
            pickle.dump(lda, f)
        f.close()
    










if __name__ == "__main__":
    model = main()