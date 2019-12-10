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
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from sklearn.cluster import KMeans, MiniBatchKMeans


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

    lsa_path = r"pickles/stemmed_lsa_5000_matrix.PICKLE"
    if os.path.exists(lsa_path):
        with open(lsa_path, 'rb') as f:
            corpus_lsa_matrix = pickle.load(f)
        f.close()
    else:
        
        print("Performing dimensionality reduction using LSA")
        svd = TruncatedSVD(5000)
        normalizer = Normalizer(copy=False)
        lsa = make_pipeline(svd, normalizer)
        
        corpus_lsa_matrix = lsa.fit_transform(corpus_tfidf_matrix)
        
        explained_variance = svd.explained_variance_ratio_.sum()
        print("Explained variance of the SVD step: {}%".format(
            int(explained_variance * 100)))
        
        with open(lsa_path, '+wb') as f:
            pickle.dump(corpus_lsa_matrix, f)
        f.close()






if __name__ == "__main__":
    model = main()