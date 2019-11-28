# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 18:14:45 2019

@author: fvice
"""

import csv
import os
from tqdm import tqdm
import spacy
nlpfr = spacy.load('fr_core_news_md')
import pickle
import multiprocessing

def setup():
    with open("testing.txt", "r") as f:
        reader = csv.reader(f)
        testing_set  = list(reader)
    
    testing_set = [element[0].split(" ") for element in testing_set]
    
    with open("training.txt", "r") as f:
        reader = csv.reader(f)
        training_set  = list(reader)
    
    training_set = [element[0].split(" ") for element in training_set]
    
    directory = r"node_information\\text\\"
    node_info = []
    IDs = []
    for filename in tqdm(os.listdir(directory)):
        with open(directory + filename, 'r', encoding='UTF-8', errors='ignore') as f:
            reader = f.read()
            node_info.append(tuple((filename[:-4], reader)))
            IDs += [filename[:-4]]

    return node_info, IDs, testing_set, training_set

# compute spaCy vector of each paper
def create_pickle(text):
    pickledDir = r"node_information\\pickle\\"
    nlp.max_length = 5000000
    try:
        doc = nlp(text[1])
        with open(pickledDir + text[0] + ".PICKLE",'+wb') as f:
            pickle.dump(doc, f)
        f.close()
        return True, text[0]
    except:
        return False, text[0]
    
    
    
node_info, IDs, testing_set, training_set = setup()

for text in tqdm(node_info):
    pickledDir = r"node_information\\pickle\\"
    nlp.max_length = 5000000
    doc = nlp.pipe(text[1])
    with open(pickledDir + text[0] + ".PICKLE",'+wb') as f:
        pickle.dump(doc, f)
    f.close()
    
for doc in nlp.pipe(node_info, disable=["tagger", "parser", "ner"]):
    pickledDir = r"node_information\\pickle\\"
    nlp.max_length = 5000000
    doc = nlp(text[1])
    with open(pickledDir + text[0] + ".PICKLE",'+wb') as f:
        pickle.dump(doc, f)
    f.close()

    
    
    

if __name__ == "__main_":
    node_info, IDs, testing_set, training_set = setup()
    parallel = False
    if parallel == True:
        pool = multiprocessing.Pool(processes=5)
        failed_ids = []
        for success, ids in tqdm(pool.imap_unordered(
                create_pickle, node_info), total=len(node_info)):
            if success == False:    
                failed_ids.append(ids)
        print(failed_ids)
        
    else:
        failed_ids = []
        for text in tqdm(node_info):
            success, ids = create_pickle(text)
            if success == False:    
                failed_ids.append(ids)
        print(failed_ids)
