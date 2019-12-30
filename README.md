# Baseline
INF554 project repository for team ==Baseline==

# Onboarding
Create virtualenv with
```
$ virtualenv -p python3 venv
```
Activate it and install depenencies
```
$ . ./env.sh  or $ . ./venv/bin/activate
$ pip install -r requirements.txt
```
Download the data from: 

[Link](https://www.kaggle.com/c/17455/download-all) 

Alternatively:
```
$ kaggle competitions download -c link-prediction-data-challenge-2019
```
Whenever you install something new, add it with
```
$ pip freeze > requirements.txt
```

# About Running the Python Notebooks and Folder Structure:

Install the necassary libraries. 

### Original data files expected in the main directory. 
'''
|-- pickles
|   |-- example_pickle.PICKLE
|-- node_information
|   |-- text
	|   +-- 0.txt
	|   +-- 1.txt
	|   +-- ...
|-- training.txt
|-- testing.txt
|-- INF554_Simple.ipynb
|-- ...
'''
Files/folders: 

+ **Preprocessing.ipynb** preprocessing of text data to create created stemmed corpus. Must be run first.
+ **INF554_Simple.ipynb** run after **Preprocessing.ipynb**, calculates the main graph and text based features.
+ **INF554_Node2Vec.ipynb** run after **INF554_Simple.ipynb**, calculated node2vec embeddings.
+ **INF554_Ensemble.ipynb** run last. This file created various models and combined their results in a model ensemble to improve performance.
+ **./pickles** stored the pickles features.
