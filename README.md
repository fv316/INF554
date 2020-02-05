# INF554 Project Repository for Team ==Baseline==

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

# Running the Notebooks and Files:

Install the necassary libraries. 

+ Original data files are expected in the main directory.

Files/folders: 

+ **Preprocessing.ipynb** preprocessing of text data to create created stemmed corpus. Must be run first.
+ **INF554_Simple.ipynb** run after **Preprocessing.ipynb**, calculates the main graph and text based features.
+ **INF554_Node2Vec.ipynb** run after **INF554_Simple.ipynb**, calculated node2vec embeddings.
+ **INF554_Ensemble.ipynb** run last. This file created various models and combined their results in a model ensemble to improve performance.
+ **./pickles** stored the pickles features.
