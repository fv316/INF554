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
