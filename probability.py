import pickle 
import pandas as pd
import numpy as np
from pandas import read_csv

with open('lgreg_method.pkl', 'rb') as f: #loading logistic regression model
    lgreg_method= pickle.load(f)

with open('column-names.pkl', 'rb') as f: #loading most important column names
    column_names = pickle.load(f)

cols=column_names.columns[:]
col_names=[]
for i in cols:
    col_names.append(i)

data = read_csv("prediction.csv") #read dataframe

########################################################## DATA PRE-PROCESSING ###################################################################
data = data.dropna() 
data.drop(data.columns[[0]], axis=1, inplace=True)
data_cols=data.columns[:]
data_col_names=[]
for i in data_cols:
    data_col_names.append(i)


if 'WeightClass' in data_col_names:
    weightClass={"Women's Featherweight":0,
    "Bantamweight":1,
    "Welterweight":2,
    "Featherweight":3,
    "Middleweight":4,
    "Flyweight":5,
    "Light Heavyweight":6,
    "Other":7,
    "Heavyweight":8,
    "Lightweight":9,
    "Women's Strawweight":10,
    "Women's Flyweight":11,
    "Women's Bantamweight":12}
    data['WeightClass']=data['WeightClass'].apply(weightClass.get)

x=data.filter(data[col_names])

d=x.iloc[np.random.choice(np.arange(len(x)), size=1)] #random row data from your given dataset

########################################################## Predicting Probabilities ###################################################################

l=lgreg_method.predict_proba(d)[0]
lr_method= {"Decision":l[1],"KO/TKO":l[0] ,"Submission":l[2],"Other":l[3]}

print("Logistic Regression Method Probabilities: ",lr_method)

with open('lgreg_fighter.pkl', 'rb') as f: #loading logistic regression model of fighter result
    lgreg_fighter= pickle.load(f)

l=lgreg_fighter.predict_proba(d)[0]
lr_fighter= {"Win":l[1] ,"Lose":l[0] ,"Draw":l[2],"Other":l[3]}
print("Logistic Regression Fighter Probabilities: ",lr_fighter)

with open('rfc_method_model.pkl', 'rb') as f: #loading Random Forest Classifier model for method
    rfc_method= pickle.load(f)

l=rfc_method.predict_proba(d)[0]
r_method= {"Decision":l[1],"KO/TKO":l[0] ,"Submission":l[2],"Other":l[3]}
print("Random Forest Method Probabilities: ",r_method)

with open('rfc_fighter_model.pkl', 'rb') as f: #loading Random Forest Classifier model for fighter
    rfc_fighter= pickle.load(f)

l=rfc_fighter.predict_proba(d)[0]
r_fighter= {"Win":l[1] ,"Lose":l[0] ,"Draw":l[2],"Other":l[3]}
print("Random Forest Classifier Fighter Probabilities: ",r_fighter)

with open('nn_method.pkl', 'rb') as f: #loading Neural Network model for method
    nn_method= pickle.load(f)

l=nn_method.predict_proba(d)[0]
n_method= {"Decision":l[1],"KO/TKO":l[0] ,"Submission":l[2],"Other":l[3]}
print("Neural Network Method Probabilities: ",n_method)

with open('nn_fighter.pkl', 'rb') as f: #loading Neural Network model for fighter
    nn_fighter= pickle.load(f)

l=nn_fighter.predict_proba(d)[0]
n_fighter= {"Win":l[1] ,"Lose":l[0] ,"Draw":l[2],"Other":l[3]}
print("Neural Network Fighter Probabilities: ",n_fighter)

