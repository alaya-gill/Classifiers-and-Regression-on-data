#!/usr/bin/env python
# coding: utf-8

# In[ ]:

from sklearn.exceptions import ConvergenceWarning 
ConvergenceWarning('ignore')



from pandas import read_csv
from sklearn.linear_model import LogisticRegression
from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import sklearn.linear_model as lm
from sklearn import metrics, model_selection
from sklearn.model_selection import KFold 
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import pickle
import warnings
warnings.filterwarnings("ignore")
data = read_csv("dataset.csv") #read dataframe


# In[ ]:




# In[ ]:


print("##############################DATA PRE-PROCESSSING############################")
data.drop(data.columns[[1, 2,8,75]], axis=1, inplace=True) #drop unnecessary columns
data.drop(data.columns[[0]], axis=1, inplace=True)


# In[ ]:


data = data.dropna() #drop rows containing nan values


# In[ ]:


method= {"Decision":1 ,"KO/TKO":0 ,"Submission":2,"Other":3}
data['Method']=data['Method'].apply(method.get) #map methods containing strings with numbers through dictionary
method= {"W":1 ,"L":0 ,"D":2,"O":3}
data['Fighter1Result']=data['Fighter1Result'].apply(method.get) #map fighter results containing strings with numbers through dictionary


# In[ ]:


df = read_csv("dataset.csv") #read data in another dataframe for input data values 
df.drop(df.columns[[2,137,138,139]], axis=1, inplace=True)
df.drop(df.columns[[0]], axis=1, inplace=True)
df=df.dropna() #drop nan value rows 


# In[ ]:


stance={"Orthodox":0,"Southpaw":1,"OpenStance":2,"Other":3,"Sideways":4,"Switch":5}
df['Fighter1Stance']=df['Fighter1Stance'].apply(stance.get) #map fighter1 stance  containing strings with numbers through dictionary
df['Fighter2Stance']=df['Fighter2Stance'].apply(stance.get) #map fighter2 stance containing strings with numbers through dictionary


# In[ ]:


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
df['WeightClass']=df['WeightClass'].apply(weightClass.get) #map weightclass containing strings with numbers through dictionary


# In[ ]:


################################PCA#####################################
print("##################Applying PCA#######################")
#using PCA to reduce the unnecessary dimensional features
model = PCA(n_components=20).fit(df)
X_pc = model.transform(df)

# number of components
n_pcs= model.components_.shape[0]

# get the index of the most important feature on EACH component
# LIST COMPREHENSION HERE
most_important = [np.abs(model.components_[i]).argmax() for i in range(n_pcs)]

x=df.filter(df[df.columns[most_important]])
print("MOST VALUABLE/IMPORTANT COLUMN NAMES:")
for i in most_important:
    print(df.columns[i])
with open('column-names.pkl','wb') as f:
    pickle.dump(df[df.columns[most_important]],f)


# In[ ]:


################################Splitting data for testing and training#####################################
print("######################Splitting data for testing and training#####################################")
X_train_org, X_test_org, y_train_org, y_test_org = train_test_split(x,data['Method'], test_size=0.2,shuffle=True)
X_train_org1, X_test_org1, y_train_org1, y_test_org1 = train_test_split(x,data['Fighter1Result'], test_size=0.2,shuffle=True)


# In[ ]:


################################LOGISTIC REGRESSION#####################################
print("######################Logistic Regression#####################################")
method_model=lm.LogisticRegression(warm_start=True, verbose = 1) #model selection Logistic Regression
kf = model_selection.RepeatedKFold(n_splits=5, n_repeats=4, random_state=None) #applying k-fold cross-validations
y=y_train_org #choosing output data column
lgreg_method_acc=[] #list for populating accuracies of validation phase 

for train_index, test_index in kf.split(X_train_org):
    X_train, X_test = X_train_org.iloc[train_index],  X_train_org.iloc[test_index] #splitting of input data rows for training and testing
    y_train, y_test = y.iloc[train_index],y.iloc[test_index] #splitting of ouput data rows for training and testing
    method_model.fit(X_train,y_train) #calculating prediction on training data
    pred = method_model.predict(X_test) #calculating prediction on testing data
    acc = accuracy_score(y_test,pred) #computing accuracy
    lgreg_method_acc.append(acc)
with open('lgreg_method.pkl','wb') as f: #saving model
    pickle.dump(method_model,f)
pred = method_model.predict(X_test_org) #calculating prediction on testing data
acc = accuracy_score(y_test_org,pred) #computing accuracy
print("Test Data Accuracy:",acc)
print("#####CONFUSION MATRIX###########")
print(confusion_matrix(y_test_org,pred))


# In[ ]:


kf = model_selection.RepeatedKFold(n_splits=5, n_repeats=4, random_state=None) 
fighter_model=lm.LogisticRegression(warm_start=True, verbose = 1)
y=y_train_org1
lgreg_fighter_acc=[]
for train_index, test_index in kf.split(X_train_org1):
    X_train, X_test = X_train_org1.iloc[train_index],  X_train_org1.iloc[test_index]
    y_train, y_test = y.iloc[train_index],y.iloc[test_index]
    fighter_model.fit(X_train,y_train)
    pred = fighter_model.predict(X_test)
    acc = accuracy_score(y_test,pred)
    lgreg_fighter_acc.append(acc)
with open('lgreg_fighter.pkl','wb') as f:
    pickle.dump(fighter_model,f)    
pred = fighter_model.predict(X_test_org1) #calculating prediction on testing data
acc = accuracy_score(y_test_org1,pred) #computing accuracy
print("Test Data Accuracy:",acc)
print("#####CONFUSION MATRIX###########")
print(confusion_matrix(y_test_org1,pred))


# In[ ]:


################################RANDOM FOREST CLASSIFIER#####################################
print("######################Random Forest Classifier#####################################")
kf = model_selection.RepeatedKFold(n_splits=5, n_repeats=4, random_state=None) #applying k-fold cross-validations
rnc_method=RandomForestClassifier(max_depth=2, random_state=0,warm_start=True) #model selection Random Forest Classifier
y=y_train_org #choosing output data column
rfc_method_acc=[] #list for populating accuracies of validation phase 
for train_index, test_index in kf.split(X_train_org):
    X_train, X_test = X_train_org.iloc[train_index],  X_train_org.iloc[test_index] #splitting of input data rows for training and testing
    y_train, y_test = y.iloc[train_index],y.iloc[test_index] #splitting of output data rows for training and testing
    rnc_method.fit(X_train,y_train) #calculating prediction on training data
    pred = rnc_method.predict(X_test) #calculating prediction on testing data
    acc = accuracy_score(y_test,pred) #computing accuracy
    rfc_method_acc.append(acc)
with open('rfc_method_model.pkl','wb') as f:
    pickle.dump(rnc_method,f)    
pred = rnc_method.predict(X_test_org) #calculating prediction on testing data
acc = accuracy_score(y_test_org,pred) #computing accuracy
print("Test Data Accuracy:",acc)
print("#####CONFUSION MATRIX###########")
print(confusion_matrix(y_test_org,pred))


# In[ ]:


kf = model_selection.RepeatedKFold(n_splits=5, n_repeats=4, random_state=None) 
rnc_fighter=RandomForestClassifier(max_depth=2, random_state=0,warm_start=True)
y=y_train_org1
rfc_fighter_acc=[]
for train_index, test_index in kf.split(X_train_org1):
    X_train, X_test = X_train_org1.iloc[train_index],  X_train_org1.iloc[test_index]
    y_train, y_test = y.iloc[train_index],y.iloc[test_index]
    rnc_fighter.fit(X_train,y_train)
    pred = rnc_fighter.predict(X_test)
    acc = accuracy_score(y_test,pred)
    rfc_fighter_acc.append(acc)
with open('rfc_fighter_model.pkl','wb') as f:
    pickle.dump(rnc_fighter,f)     
pred = rnc_fighter.predict(X_test_org1) #calculating prediction on testing data
acc = accuracy_score(y_test_org1,pred) #computing accuracy
print("Test Data Accuracy:",acc)
print("#####CONFUSION MATRIX###########")
print(confusion_matrix(y_test_org1,pred))


# In[ ]:


################################NEURAL NETWORK#####################################
kf = model_selection.RepeatedKFold(n_splits=5, n_repeats=4, random_state=None) #applying k-fold cross-validations
nn_method= MLPClassifier(random_state=1, max_iter=300,warm_start=True) #model selection Random Forest Classifier
y=y_train_org #choosing output data column
nn_method_acc=[]
for train_index, test_index in kf.split(X_train_org):
    X_train, X_test = X_train_org.iloc[train_index],  X_train_org.iloc[test_index] #splitting of input data rows for training and testing
    y_train, y_test = y.iloc[train_index],y.iloc[test_index] #splitting of output data rows for training and testing
    nn_method.fit(X_train,y_train) #calculating prediction on training data
    pred = nn_method.predict(X_test) #calculating prediction on testing data
    acc = accuracy_score(y_test,pred) #computing accuracy
    nn_method_acc.append(acc)
with open('nn_method.pkl','wb') as f:
    pickle.dump(nn_method,f)     
pred = nn_method.predict(X_test_org) #calculating prediction on testing data
acc = accuracy_score(y_test_org,pred) #computing accuracy
print("Test Data Accuracy:",acc)
print("#####CONFUSION MATRIX###########")
print(confusion_matrix(y_test_org,pred))


# In[ ]:



kf = model_selection.RepeatedKFold(n_splits=5, n_repeats=4, random_state=None) 
nn_fighter= MLPClassifier(random_state=1, max_iter=300,warm_start=True)
y=y_train_org1
nn_fighter_acc=[]
for train_index, test_index in kf.split(X_train_org1):
    X_train, X_test = X_train_org1.iloc[train_index],  X_train_org1.iloc[test_index]
    y_train, y_test = y.iloc[train_index],y.iloc[test_index]
    nn_fighter.fit(X_train,y_train)
    pred = nn_fighter.predict(X_test)
    acc = accuracy_score(y_test,pred)
    nn_fighter_acc.append(acc)
with open('nn_fighter.pkl','wb') as f:
    pickle.dump(nn_fighter,f)         
pred = nn_fighter.predict(X_test_org1) #calculating prediction on testing data
acc = accuracy_score(y_test_org1,pred) #computing accuracy
print("Test Data Accuracy:",acc)
print("#####CONFUSION MATRIX###########")
print(confusion_matrix(y_test_org1,pred))


# In[ ]:


################################PLOT OF METHODS####################
x=[i for i in range(len(nn_method_acc))] #GETTING X-AXIS FROM LENGTH OF ACCURACY'S LIST
plt.figure(figsize=(10,10)) #FIG-SIZE DEFINING
plt.ylabel('Accuracy',fontdict={'fontsize': 20, 'fontweight': 'medium'}) #X-AXIS NAME
plt.xlabel('Iterations',fontdict={'fontsize': 20, 'fontweight': 'medium'}) #Y-AXIS NAME
plt.title('Accuracies of Method',fontdict={'fontsize': 34, 'fontweight': 'medium'}) #PLOTTING TITLE
plt.plot( x, lgreg_method_acc, color='blue', linewidth=2,label="Logistic Regression") #PLOT LOGISTIC REGRESSION
plt.plot( x, rfc_method_acc,   color='red', linewidth=2,label="Random Forest") #PLOT RANDOM FOREST CLASSIFIER
plt.plot( x, nn_method_acc, color='green', linewidth=2,label="Neural Network") #PLOT NEURAL NETWORK
plt.legend()


# In[ ]:


################################PLOT OF FIGHTER RESULTS####################
plt.figure(figsize=(10,10))
plt.ylabel('Accuracy',fontdict={'fontsize': 20, 'fontweight': 'medium'})
plt.xlabel('Iterations',fontdict={'fontsize': 20, 'fontweight': 'medium'})
plt.title('Accuracies of Fighter Result',fontdict={'fontsize': 34, 'fontweight': 'medium'})
plt.plot( x, lgreg_fighter_acc, color='blue', linewidth=2,label="Logistic Regression")
plt.plot( x, rfc_fighter_acc,   color='red', linewidth=2,label="Random Forest")
plt.plot( x, nn_fighter_acc, color='green', linewidth=2,label="Neural Network")
plt.legend()


# In[ ]:





# In[ ]:




