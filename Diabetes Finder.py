#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Importing all the libraries
import pandas as pd 
import numpy as np
import seaborn as sns 
import matplotlib.pyplot as plt 
get_ipython().run_line_magic('matplotlib', 'inline')

from sklearn.linear_model import LogisticRegression


# In[2]:


df=pd.read_csv('E:\Project\diabetes.csv')
df.head(10)


# In[3]:


df.value_counts()


# In[4]:


sns.heatmap(df.isnull())


# In[37]:


sns.countplot(x='Outcome',data=df)


# In[36]:


plt.figure(figsize=(20,10))
b0 =  df['BMI'][(df['BMI'] < 16)]
b1 =  df['BMI'][(df['BMI'] > 16) & (df["BMI"] <= 20)]
b2 =  df['BMI'][(df['BMI'] > 21) & (df["BMI"] <= 25)]
b3 =  df['BMI'][(df['BMI'] > 26) & (df["BMI"] <= 30)]
b4 =  df['BMI'][(df['BMI'] > 31) & (df["BMI"] <= 36)]
b5 =  df['BMI'][(df['BMI'] > 36)]
ssx = [ "Skinny", "Under Weight", "Fit","Ower weight","Obese","Morbid Obese"]
ssy = [len(b0.values), len(b1.values), len(b2.values), len(b3.values), len(b4.values), len(b5.values)]
plt.title ("No of people in each weight category")
sns.barplot(x=ssx, y=ssy, palette="Set2")

plt.xlabel("Weight category")
plt.ylabel("No of people")
plt.show()


# In[44]:


r1=df[df.Outcome==1]
r1['Age'].plot.hist(bins=10,figsize=(10,5),edgecolor='white')


# In[45]:


r1.head()


# In[ ]:





# In[5]:


# Finding pairwise correlation
correlation=df.corr()
sns.heatmap(correlation,xticklabels=correlation.columns,yticklabels=correlation.columns)
print(correlation)


# In[6]:


# TRAINING AND TESTING
# TRAINING - 650 
# TESTING - 100
# MAKING PREDICTIONS - Last 17

train=df[:650]
test=df[650:750]
check=df[750:]

trainlabel=np.asarray(train['Outcome'])
traindata=np.asarray(train.drop('Outcome',1))
testlabel=np.asarray(test['Outcome'])
testdata=np.asarray(test.drop('Outcome',1))

# Checking if the mean is 0 and standard deviation is 1
means=np.mean(traindata,axis=0)
stds=np.std(traindata,axis=0)
traindata=(traindata-means)/stds
testdata=(testdata-means)/stds

           # TRAINING 
    
Diabetes_Check=LogisticRegression(solver='lbfgs', max_iter=1000)
Diabetes_Check.fit(traindata,trainlabel)
accuracy=Diabetes_Check.score(testdata,testlabel)

print('Accuracy is ',accuracy*100,' Percent')





# In[7]:


# TESTING

import joblib


# In[8]:


joblib.dump([Diabetes_Check,means,stds],"diabetesmodel.pkl")


# In[9]:


diabetesloadedmodel,means,stds = joblib.load('diabetesmodel.pkl')
accuracycheck=diabetesloadedmodel.score(testdata,testlabel)
print('The accuracy is',accuracycheck*100,'Percent')


# In[10]:


# Making Predctions with are final unused data

check.head()


# In[32]:


# Sample no 750
Sample_data=check[3:4]


# In[33]:


# Preparing the sample 

sampledata=np.asarray(Sample_data.drop('Outcome',1))
sampledata=(sampledata-means)/stds


#Prediction

prediction_probablity = diabetesloadedmodel.predict_proba(sampledata)
prediction = diabetesloadedmodel.predict(sampledata)
#print('The Probablity of having diabetes is ',prediction_probablity)
if prediction.any()==1:
    print('You have diabetes')
    print('Consult Your Doctor Immediatly for severity and medication')
    print('Take Care as Diabetes if uncontrolled can be Fatal ')
else :
    print ('You dont have diabetes')
    print ('You are healthy')
    print ('To prevent diabetes click on the link below')
    print ('https://link.springer.com/content/pdf/10.1007/s11606-013-2548-4.pdf')




# In[ ]:




