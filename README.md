# test
test

# coding: utf-8

# In[216]:

import matplotlib.pyplot as plt
import numpy as np 
import pandas as pd 
from sklearn.cross_validation import ShuffleSplit, train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier 
from pprint import pprint 
get_ipython().magic(u'matplotlib inline')


# In[217]:

traincsv=pd.read_csv('train.csv')
testcsv=pd.read_csv('test.csv')


# In[218]:

traincsv.Credit_History.fillna(0,inplace=True)

traincsv.Loan_Amount_Term.fillna(360,inplace=True)

traincsv.LoanAmount.fillna( traincsv.LoanAmount.mean(),inplace=True)


# In[219]:

testcsv.Credit_History.fillna(0,inplace=True)

testcsv.Loan_Amount_Term.fillna(360,inplace=True)

testcsv.LoanAmount.fillna( testcsv.LoanAmount.mean(),inplace=True)


# In[221]:

rf_initial=ExtraTreesClassifier(n_estimators=50,random_state=1)
rf_initial.fit(traincsv[['Credit_History','LoanAmount','Loan_Amount_Term','ApplicantIncome','CoapplicantIncome']],traincsv[traincsv.columns[12]])
print "Initial Traincsv score: %.2f" %rf_initial.score(traincsv[['Credit_History','LoanAmount','Loan_Amount_Term','ApplicantIncome','CoapplicantIncome']],traincsv[traincsv.columns[12]])


# In[222]:

pred_test = rf_initial.predict(testcsv[['Credit_History','LoanAmount','Loan_Amount_Term','ApplicantIncome','CoapplicantIncome']])


# In[223]:

from sklearn.cross_validation import cross_val_score
scores = cross_val_score(rf_initial,testcsv[['Credit_History','LoanAmount','Loan_Amount_Term','ApplicantIncome','CoapplicantIncome']],pred_test,cv=25)
print (scores)
print (np.mean(scores))


# In[224]:

df1 = testcsv['Loan_ID']


# In[225]:

df2 = pd.Series(pred_test,index=df1.index)


# In[226]:

result = pd.concat([df1, df2], axis=1)


# In[227]:

result.columns = ['Loan_ID','Loan_Status']


# In[228]:

result.to_csv('Submission.csv',index=False)


# In[ ]:



