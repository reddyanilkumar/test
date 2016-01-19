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














In [12]:

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns

In [3]:

user_df= pd.read_csv("train/users.csv")

In [4]:

sub_df= pd.read_csv("train/submissions.csv")

In [5]:

prob_df= pd.read_csv("train/problems.csv")

In [6]:

user_df.count()

Out[6]:

user_id         62530
skills          62271
solved_count    62530
attempts        62530
user_type       34215
dtype: int64

In [7]:

prob_df.count()

Out[7]:

problem_id      1002
level            768
accuracy        1002
solved_count    1002
error_count     1002
rating          1002
tag1             725
tag2             473
tag3             181
tag4              46
tag5               6
dtype: int64

In [8]:

sub_df.count()

Out[8]:

user_id           1198131
problem_id        1198131
solved_status     1198131
result            1198131
language_used     1198131
execution_time    1198131
dtype: int64

In [15]:

sns.factorplot('language_used',data=sub_df,kind="count",size=20)

Out[15]:

<seaborn.axisgrid.FacetGrid at 0x7f1084b66fd0>

In [25]:

merged_inner = pd.merge(left=user_df,right=sub_df,left_on='user_id', right_on='user_id')

In [47]:

merged_inner.drop('language_used', axis=1, inplace=True)

In [50]:

merged_inner.drop_duplicates(inplace=True)

In [35]:

merged_inner.drop('user_type', axis=1, inplace=True)

In [51]:

merged_inner.head(20)

Out[51]:
	user_id 	skills 	solved_count 	attempts 	problem_id
0 	1427919 	C++ 	0 	11 	913736
11 	1034704 	C 	3 	11 	906741
15 	1034704 	C 	3 	11 	909152
17 	1034704 	C 	3 	11 	909145
18 	903633 	C|PHP|Java|Python|Text|JavaScript|C++|Perl|C#|... 	66 	260 	907301
21 	903633 	C|PHP|Java|Python|Text|JavaScript|C++|Perl|C#|... 	66 	260 	920126
23 	903633 	C|PHP|Java|Python|Text|JavaScript|C++|Perl|C#|... 	66 	260 	903666
29 	903633 	C|PHP|Java|Python|Text|JavaScript|C++|Perl|C#|... 	66 	260 	906315
34 	903633 	C|PHP|Java|Python|Text|JavaScript|C++|Perl|C#|... 	66 	260 	903790
38 	903633 	C|PHP|Java|Python|Text|JavaScript|C++|Perl|C#|... 	66 	260 	903639
42 	903633 	C|PHP|Java|Python|Text|JavaScript|C++|Perl|C#|... 	66 	260 	926728
43 	903633 	C|PHP|Java|Python|Text|JavaScript|C++|Perl|C#|... 	66 	260 	903788
46 	903633 	C|PHP|Java|Python|Text|JavaScript|C++|Perl|C#|... 	66 	260 	907300
47 	903633 	C|PHP|Java|Python|Text|JavaScript|C++|Perl|C#|... 	66 	260 	903786
64 	903633 	C|PHP|Java|Python|Text|JavaScript|C++|Perl|C#|... 	66 	260 	905488
69 	903633 	C|PHP|Java|Python|Text|JavaScript|C++|Perl|C#|... 	66 	260 	903905
74 	903633 	C|PHP|Java|Python|Text|JavaScript|C++|Perl|C#|... 	66 	260 	903657
79 	903633 	C|PHP|Java|Python|Text|JavaScript|C++|Perl|C#|... 	66 	260 	903787
80 	903633 	C|PHP|Java|Python|Text|JavaScript|C++|Perl|C#|... 	66 	260 	903658
103 	903633 	C|PHP|Java|Python|Text|JavaScript|C++|Perl|C#|... 	66 	260 	903903
In [49]:

merged_inner.corr()

Out[49]:
	user_id 	solved_count 	attempts 	problem_id
user_id 	1.000000 	-0.117815 	-0.163276 	0.348288
solved_count 	-0.117815 	1.000000 	0.716177 	0.040674
attempts 	-0.163276 	0.716177 	1.000000 	-0.019638
problem_id 	0.348288 	0.040674 	-0.019638 	1.000000
In [53]:

merged_inner1 = pd.merge(left=merged_inner,right=prob_df,left_on='problem_id', right_on='problem_id')

In [55]:

merged_inner1

Out[55]:
	user_id 	skills 	solved_count_x 	attempts 	problem_id 	level 	accuracy 	solved_count_y 	error_count 	rating 	tag1 	tag2 	tag3 	tag4 	tag5
0 	1427919 	C++ 	0 	11 	913736 	M 	0.21 	524 	7868 	4.0 	Ad-Hoc 	Dynamic Programming 	Algorithms 	NaN 	NaN
1 	1187633 	Python|C++ 	14 	20 	913736 	M 	0.21 	524 	7868 	4.0 	Ad-Hoc 	Dynamic Programming 	Algorithms 	NaN 	NaN
2 	1165859 	C 	2 	14 	913736 	M 	0.21 	524 	7868 	4.0 	Ad-Hoc 	Dynamic Programming 	Algorithms 	NaN 	NaN
3 	1034822 	C|C++ 	6 	11 	913736 	M 	0.21 	524 	7868 	4.0 	Ad-Hoc 	Dynamic Programming 	Algorithms 	NaN 	NaN
4 	1384250 	C 	6 	11 	913736 	M 	0.21 	524 	7868 	4.0 	Ad-Hoc 	Dynamic Programming 	Algorithms 	NaN 	NaN
5 	1428122 	C 	0 	1 	913736 	M 	0.21 	524 	7868 	4.0 	Ad-Hoc 	Dynamic Programming 	Algorithms 	NaN 	NaN
6 	1166111 	JavaScript 	0 	1 	913736 	M 	0.21 	524 	7868 	4.0 	Ad-Hoc 	Dynamic Programming 	Algorithms 	NaN 	NaN
7 	1166158 	Ruby|Java|C++ 	18 	42 	913736 	M 	0.21 	524 	7868 	4.0 	Ad-Hoc 	Dynamic Programming 	Algorithms 	NaN 	NaN
8 	1144005 	C 	5 	18 	913736 	M 	0.21 	524 	7868 	4.0 	Ad-Hoc 	Dynamic Programming 	Algorithms 	NaN 	NaN
9 	1035196 	C|C++ 	15 	19 	913736 	M 	0.21 	524 	7868 	4.0 	Ad-Hoc 	Dynamic Programming 	Algorithms 	NaN 	NaN
10 	1035226 	C|Java|C++ 	21 	44 	913736 	M 	0.21 	524 	7868 	4.0 	Ad-Hoc 	Dynamic Programming 	Algorithms 	NaN 	NaN
11 	1035270 	C 	5 	14 	913736 	M 	0.21 	524 	7868 	4.0 	Ad-Hoc 	Dynamic Programming 	Algorithms 	NaN 	NaN
12 	1166351 	Python|C 	6 	66 	913736 	M 	0.21 	524 	7868 	4.0 	Ad-Hoc 	Dynamic Programming 	Algorithms 	NaN 	NaN
13 	1166430 	Java 	3 	2 	913736 	M 	0.21 	524 	7868 	4.0 	Ad-Hoc 	Dynamic Programming 	Algorithms 	NaN 	NaN
14 	1384339 	C 	5 	19 	913736 	M 	0.21 	524 	7868 	4.0 	Ad-Hoc 	Dynamic Programming 	Algorithms 	NaN 	NaN
15 	1166476 	C|C++ 	33 	27 	913736 	M 	0.21 	524 	7868 	4.0 	Ad-Hoc 	Dynamic Programming 	Algorithms 	NaN 	NaN
16 	1428624 	C 	0 	1 	913736 	M 	0.21 	524 	7868 	4.0 	Ad-Hoc 	Dynamic Programming 	Algorithms 	NaN 	NaN
17 	1035532 	Python|C#|C|Java|C++ 	54 	30 	913736 	M 	0.21 	524 	7868 	4.0 	Ad-Hoc 	Dynamic Programming 	Algorithms 	NaN 	NaN
18 	1035541 	C 	7 	24 	913736 	M 	0.21 	524 	7868 	4.0 	Ad-Hoc 	Dynamic Programming 	Algorithms 	NaN 	NaN
19 	1035592 	Java 	4 	14 	913736 	M 	0.21 	524 	7868 	4.0 	Ad-Hoc 	Dynamic Programming 	Algorithms 	NaN 	NaN
20 	1166670 	C++ 	0 	1 	913736 	M 	0.21 	524 	7868 	4.0 	Ad-Hoc 	Dynamic Programming 	Algorithms 	NaN 	NaN
21 	1035620 	C|C++ 	3 	13 	913736 	M 	0.21 	524 	7868 	4.0 	Ad-Hoc 	Dynamic Programming 	Algorithms 	NaN 	NaN
22 	1166694 	Python 	3 	3 	913736 	M 	0.21 	524 	7868 	4.0 	Ad-Hoc 	Dynamic Programming 	Algorithms 	NaN 	NaN
23 	1078552 	C 	5 	8 	913736 	M 	0.21 	524 	7868 	4.0 	Ad-Hoc 	Dynamic Programming 	Algorithms 	NaN 	NaN
24 	947488 	C++ 	86 	77 	913736 	M 	0.21 	524 	7868 	4.0 	Ad-Hoc 	Dynamic Programming 	Algorithms 	NaN 	NaN
25 	904686 	Python|Java 	15 	21 	913736 	M 	0.21 	524 	7868 	4.0 	Ad-Hoc 	Dynamic Programming 	Algorithms 	NaN 	NaN
26 	1166874 	C|C++ 	6 	6 	913736 	M 	0.21 	524 	7868 	4.0 	Ad-Hoc 	Dynamic Programming 	Algorithms 	NaN 	NaN
27 	1166919 	C|Java 	1 	5 	913736 	M 	0.21 	524 	7868 	4.0 	Ad-Hoc 	Dynamic Programming 	Algorithms 	NaN 	NaN
28 	1036231 	C|C++ 	14 	3 	913736 	M 	0.21 	524 	7868 	4.0 	Ad-Hoc 	Dynamic Programming 	Algorithms 	NaN 	NaN
29 	1122345 	C++ 	28 	18 	913736 	M 	0.21 	524 	7868 	4.0 	Ad-Hoc 	Dynamic Programming 	Algorithms 	NaN 	NaN
... 	... 	... 	... 	... 	... 	... 	... 	... 	... 	... 	... 	... 	... 	... 	...
421945 	1125077 	Python|C|C++ 	66 	97 	935846 	NaN 	0.96 	11 	2 	2.5 	NaN 	NaN 	NaN 	NaN 	NaN
421946 	1131415 	Python|C|Python 3|C++ 	127 	166 	935846 	NaN 	0.96 	11 	2 	2.5 	NaN 	NaN 	NaN 	NaN 	NaN
421947 	1135792 	C#|C|C++ 	46 	72 	935846 	NaN 	0.96 	11 	2 	2.5 	NaN 	NaN 	NaN 	NaN 	NaN
421948 	1009369 	Python|C|Java|C++ 	149 	163 	935846 	NaN 	0.96 	11 	2 	2.5 	NaN 	NaN 	NaN 	NaN 	NaN
421949 	1337012 	Python|C|C++ 	38 	5 	935846 	NaN 	0.96 	11 	2 	2.5 	NaN 	NaN 	NaN 	NaN 	NaN
421950 	1275822 	Python|C|C++ 	68 	103 	935846 	NaN 	0.96 	11 	2 	2.5 	NaN 	NaN 	NaN 	NaN 	NaN
421951 	1276572 	C|Java|C++ 	46 	5 	935846 	NaN 	0.96 	11 	2 	2.5 	NaN 	NaN 	NaN 	NaN 	NaN
421952 	1409736 	C|Java|Python|C++|C#|Pascal|Haskell|JavaScript... 	670 	419 	935846 	NaN 	0.96 	11 	2 	2.5 	NaN 	NaN 	NaN 	NaN 	NaN
421953 	1155218 	C|Java|Python|C++|C#|PHP|Ruby 	243 	187 	935846 	NaN 	0.96 	11 	2 	2.5 	NaN 	NaN 	NaN 	NaN 	NaN
421954 	1099417 	C|Java|Python|C++|C#|PHP 	70 	130 	935846 	NaN 	0.96 	11 	2 	2.5 	NaN 	NaN 	NaN 	NaN 	NaN
421955 	1293851 	Python|C|Java|C++ 	245 	144 	935846 	NaN 	0.96 	11 	2 	2.5 	NaN 	NaN 	NaN 	NaN 	NaN
421956 	942937 	C|C++ 	54 	56 	906717 	NaN 	0.90 	1 	1 	0.0 	NaN 	NaN 	NaN 	NaN 	NaN
421957 	955565 	C++ 	6 	16 	906717 	NaN 	0.90 	1 	1 	0.0 	NaN 	NaN 	NaN 	NaN 	NaN
421958 	954615 	C++ 	30 	16 	906717 	NaN 	0.90 	1 	1 	0.0 	NaN 	NaN 	NaN 	NaN 	NaN
421959 	959602 	Python|Java|C++ 	49 	66 	906717 	NaN 	0.90 	1 	1 	0.0 	NaN 	NaN 	NaN 	NaN 	NaN
421960 	967270 	C|C++ 	8 	8 	906717 	NaN 	0.90 	1 	1 	0.0 	NaN 	NaN 	NaN 	NaN 	NaN
421961 	945555 	C++ 	35 	85 	906717 	NaN 	0.90 	1 	1 	0.0 	NaN 	NaN 	NaN 	NaN 	NaN
421962 	1207522 	Java 	65 	50 	913270 	NaN 	0.86 	4 	3 	0.0 	NaN 	NaN 	NaN 	NaN 	NaN
421963 	1081207 	Python|Ruby|C|Java|C++ 	190 	172 	913270 	NaN 	0.86 	4 	3 	0.0 	NaN 	NaN 	NaN 	NaN 	NaN
421964 	952684 	Python|C|Java|Python 3|C++ 	505 	469 	913270 	NaN 	0.86 	4 	3 	0.0 	NaN 	NaN 	NaN 	NaN 	NaN
421965 	1088137 	Python|C#|C|Java|C++ 	98 	194 	913270 	NaN 	0.86 	4 	3 	0.0 	NaN 	NaN 	NaN 	NaN 	NaN
421966 	1088166 	Python|C|Java|C++ 	516 	457 	913270 	NaN 	0.86 	4 	3 	0.0 	NaN 	NaN 	NaN 	NaN 	NaN
421967 	1099144 	C|C++ 	53 	38 	913270 	NaN 	0.86 	4 	3 	0.0 	NaN 	NaN 	NaN 	NaN 	NaN
421968 	1100558 	C#|C|Java|JavaScript(Rhino)|C++ 	177 	112 	913270 	NaN 	0.86 	4 	3 	0.0 	NaN 	NaN 	NaN 	NaN 	NaN
421969 	1009212 	Python 	2 	2 	913270 	NaN 	0.86 	4 	3 	0.0 	NaN 	NaN 	NaN 	NaN 	NaN
421970 	1012026 	C|Java|C#|C++|Python|Objective-C 	196 	151 	913270 	NaN 	0.86 	4 	3 	0.0 	NaN 	NaN 	NaN 	NaN 	NaN
421971 	1081960 	Python|C++ 	20 	16 	913270 	NaN 	0.86 	4 	3 	0.0 	NaN 	NaN 	NaN 	NaN 	NaN
421972 	1016727 	C|Java|Scala|Python|C++|Perl|C#|Go|Ruby 	204 	141 	913270 	NaN 	0.86 	4 	3 	0.0 	NaN 	NaN 	NaN 	NaN 	NaN
421973 	968968 	Python|C|C++ 	117 	158 	937778 	NaN 	0.96 	13 	2 	4.1 	NaN 	NaN 	NaN 	NaN 	NaN
421974 	977865 	C++ 	5 	6 	917940 	NaN 	1.00 	1 	0 	0.0 	NaN 	NaN 	NaN 	NaN 	NaN

421975 rows × 15 columns
In [52]:

merged_inner.count()

Out[52]:

user_id         421975
skills          421975
solved_count    421975
attempts        421975
problem_id      421975
dtype: int64

In [43]:

merged_inner1.drop('solved_count_y', axis=1, inplace=True)

In [ ]:

 


