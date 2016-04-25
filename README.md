
In [1]:

import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
import pandas as pd
import seaborn as sns
%matplotlib inline

In [58]:

#df_train = pd.read_csv('Train.csv')
df_test = pd.read_csv('Test.csv')

In [3]:

# Get basic information and print it.
n_samples = df_train.shape[0]
n_features = df_train.shape[1]
print "Number of training samples: {}".format(n_samples)
print "Number of features: {}".format(n_features)

Number of training samples: 68353
Number of features: 12

In [4]:

perc_mvs_per_col = (df_train.isnull().sum() / n_samples) * 100
print perc_mvs_per_col[perc_mvs_per_col > 0]

Customer_Location            16.963411
Total_Past_Communications     9.984931
Total_Links                   3.220049
Total_Images                  2.453440
dtype: float64

In [89]:

df_train = df_train.fillna(value = -999)
df_test = df_test.fillna(value = -999)

In [5]:

df_train.head()

Out[5]:
	Email_ID 	Email_Type 	Subject_Hotness_Score 	Email_Source_Type 	Customer_Location 	Email_Campaign_Type 	Total_Past_Communications 	Time_Email_sent_Category 	Word_Count 	Total_Links 	Total_Images 	Email_Status
0 	EMA00081000034500 	1 	2.2 	2 	E 	2 	33 	1 	440 	8 	0 	0
1 	EMA00081000045360 	2 	2.1 	1 	-999 	2 	15 	2 	504 	5 	0 	0
2 	EMA00081000066290 	2 	0.1 	1 	B 	3 	36 	2 	962 	5 	0 	1
3 	EMA00081000076560 	1 	3.0 	2 	E 	2 	25 	2 	610 	16 	0 	0
4 	EMA00081000109720 	1 	0.0 	2 	C 	3 	18 	2 	947 	4 	0 	0
In [7]:

for i in df_train.columns:
    x=df_train[i].unique()
    print i,x

Email_ID ['EMA00081000034500' 'EMA00081000045360' 'EMA00081000066290' ...,
 'EMA00089998436500' 'EMA00089999168800' 'EMA00089999316900']
Email_Type [1 2]
Subject_Hotness_Score [ 2.2  2.1  0.1  3.   0.   1.5  3.2  0.7  2.   0.5  0.2  1.   4.   1.9  1.1
  1.6  0.3  2.3  1.4  1.7  2.8  1.2  0.8  0.6  4.2  1.8  2.4  0.9  1.3  3.3
  2.6  3.1  4.1  2.9  2.7  0.4  3.5  3.7  2.5  3.8  3.9  3.4  4.6  4.5  3.6
  4.4  4.7  5.   4.3  4.8  4.9]
Email_Source_Type [2 1]
Customer_Location ['E' -999 'B' 'C' 'G' 'D' 'F' 'A']
Email_Campaign_Type [2 3 1]
Total_Past_Communications [  33.   15.   36.   25.   18. -999.   34.   21.   40.   27.   24.   42.
   11.   23.   37.   35.   51.    9.   39.   31.   50.   30.   14.   45.
   53.   28.    7.   38.   52.   22.   43.   12.   16.   20.   41.   56.
   26.   29.    5.   32.   44.   10.   17.   46.   47.   48.    8.   49.
   13.    0.    6.   55.   19.   60.   59.   61.   54.   62.   57.   64.
   58.   65.   66.   67.   63.]
Time_Email_sent_Category [1 2 3]
Word_Count [ 440  504  962  610  947  416  116 1241  655  744  931  550  565  700  694
 1061  623  560 1082  684  733 1122  649  778  855  704  339  988  389  636
  812  880  254  490  771  353  484  922  275  392  520  458  630 1140  892
  578  311  352  902  795  577  653  524  904 1014  314 1103  721  220  673
  873  763  542  760  741  518  424   40  282  608  713  939  470  462  842
  934  806 1303  366  912  419  868 1229 1157  982 1102  841  593  251  152
  933 1216 1271  827 1189  730 1038 1280   79 1296 1203  662  145   99  631
  977  187 1173  987   67  678  605  768  661  190  521  132  770  722  840
  233  253  751  146   51 1262 1289  796  757  789 1288  773  737 1060 1252
  972  967  960  519  954  194  186 1316   50 1310  782 1309  946 1315  966
  786]
Total_Links [   8.    5.   16.    4.   11.    6.   21. -999.   31.    3.    9.   26.
   10.    7.    2.   14.   13.   28.   24.   41.    1.   19.   12.   18.
   15.   46.   17.   36.   29.   23.   39.   49.   25.   34.   20.   44.
   22.   33.]
Total_Images [   0.    2.    4.   16.   13.   15.    5.   28.   10.    3.    8. -999.
    1.   24.   12.   11.    6.   20.   21.   14.    7.   18.    9.   25.
   17.   19.   23.   27.   22.   43.   34.   26.   30.   29.   38.   40.
   32.   35.   31.   39.   36.   33.   37.   45.   41.   44.]
Email_Status [0 1 2]

In [5]:

category_variables = {'Time_Email_sent_Category','Email_Campaign_Type','Customer_Location','Email_Source_Type','Email_Type'}
numeric_variables = {'Total_Images','Total_Links','Word_Count','Total_Past_Communications','Subject_Hotness_Score'}

In [6]:

for var in numeric_variables:
    df_train[var] = df_train[var].fillna(value = df_train[var].mean())
    df_test[var] = df_test[var].fillna(value = df_test[var].mean())

In [7]:

df_train = df_train.fillna(value = -999)
df_test = df_test.fillna(value = -999)
for var in category_variables:
    lb = preprocessing.LabelEncoder()
    full_data = pd.concat((df_train[var],df_test[var]),axis=0).astype('str')
    lb.fit( full_data )
    df_train[var] = lb.transform(df_train[var].astype('str'))
    df_test[var] = lb.transform(df_test[var].astype('str'))

In [15]:

sns.factorplot('Customer_Location',data=df_train,kind="count")

Out[15]:

<seaborn.axisgrid.FacetGrid at 0x7f3600297890>

In [18]:

sns.factorplot('Total_Images',data=df_train,kind="count")

Out[18]:

<seaborn.axisgrid.FacetGrid at 0x7f35f1ca2e90>

In [6]:

from sklearn import preprocessing
le = preprocessing.LabelEncoder()

In [90]:

for i in ['Customer_Location']:
    le.fit(pd.concat([df_train,df_test])[i])
    df_train[i]  = le.transform(df_train[i])
    df_test[i]  = le.transform(df_test[i])

In [8]:

x = df_train.drop(['Email_Status','Email_ID'],axis=1)

In [33]:

from sklearn.cross_validation import train_test_split
X_train,X_test,Y_train,y_test = train_test_split(x,df_train.Email_Status)

In [35]:

df_train.columns

Out[35]:

Index([u'Email_ID', u'Email_Type', u'Subject_Hotness_Score',
       u'Email_Source_Type', u'Customer_Location', u'Email_Campaign_Type',
       u'Total_Past_Communications', u'Time_Email_sent_Category',
       u'Word_Count', u'Total_Links', u'Total_Images', u'Email_Status'],
      dtype='object')

In [47]:

from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=15,random_state=33)

In [48]:

rf.fit(X_train,Y_train)
rf.score(X_train,Y_train)

Out[48]:

0.99225577403245946

In [45]:

from sklearn.cross_validation import cross_val_score

In [59]:

scores = cross_val_score(rf,x,df_train.Email_Status,cv=5)
print (scores)
print (np.mean(scores))

[ 0.80251609  0.80184332  0.79921001  0.79868325  0.79991221]
0.800432974787

In [14]:

from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.lda import LDA
import xgboost as xgb
from sklearn.qda import QDA
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.calibration import CalibratedClassifierCV
params_xgb = {'colsample_bytree': 0.2, 'silent': 1, 'nthread': 8, 'min_child_weight': 10, 'n_estimators':100, 'subsample': 1.0,'learning_rate': 0.1, 'seed': 10, 'max_depth': 15, 'gamma': 0.75}
classifiers = [ 
    ExtraTreesClassifier(n_estimators=10),
    RandomForestClassifier(n_estimators=10),
    KNeighborsClassifier(100),
    LDA(),
    QDA(),
    GaussianNB(),
    DecisionTreeClassifier(),
    GradientBoostingClassifier(n_estimators=100, learning_rate=0.04,max_depth=7, random_state=12),
    #xgb.XGBClassifier(n_estimators=100)
]

/home/anil/anaconda/lib/python2.7/site-packages/sklearn/lda.py:4: DeprecationWarning: lda.LDA has been moved to discriminant_analysis.LinearDiscriminantAnalysis in 0.17 and will be removed in 0.19
  "in 0.17 and will be removed in 0.19", DeprecationWarning)
/home/anil/anaconda/lib/python2.7/site-packages/sklearn/qda.py:4: DeprecationWarning: qda.QDA has been moved to discriminant_analysis.QuadraticDiscriminantAnalysis in 0.17 and will be removed in 0.19.
  "in 0.17 and will be removed in 0.19.", DeprecationWarning)

In [15]:

for classifier in classifiers:
    print classifier.__class__.__name__
    scores = cross_val_score(classifier,x,df_train.Email_Status,cv=5)
    print (scores)
    print (np.mean(scores))

ExtraTreesClassifier

---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
<ipython-input-15-2e105d4557c1> in <module>()
      1 for classifier in classifiers:
      2     print classifier.__class__.__name__
----> 3     scores = cross_val_score(classifier,x,df_train.Email_Status,cv=5)
      4     print (scores)
      5     print (np.mean(scores))

NameError: name 'cross_val_score' is not defined

In [16]:

a = RandomForestClassifier(n_estimators=10)
b = KNeighborsClassifier(100)
c = LDA()
d = GaussianNB()

In [49]:

params_xgb = {'colsample_bytree':0.5, 'silent': 1, 'nthread': 8, 'min_child_weight': 8, 'n_estimators':100, 'subsample': 1.0,'learning_rate': 0.1, 'seed': 10, 'max_depth': 8}
bst = xgb.XGBClassifier(**params_xgb)

In [34]:

clf2 = GradientBoostingClassifier(n_estimators=100, learning_rate=0.04,max_depth=7, random_state=12)

In [114]:

from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin
import numpy as np
import operator

class EnsembleClassifier(BaseEstimator, ClassifierMixin):
    """
    Ensemble classifier for scikit-learn estimators.

    Parameters
    ----------

    clf : `iterable`
      A list of scikit-learn classifier objects.
    weights : `list` (default: `None`)
      If `None`, the majority rule voting will be applied to the predicted class labels.
        If a list of weights (`float` or `int`) is provided, the averaged raw probabilities (via `predict_proba`)
        will be used to determine the most confident class label.

    """
    def __init__(self, clfs, weights=None):
        self.clfs = clfs
        self.weights = weights

    def fit(self, X, y):
        """
        Fit the scikit-learn estimators.

        Parameters
        ----------

        X : numpy array, shape = [n_samples, n_features]
            Training data
        y : list or numpy array, shape = [n_samples]
            Class labels

        """
        for clf in self.clfs:
            clf.fit(X, y)

    def predict(self, X):
        """
        Parameters
        ----------

        X : numpy array, shape = [n_samples, n_features]

        Returns
        ----------

        maj : list or numpy array, shape = [n_samples]
            Predicted class labels by majority rule

        """

        self.classes_ = np.asarray([clf.predict(X) for clf in self.clfs])
        if self.weights:
            avg = self.predict_proba(X)

            maj = np.apply_along_axis(lambda x: max(enumerate(x), key=operator.itemgetter(1))[0], axis=1, arr=avg)

        else:
            maj = np.asarray([np.argmax(np.bincount(self.classes_[:,c])) for c in range(self.classes_.shape[1])])

        return maj

    def predict_proba(self, X):

        """
        Parameters
        ----------

        X : numpy array, shape = [n_samples, n_features]

        Returns
        ----------

        avg : list or numpy array, shape = [n_samples, n_probabilities]
            Weighted average probability for each class per sample.

        """
        self.probas_ = [clf.predict_proba(X) for clf in self.clfs]
        avg = np.average(self.probas_, axis=0, weights=self.weights)

        return avg

In [158]:

df = pd.DataFrame(columns=('w1', 'w2', 'w3', 'mean', 'std'))

i = 0
for w1 in range(1,4):
    for w2 in range(1,4):
        for w3 in range(1,4):

            if len(set((w1,w2,w3))) == 1: # skip if all weights are equal
                continue

            eclf = EnsembleClassifier(clfs=[a,b, clf], weights=[w1,w2,w3])
            scores = cross_validation.cross_val_score(
                                            estimator=eclf,
                                            X=x,
                                            y=df_train.Email_Status,
                                            cv=5,
                                            scoring='accuracy',
                                            n_jobs=1)

            df.loc[i] = [w1, w2, w3, scores.mean(), scores.std()]
            i += 1

df.sort(columns=['mean', 'std'], ascending=False)

Out[158]:
	w1 	w2 	w3 	mean 	std
16 	3 	1 	1 	0.806680 	0.001236
17 	3 	1 	2 	0.806563 	0.001672
8 	2 	1 	1 	0.806329 	0.000927
22 	3 	3 	1 	0.806285 	0.000990
18 	3 	1 	3 	0.805583 	0.001222
19 	3 	2 	1 	0.805583 	0.000968
20 	3 	2 	2 	0.805583 	0.000960
23 	3 	3 	2 	0.805027 	0.000928
9 	2 	1 	2 	0.804968 	0.000954
10 	2 	1 	3 	0.804910 	0.000567
21 	3 	2 	3 	0.804895 	0.000584
13 	2 	3 	1 	0.804822 	0.000858
14 	2 	3 	2 	0.804749 	0.000739
11 	2 	2 	1 	0.804690 	0.001151
5 	1 	3 	1 	0.804647 	0.000654
0 	1 	1 	2 	0.804544 	0.000396
15 	2 	3 	3 	0.804529 	0.000528
3 	1 	2 	2 	0.804383 	0.000302
4 	1 	2 	3 	0.804295 	0.000396
1 	1 	1 	3 	0.804281 	0.000736
12 	2 	2 	3 	0.804178 	0.000394
7 	1 	3 	3 	0.804091 	0.000566
2 	1 	2 	1 	0.804032 	0.000660
6 	1 	3 	2 	0.803856 	0.000485
In [138]:

from sklearn import cross_validation
np.random.seed(123)
eclf = EnsembleClassifier(clfs=[a,b,clf,c,d], weights=[1,1,1,1,1])

for clf, label in zip([a,b,clf,c,d,eclf], ['DT', 'KNN', 'XGB','LDA','GAUSS','Ensemble']):

    scores = cross_validation.cross_val_score(clf,x,df_train.Email_Status,cv=5)
    print("Accuracy: %0.6f (+/- %0.2f) [%s]" % (scores.mean(), scores.std(), label))

Accuracy: 0.799380 (+/- 0.00) [DT]
Accuracy: 0.803856 (+/- 0.00) [KNN]
Accuracy: 0.805305 (+/- 0.00) [XGB]
Accuracy: 0.804076 (+/- 0.00) [LDA]
Accuracy: 0.801852 (+/- 0.00) [GAUSS]

---------------------------------------------------------------------------
KeyboardInterrupt                         Traceback (most recent call last)
<ipython-input-138-507a495985c6> in <module>()
      5 for clf, label in zip([a,b,clf,c,d,eclf], ['DT', 'KNN', 'XGB','LDA','GAUSS','Ensemble']):
      6 
----> 7     scores = cross_validation.cross_val_score(clf,x,df_train.Email_Status,cv=5)
      8     print("Accuracy: %0.6f (+/- %0.2f) [%s]" % (scores.mean(), scores.std(), label))

/home/anil/anaconda/lib/python2.7/site-packages/sklearn/cross_validation.pyc in cross_val_score(estimator, X, y, scoring, cv, n_jobs, verbose, fit_params, pre_dispatch)
   1431                                               train, test, verbose, None,
   1432                                               fit_params)
-> 1433                       for train, test in cv)
   1434     return np.array(scores)[:, 0]
   1435 

/home/anil/anaconda/lib/python2.7/site-packages/sklearn/externals/joblib/parallel.pyc in __call__(self, iterable)
    802             self._iterating = True
    803 
--> 804             while self.dispatch_one_batch(iterator):
    805                 pass
    806 

/home/anil/anaconda/lib/python2.7/site-packages/sklearn/externals/joblib/parallel.pyc in dispatch_one_batch(self, iterator)
    660                 return False
    661             else:
--> 662                 self._dispatch(tasks)
    663                 return True
    664 

/home/anil/anaconda/lib/python2.7/site-packages/sklearn/externals/joblib/parallel.pyc in _dispatch(self, batch)
    568 
    569         if self._pool is None:
--> 570             job = ImmediateComputeBatch(batch)
    571             self._jobs.append(job)
    572             self.n_dispatched_batches += 1

/home/anil/anaconda/lib/python2.7/site-packages/sklearn/externals/joblib/parallel.pyc in __init__(self, batch)
    181         # Don't delay the application, to avoid keeping the input
    182         # arguments in memory
--> 183         self.results = batch()
    184 
    185     def get(self):

/home/anil/anaconda/lib/python2.7/site-packages/sklearn/externals/joblib/parallel.pyc in __call__(self)
     70 
     71     def __call__(self):
---> 72         return [func(*args, **kwargs) for func, args, kwargs in self.items]
     73 
     74     def __len__(self):

/home/anil/anaconda/lib/python2.7/site-packages/sklearn/cross_validation.pyc in _fit_and_score(estimator, X, y, scorer, train, test, verbose, parameters, fit_params, return_train_score, return_parameters, error_score)
   1529             estimator.fit(X_train, **fit_params)
   1530         else:
-> 1531             estimator.fit(X_train, y_train, **fit_params)
   1532 
   1533     except Exception as e:

<ipython-input-114-afa0c9bdd9e7> in fit(self, X, y)
     37         """
     38         for clf in self.clfs:
---> 39             clf.fit(X, y)
     40 
     41     def predict(self, X):

<ipython-input-114-afa0c9bdd9e7> in fit(self, X, y)
     37         """
     38         for clf in self.clfs:
---> 39             clf.fit(X, y)
     40 
     41     def predict(self, X):

<ipython-input-114-afa0c9bdd9e7> in fit(self, X, y)
     37         """
     38         for clf in self.clfs:
---> 39             clf.fit(X, y)
     40 
     41     def predict(self, X):

<ipython-input-114-afa0c9bdd9e7> in fit(self, X, y)
     37         """
     38         for clf in self.clfs:
---> 39             clf.fit(X, y)
     40 
     41     def predict(self, X):

<ipython-input-114-afa0c9bdd9e7> in fit(self, X, y)
     37         """
     38         for clf in self.clfs:
---> 39             clf.fit(X, y)
     40 
     41     def predict(self, X):

/home/anil/anaconda/lib/python2.7/site-packages/xgboost/sklearn.pyc in fit(self, X, y, sample_weight, eval_set, eval_metric, early_stopping_rounds, verbose)
    341                               early_stopping_rounds=early_stopping_rounds,
    342                               evals_result=evals_result, feval=feval,
--> 343                               verbose_eval=verbose)
    344 
    345         if evals_result:

/home/anil/anaconda/lib/python2.7/site-packages/xgboost/training.pyc in train(params, dtrain, num_boost_round, evals, obj, feval, maximize, early_stopping_rounds, evals_result, verbose_eval, learning_rates, xgb_model)
    119     if not early_stopping_rounds:
    120         for i in range(num_boost_round):
--> 121             bst.update(dtrain, i, obj)
    122             nboost += 1
    123             if len(evals) != 0:

/home/anil/anaconda/lib/python2.7/site-packages/xgboost/core.pyc in update(self, dtrain, iteration, fobj)
    692 
    693         if fobj is None:
--> 694             _check_call(_LIB.XGBoosterUpdateOneIter(self.handle, iteration, dtrain.handle))
    695         else:
    696             pred = self.predict(dtrain)

KeyboardInterrupt: 

In [18]:

from sklearn.cross_validation import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier

In [39]:

    np.random.seed(0) # seed to shuffle the train set

    n_folds = 10
    verbose = True

    skf = list(StratifiedKFold(y, n_folds))

    #clfs = [RandomForestClassifier(n_estimators=100, n_jobs=-1, criterion='gini'),
    #        RandomForestClassifier(n_estimators=100, n_jobs=-1, criterion='entropy'),
    #        ExtraTreesClassifier(n_estimators=100, n_jobs=-1, criterion='gini'),
    #        ExtraTreesClassifier(n_estimators=100, n_jobs=-1, criterion='entropy'),
     #       GradientBoostingClassifier(learning_rate=0.05, subsample=0.5, max_depth=6, n_estimators=50)]
    clfs = [c,clf2,bst]

    print "Creating train and test sets for blending."
    
    dataset_blend_train = np.zeros((x.shape[0], len(clfs)))
    dataset_blend_test = np.zeros((df_test.shape[0], len(clfs)))
    
    for j, clf in enumerate(clfs):
        print j, clf
        dataset_blend_test_j = np.zeros((df_test.shape[0], len(skf)))
        for i, (train, test) in enumerate(skf):
            print "Fold", i
            X_train = x.iloc[train]
            y_train = y.iloc[train]
            X_test = x.iloc[test]
            y_test = y.iloc[test]
            clf.fit(X_train, y_train)
            y_submission = clf.predict(X_test)
            dataset_blend_train[test, j] = y_submission
            dataset_blend_test_j[:, i] = clf.predict(df_test)
        dataset_blend_test[:,j] = dataset_blend_test_j.mean(1)

    print
    print "Blending."
    clf = xgb.XGBClassifier(n_estimators=100)
    clf.fit(dataset_blend_train,y)
    y_submission = clf.predict(dataset_blend_test)

Creating train and test sets for blending.
0 LinearDiscriminantAnalysis(n_components=None, priors=None, shrinkage=None,
              solver='svd', store_covariance=False, tol=0.0001)
Fold 0
Fold 1
Fold 2
Fold 3
Fold 4
Fold 5
Fold 6
Fold 7
Fold 8
Fold 9
1 GradientBoostingClassifier(init=None, learning_rate=0.04, loss='deviance',
              max_depth=7, max_features=None, max_leaf_nodes=None,
              min_samples_leaf=1, min_samples_split=2,
              min_weight_fraction_leaf=0.0, n_estimators=100,
              presort='auto', random_state=12, subsample=1.0, verbose=0,
              warm_start=False)
Fold 0
Fold 1
Fold 2
Fold 3
Fold 4
Fold 5
Fold 6
Fold 7
Fold 8
Fold 9
2 XGBClassifier(base_score=0.5, colsample_bylevel=1, colsample_bytree=1,
       gamma=0, learning_rate=0.1, max_delta_step=0, max_depth=8,
       min_child_weight=8, missing=None, n_estimators=100, nthread=8,
       objective='multi:softprob', reg_alpha=0, reg_lambda=1,
       scale_pos_weight=1, seed=10, silent=1, subsample=1.0)
Fold 0
Fold 1
Fold 2
Fold 3
Fold 4
Fold 5
Fold 6
Fold 7
Fold 8
Fold 9

Blending.

In [62]:

y_submission = (y_submission - y_submission.min()) / (y_submission.max() - y_submission.min())

In [40]:

np.unique(y_submission)

Out[40]:

array([0, 1])

In [32]:

%run utils.py

In [19]:

from sklearn.cluster import KMeans
kmean = KMeans(n_clusters=  10,max_iter =3000, random_state=2)
cluster_id_train = kmean.fit_predict(x1)
cluster_id_test = kmean.predict(df_test1)
    
x['cluster_id']  = cluster_id_train
df_test['cluster_id']  = cluster_id_test
   
print 'over'

over

In [118]:

params_xgb = {'colsample_bytree':0.5, 'silent': 1, 'nthread': 8, 'min_child_weight': 8, 'n_estimators':100, 'subsample': 1.0,'learning_rate': 0.1, 'seed': 10, 'max_depth': 8}
bst = xgb.XGBClassifier(**params_xgb)

In [160]:

from sklearn.ensemble import AdaBoostClassifier
clf1 = AdaBoostClassifier(n_estimators=100)

In [19]:

clf1 = RandomForestClassifier(n_estimators=300, n_jobs=-1)

In [67]:

from sklearn.ensemble import VotingClassifier

In [75]:

eclf = VotingClassifier(estimators=[('ext',a), ('knn',b),('LDA',c),('GAUSS',d),('xgb',calibrated_clf)], voting='soft')

In [150]:

calibrated_clf = CalibratedClassifierCV(bst, method='isotonic', cv=5)

In [20]:

y = df_train.Email_Status

In [151]:

scores = cross_val_score(calibrated_clf,x,df_train.Email_Status,cv=5)
print (np.mean(scores))

0.815896913521

In [31]:

x1

Out[31]:

array([-0.95662516, -0.72108637,  0.9644881 , ..., -0.77261048,
        0.12170086,  1.24419041])

In [56]:

from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
#X_scaled = preprocessing.scale(x)


scl = StandardScaler()
x1 = scl.fit_transform(x)
df_test2 = scl.transform(df_test)

In [34]:

X_scaled.mean(axis=0)

Out[34]:

array([ -1.72218813e-15,   6.73632973e-17,  -5.82140642e-16,
        -2.11264788e-15,  -1.04824493e-15,   8.75900720e-17,
        -1.11562673e-16,   3.20353909e-16,   9.49839384e-16,
        -9.07888683e-16,   7.65115558e-16])

In [88]:

gbm1 = GradientBoostingClassifier(random_state = 42, learning_rate = 0.08,subsample = 1,  min_samples_split = 30 , max_features = 'sqrt' ,  n_estimators = 500 , max_depth =  6)

In [149]:

params_xgb = {'colsample_bytree':0.5, 'silent': 1, 'nthread': 8, 'min_child_weight': 5, 'n_estimators':100, 'subsample': 1.0,'learning_rate': 0.1, 'seed': 10, 'max_depth': 8}
bst = xgb.XGBClassifier(**params_xgb)

In [148]:

scores = cross_val_score(bst,x,df_train.Email_Status,cv=5)
print (np.mean(scores))

0.815809139683

In [98]:

def importances(estimator, col_array, title): 
    # Calculate the feature ranking - Top 10
    importances = estimator.feature_importances_
    indices = np.argsort(importances)[::-1]
    print "%s Top 20 Important Features\n" %title
    for f in range(10):
        print("%d. %s (%f)" % (f + 1, col_array.columns[indices[f]], importances[indices[f]]))
#Mean Feature Importance
    print "\nMean Feature Importance %.6f" %np.mean(importances)

In [99]:

importances(a,x, "Cover Type (Initial RF)")

Cover Type (Initial RF) Top 20 Important Features

1. Word_Count (0.224534)
2. Total_Past_Communications (0.186752)
3. Subject_Hotness_Score (0.128480)
4. Total_Links (0.121808)
5. Total_Images (0.101514)
6. Customer_Location (0.098361)
7. Time_Email_sent_Category (0.052051)
8. Email_Campaign_Type (0.046454)
9. Email_Source_Type (0.025611)
10. Email_Type (0.014435)

Mean Feature Importance 0.100000

In [87]:

calibrated_clf.fit(x,df_train.Email_Status)

Out[87]:

CalibratedClassifierCV(base_estimator=XGBClassifier(base_score=0.5, colsample_bylevel=1, colsample_bytree=0.5,
       gamma=0, learning_rate=0.1, max_delta_step=0, max_depth=8,
       min_child_weight=8, missing=None, n_estimators=100, nthread=8,
       objective='binary:logistic', reg_alpha=0, reg_lambda=1,
       scale_pos_weight=1, seed=11, silent=1, subsample=1.0),
            cv=5, method='isotonic')

In [68]:

eclf1 = VotingClassifier(estimators=[('lr',calibrated_d), ('rf', calibrated_a), ('xgb', calibrated_b),('knn',calibrated_c)], voting='soft')

In [65]:

calibrated_a = CalibratedClassifierCV(a, method='isotonic', cv=5)
calibrated_b = CalibratedClassifierCV(b, method='isotonic', cv=5)
calibrated_c = CalibratedClassifierCV(c, method='isotonic', cv=5)
calibrated_d = CalibratedClassifierCV(bst, method='isotonic', cv=5)

In [ ]:

 

In [59]:

df_test1 = df_test.copy()

In [60]:

df_test.drop('Email_ID',inplace=True,axis=1)

In [91]:

preds = calibrated_clf.predict(df_test)

In [62]:

df_test1.drop('Email_Status',inplace=True,axis=1)

---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
<ipython-input-62-4661e04466d8> in <module>()
----> 1 df_test1.drop('Email_Status',inplace=True,axis=1)

/home/anil/anaconda/lib/python2.7/site-packages/pandas/core/generic.pyc in drop(self, labels, axis, level, inplace, errors)
   1595                 new_axis = axis.drop(labels, level=level, errors=errors)
   1596             else:
-> 1597                 new_axis = axis.drop(labels, errors=errors)
   1598             dropped = self.reindex(**{axis_name: new_axis})
   1599             try:

/home/anil/anaconda/lib/python2.7/site-packages/pandas/core/index.pyc in drop(self, labels, errors)
   2568         if mask.any():
   2569             if errors != 'ignore':
-> 2570                 raise ValueError('labels %s not contained in axis' % labels[mask])
   2571             indexer = indexer[~mask]
   2572         return self.delete(indexer)

ValueError: labels ['Email_Status'] not contained in axis

In [92]:

df_test1['Email_Status'] = preds

In [93]:

df_test1.to_csv('submission-seed11.csv',columns=['Email_ID','Email_Status'],index=False)

In [15]:

df_test1.columns

Out[15]:

Index([u'Email_ID', u'Email_Type', u'Subject_Hotness_Score',
       u'Email_Source_Type', u'Customer_Location', u'Email_Campaign_Type',
       u'Total_Past_Communications', u'Time_Email_sent_Category',
       u'Word_Count', u'Total_Links', u'Total_Images'],
      dtype='object')

In [ ]:

 

