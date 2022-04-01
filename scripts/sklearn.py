from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from wordcloud import WordCloud

import seaborn as sns
import matplotlib.pyplot as plt
import re
import pandas as pd
import joblib
import numpy as np
import xgboost as xgb


dataset_dir = 'sentiment140'
# dataset_dir = 'imdb'
# dataset_dir = 'coronaNLP'

n_gram = (1, 2)

# importing the processed dataframe
df = joblib.load(f'../dataframes/df_{dataset_dir}.pkl')

df.head()



X = df.iloc[:, 0]


y = df.iloc[:, 1]

X, y



tfidf = joblib.load(
    f"../vectors/vectorizer_{dataset_dir}_{n_gram}.pkl")
tfidf



X = tfidf.transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42)

X_train.shape, y_train.shape


estimators = []
estimators.append(('MNB', MultinomialNB()))
estimators.append(('BNB', BernoulliNB()))
estimators.append(('XGB',
                  xgb.XGBClassifier(random_state=42, max_depth=50, use_label_encoder=False, learning_rate=0.01)))
estimators.append(('SVC', SVC(probability=True)))
estimators.append(('LRG', LogisticRegression()))


models = [
    MultinomialNB(),
    BernoulliNB(),
    xgb.XGBClassifier(max_depth=50, use_label_encoder=False),
    SVC(probability=True),
    LogisticRegression(),
    VotingClassifier(estimators = estimators, voting ='hard')
]

model_to_use = 0

model_idx = model_to_use


params = [
    {
        'fit_prior': (False, True),
        'alpha': (1, 0.1, 0.01, 0.001)
    }, 
    {
        'fit_prior': (False, True),
        'binarize': (0.25, 0.5, 1.0),
        'alpha': (1, 0.1, 0.01, 0.001)
    },
    {
        'booster': ('gbtree', 'gblinear', 'dart'),
        'eta': (0.1, 0, 25, 0.4, 0.5), 
    },
    {
        'C': ('1', '0.5', '0.25'),
        'kernel': ('rfb', 'linear', 'poly', 'sigmoid'),
    },
    {
        'penalty': ('l2', 'none'),
        'C': np.logspace(-4, 4, 10),
        'solver': ('sag', 'saga', 'newton-cg'),
        'max_iter': (100, 1000, 2500, 5000)
    }
]


clf = models[model_idx]

parameters = params[model_idx]

clf = GridSearchCV(clf, param_grid=parameters, scoring='accuracy', cv=5, verbose=True) if model_idx != -1 else clf
clf


clf = clf.fit(X_train, y_train)
clf.best_estimator_


y_pred = clf.predict(X_test)

print(classification_report(y_test, y_pred))



print("Best: %f using %s" % (clf.best_score_,
                             clf.best_params_))
means = clf.cv_results_['mean_test_score']
stds = clf.cv_results_['std_test_score']
params = clf.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))


print(confusion_matrix(y_test, y_pred))

acc = int(accuracy_score(y_test, y_pred)*100)


test_tweet = "groceri store"
vector = tfidf.transform([test_tweet])

print(clf.predict(vector))



# exporting the pipeline
joblib.dump(clf.best_estimator_,
            f'../models/mnb_{dataset_dir}_{acc}_{n_gram}.pkl')



