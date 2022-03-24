from nltk.tokenize import RegexpTokenizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

import re
import pandas as pd
import pickle
import nltk
import preprocessor as p

dataset_dir = "imdb"
# dataset_dir = "sentiment140"
# dataset_dir = "coronaNLP"

# importing the processed dataframe
df = joblib.load(f'../dataframes/df_{dataset_dir}.pkl')


X = df.iloc[:, 0]

y = df.iloc[:, 1]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# creating our pipeline that will return an estimator
pipeline = Pipeline([('clf', BernoulliNB())])

parameters = {
    'clf__fit_prior': (False, True),
    'binarize': (0.25, 0.5, 1.0),
    'clf__alpha': (1, 0.1, 0.01, 0.001)
}

clf = GridSearchCV(pipeline, param_grid=parameters, cv=5, verbose=1)

clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

print(classification_report(y_test, y_pred))

print(confusion_matrix(y_test, y_pred))

print("Best: %f using %s" % (clf.best_score_,
                             clf.best_params_))
means = clf.cv_results_['mean_test_score']
stds = clf.cv_results_['std_test_score']
params = clf.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))


acc = int(accuracy_score(y_test, y_pred)*100)

# exporting the pipeline
pickle.dump(clf.best_estimator_, open(f'../bnb_{dataset_dir}_{acc}', 'wb'))
