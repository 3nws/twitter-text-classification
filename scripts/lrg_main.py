from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix
from wordcloud import WordCloud
from sklearn.metrics import accuracy_score

import seaborn as sns
import matplotlib.pyplot as plt
import re
import pandas as pd
import joblib
import numpy as np



# dataset_dir = 'sentiment140'
# dataset_dir = 'imdb'
dataset_dir = 'coronaNLP'

# n_gram = (1, 1)
# n_gram = (1, 2)
n_gram = (2, 2)

# importing the processed dataframe
df = joblib.load(f'./dataframes/df_{dataset_dir}.pkl')

df.head()





X = df.iloc[:, 0]


y = df.iloc[:, 1]

X, y


sns.countplot(y)



plt.figure(figsize=(15, 15))
options = [1, 2]
cond = df.iloc[:, 1].isin(options)
result = df[cond].iloc[:, 0].values
wc = WordCloud(max_words=2000, width=1600,
               height=800).generate(" ".join(result))
plt.imshow(wc, interpolation='bilinear')
plt.title('Positive words cloud')



if dataset_dir == 'coronaNLP':
    plt.figure(figsize=(15, 15))
    options = [0]
    cond = df.iloc[:, 1].isin(options)
    result = df[cond].iloc[:, 0].values
    wc = WordCloud(max_words=2000, width=1600,
                   height=800).generate(" ".join(result))
    plt.imshow(wc, interpolation='bilinear')
    plt.title('Neutral words cloud')



if dataset_dir == 'imdb' or dataset_dir == 'sentiment140':
    options = [0]
else:
    options = [-1, -2]

plt.figure(figsize=(15, 15))
cond = df.iloc[:, 1].isin(options)
result = df[cond].iloc[:, 0].values
wc = WordCloud(max_words=2000, width=1600,
               height=800).generate(" ".join(result))
plt.imshow(wc, interpolation='bilinear')
plt.title('Negative words cloud')




tfidf = joblib.load(
    f"./vectors/vectorizer_{dataset_dir}_{n_gram}.pkl")
tfidf


tfidf.vocabulary_, tfidf.idf_



X = tfidf.transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

X_train.shape, y_train.shape



# creating our pipeline that will return an estimator
pipeline = Pipeline([('clf', LogisticRegression(random_state=42, multi_class='ovr', n_jobs=4, verbose=True))])




parameters = {
    'clf__penalty': ('l2', 'none'),
    'clf__solver': ('sag', 'saga')
    }

clf = GridSearchCV(pipeline, param_grid=parameters, scoring='accuracy', cv=5, verbose=True)
clf



clf = clf.fit(X_train, y_train)




y_pred = clf.predict(X_test)

print(classification_report(y_test, y_pred))



print("Best: %f using %s" % (clf.best_score_, 
    clf.best_params_))
means = clf.cv_results_['mean_test_score']
stds = clf.cv_results_['std_test_score']
params = clf.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))




from sklearn.metrics import confusion_matrix

print(confusion_matrix(y_test, y_pred))

acc = int(accuracy_score(y_test, y_pred)*100)



test_tweet = "groceri store"
vector = tfidf.transform([test_tweet])

print(clf.predict(vector))




# exporting the pipeline
joblib.dump(clf.best_estimator_, f'./models/lrg_{dataset_dir}_{acc}_{n_gram}.pkl')


