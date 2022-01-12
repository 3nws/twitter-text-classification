from nltk.tokenize import TweetTokenizer, RegexpTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix

import re
import pandas as pd
import pickle
import nltk
import preprocessor as p

# importing the dataset
# DATASET_COLUMNS  = ["sentiment", "ids", "date", "flag", "user", "tweet"]
DATASET_ENCODING = "ISO-8859-1"
# dataset = pd.read_csv('./training.1600000.processed.noemoticon.csv', delimiter=',', encoding=DATASET_ENCODING , names=DATASET_COLUMNS)

dataset = pd.read_csv('./Corona_NLP_train.csv', delimiter=',', encoding=DATASET_ENCODING)

# removing the unnecessary columns and duplicates
dataset = dataset[['OriginalTweet','Sentiment']]
dataset.drop_duplicates()

token = RegexpTokenizer(r'[a-zA-Z0-9]+')

# tokenizing and stemming
dataset['tweet'] = dataset['OriginalTweet'].apply(p.clean)
dataset['sentiment'] = dataset['Sentiment']

# dataset.head()

X = dataset['tweet']

y = dataset['sentiment']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)

# X_train.shape, X_test.shape

# creating our pipeline that will return an estimator
pipeline = Pipeline([('tfidf', TfidfVectorizer(ngram_range=(1,2), tokenizer=token.tokenize)), ('clf', MultinomialNB())])

parameters = {
    'tfidf__max_features': (10000, 20000),
    'clf__fit_prior': (False,True),
    'clf__alpha': (1, 0.1, 0.01, 0.001, 0.0001, 0.00001)  
    }

clf = GridSearchCV(pipeline, param_grid=parameters, cv=5)

clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

print(classification_report(y_test, y_pred))

print("Best: %f using %s" % (clf.best_score_, 
    clf.best_params_))
means = clf.cv_results_['mean_test_score']
stds = clf.cv_results_['std_test_score']
params = clf.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))

# exporting the pipeline
pickle.dump(clf, open('./models/mnb_pipeline_grid', 'wb'))