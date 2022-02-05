from nltk.tokenize import TweetTokenizer, RegexpTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from xgboost import XGBClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score

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

tfidf = TfidfVectorizer(stop_words='english', max_features=20000, ngram_range=(1,2))

X = dataset['tweet']

X = tfidf.fit_transform(X)

y = dataset['sentiment']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)

# X_train.shape, X_test.shape

# creating the model usin multiple estimators

estimators = []
estimators.append(('LR', 
                  XGBClassifier(random_state=42, learning_rate=0.01)))
estimators.append(('SVC', SVC(gamma ='auto', probability = True)))
estimators.append(('DTC', MNB()))

# for name,classifier in estimators:
#     classifier = classifier
#     classifier.fit(X_train, y_train.ravel())
#     predictions = classifier.predict(X_test)
#     predictions_df[name.strip(" :")] = predictions
#     print(name, accuracy_score(y_test, predictions))

VTC = VotingClassifier(estimators = estimators, voting ='hard')

# training the model
VTC.fit(X_train, y_train)

# testing our predictions
y_pred = VTC.predict(X_test)

print(classification_report(y_test, y_pred))

# exporting the model and the trained vectorizer
pickle.dump(VTC, open('./models/voting', 'wb'))
pickle.dump(tfidf, open('./vector/tfidf_vectorizer_voting', 'wb'))