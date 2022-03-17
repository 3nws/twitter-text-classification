from nltk.tokenize import TweetTokenizer, RegexpTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import classification_report

import re
import pandas as pd
import pickle

# importing the dataset
# DATASET_COLUMNS  = ["sentiment", "ids", "date", "flag", "user", "tweet"]
# DATASET_ENCODING = "ISO-8859-1"
# dataset = pd.read_csv('./covid4.csv', delimiter=',', encoding=DATASET_ENCODING , names=DATASET_COLUMNS)

DATASET_ENCODING = "ISO-8859-1"
# dataset = pd.read_csv('./training.1600000.processed.noemoticon.csv', delimiter=',', encoding=DATASET_ENCODING , names=DATASET_COLUMNS)

# dataset = pd.read_csv('./Corona_NLP_train.csv', delimiter=',', encoding=DATASET_ENCODING)
dataset = pd.read_csv('./IMDB Dataset.csv', delimiter=',', encoding=DATASET_ENCODING)

dataset = dataset.drop_duplicates()

# removing the unnecessary columns.
# dataset = dataset[['sentiment','tweet']]

token = RegexpTokenizer(r'[a-zA-Z0-9]+')

tfidf = TfidfVectorizer(stop_words='english', max_features=20000, ngram_range=(1,2), tokenizer=token.tokenize)

X = dataset['review']

X = tfidf.fit_transform(X)

y = dataset['sentiment']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=15)

# X_train.shape, X_test.shape

# creating the model usin bernoulli naive bayes algorithm
BNB = BernoulliNB()

# training the model
BNB.fit(X_train, y_train)

# testing our predictions
y_pred = BNB.predict(X_test)

print(classification_report(y_test, y_pred))

# test_tweet = "scandinavia #news:  norway : it's illegal for employers to require covid  passports  denmark\
#     sweden : they won't be bringing in covid  vaccination passports  #holdtheline #enoughisenough #nomedicalapartheid #nomasks #nomorelockdowns #openforall #corona #coronavirus"
# vector = tfidf.transform([test_tweet])

# print(BNB.predict(vector))

# exporting the model and the trained vectorizer
pickle.dump(BNB, open('./models/BNB_model', 'wb'))
pickle.dump(tfidf, open('./vector/tfidf_vectorizer', 'wb'))