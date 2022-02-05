from sklearn.linear_model import LogisticRegression
from nltk.tokenize import TweetTokenizer, RegexpTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from xgboost import XGBClassifier

import re
import pandas as pd
import pickle
import preprocessor as p

# importing the dataset
# DATASET_COLUMNS  = ["sentiment", "ids", "date", "flag", "user", "tweet"]
DATASET_ENCODING = "ISO-8859-1"
# dataset = pd.read_csv('./training.1600000.processed.noemoticon.csv', delimiter=',', encoding=DATASET_ENCODING , names=DATASET_COLUMNS)

# dataset = pd.read_csv('./Corona_NLP_train.csv', delimiter=',', encoding=DATASET_ENCODING)
dataset = pd.read_csv('./IMDB Dataset.csv', delimiter=',', encoding=DATASET_ENCODING)

# removing the unnecessary columns and duplicates
# dataset = dataset[['OriginalTweet','Sentiment']]
dataset.drop_duplicates()

token = RegexpTokenizer(r'[a-zA-Z0-9]+')

# tokenizing and stemming
# dataset['tweet'] = dataset['OriginalTweet'].apply(p.clean)
# dataset['sentiment'] = dataset['Sentiment']

token = RegexpTokenizer(r'[a-zA-Z0-9]+')

tfidf = TfidfVectorizer(stop_words='english', max_features=20000, ngram_range=(1,2), tokenizer=token.tokenize)

X = dataset['review']

X = tfidf.fit_transform(X)

y = dataset['sentiment']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)

clf = XGBClassifier(random_state=42, learning_rate=0.01)

clf.fit(X_train,y_train)

# testing our predictions
y_pred = clf.predict(X_test)

print(classification_report(y_test, y_pred))

# test_tweet = "scandinavia #news:  norway : it's illegal for employers to require covid  passports  denmark\
#     sweden : they won't be bringing in covid  vaccination passports  #holdtheline #enoughisenough #nomedicalapartheid #nomasks #nomorelockdowns #openforall #corona #coronavirus"
# vector = tfidf.transform([test_tweet])

# print(clf.predict(vector))

# exporting the model and the trained vectorizer
pickle.dump(clf, open('./models/xgboost', 'wb'))
pickle.dump(tfidf, open('./vector/tfidf_vectorizer_xg', 'wb'))