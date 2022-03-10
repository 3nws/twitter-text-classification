from nltk.tokenize import TweetTokenizer, RegexpTokenizer
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import RegexpTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
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
p.set_options(p.OPT.URL, p.OPT.MENTION, p.OPT.HASHTAG)


nltk.download('stopwords')

stemmer = SnowballStemmer("english", ignore_stopwords=True)
token = RegexpTokenizer(r'[a-zA-Z0-9]+')

DATASET_ENCODING = "ISO-8859-1"

dataset = pd.read_csv('./Corona_NLP_train.csv',
                      delimiter=',', encoding=DATASET_ENCODING)

token = RegexpTokenizer(r'[a-zA-Z0-9]+')

dataset.head()


def preprocess_tweets(tweet):
    tweet = p.clean(tweet)
    tokens = tweet.split()
    stemmed_tokens = [stemmer.stem(token) for token in tokens]
    return ' '.join(stemmed_tokens)


X = dataset['OriginalTweet']

X = X.apply(preprocess_tweets)

y = dataset['Sentiment']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=5)

# X_train.shape, X_test.shape

# creating our pipeline that will return an estimator
pipeline = Pipeline([('tfidf', TfidfVectorizer(max_features=20000, ngram_range=(
    1, 2), tokenizer=token.tokenize)), ('clf', SVC(probability=True))])

pipeline.fit(X_train, y_train)

y_pred = pipeline.predict(X_test)

print(classification_report(y_test, y_pred))

acc = int(accuracy_score(y_test, y_pred)*100)

# exporting the model and the trained vectorizer
pickle.dump(svc, open(f'./models/SVC_model_{acc}', 'wb'))
pickle.dump(tfidf, open(f'./vector/tfidf_vectorizer_{acc}', 'wb'))
