from sklearn.linear_model import LogisticRegression
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import TweetTokenizer, RegexpTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV, ShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

import re
import pandas as pd
import pickle
import nltk
import numpy as np
import preprocessor as p
p.set_options(p.OPT.URL, p.OPT.MENTION, p.OPT.HASHTAG)


nltk.download('stopwords')

stemmer = SnowballStemmer("english", ignore_stopwords=True)
token = RegexpTokenizer(r'[a-zA-Z0-9]+')

# importing the dataset
DATASET_ENCODING = "ISO-8859-1"
# DATASET_COLUMNS  = ["sentiment", "ids", "date", "flag", "user", "tweet"]
# dataset = pd.read_csv('./training.1600000.processed.noemoticon.csv', delimiter=',', encoding=DATASET_ENCODING , names=DATASET_COLUMNS)

dataset = pd.read_csv('./IMDB Dataset.csv', delimiter=',',
  encoding=DATASET_ENCODING)
# dataset = pd.read_csv('./Corona_NLP_train.csv',
                    #   delimiter=',', encoding=DATASET_ENCODING)
dataset_dir = 'imdb'
# dataset_dir = 'coronaNLP'
# dataset_dir = 'sentiment140'
model_dir = './models/'+dataset_dir
vector_dir = './vectors/'+dataset_dir

# removing the unnecessary columns and duplicates
# dataset = dataset[['OriginalTweet','Sentiment']]
dataset.drop_duplicates()

# dataset.head()


def preprocess_tweets(tweet):
    tweet = p.clean(tweet)
    tokens = tweet.split()
    stemmed_tokens = [stemmer.stem(token) for token in tokens]
    return ' '.join(stemmed_tokens)


X = dataset['review']
# X = dataset['tweet']
# X = dataset['OriginalTweet']

X = X.apply(preprocess_tweets)

y = dataset['sentiment']
# y = dataset['Sentiment']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# X_train.shape, X_test.shape

# creating our pipeline that will return an estimator
pipeline = Pipeline([('tfidf', TfidfVectorizer(stop_words='english',
    tokenizer=token.tokenize)), ('clf', LogisticRegression(class_weight='balanced', multi_class="multinomial", random_state=42))])

parameters = {
    'tfidf__max_features': (20000, 35000, 50000),
    'tfidf__ngram_range': ((1, 1), (1, 2), (2, 2)),
    'clf__C': np.logspace(-4, 4, 20),
    'clf__penalty': ('l1', 'l2', 'elasticnet', 'none'),
    'clf__solver': ('newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga')
}

clf = GridSearchCV(pipeline, param_grid=parameters, n_jobs=-2, cv=ShuffleSplit(n_splits=1),
                    verbose=2)

clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

print(classification_report(y_test, y_pred))

print(confusion_matrix(y_test, y_pred))

# print("Best: %f using %s" % (clf.best_score_,
#                              clf.best_params_))
# means = clf.cv_results_['mean_test_score']
# stds = clf.cv_results_['std_test_score']
# params = clf.cv_results_['params']
# for mean, stdev, param in zip(means, stds, params):
#     print("%f (%f) with: %r" % (mean, stdev, param))


acc = int(accuracy_score(y_test, y_pred)*100)

# exporting the pipeline
pickle.dump(pipeline['clf'], open(f'{model_dir}/LRG_model_{acc}', 'wb'))
pickle.dump(pipeline['tfidf'], open(f'{vector_dir}/tfidf_lrg_{acc}', 'wb'))
