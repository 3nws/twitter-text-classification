from nltk.tokenize import TweetTokenizer, RegexpTokenizer
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import RegexpTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
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

# importing the dataset
DATASET_ENCODING = "ISO-8859-1"
# DATASET_COLUMNS  = ["sentiment", "ids", "date", "flag", "user", "tweet"]
# dataset = pd.read_csv('./training.1600000.processed.noemoticon.csv', delimiter=',', encoding=DATASET_ENCODING , names=DATASET_COLUMNS)

dataset = pd.read_csv('./IMDB Dataset.csv', delimiter=',', encoding=DATASET_ENCODING)
# dataset = pd.read_csv('./Corona_NLP_train.csv',
#                       delimiter=',', encoding=DATASET_ENCODING)
dataset_dir = 'imdb'
model_dir = './models/'+dataset_dir
vector_dir = './vectors/'+dataset_dir

# removing the unnecessary columns and duplicates
# dataset = dataset[['OriginalTweet','Sentiment']]
dataset.drop_duplicates()

token = RegexpTokenizer(r'[a-zA-Z0-9]+')

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
pipeline = Pipeline([('tfidf', TfidfVectorizer(
    tokenizer=token.tokenize)), ('clf', MultinomialNB())])

parameters = {
    'tfidf__max_features': (20000, 30000, 40000, 50000),
    'tfidf__ngram_range': ((1, 1), (1, 2), (2, 2)),
    'clf__fit_prior': (False, True),
    'clf__alpha': (1, 0.1, 0.01, 0.001)
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


acc = int(accuracy_score(y_test, y_pred)*100)

# exporting the pipeline
pickle.dump(pipeline['clf'], open(f'{model_dir}/MNB_model_{acc}', 'wb'))
pickle.dump(pipeline['tfidf'], open(f'{vector_dir}/tfidf_mnb_{acc}', 'wb'))
