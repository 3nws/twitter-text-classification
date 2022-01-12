# %%
from nltk.tokenize import TweetTokenizer, RegexpTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
from sklearn.metrics import classification_report
from nltk.stem.snowball import SnowballStemmer

import re
import pandas as pd
import pickle
import nltk
import preprocessor as p

nltk.download('wordnet')
nltk.download('stopwords')

# %%

stemmer = SnowballStemmer("english", ignore_stopwords=True)

ps = PorterStemmer()
lem = WordNetLemmatizer()

# importing the dataset
# DATASET_COLUMNS  = ["sentiment", "ids", "date", "flag", "user", "tweet"]
DATASET_ENCODING = "ISO-8859-1"
# dataset = pd.read_csv('./training.1600000.processed.noemoticon.csv', delimiter=',', encoding=DATASET_ENCODING , names=DATASET_COLUMNS)

dataset = pd.read_csv('./Corona_NLP_train.csv', delimiter=',', encoding=DATASET_ENCODING)

# removing the unnecessary columns and duplicates
dataset = dataset[['OriginalTweet','Sentiment']]
dataset.drop_duplicates()

token = RegexpTokenizer(r'[a-zA-Z0-9]+')


# %%

# tokenizing and stemming
dataset['tweet'] = dataset['OriginalTweet'].apply(p.clean)
dataset['sentiment'] = dataset['Sentiment']
# dataset['tokenized_tweet'] = dataset['OriginalTweet'].apply(token.tokenize)
# dataset['stemmed_tweet'] = dataset['tokenized_tweet'].apply(lambda tweets: [stemmer.stem(tweet) for tweet in tweets])
# dataset['joined_tweet'] = [" ".join(word) for word in dataset['stemmed_tweet']]

dataset.head()


# %%

# tfidf = TfidfVectorizer(stop_words='english', max_features=20000, ngram_range=(1,2))

X = dataset['tweet']

# X = [lem.lemmatize(ps.stem(x)) for x in X]

# X = tfidf.fit_transform([lem.lemmatize(ps.stem(x)) for x in X])

y = dataset['sentiment']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)

X_train.shape, X_test.shape


# %%

# creating the model usin multinomial naive bayes algorithm
MNB = MultinomialNB()

parameters = {
    'tfidf__max_features': (10000, 20000),
    'tfidf__ngram_range': [(1,1), (1,2)],
    'clf__fit_prior': (False, True),
    'clf__alpha': (1, 0.1, 0.01, 0.001, 0.0001, 0.00001)  
    }

pipeline = Pipeline([('tfidf', TfidfVectorizer()), ('clf', MultinomialNB())])

clf = GridSearchCV(pipeline, parameters, cv=5)

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

# exporting the model and the trained vectorizer
pickle.dump(MNB, open('./models/MNB_CV_model', 'wb'))
pickle.dump(tfidf, open('./vector/tfidf_vectorizer_mnb_cv', 'wb'))


