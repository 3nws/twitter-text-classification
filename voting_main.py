from nltk.tokenize import TweetTokenizer, RegexpTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score
# from nltk.stem.snowball import SnowballStemmer

import re
import pandas as pd
import pickle
import nltk
# import preprocessor as p

# nltk.download('stopwords')

# stemmer = SnowballStemmer("english", ignore_stopwords=True)

# importing the dataset
DATASET_COLUMNS  = ["sentiment", "ids", "date", "flag", "user", "tweet"]
DATASET_ENCODING = "ISO-8859-1"
dataset = pd.read_csv('./training.1600000.processed.noemoticon.csv', delimiter=',', encoding=DATASET_ENCODING , names=DATASET_COLUMNS)

# dataset = pd.read_csv('./Corona_NLP_train.csv', delimiter=',')

# removing the unnecessary columns and duplicates
dataset = dataset[['sentiment','tweet']]
dataset.drop_duplicates()

token = RegexpTokenizer(r'[a-zA-Z0-9]+')

# tokenizing and stemming
# dataset['tweet'] = dataset['tweet'].apply(p.clean)
# dataset['tokenized_tweet'] = dataset['OriginalTweet'].apply(token.tokenize)
# dataset['stemmed_tweet'] = dataset['tokenized_tweet'].apply(lambda tweets: [stemmer.stem(tweet) for tweet in tweets])
# dataset['joined_tweet'] = [" ".join(word) for word in dataset['stemmed_tweet']]

# dataset.head()

tfidf = TfidfVectorizer(stop_words='english', max_features=20000, ngram_range=(1,2))

X = dataset['tweet']

X = tfidf.fit_transform(X)

y = dataset['sentiment']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)

# X_train.shape, X_test.shape

# creating the model usin multinomial naive bayes algorithm
MNB = MultinomialNB()
# LRG = LogisticRegression(random_state=0, class_weight='balanced')

# pipe = make_pipeline(tfidf, MNB)
# pipe.fit(X_train, y_train)
# pipe.predict(y_pred)

estimator = []
estimator.append(('LR', 
                  LogisticRegression(solver ='lbfgs', 
                                     multi_class ='multinomial', 
                                     max_iter = 200)))
estimator.append(('SVC', SVC(gamma ='auto', probability = True)))
estimator.append(('DTC', MNB))

VTC = VotingClassifier(estimators = estimator, voting ='hard')

# print(cross_val_score(MNB, X, y).mean())
# print(cross_val_score(LRG, X, y).mean())
# print(cross_val_score(SVC, X, y).mean())

# print(cross_val_score(VTC, X, y).mean())

# training the model
VTC.fit(X_train, y_train)

# testing our predictions
y_pred = VTC.predict(X_test)

print(classification_report(y_test, y_pred))

test_tweet = "everyone should get vaccinated as soon as possible"
# test_tweet = "bad vaccine"
# test_tweet = "free america. end the lockdown. corona virus is a hoax, covid vaccine is a hoax"
vector = tfidf.transform([test_tweet])

print(VTC.predict(vector))

# exporting the model and the trained vectorizer
pickle.dump(MNB, open('./models/MNB_model', 'wb'))
pickle.dump(tfidf, open('./vector/tfidf_vectorizer_mnb', 'wb'))