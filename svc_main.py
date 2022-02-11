from nltk.tokenize import TweetTokenizer, RegexpTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score

import re
import pandas as pd
import pickle

DATASET_ENCODING = "ISO-8859-1"

dataset = pd.read_csv('./IMDB Dataset.csv', delimiter=',', encoding=DATASET_ENCODING)

token = RegexpTokenizer(r'[a-zA-Z0-9]+')
tt = TweetTokenizer()

dataset.head()

X = dataset['review']

tfidf = TfidfVectorizer(stop_words='english', max_features=20000, ngram_range=(1,2), tokenizer=token.tokenize)

X = tfidf.fit_transform(X)

y = dataset['sentiment']

# X.shape

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)

# X_train.shape, X_test.shape

svc = SVC(probability=True)
svc.fit(X_train, y_train)

y_pred = svc.predict(X_test)

# print(svc.predict_proba(sub_main))
print(classification_report(y_test, y_pred))

# test_tweet = "scandinavia #news:  norway : it's illegal for employers to require covid  passports  denmark\
#     sweden : they won't be bringing in covid  vaccination passports  #holdtheline #enoughisenough #nomedicalapartheid #nomasks #nomorelockdowns #openforall #corona #coronavirus"
# # test_tweet2 = "everyone should get vaccinated as soon as possible"
# vector = tfidf.transform([test_tweet])

# print(svc.predict(vector))

acc = int(accuracy_score(y_test, y_pred)*100)

# exporting the model and the trained vectorizer
pickle.dump(svc, open(f'./models/SVC_model_{acc}', 'wb'))
pickle.dump(tfidf, open(f'./vector/tfidf_vectorizer_{acc}', 'wb'))