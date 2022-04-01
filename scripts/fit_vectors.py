

from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem.snowball import SnowballStemmer

import pandas as pd
import nltk
import joblib

nltk.download('stopwords')

stemmer = SnowballStemmer("english")




# dataset_dir = 'sentiment140'
dataset_dir = 'imdb'
# dataset_dir = 'coronaNLP'

# importing the processed dataframe 
df = joblib.load(f'./dataframes/df_{dataset_dir}.pkl')


df.head()


from collections import Counter

# Count unique words


def counter_word(text_col):
    count = Counter()
    for text in text_col.values:
        for word in text.split():
            count[word] += 1
    return count


counter = counter_word(df.iloc[:, 0])

from nltk.tokenize import RegexpTokenizer

token = RegexpTokenizer(r'[a-zA-Z0-9]+')

max_f = len(counter)

# n_gram = (1, 1)
# n_gram = (1, 2)
n_gram = (2, 2)

tfidf = TfidfVectorizer(max_features=max_f, ngram_range=n_gram, tokenizer=token.tokenize)
tfidf


X = df.iloc[:, 0]

tfidf = tfidf.fit(X)


tfidf


joblib.dump(tfidf, f'./vectors/vectorizer_{dataset_dir}_{n_gram}.pkl')



new_vector = joblib.load(
    f'./vectors/vectorizer_{dataset_dir}_{n_gram}.pkl')
new_vector


