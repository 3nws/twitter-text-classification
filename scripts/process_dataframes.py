

from nltk.stem.snowball import SnowballStemmer

import pandas as pd
import nltk

nltk.download('stopwords')

stemmer = SnowballStemmer("english")




# importing the dataset
DATASET_ENCODING = "ISO-8859-1"

DATASET_COLUMNS = ["sentiment", "ids", "date", "flag", "user", "tweet"]
df = pd.read_csv('./training.1600000.processed.noemoticon.csv',
                 delimiter=',', encoding=DATASET_ENCODING, names=DATASET_COLUMNS)

# df = pd.read_csv('./IMDB Dataset.csv', delimiter=',',
#   encoding=DATASET_ENCODING)

# df = pd.read_csv('./Corona_NLP_train.csv', delimiter=',', encoding=DATASET_ENCODING)

# removing the unnecessary columns.
df = df[['tweet', 'sentiment']]
# df = df[['review','sentiment']]
# df = df[['OriginalTweet','Sentiment']]

dataset_dir = 'sentiment140'
# dataset_dir = 'imdb'
# dataset_dir = 'coronaNLP'




df = df.drop_duplicates()

df.head()



# Preprocessing
from nltk.corpus import stopwords
import re
import string

RE_EMOJI = re.compile('[\U00010000-\U0010ffff]', flags=re.UNICODE)


def strip_emoji(text):
    return RE_EMOJI.sub(r'', text)

def remove_URL(text):
    url = re.compile(r"https?://\S+|www\.\S+")
    return url.sub(r"", text)


def remove_punct(text):
    translator = str.maketrans("", "", string.punctuation)
    return text.translate(translator)


def remove_mention(text):
    return re.sub("@[A-Za-z0-9]+", "", text)


def stem_tweets(tweet):
    tokens = tweet.split()
    stemmed_tokens = [stemmer.stem(token) for token in tokens]
    return ' '.join(stemmed_tokens)


def lemmatize_tweets(tweet):
    tokens = tweet.split()
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]
    return ' '.join(lemmatized_tokens)

# remove stopwords


stop = set(stopwords.words("english"))


def remove_stopwords(text):
    stop = set(stopwords.words("english"))

    filtered_words = [word.lower()
                      for word in text.split() if word.lower() not in stop]
    return " ".join(filtered_words)


def preprocess_tweets(tweet):
    tweet = strip_emoji(tweet)
    tweet = remove_mention(tweet)
    tweet = remove_URL(tweet)
    tweet = remove_punct(tweet)
    tweet = stem_tweets(tweet)
    # tweet = lemmatize_tweets(tweet)
    tweet = remove_stopwords(tweet)
    return tweet



import numpy as np

def convert_sentiment_to_binary(sentiment):
    if dataset_dir == 'coronaNLP':
        if sentiment == 'Extremely Positive':
            return 2
        elif sentiment == 'Positive':
            return 1
        elif sentiment == 'Neutral':
            return 0
        elif sentiment == 'Negative':
            return -1
        elif sentiment == 'Extremely Negative':
            return -2
    
    if dataset_dir == 'sentiment140':
        return 1 if sentiment == 4 else 0
    
    return 1 if sentiment == 'positive' else 0


convert_sentiment_to_int_v = np.vectorize(convert_sentiment_to_binary)

df.iloc[:, 1] = convert_sentiment_to_int_v(df.iloc[:, 1])




df.iloc[:, 0] = df.iloc[:, 0].apply(preprocess_tweets)

X = df.iloc[:, 0]

df.head()



import joblib

joblib.dump(df, f'./dataframes/df_{dataset_dir}.pkl')



new_df = joblib.load(f'./dataframes/df_{dataset_dir}.pkl')
new_df.head()


