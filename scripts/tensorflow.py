import tensorflow as tf
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import joblib
import uuid


uniqueid = uuid.uuid4().int & (1 << 64)-1



use_pre_trained_embeds = False


dataset_dir = "imdb"
# dataset_dir = "sentiment140"

model_dir = "models"
visuals_dir = "visuals"

# load a preprocessed dataframe see: (https://github.com/3nws/twitter-text-classification/blob/main/notebooks/process_dataframes.ipynb)
df = joblib.load(
    "../dataframes/df_imdb.pkl") if dataset_dir == "imdb" else joblib.load("../dataframes/df_sentiment140.pkl")


df.columns=["text", "sentiment"]


df = df[730000:850000] if dataset_dir == "sentiment140" else df


df.shape


df.head()



import seaborn as sns

sns.countplot(df.sentiment)



from collections import Counter

# Count unique words
def counter_word(text_col):
    count = Counter()
    for text in text_col.values:
        for word in text.split():
            count[word] += 1
    return count


counter = counter_word(df.text)



len(counter)


counter


counter.most_common(5)


num_unique_words = len(counter)


# Split dataset into training and validation set

from sklearn.model_selection import train_test_split

X, y = df.iloc[:, 0], df.iloc[:, 1]

train_sentences, val_sentences, train_labels, val_labels = train_test_split(X, y, test_size=0.3, stratify=y ,random_state=42)

train_sentences = train_sentences.to_numpy()
val_sentences = val_sentences.to_numpy()
train_labels = train_labels.to_numpy()
val_labels = val_labels.to_numpy()



train_sentences.shape, val_sentences.shape, train_labels.shape, val_labels.shape



type(train_sentences), type(val_sentences), type(train_labels), type(val_labels),


train_sentences[:1], train_labels[:1]


# Tokenize
from tensorflow.keras.preprocessing.text import Tokenizer

max_features = 50000
# max_features = num_unique_words

# vectorize a text corpus by turning each text into a sequence of integers
tokenizer = Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(train_sentences) # fit only to training


# each word has unique index
word_index = tokenizer.word_index


word_index


len_of_vocab = len(word_index)


train_sequences = tokenizer.texts_to_sequences(train_sentences)
val_sequences = tokenizer.texts_to_sequences(val_sentences)


print(train_sentences[14:15])
print(train_sequences[14:15])


# Pad the sequences to have the same length
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Max number of words in a sequence
max_length = max([len(text) for text in train_sequences]) if dataset_dir == "sentiment140" else 50
max_length



train_padded = pad_sequences(train_sequences, maxlen=max_length, padding="post", truncating="post")
val_padded = pad_sequences(val_sequences, maxlen=max_length, padding="post", truncating="post")
train_padded.shape, val_padded.shape


train_dataset = tf.data.Dataset.from_tensor_slices(
    (train_padded, train_labels))
val_dataset = tf.data.Dataset.from_tensor_slices((val_padded, val_labels))



len(train_dataset), len(val_dataset)



BATCH_SIZE = 64
BUFFER_SIZE = 10000

train_dataset = train_dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
val_dataset = val_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)



train_dataset, val_dataset



# Check reversing the indices

# flip (key, value)
reverse_word_index = dict([(idx, word) for (word, idx) in word_index.items()])


reverse_word_index


def decode(sequence):
    return " ".join([reverse_word_index.get(idx, "?") for idx in sequence])


decoded_text = decode(train_sequences[10])

print(train_sequences[10])
print(decoded_text)


embedding_dim = 64
lstm_dim = int(embedding_dim/2)

if use_pre_trained_embeds:
    embeddings_dictionary = dict()
    glove_file = open('../embeds/glove.6B.300d.txt', 'rb')

    for line in glove_file:
        records = line.split()
        word = records[0]
        vector_dimensions = np.asarray(records[1:], dtype='float32')
        embeddings_dictionary[word] = vector_dimensions

    glove_file.close()

    embeddings_matrix = np.zeros((num_unique_words, embedding_dim))
    for word, index in tokenizer.word_index.items():
        embedding_vector = embeddings_dictionary.get(word)
        if embedding_vector is not None:
            embeddings_matrix[index] = embedding_vector



from tensorflow.keras.layers import Dense, LSTM, Embedding, Bidirectional, SpatialDropout1D, Dropout, GlobalMaxPool1D, GlobalMaxPool2D
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy, CategoricalCrossentropy, SparseCategoricalCrossentropy
from tensorflow.keras.regularizers import L1, L2


# 'softmax' activation function returns a probability distribution
# Binary for 0-1, Categorical for 2 or more classes, SparseCategorical for when labels are integers
# Dropout is used to prevent overfitting by randomly setting inputs to 0 at a low rate
# For stacked LSTMs set return_sequences to True except for the last one
# trainable parameter in Embedding layer should still be set to True when using already trained weights (it is by default anyway)

# 0
def one():
    model = Sequential()
    model.add(Embedding(max_features, embedding_dim,
                        input_length=max_length, name="embeddinglayer", weights=[embeddings_matrix], trainable=True))
    model.add(Bidirectional(LSTM(lstm_dim, dropout=0.3, return_sequences=True)))
    model.add(Bidirectional(LSTM(lstm_dim, dropout=0.3)))
    model.add(Dense(2, activation="softmax"))
    loss = SparseCategoricalCrossentropy(from_logits=False)
    optim = Adam(learning_rate=0.001)
    metrics = [
        "accuracy",
        "sparse_categorical_accuracy",
    ]
    model.compile(loss=loss, optimizer=optim, metrics=metrics)
    return model

# 1
def two():
    model = Sequential()
    model.add(Embedding(max_features, embedding_dim,
                               input_length=max_length, name="embeddinglayer"))
    model.add(LSTM(embedding_dim, dropout=0.1))
    model.add(Dense(1, activation="sigmoid"))
    loss = BinaryCrossentropy(from_logits=False)
    optim = Adam(learning_rate=0.001)
    metrics = [
        "accuracy",
        "binary_accuracy",
    ]
    model.compile(loss=loss, optimizer=optim, metrics=metrics)
    return model
    
# 2
def three():
    model = Sequential()
    model.add(Embedding(max_features, embedding_dim,
                               input_length=max_length, name="embeddinglayer"))
    model.add(LSTM(embedding_dim, dropout=0.1))
    model.add(Dense(2, activation="softmax"))
    loss = SparseCategoricalCrossentropy(from_logits=False)
    optim = Adam(learning_rate=0.001)
    metrics = [
        "accuracy",
        "sparse_categorical_accuracy",
    ]
    model.compile(loss=loss, optimizer=optim, metrics=metrics)
    return model

# 3
def four():
    model = Sequential()
    model.add(Embedding(max_features, embedding_dim,
                               input_length=max_length, name="embeddinglayer"))
    model.add(Bidirectional(LSTM(lstm_dim, dropout=0.2)))
    model.add(Dense(32, activation="relu"))
    model.add(Dense(2, activation="softmax"))
    loss = SparseCategoricalCrossentropy(from_logits=False)
    optim = Adam(learning_rate=0.001)
    metrics = [
        "accuracy",
        # "sparse_categorical_accuracy",
    ]
    model.compile(loss=loss, optimizer=optim, metrics=metrics)
    return model

# 4
def five():
    model = Sequential()
    model.add(Embedding(max_features, embedding_dim,
                        input_length=max_length, name="embeddinglayer"))
    model.add(Bidirectional(LSTM(lstm_dim, dropout=0.3, return_sequences=True)))
    model.add(Bidirectional(LSTM(lstm_dim, dropout=0.3)))
    model.add(Dense(32, activation="relu", kernel_regularizer=L1(0.01), activity_regularizer=L2(0.01)))
    model.add(Dense(2, activation="softmax"))
    loss = SparseCategoricalCrossentropy(from_logits=False)
    optim = Adam(learning_rate=0.0001)
    metrics = [
        "accuracy",
        "sparse_categorical_accuracy",
    ]
    model.compile(loss=loss, optimizer=optim, metrics=metrics)
    return model

# 5
def six():
    model = tf.keras.Sequential()
    model.add(Embedding(
        max_features, embedding_dim, input_length=max_length))
    model.add(SpatialDropout1D(0.4))
    model.add(Bidirectional(LSTM(lstm_dim, dropout=0.05, recurrent_dropout=0.2)))
    model.add(Dense(2, activation='softmax'))
    loss = SparseCategoricalCrossentropy(from_logits=False)
    optim = Adam(learning_rate=0.001)
    metrics = ["accuracy",
               "sparse_categorical_accuracy",
    ]
    model.compile(loss=loss, optimizer=optim, metrics=metrics)
    return model

# 6
def seven():
    model = Sequential()
    model.add(Embedding(max_features, embedding_dim,
                        input_length=max_length, name="embeddinglayer"))
    model.add(Bidirectional(LSTM(lstm_dim, dropout=0.3, return_sequences=True)))
    model.add(Bidirectional(LSTM(lstm_dim, dropout=0.3)))
    model.add(Dense(32, activation="relu", kernel_regularizer=L1(0.01),
                    activity_regularizer=L2(0.01)))
    model.add(Dense(1, activation="sigmoid"))
    loss = BinaryCrossentropy(from_logits=False)
    optim = Adam(learning_rate=1e-4)
    metrics = [
        "accuracy",
    ]
    model.compile(loss=loss, optimizer=optim, metrics=metrics)
    return model

# 7
def eight():
    model = Sequential()
    model.add(Embedding(max_features, 128))
    model.add(Bidirectional(LSTM(32, return_sequences=True)))
    model.add(GlobalMaxPool1D())
    model.add(Dense(20, activation="relu"))
    model.add(Dropout(0.05))
    model.add(Dense(1, activation="sigmoid"))
    loss = BinaryCrossentropy(from_logits=False)
    optim = Adam(learning_rate=1e-4)
    metrics = [
        "accuracy",
    ]
    model.compile(loss=loss, optimizer=optim, metrics=metrics)
    return model



max_features, embedding_dim, max_length



models = [
    one,
    two,
    three,
    four,
    five,
    six,
    seven,
    eight
]

model_to_use = -2

model_idx = 0 if use_pre_trained_embeds else model_to_use



model = models[model_idx]()

model.summary()


history = model.fit(train_dataset, epochs=10, batch_size=128, validation_data=val_dataset, verbose=1)




val_loss, val_acc = model.evaluate(val_dataset)
val_loss, val_acc



model_name = models[model_idx].__name__
model_export = f"NN_model_{model_name}_{uniqueid}_{val_acc}"
vis_dir = f'../{visuals_dir}/{model_export}'
model_save_dir = f'../{model_dir}/{model_export}'

# plotting training graph
plt.plot(history.history['loss'])
plt.savefig(f'{vis_dir}.png')



print(val_sentences[2])
print(val_labels[2])
print(model.predict(val_padded[2:3]))



val_predictions = model.predict(val_padded)


# Only for BinaryCrossentropy
predictions = [1 if p > 0.5 else 0 for p in val_predictions]
predictions


import matplotlib.pyplot as plt


def plot_graphs(history, metric):
  plt.plot(history.history[metric])
  plt.plot(history.history['val_'+metric], '')
  plt.xlabel("Epochs")
  plt.ylabel(metric)
  plt.legend([metric, 'val_'+metric])


plt.figure(figsize=(16, 8))
plt.subplot(1, 2, 1)
plot_graphs(history, 'accuracy')
plt.ylim(None, 1)
plt.subplot(1, 2, 2)
plot_graphs(history, 'loss')
plt.ylim(0, None)
plt.savefig(f'{vis_dir}_loss_acc.png')



model.save(model_save_dir)



loaded_model = load_model(model_save_dir)



loaded_model.summary()


# For debugging purposes


# model = keras.Model(inputs=model.input,
#                     outputs=[model.get_layer("embeddingL").output])

# feature = model.predict(val_padded)

# feature, feature.shape


