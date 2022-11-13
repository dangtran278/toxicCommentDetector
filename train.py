import os #path
import pandas as pd
import tensorflow as tf
import numpy as np

from tensorflow.keras.layers import TextVectorization
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Bidirectional, Dense, Embedding
from tensorflow.keras.metrics import Precision, Recall, CategoricalAccuracy

from matplotlib import pyplot as plt


df = pd.read_csv(os.path.join('dataset', 'train.csv'))

# PREPROCESS
X = df['comment_text']
y = df[df.columns[2:]].values # convert toxicity tags into numpy arr
MAX_FEATURES = 200000 # number of words in the vocab of TextVectorization - reduce if not enough vram

# text vectorization layer
vectorizer = TextVectorization(max_tokens=MAX_FEATURES,
                               output_sequence_length=1800, # max sentence length in token
                               output_mode='int') # map word to int
vectorizer.adapt(X.values)
vectorized_text = vectorizer(X.values)

# MCSHBAP
dataset = tf.data.Dataset.from_tensor_slices((vectorized_text, y))
dataset = dataset.cache()
dataset = dataset.shuffle(160000)
dataset = dataset.batch(16)
dataset = dataset.prefetch(8) # prevent bottlenecks

# split dataset
train = dataset.take(int(len(dataset)*.7))
val = dataset.skip(int(len(dataset)*.7)).take(int(len(dataset)*.2))
test = dataset.skip(int(len(dataset)*.9)).take(int(len(dataset)*.1))


# SEQUENTIAL MODEL
model = Sequential()

# embedding layer (positive vs negative)
model.add(Embedding(MAX_FEATURES+1, 32))

# bidirectional LSTM layer (e.g. don't hate)
model.add(Bidirectional(LSTM(32, activation='tanh')))

# feature extractor fully connected layers
model.add(Dense(128, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='relu'))

# final layer (modify to 0 or 1 and map to outputs)
model.add(Dense(6, activation='sigmoid'))

model.compile(loss='BinaryCrossentropy', optimizer='Adam') # loss for multi-output model
model.summary()

# train
history = model.fit(train, epochs=4, validation_data=val) # recommend 5-10 epochs

# plt.figure(figsize=(8,5))
# pd.DataFrame(history.history).plot()
# plt.show()


# PREDICT
input_text = vectorizer('You freaking suck! I am going to hit you.') # tokenize input
res = model.predict(np.expand_dims(input_text, 0))
(res > 0.5).astype(int)
batch_X, batch_y = test.as_numpy_iterator().next()
(model.predict(batch_X) > 0.5).astype(int)
res.shape


# # EVALUATE
# pre = Precision()
# re = Recall()
# acc = CategoricalAccuracy()

# for batch in test.as_numpy_iterator():
#     # unpack the batch
#     X_true, y_true = batch
#     # make a prediction
#     yhat = model.predict(X_true)

#     # flatten the predictions
#     y_true = y_true.flatten()
#     yhat = yhat.flatten()

#     pre.update_state(y_true, yhat)
#     re.update_state(y_true, yhat)
#     acc.update_state(y_true, yhat)

# print(f'Precision: {pre.result().numpy()}, Recall:{re.result().numpy()}, Accuracy:{acc.result().numpy()}')



# TEST
model.save('toxicity.h5')
model = tf.keras.models.load_model('toxicity.h5')
input_str = vectorizer('marius suck')
res = model.predict(np.expand_dims(input_str,0))
print(res)
