import sys
import os
import pandas as pd
import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import TextVectorization


df = pd.read_csv(os.path.join('dataset', 'train.csv'))
X = df['comment_text']

vectorizer = TextVectorization(max_tokens=200000,
                               output_sequence_length=1800, # max sentence length in token
                               output_mode='int') # map word to int
vectorizer.adapt(X.values)
model = tf.keras.models.load_model('toxicity.h5')
input_text = vectorizer(str(sys.argv[1]))
res = model.predict(np.expand_dims(input_text,0))

print('Toxic: ', res[0][0])
print('Severe Toxic: ', res[0][1])
print('Obscene: ', res[0][2])
print('Threat: ', res[0][3])
print('Insult: ', res[0][4])
print('Identity hate: ', res[0][5])
sys.stdout.flush()