from tensorflow.keras.models import load_model
import pickle
import numpy as np
from keras.preprocessing.sequence import pad_sequences
with open('trained_model/tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)
sentiment  = {'positive': 0,'neutral': 1,'negative':2}
model = load_model('trained_model/Mymodel.h5')
model.summary()

message = ['Apple lost all of their assets']
seq = tokenizer.texts_to_sequences(message)

padded = pad_sequences(seq, maxlen=50, dtype='int32', value=0)

pred = model.predict(padded)
print(message[0])
labels = ['positive','neutral','negative']
print(f"probability of positive class: {round(pred[0][sentiment['positive']]*100)}%")
print(f"probability of neutral class: {pred[0][sentiment['neutral']]}")
print(f"probability of negative class: {pred[0][sentiment['negative']]}")
print(f"predicted class: {labels[np.argmax(pred)]}")