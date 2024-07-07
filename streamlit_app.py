import streamlit as st
from tensorflow.keras.models import load_model
import pickle
import numpy as np
from keras.preprocessing.sequence import pad_sequences
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)
sentiment = {'positive': 0,'neutral': 1,'negative':2}
model = load_model('Mymodel.h5')
labels = ['Positive', 'Neutral', 'Negative']
st.title("RNN example : Financial News Sentiment Analysis")
st.write(
    "write the sentence to get an opinion"
)
input_text = st.text_area("Enter your text here:")
# Display the entered text
if input_text:

    message = [input_text]
    seq = tokenizer.texts_to_sequences(message)

    padded = pad_sequences(seq, maxlen=50, dtype='int32', value=0)

    pred = model.predict(padded)
    st.write(f"Probability of positive class: {round(pred[0][sentiment['Positive']]*100)} %")
    st.write(f"Probability of neutral class: {round(pred[0][sentiment['Neutral']]*100)} %")
    st.write(f"Probability of negative class: {round(pred[0][sentiment['Negative']]*100)} %")
    st.write(f"Predicted class: {labels[np.argmax(pred)]}")

