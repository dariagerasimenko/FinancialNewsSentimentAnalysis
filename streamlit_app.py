import streamlit as st
from tensorflow.keras.models import load_model
import pickle
import numpy as np
from keras.preprocessing.sequence import pad_sequences
with open('trained_model/tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)
sentiment = {'Positive': 0,'Neutral': 1,'Negative':2}
model = load_model('trained_model/Mymodel.h5')
labels = ['Positive', 'Neutral', 'Negative']
st.title("LSTM: Financial News Sentiment Analysis")
st.write(
    "by Daria Gerasimenko, Student of HHU"
)
default_text = "US banks suffer steeper losses"
input_text = st.text_area("Enter your sentence to predict sentiment here:", value=default_text)
# Display the entered text
if input_text:

    message = [input_text]
    seq = tokenizer.texts_to_sequences(message)

    padded = pad_sequences(seq, maxlen=50, dtype='int32', value=0)

    pred = model.predict(padded)
    st.write(f"Probability of positive class: {round(pred[0][sentiment['Positive']]*100)} %")
    st.write(f"Probability of neutral class: {round(pred[0][sentiment['Neutral']]*100)} %")
    st.write(f"Probability of negative class: {round(pred[0][sentiment['Negative']]*100)} %")
    predicted_class = labels[np.argmax(pred)]

    # Color coding for predicted class
    color_dict = {'Positive': 'green', 'Neutral': 'blue', 'Negative': 'red'}
    color = color_dict[predicted_class]

    # Display the predicted class with color
    st.markdown(f"<span style='color:{color}'>Predicted class: {predicted_class}</span>", unsafe_allow_html=True)

