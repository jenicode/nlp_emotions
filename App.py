import streamlit as st
import nlp
import pandas as pd
import numpy as np
import tensorflow as tf
from PIL import Image

from prediction import reloaded_model_predictions

st.write('Welcome!')
st.title('Emotion Analysis using RNN')
st.markdown('Use Hugging Face\'s nlp Package (tweets) to classify the emotions expressed in the texts.')

st.header('Emotions Categories: anger, sadness, surprise, love, fear, joy')

st.subheader('User inputs:')
user_input = st.text_input("Please input one text below", )

if st.button('Predict the emotion'):
	st.write('Starting predicting the text:', user_input)
	st.write('The predicted emotion is:', reloaded_model_predictions([user_input]))

st.subheader('The Training History')
image1 = Image.open('Accuracy_Loss.png')
st.image(image1)

st.subheader('Prediction -- Test')
image2 = Image.open('confusion_matrix_test.png')
st.image(image2)

st.subheader('Thank you!')

# plot the traning history




