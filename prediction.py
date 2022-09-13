import streamlit as st
import nlp
import pandas as pd
import numpy as np
import tensorflow as tf

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

saved_model_path = 'HuggingFace_nlp_emotion_RNN'
reloaded_model = tf.keras.models.load_model(saved_model_path)

dataset = nlp.load_dataset('emotion')
tweets = [d['text'] for d in dataset['train']] + [d['text'] for d in dataset['test']] + [d['text'] for d in dataset['validation']]
tokenizer = Tokenizer(num_words=10000, oov_token='<UNK>')
tokenizer.fit_on_texts(tweets)
max_length = 70 # from the trained model
nums_to_names = {0: 'anger', 1: 'sadness', 2: 'surprise', 3: 'love', 4: 'fear', 5: 'joy'}

def get_padded_sequences(pred_sentences, tokenizer):
    ### get_sequences
    sequences = tokenizer.texts_to_sequences(pred_sentences)
    ###
    padded_sequences = pad_sequences(sequences, truncating='post', maxlen=max_length, padding='post')    
    return padded_sequences

def reloaded_model_predictions(pred_sentences):
    pred_sequences = get_padded_sequences(pred_sentences, tokenizer=tokenizer)
	#st.write(pred_sequences)
    predicts = reloaded_model.predict(pred_sequences)
    classes = np.argmax(predicts, axis=-1)
    #return  [(sentence, nums_to_names[class_]) for sentence, class_ in zip(pred_sentences, classes)]
    return [nums_to_names[class_] for class_ in classes] 
#preds = reloaded_model_predictions(pred_sentences)
#preds
