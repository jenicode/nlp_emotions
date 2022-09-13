# Emotion Analysis with NLP using RNN

Notes:
The training and testing codes can be found the file "emotions_update.ipynb". 
The trained model were saved in the folder "HuggingFace_nlp_emotion_RNN". 
A app based on the trained model was built on Streamlit. Welome to hit the link: https://jenicode-nlp-emotions-app-j1e60g.streamlitapp.com/

Model Details:
The datasets are imported from Hugging Face's nlp package. The emotions are categorized into:  {'surprise', 'joy', 'love', 'anger', 'fear', 'sadness'}.
However, imbalance within the datasets are noticed, which may cause bias in the prediction.

Further improvement can be done through fixing the imbalance within the datasets or
validating the sentimental labels('positive', 'negative') using other well-trained model such as Bert.  

Thank you!

Additional Info: 

https://huggingface.co/docs/datasets/v0.3.0/installation.html
https://www.tensorflow.org/text/tutorials/text_classification_rnn 

'tf.keras.preprocessing.text.Tokenizer' can be replaced by 'tf.keras.layers.TextVectorization', which is suggested by the linke below:  https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/text/Tokenizer. 
