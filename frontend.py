import streamlit as st
from streamlit_chat import message
from transformers import BlenderbotTokenizer
from transformers import BlenderbotForConditionalGeneration

import numpy as np
import pandas as pd
import os
import time
import pickle


from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img,img_to_array
from tensorflow.python.keras import utils 

from string import punctuation
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
import nltk.stem
from nltk.stem import LancasterStemmer, SnowballStemmer
from nltk.stem.wordnet import WordNetLemmatizer
import re
from string import punctuation
from sklearn.feature_extraction.text import CountVectorizer

import easyocr
import speech_recognition as sr
import pyttsx3

# load the model from disk
filename = 'dnn_new.hdf5'
model = load_model(filename)
print(model)
le = pickle.load(open('LabelEnc.sav','rb'))
cv = pickle.load(open('CountVectorizer.sav','rb'))

def transformed_data(text): 
    stop_words = stopwords.words('english')
    stemmer = SnowballStemmer('english')
    tokens = word_tokenize(text)
    cleaned_tokens = []

    for tok, tag in pos_tag(tokens):
        tok = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+#]|[!*\(\)]|'\
                       '(?:%[0-9a-fA-F][0-9a-fA-F]))+','', tok)
        tok = re.sub("(@[A-Za-z0-9_]+)","", tok)

        if tag.startswith("NN"):
            pos = 'n'
        elif tag.startswith('VB'):
            pos = 'v'
        else:
            pos = 'a'

        if len(tok) > 3 and tok not in punctuation and tok.lower() not in stop_words:
          tok = stemmer.stem(WordNetLemmatizer().lemmatize(tok, pos))
          cleaned_tokens.append(tok.lower())
          
    return cleaned_tokens
    
def image(img):
  reader = easyocr.Reader(['en'])
  result = reader.readtext(img, paragraph="False")
  return result[0][1]

def audio(au):
  r = sr.Recognizer()
  with sr.AudioFile(au) as source:
      # listen for the data (load audio to memory)
      audio_data = r.record(source)
      # recognize (convert from speech to text)
      text = r.recognize_google(audio_data)
      return text
  
def predict(row):
    all = model.predict(row)
    label = str(all[0][0].split('__')[-1])
    percent = round(all[1][0]*100, 2)
    return label, percent

def process(text):
    preprocess = transformed_data(text)
    cv_trans = cv.transform(preprocess)
    cv_array = cv_trans.toarray()
    pred = model.predict(cv_array)
    label = le.inverse_transform(np.argmax(pred,axis=1))[0]
    percent = round(max(pred[0])/sum(pred[0])*100, 2)
    output(label, percent)
    return

def output(label, percent):
    if label != 'None':
        if percent<=30: st.success(f"The text contains {percent}% {label}, but you can still Proceed.")
        elif 30<percent<=70: st.warning(f"The text contains {percent}% {label}, you should Stay Aware from the user.")
        else: st.error(f"The text contains {percent}% {label}, we would recommend you to stop the conversation.")
    else:
        st.info(f"The message is {percent}% safe.")
    return

    

if "history" not in st.session_state:
    st.session_state.history = []

with st.expander('Samples: '):   
    message("These girls are either hand or feet models", is_user = True)
    process('These girls are either hand or feet models')
    message("Why are you so irritating")
    process('Why are you so irritating') 

while True:
    input_ = st.text_input('Enter your Message : ', key = 'text')
    placehoder = st.empty()
    message(st.session_state.text)
    process(st.session_state.text)
    st.session_state.history.append({"message": st.session_state.text, "is_user": False})
    for chat in st.session_state.history:
        message(**chat)
    st.session_state["text"] = ""
    time.sleep(10)
    




  
 


#Sexism
#These girls are either hand or feet models
#Not sexist but the quality of womens darts is terrible
#Women are responsible for childhood obesity
#I just can't enjoy a game as much with a woman announcer calling it
#Girls are not even adults when they are children playing with dolls

#Naegative
#Why are you so irritating

#safe
#would you like to hangout with me?