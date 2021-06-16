import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as s_graph
import pprint
import string
from nltk.corpus import stopwords
import streamlit as st
from PIL import Image

#Heading
st.write("""
# Emotion in Text Analysis
Information Retrival Project - Using Machine Learning & NLP
""")

#Importing Image
image = Image.open('C:/Users/Lenovo/Desktop/Emotion in Text Analysis/emotions.jpg')
st.image(image,use_column_width=True)

data = pd.read_csv('C:/Users/Lenovo/Desktop/Emotion in Text Analysis/text_emotion.csv')
st.subheader('Data Information')
st.dataframe(data)
st.write(data.describe())
st.subheader('Data Visualization')
st.bar_chart(data['sentiment'].value_counts())
st.area_chart(data['sentiment'].value_counts())


tweets = data['content']
def lower_case():
    for x in range(len(tweets)):
        tweets[x] = tweets[x].lower()

lower_case()

def remove_punctuation():
    for x in range(len(tweets)):
        word = tweets[x].translate(str.maketrans('','',string.punctuation))
        tweets[x] = word

remove_punctuation()

def tokenization_words():
    for x in range(len(tweets)):
        tweets[x] = tweets[x].split()

tokenization_words()

stopword=stopwords.words('english')

def removing_stop_words():
    for x in range(len(tweets)):
        for y in tweets[x]:
            if y in stopword:
                ind = tweets[x].index(y)
                tweets[x].pop(ind)

removing_stop_words()

def join_data():
    for x in range(len(tweets)):
        tweets[x] = ' '.join(tweets[x])

join_data()
emotion = data['sentiment'].unique().tolist()

def extract_words(dat):
    dat = dat.split()
    #print(dat)
    dat_1 = []
    dictionary = {}
    for x in dat:
        if x in dat_1:
            continue
        dat_1.append(x) 
    for y in dat_1:
        dictionary[y] = dat.count(y)
    return dictionary

anger_list = data[data['sentiment'] == 'anger']['content']
anger_list = ' '.join(anger_list)
anger_tokens = extract_words(anger_list)

from sklearn.naive_bayes import MultinomialNB,GaussianNB
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report,plot_confusion_matrix
from sklearn.model_selection import train_test_split

xfeatures = data['content']
ylabel = data['sentiment']
CV = CountVectorizer()
x=CV.fit_transform(xfeatures)

X_train,X_test,Y_train,Y_test = train_test_split(x,ylabel,test_size=0.3,random_state=42)
from sklearn import ensemble 
random_forest = ensemble.RandomForestClassifier(n_estimators=10)
random_forest = random_forest.fit(X_train,Y_train)
scores = random_forest.score(X_train,Y_train)*100
st.subheader("Model-Accuracy (Random Forest)")
st.write(str(scores)+'%')