#importing necessary libraries

import numpy as np
import pandas as pd

import string

import streamlit as st

header = st.container()
dataset = st.container()
fearure = st.container()
model_training = st.container()

def get_data(file_name):
    df = pd.read_csv(file_name , header = None)
    return df

with header:
    st.title("Emotion detection using Text")

with dataset:
    st.header("Emotion Detection Datasets")
    df = get_data("1-P-3-ISEAR.csv")
    df.columns = ['sn','Target','Sentence']
    df.drop('sn',inplace=True,axis =1)

    df.head()

    
    df.duplicated().sum()

    df.drop_duplicates(inplace = True)

    
    st.subheader("Lets check if the dataset is fairly distrributed.")

    col1 , col2 = st.columns(2)

    target_count = df['Target'].value_counts()

    col1.table(target_count)
    col2.text("Line Chart of the total output counts")

    col2.line_chart(target_count )

    st.markdown("From the above data, we can easily say the data iss fairly distributed.")

with fearure:
    st.header("Learning about Feature and converting them")

    def lowercase(text):
        text = text.lower()
        return text
    # df['Sentence'] = df['Sentence'].apply(lowercase)
    
    def remove_punc(text):
        text = "".join([char for char in text if char not in string.punctuation and not char.isdigit()])
        return text


    df['Sentence'] = df['Sentence'].apply(lowercase).apply(remove_punc)
    

    #Removing the stop words
    import nltk
    nltk.download('omw-1.4')

    nltk.download('stopwords')
    
    from nltk.corpus import stopwords
    def remove_stopwords(text):
        text = [w for w in text.split() if w not in stopwords.words('english')]
        return ' '.join(text)
    df['Sentence'] = df['Sentence'].apply(remove_stopwords)

    #Lemmatization i.e changing words into it's root form
    from nltk.stem import WordNetLemmatizer
    nltk.download('wordnet')
    from nltk.corpus import wordnet    
    lemmatizer = WordNetLemmatizer()
    def lemmatize(text):
        text = [lemmatizer.lemmatize(word,'v') for word in text.split()]
        return ' '.join(text)
    df['Sentence'] = df['Sentence'].apply(lemmatize)

    st.markdown('As the part of data pre-processing, we have done the following things:')

    st.text(" - Converting the sentence to lowercase ")
    st.text(" -Removing the Punction ")
    st.text(" -Removing the stop words ")
    st.text(" -Lemmatization i.e changing words into it is root form ,")

    st.markdown("After all these our data looks like-")

    st.dataframe(df.head())

    st.markdown("After doing Train Test split we will apply TGIF, It is technique to transform text into a meaningful vector of numbers. TFIDF penalizes words that come up too often and dont really have much use. So it rescales the frequency of words that are common which makes scoring more balanced")

with model_training:
    from sklearn.model_selection import train_test_split
    X = df['Sentence']
    y = df['Target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2,random_state=10)

    from sklearn.feature_extraction.text import TfidfVectorizer
    tfidf = TfidfVectorizer(min_df=2, max_df=0.5, ngram_range=(1, 2))
    train_tfidf = tfidf.fit_transform(X_train)
    test_tfidf = tfidf.transform(X_test)

    from sklearn.linear_model import LogisticRegression
    logistic = LogisticRegression(max_iter=1000)
    logistic.fit(train_tfidf,y_train)

    from sklearn.naive_bayes import MultinomialNB
    nb = MultinomialNB()
    nb.fit(train_tfidf,y_train)

    st.header('Checking The Accuracy using diffrent model.')

    import joblib


    joblib.dump(logistic, './mymodel/logistic_model.joblib')
    joblib.dump(nb, './mymodel/naive_bayes_model.joblib')
    joblib.dump(tfidf, './mymodel/tfidf_model.joblib')

    sel_col , disp_col = st.columns(2)

    with sel_col:

        sel_col.subheader("Logistic Regression")

        sel_col.markdown("Logistic Regression Train Error")
        sel_col.write(logistic.score(train_tfidf, y_train))

        sel_col.markdown("Logistic Regression Test Error")
        sel_col.write( logistic.score(test_tfidf, y_test))
    
    with disp_col:

        disp_col.subheader("Naive Bias")

        disp_col.markdown("Naive Bias Train Error")
        disp_col.write(nb.score(train_tfidf, y_train))

        disp_col.markdown("Naive Bias Test Error")
        disp_col.write(nb.score(test_tfidf, y_test))
