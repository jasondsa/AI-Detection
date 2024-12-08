import streamlit as st
import pickle
import numpy as np

clf_svm = pickle.load(open('clf.pkl', 'rb'))
tfidf = pickle.load(open('tfidf.pkl', 'rb'))

st.title("AI Text Classifier")

message = st.text_input("Enter your text:")
if st.button("Classify"):
    if message:
        text = tfidf.transform([message])
        result = clf_svm.predict(text)
        if result == 1:
            st.write("The text is likely written by AI")
        else:
            st.write("The text is likely written by Human")
