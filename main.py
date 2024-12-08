import streamlit as st
import os
import pickle

# Construct file paths dynamically
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
clf_path = os.path.join(BASE_DIR, 'clf.pkl')
tfidf_path = os.path.join(BASE_DIR, 'tfidf.pkl')

# Load the models
clf_svm = pickle.load(open(clf_path, 'rb'))
tfidf = pickle.load(open(tfidf_path, 'rb'))

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
