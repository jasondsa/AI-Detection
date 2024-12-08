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

# Set page configuration
st.set_page_config(
    page_title="AI Text Classifier",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Sidebar
st.sidebar.title("AI vs Human Detector")
st.sidebar.write("This tool classifies text as either written by AI or by a human.")

# Main Title
st.markdown(
    """
    <div style="text-align: center; background-color: #007bff; padding: 20px; border-radius: 10px;">
        <h1 style="color: white; font-family: Arial, sans-serif;">AI Text Classifier</h1>
    </div>
    """,
    unsafe_allow_html=True
)

# Input form
st.markdown("### Enter your text below to classify:")
message = st.text_area("Message", height=200)

# Classification button
if st.button("Classify"):
    if message.strip():
        text = tfidf.transform([message])
        result = clf_svm.predict(text)
        if result == 1:
            st.success("The text is likely written by **AI**.")
        else:
            st.info("The text is likely written by **Human**.")
    else:
        st.warning("Please enter some text to classify.")

# Footer
st.markdown(
    """
    <hr>
    <p style="text-align: center; font-size: 12px; color: grey;">Developed by Jason D'sa | © 2024</p>
    """,
    unsafe_allow_html=True
)
