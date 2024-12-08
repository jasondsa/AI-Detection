import streamlit as st
import os
import pickle
import re

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

# Dark theme styles
dark_theme_css = """
    <style>
        body {
            background-color: #121212;
            color: #ffffff;
        }
        h1 {
            color: #d31235; 
            text-align: center;
            font-family: Arial, sans-serif;
        }
        .stButton>button {
            background-color: #d31235; /* Button Color */
            color: white;
            border: none;
            padding: 10px 20px;
            border-radius: 8px;
            transition: background-color 0.3s ease, transform 0.2s ease;
            font-size: 16px;
        }
        .stButton>button:hover {
            background-color: #b00f2c; /* Darker Red on Hover */
            color: black;
            transform: scale(1.05); /* Slight zoom effect on hover */
        }
        .stAlert {
            background-color: #222222;
            border: 1px solid #444444;
            border-radius: 10px;
            padding: 15px;
        }
    </style>
"""

# Inject custom CSS for dark theme
st.markdown(dark_theme_css, unsafe_allow_html=True)

# Sidebar
st.sidebar.title("AI vs Human Detector")
st.sidebar.write("This tool classifies text as either written by AI or by a human.")
st.sidebar.markdown("---")

# Main Title
st.markdown(
    """
    <div style="text-align: center; background-color: #262730; padding: 20px; border-radius: 10px;">
        <h1 style="color: #d31235; font-family: Comic Sans MS, sans-serif;">AI Text Classifier</h1>
    </div>
    """,
    unsafe_allow_html=True
)

# Input form
st.markdown("### Enter your text below to classify:")
message = st.text_area("Message", height=200)

# Validation logic
def validate_message(msg):
    if not msg.strip():
        return "Message cannot be empty or only blank spaces."
    if not any(char.isalnum() for char in msg):
        return "Message must contain letters or numbers, not just special characters."
    if len(msg.split()) < 10:
        return "Message must contain at least 10 words."
    return None

# Classification button
if st.button("Classify"):
    validation_error = validate_message(message)
    if validation_error:
        st.markdown(
            f'<div class="stAlert stAlert-warning">{validation_error}</div>',
            unsafe_allow_html=True,
        )
    else:
        text = tfidf.transform([message])
        result = clf_svm.predict(text)
        if result == 1:
            st.markdown(
                '<div class="stAlert stAlert-success">The text is likely written by <b style="color: #FF4500;">AI</b>.</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div class="stAlert stAlert-info">The text is likely written by <b style="color: #00CED1;">Human</b>.</div>',
                unsafe_allow_html=True,
            )

# Footer
st.markdown(
    """
    <hr style="border: 1px solid #444444;">
    <p style="text-align: center; font-size: 12px; color: white;">Developed by Jason D'sa | © 2024</p>
    """,
    unsafe_allow_html=True
)
