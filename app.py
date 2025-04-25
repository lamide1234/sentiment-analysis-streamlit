import streamlit as st
import pickle
import re
import os

# Load model and vectorizer
model_path = 'logistic_regression_model.pkl'
vectorizer_path = 'tfidf_vectorizer.pkl'

with open(model_path, 'rb') as f:
    model = pickle.load(f)

with open(vectorizer_path, 'rb') as f:
    vectorizer = pickle.load(f)

# Load stopwords (assuming same list as in notebook)
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
sw = set(stopwords.words('english'))

# Preprocessing function
def preprocess_text(text):
    cleaned = re.sub(r"[^a-zA-Z\s]", '', text)
    cleaned = cleaned.lower().strip()
    cleaned = ' '.join([word for word in cleaned.split() if word not in sw])
    return cleaned

# App layout
st.title("Amazon Review Sentiment Analyzer 🛍️✨")

review = st.text_area("Enter an Amazon review:")

if st.button("Analyze Sentiment"):
    if review:
        cleaned_review = preprocess_text(review)
        vec = vectorizer.transform([cleaned_review])
        
        prediction = model.predict(vec)[0]
        proba = model.predict_proba(vec)[0]
        positive_proba = proba[1]

        if positive_proba >= 0.4:
            st.success("Predicted Sentiment: **Positive Review** 😀")
        else:
            st.success("Predicted Sentiment: **Negative Review** 😞")

        st.write(f"Prediction Probabilities: Positive: {proba[1]:.2f}, Negative: {proba[0]:.2f}")
    else:
        st.warning("Please enter a review to analyze.")
