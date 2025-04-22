import streamlit as st
import pickle

import os

model_path = 'sentiment_model.pkl'
vectorizer_path = 'tfidf_vectorizer.pkl'

with open(model_path, 'rb') as f:
    model = pickle.load(f)

with open(vectorizer_path, 'rb') as f:
    vectorizer = pickle.load(f)

from sklearn.utils.validation import check_is_fitted
check_is_fitted(vectorizer)


# App layout
st.title("Amazon Review Sentiment Analyzer 🛍️✨")

review = st.text_area("Enter an Amazon review:")

if st.button("Analyze Sentiment"):
    if review:
        # Transform the user input review using the loaded TF-IDF vectorizer
        vec = vectorizer.transform([review])
        
        # Check the number of features in the transformed input (should match the training set)
        st.write(f"Shape of transformed input: {vec.shape}")
        
        # Make a prediction using the loaded model
        prediction = model.predict(vec)[0]
        
        # Output the prediction result (assuming model outputs 1 for positive and 0 for negative)
        proba = model.predict_proba(vec)[0]
        positive_proba = proba[1]  # Probability for the positive class

        # Change the threshold to 0.4 if you want to classify reviews with a 40% chance as positive
        if positive_proba >= 0.4:
            st.success("Predicted Sentiment: **Positive Review** 😀")
        else:
            st.success("Predicted Sentiment: **Negative Review** 😞")

        # Display the prediction probabilities (if applicable)
        proba = model.predict_proba(vec)[0]
        st.write(f"Prediction Probabilities: Positive: {proba[1]:.2f}, Negative: {proba[0]:.2f}")
        
    else:
        st.warning("Please enter a review to analyze.")
