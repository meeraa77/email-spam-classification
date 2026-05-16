import streamlit as st
import pickle
import re
import nltk

# Download NLTK resources
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)

from nltk.corpus import stopwords

# Load model and vectorizer
model = pickle.load(open("spam_classifier_model.pkl", "rb"))
tfidf = pickle.load(open("tfidf_vectorizer.pkl", "rb"))

# Stopwords
stop_words = set(stopwords.words('english'))

# Text preprocessing function
def preprocess_text(text):
    text = text.lower()

    # Remove URLs
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)

    # Remove special characters and numbers
    text = re.sub(r"[^a-zA-Z]", " ", text)

    # Tokenization
    words = text.split()

    # Remove stopwords
    words = [word for word in words if word not in stop_words]

    return " ".join(words)

# Streamlit UI
st.title("📧 Email Spam Classifier")

st.write("Machine Learning based Email Spam Detection System")

user_input = st.text_area("Enter Email Message")

if st.button("Predict"):

    cleaned_input = preprocess_text(user_input)

    vector_input = tfidf.transform([cleaned_input])

    prediction = model.predict(vector_input)

    if prediction[0] == 1:
        st.error("🚨 Spam Email")
    else:
        st.success("✅ Ham Email")
