import streamlit as st
import pickle
import re
import nltk
import requests

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# ------------------ CONFIG ------------------
st.set_page_config(page_title="Fake News Detector", layout="centered")

# ------------------ LOAD MODEL ------------------
model = pickle.load(open("models/fake_news_model.pkl", "rb"))
vectorizer = pickle.load(open("models/vectorizer.pkl", "rb"))

# ------------------ NLTK SETUP ------------------
nltk.download('stopwords')
nltk.download('wordnet')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# ------------------ API KEY ------------------
API_KEY = "0e101aa5ae884683a4859278d2ffdb46"   # 🔥 Replace this

# ------------------ CLEAN TEXT ------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\s+', ' ', text)

    words = text.split()
    words = [word for word in words if word not in stop_words]
    words = [lemmatizer.lemmatize(word) for word in words]

    return " ".join(words)

# ------------------ NEWS API ------------------
def verify_with_news_api(query):
    url = "https://newsapi.org/v2/everything"

    params = {
        "q": query,
        "apiKey": API_KEY,
        "language": "en",
        "pageSize": 3
    }

    try:
        response = requests.get(url, params=params)
        data = response.json()

        if data.get("totalResults", 0) > 0:
            return data["articles"]
        else:
            return []
    except:
        return []

# ------------------ UI ------------------
st.title("📰 Fake News Detector")

st.info("⚠️ This model predicts based on writing patterns and verifies using online news sources.")

input_text = st.text_area("Enter news text:")

# ------------------ PREDICTION ------------------
if st.button("Predict"):

    # Empty input
    if input_text.strip() == "":
        st.warning("⚠️ Please enter some text")

    # Too short input
    elif len(input_text.split()) < 10:
        st.warning("⚠️ Please enter a more detailed news article (at least 10 words)")

    else:
        # Clean text
        cleaned = clean_text(input_text)

        # Vectorize
        vectorized = vectorizer.transform([cleaned])

        # Predict
        prediction = model.predict(vectorized)
        probability = model.predict_proba(vectorized)

        confidence = max(probability[0]) * 100

        # Show ML result
        if prediction[0] == 0:
            st.error(f"❌ Fake News ({confidence:.2f}% confidence)")
        else:
            st.success(f"✅ Real News ({confidence:.2f}% confidence)")

        # ------------------ API VERIFICATION ------------------
        st.markdown("### 🌐 Online Verification")

        articles = verify_with_news_api(input_text)

        if len(articles) > 0:
            st.info("📰 Similar news found online:")

            for article in articles:
                st.write("🔗", article["title"])
        else:
            st.warning("⚠️ No similar news found online")