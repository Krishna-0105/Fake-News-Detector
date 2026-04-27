import streamlit as st
import pickle
import re
import string
import requests
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# ==============================
# LOAD MODEL & VECTORIZER
# ==============================
model = pickle.load(open("models/fake_news_model.pkl", "rb"))
vectorizer = pickle.load(open("models/vectorizer.pkl", "rb"))

# ==============================
# NLP SETUP
# ==============================
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

# ==============================
# NEWS API KEY
# ==============================
API_KEY = "0e101aa5ae884683a4859278d2ffdb46"

# ==============================
# TEXT CLEANING
# ==============================
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>+', '', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub(r'\n', '', text)
    text = re.sub(r'\w*\d\w*', '', text)

    words = text.split()
    words = [word for word in words if word not in stop_words]
    words = [lemmatizer.lemmatize(word) for word in words]

    return " ".join(words)

# ==============================
# BETTER QUERY (IMPORTANT FIX)
# ==============================
def create_search_query(text):
    words = text.split()
    return " ".join(words[:8])   # first 6–8 words only

# ==============================
# API VERIFICATION (IMPROVED)
# ==============================
def verify_with_news_api(query):
    url = "https://newsapi.org/v2/everything"

    params = {
        "q": query,
        "apiKey": API_KEY,
        "pageSize": 5,
        "sortBy": "relevancy",
        "language": "en"
    }

    try:
        response = requests.get(url, params=params, timeout=5)
        data = response.json()
    except:
        return False, []

    if data.get("status") != "ok" or data.get("totalResults", 0) == 0:
        return False, []

    articles = data.get("articles", [])

    query_words = set(query.lower().split())
    relevant_articles = []

    for article in articles:
        title = (article.get("title") or "").lower()
        description = (article.get("description") or "").lower()

        content = title + " " + description

        match_count = sum(1 for word in query_words if word in content)

        # 🔥 RELAXED MATCH (IMPORTANT)
        if match_count >= 1:
            relevant_articles.append(article)

    if len(relevant_articles) >= 1:
        return True, relevant_articles

    return False, []

# ==============================
# UI
# ==============================
st.set_page_config(page_title="Fake News Detector", layout="wide")

st.title("📰 Fake News Detector")
st.subheader("🔎 Analyze News Authenticity")

st.markdown("""
Enter a news article below and the system will:
- Predict if it's Fake or Real using ML
- Cross-check with real-world news sources
""")

input_text = st.text_area("✍️ Enter News Text")

# ==============================
# PREDICTION
# ==============================
if st.button("Predict"):

    if input_text.strip() == "":
        st.warning("Please enter some text")

    else:
        # ==============================
        # ML
        # ==============================
        cleaned = clean_text(input_text)
        vectorized = vectorizer.transform([cleaned])

        prediction = model.predict(vectorized)[0]
        confidence = model.predict_proba(vectorized)[0].max()

        st.subheader("🧠 ML Prediction")

        if prediction == 0:
            st.error(f"❌ Fake News ({confidence*100:.2f}%)")
        else:
            st.success(f"✅ Real News ({confidence*100:.2f}%)")

        # ==============================
        # BETTER QUERY
        # ==============================
        search_query = create_search_query(input_text)

        # ==============================
        # API CHECK
        # ==============================
        st.subheader("🌐 Online Verification")

        is_real, articles = verify_with_news_api(search_query)

        if is_real:
            st.success("✅ Found similar real news online")

            st.markdown("### 📰 Related Articles:")
            for article in articles:
                st.markdown(f"- [{article['title']}]({article['url']})")

        else:
            st.warning("⚠️ No strong match found online")

        # ==============================
        # 🔥 FINAL VERDICT (SMART LOGIC)
        # ==============================
        st.subheader("🧾 Final Verdict")

        if confidence < 0.60:
            st.warning("🟡 FINAL: UNCERTAIN (Low ML confidence)")

        elif prediction == 1 and is_real:
            st.success("🟢 FINAL: REAL NEWS (ML + Internet match)")

        elif prediction == 0 and not is_real:
            st.error("🔴 FINAL: FAKE NEWS (Strong ML signal)")

        else:
            st.warning("🟡 FINAL: UNCERTAIN (Conflict between ML & API)")