# 📰 Fake News Detection System

## 📌 Overview
This project detects whether a news article is **Real or Fake** using Machine Learning and Online Verification.

---

## 🚀 Features
- ML-based prediction using NLP
- Real-time verification using News API
- Confidence score display
- Final verdict (Real / Fake / Uncertain)

---

## 🧠 Technologies Used
- Python
- Streamlit
- Scikit-learn
- NLTK
- News API

---

## ⚙️ How It Works
1. User enters news text
2. Text is cleaned using NLP
3. Converted into vectors (TF-IDF)
4. ML model predicts result
5. News API verifies real-world presence
6. Final verdict is generated

---

## ▶️ Run Locally

```bash
git clone https://github.com/your-username/fake-news-detector.git
cd fake-news-detector
pip install -r requirements.txt
streamlit run app/app.py