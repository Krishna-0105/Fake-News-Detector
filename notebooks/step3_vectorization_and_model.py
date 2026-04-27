import pandas as pd
import re
import nltk
import pickle
import os

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Download NLTK data
nltk.download('stopwords')
nltk.download('wordnet')

# Initialize tools
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Load data
fake = pd.read_csv("data/Fake.csv")
true = pd.read_csv("data/True.csv")

# Labels
fake["label"] = 0
true["label"] = 1

# Combine
data = pd.concat([fake, true], axis=0)
data = data.sample(frac=1).reset_index(drop=True)

# 🔥 IMPORTANT CHANGE: combine title + text
data["content"] = data["title"] + " " + data["text"]

# Cleaning function
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\s+', ' ', text)

    words = text.split()
    words = [word for word in words if word not in stop_words]
    words = [lemmatizer.lemmatize(word) for word in words]

    return " ".join(words)

# Apply cleaning
data["clean_text"] = data["content"].apply(clean_text)

# 🔥 Better TF-IDF
vectorizer = TfidfVectorizer(max_features=10000)

X = vectorizer.fit_transform(data["clean_text"])
y = data["label"]

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 🔥 Better model for text
model = SGDClassifier(loss='log_loss')
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))

print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# Save model
os.makedirs("models", exist_ok=True)

with open("models/fake_news_model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("models/vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

print("\n✅ Model retrained and saved!")