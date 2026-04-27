import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download required NLTK data (runs once)
nltk.download('stopwords')
nltk.download('wordnet')

# Initialize tools
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Load data
fake = pd.read_csv("data/Fake.csv")
true = pd.read_csv("data/True.csv")

# Add labels
fake["label"] = 0
true["label"] = 1

# Combine datasets
data = pd.concat([fake, true], axis=0)
data = data.sample(frac=1).reset_index(drop=True)

# 🔥 Text cleaning function
def clean_text(text):
    text = text.lower()  # lowercase
    text = re.sub(r'\W', ' ', text)  # remove special characters
    text = re.sub(r'\s+', ' ', text)  # remove extra spaces

    words = text.split()

    # Remove stopwords
    words = [word for word in words if word not in stop_words]

    # Lemmatization
    words = [lemmatizer.lemmatize(word) for word in words]

    return " ".join(words)

# Apply cleaning
data["clean_text"] = data["text"].apply(clean_text)

# Show output
print(data[["text", "clean_text"]].head())