import pandas as pd

# Load datasets
fake = pd.read_csv("data/Fake.csv")
true = pd.read_csv("data/True.csv")

# Add labels
fake["label"] = 0   # Fake news
true["label"] = 1   # Real news

# Combine datasets
data = pd.concat([fake, true], axis=0)

# Shuffle dataset
data = data.sample(frac=1).reset_index(drop=True)

# Display output
print(data.head())
print("\nData Info:\n")
print(data.info())
print("\nLabel Count:\n")
print(data["label"].value_counts())