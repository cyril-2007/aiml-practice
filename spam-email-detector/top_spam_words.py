import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Load dataset
data = pd.read_csv(
    "SMSSpamCollection",
    sep="\t",
    names=["Label", "Message"]
)

X = data["Message"]
y = data["Label"]

# Convert text into numbers
vectorizer = CountVectorizer()
X_vectorized = vectorizer.fit_transform(X)

# Train model
model = MultinomialNB()
model.fit(X_vectorized, y)

# Get feature names
feature_names = vectorizer.get_feature_names_out()

# Get spam probabilities
spam_index = list(model.classes_).index("spam")
spam_probs = model.feature_log_prob_[spam_index]

# Top 20 spam words
top_indices = spam_probs.argsort()[-20:][::-1]

print("Top 20 words associated with spam:\n")

for i in top_indices:
    print(feature_names[i])