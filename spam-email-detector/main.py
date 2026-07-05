import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# Load the real dataset
data = pd.read_csv(
    "SMSSpamCollection",
    sep="\t",
    names=["Label", "Message"]
)

# Features and Labels
X = data["Message"]
y = data["Label"]

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Convert text into numbers
vectorizer = CountVectorizer()

X_train_vector = vectorizer.fit_transform(X_train)
X_test_vector = vectorizer.transform(X_test)

# Train model
model = MultinomialNB()
model.fit(X_train_vector, y_train)

# Test model
predictions = model.predict(X_test_vector)

print("Accuracy:", round(accuracy_score(y_test, predictions) * 100, 2), "%")

print("\nClassification Report:")
print(classification_report(y_test, predictions))

# User prediction
message = input("\nEnter a message: ")

message_vector = vectorizer.transform([message])

prediction = model.predict(message_vector)

print("\nPrediction:", prediction[0])