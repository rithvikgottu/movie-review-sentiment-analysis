import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import pickle

df = pd.read_csv("data/sample_reviews.csv")
X = df['review']
y = df['sentiment']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

with open("vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

model = LogisticRegression(max_iter=1000)
model.fit(X_train_tfidf, y_train)

with open("sentiment_model.pkl", "wb") as f:
    pickle.dump(model, f)

y_pred = model.predict(X_test_tfidf)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

def predict_review(review_text):
    X_vec = vectorizer.transform([review_text])
    return model.predict(X_vec)[0]

if __name__ == "__main__":
    while True:
        review = input("Enter a movie review (or 'exit' to quit): ")
        if review.lower() == "exit":
            break
        print("Predicted sentiment:", predict_review(review))
