import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

if __name__ == "__main__":
    train_df = pd.read_csv("data/processed/train.csv")
    test_df = pd.read_csv("data/processed/test.csv")

    train_df["text"] = train_df["text"].fillna("")
    test_df["text"] = test_df["text"].fillna("")

    y_train = train_df["label"]
    y_test = test_df["label"]

    # Baseline 1: majority class
    dummy = DummyClassifier(strategy="most_frequent")
    dummy.fit(train_df[["text"]], y_train)
    dummy_preds = dummy.predict(test_df[["text"]])
    print(f"Majority-class baseline accuracy: {accuracy_score(y_test, dummy_preds):.4f}")

    # Shared features for both simple models
    vectorizer = TfidfVectorizer(stop_words="english", max_df=0.7)
    X_train = vectorizer.fit_transform(train_df["text"])
    X_test = vectorizer.transform(test_df["text"])

    # Baseline 2: Naive Bayes
    nb = MultinomialNB()
    nb.fit(X_train, y_train)
    nb_preds = nb.predict(X_test)
    print(f"Naive Bayes baseline accuracy: {accuracy_score(y_test, nb_preds):.4f}")

    # Baseline 3: Logistic Regression
    logreg = LogisticRegression(max_iter=1000)
    logreg.fit(X_train, y_train)
    logreg_preds = logreg.predict(X_test)
    print(f"Logistic regression baseline accuracy: {accuracy_score(y_test, logreg_preds):.4f}")