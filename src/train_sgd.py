import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score

# FutureWarning: Class PassiveAggressiveClassifier is deprecated;
#  this is deprecated in version 1.8 and will be removed in 1.10. 
# Use `SGDClassifier(loss='hinge', penalty=None, learning_rate='pa1', eta0=1.0)` instead.
#   warnings.warn(msg, category=FutureWarning)

if __name__ == "__main__":
    train_df = pd.read_csv("data/processed/train.csv")
    test_df = pd.read_csv("data/processed/test.csv")

    train_df["text"] = train_df["text"].fillna("")
    test_df["text"] = test_df["text"].fillna("")

    vectorizer = TfidfVectorizer(stop_words="english", max_df=0.7)
    X_train = vectorizer.fit_transform(train_df["text"])
    X_test = vectorizer.transform(test_df["text"])

    y_train = train_df["label"]
    y_test = test_df["label"]

    model = SGDClassifier(
        loss="hinge",
        penalty=None,
        learning_rate="pa1",
        eta0=1.0,
        max_iter=50,
        random_state=42,
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"Test accuracy: {acc:.4f}")

    joblib.dump(model, "models/pac_classifier_sgd.joblib")
    joblib.dump(vectorizer, "models/tfidf_vectorizer.joblib")
    print("Saved model and vectorizer to models/")