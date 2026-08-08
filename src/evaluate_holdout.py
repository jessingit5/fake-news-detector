import joblib
import pandas as pd
from text_utils import clean_text

if __name__ == "__main__":
    df = pd.read_csv("data/raw/news_api_real.csv")
    df["text"] = df["text"].fillna("").apply(clean_text)
    df = df[df["text"].str.strip() != ""]

    model = joblib.load("models/pac_classifier_sgd.joblib")
    vectorizer = joblib.load("models/tfidf_vectorizer.joblib")

    X = vectorizer.transform(df["text"])
    preds = model.predict(X)

    correct = (preds == 1).sum()
    total = len(preds)
    print(f"Correctly identified as real: {correct}/{total} ({correct/total:.2%})")

    misclassified = df.loc[preds == 0, "title"] if "title" in df.columns else None
    if misclassified is not None and len(misclassified) > 0:
        print("\nArticles incorrectly flagged as fake:")
        for title in misclassified:
            print(f" - {title}")
    else:
        print("\nNo misclassifications, model correctly called every article real.")
