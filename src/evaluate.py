import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from pathlib import Path

script_dir = Path(__file__).parent

if __name__ == "__main__":
    test_path = script_dir.parent / "data" / "processed" / "test.csv"
    test_df = pd.read_csv(test_path)
    test_df["text"] = test_df["text"].fillna("")
    model = joblib.load("models/pac_classifier_sgd.joblib")
    vectorizer = joblib.load("models/tfidf_vectorizer.joblib")

    X_test = vectorizer.transform(test_df["text"])
    y_test = test_df["label"]
    preds = model.predict(X_test)

    acc = accuracy_score(y_test, preds)
    print(f"Test accuracy: {acc:.4f}\n")

    report = classification_report(y_test, preds, target_names=["fake", "real"])
    print("Classification report:")
    print(report)

    with open("reports/classification_report.txt", "w") as f:
        f.write(f"Test accuracy: {acc:.4f}\n\n{report}")

    cm = confusion_matrix(y_test, preds)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["fake", "real"], yticklabels=["fake", "real"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion matrix - PA classifier")
    plt.tight_layout()
    plt.savefig("reports/confusion_matrix.png", dpi=150)
    print("\nSaved classification_report.txt and confusion_matrix.png to reports/")
