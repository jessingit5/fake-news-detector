import re
import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path
from text_utils import clean_text

script_dir = Path(__file__).parent



if __name__ == "__main__":
    combined_dataset = script_dir.parent / "data" / "processed" / "kaggle_combined.csv"
    train = script_dir.parent / "data" / "processed" / "train.csv"
    test  = script_dir.parent / "data" / "processed" / "test.csv"

    df = pd.read_csv(combined_dataset)

    df["text"] = df["text"].apply(clean_text)
    df = df[df["text"].str.strip() != ""]

    train_df, test_df = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df["label"]
    )
    

    train_df.to_csv(train, index=False)
    test_df.to_csv(test, index=False)

    print("Train shape:", train_df.shape)
    print("Test shape:", test_df.shape)
    print("\nTrain class balance:")
    print(train_df["label"].value_counts(normalize=True))
    print("\nTest class balance:")
    print(test_df["label"].value_counts(normalize=True))