import pandas as pd

def load_kaggle_data():
    fake = pd.read_csv("data/raw/Fake.csv")
    real = pd.read_csv("data/raw/True.csv")
    fake["label"] = 0
    real["label"] = 1  
    df = pd.concat([fake, real], ignore_index=True)
    df["text"] = (df["title"].fillna("") + " " + df["text"].fillna("")).str.strip()
    df = df[["text", "label"]]
    return df

if __name__=="__main__":
    df = load_kaggle_data()

    print(df.head(5))
    print("Shape:", df.shape)
    print("\nClass balance:")
    print(df["label"].value_counts())

    print("\nBlank text rows:", (df["text"].str.strip() == "").sum())
    print("Duplicate rows:", df.duplicated(subset="text").sum())

    df = df.drop_duplicates(subset="text")
    df = df[df["text"].str.strip() != ""]

    print("\nShape after cleaning:", df.shape)

    out_path = "data/processed/kaggle_combined.csv"
    df.to_csv(out_path, index=False)
    print(f"Saved to {out_path}")

    



