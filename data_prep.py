import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("data/tourism.csv")
df = df.drop(columns=[c for c in ["Unnamed: 0", "CustomerID"] if c in df.columns])
for c in df.select_dtypes(include="object").columns:
    df[c] = df[c].astype(str).str.strip().replace({"Fe Male": "Female"})
train, test = train_test_split(df, test_size=0.2, random_state=42, stratify=df["ProdTaken"])
train.to_csv("data/train.csv", index=False)
test.to_csv("data/test.csv", index=False)
print("Saved cleaned train/test datasets")
