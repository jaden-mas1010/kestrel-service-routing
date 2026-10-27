import pandas as pd

df=pd.read_csv("data/train.csv")
count=df["team_label"].value_counts()
print("Before:",df["team_label"].nunique())

df["team_label"]=df["team_label"].replace({"Installations": "Installs & Demo",
    "Consumables": "Filters & Consumables"})


print("After",df["team_label"].nunique())
df.to_csv("data/train_cleaned.csv", index=False)