import pandas as pd 

def load_data(path): 
    df = pd.read_csv(path, encoding="latin-1")
    df["text"] = df["Subject"].fillna("") + " " + df["Message"].fillna("")
    df["label"] = df["Spam/Ham"].map({"ham":0, "spam": 1})
    return df