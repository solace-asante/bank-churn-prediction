from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "Churn_Modelling.csv"

FEATURES = ["CreditScore", "Geography", "Gender", "Age", "Tenure",
            "Balance", "NumOfProducts", "HasCrCard", "IsActiveMember",
            "EstimatedSalary"]

NUMERIC = ["CreditScore", "Age", "Tenure", "Balance", "NumOfProducts",
           "HasCrCard", "IsActiveMember", "EstimatedSalary"]


def prepare_features(df):
    df = df.copy()
    df.columns = df.columns.str.strip()

    missing = set(FEATURES) - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    X = df[FEATURES].copy()
    X["Gender"] = (X["Gender"].astype(str).str.strip().str.title()
                   .map({"Female": 0, "Male": 1}))      
    for col in NUMERIC:
        X[col] = pd.to_numeric(X[col], errors="coerce")  
    return X




def prepare_training(df):
    df = df.copy()
    df.columns = df.columns.str.strip()

    df["Exited"] = pd.to_numeric(df["Exited"], errors="coerce")
    df = df[df["Exited"].isin([0, 1])]          

    return prepare_features(df), df["Exited"].astype(int)

if __name__ == "__main__":
    raw = pd.read_csv(RAW)
    X, y = prepare_training(raw)
    print(X.shape, y.shape, round(y.mean(), 4))
    