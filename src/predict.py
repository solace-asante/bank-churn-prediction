import json
import sys
from pathlib import Path

import joblib
import pandas as pd

from clean import prepare_features

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "churn_pipeline.joblib"
META_PATH = ROOT / "models" / "churn_meta.json"


def load_model():
    pipe = joblib.load(MODEL_PATH)
    meta = json.loads(META_PATH.read_text())
    return pipe, meta


def predict_churn(df, pipe, meta):
    X = prepare_features(df)                      
    probs = pipe.predict_proba(X)[:, 1]

    out = df.copy()
    out["churn_probability"] = probs
    out["flagged"] = probs >= meta["threshold"]
    return out


def main():
    if len(sys.argv) < 2:
        sys.exit("usage: python src/predict.py path/to/customers.csv")

    df = pd.read_csv(sys.argv[1])
    pipe, meta = load_model()
    result = predict_churn(df, pipe, meta)

    print(result[["churn_probability", "flagged"]].head())
    print(f"flagged {int(result['flagged'].sum())} of {len(result)} customers")


if __name__ == "__main__":
    main()