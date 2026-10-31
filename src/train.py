import json
from pathlib import Path

import joblib
import pandas as pd
import sklearn
import xgboost
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier

from clean import prepare_training

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "Churn_Modelling.csv"
MODEL_DIR = ROOT / "models"

THRESHOLD = 0.15   


def main():
    df = pd.read_csv(RAW)
    X, y = prepare_training(df)

    preprocess = ColumnTransformer(
        [("categorical", OneHotEncoder(handle_unknown="ignore"),
          ["Geography", "NumOfProducts"])],
        remainder="passthrough",
    )

    pipe = Pipeline([
        ("preprocess", preprocess),
        ("model", XGBClassifier(
            objective="binary:logistic",
            learning_rate=0.1,
            max_depth=3,
            min_child_weight=5,
            n_estimators=200,
            scale_pos_weight=1,
            random_state=42,
            n_jobs=1,
        )),
    ])

    print(f"Training final production model on all {len(X)} rows...")
    pipe.fit(X, y)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, MODEL_DIR / "churn_pipeline.joblib")

    meta = {
        "threshold": THRESHOLD,
        "features": list(X.columns),
        "n_train_rows": len(X),
        "xgboost": xgboost.__version__,
        "sklearn": sklearn.__version__,
        "test_estimate": {"auc": 0.867, "precision": 0.431, "recall": 0.835,
                          "note": "from the 8,000 row model on the held out test set"},
    }
    
    (MODEL_DIR / "churn_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"Saved pipeline and metadata to: {MODEL_DIR}")


if __name__ == "__main__":
    main()