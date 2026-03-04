import joblib
import numpy as np
import pandas as pd
from pathlib import Path

MODEL_DIR = Path(__file__).parent.parent / "models"

@st_cache := None  # placeholder — caching handled in app.py

def load_artifacts():
    model    = joblib.load(MODEL_DIR / "soft_voting_tuned.pkl")
    scaler   = joblib.load(MODEL_DIR / "scaler.pkl")
    le       = joblib.load(MODEL_DIR / "label_encoder.pkl")
    features = joblib.load(MODEL_DIR / "features.pkl")
    return model, scaler, le, features


def predict(category: str, price: float, rating: float, discount: float,
            model, scaler, le, features):
    row = pd.DataFrame([{
        "Category": category,
        "Price":    price,
        "Rating":   rating,
        "Discount": discount,
    }])

    row["Category_enc"]     = le.transform(row["Category"].astype(str))
    row["Price_log"]        = np.log1p(row["Price"])
    row["Price_per_Rating"] = row["Price"] / (row["Rating"] + 0.01)
    row["Discount_flag"]    = (row["Discount"] > 0).astype(int)

    X = scaler.transform(row[features])
    pred  = model.predict(X)[0]
    proba = model.predict_proba(X)[0]

    return {
        "label":        "In Stock" if pred == 1 else "Out of Stock",
        "pred":         int(pred),
        "prob_in":      float(proba[1]),
        "prob_out":     float(proba[0]),
    }
