"""
predict.py
==========
Prediction utility — used by Flask app to serve real-time predictions.

Usage (standalone):
    python predict.py
"""

import os
import json
import numpy as np
import joblib

BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH   = os.path.join(BASE_DIR, "model.pkl")
SCALER_PATH  = os.path.join(BASE_DIR, "scaler.pkl")
ENCODER_PATH = os.path.join(BASE_DIR, "label_encoders.pkl")
META_PATH    = os.path.join(BASE_DIR, "model_meta.json")


def load_artifacts():
    """Load trained model, scaler, label encoders and metadata."""
    model   = joblib.load(MODEL_PATH)
    scaler  = joblib.load(SCALER_PATH)
    encoders= joblib.load(ENCODER_PATH)
    with open(META_PATH, "r", encoding="utf-8") as f:
        meta = json.load(f)
    return model, scaler, encoders, meta


def predict_sales(input_data: dict) -> dict:
    """
    Predict sales from user input.

    Parameters
    ----------
    input_data : dict
        Keys: TV_Spend, Radio_Spend, Newspaper_Spend, Social_Media_Spend,
              Digital_Spend, Platform, Campaign_Type, Region, Month,
              Target_Audience

    Returns
    -------
    dict with predicted_sales, growth_pct, efficiency_score, recommendation
    """
    model, scaler, encoders, meta = load_artifacts()

    # ── Numeric base features ─────────────────────────────────
    tv        = float(input_data.get("TV_Spend", 0))
    radio     = float(input_data.get("Radio_Spend", 0))
    newspaper = float(input_data.get("Newspaper_Spend", 0))
    social    = float(input_data.get("Social_Media_Spend", 0))
    digital   = float(input_data.get("Digital_Spend", 0))

    # ── Engineered features ───────────────────────────────────
    total_spend = tv + radio + newspaper + social + digital
    roi         = 0.0  # will be estimated after prediction
    cost_per_sale = 0.0

    q4_months = {"October", "November", "December"}
    month = str(input_data.get("Month", "January")).strip().title()
    seasonal_indicator = 1 if month in q4_months else 0

    weights = {"TV_Spend": 0.30, "Radio_Spend": 0.15, "Newspaper_Spend": 0.10,
               "Social_Media_Spend": 0.20, "Digital_Spend": 0.25}
    marketing_efficiency = (
        tv * 0.30 + radio * 0.15 + newspaper * 0.10 +
        social * 0.20 + digital * 0.25
    )

    # ── Categorical encoding ──────────────────────────────────
    cat_features = {
        "Platform":        str(input_data.get("Platform", "Google Ads")).strip().title(),
        "Campaign_Type":   str(input_data.get("Campaign_Type", "Conversion")).strip().title(),
        "Region":          str(input_data.get("Region", "North")).strip().title(),
        "Month":           month,
        "Target_Audience": str(input_data.get("Target_Audience", "25-34")).strip(),
    }

    encoded_cats = []
    for col, val in cat_features.items():
        if col in encoders:
            le = encoders[col]
            # Handle unseen labels gracefully
            if val in le.classes_:
                encoded_cats.append(le.transform([val])[0])
            else:
                encoded_cats.append(0)
        else:
            encoded_cats.append(0)

    # ── Assemble feature vector ───────────────────────────────
    # Order must match training:
    # TV_Spend, Radio_Spend, Newspaper_Spend, Social_Media_Spend, Digital_Spend,
    # Platform, Campaign_Type, Region, Month, Target_Audience,
    # Total_Spend, ROI, Cost_Per_Sale, Seasonal_Indicator, Marketing_Efficiency
    feature_vector = np.array([[
        tv, radio, newspaper, social, digital,
        *encoded_cats,
        total_spend, roi, cost_per_sale, seasonal_indicator, marketing_efficiency
    ]])

    # ── Predict ───────────────────────────────────────────────
    # Try with scaler first (Linear Regression was trained on scaled data)
    try:
        predicted = float(model.predict(feature_vector)[0])
    except Exception:
        feature_scaled = scaler.transform(feature_vector)
        predicted = float(model.predict(feature_scaled)[0])

    predicted = max(predicted, 0)  # no negative sales

    # ── Post-prediction metrics ───────────────────────────────
    avg_sales = meta.get("avg_sales", predicted)
    growth_pct = round(((predicted - avg_sales) / avg_sales) * 100, 1) if avg_sales else 0

    if total_spend > 0:
        roi_val = round((predicted / total_spend) * 100, 1)
    else:
        roi_val = 0

    # Efficiency label
    if roi_val > 300:
        eff_label = "Excellent"
    elif roi_val > 200:
        eff_label = "Good"
    elif roi_val > 100:
        eff_label = "Average"
    else:
        eff_label = "Below Average"

    # Business recommendation
    if predicted > avg_sales * 1.2:
        recommendation = ("🚀 Great campaign! Allocate more budget to "
                          f"{cat_features['Platform']} for even higher returns.")
    elif predicted > avg_sales * 0.8:
        recommendation = ("📈 Solid performance. Consider increasing Digital "
                          "and Social Media spend for incremental growth.")
    else:
        recommendation = ("⚠️ Below average projection. Review campaign strategy — "
                          "shift budget to higher-ROI channels like Google Ads.")

    return {
        "predicted_sales":     round(predicted * 1000, 0),   # convert K → full ₹
        "predicted_sales_k":   round(predicted, 2),
        "growth_pct":          growth_pct,
        "roi":                 roi_val,
        "efficiency_label":    eff_label,
        "recommendation":      recommendation,
        "total_spend_k":       round(total_spend, 2),
        "best_model":          meta.get("best_model", "N/A"),
        "model_accuracy":      meta.get("best_r2", 0),
    }


# ── Standalone test ───────────────────────────────────────────
if __name__ == "__main__":
    sample = {
        "TV_Spend":          150.0,
        "Radio_Spend":        80.0,
        "Newspaper_Spend":    30.0,
        "Social_Media_Spend": 120.0,
        "Digital_Spend":      200.0,
        "Platform":          "Google Ads",
        "Campaign_Type":     "Conversion",
        "Region":            "West",
        "Month":             "November",
        "Target_Audience":   "25-34",
    }
    result = predict_sales(sample)
    print("\n── PREDICTION RESULT ──────────────────────")
    print(f"  Predicted Sales    : ₹{result['predicted_sales']:,.0f}")
    print(f"  Growth %           : {result['growth_pct']}%")
    print(f"  ROI                : {result['roi']}%")
    print(f"  Efficiency         : {result['efficiency_label']}")
    print(f"  Recommendation     : {result['recommendation']}")
