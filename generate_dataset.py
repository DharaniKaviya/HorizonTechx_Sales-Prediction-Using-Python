"""
generate_dataset.py
-------------------
Generates a realistic synthetic sales dataset (1200 rows) and saves it to
dataset/sales_data.csv. This script is called automatically by train_model.py
if the dataset does not already exist.
"""

import numpy as np
import pandas as pd
import os

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# ---------------------------------------------------------------------------
# Categorical pools
# ---------------------------------------------------------------------------
PLATFORMS = ["Facebook", "Instagram", "Google Ads", "YouTube", "Twitter", "LinkedIn"]
CAMPAIGN_TYPES = ["Brand Awareness", "Lead Generation", "Conversion", "Retargeting", "Seasonal Offer"]
REGIONS = ["North", "South", "East", "West", "Central"]
MONTHS = ["January", "February", "March", "April", "May", "June",
          "July", "August", "September", "October", "November", "December"]
AUDIENCES = ["18-24", "25-34", "35-44", "45-54", "55+"]

N = 1200  # number of records

# ---------------------------------------------------------------------------
# Base advertising budgets (₹ thousands)
# ---------------------------------------------------------------------------
tv         = np.random.uniform(10, 300, N)
radio      = np.random.uniform(5, 150, N)
newspaper  = np.random.uniform(2, 80, N)
social     = np.random.uniform(5, 200, N)
digital    = np.random.uniform(10, 250, N)

platform      = np.random.choice(PLATFORMS, N)
campaign_type = np.random.choice(CAMPAIGN_TYPES, N)
region        = np.random.choice(REGIONS, N)
month         = np.random.choice(MONTHS, N)
audience      = np.random.choice(AUDIENCES, N)

# ---------------------------------------------------------------------------
# Platform multipliers (simulate real-world channel effectiveness)
# ---------------------------------------------------------------------------
platform_mult = {
    "Facebook": 1.20, "Instagram": 1.15, "Google Ads": 1.30,
    "YouTube": 1.10, "Twitter": 0.95, "LinkedIn": 1.05
}
campaign_mult = {
    "Brand Awareness": 0.90, "Lead Generation": 1.05, "Conversion": 1.25,
    "Retargeting": 1.30, "Seasonal Offer": 1.40
}
region_mult = {
    "North": 1.10, "South": 1.05, "East": 0.95, "West": 1.20, "Central": 1.00
}
month_mult = {
    "January": 0.80, "February": 0.85, "March": 0.95, "April": 1.00,
    "May": 1.05, "June": 1.10, "July": 1.00, "August": 1.05,
    "September": 1.10, "October": 1.20, "November": 1.35, "December": 1.50
}

# ---------------------------------------------------------------------------
# Compute Sales (₹ thousands) with realistic noise
# ---------------------------------------------------------------------------
p_mult = np.array([platform_mult[p] for p in platform])
c_mult = np.array([campaign_mult[c] for c in campaign_type])
r_mult = np.array([region_mult[r] for r in region])
m_mult = np.array([month_mult[mo] for mo in month])

sales = (
    3.5 * tv
    + 2.0 * radio
    + 1.2 * newspaper
    + 2.8 * social
    + 3.0 * digital
) * p_mult * c_mult * r_mult * m_mult + np.random.normal(0, 80, N)

# Clip negative sales
sales = np.clip(sales, 50, None)

# ---------------------------------------------------------------------------
# Introduce ~4 % missing values for realism
# ---------------------------------------------------------------------------
df = pd.DataFrame({
    "TV_Spend":          np.round(tv, 2),
    "Radio_Spend":       np.round(radio, 2),
    "Newspaper_Spend":   np.round(newspaper, 2),
    "Social_Media_Spend":np.round(social, 2),
    "Digital_Spend":     np.round(digital, 2),
    "Platform":          platform,
    "Campaign_Type":     campaign_type,
    "Region":            region,
    "Month":             month,
    "Target_Audience":   audience,
    "Sales":             np.round(sales, 2),
})

# Randomly set ~4 % of spend values to NaN
for col in ["TV_Spend", "Radio_Spend", "Newspaper_Spend", "Social_Media_Spend", "Digital_Spend"]:
    mask = np.random.rand(N) < 0.04
    df.loc[mask, col] = np.nan

# Introduce ~15 duplicate rows
dup_idx = np.random.choice(N, 15, replace=False)
df = pd.concat([df, df.iloc[dup_idx]], ignore_index=True)

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
out_path = os.path.join(os.path.dirname(__file__), "dataset", "sales_data.csv")
os.makedirs(os.path.dirname(out_path), exist_ok=True)
df.to_csv(out_path, index=False)
print(f"[Dataset] Saved {len(df)} rows to {out_path}")
