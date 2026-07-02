"""
train_model.py
==============
End-to-end ML pipeline for Sales Prediction.

Steps
-----
1. Load / generate dataset
2. Data cleaning
3. Exploratory Data Analysis (saves charts to static/images/)
4. Feature engineering
5. Model training & comparison (Linear, DT, RF, GB, XGBoost)
6. Best model selection & persistence (model.pkl, scaler.pkl, label_encoders.pkl)
7. Business insights report

Run:  python train_model.py
"""

# ============================================================
# 0. Imports & Configuration
# ============================================================
import os
import sys
import json
import warnings
import subprocess

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")           # headless backend for servers
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import joblib

warnings.filterwarnings("ignore")

# ── Paths ────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
DATASET_PATH= os.path.join(DATASET_DIR, "sales_data.csv")
IMG_DIR     = os.path.join(BASE_DIR, "static", "images")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
MODEL_PATH  = os.path.join(BASE_DIR, "model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "scaler.pkl")
ENCODER_PATH= os.path.join(BASE_DIR, "label_encoders.pkl")
META_PATH   = os.path.join(BASE_DIR, "model_meta.json")

os.makedirs(IMG_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

# ── Plot style ───────────────────────────────────────────────
sns.set_theme(style="darkgrid")
PALETTE = "coolwarm"
FIG_DPI  = 120
plt.rcParams.update({
    "figure.facecolor": "#0B0B12",
    "axes.facecolor":   "#12121F",
    "axes.edgecolor":   "#5B5FFF",
    "axes.labelcolor":  "#CCCCFF",
    "xtick.color":      "#CCCCFF",
    "ytick.color":      "#CCCCFF",
    "text.color":       "#FFFFFF",
    "grid.color":       "#1E1E38",
    "axes.titlecolor":  "#FFFFFF",
    "figure.titlesize": 14,
})

# ============================================================
# 1. Load / Generate Dataset
# ============================================================
def load_dataset():
    if not os.path.exists(DATASET_PATH):
        print("[INFO] Dataset not found – generating synthetic data...")
        exec(open(os.path.join(BASE_DIR, "generate_dataset.py")).read())
    df = pd.read_csv(DATASET_PATH)
    print(f"\n{'='*60}")
    print("1. DATA LOADING")
    print(f"{'='*60}")
    print(f"   Shape            : {df.shape}")
    print(f"   Rows             : {df.shape[0]}")
    print(f"   Columns          : {df.shape[1]}")
    print(f"\n   First 5 rows:\n{df.head()}")
    print(f"\n   Data Types:\n{df.dtypes}")
    print(f"\n   Statistical Summary:\n{df.describe()}")
    return df

# ============================================================
# 2. Data Cleaning
# ============================================================
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    print(f"\n{'='*60}")
    print("2. DATA CLEANING")
    print(f"{'='*60}")

    original_shape = df.shape

    # Remove duplicates
    df.drop_duplicates(inplace=True)
    print(f"   Duplicates removed   : {original_shape[0] - df.shape[0]}")

    # Missing value report
    missing = df.isnull().sum()
    print(f"\n   Missing values per column:\n{missing[missing > 0]}")

    # Fill numeric missing values with column median
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    for col in num_cols:
        if df[col].isnull().any():
            df[col].fillna(df[col].median(), inplace=True)

    # Fill categorical missing with mode
    cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
    for col in cat_cols:
        if df[col].isnull().any():
            df[col].fillna(df[col].mode()[0], inplace=True)

    # Correct data types
    for col in num_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df.dropna(inplace=True)

    # Outlier detection & capping (IQR method) for numeric spend cols
    spend_cols = [c for c in num_cols if c != "Sales"]
    for col in spend_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lo = Q1 - 1.5 * IQR
        hi = Q3 + 1.5 * IQR
        before = ((df[col] < lo) | (df[col] > hi)).sum()
        df[col] = df[col].clip(lower=lo, upper=hi)
        if before:
            print(f"   Outliers capped in '{col}': {before} rows")

    # Validate Sales > 0
    df = df[df["Sales"] > 0]

    # Standardise string categories
    for col in cat_cols:
        df[col] = df[col].str.strip().str.title()

    print(f"\n   Clean dataset shape  : {df.shape}")
    return df.reset_index(drop=True)

# ============================================================
# 3. Exploratory Data Analysis
# ============================================================
def run_eda(df: pd.DataFrame):
    print(f"\n{'='*60}")
    print("3. EXPLORATORY DATA ANALYSIS")
    print(f"{'='*60}")

    num_cols  = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols  = df.select_dtypes(include=["object"]).columns.tolist()
    spend_cols= [c for c in num_cols if c != "Sales"]

    # ── 3.1 Sales Distribution ────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(df["Sales"], bins=40, color="#5B5FFF", edgecolor="#8A2BE2", alpha=0.85)
    ax.set_title("Sales Distribution", fontsize=16, fontweight="bold")
    ax.set_xlabel("Sales (₹ Thousands)")
    ax.set_ylabel("Frequency")
    plt.tight_layout()
    fig.savefig(os.path.join(IMG_DIR, "sales_distribution.png"), dpi=FIG_DPI)
    plt.close(fig)
    print("   Saved: sales_distribution.png")

    # ── 3.2 Advertising Spend Distribution ───────────────────
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()
    colors = ["#5B5FFF", "#8A2BE2", "#FF6B9D", "#00D4FF", "#FFB347"]
    for i, col in enumerate(spend_cols[:5]):
        axes[i].hist(df[col], bins=30, color=colors[i], edgecolor="white", alpha=0.80)
        axes[i].set_title(col.replace("_", " "), fontsize=12)
        axes[i].set_xlabel("₹ Thousands")
    axes[5].set_visible(False)
    fig.suptitle("Advertising Spend Distribution", fontsize=16, fontweight="bold")
    plt.tight_layout()
    fig.savefig(os.path.join(IMG_DIR, "spend_distribution.png"), dpi=FIG_DPI)
    plt.close(fig)
    print("   Saved: spend_distribution.png")

    # ── 3.3 Platform-wise Sales ───────────────────────────────
    if "Platform" in df.columns:
        platform_sales = df.groupby("Platform")["Sales"].mean().sort_values(ascending=True)
        fig, ax = plt.subplots(figsize=(10, 5))
        bars = ax.barh(platform_sales.index, platform_sales.values,
                       color=sns.color_palette("coolwarm", len(platform_sales)))
        ax.set_title("Average Sales by Platform", fontsize=16, fontweight="bold")
        ax.set_xlabel("Average Sales (₹ Thousands)")
        for bar, val in zip(bars, platform_sales.values):
            ax.text(bar.get_width() + 5, bar.get_y() + bar.get_height()/2,
                    f"₹{val:,.0f}", va="center", color="#CCCCFF", fontsize=10)
        plt.tight_layout()
        fig.savefig(os.path.join(IMG_DIR, "platform_sales.png"), dpi=FIG_DPI)
        plt.close(fig)
        print("   Saved: platform_sales.png")

    # ── 3.4 Campaign-wise Sales ───────────────────────────────
    if "Campaign_Type" in df.columns:
        camp_sales = df.groupby("Campaign_Type")["Sales"].mean().sort_values(ascending=False)
        fig, ax = plt.subplots(figsize=(10, 5))
        bars = ax.bar(camp_sales.index, camp_sales.values,
                      color=sns.color_palette("plasma", len(camp_sales)))
        ax.set_title("Average Sales by Campaign Type", fontsize=16, fontweight="bold")
        ax.set_ylabel("Average Sales (₹ Thousands)")
        ax.tick_params(axis="x", rotation=20)
        for bar, val in zip(bars, camp_sales.values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                    f"₹{val:,.0f}", ha="center", color="#CCCCFF", fontsize=9)
        plt.tight_layout()
        fig.savefig(os.path.join(IMG_DIR, "campaign_sales.png"), dpi=FIG_DPI)
        plt.close(fig)
        print("   Saved: campaign_sales.png")

    # ── 3.5 Region-wise Sales ─────────────────────────────────
    if "Region" in df.columns:
        region_sales = df.groupby("Region")["Sales"].mean().sort_values(ascending=False)
        fig, ax = plt.subplots(figsize=(8, 5))
        wedges, texts, autotexts = ax.pie(
            region_sales.values, labels=region_sales.index, autopct="%1.1f%%",
            colors=sns.color_palette("coolwarm", len(region_sales)),
            startangle=140, pctdistance=0.80
        )
        for t in texts + autotexts:
            t.set_color("white")
        ax.set_title("Region-wise Sales Share", fontsize=16, fontweight="bold")
        plt.tight_layout()
        fig.savefig(os.path.join(IMG_DIR, "region_sales.png"), dpi=FIG_DPI)
        plt.close(fig)
        print("   Saved: region_sales.png")

    # ── 3.6 Monthly Sales Trend ───────────────────────────────
    if "Month" in df.columns:
        month_order = ["January","February","March","April","May","June",
                       "July","August","September","October","November","December"]
        month_sales = (df.groupby("Month")["Sales"].mean()
                       .reindex([m for m in month_order if m in df["Month"].unique()]))
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(month_sales.index, month_sales.values, marker="o",
                color="#5B5FFF", linewidth=2.5, markersize=7)
        ax.fill_between(range(len(month_sales)), month_sales.values,
                        alpha=0.25, color="#8A2BE2")
        ax.set_xticks(range(len(month_sales)))
        ax.set_xticklabels(month_sales.index, rotation=30, ha="right")
        ax.set_title("Monthly Average Sales Trend", fontsize=16, fontweight="bold")
        ax.set_ylabel("Average Sales (₹ Thousands)")
        plt.tight_layout()
        fig.savefig(os.path.join(IMG_DIR, "monthly_trend.png"), dpi=FIG_DPI)
        plt.close(fig)
        print("   Saved: monthly_trend.png")

    # ── 3.7 Correlation Heatmap ───────────────────────────────
    corr = df[num_cols].corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
                linewidths=0.5, ax=ax, annot_kws={"size": 9})
    ax.set_title("Feature Correlation Heatmap", fontsize=16, fontweight="bold")
    plt.tight_layout()
    fig.savefig(os.path.join(IMG_DIR, "correlation_heatmap.png"), dpi=FIG_DPI)
    plt.close(fig)
    print("   Saved: correlation_heatmap.png")

    # ── 3.8 Scatter Plots (Spend vs Sales) ───────────────────
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()
    for i, col in enumerate(spend_cols[:5]):
        axes[i].scatter(df[col], df["Sales"], alpha=0.40,
                        color=colors[i % len(colors)], s=20)
        m, b = np.polyfit(df[col], df["Sales"], 1)
        xs = np.linspace(df[col].min(), df[col].max(), 100)
        axes[i].plot(xs, m*xs+b, color="white", linewidth=1.5)
        axes[i].set_xlabel(col.replace("_", " "))
        axes[i].set_ylabel("Sales")
        axes[i].set_title(f"{col.replace('_',' ')} vs Sales")
    axes[5].set_visible(False)
    fig.suptitle("Advertising Spend vs Sales", fontsize=16, fontweight="bold")
    plt.tight_layout()
    fig.savefig(os.path.join(IMG_DIR, "scatter_plots.png"), dpi=FIG_DPI)
    plt.close(fig)
    print("   Saved: scatter_plots.png")

    # ── 3.9 Boxplot by Platform ───────────────────────────────
    if "Platform" in df.columns:
        fig, ax = plt.subplots(figsize=(12, 6))
        platforms = df["Platform"].unique()
        data_by_platform = [df[df["Platform"] == p]["Sales"].values for p in platforms]
        bp = ax.boxplot(data_by_platform, labels=platforms, patch_artist=True,
                        notch=True, vert=True)
        box_colors = sns.color_palette("coolwarm", len(platforms))
        for patch, color in zip(bp["boxes"], box_colors):
            patch.set_facecolor(color)
        ax.set_title("Sales Distribution by Platform", fontsize=16, fontweight="bold")
        ax.set_ylabel("Sales (₹ Thousands)")
        plt.tight_layout()
        fig.savefig(os.path.join(IMG_DIR, "boxplot_platform.png"), dpi=FIG_DPI)
        plt.close(fig)
        print("   Saved: boxplot_platform.png")

    print(f"\n   All EDA charts saved to {IMG_DIR}")

# ============================================================
# 4. Feature Engineering
# ============================================================
def engineer_features(df: pd.DataFrame):
    print(f"\n{'='*60}")
    print("4. FEATURE ENGINEERING")
    print(f"{'='*60}")

    spend_cols = ["TV_Spend", "Radio_Spend", "Newspaper_Spend",
                  "Social_Media_Spend", "Digital_Spend"]
    existing_spend = [c for c in spend_cols if c in df.columns]

    # Total Advertising Spend
    df["Total_Spend"] = df[existing_spend].sum(axis=1)

    # ROI Estimation  (Sales / Total_Spend)
    df["ROI"] = df["Sales"] / df["Total_Spend"].replace(0, np.nan)
    df["ROI"].fillna(df["ROI"].median(), inplace=True)

    # Cost Per Sale (inverse of ROI)
    df["Cost_Per_Sale"] = df["Total_Spend"] / df["Sales"].replace(0, np.nan)
    df["Cost_Per_Sale"].fillna(df["Cost_Per_Sale"].median(), inplace=True)

    # Seasonal Indicator (Q4 = 1, else 0 — high-sales quarter)
    q4_months = {"October", "November", "December"}
    df["Seasonal_Indicator"] = df["Month"].isin(q4_months).astype(int) if "Month" in df.columns else 0

    # Marketing Efficiency Score (weighted spend contribution)
    weights = {"TV_Spend": 0.30, "Radio_Spend": 0.15, "Newspaper_Spend": 0.10,
               "Social_Media_Spend": 0.20, "Digital_Spend": 0.25}
    df["Marketing_Efficiency"] = sum(
        df[c] * w for c, w in weights.items() if c in df.columns
    )

    print(f"   Engineered features: Total_Spend, ROI, Cost_Per_Sale, "
          f"Seasonal_Indicator, Marketing_Efficiency")
    print(f"   New shape: {df.shape}")
    return df

# ============================================================
# 5. Encoding & Scaling
# ============================================================
def encode_and_scale(df: pd.DataFrame):
    print(f"\n{'='*60}")
    print("5. ENCODING & SCALING")
    print(f"{'='*60}")

    cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
    label_encoders = {}

    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le
        print(f"   Label-encoded '{col}'  ({len(le.classes_)} classes)")

    joblib.dump(label_encoders, ENCODER_PATH)
    print(f"\n   Saved label_encoders.pkl")
    return df, label_encoders

# ============================================================
# 6. Model Training & Evaluation
# ============================================================
def train_models(df: pd.DataFrame, label_encoders: dict):
    print(f"\n{'='*60}")
    print("6. MODEL TRAINING")
    print(f"{'='*60}")

    target = "Sales"
    feature_cols = [c for c in df.columns if c != target]
    X = df[feature_cols]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42)

    # Feature Scaling
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)
    joblib.dump(scaler, SCALER_PATH)
    print("   Saved scaler.pkl")

    # ── Define models ─────────────────────────────────────────
    models = {
        "Linear Regression":        LinearRegression(),
        "Decision Tree":            DecisionTreeRegressor(random_state=42, max_depth=10),
        "Random Forest":            RandomForestRegressor(n_estimators=150, random_state=42, n_jobs=-1),
        "Gradient Boosting":        GradientBoostingRegressor(n_estimators=150, random_state=42),
    }

    # Try adding XGBoost if available
    try:
        from xgboost import XGBRegressor
        models["XGBoost"] = XGBRegressor(n_estimators=150, random_state=42,
                                          verbosity=0, eval_metric="rmse")
        print("   XGBoost detected – included in comparison.")
    except ImportError:
        print("   XGBoost not available – skipping (install xgboost to enable).")

    results = {}
    trained  = {}

    for name, model in models.items():
        print(f"\n   Training: {name} ...")
        if name == "Linear Regression":
            model.fit(X_train_s, y_train)
            y_pred = model.predict(X_test_s)
        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

        mae  = mean_absolute_error(y_test, y_pred)
        mse  = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2   = r2_score(y_test, y_pred)

        results[name] = {"MAE": round(mae,2), "MSE": round(mse,2),
                         "RMSE": round(rmse,2), "R2": round(r2,4)}
        trained[name] = model

        print(f"      MAE={mae:.2f}  RMSE={rmse:.2f}  R²={r2:.4f}")

    # ── Select best model by R² ───────────────────────────────
    best_name = max(results, key=lambda k: results[k]["R2"])
    best_model = trained[best_name]
    print(f"\n   ★ Best Model: {best_name}  (R²={results[best_name]['R2']})")

    joblib.dump(best_model, MODEL_PATH)
    print(f"   Saved model.pkl")

    # ── Feature Importance (if supported) ────────────────────
    if hasattr(best_model, "feature_importances_"):
        importances = best_model.feature_importances_
        fi_df = pd.DataFrame({"Feature": feature_cols, "Importance": importances})
        fi_df = fi_df.sort_values("Importance", ascending=True)

        fig, ax = plt.subplots(figsize=(10, 7))
        bars = ax.barh(fi_df["Feature"], fi_df["Importance"],
                       color=sns.color_palette("coolwarm", len(fi_df)))
        ax.set_title(f"Feature Importance — {best_name}", fontsize=16, fontweight="bold")
        ax.set_xlabel("Importance Score")
        for bar, val in zip(bars, fi_df["Importance"]):
            ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2,
                    f"{val:.4f}", va="center", color="#CCCCFF", fontsize=9)
        plt.tight_layout()
        fig.savefig(os.path.join(IMG_DIR, "feature_importance.png"), dpi=FIG_DPI)
        plt.close(fig)
        print("   Saved: feature_importance.png")

    # ── Model Comparison Chart ────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(15, 6))
    model_names = list(results.keys())
    metrics     = ["MAE", "RMSE", "R2"]
    metric_labels = ["MAE (lower=better)", "RMSE (lower=better)", "R² Score (higher=better)"]
    bar_colors  = ["#5B5FFF", "#FF6B9D", "#00D4FF"]

    for ax, metric, label, color in zip(axes, metrics, metric_labels, bar_colors):
        values = [results[m][metric] for m in model_names]
        bars   = ax.bar(model_names, values, color=color, alpha=0.80)
        ax.set_title(label, fontsize=12, fontweight="bold")
        ax.tick_params(axis="x", rotation=20)
        for bar, v in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005*max(values),
                    f"{v:.2f}", ha="center", color="white", fontsize=9)

    fig.suptitle("Model Performance Comparison", fontsize=16, fontweight="bold")
    plt.tight_layout()
    fig.savefig(os.path.join(IMG_DIR, "model_comparison.png"), dpi=FIG_DPI)
    plt.close(fig)
    print("   Saved: model_comparison.png")

    return results, best_name, feature_cols, trained[best_name], X_train, scaler

# ============================================================
# 7. Generate Business Insights
# ============================================================
def generate_insights(df_clean: pd.DataFrame, results: dict,
                       best_name: str, label_encoders: dict):
    print(f"\n{'='*60}")
    print("7. GENERATING BUSINESS INSIGHTS")
    print(f"{'='*60}")

    report_lines = []
    report_lines.append("=" * 60)
    report_lines.append("  SALES PREDICTION — BUSINESS INSIGHTS REPORT")
    report_lines.append("=" * 60)
    report_lines.append("")

    # Need to decode back from encoded values for readability
    df_raw = pd.read_csv(DATASET_PATH)
    df_raw.dropna(inplace=True)
    df_raw.drop_duplicates(inplace=True)
    for col in df_raw.select_dtypes(include=["object"]).columns:
        df_raw[col] = df_raw[col].str.strip().str.title()

    insights = {}

    # Best platform
    if "Platform" in df_raw.columns:
        best_platform = df_raw.groupby("Platform")["Sales"].mean().idxmax()
        insights["best_platform"] = best_platform
        report_lines.append(f"Best Performing Platform   : {best_platform}")

    # Most effective ad channel (highest correlation with Sales)
    spend_cols = ["TV_Spend","Radio_Spend","Newspaper_Spend","Social_Media_Spend","Digital_Spend"]
    existing_spend = [c for c in spend_cols if c in df_raw.columns]
    if existing_spend:
        corr_vals = df_raw[existing_spend + ["Sales"]].corr()["Sales"].drop("Sales")
        best_channel = corr_vals.idxmax()
        insights["best_channel"] = best_channel.replace("_", " ")
        report_lines.append(f"Most Effective Ad Channel  : {best_channel.replace('_',' ')}")

    # Highest sales region
    if "Region" in df_raw.columns:
        best_region = df_raw.groupby("Region")["Sales"].mean().idxmax()
        insights["best_region"] = best_region
        report_lines.append(f"Highest Sales Region       : {best_region}")

    # Best month
    if "Month" in df_raw.columns:
        best_month = df_raw.groupby("Month")["Sales"].mean().idxmax()
        insights["best_month"] = best_month
        report_lines.append(f"Best Marketing Month       : {best_month}")

    # Model accuracy
    best_r2 = results[best_name]["R2"]
    insights["model_accuracy"] = round(best_r2 * 100, 2)
    report_lines.append(f"Best Model                 : {best_name}")
    report_lines.append(f"Model Accuracy (R²)        : {best_r2:.4f}  ({best_r2*100:.2f}%)")
    report_lines.append("")
    report_lines.append("── MODEL COMPARISON ──────────────────────────────────")
    report_lines.append(f"{'Model':<25} {'MAE':>10} {'RMSE':>10} {'R²':>10}")
    report_lines.append("-" * 57)
    for m, v in results.items():
        marker = " ★" if m == best_name else ""
        report_lines.append(f"{m:<25} {v['MAE']:>10.2f} {v['RMSE']:>10.2f} {v['R2']:>10.4f}{marker}")
    report_lines.append("")
    report_lines.append("── DATASET STATISTICS ────────────────────────────────")
    report_lines.append(f"Total Records              : {len(df_raw)}")
    report_lines.append(f"Avg Sales                  : ₹{df_raw['Sales'].mean():,.2f} K")
    report_lines.append(f"Max Sales                  : ₹{df_raw['Sales'].max():,.2f} K")
    report_lines.append(f"Min Sales                  : ₹{df_raw['Sales'].min():,.2f} K")

    # Dashboard statistics
    insights.update({
        "total_records":  len(df_raw),
        "avg_sales":      round(df_raw["Sales"].mean(), 2),
        "max_sales":      round(df_raw["Sales"].max(), 2),
        "min_sales":      round(df_raw["Sales"].min(), 2),
        "best_model":     best_name,
        "best_r2":        round(best_r2, 4),
        "model_results":  results,
        "num_features":   len([c for c in df_raw.columns if c != "Sales"]),
        "predictions_made": 0,
    })

    # Save report
    report_text = "\n".join(report_lines)
    with open(os.path.join(REPORTS_DIR, "insights_report.txt"), "w", encoding="utf-8") as f:
        f.write(report_text)
    print(report_text)

    # Save metadata for Flask app to read
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(insights, f, indent=2)
    print(f"\n   Saved model_meta.json")

    return insights

# ============================================================
# 8. Main
# ============================================================
if __name__ == "__main__":
    print("\n" + "★"*60)
    print("   SALES PREDICTION — ML PIPELINE")
    print("★"*60)

    df_raw   = load_dataset()
    df_clean = clean_data(df_raw.copy())
    run_eda(df_clean.copy())
    df_feat  = engineer_features(df_clean.copy())
    df_enc, label_encoders = encode_and_scale(df_feat.copy())
    results, best_name, feature_cols, best_model, X_train, scaler = train_models(df_enc, label_encoders)
    insights = generate_insights(df_clean, results, best_name, label_encoders)

    print("\n" + "★"*60)
    print("   PIPELINE COMPLETE ✔")
    print(f"   Best Model : {best_name}")
    print(f"   R² Score   : {results[best_name]['R2']}")
    print("★"*60 + "\n")
