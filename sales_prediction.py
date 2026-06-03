# =============================================================================
# CodeAlpha Internship — Task 4: Sales Prediction using Python
# Author  : [Your Name]
# Dataset : Advertising.csv (TV, Radio, Newspaper spend vs Sales)
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# 1. LOAD & EXPLORE DATA
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("   CODEALPHA — SALES PREDICTION PROJECT")
print("=" * 60)

df = pd.read_csv("Advertising.csv", index_col=0)

print("\n📊 First 5 rows:")
print(df.head())
print(f"\nDataset shape : {df.shape[0]} rows × {df.shape[1]} columns")
print("\nDescriptive statistics:")
print(df.describe().round(2))
print(f"\nMissing values : {df.isnull().sum().sum()}")

# ─────────────────────────────────────────────────────────────────────────────
# 2. DATA CLEANING & TRANSFORMATION
# ─────────────────────────────────────────────────────────────────────────────
# No missing values found; confirm column types
df = df.astype(float)

# ─────────────────────────────────────────────────────────────────────────────
# 3. FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────────────────────
df["Total_Spend"] = df["TV"] + df["Radio"] + df["Newspaper"]
df["TV_Radio"]    = df["TV"] * df["Radio"]          # interaction term
df["TV_pct"]      = df["TV"]    / df["Total_Spend"] # TV share of budget
df["Radio_pct"]   = df["Radio"] / df["Total_Spend"] # Radio share of budget

FEATURES = ["TV", "Radio", "Newspaper", "Total_Spend", "TV_Radio", "TV_pct", "Radio_pct"]
TARGET   = "Sales"

X = df[FEATURES]
y = df[TARGET]

# Train / Test split  (80 / 20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scale for linear models
scaler     = StandardScaler()
X_train_s  = scaler.fit_transform(X_train)
X_test_s   = scaler.transform(X_test)

# ─────────────────────────────────────────────────────────────────────────────
# 4. TRAIN MODELS
# ─────────────────────────────────────────────────────────────────────────────
models = {
    "Linear Regression" : LinearRegression(),
    "Ridge Regression"  : Ridge(alpha=1.0),
    "Random Forest"     : RandomForestRegressor(n_estimators=100, random_state=42),
    "Gradient Boosting" : GradientBoostingRegressor(n_estimators=100, random_state=42),
}

results = {}
print("\n📈 Model Performance on Test Set:")
print(f"{'Model':<22}  {'R²':>6}  {'MAE':>6}  {'RMSE':>6}  {'CV R²':>8}")
print("-" * 58)

for name, model in models.items():
    linear = "Regression" in name
    Xtr = X_train_s if linear else X_train
    Xte = X_test_s  if linear else X_test
    Xfull = scaler.transform(X) if linear else X

    model.fit(Xtr, y_train)
    preds = model.predict(Xte)
    cv    = cross_val_score(model, Xfull, y, cv=5, scoring="r2")

    r2   = r2_score(y_test, preds)
    mae  = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))

    results[name] = dict(r2=r2, mae=mae, rmse=rmse, cv_r2=cv.mean(), preds=preds)
    print(f"{name:<22}  {r2:>6.3f}  {mae:>6.3f}  {rmse:>6.3f}  {cv.mean():>8.3f}")

best_name = max(results, key=lambda k: results[k]["r2"])
print(f"\n✅ Best model : {best_name}  (R² = {results[best_name]['r2']:.3f})")

# ─────────────────────────────────────────────────────────────────────────────
# 5. FEATURE IMPORTANCE
# ─────────────────────────────────────────────────────────────────────────────
rf_importances = pd.Series(
    models["Random Forest"].feature_importances_, index=FEATURES
).sort_values(ascending=False)

print("\n🔑 Feature Importances (Random Forest):")
print(rf_importances.round(3).to_string())

# ─────────────────────────────────────────────────────────────────────────────
# 6. BUSINESS INSIGHTS
# ─────────────────────────────────────────────────────────────────────────────
coefs = {
    ch: LinearRegression().fit(df[[ch]], y).coef_[0]
    for ch in ["TV", "Radio", "Newspaper"]
}
print("\n💡 Business Insights (simple linear slope per channel):")
for ch, coef in coefs.items():
    print(f"   • Every $1K more in {ch:<10} → +{coef:.3f}K in Sales")
print("   • Radio delivers the highest ROI relative to spend.")
print("   • Newspaper has minimal impact — consider reallocating budget.")
print("   • TV × Radio combined spending creates a synergistic sales lift.")

# ─────────────────────────────────────────────────────────────────────────────
# 7. VISUALISATIONS  →  saved as sales_prediction_analysis.png
# ─────────────────────────────────────────────────────────────────────────────
sns.set_theme(style="whitegrid", palette="muted")
fig = plt.figure(figsize=(18, 14))
fig.suptitle("CodeAlpha — Sales Prediction Analysis", fontsize=16,
             fontweight="bold", y=0.98)
gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

COLORS = {"TV": "#4E79A7", "Radio": "#F28E2B", "Newspaper": "#E15759"}

# Row 0 — Scatter: each channel vs Sales
for i, col in enumerate(["TV", "Radio", "Newspaper"]):
    ax = fig.add_subplot(gs[0, i])
    ax.scatter(df[col], df["Sales"], alpha=0.55, color=COLORS[col],
               edgecolors="white", s=55)
    m, b = np.polyfit(df[col], df["Sales"], 1)
    xl = np.linspace(df[col].min(), df[col].max(), 100)
    ax.plot(xl, m * xl + b, color="black", linewidth=1.5, linestyle="--")
    ax.set_xlabel(f"{col} Spend ($K)", fontsize=10)
    ax.set_ylabel("Sales ($K)" if i == 0 else "", fontsize=10)
    ax.set_title(f"{col} vs Sales", fontsize=11, fontweight="bold")

# Row 1 left — Correlation heatmap
ax2 = fig.add_subplot(gs[1, 0])
corr = df[["TV", "Radio", "Newspaper", "Sales"]].corr()
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax2,
            linewidths=0.5, square=True, cbar_kws={"shrink": 0.8})
ax2.set_title("Correlation Heatmap", fontsize=11, fontweight="bold")

# Row 1 mid — Model R² bar chart
ax3 = fig.add_subplot(gs[1, 1])
names  = list(results.keys())
r2vals = [results[n]["r2"] for n in names]
bars   = ax3.barh(names, r2vals,
                  color=["#4E79A7", "#F28E2B", "#59A14F", "#E15759"])
for bar, val in zip(bars, r2vals):
    ax3.text(val - 0.002, bar.get_y() + bar.get_height() / 2,
             f"{val:.3f}", va="center", ha="right",
             color="white", fontweight="bold", fontsize=9)
ax3.set_xlim(0.8, 1.01)
ax3.set_xlabel("R² Score", fontsize=10)
ax3.set_title("Model R² Comparison", fontsize=11, fontweight="bold")

# Row 1 right — Actual vs Predicted
ax4 = fig.add_subplot(gs[1, 2])
preds_best = results[best_name]["preds"]
ax4.scatter(y_test, preds_best, alpha=0.7, color="#59A14F",
            edgecolors="white", s=60)
lims = [min(y_test.min(), preds_best.min()) - 1,
        max(y_test.max(), preds_best.max()) + 1]
ax4.plot(lims, lims, "k--", linewidth=1.5)
ax4.set_xlabel("Actual Sales ($K)", fontsize=10)
ax4.set_ylabel("Predicted Sales ($K)", fontsize=10)
ax4.set_title(f"Actual vs Predicted\n({best_name})", fontsize=11, fontweight="bold")

# Row 2 left — Feature importance
ax5 = fig.add_subplot(gs[2, 0])
rf_importances.plot(kind="bar", ax=ax5, color="#4E79A7", edgecolor="white")
ax5.set_title("Feature Importance (RF)", fontsize=11, fontweight="bold")
ax5.set_ylabel("Importance", fontsize=10)
ax5.tick_params(axis="x", rotation=35)

# Row 2 mid — Spend distribution
ax6 = fig.add_subplot(gs[2, 1])
for col, c in COLORS.items():
    ax6.hist(df[col], bins=20, alpha=0.6, label=col, color=c)
ax6.set_xlabel("Spend ($K)", fontsize=10)
ax6.set_ylabel("Frequency", fontsize=10)
ax6.set_title("Advertising Spend Distribution", fontsize=11, fontweight="bold")
ax6.legend()

# Row 2 right — Residuals
ax7 = fig.add_subplot(gs[2, 2])
residuals = y_test.values - preds_best
ax7.scatter(preds_best, residuals, alpha=0.7, color="#E15759",
            edgecolors="white", s=60)
ax7.axhline(0, color="black", linewidth=1.5, linestyle="--")
ax7.set_xlabel("Predicted Sales ($K)", fontsize=10)
ax7.set_ylabel("Residuals", fontsize=10)
ax7.set_title(f"Residual Plot ({best_name})", fontsize=11, fontweight="bold")

plt.savefig("sales_prediction_analysis.png", dpi=150, bbox_inches="tight")
print("\n✅ Analysis chart saved → sales_prediction_analysis.png")
plt.close()

# ─────────────────────────────────────────────────────────────────────────────
# 8. SAMPLE PREDICTIONS
# ─────────────────────────────────────────────────────────────────────────────
print("\n🔮 Sample Predictions (Gradient Boosting):")
print(f"{'TV':>6}  {'Radio':>6}  {'Newspaper':>9}  {'Predicted Sales':>15}")
print("-" * 46)
gb = models["Gradient Boosting"]
for tv, r, n in [(200, 30, 50), (50, 50, 20), (100, 20, 10),
                 (300, 60, 80), (10, 5, 5)]:
    total = tv + r + n
    inp   = pd.DataFrame(
        [[tv, r, n, total, tv * r, tv / total, r / total]],
        columns=FEATURES
    )
    pred = gb.predict(inp)[0]
    print(f"{tv:>6}  {r:>6}  {n:>9}  {pred:>14.2f}K")

print("\n🎯 Done!")
