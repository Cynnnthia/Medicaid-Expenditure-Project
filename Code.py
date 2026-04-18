import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.backends.backend_pdf import PdfPages

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split

from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.seasonal import seasonal_decompose

import os
import json

try:
    from prophet import Prophet
    HAS_PROPHET = True
except ImportError:
    HAS_PROPHET = False
    print("Prophet not installed — skipping Prophet forecast.")

# ─────────────────────────────────────────────
# 0. OUTPUT DIRECTORY
# ─────────────────────────────────────────────
OUT = "medicaid_output"
os.makedirs(OUT, exist_ok=True)

# ─────────────────────────────────────────────
# 1. DATA INGESTION
# ─────────────────────────────────────────────
YEARS = np.array([
    2013, 2014, 2015, 2016, 2017, 2018, 2019,
    2020, 2021, 2022, 2023, 2024, 2025
])

# California expenditure
VALUES = np.array([
    6.605676e+10, 6.824844e+10, 9.061436e+10, 8.660858e+10,
    8.866459e+10, 8.889577e+10, 9.410041e+10, 1.038866e+11,
    1.156904e+11, 1.250178e+11, 1.301333e+11, 1.570981e+11,
    1.734027e+11
])

# National expenditure
NATIONAL_VALUES = np.array([
    453082711064,
    488240409971,
    548190828914,
    571229555606,
    596434360108,
    611976826895,
    614908690223,
    678892781237,
    740285561048,
    823865414079,
    893144356932,
    948846651184,
    1031704529256,
])

annual_df = pd.DataFrame({"Year": YEARS, "Expenditure": VALUES})
annual_df["Date"] = pd.to_datetime(annual_df["Year"].astype(str) + "-01-01")

national_df = pd.DataFrame({"Year": YEARS, "Expenditure": NATIONAL_VALUES})
national_df["Date"] = pd.to_datetime(national_df["Year"].astype(str) + "-01-01")

# California share of national
annual_df["CA_Share_pct"] = (annual_df["Expenditure"] / national_df["Expenditure"]) * 100

print("=== California Annual Data (raw) ===")
print(annual_df[["Year","Expenditure"]].to_string(index=False))
print("\n=== National Annual Data (raw) ===")
print(national_df[["Year","Expenditure"]].to_string(index=False))

# ─────────────────────────────────────────────
# 2. GENERATE MONTHLY SERIES VIA LINEAR INTERP
# ─────────────────────────────────────────────
monthly_dates = pd.date_range("2013-01-01", "2025-12-01", freq="MS")

lin_model = LinearRegression()
X_year = YEARS.reshape(-1, 1)
lin_model.fit(X_year, VALUES)

monthly_frac = np.array([d.year + (d.month - 1) / 12 for d in monthly_dates]).reshape(-1, 1)
monthly_values = lin_model.predict(monthly_frac)

monthly_df = pd.DataFrame({
    "Date": monthly_dates,
    "Expenditure_raw": monthly_values,
    "Expenditure": monthly_values
})
monthly_df.set_index("Date", inplace=True)

monthly_csv = monthly_df.copy()
monthly_csv["Expenditure_fmt"] = monthly_csv["Expenditure_raw"].map("${:,.0f}".format)
monthly_csv.to_csv(f"{OUT}/monthly_interpolated_2013_2025.csv")
print(f"\nSaved: {OUT}/monthly_interpolated_2013_2025.csv  ({len(monthly_csv)} rows)")

# ─────────────────────────────────────────────
# 3. EXPLORATORY DATA ANALYSIS
# ─────────────────────────────────────────────
annual_df["YoY_Growth_pct"] = annual_df["Expenditure"].pct_change() * 100
national_df["YoY_Growth_pct"] = national_df["Expenditure"].pct_change() * 100

print("\n=== California Year-over-Year Growth ===")
print(annual_df[["Year", "Expenditure", "YoY_Growth_pct"]].to_string(index=False))
print("\n=== National Year-over-Year Growth ===")
print(national_df[["Year", "Expenditure", "YoY_Growth_pct"]].to_string(index=False))
print("\n=== California Share of National (%) ===")
print(annual_df[["Year", "CA_Share_pct"]].to_string(index=False))

decomp = seasonal_decompose(monthly_df["Expenditure"], model="additive", period=12)

fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
fig.suptitle("Time-Series Decomposition (Monthly Interpolated)", fontsize=14)
for ax, data, label in zip(axes,
    [monthly_df["Expenditure"], decomp.trend, decomp.seasonal, decomp.resid],
    ["Observed", "Trend", "Seasonal", "Residual"]):
    ax.plot(data.index, data.values, linewidth=1.5)
    ax.set_ylabel(label, fontsize=10)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x/1e9:.0f}B"))
    ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUT}/eda_decomposition.png", dpi=150)
plt.close()
print(f"Saved: {OUT}/eda_decomposition.png")

# ─────────────────────────────────────────────
# 4. FORECASTING MODELS — CALIFORNIA
# ─────────────────────────────────────────────
SPLIT = 10
X_train, X_test = YEARS[:SPLIT].reshape(-1, 1), YEARS[SPLIT:].reshape(-1, 1)
y_train, y_test = VALUES[:SPLIT], VALUES[SPLIT:]

FUTURE_YEARS = np.arange(2026, 2036)

results = {}

def eval_metrics(true, pred, name):
    mae  = mean_absolute_error(true, pred)
    rmse = np.sqrt(mean_squared_error(true, pred))
    mape = np.mean(np.abs((true - pred) / true)) * 100
    print(f"  {name:30s} | MAE=${mae/1e9:.2f}B | RMSE=${rmse/1e9:.2f}B | MAPE={mape:.1f}%")
    return {"MAE": mae, "RMSE": rmse, "MAPE": mape}

# ── 4a. MODEL VALIDATION ──
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.stattools import durbin_watson
from sklearn.model_selection import TimeSeriesSplit

print("\n=== Model Validation (California) ===")

adf_result = adfuller(annual_df["Expenditure"])
print(f"\n[1] ADF Stationarity Test:")
print(f"    ADF Statistic : {adf_result[0]:.4f}")
print(f"    p-value       : {adf_result[1]:.4f}")
print(f"    Conclusion    : {'Non-stationary (differencing needed)' if adf_result[1] > 0.05 else 'Stationary'}")

lr_all = LinearRegression().fit(YEARS.reshape(-1, 1), VALUES)
residuals_all = VALUES - lr_all.predict(YEARS.reshape(-1, 1))
dw_stat = durbin_watson(residuals_all)
print(f"\n[2] Durbin-Watson Residual Test (Linear Regression):")
print(f"    DW Statistic  : {dw_stat:.4f}")
print(f"    Conclusion    : ", end="")
if dw_stat < 1.5:
    print("Positive autocorrelation in residuals")
elif dw_stat > 2.5:
    print("Negative autocorrelation in residuals")
else:
    print("No significant autocorrelation (residuals look random ✓)")

print(f"\n[3] Time Series Cross-Validation (Linear Regression):")
tscv = TimeSeriesSplit(n_splits=3)
cv_maes = []
for fold, (train_idx, test_idx) in enumerate(tscv.split(YEARS)):
    X_cv_train = YEARS[train_idx].reshape(-1, 1)
    X_cv_test  = YEARS[test_idx].reshape(-1, 1)
    y_cv_train = VALUES[train_idx]
    y_cv_test  = VALUES[test_idx]
    lr_cv = LinearRegression().fit(X_cv_train, y_cv_train)
    mae_cv = mean_absolute_error(y_cv_test, lr_cv.predict(X_cv_test))
    cv_maes.append(mae_cv)
    print(f"    Fold {fold+1}: MAE = ${mae_cv/1e9:.2f}B")
print(f"    Average MAE : ${np.mean(cv_maes)/1e9:.2f}B")

# ── 4b. LINEAR REGRESSION ──
lr = LinearRegression().fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)
metrics_lr = eval_metrics(y_test, y_pred_lr, "Linear Regression")
future_lr  = lr.predict(FUTURE_YEARS.reshape(-1, 1))
residuals  = VALUES - lr.predict(YEARS.reshape(-1, 1))
std_lr     = np.std(residuals)
results["Linear Regression"] = {
    "metrics": metrics_lr,
    "forecast": future_lr,
    "ci_low":  future_lr - 1.96 * std_lr,
    "ci_high": future_lr + 1.96 * std_lr,
}

# ── 4c. POLYNOMIAL REGRESSION (degree 2) ──
poly = PolynomialFeatures(degree=2)
Xp_train = poly.fit_transform(X_train)
Xp_test  = poly.transform(X_test)
pr = LinearRegression().fit(Xp_train, y_train)
y_pred_pr = pr.predict(Xp_test)
metrics_pr = eval_metrics(y_test, y_pred_pr, "Polynomial Regression (deg 2)")
future_pr  = pr.predict(poly.transform(FUTURE_YEARS.reshape(-1, 1)))
residuals_p = VALUES - pr.predict(poly.transform(YEARS.reshape(-1, 1)))
std_pr = np.std(residuals_p)
results["Polynomial Regression"] = {
    "metrics": metrics_pr,
    "forecast": future_pr,
    "ci_low":  future_pr - 1.96 * std_pr,
    "ci_high": future_pr + 1.96 * std_pr,
}

# ── 4d. ARIMA ──
try:
    arima = SARIMAX(annual_df["Expenditure"], order=(1, 1, 1), trend="t").fit(disp=False)
    test_fcast = arima.get_forecast(steps=3)
    y_pred_arima = np.asarray(test_fcast.predicted_mean)
    metrics_arima = eval_metrics(y_test, y_pred_arima, "ARIMA(1,1,1)")
    arima_full = SARIMAX(VALUES, order=(1, 1, 1), trend="t").fit(disp=False)
    fcast = arima_full.get_forecast(steps=10)
    future_arima = np.asarray(fcast.predicted_mean)
    ci = fcast.conf_int()
    results["ARIMA(1,1,1)"] = {
        "metrics": metrics_arima,
        "forecast": future_arima,
        "ci_low":  ci.iloc[:, 0].values,
        "ci_high": ci.iloc[:, 1].values,
    }
except Exception as e:
    print(f"  ARIMA failed: {e}")

# ── 4e. HOLT-WINTERS ──
try:
    hw = ExponentialSmoothing(
        annual_df["Expenditure"], trend="add", seasonal=None, damped_trend=True
    ).fit(optimized=True)
    y_pred_hw = hw.predict(start=SPLIT, end=len(VALUES) - 1)
    metrics_hw = eval_metrics(y_test, y_pred_hw, "Holt-Winters (damped)")
    hw_full = ExponentialSmoothing(
        VALUES, trend="add", seasonal=None, damped_trend=True
    ).fit(optimized=True)
    future_hw = hw_full.forecast(10)
    rmse_hw = np.sqrt(mean_squared_error(VALUES, hw_full.fittedvalues))
    results["Holt-Winters"] = {
        "metrics": metrics_hw,
        "forecast": future_hw,
        "ci_low":  future_hw - 1.96 * rmse_hw,
        "ci_high": future_hw + 1.96 * rmse_hw,
    }
except Exception as e:
    print(f"  Holt-Winters failed: {e}")

# ── 4f. PROPHET ──
if HAS_PROPHET:
    try:
        prophet_df = annual_df[["Date", "Expenditure"]].rename(
            columns={"Date": "ds", "Expenditure": "y"}
        )
        m = Prophet(interval_width=0.95, yearly_seasonality=False)
        m.fit(prophet_df)
        future_prophet_df = m.make_future_dataframe(periods=10, freq="YS")
        forecast_prophet  = m.predict(future_prophet_df)
        future_rows = forecast_prophet[forecast_prophet["ds"].dt.year >= 2026]
        y_pred_prophet = forecast_prophet.loc[
            forecast_prophet["ds"].dt.year.isin(YEARS[SPLIT:]), "yhat"
        ].values
        metrics_prophet = eval_metrics(y_test, y_pred_prophet, "Prophet")
        results["Prophet"] = {
            "metrics": metrics_prophet,
            "forecast": future_rows["yhat"].values,
            "ci_low":  future_rows["yhat_lower"].values,
            "ci_high": future_rows["yhat_upper"].values,
        }
    except Exception as e:
        print(f"  Prophet failed: {e}")

# ─────────────────────────────────────────────
# 4g. FORECASTING MODELS — NATIONAL
# ─────────────────────────────────────────────
print("\n=== National Forecasting Models ===")

nat_y_train = NATIONAL_VALUES[:SPLIT]
nat_y_test  = NATIONAL_VALUES[SPLIT:]
national_results = {}

# Linear
lr_nat = LinearRegression().fit(X_train, nat_y_train)
y_pred_lr_nat = lr_nat.predict(X_test)
metrics_lr_nat = eval_metrics(nat_y_test, y_pred_lr_nat, "National Linear Regression")
future_lr_nat = lr_nat.predict(FUTURE_YEARS.reshape(-1, 1))
res_nat = NATIONAL_VALUES - lr_nat.predict(YEARS.reshape(-1, 1))
std_lr_nat = np.std(res_nat)
national_results["Linear Regression"] = {
    "metrics": metrics_lr_nat,
    "forecast": future_lr_nat,
    "ci_low":  future_lr_nat - 1.96 * std_lr_nat,
    "ci_high": future_lr_nat + 1.96 * std_lr_nat,
}

# Polynomial
Xp_train_nat = poly.fit_transform(X_train)
pr_nat = LinearRegression().fit(Xp_train_nat, nat_y_train)
y_pred_pr_nat = pr_nat.predict(poly.transform(X_test))
metrics_pr_nat = eval_metrics(nat_y_test, y_pred_pr_nat, "National Poly Regression (deg 2)")
future_pr_nat = pr_nat.predict(poly.transform(FUTURE_YEARS.reshape(-1, 1)))
res_nat_p = NATIONAL_VALUES - pr_nat.predict(poly.transform(YEARS.reshape(-1, 1)))
std_pr_nat = np.std(res_nat_p)
national_results["Polynomial Regression"] = {
    "metrics": metrics_pr_nat,
    "forecast": future_pr_nat,
    "ci_low":  future_pr_nat - 1.96 * std_pr_nat,
    "ci_high": future_pr_nat + 1.96 * std_pr_nat,
}

# ARIMA
try:
    arima_nat = SARIMAX(national_df["Expenditure"], order=(1, 1, 1), trend="t").fit(disp=False)
    test_fcast_nat = arima_nat.get_forecast(steps=3)
    y_pred_arima_nat = np.asarray(test_fcast_nat.predicted_mean)
    metrics_arima_nat = eval_metrics(nat_y_test, y_pred_arima_nat, "National ARIMA(1,1,1)")
    arima_full_nat = SARIMAX(NATIONAL_VALUES, order=(1, 1, 1), trend="t").fit(disp=False)
    fcast_nat = arima_full_nat.get_forecast(steps=10)
    future_arima_nat = np.asarray(fcast_nat.predicted_mean)
    ci_nat = fcast_nat.conf_int()
    national_results["ARIMA(1,1,1)"] = {
        "metrics": metrics_arima_nat,
        "forecast": future_arima_nat,
        "ci_low":  ci_nat.iloc[:, 0].values,
        "ci_high": ci_nat.iloc[:, 1].values,
    }
except Exception as e:
    print(f"  National ARIMA failed: {e}")

# Holt-Winters
try:
    hw_nat = ExponentialSmoothing(
        national_df["Expenditure"], trend="add", seasonal=None, damped_trend=True
    ).fit(optimized=True)
    y_pred_hw_nat = hw_nat.predict(start=SPLIT, end=len(NATIONAL_VALUES) - 1)
    metrics_hw_nat = eval_metrics(nat_y_test, y_pred_hw_nat, "National Holt-Winters (damped)")
    hw_full_nat = ExponentialSmoothing(
        NATIONAL_VALUES, trend="add", seasonal=None, damped_trend=True
    ).fit(optimized=True)
    future_hw_nat = hw_full_nat.forecast(10)
    rmse_hw_nat = np.sqrt(mean_squared_error(NATIONAL_VALUES, hw_full_nat.fittedvalues))
    national_results["Holt-Winters"] = {
        "metrics": metrics_hw_nat,
        "forecast": future_hw_nat,
        "ci_low":  future_hw_nat - 1.96 * rmse_hw_nat,
        "ci_high": future_hw_nat + 1.96 * rmse_hw_nat,
    }
except Exception as e:
    print(f"  National Holt-Winters failed: {e}")

# Prophet for national
if HAS_PROPHET:
    try:
        prophet_nat_df = national_df[["Date", "Expenditure"]].rename(
            columns={"Date": "ds", "Expenditure": "y"}
        )
        m_nat = Prophet(interval_width=0.95, yearly_seasonality=False)
        m_nat.fit(prophet_nat_df)
        future_prophet_nat_df = m_nat.make_future_dataframe(periods=10, freq="YS")
        forecast_prophet_nat  = m_nat.predict(future_prophet_nat_df)
        future_rows_nat = forecast_prophet_nat[forecast_prophet_nat["ds"].dt.year >= 2026]
        y_pred_prophet_nat = forecast_prophet_nat.loc[
            forecast_prophet_nat["ds"].dt.year.isin(YEARS[SPLIT:]), "yhat"
        ].values
        metrics_prophet_nat = eval_metrics(nat_y_test, y_pred_prophet_nat, "National Prophet")
        national_results["Prophet"] = {
            "metrics": metrics_prophet_nat,
            "forecast": future_rows_nat["yhat"].values,
            "ci_low":  future_rows_nat["yhat_lower"].values,
            "ci_high": future_rows_nat["yhat_upper"].values,
        }
    except Exception as e:
        print(f"  National Prophet failed: {e}")

# ─────────────────────────────────────────────
# 5. MODEL COMPARISON & BEST MODEL SELECTION
# ─────────────────────────────────────────────
# California
comparison_rows = []
for name, res in results.items():
    comparison_rows.append({
        "Model":     name,
        "MAE ($B)":  round(res["metrics"]["MAE"]  / 1e9, 2),
        "RMSE ($B)": round(res["metrics"]["RMSE"] / 1e9, 2),
        "MAPE (%)":  round(res["metrics"]["MAPE"], 1),
    })
comparison_df = pd.DataFrame(comparison_rows).sort_values("MAPE (%)")
print("\n=== California Model Comparison (sorted by MAPE) ===")
print(comparison_df.to_string(index=False))
comparison_df.to_csv(f"{OUT}/model_comparison_california.csv", index=False)

best_model_name = comparison_df.iloc[0]["Model"]
print(f"\nCalifornia Best model: {best_model_name}")

# National
nat_comparison_rows = []
for name, res in national_results.items():
    nat_comparison_rows.append({
        "Model":     name,
        "MAE ($B)":  round(res["metrics"]["MAE"]  / 1e9, 2),
        "RMSE ($B)": round(res["metrics"]["RMSE"] / 1e9, 2),
        "MAPE (%)":  round(res["metrics"]["MAPE"], 1),
    })
nat_comparison_df = pd.DataFrame(nat_comparison_rows).sort_values("MAPE (%)")
print("\n=== National Model Comparison (sorted by MAPE) ===")
print(nat_comparison_df.to_string(index=False))
nat_comparison_df.to_csv(f"{OUT}/model_comparison_national.csv", index=False)

nat_best_model_name = nat_comparison_df.iloc[0]["Model"]
print(f"\nNational Best model: {nat_best_model_name}")

# ─────────────────────────────────────────────
# 6. PROJECTION TABLES
# ─────────────────────────────────────────────
best    = results[best_model_name]
nat_best = national_results[nat_best_model_name]

# California projection
projection_df = pd.DataFrame({
    "Year":           FUTURE_YEARS,
    "Forecast ($)":   best["forecast"],
    "CI Lower ($)":   best["ci_low"],
    "CI Upper ($)":   best["ci_high"],
    "Forecast ($B)":  (best["forecast"] / 1e9).round(2),
    "CI Lower ($B)":  (best["ci_low"]   / 1e9).round(2),
    "CI Upper ($B)":  (best["ci_high"]  / 1e9).round(2),
})
projection_df.to_csv(f"{OUT}/projection_california_2026_2035.csv", index=False)
print(f"\n=== 10-Year Projection — California ({best_model_name}) ===")
print(projection_df[["Year", "Forecast ($B)", "CI Lower ($B)", "CI Upper ($B)"]].to_string(index=False))

# National projection
nat_projection_df = pd.DataFrame({
    "Year":           FUTURE_YEARS,
    "Forecast ($)":   nat_best["forecast"],
    "CI Lower ($)":   nat_best["ci_low"],
    "CI Upper ($)":   nat_best["ci_high"],
    "Forecast ($B)":  (nat_best["forecast"] / 1e9).round(2),
    "CI Lower ($B)":  (nat_best["ci_low"]   / 1e9).round(2),
    "CI Upper ($B)":  (nat_best["ci_high"]  / 1e9).round(2),
})
nat_projection_df.to_csv(f"{OUT}/projection_national_2026_2035.csv", index=False)
print(f"\n=== 10-Year Projection — National ({nat_best_model_name}) ===")
print(nat_projection_df[["Year", "Forecast ($B)", "CI Lower ($B)", "CI Upper ($B)"]].to_string(index=False))

# California share of National (historical + projected)
ca_share_hist = (VALUES / NATIONAL_VALUES) * 100
ca_share_proj = (best["forecast"] / nat_best["forecast"]) * 100
print(f"\n=== California Share of National Medicaid ===")
for yr, sh in zip(YEARS, ca_share_hist):
    print(f"  {yr}: {sh:.2f}%")
print("  --- Projected ---")
for yr, sh in zip(FUTURE_YEARS, ca_share_proj):
    print(f"  {yr}: {sh:.2f}%")

# ─────────────────────────────────────────────
# 6b. COMPLETE DATA TABLES
# ─────────────────────────────────────────────
# California full table
hist_table = pd.DataFrame({
    "Year":          annual_df["Year"],
    "Type":          "Historical",
    "CA_Expenditure ($B)": (annual_df["Expenditure"] / 1e9).round(2),
    "Nat_Expenditure ($B)": (national_df["Expenditure"] / 1e9).round(2),
    "CA_Share (%)":  annual_df["CA_Share_pct"].round(2),
    "CA CI Lower ($B)": np.nan,
    "CA CI Upper ($B)": np.nan,
    "CA YoY Growth (%)": annual_df["YoY_Growth_pct"].round(2),
    "Nat YoY Growth (%)": national_df["YoY_Growth_pct"].round(2),
})

proj_yoy = [np.nan] + list(
    np.diff(best["forecast"]) / best["forecast"][:-1] * 100
)
proj_yoy[0] = (best["forecast"][0] - VALUES[-1]) / VALUES[-1] * 100

nat_proj_yoy = [np.nan] + list(
    np.diff(nat_best["forecast"]) / nat_best["forecast"][:-1] * 100
)
nat_proj_yoy[0] = (nat_best["forecast"][0] - NATIONAL_VALUES[-1]) / NATIONAL_VALUES[-1] * 100

proj_table = pd.DataFrame({
    "Year":           FUTURE_YEARS,
    "Type":           "Forecast",
    "CA_Expenditure ($B)": (best["forecast"] / 1e9).round(2),
    "Nat_Expenditure ($B)": (nat_best["forecast"] / 1e9).round(2),
    "CA_Share (%)":   np.round(ca_share_proj, 2),
    "CA CI Lower ($B)":  (best["ci_low"]    / 1e9).round(2),
    "CA CI Upper ($B)":  (best["ci_high"]   / 1e9).round(2),
    "CA YoY Growth (%)": np.round(proj_yoy, 2),
    "Nat YoY Growth (%)": np.round(nat_proj_yoy, 2),
})

full_table = pd.concat([hist_table, proj_table], ignore_index=True)
full_table.to_csv(f"{OUT}/full_data_table_2013_2035.csv", index=False)

print("\n=== Full Data Table (2013–2035) ===")
print(full_table.to_string(index=False))
print(f"Saved: {OUT}/full_data_table_2013_2035.csv")

# ─────────────────────────────────────────────
# 7. VISUALIZATIONS
# ─────────────────────────────────────────────
COLOR = {
    "hist": "#2B6CB0",
    "nat":  "#C05621",
    "best": "#276749",
    "nat_best": "#9B2335",
    "ci": "#9AE6B4",
    "ci_nat": "#FEB2B2",
    "growth": "#C05621",
    "share": "#6B46C1",
}

def fmt_billions(ax, axis="y"):
    fmt = mticker.FuncFormatter(lambda x, _: f"${x/1e9:.0f}B")
    if axis == "y": ax.yaxis.set_major_formatter(fmt)
    else: ax.xaxis.set_major_formatter(fmt)

# ── Fig 1: California historical trend ──
fig, ax = plt.subplots(figsize=(14, 6))
ax.plot(annual_df["Year"], annual_df["Expenditure"] / 1e9, "o-",
        color=COLOR["hist"], linewidth=2, markersize=7, label="CA Historical")
for yr, val in zip(annual_df["Year"], annual_df["Expenditure"]):
    ax.annotate(f"${val/1e9:.1f}B", xy=(yr, val/1e9),
                xytext=(0, 10), textcoords="offset points",
                ha="center", fontsize=8, color=COLOR["hist"])
ax.set_title("California Medicaid Expenditure — Historical Trend (2013–2025)", fontsize=13)
ax.set_xlabel("Year"); ax.set_ylabel("Expenditure ($B)")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:.0f}B"))
ax.grid(alpha=0.3); ax.legend()
plt.tight_layout()
plt.savefig(f"{OUT}/fig1_historical_trend.png", dpi=150)
plt.close()

# ── Fig 1b: CA vs National historical comparison ──
fig, ax1 = plt.subplots(figsize=(14, 6))
ax2 = ax1.twinx()

ax1.plot(YEARS, VALUES / 1e9, "o-", color=COLOR["hist"], linewidth=2, markersize=7, label="California ($B, left)")
ax1.plot(YEARS, NATIONAL_VALUES / 1e9, "s-", color=COLOR["nat"], linewidth=2, markersize=7, label="National ($B, left)")
ax2.plot(YEARS, ca_share_hist, "^--", color=COLOR["share"], linewidth=1.5, markersize=6, label="CA Share of National (%, right)")

ax1.set_xlabel("Year")
ax1.set_ylabel("Expenditure ($B)", color="#2d3748")
ax2.set_ylabel("CA Share (%)", color=COLOR["share"])
ax2.tick_params(axis="y", labelcolor=COLOR["share"])
ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.1f}%"))
ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:.0f}B"))

for yr, ca, nat in zip(YEARS, VALUES, NATIONAL_VALUES):
    ax1.annotate(f"${ca/1e9:.0f}B", xy=(yr, ca/1e9), xytext=(0, 8),
                 textcoords="offset points", ha="center", fontsize=7, color=COLOR["hist"])
    ax1.annotate(f"${nat/1e9:.0f}B", xy=(yr, nat/1e9), xytext=(0, -14),
                 textcoords="offset points", ha="center", fontsize=7, color=COLOR["nat"])

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=9)
ax1.set_title("California vs National Medicaid Expenditure (2013–2025)", fontsize=13)
ax1.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUT}/fig1b_ca_vs_national_historical.png", dpi=150)
plt.close()
print(f"Saved: {OUT}/fig1b_ca_vs_national_historical.png")

# ── Fig 2: YoY growth rate ──
fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=False)
# CA growth
colors_ca = [COLOR["hist"] if g >= 0 else "#9B2335" for g in annual_df["YoY_Growth_pct"].fillna(0)]
axes[0].bar(annual_df["Year"], annual_df["YoY_Growth_pct"].fillna(0), color=colors_ca, edgecolor="white")
axes[0].axhline(0, color="black", linewidth=0.8)
axes[0].set_title("California YoY Growth Rate (%)", fontsize=12)
axes[0].set_xlabel("Year"); axes[0].set_ylabel("Growth (%)")
for yr, gv in zip(annual_df["Year"], annual_df["YoY_Growth_pct"].fillna(0)):
    axes[0].text(yr, gv + 0.3, f"{gv:.1f}%", ha="center", va="bottom", fontsize=7)
axes[0].grid(axis="y", alpha=0.3)
# National growth
colors_nat = [COLOR["nat"] if g >= 0 else "#9B2335" for g in national_df["YoY_Growth_pct"].fillna(0)]
axes[1].bar(national_df["Year"], national_df["YoY_Growth_pct"].fillna(0), color=colors_nat, edgecolor="white")
axes[1].axhline(0, color="black", linewidth=0.8)
axes[1].set_title("National YoY Growth Rate (%)", fontsize=12)
axes[1].set_xlabel("Year"); axes[1].set_ylabel("Growth (%)")
for yr, gv in zip(national_df["Year"], national_df["YoY_Growth_pct"].fillna(0)):
    axes[1].text(yr, gv + 0.3, f"{gv:.1f}%", ha="center", va="bottom", fontsize=7)
axes[1].grid(axis="y", alpha=0.3)
plt.suptitle("Year-over-Year Growth Comparison: California vs National", fontsize=13)
plt.tight_layout()
plt.savefig(f"{OUT}/fig2_yoy_growth.png", dpi=150)
plt.close()

# ── Fig 3: All model forecasts (California) ──
fig, ax = plt.subplots(figsize=(14, 7))
ax.plot(YEARS, VALUES / 1e9, "ko-", linewidth=2, markersize=7, label="CA Historical", zorder=5)
for yr, val in zip(YEARS, VALUES):
    ax.annotate(f"${val/1e9:.1f}B", xy=(yr, val/1e9), xytext=(0, 10),
                textcoords="offset points", ha="center", fontsize=7, color="black")
colors_models = ["#2B6CB0", "#276749", "#C05621", "#6B46C1", "#B7791F"]
for (name, res), col in zip(results.items(), colors_models):
    forecast_vals = res["forecast"] / 1e9
    ax.plot(FUTURE_YEARS, forecast_vals, "o--", color=col, linewidth=1.5, markersize=5,
            label=f"{name} {'★' if name == best_model_name else ''}")
    for yr, val in zip(FUTURE_YEARS, forecast_vals):
        ax.annotate(f"${val:.1f}B", xy=(yr, val), xytext=(0, 8),
                    textcoords="offset points", ha="center", fontsize=6, color=col)
ax.set_title("California Medicaid — All Model Projections (2026–2035)", fontsize=13)
ax.set_xlabel("Year"); ax.set_ylabel("Expenditure ($B)")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:.0f}B"))
ax.axvline(2025.5, color="gray", linestyle=":", linewidth=1)
ax.text(2025.6, ax.get_ylim()[0] * 1.02, "Forecast →", color="gray", fontsize=9)
ax.legend(fontsize=8); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUT}/fig3_all_models.png", dpi=150)
plt.close()

# ── Fig 4: Best model with CI (California) ──
fig, ax = plt.subplots(figsize=(14, 7))
ax.plot(YEARS, VALUES / 1e9, "ko-", linewidth=2, markersize=8, label="CA Historical", zorder=5)
ax.plot(FUTURE_YEARS, best["forecast"] / 1e9, "o-",
        color=COLOR["best"], linewidth=2.5, markersize=8,
        label=f"CA Forecast ({best_model_name})")
ax.fill_between(FUTURE_YEARS, best["ci_low"] / 1e9, best["ci_high"] / 1e9,
                color=COLOR["ci"], alpha=0.5, label="95% CI")
for yr, val in zip(YEARS, VALUES):
    ax.annotate(f"${val/1e9:.1f}B", xy=(yr, val/1e9), xytext=(0, 10),
                textcoords="offset points", ha="center", fontsize=7.5, color="#2B6CB0")
for yr, val in zip(FUTURE_YEARS, best["forecast"]):
    ax.annotate(f"${val/1e9:.1f}B", xy=(yr, val/1e9), xytext=(0, 10),
                textcoords="offset points", ha="center", fontsize=7.5, color=COLOR["best"])
ax.set_title(f"California Medicaid — 10-Year Projection [{best_model_name}]", fontsize=13)
ax.set_xlabel("Year"); ax.set_ylabel("Expenditure ($B)")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:.0f}B"))
ax.axvline(2025.5, color="gray", linestyle=":", linewidth=1)
ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUT}/fig4_best_forecast_ci.png", dpi=150)
plt.close()

# ── Fig 5: Rolling 12-month (monthly series) ──
monthly_df["Rolling_12M"] = monthly_df["Expenditure"].rolling(12).mean()
fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(monthly_df.index, monthly_df["Expenditure"] / 1e9,
        alpha=0.35, color=COLOR["hist"], linewidth=1, label="Monthly (interpolated)")
ax.plot(monthly_df.index, monthly_df["Rolling_12M"] / 1e9,
        color=COLOR["hist"], linewidth=2, label="12-month rolling avg")
ax.set_title("Rolling 12-Month California Medicaid Expenditure Trend", fontsize=13)
ax.set_xlabel("Date"); ax.set_ylabel("Expenditure ($B)")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:.0f}B"))
ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUT}/fig5_rolling_12m.png", dpi=150)
plt.close()

# ── Fig 6: CA vs National — Historical + Forecast (side by side) ──
fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# Left: California
axes[0].plot(YEARS, VALUES / 1e9, "ko-", linewidth=2, markersize=7, label="CA Historical", zorder=5)
axes[0].plot(FUTURE_YEARS, best["forecast"] / 1e9, "o-",
             color=COLOR["best"], linewidth=2.5, markersize=7,
             label=f"CA Forecast ({best_model_name})")
axes[0].fill_between(FUTURE_YEARS, best["ci_low"] / 1e9, best["ci_high"] / 1e9,
                     color=COLOR["ci"], alpha=0.4, label="95% CI")
axes[0].axvline(2025.5, color="gray", linestyle=":", linewidth=1)
axes[0].set_title("California Medicaid", fontsize=12)
axes[0].set_xlabel("Year"); axes[0].set_ylabel("Expenditure ($B)")
axes[0].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:.0f}B"))
axes[0].legend(fontsize=8); axes[0].grid(alpha=0.3)

# Right: National
axes[1].plot(YEARS, NATIONAL_VALUES / 1e9, "ko-", linewidth=2, markersize=7, label="National Historical", zorder=5)
axes[1].plot(FUTURE_YEARS, nat_best["forecast"] / 1e9, "o-",
             color=COLOR["nat_best"], linewidth=2.5, markersize=7,
             label=f"National Forecast ({nat_best_model_name})")
axes[1].fill_between(FUTURE_YEARS, nat_best["ci_low"] / 1e9, nat_best["ci_high"] / 1e9,
                     color=COLOR["ci_nat"], alpha=0.4, label="95% CI")
axes[1].axvline(2025.5, color="gray", linestyle=":", linewidth=1)
axes[1].set_title("National Medicaid", fontsize=12)
axes[1].set_xlabel("Year"); axes[1].set_ylabel("Expenditure ($B)")
axes[1].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:.0f}B"))
axes[1].legend(fontsize=8); axes[1].grid(alpha=0.3)

plt.suptitle("Medicaid Expenditure Forecast: California vs National (2026–2035)", fontsize=14)
plt.tight_layout()
plt.savefig(f"{OUT}/fig6_ca_vs_national_forecast.png", dpi=150)
plt.close()
print(f"Saved: {OUT}/fig6_ca_vs_national_forecast.png")

# ── Fig 7: California Share of National (historical + projected) ──
fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(YEARS, ca_share_hist, "o-", color=COLOR["share"], linewidth=2.5, markersize=8, label="Historical CA Share")
ax.plot(FUTURE_YEARS, ca_share_proj, "o--", color=COLOR["share"], linewidth=2, markersize=7,
        alpha=0.75, label="Projected CA Share")
ax.axvline(2025.5, color="gray", linestyle=":", linewidth=1)
for yr, sh in zip(YEARS, ca_share_hist):
    ax.annotate(f"{sh:.1f}%", xy=(yr, sh), xytext=(0, 10),
                textcoords="offset points", ha="center", fontsize=8, color=COLOR["share"])
for yr, sh in zip(FUTURE_YEARS, ca_share_proj):
    ax.annotate(f"{sh:.1f}%", xy=(yr, sh), xytext=(0, 10),
                textcoords="offset points", ha="center", fontsize=8, color=COLOR["share"])
ax.set_title("California's Share of National Medicaid Expenditure (2013–2035)", fontsize=13)
ax.set_xlabel("Year"); ax.set_ylabel("CA Share (%)")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.1f}%"))
ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUT}/fig7_ca_share_national.png", dpi=150)
plt.close()
print(f"Saved: {OUT}/fig7_ca_share_national.png")

print("\nAll figures saved.")

# ─────────────────────────────────────────────
# 8. EXPORT TO PDF
# ─────────────────────────────────────────────
figures = [
    f"{OUT}/fig1_historical_trend.png",
    f"{OUT}/fig1b_ca_vs_national_historical.png",
    f"{OUT}/fig2_yoy_growth.png",
    f"{OUT}/fig3_all_models.png",
    f"{OUT}/fig4_best_forecast_ci.png",
    f"{OUT}/fig5_rolling_12m.png",
    f"{OUT}/fig6_ca_vs_national_forecast.png",
    f"{OUT}/fig7_ca_share_national.png",
    f"{OUT}/eda_decomposition.png",
]
with PdfPages(f"{OUT}/medicaid_visualization_report.pdf") as pdf:
    for fpath in figures:
        img = plt.imread(fpath)
        fig, ax = plt.subplots(figsize=(11, 7))
        ax.imshow(img)
        ax.axis("off")
        pdf.savefig(fig, bbox_inches="tight")
        plt.close()
print(f"Saved: {OUT}/medicaid_visualization_report.pdf")

# ─────────────────────────────────────────────
# 9. EXPORT INTERACTIVE HTML REPORT
# ─────────────────────────────────────────────
hist_data    = list(zip(YEARS.tolist(), (VALUES / 1e9).round(2).tolist()))
nat_hist_data = list(zip(YEARS.tolist(), (NATIONAL_VALUES / 1e9).round(2).tolist()))
proj_data    = list(zip(
    FUTURE_YEARS.tolist(),
    (best["forecast"] / 1e9).round(2).tolist(),
    (best["ci_low"]   / 1e9).round(2).tolist(),
    (best["ci_high"]  / 1e9).round(2).tolist(),
))
nat_proj_data = list(zip(
    FUTURE_YEARS.tolist(),
    (nat_best["forecast"] / 1e9).round(2).tolist(),
    (nat_best["ci_low"]   / 1e9).round(2).tolist(),
    (nat_best["ci_high"]  / 1e9).round(2).tolist(),
))
share_hist_data = list(zip(YEARS.tolist(), ca_share_hist.round(2).tolist()))
share_proj_data = list(zip(FUTURE_YEARS.tolist(), ca_share_proj.round(2).tolist()))
growth_data  = list(zip(annual_df["Year"].tolist(), annual_df["YoY_Growth_pct"].fillna(0).round(1).tolist()))
nat_growth_data = list(zip(national_df["Year"].tolist(), national_df["YoY_Growth_pct"].fillna(0).round(1).tolist()))
model_table_rows = comparison_df.to_dict(orient="records")
nat_model_table_rows = nat_comparison_df.to_dict(orient="records")

html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Medicaid Expenditure Projection Report — CA vs National</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
<style>
  body{{font-family:system-ui,sans-serif;margin:0;padding:2rem;background:#f7fafc;color:#1a202c}}
  h1{{font-size:1.8rem;font-weight:700;margin-bottom:.25rem}}
  h2{{font-size:1.1rem;font-weight:600;color:#2d3748;margin:2rem 0 .75rem}}
  .subtitle{{color:#718096;font-size:.95rem;margin-bottom:2rem}}
  .cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:1rem;margin-bottom:2rem}}
  .card{{background:#fff;border:1px solid #e2e8f0;border-radius:10px;padding:1rem 1.25rem}}
  .card .label{{font-size:.75rem;color:#718096;text-transform:uppercase;letter-spacing:.05em}}
  .card .value{{font-size:1.4rem;font-weight:700;color:#2d3748;margin-top:.25rem}}
  .card.ca{{border-left:4px solid #2B6CB0}}
  .card.nat{{border-left:4px solid #C05621}}
  .card.share{{border-left:4px solid #6B46C1}}
  .chart-wrap{{background:#fff;border:1px solid #e2e8f0;border-radius:10px;padding:1.5rem;margin-bottom:1.5rem}}
  .chart-grid{{display:grid;grid-template-columns:1fr 1fr;gap:1.5rem;margin-bottom:1.5rem}}
  table{{width:100%;border-collapse:collapse;font-size:.875rem}}
  th{{background:#edf2f7;padding:.6rem .8rem;text-align:left;font-weight:600}}
  td{{padding:.55rem .8rem;border-bottom:1px solid #e2e8f0}}
  tr.best{{background:#f0fff4;font-weight:600}}
  .badge{{display:inline-block;padding:.2rem .5rem;border-radius:999px;font-size:.7rem;font-weight:600;background:#c6f6d5;color:#276749}}
  .section-title{{font-size:1.3rem;font-weight:700;color:#2d3748;margin:2.5rem 0 1rem;padding-bottom:.5rem;border-bottom:2px solid #e2e8f0}}
</style>
</head>
<body>
<h1>Medicaid Expenditure Projection — California vs National</h1>
<p class="subtitle">Historical analysis 2013–2025 &nbsp;·&nbsp; 10-year forecast 2026–2035</p>

<div class="cards">
  <div class="card ca"><div class="label">CA 2025</div><div class="value">${VALUES[-1]/1e9:.1f}B</div></div>
  <div class="card ca"><div class="label">CA 2035 Forecast</div><div class="value">${best["forecast"][-1]/1e9:.1f}B</div></div>
  <div class="card nat"><div class="label">National 2025</div><div class="value">${NATIONAL_VALUES[-1]/1e9:.0f}B</div></div>
  <div class="card nat"><div class="label">National 2035 Forecast</div><div class="value">${nat_best["forecast"][-1]/1e9:.0f}B</div></div>
  <div class="card share"><div class="label">CA Share 2025</div><div class="value">{ca_share_hist[-1]:.1f}%</div></div>
  <div class="card share"><div class="label">CA Share 2035 (proj)</div><div class="value">{ca_share_proj[-1]:.1f}%</div></div>
  <div class="card ca"><div class="label">CA Best Model</div><div class="value" style="font-size:1rem">{best_model_name}</div></div>
  <div class="card nat"><div class="label">Nat Best Model</div><div class="value" style="font-size:1rem">{nat_best_model_name}</div></div>
</div>

<div class="chart-wrap">
  <h2>California vs National — Historical + 10-Year Forecast</h2>
  <div style="position:relative;height:360px"><canvas id="forecastChart"></canvas></div>
</div>

<div class="chart-wrap">
  <h2>California Share of National Medicaid Expenditure (%)</h2>
  <div style="position:relative;height:240px"><canvas id="shareChart"></canvas></div>
</div>

<div class="chart-grid">
  <div class="chart-wrap">
    <h2>California YoY Growth (%)</h2>
    <div style="position:relative;height:220px"><canvas id="growthCAChart"></canvas></div>
  </div>
  <div class="chart-wrap">
    <h2>National YoY Growth (%)</h2>
    <div style="position:relative;height:220px"><canvas id="growthNatChart"></canvas></div>
  </div>
</div>

<div class="section-title">California Forecast Details</div>
<div class="chart-wrap">
  <h2>Model Comparison — California</h2>
  <table>
    <tr><th>Model</th><th>MAE ($B)</th><th>RMSE ($B)</th><th>MAPE (%)</th></tr>
    {''.join(
      f'<tr class="{"best" if r["Model"]==best_model_name else ""}"><td>{r["Model"]} {"<span class=badge>best</span>" if r["Model"]==best_model_name else ""}</td><td>{r["MAE ($B)"]}</td><td>{r["RMSE ($B)"]}</td><td>{r["MAPE (%)"]}</td></tr>'
      for r in model_table_rows
    )}
  </table>
</div>

<div class="section-title">National Forecast Details</div>
<div class="chart-wrap">
  <h2>Model Comparison — National</h2>
  <table>
    <tr><th>Model</th><th>MAE ($B)</th><th>RMSE ($B)</th><th>MAPE (%)</th></tr>
    {''.join(
      f'<tr class="{"best" if r["Model"]==nat_best_model_name else ""}"><td>{r["Model"]} {"<span class=badge>best</span>" if r["Model"]==nat_best_model_name else ""}</td><td>{r["MAE ($B)"]}</td><td>{r["RMSE ($B)"]}</td><td>{r["MAPE (%)"]}</td></tr>'
      for r in nat_model_table_rows
    )}
  </table>
</div>

<div class="chart-wrap">
  <h2>Full Projection Table (2026–2035): California vs National</h2>
  <table>
    <tr><th>Year</th><th>CA Forecast ($B)</th><th>CA CI Low</th><th>CA CI High</th><th>Nat Forecast ($B)</th><th>Nat CI Low</th><th>Nat CI High</th><th>CA Share (%)</th></tr>
    {''.join(
      f'<tr><td>{ca[0]}</td><td>${ca[1]:.2f}B</td><td>${ca[2]:.2f}B</td><td>${ca[3]:.2f}B</td><td>${nat[1]:.0f}B</td><td>${nat[2]:.0f}B</td><td>${nat[3]:.0f}B</td><td>{shr:.2f}%</td></tr>'
      for ca, nat, shr in zip(proj_data, nat_proj_data, ca_share_proj)
    )}
  </table>
</div>

<script>
const histCA  = {json.dumps(hist_data)};
const histNat = {json.dumps(nat_hist_data)};
const projCA  = {json.dumps(proj_data)};
const projNat = {json.dumps(nat_proj_data)};
const shareHist = {json.dumps(share_hist_data)};
const shareProj = {json.dumps(share_proj_data)};
const growCA  = {json.dumps(growth_data)};
const growNat = {json.dumps(nat_growth_data)};

const allYears = histCA.map(d=>d[0]).concat(projCA.map(d=>d[0]));
const caHistVals = histCA.map(d=>d[1]).concat(Array(projCA.length).fill(null));
const natHistVals = histNat.map(d=>d[1]).concat(Array(projNat.length).fill(null));
const caProjVals = Array(histCA.length).fill(null).concat(projCA.map(d=>d[1]));
const natProjVals = Array(histNat.length).fill(null).concat(projNat.map(d=>d[1]));
const caCI_L = Array(histCA.length).fill(null).concat(projCA.map(d=>d[2]));
const caCI_H = Array(histCA.length).fill(null).concat(projCA.map(d=>d[3]));
const natCI_L = Array(histNat.length).fill(null).concat(projNat.map(d=>d[2]));
const natCI_H = Array(histNat.length).fill(null).concat(projNat.map(d=>d[3]));

new Chart(document.getElementById("forecastChart"),{{
  type:"line",
  data:{{
    labels: allYears,
    datasets:[
      {{label:"California Historical", data:caHistVals, borderColor:"#2B6CB0", tension:.3, pointRadius:5, fill:false}},
      {{label:"CA Forecast ({best_model_name})", data:caProjVals, borderColor:"#276749", borderDash:[6,3], tension:.3, pointRadius:5, fill:false}},
      {{label:"National Historical", data:natHistVals, borderColor:"#C05621", tension:.3, pointRadius:5, fill:false}},
      {{label:"Nat Forecast ({nat_best_model_name})", data:natProjVals, borderColor:"#9B2335", borderDash:[6,3], tension:.3, pointRadius:5, fill:false}},
    ]
  }},
  options:{{
    responsive:true, maintainAspectRatio:false,
    plugins:{{legend:{{position:"top"}}}},
    scales:{{y:{{ticks:{{callback:v=>"$"+v+"B"}}}}}}
  }}
}});

const shareYears = shareHist.map(d=>d[0]).concat(shareProj.map(d=>d[0]));
const shareHistVals = shareHist.map(d=>d[1]).concat(Array(shareProj.length).fill(null));
const shareProjVals = Array(shareHist.length).fill(null).concat(shareProj.map(d=>d[1]));
new Chart(document.getElementById("shareChart"),{{
  type:"line",
  data:{{
    labels: shareYears,
    datasets:[
      {{label:"CA Share Historical (%)", data:shareHistVals, borderColor:"#6B46C1", backgroundColor:"rgba(107,70,193,.1)", tension:.3, pointRadius:5, fill:true}},
      {{label:"CA Share Projected (%)", data:shareProjVals, borderColor:"#6B46C1", borderDash:[6,3], tension:.3, pointRadius:5, fill:false}},
    ]
  }},
  options:{{
    responsive:true, maintainAspectRatio:false,
    plugins:{{legend:{{position:"top"}}}},
    scales:{{y:{{ticks:{{callback:v=>v+"%"}}}}}}
  }}
}});

new Chart(document.getElementById("growthCAChart"),{{
  type:"bar",
  data:{{
    labels: growCA.map(d=>d[0]),
    datasets:[{{label:"CA YoY (%)", data: growCA.map(d=>d[1]),
      backgroundColor: growCA.map(d=>d[1]>=0?"#2B6CB0":"#9B2335")}}]
  }},
  options:{{responsive:true, maintainAspectRatio:false,
    plugins:{{legend:{{display:false}}}},
    scales:{{y:{{ticks:{{callback:v=>v+"%"}}}}}}}}
}});

new Chart(document.getElementById("growthNatChart"),{{
  type:"bar",
  data:{{
    labels: growNat.map(d=>d[0]),
    datasets:[{{label:"Nat YoY (%)", data: growNat.map(d=>d[1]),
      backgroundColor: growNat.map(d=>d[1]>=0?"#C05621":"#9B2335")}}]
  }},
  options:{{responsive:true, maintainAspectRatio:false,
    plugins:{{legend:{{display:false}}}},
    scales:{{y:{{ticks:{{callback:v=>v+"%"}}}}}}}}
}});
</script>
</body></html>"""

with open(f"{OUT}/medicaid_report.html", "w") as f:
    f.write(html)
print(f"Saved: {OUT}/medicaid_report.html")

print("\n✓ Pipeline complete. All outputs in:", OUT)
