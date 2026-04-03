import warnings
warnings.filterwarnings("ignore")
 
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # non-interactive backend for file export
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
VALUES = np.array([
    6.605676e+10, 6.824844e+10, 9.061436e+10, 8.660858e+10,
    8.866459e+10, 8.889577e+10, 9.410041e+10, 1.038866e+11,
    1.156904e+11, 1.250178e+11, 1.301333e+11, 1.570981e+11,
    1.734027e+11
])
 
annual_df = pd.DataFrame({"Year": YEARS, "Expenditure": VALUES})
annual_df["Date"] = pd.to_datetime(annual_df["Year"].astype(str) + "-01-01")
 
print("=== Annual Data (raw) ===")
print(annual_df.to_string(index=False))
 
# ─────────────────────────────────────────────
# 2. GENERATE MONTHLY SERIES VIA LINEAR INTERP
# ─────────────────────────────────────────────
monthly_dates = pd.date_range("2013-01-01", "2025-12-01", freq="MS")
 
# Linear model on fractional year → monthly values
lin_model = LinearRegression()
X_year = YEARS.reshape(-1, 1)
lin_model.fit(X_year, VALUES)
 
monthly_frac = np.array([d.year + (d.month - 1) / 12 for d in monthly_dates]).reshape(-1, 1)
monthly_values = lin_model.predict(monthly_frac)
 
monthly_df = pd.DataFrame({
    "Date": monthly_dates,
    "Expenditure_raw": monthly_values,
    "Expenditure": monthly_values  # numeric for modelling
})
monthly_df.set_index("Date", inplace=True)
 
# Save interpolated monthly CSV
monthly_csv = monthly_df.copy()
monthly_csv["Expenditure_fmt"] = monthly_csv["Expenditure_raw"].map("${:,.0f}".format)
monthly_csv.to_csv(f"{OUT}/monthly_interpolated_2013_2025.csv")
print(f"\nSaved: {OUT}/monthly_interpolated_2013_2025.csv  ({len(monthly_csv)} rows)")
 
# ─────────────────────────────────────────────
# 3. EXPLORATORY DATA ANALYSIS
# ─────────────────────────────────────────────
annual_df["YoY_Growth_pct"] = annual_df["Expenditure"].pct_change() * 100
 
print("\n=== Year-over-Year Growth ===")
print(annual_df[["Year", "Expenditure", "YoY_Growth_pct"]].to_string(index=False))
print("\n=== Summary Statistics (annual) ===")
print(annual_df["Expenditure"].describe().apply(lambda x: f"${x:,.0f}"))
 
# Decompose monthly series
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
# 4. FORECASTING MODELS
# ─────────────────────────────────────────────
# ── 4a. TRAIN / TEST SPLIT on annual data ──
SPLIT = 10   # train on first 10 years, test on last 3
X_train, X_test = YEARS[:SPLIT].reshape(-1, 1), YEARS[SPLIT:].reshape(-1, 1)
y_train, y_test = VALUES[:SPLIT], VALUES[SPLIT:]
 
# Projection horizon
FUTURE_YEARS = np.arange(2026, 2036)
 
results = {}   # {model_name: {metrics, forecast_df}}
 
def eval_metrics(true, pred, name):
    mae  = mean_absolute_error(true, pred)
    rmse = np.sqrt(mean_squared_error(true, pred))
    mape = np.mean(np.abs((true - pred) / true)) * 100
    print(f"  {name:30s} | MAE=${mae/1e9:.2f}B | RMSE=${rmse/1e9:.2f}B | MAPE={mape:.1f}%")
    return {"MAE": mae, "RMSE": rmse, "MAPE": mape}
 
 
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
 
 
# ── 4d. ARIMA on annual data ──
try:
    arima = SARIMAX(annual_df["Expenditure"], order=(1, 1, 1), trend="t").fit(disp=False)
    test_fcast = arima.get_forecast(steps=3)
    y_pred_arima = test_fcast.predicted_mean.values
    metrics_arima = eval_metrics(y_test, y_pred_arima, "ARIMA(1,1,1)")
    # Refit on all data for projection
    arima_full = SARIMAX(VALUES, order=(1, 1, 1), trend="t").fit(disp=False)
    fcast = arima_full.get_forecast(steps=10)
    future_arima = fcast.predicted_mean.values
    ci = fcast.conf_int()
    results["ARIMA(1,1,1)"] = {
        "metrics": metrics_arima,
        "forecast": future_arima,
        "ci_low":  ci.iloc[:, 0].values,
        "ci_high": ci.iloc[:, 1].values,
    }
except Exception as e:
    print(f"  ARIMA failed: {e}")
 
 
# ── 4e. HOLT-WINTERS EXPONENTIAL SMOOTHING ──
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
    # Approximate CI from in-sample RMSE
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
# 5. MODEL COMPARISON & BEST MODEL SELECTION
# ─────────────────────────────────────────────
comparison_rows = []
for name, res in results.items():
    comparison_rows.append({
        "Model":     name,
        "MAE ($B)":  round(res["metrics"]["MAE"]  / 1e9, 2),
        "RMSE ($B)": round(res["metrics"]["RMSE"] / 1e9, 2),
        "MAPE (%)":  round(res["metrics"]["MAPE"], 1),
    })
comparison_df = pd.DataFrame(comparison_rows).sort_values("MAPE (%)")
print("\n=== Model Comparison (sorted by MAPE) ===")
print(comparison_df.to_string(index=False))
comparison_df.to_csv(f"{OUT}/model_comparison.csv", index=False)
 
best_model_name = comparison_df.iloc[0]["Model"]
print(f"\nBest model: {best_model_name}")
 
# ─────────────────────────────────────────────
# 6. 10-YEAR PROJECTION TABLE
# ─────────────────────────────────────────────
best = results[best_model_name]
projection_df = pd.DataFrame({
    "Year":           FUTURE_YEARS,
    "Forecast ($)":   best["forecast"],
    "CI Lower ($)":   best["ci_low"],
    "CI Upper ($)":   best["ci_high"],
    "Forecast ($B)":  (best["forecast"] / 1e9).round(2),
    "CI Lower ($B)":  (best["ci_low"]   / 1e9).round(2),
    "CI Upper ($B)":  (best["ci_high"]  / 1e9).round(2),
})
projection_df.to_csv(f"{OUT}/projection_2026_2035.csv", index=False)
print(f"\n=== 10-Year Projection ({best_model_name}) ===")
print(projection_df[["Year", "Forecast ($B)", "CI Lower ($B)", "CI Upper ($B)"]].to_string(index=False))
print(f"Saved: {OUT}/projection_2026_2035.csv")
 
# ─────────────────────────────────────────────
# 7. VISUALIZATIONS
# ─────────────────────────────────────────────
COLOR = {"hist": "#2B6CB0", "best": "#276749", "ci": "#9AE6B4", "growth": "#C05621"}
 
def fmt_billions(ax, axis="y"):
    fmt = mticker.FuncFormatter(lambda x, _: f"${x/1e9:.0f}B")
    if axis == "y": ax.yaxis.set_major_formatter(fmt)
    else: ax.xaxis.set_major_formatter(fmt)
 
 
# ── Fig 1: Historical trend ──
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(annual_df["Year"], annual_df["Expenditure"] / 1e9, "o-",
        color=COLOR["hist"], linewidth=2, markersize=7, label="Historical")
ax.set_title("Medicaid Expenditure — Historical Trend (2013–2025)", fontsize=13)
ax.set_xlabel("Year"); ax.set_ylabel("Expenditure ($B)")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:.0f}B"))
ax.grid(alpha=0.3); ax.legend()
plt.tight_layout()
plt.savefig(f"{OUT}/fig1_historical_trend.png", dpi=150)
plt.close()
 
# ── Fig 2: YoY growth rate ──
fig, ax = plt.subplots(figsize=(10, 4))
colors = [COLOR["growth"] if g >= 0 else "#9B2335" for g in annual_df["YoY_Growth_pct"].fillna(0)]
ax.bar(annual_df["Year"], annual_df["YoY_Growth_pct"].fillna(0), color=colors, edgecolor="white")
ax.axhline(0, color="black", linewidth=0.8)
ax.set_title("Year-over-Year Growth Rate (%)", fontsize=13)
ax.set_xlabel("Year"); ax.set_ylabel("Growth (%)")
for yr, gv in zip(annual_df["Year"], annual_df["YoY_Growth_pct"].fillna(0)):
    ax.text(yr, gv + 0.3, f"{gv:.1f}%", ha="center", va="bottom", fontsize=8)
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUT}/fig2_yoy_growth.png", dpi=150)
plt.close()
 
# ── Fig 3: All model forecasts ──
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(YEARS, VALUES / 1e9, "ko-", linewidth=2, markersize=7, label="Historical", zorder=5)
colors_models = ["#2B6CB0", "#276749", "#C05621", "#6B46C1", "#B7791F"]
for (name, res), col in zip(results.items(), colors_models):
    ax.plot(FUTURE_YEARS, res["forecast"] / 1e9, "o--",
            color=col, linewidth=1.5, markersize=5,
            label=f"{name} {'★' if name == best_model_name else ''}")
ax.set_title("Medicaid Expenditure — All Model Projections (2026–2035)", fontsize=13)
ax.set_xlabel("Year"); ax.set_ylabel("Expenditure ($B)")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:.0f}B"))
ax.axvline(2025.5, color="gray", linestyle=":", linewidth=1)
ax.text(2025.6, ax.get_ylim()[0] * 1.02, "Forecast →", color="gray", fontsize=9)
ax.legend(fontsize=8); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUT}/fig3_all_models.png", dpi=150)
plt.close()
 
# ── Fig 4: Best model with CI ──
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(YEARS, VALUES / 1e9, "ko-", linewidth=2, markersize=8, label="Historical", zorder=5)
ax.plot(FUTURE_YEARS, best["forecast"] / 1e9, "o-",
        color=COLOR["best"], linewidth=2.5, markersize=8,
        label=f"Forecast ({best_model_name})")
ax.fill_between(FUTURE_YEARS,
                best["ci_low"]  / 1e9,
                best["ci_high"] / 1e9,
                color=COLOR["ci"], alpha=0.5, label="95% Confidence Interval")
ax.set_title(f"Medicaid Expenditure — 10-Year Projection [{best_model_name}]", fontsize=13)
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
ax.set_title("Rolling 12-Month Medicaid Expenditure Trend", fontsize=13)
ax.set_xlabel("Date"); ax.set_ylabel("Expenditure ($B)")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:.0f}B"))
ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUT}/fig5_rolling_12m.png", dpi=150)
plt.close()
 
print("\nAll figures saved.")

# ─────────────────────────────────────────────
# 8. EXPORT TO PDF
# ─────────────────────────────────────────────
figures = [
    f"{OUT}/fig1_historical_trend.png",
    f"{OUT}/fig2_yoy_growth.png",
    f"{OUT}/fig3_all_models.png",
    f"{OUT}/fig4_best_forecast_ci.png",
    f"{OUT}/fig5_rolling_12m.png",
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
# Serialize data for JavaScript
hist_data = list(zip(YEARS.tolist(), (VALUES / 1e9).round(2).tolist()))
proj_data = list(zip(
    FUTURE_YEARS.tolist(),
    (best["forecast"] / 1e9).round(2).tolist(),
    (best["ci_low"]   / 1e9).round(2).tolist(),
    (best["ci_high"]  / 1e9).round(2).tolist(),
))
growth_data = list(zip(
    annual_df["Year"].tolist(),
    annual_df["YoY_Growth_pct"].fillna(0).round(1).tolist()
))
model_table_rows = comparison_df.to_dict(orient="records")
 
html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Medicaid Expenditure Projection Report</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
<style>
  body{{font-family:system-ui,sans-serif;margin:0;padding:2rem;background:#f7fafc;color:#1a202c}}
  h1{{font-size:1.8rem;font-weight:700;margin-bottom:.25rem}}
  h2{{font-size:1.1rem;font-weight:600;color:#2d3748;margin:2rem 0 .75rem}}
  .subtitle{{color:#718096;font-size:.95rem;margin-bottom:2rem}}
  .cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:1rem;margin-bottom:2rem}}
  .card{{background:#fff;border:1px solid #e2e8f0;border-radius:10px;padding:1rem 1.25rem}}
  .card .label{{font-size:.75rem;color:#718096;text-transform:uppercase;letter-spacing:.05em}}
  .card .value{{font-size:1.5rem;font-weight:700;color:#2d3748;margin-top:.25rem}}
  .chart-wrap{{background:#fff;border:1px solid #e2e8f0;border-radius:10px;padding:1.5rem;margin-bottom:1.5rem}}
  table{{width:100%;border-collapse:collapse;font-size:.875rem}}
  th{{background:#edf2f7;padding:.6rem .8rem;text-align:left;font-weight:600}}
  td{{padding:.55rem .8rem;border-bottom:1px solid #e2e8f0}}
  tr.best{{background:#f0fff4;font-weight:600}}
  .badge{{display:inline-block;padding:.2rem .5rem;border-radius:999px;font-size:.7rem;font-weight:600;background:#c6f6d5;color:#276749}}
</style>
</head>
<body>
<h1>Medicaid Expenditure Projection Model</h1>
<p class="subtitle">Historical analysis 2013–2025 &nbsp;·&nbsp; 10-year forecast 2026–2035 &nbsp;·&nbsp; Best model: <strong>{best_model_name}</strong></p>
 
<div class="cards">
  <div class="card"><div class="label">2025 Expenditure</div><div class="value">$173.4B</div></div>
  <div class="card"><div class="label">2035 Forecast</div><div class="value">${best["forecast"][-1]/1e9:.1f}B</div></div>
  <div class="card"><div class="label">Avg Annual Growth</div><div class="value">{annual_df["YoY_Growth_pct"].mean():.1f}%</div></div>
  <div class="card"><div class="label">Best Model MAPE</div><div class="value">{comparison_df.iloc[0]["MAPE (%)"]:.1f}%</div></div>
</div>
 
<div class="chart-wrap">
  <h2>Historical + 10-Year Forecast with 95% CI</h2>
  <div style="position:relative;height:340px"><canvas id="forecastChart"></canvas></div>
</div>
 
<div class="chart-wrap">
  <h2>Year-over-Year Growth Rate</h2>
  <div style="position:relative;height:240px"><canvas id="growthChart"></canvas></div>
</div>
 
<div class="chart-wrap">
  <h2>Model Comparison</h2>
  <table>
    <tr><th>Model</th><th>MAE ($B)</th><th>RMSE ($B)</th><th>MAPE (%)</th></tr>
    {''.join(
      f'<tr class="{"best" if r["Model"]==best_model_name else ""}"><td>{r["Model"]} {"<span class=badge>best</span>" if r["Model"]==best_model_name else ""}</td><td>{r["MAE ($B)"]}</td><td>{r["RMSE ($B)"]}</td><td>{r["MAPE (%)"]}</td></tr>'
      for r in model_table_rows
    )}
  </table>
</div>
 
<div class="chart-wrap">
  <h2>Projection Table (2026–2035)</h2>
  <table>
    <tr><th>Year</th><th>Forecast ($B)</th><th>CI Lower ($B)</th><th>CI Upper ($B)</th></tr>
    {''.join(f'<tr><td>{yr}</td><td>${fc:.2f}B</td><td>${lo:.2f}B</td><td>${hi:.2f}B</td></tr>'
             for yr, fc, lo, hi in proj_data)}
  </table>
</div>
 
<script>
const hist = {json.dumps(hist_data)};
const proj = {json.dumps(proj_data)};
const grow = {json.dumps(growth_data)};
 
const allYears = hist.map(d=>d[0]).concat(proj.map(d=>d[0]));
const histVals = hist.map(d=>d[1]).concat(Array(proj.length).fill(null));
const projVals = Array(hist.length).fill(null).concat(proj.map(d=>d[1]));
const ciLow  = Array(hist.length).fill(null).concat(proj.map(d=>d[2]));
const ciHigh = Array(hist.length).fill(null).concat(proj.map(d=>d[3]));
 
new Chart(document.getElementById("forecastChart"),{{
  type:"line",
  data:{{
    labels: allYears,
    datasets:[
      {{label:"Historical", data:histVals, borderColor:"#2B6CB0", backgroundColor:"rgba(43,108,176,.1)", tension:.3, pointRadius:5, fill:false}},
      {{label:"Forecast ({best_model_name})", data:projVals, borderColor:"#276749", backgroundColor:"rgba(39,103,73,.1)", borderDash:[6,3], tension:.3, pointRadius:5, fill:false}},
      {{label:"95% CI Upper", data:ciHigh, borderColor:"rgba(154,230,180,.5)", pointRadius:0, fill:"+1", backgroundColor:"rgba(154,230,180,.3)"}},
      {{label:"95% CI Lower", data:ciLow, borderColor:"rgba(154,230,180,.5)", pointRadius:0, fill:false}},
    ]
  }},
  options:{{responsive:true,maintainAspectRatio:false,plugins:{{legend:{{position:"top"}}}},scales:{{y:{{ticks:{{callback:v=>"$"+v+"B"}}}}}}}}
}});
 
new Chart(document.getElementById("growthChart"),{{
  type:"bar",
  data:{{
    labels: grow.map(d=>d[0]),
    datasets:[{{
      label:"YoY Growth (%)",
      data: grow.map(d=>d[1]),
      backgroundColor: grow.map(d=>d[1]>=0?"#276749":"#C05621"),
    }}]
  }},
  options:{{responsive:true,maintainAspectRatio:false,plugins:{{legend:{{display:false}}}},scales:{{y:{{ticks:{{callback:v=>v+"%"}}}}}}}}
}});
</script>
</body></html>"""
 
with open(f"{OUT}/medicaid_report.html", "w") as f:
    f.write(html)
print(f"Saved: {OUT}/medicaid_report.html")
 
print("\n✓ Pipeline complete. All outputs in:", OUT)
