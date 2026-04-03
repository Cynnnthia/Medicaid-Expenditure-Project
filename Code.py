import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# yearly data
years = np.array([
    2013,2014,2015,2016,2017,2018,2019,
    2020,2021,2022,2023,2024,2025
]).reshape(-1,1)

values = np.array([
    6.605676e+10, 6.824844e+10, 9.061436e+10, 8.660858e+10,
    8.866459e+10, 8.889577e+10, 9.410041e+10, 1.038866e+11,
    1.156904e+11, 1.250178e+11, 1.301333e+11, 1.570981e+11,
    1.734027e+11
])

model = LinearRegression()
model.fit(years, values)

# generate monthly dates
monthly_dates = pd.date_range(start="2013-01-01", end="2025-12-01", freq="MS")

monthly_years = np.array([
    d.year + (d.month - 1)/12 for d in monthly_dates
]).reshape(-1,1)

# generate monthly data
monthly_values = model.predict(monthly_years)

monthly_df = pd.DataFrame({
    "Date": monthly_dates,
    "Expenditure_raw": monthly_values
})

# transfer to monetary form
monthly_df["Expenditure"] = monthly_df["Expenditure_raw"].map('${:,.0f}'.format)

# print it out
print(monthly_df.head())
print(monthly_df.tail())
print("Total rows:", len(monthly_df)) 

monthly_df.to_csv("monthly_predicted_2013_2025.csv", index=False)

# generate the graphs and growth rate

import matplotlib.pyplot as plt

df = pd.read_csv("monthly_expenditure_2013_2025.csv")
df["Date"] = pd.to_datetime(df["Date"])
if "Expenditure_raw" in df.columns:
    df["Value"] = df["Expenditure_raw"]
else:
    df["Value"] = df["Expenditure"].replace('[\$,]', '', regex=True).astype(float)

df = df.sort_values("Date")

# generate overall trend
plt.figure()
plt.plot(df["Date"], df["Value"])
plt.title("Medicaid Expenditure Trend (Monthly)")
plt.xlabel("Year")
plt.ylabel("Expenditure ($)")
plt.show()

# generate yearly aggression
df["Year"] = df["Date"].dt.year

yearly = df.groupby("Year")["Value"].mean().reset_index()

plt.figure()
plt.plot(yearly["Year"], yearly["Value"], marker='o')
plt.title("Yearly Medicaid Expenditure Trend")
plt.xlabel("Year")
plt.ylabel("Expenditure ($)")
plt.show()

# generate growth rate
yearly["YoY_Growth"] = yearly["Value"].pct_change() * 100

plt.figure()
plt.plot(yearly["Year"], yearly["YoY_Growth"], marker='o')
plt.title("Year-over-Year Growth Rate (%)")
plt.xlabel("Year")
plt.ylabel("Growth (%)")
plt.axhline(0)
plt.show()

print(yearly)

# rolling trend
df["Rolling_12M"] = df["Value"].rolling(window=12).mean()

plt.figure()
plt.plot(df["Date"], df["Value"], alpha=0.4)
plt.plot(df["Date"], df["Rolling_12M"])
plt.title("Rolling 12-Month Trend")
plt.show()

# summary statistics
print("\nSummary Statistics:")
print(df["Value"].describe())

