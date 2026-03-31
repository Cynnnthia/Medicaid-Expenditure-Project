import pandas as pd

# use yearly data to generate monthly data in monetary form without scientific notation

# Yearly data
data = {
    "Year": list(range(2013, 2026)),
    "Expenditure": [
        6.605676e+10, 6.824844e+10, 9.061436e+10, 8.660858e+10,
        8.866459e+10, 8.889577e+10, 9.410041e+10, 1.038866e+11,
        1.156904e+11, 1.250178e+11, 1.301333e+11, 1.570981e+11,
        1.734027e+11
    ]
}

df = pd.DataFrame(data)

df["Date"] = pd.to_datetime(df["Year"].astype(str) + "-01-01")
df.set_index("Date", inplace=True)

monthly_index = pd.date_range(start="2013-01-01", end="2025-12-01", freq="MS")

df = df.reindex(df.index.union(monthly_index))

df = df.sort_index()
df["Expenditure"] = df["Expenditure"].interpolate(method="linear")

# only keep the monthly data
monthly_df = df.loc[monthly_index]

monthly_df = monthly_df.reset_index()
monthly_df.rename(columns={"index": "Date"}, inplace=True)

# turn it to monetary form
monthly_df["Expenditure"] = monthly_df["Expenditure"].map('${:,.0f}'.format)

# print it out
print(monthly_df.head())
print(monthly_df.tail())
print("Total rows:", len(monthly_df))

monthly_df.to_csv("monthly_expenditure_2013_2025.csv", index=False)
