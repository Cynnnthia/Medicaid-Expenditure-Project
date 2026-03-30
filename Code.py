import pandas as pd

years = range(2013, 2025)
data = []

for year in years:
    file_path = f"../data/FY_{year}_MFCU_Statistical_Chart.xlsx"
    
    df = pd.read_excel(file_path)
    
    # Find California
    CA_row = df[df["State"] == "California"]
    exp = CA_row["Total Medicaid Expenditures"].values[0]
    
    # Clean data
    exp = str(exp).replace("$", "").replace(",", "")
    exp = float(exp)
    
    data.append([year, exp])

ca_df = pd.DataFrame(data, columns=["Year", "Expenditure"])

print(ca_df)
