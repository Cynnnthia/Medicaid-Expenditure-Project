import pandas as pd

years = range(2013, 2025)
data = []

for year in years:
    file_path = f"FY_2013_MFCU_Statistical_Chart.xlsx"
    file_path = f"FY_2014_MFCU_Statistical_Chart.xlsx"
    file_path = f"FY_2015_MFCU_Statistical_Chart.xlsx"
    file_path = f"FY_2016_MFCU_Statistical_Chart.xlsx"
    file_path = f"FY_2017_MFCU_Statistical_Chart.xlsx"
    file_path = f"FY_2018_MFCU_Statistical_Chart.xlsx"
    file_path = f"FY_2019_MFCU_Statistical_Chart.xlsx"
    file_path = f"FY_2020_MFCU_Statistical_Chart.xlsx"
    file_path = f"FY_2021_MFCU_Statistical_Chart.xlsx"
    file_path = f"FY_2022_MFCU_Statistical_Chart.xlsx"
    file_path = f"FY_2023_MFCU_Statistical_Chart.xlsx"
    file_path = f"FY_2024_MFCU_Statistical_Chart.xlsx"
    file_path = f"FY_2025_MFCU_Statistical_Chart.xlsx"
    
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
