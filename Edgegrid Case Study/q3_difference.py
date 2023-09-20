import pandas as pd
from statsmodels.tsa.stattools import adfuller
import os

# Specify the directory where the CSV files are located
data_folder = 'C:\\Users\\integral computer\\Downloads\\case_study\\case_study\\daily_dataset\\daily_dataset'

# Initialize an empty DataFrame to store data from all CSV files
merged_df = pd.DataFrame()

# Iterate through each CSV file in the folder and merge data
for filename in os.listdir(data_folder):
    if filename.endswith('.csv'):
        filepath = os.path.join(data_folder, filename)
        df = pd.read_csv(filepath)
        merged_df = pd.concat([merged_df, df])

# Calculate the daily average of energy_sum for all LCLid
daily_avg_energy_sum = merged_df.groupby('day')['energy_sum'].mean()

# Calculate the first difference of the daily_avg_energy_sum
first_difference = daily_avg_energy_sum.diff().dropna()

# Perform the ADF test for stationarity on the differenced series
result = adfuller(first_difference, autolag='AIC')

# Print the ADF test results
print("ADF Statistic:", result[0])
print("P-value:", result[1])
print("Number of Lags Used:", result[2])
print("Number of Observations Used:", result[3])
print("Critical Values:")
for key, value in result[4].items():
    print(f"{key}: {value}")

# Check if the time series is stationary based on the ADF test
if result[1] <= 0.05:
    print("The time series is stationary (reject the null hypothesis)")
else:
    print("The time series is non-stationary (fail to reject the null hypothesis)")
