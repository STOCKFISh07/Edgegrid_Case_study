import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error

# Specify the directory where the CSV files are located
data_folder2 = 'C:\\Users\\integral computer\\Downloads\\case_study\\case_study\\hhblock_dataset\\hhblock_dataset'  # Replace with the actual folder path

# Initialize an empty DataFrame to store data from all CSV files
merged_df2 = pd.DataFrame()

# Iterate through each CSV file in the folder and merge data
for filename2 in os.listdir(data_folder2):
    if filename2.startswith('block_'):
        filepath2 = os.path.join(data_folder2, filename2)
        df = pd.read_csv(filepath2)

        # Merge data into the main DataFrame
        merged_df2 = pd.concat([merged_df2, df])

# Check if the 'hh_0' column exists in the merged DataFrame
if 'hh_0' in merged_df2.columns:

    # Set 'day' column as the index and resample to daily frequency
    merged_df2['day'] = pd.to_datetime(merged_df2['day'])
    merged_df2.set_index('day', inplace=True)
    daily_energy = merged_df2.resample('D').sum()

    # Choose the 'hh_0' column for analysis
    daily_energy = daily_energy[['hh_0']]

    # Convert 'hh_0' column to a numerical data type (e.g., float)
    daily_energy['hh_0'] = pd.to_numeric(daily_energy['hh_0'], errors='coerce')

    # Drop rows with missing values
    daily_energy.dropna(subset=['hh_0'], inplace=True)

    # Split the data into training and testing sets
    train_size = int(len(daily_energy) * 0.8)
    train, test = daily_energy[:train_size], daily_energy[train_size:]

    # Fit an ARIMA model (you may need to adjust order parameters)
    model = ARIMA(train['hh_0'], order=(5, 1, 0))
    model_fit = model.fit()

    # Make predictions
    forecast = model_fit.forecast(steps=len(test))

    # Calculate Mean Squared Error
    mse = mean_squared_error(test, forecast)
    print(f'Mean Squared Error: {mse}')

    # Plot actual vs. forecasted energy consumption
    plt.figure(figsize=(12, 6))
    plt.plot(test.index, test, label='Actual')
    plt.plot(test.index, forecast, color='red', label='Forecast')
    plt.xlabel('Date')
    plt.ylabel('Energy Consumption (hh_0)')
    plt.title('Energy Consumption Forecast')
    plt.legend()
    plt.show()
else:
    print("No 'hh_0' column found in the merged DataFrame.")
