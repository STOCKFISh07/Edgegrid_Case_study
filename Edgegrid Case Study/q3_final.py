import pandas as pd
from statsmodels.tsa.stattools import adfuller
import os
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
import numpy as np
import matplotlib.pyplot as plt

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

# Check if the time series is stationary based on the ADF test
if result[1] <= 0.05:
    print("The time series is stationary (reject the null hypothesis)")

    # Define the ARIMA order obtained from auto_arima
    p, d, q = 3, 0, 5

    # Split the data into training and testing sets
    train_size = int(len(first_difference) * 0.8)
    train, test = first_difference[0:train_size], first_difference[train_size:]

    # Initialize an empty list to store predictions
    predictions = []

    # Train the ARIMA model on the training data and update it at each time step
    model = ARIMA(train, order=(p, d, q))
    model_fit = model.fit()

    # Number of years to forecast
    years_to_forecast = 5
    forecast_steps = years_to_forecast * 365  # Assuming daily data

    # Forecast future values for the next 5 years
    for _ in range(forecast_steps):
        # Forecast the next time step
        forecast = model_fit.forecast(steps=1)[0]
        predictions.append(forecast)
        # Update the model with the forecasted value
        train = np.append(train, forecast)
        model_fit = ARIMA(train, order=(p, d, q)).fit()

    # Create date range for the forecasted period (next 5 years)
    last_date = daily_avg_energy_sum.index[-1]
    forecast_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=forecast_steps, closed='right')

    # Plot the original time series and forecasted values
    plt.figure(figsize=(12, 6))
    plt.plot(daily_avg_energy_sum.index, daily_avg_energy_sum, label='Original', color='blue')
    plt.plot(forecast_dates, predictions, label='Forecasted', color='red')
    plt.title('ARIMA Forecasting for the Next 5 Years')
    plt.xlabel('Time')
    plt.ylabel('Differenced Energy Sum')
    plt.legend()

    # Improve x-axis labeling
    ax = plt.gca()
    labels = ax.get_xticklabels()
    for i, label in enumerate(labels):
        if i % (len(labels) // 10) != 0:
            label.set_visible(False)

    plt.show()

else:
    print("The time series is non-stationary (fail to reject the null hypothesis)")
