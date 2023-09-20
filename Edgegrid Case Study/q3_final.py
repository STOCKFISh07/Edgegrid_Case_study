import pandas as pd
from statsmodels.tsa.stattools import adfuller
import os
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
import numpy as np
import matplotlib.pyplot as plt

data_folder = 'C:\\Users\\integral computer\\Downloads\\case_study\\case_study\\daily_dataset\\daily_dataset'

merged_df = pd.DataFrame()

for filename in os.listdir(data_folder):
    if filename.endswith('.csv'):
        filepath = os.path.join(data_folder, filename)
        df = pd.read_csv(filepath)
        merged_df = pd.concat([merged_df, df])

daily_avg_energy_sum = merged_df.groupby('day')['energy_sum'].mean()

first_difference = daily_avg_energy_sum.diff().dropna()

result = adfuller(first_difference, autolag='AIC')

if result[1] <= 0.05:
    print("The time series is stationary (reject the null hypothesis)")

    p, d, q = 3, 0, 5

    train_size = int(len(first_difference) * 0.8)
    train, test = first_difference[0:train_size], first_difference[train_size:]

    predictions = []

    model = ARIMA(train, order=(p, d, q))
    model_fit = model.fit()


    years_to_forecast = 5
    forecast_steps = years_to_forecast * 365


    for _ in range(forecast_steps):

        forecast = model_fit.forecast(steps=1)[0]
        predictions.append(forecast)

        train = np.append(train, forecast)
        model_fit = ARIMA(train, order=(p, d, q)).fit()

    last_date = daily_avg_energy_sum.index[-1]
    forecast_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=forecast_steps, closed='right')


    plt.figure(figsize=(12, 6))
    plt.plot(daily_avg_energy_sum.index, daily_avg_energy_sum, label='Original', color='blue')
    plt.plot(forecast_dates, predictions, label='Forecasted', color='red')
    plt.title('ARIMA Forecasting for the Next 5 Years')
    plt.xlabel('Time')
    plt.ylabel('Differenced Energy Sum')
    plt.legend()

    ax = plt.gca()
    labels = ax.get_xticklabels()
    for i, label in enumerate(labels):
        if i % (len(labels) // 10) != 0:
            label.set_visible(False)

    plt.show()

else:
    print("The time series is non-stationary (fail to reject the null hypothesis)")
