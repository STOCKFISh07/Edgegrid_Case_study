import pandas as pd
from statsmodels.tsa.stattools import adfuller
import os
from pmdarima.arima import auto_arima
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


print("ADF Statistic:", result[0])
print("P-value:", result[1])
print("Number of Lags Used:", result[2])
print("Number of Observations Used:", result[3])
print("Critical Values:")
for key, value in result[4].items():
    print(f"{key}: {value}")

if result[1] <= 0.05:
    print("The time series is stationary (reject the null hypothesis)")


    p, d, q = 3, 0, 5


    train_size = int(len(first_difference) * 0.8)
    train, test = first_difference[0:train_size], first_difference[train_size:]


    predictions = []


    model = ARIMA(train, order=(p, d, q))
    model_fit = model.fit()

    for t in range(len(test)):

        forecast = model_fit.forecast(steps=1)[0]
        predictions.append(forecast)

        train = np.append(train, test[t])
        model_fit = ARIMA(train, order=(p, d, q)).fit()

    # Calculate the Mean Squared Error (MSE) on the test data
    mse = mean_squared_error(test, predictions)
    rmse = np.sqrt(mse)
    print(f"Root Mean Squared Error (RMSE): {rmse}")

    test_avg = np.mean(test)
    print(f"Average of Test Data: {test_avg}")


    plt.figure(figsize=(12, 6))
    plt.plot(first_difference.index[train_size:], test, label='Actual', color='blue')
    plt.plot(first_difference.index[train_size:], predictions, label='Predicted', color='red')
    plt.title('ARIMA Forecasting')
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

#Output: Root Mean Squared Error (RMSE): 0.8862469398235996