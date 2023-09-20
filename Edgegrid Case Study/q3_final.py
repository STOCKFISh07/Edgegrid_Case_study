import pandas as pd
import os
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


data_folder = 'C:\\Users\\integral computer\\Downloads\\case_study\\case_study\\daily_dataset\\daily_dataset'

merged_df = pd.DataFrame()


for filename in os.listdir(data_folder):
    if filename.endswith('.csv'):
        filepath = os.path.join(data_folder, filename)
        df = pd.read_csv(filepath)
        merged_df = pd.concat([merged_df, df])


daily_avg_energy_sum = merged_df.groupby('day')['energy_sum'].mean()


daily_avg_energy_sum.index = pd.to_datetime(daily_avg_energy_sum.index)


train_size = int(len(daily_avg_energy_sum) * 0.95)
train, test = daily_avg_energy_sum[:train_size], daily_avg_energy_sum[train_size:]


p, d, q = 2, 1, 2
model = ARIMA(train, order=(p, d, q))
model_fit = model.fit()


forecast_steps = 365 * 5
forecast_values = model_fit.forecast(steps=forecast_steps)


forecast_dates = pd.date_range(start=daily_avg_energy_sum.index[-1] + pd.DateOffset(days=1), periods=forecast_steps)


plt.figure(figsize=(12, 6))
plt.plot(daily_avg_energy_sum.index, daily_avg_energy_sum, label='Actual', color='blue')
plt.plot(train.index, model_fit.fittedvalues, label='Fitted', color='orange')
plt.plot(forecast_dates, forecast_values, label='Forecasted (Next 5 Years)', color='green')
plt.title('ARIMA Model - Entire Dataset and Future Forecast')
plt.xlabel('Time')
plt.ylabel('Daily Average Energy Sum')
plt.legend()

ax = plt.gca()
ax.xaxis.set_major_locator(MaxNLocator(integer=True))

plt.grid(True, linestyle='--', alpha=0.7)

plt.show()
