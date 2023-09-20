import pandas as pd
import os
from pmdarima.arima import auto_arima
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt

data_folder = 'C:\\Users\\integral computer\\Downloads\\case_study\\case_study\\daily_dataset\\daily_dataset'

merged_df = pd.DataFrame()

for filename in os.listdir(data_folder):
    if filename.endswith('.csv'):
        filepath = os.path.join(data_folder, filename)
        df = pd.read_csv(filepath)
        merged_df = pd.concat([merged_df, df])


daily_avg_energy_sum = merged_df.groupby('day')['energy_sum'].mean()


train_size = int(len(daily_avg_energy_sum) * 0.8)
train, test = daily_avg_energy_sum[:train_size], daily_avg_energy_sum[train_size:]

p, d, q = 2, 1, 2
model = ARIMA(train, order=(p, d, q))
model_fit = model.fit()

predictions = model_fit.predict(start=0, end=len(train) - 1, typ='levels')

plt.figure(figsize=(12, 6))
plt.plot(train.index, train, label='Actual', color='blue')
plt.plot(train.index, predictions, label='Predicted', color='red')
plt.title('ARIMA Model - Training Set')
plt.xlabel('Time')
plt.ylabel('Daily Average Energy Sum')
plt.legend()

plt.grid(True, linestyle='--', alpha=0.7)

ax = plt.gca()
labels = ax.get_xticklabels()
for i, label in enumerate(labels):
    if i % (len(labels) // 10) != 0:
        label.set_visible(False)

plt.show()
