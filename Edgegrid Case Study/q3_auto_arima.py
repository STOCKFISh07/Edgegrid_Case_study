import pandas as pd
import os
from pmdarima.arima import auto_arima

data_folder = 'C:\\Users\\integral computer\\Downloads\\case_study\\case_study\\daily_dataset\\daily_dataset'

merged_df = pd.DataFrame()

for filename in os.listdir(data_folder):
    if filename.endswith('.csv'):
        filepath = os.path.join(data_folder, filename)
        df = pd.read_csv(filepath)
        merged_df = pd.concat([merged_df, df])


daily_avg_energy_sum = merged_df.groupby('day')['energy_sum'].mean()

model = auto_arima(daily_avg_energy_sum, seasonal=False, stepwise=True, trace=True, error_action='ignore',
                   suppress_warnings=True)
best_order = model.order

print("Best ARIMA Order (p, d, q):", best_order)
