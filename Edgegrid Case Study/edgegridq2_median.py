import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

data_folder2 = 'C:\\Users\\integral computer\\Downloads\\case_study\\case_study\\hhblock_dataset\\hhblock_dataset'  # Replace with the actual folder path

merged_df2 = pd.DataFrame()

for filename2 in os.listdir(data_folder2):
    if filename2.startswith('block_'):
        filepath2 = os.path.join(data_folder2, filename2)
        df = pd.read_csv(filepath2)
        merged_df2 = pd.concat([merged_df2, df])

half_hour_columns = merged_df2.columns[2:]

half_hour_periods = [int(col.split('_')[1]) for col in half_hour_columns]


median_consumption = merged_df2[half_hour_columns].median()

plt.figure(figsize=(12, 6))
plt.bar(half_hour_periods, median_consumption)
plt.xlabel('Half-Hour of the Day')
plt.ylabel('Median Energy Consumption')
plt.title('Median Energy Consumption for Each Half-Hour Period')
plt.xticks(half_hour_periods)
plt.show()
