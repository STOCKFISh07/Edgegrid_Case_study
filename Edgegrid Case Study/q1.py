import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os


data_folder = 'C:\\Users\\integral computer\\Downloads\\case_study\\case_study\\daily_dataset\\daily_dataset'  # Replace with the actual folder path


merged_df = pd.DataFrame()


for filename in os.listdir(data_folder):
    if filename.startswith('block_'):
        filepath = os.path.join(data_folder, filename)
        df = pd.read_csv(filepath)
        merged_df = pd.concat([merged_df, df])


daily_aggregated_df = merged_df.groupby('day')['energy_mean'].mean().reset_index()

plt.figure(figsize=(12, 6))
sns.lineplot(x='day', y='energy_mean', data=daily_aggregated_df)
plt.xlabel('Day')
plt.ylabel('Mean Energy Consumption')
plt.title('Energy Consumption Over Days')
plt.xticks(rotation=45)
plt.show()


data_folder2 = 'C:\\Users\\integral computer\\Downloads\\case_study\\case_study\\hhblock_dataset\\hhblock_dataset'  # Replace with the actual folder path

merged_df2 = pd.DataFrame()


for filename2 in os.listdir(data_folder2):
    if filename2.startswith('block_'):
        filepath2 = os.path.join(data_folder2, filename2)
        df = pd.read_csv(filepath2)
        merged_df2 = pd.concat([merged_df2, df])

merged_df2['tstp'] = pd.to_datetime(merged_df2['tstp'])
merged_df2['hour'] = merged_df2['tstp'].dt.hour

max_consumption_times = merged_df2.groupby('hour')['energy(kWh/hh)'].idxmax()
peak_hours = merged_df2.loc[max_consumption_times, 'hour']

sns.histplot(peak_hours, bins=24, kde=True)
plt.xlabel('Hour of the Day')
plt.ylabel('Frequency')
plt.title('Peak Consumption Hours Distribution')
plt.show()
