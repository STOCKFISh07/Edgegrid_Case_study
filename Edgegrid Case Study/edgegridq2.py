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


if any(col.startswith('hh_') for col in merged_df2.columns):

    max_consumption_times = merged_df2.iloc[:, 2:].idxmax(axis=1)
    peak_hours = max_consumption_times.str.replace('hh_', '').astype(float)

    sns.histplot(peak_hours, bins=48, kde=True)
    plt.xlabel('Half-Hour of the Day')
    plt.ylabel('Units Energy consumed')
    plt.title('Peak Consumption Hours Distribution')
    plt.show()
else:
    print("No 'hh_' columns found in the merged DataFrame.")
