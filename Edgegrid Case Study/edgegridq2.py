import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

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

# Check if the 'hour' columns exist in the merged DataFrame
if any(col.startswith('hh_') for col in merged_df2.columns):
    # Find the time(s) of day when consumption is at max
    max_consumption_times = merged_df2.iloc[:, 2:].idxmax(axis=1)
    peak_hours = max_consumption_times.str.replace('hh_', '').astype(float)

    # Visualize peak consumption hours
    sns.histplot(peak_hours, bins=48, kde=True)
    plt.xlabel('Hour of the Day')
    plt.ylabel('Frequency')
    plt.title('Peak Consumption Hours Distribution')
    plt.show()
else:
    print("No 'hh_' columns found in the merged DataFrame.")
