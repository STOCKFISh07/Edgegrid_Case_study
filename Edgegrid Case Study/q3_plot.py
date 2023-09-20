import pandas as pd
import matplotlib.pyplot as plt
import os

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

# Convert the index (dates) to datetime objects
daily_avg_energy_sum.index = pd.to_datetime(daily_avg_energy_sum.index)

# Plot the time series data with improved x-axis labels
plt.figure(figsize=(12, 6))
plt.plot(daily_avg_energy_sum, label='Daily Average Energy Sum')
plt.xlabel('Date')
plt.ylabel('Energy Sum')
plt.title('Daily Average Energy Sum Time Series')
plt.legend()
plt.grid(True)

# Format the x-axis labels to show date
plt.gca().xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y-%m-%d'))
plt.gca().xaxis.set_major_locator(plt.matplotlib.dates.DayLocator(interval=30))  # Adjust the interval as needed

# Rotate the x-axis labels for better visibility
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()
