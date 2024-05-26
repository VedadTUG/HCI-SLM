import pandas as pd
import matplotlib.pyplot as plt

# Define the CSV file paths
csv_file_path_1 = 'C:/Users/Paul/Documents/TuGraz/HCI/HCI-SLM/results/Logging Results/nanoGPT/extracted_data_GTX306TI_AMD.csv'
csv_file_path_2 = 'C:/Users/Paul/Documents/TuGraz/HCI/HCI-SLM/results/Logging Results/nanoGPT/extracted_data_M1_ARM.csv'

# Read the CSV files into pandas DataFrames
df1 = pd.read_csv(csv_file_path_1)
df2 = pd.read_csv(csv_file_path_2)

# Check if the columns are the same in both DataFrames
if list(df1.columns) != list(df2.columns):
    print("The columns in the two CSV files do not match.")
else:
    # Iterate over each column and plot the data from both DataFrames
    for column in df1.columns:
        plt.figure()  # Create a new figure for each column
        plt.plot(df1[column], marker='o', linestyle='-', label='GTX3060Ti-AMD')
        plt.plot(df2[column], marker='x', linestyle='--', label='M1-ARM')
        plt.title(column)
        plt.xlabel('Index')
        plt.ylabel(column)
        plt.legend()
        plt.grid(True)
        plt.show()

print("All columns have been plotted.")
