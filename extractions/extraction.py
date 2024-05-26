import re
import csv

# Define the regular expressions for each piece of data
regex_patterns = {
    "Energy consumed for RAM": r"Energy consumed for RAM\s*:\s*([\d.]+)\s*kWh",
    "RAM Power": r"RAM Power\s*:\s*([\d.]+)\s*W",
    "Energy consumed for all GPUs": r"Energy consumed for all GPUs\s*:\s*([\d.]+)\s*kWh",
    "Total GPU Power": r"Total GPU Power\s*:\s*([\d.]+)\s*W",
    "Energy consumed for all CPUs": r"Energy consumed for all CPUs\s*:\s*([\d.]+)\s*kWh",
    "Total CPU Power": r"Total CPU Power\s*:\s*([\d.]+)\s*W",
    "Electricity used since the beginning": r"([\d.]+)\s*kWh of electricity used since the beginning"
}

# Read the log file
log_file_path = 'C:/Users/Paul/Documents/TuGraz/HCI/HCI-SLM/results/Logging Results/nanoGPT/codecarbon_M1_ARM.log'
with open(log_file_path, 'r') as file:
    lines = file.readlines()

# List to store multiple sets of data
data_list = []

# Initialize a temporary dictionary to hold values for each log section
temp_data = {key: None for key in regex_patterns.keys()}

# Parse each line to find matches
for line in lines:
    for key, pattern in regex_patterns.items():
        match = re.search(pattern, line)
        if match:
            temp_data[key] = match.group(1)

    # If a complete set of data is found (all values are not None), save it and reset temp_data
    if all(value is not None for value in temp_data.values()):
        data_list.append(temp_data.copy())
        temp_data = {key: None for key in regex_patterns.keys()}

# Define the CSV file path
csv_file_path = 'C:/Users/Paul/Documents/TuGraz/HCI/HCI-SLM/results/Logging Results/nanoGPT/extracted_data.csv'

# Write the extracted values to a CSV file
with open(csv_file_path, 'w', newline='') as csv_file:
    writer = csv.DictWriter(csv_file, fieldnames=regex_patterns.keys())
    writer.writeheader()
    writer.writerows(data_list)

print(f"Data extracted and saved to {csv_file_path}")
