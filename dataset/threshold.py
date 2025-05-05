"""
Filters samples with sizings out of limit. 
"""
import torch
import numpy as np
import csv
import pandas as pd

# Define the threshold values
max_threshold_battery = 30
max_threshold_pv = 20
# Define input and output files
dataset_path = "/cluster/home/jgschwind"
input_filename = f'{dataset_path}/dataset_test_interleaved.csv'
output_filename = f'{dataset_path}/dataset_test_interleaved_threshold3020.csv'

try:
    # Load train and test csv
    data = pd.read_csv(input_filename, header=None)

    # Map charging policies to integer
    data.replace("safe_arrival", 0, inplace=True)
    data.replace("safe_departure", 1, inplace=True)
    data.replace("arrival_limit", 2, inplace=True)
    data.replace("lbn_limit", 3, inplace=True)

    data = torch.tensor(data.astype(float).to_numpy(), dtype=torch.float32)
except FileNotFoundError:
    exit()

# Extract the target columns (assuming they are the last two: battery, pv)
y = data[:, -2:]

# Create a boolean mask for rows where both battery and PV values are below their thresholds
mask_battery = y[:, 0] < max_threshold_battery
mask_pv = y[:, 1] < max_threshold_pv
combined_mask = mask_battery & mask_pv

lines_to_write = data[combined_mask].numpy().tolist()

try:
    with open(output_filename, 'w', newline='') as outfile:
        writer = csv.writer(outfile)
        writer.writerows(lines_to_write)
    print(f"Wrote {len(lines_to_write)} lines with max values below threshold to {output_filename}")
except Exception as e:
    print(f"Error writing to {output_filename}: {e}")