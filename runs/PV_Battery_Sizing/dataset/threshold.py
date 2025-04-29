"""
Filters samples with sizings out of limit. 
"""
import torch
import numpy as np
import csv

# Define the threshold values
max_threshold_battery = 20
max_threshold_pv = 10
# Define input and output files
input_filename = 'dataset_paper_test_interleaved.csv'
output_filename = 'dataset_below_threshold_test.csv'

try:
    data = torch.tensor(torch.from_numpy(np.loadtxt(input_filename, delimiter=",")), dtype=torch.float32)
except FileNotFoundError:
    print("Error: dataset_paper.csv not found. Please make sure the file exists.")
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