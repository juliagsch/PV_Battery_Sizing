"""
Convert half-hourly load traces over one day of 1000 households fetched from Faraday to one hourly load trace over one year per household.
"""
import os
import json

input_path = "./data/load/faraday"
output_path = "./data/load/original"

def process_responses(input_file):
    with open(input_file, 'r') as file:
        data = json.load(file)

    populationKeys = ["DetachedA", "DetachedD", "TerracedA"] # Adapt based on names used in fetch_load.py

    for p in populationKeys:
        response = next((item for item in data["message"]["results"] if item["name"] == p), None)
        kwh = response["kwh"]

        # Write to output text file with each hourly value on a new line
        for idx, daily_trace in enumerate(kwh):
            with open(f'{output_path}/{p}_{idx}.txt', 'a') as file:
                for i in range(0, len(daily_trace) - 1, 2):
                    # Get hourly values by adding half-hourly load
                    file.write(f"{float(daily_trace[i]) + float(daily_trace[i + 1])}\n")

# Create output directory
os.makedirs(output_path, exist_ok=True)

# Process the raw data to get yearly traces
for load_file_idx in range(365):
    process_responses(f'{input_path}/day_{load_file_idx}.json')
        
