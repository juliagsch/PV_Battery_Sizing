"""
Convert half-hourly load traces over one day of 1000 households fetched from Faraday to one hourly load trace over one year per household.
"""
import os
import json
import shutil

input_path = "./data/load/faraday_large"
output_path = "./data/load/processed_poster"

def process_responses(input_file):
    with open(input_file, 'r') as file:
        data = json.load(file)

    population = {
        "NoLCT": "NoLCT",
        "EV": "EV",
        "DetachedA": "Detached",
        "DetachedD": "Detached",
        "TerracedA": "Terraced",
        "TerracedD": "Terraced",
        "Semi-detachedA": "Semi-detached",
        "Semi-detachedD": "Semi-detached"
    }

    for p in population:
        os.makedirs(f"{output_path}/{population[p]}", exist_ok=True)

        # Extract results
        response = next((item for item in data["message"]["results"] if item["name"] == p), None)
        kwh = response["kwh"]

        # Write to output text file with each hourly value on a new line
        for idx, daily_trace in enumerate(kwh):
            with open(f'{output_path}/{population[p]}/{p}_{idx}.txt', 'a') as file:
                for i in range(0, len(daily_trace) - 1, 2):
                    # Get hourly values by adding half-hourly load
                    file.write(f"{float(daily_trace[i]) + float(daily_trace[i + 1])}\n")


if __name__ == "__main__":
    # Create output directory
    shutil.rmtree("./data/load/processed_poster/")
    os.makedirs('./data/load/processed_poster/', exist_ok=True)

    # Process the raw data to get yearly traces
    for load_file_idx in range(365):
        process_responses(f'./data/load/faraday_raw/day_{load_file_idx}.json')
        
