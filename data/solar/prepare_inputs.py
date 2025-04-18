"""
Convert PVWatts responses to solar traces in txt file format.
"""
import pandas as pd
import os

def remove_quotes_and_process(input_file, output_file):
    # Read and remove quotes from the CSV file
    with open(input_file, 'r') as file:
        content = file.read()
    modified_content = content.replace('"', '')
    
    with open(input_file, 'w') as file:
        file.write(modified_content)

    # Load and process the CSV data
    data = pd.read_csv(input_file)
    print(f"Processing {input_file} to {output_file}")

    # Extract the 'AC System Output (W)' column and convert values from watts to kilowatts
    ac_output_kw = data['AC System Output (W)'] / 1000

    # Write to output text file with each value on a new line
    with open(output_file, 'w') as file:
        for value in ac_output_kw:
            file.write(f"{value}\n")

    print(f"Total lines written: {len(ac_output_kw)}")

def get_solar_files():
    return [f"./data/solar/pvwatts/{f}" for f in os.listdir("./data/solar/pvwatts") if f.endswith('.csv')]

# Ensure the output directory exists
os.makedirs('./data/solar/processed/', exist_ok=True)

# Process the raw data
solar_files = get_solar_files()
for solar_file in solar_files:
    remove_quotes_and_process(f'./data/solar/pvwatts/{solar_file}.csv', f'./data/solar/processed/{solar_file}.txt')
    