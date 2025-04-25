import pandas as pd

# Read the CSV file into a DataFrame
path = "./data/ev/train/259.csv"
df = pd.read_csv(path)

# Sum the values in the 'Distance (km)' column
total_distance = df['Distance (km)'].sum()

total_emissions = total_distance * 132 / 1000

# Count total lines and how many contain "no trips "
total_lines = 0
no_trips_lines = 0

with open(path, 'r', encoding='utf-8') as f:
    for line in f:
        total_lines += 1
        if "No trips" in line:
            no_trips_lines += 1

kwh = (total_distance * 164) / 1000  
# Print outputs
print(f"Total Distance (km): {total_distance}")
print(f"kWh: {kwh}")
print(f"Petrol: {(total_distance/15.3052)}")
print(f"Petrol Cost: {(total_distance/15.3052)*1.36}")
print(f"Total CO2 Emissions (kg) petrol: {total_emissions}")
print(f"Days without trips': {no_trips_lines}")
print(f"Number of Trips': {total_lines-no_trips_lines-1}")
print(f"Distance per trip': {total_distance/(total_lines-no_trips_lines-1)}")
