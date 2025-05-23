import os
import pandas as pd

directory = "merged"

# Read the CSV file into a DataFrame
def get_ev_files():
    return [f for f in os.listdir(f"./data/ev/{directory}") if f.endswith('.csv')]

for file in get_ev_files():
    df = pd.read_csv(f"./data/ev/{directory}/{file}")

    num_weeks_commute = 46 # We assume 6 weeks of holidays and sick leave
    num_weeks_noncommute = 52

    # Get data to compare to expected values
    no_trips_lines = (df == "No trips").sum().sum()
    total_lines = len(df)
    num_trips = total_lines-no_trips_lines
    total_distance = df['Distance (km)'].sum()

    # Get expected values
    meta = file[:-4].split("_")
    num_commute, num_non_commute = float(meta[0]), float(meta[1])
    km_commute, km_non_commute = float(meta[2]), float(meta[3])
    expected_km = num_commute*km_commute*num_weeks_commute + num_non_commute*km_non_commute*num_weeks_noncommute

    if abs(total_distance - expected_km) > 0.1 * expected_km:
        print("Distance differs by more than 10% from expected for file ", file, f" Expected: {expected_km}, Got: {total_distance}")
    
    if directory != "merged":
        expected_trips = num_commute*num_weeks_commute + num_non_commute*num_weeks_noncommute

        if abs(num_trips - expected_trips) > 0.1 * expected_trips:
            print("Number of trips differs from expected for file ", file, f" Expected: {expected_trips}, Got: {num_trips}")
 
    # Print outputs
    # total_emissions = total_distance * 132 / 1000
    # kwh = (total_distance * 164) / 1000 
    # print(f"Total Distance (km): {total_distance}")
    # print(f"kWh: {kwh}")
    # print(f"Petrol: {(total_distance/15.3052)}")
    # print(f"Petrol Cost: {(total_distance/15.3052)*1.36}")
    # print(f"Total CO2 Emissions (kg) petrol: {total_emissions}")
    # print(f"Days without trips': {no_trips_lines}")
    # print(f"Number of Trips': {total_lines-no_trips_lines-1}")
    # print(f"Distance per trip': {total_distance/(total_lines-no_trips_lines-1)}")
