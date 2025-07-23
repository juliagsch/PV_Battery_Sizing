"""
Creates modified solar traces by shifting the original traces and by bridging and creating gaps to model cloudiness. 
Original traces fetched from PVWatts for locations around the world are used. Locations are shown in map.png
"""
import numpy as np
import os
import random

def get_solar_files(split):
    return [f"./data/solar/{split}/{f}" for f in os.listdir(f"./data/solar/{split}/") if f.endswith('.txt')]

def modify_cloudiness(time_series, p_bridge=0.3, p_gap=0.15):
    """
    Creates and bridges gaps in solar curves to model cloudiness. 
    - A gap is created by lowering the solar generation of the current hour to 25%-75% with probability p_gap. 
    - The solar curve is bridged with probability p_bridge by averaging the solar generation in the previous and in the next hour.
      This only occurs if there is a gap in the curve.
    """
    modified_series = [time_series[0]]

    for i in range(1, len(time_series) - 1):
        r = random.uniform(0,1)

        if r < p_bridge and (time_series[i-1]+time_series[i+1])/2 > time_series[i]:
            modified_series.append((time_series[i-1]+time_series[i+1])/2)
        elif r > 1.0-p_gap:
            gap_strength = random.uniform(0.25,0.75)
            modified_series.append(time_series[i]*gap_strength)
        else:
            modified_series.append(time_series[i])
    
    modified_series.append(time_series[-1])
    return modified_series

def shift_time_series(time_series, shift_range=(0.8, 1.2)):
    """
    Shifts the entire time series up or down by a random factor in shift_range.
    """
    shift_value = np.random.uniform(*shift_range)
    shifted_series = [t*shift_value for t in time_series]
    return shifted_series, [shift_value for _ in range(24)]


# Specify how many times the original trace should be duplicated.
num_runs = 3

for split in ["train", "test", "val"]:
    load_files = get_solar_files(split)

    for i in range(num_runs):
        for file in load_files:
            data = np.loadtxt(file, delimiter=",")
            modified_series= modify_cloudiness(time_series=data)
            modified_series, shift = shift_time_series(modified_series)

            file_name = os.path.basename(file)[:-4]
            noisy_file = f"./data/solar/{split}/{file_name}_{i}.txt"
            # Save noisy data to txt with each value on a new line
            with open(noisy_file, 'w') as f:
                for value in modified_series:
                    f.write(f"{value}\n")

