"""
Creates modified solar traces by shifting the original traces and by bridging and creating gaps to model cloudiness. 
The traces in ./data/solar/united are used as a base.
They contain 1000 solar traces in the UK, 1000 traces in the UK+Ireland and 700 traces from around the world. 
The locations from which the solar traces are fetched can be seen in united_map.png. As there are fewer weather stations than samples,
especially in the UK, the united folder contains many duplicates. Nevertheless, it can be used as a base to create the modified traces
which results in a suitable distribution with solar traces mostly related to the UK but including more diverse data from around the world.
"""
import numpy as np
import os
import random

def get_solar_files():
    return [f"./data/solar/united/{f}" for f in os.listdir(f"./data/solar/united/") if f.endswith('.txt')]


def modify_cloudiness(time_series, p_bridge=0.15, p_gap=0.15):
    """
    Creates and bridges gaps in solar curves to model cloudiness. 
    - A gap is created by lowering the solar generation of the current hour to 25%-75% with probability p_gap. 
    - The solar curve is bridged with probability p_bridge by averaging the solar generation in the previous and in the next hour.
    """
    modified_series = [time_series[0]]

    for i in range(1, len(time_series) - 1):
        r = random.uniform(0,1)

        if r < p_bridge:
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


load_files = get_solar_files()

for file in load_files:
    if "noisy" in file:
        continue
    
    data = np.loadtxt(file, delimiter=",")
    modified_series= modify_cloudiness(time_series=data)
    modified_series, shift = shift_time_series(modified_series)

    file_name = file.split("/")[-1]
    noisy_file = f"./data/solar/noisy/{file_name}"
    # Save noisy data to txt with each value on a new line
    with open(noisy_file, 'w') as f:
        for value in modified_series:
            f.write(f"{value}\n")

