"""
Increase number of load samples by adding noise and scaling the original traces. Additionally, the peaks in the load curves are 
shifted in time according to eating habits in different countries. The original load curves are based on UK data for which
we assume the following eating schedule:
Breakfast: 8.00
Lunch: 13.00
Dinner: 18.00
"""
import numpy as np
import os
import random

def get_load_files(split):
    return [f"./data/load/{split}/{f}" for f in os.listdir(f"./data/load/{split}") if f.endswith('.txt')]

def add_noise(time_series, mean=0.0, stddev=0.1):
    """
    Adds noise to a time series.
    """
    # Gaussian noise generation
    noise = np.random.normal(mean, stddev, len(time_series))
    # Adding noise to the original time series and ensure that values aren't negative
    noisy_series = np.clip(time_series + noise, 0.05, None)

    return noisy_series

def scale_time_series(time_series, yearly_load_range=(1000, 15000)):
    """
    Shifts the entire time series up or down by a random factor in shift_range.
    """
    current_load = np.sum(time_series)
    target_load = np.random.uniform(*yearly_load_range)
    difference = target_load - current_load
    # Add constant load in each timestep
    add = (np.random.uniform(0.1,0.5)*difference) / 8760
    # Scale the time series to match the target load
    factor = target_load / (current_load+(add*8760))
    shifted_series = [(t+add)*factor for t in time_series]
    return shifted_series

def shift_B(time_series):
    """
    Shifts the time series to match Swedish eating habits.
    Breakfast: 7.00
    Lunch: 12.00
    Dinner: 17.00
    """
    # Define hours during which we shift
    morning_hour = random.randint(0,4)
    evening_hour = random.randint(20,22)
    for day in range(365):
        daily_trace = [0 for _ in range(24)]
        daily_trace[:morning_hour] = time_series[day*24:day*24+morning_hour]
        daily_trace[morning_hour:evening_hour] = time_series[day*24+morning_hour+1:day*24+evening_hour+1]
        daily_trace[evening_hour] = (time_series[day*24+evening_hour]+time_series[day*24+evening_hour+1])/2
        daily_trace[evening_hour+1:] = time_series[day*24+evening_hour+1:(day+1)*24]

        time_series[day*24:(day+1)*24] = daily_trace
    
    return time_series

def shift_C(time_series):
    """
    Shifts the time series to match German eating habits.
    Breakfast: 8.00
    Lunch: 13.00
    Dinner: 19.00
    """
    # Define hours during which we shift
    afternoon_hour = random.randint(14,15)
    evening_hour = random.randint(21,22)
    for day in range(365):
        daily_trace = [0 for _ in range(24)]
        daily_trace[:afternoon_hour] = time_series[day*24:day*24+afternoon_hour]
        daily_trace[afternoon_hour] = (time_series[day*24+afternoon_hour-1]+time_series[day*24+afternoon_hour])/2
        daily_trace[afternoon_hour+1:evening_hour] = time_series[day*24+afternoon_hour:day*24+evening_hour-1]
        daily_trace[evening_hour:] = time_series[day*24+evening_hour:(day+1)*24]

        time_series[day*24:(day+1)*24] = daily_trace
    
    return time_series

def shift_D(time_series):
    """
    Shifts the time series to match Italian eating habits.
    Breakfast: 8.00
    Lunch: 13.00
    Dinner: 20.00
    """
    # Define hours during which we shift
    afternoon_hour = random.randint(14,15)
    evening_hour = random.randint(21,22)
    for day in range(365):
        daily_trace = [0 for _ in range(24)]
        daily_trace[:afternoon_hour] = time_series[day*24:day*24+afternoon_hour]
        daily_trace[afternoon_hour] = (time_series[day*24+afternoon_hour-1]+time_series[day*24+afternoon_hour])/2
        daily_trace[afternoon_hour+1] = (daily_trace[afternoon_hour]+time_series[day*24+afternoon_hour])/2

        daily_trace[afternoon_hour+2:evening_hour] = time_series[day*24+afternoon_hour:day*24+evening_hour-2]
        daily_trace[evening_hour:] = time_series[day*24+evening_hour-1:(day+1)*24-1]

        time_series[day*24:(day+1)*24] = daily_trace
    
    return time_series

def shift_E(time_series):
    """
    Shifts the time series to match Spanish eating habits.
    Breakfast: 9.00
    Lunch: 14.00
    Dinner: 21.00
    """
    # Define hours during which we shift
    morning_hour = random.randint(0,5)
    afternoon_hour = random.randint(14,15)
    evening_hour = 22
    for day in range(365):
        daily_trace = [0 for _ in range(24)]
        daily_trace[:morning_hour] = time_series[day*24:day*24+morning_hour]
        daily_trace[morning_hour] = (time_series[day*24+morning_hour-1]+time_series[day*24+morning_hour])/2

        daily_trace[morning_hour+1:afternoon_hour] = time_series[day*24+morning_hour:day*24+afternoon_hour-1]
        daily_trace[afternoon_hour] = (time_series[day*24+afternoon_hour-2]+time_series[day*24+afternoon_hour-1])/2
        daily_trace[afternoon_hour+1] = (daily_trace[afternoon_hour]+time_series[day*24+afternoon_hour-1])/2

        daily_trace[afternoon_hour+2:evening_hour] = time_series[day*24+afternoon_hour-1:day*24+evening_hour-3]
        daily_trace[evening_hour:] = time_series[day*24+evening_hour-2:(day+1)*24-2]

        time_series[day*24:(day+1)*24] = daily_trace
    
    return time_series


# Create noisy traces
country_profiles = [shift_D, shift_C, shift_E, shift_B]

for split in ["train", "test", "val"]:
    load_files = get_load_files(split)
    for filepath in load_files:
        trace = np.loadtxt(filepath, delimiter=",")

        for i, country in enumerate(["B", "C", "D", "E"]):
            augmented_series = country_profiles[i](trace)

            augmented_series = add_noise(augmented_series)
            augmented_series = scale_time_series(augmented_series)
            augmented_series = np.clip(augmented_series, 0.05, None)

            filename = os.path.basename(filepath).split(".")[0]
            output_path = os.path.join(f"./data/load/{split}", filename+f"_{country}.txt")

            with open(output_path, 'w') as f:
                for value in augmented_series:
                    f.write(f"{value}\n")

