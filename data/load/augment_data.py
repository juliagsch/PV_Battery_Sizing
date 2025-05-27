"""
Increase number of load samples by adding noise and shifting the original traces. Additionally, the peaks in the load curves are 
shifted in time according to eating habits in Sweden, Germany, Spain and Italy. The original load curves are based on UK data for which
we assume the following eating schedule:
Breakfast: 8.00
Lunch: 13.00
Dinner: 18.00
"""
import numpy as np
import os
import random

def get_load_files(dir):
    return [f"./data/load/faraday_yearly/averaged_{dir}/{f}" for f in os.listdir(f"./data/load/faraday_yearly/averaged_{dir}") if f.endswith('.txt')]

def add_noise(time_series, mean=0.0, stddev=0.1):
    """
    Adds noise to a time series.
    """
    # Gaussian noise generation
    noise = np.random.normal(mean, stddev, len(time_series))

    # Avoid extreme spikes by averaging with the noise in the previous time step.
    for i in range(1, len(noise)):
        noise[i] = (noise[i-1]+noise[i])/2

    # Adding noise to the original time series and ensure that values aren't negative
    noisy_series = np.clip(time_series + noise, 0, None)

    return noisy_series

def scale_time_series(time_series, shift_range=(0.5, 2)):
    """
    Shifts the entire time series up or down by a random factor in shift_range.
    """
    shift_value = np.random.uniform(*shift_range)
    shifted_series = [t*shift_value for t in time_series]
    return shifted_series

def shift_sweden(time_series):
    """
    Shifts the time series to match Swedish eating habits.
    Breakfast: 7.00
    Lunch: 12.00
    Dinner: 17.00
    """
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

def shift_germany(time_series):
    """
    Shifts the time series to match German eating habits.
    Breakfast: 8.00
    Lunch: 13.00
    Dinner: 19.00
    """
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

def shift_italy(time_series):
    """
    Shifts the time series to match Italian eating habits.
    Breakfast: 8.00
    Lunch: 13.00
    Dinner: 20.00
    """
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

def shift_spain(time_series):
    """
    Shifts the time series to match Spanish eating habits.
    Breakfast: 9.00
    Lunch: 14.00
    Dinner: 21.00
    """
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
duplicates_per_trace = 1
dir = "week"
load_files = get_load_files(dir)

out_dir = f"./data/load/faraday_yearly/noisy_{dir}"
os.makedirs(out_dir, exist_ok=True)

country_profiles = [shift_italy, shift_germany, shift_spain, shift_sweden]

for _ in range(duplicates_per_trace):
    for filepath in load_files:
        trace = np.loadtxt(filepath, delimiter=",")

        # Choose a random country profile according to which the load curve will be shifted in time
        augmented_series = random.choice(country_profiles)(trace)

        augmented_series = add_noise(augmented_series)
        augmented_series = scale_time_series(augmented_series)

        filename = os.path.basename(filepath)
        output_path = os.path.join(out_dir, filename)

        with open(output_path, 'w') as f:
            for value in augmented_series:
                f.write(f"{value}\n")

