"""
Increase number of load samples by adding noise and shifting the original traces.
"""
import numpy as np
import os

def get_load_files():
    files = []
    for type in ["Detached", "EV", "NoLCT", "Semi-detached", "Terraced"]:
        files = files + [f"./data/load/processed_poster/{type}/{f}" for f in os.listdir(f"./data/load/processed_poster/{type}") if f.endswith('.txt')]
    return files

def add_noise(time_series, mean=0.0, stddev=0.2):
    """
    Adds noise to a time series.
    """
    # Gaussian noise generation
    noise = np.random.normal(mean, stddev, len(time_series))

    # Avoid spikes by averaging with the noise in the previous time step.
    for i in range(1, len(noise)):
        noise[i] = (noise[i-1]+noise[i])/2

    # Adding noise to the original time series and ensure that values aren't negative
    noisy_series = np.clip(time_series + noise, 0, None)

    return noisy_series

def shift_time_series(time_series, shift_range=(-0.1, 0.3)):
    """
    Shifts the entire time series up or down by adding a constant random value in shift_range.
    """
    shift_value = np.random.uniform(*shift_range)  # Random shift value
    shifted_series = np.clip(time_series + shift_value, 0, None)
    return shifted_series

duplicates_per_trace = 2
load_files = get_load_files()
os.makedirs("./data/load/processed_poster/Noisy", exist_ok=True)

for i in range(duplicates_per_trace):
    for file in load_files:
        # Only add noise to original load traces
        if "noisy" in file:
            continue

        data = np.loadtxt(file, delimiter=",")
        noisy_series = add_noise(data)
        noisy_series = shift_time_series(noisy_series)

        file_name = file.split("/")[-1]
        file_name = file_name.replace(".txt", f"_noisy_{i}.txt")

        noisy_file = f"./data/load/processed_poster/Noisy/{file_name}"
        with open(noisy_file, 'w') as f:
            for value in noisy_series:
                f.write(f"{value}\n")

