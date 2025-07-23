"""
Enforce weekly and monthly patterns by averaging load traces with the average load trace during that month and weekday.
"""
import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

def smooth_electricity_profile_week(file, path):
    with open(os.path.join(path, file), 'r') as f:
        values = [float(line.strip()) for line in f if line.strip()]

    if len(values) != 8760:
        print(f"Skipping {file}: not 8760 values.")
        return

    # Create datetime index for a non-leap year
    start = datetime(2023, 1, 1)
    index = [start + timedelta(hours=i) for i in range(8760)]

    df = pd.DataFrame({'value': values}, index=index)
    
    smoothed = []
    for month in range(1, 13):
        month_data = df[df.index.month == month]

        daily_groups = [group for _, group in month_data.groupby(month_data.index.date)]
        daily_profiles = []
        weekdays = []

        for day_data in daily_groups:
            daily_profiles.append(day_data['value'].values)
            weekdays.append(day_data.index[0].weekday())

        daily_profiles = np.array(daily_profiles)

        # Create average profiles by weekday
        weekday_profiles = {}
        for weekday in range(7):
            matches = daily_profiles[np.array(weekdays) == weekday]
            if len(matches) > 0:
                weekday_profiles[weekday] = np.mean(matches, axis=0)

        # Smooth each day by averaging it with its weekday profile
        for i, day in enumerate(daily_profiles):
            weekday = weekdays[i]
            avg_weekday = weekday_profiles[weekday]
            smoothed_day = (day + avg_weekday) / 2
            smoothed.extend(smoothed_day)

    output_path = os.path.join(path, file+"smooth.txt")

    with open(output_path, 'w') as f:
        for val in smoothed:
            f.write(f"{val:.4f}\n")

def process_folder(input_folder):
    for file in os.listdir(input_folder):
        if file.endswith('.txt'):
            filepath = os.path.join(input_folder, file)
            smooth_electricity_profile_week(filepath)

for split in ["train", "test", "val"]:
    process_folder(f"./data/load/{split}")
