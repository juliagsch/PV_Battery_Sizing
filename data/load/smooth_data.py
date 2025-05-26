"""
Enforce weekly and monthly patterns by averaging load traces with the average load trace during that month or weekday.
"""
import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

def smooth_electricity_profile_week(filepath, output_dir):
    with open(filepath, 'r') as f:
        values = [float(line.strip()) for line in f if line.strip()]

    if len(values) != 8760:
        print(f"Skipping {filepath}: not 8760 values.")
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

        # Average profiles by weekday
        weekday_profiles = {}
        for wd in range(7):
            matches = daily_profiles[np.array(weekdays) == wd]
            if len(matches) > 0:
                weekday_profiles[wd] = np.mean(matches, axis=0)

        # Smooth each day by averaging it with its weekday profile
        for i, day in enumerate(daily_profiles):
            wd = weekdays[i]
            avg_wd = weekday_profiles[wd]
            smoothed_day = (day + avg_wd) / 2
            smoothed.extend(smoothed_day)

    filename = os.path.basename(filepath)
    output_path = os.path.join(output_dir, filename)

    with open(output_path, 'w') as f:
        for val in smoothed:
            f.write(f"{val:.4f}\n")



def smooth_electricity_profile_month(filepath, output_dir):
    with open(filepath, 'r') as f:
        values = [float(line.strip()) for line in f if line.strip()]

    if len(values) != 8760:
        print(f"Skipping {filepath}: not 8760 values.")
        return

    # Create datetime index for a non-leap year
    start = datetime(2023, 1, 1)
    index = [start + timedelta(hours=i) for i in range(8760)]

    df = pd.DataFrame({'value': values}, index=index)

    smoothed = []

    for month in range(1, 13):
        # Extract all days in the current month and group
        month_data = df[df.index.month == month]
        days = [group for _, group in month_data.groupby(month_data.index.date)]

        # Get average day profile (shape: 24)
        daily_profiles = [day['value'].values for day in days]
        avg_day = np.mean(daily_profiles, axis=0)

        # Smooth each day by averaging with avg_day
        for day in daily_profiles:
            smoothed_day = (day + avg_day) / 2
            smoothed.extend(smoothed_day)

    filename = os.path.basename(filepath)
    output_path = os.path.join(output_dir, filename)

    with open(output_path, 'w') as f:
        for val in smoothed:
            f.write(f"{val:.4f}\n")

def process_folder(input_folder):
    output_folder_month = './data/load/faraday_yearly/averaged_month'
    output_folder_week = './data/load/faraday_yearly/averaged_week'
    
    os.makedirs(output_folder_month, exist_ok=True)
    os.makedirs(output_folder_week, exist_ok=True)

    for file in os.listdir(input_folder):
        if file.endswith('.txt'):
            filepath = os.path.join(input_folder, file)
            smooth_electricity_profile_month(filepath, output_folder_month)
            smooth_electricity_profile_week(filepath, output_folder_week)

input_folder = './data/load/faraday_yearly/original'

process_folder(input_folder)
