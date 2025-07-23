"""Plot solar and load traces to verify data."""
import numpy as np
import matplotlib.pyplot as plt
import os
import random

def read_and_process_hourly(file_path, average, day=0, num_hours=24):
    with open(file_path, 'r') as file:
        # Read all lines and convert them to floats
        data = np.array([float(line.strip()) for line in file.readlines()])

    if(average):
        hourly = [np.mean(data[hour::num_hours]) for hour in range(num_hours)]
    else:
        hourly = data[day*24:day*24+num_hours]
    
    return hourly

def read_and_process_weekdays(file_path, average, week=0):
    with open(file_path, 'r') as file:
        data = np.array([float(line.strip()) for line in file.readlines()])

    weekday_total = [0.0 for _ in range(7)]
    weekday_count = [0 for _ in range(7)]
    if(average):
        for day in range(365):
            weekday_data = data[day*24:day*24+24]
            weekday_total[day%7] += np.sum(weekday_data)
            weekday_count[day%7] += 1
        weekday_total = [t/c for t,c in zip(weekday_total, weekday_count)]
    else:
        for day in range(7*week, 7*week+7):
            weekday_data = data[day*24:day*24+24]
            weekday_total[day%7] += np.sum(weekday_data)
    
    return weekday_total

def read_and_process_monthly(file_path):
    with open(file_path, 'r') as file:
        data = np.array([float(line.strip()) for line in file.readlines()])
    
    # Assume data has 365 days, so we split it into 12 months
    days_in_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    
    monthly_averages = []
    start_idx = 0
    for days in days_in_month:
        end_idx = start_idx + days
        monthly_averages.append(np.sum(data[start_idx * 24:end_idx * 24])/days)  # Average over the month
        start_idx = end_idx  # Move to the next month
    
    return monthly_averages

def analyze_and_plot_daily(load_files, average, datatype):
    plt.figure(figsize=(12, 6))
    if(len(load_files)>10):
        load_files = random.sample(load_files, 1)

    num_days = 1
    num_hours = num_days*24
    if(average):
        num_hours = 24
    
    for load_file in load_files:
        hourly = read_and_process_hourly(load_file, average, day=random.randint(0,364),num_hours=num_hours)
        label = load_file
        plt.plot(range(num_hours), hourly, label=label, marker='o')
    
    plt.title(f'Average Hourly {datatype}')
    plt.xlabel('Hour of the Day')
    plt.ylabel(datatype)
    plt.xticks(range(num_hours), labels=[f'{hour}:00' for hour in range(num_hours)], rotation=45)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

def analyze_and_plot_weekly(files, datatype, average):
    plt.figure(figsize=(12, 6))
    if(len(files)>10):
        files = random.sample(files, 10)

    for file in files:
        weekly = read_and_process_weekdays(file, average)
        plt.plot(range(7), weekly, label=file, marker='o')
    
    plt.title(f'Daily {datatype} Average')
    plt.xlabel('Weekday')
    plt.ylabel(datatype)
    plt.xticks(range(7), labels=[f'{month}' for month in ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]], rotation=45)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

def analyze_and_plot_monthly(files, datatype):
    plt.figure(figsize=(12, 6))

    total = [0.0 for _ in range(12)]
    for file in files:
        monthly = read_and_process_monthly(file)
        total += np.add(total, monthly)/len(files)
    
    plt.plot(range(12), total, label="average", marker='o')
    plt.title(f'Average {datatype} per Day')
    plt.xlabel('Month of the Year')
    plt.ylabel(datatype)
    plt.xticks(range(12), labels=[f'{month}' for month in ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]], rotation=45)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


def get_load_files(split):
    return [f"./data/load/{split}/{f}" for f in os.listdir(f"./data/load/{split}") if f.endswith('.txt')]

def get_solar_files(split):
    return [f"./data/solar/{split}/{f}" for f in os.listdir(f"./data/solar/{split}") if f.endswith('.txt')]

# Plot the data
average = True
load_files = get_load_files("train")
analyze_and_plot_daily(load_files, datatype="Load", average=average)
analyze_and_plot_weekly(load_files, datatype="Load", average=average)
analyze_and_plot_monthly(load_files, datatype="Load")

solar_files = get_solar_files("train")
analyze_and_plot_daily(solar_files, datatype="Solar", average=average)
analyze_and_plot_weekly(solar_files, datatype="Solar", average=average)
analyze_and_plot_monthly(solar_files, datatype="Solar")