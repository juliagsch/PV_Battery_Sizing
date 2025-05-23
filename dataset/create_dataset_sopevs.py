import os
import subprocess
import csv
import time
import multiprocessing
import random
from dataclasses import dataclass

@dataclass
class EV:
    num_commute_trips: int
    num_non_commute_trips: int
    avg_commute_distance: float
    avg_non_commute_distance: float
    battery_size_kwh: float
    min_charge_kwh: float

base_path = "/cluster/home/jgschwind/PV_Battery_Sizing"
out_path = "/cluster/scratch/jgschwind"
def get_files(filepath):
    try:
        with open(filepath, 'r') as f:
            data_list = [base_path + line.strip()[1:] for line in f]
        return data_list
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        return []

def process_pair(args):
    ev_path, solar_file, load_file, op, ev, split = args
    eue_target = random.randint(0,90)/100.0

    try:
        command = f"{base_path}/sim 1250 460 70 225 1 {eue_target} 0.9 365 {load_file} {solar_file} 0.8 0.2 {ev.battery_size_kwh} 7.4 {op} {ev_path} {ev.min_charge_kwh}"
        result = subprocess.run(command.split(), stdout=subprocess.PIPE, text=True)
        result = result.stdout.split("\t")
        battery, solar = result[0], result[1]
        if float(battery) < 30 and float(solar) < 20:
            with open(solar_file, 'r') as file:
                solar_trace = [float(line.strip()) for line in file]
            
            with open(load_file, 'r') as file:
                load_trace = [float(line.strip()) for line in file]

            ev_data = [op, ev.num_commute_trips, ev.num_non_commute_trips, ev.avg_commute_distance, ev.avg_non_commute_distance, ev.battery_size_kwh, ev.min_charge_kwh]
            line = solar_trace + load_trace + ev_data + [eue_target, battery, solar]

            with open(f"{out_path}/dataset_{split}_eveue_large_64.csv", 'a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(line)
            with open(f"{out_path}/files_processed_{split}_eveue_large_64.csv", 'a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([solar_file])
                writer.writerow([load_file])
        return True
    except Exception as e:
        print(f"Error processing {solar_file} and {load_file}: {e}")
        return False


if __name__ == "__main__":
    num_runs = 9
    ev_consumption = 0.164 #kWh/km
    train_solar_filepath = base_path + "/dataset/train_solar.txt"
    test_solar_filepath = base_path + "/dataset/test_solar.txt"
    train_load_filepath = base_path + "/dataset/train_load.txt"
    test_load_filepath = base_path + "/dataset/test_load.txt"

    id = random.randint(0,9999999999999)
    os.mkdir(f"{base_path}/data/ev/train/{id}")

    test_load = get_files(test_load_filepath)
    test_solar = get_files(test_solar_filepath)
    train_load = get_files(train_load_filepath)
    train_solar = get_files(train_solar_filepath)

    # Remove data that already includes EV
    train_load = [f for f in train_load if "EV" not in f]
    test_load = [f for f in test_load if "EV" not in f]

    num_processes = 64 #multiprocessing.cpu_count()  # Get the number of available CPU cores
    policies = ["safe_arrival", "safe_departure", "arrival_limit", "lbn_limit"]

    print(f"Using {num_processes} processes for parallel execution.")

    for round_num in range(num_runs):
        random.shuffle(train_load)
        random.shuffle(train_solar)

        tasks = []
        for idx, load_file in enumerate(train_load):

            solar_file = train_solar[idx]
            ev = EV(
                num_commute_trips=random.randint(0,5),
                num_non_commute_trips=random.randint(0,7),
                avg_commute_distance=random.randint(10, 110),
                avg_non_commute_distance=random.randint(10,20),
                battery_size_kwh=random.randint(50,100),
                min_charge_kwh=0.0
            )
            # Set min charge based on maximum expected distance
            max_distance = max(ev.avg_commute_distance, ev.avg_non_commute_distance)
            ev.min_charge_kwh = max_distance * ev_consumption + ev.battery_size_kwh * 0.2

            ev_path = f"{base_path}/data/ev/train/{id}/{idx}.csv"
            wfh_days = random.sample([0,1,2,3,4], 5-ev.num_commute_trips)

            schedule = [0 for _ in range(5)]
            for wfh_day in wfh_days:
                schedule[wfh_day] = 1
            ev_trace = f"python {base_path}/data/ev/ev_simulation.py --output {ev_path} --days 365 --ev_battery {ev.battery_size_kwh} --max_soc 0.8 --min_soc 0.2 --consumption 164 --wfh_monday {schedule[0]} --wfh_tuesday {schedule[1]} --wfh_wednesday {schedule[2]} --wfh_thursday {schedule[3]}  --wfh_friday {schedule[4]} --C_dist {ev.avg_commute_distance} --C_dept 8.00 --C_arr 18.00 --N_nc {ev.num_non_commute_trips} --Nc_dist {ev.avg_non_commute_distance}"
            _ = subprocess.run(ev_trace.split(), stdout=subprocess.PIPE, text=True)

            for op in policies:
                tasks.append((ev_path, solar_file, load_file, op, ev, "train"))

        with multiprocessing.Pool(processes=num_processes) as pool:
            results = pool.map(process_pair, tasks)

        print(f"Round {round_num + 1} completed.")

    for round_num in range(num_runs):
        random.shuffle(test_load)
        random.shuffle(test_solar)

        tasks = []
        for idx, load_file in enumerate(test_load):

            solar_file = test_solar[idx]
            ev = EV(
                num_commute_trips=random.randint(0,5),
                num_non_commute_trips=random.randint(0,7),
                avg_commute_distance=random.randint(10, 100),
                avg_non_commute_distance=random.randint(10,20),
                battery_size_kwh=random.randint(50,100),
                min_charge_kwh=0.0
            )
            # Set min charge based on maximum expected distance
            max_distance = max(ev.avg_commute_distance, ev.avg_non_commute_distance)
            ev.min_charge_kwh = max_distance * ev_consumption + ev.battery_size_kwh * 0.2

            ev_path = f"{base_path}/data/ev/test/{idx}.csv"
            wfh_days = random.sample([0,1,2,3,4], 5-ev.num_commute_trips)

            schedule = [0 for _ in range(5)]
            for wfh_day in wfh_days:
                schedule[wfh_day] = 1
            ev_trace = f"python {base_path}/data/ev/ev_simulation.py --output {ev_path} --days 365 --ev_battery {ev.battery_size_kwh} --max_soc 0.8 --min_soc 0.2 --consumption 164 --wfh_monday {schedule[0]} --wfh_tuesday {schedule[1]} --wfh_wednesday {schedule[2]} --wfh_thursday {schedule[3]}  --wfh_friday {schedule[4]} --C_dist {ev.avg_commute_distance} --C_dept 8.00 --C_arr 18.00 --N_nc {ev.num_non_commute_trips} --Nc_dist {ev.avg_non_commute_distance}"
            _ = subprocess.run(ev_trace.split(), stdout=subprocess.PIPE, text=True)

            for op in policies:
                tasks.append((ev_path, solar_file, load_file, op, ev, "test"))

        with multiprocessing.Pool(processes=num_processes) as pool:
            results = pool.map(process_pair, tasks)

        print(f"Round {round_num + 1} completed.")