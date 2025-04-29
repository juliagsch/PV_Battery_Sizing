import os
import subprocess
import csv
import multiprocessing
import random
from dataclasses import dataclass
from add_ev_load import get_load_trace

base_path = "."
base_path = "/cluster/home/jgschwind/PV_Battery_Sizing"
num_processes = 16

@dataclass
class EV:
    num_commute_trips: int
    num_non_commute_trips: int
    avg_commute_distance: float
    avg_non_commute_distance: float
    battery_size_kwh: float
    min_charge_kwh: float

def get_files(filepath):
    try:
        with open(filepath, 'r') as f:
            data_list = [line.strip() for line in f]
        return data_list
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        return []

def process_pair(args):
    solar_file, load_file, split, op = args

    try:
        command = f"{base_path}/sim_brad 1250 460 70 225 0 0.15 0.9 365 {load_file} {solar_file}"
        result = subprocess.run(command.split(), stdout=subprocess.PIPE, text=True)
        result = result.stdout.split("\t")
        battery, solar = result[0], result[1]

        with open(solar_file, 'r') as file:
            solar_trace = [float(line.strip()) for line in file]
        
        with open(load_file, 'r') as file:
            load_trace = [float(line.strip()) for line in file]

        line = solar_trace + load_trace + [op, battery, solar]

        with open(f"dataset_{split}_static.csv", 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(line)
        with open(f"files_processed_{split}_static.csv", 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([solar_file])
            writer.writerow([load_file])
        return True
    except Exception as e:
        print(f"Error processing {solar_file} and {load_file}: {e}")
        return False


if __name__ == "__main__":
    num_runs = 1
    ev_consumption = 0.164 #kWh/km
    train_solar_filepath = f"{base_path}/dataset/train_solar.txt"
    test_solar_filepath = f"{base_path}/dataset/test_solar.txt"
    train_load_filepath = f"{base_path}/dataset/train_load.txt"
    test_load_filepath = f"{base_path}/dataset/test_load.txt"

    test_load = [base_path+i[1:] for i in get_files(test_load_filepath)]
    test_solar = [base_path+i[1:] for i in get_files(test_solar_filepath)]
    train_load = [base_path+i[1:] for i in get_files(train_load_filepath)]
    train_solar = [base_path+i[1:] for i in get_files(train_solar_filepath)]

    # Remove data that already includes EV
    train_load = [f for f in train_load if "EV" not in f]
    test_load = [f for f in test_load if "EV" not in f]

    # num_processes = multiprocessing.cpu_count()  # Get the number of available CPU cores
    print(f"Using {num_processes} processes for parallel execution.")

    """
    Charging Policies:
    safe: Fully charge EV once it arrives home.
    safe limit : Charge EV at arrival until user-specified battery level.
    solar: Charge EV with solar energy. If before departure, the user-specified minimum battery level is not reached, charge until that level.
    """
    policies = ["safe", "safe_limit", "solar"] 

    for split in ["train", "test"]:
        load, solar = [], []
        for round_num in range(num_runs):
            if split == "train":
                random.shuffle(train_load)
                random.shuffle(train_solar)
                load, solar = train_load, train_solar
            else:
                random.shuffle(test_load)
                random.shuffle(test_solar)
                load, solar = test_load, test_solar

            tasks = []
            for idx, load_file in enumerate(load):

                solar_file = solar[idx]
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

                ev_path = f"{base_path}/data/ev/{split}/{idx}.csv"
                wfh_days = random.sample([0,1,2,3,4], 5-ev.num_commute_trips)

                schedule = [0 for _ in range(5)]
                for wfh_day in wfh_days:
                    schedule[wfh_day] = 1
                ev_trace = f"python {base_path}/data/ev/ev_simulation.py --output {ev_path} --days 365 --ev_battery {ev.battery_size_kwh} --max_soc 0.8 --min_soc 0.2 --consumption 164 --wfh_monday {schedule[0]} --wfh_tuesday {schedule[1]} --wfh_wednesday {schedule[2]} --wfh_thursday {schedule[3]}  --wfh_friday {schedule[4]} --C_dist {ev.avg_commute_distance} --C_dept 8.00 --C_arr 18.00 --N_nc {ev.num_non_commute_trips} --Nc_dist {ev.avg_non_commute_distance}"
                _ = subprocess.run(ev_trace.split(), stdout=subprocess.PIPE, text=True)

                for op in policies:
                    load_filename = f"{base_path}/data/load/{split}/{op}/{idx}.csv"
                    get_load_trace(load_file, solar_file, ev_path, op, load_filename)
                    tasks.append((solar_file, load_filename, split, op))

            with multiprocessing.Pool(processes=num_processes) as pool:
                results = pool.map(process_pair, tasks)

            print(f"Round {round_num + 1} in {split} completed.")
