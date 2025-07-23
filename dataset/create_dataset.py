"""
Create dataset for training, validation and testing.
"""
import os
import subprocess
import csv
import multiprocessing
import random
from dataclasses import dataclass

base_path, out_path = ".", "."

@dataclass
class EV:
    commute_mon: bool
    commute_tue: bool
    commute_wed: bool
    commute_thu: bool
    commute_fri: bool
    num_non_commute_trips: int
    avg_commute_distance: float
    avg_non_commute_distance: float
    battery_size_kwh: float
    min_charge_kwh: float

def get_ev_files():
    return [f"{base_path}/data/ev/ev_traces/{f}" for f in os.listdir(f"{base_path}/data/ev/ev_traces") if f.endswith('.csv')]

def get_files(path):
    return [f"{path}/{f}" for f in os.listdir(path) if f.endswith('.txt')]

def process_pair(args):
    ev_path, solar_file, load_file, op, ev, split = args
    eue_target = random.randint(10,80)/100.0

    try:
        if op == "no_ev":
            command = f"{base_path}/sim_noEV 1250 460 20 30 1 {eue_target} 0.9 365 {load_file} {solar_file}"
            result = subprocess.run(command.split(), stdout=subprocess.PIPE, text=True)
            result = result.stdout.split("\t")
            battery, solar = result[0], result[1]

            # Exclude sizings where the simulator was not able to find an optimum and returned inf
            if float(battery) > 30 or float(solar) > 20:
                return True

            with open(solar_file, 'r') as file:
                solar_trace = [float(line.strip()) for line in file]
            
            with open(load_file, 'r') as file:
                load_trace = [float(line.strip()) for line in file]

            ev_data = [op, False, False, False, False, False, 0, 0, 0, 0, 0]
            line = solar_trace + load_trace + ev_data + [eue_target, battery, solar]
        else:
            command = f"{base_path}/sim_EV 1250 460 20 30 1 {eue_target} 0.9 365 {load_file} {solar_file} 0.8 0.2 {ev.battery_size_kwh} 7.4 {op} {ev_path} {ev.min_charge_kwh}"
            result = subprocess.run(command.split(), stdout=subprocess.PIPE, text=True)
            result = result.stdout.split("\t")
            battery, solar = result[0], result[1]

            # Exclude sizings where the simulator was not able to find an optimum and returned inf
            if float(battery) > 30 or float(solar) > 20:
                return True
            
            with open(solar_file, 'r') as file:
                solar_trace = [float(line.strip()) for line in file]
            
            with open(load_file, 'r') as file:
                load_trace = [float(line.strip()) for line in file]

            ev_data = [op, ev.commute_mon, ev.commute_tue, ev.commute_wed, ev.commute_thu, ev.commute_fri, ev.num_non_commute_trips, ev.avg_commute_distance, ev.avg_non_commute_distance, ev.battery_size_kwh, ev.min_charge_kwh]
            line = solar_trace + load_trace + ev_data + [eue_target, battery, solar]

        with open(f"{out_path}/dataset_{split}.csv", 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(line)
        # Load and solar files are tracked for further analysis
        with open(f"{out_path}/files_processed_{split}.csv", 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([solar_file, load_file])
        return True
    except Exception as e:
        print(f"Error processing {solar_file} and {load_file}: {e}")
        return False

def get_ev_metadata(file):
    ev_consumption = 0.164 #kWh/km
    filename = os.path.basename(file)
    metadata = filename.split("_")

    ev = EV(
        commute_mon=not bool(int(metadata[0])),
        commute_tue=not bool(int(metadata[1])),
        commute_wed=not bool(int(metadata[2])),
        commute_thu=not bool(int(metadata[3])),
        commute_fri=not bool(int(metadata[4])),
        num_non_commute_trips=int(metadata[5]),
        avg_commute_distance=float(metadata[6]),
        avg_non_commute_distance=float(metadata[7]),
        battery_size_kwh=int(metadata[8]),
        min_charge_kwh=0.0
    )
    # Set min charge based on maximum expected distance
    # Should use 95th percentile instead of max for data that is more noisy.
    no_commute = not (ev.commute_mon or ev.commute_tue or ev.commute_wed or ev.commute_thu or ev.commute_fri)
    ev.avg_commute_distance = 0.0 if no_commute else ev.avg_commute_distance
    ev.avg_non_commute_distance = 0.0 if ev.num_non_commute_trips == 0 else ev.avg_non_commute_distance

    max_distance = max(ev.avg_commute_distance, ev.avg_non_commute_distance)
    # After trip, between 20% of battery should be left.
    ev.min_charge_kwh = min(
        float(max_distance) * float(ev_consumption) + float(ev.battery_size_kwh) * 0.2,
        float(ev.battery_size_kwh) * 0.8
    )
    return ev

if __name__ == "__main__":
    # Define number of combinations between load and solar traces
    num_runs = 3

    # Loading solar and load files. The load files can be replaced with ALPG or Pecan Street data to create additional test datasets.
    test_load = get_files(f"{base_path}/data/load/test")
    train_load = get_files(f"{base_path}/data/load/train")
    val_load = get_files(f"{base_path}/data/load/val")

    test_solar = get_files(f"{base_path}/data/solar/test")
    train_solar = get_files(f"{base_path}/data/solar/train")
    val_solar = get_files(f"{base_path}/data/solar/val")

    ev_files = get_ev_files()

    num_processes = multiprocessing.cpu_count()
    print(f"Using {num_processes} processes for parallel execution.")

    policies = ["safe_arrival", "safe_departure", "arrival_limit", "bidirectional", "no_ev"]

    for round_num in range(num_runs):
        tasks = []
        random.shuffle(test_load)
        random.shuffle(val_load)
        random.shuffle(train_load)

        # Test
        for idx, load_file in enumerate(test_load):
            solar_file =random.choice(test_solar)
            ev_file = random.choice(ev_files)
            ev = get_ev_metadata(ev_file)
            for op in policies:
                tasks.append((ev_file, solar_file, load_file, op, ev, "test"))

        # Val
        for idx, load_file in enumerate(val_load):
            solar_file =random.choice(val_solar)
            ev_file = random.choice(ev_files)
            ev = get_ev_metadata(ev_file)
            for op in policies:
                tasks.append((ev_file, solar_file, load_file, op, ev, "val"))

        # Train
        for idx, load_file in enumerate(train_load):
            solar_file =random.choice(train_solar)
            ev_file = random.choice(ev_files)
            ev = get_ev_metadata(ev_file)
            for op in policies:
                tasks.append((ev_file, solar_file, load_file, op, ev, "train"))

        with multiprocessing.Pool(processes=num_processes) as pool:
            results = pool.map(process_pair, tasks)

        print(f"Round {round_num + 1} completed.")