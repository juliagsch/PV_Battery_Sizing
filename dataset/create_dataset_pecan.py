"""
Create labeled dataset with Pecan Street load traces for testing.
"""
import csv
import multiprocessing
import os
import random
import subprocess

from dataclasses import dataclass

base_path = "/cluster/home/jgschwind/PV_Battery_Sizing_EV"
out_path = "/cluster/scratch/jgschwind"

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
    eue_target = random.randint(0,80)/100.0

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

            ev_data = [op, ev.commute_mon, ev.commute_tue, ev.commute_wed, ev.commute_thu, ev.commute_fri, ev.num_non_commute_trips, ev.avg_commute_distance, ev.avg_non_commute_distance, ev.battery_size_kwh, ev.min_charge_kwh]
            line = solar_trace + load_trace + ev_data + [eue_target, battery, solar]

            with open(f"{out_path}/dataset_{split}_pecan.csv", 'a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(line)
            with open(f"{out_path}/files_processed_{split}_pecan.csv", 'a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([solar_file])
                writer.writerow([load_file])
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
    # After trip, between 10% and 30% of battery should be left.
    ev.min_charge_kwh = min(
        float(max_distance) * float(ev_consumption) + float(ev.battery_size_kwh) * 0.2,
        float(ev.battery_size_kwh) * 0.8
    )
    return ev


def get_test_load_files(types):
    files = []
    for type in types:
        files = files + [f"{base_path}/data/test/load/{type}/15min/{f}" for f in os.listdir(f"{base_path}/data/test/load/{type}/15min") if f.endswith('.txt')]
    return files

if __name__ == "__main__":
    num_runs = 10

    # Load and solar traces
    test_solar_filepath = base_path + "/dataset/solar_test.txt"
    test_solar = get_files(test_solar_filepath)
    test_load_cali = get_test_load_files(["california"]) # Original pecan street load for 25 buildings in california
    test_load_scaled = get_test_load_files(["scaled"]) # The same load curves but the yearly consumption is scaled down to 2800 kWh

    ev_files = get_ev_files()

    num_processes = multiprocessing.cpu_count()  # Get the number of available CPU cores
    print(f"Using {num_processes} processes for parallel execution.")

    policies = ["safe_arrival", "safe_departure", "arrival_limit", "bidirectional"]

    for round_num in range(num_runs):
        tasks = []
        random.shuffle(test_load_cali)
        random.shuffle(test_solar)
        random.shuffle(ev_files)

        for idx, load_file in enumerate(test_load_cali):

            solar_file = test_solar[idx]
            ev_file = ev_files[idx]
            ev = get_ev_metadata(ev_file)

            for op in policies:
                tasks.append((ev_file, solar_file, load_file, op, ev, "cali"))

        random.shuffle(test_load_scaled)

        for idx, load_file in enumerate(test_load_scaled):

            solar_file = test_solar[idx]
            ev_file = ev_files[idx]
            ev = get_ev_metadata(ev_file)

            for op in policies:
                tasks.append((ev_file, solar_file, load_file, op, ev, "scaled"))

        with multiprocessing.Pool(processes=num_processes) as pool:
            results = pool.map(process_pair, tasks)

        print(f"Round {round_num + 1} completed.")