import os
import subprocess
import csv
import time
import multiprocessing
import random

self_sufficiency_target = 1.0 # Specify what percentage of load should be covered by solar production. [Range: 0.1-1.0]
confidence = 0.9 # Specify at what confidence the self sufficiency target should be met. [Range: 0.1-1.0]

eue_target = 1.0 - self_sufficiency_target

def get_solar_files():
    return [f"./data/solar/processed_all/{f}" for f in os.listdir("./data/solar/processed_all") if f.endswith('.txt')] + [f"./data/solar/noisy/{f}" for f in os.listdir("./data/solar/noisy") if f.endswith('.txt')]


def get_load_files():
    files = []
    for type in ["Detached", "EV", "NoLCT", "Semi-detached", "Terraced", "Noisy"]:
        files = files + [f"./data/load/processed_poster/{type}/{f}" for f in os.listdir(f"./data/load/processed_poster/{type}") if f.endswith('.txt')]
    return files

def get_files(filepath):
    try:
        with open(filepath, 'r') as f:
            data_list = [line.strip() for line in f]
        return data_list
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        return []

def process_pair_train(args):
    round_num, idx, solar_file, load_file = args

    try:
        command = f"./sim_huang 1250 460 70 225 1 {eue_target} {confidence} 365 {load_file} {solar_file}"
        result = subprocess.run(command.split(), stdout=subprocess.PIPE, text=True)
        result = result.stdout.split("\t")
        battery, solar = result[0], result[1]

        with open(solar_file, 'r') as file:
            solar_trace = [float(line.strip()) for line in file]
        
        with open(load_file, 'r') as file:
            load_trace = [float(line.strip()) for line in file]

        line = solar_trace + load_trace + [battery, solar]

        with open("dataset_paper_train.csv", 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(line)
        with open("files_processed_paper_train.csv", 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([solar_file])
            writer.writerow([load_file])
        return True
    except Exception as e:
        print(f"Error processing {solar_file} and {load_file} in round {round_num}: {e}")
        return False
    
def process_pair_test(args):
    round_num, idx, solar_file, load_file = args

    try:
        with open(solar_file, 'r') as file:
            solar_trace = [float(line.strip()) for line in file]

        command = f"./sim_huang 1250 460 70 225 1 {eue_target} {confidence} 365 {load_file} {solar_file}"
        result = subprocess.run(command.split(), stdout=subprocess.PIPE, text=True)
        result = result.stdout.split("\t")
        battery, solar = result[0], result[1]

        with open(load_file, 'r') as file:
            load_trace = [float(line.strip()) for line in file]

        line = solar_trace + load_trace + [battery, solar]

        # Append as a new row to the CSV file
        with open("dataset_paper_test.csv", 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(line)
        with open("files_processed_paper_test.csv", 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([solar_file])
            writer.writerow([load_file])
        return True
    except Exception as e:
        print(f"Error processing {solar_file} and {load_file} in round {round_num}: {e}")
        return False

if __name__ == "__main__":
    train_solar_filepath = "./dataset/train_solar.txt"
    test_solar_filepath = "./dataset/test_solar.txt"
    train_load_filepath = "./dataset/train_load.txt"
    test_load_filepath = "./dataset/test_load.txt"

    test_load = get_files(test_load_filepath)
    test_solar = get_files(test_solar_filepath)
    train_load = get_files(train_load_filepath)
    train_solar = get_files(train_solar_filepath)

    num_processes = multiprocessing.cpu_count()  # Get the number of available CPU cores
    print(f"Using {num_processes} processes for parallel execution.")

    for round_num in range(3):
        random.shuffle(train_load)
        random.shuffle(train_solar)

        tasks = []
        for idx, solar_file in enumerate(train_solar):
            load_file = train_load[idx]
            tasks.append((round_num, idx, solar_file, load_file))

        with multiprocessing.Pool(processes=num_processes) as pool:
            results = pool.map(process_pair_train, tasks)

        print(f"Round {round_num + 1} completed.")
    
    for round_num in range(3):
        random.shuffle(test_load)
        random.shuffle(test_solar)

        tasks = []
        for idx, solar_file in enumerate(test_solar):
            load_file = test_load[idx]
            tasks.append((round_num, idx, solar_file, load_file))

        with multiprocessing.Pool(processes=num_processes) as pool:
            results = pool.map(process_pair_test, tasks)

        print(f"Round {round_num + 1} completed.")

    print("All rounds completed.")