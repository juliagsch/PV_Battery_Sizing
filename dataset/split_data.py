import os
import random

def get_load_files():
    files = []
    for type in ["original", "averaged_month", "averaged_week", "noisy_month", "noisy_week"]:
        dir_path = f"./data/load/faraday_yearly/{type}"
        files += [os.path.join(dir_path, f) for f in os.listdir(dir_path) if f.endswith('.txt')]
    return files

def get_solar_files():
    return [f"./data/solar/pvwatts_original/{f}" for f in os.listdir("./data/solar/pvwatts_original") if f.endswith('.txt')] + [f"./data/solar/augmented/{f}" for f in os.listdir("./data/solar/augmented") if f.endswith('.txt')]

def split_and_save_file_lists(type, files):
    random.shuffle(files)  # Shuffle the list for randomness

    test_size = int(0.10 * len(files))
    test_files = files[:test_size]
    train_files = files[test_size:]

    # Write to txt files
    with open(f"./dataset/{type}_test.txt", "w") as f:
        for filename in test_files:
            f.write(f"{filename}\n")

    with open(f"./dataset/{type}_train.txt", "w") as f:
        for filename in train_files:
            f.write(f"{filename}\n")

split_and_save_file_lists("load", get_load_files())
split_and_save_file_lists("solar", get_solar_files())
