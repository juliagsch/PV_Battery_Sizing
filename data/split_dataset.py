""" Splits the data into train, test and validation using a 80/10/10 split."""

import os
import shutil
import random

for trace in ["load", "solar"]:
    original_dir = f"./data/{trace}/original"
    train_dir = f"./data/{trace}/train"
    val_dir = f"./data/{trace}/val"
    test_dir = f"./data/{trace}/test"

    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    files = [f for f in os.listdir(original_dir) if os.path.isfile(os.path.join(original_dir, f))]
    random.shuffle(files)

    # Get number of files per split
    n_total = len(files)
    n_train = int(0.8 * n_total)
    n_val = int(0.1 * n_total)
    n_test = n_total - n_train - n_val

    print(f"Total {trace} files: {n_total}, Train: {n_train}, Validation: {n_val}, Test: {n_test}")

    # Split files
    train_files = files[:n_train]
    val_files = files[n_train:n_train + n_val]
    test_files = files[n_train + n_val:]

    for f in train_files:
        shutil.copy(os.path.join(original_dir, f), os.path.join(train_dir, f))

    for f in val_files:
        shutil.copy(os.path.join(original_dir, f), os.path.join(val_dir, f))

    for f in test_files:
        shutil.copy(os.path.join(original_dir, f), os.path.join(test_dir, f))

