"""
Remove duplicate solar traces fetched from PVWatts and files with length not equal to 8760.
"""
import os
import hashlib

def hash_file(filepath):
    with open(filepath, 'rb') as f:
        return hashlib.sha256(f.read()).hexdigest()

def has_8760_values(filepath):
    with open(filepath, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]
    return len(lines) == 8760

def remove_duplicate_txt_files(folder_path):
    seen_hashes = {}
    removed_files = 0
    removed_invalid_length = 0

    for filename in os.listdir(folder_path):
        if not filename.endswith('.txt'):
            continue

        filepath = os.path.join(folder_path, filename)

        if not has_8760_values(filepath):
            os.remove(filepath)
            removed_invalid_length += 1
            print(f"Removed invalid length file: {filename}")
            continue

        file_hash = hash_file(filepath)

        if file_hash in seen_hashes:
            os.remove(filepath)
            removed_files += 1
            print(f"Removed duplicate: {filename}")
        else:
            seen_hashes[file_hash] = filepath

    print(f"Removed {removed_invalid_length} files with invalid length.")
    print(f"Removed {removed_files} duplicate files.")

remove_duplicate_txt_files('./data/solar/pvwatts_original')
