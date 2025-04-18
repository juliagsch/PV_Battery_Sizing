"""Can be used to merge datasets. 
Set interleave to true if solar and load should be ordered chronologically (solar1,load1,solar2,load2,...,battery,pv)
If interleave is set to false: (solar1, solar2,...,load1,load2,...,battery,pv)
"""
import csv

source_files = ["dataset_paper_test_actual.csv"]
destination_file = "dataset_paper_test_interleaved.csv"
interleave = True

num_rows = 0

for source_file in source_files:
    with open(source_file, 'r', newline='') as src, open(destination_file, 'a', newline='') as dest:
        reader = csv.reader(src)
        writer = csv.writer(dest)

        for row in reader:
            if 'inf' in row or 'naN' in row:
                continue
            if interleave:
                # Extract the last two rows
                first_half = row[:8760]   # First 8760 entries
                second_half = row[8760:17520]  # Next 8760 entries
                last_two = row[17520:]    # Last 2 entries

                # Create an interleaved list
                row = [None] * 17520  # Pre-allocate list for efficiency
                
                row[::2] = first_half  # Place first half at even indices (0,2,4...)
                row[1::2] = second_half  # Place second half at odd indices (1,3,5...)
                
                # Append the last two rows
                row.extend(last_two)

            writer.writerow(row)
            num_rows += 1

print(f"Appended {num_rows} lines from {source_files} to {destination_file}.")
