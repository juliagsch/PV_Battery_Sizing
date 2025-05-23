"""
Given EV data, EUE target, load files and a solar trace, this script uses the model and the 
simulator to predict sizing. This can be used to compare performance for specific traces.
"""
import joblib
import numpy as np
import os
import pandas as pd
import random
import subprocess
import torch

from dataclasses import dataclass
from train_ev_fourier import MLP_Branched

model_name = "CNN_MLP_fourier_2x68000_256"
home_path = "."
scratch_path = "./dataset"

column_names = ['17521', '17522', '17523', '17524', '17525', '17526', '17527',
'policy_arrival_limit', 'policy_lbn_limit', 'policy_safe_arrival',
'policy_safe_departure']

@dataclass
class EV:
    op: str
    num_commute_trips: int
    num_non_commute_trips: int
    avg_commute_distance: float
    avg_non_commute_distance: float
    battery_size_kwh: float
    min_charge_kwh: float

""" Get all load files in specified directories. """
def get_load_files():
    files = []
    for type in ["scaled"]:
        files = files + [f"./data/test/load/{type}/{f}" for f in os.listdir(f"./data/test/load/{type}") if f.endswith('.txt')]
    return files

""" Estract 128 FFT features from solar and load traces and concatenate them."""
def extract_fft_features(solar, load, k=128):
    fft_mag = np.abs(np.fft.rfft(solar))
    top_k_solar = np.sort(fft_mag)[-k:]
    fft_mag = np.abs(np.fft.rfft(load))
    top_k_load = np.sort(fft_mag)[-k:]
    return np.concatenate([top_k_solar, top_k_load])

""" Use model to predict optimal sizing."""
def run_model(ev, load_file, solar_file, eue_target):
    # Get solar and load traces
    with open(solar_file, 'r') as file:
        solar_trace = [float(line.strip()) for line in file]
    
    with open(load_file, 'r') as file:
        load_trace = [float(line.strip()) for line in file]
    
    print("Energy consumption", np.sum(load_trace))

    # Get metadata
    op = [ev.op in name for name in column_names][-4:]
    metadata = [ev.num_commute_trips, ev.num_non_commute_trips, ev.avg_commute_distance, ev.avg_non_commute_distance, ev.battery_size_kwh, ev.min_charge_kwh, eue_target] + op
    meta = pd.DataFrame([metadata], columns=column_names)

    traces = extract_fft_features(solar_trace, load_trace)

    # Scale traces and metadata
    scaler_ts= joblib.load(f"{home_path}/model/out/scaler_ts_{model_name}.pkl")
    scaler_meta= joblib.load(f"{home_path}/model/out/scaler_meta_{model_name}.pkl")

    traces = scaler_ts.transform([traces])
    meta = scaler_meta.transform(meta)

    X_test_tensor = torch.tensor(traces, dtype=torch.float32)
    M_test_tensor = torch.tensor(meta, dtype=torch.float32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MLP_Branched(ts_input_len=traces.shape[1], meta_input_len=meta.shape[1]).to(device)

    # Load the saved weights
    model.load_state_dict(torch.load(f"{home_path}/model/out/{model_name}_best.pth", map_location=device))

    model.eval()
    with torch.no_grad():
        test_predictions = model(X_test_tensor.to(device), M_test_tensor.to(device)).cpu().numpy()
    
    print("Model got: Battery ", test_predictions[0][0], " Solar: ", test_predictions[0][1])
    
""" Use simulator to compute optimal sizing."""
def run_sim(ev, load_file, solar_file, eue_target):
    # Create EV trace
    ev_path = f"{home_path}/temp.csv"
    wfh_days = random.sample([0,1,2,3,4], 5-ev.num_commute_trips)

    schedule = [0 for _ in range(5)]
    for wfh_day in wfh_days:
        schedule[wfh_day] = 1
    ev_trace = f"python {home_path}/data/ev/ev_simulation.py --output {ev_path} --days 365 --ev_battery {ev.battery_size_kwh} --max_soc 0.8 --min_soc 0.2 --consumption 164 --wfh_monday {schedule[0]} --wfh_tuesday {schedule[1]} --wfh_wednesday {schedule[2]} --wfh_thursday {schedule[3]}  --wfh_friday {schedule[4]} --C_dist {ev.avg_commute_distance} --C_dept 8.00 --C_arr 18.00 --N_nc {ev.num_non_commute_trips} --Nc_dist {ev.avg_non_commute_distance}"
    _ = subprocess.run(ev_trace.split(), stdout=subprocess.PIPE, text=True)
    
    # Run simulation
    command = f"{home_path}/sim 1250 460 70 225 1 {eue_target} 0.9 365 {load_file} {solar_file} 0.8 0.2 {ev.battery_size_kwh} 7.4 {ev.op} {ev_path} {ev.min_charge_kwh}"
    result = subprocess.run(command.split(), stdout=subprocess.PIPE, text=True)
    result = result.stdout.split("\t")
    battery, solar = float(result[0]), float(result[1])

    print("Simulator got: Battery ", battery, " Solar: ", solar)

if __name__ == "__main__":
    # Specify load and solar trace
    load_files = get_load_files()
    solar_file = "./data/solar/united/46.95746586463173_2.907728315231907.txt"
    
    # Specify EV metadata
    policies = ["safe_arrival", "safe_departure", "arrival_limit", "lbn_limit"]
    ev = EV(
        op="safe_departure",
        num_commute_trips=2,
        num_non_commute_trips=2,
        avg_commute_distance=40,
        avg_non_commute_distance=10,
        battery_size_kwh=70,
        min_charge_kwh=30
    )

    # Specify level of self-sufficiency in range 0.0-0.9 where 0.0 corresponds to 100% self-sufficiency
    eue_target = 0.3
    
    # Start model inference and simulations
    for load_file in load_files:
        print("Processing ",load_file)
        run_model(ev, load_file, solar_file, eue_target)
        run_sim(ev, load_file, solar_file, eue_target)
        

