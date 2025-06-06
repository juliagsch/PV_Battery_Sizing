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
from train_ev_fourier import MLP_Branched, extract_fft_features

model_name = "CNN_MLP_fourier_244000_bi_8000"
policy = "safe_departure"
home_path = "."
scratch_path = "./dataset"

column_names = ['17521', '17522', '17523', '17524', '17525', '17526', '17527', '17528', '17529', '17530', '17531',
"policy_arrival_limit", "policy_bidirectional", "policy_safe_arrival", "policy_safe_departure"]

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

""" Get all load files in specified directories. """
def get_load_files():
    files = []
    for type in ["scaled"]:
        files = files + [f"./data/test/load/{type}/15min/{f}" for f in os.listdir(f"./data/test/load/{type}/15min") if f.endswith('.txt')]
    return files

""" Use model to predict optimal sizing."""
def run_model(ev, load_file, solar_file, eue_target):
    # Get solar and load traces
    with open(solar_file, 'r') as file:
        solar_trace = [float(line.strip()) for line in file]
    
    with open(load_file, 'r') as file:
        load_trace = [float(line.strip()) for line in file]
    
    print("Energy consumption", np.sum(load_trace))

    # Get metadata
    op = [policy in name for name in column_names][-4:]
    metadata = [ev.commute_mon, ev.commute_tue, ev.commute_wed, ev.commute_thu, ev.commute_fri, ev.num_non_commute_trips, ev.avg_commute_distance, ev.avg_non_commute_distance, ev.battery_size_kwh, ev.min_charge_kwh, eue_target] + op
    meta = pd.DataFrame([metadata], columns=column_names)
    data_in = np.array([np.concatenate([np.array(solar_trace), np.array(load_trace)])])
    traces = extract_fft_features(data_in)

    # Scale traces and metadata
    scaler_ts= joblib.load(f"{home_path}/model/out/scaler_ts_{model_name}.pkl")
    scaler_meta= joblib.load(f"{home_path}/model/out/scaler_meta_{model_name}.pkl")

    traces = scaler_ts.transform(traces)
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
def run_sim(ev, load_file, solar_file, eue_target, ev_path):
    # Run simulation
    command = f"{home_path}/sim 1250 460 70 225 1 {eue_target} 0.9 365 {load_file} {solar_file} 0.8 0.2 {ev.battery_size_kwh} 7.4 {policy} {ev_path} {ev.min_charge_kwh}"
    result = subprocess.run(command.split(), stdout=subprocess.PIPE, text=True)
    result = result.stdout.split("\t")
    battery, solar = float(result[0]), float(result[1])

    print("Simulator got: Battery ", battery, " Solar: ", solar)

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

if __name__ == "__main__":
    # Specify load, EV and solar trace
    load_files = get_load_files()
    solar_file = "./data/solar/pvwatts_original/-2.104938501708162_40.06960185922972.txt"
    ev_file = "1_0_0_1_0_7_60_27_109_.csv"
    ev_path = f"./data/ev/ev_traces/{ev_file}"
    ev = get_ev_metadata(ev_file)

    # Specify EV metadata
    policies = ["safe_arrival", "safe_departure", "arrival_limit", "bidirectional"]

    # Specify level of self-sufficiency in range 0.0-0.9 where 0.0 corresponds to 100% self-sufficiency
    eue_target = 0.3
    
    # Start model inference and simulations
    for load_file in load_files:
        print("Processing ",load_file)
        run_model(ev, load_file, solar_file, eue_target)
        run_sim(ev, load_file, solar_file, eue_target, ev_path)

