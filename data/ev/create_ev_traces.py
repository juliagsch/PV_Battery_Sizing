"""
Create EV traces using SPAGHETTI by Berkes et. al. https://doi.org/10.1186/s42162-024-00314-6
"""
import os
import random
import subprocess

from dataclasses import dataclass

base_path = "."
ev_consumption = 0.164 #kWh/km

@dataclass
class EV:
    num_commute_trips: int
    num_non_commute_trips: int
    avg_commute_distance: float
    avg_non_commute_distance: float
    battery_size_kwh: float

if __name__ == "__main__":
    num_samples = 8000
    file_names = []
    
    os.makedirs("./data/ev/out", exist_ok=True)
    while len(file_names) < num_samples:
        ev = EV(
            num_commute_trips=random.randint(0,5), # per week
            num_non_commute_trips=random.randint(0,7), # per week
            avg_commute_distance=random.randint(5, 120), # return in km
            avg_non_commute_distance=random.randint(5,40), # return in km, Average: 12.9 km https://www.bfs.admin.ch/bfs/en/home/statistics/mobility-transport/passenger-transport.html
            battery_size_kwh=random.randint(21,118), # Source: https://ev-database.org/uk/cheatsheet/useable-battery-capacity-electric-car
        )

        # Set average distance to zero if there are no trips.
        ev.avg_commute_distance = 0 if ev.num_commute_trips == 0 else ev.avg_commute_distance
        ev.avg_non_commute_distance = 0 if ev.num_non_commute_trips == 0 else ev.avg_non_commute_distance

        wfh_days = random.sample([0,1,2,3,4], 5-ev.num_commute_trips)

        schedule = [0 for _ in range(5)]
        for wfh_day in wfh_days:
            schedule[wfh_day] = 1

        file_name = f"{'_'.join(str(item) for item in schedule)}_{ev.num_non_commute_trips}_{ev.avg_commute_distance}_{ev.avg_non_commute_distance}_{ev.battery_size_kwh}"
        
        # Avoid duplicates
        if file_name in file_names:
            continue

        ev_path = f"{base_path}/data/ev/out/{file_name}_.csv"

        ev_trace = f"python {base_path}/data/ev/ev_simulation.py --output {ev_path} --days 365 --ev_battery {ev.battery_size_kwh} --max_soc 0.8 --min_soc 0.2 --consumption 164 --wfh_monday {schedule[0]} --wfh_tuesday {schedule[1]} --wfh_wednesday {schedule[2]} --wfh_thursday {schedule[3]}  --wfh_friday {schedule[4]} --C_dist {ev.avg_commute_distance} --C_dept 8.00 --C_arr 18.00 --N_nc {ev.num_non_commute_trips} --Nc_dist {ev.avg_non_commute_distance}"
        _ = subprocess.run(ev_trace.split(), stdout=subprocess.PIPE, text=True)
        file_names.append(file_name)
