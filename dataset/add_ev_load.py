import numpy as np
from datetime import datetime

charging_efficiency = 0.9
eta_ev_c = 1.0/charging_efficiency
charging_rate = 7.4
num_hours = 8760
# base_path = "/cluster/home/jgschwind/PV_Battery_Sizing"
base_path = "."

def get_soc(file):
    soc = [0 for _ in range(num_hours)]
    with open(file, 'r') as file:
        ev_trace = [line.strip().split(',') for line in file][1:]
    max_battery = float(ev_trace[0][3])
    total_load = 0
    max_soc_diff = 0
    for trip in ev_trace:
        if 'No trips' in trip:
            continue
        soc_diff = float(trip[3])-float(trip[5])
        day = int(trip[0])-1
        departure_hour = datetime.strptime(trip[2], "%H:%M").hour
        arrival_hour = datetime.strptime(trip[4], "%H:%M").hour
        # If arrival hour is after midnight, add 24 hours
        if arrival_hour < departure_hour:
            arrival_hour += 24
        # -1 if car is away
        previous_soc = 0
        if not np.sum(soc[day*24+departure_hour:day*24+arrival_hour+1]) == 0:
            previous_soc = np.sum([i for i in soc[day*24+departure_hour:day*24+arrival_hour+1] if i>0])

        soc[day*24+departure_hour:day*24+arrival_hour+1] = [-1]*(arrival_hour-departure_hour+1)
        soc[day*24+arrival_hour+1] = soc_diff+previous_soc
        total_load += soc_diff
        if max_soc_diff<soc_diff:
            max_soc_diff = soc_diff
        if max_battery<float(trip[3]):
            max_battery = float(trip[3])
    return soc, max_battery, total_load, max_soc_diff

# Safe arrival: Fully charge EV once it arrives home. Pass max_battery as target battery level.
# Arrival limit: Charge EV until user-specified limit. Pass min_battery as target battery level.
def arrival_charge(hours, ev_b, target_battery):
    ev_load = [0]*hours
    if ev_b >= target_battery:
        return ev_load, ev_b
    
    for hour in range(hours):
        cur_charge = min(target_battery-ev_b, charging_rate)
        ev_load[hour] += cur_charge*eta_ev_c
        ev_b += cur_charge
    return ev_load, ev_b

# Augment load curve to ensure that we never charge more than charging rate in one timestep
def crop_charging_rate(ev_load, min_load):
    scalable = [1 if i>0 else 0 for i in ev_load]

    charging_limit = charging_rate*eta_ev_c
    while(True):
        current_load = np.sum(ev_load)
        # Crop load during each hour to charging limit.
        for idx, load in enumerate(ev_load):
            if load > charging_limit:
                current_load -= load-charging_limit
                ev_load[idx] = charging_limit
                scalable[idx] = 0
        # If EV is still charged above the minimum battery level return the ev load.
        if current_load >= min_load:
            return ev_load

        load_deficit = min_load - current_load
        # If the curve cannot be scaled anymore, the solar production is insufficient to reach min battery level, the EV is charged before departure
        if np.sum(scalable) == 0:
            for idx in reversed(range(len(ev_load))):
                ev_load[idx] += load_deficit
                if ev_load[idx] <= charging_limit:
                    return ev_load
                load_deficit = ev_load[idx] - charging_limit
                ev_load[idx] = charging_limit
            # Min battery level cannot be reached in the given time frame
            return ev_load
        
        # Scale load in the remaining time steps to reach min production
        scalable_load = np.multiply(scalable, ev_load)
        scalable_total = np.sum(scalable_load)
        factor = (load_deficit+scalable_total)/scalable_total
        scaled_load = scalable_load*factor
        ev_load = np.add(scaled_load, ev_load)
        assert np.sum(ev_load) > min_load


def get_load_trace(load_path, solar_path, ev_path, policy, out):
    with open(load_path, 'r') as file:
        load_trace = [float(line.strip()) for line in file]
    with open(solar_path, 'r') as file:
        solar_trace = [float(line.strip()) for line in file]
  
    total_solar = np.sum(solar_trace)
    soc, max_battery, total_load, max_usage = get_soc(ev_path)
    expected_pv = (total_load*eta_ev_c)/total_solar
    ev_b = max_battery

    # min battery level should be equal to 20% full battery + longest trip usage. max_battery corresponds to 80% or the full battery.
    min_battery = min(max_usage+(max_battery*0.25), max_battery)

    ev_load = [0] * num_hours
    arrival_hour = 0
    hour = 0

    while hour < num_hours:
        # Check if EV left home
        if soc[hour] == -1:
            solar_production = np.sum(solar_trace[arrival_hour:hour])
            if policy == "solar":
                if solar_production > 0:
                    # The min battery level needs to be reached in any case
                    min_battery_factor = (min_battery - ev_b)*eta_ev_c / solar_production if ev_b < min_battery else 0
                    # Do not scale PV system by more than expected_pv to meet max battery level. 
                    max_battery_factor = min((max_battery-ev_b)*eta_ev_c/solar_production, expected_pv)
                    # Charge at least to the min battery level. If realistic, keep charging until the max battery level.
                    factor = max(min_battery_factor, max_battery_factor)
                else:
                    factor = 0
                ev_load[arrival_hour:hour] = np.array(solar_trace[arrival_hour:hour])*factor
                ev_load[arrival_hour:hour] = crop_charging_rate(ev_load[arrival_hour:hour], (min_battery - ev_b)*eta_ev_c)
                ev_b += np.sum(ev_load[arrival_hour:hour])*charging_efficiency
            # Charge battery until min limit at arrival and charge to max battery or as far as possible with solar energy.
            elif policy == "safe_limit":
                ev_load[arrival_hour:hour], ev_b_arrival = arrival_charge(hour-arrival_hour, ev_b, min_battery)
                # Do not scale PV system by more than expected_pv to meet max battery level. 
                if solar_production > 0 and ev_b_arrival < max_battery:
                    factor = min((max_battery-ev_b_arrival)*eta_ev_c/solar_production, expected_pv)
                else:
                    factor = 0

                ev_load[arrival_hour:hour] += np.array(solar_trace[arrival_hour:hour])*factor
                ev_load[arrival_hour:hour] = crop_charging_rate(ev_load[arrival_hour:hour], max(0, (min_battery-ev_b)*eta_ev_c))
                ev_b += np.sum(ev_load[arrival_hour:hour])*charging_efficiency

            # Fully charge EV when it arrives home.
            elif policy == "safe":
                ev_load[arrival_hour:hour], ev_b = arrival_charge(hour-arrival_hour, ev_b, max_battery)
            else:
                raise Exception(f"Invalid charging policy: {policy}")

            # Skip to arrival of EV
            while soc[hour] == -1 and hour < num_hours:
                hour += 1

            ev_b -= soc[hour]
            arrival_hour = hour
        hour += 1
    load_trace = np.add(load_trace,ev_load)

    # Save modified load trace to txt file
    with open(out, 'w') as f:
        for value in load_trace:
            f.write(f"{value}\n")
