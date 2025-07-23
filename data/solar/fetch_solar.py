"""
Fetch solar traces from PVWatts.
"""
import csv
import requests
import os
import random
from global_land_mask import globe

from dotenv import load_dotenv
load_dotenv()  

PVWATTS_KEY= os.getenv("PVWATTS_KEY")

# API endpoint
base_url = "https://developer.nrel.gov/api/pvwatts/v8.json"

# Output dir
os.makedirs("./data/solar/original", exist_ok=True)

# Include worldwide locations
lat_min, lat_max = -70.0, 70.0
lon_min, lon_max = -180, 180

target_file_count = 1000
success_count = 0

while success_count<target_file_count:
    lat, lon = random.uniform(lat_min,lat_max), random.uniform(lon_min,lon_max)

    # Check if the coordinates are on land as otherwise the PVWatts request will fail.
    while(not globe.is_land(lat=lat, lon=lon)):
        lat, lon = random.uniform(lat_min,lat_max), random.uniform(lon_min,lon_max)

    orientation = 0 if lat < 0 else 180

    params = {
        "format": "json",
        "api_key": PVWATTS_KEY,
        "azimuth": orientation,
        "system_capacity": 1,
        "losses": 14,
        "array_type": 0,
        "module_type": 0,
        "tilt": 20,
        "lat": lat,
        "lon": lon,
        "timeframe": "hourly",
    }

    response = requests.get(base_url, params=params)
    # Check if the request was successful
    if response.status_code == 200:
        data = response.json()
        ac = data["outputs"]["ac"]
        with open(f'./data/solar/original/{lat}_{lon}.txt', 'w', newline='') as f:
            wr = csv.writer(f, delimiter=',')
            for value in ac:
                wr.writerow([value / 1000.0])
        success_count += 1
    
        
    elif response.status_code == 422:
        # If call fails, try again with international dataset
        param_intl = {
            "format": "json",
            "api_key": PVWATTS_KEY,
            "azimuth": orientation,
            "system_capacity": 1,
            "losses": 14,
            "array_type": 0,
            "module_type": 0,
            "tilt": 20,
            "lat": lat,
            "lon": lon,
            "timeframe": "hourly",
            "dataset": "intl" # Use international dataset instead of default
        }
        response = requests.get(base_url, params=param_intl)

        # Check if the request was successful
        if response.status_code == 200:
            data = response.json()
            ac = data["outputs"]["ac"]
            with open(f'./data/solar/original/{lat}_{lon}.txt', 'w', newline='') as f:
                wr = csv.writer(f, delimiter=',')
                for value in ac:
                    wr.writerow([value / 1000.0])
            success_count += 1
        else:
            print(f"Error: {response.status_code}, {response.text}")
    else:
        print(f"Error: {response.status_code}, {response.text}")
