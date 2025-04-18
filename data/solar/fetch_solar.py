"""
Fetch solar traces from PVWatts.
"""
import csv
import requests
import os
import random
from global_land_mask import globe

PVWATTS_KEY= os.getenv("PVWATTS_KEY")

# API endpoint
base_url = "https://developer.nrel.gov/api/pvwatts/v8.json"

# Define UK longitude and latitude bounds
# lat_min, lat_max = 49.0, 62.0
# lon_min, lon_max = -8.0, 2.0

# Include northern islands
# lat_min, lat_max = 59.0, 61.0
# lon_min, lon_max = -2.0, -0.7

# Include worldwide locations
lat_min, lat_max = -70.0, 70.0
lon_min, lon_max = -180, 180

success_count = 0
while success_count<1000:
    lat, lon = random.uniform(lat_min,lat_max), random.uniform(lon_min,lon_max)

    # Check if the coordinates are on land as otherwise the PVWatts request will fail.
    while(not globe.is_land(lat=lat, lon=lon)):
        lat, lon = random.uniform(lat_min,lat_max), random.uniform(lon_min,lon_max)

    params = {
        "format": "json",
        "api_key": PVWATTS_KEY,
        "azimuth": 180,
        "system_capacity": 1,
        "losses": 14,
        "array_type": 0,
        "module_type": 0,
        "tilt": 20,
        "lat": lat,
        "lon": lon,
        "timeframe": "hourly",
        "dataset": "intl"
    }

    response = requests.get(base_url, params=params)

    # Check if the request was successful
    if response.status_code == 200:
        data = response.json()
        ac = data["outputs"]["ac"]
        with open(f'./data/solar/pvwatts/{lat}_{lon}.txt', 'w', newline='') as f:
            wr = csv.writer(f, delimiter=',')
            for value in ac:
                wr.writerow([value / 1000.0])
        success_count += 1
    else:
        print(f"Error: {response.status_code}, {response.text}")
