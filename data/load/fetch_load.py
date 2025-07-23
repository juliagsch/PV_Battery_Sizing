"""
This file contains the code used to fetch the faraday raw data. We iterate through all of the days of the year and 
save the load trace for a given population on a given day. The daily load traces are saved in ./data/load/faraday_raw. Using the file ./data/load/process_load.py,
the data can be aggregated to create a yearly load trace for each household in the population.
Please request a Faraday API key here: https://developer.nrel.gov/signup/ and specify it in your .env file to run this script.
"""
import requests
import os
import json

from dotenv import load_dotenv
load_dotenv()  

FARADAY_KEY = os.getenv("FARADAY_KEY")
url = "https://faraday-api-gateway-28g4j071.nw.gateway.dev/v4/predict/"

days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
months = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]
days_per_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

os.makedirs('./data/load/faraday/', exist_ok=True)

headers = {
    "accept": "application/json",
    "content-type": "application/json",
    "x-api-key": FARADAY_KEY
}

# This is an example payload. In reality, it has to be split up into smaller junks as there will likely be timeouts for large requests.
# Experience showed that requesting 500 buildings at a time is managable.
def get_payload(day, month):
    return {
        "day_of_week": day,
        "month_of_year": month,
        "population": [
            {
                "name": "DetachedA",
                "count": 100,
                "attributes": {
                "energy_rating": "A/B/C",
                "urbanity": "Urban",
                "property_type": "Detached",
                "is_mains_gas": "Has Mains Gas",
                "lct": [
                "Has No LCTs"
                ],
                "tariff_type": "any"
                }
            },
            {
                "name": "DetachedD",
                "count": 100,
                "attributes": {
                "energy_rating": "D/E",
                "urbanity": "Urban",
                "property_type": "Detached",
                "is_mains_gas": "Has Mains Gas",
                "lct": [
                "Has No LCTs"
                ],
                "tariff_type": "any"
                }
            },
            {
                "name": "TerracedA",
                "count": 100,
                "attributes": {
                "energy_rating": "A/B/C",
                "urbanity": "Urban",
                "property_type": "Terraced",
                "is_mains_gas": "Has Mains Gas",
                "lct": [
                "Has No LCTs"
                ],
                "tariff_type": "any"
                }
            },
            {
                "name": "TerracedD",
                "count": 100,
                "attributes": {
                "energy_rating": "D/E",
                "urbanity": "Urban",
                "property_type": "Terraced",
                "is_mains_gas": "Has Mains Gas",
                "lct": [
                "Has No LCTs"
                ],
                "tariff_type": "any"
                }
            },
            {
                "name": "Semi-detachedA",
                "count": 100,
                "attributes": {
                "energy_rating": "A/B/C",
                "urbanity": "Urban",
                "property_type": "Semi-detached",
                "is_mains_gas": "Has Mains Gas",
                "lct": [
                "Has No LCTs"
                ],
                "tariff_type": "any"
                }
            },
            {
                "name": "Semi-detachedD",
                "count": 100,
                "attributes": {
                "energy_rating": "D/E",
                "urbanity": "Urban",
                "property_type": "Semi-detached",
                "is_mains_gas": "Has Mains Gas",
                "lct": [
                "Has No LCTs"
                ],
                "tariff_type": "any"
                }
            },
            {
                "name": "DetachedRemote",
                "count": 200,
                "attributes": {
                "energy_rating": "Any",
                "urbanity": "Remote",
                "property_type": "Detached",
                "is_mains_gas": "Any",
                "lct": [
                "Has No LCTs"
                ],
                "tariff_type": "any"
                }
            },
            {
                "name": "TerracedRemote",
                "count": 200,
                "attributes": {
                "energy_rating": "Any",
                "urbanity": "Remote",
                "property_type": "Terraced",
                "is_mains_gas": "Any",
                "lct": [
                "Has No LCTs"
                ],
                "tariff_type": "any"
                }
            },
            {
                "name": "Semi-detachedRemote",
                "count": 200,
                "attributes": {
                "energy_rating": "Any",
                "urbanity": "Remote",
                "property_type": "Semi-detached",
                "is_mains_gas": "Any",
                "lct": [
                "Has No LCTs"
                ],
                "tariff_type": "any"
                }
            },
        ]
    }

day_idx = 0
count = 0

# Iterate through days and months
for month_idx, total_month_days in enumerate(days_per_month):
    month_day = 0
    while month_day<total_month_days:
        print(f"Fetching profiles for {days[day_idx]} in {months[month_idx]}. Current day of the year: {count}")

        payload = get_payload(days[day_idx], months[month_idx])
        response = requests.post(url, json=payload, headers=headers)
        if response.status_code == 200:
            data = response.json()

            with open(f"./data/load/faraday/day_{count}.json", "w") as f:
                json.dump(data, f, indent=4)
        else: 
            print(f"Error: {response.status_code}, {response.text}")

        day_idx = (day_idx + 1)%7
        count += 1
        month_day += 1

print(f"Fetched profiles for {count} days")