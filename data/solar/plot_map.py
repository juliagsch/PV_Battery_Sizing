"""
Plot the coordinates from which the solar traces originate on a map.
"""
import folium
from folium.plugins import HeatMap
import os

def get_solar_files():
    files = []
    for split in ["train", "test", "val"]:
        files = files + [f[:-4] for f in os.listdir(f"./data/solar/{split}") if f.endswith('.txt')]
    return files

# Read coordinates. All solar files are named lat_lon_augmentationIdx.txt
files = get_solar_files()
coordinates = []
for file in files:
    coordinate = file.split('_')
    coordinates.append((coordinate[0], coordinate[1]))

# Initialize map
m = folium.Map(location=coordinates[0], zoom_start=6, tiles='CartoDB positron')

gradient = {
    "0.2": 'yellow',
    "0.5": 'orange',
    "0.8": 'darkorange',
    "1.0": 'red'
}

HeatMap(coordinates, radius=8, blur=12, gradient=gradient).add_to(m)
map_path = "solar_heatmap.html"
m.save(map_path)
