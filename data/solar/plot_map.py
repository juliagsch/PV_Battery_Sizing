"""
Plot the coordinates from which the solar traces originate on a map.
"""
import folium
import webbrowser
import os

def get_solar_files():
    return [f[:-4] for f in os.listdir("./data/solar/united") if f.endswith('.txt')]

# Define coordinates (latitude, longitude)
files = get_solar_files()
coordinates = []
for file in files:
    coordinate = file.split('_')
    coordinates.append((coordinate[0], coordinate[1]))

m = folium.Map(location=coordinates[0], zoom_start=7)
for lat, lon in coordinates:
    folium.Marker([lat, lon], popup=f"({lat}, {lon})").add_to(m)

# Save the map as an HTML file
map_path = "./data/solar/map.html"
m.save(map_path)
webbrowser.open("file://" + map_path)