"""
Creates a map of the geographic distribution of the battery and PV sizing errors on the test set. 
"""

import geopandas as gpd
import h3.api.basic_int as h3
import joblib
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from shapely.geometry import Polygon
from torch.utils.data import DataLoader
from MLP import preprocess, MLP_Branched, SizingDataset

model_name = "buildsys"
dataset = "dataset_test"
model_path = "./model/out"
dataset_path = "./dataset/out"

# Load files with coordinates
df_test = pd.read_csv(f'{dataset_path}/{dataset}.csv', header=None)
traces, meta, y = preprocess(df_test)
coords_df = pd.read_csv(f"./dataset/out/files_processed_{dataset}.csv", header=None, names=['coordinates', 'load'])

scaler_ts= joblib.load(f"{model_path}/scaler_ts_{model_name}.pkl")
scaler_meta= joblib.load(f"{model_path}/scaler_meta_{model_name}.pkl")

test_traces = scaler_ts.transform(traces)
test_meta = scaler_meta.transform(meta)

X_test_tensor = torch.tensor(test_traces, dtype=torch.float32)
M_test_tensor = torch.tensor(test_meta, dtype=torch.float32)
y_test_tensor = torch.tensor(y, dtype=torch.float32)
test_loader = DataLoader(SizingDataset(X_test_tensor, M_test_tensor, y_test_tensor), batch_size=64, shuffle=False)

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MLP_Branched(ts_input_len=test_traces.shape[1], meta_input_len=test_meta.shape[1], hidden_size=256).to(device)
model.load_state_dict(torch.load(f"{model_path}/{model_name}.pth", map_location=device))

# Make predictions
model.eval()
all_preds = []
with torch.no_grad():
    for X_batch, M_batch, _ in test_loader:
        preds = model(X_batch.to(device), M_batch.to(device))
        all_preds.append(preds.cpu())

# Concatenate predictions
test_predictions = torch.cat(all_preds).numpy()
test_predictions = np.clip(test_predictions, a_min=0, a_max=None)
errors = test_predictions - y_test_tensor.numpy()


# Function to extract lat / lon from the file path
def extract_lat_lon(path):
    filename = path.split('/')[-1].replace('.txt', '')
    parts = filename.split('_')
    lat = round(float(parts[0]))
    lon = round(float(parts[1]))
    return pd.Series({'latitude': lat, 'longitude': lon})

# Apply to your dataframe
coords_df[['latitude', 'longitude']] = coords_df['coordinates'].apply(extract_lat_lon)

# Example: your dataframe with lon, lat, and error
df = pd.DataFrame({
    'longitude': coords_df['longitude'],
    'latitude': coords_df['latitude'],
    'errorBattery': abs(errors[:,0]),
    'errorPV': abs(errors[:,1]),
})

# Determine cell of coordinates
df['h3_index'] = df.apply(
    lambda row: h3.latlng_to_cell(row['latitude'], row['longitude'], res=2),
    axis=1
)

# Aggregate errors by cell
hex_error = df.groupby('h3_index').agg({'errorPV': 'mean', 'errorBattery': 'mean'}).reset_index()

# Convert hexagons to polygons
def h3_to_polygon(h3_index):
    boundary = h3.cell_to_boundary(h3_index)
    return [(lng, lat) for lat, lng in boundary]

hex_error['geometry'] = hex_error['h3_index'].apply(lambda h: Polygon(h3_to_polygon(h)))

gdf = gpd.GeoDataFrame(hex_error, geometry='geometry', crs="EPSG:4326")
bivariate_colors = {
    '1-1': '#fff7bc',
    '1-2': '#fec44f',
    '1-3': '#fe9929',
    '1-4': '#cc4c02',
    '2-1': '#fdd0a2',
    '2-2': '#fdae6b',
    '2-3': '#fd8d3c',
    '2-4': '#d94801',
    '3-1': '#dadaeb',
    '3-2': '#bcbddc',
    '3-3': '#807dba',
    '3-4': '#6a51a3',
    '4-1': '#a6611a',
    '4-2': '#8c510a',
    '4-3': '#7f2704',
    '4-4': '#3f007d'
}

bins = [0.0, 0.25, 0.5, 0.75, 10]
labels = [1, 2, 3, 4]

# Cut your variables into bins
gdf['battery_bin'] = pd.cut(gdf['errorBattery'], bins=bins, labels=labels, include_lowest=True)
gdf['pv_bin'] = pd.cut(gdf['errorPV'], bins=bins, labels=labels, include_lowest=True)

# Map the colors to the GeoDataFrame
gdf['bivariate_class'] = gdf['pv_bin'].astype(str) + '-' + gdf['battery_bin'].astype(str)
gdf['bivariate_color'] = gdf['bivariate_class'].map(bivariate_colors)

world = gpd.read_file("./model/ne_110m_admin_0_countries") # needs to be downloaded from www.naturalearthdata.com
fig, ax = plt.subplots(1, 1, figsize=(15, 10))
world.plot(ax=ax, color='white', edgecolor='#8c510a', linewidth=0.4)
gdf.plot(ax=ax, color=gdf['bivariate_color'], edgecolor='grey')

# Create fiture
fig, ax = plt.subplots(figsize=(5, 5))
for x in range(1, 5):
    for y in range(1, 5):
        color = bivariate_colors[f"{x}-{y}"]
        ax.add_patch(
            mpatches.Rectangle(
                (x, y), 1, 1, facecolor=color
            )
        )

ax.set_xlim(1, 5)
ax.set_ylim(1, 5)
ax.set_xticks([1.5, 2.5, 3.5, 4.5])
ax.set_xticklabels(['0-0.25', '0.25-0.5', '0.5-0.75', '>0.75'])
ax.set_yticks([1.5, 2.5, 3.5, 4.5])
ax.set_yticklabels(['0-0.25', '0.25-0.5', '0.5-0.75', '>0.75'])
ax.set_xlabel('PV Error')
ax.set_ylabel('Battery Error')
ax.set_title('Bivariate Legend')

plt.show()
