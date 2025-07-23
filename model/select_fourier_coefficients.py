"""
Selects the fourier coefficient with the highest absolute correlation to the ground truth.
"""
import pandas as pd
import numpy as np
from scipy.stats import pearsonr

# Define how many coefficients to select per trace / label combination
top_k = 32

# Load training data
df = pd.read_csv('./dataset/out/dataset_train20.csv') # We select a partial dataset from the training data with about 10,000 samples.

X_solar = df.iloc[:, :8760].values
X_load = df.iloc[:, 8760:8760*2].values
y_battery = df.iloc[:, -2].values
y_pv = df.iloc[:, -1].values

num_samples, dim = X_solar.shape

# Compute FFT
fft_mag_solar = np.abs(np.fft.rfft(X_solar, axis=1))
fft_mag_load = np.abs(np.fft.rfft(X_load, axis=1))


# For every trace / label combination, the top_k most important coefficients are selected.
correlations_solar_battery = []
for i in range(fft_mag_solar.shape[1]):
    corr, _ = pearsonr(fft_mag_solar[:, i], y_battery)
    correlations_solar_battery.append(corr)

abs_corr_solar_battery = np.abs(correlations_solar_battery)
top_indices_solar_battery = np.argsort(-abs_corr_solar_battery)[:top_k]

print("Top FFT indices (solar & PV battery):", top_indices_solar_battery)

correlations_solar_pv = []
for i in range(fft_mag_solar.shape[1]):
    corr_solar_pv, _ = pearsonr(fft_mag_solar[:, i], y_pv)
    correlations_solar_pv.append(corr_solar_pv)

abs_corr_solar_pv = np.abs(correlations_solar_pv)
top_indices_solar_pv = np.argsort(-abs_corr_solar_pv)[:top_k]

print("Top FFT indices (Solar & PV sizing):", top_indices_solar_pv)

correlations_load_battery = []
for i in range(fft_mag_solar.shape[1]):
    corr_load_battery, _ = pearsonr(fft_mag_load[:, i], y_battery)
    correlations_load_battery.append(corr_load_battery)

abs_corr_load_battery = np.abs(correlations_load_battery)

top_indices_load_battery = np.argsort(-abs_corr_load_battery)[:top_k]

print("Top FFT indices (Load & battery sizing):", top_indices_load_battery)

correlations_load_pv = []
for i in range(fft_mag_solar.shape[1]):
    corr_load_pv, _ = pearsonr(fft_mag_load[:, i], y_pv)

    correlations_load_pv.append(corr_load_pv)

abs_corr_load_pv = np.abs(correlations_load_pv)

top_indices_load_pv = np.argsort(-abs_corr_load_pv)[:top_k]

print("Top FFT indices (Load & PV sizing):", top_indices_load_pv)

result = np.concatenate([
    top_indices_load_pv,
    top_indices_load_battery,
    top_indices_solar_pv,
    top_indices_solar_battery
])
result = np.unique(result)
result = result.tolist()
print(len(result), "coefficients were selected:", result)
