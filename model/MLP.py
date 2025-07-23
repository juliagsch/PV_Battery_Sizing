import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from torch.utils.data import Dataset

# Indices of the selected FFT coefficients
fft_indices = np.array([0, 1, 11, 15, 29, 53, 57, 61, 86, 129, 160, 175, 210, 236, 257, 279, 286, 304, 350, 355, 364, 365, 366, 368, 394, 473, 494, 573, 574, 681, 720, 728, 730, 731, 732, 780, 833, 883, 1038, 1095, 1097, 1098, 1144, 1411, 1460, 1461, 1463, 1824, 1825, 1826, 2160, 2189, 2190, 2191, 2192, 2343, 2548, 2554, 2555, 2652, 2735, 2873, 2919, 2921, 3132, 3284, 3285, 3385, 3587, 3612, 3645, 3649, 3650, 3688, 3696, 3741, 3799, 3844, 3863, 4071, 4162, 4293, 4371, 4375])

class SizingDataset(Dataset):
    def __init__(self, traces, meta, targets):
        self.traces = traces
        self.meta = meta
        self.targets = targets

    def __len__(self):
        return len(self.traces)

    def __getitem__(self, idx):
        return self.traces[idx], self.meta[idx], self.targets[idx]

class MLP_Branched(nn.Module):
    def __init__(self, ts_input_len, meta_input_len, hidden_size=256):
        super(MLP_Branched, self).__init__()
        self.seq = nn.Sequential(
            nn.Linear(ts_input_len, hidden_size*2),
            nn.ReLU(),
            nn.Linear(hidden_size*2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size//2)  # Output: [battery, solar]
        )

        self.meta_net = nn.Sequential(
            nn.Linear(meta_input_len, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size//2)
        )

        self.regressor = nn.Sequential(
            nn.Linear(hidden_size, hidden_size//2),
            nn.ReLU(),
            nn.Linear(hidden_size//2, 2)
        )

    def forward(self, ts, meta):
        ts_feat = self.seq(ts)
        meta_feat = self.meta_net(meta)
        combined = torch.cat([ts_feat, meta_feat], dim=1)
        return self.regressor(combined)

"""
Extract k FFT features from solar and load traces and concatenate them.
Assumes that solar and load traces are interleaved.
"""
def extract_fft_features(data):
    fft_features = []
    for i in range(data.shape[0]):
        solar = data[i,:8760]
        load = data[i,8760:]
        assert load.shape == solar.shape

        fft_mag = np.abs(np.fft.rfft(solar))
        top_k_solar = fft_mag[fft_indices]
        fft_mag[fft_indices] = 0
        top_k_solar = np.concatenate([top_k_solar, np.sort(fft_mag)[-16:]])
        fft_mag = np.abs(np.fft.rfft(load))
        top_k_load = fft_mag[fft_indices]
        fft_mag[fft_indices] = 0
        top_k_load = np.concatenate([top_k_load, np.sort(fft_mag)[-16:]])
        fft_features.append(np.concatenate([top_k_solar, top_k_load])) 
    
    return np.array(fft_features)


""" Extract FFT of load and solar traces, labels and metadata including EV data and EUE from test data."""
def preprocess(df):
    # Get Fourier transform of solar and load traces
    trace_len = 2 * 8760
    traces_full = df.iloc[:, :trace_len].astype(float).to_numpy()
    traces = extract_fft_features(traces_full)

    # Filter labels
    targets = df.iloc[:, -2:].astype(float).to_numpy()

    # Filter metadata and encode charging policy
    meta = df.iloc[:, trace_len:-2].copy()
    # In the no EV case, the WFH days may be set to 0 instead of False which is corrected here as booleans are expected.
    cols = [17521, 17522, 17523, 17524, 17525]
    meta[cols] = meta[cols] == "True"
    # One-hot encode policy
    meta = meta.rename(columns={trace_len: "policy"})
    meta["policy"] = meta["policy"].astype(str)
    meta = pd.get_dummies(meta, columns=["policy"])
    meta.columns = meta.columns.astype(str)

    return traces, meta, targets
