"""
Train model to predict optimal sizing based on the FFT of 
the load and solar traces as well as on EV metadata and EUE target.
"""
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset, random_split
from torch.utils.tensorboard import SummaryWriter

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
    def __init__(self, ts_input_len, meta_input_len, hidden_size=128):
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
Estract k FFT features from solar and load traces and concatenate them.
Assumes that solar and load traces are interleaved.
"""
def extract_fft_features(data, k=128):
    fft_features = []
    for i in range(data.shape[0]):
        fft_mag = np.abs(np.fft.rfft(data[i,::2]))
        top_k_pv = np.sort(fft_mag)[-k:]
        fft_mag = np.abs(np.fft.rfft(data[i,1::2]))
        top_k_battery = np.sort(fft_mag)[-k:]
        fft_features.append(np.concatenate([top_k_pv, top_k_battery]))   

    return np.array(fft_features)

""" Extract FFT of load and solar traces, labels and metadata including EV data and EUE from test data."""
def preprocess(df):
    # Get labels
    targets = df.iloc[:, -2:].astype(float).to_numpy()

    # Get fourier transform of solar and load traces
    trace_len = 2*8760
    traces = df.iloc[:, :trace_len].astype(float).to_numpy()
    traces = extract_fft_features(traces)

    # Get metadata and use one-hot encoding to encode charging policy
    meta = df.iloc[:, trace_len:-2].copy()
    meta = meta.rename(columns={trace_len: "policy"})
    meta["policy"] = meta["policy"].astype(str)
    meta = pd.get_dummies(meta, columns=["policy"])
    meta.columns = meta.columns.astype(str)

    return traces, meta, targets

if __name__ == "__main__":
    # Input data
    model_name = "CNN_MLP_fourier_2x68000_256"
    batch_size = 64
    num_epochs = 500
    validation_split = 0.1
    use_fourier = True

    home_path = "/cluster/home/jgschwind"
    scratch_path = "/cluster/scratch/jgschwind"

    # Initialize tensorboard writer
    writer = SummaryWriter(f"runs/{model_name}")

    # Load and process data
    df_train = pd.read_csv(f'{scratch_path}/dataset_train_eveue_interleaved.csv', header=None)
    df_test = pd.read_csv(f'{scratch_path}/dataset_test_eveue_interleaved.csv', header=None)

    train_traces, train_meta_df, y_train = preprocess(df_train)
    test_traces, test_meta_df, y_test = preprocess(df_test)

    test_meta_df = test_meta_df.reindex(columns=train_meta_df.columns, fill_value=0)

    # Create scalers and scale input
    scaler_ts = StandardScaler()
    scaler_meta = StandardScaler()

    train_traces = scaler_ts.fit_transform(train_traces)
    test_traces = scaler_ts.transform(test_traces)

    train_meta = scaler_meta.fit_transform(train_meta_df)
    test_meta = scaler_meta.transform(test_meta_df)

    joblib.dump(scaler_ts, f"{home_path}/scaler_ts_{model_name}.pkl")
    joblib.dump(scaler_meta, f"{home_path}/scaler_meta_{model_name}.pkl")

    # Create input tensors
    X_train_tensor = torch.tensor(train_traces, dtype=torch.float32)
    M_train_tensor = torch.tensor(train_meta, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)

    X_test_tensor = torch.tensor(test_traces, dtype=torch.float32)
    M_test_tensor = torch.tensor(test_meta, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

    # Get train and val split
    full_dataset = SizingDataset(X_train_tensor, M_train_tensor, y_train_tensor)
    val_size = int(len(full_dataset) * validation_split)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(SizingDataset(X_test_tensor, M_test_tensor, y_test_tensor), batch_size=batch_size, shuffle=False)

    # Initialize model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MLP_Branched(ts_input_len=train_traces.shape[1], meta_input_len=train_meta.shape[1]).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Start training
    best_val_loss = float("inf")
    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0
        for ts, meta, targets in train_loader:
            ts, meta, targets = ts.to(device), meta.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(ts, meta)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
        avg_train_loss = total_train_loss / len(train_loader)

        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for ts, meta, targets in val_loader:
                ts, meta, targets = ts.to(device), meta.to(device), targets.to(device)
                outputs = model(ts, meta)
                loss = criterion(outputs, targets)
                total_val_loss += loss.item()
        avg_val_loss = total_val_loss / len(val_loader)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), f"{model_name}_best.pth")
            print(f"Epoch {epoch+1}: New best model saved with val loss {best_val_loss:.4f}")

        writer.add_scalar("Loss/train", avg_train_loss, epoch)
        writer.add_scalar("Loss/val", avg_val_loss, epoch)

        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

    writer.close()

    # Final evaluation
    model.load_state_dict(torch.load(f"{model_name}_best.pth"))
    model.eval()
    all_preds = []
    with torch.no_grad():
        for X_batch, M_batch, _ in test_loader:
            preds = model(X_batch.to(device), M_batch.to(device))
            all_preds.append(preds.cpu())

    # Concatenate predictions
    test_predictions = torch.cat(all_preds).numpy()
    torch.save(model.state_dict(), f"{model_name}.pth")
    print("Final model saved.")

    # Compute Mean Squared Error (MSE)
    mse = np.mean((test_predictions - y_test_tensor.numpy()) ** 2, axis=0)
    print(f"Test MSE - Battery: {mse[0]:.4f}, Solar: {mse[1]:.4f}")
    
    # Compute Mean Absolute Error (MAE)
    mae = np.mean(np.abs(test_predictions - y_test_tensor.numpy()), axis=0)
    print(f"Test MAE - Battery: {mae[0]:.4f}, Solar: {mae[1]:.4f}")
