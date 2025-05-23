"""
Get MSE and MAE of model predicting optimal sizing based on the FFT of 
the load and solar traces as well as on EV metadata and EUE target.
"""
import joblib
import numpy as np
import pandas as pd
import torch

from torch.utils.data import DataLoader
from train_ev_fourier import MLP_Branched, SizingDataset, preprocess


batch_size = 64
model_name = "CNN_MLP_fourier_2x68000_256"
home_path = "."
scratch_path = "./dataset"

# Specify limit of labels to test on
max_threshold_battery = 20
max_threshold_pv = 10

# Specify which policies to exclude from testing if applicable
exclude_policies = ["policy_arrival_limit"]


if __name__ == "__main__":
    df_test = pd.read_csv(f'{scratch_path}/dataset_test_eveue_interleaved.csv', header=None)
    test_traces, test_meta_df, y_test = preprocess(df_test)

    # Specify which policies to test
    for policy in exclude_policies:
        mask = test_meta_df[policy] == False
        test_meta = test_meta_df[mask]
        test_traces = test_traces[mask]
        y_test = y_test[mask]

    # Scale input
    scaler_ts= joblib.load(f"{home_path}/model/out/scaler_ts_{model_name}.pkl")
    scaler_meta= joblib.load(f"{home_path}/model/out/scaler_meta_{model_name}.pkl")

    test_traces = scaler_ts.transform(test_traces)
    test_meta = scaler_meta.transform(test_meta)

    X_test_tensor = torch.tensor(test_traces, dtype=torch.float32)
    M_test_tensor = torch.tensor(test_meta, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

    # Filter test set to only include battery < 20 and pv < 10
    mask = (y_test_tensor[:, 0] < max_threshold_battery) & (y_test_tensor[:, 1] < max_threshold_pv)
    X_test_tensor = X_test_tensor[mask]
    M_test_tensor = M_test_tensor[mask]
    y_test_tensor = y_test_tensor[mask]

    test_loader = DataLoader(SizingDataset(X_test_tensor, M_test_tensor, y_test_tensor), batch_size=batch_size, shuffle=False)

    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MLP_Branched(ts_input_len=test_traces.shape[1], meta_input_len=test_meta.shape[1]).to(device)
    model.load_state_dict(torch.load(f"{home_path}/model/out/{model_name}_best.pth", map_location=device))

    # Make predictions
    model.eval()
    all_preds = []
    with torch.no_grad():
        for X_batch, M_batch, _ in test_loader:
            preds = model(X_batch.to(device), M_batch.to(device))
            all_preds.append(preds.cpu())

    # Concatenate predictions
    test_predictions = torch.cat(all_preds).numpy()

    # Compute Mean Squared Error (MSE)
    mse = np.mean((test_predictions - y_test_tensor.numpy()) ** 2, axis=0)
    print(f"Test MSE for Battery: {mse[0]:.4f}, Solar: {mse[1]:.4f}")

    # Compute Mean Absolute Error (MAE)
    mae = np.mean(abs(test_predictions - y_test_tensor.numpy()), axis=0)
    print(f"Test MAE for Battery: {mae[0]:.4f}, Solar: {mae[1]:.4f}")

    # Compute number of parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f'{model.__class__.__name__} has {total_params:,} parameters')
