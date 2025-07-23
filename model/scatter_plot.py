"""
Get scatter plots showing the model predictions against the ground truth. 
Plots include the MAE, MSE, bias and R2 metrics.
"""
import joblib
import numpy as np
import pandas as pd
import torch
import matplotlib.patches as patches
import matplotlib.pyplot as plt

from scipy.stats import gaussian_kde
from sklearn.metrics import r2_score, mean_squared_error
from torch.utils.data import DataLoader
from MLP import MLP_Branched, SizingDataset, preprocess

model_name = "buildsys"
dataset = "dataset_test"
model_path = "./model/out"
dataset_path = "./dataset/out"

def compute_metrics(y_true, y_pred):
    r2 = r2_score(y_true, y_pred)
    bias = np.mean(y_pred - y_true)
    mse = np.mean((y_pred - y_true) ** 2, axis=0)
    mae = np.mean(np.abs(y_pred - y_true), axis=0)
    return r2, bias, mse, mae


def add_ceil_match_shading(ax, max_val, step):
    """
    Shade 1x1 squares where ceil(x) == ceil(y) where we ceil to the next multiple of step.
    """
    max_int = int(np.ceil(max_val))
    i = 0
    while i < max_int:
        i += step
        square = patches.Rectangle(
            (i - step, i - step),
            step,
            step,
            facecolor=(0.3, 0.3, 0.3, 0.6),
            edgecolor=(1.0, 0.647, 0.0, 1),
            linewidth=1,
        )
        ax.add_patch(square)

if __name__ == "__main__":
    df_test = pd.read_csv(f'{dataset_path}/{dataset}.csv', header=None)

    # We plot 3000 points for every dataset for fair comparison
    df_test = df_test.iloc[:3000,:]
    traces, meta, y = preprocess(df_test)

    scaler_ts = joblib.load(f"{model_path}/scaler_ts_{model_name}.pkl")
    scaler_meta = joblib.load(f"{model_path}/scaler_meta_{model_name}.pkl")

    test_traces = scaler_ts.transform(traces)
    test_meta = scaler_meta.transform(meta)

    X_tensor = torch.tensor(test_traces, dtype=torch.float32)
    M_tensor = torch.tensor(test_meta, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)

    test_loader = DataLoader(SizingDataset(X_tensor, M_tensor, y_tensor), batch_size=64, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MLP_Branched(ts_input_len=traces.shape[1], meta_input_len=test_meta.shape[1]).to(device)
    model.load_state_dict(torch.load(f"{model_path}/{model_name}.pth", map_location=device))

    model.eval()
    all_preds = []
    with torch.no_grad():
        for X_batch, M_batch, _ in test_loader:
            preds = model(X_batch.to(device), M_batch.to(device))
            all_preds.append(preds.cpu())

    predictions = torch.cat(all_preds).numpy()
    predictions = np.clip(predictions, a_min=0, a_max=None)
    y_true = y_tensor.numpy()
            
    # Get Metrics
    fig, (ax_batt, ax_pv) = plt.subplots(2, 1, figsize=(4, 8))
    x_batt = y_true[:, 0] # The x axis corresponds to the ground truth
    y_batt = predictions[:, 0] # The y axis corresponds to the model prediction
    r2_batt, bias_batt, mse_batt, mae_batt = compute_metrics(x_batt, y_batt)

    x_pv = y_true[:, 1] # The x axis corresponds to the ground truth
    y_pv = predictions[:, 1] # The y axis corresponds to the model prediction
    r2_pv, bias_pv, mse_pv, mae_pv = compute_metrics(x_pv, y_pv)

    # Prepare data
    xy_batt = np.vstack([x_batt, y_batt])
    z_batt = gaussian_kde(xy_batt)(xy_batt)
    idx_batt = z_batt.argsort()
    x_batt, y_batt, z_batt = x_batt[idx_batt], y_batt[idx_batt], z_batt[idx_batt]

    xy_pv = np.vstack([x_pv, y_pv])
    z_pv = gaussian_kde(xy_pv)(xy_pv)
    idx_pv = z_pv.argsort()
    x_pv, y_pv, z_pv = x_pv[idx_pv], y_pv[idx_pv], z_pv[idx_pv]

    # Create plot
    ax_batt.scatter(x_batt, y_batt, c=z_batt, s=1, cmap="plasma", alpha=0.9)
    ax_batt.plot([0, max(x_batt.max(), y_batt.max())], [0, max(x_batt.max(), y_batt.max())], color='black', linestyle='-')
    ax_batt.set_xlabel("Battery Capacity (kWh)")
    ax_batt.set_ylabel("Model predictions (kWh)")
    ax_batt.set_xlim(0, 20)
    ax_batt.set_ylim(0, 20)
    ax_batt.text(0.05, 0.95, f"MAE = {mae_batt:.2f} kWh\nMSE = {mse_batt:.2f} kWh\nBias = {bias_batt:.2f} kWh\n$R^2$ = {r2_batt:.2f}",
                transform=ax_batt.transAxes, fontsize=12, verticalalignment='top')
    ax_batt.grid(False)
    add_ceil_match_shading(ax_batt, 20, step=2.5)

    ax_pv.scatter(x_pv, y_pv, c=z_pv, s=1, cmap="plasma", alpha=0.9)
    ax_pv.plot([0, max(x_pv.max(), y_pv.max())], [0, max(x_pv.max(), y_pv.max())], color='black', linestyle='-')
    ax_pv.set_xlabel("PV Size (kW)")
    ax_pv.set_ylabel("Model predictions (kW)")
    ax_pv.set_xlim(0, 10)
    ax_pv.set_ylim(0, 10)
    ax_pv.text(0.05, 0.95, f"MAE = {mae_pv:.2f} kW\nMSE = {mse_pv:.2f} kW\nBias = {bias_pv:.2f} kW\n$R^2$ = {r2_pv:.2f}",
            transform=ax_pv.transAxes, fontsize=12, verticalalignment='top')
    ax_pv.grid(False)
    add_ceil_match_shading(ax_pv, 10, step=0.4)

    plt.tight_layout()
    plt.show()
