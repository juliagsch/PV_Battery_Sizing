"""
Get MSE and MAE of model predicting optimal sizing based on the FFT of 
the load and solar traces.
"""
import joblib
import numpy as np
import torch

from torch.utils.data import DataLoader
from train_noev_fourier import MLP, SizingDataset, extract_fft_features

batch_size = 64
model_name = "PV_Battery_Sizing_fourier"
base_path = "."

# Specify limit of labels to test on
max_threshold_battery = 20
max_threshold_pv = 10

if __name__ == "__main__":
    # Load test data
    data = torch.tensor(torch.from_numpy(np.loadtxt(f'{base_path}/dataset/dataset_below_threshold_test.csv', delimiter=",")), dtype=torch.float32)

    # Assuming last two columns are battery and solar sizings (labels)
    X = data[:, :-2]
    y = data[:, -2:]

    # Get fourier transform of solar and load traces
    X = extract_fft_features(X)

    # Scale input
    scaler = joblib.load(f"{base_path}/model/out/scaler_{model_name}.pkl")
    X_test = scaler.transform(X)

    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y, dtype=torch.float32)

    # Filter test set to only include battery < 20 and pv < 10
    mask = (y_test_tensor[:, 0] < max_threshold_battery) & (y_test_tensor[:, 1] < max_threshold_pv)
    X_test_tensor = X_test_tensor[mask]
    y_test_tensor = y_test_tensor[mask]
    
    test_dataset = SizingDataset(X_test_tensor, y_test_tensor)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MLP(input_size=X_test_tensor.shape[1]).to(device)
    model.load_state_dict(torch.load(f"{base_path}/model/out/{model_name}.pth", map_location=device))

    # Make predictions
    model.eval()
    with torch.no_grad():
        test_predictions = model(X_test_tensor.to(device)).cpu().numpy()

    # Compute Mean Squared Error (MSE)
    mse = np.mean((test_predictions - y_test_tensor.numpy()) ** 2, axis=0)
    print(f"Test MSE for Battery: {mse[0]:.4f}, Solar: {mse[1]:.4f}")

    # Compute Mean Absolute Error (MAE)
    mae = np.mean(abs(test_predictions - y_test_tensor.numpy()), axis=0)
    print(f"Test MAE for Battery: {mae[0]:.4f}, Solar: {mae[1]:.4f}")

    # Compute number of parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f'{model.__class__.__name__} has {total_params:,} parameters')
