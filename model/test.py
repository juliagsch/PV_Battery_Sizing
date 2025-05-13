import torch
from torch.utils.data import DataLoader
import numpy as np
from train import MLP, SizingDataset
import joblib

base_path = "."
model_name = "model/out/sequential_below_threshold_b64_e2000_Adam"
batch_size = 64

data = torch.tensor(torch.from_numpy(np.loadtxt(f'{base_path}/dataset/dataset_below_threshold_test.csv', delimiter=",")), dtype=torch.float32)

# Assuming last two columns are battery and solar sizings (labels)
X = data[:, :-2]
y = data[:, -2:]

# Prepare data
scaler = joblib.load(f"{base_path}/{model_name}.pkl")
X_test = scaler.transform(X)

X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y, dtype=torch.float32)

# Create dataset and dataloaders
test_dataset = SizingDataset(X_test_tensor, y_test_tensor)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Recreate the model architecture
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

input_size = X_test_tensor.shape[1]
model = MLP(input_size=input_size).to(device)

# Load the saved weights
model.load_state_dict(torch.load(f"./{model_name}.pth", map_location=device))

model.eval()
with torch.no_grad():
    test_predictions = model(X_test_tensor.to(device)).cpu().numpy()

# Compute Mean Squared Error (MSE)
mse = np.mean((test_predictions - y_test_tensor.numpy()) ** 2, axis=0)
print(f"Test MSE for Battery: {mse[0]:.4f}, Solar: {mse[1]:.4f}")

# Compute Mean Absolute Error (MAE)
mae = np.mean(abs(test_predictions - y_test_tensor.numpy()), axis=0)
print(f"Test MAE for Battery: {mae[0]:.4f}, Solar: {mae[1]:.4f}")

# Count the number of model parameters
total_params = sum(p.numel() for p in model.parameters())
print(f'{model.__class__.__name__} has {total_params:,} parameters')
