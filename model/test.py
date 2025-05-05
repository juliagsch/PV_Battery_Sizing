import torch
from torch.utils.data import DataLoader
import numpy as np
from train_static import SizingDataset, MLP
from sklearn.preprocessing import StandardScaler
import pandas as pd


model_name = "PV_sizing_EV_static_interleaved"
base_path = "/cluster/home/jgschwind"
batch_size = 64

# Load train and test csv
df_train = pd.read_csv(f'{base_path}/dataset_train_static_interleaved_.csv', header=None)
df_test = pd.read_csv(f'{base_path}/dataset_test_static_interleaved_threshold3020.csv', header=None)

# Print max values
battery_col_idx = df_train.shape[1] - 2
pv_col_idx = df_train.shape[1] - 1

max_battery = df_train[battery_col_idx].max()
max_pv = df_train[pv_col_idx].max()

print(f"Max Battery Size Train: {max_battery}")
print(f"Max PV Size Train: {max_pv}")

max_battery = df_test[battery_col_idx].max()
max_pv = df_test[pv_col_idx].max()

print(f"Max Battery Size Test: {max_battery}")
print(f"Max PV Size Test: {max_pv}")

print("Loading Data...")
# Map charging policies to integer
df_train.replace("safe", 0, inplace=True)
df_train.replace("safe_limit", 1, inplace=True)
df_train.replace("solar", 2, inplace=True)

df_test.replace("safe", 0, inplace=True)
df_test.replace("safe_limit", 1, inplace=True)
df_test.replace("solar", 2, inplace=True)

print("Create Tensors...")
# Then into torch tensor
data_train = torch.tensor(df_train.astype(float).to_numpy(), dtype=torch.float32)
data_test = torch.tensor(df_test.astype(float).to_numpy(), dtype=torch.float32)

X_train = data_train[:, :-3]  # The last 3 columns are the charging policy, optimal battery and solar sizings
y_train = data_train[:, -2:]
X_test = data_test[:, :-3]
y_test = data_test[:, -2:]

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

# Create dataset and dataloaders
test_dataset = SizingDataset(X_test_tensor, y_test_tensor)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Recreate the model architecture
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

input_size = X_test_tensor.shape[1]
model = MLP(input_size=input_size).to(device)

# Load the saved weights
model.load_state_dict(torch.load(f"{base_path}/{model_name}.pth", map_location=device))

model.eval()
with torch.no_grad():
    test_predictions = model(X_test_tensor.to(device)).cpu().numpy()

# Compute Mean Squared Error (MSE)
mse = np.mean((test_predictions - y_test_tensor.numpy()) ** 2, axis=0)
print(f"Test MSE for Battery: {mse[0]:.4f}, Solar: {mse[1]:.4f}")


# Compute Mean Absolute Error (MAE)
mae = np.mean(abs(test_predictions - y_test_tensor.numpy()), axis=0)
print(f"Test MAE for Battery: {mae[0]:.4f}, Solar: {mae[1]:.4f}")


total_params = sum(p.numel() for p in model.parameters())
print(f'{model.__class__.__name__} has {total_params:,} parameters')


print("Best Model: ")
# Load the saved weights
model.load_state_dict(torch.load(f"{base_path}/{model_name}_best.pth", map_location=device))

model.eval()
with torch.no_grad():
    test_predictions = model(X_test_tensor.to(device)).cpu().numpy()

# Compute Mean Squared Error (MSE)
mse = np.mean((test_predictions - y_test_tensor.numpy()) ** 2, axis=0)
print(f"Test MSE for Battery: {mse[0]:.4f}, Solar: {mse[1]:.4f}")


# Compute Mean Absolute Error (MAE)
mae = np.mean(abs(test_predictions - y_test_tensor.numpy()), axis=0)
print(f"Test MAE for Battery: {mae[0]:.4f}, Solar: {mae[1]:.4f}")


total_params = sum(p.numel() for p in model.parameters())
print(f'{model.__class__.__name__} has {total_params:,} parameters')