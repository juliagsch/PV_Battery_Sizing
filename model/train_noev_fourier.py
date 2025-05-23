"""
Train model to predict optimal sizing based on the FFT of 
the load and solar traces.
"""
import joblib
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter

class SizingDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class MLP(nn.Module):
    def __init__(self, input_size, hidden_size=128, output_size=2):
        super(MLP, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, output_size)  # Output: [battery, solar]
        )

    def forward(self, x):
        return self.model(x)


"""
Estract k FFT features from solar and load traces and concatenate them.
Assumes that solar and load traces are interleaved.
"""
def extract_fft_features(data, k=100):
    fft_features = []
    for i in range(data.shape[0]):
        fft_mag = np.abs(np.fft.rfft(data[i,::2]))
        top_k_pv = np.sort(fft_mag)[-k:]
        fft_mag = np.abs(np.fft.rfft(data[i,1::2]))
        top_k_battery = np.sort(fft_mag)[-k:]
        fft_features.append(np.concatenate([top_k_pv, top_k_battery]))   

    return np.array(fft_features)

if __name__ == "__main__":
    batch_size = 64
    num_epochs = 200
    model_name = "PV_Battery_Sizing_fourier"
    base_path = "."

    writer = SummaryWriter(f"runs/{model_name}")

    # Load CSV files
    data_train = np.loadtxt(f'{base_path}/dataset/dataset_below_threshold_train.csv', delimiter=",")
    data_test = np.loadtxt(f'{base_path}/dataset/dataset_below_threshold_test.csv', delimiter=",")

    # Separate inputs and targets
    X_train_raw, y_train = data_train[:, :-2], data_train[:, -2:]
    X_test_raw, y_test = data_test[:, :-2], data_test[:, -2:]

    # Apply FFT feature extraction
    X_train = extract_fft_features(X_train_raw)
    X_test = extract_fft_features(X_test_raw)

    # Normalize the FFT features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    joblib.dump(scaler, f"{base_path}/scaler_{model_name}.pkl")

    # Convert to tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

    train_dataset = SizingDataset(X_train_tensor, y_train_tensor)
    test_dataset = SizingDataset(X_test_tensor, y_test_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Initialize Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MLP(input_size=X_train.shape[1]).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Start training
    best_loss = float("inf")
    for epoch in range(num_epochs):
        total_loss = 0
        model.train()
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), f"{model_name}_best.pth")
            print(f"Epoch {epoch+1}: New best model saved with loss {best_loss:.4f}")
        writer.add_scalar("Loss/train", avg_loss, epoch)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")

    writer.close()

    # Evaluation
    model.eval()
    with torch.no_grad():
        test_predictions = model(X_test_tensor.to(device)).cpu().numpy()

    torch.save(model.state_dict(), f"{model_name}.pth")
    print("Model saved successfully!")

    # Compute Mean Squared Error (MSE)
    mse = np.mean((test_predictions - y_test) ** 2, axis=0)
    print(f"Test MSE for Battery: {mse[0]:.4f}, Solar: {mse[1]:.4f}")

    # Compute Mean Absolute Error (MAE)
    mae = np.mean(np.abs(test_predictions - y_test), axis=0)
    print(f"Test MAE - Battery: {mae[0]:.4f}, Solar: {mae[1]:.4f}")

    # Compute number of parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f'{model.__class__.__name__} has {total_params:,} parameters')
