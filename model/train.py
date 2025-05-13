from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from torch.utils.data import Dataset
from torch.utils.tensorboard import SummaryWriter
import joblib

class SizingDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class MLP(nn.Module):
    def __init__(self, input_size, hidden_size=512, output_size=2):
        super(MLP, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.ReLU(),
            nn.Linear(hidden_size // 4, output_size)  # Output: [battery, solar]
        )

    def forward(self, x):
        return self.model(x)

if __name__ == "__main__":
    batch_size = 64
    num_epochs = 2000
    model_name = "PV_Battery_Sizing"
    base_path = "."

    # Tensorboard writer
    writer = SummaryWriter(f"runs/{model_name}")

    # Load train and test data
    data_train = torch.tensor(torch.from_numpy(np.loadtxt(f'{base_path}/dataset/dataset_below_threshold_train.csv', delimiter=",")), dtype=torch.float32)
    data_test = torch.tensor(torch.from_numpy(np.loadtxt(f'{base_path}/dataset/dataset_below_threshold_test.csv', delimiter=",")), dtype=torch.float32)

    X_train = data_train[:, :-2]  # The last two columns are the optimal battery and solar sizings
    y_train = data_train[:, -2:]
    X_test = data_test[:, :-2]
    y_test = data_test[:, -2:]

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    joblib.dump(scaler, f"{base_path}/scaler_{model_name}.pkl")

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

    # Create dataset and dataloaders
    train_dataset = SizingDataset(X_train_tensor, y_train_tensor)
    test_dataset = SizingDataset(X_test_tensor, y_test_tensor)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    input_size = X_train.shape[1]  # Number of features

    # Initialize model, loss, and optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MLP(input_size=input_size).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

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

        # Check if the current loss is the best so far
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), f"{model_name}_best.pth")  # Save the best model
            print(f"Epoch {epoch+1}: New best model saved with loss {best_loss:.4f}")

        # Log the loss for TensorBoard
        writer.add_scalar("Loss/train", avg_loss, epoch)

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")

    writer.close()

    model.eval()
    with torch.no_grad():
        test_predictions = model(X_test_tensor.to(device)).cpu().numpy()

    torch.save(model.state_dict(), f"{model_name}.pth")
    print("Model saved successfully!")

    # Compute Mean Squared Error (MSE)
    mse = np.mean((test_predictions - y_test.numpy()) ** 2, axis=0)
    print(f"Test MSE for Battery: {mse[0]:.4f}, Solar: {mse[1]:.4f}")