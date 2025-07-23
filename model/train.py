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
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from MLP import preprocess, MLP_Branched, SizingDataset


if __name__ == "__main__":
    model_name = "buildsys"
    batch_size = 64
    num_epochs = 200
    dataset_path = "./dataset/out"

    # Initialize tensorboard writer
    writer = SummaryWriter(f"runs/{model_name}")

    # Load and process data
    df_test = pd.read_csv(f'{dataset_path}/dataset_test.csv', header=None)
    test_traces_df, test_meta_df, y_test_df = preprocess(df_test)

    df_train = pd.read_csv(f'{dataset_path}/dataset_train.csv', header=None)
    train_traces_df, train_meta_df, y_train_df = preprocess(df_train)

    df_val = pd.read_csv(f'{dataset_path}/dataset_val.csv', header=None)
    val_traces_df, val_meta_df, y_val_df = preprocess(df_val)

    test_meta_df = test_meta_df.reindex(columns=train_meta_df.columns, fill_value=0)
    val_meta_df = val_meta_df.reindex(columns=train_meta_df.columns, fill_value=0)

    # Create scalers and scale input
    scaler_ts = StandardScaler()
    scaler_meta = StandardScaler()

    train_traces = scaler_ts.fit_transform(train_traces_df)
    test_traces = scaler_ts.transform(test_traces_df)
    val_traces = scaler_ts.transform(val_traces_df)

    train_meta = scaler_meta.fit_transform(train_meta_df)
    test_meta = scaler_meta.transform(test_meta_df)
    val_meta = scaler_meta.transform(val_meta_df)

    joblib.dump(scaler_ts, f"scaler_ts_{model_name}.pkl")
    joblib.dump(scaler_meta, f"scaler_meta_{model_name}.pkl")

    # Create input tensors
    X_train_tensor = torch.tensor(train_traces, dtype=torch.float32)
    M_train_tensor = torch.tensor(train_meta, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train_df, dtype=torch.float32)

    X_test_tensor = torch.tensor(test_traces, dtype=torch.float32)
    M_test_tensor = torch.tensor(test_meta, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test_df, dtype=torch.float32)

    X_val_tensor = torch.tensor(val_traces, dtype=torch.float32)
    M_val_tensor = torch.tensor(val_meta, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val_df, dtype=torch.float32)

    train_loader = DataLoader(SizingDataset(X_train_tensor, M_train_tensor, y_train_tensor), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(SizingDataset(X_val_tensor, M_val_tensor, y_val_tensor), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(SizingDataset(X_test_tensor, M_test_tensor, y_test_tensor), batch_size=batch_size, shuffle=False)

    # Initialize model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MLP_Branched(ts_input_len=train_traces.shape[1], meta_input_len=train_meta.shape[1]).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0005, weight_decay=0.0001)

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

    # Test model
    model.load_state_dict(torch.load(f"{model_name}_best.pth"))
    model.eval()
    all_preds = []
    with torch.no_grad():
        for X_batch, M_batch, _ in test_loader:
            preds = model(X_batch.to(device), M_batch.to(device))
            all_preds.append(preds.cpu())
    test_predictions = torch.cat(all_preds).numpy()

    # Get metrics
    mse = np.mean((test_predictions - y_test_tensor.numpy()) ** 2, axis=0)
    print(f"Test MSE: Battery: {mse[0]:.4f}, Solar: {mse[1]:.4f}")
    mae = np.mean(np.abs(test_predictions - y_test_tensor.numpy()), axis=0)
    print(f"Test MAE: Battery: {mae[0]:.4f}, Solar: {mae[1]:.4f}")
