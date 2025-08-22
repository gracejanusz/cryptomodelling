# lstm_crypto.py
import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import matplotlib.pyplot as plt

# -----------------------------
# Dataset utilities
# -----------------------------
class CryptoDataset(Dataset):
    def __init__(self, X_seq, y):
        self.X_seq = X_seq.astype(np.float32)
        self.y = y.astype(np.float32)

    def __len__(self):
        return len(self.X_seq)

    def __getitem__(self, idx):
        return self.X_seq[idx], self.y[idx]

def make_sequences(features_np, target_np, seq_len: int):
    """Build sliding windows of length seq_len to predict the next step."""
    X, y = [], []
    for i in range(seq_len, len(features_np)):
        X.append(features_np[i - seq_len:i, :])
        y.append(target_np[i])  # predict value at time i (next after the window)
    return np.stack(X), np.array(y).reshape(-1, 1)

# -----------------------------
# Model
# -----------------------------
class LSTMForecaster(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # x: [B, T, F]
        out, _ = self.lstm(x)        # out: [B, T, H]
        last = out[:, -1, :]         # [B, H]
        return self.fc(last)         # [B, 1]

# -----------------------------
# Cleaning Data
# -----------------------------

def load_and_clean(csv_path):
    df = pd.read_csv(csv_path)

    # Remove commas and convert numeric cols
    for col in ["Price", "Open", "High", "Low"]:
        df[col] = df[col].str.replace(",", "").astype(float)

    # Volume: convert "43.00K" → 43000, "1.2M" → 1,200,000
    def parse_vol(x):
        if isinstance(x, str):
            x = x.replace(",", "")
            if "K" in x:
                return float(x.replace("K", "")) * 1e3
            elif "M" in x:
                return float(x.replace("M", "")) * 1e6
            else:
                return float(x)
        return float(x)

    df["Vol."] = df["Vol."].apply(parse_vol)

    # Change %: strip %
    df["Change %"] = df["Change %"].str.replace("%", "").astype(float)

    # Date to datetime, ascending order
    df["Date"] = pd.to_datetime(df["Date"], format="%m/%d/%Y")
    df = df.sort_values("Date").reset_index(drop=True)

    return df

# -----------------------------
# Training & evaluation
# -----------------------------
def train_one_epoch(model, loader, device, criterion, optimizer):
    model.train()
    epoch_loss = 0.0
    for Xb, yb in loader:
        Xb = Xb.to(device)
        yb = yb.to(device)
        pred = model(Xb)
        loss = criterion(pred, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item() * Xb.size(0)
    return epoch_loss / len(loader.dataset)

@torch.no_grad()
def evaluate(model, loader, device, criterion):
    model.eval()
    epoch_loss = 0.0
    preds, trues = [], []
    for Xb, yb in loader:
        Xb = Xb.to(device)
        yb = yb.to(device)
        pred = model(Xb)
        loss = criterion(pred, yb)
        epoch_loss += loss.item() * Xb.size(0)
        preds.append(pred.cpu().numpy())
        trues.append(yb.cpu().numpy())
    preds = np.vstack(preds)
    trues = np.vstack(trues)
    return epoch_loss / len(loader.dataset), preds, trues

def inv_transform(arr_2d, scaler):
    """Inverse-transform a 2D target array using a fitted scaler (works for MinMax or Standard)."""
    # The scaler was fit on y (n,1), so transform expects 2D
    return scaler.inverse_transform(arr_2d)

def mape(y_true, y_pred):
    y_true = y_true.reshape(-1)
    y_pred = y_pred.reshape(-1)
    denom = np.maximum(np.abs(y_true), 1e-8)
    return np.mean(np.abs((y_true - y_pred) / denom)) * 100.0

# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser(description="LSTM RNN for cryptocurrency forecasting (PyTorch)")
    parser.add_argument("--csv", type=str, required=True, help="Path to CSV with at least Date and Close columns.")
    parser.add_argument("--date-col", type=str, default="Date", help="Date column name.")
    parser.add_argument("--symbol", type=str, default=None, help="Filter by this symbol if CSV has multiple coins.")
    parser.add_argument("--target", type=str, default="Close", help="Target column to forecast.")
    parser.add_argument("--features", type=str, nargs="*", default=None,
                        help="Feature columns to use (default will auto-pick common OHLCV if present).")
    parser.add_argument("--seq-len", type=int, default=14, help="Lookback window length (days).")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--train-split", type=float, default=0.8, help="Fraction of time-ordered data for training.")
    parser.add_argument("--normalize", choices=["standard", "minmax"], default="standard")
    parser.add_argument("--shuffle-train", action="store_true", help="Shuffle batches inside training split.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load CSV
    df = load_and_clean(args.csv)
    # Basic cleaning
    if args.date_col not in df.columns:
        raise ValueError(f"Date column '{args.date_col}' not found in CSV.")
    df[args.date_col] = pd.to_datetime(df[args.date_col], errors="coerce")
    df = df.dropna(subset=[args.date_col]).sort_values(args.date_col).reset_index(drop=True)

    if args.symbol:
        if "Symbol" not in df.columns:
            raise ValueError("No 'Symbol' column found to filter by --symbol.")
        df = df[df["Symbol"] == args.symbol].copy()
        if df.empty:
            raise ValueError(f"No rows found for symbol '{args.symbol}'.")

    # Select features
    default_candidates = ["Open", "High", "Low", "Close", "Volume"]
    if args.features is None:
        features = [c for c in default_candidates if c in df.columns]
        if args.target not in features and args.target in df.columns:
            features.append(args.target)
        if not features:
            raise ValueError("No feature columns found. Provide --features explicitly.")
    else:
        features = args.features
        missing = [c for c in features if c not in df.columns]
        if missing:
            raise ValueError(f"Features not in CSV: {missing}")

    if args.target not in df.columns:
        raise ValueError(f"Target column '{args.target}' not found in CSV.")

    # Keep only necessary columns
    keep_cols = [args.date_col] + features
    df = df[keep_cols].dropna().reset_index(drop=True)

    # Split by time
    n = len(df)
    if n < args.seq_len + 5:
        raise ValueError(f"Not enough rows ({n}) for seq-len={args.seq_len}.")
    split_idx = int(n * args.train_split)
    split_idx = max(split_idx, args.seq_len + 1)  # ensure at least one training sequence
    train_df = df.iloc[:split_idx].copy()
    test_df  = df.iloc[split_idx:].copy()

    # Scale features & target with training-only fit
    if args.normalize == "standard":
        feat_scaler = StandardScaler()
        tgt_scaler = StandardScaler()
    else:
        feat_scaler = MinMaxScaler()
        tgt_scaler = MinMaxScaler()

    X_train_feat = train_df[features].values
    y_train = train_df[[args.target]].values
    X_test_feat = test_df[features].values
    y_test = test_df[[args.target]].values

    X_train_scaled = feat_scaler.fit_transform(X_train_feat)
    y_train_scaled = tgt_scaler.fit_transform(y_train)
    X_test_scaled = feat_scaler.transform(X_test_feat)
    y_test_scaled = tgt_scaler.transform(y_test)

    # Build sequences
    Xtr_seq, ytr = make_sequences(X_train_scaled, y_train_scaled, seq_len=args.seq_len)
    # Important: the test sequences must be built only within the test block to avoid leakage
    Xte_seq, yte = make_sequences(X_test_scaled, y_test_scaled, seq_len=args.seq_len)

    train_ds = CryptoDataset(Xtr_seq, ytr)
    test_ds  = CryptoDataset(Xte_seq, yte)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=args.shuffle_train)
    test_loader  = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    # Model
    model = LSTMForecaster(
        input_size=len(features),
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Train
    best_val = math.inf
    best_state = None
    for epoch in range(1, args.epochs + 1):
        tr_loss = train_one_epoch(model, train_loader, device, criterion, optimizer)
        val_loss, _, _ = evaluate(model, test_loader, device, criterion)
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        print(f"Epoch {epoch:03d} | train MSE: {tr_loss:.6f} | valid MSE: {val_loss:.6f}")

    if best_state is not None:
        model.load_state_dict(best_state)

    # Evaluate & invert scale
    val_loss, y_pred_scaled, y_true_scaled = evaluate(model, test_loader, device, criterion)
    y_pred = inv_transform(y_pred_scaled, tgt_scaler).reshape(-1)
    y_true = inv_transform(y_true_scaled, tgt_scaler).reshape(-1)

    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    mae = np.mean(np.abs(y_true - y_pred))
    mape_val = mape(y_true, y_pred)

    print("\nTest set performance (original scale):")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE : {mae:.4f}")
    print(f"MAPE: {mape_val:.2f}%")

    # Build a date index for test predictions (align to the end of each test window)
    # Test_df length = Tt; after making sequences, we lose first seq_len rows.
    test_dates = test_df[args.date_col].iloc[args.seq_len:].reset_index(drop=True)
    out = pd.DataFrame({
        args.date_col: test_dates,
        "y_true": y_true,
        "y_pred": y_pred,
    })
    out_path = Path("predictions.csv")
    out.to_csv(out_path, index=False)
    print(f"\nSaved predictions to {out_path.resolve()}")

    # Plot
    plt.figure(figsize=(10, 5))
    plt.plot(out[args.date_col], out["y_true"], label="Actual")
    plt.plot(out[args.date_col], out["y_pred"], label="Predicted")
    plt.title("LSTM Forecast (Test Period)")
    plt.xlabel("Date")
    plt.ylabel(args.target)
    plt.legend()
    plt.tight_layout()
    fig_path = Path("pred_vs_actual.png")
    plt.savefig(fig_path, dpi=150)
    print(f"Saved plot to {fig_path.resolve()}")

    model_path = Path("lstm_crypto.pt")
    torch.save({
        "state_dict": model.state_dict(),
        "features": features,
        "target": args.target,
        "seq_len": args.seq_len,
        "feat_scaler": feat_scaler.__class__.__name__,
        "tgt_scaler": tgt_scaler.__class__.__name__,
        "feat_scaler_state": feat_scaler.__dict__,
        "tgt_scaler_state": tgt_scaler.__dict__,
        "hidden_size": args.hidden_size,
        "num_layers": args.num_layers,
        "dropout": args.dropout,
    }, model_path)
    print(f"Saved model to {model_path.resolve()}")

if __name__ == "__main__":
    main()
