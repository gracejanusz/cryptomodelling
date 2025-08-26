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
import matplotlib.dates as mdates

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
    if len(X) == 0:
        return np.empty((0, seq_len, features_np.shape[1])), np.empty((0, 1))
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
        out, _ = self.lstm(x)   # [B, T, H]
        last = out[:, -1, :]    # [B, H]
        return self.fc(last)    # [B, 1]

# -----------------------------
# Cleaning Data
# -----------------------------
def load_and_clean(csv_path):
    """
    Cleans CSVs of the form:
    "Date","Price","Open","High","Low","Vol.","Change %"
    with commas in numbers, K/M suffixes in volume, and trailing percent signs.
    """
    df = pd.read_csv(csv_path)

    # numeric with commas
    for col in ["Price", "Open", "High", "Low"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.replace(",", "", regex=False).astype(float)

    # volume: 43.00K -> 43000, 1.2M -> 1200000
    if "Vol." in df.columns:
        def parse_vol(x):
            s = str(x).replace(",", "")
            if s.endswith("K"):
                return float(s[:-1]) * 1e3
            if s.endswith("M"):
                return float(s[:-1]) * 1e6
            try:
                return float(s)
            except ValueError:
                return np.nan
        df["Vol."] = df["Vol."].apply(parse_vol)

    # change %: "-1.57%" -> -1.57
    if "Change %" in df.columns:
        df["Change %"] = (
            df["Change %"].astype(str)
            .str.replace("%", "", regex=False)
            .str.replace(",", "", regex=False)
        )
        df["Change %"] = pd.to_numeric(df["Change %"], errors="coerce")

    # date asc
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], format="%m/%d/%Y", errors="coerce")
        df = df.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)

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
    preds = np.vstack(preds) if preds else np.empty((0, 1))
    trues = np.vstack(trues) if trues else np.empty((0, 1))
    return epoch_loss / max(len(loader.dataset), 1), preds, trues

def inv_transform(arr_2d, scaler):
    """Inverse-transform a 2D target array using a fitted scaler."""
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
    parser.add_argument("--csv", type=str, required=True, help="Path to CSV with at least a Date and target column.")
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
    parser.add_argument("--train-split", type=float, default=0.8,
                        help="Fraction for time-based split (ignored if --test-start and --test-end are provided).")
    parser.add_argument("--normalize", choices=["standard", "minmax"], default="standard")
    parser.add_argument("--shuffle-train", action="store_true", help="Shuffle batches inside training split.")
    parser.add_argument("--test-start", type=str, default=None, help="Force test window start date (MM/DD/YYYY).")
    parser.add_argument("--test-end", type=str, default=None, help="Force test window end date (MM/DD/YYYY).")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load & clean CSV
    df = load_and_clean(args.csv)

    # Basic checks
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

    # Default feature candidates (supports both "Vol." and "Volume")
    default_candidates = ["Open", "High", "Low", "Close", "Price", "Vol.", "Volume", "Change %"]

    # Select features
    if args.features is None:
        features = [c for c in default_candidates if c in df.columns and c != args.target]
        if not features:
            raise ValueError("No feature columns found. Provide --features explicitly.")
    else:
        features = args.features
        missing = [c for c in features if c not in df.columns]
        if missing:
            raise ValueError(f"Features not in CSV: {missing}")

    if args.target not in df.columns:
        raise ValueError(f"Target column '{args.target}' not found in CSV.")

    # Keep only necessary columns (ensure target is present even if not in features)
    keep_cols = [args.date_col] + features
    if args.target not in keep_cols:
        keep_cols.append(args.target)
    df = df[keep_cols].dropna().reset_index(drop=True)

    # -----------------------------
    # Deterministic date-based split (if provided)
    # -----------------------------
    test_start_dt = None
    test_end_dt = None
    if args.test_start and args.test_end:
        try:
            test_start_dt = pd.to_datetime(args.test_start, format="%m/%d/%Y")
            test_end_dt = pd.to_datetime(args.test_end, format="%m/%d/%Y")
        except Exception as e:
            raise ValueError(f"Could not parse --test-start/--test-end: {e}")

        if test_end_dt < test_start_dt:
            raise ValueError("--test-end must be on/after --test-start")

        # Train: strictly before test_start; Test: between start and end inclusive
        train_df = df[df[args.date_col] < test_start_dt].copy()
        test_df  = df[(df[args.date_col] >= test_start_dt) & (df[args.date_col] <= test_end_dt)].copy()

        if test_df.empty:
            raise ValueError("Selected test window has 0 rows. Check dates or CSV.")
        if train_df.empty:
            raise ValueError("No training rows before test window. Extend your data or move the window backward.")

        # Save the exact test window rows used (cleaned)
        start_tag = test_start_dt.strftime("%Y%m%d")
        end_tag = test_end_dt.strftime("%Y%m%d")
        test_window_path = Path(f"test_window_{start_tag}_to_{end_tag}.csv")
        test_df.to_csv(test_window_path, index=False)
        print(f"Saved test window rows to {test_window_path.resolve()}")
    else:
        # -----------------------------
        # Original time-based split by fraction
        # -----------------------------
        n = len(df)
        if n < args.seq_len + 5:
            raise ValueError(f"Not enough rows ({n}) for seq-len={args.seq_len}.")

        split_idx = int(n * args.train_split)
        min_seq_rows = args.seq_len + 1
        split_idx = max(split_idx, min_seq_rows)  # training must have at least one sequence
        if n - split_idx < min_seq_rows:          # test must have at least one sequence
            split_idx = max(n - min_seq_rows, min_seq_rows)

        train_df = df.iloc[:split_idx].copy()
        test_df  = df.iloc[split_idx:].copy()

    # -----------------------------
    # Scaling (fit on train only)
    # -----------------------------
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

    # -----------------------------
    # Sequences
    # -----------------------------
    # Train sequences within train block
    Xtr_seq, ytr = make_sequences(X_train_scaled, y_train_scaled, seq_len=args.seq_len)

    # Test sequences **across the boundary**:
    # prepend last seq_len training rows as context so we can predict every test day
    if len(X_train_scaled) < args.seq_len:
        raise ValueError(
            f"Training split has only {len(X_train_scaled)} rows; need at least seq-len={args.seq_len} "
            "so the model can use train context to predict the first test day."
        )

    X_context = np.vstack([X_train_scaled[-args.seq_len:], X_test_scaled])
    y_context = np.vstack([y_train_scaled[-args.seq_len:], y_test_scaled])

    Xte_all, yte_all = make_sequences(X_context, y_context, seq_len=args.seq_len)

    # Keep exactly one sequence per test row (targets all lie in the test set)
    if len(Xte_all) >= len(test_df):
        Xte_seq, yte = Xte_all[-len(test_df):], yte_all[-len(test_df):]
    else:
        # Shouldn't happen, but guard anyway
        Xte_seq, yte = Xte_all, yte_all

    # Safety check
    if len(Xtr_seq) == 0 or len(Xte_seq) == 0:
        raise ValueError(
            f"After splitting and sequencing, got "
            f"{len(Xtr_seq)} train sequences and {len(Xte_seq)} test sequences. "
            f"Try a smaller --seq-len (e.g., 7) or adjust the test window."
        )

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

    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2)) if len(y_true) else float("nan")
    mae = np.mean(np.abs(y_true - y_pred)) if len(y_true) else float("nan")
    mape_val = mape(y_true, y_pred) if len(y_true) else float("nan")

    print("\nTest set performance (original scale):")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE : {mae:.4f}")
    print(f"MAPE: {mape_val:.2f}%")

    # One date per test prediction
    test_dates = test_df[args.date_col].reset_index(drop=True)
    out = pd.DataFrame({
        args.date_col: test_dates,
        "y_true": y_true,
        "y_pred": y_pred,
    })

    # Name outputs; if forced window was used, include it in filenames
    out_path = Path("predictions.csv")
    fig_path = Path("pred_vs_actual.png")
    if test_start_dt is not None and test_end_dt is not None:
        tag = f"{test_start_dt.strftime('%Y%m%d')}_to_{test_end_dt.strftime('%Y%m%d')}"
        out_path = Path(f"predictions_{tag}.csv")
        fig_path = Path(f"pred_vs_actual_{tag}.png")

    out.to_csv(out_path, index=False)
    print(f"\nSaved predictions to {out_path.resolve()}")

    # Plot (markers + tight daily ticks so short windows show up)
    plt.figure(figsize=(10, 5))
    plt.plot(out[args.date_col], out["y_true"], marker="o", label="Actual")
    plt.plot(out[args.date_col], out["y_pred"], marker="o", label="Predicted")
    ax = plt.gca()
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d"))
    plt.xticks(rotation=45)
    plt.xlim(out[args.date_col].min(), out[args.date_col].max())
    title_extra = ""
    if test_start_dt is not None and test_end_dt is not None:
        title_extra = f" [{test_start_dt.strftime('%m/%d/%Y')} - {test_end_dt.strftime('%m/%d/%Y')}]"
    plt.title(f"LSTM Forecast (Test Period){title_extra}")
    plt.xlabel("Date")
    plt.ylabel(args.target)
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_path, dpi=150)
    print(f"Saved plot to {fig_path.resolve()}")

    # Save model & scalers metadata
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
