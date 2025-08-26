#!/usr/bin/env python3
"""
Correlate Google Trends daily interest with BTC prices, and plot:
  1) Scatter with best-fit line
  2) Time-series line chart (price + trends)

Example:
  python btc_trends_correlation.py \
    --prices ./coin-data/BitcoinHistoricalData.csv \
    --trends ./bitcoin_us_daily_trends.csv \
    --price-date-col Date \
    --trends-date-col date \
    --price-col High \
    --trends-col bitcoin \
    --out-plot ./out/search_vs_high_scatter.png \
    --out-csv  ./out/merged_btc_high_vs_trends.csv \
    --out-line-plot ./out/price_vs_trends_line.png
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def coerce_numeric(series: pd.Series) -> pd.Series:
    return (series.astype(str)
            .str.replace(',', '', regex=False)
            .str.replace('%', '', regex=False)
            .str.strip()
            .replace({'': np.nan})
            .astype(float))

def read_prices(path: str, date_col: str, price_col: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if date_col not in df.columns:
        raise ValueError(f"Price date column '{date_col}' not found. Available: {list(df.columns)}")
    if price_col not in df.columns:
        raise ValueError(f"Price column '{price_col}' not found. Available: {list(df.columns)}")

    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    if df[date_col].isna().all():
        raise ValueError(f"Could not parse any dates in '{date_col}' from {path}")

    df[price_col] = coerce_numeric(df[price_col])

    out = df[[date_col, price_col]].rename(
        columns={date_col: 'date', price_col: 'price_value'}
    ).copy()
    out['date'] = pd.to_datetime(out['date']).dt.tz_localize(None).dt.normalize()
    out = out.dropna(subset=['date', 'price_value'])
    return out

def read_trends(path: str, date_col: str | None, trends_col: str | None) -> pd.DataFrame:
    df = pd.read_csv(path)

    # Auto-detect date column if not provided
    if date_col is None:
        if 'date' in df.columns:
            date_col = 'date'
        elif 'Date' in df.columns:
            date_col = 'Date'
        else:
            date_col = None
            for c in df.columns:
                try:
                    pd.to_datetime(df[c])
                    date_col = c
                    break
                except Exception:
                    continue
            if date_col is None:
                raise ValueError("Could not auto-detect date column in Trends CSV. Pass --trends-date-col.")

    # Auto-detect trends value column if not provided
    if trends_col is None:
        candidates = [c for c in df.columns if c.lower() in ('bitcoin', 'search_interest', 'value')]
        if not candidates:
            candidates = [c for c in df.columns if c != date_col]
            if not candidates:
                raise ValueError("Could not auto-detect trends value column. Pass --trends-col.")
        trends_col = candidates[0]

    if date_col not in df.columns:
        raise ValueError(f"Trends date column '{date_col}' not found. Available: {list(df.columns)}")
    if trends_col not in df.columns:
        raise ValueError(f"Trends value column '{trends_col}' not found. Available: {list(df.columns)}")

    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    if df[date_col].isna().all():
        raise ValueError(f"Could not parse any dates in '{date_col}' from {path}")

    df[trends_col] = pd.to_numeric(df[trends_col], errors='coerce')

    out = df[[date_col, trends_col]].rename(
        columns={date_col: 'date', trends_col: 'search_interest'}
    ).copy()
    out['date'] = pd.to_datetime(out['date']).dt.tz_localize(None).dt.normalize()
    out = out.dropna(subset=['date', 'search_interest'])
    return out

def ensure_parent(path: Path) -> None:
    if path.parent and not path.parent.exists():
        path.parent.mkdir(parents=True, exist_ok=True)

def main():
    ap = argparse.ArgumentParser(description="Plot correlation between Google Trends interest and BTC prices.")
    ap.add_argument('--prices', required=True, help="Path to BTC prices CSV")
    ap.add_argument('--trends', required=True, help="Path to Google Trends daily CSV")
    ap.add_argument('--price-date-col', default='Date', help="Date column in prices CSV (default: Date)")
    ap.add_argument('--trends-date-col', default=None, help="Date column in Trends CSV (default: auto-detect)")
    ap.add_argument('--price-col', default='High', help="Numeric price column to correlate (default: High)")
    ap.add_argument('--trends-col', default=None, help="Search interest column in Trends CSV (default: auto-detect)")
    ap.add_argument('--out-plot', default='./search_vs_price_scatter.png', help="Output scatter plot path (PNG)")
    ap.add_argument('--out-csv',  default='./merged_btc_vs_trends.csv', help="Output merged CSV path")
    ap.add_argument('--out-line-plot', default=None, help="(Optional) Output time-series line plot path (PNG)")
    ap.add_argument('--normalize-lines', action='store_true',
                    help="If set, normalizes both series to 0–100 and draws on one axis; otherwise uses dual y-axes.")
    ap.add_argument('--title', default=None, help="Custom plot title")
    args = ap.parse_args()

    prices = read_prices(args.prices, args.price_date_col, args.price_col)
    trends = read_trends(args.trends, args.trends_date_col, args.trends_col)

    merged = prices.merge(trends, on='date', how='inner')
    merged = merged.sort_values('date').dropna(subset=['price_value', 'search_interest'])

    if len(merged) < 2:
        raise SystemExit("Not enough overlapping dates to compute correlation.")

    # ---- Scatter + best-fit line ----
    corr = merged['search_interest'].corr(merged['price_value'])
    x = merged['search_interest'].to_numpy()
    y = merged['price_value'].to_numpy()

    plt.figure()
    plt.scatter(x, y)
    if not np.allclose(np.var(x), 0):
        coeffs = np.polyfit(x, y, deg=1)
        x_line = np.linspace(x.min(), x.max(), 200)
        y_line = coeffs[0] * x_line + coeffs[1]
        plt.plot(x_line, y_line)
        line_note = ""
    else:
        line_note = " (no fit: constant X)"

    title = args.title or f"Google Searches vs BTC {args.price_col} (r = {corr:.3f}){line_note}"
    plt.title(title)
    plt.xlabel("Google Trends Daily Interest")
    plt.ylabel(f"BTC {args.price_col} (USD)")

    out_plot = Path(args.out_plot)
    ensure_parent(out_plot)
    plt.savefig(out_plot, bbox_inches='tight')
    plt.close()

    # ---- Time-series line chart (two lines) ----
    if args.out_line_plot:
        out_line = Path(args.out_line_plot)
        ensure_parent(out_line)

        if args.normalize_lines:
            # Single-axis plot, both normalized to 0–100
            fig, ax = plt.subplots()
            price_norm = (merged['price_value'] / merged['price_value'].max()) * 100.0
            trends_norm = merged['search_interest']  # already 0–100

            ax.plot(merged['date'], price_norm, label=f"BTC {args.price_col} (norm)", color="tab:blue")
            ax.plot(merged['date'], trends_norm, label="Google Trends", color="tab:orange")

            ax.set_ylabel("Normalized (0–100)")
            ax.set_xlabel("Date")
            ax.set_title("BTC Price vs Google Trends (Daily)")
            ax.legend()
            fig.autofmt_xdate()
            plt.savefig(out_line, bbox_inches='tight')
            plt.close()

        else:
            # Dual y-axes with distinct colors for each series
            fig, ax = plt.subplots()

            l1, = ax.plot(merged['date'], merged['price_value'],
                        label=f"BTC {args.price_col}", color="tab:blue")
            ax.set_ylabel(f"BTC {args.price_col} (USD)", color="tab:blue")
            ax.tick_params(axis='y', labelcolor="tab:blue")
            ax.set_xlabel("Date")
            ax.set_title("BTC Price vs Google Trends (Daily)")

            ax2 = ax.twinx()
            l2, = ax2.plot(merged['date'], merged['search_interest'],
                        label="Google Trends", color="tab:orange")
            ax2.set_ylabel("Google Trends Interest (0–100)", color="tab:orange")
            ax2.tick_params(axis='y', labelcolor="tab:orange")

            # Combined legend
            lines = [l1, l2]
            labels = [l.get_label() for l in lines]
            ax.legend(lines, labels, loc="upper left")

            fig.autofmt_xdate()
            plt.savefig(out_line, bbox_inches='tight')
            plt.close()


    # ---- Save merged data ----
    out_csv = Path(args.out_csv)
    ensure_parent(out_csv)
    merged.to_csv(out_csv, index=False)

    print(f"Correlation (Pearson r): {corr:.4f}")
    print(f"Saved scatter to: {out_plot}")
    if args.out_line_plot:
        print(f"Saved line plot to: {out_line}")
    print(f"Saved merged data to: {out_csv}")

if __name__ == "__main__":
    main()
