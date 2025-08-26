#!/usr/bin/env python3
"""
Fetch daily Google Trends interest for a search term across long time ranges
by stitching overlapping <270-day windows and rescaling them, then save to CSV.

Default: term="bitcoin", start=2024-01-20, end=today (local machine date), geo="US".

Usage examples:
  python trends_daily.py                       # US only, default dates
  python trends_daily.py --geo ""              # Worldwide
  python trends_daily.py -k "bitcoin" -s 2024-01-20 -e 2025-08-24 --geo US

Requirements:
  pip install pytrends pandas python-dateutil
"""

import argparse
from datetime import date, datetime, timedelta
from dateutil.parser import isoparse
import time
import pandas as pd
from pytrends.request import TrendReq


def daterange(start: date, end: date):
    cur = start
    one_day = timedelta(days=1)
    while cur <= end:
        yield cur
        cur += one_day


def build_windows(start: date, end: date, window_days: int = 260, overlap_days: int = 30):
    assert window_days > overlap_days >= 7, "overlap should be at least one week"
    windows = []
    cur_start = start
    while cur_start <= end:
        cur_end = min(cur_start + timedelta(days=window_days - 1), end)
        windows.append((cur_start, cur_end))
        if cur_end == end:
            break
        # next window starts so that there is 'overlap_days' overlap with previous window's end
        next_start = cur_end - timedelta(days=overlap_days - 1)
        # avoid infinite loop
        if next_start <= cur_start:
            next_start = cur_start + timedelta(days=1)
        cur_start = next_start
    return windows


def fetch_daily(pytrends: TrendReq, keyword: str, start: date, end: date, geo: str):
    tf = f"{start.isoformat()} {end.isoformat()}"
    pytrends.build_payload([keyword], timeframe=tf, geo=geo)
    df = pytrends.interest_over_time()
    if df is None or df.empty:
        return pd.DataFrame(columns=["date", keyword])
    # Drop isPartial if present
    if 'isPartial' in df.columns:
        df = df.drop(columns=['isPartial'])
    df = df.reset_index().rename(columns={'date': 'date'})
    df['date'] = pd.to_datetime(df['date']).dt.tz_localize(None)
    # Keep only the keyword column
    return df[['date', keyword]]


def stitch_daily_series(pytrends: TrendReq, keyword: str, start: date, end: date, geo: str,
                        window_days: int = 260, overlap_days: int = 30, pause: float = 1.0) -> pd.DataFrame:
    windows = build_windows(start, end, window_days, overlap_days)
    stitched = None
    for i, (ws, we) in enumerate(windows):
        # backoff a bit to be polite
        time.sleep(pause)
        chunk = fetch_daily(pytrends, keyword, ws, we, geo)
        if chunk.empty:
            continue
        if stitched is None:
            stitched = chunk.copy()
        else:
            # Align via DATE overlap (last `overlap_days` of stitched vs any overlapping dates in the new chunk)
            prev_tail_start = stitched['date'].max() - timedelta(days=overlap_days - 1)
            prev_overlap = stitched[stitched['date'] >= prev_tail_start][['date', keyword]].copy()

            # Ensure chunk dates are datetime and tz-naive
            chunk = chunk.copy()
            chunk['date'] = pd.to_datetime(chunk['date']).dt.tz_localize(None)

            # Merge on dates to compute a robust scale factor
            merged = prev_overlap.merge(chunk[['date', keyword]], on='date', suffixes=('_prev', '_new'))
            if merged.empty:
                scale = 1.0
            else:
                valid = merged[merged[f'{keyword}_new'] > 0]
                if valid.empty:
                    scale = 1.0
                else:
                    ratio = valid[f'{keyword}_prev'] / valid[f'{keyword}_new']
                    scale = float(ratio.median())

            # Scale the entire new chunk and append only the non-overlapping tail
            chunk_scaled = chunk.copy()
            chunk_scaled[keyword] = (chunk_scaled[keyword] * scale).round(3)
            tail = chunk_scaled[chunk_scaled['date'] > stitched['date'].max()]
            stitched = pd.concat([stitched, tail], ignore_index=True)

    if stitched is None:
        return pd.DataFrame(columns=["date", keyword])

    # Normalize to 0..100 across the full stitched period (to match Trends semantics)
    mx = stitched[keyword].max()
    if mx and mx > 0:
        stitched[keyword] = (stitched[keyword] / mx * 100).round(2)

    # De-dup and sort by date just in case
    stitched = stitched.drop_duplicates(subset='date').sort_values('date').reset_index(drop=True)
    return stitched


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-k', '--keyword', default='bitcoin')
    parser.add_argument('-s', '--start', default='2024-01-20', help='YYYY-MM-DD')
    parser.add_argument('-e', '--end', default=date.today().isoformat(), help='YYYY-MM-DD')
    parser.add_argument('--geo', default='US', help='"US" for United States, "" for Worldwide, or any Google Trends geo code')
    parser.add_argument('--window-days', type=int, default=260)
    parser.add_argument('--overlap-days', type=int, default=30)
    parser.add_argument('--pause', type=float, default=1.0, help='seconds between requests')
    parser.add_argument('-o', '--out', default='google_trends_daily_bitcoin.csv')

    args = parser.parse_args()

    start_date = isoparse(args.start).date()
    end_date = isoparse(args.end).date()
    assert start_date <= end_date, 'start must be <= end'

    print(f"Fetching daily Google Trends for '{args.keyword}' from {start_date} to {end_date} (geo='{args.geo}')…")
    pytrends = TrendReq(hl='en-US', tz=-300)  # US Central Daylight Time (UTC-5). Use tz=0 for UTC if preferred.  # US Central offset = UTC-6; tz value is minutes offset from UTC

    # Google often returns weekly data for long ranges; our stitching uses <270-day windows to force daily
    df = stitch_daily_series(pytrends, args.keyword, start_date, end_date, args.geo,
                             window_days=args.window_days, overlap_days=args.overlap_days, pause=args.pause)

    if df.empty:
        print("No data returned. Try a shorter range, different geo, or check connectivity.")
        return

    df.to_csv(args.out, index=False)
    print(f"Saved {len(df):,} rows to {args.out}")


if __name__ == '__main__':
    main()
