"""
data_loader.py
--------------
Utilities for loading stock price data and pair CSVs.

Stock files live in:   <data_dir>/stock_csv/<ticker>.us.csv
                   or  <data_dir>/stock_txt/<ticker>.us.txt   (fallback)
"""

import os
import glob
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _find_stock_file(ticker: str, data_dir: str) -> str:
    """Return the path to a stock file, searching csv then txt subdirs."""
    ticker_lower = ticker.lower()
    for subdir in ("stock_csv", "stock_txt"):
        pattern = os.path.join(data_dir, subdir, f"{ticker_lower}.us.*")
        matches = glob.glob(pattern)
        if matches:
            return matches[0]
    raise FileNotFoundError(
        f"No data file found for ticker '{ticker}' under {data_dir}. "
        f"Tried stock_csv/ and stock_txt/."
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_stock(ticker: str, data_dir: str) -> pd.DataFrame:
    """
    Load a single stock and return a tidy DataFrame with columns:
        Date (datetime64), Close (float), Log_Close (float)

    Parameters
    ----------
    ticker   : ticker symbol, e.g. 'xom'
    data_dir : root dataset directory containing stock_csv/ and stock_txt/
    """
    path = _find_stock_file(ticker, data_dir)
    df = pd.read_csv(path)

    # Normalise column names (files have mixed capitalisation)
    df.columns = [c.strip() for c in df.columns]

    if "Date" not in df.columns:
        raise ValueError(f"'Date' column not found in {path}. Columns: {list(df.columns)}")
    if "Close" not in df.columns:
        raise ValueError(f"'Close' column not found in {path}. Columns: {list(df.columns)}")

    df["Date"] = pd.to_datetime(df["Date"])
    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
    df = df.dropna(subset=["Date", "Close"])
    df = df[df["Close"] > 0]
    df["Log_Close"] = np.log(df["Close"])

    return df[["Date", "Close", "Log_Close"]].sort_values("Date").reset_index(drop=True)


def load_pair(ticker1: str, ticker2: str, data_dir: str) -> pd.DataFrame:
    """
    Load two stocks and return a merged DataFrame on common dates with columns:
        Date, Log_Close_<TICKER1>, Log_Close_<TICKER2>

    Only rows where both tickers have data are kept (inner join).
    """
    s1 = load_stock(ticker1, data_dir).rename(columns={
        "Log_Close": f"Log_Close_{ticker1.upper()}",
        "Close":     f"Close_{ticker1.upper()}",
    })
    s2 = load_stock(ticker2, data_dir).rename(columns={
        "Log_Close": f"Log_Close_{ticker2.upper()}",
        "Close":     f"Close_{ticker2.upper()}",
    })

    merged = pd.merge(s1, s2, on="Date", how="inner").sort_values("Date").reset_index(drop=True)
    return merged


def load_pairs_csv(csv_path: str) -> pd.DataFrame:
    """
    Load a ranked pairs CSV (dbscan_pairs_ranked.csv or kmeans_pairs_ranked.csv).

    Expected columns: rank, cluster, stock_1, stock_2, crossings, half_life,
                      hedge_ratio, correlation

    Returns the DataFrame sorted by rank ascending.
    """
    df = pd.read_csv(csv_path)
    df.columns = [c.strip().lower() for c in df.columns]
    required = {"stock_1", "stock_2"}
    if not required.issubset(df.columns):
        raise ValueError(f"CSV must contain columns {required}. Found: {list(df.columns)}")
    if "rank" in df.columns:
        df = df.sort_values("rank").reset_index(drop=True)
    return df
