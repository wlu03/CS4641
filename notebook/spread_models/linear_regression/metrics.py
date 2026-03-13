"""
metrics.py
----------
Pure-function performance metrics. No side effects, no plotting.
"""

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def sharpe(returns: pd.Series, rf_daily: float = 0.00001) -> float:
    """Annualised Sharpe ratio."""
    excess = returns.dropna() - rf_daily
    std = excess.std()
    if std == 0 or np.isnan(std):
        return np.nan
    return excess.mean() / std * np.sqrt(TRADING_DAYS)


def max_drawdown(cum_returns: pd.Series) -> float:
    """Maximum drawdown (negative number, e.g. -0.12 means -12%)."""
    rolling_max = cum_returns.cummax()
    dd = (cum_returns - rolling_max) / rolling_max
    return float(dd.min())


def win_rate(returns: pd.Series) -> float:
    """Fraction of non-zero trading days that were positive."""
    active = returns[returns != 0].dropna()
    if len(active) == 0:
        return np.nan
    return float((active > 0).mean())


def summary_metrics(
    returns: pd.Series,
    cum_returns: pd.Series,
    n_trades: int,
    rf_daily: float = 0.00001,
    label: str = "",
) -> dict:
    """
    Aggregate all metrics into a dict.

    Keys: label, total_return, sharpe, max_drawdown, win_rate, n_trades
    """
    total_ret = float(cum_returns.iloc[-1] - 1)
    return {
        "label":        label,
        "total_return": total_ret,
        "sharpe":       sharpe(returns, rf_daily),
        "max_drawdown": max_drawdown(cum_returns),
        "win_rate":     win_rate(returns),
        "n_trades":     n_trades,
    }


def print_metrics(m: dict) -> None:
    width = 42
    title = m.get("label", "Performance")
    print("=" * width)
    print(f"  {title}")
    print("=" * width)
    print(f"  Cumulative Return  : {m['total_return']*100:>8.2f}%")
    print(f"  Sharpe Ratio       : {m['sharpe']:>8.2f}")
    print(f"  Max Drawdown       : {m['max_drawdown']*100:>8.2f}%")
    wr = m.get("win_rate")
    if wr is not None and not np.isnan(wr):
        print(f"  Win Rate           : {wr*100:>8.1f}%")
    print(f"  Total Trades       : {m['n_trades']:>8d}")
