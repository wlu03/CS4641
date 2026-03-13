"""
strategy.py
-----------
Pairs-trading logic: cointegration, hedge ratio, signal generation, backtest.

All functions are stateless — pass data in, get results out.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

# Older statsmodels (< 0.13) shipped a broken deprecate_kwarg decorator that
# requires 'new_arg_name' but some internal call sites omit it, causing a
# TypeError on module import.  Patch it before importing stattools.
try:
    import statsmodels.compat.pandas as _sm_compat
    _sm_compat.deprecate_kwarg("_probe", "_probe")   # will raise if broken
except TypeError:
    def _noop_deprecate_kwarg(old_arg_name, new_arg_name=None, **kwargs):
        def decorator(func):
            return func
        return decorator
    _sm_compat.deprecate_kwarg = _noop_deprecate_kwarg
except Exception:
    pass

from statsmodels.tsa.stattools import coint, adfuller

from .metrics import summary_metrics, TRADING_DAYS

# ---------------------------------------------------------------------------
# Cointegration / stationarity
# ---------------------------------------------------------------------------

def check_cointegration(
    train: pd.DataFrame,
    col1: str,
    col2: str,
) -> dict:
    """
    Run ADF on each series and Engle-Granger cointegration test.

    Returns a dict with keys:
        adf_1, adf_pval_1, adf_2, adf_pval_2,
        coint_stat, coint_pval, cointegrated (bool)
    """
    def _adf(series):
        stat, pval, *_ = adfuller(series.dropna(), autolag="AIC")
        return float(stat), float(pval)

    stat1, p1 = _adf(train[col1])
    stat2, p2 = _adf(train[col2])
    t_stat, p_coint, crit = coint(train[col1].dropna(), train[col2].dropna())

    return {
        "adf_stat_1":   stat1,
        "adf_pval_1":   p1,
        "adf_stat_2":   stat2,
        "adf_pval_2":   p2,
        "coint_stat":   float(t_stat),
        "coint_pval":   float(p_coint),
        "coint_crit":   crit,
        "cointegrated": p_coint < 0.05,
    }


# ---------------------------------------------------------------------------
# Hedge ratio
# ---------------------------------------------------------------------------

def fit_hedge_ratio(train: pd.DataFrame, col_x: str, col_y: str):
    """
    OLS: col_y = beta * col_x + alpha

    Returns (alpha, beta, r2) estimated on `train` only.
    """
    X = train[col_x].values.reshape(-1, 1)
    y = train[col_y].values
    model = LinearRegression().fit(X, y)
    return float(model.intercept_), float(model.coef_[0]), float(model.score(X, y))


# ---------------------------------------------------------------------------
# Spread & z-score
# ---------------------------------------------------------------------------

def compute_spread(df: pd.DataFrame, col_x: str, col_y: str, alpha: float, beta: float) -> pd.Series:
    """Residual spread: col_y - (beta * col_x + alpha)."""
    return df[col_y] - (beta * df[col_x] + alpha)


def compute_zscore(spread: pd.Series, window: int) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Return (rolling_mean, rolling_std, z_score)."""
    mean = spread.rolling(window).mean()
    std  = spread.rolling(window).std()
    z    = (spread - mean) / std
    return mean, std, z


# ---------------------------------------------------------------------------
# Position state machine
# ---------------------------------------------------------------------------

def generate_positions(zscore: pd.Series, entry_z: float, exit_z: float) -> np.ndarray:
    """
    Causal position state machine.

    +1 = long spread  (long stock_y, short stock_x)
    -1 = short spread (short stock_y, long stock_x)
     0 = flat
    """
    z = zscore.values
    positions = np.zeros(len(z))
    current = 0

    for i in range(len(z)):
        if np.isnan(z[i]):
            positions[i] = 0
            continue
        if current == 0:
            if z[i] < -entry_z:
                current = 1
            elif z[i] > entry_z:
                current = -1
        elif current == 1 and z[i] > -exit_z:
            current = 0
        elif current == -1 and z[i] < exit_z:
            current = 0
        positions[i] = current

    return positions


# ---------------------------------------------------------------------------
# Backtest
# ---------------------------------------------------------------------------

def _backtest_period(
    df: pd.DataFrame,
    col_x: str,
    col_y: str,
    alpha: float,
    beta: float,
    window: int,
    entry_z: float,
    exit_z: float,
    tc: float,
    rf_daily: float,
    label: str,
) -> dict:
    """Run backtest on a single period DataFrame; return metrics + annotated df."""
    df = df.copy()
    df["Spread"]       = compute_spread(df, col_x, col_y, alpha, beta)
    mean, std, z       = compute_zscore(df["Spread"], window)
    df["Rolling_Mean"] = mean
    df["Rolling_Std"]  = std
    df["Z_Score"]      = z
    df["Position"]     = generate_positions(df["Z_Score"], entry_z, exit_z)
    df["Spread_Return"]   = df["Spread"].diff()
    df["Trades"]          = df["Position"].diff().abs()
    df["Strategy_Return"] = df["Position"].shift(1) * df["Spread_Return"] - df["Trades"] * tc
    df["Cum_Return"]      = (1 + df["Strategy_Return"].fillna(0)).cumprod()

    n_trades = int(df["Trades"].sum())
    m = summary_metrics(df["Strategy_Return"].dropna(), df["Cum_Return"], n_trades, rf_daily, label)
    return {"metrics": m, "df": df}


def run_backtest(
    pair_df: pd.DataFrame,
    ticker1: str,
    ticker2: str,
    train_start: str = "2010-01-01",
    train_end:   str = "2013-12-31",
    test_end:    str = "2014-12-31",
    window:  int   = 30,
    entry_z: float = 1.0,
    exit_z:  float = 0.25,
    tc:      float = 0.0001,
    rf_daily: float = 0.00001,
) -> dict:
    """
    Full train/test backtest for a pair.

    Parameters
    ----------
    pair_df  : merged DataFrame from load_pair()
    ticker1  : name of stock_x (independent variable in OLS)
    ticker2  : name of stock_y (dependent variable in OLS)

    Returns
    -------
    dict with keys:
        train_result, test_result,
        alpha, beta, r2,
        coint_results,
        ticker1, ticker2
    """
    col1 = f"Log_Close_{ticker1.upper()}"
    col2 = f"Log_Close_{ticker2.upper()}"

    train = pair_df[(pair_df["Date"] >= train_start) & (pair_df["Date"] <= train_end)].copy()
    test  = pair_df[(pair_df["Date"] >  train_end)   & (pair_df["Date"] <= test_end)].copy()

    if len(train) < window + 10:
        raise ValueError(f"Training window too short: {len(train)} rows for pair {ticker1}/{ticker2}")

    coint_results = check_cointegration(train, col1, col2)
    alpha, beta, r2 = fit_hedge_ratio(train, col1, col2)

    train_result = _backtest_period(
        train, col1, col2, alpha, beta, window, entry_z, exit_z, tc, rf_daily,
        label=f"{ticker1.upper()}/{ticker2.upper()} — Training ({train_start[:4]}–{train_end[:4]})"
    )
    test_result = _backtest_period(
        test, col1, col2, alpha, beta, window, entry_z, exit_z, tc, rf_daily,
        label=f"{ticker1.upper()}/{ticker2.upper()} — OOS ({int(train_end[:4])+1}–{test_end[:4]})"
    ) if len(test) > 0 else None

    return {
        "ticker1":       ticker1,
        "ticker2":       ticker2,
        "alpha":         alpha,
        "beta":          beta,
        "r2":            r2,
        "coint":         coint_results,
        "train":         train_result,
        "test":          test_result,
        "col1":          col1,
        "col2":          col2,
    }


# ---------------------------------------------------------------------------
# Buy-and-hold benchmark
# ---------------------------------------------------------------------------

def buy_and_hold(
    df: pd.DataFrame,
    col1: str,
    col2: str,
    rf_daily: float = 0.00001,
    label: str = "Buy-and-Hold",
) -> dict:
    """
    Equal-weight long position in both stocks.
    Returns {"metrics": ..., "returns": pd.Series, "cum_returns": pd.Series}
    """
    df = df.copy()
    ret = 0.5 * df[col1].diff() + 0.5 * df[col2].diff()
    cum = (1 + ret.fillna(0)).cumprod()
    m   = summary_metrics(ret.dropna(), cum, n_trades=0, rf_daily=rf_daily, label=label)
    return {"metrics": m, "returns": ret, "cum_returns": cum}
