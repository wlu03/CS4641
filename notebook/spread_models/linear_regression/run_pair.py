"""
run_pair.py
-----------
CLI entry point: run the full pairs-trading pipeline for one or more pairs.

Usage examples
--------------
# Single pair
python run_pair.py --t1 xom --t2 cvx

# Top-N pairs from a ranked CSV
python run_pair.py --pairs_csv ../../../../dataset/kmeans_pairs_ranked.csv --top 5

# All pairs from DBSCAN CSV, custom date range
python run_pair.py --pairs_csv ../../../../dataset/dbscan_pairs_ranked.csv \
    --train_start 2008-01-01 --train_end 2013-12-31 --test_end 2015-12-31

# Save plots
python run_pair.py --t1 hon --t2 ups --save_dir ../../../../docs
"""

import argparse
import os
import sys
import warnings

import matplotlib
matplotlib.use("Agg")          # non-interactive backend for CLI use
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# Allow running from any cwd
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", "..", ".."))
sys.path.insert(0, os.path.join(_HERE, ".."))   # notebook/spread_models/ on path

from linear_regression.data_loader import load_pair, load_pairs_csv
from linear_regression.strategy import run_backtest, buy_and_hold
from linear_regression.metrics import print_metrics
from linear_regression.monte_carlo import (
    simulate_paths, test_3sigma_vs_bnh,
    print_3sigma_report, plot_mc_comparison, plot_realized_vs_bnh,
)


DATA_DIR = os.path.join(_ROOT, "dataset")


def run_single_pair(
    ticker1: str,
    ticker2: str,
    train_start: str,
    train_end:   str,
    test_end:    str,
    window:      int,
    entry_z:     float,
    exit_z:      float,
    tc:          float,
    n_sims:      int,
    n_days:      int,
    save_dir:    str = None,
    verbose:     bool = True,
) -> dict:
    """
    Full pipeline for a single pair. Returns a results dict.
    """
    pair_label = f"{ticker1.upper()}/{ticker2.upper()}"
    if verbose:
        print(f"\n{'─'*56}")
        print(f"  Running pair: {pair_label}")
        print(f"{'─'*56}")

    # ── 1. Load data ──────────────────────────────────────────────────────
    try:
        pair_df = load_pair(ticker1, ticker2, DATA_DIR)
    except FileNotFoundError as e:
        print(f"  [SKIP] {e}")
        return {"pair": pair_label, "error": str(e)}

    # ── 2. Backtest ───────────────────────────────────────────────────────
    try:
        bt = run_backtest(
            pair_df, ticker1, ticker2,
            train_start=train_start, train_end=train_end, test_end=test_end,
            window=window, entry_z=entry_z, exit_z=exit_z, tc=tc,
        )
    except ValueError as e:
        print(f"  [SKIP] {e}")
        return {"pair": pair_label, "error": str(e)}

    train_df   = bt["train"]["df"]
    test_df    = bt["test"]["df"] if bt["test"] else None
    col1, col2 = bt["col1"], bt["col2"]

    if verbose:
        coint = bt["coint"]
        flag  = "✓" if coint["cointegrated"] else "✗ (not cointegrated)"
        print(f"  Cointegration p-value: {coint['coint_pval']:.4f}  {flag}")
        print(f"  Hedge ratio β = {bt['beta']:.4f},  R² = {bt['r2']:.4f}")
        print()
        print_metrics(bt["train"]["metrics"])
        if test_df is not None:
            print()
            print_metrics(bt["test"]["metrics"])

    # ── 3. Buy-and-hold ───────────────────────────────────────────────────
    bnh_train = buy_and_hold(train_df, col1, col2,
                              label=f"Buy-and-Hold — Training")
    bnh_test  = buy_and_hold(test_df, col1, col2,
                              label=f"Buy-and-Hold — OOS") if test_df is not None else None

    if verbose:
        print()
        print_metrics(bnh_train["metrics"])
        if bnh_test:
            print()
            print_metrics(bnh_test["metrics"])

    # ── 4. Monte Carlo ────────────────────────────────────────────────────
    strat_sim = simulate_paths(train_df["Strategy_Return"].dropna(), n_sims=n_sims, n_days=n_days)
    bnh_sim   = simulate_paths(bnh_train["returns"].dropna(),        n_sims=n_sims, n_days=n_days)
    sigma_test = test_3sigma_vs_bnh(strat_sim, bnh_sim, n_sims=n_sims, n_days=n_days)

    if verbose:
        print_3sigma_report(sigma_test, pair_label)

    # ── 5. Plots ──────────────────────────────────────────────────────────
    safe_label = pair_label.replace("/", "_")

    mc_fig = plot_mc_comparison(
        strat_sim, bnh_sim, sigma_test,
        pair_label=pair_label, n_sims=n_sims, n_days=n_days,
        save_path=os.path.join(save_dir, f"mc_{safe_label}.png") if save_dir else None,
    )

    real_fig = plot_realized_vs_bnh(
        train_df, bnh_train["cum_returns"],
        pair_label=f"{pair_label} Training",
        save_path=os.path.join(save_dir, f"realized_{safe_label}_train.png") if save_dir else None,
    )

    if not save_dir:
        plt.show()

    plt.close("all")

    return {
        "pair":        pair_label,
        "backtest":    bt,
        "bnh_train":   bnh_train,
        "bnh_test":    bnh_test,
        "strat_sim":   strat_sim,
        "bnh_sim":     bnh_sim,
        "sigma_test":  sigma_test,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args():
    p = argparse.ArgumentParser(description="Pairs trading LR backtest + Monte Carlo")
    # Pair selection
    grp = p.add_mutually_exclusive_group(required=True)
    grp.add_argument("--t1",         help="Ticker 1 (use with --t2)")
    grp.add_argument("--pairs_csv",  help="Path to ranked pairs CSV")

    p.add_argument("--t2",           help="Ticker 2 (required when --t1 is used)")
    p.add_argument("--top",  type=int, default=5,
                   help="Number of top pairs to run from pairs CSV (default: 5)")
    # Date range
    p.add_argument("--train_start", default="2010-01-01")
    p.add_argument("--train_end",   default="2013-12-31")
    p.add_argument("--test_end",    default="2014-12-31")
    # Strategy params
    p.add_argument("--window",   type=int,   default=30)
    p.add_argument("--entry_z",  type=float, default=1.0)
    p.add_argument("--exit_z",   type=float, default=0.25)
    p.add_argument("--tc",       type=float, default=0.0001)
    # MC params
    p.add_argument("--n_sims",   type=int,   default=1000)
    p.add_argument("--n_days",   type=int,   default=252)
    # Output
    p.add_argument("--save_dir", default=None,
                   help="Directory to save plots (None = show interactively)")
    return p.parse_args()


def main():
    args = _parse_args()

    kwargs = dict(
        train_start=args.train_start,
        train_end=args.train_end,
        test_end=args.test_end,
        window=args.window,
        entry_z=args.entry_z,
        exit_z=args.exit_z,
        tc=args.tc,
        n_sims=args.n_sims,
        n_days=args.n_days,
        save_dir=args.save_dir,
    )

    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)

    if args.t1:
        if not args.t2:
            print("Error: --t2 is required when --t1 is used.")
            sys.exit(1)
        run_single_pair(args.t1, args.t2, **kwargs)
    else:
        pairs_df = load_pairs_csv(args.pairs_csv)
        pairs    = list(zip(pairs_df["stock_1"], pairs_df["stock_2"]))[:args.top]
        print(f"Running top {len(pairs)} pairs from {os.path.basename(args.pairs_csv)}")

        summary_rows = []
        for t1, t2 in pairs:
            result = run_single_pair(t1, t2, **kwargs)
            if "error" not in result:
                st = result["sigma_test"]
                summary_rows.append({
                    "pair":          result["pair"],
                    "coint_pval":    result["backtest"]["coint"]["coint_pval"],
                    "train_return":  result["backtest"]["train"]["metrics"]["total_return"],
                    "train_sharpe":  result["backtest"]["train"]["metrics"]["sharpe"],
                    "bnh_return":    result["bnh_train"]["metrics"]["total_return"],
                    "z_score":       st["z_score"],
                    "passes_3sigma": st["passes_3sigma"],
                })

        if summary_rows:
            import pandas as pd
            summary = pd.DataFrame(summary_rows)
            print(f"\n{'='*72}")
            print("  SUMMARY ACROSS ALL PAIRS")
            print(f"{'='*72}")
            print(summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
            n_pass = summary["passes_3sigma"].sum()
            print(f"\n  {n_pass}/{len(summary)} pairs pass the 3σ outperformance test.")


if __name__ == "__main__":
    main()
