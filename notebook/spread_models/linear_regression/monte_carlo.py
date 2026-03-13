"""
monte_carlo.py
--------------
Monte Carlo simulation for the pairs strategy and a 3σ significance test
against the buy-and-hold benchmark.

Statistical framing for the 3σ test
------------------------------------
Under the null hypothesis that strategy and buy-and-hold share the same
expected terminal wealth, we ask:

    z = (mean_terminal_strat - mean_terminal_bnh) / std_terminal_bnh

If z >= 3 the strategy sits at least 3 standard deviations above the
buy-and-hold distribution → reject H0 at the ~0.13% significance level.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------

def simulate_paths(
    daily_returns: pd.Series,
    n_sims: int = 1_000,
    n_days: int = 252,
    seed: int = 42,
) -> dict:
    """
    Parametric Monte Carlo: draw from N(mu, sigma) of the historical returns.

    Returns
    -------
    dict with keys:
        sim_cum   : (n_sims, n_days) cumulative wealth array
        terminal  : (n_sims,) terminal wealth
        mu, sigma : fitted parameters
        percentiles : dict  {5, 25, 50, 75, 95} → (n_days,) arrays
    """
    rng = np.random.default_rng(seed)
    clean = daily_returns.dropna()
    mu    = float(clean.mean())
    sigma = float(clean.std())

    draws   = rng.normal(mu, sigma, size=(n_sims, n_days))
    sim_cum = np.cumprod(1 + draws, axis=1)

    pcts = {q: np.percentile(sim_cum, q, axis=0) for q in (5, 25, 50, 75, 95)}

    return {
        "sim_cum":    sim_cum,
        "terminal":   sim_cum[:, -1],
        "mu":         mu,
        "sigma":      sigma,
        "percentiles": pcts,
    }


# ---------------------------------------------------------------------------
# 3σ significance test
# ---------------------------------------------------------------------------

def test_3sigma_vs_bnh(
    strat_result: dict,
    bnh_result:   dict,
    n_sims: int = 1_000,
    n_days: int = 252,
    seed: int   = 42,
) -> dict:
    """
    Test whether the pairs strategy terminal wealth is ≥ 3σ above buy-and-hold.

    Parameters
    ----------
    strat_result : output of simulate_paths() for the strategy
    bnh_result   : output of simulate_paths() for buy-and-hold

    Returns
    -------
    dict with keys:
        z_score          : (mean_strat - mean_bnh) / std_bnh
        passes_3sigma    : bool, z_score >= 3
        mean_strat       : mean terminal wealth of strategy
        mean_bnh         : mean terminal wealth of BnH
        std_bnh          : std of BnH terminal wealth
        prob_outperform  : fraction of strategy paths that beat BnH median
        t_stat, t_pval   : Welch's t-test on terminal wealth distributions
    """
    t_strat = strat_result["terminal"]
    t_bnh   = bnh_result["terminal"]

    mean_s = float(np.mean(t_strat))
    mean_b = float(np.mean(t_bnh))
    std_b  = float(np.std(t_bnh))

    z = (mean_s - mean_b) / std_b if std_b > 0 else np.inf
    prob_beat = float((t_strat > np.median(t_bnh)).mean())

    t_stat, t_pval = stats.ttest_ind(t_strat, t_bnh, equal_var=False)

    return {
        "z_score":       z,
        "passes_3sigma": z >= 3.0,
        "mean_strat":    mean_s,
        "mean_bnh":      mean_b,
        "std_bnh":       std_b,
        "prob_outperform": prob_beat,
        "t_stat":        float(t_stat),
        "t_pval":        float(t_pval),
    }


def print_3sigma_report(test_result: dict, pair_label: str = "") -> None:
    r = test_result
    label = f" — {pair_label}" if pair_label else ""
    print(f"\n{'='*56}")
    print(f"  3σ OUTPERFORMANCE TEST{label}")
    print(f"{'='*56}")
    print(f"  Mean terminal wealth (strategy)  : {r['mean_strat']:.4f}")
    print(f"  Mean terminal wealth (BnH)       : {r['mean_bnh']:.4f}")
    print(f"  Std of BnH terminal wealth       : {r['std_bnh']:.4f}")
    print(f"  Z-score vs BnH                   : {r['z_score']:+.2f}σ")
    verdict = "PASSES ✓  (strategy is ≥3σ above BnH)" if r["passes_3sigma"] \
              else f"FAILS  ✗  (need z≥3, got {r['z_score']:.2f})"
    print(f"  3σ threshold                     : {verdict}")
    print(f"  P(strategy > BnH median)         : {r['prob_outperform']*100:.1f}%")
    print(f"  Welch t-test  t={r['t_stat']:.2f}, p={r['t_pval']:.4f}")


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_mc_comparison(
    strat_sim:   dict,
    bnh_sim:     dict,
    sigma_test:  dict,
    pair_label:  str  = "",
    n_sims:      int  = 1_000,
    n_days:      int  = 252,
    save_path:   str  = None,
) -> plt.Figure:
    """
    Three-panel figure:
      Left   – fan chart for pairs strategy
      Centre – fan chart for buy-and-hold
      Right  – overlaid terminal wealth distributions with 3σ marker
    """
    days = np.arange(1, n_days + 1)
    fig  = plt.figure(figsize=(18, 6))
    gs   = gridspec.GridSpec(1, 3, figure=fig, wspace=0.35)

    # ---- helper: draw a fan chart ----
    def _fan(ax, sim_result, color, title):
        p = sim_result["percentiles"]
        ax.fill_between(days, p[5],  p[95], alpha=0.12, color=color, label="5–95th pctile")
        ax.fill_between(days, p[25], p[75], alpha=0.28, color=color, label="25–75th pctile")
        ax.plot(days, p[50], color=color, lw=2, label="Median")
        ax.axhline(1.0, color="black", linestyle="--", alpha=0.4, lw=1, label="Break-even")
        ax.set_title(f"{title}\n({n_sims:,} paths, {n_days}d)", fontsize=11)
        ax.set_xlabel("Trading Day")
        ax.set_ylabel("Cumulative Wealth")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.25)

    _fan(fig.add_subplot(gs[0]), strat_sim, "navy",       f"Pairs Strategy\n{pair_label}")
    _fan(fig.add_subplot(gs[1]), bnh_sim,   "darkorange",  "Buy-and-Hold\n(equal-weight)")

    # ---- terminal wealth distributions ----
    ax3 = fig.add_subplot(gs[2])
    t_s = strat_sim["terminal"]
    t_b = bnh_sim["terminal"]

    # KDE curves
    xs = np.linspace(min(t_s.min(), t_b.min()), max(t_s.max(), t_b.max()), 300)
    ax3.plot(xs, stats.gaussian_kde(t_s)(xs), color="navy",       lw=2, label="Pairs Strategy")
    ax3.plot(xs, stats.gaussian_kde(t_b)(xs), color="darkorange",  lw=2, label="Buy-and-Hold")

    # BnH mean ± 1σ / 2σ / 3σ bands
    mu_b  = sigma_test["mean_bnh"]
    sd_b  = sigma_test["std_bnh"]
    ymax  = ax3.get_ylim()[1] if ax3.get_ylim()[1] > 0 else 2.0
    for n, alpha_fill in [(1, 0.10), (2, 0.07), (3, 0.05)]:
        ax3.axvspan(mu_b - n * sd_b, mu_b + n * sd_b,
                    alpha=alpha_fill, color="darkorange",
                    label=f"BnH ±{n}σ" if n == 3 else None)
    ax3.axvline(mu_b + 3 * sd_b, color="red", linestyle="--", lw=1.5, label="BnH +3σ threshold")
    ax3.axvline(sigma_test["mean_strat"], color="navy", linestyle=":", lw=1.5,
                label=f"Strategy mean (z={sigma_test['z_score']:+.2f}σ)")
    ax3.axvline(1.0, color="black", linestyle="--", alpha=0.4, lw=1)

    # Badge
    badge = "3σ PASS ✓" if sigma_test["passes_3sigma"] else "3σ FAIL ✗"
    badge_color = "green" if sigma_test["passes_3sigma"] else "red"
    ax3.set_title(f"Terminal Wealth Distributions\n[{badge}]", fontsize=11, color=badge_color)
    ax3.set_xlabel("Terminal Wealth")
    ax3.set_ylabel("Density")
    ax3.legend(fontsize=7)
    ax3.grid(True, alpha=0.25)

    if pair_label:
        fig.suptitle(f"Monte Carlo Analysis — {pair_label}", fontsize=13, y=1.01)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_realized_vs_bnh(
    strat_df:    pd.DataFrame,
    bnh_cum:     pd.Series,
    date_col:    str  = "Date",
    pair_label:  str  = "",
    save_path:   str  = None,
) -> plt.Figure:
    """
    Overlay realized cumulative returns: strategy vs buy-and-hold.
    """
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(strat_df[date_col], strat_df["Cum_Return"],
            label="Pairs Strategy", color="navy", lw=1.5)
    ax.plot(strat_df[date_col], bnh_cum.values,
            label="Buy-and-Hold", color="darkorange", lw=1.5, linestyle="--")
    ax.axhline(1.0, color="black", linestyle=":", alpha=0.4)
    ax.set_title(f"Realized Cumulative Returns — {pair_label}", fontsize=12)
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative Return")
    ax.legend()
    ax.grid(True, alpha=0.25)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig
