import os
import glob
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


# ============================================================
# Configuration
# ============================================================

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

FORWARD_HORIZON = 20
TOP_K_VC2 = 50
NUM_STOCKS = 10
TRAIN_END_DATE = "2020-01-01"
REBALANCE_EVERY = 28  # trading-day step for portfolio evaluation

N_REGIMES = 3
MIN_PRICE = 1.0
MIN_CANDIDATES_PER_DATE = 10
FFILL_LIMIT = 252
TRANSACTION_COST_BPS = 10

WINSOR_LOWER = 0.01
WINSOR_UPPER = 0.99
FORWARD_RETURN_CLIP_LOW = -0.50
FORWARD_RETURN_CLIP_HIGH = 0.50


@dataclass
class RegimeResults:
    regimes: pd.DataFrame
    test_portfolio_returns: pd.DataFrame
    regime_stats: pd.DataFrame


# ============================================================
# Shared data loading / prep
# ============================================================

def load_all_csvs(data_dir: str) -> pd.DataFrame:
    paths = sorted(glob.glob(os.path.join(data_dir, "*.csv")))
    if not paths:
        raise FileNotFoundError(f"No CSV files found in {data_dir}")

    dfs = []
    for path in paths:
        ticker = os.path.splitext(os.path.basename(path))[0]
        df = pd.read_csv(path)
        if "date" not in df.columns:
            raise ValueError(f"{path} is missing a 'date' column")
        df["ticker"] = ticker
        dfs.append(df)

    out = pd.concat(dfs, ignore_index=True)
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out = out.dropna(subset=["date"])
    out = out.sort_values(["ticker", "date"]).reset_index(drop=True)
    return out


def coerce_numeric(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    for col in cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def winsorize_by_date(df: pd.DataFrame, cols: List[str], lower=0.01, upper=0.99) -> pd.DataFrame:
    out = df.copy()
    for col in cols:
        lo = out.groupby("date")[col].transform(lambda s: s.quantile(lower))
        hi = out.groupby("date")[col].transform(lambda s: s.quantile(upper))
        out[col] = out[col].clip(lo, hi)
    return out.reset_index(drop=True)


def add_lagged_features(df: pd.DataFrame, cols: List[str], lag: int = 1) -> pd.DataFrame:
    for col in cols:
        df[f"{col}_lag{lag}"] = df.groupby("ticker")[col].shift(lag)
    return df


def add_price_features(df: pd.DataFrame) -> pd.DataFrame:
    df["ret_1"] = df.groupby("ticker")["price"].pct_change(1)
    df["momentum_20"] = df.groupby("ticker")["price"].pct_change(20)
    df["volatility_20"] = (
        df.groupby("ticker")["ret_1"]
        .rolling(20)
        .std()
        .reset_index(level=0, drop=True)
    )
    return df


def add_forward_return(df: pd.DataFrame, horizon: int = FORWARD_HORIZON) -> pd.DataFrame:
    df["future_price"] = df.groupby("ticker")["price"].shift(-horizon)
    df["forward_return_raw"] = df["future_price"] / df["price"] - 1.0
    df["forward_return"] = df["forward_return_raw"].clip(
        FORWARD_RETURN_CLIP_LOW,
        FORWARD_RETURN_CLIP_HIGH,
    )
    return df


def add_cross_sectional_ranks(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    low_is_good = [
        "pb_ratio_lag1",
        "pe_ratio_lag1",
        "ps_ratio_lag1",
        "pcf_ratio_lag1",
        "ev_ebitda_lag1",
    ]
    high_is_good = ["shareholder_yield_lag1"]

    for col in low_is_good:
        if col in cols:
            df[f"{col}_rank"] = df.groupby("date")[col].rank(pct=True, ascending=True)

    for col in high_is_good:
        if col in cols:
            df[f"{col}_rank"] = df.groupby("date")[col].rank(pct=True, ascending=False)

    return df


def compute_vc2_score(df: pd.DataFrame) -> pd.DataFrame:
    rank_cols = [
        "pb_ratio_lag1_rank",
        "pe_ratio_lag1_rank",
        "ps_ratio_lag1_rank",
        "pcf_ratio_lag1_rank",
        "ev_ebitda_lag1_rank",
        "shareholder_yield_lag1_rank",
    ]
    available = [c for c in rank_cols if c in df.columns]

    if not available:
        df["vc2_raw"] = np.nan
        df["vc2"] = np.nan
        return df

    df["vc2_raw"] = df[available].mean(axis=1)
    df["vc2"] = df.groupby("date")["vc2_raw"].rank(pct=True, ascending=True)
    return df


def prepare_stock_level_dataset(data_dir: str) -> pd.DataFrame:
    df = load_all_csvs(data_dir)

    numeric_cols = [
        "price",
        "rsi",
        "pb_ratio",
        "pe_ratio",
        "ps_ratio",
        "pcf_ratio",
        "ev_ebitda",
        "shareholder_yield",
    ]
    df = coerce_numeric(df, numeric_cols)

    df = df.dropna(subset=["price"]).copy()
    df = df[df["price"] > MIN_PRICE].copy()
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)

    factor_cols = [
        "pb_ratio",
        "pe_ratio",
        "ps_ratio",
        "pcf_ratio",
        "ev_ebitda",
        "shareholder_yield",
    ]
    fill_cols = factor_cols + ["rsi"]

    df[fill_cols] = df.groupby("ticker")[fill_cols].ffill(limit=FFILL_LIMIT)
    for col in fill_cols:
        med = df.groupby("date")[col].transform("median")
        df[col] = df[col].fillna(med)

    df = winsorize_by_date(df, factor_cols, lower=WINSOR_LOWER, upper=WINSOR_UPPER)
    df = add_lagged_features(df, factor_cols + ["rsi"], lag=1)
    df = add_price_features(df)
    df = add_forward_return(df, horizon=FORWARD_HORIZON)

    lagged_cols = [f"{c}_lag1" for c in factor_cols]
    df = add_cross_sectional_ranks(df, lagged_cols)
    df = compute_vc2_score(df)

    keep_cols = [
        "date",
        "ticker",
        "price",
        "rsi",
        "rsi_lag1",
        "ret_1",
        "momentum_20",
        "volatility_20",
        "forward_return",
        "vc2",
        "pb_ratio_lag1_rank",
        "pe_ratio_lag1_rank",
        "ps_ratio_lag1_rank",
        "pcf_ratio_lag1_rank",
        "ev_ebitda_lag1_rank",
        "shareholder_yield_lag1_rank",
    ]
    return df[keep_cols].reset_index(drop=True)


def select_vc2_candidates(df: pd.DataFrame, top_k_vc2: int = TOP_K_VC2) -> pd.DataFrame:
    out = df.dropna(subset=["vc2", "rsi_lag1", "forward_return"]).copy()

    per_date_counts = out.groupby("date")["ticker"].transform("size")
    out = out[per_date_counts >= MIN_CANDIDATES_PER_DATE]

    out = out.sort_values(["date", "vc2"], ascending=[True, True])
    out = out.groupby("date", group_keys=False).head(top_k_vc2)
    out = out.sort_values(["date", "rsi_lag1"], ascending=[True, False]).reset_index(drop=True)
    return out


# ============================================================
# Market-level regime features
# ============================================================

def build_market_dataset(stock_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build one market-level row per date using only information available at date t.
    """
    grouped = stock_df.groupby("date")

    market_df = pd.DataFrame(
        {
            "date": grouped.size().index,
            "avg_market_return": grouped["ret_1"].mean().values,
            "return_dispersion": grouped["ret_1"].std().values,
            "avg_volatility": grouped["volatility_20"].mean().values,
            "avg_rsi": grouped["rsi_lag1"].mean().values,
            "avg_momentum_20": grouped["momentum_20"].mean().values,
            "vc2_dispersion": grouped["vc2"].std().values,
            "num_stocks": grouped["ticker"].nunique().values,
        }
    )

    market_df = market_df.sort_values("date").reset_index(drop=True)

    # Short rolling summaries use only current and past dates.
    market_df["avg_market_return_20"] = market_df["avg_market_return"].rolling(20).mean()
    market_df["return_dispersion_20"] = market_df["return_dispersion"].rolling(20).mean()
    market_df["avg_volatility_20"] = market_df["avg_volatility"].rolling(20).mean()

    feature_cols = get_regime_feature_cols()
    market_df = market_df.dropna(subset=feature_cols).reset_index(drop=True)
    return market_df


def get_regime_feature_cols() -> List[str]:
    return [
        "avg_market_return",
        "return_dispersion",
        "avg_volatility",
        "avg_rsi",
        "avg_momentum_20",
        "vc2_dispersion",
        "num_stocks",
        "avg_market_return_20",
        "return_dispersion_20",
        "avg_volatility_20",
    ]


def assign_regimes(market_df: pd.DataFrame, train_end_date: str = TRAIN_END_DATE) -> Tuple[pd.DataFrame, KMeans]:
    feature_cols = get_regime_feature_cols()
    work = market_df.copy()

    train_mask = work["date"] < pd.Timestamp(train_end_date)
    if train_mask.sum() < N_REGIMES:
        raise ValueError("Not enough training dates to fit the requested number of regimes.")

    scaler = StandardScaler()
    X_train = scaler.fit_transform(work.loc[train_mask, feature_cols])
    X_all = scaler.transform(work[feature_cols])

    model = KMeans(n_clusters=N_REGIMES, n_init=20, random_state=42)
    model.fit(X_train)
    work["regime"] = model.predict(X_all)

    # Order regimes from weaker to stronger market environment using
    # training-period average market return for readability.
    train_summary = (
        work.loc[train_mask]
        .groupby("regime")["avg_market_return"]
        .mean()
        .sort_values()
    )
    mapping = {old: new for new, old in enumerate(train_summary.index)}
    work["regime"] = work["regime"].map(mapping).astype(int)

    return work, model


# ============================================================
# Baseline VC2 evaluation by regime
# ============================================================

def select_rebalance_dates(dates: pd.Series, every: int = REBALANCE_EVERY) -> List[pd.Timestamp]:
    unique_dates = sorted(pd.to_datetime(pd.Series(dates).unique()))
    return unique_dates[:: max(int(every), 1)]


def max_drawdown(equity_curve: pd.Series) -> float:
    if equity_curve.empty:
        return float("nan")
    running_max = equity_curve.cummax()
    drawdown = equity_curve / running_max - 1.0
    return float(drawdown.min())


def evaluate_vc2_by_regime(
    candidates: pd.DataFrame,
    regimes: pd.DataFrame,
    num_stocks: int = NUM_STOCKS,
    rebalance_every: int = REBALANCE_EVERY,
    transaction_cost_bps: int = TRANSACTION_COST_BPS,
) -> pd.DataFrame:
    rebalance_dates = set(select_rebalance_dates(candidates["date"], every=rebalance_every))
    tc = float(transaction_cost_bps) / 10000.0

    merged = candidates.merge(regimes[["date", "regime"]], on="date", how="inner")
    rows = []

    for dt, group in merged.groupby("date"):
        if dt not in rebalance_dates:
            continue
        if len(group) < num_stocks:
            continue

        selection = group.sort_values("rsi_lag1", ascending=False).head(num_stocks)
        portfolio_return = selection["forward_return"].mean()
        portfolio_return = np.clip(portfolio_return, FORWARD_RETURN_CLIP_LOW, FORWARD_RETURN_CLIP_HIGH)
        portfolio_return -= tc

        rows.append(
            {
                "date": dt,
                "regime": int(group["regime"].iloc[0]),
                "portfolio_return": float(portfolio_return),
            }
        )

    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)


def summarize_regime_performance(returns_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    annualization = np.sqrt(252.0 / REBALANCE_EVERY)

    for regime, group in returns_df.groupby("regime"):
        rets = group["portfolio_return"]
        equity = (1.0 + rets).cumprod()
        std = float(rets.std())
        sharpe_proxy = float(rets.mean() / std * annualization) if std > 0 else float("nan")

        rows.append(
            {
                "regime": int(regime),
                "observations": int(len(group)),
                "mean_return": float(rets.mean()),
                "median_return": float(rets.median()),
                "volatility": std,
                "sharpe_proxy": sharpe_proxy,
                "hit_rate": float((rets > 0).mean()),
                "cumulative_return": float(equity.iloc[-1] - 1.0) if not equity.empty else float("nan"),
                "max_drawdown": max_drawdown(equity),
            }
        )

    return pd.DataFrame(rows).sort_values("regime").reset_index(drop=True)


# ============================================================
# Plotting
# ============================================================

def plot_regime_timeline(regimes: pd.DataFrame, save_path: str = "regime_timeline.png") -> None:
    fig, ax = plt.subplots(figsize=(11, 3.2))

    plot_df = regimes.sort_values("date").copy()
    cmap = plt.get_cmap("tab10")

    for regime in sorted(plot_df["regime"].unique()):
        subset = plot_df[plot_df["regime"] == regime]
        ax.scatter(
            subset["date"],
            np.repeat(regime, len(subset)),
            s=10,
            color=cmap(regime),
            label=f"Regime {regime}",
        )

    ax.set_xlabel("Date")
    ax.set_ylabel("Regime")
    ax.set_title("Market Regimes Over Time")
    ax.set_yticks(sorted(plot_df["regime"].unique()))
    ax.legend(ncol=min(N_REGIMES, 4), loc="upper right")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_regime_performance(regime_stats: pd.DataFrame, save_path: str = "regime_performance.png") -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    x = regime_stats["regime"].astype(str)

    axes[0].bar(x, regime_stats["mean_return"] * 100.0, color="steelblue")
    axes[0].set_title("Mean VC2 Forward Return by Regime")
    axes[0].set_xlabel("Regime")
    axes[0].set_ylabel("Mean Return (%)")

    axes[1].bar(x, regime_stats["sharpe_proxy"], color="darkorange")
    axes[1].set_title("Sharpe Proxy by Regime")
    axes[1].set_xlabel("Regime")
    axes[1].set_ylabel("Sharpe Proxy")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)


# ============================================================
# Main experiment
# ============================================================

def run_experiment(data_dir: str = DATA_DIR) -> RegimeResults:
    stock_df = prepare_stock_level_dataset(data_dir)
    market_df = build_market_dataset(stock_df)
    regimes, _ = assign_regimes(market_df, train_end_date=TRAIN_END_DATE)

    candidates = select_vc2_candidates(stock_df, top_k_vc2=TOP_K_VC2)
    test_candidates = candidates[candidates["date"] >= pd.Timestamp(TRAIN_END_DATE)].copy()
    test_regimes = regimes[regimes["date"] >= pd.Timestamp(TRAIN_END_DATE)].copy()

    if test_candidates.empty:
        raise ValueError("Test candidate set is empty after preprocessing and VC2 selection.")
    if test_regimes.empty:
        raise ValueError("No test-period regimes were generated.")

    portfolio_returns = evaluate_vc2_by_regime(
        test_candidates,
        test_regimes,
        num_stocks=NUM_STOCKS,
        rebalance_every=REBALANCE_EVERY,
        transaction_cost_bps=TRANSACTION_COST_BPS,
    )
    if portfolio_returns.empty:
        raise ValueError("No portfolio returns were generated. Check candidate counts and rebalance dates.")

    regime_stats = summarize_regime_performance(portfolio_returns)

    plot_regime_timeline(test_regimes, save_path="regime_timeline.png")
    plot_regime_performance(regime_stats, save_path="regime_performance.png")

    return RegimeResults(
        regimes=test_regimes,
        test_portfolio_returns=portfolio_returns,
        regime_stats=regime_stats,
    )


if __name__ == "__main__":
    results = run_experiment(DATA_DIR)

    print("\n=== Regime Clustering Summary ===")
    print(f"Number of regimes: {N_REGIMES}")
    print(f"Clustered test dates: {len(results.regimes)}")
    print(f"Test rebalances evaluated: {len(results.test_portfolio_returns)}")

    print("\n=== VC2 Performance by Regime ===")
    for row in results.regime_stats.itertuples(index=False):
        print(
            f"Regime {row.regime}: "
            f"obs={row.observations}, "
            f"mean={row.mean_return:.4%}, "
            f"median={row.median_return:.4%}, "
            f"sharpe_proxy={row.sharpe_proxy:.3f}, "
            f"hit_rate={row.hit_rate:.1%}, "
            f"cum_return={row.cumulative_return:.4%}, "
            f"max_dd={row.max_drawdown:.2%}"
        )

    best_regime = results.regime_stats.sort_values("mean_return", ascending=False).iloc[0]
    worst_regime = results.regime_stats.sort_values("mean_return", ascending=True).iloc[0]

    print("\n=== Interpretation ===")
    print(
        f"Best VC2 regime: Regime {int(best_regime['regime'])} "
        f"(mean return {best_regime['mean_return']:.4%}, "
        f"Sharpe proxy {best_regime['sharpe_proxy']:.3f})"
    )
    print(
        f"Worst VC2 regime: Regime {int(worst_regime['regime'])} "
        f"(mean return {worst_regime['mean_return']:.4%}, "
        f"Sharpe proxy {worst_regime['sharpe_proxy']:.3f})"
    )

    print("\nSaved plots:")
    print("- regime_timeline.png")
    print("- regime_performance.png")
