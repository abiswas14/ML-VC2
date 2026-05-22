import os
import glob
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    log_loss,
    precision_recall_curve,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    roc_curve,
)
from sklearn.pipeline import Pipeline


# ============================================================
# Configuration
# ============================================================

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

FORWARD_HORIZON = 20
TOP_K_VC2 = 50
NUM_STOCKS = 10
TRAIN_END_DATE = "2020-01-01"
REBALANCE_EVERY = 28  # trading-day step for portfolio evaluation

# preprocessing / robustness settings
WINSOR_LOWER = 0.01
WINSOR_UPPER = 0.99
MIN_PRICE = 1.0
MIN_CANDIDATES_PER_DATE = 10
FFILL_LIMIT = 252  # max days to forward-fill within ticker (prevents stale fundamentals)

# return handling
FORWARD_RETURN_CLIP_LOW = -0.50
FORWARD_RETURN_CLIP_HIGH = 0.50

# label threshold: require >X forward return to count as positive (0.0 = "up vs down")
POSITIVE_RETURN_THRESHOLD = 0.0

# transaction cost (applied once per rebalance period, equal-weight portfolio)
TRANSACTION_COST_BPS = 10  # 10 bps
FILTER_MIN_PROB = 0.55  # if enough names pass, treat model as a filter then use RSI

# label mode:
# - "absolute": forward return > POSITIVE_RETURN_THRESHOLD
# - "relative": forward return > cross-sectional median on that date
LABEL_MODE = "relative"


# ============================================================
# Result container
# ============================================================

@dataclass
class ExperimentResults:
    metrics: Dict[str, float]
    baseline_portfolio_returns: pd.Series
    filtered_portfolio_returns: pd.Series
    test_candidates: pd.DataFrame
    feature_importances: pd.Series


# ============================================================
# Data loading
# ============================================================

def load_all_csvs(data_dir: str) -> pd.DataFrame:
    """
    Load all stock CSVs from the data directory.

    Expected columns:
    date,price,rsi,pb_ratio,pe_ratio,ps_ratio,pcf_ratio,ev_ebitda,shareholder_yield
    """
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
    """Convert columns to numeric."""
    for col in cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


# ============================================================
# Preprocessing
# ============================================================

def winsorize_by_date(df: pd.DataFrame, cols: List[str], lower=0.01, upper=0.99) -> pd.DataFrame:
    """
    Winsorize columns cross-sectionally by date to reduce outlier impact.
    """
    out = df.copy()
    for col in cols:
        lo = out.groupby("date")[col].transform(lambda s: s.quantile(lower))
        hi = out.groupby("date")[col].transform(lambda s: s.quantile(upper))
        out[col] = out[col].clip(lo, hi)
    return out.reset_index(drop=True)


def add_lagged_features(df: pd.DataFrame, cols: List[str], lag: int = 1) -> pd.DataFrame:
    """
    Lag features by ticker to reduce lookahead bias.
    """
    for col in cols:
        df[f"{col}_lag{lag}"] = df.groupby("ticker")[col].shift(lag)
    return df


def add_price_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add momentum and volatility features from price.
    """
    df["ret_1"] = df.groupby("ticker")["price"].pct_change(1)
    df["momentum_20"] = df.groupby("ticker")["price"].pct_change(20)
    df["momentum_63"] = df.groupby("ticker")["price"].pct_change(63)

    df["volatility_20"] = (
        df.groupby("ticker")["ret_1"]
        .rolling(20)
        .std()
        .reset_index(level=0, drop=True)
    )
    return df


def add_forward_label(df: pd.DataFrame, horizon: int = 20) -> pd.DataFrame:
    """
    Create forward return and binary label.
    """
    df["future_price"] = df.groupby("ticker")["price"].shift(-horizon)
    df["forward_return_raw"] = df["future_price"] / df["price"] - 1.0

    # Cap extreme returns to keep evaluation realistic/stable.
    df["forward_return"] = df["forward_return_raw"].clip(
        FORWARD_RETURN_CLIP_LOW,
        FORWARD_RETURN_CLIP_HIGH,
    )

    if LABEL_MODE == "relative":
        median_by_date = df.groupby("date")["forward_return"].transform("median")
        df["label"] = (df["forward_return"] > median_by_date).astype(int)
    else:
        df["label"] = (df["forward_return"] > POSITIVE_RETURN_THRESHOLD).astype(int)
    return df


def add_cross_sectional_ranks(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    """
    Add percentile ranks by date.
    Lower valuation ratios are better.
    Higher shareholder yield is better.
    """
    if "date" not in df.columns:
        if isinstance(df.index, pd.DatetimeIndex):
            df = df.copy()
            df["date"] = df.index
            df = df.reset_index(drop=True)
        else:
            df = df.reset_index()
        if "date" not in df.columns:
            raise KeyError(
                "Missing required 'date' column for cross-sectional ranking. "
                "Ensure earlier preprocessing steps preserve 'date'."
            )

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
    """
    Compute VC2-style score from ranked lagged factors.
    Lower score is better.
    """
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


def get_feature_cols() -> List[str]:
    return [
        "pb_ratio_lag1_rank",
        "pe_ratio_lag1_rank",
        "ps_ratio_lag1_rank",
        "pcf_ratio_lag1_rank",
        "ev_ebitda_lag1_rank",
        "shareholder_yield_lag1_rank",
        "rsi_lag1",
        "momentum_20",
        "momentum_63",
        "volatility_20",
        "vc2",
    ]


def prepare_dataset(data_dir: str) -> pd.DataFrame:
    """
    Full preprocessing pipeline for the gradient boosting experiment.
    Mirrors the logistic regression pipeline to keep comparisons fair.
    """
    df = load_all_csvs(data_dir)

    base_numeric = [
        "price",
        "rsi",
        "pb_ratio",
        "pe_ratio",
        "ps_ratio",
        "pcf_ratio",
        "ev_ebitda",
        "shareholder_yield",
    ]
    df = coerce_numeric(df, base_numeric)

    # Remove obviously bad price rows.
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

    # Fill missing fundamentals/RSI using only past information:
    # - forward-fill within ticker
    # - then fill remaining NAs cross-sectionally by date median
    fill_cols = factor_cols + ["rsi"]
    df[fill_cols] = df.groupby("ticker")[fill_cols].ffill(limit=FFILL_LIMIT)
    for col in fill_cols:
        med = df.groupby("date")[col].transform("median")
        df[col] = df[col].fillna(med)

    df = winsorize_by_date(df, factor_cols, lower=WINSOR_LOWER, upper=WINSOR_UPPER)
    df = add_lagged_features(df, factor_cols + ["rsi"], lag=1)
    df = add_price_features(df)
    df = add_forward_label(df, horizon=FORWARD_HORIZON)

    lagged_cols = [f"{c}_lag1" for c in factor_cols]
    df = add_cross_sectional_ranks(df, lagged_cols)
    df = compute_vc2_score(df)

    keep_cols = [
        "date",
        "ticker",
        "price",
        "rsi_lag1",
        "forward_return",
        "forward_return_raw",
        "label",
        "vc2",
        "momentum_20",
        "momentum_63",
        "volatility_20",
        "pb_ratio_lag1_rank",
        "pe_ratio_lag1_rank",
        "ps_ratio_lag1_rank",
        "pcf_ratio_lag1_rank",
        "ev_ebitda_lag1_rank",
        "shareholder_yield_lag1_rank",
    ]
    df = df[keep_cols]

    feature_cols = get_feature_cols()
    needed = feature_cols + ["forward_return", "label", "date", "ticker", "vc2"]
    df = df.dropna(subset=needed).reset_index(drop=True)

    return df


# ============================================================
# Candidate selection
# ============================================================

def select_vc2_candidates(df: pd.DataFrame, top_k_vc2: int = 50) -> pd.DataFrame:
    """
    Mimic baseline VC2 selection:
    1. sort by VC2 ascending (cheapest first)
    2. keep top K
    3. sort by RSI descending
    """
    out = df.dropna(subset=["vc2", "rsi_lag1", "forward_return"]).copy()

    per_date_counts = out.groupby("date")["ticker"].transform("size")
    out = out[per_date_counts >= MIN_CANDIDATES_PER_DATE]

    out = out.sort_values(["date", "vc2"], ascending=[True, True])
    out = out.groupby("date", group_keys=False).head(top_k_vc2)

    out = out.sort_values(["date", "rsi_lag1"], ascending=[True, False]).reset_index(drop=True)
    return out


# ============================================================
# Model
# ============================================================

def train_gradient_boosting_model(train_df: pd.DataFrame) -> Tuple[Pipeline, List[str]]:
    """
    Train a conservative histogram-based gradient boosting classifier.
    HistGradientBoosting handles tabular numeric data well and is stable on medium-sized datasets.
    """
    feature_cols = get_feature_cols()

    X_train = train_df[feature_cols]
    y_train = train_df["label"]

    model = Pipeline(
        steps=[
            (
                "gb",
                HistGradientBoostingClassifier(
                    loss="log_loss",
                    learning_rate=0.05,
                    max_iter=250,
                    max_leaf_nodes=15,
                    min_samples_leaf=40,
                    l2_regularization=1.0,
                    random_state=42,
                ),
            )
        ]
    )
    model.fit(X_train, y_train)
    return model, feature_cols


def evaluate_classifier(model: Pipeline, test_df: pd.DataFrame, feature_cols: List[str]) -> Dict[str, float]:
    """
    Compute classification metrics.
    """
    X_test = test_df[feature_cols]
    y_test = test_df["label"]

    probs = model.predict_proba(X_test)[:, 1]
    preds = (probs >= 0.5).astype(int)

    metrics = {
        "roc_auc": roc_auc_score(y_test, probs),
        "avg_precision": average_precision_score(y_test, probs),
        "precision": precision_score(y_test, preds, zero_division=0),
        "recall": recall_score(y_test, preds, zero_division=0),
        "f1": f1_score(y_test, preds, zero_division=0),
        "accuracy": float((preds == y_test).mean()),
        "balanced_accuracy": balanced_accuracy_score(y_test, preds),
        "log_loss": log_loss(y_test, probs, labels=[0, 1]),
        "brier": brier_score_loss(y_test, probs),
        "base_rate": float(y_test.mean()),
        "positive_rate": float(preds.mean()),
        "num_test_samples": float(len(test_df)),
    }
    return metrics


def build_test_candidate_predictions(
    model: Pipeline,
    test_candidates: pd.DataFrame,
    feature_cols: List[str],
) -> pd.DataFrame:
    """
    Add predicted probabilities to the test candidates.
    """
    out = test_candidates.copy()
    out["pred_prob"] = model.predict_proba(out[feature_cols])[:, 1]
    return out


def get_feature_importances(model: Pipeline, feature_cols: List[str]) -> pd.Series:
    gb = model.named_steps.get("gb")
    if gb is None or not hasattr(gb, "feature_importances_"):
        return pd.Series(dtype=float)
    return pd.Series(gb.feature_importances_, index=feature_cols).sort_values(ascending=False)


# ============================================================
# Portfolio comparison
# ============================================================

def select_rebalance_dates(dates: pd.Series, every: int = REBALANCE_EVERY) -> List[pd.Timestamp]:
    """
    Pick rebalance dates by stepping through sorted unique trading dates.
    This avoids treating overlapping forward-return windows as independent periods.
    """
    unique_dates = sorted(pd.to_datetime(pd.Series(dates).unique()))
    return unique_dates[:: max(int(every), 1)]


def max_drawdown(equity_curve: pd.Series) -> float:
    if equity_curve.empty:
        return float("nan")
    running_max = equity_curve.cummax()
    drawdown = equity_curve / running_max - 1.0
    return float(drawdown.min())


def compare_portfolios(
    test_candidates: pd.DataFrame,
    num_stocks: int = 10,
    rebalance_every: int = REBALANCE_EVERY,
    transaction_cost_bps: int = TRANSACTION_COST_BPS,
) -> Tuple[pd.Series, pd.Series]:
    """
    Compare:
    - baseline VC2 = top N by RSI among candidate set
    - filtered VC2 = top N by predicted probability among same candidate set

    Returns one-period forward equal-weight portfolio returns by rebalance date.
    """
    baseline_returns = {}
    filtered_returns = {}

    rebalance_dates = set(select_rebalance_dates(test_candidates["date"], every=rebalance_every))

    for dt, group in test_candidates.groupby("date"):
        if dt not in rebalance_dates:
            continue
        if len(group) < num_stocks:
            continue

        base_sel = group.sort_values("rsi_lag1", ascending=False).head(num_stocks)

        filt_pool = group[group["pred_prob"] >= FILTER_MIN_PROB]
        if len(filt_pool) >= num_stocks:
            filt_sel = filt_pool.sort_values("rsi_lag1", ascending=False).head(num_stocks)
        else:
            filt_sel = group.sort_values("pred_prob", ascending=False).head(num_stocks)

        base_ret = base_sel["forward_return"].mean()
        filt_ret = filt_sel["forward_return"].mean()

        # Clip portfolio-period returns for stability / realism.
        base_ret = np.clip(base_ret, FORWARD_RETURN_CLIP_LOW, FORWARD_RETURN_CLIP_HIGH)
        filt_ret = np.clip(filt_ret, FORWARD_RETURN_CLIP_LOW, FORWARD_RETURN_CLIP_HIGH)

        tc = float(transaction_cost_bps) / 10000.0
        base_ret -= tc
        filt_ret -= tc

        baseline_returns[dt] = base_ret
        filtered_returns[dt] = filt_ret

    baseline_returns = pd.Series(baseline_returns).sort_index()
    filtered_returns = pd.Series(filtered_returns).sort_index()

    return baseline_returns, filtered_returns


# ============================================================
# Plotting
# ============================================================

def plot_roc_curve(
    model: Pipeline,
    test_df: pd.DataFrame,
    feature_cols: List[str],
    save_path: str = "gb_roc_curve.png",
) -> None:
    X_test = test_df[feature_cols]
    y_test = test_df["label"]
    probs = model.predict_proba(X_test)[:, 1]

    fpr, tpr, _ = roc_curve(y_test, probs)

    plt.figure(figsize=(6, 4))
    plt.plot(fpr, tpr, label="Gradient Boosting")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_precision_recall(
    model: Pipeline,
    test_df: pd.DataFrame,
    feature_cols: List[str],
    save_path: str = "gb_pr_curve.png",
) -> None:
    X_test = test_df[feature_cols]
    y_test = test_df["label"]
    probs = model.predict_proba(X_test)[:, 1]

    precision, recall, _ = precision_recall_curve(y_test, probs)
    ap = average_precision_score(y_test, probs)

    plt.figure(figsize=(6, 4))
    plt.plot(recall, precision, label=f"AP={ap:.3f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_cumulative_returns(
    baseline_returns: pd.Series,
    filtered_returns: pd.Series,
    save_path: str = "gb_portfolio_comparison.png",
) -> None:
    baseline_curve = (1 + baseline_returns).cumprod()
    filtered_curve = (1 + filtered_returns).cumprod()

    fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
    ax = axes[0]
    ax.plot(baseline_curve.index, baseline_curve.values, label="Baseline VC2")
    ax.plot(filtered_curve.index, filtered_curve.values, label="VC2 + Gradient Boosting Filter")
    ax.set_ylabel("Cumulative Growth")
    ax.set_title("Baseline vs Gradient-Boosting-Filtered VC2 (Rebalance Periods)")
    ax.legend()

    dd_base = baseline_curve / baseline_curve.cummax() - 1.0
    dd_filt = filtered_curve / filtered_curve.cummax() - 1.0
    ax2 = axes[1]
    ax2.plot(dd_base.index, dd_base.values, label="Baseline DD")
    ax2.plot(dd_filt.index, dd_filt.values, label="Filtered DD")
    ax2.set_ylabel("Drawdown")
    ax2.set_xlabel("Date")
    ax2.legend(loc="lower left")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)


# ============================================================
# Main experiment
# ============================================================

def run_experiment(data_dir: str = DATA_DIR) -> ExperimentResults:
    """
    Full experiment pipeline.
    """
    df = prepare_dataset(data_dir)
    candidates = select_vc2_candidates(df, top_k_vc2=TOP_K_VC2)

    if candidates.empty:
        raise ValueError("Candidate set is empty after preprocessing and VC2 selection.")

    candidates = candidates.reset_index(drop=True)

    train_df = candidates[candidates["date"] < pd.Timestamp(TRAIN_END_DATE)].copy()
    test_df = candidates[candidates["date"] >= pd.Timestamp(TRAIN_END_DATE)].copy()

    if train_df.empty:
        raise ValueError("Training set is empty. Adjust TRAIN_END_DATE or inspect preprocessing.")
    if test_df.empty:
        raise ValueError("Test set is empty. Adjust TRAIN_END_DATE or inspect preprocessing.")

    model, feature_cols = train_gradient_boosting_model(train_df)
    metrics = evaluate_classifier(model, test_df, feature_cols)
    importances = get_feature_importances(model, feature_cols)

    test_pred = build_test_candidate_predictions(model, test_df, feature_cols)
    baseline_returns, filtered_returns = compare_portfolios(
        test_pred,
        num_stocks=NUM_STOCKS,
        rebalance_every=REBALANCE_EVERY,
        transaction_cost_bps=TRANSACTION_COST_BPS,
    )

    if baseline_returns.empty or filtered_returns.empty:
        raise ValueError("Portfolio comparison series are empty. Check candidate counts per date.")

    plot_roc_curve(model, test_df, feature_cols, save_path="gb_roc_curve.png")
    plot_precision_recall(model, test_df, feature_cols, save_path="gb_pr_curve.png")
    plot_cumulative_returns(
        baseline_returns,
        filtered_returns,
        save_path="gb_portfolio_comparison.png",
    )

    return ExperimentResults(
        metrics=metrics,
        baseline_portfolio_returns=baseline_returns,
        filtered_portfolio_returns=filtered_returns,
        test_candidates=test_pred,
        feature_importances=importances,
    )


# ============================================================
# Runner
# ============================================================

if __name__ == "__main__":
    results = run_experiment(DATA_DIR)

    print("\n=== Classification Metrics ===")
    for k, v in results.metrics.items():
        if "num_" in k:
            print(f"{k}: {int(v)}")
        else:
            print(f"{k}: {v:.4f}")

    if not results.feature_importances.empty:
        print("\n=== Top Feature Importances (Gradient Boosting) ===")
        for name, val in results.feature_importances.head(8).items():
            print(f"{name}: {val:.4f}")

    baseline_mean = results.baseline_portfolio_returns.mean()
    filtered_mean = results.filtered_portfolio_returns.mean()
    baseline_med = results.baseline_portfolio_returns.median()
    filtered_med = results.filtered_portfolio_returns.median()
    baseline_hit = float((results.baseline_portfolio_returns > 0).mean())
    filtered_hit = float((results.filtered_portfolio_returns > 0).mean())

    baseline_cum = (1 + results.baseline_portfolio_returns).prod() - 1
    filtered_cum = (1 + results.filtered_portfolio_returns).prod() - 1

    baseline_curve = (1 + results.baseline_portfolio_returns).cumprod()
    filtered_curve = (1 + results.filtered_portfolio_returns).cumprod()
    baseline_dd = max_drawdown(baseline_curve)
    filtered_dd = max_drawdown(filtered_curve)
    num_rebalances = int(len(results.baseline_portfolio_returns))

    def _cagr(curve: pd.Series) -> float:
        if curve.empty:
            return float("nan")
        start = pd.to_datetime(curve.index.min())
        end = pd.to_datetime(curve.index.max())
        years = max((end - start).days / 365.25, 1e-9)
        return float(curve.iloc[-1] ** (1.0 / years) - 1.0)

    print("\n=== Portfolio Comparison on Test Rebalances ===")
    print(f"rebalances: {num_rebalances} (every ~{REBALANCE_EVERY} trading days, tc={TRANSACTION_COST_BPS} bps)")
    print(f"Baseline mean forward portfolio return: {baseline_mean:.4%}")
    print(f"Baseline median forward portfolio return: {baseline_med:.4%} (hit rate: {baseline_hit:.1%})")
    print(f"Filtered mean forward portfolio return: {filtered_mean:.4%}")
    print(f"Filtered median forward portfolio return: {filtered_med:.4%} (hit rate: {filtered_hit:.1%})")
    print(f"Baseline cumulative test-period return: {baseline_cum:.4%}")
    print(f"Filtered cumulative test-period return: {filtered_cum:.4%}")
    print(f"Baseline max drawdown: {baseline_dd:.2%}")
    print(f"Filtered max drawdown: {filtered_dd:.2%}")
    print(f"Baseline CAGR (approx): {_cagr(baseline_curve):.2%}")
    print(f"Filtered CAGR (approx): {_cagr(filtered_curve):.2%}")

    print("\nSaved plots:")
    print("- gb_roc_curve.png")
    print("- gb_pr_curve.png")
    print("- gb_portfolio_comparison.png")
