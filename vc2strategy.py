import os
import glob
import logging
from datetime import datetime, date
from typing import Dict, List, Tuple

import backtrader as bt
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# Configuration
# ============================================================

DEFAULT_DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)


# ============================================================
# Custom Pandas Data Feed
# Expected CSV columns:
# date,price,rsi,pb_ratio,pe_ratio,ps_ratio,pcf_ratio,ev_ebitda,shareholder_yield
# ============================================================

BACKTEST_START = datetime(2013, 1, 1)
BACKTEST_END   = datetime(2024, 2, 28)

class VC2Data(bt.feeds.PandasData):
    lines = (
        "rsi",
        "pb_ratio",
        "pe_ratio",
        "ps_ratio",
        "pcf_ratio",
        "ev_ebitda",
        "shareholder_yield",
    )

    params = (
        ("datetime", None),   # use DataFrame index
        ("open",  "price"),
        ("high",  "price"),
        ("low",   "price"),
        ("close", "price"),
        ("volume", -1),
        ("openinterest", -1),
        ("rsi",              "rsi"),
        ("pb_ratio",         "pb_ratio"),
        ("pe_ratio",         "pe_ratio"),
        ("ps_ratio",         "ps_ratio"),
        ("pcf_ratio",        "pcf_ratio"),
        ("ev_ebitda",        "ev_ebitda"),
        ("shareholder_yield","shareholder_yield"),
        ("valid_from",       None),   # actual first data date
    )


def load_padded_csv(path: str) -> Tuple[pd.DataFrame, date]:
    """
    Load a stock CSV, reindex to the full backtest date range, and
    fill padded rows (before the stock's first real date) with the
    stock's first available values so backtrader always has a valid
    execution price. Returns the DataFrame and the actual start date.
    """
    df = pd.read_csv(path, index_col="date", parse_dates=True)
    actual_start: date = df.index[0].date()
    full_idx = pd.date_range(BACKTEST_START, BACKTEST_END, freq="D")
    df = df.reindex(full_idx).bfill().ffill()
    return df, actual_start


# ============================================================
# VC2 Strategy
# ============================================================

class VC2Strategy(bt.Strategy):
    params = (
        ("num_stocks", 10),
        ("rebalance_period", 28),
        ("top_k_pre_rsi", 50),
        ("printlog", True),
    )

    def __init__(self) -> None:
        self.last_rebalance_bar = None

    def log(self, txt: str, level: int = logging.INFO) -> None:
        if self.p.printlog and len(self.datas) > 0 and len(self.datas[0]) > 0:
            dt = self.datas[0].datetime.date(0)
            logging.log(level, f"{dt} | {txt}")

    def next(self) -> None:
        """
        Rebalance every self.p.rebalance_period bars.
        """
        if self.last_rebalance_bar is None:
            self.rebalance()
            self.last_rebalance_bar = len(self)
            return

        if len(self) - self.last_rebalance_bar >= self.p.rebalance_period:
            self.rebalance()
            self.last_rebalance_bar = len(self)

    def rebalance(self) -> None:
        self.log("Starting rebalance")

        universe = [d._name for d in self.datas]
        curr_data: Dict[str, Dict[str, float]] = {}

        current_date: date = self.datas[0].datetime.date(0)

        for d in self.datas:
            if len(d) == 0:
                continue

            # skip stocks whose real data hasn't started yet
            if d.p.valid_from is not None and current_date < d.p.valid_from:
                continue

            required_lines = [
                "pb_ratio",
                "pe_ratio",
                "ps_ratio",
                "pcf_ratio",
                "ev_ebitda",
                "shareholder_yield",
                "rsi",
                "close",
            ]
            if any(not hasattr(d, ln) for ln in required_lines):
                continue

            curr_data[d._name] = {
                "pb_ratio": d.pb_ratio[0],
                "pe_ratio": d.pe_ratio[0],
                "ps_ratio": d.ps_ratio[0],
                "pcf_ratio": d.pcf_ratio[0],
                "ev_ebitda": d.ev_ebitda[0],
                "shareholder_yield": d.shareholder_yield[0],
                "rsi": d.rsi[0],
                "price": d.close[0],
            }

        ranked_df, coverage = self.rank_tickers(curr_data, universe)

        if ranked_df.empty:
            self.log("No eligible tickers after ranking", level=logging.WARNING)
            return

        selected = self.get_buys(ranked_df, self.p.num_stocks)
        selected_set = set(selected)

        self.log(f"Coverage: {coverage:.2%}")
        self.log(f"Selected tickers: {selected}")

        # Close positions not selected anymore
        for d in self.datas:
            pos = self.getposition(d)
            if pos.size != 0 and d._name not in selected_set:
                self.order_target_percent(data=d, target=0.0)

        # Equal-weight selected names
        target_weight = 1.0 / max(len(selected), 1)
        for ticker in selected:
            data = self.getdatabyname(ticker)
            self.order_target_percent(data=data, target=target_weight)

    def rank_tickers(
        self,
        vc2_data: Dict[str, Dict[str, float]],
        universe: List[str],
    ) -> Tuple[pd.DataFrame, float]:
        """
        Rank stocks based on VC2 factors and then use RSI as a secondary filter.

        Lower valuation ratios are better:
        - pb_ratio
        - pe_ratio
        - ps_ratio
        - pcf_ratio
        - ev_ebitda

        Higher shareholder_yield is better.
        """
        df = pd.DataFrame(vc2_data).T
        available_tickers = df.index.intersection(universe)
        df = df.loc[available_tickers]

        numeric_cols = [
            "pb_ratio",
            "pe_ratio",
            "ps_ratio",
            "pcf_ratio",
            "ev_ebitda",
            "shareholder_yield",
            "rsi",
            "price",
        ]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        factor_cols = [
            "pb_ratio",
            "pe_ratio",
            "ps_ratio",
            "pcf_ratio",
            "ev_ebitda",
            "shareholder_yield",
        ]
        factor_cols = [c for c in factor_cols if c in df.columns and df[c].notna().any()]

        if not factor_cols:
            return pd.DataFrame(), 0.0

        low_is_good = ["pb_ratio", "pe_ratio", "ps_ratio", "pcf_ratio", "ev_ebitda"]
        high_is_good = ["shareholder_yield"]

        # Lower values should get smaller percentile ranks
        for col in low_is_good:
            if col in factor_cols:
                df[f"{col}_rank"] = df[col].rank(pct=True, ascending=True)

        # Higher shareholder yield should get smaller percentile ranks
        for col in high_is_good:
            if col in factor_cols:
                df[f"{col}_rank"] = df[col].rank(pct=True, ascending=False)

        rank_cols = [c for c in df.columns if c.endswith("_rank")]
        if not rank_cols:
            return pd.DataFrame(), 0.0

        df["vc2_raw"] = df[rank_cols].mean(axis=1)
        df["vc2"] = df["vc2_raw"].rank(pct=True, ascending=True)

        # lower vc2 is better
        df = df.dropna(subset=["vc2", "rsi"])
        coverage = len(df) / max(len(universe), 1)

        # first filter by cheapest names
        df = df.sort_values("vc2", ascending=True).head(self.p.top_k_pre_rsi)

        # then sort by RSI descending
        df = df.sort_values("rsi", ascending=False)

        return df, coverage

    def get_buys(self, df: pd.DataFrame, num_stocks: int) -> List[str]:
        buys = df.head(num_stocks).index.tolist()
        self.log(f"Top {num_stocks} stocks: {buys}")
        return buys

    def stop(self) -> None:
        self.log(f"Final Portfolio Value: {self.broker.getvalue():.2f}")


# ============================================================
# Helpers
# ============================================================

def ticker_from_path(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0]


def discover_csvs(data_dir: str, exclude: List[str] = ["SPY"]) -> List[str]:
    pattern = os.path.join(data_dir, "*.csv")
    return sorted(
        p for p in glob.glob(pattern)
        if ticker_from_path(p) not in exclude
    )


def build_cerebro(
    data_dir: str = DEFAULT_DATA_DIR,
    cash: float = 100000.0,
    commission: float = 0.001,
    num_stocks: int = 10,
    rebalance_period: int = 28,
) -> bt.Cerebro:
    cerebro = bt.Cerebro()
    cerebro.broker.setcash(cash)
    cerebro.broker.setcommission(commission=commission)

    csv_files = discover_csvs(data_dir)
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in: {data_dir}")

    for path in csv_files:
        ticker = ticker_from_path(path)
        df, valid_from = load_padded_csv(path)
        data = VC2Data(dataname=df, valid_from=valid_from)
        cerebro.adddata(data, name=ticker)

    cerebro.addstrategy(
        VC2Strategy,
        num_stocks=num_stocks,
        rebalance_period=rebalance_period,
    )

    # Analyzers
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name="sharpe")
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name="drawdown")
    cerebro.addanalyzer(bt.analyzers.Returns, _name="returns")
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name="trades")
    cerebro.addanalyzer(bt.analyzers.TimeReturn, _name="timereturn")

    return cerebro


def safe_get_sharpe(strat) -> float:
    analysis = strat.analyzers.sharpe.get_analysis()
    if isinstance(analysis, dict):
        return analysis.get("sharperatio", float("nan"))
    return float("nan")


def safe_get_log_total_return(strat) -> float:
    """
    Backtrader's Returns analyzer 'rtot' is a log-style total return,
    not the simple cumulative return.
    """
    analysis = strat.analyzers.returns.get_analysis()
    if isinstance(analysis, dict):
        return analysis.get("rtot", float("nan"))
    return float("nan")


def safe_get_cagr(strat) -> float:
    analysis = strat.analyzers.returns.get_analysis()
    if isinstance(analysis, dict):
        # Backtrader returns rnorm100 as annualized percent return
        return analysis.get("rnorm100", float("nan"))
    return float("nan")


def safe_get_max_drawdown(strat) -> float:
    analysis = strat.analyzers.drawdown.get_analysis()
    try:
        return analysis.max.drawdown
    except Exception:
        return float("nan")


def safe_get_trade_count(strat) -> int:
    analysis = strat.analyzers.trades.get_analysis()
    try:
        return analysis.total.closed
    except Exception:
        return 0


def safe_get_win_rate(strat) -> float:
    analysis = strat.analyzers.trades.get_analysis()
    try:
        won = analysis.won.total
        closed = analysis.total.closed
        if closed and closed > 0:
            return won / closed
        return float("nan")
    except Exception:
        return float("nan")


def save_equity_curve(strat, save_path: str = "vc2_equity_curve.png") -> None:
    """
    Save cumulative equity curve from TimeReturn analyzer.
    """
    timereturn = strat.analyzers.timereturn.get_analysis()
    returns_series = pd.Series(timereturn).sort_index()

    if returns_series.empty:
        print("No time return data available; equity curve not saved.")
        return

    equity_curve = (1 + returns_series).cumprod()

    plt.figure(figsize=(8, 5))
    plt.plot(equity_curve.index, equity_curve.values, label="VC2 Baseline")
    plt.title("VC2 Baseline Equity Curve")
    plt.xlabel("Date")
    plt.ylabel("Portfolio Growth")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


# ============================================================
# Runner
# ============================================================

if __name__ == "__main__":
    DATA_DIR = DEFAULT_DATA_DIR

    cerebro = build_cerebro(
        data_dir=DATA_DIR,
        cash=100000.0,
        commission=0.001,  # 10 bps
        num_stocks=10,
        rebalance_period=28,
    )

    starting_value = cerebro.broker.getvalue()
    print(f"Starting Portfolio Value: {starting_value:.2f}")

    results = cerebro.run()
    strat = results[0]

    final_value = cerebro.broker.getvalue()
    simple_total_return = final_value / starting_value - 1.0
    log_total_return = safe_get_log_total_return(strat)
    cagr = safe_get_cagr(strat)
    sharpe = safe_get_sharpe(strat)
    max_dd = safe_get_max_drawdown(strat)
    total_trades = safe_get_trade_count(strat)
    win_rate = safe_get_win_rate(strat)

    save_equity_curve(strat, save_path="vc2_equity_curve.png")

    print("\n=== VC2 BASELINE RESULTS ===")
    print(f"Final Portfolio Value: {final_value:.2f}")
    print(f"Simple Total Return: {simple_total_return * 100:.2f}%")
    print(f"Backtrader Log Total Return (rtot): {log_total_return:.4f}" if pd.notna(log_total_return) else "Backtrader Log Total Return (rtot): N/A")
    print(f"CAGR: {cagr:.2f}%")
    print(f"Sharpe Ratio: {sharpe:.4f}" if pd.notna(sharpe) else "Sharpe Ratio: N/A")
    print(f"Max Drawdown: {max_dd:.2f}%")
    print(f"Total Trades: {total_trades}")
    print(f"Win Rate: {win_rate:.2%}" if pd.notna(win_rate) else "Win Rate: N/A")
    print("\nSaved plot:")
    print("- vc2_equity_curve.png")

    # Uncomment if you want interactive Backtrader plotting
    # cerebro.plot()
