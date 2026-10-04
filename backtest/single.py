"""
backtest.py - a small, vectorized backtesting module built on yfinance data.

Quick start
-----------
    from backtest import load_data, Backtester, SMACrossover

    df = load_data("AAPL", start="2015-01-01")
    result = Backtester(commission_bps=1, slippage_bps=2).run(df, SMACrossover(50, 200))
    print(result.summary())
    result.plot()

CLI
---
    python backtest.py AAPL --strategy sma --fast 50 --slow 200 --start 2015-01-01

Design notes
------------
* Strategies output a *target position* per bar (+1 long, 0 flat, -1 short,
  or any fractional weight). The position is applied on the NEXT bar, so a
  signal computed from today's close never earns today's return (no lookahead).
* Costs are charged on turnover (change in position) in basis points.
* Prices are dividend/split adjusted (yfinance auto_adjust=True).
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def load_data(
    ticker: str,
    start: str | None = None,
    end: str | None = None,
    interval: str = "1d",
) -> pd.DataFrame:
    """Download OHLCV data from Yahoo Finance as a flat-column DataFrame."""
    import yfinance as yf

    df = yf.download(
        ticker,
        start=start,
        end=end,
        interval=interval,
        auto_adjust=True,
        progress=False,
    )
    if df is None or df.empty:
        raise ValueError(f"No data returned for {ticker!r}")
    # Newer yfinance versions return MultiIndex columns even for one ticker.
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df.dropna()


# --------------------------------------------------------------------------- #
# Strategies
# --------------------------------------------------------------------------- #
class Strategy:
    """Subclass and implement `signals`, returning a target-position Series."""

    name = "Strategy"

    def signals(self, df: pd.DataFrame) -> pd.Series:
        raise NotImplementedError


class BuyAndHold(Strategy):
    name = "Buy & Hold"

    def signals(self, df):
        return pd.Series(1.0, index=df.index)


class SMACrossover(Strategy):
    """Long when fast SMA > slow SMA. Set long_only=False to short otherwise."""

    def __init__(self, fast: int = 50, slow: int = 200, long_only: bool = True):
        if fast >= slow:
            raise ValueError("fast must be < slow")
        self.fast, self.slow, self.long_only = fast, slow, long_only
        self.name = f"SMA({fast}/{slow})"

    def signals(self, df):
        close = df["Close"]
        fast = close.rolling(self.fast).mean()
        slow = close.rolling(self.slow).mean()
        off = 0.0 if self.long_only else -1.0
        sig = pd.Series(np.where(fast > slow, 1.0, off), index=df.index)
        sig[slow.isna()] = 0.0  # no signal until both averages exist
        return sig


class RSIMeanReversion(Strategy):
    """Enter long when RSI < oversold; exit when RSI > exit_level."""

    def __init__(self, period: int = 14, oversold: float = 30, exit_level: float = 55):
        self.period, self.oversold, self.exit_level = period, oversold, exit_level
        self.name = f"RSI({period},{oversold}/{exit_level})"

    @staticmethod
    def rsi(close: pd.Series, period: int) -> pd.Series:
        delta = close.diff()
        gain = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
        loss = (-delta.clip(upper=0)).ewm(alpha=1 / period, adjust=False).mean()
        rs = gain / loss.replace(0, np.nan)
        return 100 - 100 / (1 + rs)

    def signals(self, df):
        rsi = self.rsi(df["Close"], self.period).to_numpy()
        pos = np.zeros(len(rsi))
        holding = 0.0
        for i, r in enumerate(rsi):
            if np.isnan(r):
                pass
            elif holding == 0.0 and r < self.oversold:
                holding = 1.0
            elif holding == 1.0 and r > self.exit_level:
                holding = 0.0
            pos[i] = holding
        return pd.Series(pos, index=df.index)


# --------------------------------------------------------------------------- #
# Results & metrics
# --------------------------------------------------------------------------- #
@dataclass
class BacktestResult:
    strategy_name: str
    equity: pd.Series
    benchmark_equity: pd.Series
    returns: pd.Series
    positions: pd.Series
    trades: pd.DataFrame
    metrics: dict
    benchmark_metrics: dict

    def summary(self) -> pd.DataFrame:
        return pd.DataFrame(
            {self.strategy_name: self.metrics, "Buy & Hold": self.benchmark_metrics}
        )

    def plot(self, show: bool = True):
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(
            3, 1, figsize=(11, 8), sharex=True, gridspec_kw={"height_ratios": [3, 1, 1]}
        )
        axes[0].plot(self.equity, label=self.strategy_name)
        axes[0].plot(self.benchmark_equity, label="Buy & Hold", alpha=0.7)
        axes[0].set_ylabel("Equity")
        axes[0].legend()
        axes[0].grid(alpha=0.3)

        dd = self.equity / self.equity.cummax() - 1
        axes[1].fill_between(dd.index, dd, 0, color="tab:red", alpha=0.4)
        axes[1].set_ylabel("Drawdown")
        axes[1].grid(alpha=0.3)

        axes[2].fill_between(self.positions.index, self.positions, 0, step="post", alpha=0.5)
        axes[2].set_ylabel("Position")
        axes[2].grid(alpha=0.3)

        fig.tight_layout()
        if show:
            plt.show()
        return fig


def _infer_periods_per_year(index: pd.DatetimeIndex) -> float:
    if len(index) < 3:
        return 252.0
    years = (index[-1] - index[0]).days / 365.25
    return len(index) / years if years > 0 else 252.0


def compute_metrics(returns: pd.Series, equity: pd.Series, positions: pd.Series | None,
                    trades: pd.DataFrame | None, ppy: float, rf: float = 0.0) -> dict:
    n = len(returns)
    years = n / ppy
    total = equity.iloc[-1] / equity.iloc[0] - 1
    cagr = (1 + total) ** (1 / years) - 1 if years > 0 and total > -1 else np.nan
    vol = returns.std() * np.sqrt(ppy)
    excess = returns - rf / ppy
    sharpe = excess.mean() / returns.std() * np.sqrt(ppy) if returns.std() > 0 else np.nan
    downside = returns[returns < 0].std()
    sortino = excess.mean() / downside * np.sqrt(ppy) if downside and downside > 0 else np.nan
    max_dd = (equity / equity.cummax() - 1).min()
    calmar = cagr / abs(max_dd) if max_dd < 0 and not np.isnan(cagr) else np.nan

    out = {
        "Total Return": total,
        "CAGR": cagr,
        "Volatility (ann.)": vol,
        "Sharpe": sharpe,
        "Sortino": sortino,
        "Max Drawdown": max_dd,
        "Calmar": calmar,
    }
    if positions is not None:
        out["Exposure"] = (positions != 0).mean()
    if trades is not None:
        out["# Trades"] = len(trades)
        out["Win Rate"] = (trades["return"] > 0).mean() if len(trades) else np.nan
        out["Avg Trade"] = trades["return"].mean() if len(trades) else np.nan
    return out


# --------------------------------------------------------------------------- #
# Engine
# --------------------------------------------------------------------------- #
class Backtester:
    def __init__(
        self,
        initial_capital: float = 10_000.0,
        commission_bps: float = 1.0,
        slippage_bps: float = 1.0,
        risk_free_rate: float = 0.0,
        price_col: str = "Close",
    ):
        self.initial_capital = initial_capital
        self.cost_rate = (commission_bps + slippage_bps) / 1e4
        self.rf = risk_free_rate
        self.price_col = price_col

    def run(self, df: pd.DataFrame, strategy: Strategy) -> BacktestResult:
        price = df[self.price_col]
        asset_ret = price.pct_change().fillna(0.0)

        target = strategy.signals(df).reindex(df.index).fillna(0.0)
        pos = target.shift(1).fillna(0.0)  # act on next bar -> no lookahead

        turnover = pos.diff().abs()
        turnover.iloc[0] = abs(pos.iloc[0])
        strat_ret = pos * asset_ret - turnover * self.cost_rate

        equity = self.initial_capital * (1 + strat_ret).cumprod()
        bench_equity = self.initial_capital * (1 + asset_ret).cumprod()

        trades = self._extract_trades(pos, strat_ret, price)
        ppy = _infer_periods_per_year(df.index)

        return BacktestResult(
            strategy_name=strategy.name,
            equity=equity,
            benchmark_equity=bench_equity,
            returns=strat_ret,
            positions=pos,
            trades=trades,
            metrics=compute_metrics(strat_ret, equity, pos, trades, ppy, self.rf),
            benchmark_metrics=compute_metrics(asset_ret, bench_equity, None, None, ppy, self.rf),
        )

    @staticmethod
    def _extract_trades(pos: pd.Series, strat_ret: pd.Series, price: pd.Series) -> pd.DataFrame:
        """A trade = a contiguous run of the same non-zero position."""
        trade_id = (pos != pos.shift()).cumsum()
        rows = []
        for _, idx in pos.groupby(trade_id).groups.items():
            p = pos.loc[idx[0]]
            if p == 0:
                continue
            rows.append(
                {
                    "entry": idx[0],
                    "exit": idx[-1],
                    "direction": "long" if p > 0 else "short",
                    "size": p,
                    "entry_price": price.loc[idx[0]],
                    "exit_price": price.loc[idx[-1]],
                    "return": (1 + strat_ret.loc[idx]).prod() - 1,
                    "bars": len(idx),
                }
            )
        cols = ["entry", "exit", "direction", "size", "entry_price", "exit_price", "return", "bars"]
        return pd.DataFrame(rows, columns=cols)


# --------------------------------------------------------------------------- #
# Parameter sweep helper
# --------------------------------------------------------------------------- #
def grid_search(df: pd.DataFrame, make_strategy, param_grid: dict, backtester: Backtester | None = None,
                sort_by: str = "Sharpe") -> pd.DataFrame:
    """Run a strategy factory over a parameter grid and rank the results.

    Beware: picking the best in-sample params overfits. Validate on held-out data.
    """
    import itertools

    bt = backtester or Backtester()
    keys = list(param_grid)
    rows = []
    for combo in itertools.product(*param_grid.values()):
        params = dict(zip(keys, combo))
        try:
            res = bt.run(df, make_strategy(**params))
        except ValueError:
            continue
        rows.append({**params, **res.metrics})
    return pd.DataFrame(rows).sort_values(sort_by, ascending=False).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _fmt(summary: pd.DataFrame) -> pd.DataFrame:
    pct = {"Total Return", "CAGR", "Volatility (ann.)", "Max Drawdown", "Exposure", "Win Rate", "Avg Trade"}
    out = summary.astype(object).copy()
    for metric in summary.index:
        for col in summary.columns:
            v = summary.loc[metric, col]
            if pd.isna(v):
                out.loc[metric, col] = "-"
            elif metric in pct:
                out.loc[metric, col] = f"{v:.2%}"
            elif metric == "# Trades":
                out.loc[metric, col] = f"{int(v)}"
            else:
                out.loc[metric, col] = f"{v:.2f}"
    return out


def main():
    ap = argparse.ArgumentParser(description="Simple yfinance backtester")
    ap.add_argument("ticker")
    ap.add_argument("--start", default="2015-01-01")
    ap.add_argument("--end", default=None)
    ap.add_argument("--interval", default="1d")
    ap.add_argument("--strategy", choices=["sma", "rsi", "hold"], default="sma")
    ap.add_argument("--fast", type=int, default=50)
    ap.add_argument("--slow", type=int, default=200)
    ap.add_argument("--capital", type=float, default=10_000)
    ap.add_argument("--commission-bps", type=float, default=1.0)
    ap.add_argument("--slippage-bps", type=float, default=1.0)
    ap.add_argument("--plot", action="store_true")
    args = ap.parse_args()

    strat = {
        "sma": lambda: SMACrossover(args.fast, args.slow),
        "rsi": lambda: RSIMeanReversion(),
        "hold": lambda: BuyAndHold(),
    }[args.strategy]()

    df = load_data(args.ticker, args.start, args.end, args.interval)
    bt = Backtester(args.capital, args.commission_bps, args.slippage_bps)
    result = bt.run(df, strat)
    print(f"\n{args.ticker} | {df.index[0].date()} -> {df.index[-1].date()} | {len(df)} bars\n")
    print(_fmt(result.summary()).to_string())
    if args.plot:
        result.plot()


if __name__ == "__main__":
    main()