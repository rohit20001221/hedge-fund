"""
portfolio_backtest.py - a multi-asset portfolio backtester built on yfinance data.

Quick start
-----------
    from portfolio_backtest import load_data, PortfolioBacktester, Momentum

    tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "JPM", "XOM"]
    data = load_data(tickers, start="2015-01-01")
    bt = PortfolioBacktester(rebalance="M", commission_bps=2, slippage_bps=2)
    result = bt.run(data, Momentum(lookback=126, skip=21, top_n=3))
    print(result.summary())
    result.plot()

How strategies work
-------------------
At every rebalance date the engine hands the strategy the basket's history
UP TO AND INCLUDING that date and asks for target weights:

    class MyStrategy(Strategy):
        min_history = 60                      # bars needed before first allocation
        def allocate(self, history) -> pd.Series:
            prices = history["Close"]         # DataFrame: dates x tickers
            ...                               # (also "Open", "High", "Low", "Volume")
            return pd.Series({...})           # ticker -> weight (cash = 1 - sum)

Because the strategy only ever sees data up to date t, it cannot look ahead.
Weights chosen at the close of day t earn the returns from day t+1 onward.
Between rebalances the portfolio drifts with prices (buy-and-hold).
"""
from __future__ import annotations

import argparse
import itertools
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def load_data(
    tickers: list[str] | str,
    start: str | None = None,
    end: str | None = None,
    interval: str = "1d",
) -> pd.DataFrame:
    """Download adjusted OHLCV for a basket.

    Returns a DataFrame with MultiIndex columns (field, ticker), so
    ``data["Close"]`` is a dates x tickers price table.
    """
    import yfinance as yf

    if isinstance(tickers, str):
        tickers = [tickers]
    df = yf.download(
        tickers, start=start, end=end, interval=interval,
        auto_adjust=True, progress=False, group_by="column",
    )
    if df is None or df.empty:
        raise ValueError(f"No data returned for {tickers}")
    if not isinstance(df.columns, pd.MultiIndex):  # single-ticker download
        df.columns = pd.MultiIndex.from_product([df.columns, tickers])

    close = df["Close"]
    dead = [t for t in close.columns if close[t].isna().all()]
    if dead:
        warnings.warn(f"No data for: {dead} (dropped)")
        df = df.drop(columns=dead, level=1)

    df = df.loc[df["Close"].notna().any(axis=1)]
    return df.ffill()  # fill gaps (holidays); leading NaNs (pre-listing) stay NaN


# --------------------------------------------------------------------------- #
# Strategies
# --------------------------------------------------------------------------- #
class Strategy:
    """Subclass and implement `allocate`."""

    name = "Strategy"
    min_history = 1  # number of bars required before the first allocation

    def allocate(self, history: pd.DataFrame) -> pd.Series | None:
        """Return target weights (ticker -> weight) or None to keep current holdings.

        `history` has MultiIndex columns (field, ticker) and ends at the current date.
        Assets that hadn't listed yet contain NaN - handle with .dropna().
        """
        raise NotImplementedError


class EqualWeight(Strategy):
    name = "Equal Weight"

    def allocate(self, history):
        last = history["Close"].iloc[-1].dropna()
        return pd.Series(1.0 / len(last), index=last.index)


class FixedWeights(Strategy):
    """Constant target weights, e.g. FixedWeights({"SPY": 0.6, "TLT": 0.4})."""

    def __init__(self, weights: dict[str, float]):
        self.weights = pd.Series(weights, dtype=float)
        self.name = "Fixed(" + ", ".join(f"{k}:{v:.0%}" for k, v in weights.items()) + ")"

    def allocate(self, history):
        return self.weights


class InverseVolatility(Strategy):
    """Weight each asset by 1 / trailing volatility (a simple risk-parity)."""

    def __init__(self, lookback: int = 60):
        self.lookback = lookback
        self.min_history = lookback + 1
        self.name = f"InvVol({lookback})"

    def allocate(self, history):
        rets = history["Close"].pct_change(fill_method=None).iloc[-self.lookback:]
        rets = rets.dropna(axis=1, how="any")  # need a full window
        vol = rets.std()
        vol = vol[vol > 0]
        if vol.empty:
            return None
        w = 1.0 / vol
        return w / w.sum()


class Momentum(Strategy):
    """Hold the top_n assets by trailing return (skipping the most recent `skip` bars).

    abs_filter=True only holds assets with positive momentum; the rest sits in cash.
    """

    def __init__(self, lookback: int = 126, skip: int = 21, top_n: int = 5,
                 abs_filter: bool = False):
        self.lookback, self.skip, self.top_n, self.abs_filter = lookback, skip, top_n, abs_filter
        self.min_history = lookback + skip + 1
        self.name = f"Momentum({lookback},{skip},top{top_n})"

    def allocate(self, history):
        px = history["Close"]
        end = px.iloc[-1 - self.skip]
        start = px.iloc[-1 - self.skip - self.lookback]
        mom = (end / start - 1).dropna()
        mom = mom[px.iloc[-1].reindex(mom.index).notna()]  # must be tradable today
        if self.abs_filter:
            mom = mom[mom > 0]
        top = mom.nlargest(self.top_n)
        if top.empty:
            return pd.Series(dtype=float)  # all cash
        return pd.Series(1.0 / self.top_n, index=top.index)


# --------------------------------------------------------------------------- #
# Results & metrics
# --------------------------------------------------------------------------- #
def compute_metrics(returns: pd.Series, equity: pd.Series, ppy: float, rf: float = 0.0) -> dict:
    years = len(returns) / ppy
    total = equity.iloc[-1] / equity.iloc[0] - 1
    cagr = (1 + total) ** (1 / years) - 1 if years > 0 and total > -1 else np.nan
    std = returns.std()
    vol = std * np.sqrt(ppy)
    excess = returns - rf / ppy
    sharpe = excess.mean() / std * np.sqrt(ppy) if std > 0 else np.nan
    dstd = returns[returns < 0].std()
    sortino = excess.mean() / dstd * np.sqrt(ppy) if dstd and dstd > 0 else np.nan
    max_dd = (equity / equity.cummax() - 1).min()
    calmar = cagr / abs(max_dd) if max_dd < 0 and not np.isnan(cagr) else np.nan
    return {
        "Total Return": total, "CAGR": cagr, "Volatility (ann.)": vol,
        "Sharpe": sharpe, "Sortino": sortino, "Max Drawdown": max_dd, "Calmar": calmar,
        "Best Day": returns.max(), "Worst Day": returns.min(),
    }


def _infer_periods_per_year(index: pd.DatetimeIndex) -> float:
    if len(index) < 3:
        return 252.0
    years = (index[-1] - index[0]).days / 365.25
    return len(index) / years if years > 0 else 252.0


@dataclass
class PortfolioResult:
    strategy_name: str
    benchmark_name: str
    equity: pd.Series
    benchmark_equity: pd.Series
    returns: pd.Series
    weights: pd.DataFrame      # daily weights held going into the next bar
    turnover: pd.Series        # one-way turnover on each rebalance date (0 otherwise)
    metrics: dict
    benchmark_metrics: dict

    @property
    def rebalances(self) -> pd.DataFrame:
        """Target weights on each date a trade occurred."""
        return self.weights.loc[self.turnover > 0]

    def summary(self) -> pd.DataFrame:
        return pd.DataFrame({self.strategy_name: self.metrics,
                             self.benchmark_name: self.benchmark_metrics})

    def plot(self, show: bool = True, save_to: str | None = None):
        """Draw equity / drawdown / weights.

        save_to : path (e.g. "result.png") to write the figure to.
        If a window can't be opened (non-interactive backend such as Agg),
        the figure is saved to "backtest.png" instead of just warning.
        """
        import matplotlib
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True,
                                 gridspec_kw={"height_ratios": [3, 1, 2]})
        axes[0].plot(self.equity, label=self.strategy_name)
        axes[0].plot(self.benchmark_equity, label=self.benchmark_name, alpha=0.7)
        axes[0].set_ylabel("Equity")
        axes[0].legend()
        axes[0].grid(alpha=0.3)

        dd = self.equity / self.equity.cummax() - 1
        bdd = self.benchmark_equity / self.benchmark_equity.cummax() - 1
        axes[1].fill_between(dd.index, dd, 0, color="tab:red", alpha=0.4)
        axes[1].plot(bdd, color="gray", lw=0.8, alpha=0.7)
        axes[1].set_ylabel("Drawdown")
        axes[1].grid(alpha=0.3)

        w = self.weights.loc[self.equity.index]
        w = w.loc[:, (w.abs() > 1e-9).any()].clip(lower=0)
        axes[2].stackplot(w.index, w.T.to_numpy(), labels=w.columns)
        axes[2].set_ylabel("Weights")
        axes[2].set_ylim(0, max(1.0, float(w.sum(axis=1).max())))
        if w.shape[1] <= 15:
            axes[2].legend(loc="upper left", ncol=min(w.shape[1], 5), fontsize=8)
        axes[2].grid(alpha=0.3)

        fig.tight_layout()

        interactive = matplotlib.get_backend().lower() not in {"agg", "pdf", "svg", "ps", "cairo", "template"}
        if show and not interactive and not save_to:
            save_to = "backtest.png"
            print("No interactive matplotlib backend available; saving chart to backtest.png")
        if save_to:
            fig.savefig(save_to, dpi=150, bbox_inches="tight")
            print(f"Saved chart to {save_to}")
        if show and interactive:
            plt.show()
        return fig


# --------------------------------------------------------------------------- #
# Engine
# --------------------------------------------------------------------------- #
_FREQ = {"W": "W", "M": "M", "Q": "Q", "Y": "Y"}


def _rebalance_mask(index: pd.DatetimeIndex, freq: str | None) -> np.ndarray:
    """True on the last trading day of each period ('D' = every bar, None = never after start)."""
    n = len(index)
    if freq is None:
        return np.zeros(n, dtype=bool)
    freq = freq.upper()
    if freq == "D":
        return np.ones(n, dtype=bool)
    if freq not in _FREQ:
        raise ValueError("rebalance must be one of: None, 'D', 'W', 'M', 'Q', 'Y'")
    pos = pd.Series(np.arange(n), index=index)
    last = pos.groupby(index.to_period(_FREQ[freq])).last().to_numpy()
    mask = np.zeros(n, dtype=bool)
    mask[last] = True
    return mask


class PortfolioBacktester:
    def __init__(
        self,
        initial_capital: float = 100_000.0,
        rebalance: str | None = "M",
        commission_bps: float = 1.0,
        slippage_bps: float = 1.0,
        risk_free_rate: float = 0.0,
        allow_short: bool = False,
        max_gross_leverage: float = 1.0,
    ):
        """
        rebalance          : None (initial allocation only), 'D', 'W', 'M', 'Q' or 'Y'
        commission/slippage: bps charged on traded notional (sum |w_new - w_drifted|)
        max_gross_leverage : weights are scaled down if sum(|w|) exceeds this
        """
        self.initial_capital = initial_capital
        self.rebalance = rebalance
        self.cost_rate = (commission_bps + slippage_bps) / 1e4
        self.rf = risk_free_rate
        self.allow_short = allow_short
        self.max_lev = max_gross_leverage

    # -- public ----------------------------------------------------------- #
    def run(self, data: pd.DataFrame, strategy: Strategy,
            benchmark: pd.Series | None = None) -> PortfolioResult:
        """Backtest `strategy` on `data` (MultiIndex columns: field, ticker).

        benchmark: optional price Series (e.g. SPY close). Default: equal-weight,
        rebalanced monthly, zero cost, over the same basket.
        """
        close = data["Close"]
        rets, weights, turnover = self._simulate(data, strategy, self.rebalance, self.cost_rate)

        held = (weights.abs().sum(axis=1) > 0).to_numpy()
        if not held.any():
            raise ValueError("Strategy never produced an allocation (not enough history?)")
        start = int(np.argmax(held))  # first bar with weights set; earns from start+1

        # Benchmark
        if benchmark is None:
            b_rets, _, _ = self._simulate(data, EqualWeight(), "M", 0.0)
            bench_name = "Equal Weight (M)"
        else:
            b_rets = benchmark.reindex(close.index).ffill().pct_change(fill_method=None).fillna(0.0)
            bench_name = str(benchmark.name or "Benchmark")

        rets = rets.iloc[start:].copy()
        b_rets = b_rets.iloc[start:].copy()
        rets.iloc[0] = 0.0
        b_rets.iloc[0] = 0.0

        equity = self.initial_capital * (1 + rets).cumprod()
        b_equity = self.initial_capital * (1 + b_rets).cumprod()
        ppy = _infer_periods_per_year(rets.index)

        metrics = compute_metrics(rets, equity, ppy, self.rf)
        w = weights.iloc[start:]
        years = len(rets) / ppy
        metrics["Exposure (avg gross)"] = w.abs().sum(axis=1).mean()
        metrics["Avg # Holdings"] = (w.abs() > 1e-9).sum(axis=1).mean()
        metrics["Annual Turnover (1-way)"] = turnover.iloc[start:].sum() / years if years > 0 else np.nan
        metrics["# Rebalances"] = int((turnover.iloc[start:] > 0).sum())

        return PortfolioResult(
            strategy_name=strategy.name, benchmark_name=bench_name,
            equity=equity, benchmark_equity=b_equity, returns=rets,
            weights=weights, turnover=turnover,
            metrics=metrics, benchmark_metrics=compute_metrics(b_rets, b_equity, ppy, self.rf),
        )

    # -- internals -------------------------------------------------------- #
    def _simulate(self, data, strategy, rebalance, cost_rate):
        close = data["Close"]
        tickers = list(close.columns)
        rets_arr = close.pct_change(fill_method=None).fillna(0.0).to_numpy()
        avail = close.notna().to_numpy()
        n, m = rets_arr.shape

        mask = _rebalance_mask(close.index, rebalance)
        min_h = max(1, int(strategy.min_history))

        w = np.zeros(m)                 # current (drifted) weights; cash is implicit
        port = np.zeros(n)
        turn = np.zeros(n)
        W = np.zeros((n, m))
        started = False

        for i in range(n):
            gross = float(w @ rets_arr[i])
            if 1.0 + gross > 0:
                w = w * (1.0 + rets_arr[i]) / (1.0 + gross)   # drift
            r = gross

            if i >= min_h - 1 and (mask[i] or not started):
                target = self._clean(strategy.allocate(data.iloc[: i + 1]), tickers, avail[i])
                if target is not None:
                    t = float(np.abs(target - w).sum())
                    r -= t * cost_rate
                    turn[i] = t / 2.0                          # one-way turnover
                    w = target
                    started = True

            port[i] = r
            W[i] = w

        idx = close.index
        return (pd.Series(port, index=idx), pd.DataFrame(W, index=idx, columns=tickers),
                pd.Series(turn, index=idx))

    def _clean(self, target, tickers, avail_row):
        if target is None:
            return None
        s = pd.Series(target, dtype=float)
        unknown = set(s.index) - set(tickers)
        if unknown:
            raise ValueError(f"Strategy returned weights for unknown tickers: {sorted(unknown)}")
        arr = s.reindex(tickers).fillna(0.0).to_numpy()
        arr = np.where(avail_row, arr, 0.0)                     # can't hold what isn't trading
        if not self.allow_short and (arr < -1e-12).any():
            raise ValueError("Negative weights returned but allow_short=False")
        gross = np.abs(arr).sum()
        if gross > self.max_lev + 1e-9:
            arr = arr * self.max_lev / gross
        return arr


# --------------------------------------------------------------------------- #
# Parameter sweep
# --------------------------------------------------------------------------- #
def grid_search(data, make_strategy, param_grid: dict, backtester: PortfolioBacktester | None = None,
                sort_by: str = "Sharpe") -> pd.DataFrame:
    """Rank strategy parameters. In-sample optimisation overfits - validate out of sample."""
    bt = backtester or PortfolioBacktester()
    keys = list(param_grid)
    rows = []
    for combo in itertools.product(*param_grid.values()):
        params = dict(zip(keys, combo))
        try:
            res = bt.run(data, make_strategy(**params))
        except ValueError:
            continue
        rows.append({**params, **res.metrics})
    return pd.DataFrame(rows).sort_values(sort_by, ascending=False).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _fmt(summary: pd.DataFrame) -> pd.DataFrame:
    plain = {"Sharpe", "Sortino", "Calmar", "Avg # Holdings"}
    ints = {"# Rebalances"}
    out = summary.astype(object).copy()
    for metric in summary.index:
        for col in summary.columns:
            v = summary.loc[metric, col]
            if pd.isna(v):
                out.loc[metric, col] = "-"
            elif metric in ints:
                out.loc[metric, col] = f"{int(v)}"
            elif metric in plain:
                out.loc[metric, col] = f"{v:.2f}"
            else:
                out.loc[metric, col] = f"{v:.2%}"
    return out


def main():
    ap = argparse.ArgumentParser(description="Portfolio backtester (yfinance)")
    ap.add_argument("tickers", nargs="+")
    ap.add_argument("--start", default="2015-01-01")
    ap.add_argument("--end", default=None)
    ap.add_argument("--strategy", choices=["ew", "invvol", "mom"], default="ew")
    ap.add_argument("--rebalance", default="M", help="D, W, M, Q, Y")
    ap.add_argument("--lookback", type=int, default=126)
    ap.add_argument("--top-n", type=int, default=3)
    ap.add_argument("--benchmark", default=None, help="ticker, e.g. SPY")
    ap.add_argument("--capital", type=float, default=100_000)
    ap.add_argument("--commission-bps", type=float, default=1.0)
    ap.add_argument("--slippage-bps", type=float, default=1.0)
    ap.add_argument("--plot", action="store_true")
    args = ap.parse_args()

    strat = {
        "ew": lambda: EqualWeight(),
        "invvol": lambda: InverseVolatility(args.lookback),
        "mom": lambda: Momentum(args.lookback, 21, args.top_n),
    }[args.strategy]()

    data = load_data(args.tickers, args.start, args.end)
    bench = None
    if args.benchmark:
        bench = load_data(args.benchmark, args.start, args.end)["Close"].iloc[:, 0]
        bench.name = args.benchmark

    bt = PortfolioBacktester(args.capital, args.rebalance, args.commission_bps, args.slippage_bps)
    res = bt.run(data, strat, benchmark=bench)
    idx = res.equity.index
    print(f"\n{len(data['Close'].columns)} assets | {idx[0].date()} -> {idx[-1].date()} | "
          f"rebalance={args.rebalance}\n")
    print(_fmt(res.summary()).to_string())
    print("\nLatest weights:")
    print(res.weights.iloc[-1][lambda s: s.abs() > 1e-9].round(3).to_string())
    if args.plot:
        res.plot()


if __name__ == "__main__":
    main()