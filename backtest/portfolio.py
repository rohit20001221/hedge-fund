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

After each executed rebalance the engine calls `strategy.on_rebalance(stats)`
(override it; default is a no-op). `stats` is a RebalanceStats with the date,
equity, drawdown, weights before/after, turnover, cost and performance
metrics since inception:

        def on_rebalance(self, stats):
            print(stats.date.date(), f"equity={stats.equity:,.0f}",
                  f"dd={stats.drawdown:.1%}", f"sharpe={stats.metrics['Sharpe']:.2f}")

`strategy.on_period_end(stats, history)` reports the OTHER direction: how the
previous allocation performed. It fires on each rebalance date after the first,
once prices have moved and before the new allocation is chosen. `history` is the
same data slice `allocate` is about to receive. Order on a rebalance date:

    prices applied -> on_period_end(PeriodStats, history) -> allocate() -> trade -> on_rebalance(RebalanceStats)

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


CASH_TICKER = "FREE:CASH"
CASH_DAILY_RETURN = 0.0003

# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def _make_cash_frame(
    index: pd.DatetimeIndex,
    fields: list[str],
    daily_return: float,
    base: float = 100.0,
) -> pd.DataFrame:
    """Synthetic risk-free asset in the same (field, ticker) layout."""
    level = base * (1 + daily_return) ** np.arange(len(index))
    data = {}
    for f in fields:
        data[(f, CASH_TICKER)] = np.zeros(len(index)) if f == "Volume" else level
    out = pd.DataFrame(data, index=index)
    out.columns = pd.MultiIndex.from_tuples(out.columns)
    return out


def load_data(
    tickers: list[str] | str,
    start: str | None = None,
    end: str | None = None,
    interval: str = "1d",
    cash_daily_return: float = CASH_DAILY_RETURN,
) -> pd.DataFrame:
    """Download adjusted OHLCV for a basket.

    Returns a DataFrame with MultiIndex columns (field, ticker), so
    ``data["Close"]`` is a dates x tickers price table.

    The ticker "FREE:CASH" is not downloaded. It is generated as a synthetic
    risk-free asset compounding at ``cash_daily_return`` per row (default
    0.03%), on the same dates as the other tickers.
    """
    import yfinance as yf

    if isinstance(tickers, str):
        tickers = [tickers]

    want_cash = CASH_TICKER in tickers
    real = [t for t in tickers if t != CASH_TICKER]

    if real:
        df = yf.download(
            real, start=start, end=end, interval=interval,
            auto_adjust=True, progress=False, group_by="column",
        )
        if df is None or df.empty:
            raise ValueError(f"No data returned for {real}")
        if not isinstance(df.columns, pd.MultiIndex):  # single-ticker download
            df.columns = pd.MultiIndex.from_product([df.columns, real])

        close = df["Close"]
        dead = [t for t in close.columns if close[t].isna().all()]
        if dead:
            warnings.warn(f"No data for: {dead} (dropped)")
            df = df.drop(columns=dead, level=1)

        df = df.loc[df["Close"].notna().any(axis=1)]
        df = df.ffill()  # fill gaps (holidays); leading NaNs (pre-listing) stay NaN
        fields = list(df.columns.get_level_values(0).unique())
    else:
        # cash-only basket: no market calendar to align to
        if start is None or end is None:
            raise ValueError("start and end are required when only FREE:CASH is requested")
        df = None
        fields = ["Open", "High", "Low", "Close", "Volume"]

    if want_cash:
        index = df.index if df is not None else pd.bdate_range(start, end)
        cash = _make_cash_frame(index, fields, cash_daily_return)
        df = cash if df is None else pd.concat([df, cash], axis=1)

    # restore requested ticker order (minus any dropped dead tickers)
    kept = [t for t in tickers if t in df.columns.get_level_values(1)]
    cols = pd.MultiIndex.from_product([fields, kept])
    return df.reindex(columns=cols)


@dataclass
class RebalanceStats:
    """Snapshot handed to `Strategy.on_rebalance` right after a rebalance trade."""

    date: pd.Timestamp
    index: int                  # bar number of this date in the data
    rebalance_number: int       # 1 for the first allocation, 2 for the next, ...
    equity: float               # portfolio value after costs
    drawdown: float             # current drawdown from the equity peak (<= 0)
    weights_before: pd.Series   # weights just before trading (after price drift)
    weights_after: pd.Series    # new target weights
    turnover: float             # one-way turnover of this rebalance
    cost: float                 # trading cost in currency
    metrics: dict               # performance since inception, up to this date
                                # (same keys as PortfolioResult.metrics' core stats)


@dataclass
class PeriodStats:
    """How the PREVIOUS allocation fared. Handed to `Strategy.on_period_end`
    on the next rebalance date, after prices have moved and before the new
    allocation is chosen."""

    date: pd.Timestamp           # end of the holding period (today)
    start_date: pd.Timestamp     # when the period began (previous rebalance)
    bars: int                    # number of bars in the period
    rebalance_number: int        # which rebalance's holding period just ended
    period_return: float         # portfolio return over the period (after drift, before today's trade)
    period_max_drawdown: float   # worst peak-to-trough inside the period
    equity: float                # portfolio value now, before today's trade
    drawdown: float              # current drawdown from the equity peak (<= 0)
    weights_start: pd.Series     # weights right after the previous rebalance
    weights_end: pd.Series       # weights now (drifted with prices)
    asset_returns: pd.Series     # each asset's price return over the period
    contributions: pd.Series     # weights_start * asset_returns; sums to period_return
    metrics: dict                # performance since inception, up to today (pre-trade)


# --------------------------------------------------------------------------- #
# Strategies
# --------------------------------------------------------------------------- #
class Strategy:
    """Subclass and implement `allocate`; optionally override `on_rebalance`
    and `on_period_end`."""

    name = "Strategy"
    min_history = 1  # number of bars required before the first allocation

    def allocate(self, history: pd.DataFrame) -> pd.Series | None:
        """Return target weights (ticker -> weight) or None to keep current holdings.

        `history` has MultiIndex columns (field, ticker) and ends at the current date.
        Assets that hadn't listed yet contain NaN - handle with .dropna().
        """
        raise NotImplementedError

    def on_rebalance(self, stats: RebalanceStats) -> None:
        """Called after every executed rebalance with summary stats. Default: no-op.

        Override to log, record state, or adapt parameters for later rebalances
        (e.g. de-risk after a deep drawdown). It is informational: the trade has
        already happened, and `allocate` is the only place weights are chosen.
        """
        return None

    def on_period_end(self, stats: PeriodStats, history: pd.DataFrame) -> None:
        """Called when a holding period ends, i.e. on each rebalance date AFTER
        the first one, once prices have moved and BEFORE `allocate` runs.

        `stats` describes how the previous allocation performed (period return,
        per-asset contributions, drawdown, ...). `history` is the basket's data
        up to and including today (MultiIndex columns: field, ticker), exactly
        what `allocate` is about to receive. Default: no-op. Use it to record
        results or adapt state that `allocate` will read a moment later.
        """
        return None


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
    total = (1 + returns).prod() - 1
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
        # Strategy's first bar holds nothing yet, so its return is just the entry cost.
        b_rets.iloc[0] = 0.0   # benchmark starts at the same point, with no gain on day one

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
        E = np.full(n, self.initial_capital, dtype=float)   # running equity
        equity = self.initial_capital
        started = False
        start_i = 0
        n_reb = 0
        idx = close.index
        # Only pay for the stats snapshot if the strategy actually overrides the hook.
        wants_cb = type(strategy).on_rebalance is not Strategy.on_rebalance
        wants_period_cb = type(strategy).on_period_end is not Strategy.on_period_end
        close_arr = close.to_numpy()
        last_i = 0                      # bar of the most recent rebalance

        for i in range(n):
            gross = float(w @ rets_arr[i])
            if 1.0 + gross > 0:
                w = w * (1.0 + rets_arr[i]) / (1.0 + gross)   # drift
            r = gross
            equity_pre_cost = equity * (1.0 + gross)
            w_before = w.copy()
            traded = False

            due = i >= min_h - 1 and (mask[i] or not started)

            # Holding period just ended: report it BEFORE choosing new weights.
            if due and started and wants_period_cb:
                strategy.on_period_end(
                    self._make_period_stats(
                        idx, close_arr, i, last_i, n_reb, start_i, port, E, W,
                        tickers, w, equity_pre_cost, gross),
                    data.iloc[: i + 1])

            if due:
                target = self._clean(strategy.allocate(data.iloc[: i + 1]), tickers, avail[i])
                if target is not None:
                    t = float(np.abs(target - w).sum())
                    cost_frac = t * cost_rate
                    r -= cost_frac
                    turn[i] = t / 2.0                          # one-way turnover
                    w = target
                    if not started:
                        start_i = i
                    started = True
                    traded = True
                    n_reb += 1
                    last_i = i

            port[i] = r
            equity *= (1.0 + r)
            E[i] = equity
            W[i] = w

            if traded and wants_cb:
                strategy.on_rebalance(self._make_stats(
                    idx, i, n_reb, start_i, port, E, tickers, w_before, w,
                    turn[i], cost_frac * equity_pre_cost))

        return (pd.Series(port, index=idx), pd.DataFrame(W, index=idx, columns=tickers),
                pd.Series(turn, index=idx))

    def _make_period_stats(self, idx, close_arr, i, last_i, n_reb, start_i, port, E, W,
                           tickers, w_end, equity_pre, gross) -> PeriodStats:
        w_start = W[last_i]
        with np.errstate(divide="ignore", invalid="ignore"):
            asset_ret = close_arr[i] / close_arr[last_i] - 1.0
        contrib = w_start * np.nan_to_num(asset_ret)

        # equity path inside the period: bars last_i..i-1 are stored, bar i is pre-trade
        path = np.append(E[last_i:i], equity_pre)
        period_dd = float((path / np.maximum.accumulate(path) - 1.0).min())

        # performance since inception through today (pre-trade); port[i]/E[i] not yet written
        rets = pd.Series(np.append(port[start_i:i], gross), index=idx[start_i: i + 1])
        eq = pd.Series(np.append(E[start_i:i], equity_pre), index=idx[start_i: i + 1])
        metrics = compute_metrics(rets, eq, _infer_periods_per_year(idx[: i + 1]), self.rf)

        return PeriodStats(
            date=idx[i], start_date=idx[last_i], bars=i - last_i, rebalance_number=n_reb,
            period_return=float(equity_pre / E[last_i] - 1.0),
            period_max_drawdown=period_dd,
            equity=float(equity_pre),
            drawdown=float(equity_pre / max(E[:i].max(), equity_pre) - 1.0),
            weights_start=pd.Series(w_start, index=tickers),
            weights_end=pd.Series(w_end, index=tickers),
            asset_returns=pd.Series(asset_ret, index=tickers),
            contributions=pd.Series(contrib, index=tickers),
            metrics=metrics,
        )

    def _make_stats(self, idx, i, n_reb, start_i, port, E, tickers, w_before, w_after,
                    turnover, cost) -> RebalanceStats:
        sl = slice(start_i, i + 1)
        rets = pd.Series(port[sl], index=idx[sl])
        eq = pd.Series(E[sl], index=idx[sl])
        metrics = compute_metrics(rets, eq, _infer_periods_per_year(idx[: i + 1]), self.rf)
        return RebalanceStats(
            date=idx[i], index=i, rebalance_number=n_reb,
            equity=float(E[i]), drawdown=float(E[i] / E[: i + 1].max() - 1.0),
            weights_before=pd.Series(w_before, index=tickers),
            weights_after=pd.Series(w_after, index=tickers),
            turnover=float(turnover), cost=float(cost), metrics=metrics,
        )

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