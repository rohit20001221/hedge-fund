"""
PortfolioEnv: a Gymnasium environment for allocating between
NIFTYBEES.NS, GOLDBEES.NS, JUNIORBEES.NS and FREE_CASH, rendered with pygame.

Action      : Box(0, 1, shape=(4,)) -> target weights for
              [NIFTYBEES.NS, GOLDBEES.NS, JUNIORBEES.NS, FREE_CASH].
              Any non-negative vector is accepted and normalised to sum to 1
              (an all-zero action means 100% cash).
Observation : (11,) vector
                [0:3]  mean over the window of log(Close / rolling VWAP) per ETF
                [3:6]  std  over the window of log(Close / rolling VWAP) per ETF
                [6:10] current portfolio weights (post-drift)
                [10]   Differential Sharpe Ratio of the portfolio (Moody & Saffell)
Reward      : differential Sharpe ratio of the portfolio (default, reward_type="dsr");
              reward_type="log_return" gives the log growth of portfolio value instead.

Install:  pip install gymnasium pygame numpy pandas yfinance
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces

ETFS = ["NIFTYBEES.NS", "GOLDBEES.NS", "JUNIORBEES.NS"]
ASSETS = ETFS + ["FREE_CASH"]


# --------------------------------------------------------------------------- #
# Data helpers
# --------------------------------------------------------------------------- #
def load_market_data(start="2017-09-01", end=None, tickers=ETFS) -> dict:
    """Download adjusted OHLCV from Yahoo Finance up to today (end=None).
    Starts before 2018 so the feature windows have warm-up history."""
    import yfinance as yf

    raw = yf.download(list(tickers), start=start, end=end, auto_adjust=True, progress=False)
    close = raw["Close"][list(tickers)].dropna()
    if close.empty:
        raise RuntimeError("No price data downloaded.")
    out = {"close": close}
    for key, field in [("high", "High"), ("low", "Low"), ("volume", "Volume")]:
        out[key] = raw[field][list(tickers)].reindex(close.index).fillna(0.0)
    return out


def synthetic_market_data(n_days=2200, seed=0, tickers=ETFS) -> dict:
    """Correlated GBM close prices + synthetic high/low/volume (fallback when offline)."""
    rng = np.random.default_rng(seed)
    mu = np.array([0.12, 0.09, 0.15]) / 252
    vol = np.array([0.16, 0.14, 0.20]) / np.sqrt(252)
    corr = np.array([[1.0, 0.0, 0.9], [0.0, 1.0, 0.05], [0.9, 0.05, 1.0]])
    L = np.linalg.cholesky(corr)
    z = rng.standard_normal((n_days, 3)) @ L.T
    close = 100 * np.cumprod(1 + mu + vol * z, axis=0)
    spread = np.abs(rng.normal(0, 0.006, (n_days, 3)))
    idx = pd.bdate_range("2017-09-01", periods=n_days)
    mk = lambda x: pd.DataFrame(x, index=idx, columns=list(tickers))
    return {
        "close": mk(close),
        "high": mk(close * (1 + spread)),
        "low": mk(close * (1 - spread)),
        "volume": mk(rng.lognormal(12, 0.4, (n_days, 3))),
    }


# --------------------------------------------------------------------------- #
# Environment
# --------------------------------------------------------------------------- #
class PortfolioEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 10}

    def __init__(
        self,
        data: dict | pd.DataFrame | None = None,
        window: int = 20,
        episode_length: int | None = 256,
        sample_start: str = "2018-01-01",     # earliest date an episode may begin
        initial_cash: float = 1_000_000.0,
        transaction_cost: float = 0.001,      # 10 bps per unit of traded notional
        cash_return_annual: float = 0.0,      # return earned by FREE_CASH
        dsr_eta: float = 0.01,                # EMA rate for the differential Sharpe ratio
        reward_type: str = "dsr",             # "dsr" (default) or "log_return"
        render_mode: str | None = None,
    ):
        super().__init__()
        if data is None:
            try:
                data = load_market_data()
            except Exception as e:  # no network / yfinance missing
                print(f"[PortfolioEnv] Could not download data ({e}); using synthetic data.")
                data = synthetic_market_data()
        if isinstance(data, pd.DataFrame):            # close only -> VWAP degenerates to price average
            data = {"close": data}
        close = data["close"][ETFS].dropna()
        assert len(close) > 2 * window + 2, "Not enough price history."
        high = data.get("high", close)[ETFS].reindex(close.index).fillna(close)
        low = data.get("low", close)[ETFS].reindex(close.index).fillna(close)
        volume = data.get("volume", pd.DataFrame(1.0, index=close.index, columns=ETFS))
        volume = volume[ETFS].reindex(close.index).fillna(0.0)
        assert reward_type in ("log_return", "dsr")

        self.window = window
        self.sample_start = pd.Timestamp(sample_start)
        self.episode_length = episode_length
        self.initial_cash = initial_cash
        self.tc = transaction_cost
        self.cash_ret = (1 + cash_return_annual) ** (1 / 252) - 1
        self.eta = dsr_eta
        self.reward_type = reward_type
        self.n_assets = len(ASSETS)

        self.prices_df = close
        self.prices = close.to_numpy(dtype=np.float64)
        self.dates = close.index
        self.returns = self.prices[1:] / self.prices[:-1] - 1.0   # returns[i] : day i -> i+1
        self.features = self._vwap_features(close, high, low, volume)   # (T, 6)

        n_obs = 2 * len(ETFS) + self.n_assets + 1
        self.action_space = spaces.Box(0.0, 1.0, shape=(self.n_assets,), dtype=np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(n_obs,), dtype=np.float32)

        # rendering
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        self.W, self.H = 1000, 620
        self.screen = None
        self.clock = None
        self.font = self.small = self.big = None

    # ------------------------------------------------------------------ #
    def _vwap_features(self, close, high, low, volume) -> np.ndarray:
        """NOTE ON EPISODE START: these features are computed once over the *whole*
        downloaded history (which begins ~4 months before 2018), not per episode.
        An episode may only start at index >= 2*window, so at reset the features
        for day t already use real market data from days t-2*window+1 .. t
        (nothing after t, so no look-ahead). Only the portfolio state (weights,
        DSR moments) starts fresh at reset.

        Per ETF: mean and std over `window` days of x_t = log(Close_t / VWAP_t),
        where VWAP_t is the rolling `window`-day volume-weighted average of the
        typical price (H+L+C)/3 (daily bars, so this is a daily-bar VWAP proxy)."""
        w = self.window
        tp = (high + low + close) / 3.0
        vwap = (tp * volume).rolling(w).sum() / volume.rolling(w).sum().replace(0.0, np.nan)
        dev = np.log(close / vwap)
        mean = dev.rolling(w).mean()
        std = dev.rolling(w).std()
        feats = pd.concat([mean, std], axis=1).replace([np.inf, -np.inf], np.nan)
        return feats.fillna(0.0).to_numpy(dtype=np.float64)

    def _obs(self):
        return np.concatenate([self.features[self.t], self.weights, [self.dsr]]).astype(np.float32)

    def _update_dsr(self, R: float) -> float:
        """Differential Sharpe ratio (Moody & Saffell, 2001) with EMA moments A, B."""
        dA, dB = R - self.A, R * R - self.B
        var = self.B - self.A ** 2
        d = 0.0 if var < 1e-10 else (self.B * dA - 0.5 * self.A * dB) / var ** 1.5
        self.A += self.eta * dA
        self.B += self.eta * dB
        return float(np.clip(d, -10.0, 10.0))

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Any random block of `episode_length` days, beginning on/after sample_start
        # and ending on the latest available date.
        first_start = max(2 * self.window, int(self.dates.searchsorted(self.sample_start)))
        ep = self.episode_length or 1
        last_start = len(self.returns) - ep
        if last_start < first_start:
            raise RuntimeError(
                f"Only {len(self.returns) - first_start} usable days since {self.sample_start.date()}; "
                f"need at least {ep}."
            )
        self.start = int(self.np_random.integers(first_start, last_start + 1))
        self.t = self.start
        self.end = len(self.returns) if self.episode_length is None else min(
            self.start + self.episode_length, len(self.returns)
        )

        self.value = self.initial_cash
        self.weights = np.array([0, 0, 0, 1.0])            # start 100% in cash
        self.value_hist = [self.value]
        self.weight_hist = [self.weights.copy()]
        self.bench_hist = [1.0]                            # equal-weight ETF buy&hold (normalised)
        self.bench_units = np.ones(3) / 3
        # DSR moment estimates A (mean return) and B (mean squared return), seeded from the
        # equal-weight ETF returns of the `window` days *before* the episode. Without this,
        # the variance estimate is ~0 on the first steps and the DSR explodes.
        ew = self.returns[self.start - self.window : self.start].mean(axis=1)
        self.A, self.B = float(ew.mean()), float((ew ** 2).mean())
        self.dsr = 0.0
        self.last_reward = 0.0
        self.last_cost = 0.0
        self.total_cost = 0.0
        self.last_action = self.weights.copy()
        return self._obs(), self._info()

    def step(self, action):
        a = np.clip(np.asarray(action, dtype=np.float64), 0.0, None)
        s = a.sum()
        target = a / s if s > 1e-8 else np.array([0, 0, 0, 1.0])

        # rebalance at today's close; pay costs on traded ETF notional
        traded = np.abs(target[:3] - self.weights[:3]).sum()
        cost = self.tc * traded
        v_after_cost = self.value * (1 - cost)

        # market moves to the next day
        r = np.append(self.returns[self.t], self.cash_ret)
        growth = float(target @ (1 + r))
        new_value = v_after_cost * growth
        self.weights = target * (1 + r) / growth           # drifted weights
        log_ret = float(np.log(new_value / self.value))
        self.dsr = self._update_dsr(new_value / self.value - 1.0)
        reward = log_ret if self.reward_type == "log_return" else self.dsr

        self.value = new_value
        self.t += 1
        self.last_reward, self.last_cost, self.last_action = reward, cost * v_after_cost, target
        self.total_cost += self.last_cost
        self.value_hist.append(self.value)
        self.weight_hist.append(self.weights.copy())
        self.bench_hist.append(self.bench_hist[-1] * float(np.mean(1 + r[:3])))  # daily-rebalanced EW

        terminated = self.t >= self.end
        truncated = False
        if self.render_mode == "human":
            self.render()
        obs = self._obs() if self.t < len(self.features) else self._obs_last()
        return obs, reward, terminated, truncated, self._info()

    def _obs_last(self):
        return np.concatenate([self.features[-1], self.weights, [self.dsr]]).astype(np.float32)

    def _info(self):
        return {
            "portfolio_value": self.value,
            "weights": dict(zip(ASSETS, self.weights.round(4))),
            "date": str(self.dates[min(self.t, len(self.dates) - 1)].date()),
            "total_cost": self.total_cost,
            "dsr": self.dsr,
        }

    # ------------------------------------------------------------------ #
    # Rendering
    # ------------------------------------------------------------------ #
    COLORS = {
        "NIFTYBEES.NS": (66, 133, 244),
        "GOLDBEES.NS": (255, 193, 7),
        "JUNIORBEES.NS": (52, 168, 83),
        "FREE_CASH": (160, 160, 170),
    }
    BG, PANEL, TXT, MUTED = (18, 20, 26), (28, 31, 40), (235, 237, 242), (130, 135, 150)

    def _init_pygame(self):
        import pygame

        pygame.init()
        pygame.font.init()
        if self.render_mode == "human":
            pygame.display.init()
            self.screen = pygame.display.set_mode((self.W, self.H))
            pygame.display.set_caption("Portfolio Environment")
        else:
            self.screen = pygame.Surface((self.W, self.H))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("dejavusans,arial", 16)
        self.small = pygame.font.SysFont("dejavusans,arial", 13)
        self.big = pygame.font.SysFont("dejavusans,arial", 26, bold=True)

    def _text(self, s, pos, font=None, color=None):
        surf = (font or self.font).render(s, True, color or self.TXT)
        self.screen.blit(surf, pos)

    def _line(self, rect, series, color, lo, hi, width=2):
        import pygame

        if len(series) < 2:
            return
        x0, y0, w, h = rect
        n = len(series)
        pts = []
        for i, v in enumerate(series):
            x = x0 + w * i / max(self.end - self.start, n - 1)
            y = y0 + h - h * (v - lo) / max(hi - lo, 1e-9)
            pts.append((x, y))
        pygame.draw.lines(self.screen, color, False, pts, width)

    def render(self):
        if self.render_mode is None:
            return None
        import pygame

        if self.screen is None:
            self._init_pygame()
        if self.render_mode == "human":
            pygame.event.pump()

        S = self.screen
        S.fill(self.BG)

        # ---- header -------------------------------------------------- #
        ret = self.value / self.initial_cash - 1
        self._text("Portfolio Value", (24, 16), self.small, self.MUTED)
        self._text(f"₹ {self.value:,.0f}".replace("₹", "Rs."), (24, 34), self.big)
        col = (80, 200, 120) if ret >= 0 else (235, 90, 90)
        self._text(f"{ret * 100:+.2f}%", (260, 40), self.font, col)
        d = self.dates[min(self.t, len(self.dates) - 1)].date()
        self._text(f"Date: {d}   Step: {self.t - self.start}/{self.end - self.start}", (24, 74), self.small, self.MUTED)
        self._text(f"DSR: {self.dsr:+.3f}   Reward: {self.last_reward:+.5f}   Last cost: {self.last_cost:,.0f}   Total cost: {self.total_cost:,.0f}",
                   (24, 94), self.small, self.MUTED)

        # ---- equity chart -------------------------------------------- #
        cx, cy, cw, ch = 24, 140, 600, 250
        pygame.draw.rect(S, self.PANEL, (cx - 10, cy - 28, cw + 20, ch + 60), border_radius=10)
        self._text("Equity curve (normalised)", (cx, cy - 22), self.small, self.MUTED)
        eq = np.array(self.value_hist) / self.initial_cash
        bm = np.array(self.bench_hist)
        lo, hi = min(eq.min(), bm.min()) * 0.98, max(eq.max(), bm.max()) * 1.02
        for k in range(5):
            gy = cy + ch * k / 4
            pygame.draw.line(S, (45, 49, 62), (cx, gy), (cx + cw, gy))
            self._text(f"{hi - (hi - lo) * k / 4:.3f}", (cx + cw - 50, gy - 15), self.small, self.MUTED)
        self._line((cx, cy, cw, ch), bm, (120, 125, 140), lo, hi, 1)
        self._line((cx, cy, cw, ch), eq, (80, 200, 255), lo, hi, 2)
        pygame.draw.line(S, (200, 200, 90), (cx, cy + ch - ch * (1 - lo) / (hi - lo)),
                         (cx + cw, cy + ch - ch * (1 - lo) / (hi - lo)), 1)
        pygame.draw.circle(S, (80, 200, 255), (cx + 40, cy + ch + 22), 5)
        self._text("Agent", (cx + 50, cy + ch + 14), self.small)
        pygame.draw.circle(S, (120, 125, 140), (cx + 130, cy + ch + 22), 5)
        self._text("Equal-weight ETFs", (cx + 140, cy + ch + 14), self.small)

        # ---- stacked weight history ---------------------------------- #
        wx, wy, ww, wh = 24, 450, 600, 130
        pygame.draw.rect(S, self.PANEL, (wx - 10, wy - 28, ww + 20, wh + 40), border_radius=10)
        self._text("Allocation history", (wx, wy - 22), self.small, self.MUTED)
        wh_arr = np.array(self.weight_hist)
        total_steps = max(self.end - self.start, len(wh_arr) - 1)
        for i in range(len(wh_arr) - 1):
            x1 = wx + ww * i / total_steps
            x2 = wx + ww * (i + 1) / total_steps
            base = wy + wh
            for j, name in enumerate(ASSETS):
                hgt = wh * wh_arr[i, j]
                pygame.draw.rect(S, self.COLORS[name], (x1, base - hgt, max(x2 - x1, 1) + 1, hgt))
                base -= hgt

        # ---- right panel: bars + legend ------------------------------ #
        px, py, pw, ph = 660, 140, 316, 440
        pygame.draw.rect(S, self.PANEL, (px, py - 28, pw, ph + 28 + 10), border_radius=10)
        self._text("Current weights (post-drift)", (px + 14, py - 22), self.small, self.MUTED)
        bar_w = pw - 160
        for i, name in enumerate(ASSETS):
            y = py + 20 + i * 62
            self._text(name, (px + 14, y), self.small)
            pygame.draw.rect(S, (45, 49, 62), (px + 14, y + 22, bar_w, 16), border_radius=4)
            pygame.draw.rect(S, self.COLORS[name], (px + 14, y + 22, bar_w * self.weights[i], 16), border_radius=4)
            self._text(f"{self.weights[i] * 100:5.1f}%", (px + 24 + bar_w, y + 20), self.small)
            self._text(f"target {self.last_action[i] * 100:4.1f}%", (px + 24 + bar_w, y + 4), self.small, self.MUTED)

        # pie chart
        import math

        cxp, cyp, R = px + pw // 2, py + 340, 70
        start = -math.pi / 2
        for i, name in enumerate(ASSETS):
            ang = 2 * math.pi * self.weights[i]
            if ang < 1e-4:
                continue
            pts = [(cxp, cyp)]
            steps = max(2, int(ang * 20))
            for k in range(steps + 1):
                a = start + ang * k / steps
                pts.append((cxp + R * math.cos(a), cyp + R * math.sin(a)))
            pygame.draw.polygon(S, self.COLORS[name], pts)
            start += ang
        pygame.draw.circle(S, self.PANEL, (cxp, cyp), R // 2)

        if self.render_mode == "human":
            pygame.event.pump()
            pygame.display.flip()
            self.clock.tick(self.metadata["render_fps"])
            return None
        return np.transpose(np.array(pygame.surfarray.pixels3d(S)), (1, 0, 2))

    def close(self):
        if self.screen is not None:
            import pygame

            pygame.display.quit()
            pygame.quit()
            self.screen = None
