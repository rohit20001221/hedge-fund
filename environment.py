from typing import Any, Dict, Optional, Tuple
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import pygame


class StockPortfolioEnv(gym.Env):
    """Custom Gymnasium Environment for Stock Portfolio Allocation using Total Sharpe Ratio as Reward."""

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(
        self,
        stocks: Dict[str, pd.DataFrame],
        initial_capital: float = 100000.0,
        step_size: int = 5,
        window_size: int = 365,
        max_drawdown_limit: float = 0.20,
        risk_free_rate: float = 0.0,  # Annualized risk-free rate (e.g., 0.02 for 2%)
        render_mode: Optional[str] = None,
    ):
        super().__init__()

        self.stocks = stocks
        self.stock_names = sorted(list(stocks.keys()))
        self.num_stocks = len(self.stock_names)

        self.initial_capital = float(initial_capital)
        self.step_size = int(step_size)
        self.window_size = int(window_size)
        self.max_drawdown_limit = float(max_drawdown_limit)
        self.render_mode = render_mode

        # Daily risk-free rate assuming 252 trading days/year
        self.daily_risk_free_rate = float(risk_free_rate) / 252.0

        # Color palette for rendering individual stock allocations
        self.stock_colors = [
            (255, 99, 132),
            (54, 162, 235),
            (255, 206, 86),
            (75, 192, 192),
            (153, 102, 255),
            (255, 159, 64),
            (46, 204, 113),
            (231, 76, 60),
        ]

        for name in self.stock_names:
            self.stocks[name] = (
                self.stocks[name].sort_values("Date").reset_index(drop=True)
            )

        self.total_records = len(self.stocks[self.stock_names[0]])
        if self.total_records < self.window_size:
            raise ValueError(
                f"Dataframe length ({self.total_records}) must be at least window_size ({self.window_size})"
            )

        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.num_stocks,), dtype=np.float32
        )

        self.observation_space = spaces.Dict(
            {
                "portfolio_value": spaces.Box(
                    low=0, high=np.inf, shape=(1,), dtype=np.float32
                ),
                "initial_capital": spaces.Box(
                    low=0, high=np.inf, shape=(1,), dtype=np.float32
                ),
                "stock_history": spaces.Dict(
                    {
                        name: spaces.Box(
                            low=-np.inf,
                            high=np.inf,
                            shape=(self.step_size, 5),
                            dtype=np.float32,
                        )
                        for name in self.stock_names
                    }
                ),
            }
        )

        self.window_width = 1000
        self.window_height = 600
        self.window = None
        self.clock = None

        self.window_start_idx = 0
        self.window_end_idx = 0
        self.current_idx = 0
        self.portfolio_value = self.initial_capital
        self.max_portfolio_value = self.initial_capital
        self.max_drawdown = 0.0
        self.weights = np.ones(self.num_stocks, dtype=np.float32) / self.num_stocks
        self.portfolio_history = []
        self.all_portfolio_returns = []

    def _calculate_total_sharpe(self) -> float:
        """Calculates annualized Total Sharpe Ratio from all daily returns since reset."""
        if len(self.all_portfolio_returns) < 2:
            return 0.0
        returns_array = np.array(self.all_portfolio_returns, dtype=np.float32)
        mean_return = np.mean(returns_array)
        std_return = np.std(returns_array, ddof=1)
        excess_returns = mean_return - self.daily_risk_free_rate
        return float((excess_returns / (std_return + 1e-8)) * np.sqrt(252))

    def reset(
        self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        super().reset(seed=seed)

        max_start_idx = self.total_records - self.window_size
        self.window_start_idx = self.np_random.integers(0, max_start_idx + 1)
        self.window_end_idx = self.window_start_idx + self.window_size

        self.current_idx = self.window_start_idx + self.step_size
        self.portfolio_value = self.initial_capital
        self.max_portfolio_value = self.initial_capital
        self.max_drawdown = 0.0

        self.weights = np.ones(self.num_stocks, dtype=np.float32) / self.num_stocks
        self.portfolio_history = [self.portfolio_value]
        self.all_portfolio_returns = []

        obs = self._get_observation()
        info = {
            "window_start_idx": self.window_start_idx,
            "window_end_idx": self.window_end_idx,
            "max_drawdown": self.max_drawdown,
            "drawdown_exceeded": False,
            "total_sharpe_ratio": 0.0,
        }

        if self.render_mode in ["human", "rgb_array"]:
            self.render()

        return obs, info

    def step(
        self, action: np.ndarray
    ) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        exp_actions = np.exp(action - np.max(action))
        self.weights = exp_actions / np.sum(exp_actions)

        target_idx = min(self.current_idx + self.step_size, self.window_end_idx)
        actual_steps = target_idx - self.current_idx

        drawdown_exceeded = False

        for idx in range(self.current_idx, target_idx):
            daily_returns = np.zeros(self.num_stocks, dtype=np.float32)
            for i, name in enumerate(self.stock_names):
                df = self.stocks[name]
                prev_close = df.loc[idx - 1, "Close"]
                curr_close = df.loc[idx, "Close"]
                daily_returns[i] = (curr_close - prev_close) / prev_close

            portfolio_return = float(np.dot(self.weights, daily_returns))
            self.all_portfolio_returns.append(portfolio_return)

            self.portfolio_value *= 1.0 + portfolio_return
            self.portfolio_history.append(self.portfolio_value)

            # Update Peak and Max Drawdown
            if self.portfolio_value > self.max_portfolio_value:
                self.max_portfolio_value = self.portfolio_value

            current_dd = (
                self.max_portfolio_value - self.portfolio_value
            ) / self.max_portfolio_value
            if current_dd > self.max_drawdown:
                self.max_drawdown = current_dd

            # Check Max Drawdown limit breach inside step loop
            if self.max_drawdown >= self.max_drawdown_limit:
                drawdown_exceeded = True
                self.current_idx = idx + 1
                break

        if not drawdown_exceeded:
            self.current_idx = target_idx

        # --- Total Sharpe Ratio Reward Calculation ---
        total_sharpe_ratio = self._calculate_total_sharpe()

        reward = total_sharpe_ratio
        if drawdown_exceeded:
            reward -= 1.0  # Penalty for hitting max drawdown limit

        # Termination conditions
        window_finished = self.current_idx >= self.window_end_idx
        terminated = window_finished or drawdown_exceeded
        truncated = False

        obs = self._get_observation()
        info = {
            "portfolio_value": self.portfolio_value,
            "max_drawdown": self.max_drawdown,
            "total_sharpe_ratio": total_sharpe_ratio,
            "drawdown_exceeded": drawdown_exceeded,
            "steps_advanced": actual_steps,
        }

        if self.render_mode in ["human", "rgb_array"]:
            self.render()

        return obs, reward, terminated, truncated, info

    def _get_observation(self) -> Dict[str, Any]:
        history_start = max(self.window_start_idx, self.current_idx - self.step_size)

        stock_history = {}
        for name in self.stock_names:
            df = self.stocks[name]
            history_slice = df.loc[
                history_start : self.current_idx - 1,
                ["Open", "High", "Low", "Close", "Volume"],
            ].values.astype(np.float32)

            if len(history_slice) < self.step_size:
                pad_width = self.step_size - len(history_slice)
                history_slice = np.pad(
                    history_slice, ((pad_width, 0), (0, 0)), mode="constant"
                )

            stock_history[name] = history_slice

        return {
            "portfolio_value": np.array([self.portfolio_value], dtype=np.float32),
            "initial_capital": np.array([self.initial_capital], dtype=np.float32),
            "stock_history": stock_history,
        }

    def render(self) -> Optional[np.ndarray]:
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(
                (self.window_width, self.window_height)
            )
            pygame.display.set_caption("Stock Portfolio Environment")

        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_width, self.window_height))
        canvas.fill((20, 24, 33))

        font = pygame.font.SysFont("Helvetica", 18)
        bold_font = pygame.font.SysFont("Helvetica", 22, bold=True)

        profit_pct = (
            (self.portfolio_value - self.initial_capital) / self.initial_capital
        ) * 100
        pnl_color = (46, 204, 113) if profit_pct >= 0 else (231, 76, 60)

        title_surf = bold_font.render("PORTFOLIO DASHBOARD", True, (255, 255, 255))
        val_text = f"Value: ${self.portfolio_value:,.2f} ({profit_pct:+.2f}%)"
        val_surf = bold_font.render(val_text, True, pnl_color)

        mdd_color = (
            (231, 76, 60)
            if self.max_drawdown >= self.max_drawdown_limit
            else (180, 180, 180)
        )
        mdd_surf = font.render(
            f"Max DD: {self.max_drawdown * 100:.2f}% / Limit: {self.max_drawdown_limit * 100:.0f}%",
            True,
            mdd_color,
        )

        total_sharpe = self._calculate_total_sharpe()
        sharpe_color = (
            (46, 204, 113)
            if total_sharpe >= 1.0
            else ((255, 206, 86) if total_sharpe >= 0 else (231, 76, 60))
        )
        sharpe_surf = font.render(
            f"Total Sharpe: {total_sharpe:.2f}",
            True,
            sharpe_color,
        )

        day_surf = font.render(
            f"Day: {self.current_idx - self.window_start_idx}/{self.window_size}",
            True,
            (180, 180, 180),
        )

        canvas.blit(title_surf, (30, 12))
        canvas.blit(val_surf, (30, 42))
        canvas.blit(mdd_surf, (30, 72))
        canvas.blit(sharpe_surf, (420, 42))
        canvas.blit(day_surf, (self.window_width - 180, 12))

        # 1. Draw Portfolio Performance Chart
        chart_x, chart_y, chart_w, chart_h = 50, 120, 580, 420
        pygame.draw.rect(canvas, (30, 36, 48), (chart_x, chart_y, chart_w, chart_h))

        if len(self.portfolio_history) > 1:
            min_val = min(self.portfolio_history) * 0.98
            max_val = max(self.portfolio_history) * 1.02
            val_range = max(max_val - min_val, 1e-5)

            points = []
            for idx, val in enumerate(self.portfolio_history):
                px = chart_x + int((idx / self.window_size) * chart_w)
                py = chart_y + chart_h - int(((val - min_val) / val_range) * chart_h)
                points.append((px, py))

            if len(points) >= 2:
                pygame.draw.lines(canvas, (54, 162, 235), False, points, 3)

        # 2. Draw Allocation Pie Chart
        pie_cx, pie_cy, pie_radius = 800, 260, 110
        start_angle = 0.0

        for i, weight in enumerate(self.weights):
            if weight <= 0:
                continue
            extent = weight * 2 * np.pi
            color = self.stock_colors[i % len(self.stock_colors)]

            num_points = max(int(extent * 20), 10)
            angles = np.linspace(start_angle, start_angle + extent, num_points)
            slice_points = [(pie_cx, pie_cy)]
            for angle in angles:
                x = pie_cx + int(pie_radius * np.cos(angle))
                y = pie_cy + int(pie_radius * np.sin(angle))
                slice_points.append((x, y))

            if len(slice_points) > 2:
                pygame.draw.polygon(canvas, color, slice_points)

            start_angle += extent

        # 3. Draw Stock Allocation Legend
        legend_x, legend_y = 680, 410
        legend_title = bold_font.render("Current Allocation", True, (255, 255, 255))
        canvas.blit(legend_title, (legend_x, legend_y - 35))

        for i, name in enumerate(self.stock_names):
            color = self.stock_colors[i % len(self.stock_colors)]
            weight_pct = self.weights[i] * 100

            pygame.draw.rect(
                canvas, color, (legend_x, legend_y + (i * 28), 16, 16)
            )
            legend_text = font.render(
                f"{name}: {weight_pct:.1f}%", True, (220, 220, 220)
            )
            canvas.blit(legend_text, (legend_x + 26, legend_y + (i * 28) - 2))

        if self.render_mode == "human":
            self.window.blit(canvas, (0, 0))
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])

        elif self.render_mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), (1, 0, 2)
            )

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
            self.window = None