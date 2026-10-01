import time
import pandas as pd
from environment import StockPortfolioEnv
from constants.stocks import STOCK_LIST

stocks = dict()
for stock in STOCK_LIST:
    path = f"data/{stock}.csv"

    df = pd.read_csv(path, parse_dates=['Date'])
    stocks[stock] = df

# Initialize environment with render_mode="human"
env = StockPortfolioEnv(
    stocks=stocks,
    initial_capital=100000,
    step_size=30,
    window_size=365,
    max_drawdown_limit=0.15,
    render_mode="human",
)

for _ in range(10):
    obs, info = env.reset()
    terminated = False

    while not terminated:
        action = env.action_space.sample()  # Random allocation step
        obs, reward, terminated, truncated, info = env.step(action)

        time.sleep(1)  # Slow down execution slightly for visual inspection

env.close()