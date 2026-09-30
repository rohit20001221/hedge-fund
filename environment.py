import gymnasium as gym
import pandas as pd
import datetime
import numpy as np
import talib as ta

def load_historical_data(path: str):
    df = pd.read_csv(path, parse_dates=["Date"])
    
    df.dropna(inplace=True)
    return df

def slice_historical_data(df: pd.DataFrame, start_date: datetime.date, end_date: datetime.date):
    start_date = pd.Timestamp(start_date)
    end_date = pd.Timestamp(end_date)
    
    return df[
        (df['Date'] >= start_date) 
        &
        (df['Date'] <= end_date) 
    ]

def sample_window_idx(start_date: str, end_date: str, window_size=240):
    """Samples a random window of `window_size` days between start_date and end_date.

    Parameters:
    - start_date: String or datetime (e.g. '2023-01-01')
    - end_date: String or datetime (e.g. '2023-12-31')
    - window_size: Number of days in the window

    Returns:
    - Dict with random start_date, end_date, and window_size
    """
    # Convert string inputs to date objects if needed
    if isinstance(start_date, str):
        start_date = datetime.datetime.strptime(start_date, "%Y-%m-%d").date()
    if isinstance(end_date, str):
        end_date = datetime.datetime.strptime(end_date, "%Y-%m-%d").date()

    # Total days in the given range
    total_days = (end_date - start_date).days + 1

    if total_days < window_size:
        raise ValueError(
            f"Date range span ({total_days} days) is smaller than window_size ({window_size} days)."
        )

    # Calculate maximum random offset in days
    max_start_offset = total_days - window_size
    random_day_offset = np.random.randint(0, max_start_offset + 1)

    # Sample random start and end dates
    sampled_start_date = start_date + datetime.timedelta(days=int(random_day_offset))
    sampled_end_date = sampled_start_date + datetime.timedelta(days=window_size - 1)

    return {
        "start_date": sampled_start_date,
        "end_date": sampled_end_date,
        "window_size": window_size,
    }

class TradingEnvironment(gym.Env):
    def __init__(
            self, 
            historical_data_paths: list[str],
            start_date: datetime.date,
            end_date: datetime.date,
            initial_captial=100000,
            step_count=5,
            max_drawdown=0.15
        ):
        super(TradingEnvironment, self).__init__()

        self.total_stocks = len(historical_data_paths)
        self.action_space = gym.spaces.Box(
            np.zeros(self.total_stocks), 
            np.ones(self.total_stocks),
            dtype=np.float64
        )

        self.initial_captial = initial_captial
        self.portfolio_value = initial_captial
        self.current_day = 0   # for each day increment the day by 1
        self.step_count  = step_count    # holding period of the portfolio before rebalancing
        self.max_drawdown = max_drawdown

        self.weights = np.zeros(self.total_stocks)

        self.stocks: dict[str, pd.DataFrame] = dict()
        for stock in historical_data_paths:
            df = load_historical_data(f"data/{stock}.csv")
            df = slice_historical_data(df, start_date, end_date)

            self.stocks[stock] = df
            print(df)

    def _get_obs(self):
        pass

    def _get_info(self):
        pass

    def reset(self, seed = None, options = None):
        super().reset(seed=seed, options=options)
        pass

    

