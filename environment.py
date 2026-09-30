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
        self.action_space = gym.spaces.Box(
            np.zeros(self.total_stocks), 
            np.ones(self.total_stocks),
            dtype=np.float64
        )

        self.total_stocks = len(historical_data_paths)
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

    def _get_obs(self):
        pass

    def _get_info(self):
        pass

    def reset(self, seed = None, options = None):
        super().reset(seed=seed, options=options)
        pass

    

