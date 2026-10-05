from backtest.portfolio import load_data, PortfolioBacktester, Strategy
from strategies.model import PortfolioAgent, prepare_data, flatten
from datetime import datetime, timedelta
import pandas as pd
import torch

tickers = ["NIFTYBEES.NS", "JUNIORBEES.NS", "GOLDBEES.NS", "FREE:CASH"]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")   

class AgentStrategy(Strategy):
    min_history = 30

    def __init__(self, num_stocks):
        super().__init__()

        self.num_stocks = num_stocks
        self.count = 0
        self.weights = None
        self.model = PortfolioAgent(num_stocks=4).to(device)

    def allocate(self, history):
        prices = history["Close"].tail(30)
        x = prepare_data(prices, self.num_stocks, self.min_history, self.weights)
        
        weights = self.model.actor(x)
        self.weights = flatten(weights)

        return pd.Series({
            ticker: weight
            for ticker, weight in zip(tickers, self.weights)
        })

    def on_period_end(self, stats, history):
        pass

end_date = datetime.today()
start_date = end_date - timedelta(days=365 * 2)

data = load_data(tickers, start=start_date, end=end_date, interval="1d")
backtester = PortfolioBacktester(
    rebalance="M",
    commission_bps=2,
    slippage_bps=2
)

result = backtester.run(
    data,
    AgentStrategy(num_stocks=len(tickers)),
    benchmark=None
)

result.plot()