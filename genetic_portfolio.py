from backtest.portfolio import load_data, PortfolioBacktester, Strategy
from strategies.genetic import genetic_algorithm
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

tickers = ["NIFTYBEES.NS", "JUNIORBEES.NS", "LIQUIDCASE.NS", "GOLDBEES.NS"]

class GeneticStrategy(Strategy):
    min_history = 30

    def allocate(self, history):
        prices = history["Close"].tail(30)

        log_returns = np.log(prices / prices.shift(1)).dropna()
        cov_matrix  = prices.cov()

        solution = genetic_algorithm(log_returns, cov_matrix, len(tickers), population_size=300, num_generations=100)
        weights = solution["weights"]


        return pd.Series({
            ticker: weight
            for ticker, weight in zip(tickers, weights)
        })

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
    GeneticStrategy(),
    benchmark=None
)

result.plot()