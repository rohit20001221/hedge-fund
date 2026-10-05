from backtest.portfolio import load_data, PortfolioBacktester, Strategy
from strategies.model import PortfolioAgent, prepare_data, train
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import torch

tickers = ["NIFTYBEES.NS", "JUNIORBEES.NS", "GOLDBEES.NS", "FREE:CASH"]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")   

model = PortfolioAgent(num_stocks=len(tickers)).to(device)

actorOptimizer = torch.optim.Adam(model.actor.parameters())
criticOptimizer = torch.optim.Adam(model.critic.parameters())

epochs = 1000

class AgentStrategy(Strategy):
    min_history = 30

    def __init__(self, num_stocks, model):
        super().__init__()

        self.num_stocks = num_stocks
        self.count = 0
        self.weights = None
        self.model = model

        self.value = None
        self.action = None
        self.dist = None

        self.log_probs = []
        self.values = []
        self.rewards = []
        self.masks = []
        self.entropy = 0

        self.state = None

    def allocate(self, history):
        prices = history["Close"].tail(30)
        x = prepare_data(prices, self.num_stocks, self.min_history, self.weights)

        self.state = x

        dist, value = self.model(x)
        action = dist.sample()

        self.weights = np.reshape(action.cpu().numpy(), -1)
        self.action = action
        self.value = value
        self.dist = dist

        return pd.Series({
            ticker: weight
            for ticker, weight in zip(tickers, self.weights)
        })

    def on_period_end(self, stats, history):
        log_prob = self.dist.log_prob(self.action).unsqueeze(0)

        self.entropy += self.dist.entropy().mean()
        
        self.log_probs.append(log_prob)
        self.values.append(self.value)
        self.rewards.append(torch.tensor(stats.metrics["Sharpe"], dtype=torch.float32, device=device))
        self.masks.append(torch.tensor([1], dtype=torch.float, device=device))

end_date = datetime.today()
start_date = end_date - timedelta(days=365 * 4)

data = load_data(tickers, start=start_date, end=end_date, interval="1d")
backtester = PortfolioBacktester(
    rebalance="M",
    commission_bps=2,
    slippage_bps=2
)

for i in range(epochs):
    strategy = AgentStrategy(num_stocks=len(tickers), model=model)

    result = backtester.run(
        data,
        strategy,
        benchmark=None
    )

    train(
        model.actor, 
        model.critic, 
        actorOptimizer, 
        criticOptimizer, 
        strategy.state, 
        strategy.rewards, 
        strategy.masks, 
        strategy.log_probs, 
        strategy.values
    )

    if i % 20 == 0:
        result.plot(save_to=f"out/results_{i}.png")