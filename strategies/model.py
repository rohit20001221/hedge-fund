import torch.nn as nn
import numpy as np
import pandas as pd
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")   

def prepare_data(prices: pd.Series, look_back=30, initial_weights=None):
    log_returns = np.log(prices / prices.shift(1)).dropna().values
    
    cov_matrix = prices.cov()

    cov_matrix_flatten = np.reshape(cov_matrix.values, -1)
    mean = np.mean(log_returns, axis=0) * look_back
    std = np.std(log_returns, axis=0) * np.sqrt(look_back)

    if initial_weights is None:
        initial_weights = np.array([0,0,0,1])

    x = np.concatenate([cov_matrix_flatten, mean, std, initial_weights])
    x = torch.tensor(x).to(device, dtype=torch.float32).view(1, -1)

    return x

def flatten(weights):
    return weights.clone().detach().view(-1).cpu().numpy()

class PortfolioAgent(nn.Module):
    def __init__(self, num_stocks=4):
        super(PortfolioAgent, self).__init__()

        # n x n covariance matrix / n expected log_returns / n std deviations / n weights initially [0,0,0,1] where 1 is the cash
        # flatten the covriance matrix

        self.input_size = num_stocks * num_stocks + num_stocks + num_stocks + num_stocks
        self.hidden_size = self.input_size * 5

        self.actor_network = nn.Sequential(
            nn.Linear(self.input_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, num_stocks),
            nn.Softmax(dim=1)
        )

        self.critic_network = nn.Sequential(
            nn.Linear(self.input_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, 1)
        )

    def forward(self, x):
        weights = self.actor_network(x)
        value = self.critic_network(x)

        return weights, value