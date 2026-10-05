import torch.nn as nn
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.distributions import Dirichlet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")   

def prepare_data(prices: pd.Series, num_stocks, look_back=30, initial_weights=None):
    log_returns = np.log(prices / prices.shift(1)).dropna().values
    
    cov_matrix = prices.cov()

    cov_matrix_flatten = np.reshape(cov_matrix.values, -1)
    mean = np.mean(log_returns, axis=0) * look_back
    std = np.std(log_returns, axis=0) * np.sqrt(look_back)

    if initial_weights is None:
        initial_weights = [1 / num_stocks for _ in range(num_stocks)]
        
        initial_weights = np.array(initial_weights)

    x = np.concatenate([cov_matrix_flatten, mean, std, initial_weights])
    x = torch.tensor(x).to(device, dtype=torch.float32).view(1, -1)

    return x

def flatten(weights):
    return weights.detach().view(-1).cpu().numpy()

class Actor(nn.Module):
    def __init__(self, state_size, hidden_Size, action_size):
        super(Actor, self).__init__()

        self.linear_a = nn.Linear(state_size, hidden_Size)
        self.linear_b = nn.Linear(hidden_Size, hidden_Size)
        self.linear_c = nn.Linear(hidden_Size, action_size)

    def forward(self, state):
        h = F.relu(self.linear_a(state))
        h = F.relu(self.linear_b(h))

        h = self.linear_c(h)
        distribution = Dirichlet(F.softmax(h, dim=0))

        return distribution

class Critic(nn.Module):
    def __init__(self, state_size, hidden_size):
        super(Critic, self).__init__()

        self.linear_a = nn.Linear(state_size, hidden_size)
        self.linear_b = nn.Linear(hidden_size, hidden_size)
        self.linear_c = nn.Linear(hidden_size, 1)

    def forward(self, state):
        h = F.relu(self.linear_a(state))
        h = F.relu(self.linear_b(h))

        value = self.linear_c(h)
        return value

class PortfolioAgent(nn.Module):
    def __init__(self, num_stocks=4):
        super(PortfolioAgent, self).__init__()

        # n x n covariance matrix / n expected log_returns / n std deviations / n weights initially [0,0,0,1] where 1 is the cash
        # flatten the covriance matrix

        self.state_size = num_stocks * num_stocks + num_stocks + num_stocks + num_stocks
        self.hidden_size = self.state_size * 5
        self.action_size = num_stocks

        self.actor = Actor(self.state_size, self.hidden_size, self.action_size)
        self.critic = Critic(self.state_size, self.hidden_size)


    def forward(self, x):
        dist = self.actor(x)
        value = self.critic(x)

        return dist, value

def compute_returns(next_value, rewards, masks, gamma=0.99):
    R = next_value
    returns = []

    for step in reversed(range(len(rewards))):
        R = rewards[step] + gamma * R * masks[step]
        returns.insert(0, R)

    return returns

def train(
        actor, 
        critic, 
        actorOptimizer, 
        criticOptimizer, 
        next_state,
        rewards,
        masks,
        log_probs,
        values,
    ):

    # next_state = torch.FloatTensor(next_state).to(device)
    next_value = critic(next_state)
    returns = compute_returns(next_value, rewards, masks)

    log_probs = torch.cat(log_probs)
    returns = torch.cat(returns).detach()
    values = torch.cat(values)

    advantage = returns - values

    actor_loss = -(log_probs * advantage.detach()).mean()
    critic_loss = advantage.pow(2).mean()

    actorOptimizer.zero_grad()
    criticOptimizer.zero_grad()

    actor_loss.backward()
    critic_loss.backward()

    actorOptimizer.step()
    criticOptimizer.step()