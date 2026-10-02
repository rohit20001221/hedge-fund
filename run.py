from environment import PortfolioEnv
import numpy as np
import pygame
import sys

if __name__ == "__main__":
    env = PortfolioEnv(render_mode="human", cash_return_annual=0.06)      # defaults: sample_start=2018-01-01, episode_length=256
    obs, info = env.reset()                      # no seed -> new random window each run

    start_date = env.dates[env.start].date()
    end_date = env.dates[min(env.end, len(env.dates) - 1)].date()
    print(f"Episode: {start_date} -> {end_date} ({env.end - env.start} days)")

    policy = sys.argv[1] if len(sys.argv) > 1 else "random"
    done, total = False, 0.0
    while not done:
        if policy == "equal":
            action = np.array([1, 1, 1, 0], dtype=np.float32)
        else:
            action = env.action_space.sample()
    
        obs, r, terminated, truncated, info = env.step(action)

        total += r
        done = terminated or truncated

        if any(e.type == pygame.QUIT for e in pygame.event.get()):
            break

    print("Final value:", round(info["portfolio_value"], 2), "| log return:", round(total, 4))
    env.close()