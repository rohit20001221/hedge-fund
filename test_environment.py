from environment import TradingEnvironment, sample_window_idx

from constants.stocks import STOCK_LIST
import json

config = json.load(open("data/meta.json"))

sample = sample_window_idx(
    config["start_date"],
    config["end_date"],
    window_size=360
)

start_date = sample["start_date"]
end_date = sample["end_date"]

env = TradingEnvironment(
    historical_data_paths=STOCK_LIST,
    start_date=start_date,
    end_date=end_date
)
