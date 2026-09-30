from environment import TradingEnvironment

from constants.stocks import STOCK_LIST
import json
import datetime

config = json.load(open("data/meta.json"))

start_date = datetime.datetime.strptime(config["start_date"], "%Y-%m-%d").date()
end_date = start_date + datetime.timedelta(days=360)

env = TradingEnvironment(
    historical_data_paths=STOCK_LIST,
    start_date=start_date,
    end_date=end_date
)

print(env._get_info())