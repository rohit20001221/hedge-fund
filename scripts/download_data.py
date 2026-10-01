import yfinance
import datetime
import os
import json
from constants.stocks import STOCK_LIST

if not os.path.exists("data"):
    os.mkdir("data")

end_date = datetime.date.today()
start_date = end_date - datetime.timedelta(days= 365 * 10) # trying to download 20 years of data

for ticker in STOCK_LIST:
    df = yfinance.download(ticker, start_date, end_date, multi_level_index=False)
    df.to_csv(f"data/{ticker}.csv")

with open("data/meta.json", "w") as f:
    f.write(json.dumps({
        "start_date": str(start_date),
        "end_date": str(end_date)
    }))