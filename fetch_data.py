import requests, pandas as pd, datetime as dt, pathlib

sym = 'BTC'
tf = '1m'
days = 3.4

end = int(dt.datetime.utcnow().timestamp() * 1000)
start = int((dt.datetime.utcnow() - dt.timedelta(days=days)).timestamp() * 1000)
url = 'https://api.binance.com/api/v3/klines'
params = {'symbol': f'{sym}USDT', 'interval': tf, 'startTime': start, 'endTime': end, 'limit': 1000}

rows = []
while start < end:
    res = requests.get(url, params=params, timeout=15).json()
    if not res:
        break
    rows.extend(res)
    start = res[-1][0] + 60_000
    params['startTime'] = start

print('Downloaded', len(rows), 'candles')

cols = ['ts', 'open', 'high', 'low', 'close', 'vol', 'close_time', 'quote_vol', 'trades', 'taker_base', 'taker_quote', 'ignore']
df = pd.DataFrame(rows, columns=cols)
# Keep only the main columns
df = df[['ts', 'open', 'high', 'low', 'close', 'vol']]
out = pathlib.Path('data/BTC_1m.parquet')
df.to_parquet(out)
print('Saved ->', out)