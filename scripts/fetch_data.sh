#!/usr/bin/env bash
# Usage: ./scripts/fetch_data.sh BTC 7

sym=${1:-BTC}
days=${2:-7}

cd "$(dirname "$0")/.."
source .venv/bin/activate

python - <<EOF
import requests, pandas as pd, datetime as dt, pathlib

sym = '$sym'
days = $days
tf = '1m'

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

print(f'Downloaded {len(rows)} candles for {sym}')

cols = ['ts', 'open', 'high', 'low', 'close', 'vol', 'close_time', 'quote_vol', 'trades', 'taker_base', 'taker_quote', 'ignore']
df = pd.DataFrame(rows, columns=cols)
df = df[['ts', 'open', 'high', 'low', 'close', 'vol']]
out = pathlib.Path(f'data/{sym}_1m.parquet')
df.to_parquet(out)
print(f'Saved -> {out}')
EOF