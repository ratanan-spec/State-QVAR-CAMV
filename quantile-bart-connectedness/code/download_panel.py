"""Rebuild a one-step return panel from Binance monthly public spot klines.
Raw downloads and reconstructed prices/returns are local outputs, not bundled.
"""
import argparse,calendar,csv,io,json,zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.request import Request,urlopen
from urllib.error import HTTPError
import numpy as np,pandas as pd
ASSETS=['BTCUSDT','ETHUSDT','BNBUSDT','XRPUSDT','ADAUSDT']
def main():
 p=argparse.ArgumentParser();p.add_argument('--start',default='2018-05');p.add_argument('--end',default='2026-10');p.add_argument('--out',required=True);p.add_argument('--workers',type=int,default=8);a=p.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
 months=pd.period_range(a.start,a.end,freq='M');jobs=[(asset,str(m)) for asset in ASSETS for m in months]
 def fetch(job):
  asset,month=job;name=f'{asset}-1d-{month}.zip';cache=out/'cache'/name;cache.parent.mkdir(exist_ok=True)
  if cache.exists():blob=cache.read_bytes()
  else:
   url=f'https://data.binance.vision/data/spot/monthly/klines/{asset}/1d/{name}'
   try:
    with urlopen(Request(url,headers={'User-Agent':'Q-BART-replication/0.2'}),timeout=60) as r:blob=r.read()
   except HTTPError as e:
    if e.code==404:return asset,[],{'file':name,'status':'not_available_404'}
    raise
   cache.write_bytes(blob)
  vals=[]
  with zipfile.ZipFile(io.BytesIO(blob)) as z:
   with z.open(z.namelist()[0]) as h:
    for row in csv.reader(io.TextIOWrapper(h,encoding='utf-8')):
     if not row or not row[0].isdigit():continue
     ts=int(row[0]);dt=pd.to_datetime(ts,unit='us' if ts>=100000000000000 else 'ms',utc=True).normalize();vals.append((dt,float(row[4])))
  return asset,vals,{'file':name,'status':'ok'}
 rows={x:[] for x in ASSETS};manifest=[]
 with ThreadPoolExecutor(max_workers=a.workers) as ex:
  for asset,values,status in ex.map(fetch,jobs):rows[asset].extend(values);manifest.append(status)
 series=[]
 for asset in ASSETS:series.append(pd.DataFrame(rows[asset],columns=['date',asset]).drop_duplicates('date').set_index('date'))
 prices=pd.concat(series,axis=1,join='inner').sort_index();returns=100*np.log(prices).diff();panel=pd.DataFrame(index=prices.index)
 for asset in ASSETS:
  panel[asset+'_ret']=returns[asset];panel[asset+'_ret_lag1']=returns[asset].shift(1);panel[asset+'_ret_lead1']=returns[asset].shift(-1)
 # Keep the last return row so target-date construction retains the final response.
 panel=panel.loc[panel[[x+'_ret' for x in ASSETS]].notna().all(axis=1)];panel.to_csv(out/'model_panel.csv',index_label='date')
 (out/'download_manifest.json').write_text(json.dumps({'start':a.start,'end':a.end,'files':manifest,'predictor':'current return at t','target':'return at t+1','terminal_return_retained':True},indent=2))
if __name__=='__main__':main()
