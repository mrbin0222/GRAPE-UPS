#!/usr/bin/env python3
"""Paired exact sign-flip tests, Holm correction and bootstrap mean differences.

CSV columns: difficulty, seed, method, metric. Each row is one independent
realization, not a battery event. All requested comparisons form one Holm family.
"""
import argparse,json
import numpy as np
import pandas as pd
p=argparse.ArgumentParser();p.add_argument('csv');p.add_argument('--reference',default='GRAPE-UPS');p.add_argument('--output',required=True);p.add_argument('--bootstrap',type=int,default=100000);p.add_argument('--seed',type=int,default=20260926);a=p.parse_args()
f=pd.read_csv(a.csv)
if f.duplicated(['difficulty','seed','method']).any():raise ValueError('Duplicate realization/method rows.')
rows=[];rng=np.random.default_rng(a.seed)
for difficulty,g in f.groupby('difficulty'):
 w=g.pivot(index='seed',columns='method',values='metric')
 if a.reference not in w:raise ValueError('Missing reference method.')
 for method in w.columns:
  if method==a.reference:continue
  pair=w[[a.reference,method]]
  if pair.isna().any().any():raise ValueError('Incomplete pairs must be resolved explicitly.')
  d=(pair[a.reference]-pair[method]).to_numpy(float);n=len(d)
  if n<2 or n>20 or not np.isfinite(d).all():raise ValueError('Exact test requires 2..20 finite pairs.')
  signs=1-2*((np.arange(2**n,dtype=np.uint32)[:,None]>>np.arange(n))&1).astype(float)
  null=np.abs((signs@d)/n);raw=float(np.mean(null>=abs(d.mean())-1e-12))
  boot=np.mean(rng.choice(d,size=(a.bootstrap,n),replace=True),axis=1)
  lo,hi=np.quantile(boot,[.025,.975]);rows.append(dict(difficulty=difficulty,method=method,n=n,mean_difference=float(d.mean()),bootstrap_low=float(lo),bootstrap_high=float(hi),raw_p=raw))
order=np.argsort([r['raw_p'] for r in rows]);previous=0.
for rank,i in enumerate(order):
 previous=max(previous,min(1.,(len(rows)-rank)*rows[i]['raw_p']));rows[i]['holm_p']=previous
pd.DataFrame(rows).to_csv(a.output,index=False)
