#!/usr/bin/env python3
"""Re-select five-component weights on explicitly supplied development scores.

Only validation rows are used. Do not pass confirmation runs as development data.
"""
import argparse,sys,json
from pathlib import Path
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from ups_ai.selection import candidates,select_weights,COLS,NAMES
p=argparse.ArgumentParser();p.add_argument('--development-scores',nargs='+',required=True);p.add_argument('--output',required=True);a=p.parse_args()
bundles=[]
for path in a.development_scores:
 f=pd.read_csv(path);f=f[f.split.eq('validation')]
 bundles.append((f[COLS].to_numpy(float),f.label_anomaly.to_numpy(int)))
w=candidates();idx,_,_=select_weights(bundles,w)
payload={'component_order':['relative_temporal','relative_physics','group_relative','drift','absolute_temporal','absolute_physics'],'weights':dict(zip(NAMES,w[idx].tolist())),'selection_scope':'supplied_development_validation_only','used_splits':['validation'],'candidate_count':len(w),'stability_penalty':0.15}
payload['weights']['drift']=0.
Path(a.output).parent.mkdir(parents=True,exist_ok=True)
Path(a.output).write_text(json.dumps(payload,indent=2)+'\n')
