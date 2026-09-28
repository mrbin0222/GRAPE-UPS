#!/usr/bin/env python3
"""Evaluate any supplied battery-event scores; no comparator implementations."""
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score,roc_auc_score,f1_score,precision_score,recall_score
p=argparse.ArgumentParser();p.add_argument('csv');p.add_argument('--score',required=True);p.add_argument('--label',default='label_anomaly');p.add_argument('--split',default='test_external');p.add_argument('--threshold',type=float);p.add_argument('--output',required=True);a=p.parse_args()
f=pd.read_csv(a.csv)
if 'split' in f:f=f[f.split.eq(a.split)]
y=f[a.label].to_numpy(int);s=f[a.score].to_numpy(float)
if not len(y) or set(np.unique(y))!={0,1} or not np.isfinite(s).all():raise ValueError('Need finite scores and both binary classes.')
r={'n':len(y),'AUPRC':float(average_precision_score(y,s)),'AUROC':float(roc_auc_score(y,s))}
if a.threshold is not None:
 pred=s>=a.threshold
 r.update(F1=float(f1_score(y,pred)),precision=float(precision_score(y,pred,zero_division=0)),recall=float(recall_score(y,pred)),FPR=float(pred[y==0].mean()))
Path(a.output).parent.mkdir(parents=True,exist_ok=True)
Path(a.output).write_text(json.dumps(r,indent=2)+'\n')
