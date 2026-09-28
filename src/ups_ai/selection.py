from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
NAMES=['relative_temporal','relative_physics','group_relative','absolute_temporal','absolute_physics']
COLS=['component_sequence_error_z','component_physics_error_z','component_group_score_z','component_abs_sequence_error_z','component_abs_physics_error_z']
ALPHAS=[.25,.30,.40,.50,.60,.70,.80,.85,.90,.95,1.]
FAMILIES=['high_resistance','low_capacity','sensor_offset','thermal_drift','slow_recovery','voltage_drift','intermittent_fault','accelerated_aging','sulfation']
TARGETS=['voltage_drift','intermittent_fault','slow_recovery']
def candidates():
 rng=np.random.default_rng(2026092106)
 a=np.vstack([np.eye(5),np.array([.35,.30,.25,0,0])/.9,[0,0,0,.6,.4],np.array([.2,.2,.25,.1,.15])/.9,rng.dirichlet(np.full(5,.35),30000),rng.dirichlet(np.ones(5),10000)])
 _,ix=np.unique(a,axis=0,return_index=True);a=a[np.sort(ix)]
 assert a.shape==(40008,5) and np.all(a>=0) and np.allclose(a.sum(1),1)
 return a

def ap_columns(y,scores):
 # Exact non-interpolated AP: precision evaluated at each tied-score block end.
 y=np.asarray(y,int);s=np.asarray(scores,float)
 if s.ndim==1:s=s[:,None]
 if y.sum()==0:raise ValueError('AP selection requires positives')
 order=np.argsort(-s,axis=0,kind='stable');ss=np.take_along_axis(s,order,axis=0)
 yy=y[order];tp=np.cumsum(yy,axis=0);rank=np.arange(1,len(y)+1)[:,None]
 ends=np.vstack([ss[:-1]!=ss[1:],np.ones((1,s.shape[1]),bool)])
 last=np.maximum.accumulate(np.where(ends,np.arange(len(y))[:,None],-1),axis=0)
 prev=np.vstack([np.full((1,s.shape[1]),-1),last[:-1]])
 prev_tp=np.take_along_axis(np.vstack([np.zeros((1,s.shape[1])),tp]),prev+1,axis=0)
 return (((tp-prev_tp)*(tp/rank)*ends).sum(0)/y.sum())

def select_weights(bundles,w):
 values=np.empty((len(bundles),len(w)))
 for i,(x,y) in enumerate(bundles):
  if len(np.unique(y))!=2:raise ValueError('No two classes in development budget')
  for start in range(0,len(w),128):values[i,start:start+128]=ap_columns(y,x@w[start:start+128].T)
 objective=values.mean(0)-.15*values.std(0,ddof=0);idx=int(np.argmax(objective))
 return idx,values,objective

def ewma(frame,scores,alpha):
 f=frame.reset_index(drop=True);out=np.empty(len(f));scores=np.asarray(scores)
 for _,part in f.groupby(['string_id','battery_id'],sort=False):
  ix=part.sort_values('event_index',kind='stable').index.to_numpy()
  if part.event_index.duplicated().any():raise ValueError('ambiguous event ordering')
  out[ix]=pd.Series(scores[ix]).ewm(alpha=alpha,adjust=False).mean()
 return out

def choose_alpha(bundles):
 for frame,score,mask in bundles:
  if len(np.unique(frame.label_anomaly.to_numpy(int)[mask]))<2:
   return 1.,'no_two_classes',pd.DataFrame([{'alpha':a,'eligible':False,'reason':'no_two_classes'} for a in ALPHAS])
 rows=[]
 for alpha in ALPHAS:
  aps=[];deltas=[];targets=[]
  for frame,score,mask in bundles:
   z=ewma(frame,score,alpha);y=frame.label_anomaly.to_numpy(int)
   if len(np.unique(y[mask]))<2:raise ValueError('No two classes for alpha')
   ap=average_precision_score(y[mask],z[mask]);aps.append(ap);deltas.append(ap-average_precision_score(y[mask],score[mask]))
   for fam in TARGETS:
    m=mask&frame.active_fault_type.isin(['normal',fam]).to_numpy()
    if len(np.unique(y[m]))==2:targets.append(average_precision_score(y[m],z[m]))
  rows.append(dict(alpha=alpha,global_ap=np.mean(aps),minimum_delta=min(deltas),target_ap=np.mean(targets) if targets else np.nan,target_cells=len(targets),eligible=min(deltas)>1e-14 and bool(targets)))
 table=pd.DataFrame(rows);ok=table[table.eligible]
 if ok.empty:return 1.,'no_eligible_improvement_or_target',table
 win=ok.sort_values(['target_ap','global_ap'],ascending=False,kind='stable').iloc[0]
 return float(win.alpha),'selected',table

def threshold(y,s):
 c=np.unique(np.quantile(s,np.linspace(.5,.995,250)));pr=s[:,None]>=c;tp=(pr*y[:,None]).sum(0);den=pr.sum(0)+y.sum()
 return float(c[np.argmax(np.divide(2*tp,den,out=np.zeros(len(c)),where=den>0))])

def composition(frame,m,kind,seed=2026092106,excluded=None):
 f=frame.copy();f['unit']=f.string_id.astype(str)+'/'+f.battery_id.astype(str)
 positive=f[f.label_anomaly>0].groupby('unit').active_fault_type
 if (positive.nunique()>1).any():raise ValueError('ambiguous multi-family trajectory')
 pos=positive.first()
 health=np.setdiff1d(f.unit.unique(),pos.index)
 if excluded is not None:
  # Drop entire units where this family occurs, not just positive event rows.
  removed=f.loc[f.active_fault_type.eq(excluded),'unit'].unique();chosen=np.setdiff1d(f.unit.unique(),removed)
  return f.unit.isin(chosen).to_numpy(),{'status':'ok' if len(removed) else 'not_estimable','excluded_units':list(removed)}
 if m>len(pos):return np.zeros(len(f),bool),{'status':'insufficient_trajectories'}
 rng=np.random.default_rng(seed);pool={fam:list(rng.permutation(pos[pos==fam].index)) for fam in sorted(pos.unique())};family_order=list(rng.permutation(sorted(pool)));sel=[]
 if kind=='natural':sel=list(rng.choice(pos.index,m,replace=False))
 else:
  if kind=='skewed':
   q=pool.get('high_resistance',[]);sel=q[:int(np.ceil(.75*m))];pool['high_resistance']=q[len(sel):]
  while len(sel)<m:
   for fam in family_order:
    if pool[fam] and len(sel)<m:sel.append(pool[fam].pop(0))
 attained=bool((pos.loc[sel]=='high_resistance').sum()>=np.ceil(.75*m)) if kind=='skewed' else True
 return f.unit.isin(list(health)+sel).to_numpy(),{'status':'ok' if attained else 'skew_target_unattainable','selected_fault_units':sel,'actual_families':pos.loc[sel].value_counts().to_dict(),'requested_m':m,'requested_high_resistance':int(np.ceil(.75*m)) if kind=='skewed' else None,'target_attained':attained}
