"""Duration-aware features and lossless ragged native-profile libraries."""
import json
from pathlib import Path
import numpy as np
FEATURES=['pre_voltage_v','voltage_sag_v','dynamic_resistance_mohm','discharge_slope_v_per_s','minimum_voltage_v','recovery_30s_v','recovery_ratio','temperature_rise_c']
CHANNELS=['time_s','voltage_v','temperature_k','current_a']
def extract(profile,meta):
    t=np.asarray(profile['time_s'],float)-float(meta['pre_event_rest_s']);D=float(meta['discharge_s'])
    if D<=0:raise ValueError('duration must be positive')
    v=np.asarray(profile['voltage_v'],float);T=np.asarray(profile['temperature_k'],float)-273.15;I=np.asarray(profile['current_a'],float)
    if not (len(t)==len(v)==len(T)==len(I)):raise ValueError('channel lengths differ')
    def window(x,a,b):return x[(t>=a)&(t<=b)&np.isfinite(t)&np.isfinite(x)]
    def mean(x,a,b):
        z=window(x,a,b);return float(z.mean()) if len(z) else np.nan
    pre=mean(v,-5,-1);sag=pre-mean(v,0,3);step=abs(mean(I,-5,-1)-mean(I,0,3))
    discharge=window(v,0,D);minimum=float(discharge.min()) if len(discharge) else np.nan
    recovery=mean(v,D+20,D+40);sm=(t>=10)&(t<=min(D-5,60))&np.isfinite(v)&np.isfinite(t)
    slope=float(np.polyfit(t[sm],v[sm],1)[0]) if sm.sum()>=3 else np.nan
    hot=window(T,0,min(float(np.max(t)),D+60));rise=float(hot.max()-mean(T,-5,-1)) if len(hot) else np.nan
    values=dict(zip(FEATURES,[pre,sag,1000*sag/max(step,1e-6),slope,minimum,recovery,(recovery-minimum)/max(sag,1e-6),rise]))
    support={'pre_points':len(window(v,-5,-1)),'early_points':len(window(v,0,3)),'slope_points':int(sm.sum()),'recovery_points':len(window(v,D+20,D+40)),'discharge_points':len(discharge),'record_end_event_time':float(t.max()),'recovery_end_covered':bool(t.max()>=D+40),'current_step_a':step,'finite_features':bool(np.isfinite(list(values.values())).all())}
    return values,support

def assemble(path,records):
    """records = [(profile_dict, per_profile_metadata), ...], native grids retained."""
    if not records:raise ValueError('empty profile library')
    offsets=[0];meta=[]
    for p,m in records:
        sizes={len(p[k]) for k in CHANNELS}
        if len(sizes)!=1:raise ValueError('unequal channels')
        if not np.all(np.diff(p['time_s'])>=0):raise ValueError('time order violated')
        if any(k not in m for k in ['pre_event_rest_s','discharge_s','profile_id']):raise ValueError('missing duration metadata')
        offsets.append(offsets[-1]+len(p['time_s']));meta.append(m)
    np.savez_compressed(path,**{k:np.concatenate([p[k] for p,m in records]) for k in CHANNELS},offsets=np.array(offsets,dtype=np.int64),metadata_json=np.array(json.dumps(meta,sort_keys=True)))

def read(path):
    with np.load(path,allow_pickle=False) as a:
        offsets=a['offsets'];meta=json.loads(str(a['metadata_json']))
        return [({k:a[k][int(offsets[i]):int(offsets[i+1])].copy() for k in CHANNELS},m) for i,m in enumerate(meta)]

def parameter_draw(seed,ranges):
    rng=np.random.default_rng(int(seed))
    return {key:float(rng.uniform(*limits)) for key,limits in ranges.items()}

def apply_background(parameters,draw,scaled_function):
    scalars={'acid_concentration':'Initial concentration in electrolyte [mol.m-3]','negative_thickness':'Negative electrode thickness [m]','positive_thickness':'Positive electrode thickness [m]','negative_porosity':'Maximum porosity of negative electrode','positive_porosity':'Maximum porosity of positive electrode'}
    applied={}
    for key,param in scalars.items():
        before=float(parameters[param]);after=before*draw[key]
        if 'porosity' in key and not 0<after<1:raise ValueError('unphysical porosity')
        if after<=0:raise ValueError('nonpositive physical parameter')
        parameters.update({param:after});applied[param]={'default':before,'background':after,'multiplier':draw[key]}
    for key,param in [('negative_kinetics','Negative electrode exchange-current density [A.m-2]'),('positive_kinetics','Positive electrode exchange-current density [A.m-2]')]:
        parameters.update({param:scaled_function(parameters[param],draw[key])});applied[param]={'type':'function_multiplier','multiplier':draw[key]}
    return applied
