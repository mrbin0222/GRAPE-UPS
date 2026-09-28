"""Observation perturbations used by the telemetry experiment."""
import numpy as np

def perturb(t,v,temp,s,affected,rng):
 t=t.copy();v=v.copy();temp=temp.copy();n=len(t);kind=s['kind'];a=s['value'];keep=np.ones(n,bool)
 if kind=='random_drop':keep=rng.random(n)>=a
 elif kind=='block_drop':
  length=max(1,int(round(n*a)));start=int(rng.integers(0,n-length+1));keep[start:start+length]=False
 elif affected:
  if kind=='voltage_offset':v+=a
  elif kind=='voltage_gain':v*=1+a
  elif kind=='temperature_offset':temp+=a
  elif kind=='temperature_gain':temp*=1+a
  elif kind=='voltage_quantization':v=np.round(v/a)*a
  elif kind=='voltage_drift':v+=np.linspace(0,a,n)
  elif kind=='clock_shift':t+=a
 return t[keep],v[keep],temp[keep],keep
