from physical_common import *
import argparse,time,importlib.util
import numpy as np,pybamm
from duration_library import parameter_draw,apply_background,extract
p=argparse.ArgumentParser();p.add_argument('--seed',type=int,required=True);p.add_argument('--domain',required=True);p.add_argument('--variant',type=int,required=True);p.add_argument('--role',choices=['development','formal'],default='formal');p.add_argument('--output',type=Path,required=True);p.add_argument('--config',type=Path,default=ROOT/'configs/simulation/physical_domains.json');a=p.parse_args();c=json.loads(a.config.read_text());domain=next(x for x in c['domains'] if x['id']==a.domain);variant=c['variants'][a.variant]
out=a.output/f'seed{a.seed}'/a.domain/f'variant{a.variant:02d}';out.mkdir(parents=True,exist_ok=True)
if (out/'result.json').exists():raise RuntimeError('Attempt exists; no silent rerun')
sp=importlib.util.spec_from_file_location('original_generator',ROOT/'scripts/generate_independent_pybamm_suite.py');gen=importlib.util.module_from_spec(sp);sp.loader.exec_module(gen)
class ErrorCapture(pybamm.callbacks.Callback):
 def __init__(self):self.errors=[]
 def on_experiment_error(self,logs):self.errors.append(str(logs['error']))
capture=ErrorCapture()
start=time.time();result=dict(seed=a.seed,domain=a.domain,variant_index=a.variant,**variant,role=a.role)
try:
 pv=pybamm.ParameterValues(c['models']['parameter_set']);pv.update({'Ambient temperature [K]':domain['ambient_k'],'Initial temperature [K]':domain['ambient_k']});draw=parameter_draw(a.seed,c['background_multipliers']);apply_background(pv,draw,gen.scaled_function)
 for key in ['Negative electrode thickness [m]','Positive electrode thickness [m]']:pv.update({key:float(pv[key])*domain['design_thickness']})
 pv.update({'Initial concentration in electrolyte [mol.m-3]':float(pv['Initial concentration in electrolyte [mol.m-3]'])*domain['age_acid']})
 for key in ['Negative electrode exchange-current density [A.m-2]','Positive electrode exchange-current density [A.m-2]']:pv.update({key:gen.scaled_function(pv[key],domain['age_kinetics'])})
 gen.apply_fault(pv,variant['family'],variant['multiplier'])
 e=c['event'];pre_s=domain['removed_fraction']*e['nominal_capacity_ah']/e['conditioning_current_a']*3600;event_start=pre_s+e['rest_after_conditioning_s']+e['pre_event_rest_s']
 steps=([f"Discharge at {e['conditioning_current_a']} A for {pre_s} seconds"] if pre_s>0 else [])+[f"Rest for {e['rest_after_conditioning_s']} seconds",f"Rest for {e['pre_event_rest_s']} seconds",f"Discharge at {domain['current_a']} A for {domain['discharge_s']} seconds",f"Rest for {e['post_event_rest_s']} seconds"]
 meta=dict(profile_id=f'{a.seed}_{a.domain}_{a.variant}',pre_event_rest_s=event_start,**domain,**variant,physical_seed=a.seed,background_draw=draw,solver_configuration=c['models'],conditioning_duration_s=pre_s,removed_nominal_charge_ah=domain['removed_fraction']*17.,steps=steps,final_scalar_parameters={k:float(pv[k]) for k in ['Initial concentration in electrolyte [mol.m-3]','Negative electrode thickness [m]','Positive electrode thickness [m]','Maximum porosity of negative electrode','Maximum porosity of positive electrode']})
 save(out/'requested.json',meta)
 m=pybamm.lead_acid.Full(options={'thermal':'lumped'});pts=m.default_var_pts.copy();pts.update(c['models']['spatial_points']);solver=pybamm.IDAKLUSolver(rtol=c['models']['rtol'],atol=c['models']['atol'],options={'num_threads':1})
 sol=pybamm.Simulation(m,parameter_values=pv,experiment=pybamm.Experiment([pybamm.step.string(step,period=(2.0 if e["sample_period_s"]<2 and i<len(steps)-3 else e["sample_period_s"])) for i,step in enumerate(steps)]),var_pts=pts,solver=solver).solve(callbacks=capture)
 z=dict(time_s=np.asarray(sol.t),voltage_v=np.asarray(sol['Battery voltage [V]'].entries),temperature_k=np.asarray(sol['Volume-averaged cell temperature [K]'].entries),current_a=np.asarray(sol['Current [A]'].entries));np.savez_compressed(out/'history.npz',**z)
 result.update(end_s=float(sol.t[-1]),expected_end_s=event_start+domain['discharge_s']+e['post_event_rest_s'])
 if capture.errors:result.update(status='solver_failure',errors=capture.errors,partial_history=True)
 elif sol.t[-1]<result['expected_end_s']-1e-6:result.update(status='physical_early_termination',termination=str(sol.termination))
 else:
  mask=z['time_s']>=event_start-e['pre_event_rest_s'];event={k:v[mask] for k,v in z.items()};np.savez_compressed(out/'event.npz',**event);f,q=extract(event,meta)
  valid=all(np.isfinite(v).all() for v in event.values()) and q['finite_features'] and q['recovery_end_covered'] and q['slope_points']>=3
  save(out/'features.json',dict(values=f,support=q));result.update(status='success' if valid else 'invalid_features',points=len(event['time_s']))
except pybamm.SolverError as ex:result.update(status='solver_failure',error=repr(ex))
except Exception as ex:result.update(status='unexpected_profile_error',error=repr(ex))
result['elapsed_s']=time.time()-start;save(out/'result.json',result);print(json.dumps(result),flush=True)

if result['status']=='unexpected_profile_error':sys.exit(2)
