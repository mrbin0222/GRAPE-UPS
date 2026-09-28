import os,json,sys
from pathlib import Path
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:os.environ[k]='1'
os.environ['PYBAMM_DISABLE_TELEMETRY']='true'
ROOT=Path(__file__).resolve().parents[1]
def cfg():return json.loads((ROOT/'configs/simulation/physical_domains.json').read_text())
def save(p,obj):Path(p).write_text(json.dumps(obj,indent=2)+'\n')
