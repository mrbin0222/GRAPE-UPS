from pathlib import Path
import pandas as pd
root=Path(__file__).resolve().parents[1]/'data/field'
f=pd.read_csv(root/'battery.csv')
dt=f.sort_values(['cell_id','sample_time_s']).groupby('cell_id').sample_time_s.diff()
print('battery rows:',len(f),'cells:',f.cell_id.nunique(),'median interval (s):',dt.median(),'max interval (s):',dt.max())
