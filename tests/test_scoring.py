import sys,unittest
from pathlib import Path
import numpy as np,pandas as pd
from sklearn.metrics import average_precision_score
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from ups_ai.models import _directed_group_score,GROUP_DIRECTIONAL_FEATURES
from ups_ai.common import robust_positive_z
from ups_ai.evaluation import repeated_event_ewma
from ups_ai.selection import ap_columns,candidates
class ScoringTests(unittest.TestCase):
 def test_adverse_direction_and_top_three(self):
  cols=list(GROUP_DIRECTIONAL_FEATURES);f=pd.DataFrame(np.zeros((2,8)),columns=cols)
  for k in cols:f.loc[0,k]=GROUP_DIRECTIONAL_FEATURES[k]*3;f.loc[1,k]=-GROUP_DIRECTIONAL_FEATURES[k]*3
  np.testing.assert_allclose(_directed_group_score(f),[3,0])
 def test_normalization_clips_below_median(self):
  np.testing.assert_array_equal(robust_positive_z(np.array([-2,-1,0]),np.array([-1,0,1])),[0,0,0])
 def test_accumulator_is_causal(self):
  f=pd.DataFrame({'string_id':['s']*3,'battery_id':['b']*3,'event_index':[1,2,3]})
  a=repeated_event_ewma(f,np.array([1.,2.,3.]),.8);b=repeated_event_ewma(f,np.array([1.,2.,100.]),.8)
  np.testing.assert_array_equal(a[:2],b[:2]);np.testing.assert_allclose(a,[1,1.8,2.76])
 def test_tied_average_precision(self):
  y=np.array([0,1,0,1]);s=np.array([[0,1],[0,1],[1,1],[2,0]],float)
  np.testing.assert_allclose(ap_columns(y,s),[average_precision_score(y,s[:,i]) for i in range(2)])
 def test_candidates(self):
  a=candidates();self.assertEqual(a.shape,(40008,5));np.testing.assert_allclose(a.sum(1),1)
if __name__=='__main__':unittest.main()
