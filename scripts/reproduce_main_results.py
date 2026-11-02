"""Reconstruct the two principal result tables from aggregate CSV evidence."""
import argparse,pathlib
import pandas as pd
R=pathlib.Path(__file__).resolve().parents[1]
a=argparse.ArgumentParser();a.add_argument('--root',default='evidence/results');args=a.parse_args();root=R/args.root
f=pd.read_csv(root/'full/metrics.csv');f=f[f.budget.eq(100)&f.evaluation.eq('target_test')&f.method.isin(['random','proposed'])]
t1=f.groupby(['model','method'])[['brier','fpr','recall','mcc']].mean()
assert f.groupby(['model','method']).size().eq(5).all()
print('Main manuscript Table 1: original 100-label evaluation');print(t1.round(5).to_string())
rows=[]
for name,file,domain in [('CIC','revision/common_metrics.csv','target_test'),('IoT-23 expanded','strengthening/external_metrics.csv','expanded_external')]:
 d=pd.read_csv(root/file);d=d[d.model.eq('rf')&d.budget.eq(250)&d['mode'].eq('target_validation_05')&d.domain.eq(domain)]
 for method,label in [('cal_random','Calibration'),('target_random','Target RF'),('mix_random','Mixed RF')]:
  q=d[d.method.eq(method)];assert len(q)==5,(name,method,len(q))
  row={'evaluation':name,'adaptation':label,**q[['fpr','recall','mcc']].mean().to_dict(),'joint_success':str(int(q.joint_goal_5pct.sum()))+'/5'};rows.append(row)
t2=pd.DataFrame(rows)
print('\nMain manuscript Table 2: common-feature and expanded external task');print(t2.round(4).to_string(index=False))
# Published table values, checked at displayed precision; no retuning occurs.
expected=[(.2591,.9978,.8063),(.2150,.9989,.8402),(.0208,.9981,.9816),(.8410,.8378,-.0196),(.1875,.9150,.7379),(.9374,.9989,.1787)]
for row,exp in zip(rows,expected):
 for key,v in zip(['fpr','recall','mcc'],exp):assert abs(row[key]-v)<.000051,(row,key,v)
print('\nTable 2 agrees with the latest manuscript at four decimal places.')
