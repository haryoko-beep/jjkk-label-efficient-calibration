"""Predict with frozen models/cutoffs only; no fit or selection on added captures."""
import pathlib,json,hashlib,warnings,datetime
import numpy as np,pandas as pd,joblib
from scipy.special import expit,logit
from sklearn.metrics import roc_auc_score
from tune_revision import prep,evaluate
R=pathlib.Path(__file__).resolve().parents[1];O=R/'results/strengthening'
def metric(d,p,t):
 y=d._y.to_numpy(np.int8)
 with warnings.catch_warnings():warnings.simplefilter('ignore',RuntimeWarning);m=evaluate(y,p,t)
 if not np.any(y==0):m['fpr']=np.nan
 if not np.any(y==1):m['recall']=np.nan
 m['joint_goal_5pct']=bool(np.isfinite(m['fpr']) and np.isfinite(m['recall']) and m['fpr']<=.05 and m['recall']>=.8)
 m['auroc']=float(roc_auc_score(y,p)) if len(set(y))==2 else np.nan
 attack_recalls=[];benign_fprs=[]
 for _,idx in d.groupby('_capture').indices.items():
  a=y[idx]==1;b=y[idx]==0;pr=p[idx]>=t
  if a.any():attack_recalls.append(float(pr[a].mean()))
  if b.any():benign_fprs.append(float(pr[b].mean()))
 m.update(macro_capture_recall=float(np.mean(attack_recalls)) if attack_recalls else np.nan,worst_capture_recall=float(np.min(attack_recalls)) if attack_recalls else np.nan,macro_capture_fpr=float(np.mean(benign_fprs)) if benign_fprs else np.nan,attack_bearing_captures=len(attack_recalls),benign_bearing_captures=len(benign_fprs));return m
def main():
 features=json.loads((R/'results/revision/common_extractor.json').read_text())['features'];old=pd.read_csv(R/'results/revision/common_pcap_windows.csv.gz',dtype={'_hash':str});seen=set();olds={};ret=[]
 for name,mask in [('source',old._domain.eq('CICIoT2023')&old._partition.eq('train')),('source_holdout',old._partition.eq('source_holdout')),('target_pool',old._domain.eq('CICIoMT2024')&old._partition.eq('train')),('target_test',old._domain.eq('CICIoMT2024')&old._partition.eq('test')),('old_external',old._domain.eq('IoT23'))]:
  d=old[mask].copy();d=d.drop_duplicates('_hash');d=d[~d._hash.isin(seen)].reset_index(drop=True);seen.update(d._hash);olds[name]=d
 fresh=pd.read_csv(O/'new_external_windows.csv.gz',dtype={'_hash':str});parts=[]
 for name,g in fresh.groupby('_capture',sort=True):
  before=len(g);g=g.drop_duplicates('_hash');g=g[~g._hash.isin(seen)].reset_index(drop=True);seen.update(g._hash);parts.append(g);ret.append({'capture':name,'before':before,'retained':len(g),'benign':int(g._y.eq(0).sum()),'attack':int(g._y.eq(1).sum()),'excluded_duplicate_vectors':before-len(g)})
 fresh=pd.concat(parts,ignore_index=True);expanded=pd.concat([olds['old_external'],fresh],ignore_index=True);domains={'old_external':olds['old_external'],'new_external':fresh,'expanded_external':expanded};pd.DataFrame(ret).to_csv(O/'new_external_retention.csv',index=False);expanded.to_csv(O/'expanded_external_windows.csv.gz',index=False);base=pd.read_csv(R/'results/revision/common_metrics.csv');rules=base[base.domain.eq('external_iot23')].copy();cal=[json.loads(x) for x in (R/'logs/common_training.jsonl').read_text().splitlines()];rows=[];caprows=[];preds=[]
 for kind in ['lr','rf']:
  obj=joblib.load(R/f'results/revision/common_source_{kind}.joblib');xx={n:prep(d,features,obj['imputer'],obj['scaler']) for n,d in domains.items()};source={n:obj['model'].predict_proba(x)[:,1] for n,x in xx.items()}
  for rule in rules[rules.model.eq(kind)].itertuples():
   cut=float.fromhex(rule.threshold_hex)
   if rule.method=='source_only':probs=source
   elif rule.method.startswith('cal_'):
    method=rule.method.split('_',1)[1];coef=next(x['coef'] for x in cal if x['model']==kind and x['seed']==rule.seed and x['method']==method);probs={n:expit(coef[0]*logit(np.clip(p,1e-6,1-1e-6))+coef[1]) for n,p in source.items()}
   else:
    mix,method=rule.method.split('_',1);m=joblib.load(R/f'results/revision/common_{kind}_{rule.seed}_{mix}_{method}.joblib');probs={n:m.predict_proba(x)[:,1] if len(m.classes_)==2 else np.full(len(x),float(m.classes_[0])) for n,x in xx.items()}
   for domain,d in domains.items():
    p=probs[domain];meta={'model':kind,'method':rule.method,'seed':rule.seed,'mode':rule.mode,'domain':domain,'threshold_hex':rule.threshold_hex,'budget':rule.budget};rows.append({**meta,**metric(d,p,cut)})
    for capture,idx in d.groupby('_capture').indices.items():caprows.append({**meta,'capture':capture,**metric(d.iloc[idx].reset_index(drop=True),p[idx],cut)})
    if rule.mode=='target_validation_05':preds.append(pd.DataFrame({**meta,'capture':d._capture,'hash':d._hash,'y':d._y,'probability':p}))
  print('Frozen external evaluation',kind,flush=True)
 pd.DataFrame(rows).to_csv(O/'external_metrics.csv',index=False);pd.DataFrame(caprows).to_csv(O/'external_capture_metrics.csv',index=False);pd.concat(preds).to_csv(O/'external_predictions.csv.gz',index=False);(O/'external_complete.json').write_text(json.dumps({'evaluations':len(rows),'frozen_rules':len(rules),'new_training_fits':0,'new_captures_acquired':len(json.loads((O/'new_external_manifest.json').read_text())),'new_eligible_captures':int(fresh._capture.nunique()),'expanded_attack_bearing_captures':int(expanded[expanded._y.eq(1)]._capture.nunique()),'old_windows':len(olds['old_external']),'new_windows':len(fresh),'expanded_windows':len(expanded),'protocol_sha256':hashlib.sha256((R/'STRENGTHENING_PROTOCOL.md').read_bytes()).hexdigest()},indent=2));print('Frozen external complete',len(rows),flush=True)
if __name__=='__main__':main()
