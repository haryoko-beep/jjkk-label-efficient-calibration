import pathlib,json,hashlib,datetime
import pandas as pd,numpy as np,joblib
from scipy.special import expit,logit
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import roc_auc_score
from revision_factorial import one_source,choose
from tune_revision import prep,threshold,evaluate
from experiment import calibrate
from sklearn.ensemble import RandomForestClassifier
R=pathlib.Path(__file__).resolve().parents[1];O=R/'results/revision'
def main():
 df=pd.read_csv(O/'common_pcap_windows.csv.gz');features=json.loads((O/'common_extractor.json').read_text())['features'];src=df[df._domain.eq('CICIoT2023')&df._partition.eq('train')].copy();pool=df[df._domain.eq('CICIoMT2024')&df._partition.eq('train')].copy();test=df[df._domain.eq('CICIoMT2024')&df._partition.eq('test')].copy();external=df[df._domain.eq('IoT23')].copy();hold=df[df._partition.eq('source_holdout')].copy();retention=[]
 # Training wins ownership of exact feature duplicates. Do not manufacture new
 # independent observations by retaining copies in evaluation partitions.
 seen=set()
 for name,d in [('source',src),('source_holdout',hold),('target_pool',pool),('target_test',test),('external',external)]:
  before=len(d);d.drop_duplicates('_hash',inplace=True);d.drop(d[d._hash.isin(seen)].index,inplace=True);seen.update(d._hash);d.reset_index(drop=True,inplace=True);retention.append({'partition':name,'before':before,'retained':len(d),'benign':int(d._y.eq(0).sum()),'attack':int(d._y.eq(1).sum())})
 pd.DataFrame(retention).to_csv(O/'common_partition_retention.csv',index=False)
 for name,d in [('source',src),('pool',pool),('test',test),('external',external)]:
  if d._y.nunique()<2:raise RuntimeError('Insufficient classes after label mapping in '+name)
 rows=[];bycapture=[];logs=[];preds=[];queries=[]
 for kind in ['lr','rf']:
  obj=one_source(src,features,kind,101);joblib.dump(obj,O/f'common_source_{kind}.joblib');xs=prep(src,features,obj['imputer'],obj['scaler']);xp=prep(pool,features,obj['imputer'],obj['scaler']);pp=obj['model'].predict_proba(xp)[:,1];domains={'target_test':test,'external_iot23':external,'source_holdout':hold};xx={n:prep(d,features,obj['imputer'],obj['scaler']) for n,d in domains.items()};ps={n:obj['model'].predict_proba(x)[:,1] for n,x in xx.items()}
  def record(method,seed,probabilities,cut,mode):
   for domain,d in domains.items():
    p=probabilities[domain];y=d._y.to_numpy(np.int8);row={'model':kind,'method':method,'seed':seed,'domain':domain,'mode':mode,'threshold_hex':float(cut).hex(),'budget':0 if method=='source_only' else 250,'auroc':float(roc_auc_score(y,p)) if len(set(y))==2 else np.nan,**evaluate(y,p,cut)};rows.append(row)
    for capture,idx in d.groupby('_capture').indices.items():
     yy=y[idx];pred=p[idx]>=cut;bycapture.append({'model':kind,'method':method,'seed':seed,'domain':domain,'mode':mode,'capture':capture,'n':len(idx),'benign':int((yy==0).sum()),'attack':int((yy==1).sum()),'fp':int(np.sum(pred&(yy==0))),'tp':int(np.sum(pred&(yy==1)))})
    if mode=='target_validation_05':preds.append(pd.DataFrame({'model':kind,'method':method,'seed':seed,'domain':domain,'hash':d._hash,'y':y,'probability':p,'threshold_hex':float(cut).hex()}))
  record('source_only',0,ps,obj['source_threshold'],'source_validation_05')
  for seed in [11,23,37,53,71]:
   vi=np.random.default_rng(seed).choice(len(pool),100,replace=False);remain=np.setdiff1d(np.arange(len(pool)),vi);vl=pool._y.to_numpy()[vi];strata=MiniBatchKMeans(n_clusters=8,n_init=3,batch_size=2048,random_state=seed).fit_predict(xp)
   for method in ['random','diversity']:
    if method=='random':ix=np.random.default_rng(seed).choice(remain,150,replace=False);w=np.ones(150);detail={}
    else:ii,w,detail=choose(pp[remain],strata[remain],150,seed,True,True);ix=remain[ii]
    lab=pool._y.to_numpy()[ix];coef,info=calibrate(pp[ix],lab,w);queries.append({'model':kind,'seed':seed,'method':method,'adaptation':150,'validation':100,'benign_adaptation':int((lab==0).sum()),'benign_validation':int((vl==0).sum()),'overlap':len(set(ix)&set(vi))});logs.append({'event':'calibration','model':kind,'seed':seed,'method':method,'hashes':pool._hash.iloc[ix].tolist(),'validation_hashes':pool._hash.iloc[vi].tolist(),'labels':lab.tolist(),'weights':w.tolist(),'coef':coef.tolist(),**detail,**info})
    probs={n:expit(coef[0]*logit(np.clip(p,1e-6,1-1e-6))+coef[1]) for n,p in ps.items()};vp=expit(coef[0]*logit(np.clip(pp[vi],1e-6,1-1e-6))+coef[1]);configs=[('cal_'+method,probs,vp)]
    for mix in [False,True]:
     m=RandomForestClassifier(n_estimators=120,min_samples_leaf=2,class_weight='balanced',random_state=seed,n_jobs=4);x=xp[ix];yy=lab;weights=np.ones(len(ix))
     if mix:x=np.concatenate([xs,x]);yy=np.concatenate([src._y.to_numpy(),lab]);weights=np.concatenate([np.full(len(src),.5/len(src)),np.full(len(ix),.5/len(ix))])
     m.fit(x,yy,sample_weight=weights);pr=lambda x: m.predict_proba(x)[:,1] if len(m.classes_)==2 else np.full(len(x),float(m.classes_[0]));configs.append((('mix_' if mix else 'target_')+method,{n:pr(x) for n,x in xx.items()},pr(xp[vi])));joblib.dump(m,O/f'common_{kind}_{seed}_{"mix" if mix else "target"}_{method}.joblib',compress=3)
    for name,probs,vp in configs:
     for mode,cut in [('native_05',.5),('target_validation_05',threshold(vl,vp,.05)[0])]:record(name,seed,probs,cut,mode)
   print('common condition',kind,seed,flush=True)
 pd.DataFrame(rows).to_csv(O/'common_metrics.csv',index=False);pd.DataFrame(bycapture).to_csv(O/'common_capture_metrics.csv',index=False);pd.DataFrame(queries).to_csv(O/'common_queries.csv',index=False);pd.concat(preds).to_csv(O/'common_predictions.csv.gz',index=False);(R/'logs/common_training.jsonl').write_text('\n'.join(json.dumps(x) for x in logs)+'\n');(O/'common_complete.json').write_text(json.dumps({'evaluations':len(rows),'source_models':2,'calibration_fits':20,'retraining_fits':40,'extractor_sha256':hashlib.sha256((R/'scripts/common_pcap_features.py').read_bytes()).hexdigest(),'external_status':'new external subset; fixed pipeline; capture-dependent; no deployment guarantee'},indent=2));print('common complete',len(rows),flush=True)
if __name__=='__main__':main()
