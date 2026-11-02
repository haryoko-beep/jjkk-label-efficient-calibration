"""Prespecified capture/cap sensitivity and paired full-factorial calibration."""
import pathlib,json,hashlib,time,itertools,datetime
import numpy as np,pandas as pd,joblib
from scipy.special import expit,logit
from sklearn.cluster import MiniBatchKMeans
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from experiment import allocate,calibrate
from tune_revision import prep,threshold,evaluate
R=pathlib.Path(__file__).resolve().parents[1];O=R/'results/revision';O.mkdir(exist_ok=True)
def log(event,**kw):
 d={'time_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'event':event,**kw}
 with (R/'logs/revision_factorial.jsonl').open('a') as f:f.write(json.dumps(d,default=str)+'\n')
 print(event,{k:v for k,v in kw.items() if k in ['cap','split','model','seed','budget','rows']},flush=True)
def choose(p,strata,B,seed,entropy,weight,coverage=True):
 h=np.unique(strata);counts=np.array([(strata==k).sum() for k in h]);u=-(p*np.log(np.clip(p,1e-8,1))+(1-p)*np.log(np.clip(1-p,1e-8,1)));uh=np.array([u[strata==k].mean() for k in h]);a=allocate(counts,uh,B,coverage,entropy);rng=np.random.default_rng(seed);ix=[];w=[]
 for j,k in enumerate(h):
  if not a[j]:continue
  ids=rng.choice(np.flatnonzero(strata==k),int(a[j]),replace=False);ix.extend(ids);w.extend([counts[j]/a[j]]*len(ids))
 w=np.array(w) if weight else np.ones(B);w=w/w.mean();return np.array(ix),w,{'allocation':a.tolist(),'cluster_sizes':counts.tolist(),'clusters_reached':int((a>0).sum()),'pool_mass_represented':float(counts[a>0].sum()/counts.sum()),'ess':float(w.sum()**2/(w*w).sum())}
def groupsplit(src,seed):
 rng=np.random.default_rng(seed);held=[]
 for fam,g in src.groupby('_family'):
  caps=sorted(g._capture.unique());held.extend(rng.choice(caps,max(1,len(caps)//4),replace=False).tolist() if len(caps)>1 else [])
 return src[~src._capture.isin(held)].copy(),src[src._capture.isin(held)].copy(),held
def families(y,p,t,test):
 vals=[]
 for fam,idx in test.groupby('_family').indices.items():
  if np.any(y[idx]==1):vals.append(float(np.mean(p[idx][y[idx]==1]>=t)))
 return float(np.mean(vals))
def one_source(src,features,kind,seed):
 benign=src[src._y.eq(0)];attack=src[src._y.eq(1)].sample(n=min(30000,int(src._y.sum())),random_state=20261101);s=pd.concat([benign,attack]).reset_index(drop=True);fi,vi=train_test_split(np.arange(len(s)),test_size=.2,stratify=s._y,random_state=20261101);fit=s.iloc[fi];v=s.iloc[vi];imp=SimpleImputer(strategy='median',keep_empty_features=True);sc=RobustScaler();x=prep(fit,features,imp,sc,True);xv=prep(v,features,imp,sc);m=LogisticRegression(C=1,max_iter=1000,random_state=seed) if kind=='lr' else RandomForestClassifier(n_estimators=120,min_samples_leaf=20,class_weight='balanced',n_jobs=4,random_state=seed);m.fit(x,fit._y);cut,_=threshold(v._y.to_numpy(),m.predict_proba(xv)[:,1],.05);return {'model':m,'imputer':imp,'scaler':sc,'features':features,'source_threshold':cut,'fit_n':len(fit),'fit_benign':int(fit._y.eq(0).sum())}
def main():
 if (O/'factorial_complete.json').exists():print('complete; refusing duplicate run');return
 if (R/'logs/revision_factorial.jsonl').exists():raise RuntimeError('Partial run exists; inspect before explicit resume')
 log('run_start',protocol_sha256=hashlib.sha256((R/'REVIEW_REVISION_PROTOCOL.md').read_bytes()).hexdigest())
 dom,features=joblib.load(R/'data/processed_cap1000.joblib');pool=dom['CICIoMT2024'];pool=pool[pool._partition.eq('pool')].copy();pool[features]=pool[features].astype(float);pool['_hash']=pd.util.hash_pandas_object(pool[features],index=False).astype(str);ids=pd.read_csv(R/'results/full/target_pool_manifest.csv.gz',dtype={'_hash':str});pool=pool[pool._hash.isin(set(ids._hash))].reset_index(drop=True);test=joblib.load(R/'results/full/complete_target_test.joblib');y=test._y.to_numpy(np.int8);rows=[];queries=[];diagnostics=[];partitions=[]
 for cap in [1000,5000]:
  src=dom['CICIoT2023'].copy() if cap==1000 else joblib.load(R/'data/source_cap5000.joblib')[0];src[features]=src[features].astype(float)
  for split in [101,202,303]:
   train,hold,held=groupsplit(src,split);partitions.append({'cap':cap,'split':split,'held_captures':held,'train_rows':len(train),'holdout_rows':len(hold),'train_benign':int(train._y.eq(0).sum()),'holdout_benign':int(hold._y.eq(0).sum()),'overlapping_captures':len(set(train._capture)&set(hold._capture))})
   for kind in ['lr','rf']:
    obj=one_source(train,features,kind,split);key=f'{cap}_{split}_{kind}';joblib.dump(obj,O/f'source_{key}.joblib');xp=prep(pool,features,obj['imputer'],obj['scaler']);xt=prep(test,features,obj['imputer'],obj['scaler']);xh=prep(hold,features,obj['imputer'],obj['scaler']);pp=obj['model'].predict_proba(xp)[:,1];pt=obj['model'].predict_proba(xt)[:,1];ph=obj['model'].predict_proba(xh)[:,1];np.savez_compressed(O/f'scores_{key}.npz',test_probability=pt,pool_probability=pp);log('source_fit',cap=cap,split=split,model=kind,fit_n=obj['fit_n'],fit_benign=obj['fit_benign'])
    for domain,yy,p in [('source_holdout',hold._y.to_numpy(),ph),('target_test',y,pt)]:
     rows.append({'cap':cap,'split':split,'model':kind,'seed':0,'budget':0,'method':'source_only','domain':domain,'mode':'source_validation_05','threshold_hex':float(obj['source_threshold']).hex(),**evaluate(yy,p,obj['source_threshold'])})
    for j,f in enumerate(features):diagnostics.append({'cap':cap,'split':split,'model':kind,'domain':'target_test','feature':f,'clipped_fraction':float((np.abs(xt[:,j])>=100).mean()),'missing_fraction':float(test[f].isna().mean())})
    for seed in [11,23,37]:
     strata=MiniBatchKMeans(n_clusters=8,n_init=3,batch_size=2048,random_state=seed).fit_predict(xp);vi=np.random.default_rng(seed).choice(len(pool),100,replace=False);remain=np.setdiff1d(np.arange(len(pool)),vi);vl=pool._y.to_numpy()[vi]
     for entropy,weight in itertools.product([False,True],repeat=2):
      ix,w,detail=choose(pp[remain],strata[remain],150,seed,entropy,weight);ix=remain[ix];lab=pool._y.to_numpy()[ix];coef,info=calibrate(pp[ix],lab,w);p=expit(coef[0]*logit(np.clip(pt,1e-6,1-1e-6))+coef[1]);vp=expit(coef[0]*logit(np.clip(pp[vi],1e-6,1-1e-6))+coef[1]);method=f'E{int(entropy)}W{int(weight)}';query={'cap':cap,'split':split,'model':kind,'seed':seed,'budget':250,'method':method,'entropy':entropy,'weighting':weight,'benign_adaptation':int((lab==0).sum()),'benign_validation':int((vl==0).sum()),'unique_captures':int(pool.iloc[ix]._capture.nunique()),'overlap':len(set(ix)&set(vi)),**detail};queries.append(query);log('calibration_fit',**query,hashes=pool._hash.iloc[ix].tolist(),validation_hashes=pool._hash.iloc[vi].tolist(),labels=lab.tolist(),weights=w.tolist(),coef=coef.tolist(),**info)
      for mode,cut in [('native_05',.5),('target_validation_05',threshold(vl,vp,.05)[0])]:rows.append({'cap':cap,'split':split,'model':kind,'seed':seed,'budget':250,'method':method,'entropy':entropy,'weighting':weight,'domain':'target_test','mode':mode,'threshold_hex':float(cut).hex(),'macro_family_recall':families(y,p,cut,test),**evaluate(y,p,cut)})
    pd.DataFrame(rows).to_csv(O/'factorial_metrics.csv',index=False);pd.DataFrame(queries).to_csv(O/'factorial_queries.csv',index=False);pd.DataFrame(diagnostics).to_csv(O/'preprocessing_diagnostics.csv',index=False);(O/'group_partitions.json').write_text(json.dumps(partitions,indent=2));log('condition_complete',cap=cap,split=split,model=kind,rows=len(rows))
 # Separate coverage intervention, preserving original representation/model.
 obj=joblib.load(R/'results/tuning/selected_rf.joblib');xp=prep(pool,features,obj['imputer'],obj['scaler']);pp=obj['model'].predict_proba(xp)[:,1];pt=np.load(R/'results/tuning/source_scores_rf.npz')['probability'];stress=[]
 for seed in [11,23,37,53,71]:
  strata=MiniBatchKMeans(n_clusters=32,n_init=3,batch_size=2048,random_state=seed).fit_predict(xp)
  for B,coverage,entropy,weight in itertools.product([32,40,64],[False,True],[False,True],[False,True]):
   ix,w,detail=choose(pp,strata,B,seed,entropy,weight,coverage);lab=pool._y.to_numpy()[ix];coef,info=calibrate(pp[ix],lab,w);p=expit(coef[0]*logit(np.clip(pt,1e-6,1-1e-6))+coef[1]);d={'seed':seed,'budget':B,'coverage':coverage,'entropy':entropy,'weighting':weight,'benign':int((lab==0).sum()),'unique_captures':int(pool.iloc[ix]._capture.nunique()),**detail,**evaluate(y,p,.5)};stress.append(d);log('coverage_fit',**{k:v for k,v in d.items() if k not in ['allocation','cluster_sizes']},hashes=pool._hash.iloc[ix].tolist(),labels=lab.tolist(),weights=w.tolist(),coef=coef.tolist(),**info)
  pd.DataFrame(stress).to_csv(O/'coverage_stress.csv',index=False);log('coverage_seed_complete',seed=seed,rows=len(stress))
 # Six fixed feature controls share the source cohort, not source-selected target winners.
 src,_=groupsplit(dom['CICIoT2023'],101)[:2];controls=[]
 sets={'full':features,'protocol_flags':[c for c in features if c in ['protocoltype','http','https','dns','telnet','smtp','ssh','irc','tcp','udp','dhcp','arp','icmp','igmp','ipv','llc'] or 'flag' in c or c in ['ackcount','syncount','fincount','rstcount']],'packet_lengths':[c for c in features if c in ['min','max','avg','std','totsize','totsum','variance','headerlength']]}
 for fs,ff in sets.items():
  for kind in ['lr','rf']:
   obj=one_source(src,ff,kind,101);p=obj['model'].predict_proba(prep(test,ff,obj['imputer'],obj['scaler']))[:,1];controls.append({'feature_set':fs,'model':kind,'features':len(ff),'auroc':float(roc_auc_score(y,p)),**evaluate(y,p,obj['source_threshold'])})
 pd.DataFrame(controls).to_csv(O/'feature_controls.csv',index=False);(O/'factorial_complete.json').write_text(json.dumps({'source_models':12,'factorial_fits':len(queries),'evaluation_rows':len(rows),'coverage_fits':len(stress),'feature_controls':len(controls),'test_status':'reused_exploratory'},indent=2));log('run_complete',rows=len(rows))
if __name__=='__main__':main()
