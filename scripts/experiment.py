"""Reproducible source-frozen, label-budgeted calibration experiment on official CSVs.
Run only after acquisition completes. No synthetic benchmark results are produced.
"""
import os
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ.setdefault(k,'4')
import argparse, datetime, hashlib, io, json, pathlib, platform, re, time, zipfile
import joblib, numpy as np, pandas as pd, scipy, sklearn
from scipy.optimize import minimize
from scipy.special import expit, logit
from scipy.stats import ks_2samp
from sklearn.cluster import MiniBatchKMeans
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score, brier_score_loss, log_loss, confusion_matrix, f1_score, matthews_corrcoef
from sklearn.preprocessing import RobustScaler

ROOT=pathlib.Path(__file__).resolve().parents[1]
def log(event,**kw):
 row={'time_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'event':event,**kw}
 with (ROOT/'logs/training.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(row,allow_nan=False,default=str)+'\n')
 print(json.dumps(row,default=str),flush=True)
def canonical(s):return re.sub('[^a-z0-9]','',s.lower())
def reservoir(stream,cap,seed):
 rng=np.random.default_rng(seed);sample=None;seen=0
 for chunk in pd.read_csv(stream,chunksize=100000,low_memory=False):
  chunk.columns=[canonical(str(c)) for c in chunk.columns]
  chunk['_key']=rng.random(len(chunk));seen+=len(chunk)
  sample=chunk if sample is None else pd.concat([sample,chunk],ignore_index=True)
  if len(sample)>cap:sample=sample.nsmallest(cap,'_key')
 return sample.drop(columns='_key'),seen
def read_data(cap):
 cache=ROOT/f'data/processed_cap{cap}.joblib'
 if cache.exists():return joblib.load(cache)
 domains={};inventory=[]
 for domain in ['CICIoT2023','CICIoMT2024']:
  frames=[];base=ROOT/'data/raw'/domain
  if domain=='CICIoT2023':
   zp=base/'CSV/CSV.zip'
   if not zp.exists() or not zipfile.is_zipfile(zp):raise FileNotFoundError('Complete official CICIoT2023 CSV.zip required; acquisition has not finished')
   z=zipfile.ZipFile(zp);entries=[n for n in z.namelist() if n.lower().endswith('.csv')];entries.sort()
   it=[(n,lambda n=n:z.open(n)) for n in entries]
  else:
   paths=sorted((base/'WiFI_and_MQTT/attacks/CSV').rglob('*.csv'))
   if not paths:raise FileNotFoundError('No official target CSVs')
   it=[(str(p.relative_to(base)).replace('\\','/'),lambda p=p:p.open('rb')) for p in paths]
  for name,opener in it:
   digest=int(hashlib.sha256((domain+name).encode()).hexdigest()[:8],16)
   with opener() as f:df,n=reservoir(f,cap,digest)
   family=pathlib.PurePosixPath(name).parent.name if domain=='CICIoT2023' else re.sub(r'_(?:train|test).*','',pathlib.PurePosixPath(name).name,flags=re.I)
   df['_family']=family;df['_capture']=name;df['_y']=0 if 'benign' in name.lower() else 1
   df['_partition']='test' if '/test/' in name else 'pool'
   frames.append(df);inventory.append({'domain':domain,'capture':name,'family':family,'rows_original':n,'rows_retained':len(df)})
   log('capture_read',**inventory[-1])
  domains[domain]=pd.concat(frames,ignore_index=True)
 common=sorted(set(domains['CICIoT2023'].columns)&set(domains['CICIoMT2024'].columns))
 forbidden=['label','attack','target','flowid','srcip','dstip','timestamp','sourceip','destinationip']
 features=[c for c in common if not c.startswith('_') and not any(x in c for x in forbidden)]
 for dom,df in domains.items():
  for c in features:df[c]=pd.to_numeric(df[c],errors='coerce')
  df[features]=df[features].replace([np.inf,-np.inf],np.nan)
  df['_hash']=pd.util.hash_pandas_object(df[features],index=False).astype(str)
  before=len(df);df=df.drop_duplicates('_hash').reset_index(drop=True);log('dedup_within_domain',domain=dom,before=before,after=len(df));domains[dom]=df
 pd.DataFrame(inventory).to_csv(ROOT/'results/data_inventory.csv',index=False)
 (ROOT/'results/common_features.json').write_text(json.dumps(features,indent=2))
 joblib.dump((domains,features),cache);return domains,features
def source_split(df):
 train=[];test=[]
 for fam,g in df.groupby('_family'):
  captures=sorted(g['_capture'].unique())
  # Whole capture files; singleton families occur in training only.
  held=set(captures[-max(1,len(captures)//4):]) if len(captures)>1 else set()
  train.append(g[~g['_capture'].isin(held)]);test.append(g[g['_capture'].isin(held)])
 return pd.concat(train),pd.concat(test)
def metrics(y,p):
 p=np.clip(p,1e-7,1-1e-7);pred=p>=.5;tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel();n=len(y)
 ece=0.
 for lo in np.linspace(0,1,11)[:-1]:
  mask=(p>=lo)&(p<lo+.1)
  if mask.any():ece+=mask.mean()*abs(y[mask].mean()-p[mask].mean())
 return dict(ap=float(average_precision_score(y,p)),auroc=float(roc_auc_score(y,p)),brier=float(brier_score_loss(y,p)),nll=float(log_loss(y,p,labels=[0,1])),ece=float(ece),fpr=float(fp/max(1,fp+tn)),recall=float(tp/max(1,tp+fn)),precision=float(tp/max(1,tp+fp)),macro_f1=float(f1_score(y,pred,average='macro',zero_division=0)),mcc=float(matthews_corrcoef(y,pred)),tn=int(tn),fp=int(fp),fn=int(fn),tp=int(tp),alarms_per_10000=float(10000*(fp+tp)/n),prevalence=float(np.mean(y)),n=n)
def allocate(counts,uncertainty,budget,coverage=True,use_uncertainty=True):
 active=counts>0;allocation=np.zeros(len(counts),dtype=int)
 if coverage:
  if budget<active.sum():raise ValueError('Budget smaller than stratum count')
  allocation[active]=1
 priorities=counts*(.05+uncertainty if use_uncertainty else np.ones(len(counts)))
 # Greedy proportional allocation with capacity limits; labels never used.
 while allocation.sum()<budget:
  score=priorities/(allocation+1);score[allocation>=counts]=-np.inf
  j=int(np.argmax(score))
  if not np.isfinite(score[j]):raise ValueError('Insufficient pool')
  allocation[j]+=1
 return allocation
def select(method,z,p,strata,budget,seed):
 rng=np.random.default_rng(seed);n=len(p);unc=-(p*np.log(np.clip(p,1e-8,1))+(1-p)*np.log(np.clip(1-p,1e-8,1)))
 if method=='random':return rng.choice(n,budget,replace=False),np.ones(budget),{}
 if method=='uncertainty':return np.argsort(-unc,kind='stable')[:budget],np.ones(budget),{}
 if method=='coreset':
  selected=[int(rng.integers(n))];dist=np.sum((z-z[selected[0]])**2,axis=1)
  for _ in range(1,budget):
   i=int(np.argmax(dist));selected.append(i);dist=np.minimum(dist,np.sum((z-z[i])**2,axis=1));dist[selected]=-np.inf
  return np.array(selected),np.ones(budget),{}
 h=np.unique(strata);counts=np.array([(strata==i).sum() for i in h]);uh=np.array([unc[strata==i].mean() for i in h]);alloc=allocate(counts,uh,budget,coverage=method!='no_coverage',use_uncertainty=method!='no_uncertainty')
 ix=[];w=[]
 for j,k in enumerate(h):
  if alloc[j]==0:continue
  ids=rng.choice(np.flatnonzero(strata==k),int(alloc[j]),replace=False);ix.extend(ids);w.extend([counts[j]/alloc[j]]*len(ids))
 w=np.array(w,dtype=float)
 if method=='no_weights':w[:]=1
 return np.array(ix),w/w.mean(),{'stratum_counts':counts.tolist(),'stratum_allocations':alloc.tolist(),'zero_inclusion_strata':int(np.sum(alloc==0))}
def calibrate(p,y,w):
 s=logit(np.clip(p,1e-6,1-1e-6));w=w/w.sum()
 def objective(v):
  z=v[0]*s+v[1];loss=np.sum(w*(np.logaddexp(0,z)-y*z))+.01*((v[0]-1)**2+v[1]**2)
  e=w*(expit(z)-y);grad=np.array([np.dot(e,s),e.sum()])+.02*np.array([v[0]-1,v[1]])
  return loss,grad
 history=[]
 fit=minimize(objective,[1.,0.],jac=True,bounds=[(1e-4,20),(-20,20)],method='L-BFGS-B',callback=lambda v:history.append(float(objective(v)[0])))
 return fit.x,{'success':bool(fit.success),'iterations':int(fit.nit),'loss':float(fit.fun),'loss_history':history,'message':str(fit.message)}
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--cap-per-capture',type=int,default=3000);ap.add_argument('--seeds',default='11,23,37,53,71');ap.add_argument('--models',default='lr,rf,hgb');ap.add_argument('--allow-incomplete',action='store_true');a=ap.parse_args()
 manifest=json.loads((ROOT/'data/official_manifest.json').read_text())
 missing=[r['relative_path'] for r in manifest if (r['relative_path'].lower().endswith('.csv') or r['relative_path']=='CSV/CSV.zip') and not (ROOT/'data/raw'/r['dataset']/r['relative_path']).exists()]
 if missing and not a.allow_incomplete:
  log('blocked',reason='incomplete_official_dataset',missing_files=len(missing));raise SystemExit('Acquisition incomplete: refusing to generate full-study results')
 log('run_start',status='partial' if missing else 'complete_csv_acquisition',python=platform.python_version(),sklearn=sklearn.__version__,numpy=np.__version__,pandas=pd.__version__,cap_per_capture=a.cap_per_capture,arguments=vars(a))
 domains,features=read_data(a.cap_per_capture);source=domains['CICIoT2023'];target=domains['CICIoMT2024'];src,st=source_split(source);pool=target[target['_partition']=='pool'].copy();test=target[target['_partition']=='test'].copy()
 # Cross-partition exact-feature duplicates are removed before fitting.
 st=st[~st['_hash'].isin(set(src['_hash']))];pool=pool[~pool['_hash'].isin(set(src['_hash']))];test=test[~test['_hash'].isin(set(src['_hash'])|set(pool['_hash']))]
 partitions={'source_train':src,'source_test':st,'target_pool':pool,'target_test':test}
 for name,df in partitions.items():
  if len(df)==0 or df['_y'].nunique()<2:raise RuntimeError('Empty or single-class partition: '+name)
  log('partition',name=name,n=len(df),benign=int((df['_y']==0).sum()),attack=int((df['_y']==1).sum()),captures=int(df['_capture'].nunique()))
 pd.concat([df[['_capture','_hash','_y']].assign(partition=k) for k,df in partitions.items()]).to_csv(ROOT/'results/split_manifest.csv',index=False)
 transform=lambda df:np.sign(df[features].to_numpy(dtype=float))*np.log1p(np.abs(df[features].to_numpy(dtype=float)))
 imputer=SimpleImputer(strategy='median',keep_empty_features=True);scaler=RobustScaler();xs=scaler.fit_transform(imputer.fit_transform(transform(src)));xp=scaler.transform(imputer.transform(transform(pool)));xt=scaler.transform(imputer.transform(transform(test)));xst=scaler.transform(imputer.transform(transform(st)))
 for name,x in [('source',xs),('pool',xp),('test',xt),('source_test',xst)]:np.clip(x,-100,100,out=x)
 drift=[{'feature':c,'ks_source_pool':float(ks_2samp(xs[:,j],xp[:,j]).statistic)} for j,c in enumerate(features)];pd.DataFrame(drift).to_csv(ROOT/'results/drift.csv',index=False)
 seeds=[int(s) for s in a.seeds.split(',')];results=[];pred_dir=ROOT/'results/predictions';pred_dir.mkdir(exist_ok=True)
 for seed in seeds:
  km=MiniBatchKMeans(n_clusters=8,random_state=seed,n_init=3,batch_size=2048);strata=km.fit_predict(xp)
  for model_name in a.models.split(','):
   model={'lr':lambda:LogisticRegression(max_iter=1000,C=1,random_state=seed),'rf':lambda:RandomForestClassifier(n_estimators=150,min_samples_leaf=5,n_jobs=4,random_state=seed),'hgb':lambda:HistGradientBoostingClassifier(max_iter=120,max_leaf_nodes=15,l2_regularization=1,random_state=seed)}[model_name]()
   t=time.monotonic();model.fit(xs,src['_y']);elapsed=time.monotonic()-t;ps=model.predict_proba(xst)[:,1];pp=model.predict_proba(xp)[:,1];pt=model.predict_proba(xt)[:,1];y=test['_y'].to_numpy();score=logit(np.clip(pt,1e-6,1-1e-6))
   log('source_fit',seed=seed,model=model_name,seconds=elapsed,n=len(src),iterations=getattr(model,'n_iter_',None));joblib.dump({'model':model,'imputer':imputer,'scaler':scaler,'features':features},ROOT/'results'/f'model_{model_name}_{seed}.joblib')
   for domain,yy,prob in [('source_test',st['_y'].to_numpy(),ps),('target_test',y,pt)]:
    row=dict(seed=seed,model=model_name,method='source_only',budget=0,evaluation=domain,**metrics(yy,prob));results.append(row);log('evaluation',**row)
   for budget in [25,50,100,250]:
    if len(pool)<budget:continue
    for method in ['random','uncertainty','coreset','no_coverage','no_uncertainty','no_weights','proposed']:
     ix,w,detail=select(method,xp,pp,strata,budget,seed);cal,info=calibrate(pp[ix],pool['_y'].to_numpy()[ix],w);prob=expit(cal[0]*score+cal[1]);row=dict(seed=seed,model=model_name,method=method,budget=budget,evaluation='target_test',a=float(cal[0]),b=float(cal[1]),effective_sample_size=float(w.sum()**2/np.sum(w*w)),**metrics(y,prob));results.append(row)
     log('calibration_fit',seed=seed,model=model_name,method=method,budget=budget,selected_hashes=pool['_hash'].iloc[ix].tolist(),selected_labels=pool['_y'].iloc[ix].tolist(),weights=w.tolist(),parameters=cal.tolist(),**detail,**info);log('evaluation',**row)
     pd.DataFrame({'capture':test['_capture'],'family':test['_family'],'y':y,'probability':prob,'feature_hash':test['_hash']}).to_csv(pred_dir/f'{model_name}_{seed}_{method}_{budget}.csv.gz',index=False)
   pd.DataFrame(results).to_csv(ROOT/'results/metrics.csv',index=False)
 agg=pd.DataFrame(results).groupby(['model','method','budget','evaluation']).agg({c:['mean','std'] for c in ['brier','nll','ece','fpr','recall','macro_f1','mcc','ap']});agg.to_csv(ROOT/'results/metrics_summary.csv');log('run_complete',rows=len(results),seeds=seeds,models=a.models)
if __name__=='__main__':main()
