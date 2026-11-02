"""All-capture source study and complete official target-test evaluation.
Training uses a deterministic capped reservoir, not all source rows.
Persist raw source scores plus coefficients to reconstruct every prediction.
"""
import os
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='4'
import datetime,hashlib,json,pathlib,time,re,platform
import numpy as np,pandas as pd,joblib,sklearn
from scipy.special import expit,logit
from sklearn.cluster import MiniBatchKMeans
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier,HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score,roc_auc_score
from experiment import read_data,source_split,canonical,select,calibrate
R=pathlib.Path(__file__).resolve().parents[1];O=R/'results/full';O.mkdir(exist_ok=True)
SEEDS=[11,23,37,53,71];BUDGETS=[25,50,100,250];METHODS=['random','uncertainty','coreset','no_coverage','no_uncertainty','no_weights','proposed']
def log(event,**kw):
 row={'time_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'event':event,**kw}
 with (R/'logs/full_training.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(row,default=str)+'\n')
 print(json.dumps({k:v for k,v in row.items() if k not in ['selected_hashes','selected_labels','weights','loss_history']},default=str),flush=True)
def normalize(df,features):
 df=df.copy();df.columns=[canonical(str(c)) for c in df.columns];df=df.reindex(columns=features);return df.apply(pd.to_numeric,errors='coerce').replace([np.inf,-np.inf],np.nan).astype(np.float64)
def transform(df,features,imp,sc,fit=False):
 a=df[features].to_numpy(float);a=np.sign(a)*np.log1p(np.abs(a));a=sc.fit_transform(imp.fit_transform(a)) if fit else sc.transform(imp.transform(a));return np.clip(a,-100,100)
def fastmetrics(y,p,ranks=None):
 p=np.clip(p,1e-7,1-1e-7);pred=p>=.5;yy=y.astype(bool);tp=int(np.sum(pred&yy));fp=int(np.sum(pred&~yy));tn=int(np.sum(~pred&~yy));fn=int(np.sum(~pred&yy));n=len(y)
 f1a=2*tp/max(1,2*tp+fp+fn);f1b=2*tn/max(1,2*tn+fp+fn);den=np.sqrt(float((tp+fp)*(tp+fn))*float((tn+fp)*(tn+fn)))
 bins=np.minimum((p*10).astype(int),9);cnt=np.bincount(bins,minlength=10);sumy=np.bincount(bins,weights=y,minlength=10);sump=np.bincount(bins,weights=p,minlength=10)
 return {'n':n,'prevalence':float(y.mean()),'tp':tp,'fp':fp,'tn':tn,'fn':fn,'fpr':fp/max(1,fp+tn),'recall':tp/max(1,tp+fn),'precision':tp/max(1,tp+fp),'macro_f1':(f1a+f1b)/2,'mcc':(tp*tn-fp*fn)/den if den else 0.,'brier':float(np.mean((p-y)**2)),'nll':float(np.mean(-y*np.log(p)-(1-y)*np.log1p(-p))),'ece':float(np.sum(np.abs(sumy-sump))/n),'alarms_per_10000':10000*(tp+fp)/n,**(ranks or {})}
def family(name):
 s=re.sub(r'_(train|test).*','',name,flags=re.I);return re.sub(r'\d+$','',s)
def alltest(features,exclude):
 cache=O/'complete_target_test.joblib'
 if cache.exists():return joblib.load(cache)
 chunks=[];audit=[];seen=set(exclude)
 for path in sorted((R/'data/raw/CICIoMT2024/WiFI_and_MQTT/attacks/CSV/test').glob('*.csv')):
  raw_n=0;kept=0;within=0;cross=0
  for chunk in pd.read_csv(path,chunksize=100000,low_memory=False):
   df=normalize(chunk,features);raw_n+=len(df);hashes=pd.util.hash_pandas_object(df,index=False).to_numpy(dtype=np.uint64);mask=np.ones(len(df),bool)
   for j,h in enumerate(hashes):
    if int(h) in seen:mask[j]=False
    else:seen.add(int(h))
   cross+=int(np.sum(~mask));df=df.loc[mask].copy();df['_hash']=hashes[mask];df['_capture']=path.name;df['_family']=family(path.name);df['_y']=int('benign' not in path.name.lower());chunks.append(df);kept+=len(df)
  row={'capture':path.name,'family':family(path.name),'raw_rows':raw_n,'kept_unique_rows':kept,'excluded_overlap_or_duplicate':cross};audit.append(row);log('complete_target_read',**row)
 df=pd.concat(chunks,ignore_index=True);pd.DataFrame(audit).to_csv(O/'target_test_inventory.csv',index=False);joblib.dump(df,cache,compress=0);return df
def main():
 manifest=json.loads((R/'data/official_manifest.json').read_text());required=[x for x in manifest if x['relative_path'].endswith('.csv') or x['relative_path']=='CSV/CSV.zip'];missing=[x['relative_path'] for x in required if not (R/'data/raw'/x['dataset']/x['relative_path']).exists()]
 if missing:raise RuntimeError('Required CSV acquisition incomplete: '+str(missing))
 log('run_start',cap_per_file=1000,seeds=SEEDS,budgets=BUDGETS,models=['lr','rf','hgb'],sklearn=sklearn.__version__,python=platform.python_version(),test_policy='all_official_test_rows_then_exact_feature_dedup',training_policy='reservoir_cap_1000_per_capture',ece_policy='ten_disjoint_equal_width_bins')
 domains,features=read_data(1000);src,st=source_split(domains['CICIoT2023']);pool=domains['CICIoMT2024'];pool=pool[pool['_partition'].eq('pool')].copy()
 # Recompute hashes identically as uint64 for the complete test overlap audit.
 for df in [src,st,pool]:
  df[features]=df[features].astype(np.float64)
  df['_hash']=pd.util.hash_pandas_object(df[features],index=False).to_numpy(dtype=np.uint64)
 st=st[~st['_hash'].isin(set(src['_hash']))];pool=pool[~pool['_hash'].isin(set(src['_hash']))];test=alltest(features,set(src['_hash'])|set(pool['_hash']))
 parts={'source_train':src,'source_holdout':st,'target_pool':pool,'complete_target_test':test};stats=[]
 for name,df in parts.items():
  row={'partition':name,'n':len(df),'benign':int((df['_y']==0).sum()),'attack':int((df['_y']==1).sum()),'files':int(df['_capture'].nunique()),'families':int(df['_family'].nunique())};stats.append(row);log('partition',**row)
  df[['_hash','_capture','_family','_y']].to_csv(O/f'{name}_manifest.csv.gz',index=False)
 pd.DataFrame(stats).to_csv(O/'partitions.csv',index=False);(O/'features.json').write_text(json.dumps(features,indent=2))
 imp=SimpleImputer(strategy='median',keep_empty_features=True);sc=RobustScaler();xs=transform(src,features,imp,sc,True);xh=transform(st,features,imp,sc);xp=transform(pool,features,imp,sc);xt=transform(test,features,imp,sc);y=test['_y'].to_numpy(np.int8);yp=pool['_y'].to_numpy(np.int8);ys=src['_y'].to_numpy(np.int8)
 joblib.dump({'imputer':imp,'scaler':sc,'features':features},O/'preprocessing.joblib');caps=sorted(test['_capture'].unique());capids=pd.Categorical(test['_capture'],categories=caps).codes;np.savez_compressed(O/'test_labels_and_ids.npz',y=y,capture_id=capids,feature_hash=test['_hash'].to_numpy());(O/'capture_names.json').write_text(json.dumps(caps));rows=[];files=[];coeffs=[]
 # Frozen source-prior baseline uses no target labels and exposes calibration-only collapse.
 prior=float(ys.mean());const=fastmetrics(y,np.full(len(y),prior));rows.append({'model':'constant','seed':0,'method':'source_prior','budget':0,'evaluation':'target_test',**const});log('evaluation',**rows[-1])
 # Diagnostic reference uses every retained target-pool label; never budget-matched.
 for name in ['lr','rf','hgb']:
  ref={'lr':lambda:LogisticRegression(max_iter=1000,C=1,random_state=11),'rf':lambda:RandomForestClassifier(n_estimators=150,min_samples_leaf=5,n_jobs=4,random_state=11),'hgb':lambda:HistGradientBoostingClassifier(max_iter=120,max_leaf_nodes=15,l2_regularization=1,random_state=11)}[name]();t=time.monotonic();ref.fit(xp,yp);prob=ref.predict_proba(xt)[:,1];row={'model':name,'seed':11,'method':'target_only_reference','budget':len(pool),'evaluation':'target_test',**fastmetrics(y,prob,{'ap':float(average_precision_score(y,prob)),'auroc':float(roc_auc_score(y,prob))})};rows.append(row);log('target_reference_fit',model=name,seed=11,labels=len(pool),seconds=time.monotonic()-t);log('evaluation',**row);joblib.dump(ref,O/f'target_reference_{name}.joblib',compress=3);np.savez_compressed(O/f'target_reference_scores_{name}.npz',probability=prob)
 for seed in SEEDS:
  strata=MiniBatchKMeans(n_clusters=8,random_state=seed,n_init=3,batch_size=2048).fit_predict(xp)
  for name in ['lr','rf','hgb']:
   model={'lr':lambda:LogisticRegression(max_iter=1000,C=1,random_state=seed),'rf':lambda:RandomForestClassifier(n_estimators=150,min_samples_leaf=5,n_jobs=4,random_state=seed),'hgb':lambda:HistGradientBoostingClassifier(max_iter=120,max_leaf_nodes=15,l2_regularization=1,random_state=seed)}[name]();t=time.monotonic();model.fit(xs,ys);log('source_fit',model=name,seed=seed,n=len(src),seconds=time.monotonic()-t,iterations=getattr(model,'n_iter_',None));joblib.dump(model,O/f'model_{name}_{seed}.joblib',compress=3)
   pp=model.predict_proba(xp)[:,1];pt=model.predict_proba(xt)[:,1];ph=model.predict_proba(xh)[:,1];score=logit(np.clip(pt,1e-6,1-1e-6));rank={'ap':float(average_precision_score(y,pt)),'auroc':float(roc_auc_score(y,pt))};np.savez_compressed(O/f'scores_{name}_{seed}.npz',target_probability=pt,target_logit=score,pool_probability=pp,source_holdout_probability=ph)
   row={'model':name,'seed':seed,'method':'source_only','budget':0,'evaluation':'source_holdout',**fastmetrics(st['_y'].to_numpy(),ph,{'ap':float(average_precision_score(st['_y'],ph)),'auroc':float(roc_auc_score(st['_y'],ph))})};rows.append(row);log('evaluation',**row)
   row={'model':name,'seed':seed,'method':'source_only','budget':0,'evaluation':'target_test',**fastmetrics(y,pt,rank)};rows.append(row);log('evaluation',**row)
   configs=[('source_only',0,pt,None)]
   # Cache acquisition sets; budgets are separate experiments rather than cumulative query accounting.
   for budget in BUDGETS:
    for method in METHODS:
     t=time.monotonic();ix,w,detail=select(method,xp,pp,strata,budget,seed);cal,info=calibrate(pp[ix],yp[ix],w);prob=expit(cal[0]*score+cal[1]);row={'model':name,'seed':seed,'method':method,'budget':budget,'evaluation':'target_test','a':float(cal[0]),'b':float(cal[1]),'ess':float(w.sum()**2/np.sum(w*w)),**fastmetrics(y,prob,rank)};rows.append(row);coeffs.append({k:row[k] for k in ['model','seed','method','budget','a','b']});log('calibration_fit',model=name,seed=seed,method=method,budget=budget,seconds=time.monotonic()-t,selected_hashes=pool['_hash'].iloc[ix].astype(str).tolist(),selected_labels=yp[ix].tolist(),weights=w.tolist(),parameters=cal.tolist(),**detail,**info);log('evaluation',**row)
     if budget==100:configs.append((method,budget,prob,cal))
   for method,budget,prob,_ in configs:
    for ci,cap in enumerate(caps):
     mask=capids==ci;files.append({'model':name,'seed':seed,'method':method,'budget':budget,'capture':cap,'family':family(cap),**fastmetrics(y[mask],prob[mask])})
   pd.DataFrame(rows).to_csv(O/'metrics.csv',index=False);pd.DataFrame(files).to_csv(O/'per_capture_metrics.csv',index=False);pd.DataFrame(coeffs).to_csv(O/'calibration_coefficients.csv',index=False);log('model_complete',model=name,seed=seed,total_evaluations=len(rows))
 # Sensitivity uses fixed RF seed 11, same budgets and test, K chosen in advance.
 obj=joblib.load(O/'scores_rf_11.npz') if False else np.load(O/'scores_rf_11.npz');sensitivity=[]
 for k in [4,16]:
  strata=MiniBatchKMeans(n_clusters=k,random_state=11,n_init=3,batch_size=2048).fit_predict(xp)
  for budget in BUDGETS:
   ix,w,detail=select('proposed',xp,obj['pool_probability'],strata,budget,11);cal,info=calibrate(obj['pool_probability'][ix],yp[ix],w);prob=expit(cal[0]*obj['target_logit']+cal[1]);sensitivity.append({'model':'rf','seed':11,'k':k,'budget':budget,**fastmetrics(y,prob)});log('sensitivity_fit',k=k,budget=budget,parameters=cal.tolist(),**detail,**info)
 pd.DataFrame(sensitivity).to_csv(O/'cluster_sensitivity.csv',index=False);log('run_complete',evaluation_rows=len(rows),source_fits=15,target_reference_fits=3,calibration_fits=420,sensitivity_fits=8,source_prior=prior)
if __name__=='__main__':main()
