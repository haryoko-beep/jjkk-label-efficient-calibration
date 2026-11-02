import pathlib,json,hashlib,datetime,itertools,time
import numpy as np,pandas as pd,joblib
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import MiniBatchKMeans
from scipy.special import expit,logit
from experiment import calibrate
from revision_factorial import choose,families
from tune_revision import prep,threshold,evaluate
R=pathlib.Path(__file__).resolve().parents[1];O=R/'results/strengthening';O.mkdir(exist_ok=True)
def log(event,**kw):
 d={'time_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'event':event,**kw}
 with (R/'logs/conflict_sensitivity.jsonl').open('a') as f:f.write(json.dumps(d,default=str)+'\n')
 print(event,{k:v for k,v in kw.items() if k in ['cap','split','model','seed','rows','removed_fit','removed_validation']},flush=True)
def membership(train):
 benign=train[train._y.eq(0)];attack=train[train._y.eq(1)].sample(n=min(30000,int(train._y.sum())),random_state=20261101);s=pd.concat([benign,attack]).reset_index(drop=True);fi,vi=train_test_split(np.arange(len(s)),test_size=.2,stratify=s._y,random_state=20261101);return s.iloc[fi].copy(),s.iloc[vi].copy()
def main():
 if (O/'conflict_complete.json').exists():print('already complete');return
 resume=(O/'conflict_clean_metrics.csv').exists()
 inventory=json.loads((O/'conflict_inventory.json').read_text());log('run_start',protocol_sha256=hashlib.sha256((R/'STRENGTHENING_PROTOCOL.md').read_bytes()).hexdigest());dom,features=joblib.load(R/'data/processed_cap1000.joblib');pool=dom['CICIoMT2024'];pool=pool[pool._partition.eq('pool')].copy();pool[features]=pool[features].astype(float);pool['_hash']=pd.util.hash_pandas_object(pool[features],index=False).astype(str);ids=pd.read_csv(R/'results/full/target_pool_manifest.csv.gz',dtype={'_hash':str});pool=pool[pool._hash.isin(set(ids._hash))].reset_index(drop=True);test=joblib.load(R/'results/full/complete_target_test.joblib');y=test._y.to_numpy(np.int8);parts=json.loads((R/'results/revision/group_partitions.json').read_text());rows=[];account=[];fits=[]
 if resume:
  rows=pd.read_csv(O/'conflict_clean_metrics.csv').to_dict('records');account=pd.read_csv(O/'conflict_queries.csv').to_dict('records');fits=pd.read_csv(O/'conflict_fit_accounting.csv').to_dict('records');log('audited_resume',completed_conditions=len(fits),reason='RF parallel tree reduction gives one-ULP cutoff variation; fixed membership preserved')
 for cap in [1000,5000]:
  src=dom['CICIoT2023'].copy() if cap==1000 else joblib.load(R/'data/source_cap5000.joblib')[0].copy();src[features]=src[features].astype(float);src['_hash']=pd.util.hash_pandas_object(src[features],index=False).astype(str);bad=set(json.loads((O/f'conflict_hashes_cap{cap}.json').read_text()))
  for split in [101,202,303]:
   held=set(next(p['held_captures'] for p in parts if p['cap']==cap and p['split']==split));train=src[~src._capture.isin(held)].copy();hold0=src[src._capture.isin(held)].copy();hold=hold0[~hold0._hash.isin(bad)].copy();fit0,val0=membership(train);fit=fit0[~fit0._hash.isin(bad)].copy();val=val0[~val0._hash.isin(bad)].copy();nf=len(fit0)-len(fit);nv=len(val0)-len(val)
   for kind in ['lr','rf']:
    if any(x['cap']==cap and x['split']==split and x['model']==kind for x in fits):continue
    old=joblib.load(R/f'results/revision/source_{cap}_{split}_{kind}.joblib');oldpv=old['model'].predict_proba(prep(val0,features,old['imputer'],old['scaler']))[:,1];oldcut,_=threshold(val0._y.to_numpy(),oldpv,.05);ulp=abs(oldcut-old['source_threshold'])/abs(np.spacing(float(old['source_threshold'])));assert ulp<=8,'Old membership threshold mismatch exceeds floating-point reduction tolerance';log('membership_verification',cap=cap,split=split,model=kind,cutoff_ulp_difference=float(ulp),fit_n=len(fit0),validation_n=len(val0))
    imp=SimpleImputer(strategy='median',keep_empty_features=True);sc=RobustScaler();xx=prep(fit,features,imp,sc,True);xv=prep(val,features,imp,sc);m=LogisticRegression(C=1,max_iter=1000,random_state=split) if kind=='lr' else RandomForestClassifier(n_estimators=120,min_samples_leaf=20,class_weight='balanced',n_jobs=4,random_state=split);m.fit(xx,fit._y);cut,_=threshold(val._y.to_numpy(),m.predict_proba(xv)[:,1],.05);obj={'model':m,'imputer':imp,'scaler':sc,'features':features,'source_threshold':cut};joblib.dump(obj,O/f'clean_source_{cap}_{split}_{kind}.joblib');xp=prep(pool,features,imp,sc);xt=prep(test,features,imp,sc);pp=m.predict_proba(xp)[:,1];pt=m.predict_proba(xt)[:,1];np.savez_compressed(O/f'clean_scores_{cap}_{split}_{kind}.npz',test_probability=pt,pool_probability=pp);rec={'cap':cap,'split':split,'model':kind,'old_fit_n':len(fit0),'old_validation_n':len(val0),'clean_fit_n':len(fit),'clean_validation_n':len(val),'removed_fit':nf,'removed_validation':nv,'removed_holdout':len(hold0)-len(hold),'old_cutoff_hex':float(oldcut).hex(),'new_cutoff_hex':float(cut).hex(),'reconstructed_membership_verified':True};fits.append(rec);log('source_refit',**rec,fit_hashes=fit._hash.tolist(),validation_hashes=val._hash.tolist(),excluded_fit_hashes=fit0[fit0._hash.isin(bad)]._hash.tolist(),excluded_validation_hashes=val0[val0._hash.isin(bad)]._hash.tolist())
    for domain,d,p in [('target_test',test,pt),('source_holdout',hold,m.predict_proba(prep(hold,features,imp,sc))[:,1])]:rows.append({'cap':cap,'split':split,'model':kind,'seed':0,'budget':0,'method':'source_only','domain':domain,'mode':'source_validation_05','threshold_hex':float(cut).hex(),**evaluate(d._y.to_numpy(np.int8),p,cut)})
    for seed in [11,23,37]:
     strata=MiniBatchKMeans(n_clusters=8,n_init=3,batch_size=2048,random_state=seed).fit_predict(xp);vi=np.random.default_rng(seed).choice(len(pool),100,replace=False);remain=np.setdiff1d(np.arange(len(pool)),vi);vl=pool._y.to_numpy()[vi]
     for entropy,weight in itertools.product([False,True],repeat=2):
      ii,w,detail=choose(pp[remain],strata[remain],150,seed,entropy,weight);ix=remain[ii];lab=pool._y.to_numpy()[ix];coef,info=calibrate(pp[ix],lab,w);p=expit(coef[0]*logit(np.clip(pt,1e-6,1-1e-6))+coef[1]);vp=expit(coef[0]*logit(np.clip(pp[vi],1e-6,1-1e-6))+coef[1]);method=f'E{int(entropy)}W{int(weight)}';acct={'cap':cap,'split':split,'model':kind,'seed':seed,'budget':250,'method':method,'entropy':entropy,'weighting':weight,'overlap':len(set(ix)&set(vi)),'benign_adaptation':int((lab==0).sum()),'benign_validation':int((vl==0).sum()),**detail};account.append(acct);log('calibration_fit',**acct,hashes=pool._hash.iloc[ix].tolist(),validation_hashes=pool._hash.iloc[vi].tolist(),labels=lab.tolist(),weights=w.tolist(),coef=coef.tolist(),**info)
      for mode,ct in [('native_05',.5),('target_validation_05',threshold(vl,vp,.05)[0])]:rows.append({'cap':cap,'split':split,'model':kind,'seed':seed,'budget':250,'method':method,'entropy':entropy,'weighting':weight,'domain':'target_test','mode':mode,'threshold_hex':float(ct).hex(),'macro_family_recall':families(y,p,ct,test),**evaluate(y,p,ct)})
    pd.DataFrame(rows).to_csv(O/'conflict_clean_metrics.csv',index=False);pd.DataFrame(fits).to_csv(O/'conflict_fit_accounting.csv',index=False);pd.DataFrame(account).to_csv(O/'conflict_queries.csv',index=False);log('condition_complete',cap=cap,split=split,model=kind,rows=len(rows))
 old=pd.read_csv(R/'results/revision/factorial_metrics.csv');new=pd.DataFrame(rows);keys=['cap','split','model','seed','budget','method','domain','mode'];paired=new.merge(old,on=keys,suffixes=('_clean','_original'),validate='one_to_one');
 for k in ['fpr','recall','mcc','brier','balanced_brier','macro_family_recall']:paired['delta_'+k]=paired[k+'_clean']-paired[k+'_original']
 paired.to_csv(O/'conflict_paired_metrics.csv',index=False);(O/'conflict_complete.json').write_text(json.dumps({'source_refits':len(fits),'calibration_fits':len(account),'evaluations':len(rows),'paired_rows':len(paired),'fixed_fit_membership':True,'test_status':'reused exploratory sensitivity'},indent=2));log('run_complete',rows=len(rows))
if __name__=='__main__':main()
