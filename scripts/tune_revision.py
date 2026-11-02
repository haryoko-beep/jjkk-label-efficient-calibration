import os
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='4'
import pathlib,json,time,datetime,hashlib
import numpy as np,pandas as pd,joblib
from scipy.special import expit,logit
from scipy.optimize import minimize
from scipy.stats import beta
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import roc_auc_score,average_precision_score
from experiment import select,calibrate
from full_study import fastmetrics
R=pathlib.Path(__file__).resolve().parents[1];O=R/'results/tuning';O.mkdir(exist_ok=True)
SEEDS=[11,23,37,53,71];BUDGETS=[100,250,1000]
def log(event,**kw):
 row={'time_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'event':event,**kw}
 with (R/'logs/tuning_training.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(row,default=str)+'\n')
 print(json.dumps({k:v for k,v in row.items() if k not in ['hashes','labels','weights','loss_history']},default=str),flush=True)
def threshold(y,p,alpha):
 benign=np.sort(np.asarray(p)[np.asarray(y)==0]);n=len(benign)
 if n==0:return float('inf'),0
 # Smallest upper-tail cutoff with empirical FPR <= alpha, handling ties conservatively.
 allowed=int(np.floor(alpha*n));idx=max(0,n-allowed-1);return float(np.nextafter(benign[idx],np.inf)),n
def evaluate(y,p,t):
 d=fastmetrics(y,p);pr=p>=t;yy=y.astype(bool);tn=int(np.sum(~pr&~yy));fp=int(np.sum(pr&~yy));fn=int(np.sum(~pr&yy));tp=int(np.sum(pr&yy));den=np.sqrt(float((tp+fp)*(tp+fn))*float((tn+fp)*(tn+fn)));d.update(tn=tn,fp=fp,fn=fn,tp=tp,fpr=fp/max(1,fp+tn),recall=tp/max(1,tp+fn),precision=tp/max(1,tp+fp),macro_f1=.5*(2*tp/max(1,2*tp+fp+fn)+2*tn/max(1,2*tn+fp+fn)),mcc=(tp*tn-fp*fn)/den if den else 0.)
 for c,n in [(0,'benign'),(1,'attack')]:
  q=np.clip(p[y==c],1e-7,1-1e-7);d['brier_'+n]=float(np.mean((q-c)**2));d['nll_'+n]=float(np.mean(-c*np.log(q)-(1-c)*np.log1p(-q)))
 d['alarms_per_10000']=10000*(tp+fp)/len(y);d['balanced_brier']=(d['brier_benign']+d['brier_attack'])/2;d['balanced_nll']=(d['nll_benign']+d['nll_attack'])/2;d['joint_goal_5pct']=bool(d['fpr']<=.05 and d['recall']>=.8);d['joint_goal_1pct']=bool(d['fpr']<=.01 and d['recall']>=.8);return d
def prep(df,features,imp,sc,fit=False):
 a=df[features].to_numpy(float);a=np.sign(a)*np.log1p(np.abs(a));a=sc.fit_transform(imp.fit_transform(a)) if fit else sc.transform(imp.transform(a));return np.clip(a,-100,100)
def main():
 log('run_start',protocol_sha256=hashlib.sha256((R/'TUNING_PROTOCOL.md').read_bytes()).hexdigest(),test_status='previously_inspected_exploratory_reuse',seeds=SEEDS,budgets=BUDGETS)
 dom,allfeatures=joblib.load(R/'data/processed_cap1000.joblib');src=dom['CICIoT2023'].copy();pool=dom['CICIoMT2024'];pool=pool[pool['_partition'].eq('pool')].copy()
 for df in [src,pool]:df[allfeatures]=df[allfeatures].astype(float);df['_hash']=pd.util.hash_pandas_object(df[allfeatures],index=False).astype(str)
 def recover(df,name):
  ids=pd.read_csv(R/f'results/full/{name}_manifest.csv.gz',dtype={'_hash':str});return df[df['_hash'].isin(set(ids['_hash']))].copy()
 src=recover(src,'source_train');pool=recover(pool,'target_pool').reset_index(drop=True)
 neg=src[src['_y']==0];pos=src[src['_y']==1].sample(n=min(30000,int((src['_y']==1).sum())),random_state=20261101);src=pd.concat([neg,pos]).reset_index(drop=True)
 fitids,validids=train_test_split(np.arange(len(src)),test_size=.2,stratify=src['_y'],random_state=20261101);train=src.iloc[fitids];val=src.iloc[validids];ys=train['_y'].to_numpy();yv=val['_y'].to_numpy()
 sets={'full':allfeatures,'protocol_flags':[c for c in allfeatures if c in ['protocoltype','http','https','dns','telnet','smtp','ssh','irc','tcp','udp','dhcp','arp','icmp','igmp','ipv','llc'] or 'flag' in c or c in ['ackcount','syncount','fincount','rstcount']],'packet_lengths':[c for c in allfeatures if c in ['min','max','avg','std','totsize','totsum','variance','headerlength']]}
 candidates=[];objects={};i=0
 for fs,features in sets.items():
  imp=SimpleImputer(strategy='median',keep_empty_features=True);sc=RobustScaler();x=prep(train,features,imp,sc,True);v=prep(val,features,imp,sc)
  for kind in ['lr','rf']:
   for param in ([.1,1.] if kind=='lr' else [5,20]):
    for cw in [None,'balanced']:
     i+=1;t=time.monotonic();m=LogisticRegression(C=param,class_weight=cw,max_iter=1000,random_state=11) if kind=='lr' else RandomForestClassifier(n_estimators=120,min_samples_leaf=param,class_weight=cw,n_jobs=4,random_state=11);m.fit(x,ys);prob=m.predict_proba(v)[:,1];cut,n=threshold(yv,prob,.05);a=evaluate(yv,prob,cut);row={'candidate':i,'kind':kind,'feature_set':fs,'features':len(features),'parameter':param,'class_weight':str(cw),'source_threshold':cut,'validation_auroc':roc_auc_score(yv,prob),'validation_recall':a['recall'],'validation_fpr':a['fpr'],'seconds':time.monotonic()-t};candidates.append(row);objects[i]={'model':m,'imputer':imp,'scaler':sc,'features':features,'source_threshold':cut};log('source_grid_fit',**row)
 pd.DataFrame(candidates).to_csv(O/'source_grid.csv',index=False);chosen={}
 for kind in ['lr','rf']:
  best=sorted([r for r in candidates if r['kind']==kind],key=lambda r:(-r['validation_recall'],-r['validation_auroc'],r['features'],r['candidate']))[0];chosen[kind]=objects[best['candidate']];joblib.dump(chosen[kind],O/f'selected_{kind}.joblib');log('source_selected',**best)
 (O/'selected_candidates.json').write_text(json.dumps({k:next(r for r in candidates if objects[r['candidate']] is v) for k,v in chosen.items()},indent=2))
 # Test is loaded only after source candidate selection is frozen.
 test=joblib.load(R/'results/full/complete_target_test.joblib');y=test['_y'].to_numpy(np.int8);rows=[];families=[];account=[];thresholds=[]
 for kind,obj in chosen.items():
  features=obj['features'];xp=prep(pool,features,obj['imputer'],obj['scaler']);xt=prep(test,features,obj['imputer'],obj['scaler']);xs=prep(train,features,obj['imputer'],obj['scaler']);pp=obj['model'].predict_proba(xp)[:,1];pt=obj['model'].predict_proba(xt)[:,1];np.savez_compressed(O/f'source_scores_{kind}.npz',probability=pt)
  for mode,t in [('native_05',.5),('source_validation_05',obj['source_threshold'])]:rows.append({'model':kind,'seed':0,'budget':0,'method':'tuned_source','threshold_mode':mode,'threshold':t,**evaluate(y,pt,t)})
  for seed in SEEDS:
   rng=np.random.default_rng(seed);strata=MiniBatchKMeans(n_clusters=8,n_init=3,batch_size=2048,random_state=seed).fit_predict(xp)
   for B in BUDGETS:
    nv=int(.4*B);na=B-nv;vi=rng.choice(len(pool),nv,replace=False);remain=np.setdiff1d(np.arange(len(pool)),vi);vlabels=pool['_y'].to_numpy()[vi];pv=pp[vi];adapt={}
    for method in ['random','proposed','no_weights','no_uncertainty']:
     ix,w,detail=select(method,xp[remain],pp[remain],strata[remain],na,seed);ids=remain[ix];labels=pool['_y'].to_numpy()[ids];coef,info=calibrate(pp[ids],labels,w);adapt[method]=(ids,w,labels);p=expit(coef[0]*logit(np.clip(pt,1e-6,1-1e-6))+coef[1]);vp=expit(coef[0]*logit(np.clip(pv,1e-6,1-1e-6))+coef[1]);log('calibration_fit',model=kind,seed=seed,budget=B,method=method,adaptation_labels=na,validation_labels=nv,hashes=pool['_hash'].iloc[ids].tolist(),labels=labels.tolist(),weights=w.tolist(),coef=coef.tolist(),**detail,**info)
     configs=[('cal_'+method,p,vp)]
     if method in ['random','proposed']:
      # RF target-only and source+target retraining use exactly the same queried labels.
      for mixing in [False,True]:
       rf=RandomForestClassifier(n_estimators=120,min_samples_leaf=2,class_weight='balanced',n_jobs=4,random_state=seed)
       xx=xp[ids];yy=labels;weights=np.ones(len(ids))
       if mixing:
        ss=np.arange(len(train));weights=np.concatenate([np.full(len(ss),.5/len(ss)),np.full(len(ids),.5/len(ids))]);xx=np.concatenate([xs,xp[ids]]);yy=np.concatenate([ys,labels])
       rf.fit(xx,yy,sample_weight=weights);name=('mix_' if mixing else 'target_')+method
       if len(rf.classes_)==1:p2=np.full(len(y),float(rf.classes_[0]));vp2=np.full(nv,float(rf.classes_[0]))
       else:p2=rf.predict_proba(xt)[:,1];vp2=rf.predict_proba(xp[vi])[:,1]
       configs.append((name,p2,vp2));joblib.dump(rf,O/f'{kind}_{seed}_{B}_{name}.joblib',compress=3);log('retraining_fit',model=kind,seed=seed,budget=B,method=name,adaptation_labels=na,validation_labels=nv,classes=rf.classes_.tolist())
     if method=='random':
      scores=logit(np.clip(pp[ids],1e-6,1-1e-6));fit=minimize(lambda b:np.mean(np.logaddexp(0,scores+b[0])-labels*(scores+b[0]))+.01*b[0]**2,[0.],method='L-BFGS-B');b=float(fit.x[0]);configs.append(('intercept_random',expit(logit(np.clip(pt,1e-6,1-1e-6))+b),expit(logit(np.clip(pv,1e-6,1-1e-6))+b)));log('intercept_fit',model=kind,seed=seed,budget=B,intercept=b,success=bool(fit.success))
     account.append({'model':kind,'seed':seed,'budget':B,'selector':method,'adaptation_labels':na,'validation_labels':nv,'total_labels':na+nv,'adaptation_benign':int((labels==0).sum()),'validation_benign':int((vlabels==0).sum()),'overlap':len(set(ids)&set(vi))});log('query_validation',model=kind,seed=seed,budget=B,method=method,hashes=pool['_hash'].iloc[vi].tolist(),labels=vlabels.tolist())
     for name,prob,vprob in configs:
      np.savez_compressed(O/f'scores_{kind}_{seed}_{B}_{name}.npz',probability=prob,validation_probability=vprob)
      for mode,alpha in [('native_05',None),('target_validation_05',.05),('target_validation_01',.01)]:
       cut=.5;nb=int((vlabels==0).sum())
       if alpha is not None:cut,nb=threshold(vlabels,vprob,alpha)
       ev=evaluate(y,prob,cut);row={'model':kind,'seed':seed,'budget':B,'method':name,'threshold_mode':mode,'threshold':cut,'validation_benign':nb,'fallback_no_benign':nb==0,**ev};rows.append(row);ve=evaluate(vlabels,vprob,cut) if len(np.unique(vlabels))==2 else None;upper=float(beta.ppf(.95,(ve['fp'] if ve else 0)+1,nb-(ve['fp'] if ve else 0))) if nb and (ve['fp'] if ve else 0)<nb else 1.;thresholds.append({'model':kind,'seed':seed,'budget':B,'method':name,'mode':mode,'threshold':cut,'benign_n':nb,'validation_fpr_upper95':upper});log('evaluation',**row)
       if B==250 and mode=='target_validation_05':
        for fam,df in test.groupby('_family'):
         ii=test.index.get_indexer(df.index);families.append({'model':kind,'seed':seed,'method':name,'family':fam,**evaluate(y[ii],prob[ii],cut)})
    pd.DataFrame(rows).to_csv(O/'metrics.csv',index=False);pd.DataFrame(account).to_csv(O/'label_accounting.csv',index=False);pd.DataFrame(thresholds).to_csv(O/'thresholds.csv',index=False);pd.DataFrame(families).to_csv(O/'family_metrics.csv',index=False);log('budget_complete',model=kind,seed=seed,budget=B,evaluations=len(rows))
 log('run_complete',evaluations=len(rows),source_grid_fits=len(candidates),target_validation_policy='40pct_of_total_budget',test_status='exploratory_reuse')
if __name__=='__main__':main()
