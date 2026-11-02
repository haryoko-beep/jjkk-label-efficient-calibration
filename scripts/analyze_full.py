"""Exploratory paired capture bootstrap, budget integrals and shift diagnostics."""
import pathlib,json,numpy as np,pandas as pd
from scipy.stats import wilcoxon,ks_2samp
from full_study import R,O
def main():
 d=pd.read_csv(O/'metrics.csv');f=pd.read_csv(O/'per_capture_metrics.csv');out=[];rng=np.random.default_rng(20261031)
 # Paired bootstrap over retained test files, averaging fixed seeds first.
 # Intervals are conditional on the distributed captures; not independent-domain CIs.
 for model in ['lr','rf','hgb']:
  for comparator in ['random','uncertainty','coreset','no_coverage','no_uncertainty','no_weights']:
   a=f[(f.model==model)&f.method.eq('proposed')].groupby('capture')[['brier','nll','n']].mean();b=f[(f.model==model)&f.method.eq(comparator)].groupby('capture')[['brier','nll','n']].mean();a,b=a.align(b,join='inner');n=a.n.to_numpy();ix=rng.integers(0,len(n),size=(5000,len(n)))
   for metric in ['brier','nll']:
    diff=a[metric].to_numpy()-b[metric].to_numpy();boot=np.sum(diff[ix]*n[ix],axis=1)/np.sum(n[ix],axis=1);lo,hi=np.quantile(boot,[.025,.975]);p=1. if np.allclose(diff,0) else float(wilcoxon(diff,zero_method='wilcox',alternative='two-sided').pvalue)
    out.append({'model':model,'comparator':comparator,'metric':metric,'budget':100,'delta_full_minus_control':float(np.sum(diff*n)/n.sum()),'bootstrap_lower':float(lo),'bootstrap_upper':float(hi),'files':len(n),'wilcoxon_file_p_exploratory':p})
 # Holm adjustment is exploratory and does not solve correlated capture dependence.
 pvals=np.array([x['wilcoxon_file_p_exploratory'] for x in out]);order=np.argsort(pvals);adj=np.maximum.accumulate((len(pvals)-np.arange(len(pvals)))*pvals[order]);adj=np.minimum(adj,1);adjusted=np.empty(len(out));adjusted[order]=adj
 for row,p in zip(out,adjusted):row['holm_p_exploratory']=p
 pd.DataFrame(out).to_csv(O/'paired_capture_intervals.csv',index=False)
 curves=[]
 for (model,seed,method),g in d[d.budget.gt(0)&~d.method.eq('target_only_reference')].groupby(['model','seed','method']):
  g=g.sort_values('budget');x=np.log(g.budget.to_numpy());curves.append({'model':model,'seed':seed,'method':method,'aubc_brier':np.trapezoid(g.brier,x)/(x[-1]-x[0]),'aubc_nll':np.trapezoid(g.nll,x)/(x[-1]-x[0])})
 pd.DataFrame(curves).to_csv(O/'budget_integrals.csv',index=False)
 # Balanced-prior stress diagnostic computed exactly from conditional losses,
 # not synthetic observations, not an operating threshold selected on test.
 stress=[]
 for (model,seed,method),g in f.groupby(['model','seed','method']):
  for prevalence in [.01,.05,.2]:
   benign=g[g.capture.str.contains('Benign')];attack=g[~g.capture.str.contains('Benign')]
   def av(x,k):return np.average(x[k],weights=x.n)
   stress.append({'model':model,'seed':seed,'method':method,'attack_prevalence':prevalence,'brier':(1-prevalence)*av(benign,'brier')+prevalence*av(attack,'brier'),'nll':(1-prevalence)*av(benign,'nll')+prevalence*av(attack,'nll'),'fpr':av(benign,'fpr'),'recall':av(attack,'recall')})
 pd.DataFrame(stress).to_csv(O/'prevalence_stress.csv',index=False)
 # Query composition is observed only after selection; never fed back to the rule.
 events=[json.loads(s) for s in (R/'logs/full_training.jsonl').read_text().splitlines()];query=[]
 for e in events:
  if e['event']!='calibration_fit':continue
  w=np.asarray(e['weights'],float);yy=np.asarray(e['selected_labels']);query.append({'model':e['model'],'seed':e['seed'],'method':e['method'],'budget':e['budget'],'benign_labels':int(np.sum(yy==0)),'attack_labels':int(np.sum(yy==1)),'single_class_query':int(np.unique(yy).size==1),'zero_inclusion_strata':e.get('zero_inclusion_strata',0),'ess':float(w.sum()**2/np.sum(w*w)),'maximum_weight':float(w.max()),'slope':e['parameters'][0],'slope_at_lower_bound':int(e['parameters'][0]<=.00010001),'converged':e['success']})
 pd.DataFrame(query).to_csv(O/'query_diagnostics.csv',index=False)
 # Drift measured on source and pool only, never used to choose features.
 import joblib
 domains,features=joblib.load(R/'data/processed_cap1000.joblib');src=domains['CICIoT2023'];pool=domains['CICIoMT2024'];pool=pool[pool['_partition'].eq('pool')];pd.DataFrame([{'feature':c,'ks_source_pool':ks_2samp(src[c].dropna(),pool[c].dropna()).statistic,'source_median':src[c].median(),'pool_median':pool[c].median(),'source_missing':int(src[c].isna().sum()),'pool_missing':int(pool[c].isna().sum())} for c in features]).to_csv(O/'feature_drift.csv',index=False)
 d.groupby(['model','method','budget','evaluation'])[['brier','nll','fpr','recall','macro_f1','mcc','ece','ap','auroc']].agg(['mean','std']).to_csv(O/'summary.csv')
 print(json.dumps({'evaluation_rows':len(d),'paired_contrasts':len(out),'budget_curves':len(curves),'stress_conditions':len(stress)},indent=2))
if __name__=='__main__':main()
