import pathlib,json,numpy as np,pandas as pd
R=pathlib.Path(__file__).resolve().parents[1];O=R/'results/tuning'
def main():
 e=[json.loads(x) for x in (R/'logs/tuning_training.jsonl').read_text().splitlines()];assert e[-1]['event']=='run_complete'
 d=pd.read_csv(O/'metrics.csv',float_precision='round_trip')
 # Preserve nextafter cutoffs exactly: decimal CSV parsing can move a cutoff by
 # one ULP and change tied-score decisions. JSON raw logs retain binary64 values.
 cuts={(x['model'],x['seed'],x['budget'],x['method'],x['threshold_mode']):x['threshold'] for x in e if x['event']=='evaluation'}
 chosen=json.loads((O/'selected_candidates.json').read_text())
 exact=[]
 for row in d.itertuples():
  cut=cuts[(row.model,row.seed,row.budget,row.method,row.threshold_mode)] if row.budget else (.5 if row.threshold_mode=='native_05' else chosen[row.model]['source_threshold'])
  exact.append(float(cut))
 d['threshold']=exact;d['threshold_hex']=[x.hex() for x in exact];d['alarms_per_10000']=10000*(d.tp+d.fp)/d.n;d.to_csv(O/'metrics.csv',index=False)
 ff=pd.read_csv(O/'family_metrics.csv');ff['alarms_per_10000']=10000*(ff.tp+ff.fp)/ff.n;ff.to_csv(O/'family_metrics.csv',index=False)
 d.groupby(['model','budget','method','threshold_mode']).agg({k:['mean','std'] for k in ['fpr','recall','mcc','balanced_brier','brier','joint_goal_5pct','joint_goal_1pct']}).to_csv(O/'summary.csv')
 acc=pd.read_csv(O/'label_accounting.csv');assert (acc.total_labels==acc.budget).all() and acc.overlap.eq(0).all();thr=pd.read_csv(O/'thresholds.csv');thr.groupby(['budget','mode']).agg(benign_mean=('benign_n','mean'),benign_min=('benign_n','min'),upper95_mean=('validation_fpr_upper95','mean')).to_csv(O/'threshold_support.csv')
 # All candidates reported; no best-test selection is used for a deployment claim.
 success=d[d.budget.gt(0)].groupby(['model','method','budget','threshold_mode']).agg(joint_success_runs=('joint_goal_5pct','sum'),runs=('seed','count'),fpr=('fpr','mean'),recall=('recall','mean'),mcc=('mcc','mean'));success.to_csv(O/'joint_goal_matrix.csv')
 contrasts=[]
 for model in ['lr','rf']:
  for B in [100,250,1000]:
   for ab in ['cal_no_weights','cal_no_uncertainty','cal_random']:
    a=d[(d.model==model)&d.budget.eq(B)&d.method.eq('cal_proposed')&d.threshold_mode.eq('target_validation_05')].set_index('seed');b=d[(d.model==model)&d.budget.eq(B)&d.method.eq(ab)&d.threshold_mode.eq('target_validation_05')].set_index('seed')
    contrasts.append({'model':model,'budget':B,'comparison':ab,'delta_balanced_brier':float((a.balanced_brier-b.balanced_brier).mean()),'delta_fpr':float((a.fpr-b.fpr).mean()),'delta_recall':float((a.recall-b.recall).mean()),'delta_mcc':float((a.mcc-b.mcc).mean())})
 pd.DataFrame(contrasts).to_csv(O/'ablation_contrasts.csv',index=False)
 details={'evaluations':len(d),'source_grid_fits':sum(x['event']=='source_grid_fit' for x in e),'calibration_fits':sum(x['event']=='calibration_fit' for x in e),'retraining_fits':sum(x['event']=='retraining_fit' for x in e),'intercept_fits':sum(x['event']=='intercept_fit' for x in e),'budget_conditions':sum(x['event']=='budget_complete' for x in e),'fivepct_joint_success_rows':int(d.joint_goal_5pct.sum()),'test_status':'previously_inspected_exploratory_reuse','label_accounting_pass':True};(O/'run_summary.json').write_text(json.dumps(details,indent=2));print(json.dumps(details,indent=2));print(success[success.joint_success_runs>0].to_string())
if __name__=='__main__':main()
