import pathlib,json,itertools
import pandas as pd,numpy as np,matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=pathlib.Path(__file__).resolve().parents[1];O=R/'results/revision';F=R/'figures'
def main():
 d=pd.read_csv(O/'factorial_metrics.csv');a=d[d.budget.eq(250)].copy();out=[]
 for (cap,split,model,mode),g in a.groupby(['cap','split','model','mode']):
  for metric in ['balanced_brier','brier_benign','brier_attack','fpr','recall','mcc']:
   v=g.pivot(index='seed',columns=['entropy','weighting'],values=metric);e0=(v[(False,True)]-v[(False,False)]);e1=(v[(True,True)]-v[(True,False)]);out.append({'cap':cap,'split':split,'model':model,'mode':mode,'metric':metric,'weight_effect_entropy_off':e0.mean(),'weight_effect_entropy_on':e1.mean(),'interaction':(e1-e0).mean(),'entropy_effect_unweighted':(v[(True,False)]-v[(False,False)]).mean(),'entropy_effect_weighted':(v[(True,True)]-v[(False,True)]).mean()})
 pd.DataFrame(out).to_csv(O/'factorial_interactions.csv',index=False);a.groupby(['cap','split','model','method','mode'])[['fpr','recall','precision','mcc','balanced_brier','macro_family_recall']].mean().to_csv(O/'split_sensitivity.csv');s=pd.read_csv(O/'coverage_stress.csv');s.groupby(['budget','coverage','entropy','weighting'])[['clusters_reached','pool_mass_represented','ess','benign','balanced_brier','fpr','recall']].mean().to_csv(O/'coverage_summary.csv')
 # Independent factors are not independent repeated-domain trials.
 plt.rcParams.update({'svg.fonttype':'none','font.size':9});fig,axes=plt.subplots(1,2,figsize=(8.4,3.3))
 for ax,model in zip(axes,['lr','rf']):
  g=a[a.model.eq(model)&a['mode'].eq('target_validation_05')].groupby(['cap','split'])[['fpr','recall']].mean()
  from matplotlib.lines import Line2D
  for cap,color in [(1000,'#1f77b4'),(5000,'#ff7f0e')]:
   for split,v in g.loc[cap].iterrows():ax.scatter(v.fpr,v.recall,marker={101:'o',202:'s',303:'^'}[split],color=color,s=36,alpha=.8)
  ax.axvline(.05,c='black',ls=':');ax.axhline(.8,c='black',ls=':');ax.set(xlabel='Target FPR',ylabel='Attack recall',title=model.upper(),xlim=(-.02,1.02),ylim=(-.02,1.02))
  handles=[Line2D([],[],color=color,lw=2,label=f'cap {cap}') for cap,color in [(1000,'#1f77b4'),(5000,'#ff7f0e')]]+[Line2D([],[],color='black',marker=marker,ls='none',label=f'split {split}') for split,marker in [(101,'o'),(202,'s'),(303,'^')]];ax.legend(handles=handles,fontsize=6,ncol=2)
 fig.tight_layout()
 for ext in ['png','svg']:fig.savefig(F/f'revision_splits.{ext}',dpi=220,bbox_inches='tight')
 plt.close(fig);fig,ax=plt.subplots(figsize=(6.7,3.2))
 for cov,style in [(False,'--'),(True,'-')]:
  z=s[s.coverage.eq(cov)].groupby('budget')[['clusters_reached','pool_mass_represented']].mean();ax.plot(z.index,z.clusters_reached,'o'+style,label='Coverage '+('on' if cov else 'off'))
 ax.set(xlabel='Adaptation labels',ylabel='Mean reached clusters out of 32');ax.legend();ax.grid(alpha=.2);fig.tight_layout()
 for ext in ['png','svg']:fig.savefig(F/f'revision_coverage.{ext}',dpi=220,bbox_inches='tight')
 plt.close(fig)
 # Show class-conditioned score distributions on the exact original test.
 y=np.load(R/'results/full/test_labels_and_ids.npz')
 print('test label archive keys',list(y.keys()),flush=True)
 if (O/'common_complete.json').exists():
  c=pd.read_csv(O/'common_metrics.csv');c[c['mode'].eq('target_validation_05')].groupby(['model','method','domain'])[['fpr','recall','precision','mcc','balanced_brier']].agg(['mean','std']).to_csv(O/'common_summary.csv')
  p=pd.read_csv(O/'common_predictions.csv.gz');fig,axes=plt.subplots(1,2,figsize=(8.4,3.3))
  for ax,domain in zip(axes,['target_test','external_iot23']):
   for lab,name in [(0,'Benign'),(1,'Attack')]:
    z=p[p.model.eq('rf')&p.method.eq('cal_diversity')&p.seed.eq(11)&p.domain.eq(domain)&p.y.eq(lab)];ax.hist(z.probability,bins=np.linspace(0,1,21),density=True,histtype='step',lw=1.5,label=name)
   ax.set(xlabel='Calibrated probability',ylabel='Density',title=domain.replace('_',' '));ax.legend()
  fig.tight_layout()
  for ext in ['png','svg']:fig.savefig(F/f'revision_score_distributions.{ext}',dpi=220,bbox_inches='tight')
  plt.close(fig)
 print('analysis complete',flush=True)
if __name__=='__main__':main()
