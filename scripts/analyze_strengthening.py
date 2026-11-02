import pathlib,json
import numpy as np,pandas as pd,matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=pathlib.Path(__file__).resolve().parents[1];O=R/'results/strengthening';F=R/'figures'
def main():
 p=pd.read_csv(O/'conflict_paired_metrics.csv');d=p[p.budget.eq(250)&p['mode'].eq('target_validation_05')];d.groupby(['cap','model'])[['delta_fpr','delta_recall','delta_mcc','delta_balanced_brier']].agg(['mean','min','max']).to_csv(O/'conflict_summary.csv');e=pd.read_csv(O/'external_metrics.csv');e.groupby(['model','method','domain','mode'])[['fpr','recall','macro_capture_recall','worst_capture_recall','mcc','joint_goal_5pct']].agg(['mean','std']).to_csv(O/'external_summary.csv');c=pd.read_csv(O/'external_capture_metrics.csv');c['benign']=c.tn+c.fp;c['attack']=c.tp+c.fn;c[c['mode'].eq('target_validation_05')].groupby(['model','method','domain','capture'])[['fpr','recall','n','benign','attack']].mean().to_csv(O/'capture_summary.csv')
 plt.rcParams.update({'svg.fonttype':'none','font.size':8});methods=['cal_diversity','cal_random','target_diversity','target_random','mix_diversity','mix_random'];names=['Calibration diversity','Calibration random','Target RF diversity','Target RF random','Mixed RF diversity','Mixed RF random'];z=c[c.model.eq('rf')&c['mode'].eq('target_validation_05')&c.domain.eq('expanded_external')];fig,axes=plt.subplots(1,2,figsize=(9,3.7),gridspec_kw={'width_ratios':[1.6,1]})
 for ax,value,label in [(axes[0],'recall','Attack recall'),(axes[1],'fpr','Benign FPR')]:
  piv=z.groupby(['method','capture'])[value].mean().unstack('capture').reindex(methods);piv=piv.dropna(axis=1,how='all');a=piv.to_numpy();im=ax.imshow(a,aspect='auto',vmin=0,vmax=1,cmap='YlGnBu');ax.set_xticks(np.arange(len(piv.columns)),[n.replace('CTU-IoT-Malware-Capture-','M').replace('CTU-Honeypot-Capture-','H') for n in piv.columns],rotation=45,ha='right');ax.set_yticks(range(6),names if ax==axes[0] else []);ax.set_title(label)
  for (y,x),v in np.ndenumerate(a):
   if np.isfinite(v):ax.text(x,y,f'{v:.2f}',ha='center',va='center',fontsize=7,color='white' if v>.65 else 'black')
 fig.subplots_adjust(left=.18,right=.87,bottom=.25,wspace=.16);cax=fig.add_axes([.91,.25,.015,.63]);fig.colorbar(im,cax=cax)
 for ext in ['svg','png']:fig.savefig(F/f'strengthening_capture_metrics.{ext}',dpi=220,bbox_inches='tight')
 plt.close(fig);summary={'conflict_evaluations':len(p),'external_evaluations':len(e),'expanded_joint_success':e[e['mode'].eq('target_validation_05')&e.domain.eq('expanded_external')].groupby(['model','method']).joint_goal_5pct.sum().to_dict()};summary['expanded_joint_success']={str(k):int(v) for k,v in summary['expanded_joint_success'].items()};(O/'analysis_summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
