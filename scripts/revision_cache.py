import pathlib,zipfile,hashlib,json,time
import pandas as pd,numpy as np,joblib
from experiment import reservoir
R=pathlib.Path(__file__).resolve().parents[1];O=R/'results/revision';O.mkdir(exist_ok=True)
def main():
 p=R/'data/source_cap5000.joblib'
 if p.exists():print('source cache exists');return
 _,features=joblib.load(R/'data/processed_cap1000.joblib');frames=[];rows=[]
 with zipfile.ZipFile(R/'data/raw/CICIoT2023/CSV/CSV.zip') as z:
  names=sorted(n for n in z.namelist() if n.lower().endswith('.csv'))
  for i,name in enumerate(names):
   seed=int(hashlib.sha256(('CICIoT2023'+name).encode()).hexdigest()[:8],16)
   with z.open(name) as f:df,n=reservoir(f,5000,seed)
   df=df.reindex(columns=features).apply(pd.to_numeric,errors='coerce').replace([np.inf,-np.inf],np.nan);df['_capture']=name;df['_family']=pathlib.PurePosixPath(name).parent.name;df['_y']=int('benign' not in name.lower());df['_hash']=pd.util.hash_pandas_object(df[features],index=False).astype(str);frames.append(df);rows.append({'capture':name,'original':n,'retained_before_dedup':len(df)})
   if i%20==0:print('capture',i+1,len(names),flush=True)
 df=pd.concat(frames,ignore_index=True);before=len(df);conflicts=df.groupby('_hash')._y.nunique();pd.DataFrame([{'before':before,'unique':df._hash.nunique(),'conflicting_feature_hashes':int((conflicts>1).sum())}]).to_csv(O/'source_dedup_audit.csv',index=False);df=df.drop_duplicates('_hash').reset_index(drop=True);joblib.dump((df,features),p,compress=0);pd.DataFrame(rows).to_csv(O/'cap5000_inventory.csv',index=False);print('cache complete',len(df),flush=True)
if __name__=='__main__':main()
