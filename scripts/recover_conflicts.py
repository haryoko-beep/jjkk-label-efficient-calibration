"""Reconstruct pre-dedup source reservoirs without altering prior caches."""
import pathlib,json,hashlib,zipfile,datetime
import pandas as pd,numpy as np,joblib
R=pathlib.Path(__file__).resolve().parents[1];O=R/'results/strengthening';O.mkdir(exist_ok=True)
def main():
 if (O/'conflict_inventory.json').exists():print('Conflict inventory already complete');return
 dom,features=joblib.load(R/'data/processed_cap1000.joblib');parts={1000:[],5000:[]};inventory=[]
 with zipfile.ZipFile(R/'data/raw/CICIoT2023/CSV/CSV.zip') as z:
  names=sorted(n for n in z.namelist() if n.lower().endswith('.csv'))
  for i,name in enumerate(names):
   seed=int(hashlib.sha256(('CICIoT2023'+name).encode()).hexdigest()[:8],16);rng=np.random.default_rng(seed);sample=None;seen=0
   with z.open(name) as stream:
    for chunk in pd.read_csv(stream,chunksize=100000,low_memory=False):
     chunk.columns=[''.join(c for c in str(x).lower() if c.isalnum()) for x in chunk.columns];chunk['_key']=rng.random(len(chunk));seen+=len(chunk);sample=chunk if sample is None else pd.concat([sample,chunk],ignore_index=True)
     if len(sample)>5000:sample=sample.nsmallest(5000,'_key')
   for cap in [1000,5000]:
    d=sample.nsmallest(cap,'_key').reindex(columns=features).apply(pd.to_numeric,errors='coerce').replace([np.inf,-np.inf],np.nan).astype(float);h=pd.util.hash_pandas_object(d,index=False).astype(str);parts[cap].append(pd.DataFrame({'_hash':h.to_numpy(),'_y':int('benign' not in name.lower()),'_capture':name}));inventory.append({'cap':cap,'capture':name,'original_rows':seen,'sampled':len(d)})
   if i%20==0:print('Conflict scan',i+1,len(names),flush=True)
 result={}
 for cap in [1000,5000]:
  p=pd.concat(parts[cap],ignore_index=True);counts=p.groupby('_hash')._y.nunique();bad=set(counts[counts>1].index);p[p._hash.isin(bad)].to_csv(O/f'conflict_provenance_cap{cap}.csv',index=False);(O/f'conflict_hashes_cap{cap}.json').write_text(json.dumps(sorted(bad)));p.to_csv(O/f'reservoir_provenance_cap{cap}.csv.gz',index=False)
  old=dom['CICIoT2023'] if cap==1000 else joblib.load(R/'data/source_cap5000.joblib')[0];h=pd.util.hash_pandas_object(old[features].astype(float),index=False).astype(str);recovered=set(p._hash);oldset=set(h);excluded=h.isin(bad);result[str(cap)]={'sampled_rows_before_dedup':len(p),'reconstructed_unique_vectors':len(recovered),'old_retained_rows':len(old),'old_unique_vectors_canonical':len(oldset),'conflict_hashes':len(bad),'old_rows_excluded':int(excluded.sum()),'excluded_benign':int(((old._y.to_numpy()==0)&excluded.to_numpy()).sum()),'excluded_attack':int(((old._y.to_numpy()==1)&excluded.to_numpy()).sum()),'old_vectors_missing_from_reconstruction':len(oldset-recovered),'reconstructed_vectors_missing_from_old':len(recovered-oldset),'old_hash_recomputed_mismatch':int(np.sum(h.to_numpy()!=old._hash.astype(str).to_numpy()))};assert oldset==recovered,'Source cohort mismatch; investigate before fitting';print(cap,result[str(cap)],flush=True)
 pd.DataFrame(inventory).to_csv(O/'reservoir_reconstruction_inventory.csv',index=False);result['time_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat();result['protocol_sha256']=hashlib.sha256((R/'STRENGTHENING_PROTOCOL.md').read_bytes()).hexdigest();(O/'conflict_inventory.json').write_text(json.dumps(result,indent=2));print('Conflict reconstruction complete',flush=True)
if __name__=='__main__':main()
