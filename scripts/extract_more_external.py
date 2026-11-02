import pathlib,json,struct,hashlib,datetime
import numpy as np,pandas as pd
from common_pcap_features import FEATURES,decode,labels,packetlabel,vector
from acquire_revision_pcaps import header,R
O=R/'results/strengthening'
def main():
 rows=json.loads((O/'new_external_manifest.json').read_text());frames=[];audits=[]
 for row in rows:
  table,lc,bad=labels(R/row['label_file']);window=[];wl=[];vectors=[];ys=[];starts=[];seen=valid=unknown=mixed=0
  with (R/row['local_path']).open('rb') as f:
   e,div,link=header(f.read(24));assert link==1
   while rh:=f.read(16):
    assert len(rh)==16;sec,frac,n,wire=struct.unpack(e+'IIII',rh);b=f.read(n);assert len(b)==n;seen+=1;p=decode(b,sec+frac/div,wire)
    if p is None:continue
    valid+=1;window.append(p);wl.append(packetlabel(p,table))
    if len(window)==100:
     known=[x for x in wl if x is not None];y=None
     if len(known)<90:unknown+=1
     elif len(set(known))!=1:mixed+=1
     else:y=known[0]
     if y is not None:vectors.append(vector(window));ys.append(y);starts.append(window[0][0])
     window=[];wl=[]
  df=pd.DataFrame(vectors,columns=FEATURES);df['_y']=ys;df['_start']=starts;df['_capture']=row['capture_name'];df['_domain']='IoT23';df['_partition']='new_external';df['_family']=row['capture_name'];df['_hash']=pd.util.hash_pandas_object(df[FEATURES],index=False).astype(str);frames.append(df);audit={'capture':row['capture_name'],'packets':seen,'valid_packets':valid,'complete_windows':valid//100,'eligible':len(df),'benign':int(df._y.eq(0).sum()),'attack':int(df._y.eq(1).sum()),'excluded_unknown':unknown,'excluded_mixed':mixed,'incomplete_packets':len(window),'reached_eof':row['reached_eof'],'connection_labels':lc,'bad_connection_rows':bad};audits.append(audit);print(audit,flush=True)
 pd.concat(frames,ignore_index=True).to_csv(O/'new_external_windows.csv.gz',index=False);pd.DataFrame(audits).to_csv(O/'new_external_extraction_audit.csv',index=False);(O/'new_extractor_provenance.json').write_text(json.dumps({'feature_definition_sha256':hashlib.sha256((R/'scripts/common_pcap_features.py').read_bytes()).hexdigest(),'driver_sha256':hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),'features':FEATURES,'protocol_sha256':hashlib.sha256((R/'STRENGTHENING_PROTOCOL.md').read_bytes()).hexdigest()},indent=2))
if __name__=='__main__':main()
