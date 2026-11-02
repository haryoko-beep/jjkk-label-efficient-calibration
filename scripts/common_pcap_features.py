"""One transparent Ethernet packet-window implementation for all three datasets."""
import pathlib,json,struct,socket,collections,hashlib,ipaddress
import numpy as np,pandas as pd
from acquire_revision_pcaps import header
R=pathlib.Path(__file__).resolve().parents[1];O=R/'results/revision'
FEATURES=['length_mean','length_std','length_min','length_max','length_q25','length_q75','iat_mean','iat_std','iat_max','duration','tcp_fraction','udp_fraction','icmp_fraction','arp_fraction','ipv6_fraction','syn_fraction','ack_fraction','rst_fraction','fin_fraction','negative_iat_fraction']
def decode(b,t,wire):
 if len(b)<14:return None
 et=struct.unpack('!H',b[12:14])[0];off=14
 while et in [0x8100,0x88a8]:
  if len(b)<off+4:return None
  et=struct.unpack('!H',b[off+2:off+4])[0];off+=4
 proto=0;key=None;flags=0
 if et==0x0800:
  if len(b)<off+20:return None
  hl=(b[off]&15)*4
  if hl<20 or len(b)<off+hl:return None
  proto=b[off+9];src=socket.inet_ntop(socket.AF_INET,b[off+12:off+16]);dst=socket.inet_ntop(socket.AF_INET,b[off+16:off+20]);frag=struct.unpack('!H',b[off+6:off+8])[0]&8191;off+=hl
 elif et==0x86dd:
  if len(b)<off+40:return None
  proto=b[off+6];src=socket.inet_ntop(socket.AF_INET6,b[off+8:off+24]);dst=socket.inet_ntop(socket.AF_INET6,b[off+24:off+40]);off+=40;frag=0
 elif et==0x0806:return (t,wire,0,et,0,None)
 else:return None
 if proto in [6,17] and not frag and len(b)>=off+4:
  sp,dp=struct.unpack('!HH',b[off:off+4]);key=(src,sp,dst,dp,'tcp' if proto==6 else 'udp')
  if proto==6 and len(b)>=off+14:flags=b[off+13]
 return (t,wire,proto,et,flags,key)
def labels(path):
 table=collections.defaultdict(list);counts=collections.Counter();bad=0
 with path.open(encoding='utf-8',errors='replace') as f:
  for line in f:
   if line.startswith('#'):continue
   a=line.split()
   if len(a)<23:bad+=1;continue
   try:
    ts=float(a[0]);dur=0 if a[8]=='-' else float(a[8]);lab=a[-2].lower();y=0 if lab=='benign' else 1 if lab=='malicious' else None
    if y is None:counts['unknown']+=1;continue
    key=(str(ipaddress.ip_address(a[2])),int(a[3]),str(ipaddress.ip_address(a[4])),int(a[5]),a[6]);val=(ts-1,ts+dur+1,y);table[key].append(val);table[(key[2],key[3],key[0],key[1],key[4])].append(val);counts[lab]+=1
   except (ValueError,IndexError):bad+=1
 return table,dict(counts),bad
def packetlabel(p,table):
 if p[5] is None:return None
 labs={y for start,end,y in table.get(p[5],[]) if start<=p[0]<=end}
 return next(iter(labs)) if len(labs)==1 else None
def vector(window):
 a=np.array([p[:5] for p in window],float);dt=np.diff(a[:,0]);length=a[:,1];proto=a[:,2].astype(int);etype=a[:,3].astype(int);flags=a[:,4].astype(int)
 return [length.mean(),length.std(),length.min(),length.max(),*np.quantile(length,[.25,.75]),dt.mean(),dt.std(),dt.max(),a[-1,0]-a[0,0],np.mean(proto==6),np.mean(proto==17),np.mean(np.isin(proto,[1,58])),np.mean(etype==0x806),np.mean(etype==0x86dd),*[np.mean((flags&bit)>0) for bit in [2,16,4,1]],np.mean(dt<0)]
def main():
 manifest=json.loads((O/'pcap_manifest.json').read_text())
 if (O/'external_eligibility_extension.json').exists():manifest.append(json.loads((O/'external_eligibility_extension.json').read_text()))
 frames=[];audits=[]
 for row in manifest:
  path=R/row['local_path'];capture=row.get('capture_name',path.stem);table=None;lc={};bad=0
  if row['domain']=='IoT23':table,lc,bad=labels(R/row['label_file'])
  window=[];wl=[];vectors=[];ys=[];starts=[];seen=valid=unknown=ambiguous=0
  with path.open('rb') as f:
   h=f.read(24);e,div,link=header(h)
   if link!=1:raise ValueError('Unsupported link type '+str(link))
   while rh:=f.read(16):
    if len(rh)!=16:raise ValueError('Truncated record')
    sec,frac,caplen,wire=struct.unpack(e+'IIII',rh);b=f.read(caplen)
    if len(b)!=caplen:raise ValueError('Truncated packet')
    seen+=1;p=decode(b,sec+frac/div,wire)
    if p is None:continue
    valid+=1;window.append(p)
    if table is not None:wl.append(packetlabel(p,table))
    if len(window)==100:
     if table is None:y=row['scenario_label']
     else:
      known=[x for x in wl if x is not None]
      if len(known)<90:unknown+=1;y=None
      elif len(set(known))!=1:ambiguous+=1;y=None
      else:y=known[0]
     if y is not None:vectors.append(vector(window));ys.append(y);starts.append(window[0][0])
     window=[];wl=[]
  df=pd.DataFrame(vectors,columns=FEATURES);df['_y']=ys;df['_start']=starts;df['_capture']=capture;df['_domain']=row['domain'];df['_partition']=row['partition'];df['_family']='benign' if row.get('scenario_label')==0 else 'scan' if 'Scan' in capture else 'syn' if 'SYN' in capture else capture;df['_hash']=pd.util.hash_pandas_object(df[FEATURES],index=False).astype(str);frames.append(df);audits.append({'domain':row['domain'],'capture':capture,'partition':row['partition'],'packets':seen,'valid_packets':valid,'incomplete_window_packets':len(window),'windows_eligible':len(df),'windows_benign':int(df._y.eq(0).sum()),'windows_attack':int(df._y.eq(1).sum()),'excluded_low_label_coverage':unknown,'excluded_mixed_labels':ambiguous,'connection_labels':lc,'bad_connection_rows':bad,'complete_capture':row['reached_eof']});print(audits[-1],flush=True)
 data=pd.concat(frames,ignore_index=True);data.to_csv(O/'common_pcap_windows.csv.gz',index=False);pd.DataFrame(audits).to_csv(O/'common_pcap_extraction_audit.csv',index=False);(O/'common_extractor.json').write_text(json.dumps({'features':FEATURES,'window_packets':100,'extractor_sha256':hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),'protocol_sha256':hashlib.sha256((R/'REVIEW_REVISION_PROTOCOL.md').read_bytes()).hexdigest(),'rows':len(data)},indent=2))
if __name__=='__main__':main()
