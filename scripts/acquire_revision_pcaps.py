"""Official chronological packet prefixes, never represented as full captures."""
import pathlib,requests,json,re,html,struct,lzma,hashlib,concurrent.futures,datetime
from urllib.parse import quote,urljoin
R=pathlib.Path(__file__).resolve().parents[1];O=R/'results/revision';O.mkdir(exist_ok=True)
BASE={'CICIoT2023':'https://cicresearch.ca/IOTDataset/CIC_IOT_Dataset2023/','CICIoMT2024':'https://cicresearch.ca/IOTDataset/CICIoMT2024/','IoT23':'https://mcfp.felk.cvut.cz/publicDatasets/IoT-23-Dataset/IndividualScenarios/'}
def header(h):
 fmt={b'\xd4\xc3\xb2\xa1':('<',1e6),b'\xa1\xb2\xc3\xd4':('>',1e6),b'\x4d\x3c\xb2\xa1':('<',1e9),b'\xa1\xb2\x3c\x4d':('>',1e9)}
 if len(h)!=24 or h[:4] not in fmt:raise ValueError('Unsupported/non-PCAP response')
 e,precision=fmt[h[:4]];return e,precision,struct.unpack(e+'I',h[20:24])[0]
def get(job):
 domain,path,partition,label=job;s=requests.Session()
 url=BASE[domain]+('download.php?file='+quote(path) if domain!='IoT23' else path)
 out=R/'data/pcap_revision'/domain/(path.split('/')[-1].replace('.xz',''));meta=out.with_suffix('.json');out.parent.mkdir(parents=True,exist_ok=True)
 if out.exists() and meta.exists():return json.loads(meta.read_text())
 if domain!='IoT23':
  # PHP download requests can hold a session lock during a large transfer.
  # A separate official registration session per capture avoids serialized waits.
  s.get(BASE[domain],timeout=30);reg=s.post(BASE[domain]+'insert.php',data=json.loads((R/'tmp/cic_identity.json').read_text()),timeout=30);reg.raise_for_status();assert reg.json().get('ok')
 with s.get(url,stream=True,timeout=(30,90)) as resp:
  resp.raise_for_status();resp.raw.decode_content=True;stream=lzma.LZMAFile(resp.raw) if path.endswith('.xz') else resp.raw
  h=stream.read(24);e,precision,link=header(h);n=0;eof=False;tmp=out.with_suffix('.part')
  with tmp.open('wb') as f:
   f.write(h)
   while n<100000:
    rh=stream.read(16)
    if not rh:eof=True;break
    if len(rh)!=16:raise ValueError('Truncated record header')
    sec,frac,caplen,wirelen=struct.unpack(e+'IIII',rh)
    if caplen>2000000:raise ValueError('Invalid packet length')
    pkt=stream.read(caplen)
    if len(pkt)!=caplen:raise ValueError('Truncated packet')
    f.write(rh);f.write(pkt);n+=1
  tmp.replace(out)
 row={'domain':domain,'relative_path':path,'url':url,'partition':partition,'scenario_label':label,'local_path':out.relative_to(R).as_posix(),'packets':n,'reached_eof':eof,'linktype':link,'timestamp_divisor':precision,'bytes':out.stat().st_size,'sha256_prefix':hashlib.sha256(out.read_bytes()).hexdigest(),'time_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()};meta.write_text(json.dumps(row,indent=2));print('PCAP',domain,path,n,eof,flush=True);return row
def main():
 # Session cookies expire between workdays. Renew using the previously authorized
 # local registration identity without recording personal fields in the manifest.
 for domain in ['CICIoT2023','CICIoMT2024']:
  s=requests.Session();s.get(BASE[domain],timeout=30);r=s.post(BASE[domain]+'insert.php',data=json.loads((R/'tmp/cic_identity.json').read_text()),timeout=30);r.raise_for_status();assert r.json().get('ok');(R/'tmp'/f'{domain}_session.json').write_text(json.dumps(s.cookies.get_dict()))
 jobs=[]
 for i in range(4):jobs.append(('CICIoT2023','PCAP/Benign_Final/BenignTraffic'+(str(i) if i else '')+'.pcap','train' if i<2 else 'source_holdout',0))
 jobs.append(('CICIoT2023','PCAP/Recon-PortScan/Recon-PortScan.pcap','train',1))
 for i in range(3):jobs.append(('CICIoT2023','PCAP/DoS-SYN_Flood/DoS-SYN_Flood'+(str(i) if i else '')+'.pcap','train' if i<2 else 'source_holdout',1))
 for part in ['train','test']:
  for name in ['Benign','Recon-Port_Scan','TCP_IP-DoS-SYN']:
   filename=name+('1' if part=='train' and name=='TCP_IP-DoS-SYN' else '')
   jobs.append(('CICIoMT2024',f'WiFI_and_MQTT/attacks/PCAP/{part}/{filename}_{part}.pcap',part,int(name!='Benign')))
 rows=[];errors=[]
 with concurrent.futures.ThreadPoolExecutor(2) as ex:
  futs={ex.submit(get,j):j for j in jobs}
  for f in concurrent.futures.as_completed(futs):
   try:rows.append(f.result())
   except Exception as e:errors.append({'job':futs[f],'error':str(e)});print('FAIL',futs[f],str(e),flush=True)
 (O/'pcap_manifest.json').write_text(json.dumps(rows,indent=2));(O/'pcap_acquisition_errors.json').write_text(json.dumps(errors,indent=2))
 # Preserve external provider metadata and connection labels; no model outcomes here.
 external=[]
 for name in ['CTU-IoT-Malware-Capture-1-1','CTU-IoT-Malware-Capture-21-1','CTU-Honeypot-Capture-4-1','CTU-Honeypot-Capture-5-1','CTU-Honeypot-Capture-7-1']:
  try:
   suffixdir='Somfy-01/' if name=='CTU-Honeypot-Capture-7-1' else ''
   b=BASE['IoT23']+name+'/'+suffixdir;r=requests.get(b,timeout=40);r.raise_for_status();links=[html.unescape(x) for x in re.findall('href="([^"]+)"',r.text)];pc=next(x for x in links if x.endswith(('.pcap','.pcap.xz')));local=R/'data/pcap_revision/IoT23'/name;local.mkdir(parents=True,exist_ok=True)
   for suffix,target in [('README.md','README.md'),('bro/conn.log.labeled','conn.log.labeled')]:
    dest=local/target
    if not dest.exists():
     resource=(BASE['IoT23']+name+'/README.md') if target=='README.md' and suffixdir else b+suffix
     with requests.get(resource,stream=True,timeout=(30,90)) as rr:
      if rr.status_code==404 and target=='README.md' and suffixdir:
       (local/'provider_index.html').write_text(r.text,encoding='utf-8');continue
      rr.raise_for_status()
      with dest.with_suffix('.part').open('wb') as out:
       for chunk in rr.iter_content(1024*1024):out.write(chunk)
     dest.with_suffix('.part').replace(dest)
   row=get(('IoT23',name+'/'+suffixdir+pc,'external',None));row['label_file']=(local/'conn.log.labeled').relative_to(R).as_posix();row['capture_name']=name;row['label_sha256']=hashlib.sha256((local/'conn.log.labeled').read_bytes()).hexdigest();external.append(row)
  except Exception as e:errors.append({'capture':name,'error':str(e)});print('External FAIL',name,str(e),flush=True)
 (O/'pcap_manifest.json').write_text(json.dumps(rows+external,indent=2));(O/'pcap_acquisition_errors.json').write_text(json.dumps(errors,indent=2));print('finished',len(rows),len(external),len(errors),flush=True)
if __name__=='__main__':main()
