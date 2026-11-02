import pathlib,json,re,requests,hashlib,concurrent.futures,time,struct
from acquire_revision_pcaps import BASE,get,R,header
O=R/'results/strengthening';O.mkdir(exist_ok=True)
CAPTURES=['CTU-IoT-Malware-Capture-'+x for x in ['8-1','20-1','42-1','60-1']]
def acquire(name):
 local=R/'data/pcap_revision/IoT23'/name;local.mkdir(exist_ok=True);meta=O/(name+'_manifest.json')
 if meta.exists():return json.loads(meta.read_text())
 b=BASE['IoT23']+name+'/';r=requests.get(b,timeout=40);r.raise_for_status();(local/'provider_index.html').write_text(r.text,encoding='utf-8');pcs=sorted(x for x in re.findall('href="([^"]+)"',r.text) if x.endswith(('.pcap','.pcap.xz')));assert pcs
 for suffix,target in [('README.md','README.md'),('bro/conn.log.labeled','conn.log.labeled')]:
  p=local/target
  if not p.exists():
   for attempt in range(3):
    try:
     with requests.get(b+suffix,stream=True,timeout=(30,90)) as rr:
      rr.raise_for_status()
      with p.with_suffix('.part').open('wb') as f:
       for chunk in rr.iter_content(1024*1024):f.write(chunk)
     p.with_suffix('.part').replace(p);break
    except Exception:
     if attempt==2:raise
  print('metadata',name,target,p.stat().st_size,flush=True)
 try:row=get(('IoT23',name+'/'+pcs[0],'new_external',None))
 except ValueError as error:
  if str(error) not in ['Truncated packet','Truncated record header']:raise
  # Validate a complete HTTP response independently before preserving its valid
  # packet prefix. A malformed final source record is never invented or padded.
  rr=requests.get(b+pcs[0],timeout=(30,90));rr.raise_for_status();body=rr.content
  if rr.headers.get('Content-Length'):assert len(body)==int(rr.headers['Content-Length'])
  e,div,link=header(body[:24]);offset=24;n=0;terminal=None
  while offset<len(body) and n<100000:
   start=offset
   if len(body)-offset<16:terminal='truncated_terminal_header';break
   sec,frac,cl,wl=struct.unpack(e+'IIII',body[offset:offset+16]);assert cl<=2000000
   if len(body)-offset-16<cl:terminal='truncated_terminal_packet';break
   offset+=16+cl;n+=1
  assert terminal is not None and n>0
  out=R/'data/pcap_revision/IoT23'/pcs[0].replace('.xz','');out.write_bytes(body[:offset]);row={'domain':'IoT23','relative_path':name+'/'+pcs[0],'url':b+pcs[0],'partition':'new_external','scenario_label':None,'local_path':out.relative_to(R).as_posix(),'packets':n,'reached_eof':False,'source_eof_reached':True,'termination':terminal,'source_http_bytes':len(body),'source_http_sha256':hashlib.sha256(body).hexdigest(),'discarded_terminal_bytes':len(body)-offset,'linktype':link,'timestamp_divisor':div,'bytes':offset,'sha256_prefix':hashlib.sha256(body[:offset]).hexdigest()};out.with_suffix('.json').write_text(json.dumps(row,indent=2));print('Malformed source terminal record excluded',name,n,len(body)-offset,flush=True)
 row.update(capture_name=name,label_file=(local/'conn.log.labeled').relative_to(R).as_posix(),label_sha256=hashlib.sha256((local/'conn.log.labeled').read_bytes()).hexdigest(),protocol_sha256=hashlib.sha256((R/'STRENGTHENING_PROTOCOL.md').read_bytes()).hexdigest());meta.write_text(json.dumps(row,indent=2));return row
def main():
 rows=[];errors=[]
 with concurrent.futures.ThreadPoolExecutor(2) as ex:
  futures={ex.submit(acquire,n):n for n in CAPTURES}
  for f in concurrent.futures.as_completed(futures):
   try:rows.append(f.result())
   except Exception as e:errors.append({'capture':futures[f],'error':str(e)});print('FAILED',errors[-1],flush=True)
 (O/'new_external_manifest.json').write_text(json.dumps(sorted(rows,key=lambda x:x['capture_name']),indent=2));(O/'acquisition_errors.json').write_text(json.dumps(errors,indent=2));print('Finished',len(rows),'captures',len(errors),'errors',flush=True)
if __name__=='__main__':main()
