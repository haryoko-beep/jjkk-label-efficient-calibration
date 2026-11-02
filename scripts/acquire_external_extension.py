"""Reproduce the disclosed pre-model label-feasibility amendment, not model search."""
import pathlib,re,requests,json,hashlib
from acquire_revision_pcaps import get,BASE,R
def main():
 name='CTU-IoT-Malware-Capture-3-1';b=BASE['IoT23']+name+'/';r=requests.get(b,timeout=30);r.raise_for_status();pc=next(x for x in re.findall('href="([^"]+)"',r.text) if x.endswith(('.pcap','.pcap.xz')));local=R/'data/pcap_revision/IoT23'/name;local.mkdir(exist_ok=True)
 for suffix,target in [('README.md','README.md'),('bro/conn.log.labeled','conn.log.labeled')]:
  p=local/target
  if not p.exists():
   with requests.get(b+suffix,stream=True,timeout=(30,90)) as rr:
    rr.raise_for_status()
    with p.with_suffix('.part').open('wb') as f:
     for chunk in rr.iter_content(1024*1024):f.write(chunk)
   p.with_suffix('.part').replace(p)
 row=get(('IoT23',name+'/'+pc,'external',None));row.update(label_file=(local/'conn.log.labeled').relative_to(R).as_posix(),capture_name=name,label_sha256=hashlib.sha256((local/'conn.log.labeled').read_bytes()).hexdigest());(R/'results/revision/external_eligibility_extension.json').write_text(json.dumps(row,indent=2));print('extension acquired')
if __name__=='__main__':main()
