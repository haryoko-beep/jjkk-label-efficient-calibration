"""Official CIC registration, inventory, resumable download and integrity logs.
Credentials are read from a local JSON file; never written to the public manifest.
"""
import argparse, concurrent.futures, datetime, hashlib, html, json, pathlib, re, shutil, time
from urllib.parse import urljoin, urlparse, parse_qs, unquote
import requests

ROOT = pathlib.Path(__file__).resolve().parents[1]
BASES = {'CICIoT2023':'https://cicresearch.ca/IOTDataset/CIC_IOT_Dataset2023/',
         'CICIoMT2024':'https://cicresearch.ca/IOTDataset/CICIoMT2024/'}
def stamp(): return datetime.datetime.now(datetime.timezone.utc).isoformat()
def log(event, **kw):
    row={'time_utc':stamp(),'event':event,**kw}
    with (ROOT/'logs/acquisition.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(row)+'\n')
    print(json.dumps(row),flush=True)
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--identity',required=True);ap.add_argument('--download',action='store_true');ap.add_argument('--all-raw',action='store_true');ap.add_argument('--workers',type=int,default=3);a=ap.parse_args()
    identity=json.loads(pathlib.Path(a.identity).read_text(encoding='utf-8'));inventory=[];sessions={}
    for name,base in BASES.items():
        s=requests.Session();s.get(base,timeout=30)
        r=s.post(urljoin(base,'insert.php'),data=identity,timeout=30);r.raise_for_status();result=r.json()
        log('registration',dataset=name,ok=result.get('ok'),response_type=result.get('type'))
        if not result.get('ok'):raise RuntimeError(result)
        sessions[name]=s
        queue=[urljoin(base,result.get('redirect','browse.php'))];seen=set()
        while queue:
            u=queue.pop(0)
            if u in seen:continue
            seen.add(u);r=s.get(u,timeout=60);r.raise_for_status();r.encoding='utf-8'
            rel=parse_qs(urlparse(u).query).get('p',[''])[0]
            (ROOT/'sources'/f'{name}_listing_{len(seen):04d}.html').write_text(r.text,encoding='utf-8')
            links=[html.unescape(x) for x in re.findall(r'href="([^"]+)"',r.text)]
            for href in links:
                dest=urljoin(base,href)
                if 'browse.php?' in href:
                    sub=parse_qs(urlparse(dest).query).get('p',[''])[0]
                    if '..' in pathlib.PurePosixPath(sub).parts:continue
                    if not a.all_raw and (any(x.lower() in sub.lower() for x in ['pcap','bluetooth','profiling']) or (name=='CICIoT2023' and sub.startswith(('CSV/CSV','CSV/MERGED_CSV')))):continue
                    queue.append(dest)
                elif 'download.php?' in href:
                    fn=parse_qs(urlparse(dest).query).get('file',[''])[0]
                    if not fn:continue
                    if not a.all_raw and not (any(x in fn.lower() for x in ['csv','readme','example','supplement','feature']) or fn.lower().endswith(('.py','.ipynb','.pdf'))):continue
                    if not a.all_raw and name=='CICIoT2023' and fn=='CSV/MERGED_CSV.zip':continue
                    inventory.append({'dataset':name,'relative_path':fn,'url':dest,'status':'discovered'})
            log('directory_inventory',dataset=name,path=rel,file_count=len(inventory),pending_directories=len(queue))
        (ROOT/'tmp'/f'{name}_session.json').write_text(json.dumps(s.cookies.get_dict()))
    uniq={(r['dataset'],r['relative_path']):r for r in inventory};inventory=list(uniq.values())
    inventory.sort(key=lambda r:(not r['relative_path'].lower().endswith('.pdf'),r['dataset']=='CICIoT2023',not any(k in r['relative_path'].lower() for k in ['benign','arp_spoof','recon']),'/train/' not in r['relative_path'],r['relative_path']))
    manifest=ROOT/'data/official_manifest.json';manifest.write_text(json.dumps(inventory,indent=2))
    if not a.download:return
    def get(row):
        name=row['dataset'];rel=pathlib.PurePosixPath(row['relative_path'])
        if rel.is_absolute() or '..' in rel.parts:raise ValueError('Unsafe server path')
        p=ROOT/'data/raw'/name/pathlib.Path(*rel.parts);p.parent.mkdir(parents=True,exist_ok=True);part=p.with_name(p.name+'.part')
        s=requests.Session();s.cookies.update(sessions[name].cookies)
        for attempt in range(4):
            try:
                if p.exists():
                    row.update(status='complete',bytes=p.stat().st_size,sha256=sha(p));return row
                if shutil.disk_usage(ROOT).free<6*1024**3:raise RuntimeError('Less than 6 GiB free; download stopped without deleting files')
                offset=part.stat().st_size if part.exists() else 0
                r=s.get(row['url'],headers={'Range':f'bytes={offset}-'} if offset else {},stream=True,timeout=(20,120));r.raise_for_status()
                if 'text/html' in r.headers.get('Content-Type',''):raise RuntimeError('HTML returned instead of data')
                append=offset and r.status_code==206
                if append and not r.headers.get('Content-Range','').startswith(f'bytes {offset}-'):
                    raise RuntimeError('Server returned an inconsistent Content-Range')
                written=0
                with part.open('ab' if append else 'wb') as f:
                    for chunk in r.iter_content(65536):
                        if chunk:
                            f.write(chunk);written+=len(chunk)
                            if written%(32*1024*1024)<len(chunk) and shutil.disk_usage(ROOT).free<6*1024**3:
                                raise RuntimeError('Less than 6 GiB free during transfer; partial file preserved')
                if r.headers.get('Content-Length') and written!=int(r.headers['Content-Length']):
                    raise RuntimeError('Downloaded byte count does not match Content-Length')
                part.replace(p);row.update(status='complete',bytes=p.stat().st_size,sha256=sha(p));log('download_complete',dataset=name,path=str(rel),bytes=row['bytes'],sha256=row['sha256']);return row
            except Exception as e:
                log('download_retry',dataset=name,path=str(rel),attempt=attempt,error=str(e));time.sleep(min(2**attempt,8))
        row.update(status='failed');return row
    with concurrent.futures.ThreadPoolExecutor(max_workers=a.workers) as ex:
        for row in ex.map(get,inventory):manifest.write_text(json.dumps(inventory,indent=2))
    log('acquisition_finished',complete=sum(r['status']=='complete' for r in inventory),total=len(inventory))
if __name__=='__main__':main()
