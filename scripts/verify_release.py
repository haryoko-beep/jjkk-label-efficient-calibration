"""Verify every tracked release artifact using the standard library."""
import hashlib, json, pathlib
R=pathlib.Path(__file__).resolve().parents[1]
manifest=json.loads((R/'MANIFEST.sha256.json').read_text(encoding='utf-8'))
errors=[]
for name,digest in manifest.items():
 p=R/name
 if not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest()!=digest:errors.append(name)
if errors:raise SystemExit('Checksum mismatch or missing file: '+', '.join(errors))
print('Verified',len(manifest),'release artifacts.')
