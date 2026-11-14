"""Put unchanged cached text weights on local SSD for fast worker restarts."""
from pathlib import Path
import hashlib,json,os,shutil

ROOT=Path('/workspace/freegraftor')
SOURCE=Path('/workspace/instantx-0807/cache/huggingface/hub/models--black-forest-labs--FLUX.1-dev/snapshots/3de623fc3c33e44ffbe2bad470d0f45bccf2eb21')
DEST=Path('/opt/freegraftor-runtime/text-encoders')
files=[p for folder in ['text_encoder','text_encoder_2'] for p in (SOURCE/folder).iterdir() if p.is_file()]
required=sum(p.stat().st_size for p in files if not (DEST/p.relative_to(SOURCE)).exists())
assert shutil.disk_usage('/opt').free>required+2*1024**3,'Keep at least 2GiB local disk headroom'
audit={}
for p in files:
    target=DEST/p.relative_to(SOURCE);target.parent.mkdir(parents=True,exist_ok=True)
    if not target.exists():
        temporary=target.with_suffix(target.suffix+'.tmp')
        h=hashlib.sha256()
        with p.open('rb') as src,temporary.open('wb') as dst:
            for b in iter(lambda:src.read(16*1024*1024),b''):dst.write(b);h.update(b)
        if p.suffix=='.safetensors':assert h.hexdigest()==p.resolve().name,'HF cached blob checksum mismatch'
        temporary.replace(target)
        print('Copied/verified',p.relative_to(SOURCE),flush=True)
    audit[str(p.relative_to(SOURCE))]=dict(size=target.stat().st_size,original_blob=p.resolve().name)
for name in ['text_encoder','text_encoder_2']:
    link=ROOT/'cache/flux-native'/name
    temp=link.with_name(name+'.local-link')
    temp.unlink(missing_ok=True);temp.symlink_to(DEST/name,target_is_directory=True);os.replace(temp,link)
(ROOT/'logs/text-encoders-local.json').write_text(json.dumps(audit,indent=2))
print('Text encoders now use unchanged local SSD copies',flush=True)
