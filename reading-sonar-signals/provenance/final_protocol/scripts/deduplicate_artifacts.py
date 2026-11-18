"""Hard-link byte-identical rendered inputs to reduce backup/storage overhead."""
from pathlib import Path
import hashlib
import os
import json

seen={};links=0;bytes_saved=0
for root in ['results','data','figures','viewer']:
    for path in sorted(Path(root).rglob('*.png')):
        if 'raw' in path.parts:continue
        digest=hashlib.sha256(path.read_bytes()).hexdigest()
        if digest in seen:
            original=seen[digest]
            if path.stat().st_ino==original.stat().st_ino:continue
            size=path.stat().st_size
            temporary=path.with_name(path.name+'.hardlink.tmp')
            if temporary.exists():temporary.unlink()
            os.link(original,temporary);os.replace(temporary,path)
            links+=1;bytes_saved+=size
        else:seen[digest]=path
print(json.dumps({'identical_png_hardlinks_created':links,'physical_bytes_saved':bytes_saved}))
