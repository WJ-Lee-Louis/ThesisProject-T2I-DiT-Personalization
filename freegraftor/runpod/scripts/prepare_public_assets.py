from pathlib import Path
import hashlib,urllib.request
from huggingface_hub import snapshot_download

root=Path('/workspace/freegraftor')
snapshot_download('IDEA-Research/grounding-dino-tiny',revision='a2bb814dd30d776dcf7e30523b00659f4f141c71',allow_patterns=['*.json','*.safetensors','*.txt'],max_workers=4)
p=root/'cache/sam_vit_h_4b8939.pth'
if not p.exists():
    temporary=p.with_suffix('.pth.tmp')
    urllib.request.urlretrieve('https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth',temporary)
    temporary.replace(p)
h=hashlib.sha256()
with p.open('rb') as f:
    for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
assert h.hexdigest()=='a7bf3b02f3ebf1267aba913ff637d9a2d5c33d3173bb679e46d9f338c26f262e'
from modelscope import snapshot_download as ms_download
print(ms_download('iic/cv_fft_inpainting_lama',revision='v1.0.0'))
