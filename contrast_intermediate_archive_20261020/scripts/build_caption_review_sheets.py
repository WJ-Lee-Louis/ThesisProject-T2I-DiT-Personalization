"""Contact sheets for direct review; original images remain unchanged."""
from pathlib import Path
import json
from PIL import Image, ImageDraw, ImageFont, ImageOps
ROOT=Path(__file__).resolve().parents[1]
CAP=ROOT/'datasets/caption_monitoring/module_a_v2'
OUT=ROOT/'outputs/joint_pilot_v2/caption_review'
OUT.mkdir(parents=True,exist_ok=True)
refs=json.loads((CAP/'references.json').read_text(encoding='utf-8'))
font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',18)
for start in range(0,180,12):
    sheet=Image.new('RGB',(1200,1440),'white'); d=ImageDraw.Draw(sheet)
    for slot,c in enumerate(refs[start:start+12]):
        x=(slot%3)*400;y=(slot//3)*360
        with Image.open(ROOT/c['image_path']) as im:
            tile=ImageOps.contain(im.convert('RGB'),(390,315))
            sheet.paste(tile,(x+(400-tile.width)//2,y+30))
        d.text((x+5,y+5),f"{start+slot:03d} {c['identity_term']}",font=font,fill='black')
    sheet.save(OUT/f'sheet_{start//12:02d}.jpg',quality=92)
