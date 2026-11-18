"""Display likely repeated Woo views using surface or near-pixel evidence, for direct adjudication."""

import json
from pathlib import Path
from zipfile import ZipFile
from io import BytesIO
import numpy as np
from PIL import Image,ImageDraw
from class0_patch_refinement import read,atomic_json
from defect_public_intake import OUT
from defect_public_foreground import fruit_view


def run(snapshot):
    dst=Path(snapshot);rows={r["key"]:r for r in read(dst/"image_decisions.json")}
    edges=read(dst/"overlap_audit.json")["edges"]
    spec=next(r for r in read(dst/"receipt.json")["source_specs"] if r["project_id"].startswith("woo"))
    folder=dst/"pair_review";folder.mkdir(exist_ok=True);frames={};pairs=[]
    with ZipFile(spec["archive"]) as z:
        for key,row in rows.items():
            if row["project_id"]!=spec["project_id"]:continue
            with Image.open(BytesIO(z.read(row["member"]))) as src:
                im=src.convert("RGB");t=np.asarray(im.resize((112,64),Image.Resampling.BILINEAR)).astype(np.int16)
                v=fruit_view(im);w,h=im.size
                if v is not None:
                    f=np.array(v["xyxy"])*[w/256,h/176,w/256,h/176]
                elif row["labels"]:
                    xy=np.asarray([a["box"] for a in row["labels"]]);ctr=np.median((xy[:,:2]+xy[:,2:])/2,axis=0)
                    f=np.r_[ctr-[.18*w,.26*h],ctr+[.18*w,.26*h]]
                else:f=[.14*w,.14*h,.87*w,.86*h]
                crop=im.crop(tuple(f));crop.thumbnail((98,80));frames[key]=(t,crop)
        for e in edges:
            ka,kb=e["a_key"],e["b_key"]
            if ka not in frames or kb not in frames or e["rule"]=="same_source_lineage":continue
            aa,bb=frames[ka][0],frames[kb][0]
            d=np.abs(aa-bb).mean(2);worst=np.sort(d.reshape(8,8,7,16).mean((1,3)).ravel())[-2:].mean()
            m=e["foreground"]
            surface=bool(m and m["texture"]>=.50 and m["correlation"]>.74 and m["absolute_error"]<9 and m["mask_iou"]>.75)
            pixels=bool(d.mean()<3.3 and worst<15 and np.percentile(d,97)<13)
            if surface or pixels:pairs.append(dict(a=ka,b=kb,ss=surface,px=pixels))
    pairs.sort(key=lambda r:(not(r["ss"] and r["px"]),not r["ss"],r["a"],r["b"]))
    for i,p in enumerate(pairs):p["code"]=f"W{i+1:03}"
    atomic_json(folder/"candidates.json",pairs)
    for start in range(0,len(pairs),30):
        batch=pairs[start:start+30];canvas=Image.new("RGB",(1005,((len(batch)+4)//5)*100),"white");draw=ImageDraw.Draw(canvas)
        for i,p in enumerate(batch):
            left=i%5*201;top=i//5*100
            for h,k in enumerate(("a","b")):canvas.paste(frames[p[k]][1],(left+h*100,top))
            draw.text((left+3,top+81),p["code"]+"   "+("S" if p["ss"] else "")+("P" if p["px"] else ""),fill="black")
        canvas.save(folder/f"pairs_{start//30+1:02d}.jpg",quality=92)
    print(len(pairs),folder)


if __name__=="__main__":
    import sys
    run(sys.argv[1])
