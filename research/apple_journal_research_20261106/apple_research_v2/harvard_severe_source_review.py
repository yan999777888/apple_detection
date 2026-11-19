"""Inspect the CC0 white-studio series for visibly severe fruit with matched normal sides."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import time

import cv2
import numpy as np
from PIL import Image,ImageDraw
import requests

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"datasets/harvard_grade_severe_20261119"
INDEX=ROOT/"datasets/apple_matched_quality_v1/KFQKXL.json"
OLD=ROOT/"datasets/apple_matched_quality_v1/harvard_preview"
SOURCE="https://dataverse.harvard.edu/api/access/datafile/"


def refpath(row):
    n=row["filename"]
    old=OLD/n
    return old if old.exists() else OUT/"originals"/n


def materialize(row):
    p=refpath(row)
    p.parent.mkdir(parents=True,exist_ok=True)
    if p.exists() and hashlib.md5(p.read_bytes()).hexdigest()==row["md5"]:
        return True
    for attempt in range(4):
        try:
            with requests.get(SOURCE+str(row["id"]),timeout=(7,35)) as r:
                r.raise_for_status()
                d=r.content
            if len(d)!=row["filesize"] or hashlib.md5(d).hexdigest()!=row["md5"]:
                continue
            p.write_bytes(d);return True
        except requests.RequestException:
            time.sleep(min(2,.3*(attempt+1)))
    return False


def crop_view(p):
    with Image.open(p) as raw:
        w,h=raw.size
        exif=raw.getexif()
        raw.draft("RGB",(1000,700))
        image=raw.convert("RGB")
    arr=np.asarray(image);height,width=arr.shape[:2]
    hsv=cv2.cvtColor(arr,cv2.COLOR_RGB2HSV)
    colorful=np.uint8((hsv[:,:,1]>42)&(hsv[:,:,2]>32))*255
    contours,_=cv2.findContours(cv2.morphologyEx(colorful,cv2.MORPH_CLOSE,np.ones((11,11),np.uint8)),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        x,y,cw,ch=0,0,width,height
    else:
        x,y,cw,ch=cv2.boundingRect(max(contours,key=cv2.contourArea))
    pad=round(.1*max(cw,ch))
    box=(max(0,x-pad),max(0,y-pad),min(width,x+cw+pad),min(height,y+ch+pad))
    rawbox=[round(v*(h/height if i%2 else w/width)) for i,v in enumerate(box)]
    return image.crop(box),rawbox,str(exif.get(306,"")),str(exif.get(272,""))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("stage",choices=["sample","all","gallery"])
    args=parser.parse_args()
    OUT.mkdir(exist_ok=True)
    d=json.loads(INDEX.read_text(encoding="utf-8"))
    assert d["license"]["rightsIdentifier"]=="CC0-1.0"
    series=[r["dataFile"] for r in d["files"] if r["label"].startswith("Anjali_")]
    assert len(series)==204
    if args.stage!="gallery":
        selected=series if args.stage=="all" else [r for i,r in enumerate(series) if i%6==0 or (OLD/r["filename"]).exists()]
        with ThreadPoolExecutor(6) as pool:
            done=list(pool.map(materialize,selected))
        print("verified",sum(done),"/",len(selected),flush=True)
    available=[r for r in series if refpath(r).is_file()]
    summary=[]
    for start in range(0,len(available),30):
        sheet=Image.new("RGB",(176*6,166*5),"white");draw=ImageDraw.Draw(sheet)
        for pos,row in enumerate(available[start:start+30]):
            p=refpath(row);view,box,date,camera=crop_view(p);view.thumbnail((169,142))
            x=pos%6*176;y=pos//6*166
            number=int(p.stem.rsplit("_",1)[1])
            sheet.paste(view,(x,y));draw.text((x,y+143),f"{number:03d}  {date[11:]}",fill="black")
            summary.append(dict(number=number,path=p.relative_to(ROOT).as_posix(),camera=camera,time=date,
               raw_frame_crop=box,source_file_id=row["id"],md5=row["md5"]))
        sheet.save(OUT/f"review_{start:03d}.jpg",quality=95)
    (OUT/"original_audit.json").write_text(json.dumps(summary,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")


if __name__=="__main__":
    main()
