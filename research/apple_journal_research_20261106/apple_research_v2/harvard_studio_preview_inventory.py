"""Read official embedded camera thumbnails to screen the complete burst sequence."""
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
import json
import time

import cv2
import numpy as np
from PIL import Image,ImageDraw
import requests

from harvard_severe_source_review import INDEX, OUT, SOURCE, refpath


def extract(row):
    number=int(row["filename"].split("_")[-1].split(".")[0])
    p=OUT/"camera_thumbnails"/f"{number:03d}.jpg";p.parent.mkdir(exist_ok=True)
    src=refpath(row)
    prefix=src.read_bytes()[:65536] if src.is_file() else None
    if prefix is None:
        for attempt in range(4):
            try:
                with requests.get(SOURCE+str(row["id"]),stream=True,timeout=(6,12),
                                  headers={"Range":"bytes=0-65535","Accept-Encoding":"identity"}) as r:
                    r.raise_for_status()
                    if r.status_code==206:
                        assert r.headers["Content-Range"]==f"bytes 0-65535/{row['filesize']}"
                    prefix=next(r.iter_content(65536))
                    assert len(prefix)==65536
                    break
            except (requests.RequestException,AssertionError):
                time.sleep(.3*(attempt+1))
    if prefix is None:return dict(number=number,error="unavailable")
    start=prefix.find(b"\xff\xd8",2);end=prefix.find(b"\xff\xd9",start)
    if start<0 or end<0:return dict(number=number,error="embedded thumbnail missing")
    thumb=prefix[start:end+2];im=Image.open(BytesIO(thumb));exif=Image.open(BytesIO(prefix)).getexif()
    assert im.size==(160,120);p.write_bytes(thumb)
    return dict(number=number,time=str(exif.get(306)),camera=str(exif.get(272)),official_file_id=row['id'],
                full_md5=row["md5"],thumbnail_bytes=len(thumb))


def main():
    rows=[r["dataFile"] for r in json.loads(INDEX.read_text(encoding="utf-8"))["files"]if r["label"].startswith("Anjali_")]
    prior=OUT/"camera_thumbnails_audit.json"
    cache={r["number"]:r for r in json.loads(prior.read_text(encoding="utf-8")) if "error" not in r} if prior.exists() else {}
    def get_one(row):
        n=int(row["filename"].rsplit("_",1)[1].split(".")[0])
        if n in cache and (OUT/"camera_thumbnails"/f"{n:03d}.jpg").is_file():
            return cache[n]
        return extract(row)
    with ThreadPoolExecutor(9) as pool: summaries=list(pool.map(get_one,rows))
    (OUT/"camera_thumbnails_audit.json").write_text(json.dumps(summaries,indent=2)+"\n",encoding="utf-8")
    for start in range(0,len(rows),48):
        sheet=Image.new("RGB",(144*8,136*6),"white");draw=ImageDraw.Draw(sheet)
        for idx,r in enumerate(summaries[start:start+48]):
            n=r['number'];p=OUT/"camera_thumbnails"/f"{n:03d}.jpg"
            x=idx%8*144;y=idx//8*136
            if p.exists():
                obj=Image.open(p);rgb=np.asarray(obj.convert("RGB"))
                hsv=cv2.cvtColor(rgb,cv2.COLOR_RGB2HSV)
                mask=np.uint8((hsv[:,:,1]>42)&(hsv[:,:,2]>38))*255
                contours,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                    l,t,w,h=cv2.boundingRect(max(contours,key=cv2.contourArea));pad=3
                    obj=obj.crop((max(0,l-pad),max(0,t-pad),min(rgb.shape[1],l+w+pad),min(rgb.shape[0],t+h+pad)))
                obj=obj.resize((int(obj.width*min(132/obj.width,108/obj.height)),int(obj.height*min(132/obj.width,108/obj.height))),Image.Resampling.BICUBIC)
                sheet.paste(obj,(x,y))
            draw.text((x,y+111),f"{n:03d}  {r.get('time','')[-8:]}",fill="black")
        sheet.save(OUT/f"exif_{start:03d}.jpg",quality=92)
    print("embedded",sum("error" not in x for x in summaries),"/",len(rows))


if __name__=="__main__":main()
