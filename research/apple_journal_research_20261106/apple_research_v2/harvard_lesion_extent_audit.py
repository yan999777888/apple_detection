"""Measure the clearest isolated depressed region; retain it for audit only."""
import hashlib
import json

import cv2
import numpy as np
from PIL import Image,ImageDraw

from harvard_severe_source_review import OUT,ROOT,INDEX,crop_view,refpath


# Review coordinates on a 400x400 rendering of the full target crop (no mask
# filtering applied before review). Stay within visibly damaged brown tissue.
OUTLINES={
    175:[(265,189),(268,177),(276,168),(278,159),(287,155),(300,158),(309,163),
         (318,177),(321,189),(323,203),(317,219),(310,228),(300,234),
         (287,234),(278,231),(270,226),(265,217),(263,207)],
    187:[(49,167),(53,152),(60,143),(67,133),(78,130),(87,132),(97,139),
         (99,153),(95,172),(86,195),(76,207),(68,211),(60,208),(53,201),(48,189)],
}


def sample(row,number):
    p=refpath(row)
    data=p.read_bytes()
    assert hashlib.md5(data).hexdigest()==row["md5"]
    _,box,date,camera=crop_view(p)
    with Image.open(p) as raw:
        rgb=np.array(raw.convert("RGB").crop(box))
    height,width=rgb.shape[:2];hsv=cv2.cvtColor(rgb,cv2.COLOR_RGB2HSV)
    pts=np.int32([(round(x*width/400),round(y*height/400)) for x,y in OUTLINES[number]])
    lesion=np.zeros((height,width),np.uint8);cv2.fillPoly(lesion,[pts],1)
    estimates={};center_mask=None
    for thresh in (28,42,58):
        pigment=np.uint8((hsv[:,:,1]>thresh)&(hsv[:,:,2]>28))*255
        kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(21,21))
        contours,_=cv2.findContours(cv2.morphologyEx(pigment,cv2.MORPH_CLOSE,kernel),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        mask=np.zeros_like(lesion);cv2.drawContours(mask,[max(contours,key=cv2.contourArea)],-1,1,cv2.FILLED)
        a=int(mask.sum());b=int(np.count_nonzero(lesion&mask))
        estimates[str(thresh)]=dict(fruit_projected_pixels=a,confirmed_lesion_pixels=b,fraction=b/a)
        if thresh==42:center_mask=mask
    show=Image.fromarray(rgb).resize((400,400))
    smallmask=cv2.resize(center_mask,(400,400),interpolation=cv2.INTER_NEAREST)
    outer,_=cv2.findContours(smallmask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    drawn=np.array(show);cv2.drawContours(drawn,outer,-1,(35,168,213),2)
    cv2.polylines(drawn,[np.int32(OUTLINES[number])],True,(230,25,74),2)
    r=dict(source_id=row["id"],name=row["filename"],source_path=p.relative_to(ROOT).as_posix(),
           source_sha256=hashlib.sha256(data).hexdigest(),source_md5=row["md5"],capture_time=date,
           camera=camera,crop_xyxy=box,polygon_raw_xy=[[int(x+box[0]),int(y+box[1])] for x,y in pts],
           segmentation_sensitivity=estimates,observation="Localized brown depressed/wrinkled area; neighboring gray-brown flat spots excluded. Etiology and interior condition undetermined.",
           group_id="Canon_2024-04-20_same_marked_fruit_10h50",training_eligible=False)
    return r,Image.fromarray(drawn)


def main():
    rows={int(r["label"].rsplit("_",1)[1].split(".")[0]):r["dataFile"] for r in json.loads(INDEX.read_text(encoding="utf-8"))["files"] if r["label"].startswith("Anjali_")}
    records=[];sheet=Image.new("RGB",(820,444),"white");d=ImageDraw.Draw(sheet)
    for i,n in enumerate(OUTLINES):
        r,im=sample(rows[n],n);records.append(r);sheet.paste(im,(i*410,0))
        d.text((i*410+5,410),f"{n}  visible lesion / projected fruit = {r['segmentation_sensitivity']['42']['fraction']:.2%}",fill="black")
    sheet.save(OUT/"lesion_extent_review.jpg",quality=94)
    (OUT/"lesion_extent_diagnostic.json").write_text(json.dumps(records,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    print([(r['name'],r['segmentation_sensitivity']['42']['fraction']) for r in records])


if __name__=="__main__":main()
