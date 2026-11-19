"""Prepare original official Manalagi-v6 test fruit photos for a blind grading review."""
import hashlib
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests
from PIL import Image,ImageDraw

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"datasets/grade_pairs_blind_20261119"
API="https://data.mendeley.com/public-api/datasets/9zgkwwv9j8"
DISEASES=("Anthracnose","Black Pox","Healthy","Powdery Mildew")


def get(url,**kw):
    error=None
    for n in range(5):
        try:
            r=requests.get(url,timeout=(10,30),**kw)
            r.raise_for_status()
            return r
        except requests.RequestException as e:
            error=type(e).__name__
            time.sleep(.2*(n+1))
    raise OSError(f"Public Mendeley transfer failed: {error}")


def inventory():
    dst=OUT/"manalagi_official_test.json"
    if dst.exists():
        return json.loads(dst.read_text(encoding="utf-8"))
    folders=get(API+"/folders/6").json()
    ix={x['id']:x for x in folders}
    def parts(f):
        path=[f['name']]
        while f.get('parent_id') in ix:
            f=ix[f['parent_id']]
            path.insert(0,f['name'])
        return "/".join(path)
    selected={f['name']:(f['id'],parts(f)) for f in folders if f['name'] in DISEASES and "/Test_Set/" in parts(f) and "Original" in parts(f)}
    if len(selected)!=4:
        raise ValueError([parts(f) for f in folders if f['name']=='Healthy'])
    def one(pair):
        disease,(fid,parent)=pair
        url=API+"/files"
        files=get(url,params={"folder_id":fid,"version":6,"$start":0,"$limit":1000}).json()
        rows=[]
        for f in files:
            detail=f["content_details"]
            rows.append({"condition":disease,"folder":parent,"id":f["id"],"name":f["filename"],
                         "sha256":detail["sha256_hash"],"bytes":detail["size"],
                         "url":detail["download_url"].replace("https://api.mendeley.com", "https://data.mendeley.com")})
        return sorted(rows,key=lambda r:r["name"])
    blocks=ThreadPoolExecutor(4).map(one,sorted(selected.items()))
    rows=[r for b in blocks for r in b]
    result={"doi":"10.17632/9zgkwwv9j8.6","license":"CC BY 4.0","samples":rows}
    dst.write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
    return result


def original(row):
    return OUT/"Manalagi_original_test"/row["condition"]/row["name"]


def fetch(row):
    path=original(row)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and hashlib.sha256(path.read_bytes()).hexdigest()==row["sha256"]:
        return True
    try:
        data=get(row["url"]).content
        assert len(data)==row["bytes"] and hashlib.sha256(data).hexdigest()==row["sha256"]
        path.write_bytes(data);return True
    except OSError:
        return False


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    rows=inventory()["samples"]
    good=list(ThreadPoolExecutor(7).map(fetch,rows))
    for start in range(0,len(rows),20):
        panel=Image.new("RGB",(180*5,191*4),"white");draw=ImageDraw.Draw(panel)
        for idx in range(start,min(start+20,len(rows))):
            if not good[idx]: continue
            row=rows[idx];im=Image.open(original(row)).convert("RGB")
            im.thumbnail((176,166));x=(idx-start)%5*180;y=(idx-start)//5*191
            panel.paste(im,(x,y));draw.text((x,y+166),f"{idx:03d} {row['condition']} | {row['name'][:12]}",fill='black')
        panel.save(OUT/f"manalagi_{start:03d}.jpg",quality=96)
    print(len(rows),sum(good))


if __name__=="__main__":
    main()
