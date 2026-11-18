"""Prepare external evaluation photographs of apple rot with credited source records."""
import argparse
import hashlib
import json
from pathlib import Path
import time
from urllib.parse import quote

from PIL import Image, ImageDraw
import requests


ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/"datasets/commons_rot_stress_v1"
TITLES=[
    ("Brown Rot on Apple.jpg","Arthuc01","CC BY-SA 3.0"),
    ("Rotten apple.jpg","Kulmalukko","CC BY-SA 3.0"),
    ("Rotten apples.JPG","WikiBCS","CC0 1.0"),
    ("Rotting apple.jpg","Nicko Margolies (Nickomargolies)","CC BY 2.5"),
    ("Rotten Apple 01.jpg","Shams948","CC0 1.0"),
    ("Rotten apple under the tree.jpg","Sally V","CC BY-SA 4.0"),
    ("Monilia sp. on apple.jpg","Mnolf; source edition cropped by Lämpel","CC BY-SA 3.0"),
]


def fetch(entry):
    title,author,license=entry
    filename=title.replace(" ","_")
    md5=hashlib.md5(filename.encode()).hexdigest()
    media=f"https://upload.wikimedia.org/wikipedia/commons/{md5[0]}/{md5[:2]}/{quote(filename)}"
    path=BASE/"originals"/title
    path.parent.mkdir(parents=True,exist_ok=True)
    if not path.exists():
        for attempt in range(4):
            try:
                contact="AppleVisibleGradeResearch/1.0 (https://github.com/yan999777888/apple_detection) python-requests/"+requests.__version__
                r=requests.get(media,timeout=12,headers={"User-Agent":contact})
                r.raise_for_status()
                assert r.headers.get("Content-Type","").startswith("image/")
                path.write_bytes(r.content)
                break
            except requests.HTTPError as exc:
                if exc.response is not None and exc.response.status_code in (403,429):
                    break
                time.sleep(.5+attempt)
            except requests.RequestException:
                time.sleep(.5+attempt)
    return path,media


def download():
    s=Image.new("RGB",(245*3,225*3),"white");d=ImageDraw.Draw(s)
    result=[]
    for entry in TITLES:
        result.append(fetch(entry))
        time.sleep(1)
    for idx,(p,url) in enumerate(result):
        if p.exists():
            im=Image.open(p).convert("RGB")
            w,h=im.size
            im.thumbnail((239,196))
            x=idx%3*245;y=idx//3*225
            s.paste(im,(x,y));d.text((x,y+197),f"{idx} {p.stem[:19]} {w}x{h}",fill="black")
    s.save(BASE/"review.jpg",quality=95)
    print([(i,p.exists()) for i,(p,u) in enumerate(result)])


BOXES={
    0:(.23,.33,.65,.65), 1:(.08,.07,.84,.97), 2:(.36,.32,.56,.61),
    3:(.34,.20,1,.97), 4:(.245,.19,.70,.65), 5:(.18,.23,.70,.94),6:(.04,.02,.98,.97),
}
REASONS=[
    "On-tree target with widespread brown tissue, depressions and dense fungal structures.",
    "Extensive wrinkling and gross collapse around a deep cavity on the fallen fruit.",
    "Target fallen fruit covered by conspicuous pale fungal growth.",
    "On-branch fruit with severe collapse, tissue loss and extensive shrivelling.",
    "Broad brown sunken region covering much of a yellow apple.",
    "Extensive surface decay and a large cavity through the visible rotten tissue.",
    "Multiple rings of fungal structures extend over nearly the entire discolored surface.",
]


def official_receipt():
    dest=BASE/"wikimedia_receipt.json"
    if not dest.exists():
        q=dict(action="query",format="json",prop="imageinfo",iiprop="url|sha1|extmetadata",
               titles="|".join("File:"+name for name,author,lic in TITLES),
               iiextmetadatafilter="LicenseShortName|LicenseUrl|Artist|ImageDescription|DateTimeOriginal")
        r=requests.get("https://commons.wikimedia.org/w/api.php",params=q,timeout=20,headers={"User-Agent":"AppleVisibleGradeResearch/1.0 (https://github.com/yan999777888/apple_detection) python-requests/"+requests.__version__})
        r.raise_for_status()
        dest.write_text(json.dumps(r.json(),ensure_ascii=False,indent=2),encoding="utf-8")
    pages=json.loads(dest.read_text(encoding="utf-8"))["query"]["pages"].values()
    return {row["title"][5:]: row["imageinfo"][0] for row in pages}


def build(freeze):
    from prepare_grocery_normal_stress import sha, audit_overlap
    receipt=official_receipt()
    s=Image.new("RGB",(230*4,229*2),"white");draw=ImageDraw.Draw(s)
    out=BASE/"crops";out.mkdir(exist_ok=True)
    samples=[]
    for i, (name,author,license) in enumerate(TITLES):
        p=BASE/"originals"/name
        info=receipt[name]
        assert hashlib.sha1(p.read_bytes()).hexdigest()==info["sha1"]
        im=Image.open(p).convert("RGB")
        width,height=im.size
        box=[round(v*(height if j%2 else width)) for j,v in enumerate(BOXES[i])]
        crop=im.crop(box)
        dst=out/f"c{i:02d}.png"
        crop.save(dst);crop.thumbnail((220,200))
        x=i%4*230;y=i//4*229
        s.paste(crop,(x,y));draw.text((x,y+201),name[:22],fill="black")
        metadata=info["extmetadata"]
        source=metadata["LicenseShortName"]["value"]
        assert source==license or source=="CC0" and license=="CC0 1.0"
        samples.append(dict(path=dst.relative_to(ROOT).as_posix(), sha256=sha(dst), grade=2, split="test",
            source="Wikimedia_Commons_photographed_apple_decay",original_path=p.relative_to(ROOT).as_posix(),
            source_sha256=sha(p),official_sha1=info["sha1"],author=author,source_member=name,
            source_url=info["descriptionurl"],license=license,license_url=metadata["LicenseUrl"]["value"],
            derivative="Single target crop; same license as the attributed photograph.",crop_xyxy=box,
            group_id="Commons_"+author.split(";")[0],condition_id="Commons_"+name,visible_evidence=REASONS[i]))
    s.save(BASE/"review_crops.jpg",quality=95)
    if freeze:
        raw=ROOT/"apple_research_v2/sources/additional_attribute_sources/hdf_real_only/raw"
        history=[p for p in raw.glob("*/*") if p.suffix.lower() in {".jpg",".jpeg",".png"}]
        assert len(history)==5000
        n=audit_overlap(samples,history)
        result=dict(samples=samples,counts={"test":{"0":0,"1":0,"2":len(samples)}},
              reference_images_screened=n,source_provenance="Official per-photograph source, creator, license and SHA-1 verified. Independent photographers. Obvious decay/collapse confirmed on each single target. Observer quality proxies; no commercial-grade standard, internal quality or prevalence inferred.",
              evaluation="External test only; manual preselection for unmistakable severe visible defects, no scores accessed in selection or cropping.")
        (BASE/"manifest.json").write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
        notice=["# Source credit and crop licensing",""]
        for r in samples:
            notice.append(f"- {r['source_member']} — {r['author']}. [Original]({r['source_url']}). [{r['license']}]({r['license_url']}). Single-target crop distributed on the same terms.")
        (BASE/"ATTRIBUTION.md").write_text("\n".join(notice)+"\n",encoding="utf-8")


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("stage", choices=["download","review","freeze"],nargs="?",default="download")
    args=parser.parse_args()
    if args.stage=="download":
        download()
    else:
        build(args.stage=="freeze")
