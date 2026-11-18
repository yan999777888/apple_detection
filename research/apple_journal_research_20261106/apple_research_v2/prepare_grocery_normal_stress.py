"""Prepare a small external, grade-proxy stress panel from real smartphone fruit images."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from collections import Counter
import hashlib
import json
from pathlib import Path
import time

import cv2
import numpy as np
from PIL import Image, ImageDraw
import requests


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "datasets/grocery_normal_review_v1"
URL = "https://raw.githubusercontent.com/marcusklasson/GroceryStoreDataset/master/"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fetch(member, destination):
    if destination.exists():
        return True
    destination.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(7):
        try:
            response = requests.get(URL + member, timeout=12)
            response.raise_for_status()
            destination.write_bytes(response.content)
            return True
        except requests.RequestException:
            time.sleep(0.2 + 0.2*attempt)
    return False


def read_source():
    for src in ["README.md", "LICENSE", "dataset/val.txt", "dataset/test.txt", "dataset/train.txt"]:
        assert fetch(src, BASE/Path(src).name)
    entries = []
    for split, count in [("val", 3), ("test", 7), ("train", 5)]:
        paths = [r.split(",")[0] for r in (BASE/f"{split}.txt").read_text().splitlines() if "/Fruit/Apple/" in r]
        varieties = sorted({x.split("/")[3] for x in paths})
        for variety in varieties:
            pool = [x for x in paths if x.split("/")[3] == variety]
            for index in np.linspace(0, len(pool)-1, count, dtype=int):
                entries.append(pool[index])
    return entries


def local_path(row):
    return BASE / "originals" / row


def download():
    entries = read_source()
    okay = list(ThreadPoolExecutor(8).map(lambda row: fetch("dataset/"+row, local_path(row)), entries))
    for group in range(0, len(entries), 12):
        sheet = Image.new("RGB", (232*4, 206*3), "white")
        text = ImageDraw.Draw(sheet)
        for ag in range(group, min(group+12, len(entries))):
            if not okay[ag]:
                continue
            row = entries[ag]
            p=local_path(row)
            with Image.open(p) as raw:
                im = raw.convert("RGB")
                w,h=im.size
                im.thumbnail((228,180))
            x=(ag-group)%4*232
            y=(ag-group)//4*206
            sheet.paste(im,(x,y));text.text((x,y+180),f"{ag:02d} {Path(row).stem} {w}x{h}",fill="black")
        sheet.save(BASE/f"inspect_{group//12}.jpg",quality=93)
    print(f"{sum(okay)} of {len(entries)} downloaded")


# Boxes were specified directly on the source photos, before any scoring.
# Coordinates use a square 348x348 reference grid for the complete photograph.
REVIEWED_BOXES = {
    0: (126, 151, 250, 282), 4: (89, 61, 298, 276), 5: (101, 197, 202, 292),
    7: (57, 218, 174, 342), 9: (106, 61, 182, 157),
    14: (85, 204, 163, 271), 17: (109, 210, 188, 296),
    18: (138, 131, 281, 278), 30: (88, 217, 219, 342),
    31: (27, 80, 155, 207), 37: (137, 195, 220, 277),
    39: (67, 54, 189, 174), 44: (179, 87, 269, 168),
    47: (64, 95, 269, 309), 49: (115, 111, 294, 300),
    53: (83, 109, 277, 310), 56: (140, 132, 305, 307),
    57: (78, 129, 275, 318), 61: (109, 121, 286, 298),
    62: (91, 100, 271, 302),
}
REJECTED_AFTER_CROP_REVIEW = {
    7:"localized blur across calyx", 9:"small mottled fruit, neighboring fruit crowds the crop",
    37:"small grey/brown area on the upper skin, uncertain",44:"small ambiguous fruit",
    57:"elongated dark line near the calyx, uncertain normal stem versus skin defect",
}


def audit_overlap(samples, extra_paths=()):
    from prepare_harvard_normal_anchors import fingerprints, HDF

    paths=set(HDF.glob("*.jpg"))
    paths.update(extra_paths)
    for rel in ["grade_guard_sources_v2","harvard_grade_normal_anchors_v1","apple_matched_fruitlife_woo_v1",
                "apple_matched_quality_v1","afruitdb_apples_v1","penn_surface_challenge_v1"]:
        m=ROOT/"datasets"/rel/"manifest.json"
        manifest=json.loads(m.read_text(encoding="utf-8"))
        for item in manifest.get("samples",[]):
            if isinstance(item,dict) and item.get("path") and (ROOT/item["path"]).is_file():
                paths.add(ROOT/item["path"])
    samples_sorted=sorted(paths)
    hashes={sha(p) for p in samples_sorted}
    features=[]
    for p in samples_sorted:
        with Image.open(p) as im:
            features.append(fingerprints(im)[0])
    features=np.stack(features).astype(np.int16)
    for row in samples:
        assert row["sha256"] not in hashes and row["source_sha256"] not in hashes
        with Image.open(ROOT/row["path"]) as im:
            dist=min(float(np.abs(features-v).mean(axis=(1,2,3)).min()) for v in fingerprints(im))
        assert dist>12, (row["source_member"],dist)
        row["reference_minimum_RGB_MAE_32_D4"]=round(dist,3)
    return len(samples_sorted)


def build(freeze=False):
    entries = read_source()
    target = BASE / "crops"
    target.mkdir(parents=True, exist_ok=True)
    sheet = Image.new("RGB",(170*5, 178*4),"white")
    drawer = ImageDraw.Draw(sheet)
    samples = []
    for index, box in REVIEWED_BOXES.items():
        if index in REJECTED_AFTER_CROP_REVIEW:
            continue
        pos=len(samples)
        row=entries[index]
        src=local_path(row)
        with Image.open(src) as im:
            im=im.convert("RGB")
            width,height=im.size
            box_px=[round(v*(height if k%2 else width)/348) for k,v in enumerate(box)]
            view=im.crop(box_px)
        p=target/f"g{index:02d}.png"
        view.save(p)
        v=view.copy()
        v.thumbnail((165,151))
        x=pos%5*170;y=pos//5*178
        sheet.paste(v,(x,y));drawer.text((x,y+153),f"{index:02d} {Path(row).stem}",fill="black")
        variety=row.split("/")[3]
        samples.append(dict(path=p.relative_to(ROOT).as_posix(), sha256=sha(p), grade=0, split="test",
                            source="GroceryStoreDataset_visible_fruit_normal_proxy", source_split=row.split("/")[0],
                            source_member=row, source_sha256=sha(src), original_path=src.relative_to(ROOT).as_posix(),
                            crop_xyxy=box_px, variety=variety, group_id=f"GroceryStoreDataset_{variety}", condition_id=f"GroceryStoreDataset_{row.split('/')[0]}_{variety}"))
    sheet.save(BASE/"accepted_crop_contact.jpg",quality=95)
    if freeze:
        reference_count=audit_overlap(samples)
        # A single source/variety grouping prevents overclaiming independent fruits.
        # No model output is accessed at any point during screening or freezing.
        manifest=dict(samples=samples,license="MIT (repository). Author README explicitly permits application to models.",
            root_url="https://github.com/marcusklasson/GroceryStoreDataset",
            source_provenance="Author smartphone photographs of retail fruit. The training, val, and test source designations are preserved. All selected photographs are external-test-only for this quality study.",
            grade_basis="Image review for a visibly intact target face, no author cosmetic grade or expert assessment. Only this visible face was examined. Differing views or nearby photographs may share physical fruits; source group identifiers are coarse variety groups.",
            excluded="Downloaded iconic images, ambiguous damage, heavily blurred targets, largely occluded targets. Neighboring fruit can be present near a crop boundary.",
            source_splits=dict(Counter(r["source_split"] for r in samples)),
            reference_images_screened=reference_count,
            overlap_rule="Original and crop SHA-256 and crop RGB mean absolute differences at 32x32 over D4 orientation transforms vs reachable quality manifests and entire HDF fresh. Views of unrecorded same fruits can still evade image screening.",
            counts={"test":{"0":len(samples),"1":0,"2":0}})
        (BASE/"manifest.json").write_text(json.dumps(manifest,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("stage",choices=["download","review","freeze"],nargs="?",default="download")
    args=parser.parse_args()
    if args.stage=="download":
        download()
    else:
        build(args.stage=="freeze")
