"""A training-source-only check of whether visible injury texture transfers across apples."""
from collections import Counter
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "datasets/grade_pairs_sources_20261119/severity_evidence.json"
DEST = ROOT / "experiments/grade_pairs_20261119/severity_transfer"


def read_asset(relative, checksum, mono=False):
    data = (ROOT / relative).read_bytes()
    if hashlib.sha256(data).hexdigest() != checksum:
        raise ValueError(f"Changed asset {relative}")
    flag = cv2.IMREAD_GRAYSCALE if mono else cv2.IMREAD_COLOR
    return cv2.imdecode(np.frombuffer(data, np.uint8), flag)


def rectangle(shape, box):
    m = np.zeros(shape[:2], np.uint8)
    x, y, r, b = box
    m[y:b, x:r] = 1
    return m


def polygons(shape, rings):
    m = np.zeros(shape[:2], np.uint8)
    for coords in rings:
        cv2.fillPoly(m, [np.rint(np.asarray(coords).reshape(-1, 2)).astype(np.int32)], 1)
    return m


def face_width(mask):
    _, _, width, height = cv2.boundingRect((mask != 0).astype(np.uint8))
    return max(width, height)


def features(image, mask, width):
    scale = 320 / width
    sz = (round(image.shape[1] * scale), round(image.shape[0] * scale))
    im = cv2.resize(image, sz, interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
    m = cv2.resize(mask.astype(np.uint8), sz, interpolation=cv2.INTER_NEAREST) != 0
    safe = cv2.erode(m.astype(np.uint8), np.ones((9, 9), np.uint8)) != 0
    if not np.any(safe):
        raise ValueError("Marked region smaller than its measurement footprint")
    lab = cv2.cvtColor(im.astype(np.float32)/255., cv2.COLOR_BGR2LAB)
    grey = lab[:, :, 0]
    # Each feature uses only reviewed pixels (two 3x3 filters at most).
    sm = cv2.GaussianBlur(grey, (3, 3), .8)
    grad = np.hypot(cv2.Sobel(sm, cv2.CV_32F, 1, 0, ksize=3),
                    cv2.Sobel(sm, cv2.CV_32F, 0, 1, ksize=3))
    curvature = np.abs(cv2.Laplacian(sm, cv2.CV_32F, ksize=3))
    structure = np.concatenate([np.quantile(q[safe], [.5,.8,.95]) for q in (grad, curvature, np.abs(grey-sm))])
    colour = np.quantile(lab[safe], [.25,.5,.75], axis=0).reshape(-1)
    return np.r_[structure, colour]


def data():
    d = json.loads(SOURCE.read_text(encoding="utf8"))
    cases = []
    for r in d["fruitlife_visible_collapse"]:
        im = read_asset(r["path"], r["sha256"])
        fruit = read_asset(r["fruit_mask_path"], r["fruit_mask_sha256"], True)
        lesion = read_asset(r["region_mask_path"], r["region_mask_sha256"], True)
        n = r["paired_earlier_normal"]
        early = read_asset(n["path"], n["sha256"])
        # Earlier source crops are tight around an individual fruit. Fit the crop's
        # long edge; only the explicitly reviewed normal patch provides features.
        clean = features(early, rectangle(early.shape,n["normal_region_xyxy"]), max(early.shape[:2]))
        cases.append((r["specimen_id"], "FruitLife", r["path"], n["path"],
                      features(im, (fruit!=0)&(lesion!=0), face_width(fruit)), clean))
    for r in d["penn_large_author_sunburn"]:
        if not r["use_for_lesion_pixel_overlap"]:
            continue
        im = read_asset(r["path"], r["sha"])
        fruit = polygons(im.shape, r["fruit_polygons_native"])
        mask = polygons(im.shape, r["lesion_polygons_native"]) & fruit
        clean = rectangle(im.shape,r["normal_region_xyxy"])
        w = face_width(fruit)
        cases.append((r["group_id"], "PennState", r["path"], r["path"],
                      features(im, mask, w), features(im, clean, w)))
    return cases


def fit(training, channels):
    # One row per photographed lesion and one per unique normal reference.
    # Balance the two classes and weight every training capture group equally.
    x, y, weights = [], [], []
    for group in sorted({c[0] for c in training}):
        examples = [r for r in training if r[0] == group]
        for label, index, key in ((1,4,2), (0,5,3)):
            unique = {row[key]:row[index][:channels] for row in examples}
            for f in unique.values():
                x.append(f); y.append(label); weights.append(1/(2*len(unique)))
    x=np.asarray(x); y=np.asarray(y); weights=np.asarray(weights)
    mean=np.average(x, axis=0, weights=weights)
    scale=np.maximum(np.sqrt(np.average((x-mean)**2,axis=0,weights=weights)),1e-5)
    z=(x-mean)/scale
    way=np.average(z[y==1],axis=0,weights=weights[y==1])-np.average(z[y==0],axis=0,weights=weights[y==0])
    return lambda f: float(((f[:channels]-mean)/scale)@way)


def normal_controls(train):
    """Diagnostic operating point fixed at the training class-centroid midpoint."""
    score = fit(train, 9)
    manifest=ROOT/"datasets/grade_pairs_sources_20261119/regions.json"
    controls=[]
    for r in json.loads(manifest.read_text(encoding="utf8"))["regions"]:
        if r["split"]!="val":
            continue
        frame=read_asset(r["path"],r["image_sha256"])
        x,y,b,bot=r["parent_fruit_xyxy"]
        mask=rectangle(frame.shape,np.rint(r["source_image_xyxy"]).astype(int))
        try:
            obs=score(features(frame,mask,max(b-x,bot-y)))
        except ValueError as err:
            if str(err)!="Marked region smaller than its measurement footprint":
                raise
            obs=None
        controls.append(dict(id=r["id"],source=r["source"],label=r["target"],type=r["subtype"],score=obs))
    return dict(regions_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),
                scope="Development normal anatomy / limited defects. Fixed midpoint; positives indicate similarity to severe texture only.",
                rows=controls)


def main():
    cv2.setNumThreads(1)
    protocol = dict(data_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        units="Six advanced views of three distinct FruitLife apples; two nonrectangular sunburn regions from one Penn capture day.",
        conditions=["leave one physical fruit/capture day out", "leave entire source out"],
        features="9 geometry/texture summaries, compared with same features plus 9 Lab colour summaries; fixed linear difference of class means standardized exclusively on fit groups.",
        outcome="Does the lesion score exceed its matched normal patch? Positive ranking is not a whole-fruit severity decision. No AF or external panel inputs.")
    DEST.mkdir(parents=True, exist_ok=True)
    dest = DEST/"result.json"
    if dest.exists():
        raise FileExistsError(dest)
    (DEST/"protocol.json").write_text(json.dumps(protocol,indent=2)+"\n",encoding="utf8")
    rows=data()
    results={}
    for axis, idx in (("held_group",0),("held_source",1)):
        for mode,n in (("texture",9),("texture_colour",18)):
            measured=[]
            for key in sorted({r[idx] for r in rows}):
                training=[r for r in rows if r[idx]!=key]
                test=[r for r in rows if r[idx]==key]
                fn=fit(training,n)
                measured.extend(dict(group=r[0],source=r[1],image=r[2],normal_image=r[3],
                                     lesion_score=fn(r[4]),normal_score=fn(r[5])) for r in test)
            counts=Counter(r["source"] for r in measured if r["lesion_score"]>r["normal_score"])
            results[f"{axis}_{mode}"]=dict(correct_by_source=dict(counts), rows=measured)
    dest.write_text(json.dumps(dict(photos=len(rows),results=results),indent=2)+"\n",encoding="utf8")
    print({k:v["correct_by_source"] for k,v in results.items()})


if __name__ == "__main__":
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--development-controls",action="store_true")
    if parser.parse_args().development_controls:
        path=DEST/"development_controls.json"
        if path.exists():raise FileExistsError(path)
        report=normal_controls(data())
        path.write_text(json.dumps(report,indent=2)+"\n",encoding="utf8")
        print(Counter((r["label"], "small" if r["score"] is None else "flag" if r["score"]>0 else "clear") for r in report["rows"]))
    else:
        main()
