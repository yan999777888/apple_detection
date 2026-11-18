"""Versioned grading inputs with HDF entirely excluded from fitting and selection."""
from collections import Counter
import hashlib
from zipfile import ZipFile
import cv2
import numpy as np

import afruit_grade_model as grade
import afruit_grade_train as io
import fruitlife_apple_source as fruitlife
import penn_surface_challenge as penn

ROOT = grade.ROOT
OUT = ROOT / "datasets/grade_guard_sources_v2"
HARVARD = ROOT / "datasets/harvard_grade_normal_anchors_v1/manifest.json"


def penn_rows():
    old = io.read(penn.OUT / "manifest.json")
    dates = set(old["excluded_complete_dates"])
    if dates & {r["timestamp"][:10] for r in old["samples"]}:
        raise ValueError("Historical Penn challenge dates overlap")
    if grade.file_sha(penn.ARCHIVE) != penn.ARCHIVE_SHA:
        raise ValueError("Penn source archive changed")
    records=[]
    with ZipFile(penn.ARCHIVE) as z:
        all_dates={penn.sample_time(p).date().isoformat() for p in z.namelist() if p.endswith("jpg")}
        for r in penn.candidates(z, all_dates-dates):
            raw=z.read(r["original_member"])
            image=cv2.imdecode(np.frombuffer(raw,np.uint8),cv2.IMREAD_COLOR)
            x,y,right,bottom=r["source_box"]
            payload=cv2.imencode(".png",image[y:bottom,x:right])[1].tobytes()
            sha=hashlib.sha256(payload).hexdigest()
            path=OUT/"penn"/(sha[:20]+".png")
            path.parent.mkdir(parents=True,exist_ok=True)
            if path.exists() and grade.file_sha(path)!=sha:raise ValueError("Crop bytes changed")
            if not path.exists():path.write_bytes(payload)
            day=r["timestamp"][:10]
            split={"2024-10-10":"val","2024-10-30":"test"}.get(day,"train")
            records.append(dict(**r,split=split,source="PSU_training_dates",path=path.relative_to(ROOT).as_posix(),
                                source_label="original apple and insect/scab polygons",train_target="small visible lesion proxy",
                                sha256=sha,source_sha256=hashlib.sha256(raw).hexdigest()))
    return records


def prepare(harvard=HARVARD):
    records = []
    fruit = io.read(fruitlife.OUT / "apple_manifest.json")["samples"]
    assignments = io.read(fruitlife.OUT / "physical_id_split.json")["assignment"]
    for r in fruit:
        stage = r["source_stage"]
        if assignments[r["specimen_id"]] != "train" or stage not in (0, 1, 3, 4):
            continue
        records.append(dict(path=r["path"], sha256=r["sha256"], source_sha256=r["source_sha256"],
                            source="FruitLife", group_id="FruitLife_" + r["specimen_id"], split="train",
                            grade=2 if stage >= 3 else stage, source_label="author_stage_" + str(stage),
                            train_target="author graded visual storage condition; 0 fresh,1 minor change,3/4 advanced degradation"))
    records.extend(penn_rows())
    anchors = io.read(harvard)["samples"]
    for r in anchors:
        if r["grade"] != 0 or r["split"] not in ("train", "val", "test"):
            raise ValueError("Unexpected healthy anchor protocol")
        row = dict(r)
        row.setdefault("source_label","author apple specimen plus independent visible-face screen")
        row.setdefault("train_target","visible clean face only")
        records.append(row)
    hashes=set()
    groups={}
    for r in records:
        if "hdf" in str(r["source"]).lower():
            raise ValueError("Historical HDF source cannot be used")
        key=r["source"],r["group_id"]
        if groups.setdefault(key,r["split"]) != r["split"]:
            raise ValueError("Cross-split source capture/fruit")
        if r["sha256"] in hashes or grade.file_sha(ROOT/r["path"]) != r["sha256"]:
            raise ValueError("Duplicate or modified source image")
        hashes.add(r["sha256"])
    src=Counter((r["source"],r["split"],r["grade"]) for r in records)
    counts={s:{str(c):sum(r["split"]==s and r["grade"]==c for r in records) for c in range(3)}
            for s in ("train","val","test")}
    doc=dict(version=2, scope="Pure external supervision appendage. AF and Woo continuation replay keep their prior published train/validation boundaries. Entire HDF archive excluded from fit and selection. FruitLife source-linked train IDs only, assigned before this experiment; archived external FruitLife fruits excluded even in selection. PSU uses defect training dates, with dates disjoint across rows and from prior Penn challenge.",
             counts=counts, source_summary=[dict(source=k[0],split=k[1],grade=k[2],n=v) for k,v in sorted(src.items())],
             input_sha256=dict(fruit=grade.file_sha(fruitlife.OUT/"apple_manifest.json"),fruit_split=grade.file_sha(fruitlife.OUT/"physical_id_split.json"),
                               psu=grade.file_sha(penn.ARCHIVE),psu_dates=grade.file_sha(penn.OUT/"manifest.json"),harvard=grade.file_sha(harvard)),
             provenance="Fresh/minor from storage are appearance proxies, anatomically distinct from a scratch or bruise. Source and crop bytes and available fruit/capture grouping retained.",
             samples=records)
    p=OUT/"manifest.json"
    if p.is_file() and io.read(p)!=doc:raise ValueError("Existing v2 manifest changed")
    if not p.is_file():io.atomic(p,doc)
    print(counts,flush=True)
    return p


if __name__=="__main__":
    prepare()

