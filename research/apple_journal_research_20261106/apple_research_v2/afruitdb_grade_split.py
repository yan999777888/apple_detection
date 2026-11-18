"""Freeze an apple-only image split by timestamp bursts and verified near duplicates."""
from collections import Counter, defaultdict
from datetime import datetime
import hashlib
import itertools
import json
from pathlib import Path
import re

import cv2
import numpy as np
from PIL import Image, ImageOps

from afruitdb_grade_source import OUT, ROOT, correct, dump, index, path_for


NAMES = ["good","medium","bad"]
SEED = 20261118
CAPTURE_GAP_SECONDS = 20
TARGET = np.array([.60,.18,.22])
OPS = [None,Image.Transpose.ROTATE_90,Image.Transpose.ROTATE_180,Image.Transpose.ROTATE_270,
       Image.Transpose.FLIP_LEFT_RIGHT,Image.Transpose.FLIP_TOP_BOTTOM,
       Image.Transpose.TRANSPOSE,Image.Transpose.TRANSVERSE]


class Components:
    def __init__(self,n):
        self.roots=list(range(n))

    def root(self,i):
        while self.roots[i]!=i:
            self.roots[i]=self.roots[self.roots[i]]
            i=self.roots[i]
        return i

    def join(self,i,j):
        a,b=self.root(i),self.root(j)
        self.roots[max(a,b)]=min(a,b)


def fingerprint(im):
    gray=cv2.cvtColor(im,cv2.COLOR_RGB2GRAY)
    small=cv2.resize(gray,(32,32),interpolation=cv2.INTER_AREA).astype(np.float32)
    coeff=cv2.dct(small)[:8,:8].ravel()
    ph=int.from_bytes(np.packbits(coeff>np.median(coeff[1:])).tobytes(),"big")
    small=cv2.resize(gray,(9,8),interpolation=cv2.INTER_AREA).astype(np.int16)
    dh=int.from_bytes(np.packbits(small[:,1:]>small[:,:-1]).tobytes(),"big")
    return ph,dh


def proxy_features(im):
    full=np.asarray(im)
    # The central feature verifies foreground agreement, reducing matches between
    # distinct apples on similar trays. Only same-size normalized views are compared.
    center=full[16:-16,16:-16]
    return (*fingerprint(full),*fingerprint(center))


def describe(src):
    path=path_for(src); blob=path.read_bytes()
    if not correct(blob,src):
        raise ValueError(f"CRC error for {path}")
    with Image.open(path) as reader:
        width,height=reader.size
        orientation=reader.getexif().get(274,1)
        im=ImageOps.exif_transpose(reader).convert("RGB")
        small=im.resize((80,80),Image.Resampling.LANCZOS)
    hh=hashlib.sha256(blob).hexdigest()
    ts=datetime.strptime(re.search(r"\d{8}_\d{6}",path.name)[0],"%Y%m%d_%H%M%S")
    row=dict(path=path.relative_to(ROOT).as_posix(),source_member=src["source_member"],
             grade=src["grade"],grade_name=NAMES[src["grade"]],sha256=hh,width=width,height=height,
             exif_orientation=orientation,source_crc32=src["crc32"],
             capture_time_filename=ts.isoformat(),capture_date_filename=str(ts.date()))
    transformed=[small if op is None else small.transpose(op) for op in OPS]
    return row,ts,[proxy_features(v) for v in transformed],np.stack([np.asarray(v) for v in transformed])


def verified_pair(a,b):
    middle=(slice(16,-16),slice(16,-16))
    x=a[middle].astype(np.float32);y=b[middle].astype(np.float32)
    mae=float(np.abs(x-y).mean()/255)
    if mae>.055: return False
    x-=x.mean(); y-=y.mean()
    corr=(x*y).sum()/max(float(np.sqrt((x*x).sum()*(y*y).sum())),1.0)
    return bool(corr>.955 and np.abs(a.astype(np.int16)-b.astype(np.int16)).mean()/255<.065)


def allocate(rows,groups):
    counts=np.array([np.bincount([rows[i]['grade'] for i in g],minlength=3) for g in groups])
    if len(groups)<3: raise ValueError("Need independent groups")
    totals=counts.sum(0)
    rng=np.random.default_rng(SEED)
    best_cost=np.inf;best=None
    for _ in range(256):
        mapping=rng.choice(3,size=(512,len(groups)),p=[.52,.23,.25])
        got=np.stack([(mapping==s).astype(int)@counts for s in range(3)],axis=1)
        support=np.stack([(mapping==s).astype(int)@(counts>0) for s in range(3)],axis=1)
        fraction=got/totals[None,None,:]
        loss=((fraction-TARGET[None,:,None])**2).sum((1,2))
        # Encourage multiple detached acquisition groups per class in both
        # evaluation partitions. Do not put any model-based metric in this search.
        loss+=.03*np.maximum(2-support,0).sum((1,2))
        loss+=5*(got<8).sum((1,2))
        ids=np.argmin(loss)
        if loss[ids]<best_cost:
            best_cost=float(loss[ids]);best=mapping[ids].copy()
    if (np.stack([counts[best==s].sum(0) for s in range(3)])<8).any():
        raise ValueError("Insufficient separated groups per grade")
    return best


def build():
    inv=index()
    desc=[describe(r) for r in sorted(inv["rows"],key=lambda r:r["source_member"])]
    rows,times,hashes,thumbs=map(list,zip(*desc))
    comps=Components(len(rows))
    temporal=[]; visual=[]
    order=sorted(range(len(rows)),key=lambda i:(times[i],i))
    for a,b in zip(order,order[1:]):
        if (times[b]-times[a]).total_seconds()<=CAPTURE_GAP_SECONDS:
            comps.join(a,b);temporal.append((a,b))
    for a,b in itertools.combinations(range(len(rows)),2):
        pb,db,cb,eb=hashes[b][0]
        for t,(pa,da,ca,ea) in enumerate(hashes[a]):
            if ((pb^pa).bit_count()<=8 and (db^da).bit_count()<=9
                and (cb^ca).bit_count()<=9 and (eb^ea).bit_count()<=10
                and verified_pair(thumbs[a][t],thumbs[b][0])):
                visual.append((a,b,t));comps.join(a,b);break
    exact=defaultdict(list)
    for i,r in enumerate(rows): exact[r["sha256"]].append(i)
    remove=set();conflicting=[]
    for ids in exact.values():
        for i in ids[1:]:comps.join(ids[0],i)
        if len({rows[i]["grade"] for i in ids})>1:
            conflicting.extend(ids);remove.update(ids)
        else:
            remove.update(ids[1:])
    groups=defaultdict(list)
    for i in range(len(rows)):
        if i not in remove: groups[comps.root(i)].append(i)
    labels=sorted(groups)
    inds=[groups[k] for k in labels]
    split=allocate(rows,inds)
    for g,k in enumerate(labels):
        for i in groups[k]:
            rows[i].update(split=("train","val","test")[split[g]],group_id=f"burst_{k:04d}")
    committed=[r for i,r in enumerate(rows) if i not in remove]
    for a,b in temporal+[(a,b) for a,b,_ in visual]:
        assert comps.root(a)==comps.root(b)
    assert len(set(r["sha256"] for r in committed))==len(committed)
    counts={s:dict(Counter(r["grade_name"] for r in committed if r["split"]==s)) for s in ("train","val","test")}
    group_counts={s:{k:len({r["group_id"] for r in committed if r["split"]==s and r["grade_name"]==k})
                     for k in NAMES} for s in counts}
    audit=dict(source="10.17632/bz65dz2pbj.1",archive_index_sha256=inv["central_directory_sha256"],
               raw_photos=len(rows),usable=len(committed),exact_duplicate_removals=len(remove),
               conflicting_exact_rows=len(conflicting),date_histograms={g:dict(Counter(r["capture_date_filename"]
                    for r in committed if r["grade_name"]==g)) for g in NAMES},
               seed=SEED,groups_total=len(groups),groups_per_grade=group_counts,images_per_grade=counts,
               correlated_burst_gap_seconds=CAPTURE_GAP_SECONDS,total_temporal_edges=len(temporal),
               verified_near_duplicate_edges=len(visual),near_duplicate_cross_split=0,
               physical_apple_id_available=False,
               caveat="Filename date confounds grade. Burst and conservative visual grouping do not certify independence of physical fruits or orchard domains. A grade is for the provided photo.",
               mapping={"1st grade":"good","2nd grade":"medium","3rd grade":"bad"},
               input_label="Author-labelled postharvest appearance, a proxy for the proposed A/B/C severity labels; no per-image physiological maturity truth.")
    dump(OUT/"split_audit.json",audit)
    dump(OUT/"proxy_groups.json",dict(temporal_pairs=temporal,verified_visual_pairs=visual,
                                       keys=[r["source_member"] for r in rows],removed_indices=sorted(remove)))
    manifest=dict(schema=1,source_doi=inv["doi"],source_license=inv["license"],grade_names=NAMES,
                  group_method="20-second temporary capture sequence spanning grades plus verified D4 near duplicates",
                  audit_file="datasets/afruitdb_apples_v1/split_audit.json",samples=committed)
    dump(OUT/"manifest.json",manifest)
    print(json.dumps(audit,ensure_ascii=True,indent=2),flush=True)


if __name__=="__main__":
    cv2.setNumThreads(1)
    build()
