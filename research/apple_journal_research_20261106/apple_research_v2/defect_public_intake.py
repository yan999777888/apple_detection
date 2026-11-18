"""Verify public COCO export ZIPs and build a cumulative training-only localized-positive crop manifest.

Run: defect_public_intake.py --spec path/to/source.json [--spec second.json]
A UTF-8 spec has project_id, version, record_url, license, license_evidence_url, archive,
archive_sha256, coco_members (exact ZIP paths) and category_map (exact COCO names -> SSDA 0/1).
The mapping and license are supplied after checking the publisher's evidence. The tool does not infer either.
Use active.json to find the complete current crop manifest. Never concatenate snapshots.
"""

import argparse
from collections import Counter,defaultdict
from contextlib import ExitStack
import hashlib
from io import BytesIO
import json
from pathlib import Path,PurePosixPath
import re
from zipfile import ZipFile
from urllib.parse import urlparse
import numpy as np
from PIL import Image,ImageDraw
from class0_patch_refinement import ROOT,configure,sha,read,atomic_json
from sources.additional_attribute_sources.audit_hdf_perceptual import OPS,NAMES,BK,UF,phash,dhash
from defect_public_foreground import fruit_view,visible_surface,lesion_context,copied_lesion

OUT=ROOT/"experiments/defect_public_intake"
SSDA=ROOT/"apple_research_v2/datasets/ssda_date_v2/manifest.json"


def digest(blob):return hashlib.sha256(blob).hexdigest()


def load_spec(path):
    data=json.loads(Path(path).read_text("utf-8-sig"))
    for key in ("project_id","version","license","record_url","license_evidence_url","archive_sha256"):
        if not isinstance(data.get(key),str) or not data[key].strip():raise ValueError(f"Missing {key}")
    if any(urlparse(data[k]).scheme not in ("https","http") or not urlparse(data[k]).netloc
           for k in ("record_url","license_evidence_url")):raise ValueError("Publisher and license evidence URLs required")
    if data["license"].strip().lower() in ("unknown","unspecified","unverified","none"):raise ValueError("Unverified license")
    if not re.fullmatch(r"[0-9a-f]{64}",data["archive_sha256"]):raise ValueError("Invalid archive hash")
    mapping=data.get("category_map")
    if not isinstance(mapping,dict) or not mapping or any(type(v) is not int or v not in (0,1) for v in mapping.values()):
        raise ValueError("Explicit approved category names and target ID 0/1 required")
    if not all(isinstance(k,str) and k.strip() for k in mapping):raise ValueError("Invalid category name")
    if not data.get("coco_members") or len(set(data["coco_members"]))!=len(data["coco_members"]):
        raise ValueError("Unique COCO ZIP members required")
    source=(Path(path).resolve().parent / data["archive"]).resolve()
    if source.drive.lower()!="f:":raise ValueError("Archive must reside on F")
    data["archive"]=str(source)
    return data


def signatures(im,oblique=False):
    t=im.convert("L").resize((48,48),Image.Resampling.LANCZOS)
    small=t.crop((3,3,45,45)).resize((48,48),Image.Resampling.LANCZOS)
    views=[t,small]
    if oblique:
        for degrees in (-24,-12,12,24):
            views.append(t.rotate(degrees,Image.Resampling.BICUBIC).crop((3,3,45,45)).resize((48,48),Image.Resampling.LANCZOS))
    return [[(phash(o),dhash(o)) for op in OPS for o in [frame if op is None else frame.transpose(op)]]
            for frame in views]


def fingerprint(im,key,filehash,partition,lineage=None):
    return dict(key=key,sha=filehash,part=partition,lineage=lineage,prints=signatures(im,partition=="external"))


def original_bank(path):
    rows=read(path)["samples"]
    counts=Counter(r["partition"] for r in rows)
    if counts!={"dev":113,"train":596,"quarantine":1} or len({r["image_id"] for r in rows})!=710:
        raise ValueError("SSDA source must include all 710 frames and quarantined frame")
    bank=[]
    for r in rows:
        p=Path(r["rgb_path"]);expected=r["rgb_sha256"]
        if sha(p)!=expected:raise ValueError(f"Changed SSDA frame {r['image_id']}")
        with Image.open(p) as im:bank.append(dict(fingerprint(im,r["image_id"],expected,r["partition"]),src=str(p)))
    return bank


def finite_box(values,w,h):
    b=np.asarray(values,dtype=float)
    if b.shape!=(4,) or not np.isfinite(b).all():raise ValueError("Invalid bbox")
    x,y,bw,bh=b
    if min(x,y)<-1e-5 or min(bw,bh)<=0 or x+bw>w+1e-4 or y+bh>h+1e-4:
        raise ValueError("Box empty or exceeds image")
    return [float(max(0,x)),float(max(0,y)),float(min(w,x+bw)),float(min(h,y+bh))]


def rle_counts(counts,w,h):
    # COCO differential signed 5-bit ASCII RLE. Validate all run lengths before bounds arithmetic.
    if isinstance(counts,str):
        decoded=[];j=0
        while j<len(counts):
            val=0;shift=0
            while True:
                if j>=len(counts) or shift>60:raise ValueError("Truncated/oversize RLE run")
                c=ord(counts[j])-48;j+=1
                if not 0<=c<64:raise ValueError("Invalid compressed RLE character")
                val|=(c&31)<<shift;shift+=5
                if not c&32:break
            if c&16:val-=1<<shift
            if len(decoded)>2:val+=decoded[-2]
            decoded.append(val)
        counts=decoded
    if not isinstance(counts,list) or any(type(v) is not int or v<0 for v in counts) or sum(counts)!=w*h:
        raise ValueError("RLE counts do not cover dimensions")
    return counts


def rle_bounds(counts,w,h):
    counts=rle_counts(counts,w,h)
    pos=0;bounds=[w,h,0,0]
    for k,n in enumerate(counts):
        if k%2 and n:
            x0,y0=divmod(pos,h);x1,y1=divmod(pos+n-1,h)
            bounds=[min(bounds[0],x0),min(bounds[1],y0 if x0==x1 else 0),
                    max(bounds[2],x1+1),max(bounds[3],y1+1 if x0==x1 else h)]
        pos+=n
    if bounds[2]<=bounds[0]:raise ValueError("Empty RLE")
    return bounds


def geometry(ann,w,h):
    box=finite_box(ann["bbox"],w,h)
    seg=ann.get("segmentation")
    if seg is None or seg==[]:return box,"box"
    if isinstance(seg,list):
        lo=[];hi=[]
        for poly in seg:
            pts=np.asarray(poly,float)
            if pts.ndim!=1 or pts.size<6 or pts.size%2 or not np.isfinite(pts).all():raise ValueError("Invalid polygon")
            pts=pts.reshape(-1,2);x,y=pts.T
            area=abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)))*.5
            if area<=0 or np.any(pts<-.0001) or np.any(pts>[w+.0001,h+.0001]):raise ValueError("Invalid polygon bounds/area")
            lo.append(pts.min(0));hi.append(pts.max(0))
        bounds=np.r_[np.min(lo,axis=0),np.max(hi,axis=0)]
        kind="polygon"
    elif isinstance(seg,dict) and seg.get("size")==[h,w]:
        bounds=rle_bounds(seg["counts"],w,h)
        kind="RLE"
    else:raise ValueError("Unsupported mask or mismatched mask dimensions")
    if min(bounds[0]-box[0],bounds[1]-box[1],box[2]-bounds[2],box[3]-bounds[3]) < -1.01:
        raise ValueError("Mask extends beyond bbox")
    return box,kind


def image_member(files,cocopath,name):
    item=PurePosixPath(name)
    if item.is_absolute() or ".." in item.parts or "\\" in name:raise ValueError("Unsafe COCO member name")
    opts={str(item),str(PurePosixPath(cocopath).parent/item)}
    found=opts&files
    if len(found)!=1:raise ValueError(f"Unresolved or ambiguous image {name}")
    return found.pop()


def lineage(project,member):
    base=PurePosixPath(member).stem
    # Roboflow original filename preceding .rf.<export hash> is meaningful across versions.
    m=re.fullmatch(r"(.+)\.rf\.[a-zA-Z0-9_-]+",base,re.I)
    if project=="woo-wwyog/orchard-apple-defect-segmentatio":
        capture=re.search(r"(FRUIT_\d{8}_\d{6}_\d+)_(?:LEFT|RIGHT)",base,re.I)
        if capture:return f"{project}/capture/{capture.group(1).casefold()}"
    return f"{project}/{m.group(1).casefold()}" if m else None


def native_split(member,image):
    for source in (member,image):
        for token in PurePosixPath(source).parts:
            if token.casefold() in ("train","valid","val","test"):
                return token.casefold()
    return "unspecified"


def scan(spec,z,external):
    files=set(z.namelist())
    if len(files)!=len(z.namelist()):raise ValueError("Ambiguous ZIP paths")
    approved=set(spec["category_map"]);seen=set()
    adjudicated={}
    if spec.get("pair_review"):
        reviewed=spec["pair_review"]
        if sha(reviewed["path"])!=reviewed["sha256"]:raise ValueError("Changed photo-pair review")
        for v in read(Path(reviewed["path"]))["pairs"]:
            adjudicated.setdefault(v["a"],{})[v["b"]]=v["relationship"]
            adjudicated.setdefault(v["b"],{})[v["a"]]=v["relationship"]
    hold={(v["coco_member"],str(v["image_id"]),str(v["annotation"])):v["reason"] for v in spec.get("exclude_annotations",[])}
    applied=set()
    for member in spec["coco_members"]:
        raw=json.loads(z.read(member).decode("utf-8-sig"))
        cats={r["id"]:r["name"] for r in raw["categories"]}
        if len(cats)!=len(raw["categories"]):raise ValueError("Duplicate COCO category ID")
        imageids={r["id"] for r in raw["images"]}
        if len(imageids)!=len(raw["images"]):raise ValueError("Duplicate COCO image ID")
        annids=set();group=defaultdict(list)
        for ann in raw["annotations"]:
            if ann["image_id"] not in imageids or ann["category_id"] not in cats or ann["id"] in annids:
                raise ValueError("Ambiguous annotation ID/image/category")
            annids.add(ann["id"]);group[ann["image_id"]].append(ann)
        for r in raw["images"]:
            key="|".join(map(str,(spec["project_id"],spec["version"],member,r["id"])))
            path=image_member(files,member,r["file_name"]);blob=z.read(path)
            with Image.open(BytesIO(blob)) as im:
                im.load();w,h=im.size
                fp=fingerprint(im,key,digest(blob),"external",lineage(spec["project_id"],path))
                fp["adjudicated"]=adjudicated.get(key,{})
                if spec["project_id"]=="woo-wwyog/orchard-apple-defect-segmentatio":
                    fp["fruit"]=fruit_view(im)
                defects=[];excluded=[]
                bad=(w,h)!=(r["width"],r["height"]) or im.getexif().get(274,1)!=1
                for ann in group[r["id"]]:
                    name=cats[ann["category_id"]]
                    if name not in approved:
                        excluded.append(dict(id=ann["id"],reason="outside approved categories",name=name));continue
                    seen.add(name)
                    tag=member,str(r["id"]),str(ann["id"])
                    if tag in hold:
                        excluded.append(dict(id=ann["id"],reason=hold[tag],name=name));applied.add(tag);continue
                    if ann.get("iscrowd",0):
                        excluded.append(dict(id=ann["id"],reason="crowd",name=name));continue
                    try:box,kind=geometry(ann,w,h)
                    except (ValueError,KeyError,TypeError,ImportError,OverflowError) as ex:
                        excluded.append(dict(id=ann["id"],reason=str(ex),name=name));continue
                    defects.append(dict(ann_id=ann["id"],name=name,target=spec["category_map"][name],box=box,
                                        geometry=kind,segmentation=ann.get("segmentation"),bbox_xywh=ann["bbox"]))
                external.append(dict(fp=fp,key=key,project_id=spec["project_id"],version=spec["version"],
                                     coco=member,image_id=r["id"],member=path,labels=defects,excluded=excluded,
                                     image_bad=bad,raw_split=native_split(member,path),
                                     license=spec["license"],archive_sha256=spec["archive_sha256"]))
    if not approved<=seen:raise ValueError(f"Approved name absent from annotations: {sorted(approved-seen)}")
    if applied!=hold.keys():raise ValueError("Unresolved instance-level review exclusions")


def compare(bank,base_count):
    tree=BK();exact=defaultdict(list);ancestry=defaultdict(list);uf=UF(len(bank));edges=[]
    for i,r in enumerate(bank):
        external=i>=base_count
        best={}
        if external:
            for j in exact[r["sha"]]:best[j]=("file_sha256",0,0,0,0,0)
            if r["lineage"]:
                for j in ancestry[r["lineage"]]:best.setdefault(j,("same_source_lineage",0,0,0,0,0))
            for qi,view in enumerate(r["prints"]):
                for orient,(ph,dh) in enumerate(view):
                    for (j,variant),hd in tree.query(ph,6):
                        other_dh=bank[j]["prints"][variant][0][1]
                        dd=(dh^other_dh).bit_count()
                        if dd<=9 and (j not in best or best[j][0]=="perceptual" and hd+dd<best[j][1]+best[j][2]):
                            best[j]=("perceptual",hd,dd,orient,qi,variant)
            for j,(rule,ph,dh,orient,qi,variant) in best.items():
                local=None;join=True
                if rule=="perceptual" and "fruit" in r and "fruit" in bank[j]:
                    local=visible_surface(r["fruit"],bank[j]["fruit"])
                    # Fixed-camera backgrounds hash together across clearly different apples. An indoor
                    # turntable image alone cannot bridge any groups unless fruit pixels corroborate it.
                    join=bool(local and local["relationship"]=="near_same_surface")
                    if join:rule="verified_near_surface"
                    judgment=r.get("adjudicated",{}).get(bank[j]["key"])
                    if judgment=="near_same_visible_view":
                        rule="reviewed_near_same_view";join=True
                    elif judgment=="too_similar_to_repeat_train":
                        rule="reviewed_similar_view";join=True
                if join:uf.join(i,j)
                edges.append(dict(a=i,b=j,rule=rule,phash_distance=ph,dhash_distance=dh,transform=NAMES[orient],
                                  linked=join,foreground=local,query_view=qi,reference_view=variant))
        exact[r["sha"]].append(i)
        if r["lineage"]:ancestry[r["lineage"]].append(i)
        for variant,v in enumerate(r["prints"][:2]):tree.add(v[0][0],(i,variant))
    groups=defaultdict(list)
    for i in range(len(bank)):groups[uf.find(i)].append(i)
    return groups,edges


def build(specs,originals,dst,ssda_manifest_sha256):
    """Construct a fresh snapshot from inputs. A caller publishes only after all outputs are complete."""
    crops=dst/"positive_crops";crops.mkdir(parents=True,exist_ok=True)
    external=[];kept=[];decisions=[];review=[];lesion_pool=defaultdict(list);lesion_rejections=[]
    with ExitStack() as stack:
        zips={}
        for spec in specs:
            if sha(spec["archive"])!=spec["archive_sha256"]:raise ValueError("Public ZIP checksum mismatch")
            z=stack.enter_context(ZipFile(spec["archive"]))
            zips[spec["archive_sha256"]]=z
            scan(spec,z,external)
        base=len(originals);bank=originals+[r["fp"] for r in external]
        groups,edges=compare(bank,base)
        relations=defaultdict(set)
        for edge in edges:
            a=edge["a"];b=edge["b"]
            if edge["linked"]:
                relations[a].add(edge["rule"]);relations[b].add(edge["rule"])
            edge.update(a_key=bank[a]["key"],b_key=bank[b]["key"],b_partition=bank[b]["part"])
        for members in groups.values():
            ext=[i for i in members if i>=base]
            if not ext:continue
            group_key=digest("\n".join(sorted(bank[i]["key"] for i in members)).encode())[:18]
            touches=any(i<base for i in members)
            ambiguous=any("perceptual" in relations[i] for i in members)
            has_holdout=any(external[i-base]["raw_split"] in ("valid","val","test") for i in ext)
            # Matching left/right camera timestamps identify shared acquisition, not identical pixels.
            # Native validation/test photos are discarded; one official train view may represent the capture.
            has_photo_evidence=any("file_sha256" in relations[i] or "perceptual" in relations[i] or "verified_near_surface" in relations[i] or "reviewed_near_same_view" in relations[i]
                                   or "same_source_lineage" in relations[i] and "/capture/" not in str(bank[i]["lineage"]) for i in ext)
            holdout=has_holdout and has_photo_evidence
            eligible=[i for i in ext if external[i-base]["raw_split"]=="train" and not external[i-base]["image_bad"] and external[i-base]["labels"]]
            chosen=None if touches or holdout or ambiguous or not eligible else min(eligible,key=lambda i:(-len(external[i-base]["labels"]),bank[i]["key"]))
            for i in ext:
                r=external[i-base];info={k:v for k,v in r.items() if k!="fp"}
                if touches:reason="SSDA overlap candidate"
                elif r["raw_split"]!="train":reason="author partition excluded from training"
                elif holdout:reason="overlaps external author validation/test group"
                elif ambiguous:reason="external near duplicate for review"
                elif r["image_bad"]:reason="image size/orientation mismatch"
                elif not r["labels"]:reason="no valid approved localized positive"
                elif i!=chosen:reason="same-source group; representative only"
                else:reason="export positive crops"
                decisions.append(dict(**info,sha256=bank[i]["sha"],group_id=group_key,decision=reason))
                if touches or ambiguous or holdout and len(ext)>1:review.append(dict(key=r["key"],group_id=group_key,reason=reason))
                if i!=chosen:continue
                blob=zips[r["archive_sha256"]].read(r["member"])
                with Image.open(BytesIO(blob)) as src:
                    im=src.convert("RGB")
                    full=dst/"source_images"/(bank[i]["sha"]+PurePosixPath(r["member"]).suffix.lower())
                    full.parent.mkdir(exist_ok=True);full.write_bytes(blob)
                    if sha(full)!=bank[i]["sha"]:raise ValueError("Source image copy mismatch")
                    for a in r["labels"]:
                        # Minimal lesion rectangle. All pixels outside that box are unlabelled for this task.
                        b=a["box"];rect=(int(np.floor(b[0])),int(np.floor(b[1])),int(np.ceil(b[2])),int(np.ceil(b[3])))
                        context=lesion_context(im,b)
                        duplicate=None
                        for previous,signature in lesion_pool[(r["project_id"],a["target"])]:
                            if previous["source_id"]==r["key"]:continue
                            metrics=copied_lesion(context,signature)
                            if metrics is not None:
                                duplicate=dict(source_id=r["key"],annotation=a["ann_id"],matched_source=previous["source_id"],
                                               matched_annotation=previous["annotation"],metrics=metrics)
                                break
                        if duplicate:
                            lesion_rejections.append(duplicate);continue
                        sample=digest(f"{r['key']}#{a['ann_id']}".encode())[:26]
                        target=crops/f"{sample}.png";im.crop(rect).save(target)
                        mask_name=None;polygon=a["segmentation"] if a["geometry"]=="polygon" else []
                        local=[(np.asarray(poly,float).reshape(-1,2)-np.array(rect[:2])).tolist() for poly in polygon]
                        if a["geometry"]!="box":
                            shape=(rect[2]-rect[0],rect[3]-rect[1])
                            if a["geometry"]=="polygon":
                                lesion=Image.new("L",shape);painter=ImageDraw.Draw(lesion)
                                for poly in local:painter.polygon([tuple(xy) for xy in poly],fill=255)
                            else:
                                seg=a["segmentation"];w,h=im.size
                                runs=rle_counts(seg["counts"],w,h);flat=np.zeros(w*h,np.uint8);pos=0
                                for k,n in enumerate(runs):
                                    if k%2:flat[pos:pos+n]=255
                                    pos+=n
                                lesion=Image.fromarray(flat.reshape(w,h).T).crop(rect)
                            maskfile=dst/"positive_masks"/f"{sample}.png";maskfile.parent.mkdir(exist_ok=True)
                            lesion.save(maskfile);mask_name=maskfile.relative_to(dst).as_posix()
                        kept.append(dict(source_id=r["key"],group_id=group_key,annotation=a["ann_id"],category_name=a["name"],
                                         target_class=a["target"],box_xyxy=b,raw_bbox_xywh=a["bbox_xywh"],crop_xyxy=list(rect),image_member=r["member"],
                                         image=full.relative_to(dst).as_posix(),full_image=full.relative_to(dst).as_posix(),
                                         mask=mask_name,polygon_xy=[np.asarray(p,float).reshape(-1,2).tolist() for p in polygon],
                                         polygon_in_crop=local,raw_split=r["raw_split"],geometry=a["geometry"],
                                         crop=target.relative_to(dst).as_posix(),crop_sha256=sha(target),
                                         image_sha256=bank[i]["sha"],mask_sha256=sha(dst/mask_name) if mask_name else None,
                                         role="training_only_local_positive",background_supervision=False,mask_supervision=bool(mask_name),
                                         license=r["license"],archive_sha256=r["archive_sha256"]))
                        lesion_pool[(r["project_id"],a["target"])].append((kept[-1],context))
        show={}
        review_keys={x["key"] for x in review}
        for e in edges:
            if not e["linked"] or e["a_key"] not in review_keys:continue
            priority=(e["b"]>=base,e["rule"]!="file_sha256",e["phash_distance"]+e["dhash_distance"])
            if e["a"] not in show or priority<show[e["a"]][0]:show[e["a"]]=(priority,e)
        def preview(i):
            if i<base and "src" in bank[i]:
                with Image.open(bank[i]["src"]) as im:return im.convert("RGB")
            if i>=base:
                obj=external[i-base]
                with Image.open(BytesIO(zips[obj["archive_sha256"]].read(obj["member"]))) as im:return im.convert("RGB")
            return Image.new("RGB",(100,100),"gray")
        display=[item[1] for item in show.values()]
        for offset in range(0,len(display),8):
            items=display[offset:offset+8];panel=Image.new("RGB",(500,len(items)*155),"white");draw=ImageDraw.Draw(panel)
            for row,e in enumerate(items):
                for j,i in enumerate((e["a"],e["b"])):
                    im=preview(i);im.thumbnail((238,128));panel.paste(im,(250*j,155*row))
                    draw.text((250*j,155*row+128),f"#{i} {bank[i]['part']} {e['rule']}",fill="black")
                e["review_sheet"]=f"review_{offset//8+1:04d}.jpg";e["review_row"]=row+1
            panel.save(dst/items[0]["review_sheet"],quality=88)
    atomic_json(dst/"positives.json",kept);atomic_json(dst/"image_decisions.json",decisions)
    atomic_json(dst/"overlap_audit.json",dict(SSDA_partitions=Counter(r["part"] for r in originals),edges=edges,review=review))
    atomic_json(dst/"repeated_lesion_contexts.json",lesion_rejections)
    counts=dict(images=len(external),positive_crops=len(kept),decisions=Counter(r["decision"] for r in decisions))
    atomic_json(dst/"receipt.json",dict(source_specs=specs,SSDA_manifest_sha256=ssda_manifest_sha256,
                                       code_sha256=sha(__file__),counts=counts,
                                       duplicate_screen="byte SHA, source lineage and D4 perceptual hashes on full/central views; arbitrary crops, oblique rotations or fruit identities may be unresolved",
                                       manifest_sha256=sha(dst/"positives.json")))
    return counts


def run(paths):
    from uuid import uuid4
    configure(OUT,"cpu");active=OUT/"active.json";lock=OUT/"intake.lock"
    # Prevent simultaneous imports racing on the list of cumulative, version-pinned sources.
    guard=lock.open("x");guard.close()
    try:
        prior=read(OUT/read(active)["snapshot"]/"receipt.json")["source_specs"] if active.exists() else []
        unique={}
        for spec in prior+[load_spec(p) for p in paths]:
            key=(spec["project_id"],spec["version"])
            if key in unique and unique[key]!=spec:
                prev=unique[key]
                # Audited additional classes may be admitted to the same pinned ZIP after checking publisher
                # semantics and actual contours, while keeping all previous mappings and provenance immutable.
                audited={"category_map","exclude_annotations","pair_review"}
                if ({k:v for k,v in spec.items() if k not in audited}!={k:v for k,v in prev.items() if k not in audited}
                    or not prev["category_map"].items()<=spec["category_map"].items()):
                    raise ValueError("Conflicting inputs/mapping for one version")
            unique[key]=spec
        specs=[unique[k] for k in sorted(unique)]
        if not specs:raise ValueError("No public source supplied")
        dst=OUT/"snapshots"/uuid4().hex
        counts=build(specs,original_bank(SSDA),dst,sha(SSDA))
        atomic_json(active,dict(snapshot=dst.relative_to(OUT).as_posix(),**counts))
        print(counts)
    finally:lock.unlink()


if __name__=="__main__":
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--spec",type=Path,action="append",required=True)
    run(ap.parse_args().spec)
