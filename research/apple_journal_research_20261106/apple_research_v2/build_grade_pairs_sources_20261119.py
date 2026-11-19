"""Training-only quality-pair and spot provenance audit, grounded in original source coordinates."""
from collections import Counter,defaultdict
from pathlib import Path
from zipfile import ZipFile
import hashlib
import json
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageOps

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"datasets/grade_pairs_sources_20261119"
AF=ROOT/"datasets/afruitdb_apples_v1/manifest.json"
SPOTS=ROOT/"datasets/afruit_train_spots_v1/manifest.json"
WOO=ROOT/"datasets/apple_matched_quality_v1/manifest.json"
WOOZIP=next((ROOT/"datasets/public_defect_intake").glob("orchard-apple*.zip"))
PSUZIP=next((ROOT/"datasets/public_defect_intake").glob("Fruit_defects*.zip"))
GUARD=ROOT/"datasets/grade_guard_sources_v2/manifest.json"
SSA=ROOT/"apple_research_v2/datasets/ssda_date_v2/manifest.json"
BLIND=next((ROOT/"experiments/ssda_blind_defects_v1").rglob("34处最终核对/cases_audit.json"))
ALIGN=ROOT/"experiments/afruit_grade_crop_alignment_v1"
SOURCE=ROOT/"datasets/penn_surface_challenge_v1/manifest.json"


def read(p):return json.loads(Path(p).read_bytes())
def save(name,obj):
    p=OUT/name;p.parent.mkdir(exist_ok=True,parents=True)
    p.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+"\n",encoding="utf8")
def digest(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def listed(im):
    return Path(im).relative_to(ROOT).as_posix()
def pixels(p):
    rgb=cv2.imdecode(np.fromfile(ROOT/p,np.uint8),cv2.IMREAD_COLOR)
    if rgb is None:raise ValueError(p)
    return rgb
def hashid(*parts):return hashlib.sha256(json.dumps(parts,sort_keys=True).encode()).hexdigest()[:20]


def region(path,sha,hw,xyxy,fruit,source,split,kind,sub,group,record,trace,**extra):
    h,w=map(int,hw)
    x,y,x2,y2=map(float,xyxy)
    if not (0<=x<x2<=w and 0<=y<y2<=h):raise ValueError((path,xyxy,hw))
    return dict(id=hashid(path,xyxy,sub,record),source=source,split=split,path=path,image_sha256=sha,
                image_hw=[h,w],source_image_xyxy=[round(x,3),round(y,3),round(x2,3),round(y2,3)],
                parent_fruit_xyxy=fruit,target=kind,subtype=sub,group_id=group,
                source_record=record,trace=trace,**extra)


def real_af():
    rows=read(AF)["samples"]; regions=[]
    for r in read(SPOTS)["samples"]:
        if r["split"]!="train":raise ValueError("AF spot training only")
        h,w=r["source_hw"];dh,dw=r["view_hw"]
        a=r.get("crop_xyxy_1280");a=[round(x*k) for x,k in zip(a,(w/dw,h/dh,w/dw,h/dh))] if a else None
        for tag,kind in (("positives_xyxy","surface_defect"),("negatives_xyxy","confirmed_normal")):
            for j,box in enumerate(r[tag]):
                regions.append(region(r["path"],r["sha256"],(h,w),box,a,"AFruitDB","train",kind,
                                      "reviewed_surface_mark" if tag.startswith("positive") else "reviewed_normal_structure",
                                      r["group_id"],r["path"]+f"#{tag}:{j}","individually reviewed and localized from registered training crop"))
    return regions


def crop_box(box,offset,hw):
    x,y,w,h=box;left,top=offset;hh,ww=hw
    return [max(0,x-left),max(0,y-top),min(ww,x+w-left),min(hh,y+h-top)]


def real_woo():
    rows=read(WOO)["samples"];output=[]
    with ZipFile(WOOZIP) as z:
        coco=json.loads(z.read("train/_annotations.coco.json"))
    names={"train/"+r["file_name"]:r["id"] for r in coco["images"]}
    grouped=defaultdict(list)
    for a in coco["annotations"]:grouped[a["image_id"]].append(a)
    for r in rows:
        if r["split"]!="train":raise ValueError("Woo training only")
        if not r["grade"]:continue
        h,w=pixels(r["path"]).shape[:2];x,y=r["source_box"][:2]
        for a in grouped[names[r["source_member"]]]:
            if a["category_id"]!=3:raise ValueError("Unexpected Woo lesion severity")
            box=crop_box(a["bbox"],(x,y),(h,w))
            output.append(region(r["path"],r["sha256"],(h,w),box,[0,0,w,h],"Woo","train","surface_defect","author_scratch",r["group_id"],r["source_member"]+f"#ann{a['id']}","Reviewed shallow scratch; original author polygon and crop alignment",condition=r["condition_id"],capture=r["capture_key"],source_photo_sha256=r["source_sha256"]))
    return output


def real_penn():
    with ZipFile(PSUZIP) as z:
        coco=json.loads(z.read("train/_annotations.coco.json"))
    ann={r["id"]:r for r in coco["annotations"]};out=[]
    for r in read(GUARD)["samples"]:
        if r["source"]!="PSU_training_dates" or r["split"] not in ("train","val"):continue
        h,w=pixels(r["path"]).shape[:2];x,y=r["source_box"][:2]
        for lesion in r["original_lesions"]:
            a=ann[lesion["id"]];box=crop_box(a["bbox"],(x,y),(h,w))
            out.append(region(r["path"],r["sha256"],(h,w),box,[0,0,w,h],"PennState",r["split"],"surface_defect",lesion["name"],r["group_id"],
                              r["original_member"]+f"#ann{a['id']}","Curated fruit owner, source lesion polygon, visually confirmed limited damage",marked_fraction=float(r["marked_fraction"]),source_photo_sha256=r["source_sha256"]))
    return out


def real_ssda():
    img={r["image_id"]:r for r in read(SSA)["samples"] if r["partition"]=="train"};out=[]
    for r in read(BLIND)["rows"]:
        if r["manual"] not in ("scratch","insect"):continue
        v=img[r["image_id"]];p=Path(v["rgb_path"]);h,w=pixels(listed(p)).shape[:2]
        key=v["same_second_group"]
        out.append(region(listed(p),digest(p),(h,w),r["box_xyxy"],None,"SSDA","train","surface_defect",r["manual"],"SSDA_"+key,
                          r["id"],"Operator blind 34-site diagnosis, including explicitly sourced final conversational supplement. Severity per visible fruit undocumented."))
    return out


def board(name,records,cols=5,rows=4):
    for st in range(0,len(records),cols*rows):
        group=records[st:st+cols*rows]
        canvas=Image.new("RGB",(cols*215,rows*218),"white");draw=ImageDraw.Draw(canvas)
        for k,r in enumerate(group):
            xx,yy=(k%cols)*215,(k//cols)*218
            im=Image.fromarray(cv2.cvtColor(pixels(r["path"]),cv2.COLOR_BGR2RGB))
            if r.get("view"):im=im.crop(r["view"])
            im=im.resize((210,190),Image.Resampling.LANCZOS)
            canvas.paste(im,(xx,yy));draw.text((xx+2,yy+191),str(r["id"])+" "+r.get("note","")[:30],fill="black")
        dest=OUT/f"review/{name}_{st:03d}.jpg";dest.parent.mkdir(parents=True,exist_ok=True);canvas.save(dest,quality=91)


def collect_review():
    clean=sorted([r for r in read(WOO)["samples"] if r["grade"]==0],key=lambda r:r["review_index"])
    a=[dict(id=i,path=r["path"],note=f"c{r['review_index']}",sha=r["sha256"],group=r["group_id"],condition=r["condition_id"],capture=r["capture_key"]) for i,r in enumerate(clean)]
    afrows=[r for r in read(AF)["samples"] if r["split"]=="val"]
    footprints={r["sha256"]:r for r in read(ALIGN/"val_regions.json")}
    b=[]
    for r in afrows:
        if r["grade"]>1:continue
        k=footprints[r["sha256"]];c=k["crop"]
        if c is None:continue
        h,w=k["hw"];hh,ww=r["height"],r["width"]
        view=[round(x*t) for x,t in zip(c,(ww/w,hh/h,ww/w,hh/h))]
        b.append(dict(id=len(b),path=r["path"],note=f"AF:{r['grade']}",sha=r["sha256"],group=r["group_id"],view=view))
    board("woo",a);board("af_dev",b)
    return a,b


def initial():
    a,b=collect_review()
    save("review/index.json",dict(woo=a,af_dev=b))
    rr=real_af()+real_woo()+real_penn()+real_ssda()
    save("core_regions.json",dict(schema="source_xyxy_v1",regions=rr))
    print(Counter((r["source"],r["split"],r["target"]) for r in rr))


def expanded_woo():
    from prepare_matched_woo_quality import descriptor
    src=ROOT/"experiments/defect_public_intake/snapshots/75e57d602dc84195b8d705105909e37c"
    decisions=[r for r in read(src/"image_decisions.json") if r["project_id"].startswith("woo-")]
    prior={r["member"]:r for r in decisions}
    def stamp(member):return Path(member).name.split("_jpg.rf.")[0].rsplit("_",1)[0]
    split_by_timestamp=defaultdict(set)
    for r in decisions:split_by_timestamp[stamp(r["member"])].add(r["raw_split"])
    excluded_groups={r["group_id"] for r in decisions if r["raw_split"]!="train"}
    eligible=[];frames={}
    for p in read(src/"positives.json"):
        if not p["source_id"].startswith("woo-") or p["category_name"]!="scratch":continue
        file=p["image_member"];q=prior[file]
        if split_by_timestamp[stamp(file)]!={"train"} or q["group_id"] in excluded_groups or q["decision"]!="export positive crops":continue
        if file not in frames:
            path=src/p["full_image"];path=listed(path);im=pixels(path)
            if digest(ROOT/path)!=p["image_sha256"]:raise ValueError("Woo source signature changed")
            fruit=descriptor(im)
            frames[file]=dict(id=len(frames),path=path,sha=p["image_sha256"],hw=im.shape[:2],view=list(fruit[2]) if fruit else None,
                              member=file,group="Woo_20260814_session",condition=file.split("_jpg.rf.")[0].rsplit("_",1)[1],capture=stamp(file))
        r=frames[file]
        eligible.append(region(r["path"],r["sha"],r["hw"],p["box_xyxy"],r["view"],"Woo","train","surface_defect","author_scratch",
                               r["group"],file+f"#ann{p['annotation']}","Original author positive geometry, frozen public-intake overlap screen, whole train-only timestamp; whole-fruit grade unspecified",
                               condition=r["condition"],capture=r["capture"],author_polygon_xy=p["polygon_xy"],prior_group_id=q["group_id"]))
    return eligible,list(frames.values())


def extra_initial():
    positives,frames=expanded_woo()
    save("woo_expanded_regions.json",dict(schema="source_xyxy_v1",regions=positives,usage="Full native frames. Replace overlapping source_record entries from core Woo crops; no duplicate training exposure."))
    save("review/woo_positive_frames.json",frames);board("woo_marks",frames,5,4)
    print("expanded",len(positives),len(frames),flush=True)


# Pixel centers and half extents on the 210 x 190 inspection plane exported by board().
# A coordinate refers to the visible region itself, never the entire fruit bounding box.
WOO_CLEAN=[
(0,147,39,9,7,"highlight"),(1,126,54,10,8,"colour_gradient"),(2,116,82,9,8,"calyx"),
(3,142,42,10,6,"highlight"),(4,99,109,13,10,"lenticels"),(5,108,107,14,8,"colour_stripe"),
(6,134,44,9,6,"highlight"),(7,114,114,8,10,"stem_cavity"),(8,133,61,9,8,"colour_gradient"),
(9,96,74,8,8,"calyx"),(10,117,102,13,9,"lenticels"),(11,150,43,8,7,"highlight"),
(12,122,118,13,9,"colour_stripe"),(13,127,105,9,9,"stem_cavity"),(14,58,70,8,12,"stem_cavity"),
(15,117,105,9,8,"colour_stripe"),(16,111,103,9,9,"stem_cavity"),(17,95,98,9,9,"stem_cavity"),
(18,98,87,10,7,"stem_cavity"),(19,141,89,9,11,"stem_cavity"),(20,90,67,15,9,"colour_gradient"),
(21,91,86,9,9,"stem_cavity"),(22,122,75,9,8,"colour_gradient"),(23,97,87,9,9,"stem_cavity"),
(24,125,82,10,7,"stem_cavity"),(25,84,76,9,10,"stem_cavity"),(26,113,96,6,11,"stem_cavity"),
(28,108,114,9,10,"stem"),(29,99,98,9,7,"calyx"),(30,145,106,11,10,"lenticels"),
(31,104,91,9,8,"calyx"),(32,80,103,9,13,"lenticels"),(33,82,111,9,12,"colour_stripe"),
(34,96,106,8,7,"calyx"),(35,77,115,9,12,"lenticels"),(36,75,95,8,11,"colour_stripe"),
(37,95,122,9,7,"calyx"),(38,117,101,9,10,"colour_stripe"),(39,133,102,11,13,"lenticels"),
(40,125,105,10,8,"calyx"),(41,100,106,9,10,"stem"),(42,130,44,6,8,"highlight"),
(43,70,100,9,12,"colour_stripe"),(44,85,96,11,7,"calyx"),(45,108,115,7,9,"colour_stripe"),
(46,72,89,11,9,"lenticels"),(47,110,105,10,8,"calyx")]
WOO_SAME_FACE=[
(0,108,42,9,7,"highlight"),(0,116,134,9,7,"healthy_skin"),(1,137,114,15,10,"healthy_skin"),
(2,145,135,10,7,"healthy_skin"),(3,126,115,4,6,"healthy_skin"),(4,101,107,6,7,"healthy_skin"),
(5,134,78,5,8,"highlight"),(6,126,107,12,8,"healthy_skin"),(7,96,122,10,7,"healthy_skin"),
(8,136,122,10,8,"healthy_skin"),(9,153,90,8,6,"healthy_skin"),(10,95,99,10,7,"highlight"),
(11,105,118,10,9,"healthy_skin"),(12,132,105,9,7,"highlight"),(13,134,45,10,5,"highlight"),
(14,97,90,9,7,"calyx"),(15,139,86,5,9,"calyx"),(16,111,88,9,7,"calyx"),
(17,111,133,3,4,"healthy_skin"),(18,91,145,9,5,"healthy_skin"),(19,115,84,4,5,"healthy_skin"),
(20,110,86,9,9,"calyx"),(20,80,115,12,9,"healthy_skin"),(21,111,112,9,8,"highlight"),
(22,145,73,7,7,"healthy_skin"),(23,98,92,8,7,"calyx"),(24,114,112,10,9,"highlight"),
(25,69,137,8,5,"healthy_skin"),(26,131,121,9,7,"healthy_skin"),(27,138,132,6,7,"healthy_skin"),
(28,95,127,8,7,"healthy_skin"),(29,112,107,9,7,"calyx")]
AF_DEV_POS=[(24,143,126,16,8,"elongated_gouge"),(25,136,43,22,6,"elongated_gouge"),
            (25,141,111,8,8,"brown_gouge"),(25,165,88,6,11,"brown_gouge"),
            (30,109,127,17,6,"abraded_patch"),(31,107,151,12,7,"brown_abraded_patch"),(32,88,113,10,7,"brown_abraded_patch")]
AF_DEV_NEG=[(0,78,115,9,6,"calyx"),(1,101,93,9,8,"stem_cavity"),(2,105,88,10,9,"calyx"),
            (3,137,51,10,9,"calyx"),(4,109,98,13,11,"lenticels"),(5,149,124,7,10,"stem_cavity"),
            (6,114,83,9,10,"stem_cavity"),(7,73,93,10,9,"calyx"),(8,126,90,10,10,"calyx"),
            (9,61,64,10,8,"calyx"),(10,157,78,9,9,"calyx"),(11,153,99,9,9,"calyx"),
            (12,103,127,5,11,"stem"),(13,43,124,10,6,"stem"),(15,139,74,5,8,"stem"),
            (16,101,91,10,9,"calyx"),(17,157,102,10,9,"calyx"),(18,150,93,10,9,"calyx"),
            (19,148,79,10,10,"calyx"),(20,165,123,9,8,"calyx"),(24,41,91,6,8,"calyx"),
            (25,40,100,7,8,"calyx"),(26,43,93,10,5,"stem"),(27,79,134,8,4,"stem"),
            (28,115,91,9,7,"highlight"),(30,68,112,5,6,"highlight"),(32,102,83,7,6,"stem"),(33,159,56,6,7,"calyx")]


def extra_marks():
    q=read(OUT/"review/index.json");pos=read(OUT/"review/woo_positive_frames.json")
    with ZipFile(WOOZIP) as z:coco=json.loads(z.read("train/_annotations.coco.json"))
    image_index={r["id"]:"train/"+r["file_name"] for r in coco["images"]}
    polys=defaultdict(list)
    for a in coco["annotations"]:polys[image_index[a["image_id"]]].extend(a["segmentation"])
    marks=[];counts=Counter()
    for source,split,target,idx,manual in (("Woo","train","confirmed_normal",q["woo"],WOO_CLEAN),
                                          ("Woo","train","confirmed_normal",pos,WOO_SAME_FACE),
                                          ("AFruitDB","val","surface_defect",q["af_dev"],AF_DEV_POS),
                                          ("AFruitDB","val","confirmed_normal",q["af_dev"],AF_DEV_NEG)):
        for i,cx,cy,rx,ry,name in manual:
            r=idx[i];im=pixels(r["path"]);h,w=im.shape[:2]
            x,y,t,b=r.get("view") or (0,0,w,h)
            box=[round(x+(cx-rx)*(t-x)/210),round(y+(cy-ry)*(b-y)/190),
                 round(x+(cx+rx)*(t-x)/210),round(y+(cy+ry)*(b-y)/190)]
            if r.get("member"):
                all_poly=np.zeros((h,w),np.uint8)
                for poly in polys[r["member"]]:
                    cv2.fillPoly(all_poly,[np.rint(np.array(poly).reshape(-1,2)).astype(np.int32)],1)
                l,u,v,k=box
                if all_poly[u:k,l:v].any():
                    counts["excluded_touching_author_damage"]+=1;continue
            entry=region(r["path"],r["sha"],(h,w),box,r.get("view") or ([0,0,w,h] if "member" not in r else None),
                         source,split,target,name,r["group"],f"review_20261119:{source}:{split}:{r['path']}:{i}:{cx}:{cy}",
                         "Individually visually reviewed original or registered crop; marked structure only; rest unknown",
                         capture=r.get("capture"),condition=r.get("condition"),same_frame_has_author_scratch=bool(r.get("member")))
            marks.append(entry)
    save("reviewed_regions.json",dict(schema="source_xyxy_v1",regions=marks,audit=dict(counts)))
    print("extra",Counter((r["source"],r["split"],r["target"],r["same_frame_has_author_scratch"]) for r in marks),counts)



def penn_areas():
    prior=read(SOURCE)
    allowed={"2025-10-01":"train","2025-10-02":"train","2025-10-03":"train","2024-10-10":"val"}
    if not set(allowed)<=set(prior["excluded_complete_dates"]):raise ValueError("Unexpected PSU provenance")
    results=[]
    with ZipFile(PSUZIP) as z:
        d=json.loads(z.read("train/_annotations.coco.json"));catalog={c["id"]:c["name"] for c in d["categories"]}
        by_image=defaultdict(list)
        for a in d["annotations"]:by_image[a["image_id"]].append(a)
        for im in sorted(d["images"],key=lambda r:r["file_name"]):
            date=im["file_name"][4:12]
            day=date[:4]+"-"+date[4:6]+"-"+date[6:]
            if day not in allowed:continue
            h,w=im["height"],im["width"];anns=by_image[im["id"]]
            def bits(a):
                m=np.zeros((h,w),np.uint8)
                for p in a["segmentation"]:
                    cv2.fillPoly(m,[np.rint(np.array(p).reshape(-1,2)).astype(np.int32)],1)
                return m.astype(bool)
            fruit={a["id"]:(a,bits(a)) for a in anns if catalog[a["category_id"]]=="Apple"}
            assigned=defaultdict(list)
            for a in anns:
                if catalog[a["category_id"]]=="Apple":continue
                m=bits(a);owners=[k for k,(_,fg) in fruit.items() if np.count_nonzero(fg&m)>.5*m.sum()]
                if len(owners)==1:assigned[owners[0]].append((a,m))
            member="train/"+im["file_name"];raw=z.read(member)
            hashed=hashlib.sha256(raw).hexdigest()
            path=OUT/"penn_train_dev"/(hashed[:20]+".jpg")
            path.parent.mkdir(parents=True,exist_ok=True)
            if not path.exists():path.write_bytes(raw)
            if digest(path)!=hashed:raise ValueError("PSU source changed")
            for k,(a,fg) in fruit.items():
                damage=np.zeros((h,w),bool);x,y,b,t=[float(v) for v in a["bbox"]]
                lesions=[]
                for obj,m in assigned[k]:
                    inside=fg&m;damage|=inside
                    lx,ly,dw,dh=obj["bbox"];sub=catalog[obj["category_id"]]
                    if inside.any():
                        lesions.append(dict(id=obj["id"],subtype=sub,source_image_xyxy=[lx,ly,lx+dw,ly+dh],pixel_area_on_fruit=int(inside.sum())))
                area=int(fg.sum());n=int(damage.sum())
                results.append(dict(id=len(results),path=listed(path),sha=hashed,member=member,group_id="PSU_day_"+day,split=allowed[day],
                                    view=[max(0,int(x-.05*b)),max(0,int(y-.05*t)),min(w,int(x+1.05*b)),min(h,int(y+1.05*t))],
                                    fruit_annotation=k,fruit_bbox_xyxy=[x,y,x+b,y+t],image_hw=[h,w],visible_fruit_pixels=area,
                                    lesion_union_pixels=n,lesion_union_fraction=n/area,lesions=lesions,
                                    note=f"{100*n/area:.1f}% "+','.join(sorted({u["subtype"] for u in lesions}))))
    save("penn_area_train_dev.json",dict(rules="Original PSU polygons projected at original COCO resolution. Require lesion majority overlap with exactly one fruit. Area is the pixel union of all accepted lesions intersecting that fruit. Absence of annotations does not certify healthy. No grading labels inferred.",fruits=results))
    board("penn_area",results)
    print("PSU areas, descending",[(r["id"],r["note"]) for r in sorted(results,key=lambda x:-x["lesion_union_fraction"])[:12]])

def freeze_regions():
    core=read(OUT/"core_regions.json")["regions"]
    expansion=read(OUT/"woo_expanded_regions.json")["regions"]
    manual=read(OUT/"reviewed_regions.json")["regions"]
    lookup={r["source_record"]:r for r in expansion}
    dupe=[r for r in core if r["source_record"] in lookup]
    rows=[r for r in core if r["source_record"] not in lookup]+expansion+manual
    if len({r["id"] for r in rows})!=len(rows):raise ValueError("Duplicate regional identity")
    for case in rows:
        if case["split"] not in ("train","val"):raise ValueError("Unexpected holdout")
    out=dict(schema="source_xyxy_v1",coordinate_system="Native pixel xyxy on the file in path",source_record_dedup=len(dupe),
             partitions="Train grade supporting sources plus original AF development and a different eligible Penn development date. No PLOS, Manalagi, HDF or previously frozen tests used for fitting or selection.",
             note="Only labelled visible sites are qualified. Confirmed normal is specific to the reviewed patch. A surface defect label does not establish a whole-fruit grade.",
             counts=[dict(source=k[0],split=k[1],target=k[2],n=v) for k,v in sorted(Counter((r['source'],r['split'],r['target']) for r in rows).items())],regions=rows)
    save("regions.json",out)
    print(len(rows),len(dupe),out["counts"])


def species_pages():
    f=read(ROOT/"datasets/public_defect_intake/fruitlife_v1/apple_manifest.json")["samples"]
    s=read(ROOT/"datasets/public_defect_intake/fruitlife_v1/physical_id_split.json")["assignment"]
    chosen=[]
    for obj in sorted(k for k,v in s.items() if v=="train"):
        for stage in (0,1,3,4):
            items=sorted([r for r in f if r["specimen_id"]==obj and r["source_stage"]==stage],key=lambda r:r["time"])
            for index in sorted({len(items)//2}):
                r=items[index];chosen.append(dict(id=len(chosen),path=r["path"],sha=r["sha256"],note=f"{obj} stage{stage}",original=r))
    af=defaultdict(list)
    for r in read(AF)["samples"]:
        if r["split"]=="train" and r["grade"]==2:af[r["group_id"]].append(r)
    extreme=[]
    for group,members in sorted(af.items()):
        for idx in sorted({0,len(members)//2,len(members)-1}):
            r=members[idx];extreme.append(dict(id=len(extreme),path=r["path"],sha=r["sha256"],note=group,original=r))
    save("review/severity_frames.json",dict(fruitlife=chosen,af=extreme))
    board("fruitlife_stage",chosen,4,3);board("af_severe",extreme,4,4)

def pairs_index():
    regs=read(OUT/"regions.json")["regions"]
    pairs=[];grouped=defaultdict(list)
    for r in regs:grouped[r["path"],r["split"]].append(r)
    for (path,split),subset in grouped.items():
        p=[r for r in subset if r["target"]=="surface_defect"]
        n=[r for r in subset if r["target"]=="confirmed_normal"]
        for d in p:
            for c in n:
                z=d["source_image_xyxy"];q=c["source_image_xyxy"]
                overlap=max(0,min(z[2],q[2])-max(z[0],q[0]))*max(0,min(z[3],q[3])-max(z[1],q[1]))
                if overlap:continue
                pairs.append(dict(scope="same_frame_observed_fruit_surface",source=d["source"],split=split,positive=d["id"],normal=c["id"],source_path=path,group_id=d["group_id"],fruit_id="visible fruit in this single frame; no cross-photo identity"))
    woo=read(WOO)["samples"];by_stamp=defaultdict(list);by_camera=defaultdict(list)
    for r in woo:
        by_stamp[r["capture_key"]].append(r)
        by_camera[r["condition_id"]].append(r)
    for k,records in by_stamp.items():
        for p in [r for r in records if r["grade"]==1]:
            for n in [r for r in records if r["grade"]==0]:
                pairs.append(dict(scope="same_camera_trigger_different_views",source="Woo",split="train",positive=p["path"],normal=n["path"],group_id=k,fruit_id="same synchronized station trigger; persistent specimen ID absent"))
    def paired_conditions(groups,origin):
        all_eligible=0
        for k,m in groups.items():
            p=[r for r in m if r["grade"]==1];n=[r for r in m if r["grade"]==0]
            if not p or not n:continue
            for r in p:
                candidates=[a for a in n if a.get("capture_key",a["path"])!=r.get("capture_key",r["path"])]
                all_eligible+=len(candidates)
                if not candidates:continue
                from datetime import datetime
                def tm(x):
                    if origin=="Woo":return datetime.strptime(x["capture_key"][6:],"%Y%m%d_%H%M%S_%f")
                    return datetime.fromisoformat(x["capture_time_filename"])
                target=min(candidates,key=lambda a:(abs(tm(a)-tm(r)),a["path"]))
                pairs.append(dict(scope="same_camera_and_session" if origin=="Woo" else "same_inferred_capture_burst",source=origin,split="train",
                                  positive=r["path"],normal=target["path"],group_id=k,delta_seconds=abs((tm(r)-tm(target)).total_seconds()),
                                  fruit_id="unknown; temporal nearest qualifying opposite grade in that source condition"))
        return all_eligible
    af=defaultdict(list)
    for r in read(AF)["samples"]:
        if r["split"]=="train":af[r["group_id"]].append(r)
    potential=dict(woo_camera=paired_conditions(by_camera,"Woo"),af_burst=paired_conditions(af,"AFruitDB"))
    f=read(OUT/"review/severity_frames.json")["fruitlife"]
    specimens=defaultdict(dict)
    for r in f:specimens[r["original"]["specimen_id"]][r["original"]["source_stage"]]=r["original"]
    for sid,phases in specimens.items():
        pairs.append(dict(scope="same_documented_physical_apple_across_time",source="FruitLife",split="train",group_id=sid,fruit_id=sid,
                          earlier=phases[0]["path"],minor_stage=phases[1]["path"],advanced_stage3=phases[3]["path"],advanced_stage4=phases[4]["path"],
                          storage=phases[0]["storage"],meaning="Author stage trajectory, one representative per stage, not mechanical injury and not commercial whole-fruit grading."))
    counts=[dict(scope=k[0],source=k[1],split=k[2],n=v) for k,v in sorted(Counter((r["scope"],r["source"],r["split"]) for r in pairs).items())]
    save("pairs.json",dict(unit="Each explicitly saved record describes its relation; same-photo pairs reuse sites and are not independent biological replicates. Same-station/burst = time-nearest opposite grade. Source and filename only do not establish physical identity.",eligible_cross_products=potential,colour_only_used_for_fit=0,counts=counts,pairs=pairs))
    print("Pairs",counts,potential)

# Standard review-plane conservative polygons. The selected polygon covers clearly visible
# collapse or shrivelling only and is explicitly a lower bound, not complete tissue diagnosis.
LATE_REGIONS={
2:[[34,61],[54,36],[86,24],[126,27],[155,42],[174,64],[178,103],[169,125],[144,148],[121,158],[84,159],[58,140],[42,116]],
3:[[34,61],[51,38],[93,25],[127,24],[156,40],[176,61],[183,95],[177,117],[158,143],[121,158],[82,158],[59,143],[40,117]],
6:[[129,105],[143,109],[160,117],[180,119],[171,132],[158,138],[154,150],[141,155],[131,148],[128,134]],
7:[[96,105],[111,93],[125,95],[142,110],[161,118],[155,143],[137,156],[107,153],[93,143]],
10:[[101,112],[120,101],[133,90],[143,83],[141,104],[153,108],[184,108],[173,129],[151,143],[133,149],[108,140],[91,126]],
11:[[38,61],[63,36],[94,25],[125,30],[153,43],[173,64],[179,103],[168,124],[148,142],[119,157],[81,156],[55,139],[39,118]],
}
FRESH_AREAS={0:[56,63,84,85],4:[42,88,62,108],8:[57,72,79,90]}
PENN_NORMAL={7:[142,95,163,114],21:[43,113,60,128],30:[78,121,102,136]}


def severe_records():
    import tarfile
    fbase=ROOT/"datasets/public_defect_intake/fruitlife_v1"
    frames=read(OUT/"review/severity_frames.json")["fruitlife"]
    fruit_rows=[];controls={}
    def native_x(coords,w,h):
        v=np.asarray(coords,dtype=float)*[w/210,h/190]
        return np.rint(v).astype(np.int32)
    cache_dir=OUT/"area_masks";cache_dir.mkdir(parents=True,exist_ok=True)
    with tarfile.open(fbase/"fruitlife_masks_v1.3_shadowfree.tar.gz") as tf:
        for r in frames:
            i=r["id"]
            if i not in LATE_REGIONS and i not in FRESH_AREAS:continue
            s=r["original"];hw=s["source_hw"];orig_h,orig_w=hw
            p=ROOT/s["path"];sid=s["specimen_id"];stem=Path(s["source_path"]).stem
            data=tf.extractfile(f"masks/APPLE/{sid}/{stem}.png").read()
            if hashlib.sha256(data).hexdigest()!=s["author_mask_sha256"]:raise ValueError("Changed author fruit mask")
            l,t,rr,b=s["xyxy"]
            mask=cv2.imdecode(np.frombuffer(data,np.uint8),cv2.IMREAD_GRAYSCALE)
            mask=cv2.resize(mask,(orig_w,orig_h),interpolation=cv2.INTER_NEAREST)[t:b,l:rr]>0
            h,w=mask.shape
            if (h,w)!=pixels(s["path"]).shape[:2]:raise ValueError("Fruit mask/crop mismatch")
            if i in FRESH_AREAS:
                x,y,x2,y2=FRESH_AREAS[i]
                box=native_x([[x,y],[x2,y2]],w,h).astype(int).flatten().tolist()
                assert mask[box[1]:box[3],box[0]:box[2]].all()
                controls[sid]=dict(path=s["path"],sha256=s["sha256"],specimen_id=sid,normal_region_xyxy=box,
                                   source_stage=s["source_stage"],basis="Visibly intact reviewed patch on same identified physical fruit at stage 0; no whole-fruit zero lesion label.")
                continue
            poly=native_x(LATE_REGIONS[i],w,h)
            damage=np.zeros((h,w),np.uint8);cv2.fillPoly(damage,[poly],1)
            inside=damage.astype(bool)&mask
            if inside.sum()<.96*damage.sum():raise ValueError("Poorly registered visible region")
            claim="manually delineated conservative region of visibly sunken or wrinkled skin"
            base=dict(source="FruitLife",split="train",path=s["path"],sha256=s["sha256"],image_hw=[h,w],
                      specimen_id=sid,group_id=sid,storage=s["storage"],source_stage=s["source_stage"],
                      source_mask_sha256=s["author_mask_sha256"],source_subject_stage_path=s["source_path"],
                      evidence=claim,lesion_polygon_xy=poly.tolist(),lesion_union_pixels=int(inside.sum()),
                      visible_fruit_pixels=int(mask.sum()),lesion_union_fraction=float(inside.sum()/mask.sum()),
                      area_semantics="Conservative region restricted to visible fruit pixels. Lower bound of visibly changed region, not an author lesion area or a microbiological diagnosis.")
            for name,im in (("fruit",mask),("region",inside)):
                pp=cache_dir/f"FruitLife_{sid}_{stem}_{name}.png"
                pp.write_bytes(cv2.imencode(".png",im.astype(np.uint8)*255)[1].tobytes())
                base[name+"_mask_path"]=listed(pp);base[name+"_mask_sha256"]=digest(pp)
            fruit_rows.append(base)
    for r in fruit_rows:r["paired_earlier_normal"]=controls[r["specimen_id"]]
    areas=read(OUT/"penn_area_train_dev.json")["fruits"];penn_rows=[]
    with ZipFile(PSUZIP) as z:
        orig=json.loads(z.read("train/_annotations.coco.json"));anns={r["id"]:r for r in orig["annotations"]}
        photos={i["file_name"]:i for i in orig["images"]}
        ages=defaultdict(list)
        for a in orig["annotations"]:
            if a["category_id"]==7:ages[a["image_id"]].append(a["id"])
        rot_dates=Counter()
        for name,im in photos.items():
            if ages[im["id"]]:rot_dates[name[4:12]]+=len(ages[im["id"]])
        assert rot_dates=={"20241005":8} and "2024-10-05" in {r["timestamp"][:10] for r in read(SOURCE)["samples"]}
        for i,view_box in PENN_NORMAL.items():
            r=areas[i];h,w=r["image_hw"];l,t,rr,b=r["view"]
            manual=native_x([[view_box[0],view_box[1]],[view_box[2],view_box[3]]],rr-l,b-t)+[l,t]
            x,y,x2,y2=manual.flatten().tolist()
            m=np.zeros((h,w),np.uint8)
            polys=anns[r["fruit_annotation"]]["segmentation"]
            cv2.fillPoly(m,[np.rint(np.array(a).reshape(-1,2)).astype(np.int32) for a in polys],1)
            v=np.zeros_like(m);surface=[]
            for lesion in r["lesions"]:
                a=anns[lesion["id"]];surface+=a["segmentation"]
                cv2.fillPoly(v,[np.rint(np.array(p).reshape(-1,2)).astype(np.int32) for p in a["segmentation"]],1)
            v=v&m
            if not m[y:y2,x:x2].all() or v[y:y2,x:x2].any():raise ValueError(("Unconfirmed clean source patch",i))
            assert int(v.sum())==r["lesion_union_pixels"] and int(m.sum())==r["visible_fruit_pixels"]
            shapes=[np.asarray(p,dtype=np.float32).reshape(-1,2) for p in surface]
            rectangular=all(len(p)<=5 and cv2.contourArea(p)/max(1,float(np.prod(np.ptp(p,axis=0))))>.97 for p in shapes)
            rr=dict(r,source="PennState",evidence="Author sunburn annotation over a visibly discoloured region.",
                    lesion_polygons_native=surface,fruit_polygons_native=polys,
                    lesion_vertex_counts=[len(p) for p in shapes],is_coarse_rectangle=bool(rectangular),use_for_lesion_pixel_overlap=not rectangular,
                    normal_region_xyxy=[x,y,x2,y2],
                    area_semantics=("Coarse rectangular author annotation. Its percentage is box coverage only, including healthy boundary skin. Excluded from lesion pixel area validation." if rectangular else
                                    "Nonrectangular author sunburn mask clipped to segmented visible fruit, pixel union. No assignment of pathological rot or commercial grade."))
            penn_rows.append(rr)
    doc=dict(scope="Training only. PSU true rot author annotations all belonged to 2024-10-05, a historical external holdout day. No true Penn rot admitted. Natural stage 3/4 storage changes and PSU sunburn evaluated separately.",
             penn_original_rot_by_date=rot_dates,fruitlife_visible_collapse=fruit_rows,penn_large_author_sunburn=penn_rows,
             unique_physical_FruitLife_subjects=3,penn_capture_days=sorted({r["group_id"] for r in penn_rows}))
    save("severity_evidence.json",doc)
    print("affected",[(r["specimen_id"],r["source_stage"],round(r["lesion_union_fraction"],4)) for r in fruit_rows])

def severity_board():
    data=read(OUT/"severity_evidence.json")
    show=[]
    for r in data["fruitlife_visible_collapse"]:
        im=Image.open(ROOT/r["path"]).convert("RGB");draw=ImageDraw.Draw(im)
        xy=[tuple(p) for p in r["lesion_polygon_xy"]]
        draw.line(xy+[xy[0]],fill=(255,215,20),width=max(2,im.width//155))
        show.append((im,f"{r['specimen_id']} / stage {r['source_stage']} / {r['lesion_union_fraction']*100:.1f}% min"))
    for r in data["penn_large_author_sunburn"]:
        im=Image.open(ROOT/r["path"]).convert("RGB");draw=ImageDraw.Draw(im)
        for p in r["lesion_polygons_native"]:
            xy=[tuple(z) for z in np.array(p).reshape(-1,2)]
            draw.line(xy+[xy[0]],fill=(255,215,20),width=max(2,im.width//280))
        draw.rectangle(r["normal_region_xyxy"],outline=(28,224,147),width=max(2,im.width//300))
        shape="box only" if r["is_coarse_rectangle"] else "mask"
        show.append((im.crop(r["view"]),f"Penn / {shape} / {r['lesion_union_fraction']*100:.1f}%"))
    canvas=Image.new("RGB",(3*250,3*250),(247,246,244))
    D=ImageDraw.Draw(canvas)
    for j,(im,note) in enumerate(show):
        im.thumbnail((241,220),Image.Resampling.LANCZOS);x=j%3*250;y=j//3*250
        canvas.paste(im,(x+(250-im.width)//2,y+(225-im.height)//2))
        D.text((x+4,y+226),note,fill=(15,29,40))
    canvas.save(OUT/"review/severity_verified.jpg",quality=91)


def integrity():
    regions=read(OUT/"regions.json");rp=regions["regions"]
    known=defaultdict(set);hash_of={};partition=defaultdict(set)
    for r in rp:
        p=ROOT/r["path"]
        if p not in hash_of:hash_of[p]=digest(p)
        if hash_of[p]!=r["image_sha256"]:raise ValueError("Changed region photo")
        partition[r["image_sha256"]].add(r["split"])
        known[(r["source"],r["split"],r["target"])].add(r["path"])
    if any(len(v)>1 for v in partition.values()):raise ValueError("Train-dev exact cross-photos")
    pairs=read(OUT/"pairs.json")
    ids={r["id"]:r for r in rp}
    for z in pairs["pairs"]:
        if z["scope"]=="same_frame_observed_fruit_surface":
            a,b=ids[z["positive"]],ids[z["normal"]]
            if (a["path"],a["split"],a["group_id"])!=(b["path"],b["split"],b["group_id"]):raise ValueError("Invalid within-frame comparison")
    sr=read(OUT/"severity_evidence.json")
    for r in sr["fruitlife_visible_collapse"]:
        for k in ("fruit","region"):
            if digest(ROOT/r[k+"_mask_path"])!=r[k+"_mask_sha256"]:raise ValueError("Wrong severity masks")
    exclusions=["PLOS Dataset_1 (all)", "Manalagi (all)", "HDF (all)", "AFruitDB old 110 test", "author train-excluded Woo trigger/group",
                "PSU historical external dates incl. 2024-10-05 (all 8 true Rot annotations), additional reserved date 2024-10-30",
                "SSDA 113 development and quarantine (not used as training regions)","FruitLife external 3 physical specimens",
                "Grocery/Commons historical stress sources"]
    files=["regions.json","pairs.json","severity_evidence.json","penn_area_train_dev.json","review/severity_verified.jpg"]
    summary=dict(schema="grade_pairs_evidence_20261119_v1",files={f:digest(OUT/f) for f in files},code_sha256=digest(__file__),
                 region_counts=regions["counts"],unique_representations=[dict(source=k[0],split=k[1],target=k[2],n_images=len(v)) for k,v in sorted(known.items())],
                 source_traces={"AFruitDB DOI":read(AF)["source_doi"],
                                "AF local":digest(AF),"AF individual original spot review":digest(SPOTS),"Woo source":digest(WOOZIP),"Woo grade review":digest(WOO),"PSU source":digest(PSUZIP),"SSDA date split":digest(SSA),"SSDA manual review":digest(BLIND)},
                 history="AF and confirmed Woo faces/SSDA are previously used training sources. More Woo regions and normal boxes add localized supervision; they are not a new out-of-sample trial. AF development also previously used. Exact source hashes and family filters cannot certify physical identity unless the author supplied IDs.",
                 exclusions=exclusions,source_authority="Author disease polygons and author grade/stage labels are identified separately from locally reviewed normal structures or conservative extent polygons. Appearance-only grades do not substitute for pathological rotting.",
                 pairs=pairs["counts"],cross_split_equal_source_file_hash=0,qa="Every localized image path and SHA checked, all reported same-frame pairs bind to matched regions in that exact image. Stored FruitLife masks rehashed after serialization.")
    save("manifest.json",summary)
    print("verified",len(rp),"regions",len(hash_of),"distinct image files",len(pairs["pairs"]),"pair/trajectory records",summary["source_traces"]["AFruitDB DOI"])

if __name__=="__main__":
    initial();extra_initial();extra_marks();penn_areas();freeze_regions();species_pages();pairs_index();severe_records();severity_board();integrity()

