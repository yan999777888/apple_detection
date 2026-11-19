"""Build same-view, same-scale patch pairs from verified studio originals, without whole-fruit grades."""
from collections import Counter
import hashlib
import json

import cv2
import numpy as np
from PIL import Image, ImageDraw

from harvard_severe_source_review import INDEX, OUT, ROOT, crop_view, refpath
from penn_surface_challenge import reference_paths,signature,two_views


# Coordinates refer to a 400 x 400 display of the same automatically bounded fruit crop.
PAIRS={
    19:dict(polygon=[(126,205),(131,201),(140,199),(147,202),(151,208),(150,222),(146,230),(139,234),(131,230),(126,220)],
            positive=(116,191,163,238),control=(164,220,211,267),
            observation="A bounded olive-brown stippled mark on the lower-left yellow cheek."),
    75:dict(polygon=[(245,258),(248,252),(254,249),(261,245),(267,242),(274,244),(278,249),(278,257),(273,264),(267,271),(258,275),(251,279),(246,273)],
            positive=(234,235,286,287),control=(167,231,219,283),
            observation="A discrete irregular matte brown patch on the blushed cheek."),
    133:dict(polygon=[(139,282),(144,277),(150,276),(156,277),(162,282),(164,287),(158,293),(150,295),(144,292),(139,287)],
             positive=(135,270,167,302),control=(158,237,190,269),
             observation="One small round tan/olive spot on a yellow-orange cheek."),
}


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(data):
    return hashlib.sha256(data).hexdigest()


def train_references():
    old={}
    def add(key,p):
        if p.is_file() and p.suffix.lower() in {".jpg",".jpeg",".png",".bmp"}:
            old.setdefault(p.resolve(),key)
    for g,p in reference_paths():add(g,p)
    for p in (ROOT/"apple_research_v2/sources/additional_attribute_sources/hdf_real_only/raw").glob("*/*"):add("additional_HDF",p)
    for name in ("afruitdb_apples_v1","grade_guard_sources_v2","apple_matched_quality_v1",
                 "apple_matched_fruitlife_woo_v1","grade_guard_hdf_v1","harvard_grade_normal_anchors_v1"):
        for row in read(ROOT/"datasets"/name/"manifest.json")["samples"]:
            if row.get("split")!="train":continue
            for key in ("path","original_path"):
                if row.get(key):add(name,ROOT/row[key])
    return old


def audit_overlap(records):
    old=train_references()
    old_bytes=set()
    sig=[]
    for r in records:
        p=ROOT/r["source_path"]
        with Image.open(p) as raw:fruit=np.array(raw.crop(r["fruit_crop_xyxy"]).convert("RGB"))[:,:,::-1]
        files=[(p,r['sha256']),(ROOT/r["positive_path"],r["positive_sha256"]),(ROOT/r["control_path"],r["control_sha256"])]
        checks=[("fruit_crop",fruit)]
        for q,h in files:
            old_bytes.add(h)
            checks.append((q.name,cv2.imdecode(np.frombuffer(q.read_bytes(),np.uint8),cv2.IMREAD_REDUCED_COLOR_4)))
        for kind,a in checks:
            for v in two_views(a):
                for k in range(4):
                    rot=np.rot90(v,k)
                    for c in (rot,rot[:,::-1]):
                        h,pix=signature(c);sig.append((r["number"],kind,h,pix.astype(np.int16)))
    near=[]
    exact=[]
    for p,group in old.items():
        buf=p.read_bytes()
        if sha(buf) in old_bytes:exact.append(str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p))
        im=cv2.imdecode(np.frombuffer(buf,np.uint8),cv2.IMREAD_REDUCED_COLOR_4)
        assert im is not None,p
        for v in two_views(im):
            h,pix=signature(v)
            for n,kind,code,ref in sig:
                if (h^code).bit_count()<=6:
                    err=float(np.abs(pix.astype(np.int16)-ref).mean())
                    if err<14:
                        near.append(dict(number=n,view=kind,reference=p.name,source=group,mae=round(err,3)))
    result=dict(reference_count=len(old),counts=dict(Counter(old.values())),exact_matches=exact,near_matches=near,
                rule="SHA-256 and full/central 2/3 D4 view screening of original, fruit, positive and control. pHash Hamming <=6 and RGB32 MAD <14/255. Covers listed historical training references and all 5000 historical HDF files. No current blind test opened.")
    (OUT/"patch_overlap_audit.json").write_text(json.dumps(result,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    return result


def build_local():
    rows={int(r["label"].rsplit("_",1)[1].split(".")[0]):r["dataFile"] for r in read(INDEX)["files"] if r["label"].startswith("Anjali_")}
    pairs=[]
    panel=Image.new("RGB",(750,238*len(PAIRS)),"white");draw=ImageDraw.Draw(panel)
    patches=OUT/"training_patch_pairs";patches.mkdir(exist_ok=True)
    for j,(number,ann) in enumerate(PAIRS.items()):
        row=rows[number];p=refpath(row)
        data=p.read_bytes();assert hashlib.md5(data).hexdigest()==row["md5"]
        _,box,ts,camera=crop_view(p)
        with Image.open(p) as raw:fruit=raw.convert("RGB").crop(box)
        w,h=fruit.size
        def raw_box(b):
            return [round(val*(h/400 if i%2 else w/400))+(box[1] if i%2 else box[0]) for i,val in enumerate(b)]
        poly=np.int32([(round(x*w/400),round(y*h/400)) for x,y in ann["polygon"]])
        hsv=cv2.cvtColor(np.array(fruit),cv2.COLOR_RGB2HSV)
        f=np.uint8((hsv[:,:,1]>42)&(hsv[:,:,2]>28))*255
        kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(21,21))
        cs,_=cv2.findContours(cv2.morphologyEx(f,cv2.MORPH_CLOSE,kernel),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        mask=np.zeros((h,w),np.uint8);cv2.drawContours(mask,[max(cs,key=cv2.contourArea)],-1,1,cv2.FILLED)
        damage=np.zeros_like(mask);cv2.fillPoly(damage,[poly],1)
        denom=int(mask.sum());area=int((damage&mask).sum())
        record=dict(number=number,source_path=p.relative_to(ROOT).as_posix(),source_file_id=row["id"],
                    sha256=sha(data),official_md5=row["md5"],camera=camera,capture_time=ts,
                    source_doi="10.7910/DVN/KFQKXL",license="CC0-1.0",source_class="Imperfect",
                    split="train",group_id="Harvard_Anjali_Canon_20240420_white_studio_single_session",
                    observation=ann["observation"],fruit_crop_xyxy=box,fruit_projected_pixels=denom,
                    conservative_blemish_pixels=area,visible_projected_fraction=area/denom,
                    polygon_xy_raw=(poly+np.int32(box[:2])).tolist())
        frame=fruit.resize((210,210));overlay=ImageDraw.Draw(frame)
        for key,color in [("positive","#DB2872"),("control","#167C9D")]:
            b=raw_box(ann[key]);local=[v-box[i%2] for i,v in enumerate(b)]
            # Round to the positive patch's exact pixel extent, including non-square fruit crops.
            if key=="positive":extent=(local[2]-local[0],local[3]-local[1])
            else:
                local[2]=local[0]+extent[0];local[3]=local[1]+extent[1]
                b=[v+box[i%2] for i,v in enumerate(local)]
            assert mask[local[1]:local[3],local[0]:local[2]].min()==1
            out=patches/f"{number:03d}_{key}.png"
            patch=fruit.crop(local);patch.save(out)
            record[key+"_path"]=out.relative_to(ROOT).as_posix()
            record[key+"_sha256"]=sha(out.read_bytes());record[key+"_xyxy_raw"]=b
            overlay.rectangle([round(v*210/400) for v in ann[key]],outline=color,width=2)
            rot=patch.resize((206,206));panel.paste(rot,(246 if key=="positive" else 482,j*238))
        panel.paste(frame,(0,j*238));draw.text((0,j*238+211),f"{number} marked vs nearby intact-looking; conservative core = {area/denom:.2%}",fill="black")
        pairs.append(record)
    panel.save(OUT/"local_training_pair_review.jpg",quality=96)
    return pairs


def main():
    pairs=build_local()
    overlap=audit_overlap(pairs)
    assert not overlap["exact_matches"] and not overlap["near_matches"],"Inspect possible source overlap first"
    study=read(OUT/"camera_thumbnails_audit.json");assert all("error" not in r for r in study) and len(study)==204
    roots=read(OUT/"hdf_geometric_overlap.json")["pairs"]
    result=dict(samples=[],counts={"whole_fruit_severe_train":0,"whole_fruit_good_train":0,"training_local_pairs":len(pairs)},
             local_pairs=pairs,source="Harvard Dataverse Anjali Sharma, CC0",primary="10.7910/DVN/KFQKXL",
             note="Only local visual supervision: positively marked patches vs equal-scale nearby patches on the same fruit photograph without the same visible spot. Coloration or physiological origin is uncertain. No whole-fruit grades, no new severe training examples, no test examples; permutations of crops are not independent fruits.",
             overlap="0 earlier-used source/fruit/patch matches under listed checks. Imperfect #75 was viewed previously for source screening only. Entire Canon burst assigned to one training-only group. Bad screenshots all previously seen as HDF source images.",
             audit_counts={"embedded_studio_images":len(study),"whole_good_archive_exif_records":682,"bad_screenshots_matching_HDF":len(roots)})
    (OUT/"manifest.json").write_text(json.dumps(result,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    print(result['counts'],overlap['reference_count'])


if __name__=="__main__":main()
