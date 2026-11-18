"""Synthetic COCO intake checks plus actual integrity verification of all 710 SSDA originals."""

from io import BytesIO
from pathlib import Path
import json
from uuid import uuid4
from zipfile import ZipFile
import numpy as np
from PIL import Image
from class0_patch_refinement import ROOT,configure,read,sha,atomic_json
from defect_public_intake import SSDA,original_bank,build,load_spec,rle_bounds,geometry
from defect_public_foreground import lesion_context,copied_lesion


OUT=ROOT/"experiments/defect_public_intake_checks"


def run():
    configure(OUT,"cpu")
    dst=OUT/uuid4().hex;dst.mkdir()
    bank=original_bank(SSDA)
    quarantined=next(b for b in bank if b["part"]=="quarantine")
    orig=Path(quarantined["src"])
    with Image.open(orig) as im:
        other=BytesIO()
        im.transpose(Image.Transpose.ROTATE_90).transpose(Image.Transpose.FLIP_LEFT_RIGHT).resize((256,256)).save(other,format="JPEG",quality=84)
    def random_image(seed):
        g=np.random.default_rng(seed)
        a=g.integers(0,256,(10,10,3),dtype=np.uint8)
        im=Image.fromarray(a).resize((90,86),Image.Resampling.BILINEAR);p=BytesIO();im.save(p,format="PNG");return p.getvalue()
    source=[("a.rf.a1.png",random_image(1)),("b.png",random_image(2)),("same.jpg",orig.read_bytes()),
            ("rotated.jpg",other.getvalue()),("negative.png",random_image(3)),("bad.png",random_image(4)),
            ("a.rf.b2.png",random_image(5)),("crowd.png",random_image(6)),("shared_holdout.png",random_image(7))]
    images=[];anns=[]
    for i,(name,blob) in enumerate(source):
        with Image.open(BytesIO(blob)) as im:w,h=im.size
        images.append(dict(id=i,file_name=name,width=w,height=h))
        g=dict(id=i,image_id=i,category_id=1,bbox=[5.,8.,11.,12.],segmentation=[[5,8,16,8,16,19,5,20]],iscrowd=0)
        if i==1:g["segmentation"]=[]
        if i==4:g["category_id"]=3
        if i==5:g["segmentation"]=[[0,0,4,0,2,3]]
        if i==7:g["iscrowd"]=1
        anns.append(g)
    coco=dict(categories=[dict(id=1,name="Scratch"),dict(id=3,name="Fungus")],images=images,annotations=anns)
    archive=dst/"synthetic.zip"
    with ZipFile(archive,"w") as z:
        z.writestr("train/_annotations.coco.json",json.dumps(coco))
        for n,b in source:z.writestr("train/"+n,b)
        z.writestr("val/holdout.png",source[8][1])
        z.writestr("val/_annotations.coco.json",json.dumps(dict(categories=coco["categories"],
                        images=[dict(id=9,file_name="holdout.png",width=90,height=86)],
                        annotations=[dict(anns[8],id=9,image_id=9)])))
    spec=dict(project_id="SYNTHETIC-TEST",version="1",record_url="https://example.org/synthetic",
              license="synthetic internal fixture",license_evidence_url="https://example.org/synthetic-license",
              archive=archive.name,archive_sha256=sha(archive),coco_members=["train/_annotations.coco.json","val/_annotations.coco.json"],
              category_map={"Scratch":0})
    cfg=dst/"source.json";atomic_json(cfg,spec)
    parsed=load_spec(cfg)
    dest=dst/"build"
    result=build([parsed],bank,dest,sha(SSDA))
    rows=read(dest/"positives.json");decisions={d["image_id"]:d for d in read(dest/"image_decisions.json")}
    done=[]
    def check(name,test):
        if not test:raise AssertionError(name)
        done.append(name)
    check("all original 710 frames independently byte-hashed with one quarantined",len(bank)==710)
    check("exact SHA quarantine",decisions[2]["decision"]=="SSDA overlap candidate")
    check("rotated + flipped + resized + JPEG of quarantined frame excluded",decisions[3]["decision"]=="SSDA overlap candidate")
    check("unmapped category only never trains background",decisions[4]["decision"]=="no valid approved localized positive")
    check("bad polygon excluded",decisions[5]["decision"]=="no valid approved localized positive")
    check("crowd ignored",decisions[7]["decision"]=="no valid approved localized positive")
    check("training duplicate of author holdout quarantined",decisions[8]["decision"]=="overlaps external author validation/test group")
    check("author val excluded",decisions[9]["decision"]=="author partition excluded from training")
    check("lineage augmentation group exported once",sum(decisions[i]["decision"]=="export positive crops" for i in (0,6))==1)
    check("only two eligible positives",len(rows)==2)
    same=Image.open(BytesIO(random_image(1))).convert("RGB")
    context=lesion_context(same,[10,12,31,36])
    check("repeated local lesion context detected",copied_lesion(context,context) is not None)
    check("unrelated local patterns remain distinguishable",copied_lesion(context,lesion_context(Image.open(BytesIO(random_image(22))),[10,12,31,36])) is None)
    check("no background supervision",not any(r["background_supervision"] for r in rows))
    poly=next(r for r in rows if r["mask"]);box=next(r for r in rows if not r["mask"])
    with Image.open(dest/poly["mask"]) as mask,Image.open(dest/poly["crop"]) as crop:
        check("crop and polygon mask match dimensions",mask.size==crop.size==(11,12) and np.asarray(mask).max()==255)
    check("crop and source polygon offsets correct",poly["polygon_xy"][0][0]==[5.,8.] and poly["polygon_in_crop"][0][0]==[0.,0.])
    check("original image bytes preserved",sha(dest/box["image"])==decisions[1]["sha256"] and box["raw_split"]=="train")
    check("review contact sheets generated",any(dest.glob("review_*.jpg")))
    check("column-major RLE geometric bounds",rle_bounds([4,2,6],3,4)==[1,0,2,2])
    check("ASCII COCO RLE can be decoded",rle_bounds("426",3,4)==[1,0,2,2])
    check("compressed differential negative run encoding",rle_bounds("131N9",4,4)==rle_bounds([1,3,1,1,10],4,4))
    for name,fn in [
            ("unapproved case variant rejected",lambda:build([{**parsed,"category_map":{"scratch":0}}],bank,dst/"wrong_name",sha(SSDA))),
            ("wrong checksum refused",lambda:build([{**parsed,"archive_sha256":"0"*64}],bank,dst/"wrong_sha",sha(SSDA))),
            ("mask RLE dimensions rejected",lambda:geometry(dict(bbox=[0,0,2,2],segmentation=dict(size=[1,2],counts=[2])),2,2)),
            ("bad unordered counts rejected",lambda:rle_bounds([4,-1,9],3,4))]:
        try:fn()
        except ValueError:done.append(name)
        else:raise AssertionError(name)
    report=dict(tests=done,passed=len(done),synthetic_images=len(images)+1,synthetic_result=result,
                actual_SSDA_checked=len(bank),real_external_training_samples=0,workspace=str(dst))
    atomic_json(OUT/"cpu_check.json",report)
    print(report,flush=True)


if __name__=="__main__":
    run()
