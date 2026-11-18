"""Preview pinned public COCO version labels in original context before admitting any crops."""
from collections import Counter,defaultdict
from io import BytesIO
import argparse,json
from pathlib import Path
from zipfile import ZipFile
import numpy as np
from PIL import Image,ImageDraw
from class0_patch_refinement import ROOT,atomic_json,sha,configure
from defect_public_intake import image_member,native_split,geometry,rle_counts

OUT=ROOT/"experiments/defect_public_previews"
VERSIONS={
    "woo":dict(project_id="woo-wwyog/orchard-apple-defect-segmentatio",version="1",name="scratch",target=0),
    "psu":dict(project_id="penn-state-university/fruit_defects",version="2",name="Codling Moth Entry",target=1),
}


def preview(archive,source,category=None):
    configure(OUT,"cpu")
    spec=dict(VERSIONS[source])
    if category:
        if source!="psu" or category not in ("Plum Curculio","Stink Bug"):raise ValueError("Unexpected candidate category")
        spec["name"]=category
    arc=Path(archive).resolve();checksum=sha(arc)
    out=OUT/(source+"_"+checksum[:12]+("_"+category.lower().replace(" ","_") if category else ""));out.mkdir(exist_ok=True)
    with ZipFile(arc) as z:
        files=set(z.namelist())
        members=sorted(n for n in files if n.endswith(".json") and ("_annotations.coco" in n.lower() or "/instances_" in n.lower()))
        if not members:raise ValueError("No COCO instances member")
        categories=set();pool=[];counts=Counter()
        for member in members:
            data=json.loads(z.read(member).decode("utf-8-sig"))
            cats={r["id"]:r["name"] for r in data["categories"]};categories.update(cats.values())
            idx={r["id"]:r for r in data["images"]}
            for a in data["annotations"]:
                name=cats[a["category_id"]];r=idx[a["image_id"]]
                path=image_member(files,member,r["file_name"]);part=native_split(member,path)
                counts[f"{part}/{name}"]+=1
                if part=="train" and name.casefold()==spec["name"].casefold() and not a.get("iscrowd"):
                    b,typ=geometry(a,r["width"],r["height"])
                    pool.append((path,member,r,a,name,b,typ))
        eligible={i[4] for i in pool}
        if len(eligible)!=1:raise ValueError(f"Expected one exact candidate name in native train; categories: {sorted(categories)}")
        member_names=[n for n in files if n.lower().endswith("readme.roboflow.txt") or n.lower().endswith("readme.dataset.txt")]
        notes=[z.read(n).decode("utf-8-sig",errors="replace") for n in member_names]
        urls=[f"https://universe.roboflow.com/{spec['project_id']}/dataset/{spec['version']}",
              f"https://universe.roboflow.com/{spec['project_id']}"]
        # The project and version are checked against the publisher's export details/README by the reviewer.
        candidates=sorted(pool,key=lambda q:np.sqrt((q[5][2]-q[5][0])*(q[5][3]-q[5][1]))/max(q[2]["width"],q[2]["height"]))
        n=min(18,len(candidates));sample=[candidates[k] for k in np.linspace(0,len(candidates)-1,n,dtype=int)]
        annotated=[]
        for start in range(0,len(sample),6):
            items=sample[start:start+6]
            page=Image.new("RGB",(680,220*len(items)),(250,250,250));d=ImageDraw.Draw(page)
            for j,(path,member,r,a,name,b,typ) in enumerate(items):
                with Image.open(BytesIO(z.read(path))) as original:im=original.convert("RGB")
                if im.size!=(r["width"],r["height"]):raise ValueError("Declared dimensions differ from actual pixels")
                overlay=im.copy();od=ImageDraw.Draw(overlay);od.rectangle(b,outline="cyan",width=max(2,im.width//300))
                if typ=="polygon":
                    for part in a["segmentation"]:
                        pts=[tuple(x) for x in np.asarray(part).reshape(-1,2)];od.line(pts+[pts[0]],fill="yellow",width=max(2,im.width//400))
                elif typ=="RLE":
                    runs=rle_counts(a["segmentation"]["counts"],*im.size);p=0;arr=np.zeros(im.width*im.height,np.uint8)
                    for kk,nr in enumerate(runs):
                        if kk%2:arr[p:p+nr]=100
                        p+=nr
                    overlay.paste((255,230,0),(0,0,im.width,im.height),Image.fromarray(arr.reshape(im.width,im.height).T))
                crop=im.crop((np.floor(b[0]),np.floor(b[1]),np.ceil(b[2]),np.ceil(b[3])))
                for k,v in enumerate((im,overlay,crop)):
                    if k==2:
                        gain=min(218/v.width,183/v.height,5.)
                        v=v.resize((max(1,round(v.width*gain)),max(1,round(v.height*gain))),Image.Resampling.LANCZOS)
                    else:v.thumbnail((218,183))
                    page.paste(v,(k*226,j*220))
                d.text((0,j*220+184),f"{start+j+1}. {name} image {r['id']} annotation {a['id']} | {typ}  bbox {list(map(round,b))}",fill="black")
                annotated.append(dict(member=member,image=path,id=r["id"],annotation=a["id"],category=name,box=b))
            page.save(out/f"preview_{start//6+1}.jpg",quality=93)
        exact=next(iter(eligible))
        atomic_json(out/"inspection.json",dict(zip_name=arc.name,archive_sha256=checksum,readme_members=member_names,
                    mentions_project=any(spec["project_id"].split("/")[-1].casefold() in t.casefold() for t in notes),
                    categories=sorted(categories),counts=counts,samples=annotated,preliminary_candidate=exact,
                    accepted=False,expected_official=urls[0]))
        atomic_json(out/"proposed_pin.json",dict(project_id=spec["project_id"],version=spec["version"],record_url=urls[0],
                    license="CC BY 4.0",license_evidence_url=urls[1],archive=str(arc),archive_sha256=checksum,
                    coco_members=members,category_map={exact:spec["target"]}))
    print(out,counts,flush=True)


if __name__=="__main__":
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--archive",type=Path,required=True);ap.add_argument("--source",choices=VERSIONS,required=True)
    ap.add_argument("--category",choices=("Plum Curculio","Stink Bug"))
    args=ap.parse_args();preview(args.archive,args.source,args.category)
