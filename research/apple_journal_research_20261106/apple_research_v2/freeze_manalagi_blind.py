"""Freeze a visible-grade panel from a source used solely for external testing."""
from collections import Counter
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

from manalagi_blind_source import ROOT, OUT, inventory, original
from penn_surface_challenge import reference_paths,signature,two_views


# The numerator is visible lesion extent and apparent outer surface integrity,
# not the diagnosed disease alone. The exclusions remain available for inspection.
GOOD={
    43:"Uniform intact green face under a shaded branch.",44:"Even light-green peel, normal fine lenticels.",
    46:"Light yellow-green peel with faint natural blush near shoulder.",47:"Even lime-green peel with a light reflection.",
    48:"Intact cheek, natural yellow blush at the shoulder.",49:"Small sunlit areas and dappled shade; peel looks intact.",
    52:"Intact olive-green face, fine pale lenticels and shadow.",53:"Intact smooth green cheek and visible calyx.",
    54:"Glare on a smooth green cheek; a neighboring fruit is visible at the corner.",55:"Intact yellow-green face with soft sunlit gradient.",
}
MILD={
    24:"Scattered small reddish-brown points, mainly toward the right; most visible peel intact.",
    25:"Numerous fine, mostly separate points, without a broad coalescent area or contour collapse.",
    26:"Small isolated round spots around and below the calyx; no broad damaged patch.",
    28:"Widely spaced, small reddish points across intact yellow-green skin.",
    31:"Separate small circular spots; green peel remains dominant.",
    34:"Scattered speck-size marks, no large dark plaque or gross deformation.",
    35:"Spaced small dark specks and sparse brown rings; intact central peel.",
    38:"Isolated sharply edged dark spots separated by mostly intact green peel.",
    58:"Fine yellow-brown reticulated lines on the left cheek, visible through water droplets.",
    59:"Fine pale cork-like lines toward the top/right; underlying green cheek intact.",
    69:"Narrow brown reticulated streaks chiefly at the left shoulder; right cheek comparatively intact.",
}
SEVERE={
    0:"Several large dark necrotic patches on the same surface.",1:"Multiple large confluent dark plaques, broad discoloration.",
    2:"One broad, nearly black round plaque covers a substantial part of the visible cheek.",
    3:"Several large dark necrotic areas adjoining the calyx and extending across the top.",
    8:"Large centrally darkened plaque, halo and numerous satellite lesions.",
    10:"Broad coalescing central patches and further satellite plaques.",
    12:"Large brown lesion with visibly collapsed, cracked texture.",
    15:"Broad contiguous blackened areas and extensive calyx-region damage.",
    19:"Two broad blackened regions, one extending across the calyx.",
    22:"Dense dark lesions across much of the visible fruit; broad distributed affected surface.",
    36:"Numerous lesions with coalescent dark damage around calyx.",
}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def old_paths():
    paths={}
    def add(group,p):
        p=Path(p)
        if p.is_file() and p.suffix.lower() in {".jpg",".jpeg",".png",".bmp",".webp",".tif",".tiff"}:
            paths.setdefault(p.resolve(),group)
    for origin,p in reference_paths():
        add(origin,p)
    datasets=ROOT/"datasets"
    for m in sorted(datasets.glob("*/manifest.json")):
        if m.parent==OUT:continue
        for r in json.loads(m.read_text(encoding="utf-8-sig")).get("samples",[]):
            if isinstance(r,dict):
                for key in ["path","original_path","source_path"]:
                    if isinstance(r.get(key),str):
                        add(m.parent.name,ROOT/r[key])
    for p in (ROOT/"apple_research_v2/sources/additional_attribute_sources/hdf_real_only/raw").glob("*/*"):
        add("remaining_HDF",p)
    for p in (datasets/"public_defect_intake/fruitlife_v1").rglob("*.jpg"):
        add("remaining_FruitLife",p)
    for p in (datasets/"public_defect_intake/plos_apple_quality_original/Dataset_1").rglob("*.jpeg"):
        add("excluded_PLOS",p)
    return paths


def compare(rows):
    sums={r["sha256"]:i for i,r in enumerate(rows)}
    if len(sums)!=len(rows): raise ValueError("Exact repeats inside test source")
    signatures=[]
    for i,r in enumerate(rows):
        raw=original(r).read_bytes()
        assert sha(raw)==r["sha256"]
        im=cv2.imdecode(np.frombuffer(raw,np.uint8),cv2.IMREAD_REDUCED_COLOR_4)
        for view in two_views(im):
            for k in range(4):
                rot=np.rot90(view,k)
                for tr in (rot,rot[:,::-1]):
                    h,pix=signature(tr)
                    signatures.append((i,h,pix.astype(np.int16)))
    paths=old_paths()
    collisions=[]
    internal_pairs=[]
    counts=Counter(paths.values())
    for p,origin in list(paths.items())+[(original(r),"__internal") for r in rows]:
        buf=p.read_bytes();digest=sha(buf)
        own=sums.get(digest)
        if origin!="__internal" and own is not None:
            collisions.append([own,origin,str(p),"exact"]);continue
        dec=cv2.imdecode(np.frombuffer(buf,np.uint8),cv2.IMREAD_REDUCED_COLOR_4)
        assert dec is not None, p
        for view in two_views(dec):
            code,pix=signature(view)
            for i,h,other in signatures:
                if origin=="__internal" and own is not None and own>=i:continue
                if (code^h).bit_count()<=8:
                    err=float(np.abs(pix.astype(np.int16)-other).mean())
                    if err<18:
                        if origin=="__internal":internal_pairs.append((own,i))
                        else:collisions.append([i,origin,str(p),round(err,3)])
    result=dict(reference_counts=dict(counts),reference_total=len(paths),historical_matches=collisions,
                within_test_pairs=sorted(set(internal_pairs)),
                protocol="Source SHA-256, full and central 2/3 crop D4 matching with pHash distance <=8 AND 32x32 mean absolute RGB difference <18/255. Compare both full and central views of all references. Unknown provenance or substantial recrops may evade the screen.")
    return result


def excluded_reason(idx,condition):
    if idx in {45,50,51}:return "Author healthy; atypical bluish patches or surface deposits make visible grade ambiguous."
    if condition=="Anthracnose":return "Limited but prominent necrotic focus; uncertain boundary between minor and major external damage."
    if condition=="Black Pox":return "Density, size or ring texture intermediate; no confident visible minor/severe assignment."
    return "Russet-like or mottled surface covers a substantial/uncertain extent; mild versus heavy uncertain."


def main():
    rows=inventory()["samples"]
    assert len(rows)==73
    check=compare(rows)
    (OUT/"overlap_audit.json").write_text(json.dumps(check,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    flagged={x[0] for x in check["historical_matches"]}
    for a,b in check["within_test_pairs"]:
        flagged.add(b)
    grades={k:(g,n) for g,notes in enumerate([GOOD,MILD,SEVERE]) for k,n in notes.items()}
    records=[];excluded=[]
    for i,r in enumerate(rows):
        if i not in grades or i in flagged:
            excluded.append(dict(index=i,source_id=r["id"],source_disease=r["condition"],
                reason="Near/equal-content match excluded conservatively." if i in flagged else excluded_reason(i,r["condition"])))
            continue
        grade,why=grades[i]
        records.append(dict(index=i,path=original(r).relative_to(ROOT).as_posix(),sha256=r["sha256"],grade=grade,split="test",
            source="Manalagi_v6_heldout_visible_grade_proxy",source_split="Test_Set/01_Original_Images",
            group_id="Manalagi_v6_specimen_"+r["id"],condition_id="Manalagi_Indonesia_orchard",
            source_disease=r["condition"],source_id=r["id"],source_member=r["folder"]+"/"+r["name"],
            grade_basis=why,grade_annotator="Codex visual audit, one model-independent review; observer-assigned external appearance grade. Original disease categories expert-validated by authors.",
            license="CC BY 4.0",doi="10.17632/9zgkwwv9j8.6"))
    ncols=6
    canvas=Image.new("RGB",(ncols*160,(len(records)+ncols-1)//ncols*170),"white");draw=ImageDraw.Draw(canvas)
    for j,r in enumerate(sorted(records,key=lambda item:(item["grade"],item["index"]))):
        with Image.open(ROOT/r['path']) as raw:
            img=raw.convert("RGB");img.thumbnail((154,145))
        x=j%ncols*160;y=j//ncols*170;canvas.paste(img,(x,y));draw.text((x,y+146),f"{r['index']:02d} | {['good','mild','severe'][r['grade']]} | {r['source_disease'][:7]}",fill="black")
    canvas.save(OUT/"reviewed_grade_panel.jpg",quality=95)
    result=dict(samples=records,counts={"test":dict(Counter(r["grade"] for r in records))},source_disease_by_grade={str(i):dict(Counter(r["source_disease"] for r in records if r["grade"]==i)) for i in [0,1,2]},
        source_doi="10.17632/9zgkwwv9j8.6",creators="M. Aldiki Febriantono, Abba Suganda Girsang, Suharjito Suharjito, Maria Susan Anggreainy",
        study="https://pmc.ncbi.nlm.nih.gov/articles/PMC13452390/",license_url="https://creativecommons.org/licenses/by/4.0/",
        policy="Whole source reserved for external test. Author-cropped and standardized original 1024x1024 photos, one specimen per image according to the authors; no additional processing. Source disease label is expert-validated. The three visible grades are our non-expert proxies. Rejected ambiguous visible surfaces retained in exclusion log. No model predictions used in review.",
        untested="Fruit color/maturity, depth of damage, hidden surfaces, commercial grades, mechanical scratch detection and prevalence are not validated by this panel.")
    (OUT/"excluded.json").write_text(json.dumps(excluded,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    (OUT/"manifest.json").write_text(json.dumps(result,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    print(result["counts"],check["reference_total"],len(flagged))


if __name__=="__main__":
    main()
