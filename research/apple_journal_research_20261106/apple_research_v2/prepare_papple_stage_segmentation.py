"""Prepare author-labelled PApple visible stage segmentation, retaining provenance.

All writes are restricted to a new F-drive dataset. Author west/east splits,
polygons and ripe/unripe strings remain unchanged. No test data or GPU is used.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import io
import json
import math
import os
from pathlib import Path
import shutil
import sys

sys.dont_write_bytecode=True
CLASSES=["ripe","unripe"]
REQUIRED=["papple_maturity_candidate_manifest.json","aligned_image_manifest.json","candidate_validation.json",
          "linkage_details.json","component_audit_v2.json","label_consistency_audit.json",
          "identifier_scope_reconciliation.json","primary_provenance.json",
          "train_via_final.json","val_via_final.json","kaggle_v4_metadata.json","download_receipt.json",
          "remote_image_checksum_samples_v2.json","alignment_audit.json"]


def require(ok,message):
    if not ok: raise ValueError(message)


def sha(path):
    with Path(path).open("rb") as f:return hashlib.file_digest(f,"sha256").hexdigest()


def shape_sha(shape):
    return hashlib.sha256(json.dumps(shape,sort_keys=True,separators=(",",":"),ensure_ascii=False).encode()).hexdigest()


def save(path,value):
    with Path(path).open("x",encoding="utf-8") as f:
        json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False);f.write("\n")


def regions(record):
    value=record["regions"]
    return value if isinstance(value,list) else list(value.values())


def geometry(region):
    s=region["shape_attributes"]
    return tuple(s["all_points_x"]),tuple(s["all_points_y"])


def fpath(value):
    p=Path(value).resolve()
    require(p.drive.upper()=="F:",f"F drive required: {p}")
    require(not any(part.lower() in ("test","test_resize","tests") for part in p.parts),"Test path forbidden")
    return p


def cpu_runtime(output):
    runtime=output/"verification_runtime"
    for folder in ("tmp","yolo","matplotlib","torch","hf","cuda","torch_extensions","pip","cache"):
        (runtime/folder).mkdir(parents=True,exist_ok=True)
    os.environ.update(PYTHONDONTWRITEBYTECODE="1",CUDA_VISIBLE_DEVICES="",YOLO_AUTOINSTALL="False",YOLO_OFFLINE="true",
        YOLO_CONFIG_DIR=str(runtime/"yolo"),MPLCONFIGDIR=str(runtime/"matplotlib"),TORCH_HOME=str(runtime/"torch"),
        HF_HOME=str(runtime/"hf"),HF_HUB_OFFLINE="1",TRANSFORMERS_OFFLINE="1",
        CUDA_CACHE_PATH=str(runtime/"cuda"),TORCH_EXTENSIONS_DIR=str(runtime/"torch_extensions"),
        PIP_CACHE_DIR=str(runtime/"pip"),XDG_CACHE_HOME=str(runtime/"cache"),TMP=str(runtime/"tmp"),TEMP=str(runtime/"tmp"))


def full_loader(output,audit):
    cpu_runtime(output)
    import torch
    torch.set_num_threads(2)
    import cv2
    cv2.setNumThreads(1)
    import ultralytics
    from ultralytics.cfg import get_cfg
    from ultralytics.data.dataset import YOLODataset
    require(ultralytics.__version__=="8.4.115","Unexpected Ultralytics version")
    names=dict(enumerate(CLASSES))
    entries={r["image_path"]:r for r in audit["images"]}
    report={"ultralytics_version":ultralytics.__version__,"GPU_used":False,
        "method":"YOLODataset(task=segment) every image __getitem__, augment=False",
        "imgsz":640,"mask_ratio":4,"overlap_mask":False,"splits":{}}
    for split in ("train","val"):
        dataset=YOLODataset(img_path=str(output/"images"/split),imgsz=640,batch_size=4,augment=False,
            cache=False,hyp=get_cfg(overrides={"mask_ratio":4,"overlap_mask":False}),rect=False,stride=32,pad=0.0,
            data={"names":names,"nc":2,"channels":3},task="segment",prefix=split+": ")
        expected=audit["counts"][split]
        require(len(dataset)==expected["images"],"Loader lost images")
        require(sum(len(x["cls"]) for x in dataset.labels)==expected["instances"],"Loader removed author instances")
        counts=Counter();zero=[];empty=0;mask_shapes=Counter()
        for i in range(len(dataset)):
            sample=dataset[i]
            entry=entries[str(Path(sample["im_file"]).resolve())]
            labels=[int(v) for v in sample["cls"].reshape(-1).tolist()]
            require(labels==[r["label_id"] for r in entry["instances"]],"Loaded instance/class ordering changed")
            masks=sample["masks"]
            require(masks.ndim==3 and masks.shape[0]==len(labels),"Mask count differs from author instances")
            mask_shapes[str(list(masks.shape[1:]))]+=1
            counts.update(names[v] for v in labels)
            if not labels:empty+=1
            else:
                areas=masks.flatten(1).sum(1)
                for j,area in enumerate(areas):
                    if area.item()==0:
                        zero.append({"image":sample["im_file"],"source_region_index":entry["instances"][j]["source_region_index"],
                                     "label":names[labels[j]],"source_shape_sha256":entry["instances"][j]["source_shape_sha256"]})
            if (i+1)%250==0:
                print(json.dumps({"phase":"CPU_loader_progress","split":split,"images_checked":i+1,"total":len(dataset)}),flush=True)
        require(dict(counts)==expected["class_counts"],"Loader class counts changed")
        require(empty==expected["empty_annotation_images"],"Loader empty-image counts changed")
        report["splits"][split]={"images":len(dataset),"instances":sum(counts.values()),"class_counts":dict(counts),
            "empty_annotation_images":empty,"mask_shapes":dict(mask_shapes),"zero_rasterized_mask_count":len(zero),
            "zero_rasterized_masks":zero}
    report["zero_mask_policy"]="No author polygon or class is dropped; any zero rasterized masks are disclosed for training-configuration review"
    save(output/"full_loader_verification.json",report)
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-audit",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    args=p.parse_args()
    project=Path(__file__).resolve().parent
    require(project.drive.upper()=="F:","Script must live on F")
    source,output=fpath(args.source_audit),fpath(args.output)
    require(source.is_relative_to(project) and output.is_relative_to(project),"Source audit/output must stay in project")
    require(not output.exists(),"Output exists; refusing overwrite")
    documents={name:json.loads((source/name).read_text(encoding="utf-8-sig")) for name in REQUIRED}
    input_hashes={str(source/name):sha(source/name) for name in REQUIRED}
    candidate=documents["papple_maturity_candidate_manifest.json"]
    validation=documents["candidate_validation.json"]
    aligned=documents["aligned_image_manifest.json"]
    consistency=documents["label_consistency_audit.json"]
    identity=documents["identifier_scope_reconciliation.json"]
    require(validation["status"]=="passed" and validation["candidate_sha256"]==sha(source/"papple_maturity_candidate_manifest.json"),"Candidate validation/hash mismatch")
    require(candidate["class_names"]==CLASSES and candidate["class_mapping"]=={"ripe":0,"unripe":1},"Original class mapping mismatch")
    require(candidate["record_count"]==len(candidate["records"])==12014,"Unexpected author region count")
    require(len(aligned)==3118,"Unexpected aligned image count")
    require(identity["composite_counts"]["shared"]==309 and consistency["IDs_with_both_ripe_and_unripe"]==134,"Identity/consistency audit changed")
    records=defaultdict(list)
    for row in candidate["records"]:
        require(row["split"] in ("train","val"),"Unexpected candidate split")
        records[(row["split"],row["source_image_id"])].append(row)
    image_keys={(r["author_split"],r["filename"]) for r in aligned}
    require(len(image_keys)==len(aligned) and set(records)<=image_keys,"Candidate image alignment mismatch")
    require({r["sha256"] for r in aligned if r["author_split"]=="train"}.isdisjoint(
            {r["sha256"] for r in aligned if r["author_split"]=="val"}),"Exact image hash crosses split")
    require({r["source_frame_proxy"] for r in aligned if r["author_split"]=="train"}.isdisjoint(
            {r["source_frame_proxy"] for r in aligned if r["author_split"]=="val"}),"Original frame proxy crosses split")
    source_docs={split:documents[split+"_via_final.json"] for split in ("train","val")}
    for split in ("train","val"):
        require(candidate["source_files_sha256"][split]==sha(source/(split+"_via_final.json")),"Author annotation hash changed")
    old_val_sources=[r for r in documents["component_audit_v2.json"]["sources"]
                     if Path(r["path"]).name=="via_region_data_instance.json" and Path(r["path"]).parent.name=="val"]
    require(len(old_val_sources)==1,"Original val polygon provenance unresolved")
    old_val_path=fpath(old_val_sources[0]["path"])
    require(sha(old_val_path)==old_val_sources[0]["sha256"],"Original val source hash changed")
    input_hashes[str(old_val_path)]=sha(old_val_path)
    old_val=json.loads(old_val_path.read_text(encoding="utf-8-sig"))
    old_val={r["filename"]:r for r in old_val.values()}
    new_val={r["filename"]:r for r in source_docs["val"].values()}
    omitted=[]
    for name,old in old_val.items():
        original,new=regions(old),regions(new_val[name])
        if len(original)>len(new):
            new_geometry={geometry(r) for r in new}
            unmatched=[(i,r) for i,r in enumerate(original) if geometry(r) not in new_geometry]
            require(len(unmatched)==len(original)-len(new),"Missing/modified regions ambiguous in an omission image")
            for index,reg in unmatched:
                omitted.append({"filename":name,"split":"val","original_region_index":index,
                    "original_apple_ID":reg["region_attributes"].get("apple_ID"),"source_annotation_path":str(old_val_path),
                    "source_annotation_sha256":input_hashes[str(old_val_path)],"original_region":reg,
                    "original_shape_sha256":shape_sha(reg["shape_attributes"]),"ripeness_label":None,
                    "reason":"Original visible polygon absent from author maturity annotations; no stage label inferred"})
    require(len(omitted)==14,"Unexpected original val omission count")
    output.mkdir(parents=True,exist_ok=False)
    (output/"source").mkdir()
    for name in REQUIRED:shutil.copyfile(source/name,output/"source"/name)
    shutil.copyfile(old_val_path,output/"source"/"original_val_visible_regions.json")
    save(output/"ignore_candidates_provenance.json",{"count":14,"candidates":omitted,
        "enabled_in_standard_yolo_loader_or_evaluation":False,
        "policy":"Provenance only. No labels manufactured, no ignore threshold selected, and no missing polygon silently treated as a new negative truth. Default YOLO evaluator will not use this sidecar.",
        "limitation":"A detection at an omitted original val fruit may count as a false positive under author-label evaluation; review a predeclared ignore protocol separately."})
    audit={"schema":"papple_author_visible_stage_segmentation_v1","created_utc":datetime.now(timezone.utc).isoformat(),
        "converter_sha256":sha(__file__),"source_audit":str(source),"source_input_hashes":input_hashes,
        "class_names":CLASSES,"class_mapping":candidate["class_mapping"],"label_semantics":candidate["label_semantics"],
        "source_license":candidate["source_license"],"data_version":candidate["data_version"],
        "polygon_semantics":"Exact new author VIA polygons; 11996 exactly match unique original visible polygons, 18 have changed geometry and are retained as supplied; not amodal/hidden-surface truth",
        "coordinate_policy":"Original point order and endpoints unchanged; divide x by native width and y by native height, serialize with 17 significant digits; no clipping, repairs or substitutions",
        "split_policy":dict(candidate["split_policy"],independent_group_holdout=False,
            derived_composite_fruit_IDs_shared_across_splits=309,raw_IDs_shared_across_splits=266,
            label_mixed_derived_fruit_ID_count=134,contradictory_view_policy="Keep author labels; no contradiction-based sample exclusion"),
        "limitations":["Author surface-color/capture-date proxy is not physiological maturity or commercial grade",
            "West/east split is development with 309 shared derived composite identities; namespace is a local protocol, not independently verified physical-fruit identity",
            "134 derived identities have both labels across views; labels are unchanged and not silently reconciled",
            "14 old val visible regions lack new author stage annotations; ignored only in a provenance sidecar, not automatically excluded by YOLO",
            "Only four sampled remote MD5 checks were verified; all local image SHA256 hashes are checked, not every remote image",
            "Author-labelled subset excludes 91 unavailable OpenAccess train/val images; this is not the full published Fuji collection"],
        "test_data_read":False,"training_performed":False,"diameter_regression_enabled":False,
        "counts":{},"images":[],"changed_author_geometry_preserved":0}
    from PIL import Image
    for split in ("train","val"):
        (output/"images"/split).mkdir(parents=True);(output/"labels"/split).mkdir(parents=True)
        counts=Counter();empty=0;closures=Counter();zero_area=0
        subset=[r for r in aligned if r["author_split"]==split]
        for number,item in enumerate(subset):
            name=item["filename"]
            require(Path(name).name==name and Path(name).suffix.lower()==".png","Unsafe image filename")
            src=fpath(item["image_path"])
            raw=src.read_bytes()
            require(len(raw)==item["bytes"] and hashlib.sha256(raw).hexdigest()==item["sha256"],"Aligned source image changed")
            with Image.open(io.BytesIO(raw)) as im:
                w,h=im.size;im.verify()
            require([w,h]==item["native_wh"]==[1300,1300],"Native dimensions differ from audit")
            annotation=source_docs[split][item["source_via_key"]]
            require(annotation["filename"]==name,"VIA image mapping mismatch")
            original_regions=regions(annotation)
            objects=sorted(records.get((split,name),[]),key=lambda r:r["source_region_index"])
            require([r["source_region_index"] for r in objects]==list(range(len(original_regions))),"Candidate omits an author instance")
            lines=[];instances=[]
            for row in objects:
                region=original_regions[row["source_region_index"]];shape=region["shape_attributes"]
                label=region["region_attributes"]["Ripeness"]
                require(label==row["label_string_as_provided"]==CLASSES[row["label"]],"Author label mismatch")
                require(shape_sha(shape)==row["source_shape_sha256"],"Author polygon changed")
                require(shape["name"]=="polygon","Unexpected author shape type")
                x,y=shape["all_points_x"],shape["all_points_y"]
                require(len(x)==len(y) and len(x)>=3,"Invalid point count")
                require(all(isinstance(v,(int,float)) and not isinstance(v,bool) and math.isfinite(v) for v in x+y),"Nonfinite source vertex")
                require(all(0<=xx<=w and 0<=yy<=h for xx,yy in zip(x,y)),"Source coordinate outside image")
                require(len(set(zip(x,y)))>=3 and min(x)<max(x) and min(y)<max(y),"Degenerate author bbox")
                area=abs(sum(x[i]*y[(i+1)%len(x)]-x[(i+1)%len(x)]*y[i] for i in range(len(x))))*.5
                zero_area+=int(area==0)
                explicit=x[0]==x[-1] and y[0]==y[-1];closures["explicit" if explicit else "implicit_polygon"]+=1
                normalized=[v for xx,yy in zip(x,y) for v in (xx/w,yy/h)]
                tokens=[format(v,".17g") for v in normalized]
                require(all(0<=float(v)<=1 for v in tokens),"Normalized coordinates outside [0,1]")
                lines.append(str(row["label"])+" "+" ".join(tokens))
                counts[label]+=1
                audit["changed_author_geometry_preserved"]+=int(not row["exact_original_modal_geometry"])
                instances.append({"source_region_index":row["source_region_index"],"source_via_key":row["source_via_key"],
                    "source_shape_sha256":row["source_shape_sha256"],"label_id":row["label"],"label_string_as_provided":label,
                    "exact_original_modal_geometry":row["exact_original_modal_geometry"],
                    "matched_original_modal_region_index":row["matched_original_modal_region_index"],
                    "matched_original_apple_ID":row["matched_original_apple_ID"],
                    "original_vertex_count":len(x),"explicitly_closed":explicit})
            if not objects:empty+=1
            target=output/"images"/split/name;label_path=output/"labels"/split/(Path(name).stem+".txt")
            with target.open("xb") as f:f.write(raw)
            with label_path.open("x",encoding="utf-8",newline="\n") as f:f.write("\n".join(lines)+("\n" if lines else ""))
            require(sha(target)==item["sha256"],"Copied image mismatch")
            audit["images"].append({"split":split,"image_path":str(target),"label_path":str(label_path),
                "source_image_path":str(src),"source_image_id":name,"source_frame_proxy":item["source_frame_proxy"],
                "native_hw":[h,w],"image_sha256":item["sha256"],"label_sha256":sha(label_path),
                "annotation_path":str(output/"source"/(split+"_via_final.json")),
                "annotation_sha256":candidate["source_files_sha256"][split],"instances":instances})
            if (number+1)%250==0:
                print(json.dumps({"phase":"copy_convert","split":split,"images_completed":number+1,"total":len(subset)}),flush=True)
        audit["counts"][split]={"images":len(subset),"instances":sum(counts.values()),"class_counts":dict(counts),
            "empty_annotation_images":empty,"closure_counts":dict(closures),"zero_shoelace_area_regions":zero_area}
        require(dict(counts)==validation["class_counts"][split],"Converted class counts differ")
        require(empty=={"train":14,"val":13}[split],"Empty annotation image count changed")
    require(audit["changed_author_geometry_preserved"]==18,"Changed author polygon count mismatch")
    yaml_text="path: "+json.dumps(str(output).replace("\\","/"),ensure_ascii=False)+"\ntrain: images/train\nval: images/val\nnames:\n  0: ripe\n  1: unripe\n"
    with (output/"data.yaml").open("x",encoding="utf-8") as f:f.write(yaml_text)
    audit["data_yaml_sha256"]=sha(output/"data.yaml")
    save(output/"conversion_audit.json",audit)
    loader=full_loader(output,audit)
    for entry in audit["images"]:
        require(sha(entry["source_image_path"])==entry["image_sha256"],"Original source image changed")
        require(sha(entry["image_path"])==entry["image_sha256"] and sha(entry["label_path"])==entry["label_sha256"],"Loader modified a copied image or label")
    for name,digest in input_hashes.items():require(sha(name)==digest,"Input audit/annotation changed")
    ready={"data_yaml":str(output/"data.yaml"),"conversion_audit_sha256":sha(output/"conversion_audit.json"),
        "full_loader_verification_sha256":sha(output/"full_loader_verification.json"),
        "counts":audit["counts"],"all_original_sources_and_copied_images_labels_unchanged":True,
        "source_image_bytes_copied":sum(r["bytes"] for r in aligned),
        "author_polygons_changed_vs_original_preserved":18,"old_val_omissions_documented_without_stage_labels":14,
        "independent_group_holdout":False,"GPU_used":False,"training_performed":False,
        "zero_rasterized_masks":{s:loader["splits"][s]["zero_rasterized_mask_count"] for s in ("train","val")}}
    save(output/"DATA_READY.json",ready)
    print(json.dumps(ready,ensure_ascii=False),flush=True)


if __name__=="__main__":main()
