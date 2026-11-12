"""Independently compare exported YOLO point rows with every original author VIA polygon."""
from pathlib import Path
import json,hashlib
import sys
sys.dont_write_bytecode=True
root=Path(__file__).resolve().parent/"datasets"/"papple_visible_stage_seg_v1"
audit=json.loads((root/"conversion_audit.json").read_text(encoding="utf-8"))
docs={split:json.loads((root/"source"/(split+"_via_final.json")).read_text(encoding="utf-8"))
      for split in ("train","val")}
maximum_error=0.0
points=instances=changed=empty=0
split_counts={"train":{"images":0,"instances":0},"val":{"images":0,"instances":0}}
for image in audit["images"]:
    exported=Path(image["label_path"]).read_text(encoding="utf-8").splitlines()
    assert len(exported)==len(image["instances"])
    split_counts[image["split"]]["images"]+=1
    if not exported:empty+=1
    h,w=image["native_hw"]
    for line,provenance in zip(exported,image["instances"]):
        fields=line.split()
        assert int(fields[0])==provenance["label_id"]
        normalized=list(map(float,fields[1:]))
        record=docs[image["split"]][provenance["source_via_key"]]
        raw_regions=record["regions"]
        if not isinstance(raw_regions,list):raw_regions=list(raw_regions.values())
        original=raw_regions[provenance["source_region_index"]]
        assert original["region_attributes"]["Ripeness"]==provenance["label_string_as_provided"]
        shape=original["shape_attributes"]
        canonical=json.dumps(shape,sort_keys=True,separators=(",",":"),ensure_ascii=False).encode()
        assert hashlib.sha256(canonical).hexdigest()==provenance["source_shape_sha256"]
        xs,ys=shape["all_points_x"],shape["all_points_y"]
        assert len(normalized)==2*len(xs)==2*len(ys)
        for i,(x,y) in enumerate(zip(xs,ys)):
            err=max(abs(normalized[2*i]*w-x),abs(normalized[2*i+1]*h-y))
            maximum_error=max(maximum_error,err)
        assert all(0<=v<=1 for v in normalized)
        instances+=1;points+=len(xs)
        changed+=not provenance["exact_original_modal_geometry"]
        split_counts[image["split"]]["instances"]+=1
assert maximum_error<=1e-9 and instances==12014 and changed==18 and empty==27
report={"validation":"passed","method":"Independent exported label parse compared point-by-point to copied original author VIA regions",
    "images":len(audit["images"]),"instances":instances,"vertices":points,"empty_annotation_images":empty,
    "changed_author_polygons_vs_old_original_retained":changed,"split_counts":split_counts,
    "max_absolute_coordinate_roundtrip_error_pixels":maximum_error,"pixel_tolerance":1e-9,
    "point_count_order_class_string_and_source_shape_sha_verified":True,"GPU_used":False,
    "conversion_audit_sha256":hashlib.sha256((root/"conversion_audit.json").read_bytes()).hexdigest()}
with (root/"polygon_roundtrip_verification.json").open("x",encoding="utf-8") as f:
    json.dump(report,f,ensure_ascii=False,indent=2);f.write("\n")
print(json.dumps(report,ensure_ascii=False))
