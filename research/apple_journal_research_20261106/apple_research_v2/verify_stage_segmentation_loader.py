"""CPU-only full YOLODataset verification of exported stage segmentation."""
from pathlib import Path
import os, sys, json, hashlib
from collections import Counter
sys.dont_write_bytecode=True
root=Path(__file__).resolve().parent/"datasets"/"mendeley_stage_seg_v1"
runtime=root/"verification_runtime"
os.environ.update(PYTHONDONTWRITEBYTECODE="1",CUDA_VISIBLE_DEVICES="",YOLO_AUTOINSTALL="False",
    YOLO_CONFIG_DIR=str(runtime/"yolo"),MPLCONFIGDIR=str(runtime/"matplotlib"),
    TORCH_HOME=str(runtime/"torch"),TMP=str(runtime/"tmp"),TEMP=str(runtime/"tmp"))
import torch
torch.set_num_threads(2)
import cv2
cv2.setNumThreads(1)
import ultralytics
from ultralytics.cfg import get_cfg
from ultralytics.data.dataset import YOLODataset
audit=json.loads((root/"conversion_audit.json").read_text(encoding="utf-8"))
names={i:v for i,v in enumerate(audit["class_names"])}
summary={"ultralytics_version":ultralytics.__version__,"GPU_used":False,
    "method":"YOLODataset(task=segment), every image __getitem__, augment=False; all returned masks and labels checked",
    "imgsz":640,"mask_ratio":4,"overlap_mask":False,"splits":{}}
for split in ("train","val"):
    cfg=get_cfg(overrides={"mask_ratio":4,"overlap_mask":False})
    dataset=YOLODataset(img_path=str(root/"images"/split),imgsz=640,batch_size=4,augment=False,
        cache=False,hyp=cfg,rect=False,stride=32,pad=0.0,
        data={"names":names,"nc":3,"channels":3},task="segment",prefix=split+": ")
    assert len(dataset)==audit["counts"][split]["images"]
    assert sum(len(x["cls"]) for x in dataset.labels)==audit["counts"][split]["instances"]
    classes=Counter(); zero_masks=[]; mask_shapes=Counter(); instance_count=0
    for i in range(len(dataset)):
        sample=dataset[i]
        labels=[int(v) for v in sample["cls"].reshape(-1).tolist()]
        masks=sample["masks"]
        assert masks.ndim==3 and masks.shape[0]==len(labels)
        assert sample["img"].ndim==3 and sample["img"].shape[0]==3
        classes.update(names[k] for k in labels)
        instance_count+=len(labels)
        mask_shapes[str(list(masks.shape[1:]))]+=1
        areas=masks.reshape(len(labels),-1).sum(1)
        for j,area in enumerate(areas):
            if area.item()==0:
                zero_masks.append({"image":sample["im_file"],"instance_index":j,"label":names[labels[j]]})
    assert instance_count==audit["counts"][split]["instances"]
    assert dict(classes)==audit["counts"][split]["class_counts"]
    summary["splits"][split]={"images":len(dataset),"instances":instance_count,
        "class_counts":dict(classes),"mask_shapes":dict(mask_shapes),"zero_rasterized_masks":zero_masks}
    print(json.dumps({"split":split,**summary["splits"][split]},ensure_ascii=False),flush=True)
for entry in audit["images"]:
    for path,key in ((entry["image_path"],"image_sha256"),(entry["label_path"],"label_sha256")):
        digest=hashlib.sha256(Path(path).read_bytes()).hexdigest()
        assert digest==entry[key],f"Loader modified data: {path}"
summary["all_source_images_and_labels_unchanged"]=True
summary["conversion_audit_sha256"]=hashlib.sha256((root/"conversion_audit.json").read_bytes()).hexdigest()
output=root/"full_loader_verification.json"
with output.open("x",encoding="utf-8") as f:
    json.dump(summary,f,ensure_ascii=False,indent=2);f.write("\n")
print(json.dumps({"verification_report":str(output),"GPU_used":False},ensure_ascii=False))
