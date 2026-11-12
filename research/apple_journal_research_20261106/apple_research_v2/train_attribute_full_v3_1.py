"""Frozen COCO ResNet50 GT-ROI full baselines: one feature pass, CPU heads.
20 epochs, three seeds, two loss arms; fixed final epoch, non-independent dev.
No test data, no physiological maturity/commodity grade/detector FPS claims.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
import time
import traceback
sys.dont_write_bytecode=True
os.environ.update(PYTHONDONTWRITEBYTECODE="1",HF_HUB_OFFLINE="1",TRANSFORMERS_OFFLINE="1",YOLO_AUTOINSTALL="False")
WEIGHT_SHA="bf2d0c1efbc936eeee2bc95a48e80ebc86b891f61b0106485937fc29f9315fc0"
SCHEMA="attribute_full_frozen_resnet50_v3_1"
ROLE="development_non_independent"
SEEDS=(20261106,20261107,20261108)
ARMS=("none","inverse-frequency")
LIMITS="Dataset-specific visible ROI labels; no physiological maturity, internal quality, commodity grade, whole-fruit quality, independent generalization, or whole-image detection claim. Inputs are ground-truth annotated ROIs."
TRANSFORM={"crop":"floor x1/y1, ceil x2/y2; bounds checked; no clipping","resize":[224,224],"interpolation":"PIL bilinear","mean":[.485,.456,.406],"std":[.229,.224,.225],"augmentation":None,"dtype":"float32"}
RECIPE={"epochs":20,"batch_size":32,"learning_rate":.001,"weight_decay":.0001,"optimizer":"AdamW","scheduler":None,"selection":"fixed_final_epoch_20","early_stopping":False,"seed_order":"random.Random(seed + epoch).shuffle(all_train_indices); epoch starts at 1","head":"Linear(2048,class_count)","initialization":"fresh Linear after torch.manual_seed(seed); no pilot warm start","dev_loss_weighting":"none","dtype":"FP32","device":"cpu"}
def require(condition,message):
    if not condition: raise ValueError(message)
def now(): return datetime.now(timezone.utc).isoformat()
def sha(path):
    with Path(path).open("rb") as stream: return hashlib.file_digest(stream,"sha256").hexdigest()
def canonical(value): return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False,allow_nan=False,separators=(",",":")).encode()).hexdigest()
def write_json(path,value):
    path=Path(path); tmp=path.with_name(path.name+f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding="utf-8"); os.replace(tmp,path)
def save_tensor(path,value):
    import torch
    path=Path(path); tmp=path.with_name(path.name+f".tmp.{os.getpid()}")
    torch.save(value,tmp); torch.load(tmp,map_location="cpu",weights_only=True); os.replace(tmp,path)
def fingerprint_state(state):
    h=hashlib.sha256()
    for name,tensor in sorted(state.items()):
        data=tensor.detach().cpu().contiguous().numpy()
        h.update(name.encode()); h.update(str(data.dtype).encode()); h.update(str(data.shape).encode()); h.update(memoryview(data).cast("B"))
    return h.hexdigest()
def fingerprint(module): return fingerprint_state(module.state_dict())
def configure_runtime(output,device):
    output=Path(output).resolve(); require(output.drive.lower()=="f:" and output.is_relative_to(Path(__file__).resolve().parents[1]),"All outputs/caches must stay in this migrated F project")
    output.mkdir(parents=True,exist_ok=True)
    for variable in ("TEMP","TMP","TORCH_HOME","HF_HOME","MPLCONFIGDIR","XDG_CACHE_HOME","CUDA_CACHE_PATH","TORCH_EXTENSIONS_DIR","NUMBA_CACHE_DIR"):
        path=output/"runtime"/variable.lower(); path.mkdir(parents=True,exist_ok=True); os.environ[variable]=str(path)
    os.environ["CUBLAS_WORKSPACE_CONFIG"]=":4096:8"
    if device=="cpu": os.environ["CUDA_VISIBLE_DEVICES"]=""
def torch_setup(device):
    import torch
    torch.set_num_threads(4); torch.set_num_interop_threads(1); torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False; torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    if device.startswith("cuda"): require(torch.cuda.is_available(),"CUDA unavailable; no fallback")
    return torch
def resolve_path(value,parent):
    path=Path(value); return (path if path.is_absolute() else parent/path).resolve()
def manifest_metadata(document,parent,allow):
    require(allow,"Explicit --allow-non-independent-development required")
    require(isinstance(document.get("label_semantics"),str) and document["label_semantics"].strip(),"Missing real label_semantics")
    policy=document.get("split_policy",{})
    require(policy.get("evaluation_role")==ROLE,"split_policy.evaluation_role must be development_non_independent")
    require(policy.get("independent_group_holdout") is not True,"Cannot claim independent grouping")
    rows=document.get("records"); require(isinstance(rows,list) and rows,"Missing real records")
    require(all(isinstance(r,dict) and r.get("split") in ("train","dev","val") for r in rows),"Test/unknown split refused before source reads")
    names=document.get("class_names")
    require(isinstance(names,list) and len(names)>=2 and all(isinstance(x,str) and x.strip() for x in names) and len(set(names))==len(names),"Invalid class_names")
    seen=set(); memberships={k:defaultdict(set) for k in ("source_image_id","group_id","image_path")}; normalized=[]
    for index,record in enumerate(rows):
        for field in ("image_path","image_sha256","annotation_path","annotation_sha256","bbox_xyxy","label","source_image_id","group_id","split"):
            require(record.get(field) is not None,f"Record {index} missing {field}")
        require(all(isinstance(record[k],str) and record[k].strip() for k in ("source_image_id","group_id")),"Invalid identity")
        for field in ("image_sha256","annotation_sha256"):
            value=record[field]; require(isinstance(value,str) and len(value)==64 and all(c in "0123456789abcdef" for c in value.lower()),f"Invalid {field}")
        label=record["label"]
        if isinstance(label,int) and not isinstance(label,bool):
            require(0<=label<len(names),"Invalid label"); label=names[label]
        require(label in names,"Unknown label")
        bbox=record["bbox_xyxy"]
        require(isinstance(bbox,list) and len(bbox)==4 and all(isinstance(v,(int,float)) and not isinstance(v,bool) and math.isfinite(v) for v in bbox),"Invalid bbox")
        x1,y1,x2,y2=map(float,bbox); require(0<=x1<x2 and 0<=y1<y2,"Invalid/empty bbox")
        row=dict(record,record_index=index,split="train" if record["split"]=="train" else "dev",label=label,label_id=names.index(label),bbox_xyxy=[x1,y1,x2,y2])
        for field in ("image_path","annotation_path"): row[field]=str(resolve_path(row[field],parent))
        identity=(row["image_path"],tuple(row["bbox_xyxy"])); require(identity not in seen,"Duplicate ROI"); seen.add(identity)
        for key,values in memberships.items(): values[row[key]].add(row["split"])
        row["record_id"]=canonical({k:row[k] for k in ("source_image_id","image_sha256","annotation_sha256","bbox_xyxy","label_id","group_id","split")}); normalized.append(row)
    for key,values in memberships.items(): require(all(len(v)==1 for v in values.values()),f"{key} crosses split")
    require(len({r["record_id"] for r in normalized})==len(normalized),"Duplicate stable record identity")
    for split in ("train","dev"): require(set(r["label"] for r in normalized if r["split"]==split)==set(names),f"Every class needs real {split} support")
    return normalized,names
def preflight(args):
    path=args.manifest.resolve(); document=json.loads(path.read_text(encoding="utf-8-sig")); rows,names=manifest_metadata(document,path.parent,args.allow_non_independent_development)
    weights=args.weights.resolve(); receipts={str(path):sha(path),str(weights):sha(weights)}; require(receipts[str(weights)]==WEIGHT_SHA,"Exact official COCO weights required")
    from PIL import Image
    sizes={}; content_splits=defaultdict(set)
    for row in rows:
        for field,hfield in (("image_path","image_sha256"),("annotation_path","annotation_sha256")):
            name=row[field]
            if name not in receipts: receipts[name]=sha(name)
            require(receipts[name]==row[hfield].lower(),f"Input hash mismatch: {name}")
        name=row["image_path"]
        if name not in sizes:
            with Image.open(name) as im: sizes[name]=im.size; im.verify()
        w,h=sizes[name]; require(row["bbox_xyxy"][2]<=w and row["bbox_xyxy"][3]<=h,f"Out-of-bounds ROI {row['record_index']}")
        content_splits[receipts[name]].add(row["split"])
    require(all(len(v)==1 for v in content_splits.values()),"Exact image content crosses split")
    counts={s:{n:sum(r["split"]==s and r["label"]==n for r in rows) for n in names} for s in ("train","dev")}
    identity={"schema":SCHEMA,"manifest_sha256":receipts[str(path)],"weights_sha256":WEIGHT_SHA,"transform":TRANSFORM,"class_names":names,"records":[{k:r[k] for k in ("record_index","record_id","label_id","split")} for r in rows],"script_sha256":sha(__file__)}
    return document,rows,names,receipts,counts,identity
def verify_inputs(receipts):
    after={path:sha(path) for path in receipts}; require(after==receipts,"Inputs changed"); return after
def class_weights(counts,names,mode):
    require(mode in ARMS,"Unknown weighting"); require(set(counts)==set(names) and all(isinstance(counts[n],int) and counts[n]>0 for n in names),"Weights need actual positive train counts")
    return None if mode=="none" else [sum(counts.values())/(len(names)*counts[n]) for n in names]
def metric_summary(confusion,nll_sum,names):
    size=len(names); require(len(confusion)==size and all(len(r)==size for r in confusion),"Bad confusion shape")
    support=[sum(r) for r in confusion]; total=sum(support); require(total>0,"Empty evaluation"); rows=[]
    for i,name in enumerate(names):
        tp=confusion[i][i]; predicted=sum(confusion[j][i] for j in range(size)); precision=tp/predicted if predicted else 0.; recall=tp/support[i] if support[i] else 0.; f1=2*precision*recall/(precision+recall) if precision+recall else 0.
        rows.append({"class_name":name,"support":support[i],"predicted_count":predicted,"precision":precision,"recall":recall,"F1":f1})
    return {"n_ROIs":total,"unweighted_NLL":nll_sum/total,"accuracy":sum(confusion[i][i] for i in range(size))/total,"macro_F1":sum(r["F1"] for r in rows)/size,"balanced_accuracy":sum(r["recall"] for r in rows)/size,"per_class":rows,"confusion_matrix":confusion,"confusion_orientation":"rows=true, columns=predicted","class_order":names,"undefined_precision_rule":"0 if no predicted examples"}
def load_cache(cache,identity,torch):
    complete=json.loads((cache/"complete.json").read_text(encoding="utf-8")); require(complete["identity"]==identity,"Feature cache identity differs")
    require(sha(cache/"features.pt")==complete["features_sha256"],"Feature cache bytes changed"); payload=torch.load(cache/"features.pt",map_location="cpu",weights_only=True)
    require(payload["identity"]==identity and payload["record_ids"]==[r["record_id"] for r in identity["records"]],"Cached IDs differ")
    x,y=payload["features"],payload["labels"]; require(tuple(x.shape)==(len(identity["records"]),2048) and x.dtype==torch.float32 and torch.isfinite(x).all().item(),"Invalid features")
    require(y.tolist()==[r["label_id"] for r in identity["records"]],"Cached labels differ")
    require(complete["frozen_backbone_unchanged"] and complete["input_sha256_before"]==complete["input_sha256_after"],"Unverified cache")
    return payload,complete
def finalize_feature_cache(output,state,rows,identity,receipts,counts,document,runtime,before,after,no_grad):
    """Commit the actual shard assembly only after validating bytes, IDs and values.

    Kept as the production finalization path so CPU regression tests can exercise
    real shard I/O and the complete marker without GPU or synthetic research runs.
    """
    import torch
    output=Path(output); require(bool(rows) and bool(state["shards"]),"Cannot finalize empty cache")
    require(before==after and no_grad,"Backbone changed/received gradients")
    require([r["record_id"] for r in rows]==[r["record_id"] for r in identity["records"]],"Finalization identity IDs differ")
    require([r["label_id"] for r in rows]==[r["label_id"] for r in identity["records"]],"Finalization identity labels differ")
    pieces=[]; cursor=0
    for item in state["shards"]:
        require(isinstance(item["n"],int) and item["n"]>0 and item["offset"]==cursor,"Invalid/noncontiguous shard metadata")
        require(Path(item["file"]).name==item["file"],"Shard filename must stay inside cache directory")
        shard_path=output / item["file"]
        require(sha(shard_path)==item["sha256"],"Finalization shard bytes changed")
        piece=torch.load(shard_path,map_location="cpu",weights_only=True)
        expected=rows[cursor:cursor+item["n"]]
        require(len(expected)==item["n"],"Shard exceeds manifest")
        require(piece["record_indices"]==list(range(cursor,cursor+item["n"])) and piece["record_ids"]==[r["record_id"] for r in expected],"Finalization shard order/IDs differ")
        x,y=piece["features"],piece["labels"]
        require(tuple(x.shape)==(item["n"],2048) and x.dtype==torch.float32 and torch.isfinite(x).all().item(),"Invalid/nonfinite shard features")
        require(y.dtype==torch.int64 and tuple(y.shape)==(item["n"],) and y.tolist()==[r["label_id"] for r in expected],"Finalization shard labels differ")
        pieces.append(piece); cursor+=item["n"]
    require(cursor==len(rows),"Finalization missing records")
    payload={"identity":identity,"features":torch.cat([p["features"] for p in pieces]),"labels":torch.cat([p["labels"] for p in pieces]),"record_ids":[i for p in pieces for i in p["record_ids"]]}
    require(sha(__file__)==identity["script_sha256"],"Source changed")
    input_after=verify_inputs(receipts); save_tensor(output/"features.pt",payload)
    complete={"schema":SCHEMA,"phase":"complete","completed_utc":now(),"identity":identity,"runtime":runtime,"counts":counts,"features_sha256":sha(output/"features.pt"),"backbone_tensor_sha256_before":before,"backbone_tensor_sha256_after":after,"frozen_backbone_unchanged":True,"input_sha256_before":receipts,"input_sha256_after":input_after,"evaluation_role":ROLE,"independent_test_claimed":False,"test_used":False,"label_semantics":document["label_semantics"],"semantic_limits":LIMITS,"elapsed_scope":"one real feature extraction for every train/development annotated ROI; not online inference speed"}
    write_json(output/"complete.json",complete)
    return complete

def extract(args,document,rows,names,receipts,counts,identity):
    output=args.output.resolve(); require(not output.exists() or args.resume,"Existing output needs --resume")
    configure_runtime(output,args.device); torch=torch_setup(args.device)
    import numpy as np
    import torchvision
    import PIL
    from PIL import Image
    from torchvision.models import resnet50
    from torchvision.models._utils import IntermediateLayerGetter
    from torchvision.ops.misc import FrozenBatchNorm2d
    from torchvision.transforms import functional as TF
    runtime={"torch":str(torch.__version__),"torchvision":str(torchvision.__version__),"numpy":np.__version__,"Pillow":PIL.__version__,"device":args.device,"feature_batch_size":args.feature_batch_size}
    if (output/"complete.json").exists():
        _,complete=load_cache(output,identity,torch); require(complete["runtime"]==runtime,"Cache runtime differs"); print("Feature cache complete and verified",flush=True); return
    state_path=output/"extract_state.json"
    if state_path.exists():
        state=json.loads(state_path.read_text(encoding="utf-8")); require(state["identity"]==identity and state["runtime"]==runtime,"Extraction resume mismatch")
    else:
        state={"identity":identity,"runtime":runtime,"created_utc":now(),"shards":[],"input_sha256_before":receipts}; write_json(state_path,state)
    for item in state["shards"]: require(sha(output/item["file"])==item["sha256"],"Shard changed")
    torch.manual_seed(20261106)
    body=IntermediateLayerGetter(resnet50(weights=None,norm_layer=FrozenBatchNorm2d),return_layers={"layer4":"features"})
    source=torch.load(args.weights,map_location="cpu",weights_only=True); values={k[len("backbone.body."):]:v for k,v in source.items() if k.startswith("backbone.body.")}
    require(len(values)==265,"Unexpected backbone keys"); body.load_state_dict(values,strict=True); del source,values
    body.requires_grad_(False).eval().to(args.device); before=fingerprint(body)
    require(all(not p.requires_grad for p in body.parameters()),"Backbone not frozen"); require(state.get("backbone_tensor_sha256",before)==before,"Resume backbone differs"); state["backbone_tensor_sha256"]=before
    completed=sum(item["n"] for item in state["shards"]); require(completed<=len(rows),"Too many records")
    for offset in range(completed,len(rows),args.feature_batch_size):
        batch=rows[offset:offset+args.feature_batch_size]; tensors=[]
        for row in batch:
            with Image.open(row["image_path"]) as im:
                a,b,c,d=row["bbox_xyxy"]; rgb=im.convert("RGB").crop((math.floor(a),math.floor(b),math.ceil(c),math.ceil(d))).resize((224,224),Image.Resampling.BILINEAR)
                tensors.append(TF.normalize(TF.to_tensor(rgb),TRANSFORM["mean"],TRANSFORM["std"]))
        require(not body.training and all(not m.training for m in body.modules()),"Backbone/BN left eval")
        with torch.no_grad(): feat=torch.nn.functional.adaptive_avg_pool2d(body(torch.stack(tensors).to(args.device))["features"],1).flatten(1).cpu()
        require(tuple(feat.shape)==(len(batch),2048) and torch.isfinite(feat).all().item(),"Invalid feature batch")
        shard={"features":feat,"labels":torch.tensor([r["label_id"] for r in batch],dtype=torch.long),"record_ids":[r["record_id"] for r in batch],"record_indices":[r["record_index"] for r in batch]}
        filename=f"shard_{offset:06d}.pt"; save_tensor(output/filename,shard); state["shards"].append({"file":filename,"sha256":sha(output/filename),"n":len(batch),"offset":offset})
        state.update(completed_ROIs=offset+len(batch),updated_utc=now()); write_json(state_path,state); print(f"FEATURES {offset+len(batch)}/{len(rows)}",flush=True)
    finalize_feature_cache(output,state,rows,identity,receipts,counts,document,runtime,before,fingerprint(body),all(p.grad is None for p in body.parameters()))
    print("FEATURE_CACHE_COMPLETE",flush=True)

def evaluate_head(head,x,y,names,ids):
    import torch
    head.eval(); confusion=[[0]*len(names) for _ in names]; total=0.; predictions=[]
    with torch.no_grad():
        for start in range(0,len(y),256):
            logits=head(x[start:start+256]); labels=y[start:start+256]; values=torch.nn.functional.cross_entropy(logits,labels,reduction="none")
            require(torch.isfinite(values).all().item(),"Nonfinite evaluation loss")
            probs=logits.softmax(1).tolist(); predicted=logits.argmax(1).tolist(); true=labels.tolist(); total+=float(values.double().sum().item())
            for j,(actual,pred,prob) in enumerate(zip(true,predicted,probs)):
                confusion[actual][pred]+=1; predictions.append({"record_id":ids[start+j],"label_id":actual,"predicted_id":pred,"probabilities":prob})
    return metric_summary(confusion,total,names),predictions
def optimizer_finite(optimizer):
    import torch
    require(all(not isinstance(value,torch.Tensor) or torch.isfinite(value).all().item() for state in optimizer.state.values() for value in state.values()),"Nonfinite optimizer state")
def expected_updates(n): return math.ceil(n/RECIPE["batch_size"])*RECIPE["epochs"]
def train(args,document,rows,names,receipts,counts,identity):
    output=args.output.resolve(); require(not output.exists() or args.resume,"Existing suite needs --resume")
    configure_runtime(output,"cpu"); torch=torch_setup("cpu"); payload,complete=load_cache(args.cache.resolve(),identity,torch)
    train_ids=[i for i,r in enumerate(rows) if r["split"]=="train"]; dev_ids=[i for i,r in enumerate(rows) if r["split"]=="dev"]
    tx,ty=payload["features"][train_ids],payload["labels"][train_ids]; dx,dy=payload["features"][dev_ids],payload["labels"][dev_ids]
    train_record_ids=[rows[i]["record_id"] for i in train_ids]; dev_record_ids=[rows[i]["record_id"] for i in dev_ids]
    suite_identity={"identity":identity,"cache_features_sha256":complete["features_sha256"],"recipe":RECIPE,"seeds":list(SEEDS),"arms":list(ARMS),"runtime":{"torch":str(torch.__version__),"python":sys.version,"threads":torch.get_num_threads()}}
    identity_path=output/"suite_identity.json"
    if identity_path.exists(): require(json.loads(identity_path.read_text(encoding="utf-8"))==suite_identity,"Suite resume identity differs")
    else: write_json(identity_path,suite_identity)
    if (output/"suite_report.json").exists():
        done=json.loads((output/"suite_report.json").read_text(encoding="utf-8"))
        require(done["phase"]=="complete" and done["suite_identity"]==suite_identity,"Completed suite identity mismatch")
        require(verify_inputs(receipts)==done["input_sha256_after"],"Completed suite inputs differ")
        for result in done["results"]:
            require(sha(result["checkpoint"])==result["checkpoint_sha256"] and sha(result["report"])==result["report_sha256"],"Completed suite artifact changed")
        print("ATTRIBUTE_FULL_SUITE_ALREADY_COMPLETE_VERIFIED",flush=True); return
    results=[]
    for seed in SEEDS:
        for arm in ARMS:
            run_dir=output/f"seed{seed}_{arm.replace('-','_')}"; run_dir.mkdir(exist_ok=True); weights=class_weights(counts["train"],names,arm)
            torch.manual_seed(seed); head=torch.nn.Linear(2048,len(names)); initial_hash=fingerprint(head)
            optimizer=torch.optim.AdamW(head.parameters(),lr=RECIPE["learning_rate"],weight_decay=RECIPE["weight_decay"])
            weight_tensor=None if weights is None else torch.tensor(weights,dtype=torch.float32)
            run_identity={"suite_identity_sha256":canonical(suite_identity),"seed":seed,"class_weighting":arm,"training_CE_weights":weights,"initial_head_sha256":initial_hash}
            checkpoint_path=run_dir/"last.pt"; completed=0; history=[]; global_step=0
            if checkpoint_path.exists():
                require(args.resume,"Existing checkpoint needs --resume"); ck=torch.load(checkpoint_path,map_location="cpu",weights_only=True); require(ck["run_identity"]==run_identity,"Checkpoint identity differs")
                head.load_state_dict(ck["head_state_dict"],strict=True); optimizer.load_state_dict(ck["optimizer_state_dict"]); optimizer_finite(optimizer); completed=ck["completed_epoch"]; history=ck["history"]; global_step=ck["actual_optimizer_updates"]
                require(0<=completed<=20 and len(history)==completed and global_step==completed*math.ceil(len(ty)/32),"Checkpoint budget differs")
                require(fingerprint(head)==ck["head_tensor_sha256"],"Checkpoint head hash differs"); torch.set_rng_state(ck["torch_rng_state"])
            for epoch in range(completed+1,RECIPE["epochs"]+1):
                order=list(range(len(ty))); random.Random(seed+epoch).shuffle(order); require(sorted(order)==list(range(len(ty))),"Epoch not exactly once per ROI")
                order_hash=canonical([train_record_ids[i] for i in order]); head.train(); numerator=0.; denominator=0.; steps=[]; started=time.perf_counter()
                for offset in range(0,len(order),RECIPE["batch_size"]):
                    indices=order[offset:offset+RECIPE["batch_size"]]; labels=ty[indices]; optimizer.zero_grad(set_to_none=True); logits=head(tx[indices])
                    summed=torch.nn.functional.cross_entropy(logits,labels,weight=weight_tensor,reduction="sum"); denom=float(len(indices)) if weight_tensor is None else float(weight_tensor[labels].sum().item()); loss=summed/denom
                    require(torch.isfinite(loss).item(),"Nonfinite training loss"); loss.backward(); require(all(p.grad is not None and torch.isfinite(p.grad).all().item() for p in head.parameters()),"Missing/nonfinite gradient")
                    optimizer.step(); require(all(torch.isfinite(p).all().item() for p in head.parameters()),"Nonfinite head"); global_step+=1; numerator+=float(summed.detach().item()); denominator+=denom
                    steps.append({"step":global_step,"n_ROIs":len(indices),"loss":float(loss.item()),"gradient_finite":True,"optimizer_step_completed":True})
                dev_metrics,_=evaluate_head(head,dx,dy,names,dev_record_ids)
                history.append({"epoch":epoch,"actual_optimizer_updates":global_step,"train_ROIs_used":len(order),"train_order_sha256":order_hash,"train_weighted_CE":numerator/denominator,"train_CE_numerator":numerator,"train_CE_denominator":denominator,"development":dev_metrics,"steps":steps,"wall_seconds":time.perf_counter()-started})
                optimizer_finite(optimizer)
                ck={"schema":SCHEMA,"run_identity":run_identity,"class_names":names,"label_semantics":document["label_semantics"],"semantic_limits":LIMITS,"recipe":RECIPE,"completed_epoch":epoch,"actual_optimizer_updates":global_step,"head_state_dict":head.state_dict(),"optimizer_state_dict":optimizer.state_dict(),"torch_rng_state":torch.get_rng_state(),"next_epoch_shuffle_seed":seed+epoch+1,"head_tensor_sha256":fingerprint(head),"history":history,"backbone_source_weights_sha256":WEIGHT_SHA,"backbone_tensor_sha256":complete["backbone_tensor_sha256_after"],"manifest_sha256":identity["manifest_sha256"],"cache_features_sha256":complete["features_sha256"],"script_sha256":identity["script_sha256"],"evaluation_role":ROLE,"independent_test_claimed":False,"test_used":False,"checkpoint_role":"fixed epoch full-data frozen-feature linear baseline; GT-ROI development only"}
                save_tensor(checkpoint_path,ck); write_json(run_dir/"progress.json",{"phase":"training","updated_utc":now(),"seed":seed,"arm":arm,"completed_epoch":epoch,"actual_optimizer_updates":global_step,"last_development":dev_metrics})
                print(f"FULL_ATTRIBUTE seed={seed} arm={arm} epoch={epoch}/20 steps={global_step} dev_macroF1={dev_metrics['macro_F1']:.6f}",flush=True)
            require(global_step==expected_updates(len(ty)),"Final budget incomplete"); optimizer_steps=[int(v["step"].item()) for v in optimizer.state.values()]; require(optimizer_steps==[global_step,global_step],"AdamW steps differ")
            train_metrics,_=evaluate_head(head,tx,ty,names,train_record_ids); dev_metrics,predictions=evaluate_head(head,dx,dy,names,dev_record_ids)
            ck=torch.load(checkpoint_path,map_location="cpu",weights_only=True); require(ck["completed_epoch"]==20,"Final checkpoint not epoch20")
            final_path=run_dir/"final_epoch20.pt"
            if final_path.exists():
                final=torch.load(final_path,map_location="cpu",weights_only=True)
                require(final["run_identity"]==run_identity and final["completed_epoch"]==20 and final["head_tensor_sha256"]==fingerprint(head) and fingerprint_state(final["head_state_dict"])==fingerprint(head) and final["actual_optimizer_updates"]==global_step,"Existing final checkpoint differs")
            else: save_tensor(final_path,ck)
            prediction_path=run_dir/"development_predictions.json"
            if prediction_path.exists(): require(json.loads(prediction_path.read_text(encoding="utf-8"))==predictions,"Existing development predictions differ")
            else: write_json(prediction_path,predictions)
            report={"schema":SCHEMA,"phase":"pending_suite_integrity","trained_utc":now(),"run_identity":run_identity,"recipe":RECIPE,"evaluation_role":ROLE,"independent_test_claimed":False,"test_used":False,"GT_ROI_input":True,"label_semantics":document["label_semantics"],"semantic_limits":LIMITS,"split_policy":document["split_policy"],"class_names":names,"label_counts":counts,"actual_optimizer_updates":global_step,"actual_AdamW_parameter_steps":optimizer_steps,"history":history,"training_final":train_metrics,"development_final":dev_metrics,"development_used_for_model_selection":False,"class_weight_formula":"N_train/(C*n_class), complete train counts only","head_tensor_sha256_initial":initial_hash,"head_tensor_sha256_final":fingerprint(head),"head_changed":initial_hash!=fingerprint(head),"cache_features_sha256":complete["features_sha256"],"backbone_frozen_unchanged":complete["frozen_backbone_unchanged"],"input_sha256_before":receipts,"checkpoint_sha256":sha(run_dir/"final_epoch20.pt"),"full_network_finetuning":False,"protocol_frozen_after_exploratory_pilots":True,"preregistered":False}
            write_json(run_dir/"full_report.json",report); results.append({"seed":seed,"arm":arm,"report":str(run_dir/"full_report.json"),"checkpoint":str(run_dir/"final_epoch20.pt"),"checkpoint_sha256":report["checkpoint_sha256"],"initial_head_sha256":initial_hash,"train_order_sha256":[h["train_order_sha256"] for h in history],"development":dev_metrics})
    for seed in SEEDS:
        pair=[r for r in results if r["seed"]==seed]; require(pair[0]["initial_head_sha256"]==pair[1]["initial_head_sha256"] and pair[0]["train_order_sha256"]==pair[1]["train_order_sha256"],"Arms differ in initialization/order")
    after=verify_inputs(receipts); require(sha(__file__)==identity["script_sha256"] and sha(args.cache/"features.pt")==complete["features_sha256"],"Source/cache changed")
    for result in results:
        report_path=Path(result["report"]); report=json.loads(report_path.read_text(encoding="utf-8"))
        report.update(phase="complete",completed_utc=now(),input_sha256_after=after,source_and_cache_verified=True)
        write_json(report_path,report); result["report_sha256"]=sha(report_path)
    summary={"schema":SCHEMA,"phase":"complete","completed_utc":now(),"suite_identity":suite_identity,"results":results,"evaluation_role":ROLE,"independent_test_claimed":False,"test_used":False,"GT_ROI_input":True,"input_sha256_before":receipts,"input_sha256_after":after,"class_names":names,"label_counts":counts,"label_semantics":document["label_semantics"],"semantic_limits":LIMITS,"same_seed_arms_have_same_initialization_and_order":True,"selection":"all six fixed final-epoch results; no winner promotion/test selection"}
    write_json(output/"suite_report.json",summary); print("ATTRIBUTE_FULL_SUITE_COMPLETE",flush=True)
def evaluate(args,document,rows,names,receipts,counts,identity):
    output=args.output.resolve(); require(not output.exists(),"Evaluation output exists"); configure_runtime(output,"cpu"); torch=torch_setup("cpu")
    payload,complete=load_cache(args.cache.resolve(),identity,torch); ck=torch.load(args.checkpoint,map_location="cpu",weights_only=True)
    require(ck["manifest_sha256"]==identity["manifest_sha256"] and ck["cache_features_sha256"]==complete["features_sha256"] and ck["script_sha256"]==identity["script_sha256"],"Checkpoint/source/cache differs")
    require(ck["class_names"]==names and ck["completed_epoch"]==20 and ck["recipe"]==RECIPE,"Only fixed final compatible checkpoint accepted")
    head=torch.nn.Linear(2048,len(names)); head.load_state_dict(ck["head_state_dict"],strict=True); require(fingerprint(head)==ck["head_tensor_sha256"],"Head hash differs")
    indices=[i for i,r in enumerate(rows) if r["split"]=="dev"]; metrics,predictions=evaluate_head(head,payload["features"][indices],payload["labels"][indices],names,[rows[i]["record_id"] for i in indices])
    write_json(output/"development_predictions.json",predictions)
    write_json(output/"evaluation_report.json",{"phase":"complete","metrics":metrics,"checkpoint_sha256":sha(args.checkpoint),"feature_cache_sha256":complete["features_sha256"],"evaluation_role":ROLE,"independent_test_claimed":False,"GT_ROI_input":True,"test_used":False,"label_semantics":document["label_semantics"],"input_sha256_before":receipts,"input_sha256_after":verify_inputs(receipts)})
    print(json.dumps(metrics,ensure_ascii=False,indent=2),flush=True)
def parser():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("mode",choices=("check","extract","train","evaluate"))
    p.add_argument("--manifest",type=Path,required=True); p.add_argument("--weights",type=Path,required=True); p.add_argument("--allow-non-independent-development",action="store_true")
    p.add_argument("--output",type=Path); p.add_argument("--cache",type=Path); p.add_argument("--checkpoint",type=Path); p.add_argument("--resume",action="store_true")
    p.add_argument("--device",choices=("cuda:0","cpu"),default="cuda:0"); p.add_argument("--feature-batch-size",type=int,default=16); return p
def main():
    args=parser().parse_args(); require(1<=args.feature_batch_size<=32,"Feature batch size [1,32]")
    require(args.mode=="check" or args.output is not None,"--output required"); require(args.mode not in ("train","evaluate") or args.cache is not None,"--cache required")
    require(args.mode!="evaluate" or args.checkpoint is not None,"--checkpoint required"); values=preflight(args)
    if args.mode=="check":
        document,rows,names,receipts,counts,identity=values
        print(json.dumps({"phase":"check_passed","counts":counts,"class_names":names,"label_semantics":document["label_semantics"],"evaluation_role":ROLE,"independent_test_claimed":False,"GT_ROI_input":True,"test_used":False,"input_hash_count":len(receipts),"manifest_sha256":identity["manifest_sha256"],"recipe":RECIPE,"seeds":SEEDS,"arms":ARMS,"expected_updates_per_run":expected_updates(sum(counts["train"].values()))},ensure_ascii=False,indent=2)); return 0
    try: {"extract":extract,"train":train,"evaluate":evaluate}[args.mode](args,*values)
    except BaseException as error:
        if args.output and args.output.exists() and args.output.resolve().drive.lower()=="f:" and args.output.resolve().is_relative_to(Path(__file__).resolve().parents[1]): write_json(args.output/"failure.json",{"failed_utc":now(),"error":str(error),"traceback":traceback.format_exc()})
        raise
    return 0
if __name__=="__main__": raise SystemExit(main())
