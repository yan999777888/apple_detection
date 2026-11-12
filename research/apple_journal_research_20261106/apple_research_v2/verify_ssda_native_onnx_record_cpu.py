"""Independently audit saved native/ONNX predictions, timings and ORT profile on CPU."""
from pathlib import Path
from collections import Counter,defaultdict
from datetime import datetime,timezone
import hashlib,json,math,statistics,random,sys
sys.dont_write_bytecode=True
import numpy as np
import cv2
from scipy.optimize import linear_sum_assignment
import yaml

ROOT=Path(__file__).resolve().parents[1]
DIRECTORY=ROOT/"benchmarks"/"ssda_native_onnx_parity_cuda_v1"
REPORT=DIRECTORY/"parity_benchmark.json"
OUTPUT=DIRECTORY/"independent_cpu_verification.json"


def require(ok,msg):
    if not ok:raise ValueError(msg)


def sha(path):
    with Path(path).open("rb") as f:return hashlib.file_digest(f,"sha256").hexdigest()


def near(a,b,tol=1e-10):
    require(abs(float(a)-float(b))<=tol,f"Numeric mismatch {a} vs {b}")


def overlap(a,b):
    intersection=max(0.,min(a[2],b[2])-max(a[0],b[0]))*max(0.,min(a[3],b[3])-max(a[1],b[1]))
    union=(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-intersection
    return intersection/union if union>0 else 0.


def stats(values):
    s=sorted(values);n=len(s)
    def q(p):
        position=p*(n-1);i=math.floor(position);j=math.ceil(position)
        return s[i]+(s[j]-s[i])*(position-i)
    mean=statistics.mean(values)
    return {"n_pipeline_frames":n,"mean_ms":mean,"p50_ms":q(.5),"p95_ms":q(.95),
            "fps_1000_over_mean_ms":1000/mean}


def main():
    require(ROOT.drive.upper()=="F:","F-only verifier")
    require(not OUTPUT.exists(),"Verification output exists; refusing overwrite")
    original_sha=sha(REPORT)
    d=json.loads(REPORT.read_text(encoding="utf-8"))
    require(d["phase"]=="complete","Original run not complete")
    fixed={"conf":.25,"matching_box_iou_min":.99,"box_coordinate_absolute_tolerance_px":.1,
        "score_absolute_tolerance":1e-4,"mask_iou_min":.999,"confidence_boundary_epsilon":1e-4,
        "count_changes_allowed":False,"class_count_changes_allowed":False,"threshold_adaptation_allowed":False,
        "imgsz":640,"batch":1,"quantize":32,"retina_masks":True,"rect":False,"max_det":300,"iou":.7,
        "timing_distinct_frames":40,"timing_repeats":3,"warmup_each_backend":5}
    require(all(d["protocol"][k]==v for k,v in fixed.items()),"Fixed protocol changed")
    m=d["manifest"];hash_checks=[]
    def check_file(path,expected,kind):
        path=Path(path).resolve();require(path.drive.upper()=="F:","Non-F input")
        actual=sha(path);require(actual==expected,f"Input hash changed: {path}")
        hash_checks.append({"path":str(path),"sha256":actual,"kind":kind})
    for key in ("native","onnx","data_yaml","export_receipt"):
        check_file(m[key],m[key+"_sha256"],key)
    for path,digest in m["source_sha256"].items():check_file(path,digest,"producer_source")
    require(m["split"]=="val" and len(m["images"])==113,"Expected full 113-image val")
    for image in m["images"]:
        check_file(image["path"],image["image_sha256"],"image")
        data=cv2.imdecode(np.frombuffer(Path(image["path"]).read_bytes(),np.uint8),cv2.IMREAD_COLOR)
        require(data is not None and list(data.shape)==image["shape"],"Image dimensions/decode mismatch")
        require(hashlib.sha256(memoryview(data)).hexdigest()==image["decoded_bgr_sha256"],"Decoded BGR changed")
    config=yaml.safe_load(Path(m["data_yaml"]).read_text(encoding="utf-8"))
    data_root=Path(config["path"])
    if not data_root.is_absolute():data_root=Path(m["data_yaml"]).parent/data_root
    val=Path(config["val"]);val=val if val.is_absolute() else data_root/val
    if val.is_file() and val.suffix.lower()==".txt":
        declared=[Path(line.strip()) for line in val.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
        resolved=[p.resolve() if p.is_absolute() else (val.parent/p).resolve() for p in declared]
        require(len(set(resolved))==len(resolved),"Duplicate declared validation image")
        actual_images={str(p) for p in resolved}
    else:
        actual_images={str(p.resolve()) for p in val.rglob("*") if p.suffix.lower() in (".jpg",".jpeg",".png")}
    require(actual_images=={str(Path(x["path"]).resolve()) for x in m["images"]},"Val image coverage changed")
    parity=d["parity_rows"]
    require(len(parity)==113 and {r["frame_index"] for r in parity}==set(range(113)),"Missing/duplicate parity frames")
    maxima={"box_coordinate_error_px":0.,"score_error":0.,"recorded_mask_xor_pixels":None}
    minima={"box_iou":1.,"recorded_mask_iou":None}
    class_total=Counter();frame_checks=[];match_count=0;boundary_count=0;min_conf_margin=1.
    for row in parity:
        idx=row["frame_index"]
        require(row["image"]==m["images"][idx]["path"],"Frame/image binding mismatch")
        require(row["preprocessing_equal"] is True and len(row["preprocessing_sha256"])==64,"Preprocessing evidence missing")
        a=row["native_boxes_xyxy_score_class"];b=row["onnx_boxes_xyxy_score_class"]
        require(len(a)==len(b)==row["native_count"]==row["onnx_count"],"Count changed")
        for name,boxes in (("native",a),("onnx",b)):
            require(all(len(x)==6 and all(math.isfinite(v) for v in x) and float(x[5]).is_integer() and x[4]>=.25 for x in boxes),"Invalid raw prediction")
            counts=Counter(str(int(x[5])) for x in boxes)
            require(dict(counts)==row[name+"_class_counts"],"Saved class counts inconsistent")
            boundaries=[{"index":i,"class":int(x[5]),"score":x[4]} for i,x in enumerate(boxes) if abs(x[4]-.25)<=1e-4]
            require(boundaries==row[name+"_threshold_boundary"],"Inclusive confidence-boundary mismatch")
            boundary_count+=len(boundaries)
            if boxes:min_conf_margin=min(min_conf_margin,min(abs(x[4]-.25) for x in boxes))
        require(Counter(int(x[5]) for x in a)==Counter(int(x[5]) for x in b),"Class count changed")
        reconstructed=set()
        for cls in sorted({int(x[5]) for x in a}):
            ai=sorted([i for i,x in enumerate(a) if int(x[5])==cls],key=lambda i:(*a[i][:5],i))
            bi=sorted([i for i,x in enumerate(b) if int(x[5])==cls],key=lambda i:(*b[i][:5],i))
            matrix=np.asarray([[overlap(a[i],b[j]) for j in bi] for i in ai])
            assignments=linear_sum_assignment(np.where(matrix>=.99,1-matrix,1e6))
            reconstructed.update((ai[int(i)],bi[int(j)]) for i,j in zip(*assignments))
        saved={(x["native_index"],x["onnx_index"]) for x in row["matches"]}
        require(saved==reconstructed and len(saved)==len(a)==len(row["matches"]),"Independent Hungarian pairs differ")
        require(len({i for i,j in saved})==len(a) and len({j for i,j in saved})==len(b),"Match is not one-to-one")
        for result in row["matches"]:
            i,j=result["native_index"],result["onnx_index"];x,y=a[i],b[j]
            require(int(x[5])==int(y[5])==result["class"],"Match class mismatch")
            iou=overlap(x,y);error=max(abs(v-w) for v,w in zip(x[:4],y[:4]));score=abs(x[4]-y[4])
            near(iou,result["box_iou"]);near(error,result["max_box_abs_error_px"]);near(score,result["score_abs_error"])
            # Inclusive gates: equality is permitted. Mask evidence is recorded
            # pixel-comparison metrics, not independently available mask arrays.
            require(iou>=.99 and error<=.1 and score<=1e-4,"Box/score parity threshold failed")
            mask_iou=result["mask_iou"];xor=result["mask_xor_pixels"]
            require(mask_iou is None and xor is None,"Detection benchmark unexpectedly reports segmentation masks")
            require(result["failures"]==[],"Original match has failures")
            maxima["box_coordinate_error_px"]=max(maxima["box_coordinate_error_px"],error)
            maxima["score_error"]=max(maxima["score_error"],score)
            minima["box_iou"]=min(minima["box_iou"],iou)
            match_count+=1;class_total[str(int(x[5]))]+=1
        require(row["passed"] is True and row["issues"]==[],"Original frame not marked clean")
        frame_checks.append({"frame_index":idx,"detections_each_backend":len(a),"independent_box_score_match_passed":True,"segmentation_mask_comparison":"not_applicable_detection_model"})
    require(match_count==278 and d["parity_summary"]["native_detections"]==278 and d["parity_summary"]["onnx_detections"]==278,"Detection totals mismatch")
    require(d["parity_summary"]["all_images"]==113 and d["parity_summary"]["failed_frames"]==0 and d["parity_summary"]["passed"] is True,"Parity summary mismatch")
    require(boundary_count==0 and d["parity_summary"]["count_change_frames"]==0,"Unexpected count/boundary cases")
    selected=d["timing_frame_indices"]
    require(selected==random.Random(20261106).sample(range(113),40),"Timing sample selection mismatch")
    times=d["timing_rows"];require(len(times)==240,"Expected 240 timing records")
    expected_order=[(backend,repeat,index) for repeat in range(3) for index in selected
                    for backend in (("native","onnx") if repeat%2==0 else ("onnx","native"))]
    require([(r["backend"],r["repeat"],r["frame_index"]) for r in times]==expected_order,"Paired timing/order mismatch")
    by_key={};values=defaultdict(list)
    for row in times:
        key=(row["backend"],row["repeat"],row["frame_index"])
        require(key not in by_key,"Duplicate timing record");by_key[key]=row
        require(isinstance(row["elapsed_ns"],int) and row["elapsed_ns"]>0,"Invalid elapsed ns")
        near(row["elapsed_ms"],row["elapsed_ns"]/1e6,1e-12)
        require(row["detections"]==parity[row["frame_index"]][row["backend"]+"_count"],"Timing count differs from parity")
        h,w,c=m["images"][row["frame_index"]]["shape"]
        require(row["cpu_output_bytes"]==row["detections"]*(6*4),"CPU output bytes inconsistent with detection-only FP32 boxes")
        values[row["backend"]].append(row["elapsed_ns"]/1e6)
    summaries={backend:stats(v) for backend,v in values.items()}
    for backend,summary in summaries.items():
        require(len(values[backend])==120,"Timing backend count mismatch")
        for key,value in summary.items():near(value,d["timing_summary"][backend][key])
    differences=[by_key[("native",r,i)]["elapsed_ms"]-by_key[("onnx",r,i)]["elapsed_ms"] for r in range(3) for i in selected]
    evidence=d["provider_evidence"];check_file(evidence["profile"],evidence["profile_sha256"],"actual_wrapper_profile")
    events=json.loads(Path(evidence["profile"]).read_text(encoding="utf-8"))
    categories=Counter(e.get("cat","<missing>") for e in events)
    nodes=[e for e in events if e.get("cat")=="Node"]
    providers=Counter();ops=defaultdict(Counter);malformed=[];unique_nodes=set()
    for n,e in enumerate(nodes):
        args=e.get("args",{});provider=args.get("provider","<missing>");op=args.get("op_name","<missing>")
        providers[provider]+=1;ops[provider][op]+=1;unique_nodes.add((provider,args.get("node_index"),op))
        if not e.get("name","").endswith("_kernel_time") or provider=="<missing>" or op=="<missing>":malformed.append(n)
    require(len(nodes)==34800 and not malformed,"Unaccounted profile Node events")
    require(dict(providers)==evidence["executed_node_events_by_provider"],"Provider counts differ")
    require({k:dict(v) for k,v in ops.items()}==evidence["executed_ops_by_provider"],"Provider operation counts differ")
    require(providers==Counter({"CUDAExecutionProvider":34800}),"CPU/unknown nodes observed")
    provider_file=DIRECTORY/"provider_execution_evidence.json"
    require(json.loads(provider_file.read_text(encoding="utf-8"))==evidence,"Standalone provider evidence differs")
    require(sha(REPORT)==original_sha,"Original report changed during verification")
    require("torch" not in sys.modules and "onnxruntime" not in sys.modules,"Unexpected inference runtime import")
    out={"schema":"independent_saved_parity_timing_CPU_verification_v1","verified_utc":datetime.now(timezone.utc).isoformat(),
        "verification_passed":True,"original_report":str(REPORT),"original_report_sha256":original_sha,
        "verifier_source_sha256":sha(__file__),"GPU_used":False,"inference_rerun":False,
        "fixed_protocol_verified":fixed,"inclusive_gates":{"box_iou":">= 0.99","coordinate_error_px":"<= 0.1","score_error":"<= 0.0001","recorded_mask_iou":"not applicable to detection-only checkpoint","confidence_boundary":"absolute distance <= 0.0001"},
        "images":113,"detections_each_backend":278,"class_counts_each_backend":dict(class_total),
        "matched_boxes_scores_independently_recomputed":True,"maxima":maxima,"minima":minima,
        "minimum_confidence_distance_from_0_25":min_conf_margin,"threshold_boundary_count_both_backends":boundary_count,
        "frame_checks":frame_checks,"timing_summary_recomputed":summaries,
        "paired_timing":{"pairs":120,"unique_images":40,"repeats":3,"alternating_backend_order_verified":True,
            "mean_native_minus_onnx_ms":statistics.mean(differences),"onnx_faster_pairs":sum(x>0 for x in differences),
            "native_mean_divided_by_onnx_mean":summaries["native"]["mean_ms"]/summaries["onnx"]["mean_ms"],
            "mean_latency_reduction_percent":100*(1-summaries["onnx"]["mean_ms"]/summaries["native"]["mean_ms"])},
        "all_profile_events_audited":{"total_events":len(events),"categories":dict(categories),"node_events":len(nodes),
            "node_events_by_provider":dict(providers),"ops_by_provider":{k:dict(v) for k,v in ops.items()},
            "unique_provider_node_index_op_tuples":len(unique_nodes),"unclassified_node_events":malformed,
            "session_model_run_events":sum(e.get("cat")=="Session" and e.get("name")=="model_run" for e in events),
            "profile_ended_before_timed_loop_per_frozen_source":True},
        "current_file_hash_checks":hash_checks,"current_decoded_BGR_hashes_all_match":True,
        "original_report_unchanged":True,"ground_truth_labels_read":False,
        "limitations":["This is a detection-only checkpoint: no segmentation masks are returned or benchmarked. Both preprocessing tensors were not persisted; their equality is producer evidence rather than a current tensor recomputation.",
            "ORT profile audits actual wrapper-session node events before timing. All recorded nodes use CUDAExecutionProvider; this does not place CPU preprocessing/postprocessing or the entire pipeline on GPU.",
            "A CPU audit of saved results is not a rerun of inference, a new accuracy evaluation, or robot/camera end-to-end timing.",
            "Other GPU workloads were not verified absent in the producer report; retain that measurement limitation."]}
    with OUTPUT.open("x",encoding="utf-8") as f:json.dump(out,f,ensure_ascii=False,indent=2);f.write("\n")
    print(json.dumps({"verification":str(OUTPUT),"passed":True,"images":113,"detections_each":278,
        "maxima":maxima,"minima":minima,"timings":summaries,"paired_timing":out["paired_timing"],
        "profile_categories":dict(categories),"profile_providers":dict(providers),
        "file_hashes_checked":len(hash_checks),"GPU_used":False},ensure_ascii=False))


if __name__=="__main__":main()
