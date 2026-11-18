"""Evaluate the frozen apple pipeline on speed counterbalanced frames and AFruitDB test."""

import argparse
from collections import Counter
from pathlib import Path
import time

import cv2
import numpy as np
import torch

from afruit_visible_quality import (
    OUT, ROOT, CROP_CONTEXT, GRADES, PRIMARY, QualityPipeline, STAGE_WEIGHT_SHA,
    SOURCE_SHA, WARM_SHA, atomic_json, configure, load_image, overlay, prepare_input, read, sha,
)
from benchmark_joint_20261112 import summary
from joint_eval_v1 import source_spec


MANIFEST = ROOT / "datasets/afruitdb_apples_v1/manifest.json"
GRADE_OUT = ROOT / "experiments/afruit_grade_release_v1"


def freeze_policy():
    policy = dict(
        version=1, crop_context_each_side=CROP_CONTEXT, stage_score=0.25,
        defect_thresholds=PRIMARY, grade_rule="locked AFruitDB validation-selected threshold",
        grade_dtype="FP32", scene="grade every distinct predicted stage instance's contextual bounding box",
        known_single_fruit="grade the full supplied fruit image once, regardless of maturity branch detections",
        test_single="only exactly one predicted instance with a valid crop is linkable to the image label; no or multiple instances count as unresolved; no defect or stage override of appearance grade",
        severe_denominator="display all true severe images, resolved only, and unresolved count separately",
        speed="20 legacy Mendeley + 20 SSDA development images at fixed evenly spaced ranks, 15 rounds; up to 20 evenly spaced fixed-test AFruitDB photos with >=600 measured frames in each market mode, 5 warmup per mode",
        stage_fp32_checkpoint=STAGE_WEIGHT_SHA,defect_warm_checkpoint=WARM_SHA,
        defect_scratch_checkpoint=SOURCE_SHA,
    )
    file = OUT/"frozen_policy.json"
    if file.exists() and policy != read(file):
        raise ValueError("Frozen association or evaluation policy changed")
    atomic_json(file, policy)


def freeze_camera_profile():
    p=OUT/"camera_resolution_policy.json"
    setting=dict(long_edge=1280, resize="aspect-preserving, performed before timing", rounds=30,
                 n=20, mode="AFruitDB full scene with per-instance grades",
                 note="Simulates receiving frames already delivered at these pixel dimensions. Not real camera latency or grading accuracy at 1280.")
    if p.exists() and read(p)!=setting:
        raise ValueError("Changed camera resolution profile")
    atomic_json(p,setting)
    follow=OUT/"camera_grade_protocol.json"
    companion=dict(long_edge=1280, frozen_same_policy_as_native=True,
                   test="same complete fixed AFruitDB test images; classify only an exact single predicted fruit for the primary provisional image link; report severe-to-good plus unresolved",
                   timing="additional measured source-to-output time includes OpenCV INTER_AREA downsampling before network inference; deliver boxes and overlays on inference grid")
    if follow.exists() and read(follow)!=companion:
        raise ValueError("Changed paired resolution accuracy protocol")
    atomic_json(follow,companion)


def spaced(items, n):
    n = min(n, len(items))
    if n < 2:
        return items[:n]
    return [items[round(i * (len(items)-1)/(n-1))] for i in range(n)]


def market_image(path):
    # Match the AFruitDB classifier loader (including OpenCV's JPEG orientation policy).
    image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Invalid AFruitDB image: {path}")
    return image


def timed_frame(pipe, frame, known, long_edge):
    torch.cuda.synchronize()
    start=time.perf_counter()
    frame=prepare_input(frame,long_edge)
    prepared=time.perf_counter()
    result,masks=pipe(frame,known_single_fruit=known)
    torch.cuda.synchronize()
    middle=time.perf_counter()
    overlay(frame,result,masks)
    end=time.perf_counter()
    return result, (middle-prepared)*1000, (end-prepared)*1000, (middle-start)*1000, (end-start)*1000


def distribution(rows):
    if not rows:
        return None
    counts = Counter(r["fruits"] for r in rows)
    number = len(rows)
    return dict(
        n=number, unique_files=len({r["path"] for r in rows}),
        input_to_cpu_results=summary([r["ms"] for r in rows]),
        input_to_annotated_frame=summary([r["annotated_ms"] for r in rows]),
        source_including_resize_to_results=summary([r["source_ms"] for r in rows]),
        source_including_resize_to_overlay=summary([r["source_annotated_ms"] for r in rows]),
        grade_items=sum(r["grade_items"] for r in rows),
        grade_batches=sum(r["grade_batches"] for r in rows),
        frames_with_grade=sum(r["grade_items"]>0 for r in rows),
        zero_fruit_frames=counts[0], single_fruit_frames=counts[1],
        multiple_fruit_frames=number-counts[0]-counts[1],
        fruit_instances=sum(r["fruits"] for r in rows),
        defect_boxes=sum(r["defects"] for r in rows),
        unique_linked_defects=sum(r["linked"] for r in rows),
        per_frame_fruit_counts=dict(sorted(counts.items())),
    )


def benchmark(pipe, name, paths, rounds, known=False, long_edge=None):
    data=[(source,str(path),market_image(path) if source=="AFruitDB" else load_image(path)) for source,path in paths]
    for _,_,im in data[:5]:
        timed_frame(pipe,im,known,long_edge)
    rows=[]
    for repeat in range(rounds):
        for source,path,im in data:
            output,ms,annotated,source_ms,source_annotated=timed_frame(pipe,im,known,long_edge)
            rows.append(dict(source=source,path=path,repeat=repeat,
                             fruits=len(output["fruits"]),defects=len(output["defects"]),
                             linked=sum(d["association"]=="unique" for d in output["defects"]),
                             grade_items=output["grade_inference_items"],grade_batches=output["grade_batches"],
                             ms=ms,annotated_ms=annotated,source_ms=source_ms,source_annotated_ms=source_annotated))
    dataset={s:distribution([r for r in rows if r["source"]==s]) for s in sorted({r["source"] for r in rows})}
    counts={name:distribution([r for r in rows if cond(r)]) for name,cond in (
        ("zero",lambda r:r["fruits"]==0),("single",lambda r:r["fruits"]==1),("multiple",lambda r:r["fruits"]>1),
        ("grade_invoked",lambda r:r["grade_items"]>0))}
    answer=dict(mode=name,rounds=rounds,known_single_fruit=known,all=distribution(rows),
                by_source=dataset,by_instance_count=counts,
                long_edge=long_edge,
                files=[dict(source=s,path=p,source_hw=list(im.shape[:2]),prediction_hw=list(prepare_input(im,long_edge).shape[:2])) for s,p,im in data])
    atomic_json(OUT/f"latency_{name}.json",answer)
    return answer


def quality_counts(rows, prediction):
    # Rows 0/1/2 are image-level ground truth; column 3 means no unambiguous single instance.
    matrix=np.zeros((3,4),dtype=int)
    for r in rows:
        answer=r[prediction]
        matrix[r["truth"],answer if answer is not None else 3]+=1
    bad=matrix[2]
    return dict(
        rows_by_true_grade=list(GRADES),columns_by_prediction=[*GRADES,"unresolved"],
        confusion=matrix.tolist(),images=int(matrix.sum()),resolved=int(matrix[:,:3].sum()),
        correct=int(matrix[:,:3].trace()),severe_total=int(bad.sum()),
        severe_to_good=int(bad[0]),severe_to_minor=int(bad[1]),
        severe_to_severe=int(bad[2]),severe_unresolved=int(bad[3]),
        severe_to_good_fraction_of_all=float(bad[0]/bad.sum()) if bad.sum() else None,
        severe_to_good_fraction_among_resolved=float(bad[0]/bad[:3].sum()) if bad[:3].sum() else None,
        severe_good_or_unresolved_fraction=float((bad[0]+bad[3])/bad.sum()) if bad.sum() else None,
    )


def evaluate_frozen_test(pipe, cases, long_edge=None):
    rows=[]
    bucket=Counter()
    tag="" if long_edge is None else f"_{long_edge}"
    dirs=OUT/("examples"+tag)
    dirs.mkdir(exist_ok=True)
    for rec in cases:
        path=ROOT/rec["path"]
        im=prepare_input(market_image(path),long_edge)
        out,masks=pipe(im)
        direct=pipe.classify([im])[0]["grade_id"]
        n=len(out["fruits"])
        response=out["fruits"][0]["grade_id"] if n==1 else None
        case=dict(path=rec["path"],truth=int(rec["grade"]),group=rec["group_id"],fruit_count=n,
                  linked_grade=response,direct_grade=direct, fruits=out["fruits"], defects=out["defects"])
        rows.append(case)
        key=(case["truth"],"one" if n==1 else "none" if n==0 else "multiple")
        if not bucket[key]:
            filename=f"{GRADES[key[0]]}_{key[1]}.jpg"
            path_out=dirs/filename
            success,encoded=cv2.imencode(".jpg",overlay(im,out,masks))
            if not success:
                raise IOError(path_out)
            encoded.tofile(str(path_out))
            case["sample_preview"] = str(path_out.relative_to(OUT))
        bucket[key]+=1
    independent={r["path"]:r for r in read(GRADE_OUT/"test_predictions.json")}
    if set(independent)!={r["path"] for r in rows}:
        raise ValueError("Standalone classifier test IDs and pipeline test IDs differ")
    direct_disagreements=sum(independent[r["path"]]["predicted"]!=r["direct_grade"] for r in rows)
    result=dict(
        pipeline_single_instance=quality_counts(rows,"linked_grade"),
        supplied_single_fruit_image=quality_counts(rows,"direct_grade"),
        direct_predictions_differing_from_native_standalone=direct_disagreements,
        long_edge=long_edge,
        grade_images_with_no_instance=sum(r["fruit_count"]==0 for r in rows),
        grade_images_with_multiple_instances=sum(r["fruit_count"]>1 for r in rows),
        fruit_instances=sum(r["fruit_count"] for r in rows),
        instance_coverage_by_true_grade={
            label:dict(Counter("none" if r["fruit_count"]==0 else "single" if r["fruit_count"]==1 else "multiple"
                               for r in rows if r["truth"]==i)) for i,label in enumerate(GRADES)},
        all_defect_alerts=sum(len(r["defects"]) for r in rows),
        stage_output_counts=dict(Counter(f["maturity"] for r in rows for f in r["fruits"])),
        conditional_by_predicted_maturity={
            stage:quality_counts([r for r in rows if r["fruit_count"]==1 and r["fruits"][0]["maturity"]==stage],"linked_grade")
            for stage in ("immature","semi-mature","mature")},
        note="AFruitDB image-level grade only. Single predicted fruit provisionally linked to the image grade; no boxes or maturity labels to validate identity, segmentation, maturity or lesion accuracy. Unresolved frames remain in denominators.",
    )
    atomic_json(OUT/f"test_predictions{tag}.json",rows)
    atomic_json(OUT/f"test_summary{tag}.json",result)
    return result


def run():
    from afruit_grade_release import __file__ as grade_source

    if not (OUT/"frozen_policy.json").is_file() or not (OUT/"camera_resolution_policy.json").is_file():
        raise ValueError("Freeze the integration policy before evaluating")
    freeze_policy()
    freeze_camera_profile()
    selection=read(GRADE_OUT/"selection.json")
    standalone=read(GRADE_OUT/"test.json")
    if sha(MANIFEST) != selection["manifest_sha256"] or sha(GRADE_OUT/"test_predictions.json")!=standalone["predictions_sha256"]:
        raise ValueError("Standalone test evidence is not bound to frozen predictions")
    manifest=read(MANIFEST)
    cases=sorted((r for r in manifest["samples"] if r["split"]=="test"),key=lambda r:r["path"])
    if set(int(r["grade"]) for r in cases)!={0,1,2}:
        raise ValueError("Frozen test split missing apple grades")
    for r in cases:
        if sha(ROOT/r["path"]) != r["sha256"]:
            raise ValueError("AFruitDB held-out input hash mismatch")
    protocol=dict(
        frozen_policy_sha256=sha(OUT/"frozen_policy.json"),
        manifest_sha256=sha(MANIFEST),grade_weight_sha256=sha(GRADE_OUT/"weights.pt"),
        grade_policy_sha256=sha(GRADE_OUT/"selection.json"),grade_code_sha256=sha(Path(grade_source)),
        stage_composite_sha256=STAGE_WEIGHT_SHA,defect_shared_sha256=WARM_SHA,
        defect_scratch_sha256=SOURCE_SHA,stage_threshold=0.25,defect_workpoints=PRIMARY,
        crop_context_per_side=CROP_CONTEXT, stages=["immature","semi-mature","mature"],
        GPU=torch.cuda.get_device_name(0), cpu_threads=torch.get_num_threads(),warm_frames=5,
        timing="Serial batch-one image arrays in memory, full-frame maturity instance masks and lesion detector each invoked on every frame. All fruit crops classified in batches of up to 12. Return CPU-accessible masks, boxes, per-instance predictions, associations. Separate synchronous overlay timing; excludes camera, image decode, GUI refresh, disk and robot control.",
        scenarios=["historic: 20 stage and 20 SSDA images, 15 rounds, grades on every detected fruit",
                   "market_frame: up to 20 held-out AFruitDB images, grades on detected instances",
                   "market_known_crop: the same provided single-fruit crops, grade always invoked once alongside both prior tasks"],
    )
    atomic_json(OUT/"protocol.json",protocol)
    pipe=QualityPipeline()
    with torch.inference_mode():
        test=evaluate_frozen_test(pipe,cases)
        evaluate_frozen_test(pipe,cases,long_edge=1280)
        historic=[(task,p) for task in ("stage","defect") for p in spaced(source_spec(task)["images"],20)]
        market=[("AFruitDB",ROOT/r["path"]) for r in spaced(cases,20)]
        speed=benchmark(pipe,"historic",historic,15)
        benchmark(pipe,"market_frame",market,int(np.ceil(600/len(market))))
        benchmark(pipe,"market_known_crop",market,int(np.ceil(600/len(market))),known=True)
        benchmark(pipe,"market_frame_1280",market,int(np.ceil(600/len(market))),long_edge=1280)
    print("Held-out",test["pipeline_single_instance"]["confusion"],"historic",speed["all"],flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze-policy",action="store_true")
    parser.add_argument("--repeat-known-speed",action="store_true",
                        help="Repeat just the supplied-crop speed measurements after an overlapping system workload")
    args=parser.parse_args()
    configure(OUT,"cpu" if args.freeze_policy else "cuda")
    torch.set_num_threads(2)
    cv2.setNumThreads(1)
    if args.freeze_policy:
        freeze_policy()
        freeze_camera_profile()
    elif args.repeat_known_speed:
        if not (OUT/"latency_market_known_crop.json").exists():
            raise ValueError("No original speed run to repeat")
        seen=read(OUT/"latency_market_known_crop.json")
        if (OUT/"latency_market_known_crop_repeat.json").exists():
            raise ValueError("A repeat speed record already exists")
        paths=[(r["source"],r["path"]) for r in seen["files"]]
        final_pipe=QualityPipeline()
        benchmark(final_pipe,"market_known_crop_repeat",paths,seen["rounds"],known=True)
        benchmark(final_pipe,"market_known_crop_1280",paths,seen["rounds"],known=True,long_edge=1280)
    else:
        run()
