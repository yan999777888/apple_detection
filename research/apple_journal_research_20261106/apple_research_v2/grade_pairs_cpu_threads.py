"""Check image-byte equivalence before timing an opt-in OpenCV worker-count change."""
from time import perf_counter
import argparse
import cv2
import numpy as np
import torch
import grade_pairs_dev_speed as speed
import afruit_visible_quality as visible
from matched_grade_loader import install

DEST=speed.OUT/"opencv_thread_trial"
MODES=(1,4)
FULL_MODES=(1,4,8)
TRIAL_ROUNDS=15


def cpu_check():
    frames=[speed.af.read_frame(r) for r in speed.select_images()]
    timings={str(n):[] for n in MODES}
    checks=[]
    for frame in frames:
        variants=[]
        for n in MODES:
            cv2.setNumThreads(n);variants.append(visible.prepare_input(frame,1280))
        checks.append(np.array_equal(*variants))
    for turn in range(5):
        for n in (MODES if turn%2==0 else MODES[::-1]):
            cv2.setNumThreads(n)
            for frame in frames:
                t=perf_counter();visible.prepare_input(frame,1280)
                timings[str(n)].append((perf_counter()-t)*1000)
    rec=dict(images=len(frames),same_rgb8=sum(checks),threads={n:speed.about(t) for n,t in timings.items()})
    speed.io.atomic(DEST/"cpu_check.json",rec)
    print(rec,flush=True)
    return all(checks)


@torch.inference_mode()
def end_to_end():
    if (DEST/"timings.json").is_file():raise ValueError("Finished trial exists")
    check=speed.io.read(DEST/"cpu_check.json")
    if (check["same_rgb8"]!=20 or check["threads"]["4"]["mean_ms"]>=check["threads"]["1"]["mean_ms"]):
        raise ValueError("Only benchmark a byte-exact CPU optimization that improved resize latency")
    rows=speed.select_images();frames=[speed.af.read_frame(r) for r in rows]
    speed.io.atomic(DEST/"protocol.json",dict(development_only=True,frame_hashes=[r["sha256"] for r in rows],
          cpu_check_sha=speed.af.grade.file_sha(DEST/"cpu_check.json"),model=speed.af.grade.file_sha(speed.BASE/"weights.pt"),
          method="3 CPU settings (1,4,8). 5 full warmups each. Balance measured block order using all permutations and a rotating final three-round Latin cycle. Same 20 frames in each mode and round, 300 frames per setting. Baseline model in ordinary eager mode. All modules + draw. Measure with GPUs otherwise reserved.",
          extra="Secondary latency diagnostic only: use all six remaining AF development scenes with zero or multiple prior native stage instances, 10 repeated passes at the fastest native-to-overlay setting. Report cohorts separately; do not fold them into the preselected 20.",
          trial_code=speed.af.grade.file_sha(__file__)))
    telemetry=[speed.audit_processes()];torch.set_num_threads(2)
    pipe=visible.QualityPipeline();install(pipe,speed.BASE)
    flags=[]
    for frame in frames:
        variants=[]
        for n in FULL_MODES:
            cv2.setNumThreads(n);im=visible.prepare_input(frame,1280)
            a,m=pipe(im);variants.append((im,a,m,visible.overlay(im,a,m)))
        source,a,m,ov=variants[0]
        flags.append(all(np.array_equal(source,other) and speed.comparable(a,b) and np.array_equal(m,m2) and np.array_equal(ov,ov2) for other,b,m2,ov2 in variants[1:]))
    speed.io.atomic(DEST/"equivalence.json",dict(total=len(flags),passed=sum(flags),score_tolerance=2e-6,identical_overlay=True if all(flags) else None))
    if not all(flags):raise ValueError("Change affected final outputs")
    for n in FULL_MODES:
        cv2.setNumThreads(n)
        for frame in frames[:5]:
            im=visible.prepare_input(frame,1280);r,m=pipe(im);visible.overlay(im,r,m)
    from itertools import permutations
    sequence=list(permutations(FULL_MODES))*2+[(1,4,8),(4,8,1),(8,1,4)]
    results={n:[] for n in FULL_MODES};rng=np.random.default_rng(speed.SEED)
    for turn in range(TRIAL_ROUNDS):
        if turn%4==0:telemetry.append(speed.audit_processes())
        ids=rng.permutation(len(frames))
        for n in sequence[turn]:
            cv2.setNumThreads(n)
            for i in ids:
                torch.cuda.synchronize();a=perf_counter();im=visible.prepare_input(frames[i],1280);b=perf_counter()
                r,m=pipe(im);torch.cuda.synchronize();c=perf_counter();visible.overlay(im,r,m);d=perf_counter()
                results[n].append(dict(fruit=len(r["fruits"]),index=int(i),round=turn,from_ready_to_overlay=(d-b)*1000,
                                       from_native_to_overlay=(d-a)*1000,from_ready_to_results=(c-b)*1000))
    report=dict(telemetry=telemetry,per_threads={n:speed.stats(v) for n,v in results.items()},records=results)
    best=min(FULL_MODES,key=lambda n:report["per_threads"][n]["all"]["from_native_to_overlay"]["mean_ms"])
    cv2.setNumThreads(best)
    rare_ids={r["sha256"]:r for r in speed.io.read(speed.af.OUT/"val_regions.json") if r["count"]!=1}
    rare=[r for r in speed.af.split_rows("val") if r["sha256"] in rare_ids]
    other=[speed.af.read_frame(r) for r in rare]
    for f in other:
        im=visible.prepare_input(f,1280);z,m=pipe(im);visible.overlay(im,z,m)
    rare_times=[]
    for turn in range(10):
        for i in rng.permutation(len(other)):
            torch.cuda.synchronize();a=perf_counter();im=visible.prepare_input(other[i],1280);b=perf_counter()
            z,m=pipe(im);torch.cuda.synchronize();c=perf_counter();visible.overlay(im,z,m);d=perf_counter()
            if len(z["fruits"])!=rare_ids[rare[i]["sha256"]]["count"]:raise ValueError("Previously cached development count changed")
            rare_times.append(dict(fruit=len(z["fruits"]),index=int(i),round=turn,from_ready_to_overlay=(d-b)*1000,
                                   from_native_to_overlay=(d-a)*1000,from_ready_to_results=(c-b)*1000))
    report["other_dev_instance_counts"]=dict(threads=best,hashes=[r["sha256"] for r in rare],
                                            n_views=len(rare),summary=speed.stats(rare_times),records=rare_times)
    telemetry.append(speed.audit_processes())
    speed.io.atomic(DEST/"timings.json",report)
    for n in FULL_MODES:print(n,report["per_threads"][n]["all"],flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("action",choices=["cpu","full"])
    if p.parse_args().action=="cpu":cpu_check()
    else:end_to_end()

