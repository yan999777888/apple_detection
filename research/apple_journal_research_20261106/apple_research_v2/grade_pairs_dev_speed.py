"""Frozen AF development speed audit, with optional equivalent single-fruit CUDA graph."""
import argparse
from collections import Counter
from contextlib import ExitStack
from pathlib import Path
import os
import subprocess
from time import perf_counter
from unittest.mock import patch
import cv2
import numpy as np
import torch
from torch import nn
import afruit_grade_crop_alignment as af
import afruit_grade_train as io
import afruit_visible_quality as visible
import matched_grade_loader as loaders
from grade_pairs_20261119 import PairBatch

DIR=af.grade.ROOT/"experiments/grade_pairs_20261119"
BASE=DIR/"sparse_verifier_v1"
OUT=DIR/"development_full_speed_v1"
SEED=19112026


def about(data):
    q=np.asarray(data,float)
    if not len(q):return None
    return dict(n=len(q),mean_ms=float(q.mean()),p50_ms=float(np.median(q)),p95_ms=float(np.percentile(q,95)),fps_from_mean=float(1000/q.mean()))


def audit_processes():
    text=subprocess.check_output(["nvidia-smi","--query-compute-apps=pid,process_name","--format=csv,noheader"],text=True,errors="replace")
    matches=[]
    for line in text.splitlines():
        if "," not in line:continue
        pid,name=line.split(",",1)
        if pid.strip().isdigit() and int(pid)!=os.getpid() and any(x in name.lower() for x in ("python","onnxruntime","trtexec","tensor","cuda_sample","ffmpeg")):
            matches.append([pid.strip(),Path(name.strip()).name])
    if matches:raise RuntimeError(f"Other candidate compute processes active: {matches}")
    env=subprocess.check_output(["nvidia-smi","--query-gpu=name,driver_version,temperature.gpu,clocks.sm,power.draw,utilization.gpu,memory.used","--format=csv,noheader"],text=True,errors="replace")
    return dict(time_local=__import__("datetime").datetime.now().isoformat(timespec="seconds"),gpu=env.strip(),foreign_compute_matches=matches,desktop_graphics_clients=len(text.splitlines()))


class SingleGraph(nn.Module):
    """Serial inference on one fixed-size fruit. Other batch sizes use ordinary forward."""
    def __init__(self,base):
        super().__init__()
        self.base=base;self.graph=None
    def forward(self,args):
        if args.parent.shape[0]!=1:return self.base(args)
        if self.graph is None:
            self.fixed=PairBatch(*(torch.empty_like(a).copy_(a) for a in args))
            side=torch.cuda.Stream();side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                for _ in range(4):self.base(self.fixed)
            torch.cuda.current_stream().wait_stream(side)
            self.graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph):
                self.reused=self.base(self.fixed)
        for d,x in zip(self.fixed,args,strict=True):d.copy_(x)
        self.graph.replay()
        return self.reused


def select_images():
    rows=af.split_rows("val")
    out=[]
    for c,n in ((0,8),(1,6),(2,6)):
        part=sorted((r for r in rows if r["grade"]==c),key=lambda r:r["sha256"])
        out+=[part[j] for j in np.linspace(0,len(part)-1,n).round().astype(int)]
    rng=np.random.default_rng(SEED);rng.shuffle(out)
    return out


def stats(records):
    return {group:{field:about([r[field] for r in records if fn(r)]) for field in ("from_ready_to_overlay","from_native_to_overlay","from_ready_to_results")} for group,fn in
            (("all",lambda r:True),("zero",lambda r:r["fruit"]==0),("one",lambda r:r["fruit"]==1),("multi",lambda r:r["fruit"]>1))}


def run_primary(pipe,frames,name,snapshots):
    rng=np.random.default_rng(SEED);rows=[]
    with torch.inference_mode():
        for i in range(5):
            im=visible.prepare_input(frames[i],1280)
            v,m=pipe(im);visible.overlay(im,v,m)
        for r in range(30):
            if r%6==0:snapshots.append(audit_processes())
            for i in rng.permutation(len(frames)):
                torch.cuda.synchronize();t=perf_counter()
                im=visible.prepare_input(frames[i],1280);t1=perf_counter()
                result,mask=pipe(im);torch.cuda.synchronize();t2=perf_counter()
                visible.overlay(im,result,mask);t3=perf_counter()
                rows.append(dict(round=r,index=int(i),fruit=len(result["fruits"]),hints=len(result["defects"]),
                                 from_ready_to_overlay=(t3-t1)*1000,from_native_to_overlay=(t3-t)*1000,from_ready_to_results=(t2-t1)*1000))
    report=dict(mode=name,n=len(rows),fruit_counts=dict(Counter(z["fruit"] for z in rows)),summary=stats(rows),rows=rows)
    io.atomic(OUT/(name+".json"),report)
    print(name,report["summary"]["all"],flush=True)
    return report


def profile(pipe,frames,name):
    rows=[];clock={};sync=torch.cuda.synchronize
    def mark(key):
        sync();clock[key]=perf_counter()
    def timed(key,fn):
        def wrap(*args,**kwargs):
            mark(key+"_begin");r=fn(*args,**kwargs);mark(key+"_end");return r
        return wrap
    def start(key,fn):
        def wrap(*args,**kwargs):
            mark(key);return fn(*args,**kwargs)
        return wrap
    with ExitStack() as stack,torch.inference_mode():
        for obj,key,alias in ((visible,"defect_boxes","defect"),(visible,"associate","assign"),(pipe,"preprocess_grade_with_boxes","prep"),(pipe.grade,"forward","grade")):
            stack.enter_context(patch.object(obj,key,timed(alias,getattr(obj,key))))
        stack.enter_context(patch.object(pipe.predictor,"preprocess",start("stage_begin",pipe.predictor.preprocess)))
        rng=np.random.default_rng(SEED)
        for _ in range(5):
            for i in rng.permutation(len(frames)):
                clock.clear();mark("start");im=visible.prepare_input(frames[i],1280);mark("ready")
                result,m=pipe(im);mark("reported");visible.overlay(im,result,m);mark("end")
                c=clock
                vals=dict(resize=c["ready"]-c["start"],defect=c["defect_end"]-c["defect_begin"],
                          maturity_and_crops=c["assign_begin"]-c["stage_begin"],
                          lesion_association=c["assign_end"]-c["assign_begin"],
                          grade_prepare=c.get("prep_end",0)-c.get("prep_begin",0),
                          grade_forward=c.get("grade_end",0)-c.get("grade_begin",0),
                          draw=c["end"]-c["reported"])
                vals["remaining_host_and_scores"]=c["end"]-c["start"]-sum(vals.values())
                rows.append({k:v*1000 for k,v in vals.items()})
    out={k:about([r[k] for r in rows]) for k in rows[0]}
    io.atomic(OUT/(name+"_stages.json"),out)
    return out


def comparable(a,b):
    if isinstance(a,dict):return a.keys()==b.keys() and all(comparable(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)):return len(a)==len(b) and all(comparable(x,y) for x,y in zip(a,b))
    if isinstance(a,float):return bool(np.isclose(a,b,atol=2e-6,rtol=2e-6))
    return a==b


def work():
    if (OUT/"receipt.json").exists():raise ValueError("Completed benchmark already exists")
    cv2.setNumThreads(1);torch.set_num_threads(2)
    selected=select_images()
    sel=io.read(BASE/"selection.json");images=[af.read_frame(r) for r in selected]
    p=dict(selected=[r["sha256"] for r in selected],model=sel["weights_sha256"],source_split="AFruitDB development only. 8 good, 6 minor and 6 severe evenly spaced over sorted source hashes. No blind cohort access.",
           policy="Preheat 5. Shuffle 20 images every round with a fixed seed, 30 serial rounds per ordinary end-to-end, 5 rounds in independent synchronized profiling. 1280 long edge from original image, no I/O, same original RGB scene mode. All per-fruit boxes/masks/maturity/visible defect boxes, grade, drawing. Fixed 1-fruit optional graph benchmark if native-to-overlay >50 ms; mandatory direct whole-output comparison.",
           torch=str(torch.__version__),cuda=str(torch.version.cuda),dtype="original native branch dtypes, classifier FP32",source_code=af.grade.file_sha(__file__))
    if (OUT/"protocol.json").exists() and io.read(OUT/"protocol.json")!=p:raise ValueError("Frozen benchmark modified")
    io.atomic(OUT/"protocol.json",p);snapshots=[audit_processes()]
    pipe=visible.QualityPipeline();loaders.install(pipe,BASE)
    eager=run_primary(pipe,images,"eager",snapshots);profile(pipe,images,"eager")
    trial=None
    if eager["summary"]["all"]["from_native_to_overlay"]["mean_ms"]>50:
        eager_model=pipe.grade;graph_model=SingleGraph(eager_model).eval()
        # Warm capture and assert complete scene output equivalence before timing.
        with torch.inference_mode():
            checks=[]
            for frame in images:
                im=visible.prepare_input(frame,1280)
                pipe.grade=eager_model;a,ma=pipe(im)
                pipe.grade=graph_model;b,mb=pipe(im)
                checks.append(bool(comparable(a,b) and np.array_equal(ma,mb) and
                                   np.array_equal(visible.overlay(im,a,ma),visible.overlay(im,b,mb))))
        io.atomic(OUT/"graph_equivalence.json",dict(total=len(checks),passed=sum(checks),tolerance_for_probabilities=2e-6))
        if not all(checks):raise ValueError("Graph changed scene prediction")
        trial=run_primary(pipe,images,"single_graph",snapshots);profile(pipe,images,"single_graph")
    snapshots.append(audit_processes());io.atomic(OUT/"receipt.json",dict(protocol_sha=af.grade.file_sha(OUT/"protocol.json"),telemetry=snapshots,
             no_parallel_research_process_observed=True,hardware="RTX4060 Laptop under Windows WDDM, ordinary desktop processes still present.",
             output=dict(eager=eager["summary"]["all"],single_graph=trial and trial["summary"]["all"])))
    print("Finished",flush=True)


if __name__=="__main__":work()

