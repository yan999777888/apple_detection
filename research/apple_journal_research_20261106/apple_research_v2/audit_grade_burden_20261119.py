"""Replay the frozen native-fruit grader on the actual development inputs."""
import time
import numpy as np
import torch
import afruit_grade_train as io
from matched_grade_loader import load
import grade_pairs_burden_20261119 as b
import train_grade_burden_20261119 as fit
import train_grade_pairs_20261119 as prior


@torch.inference_mode()
def run():
    torch.set_num_threads(2);prior.cv2.setNumThreads(1)
    torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    model,_,side=load(fit.OUT,"cuda")
    if not isinstance(side,b.BurdenSide):raise ValueError("Incorrect grader input")
    reference=io.read(fit.OUT/"development_predictions.json")
    lookup={(r["path"],r["panel"]):r["proposed"] for r in reference["rows"]}
    changes=[];n=0;timed=[]
    for r,im,h in prior.views("val"):
        if im is None:continue
        pred=int(model(b.prepare([im],"cuda",[h])).argmax(1).item())
        if pred!=lookup[r["path"],r["panel"]]:changes.append(dict(row=r,replayed=pred,prior=lookup[r["path"],r["panel"]]))
        if len(timed)<12 and r["panel"]=="auto":timed.append((im,h))
        n+=1
    ms=[]
    for i in range(101):
        im,h=timed[i%12]
        torch.cuda.synchronize();start=time.perf_counter()
        out=model(b.prepare([im],"cuda",[h])).argmax(1).item()
        torch.cuda.synchronize()
        if i>=5:ms.append(1000*(time.perf_counter()-start))
    summary=dict(inputs_replayed=n,argmax_match=n-len(changes),changes=changes,
                 grade_only=dict(measurements=96,warmup=5,frames=12,mean_ms=float(np.mean(ms)),
                        p95_ms=float(np.percentile(ms,95)),fps=float(1000/np.mean(ms)),
                        includes="decoded native fruit crop and existing proposals to predicted fruit grade; no scene networks or overlay"))
    io.atomic(fit.OUT/"replay_and_latency.json",summary)
    print(summary,flush=True)
    if changes or len(lookup)!=n:raise ValueError("Development results differ from production replay")


if __name__=="__main__":run()
