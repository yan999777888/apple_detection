"""Dedicated strong Base C0/C1 launcher; no query student path exists."""
from pathlib import Path
import argparse
import json
import sys
import traceback

ROOT=Path(__file__).absolute().parents[1]
sys.path.insert(0,str(ROOT/"src"))
sys.path.insert(0,str(ROOT/"strong_runtime/src"))


class Tee:
    def __init__(self,console,log):self.console,self.log=console,log
    def write(self,value):self.console.write(value);self.log.write(value);self.log.flush()
    def flush(self):self.console.flush();self.log.flush()
    def __getattr__(self,name):return getattr(self.console,name)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",type=Path,required=True)
    parser.add_argument("--device",choices=("cuda",),default="cuda")
    parser.add_argument("--smoke",action="store_true")
    parser.add_argument("--resume",action="store_true")
    parser.add_argument("--stop-after-updates",type=int)
    parser.add_argument("--continue-after-gate",type=Path)
    parser.add_argument("--max-smoke-updates",type=int,default=2)
    parser.add_argument("--smoke-indices",type=int,nargs=4)
    args=parser.parse_args()
    from apple_strong.training import run,load_config,atomic_json
    cfg=load_config(args.config);out=cfg["paths"]["output"]
    if out.exists() and any(out.iterdir()) and not args.resume:raise FileExistsError("Use fresh strong output or exact resume")
    out.mkdir(parents=True,exist_ok=True)
    with (out/"runtime.log").open("a" if args.resume else "x",encoding="utf-8") as log:
        stdout,stderr=sys.stdout,sys.stderr
        sys.stdout,sys.stderr=Tee(stdout,log),Tee(stderr,log)
        try:
            print(json.dumps(dict(event="strong_launch",arm=cfg["arm"],smoke=args.smoke,
                                 python=sys.executable,config=str(args.config),output=str(out))),flush=True)
            result=run(args.config,device=args.device,smoke=args.smoke,resume=args.resume,
                       stop_after_updates=args.stop_after_updates,continue_after_gate=args.continue_after_gate,
                       max_smoke_updates=args.max_smoke_updates,smoke_indices=args.smoke_indices)
            print(json.dumps({k:result[k] for k in ("phase","arm","updates","checkpoint")}),flush=True)
            return 0
        except BaseException as error:
            traceback.print_exc()
            status_path=out/"training_status.json"
            current=json.loads(status_path.read_text(encoding="utf-8")) if status_path.exists() else {}
            if current.get("phase")!="failed":
                atomic_json(status_path,dict(phase="failed",error_type=type(error).__name__,error=str(error),last_status=current))
            return 1
        finally:
            sys.stdout,sys.stderr=stdout,stderr


if __name__=="__main__":
    raise SystemExit(main())
