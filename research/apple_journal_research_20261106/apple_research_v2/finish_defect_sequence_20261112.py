"""Wait for the already queued six-arm runner, then calibrate and audit serially."""
import sys,os,json,time,subprocess,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'apple_research_v2/joint'))
from joint_state_v1 import configure,atomic_json,sha
from continue_defect_steps_20261112 import read_shared
def main():
    out=ROOT/'experiments/defect_algorithm_steps_20261112_v2/finish_sequence'
    if out.exists():raise ValueError('Single finishing queue only')
    configure(out,'cpu')
    scripts=['calibrate_defect_modes_oof_20261112.py','report_defect_algorithm_steps_20261112.py','finalize_defect_experiments_20261112.py'];frozen={s:sha(ROOT/'apple_research_v2'/s) for s in scripts};atomic_json(out/'protocol.json',{'sources':frozen,'no_parallel_GPU':True})
    try:
        atomic_json(out/'status.json',{'phase':'waiting_existing_training','pid':os.getpid()})
        while True:
            state=read_shared(out.parent/'continuation/status.json')
            if state['phase']=='experiments_complete_report_pending':break
            if state['phase']=='failed':raise RuntimeError('Previous phase failed: '+state.get('error',''))
            time.sleep(15)
        for script in scripts:
            assert sha(ROOT/'apple_research_v2'/script)==frozen[script]
            atomic_json(out/'status.json',{'phase':'running','script':script})
            with (out/(script+'.log')).open('w',encoding='utf8') as f:subprocess.run([sys.executable,'-B',str(ROOT/'apple_research_v2'/script)],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,check=True)
        atomic_json(out/'status.json',{'phase':'experiments_complete_delivery_pending'})
    except BaseException as e:atomic_json(out/'status.json',{'phase':'failed','error':repr(e),'traceback':traceback.format_exc()});raise
if __name__=='__main__':main()
