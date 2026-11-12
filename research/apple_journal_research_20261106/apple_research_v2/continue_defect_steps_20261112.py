"""Sequential continuation; waits for existing OOF, never launches a second GPU worker."""
import sys,os,json,time,subprocess,traceback,ctypes,msvcrt
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'apple_research_v2/joint'))
from joint_state_v1 import configure,atomic_json,sha
def read_shared(path):
    api=ctypes.WinDLL('kernel32',use_last_error=True);api.CreateFileW.argtypes=[ctypes.c_wchar_p,ctypes.c_uint32,ctypes.c_uint32,ctypes.c_void_p,ctypes.c_uint32,ctypes.c_uint32,ctypes.c_void_p];api.CreateFileW.restype=ctypes.c_void_p
    h=api.CreateFileW(str(path),0x80000000,7,None,3,128,None)
    if h==ctypes.c_void_p(-1).value:raise OSError(ctypes.get_last_error(),str(path))
    fd=msvcrt.open_osfhandle(h,os.O_RDONLY)
    with os.fdopen(fd,'r',encoding='utf8') as f:return json.load(f)
def main():
    out=ROOT/'experiments/defect_algorithm_steps_20261112_v2/continuation'
    if out.exists():raise ValueError('One continuation only')
    configure(out,'cpu');status=out/'status.json'
    jobs=[('deployment_parity','verify_defect_modes_20261112.py',[]),('GPU_smoke','defect_shield_train_20261112.py',['--smoke']),('six_arm_training','defect_shield_train_20261112.py',[])]
    sources={str(ROOT/'apple_research_v2'/name):sha(ROOT/'apple_research_v2'/name) for _,name,_ in jobs}
    sources[str(ROOT/'apple_research_v2/defect_algorithm_modes_20261112.py')]=sha(ROOT/'apple_research_v2/defect_algorithm_modes_20261112.py')
    atomic_json(out/'protocol.json',{'jobs':jobs,'frozen_sources':sources,'wait_for_existing_OOF':True,'no_parallel_GPU':True})
    try:
        atomic_json(status,{'phase':'waiting_existing_OOF','pid':os.getpid()})
        while True:
            state=read_shared(out.parent/'oof_v2/status.json')
            if state['phase']=='complete':break
            if state['phase']=='failed':raise RuntimeError('OOF failed: '+state.get('error','unknown'))
            time.sleep(15)
        for name,script,args in jobs:
            for path,digest in sources.items():assert sha(path)==digest,'Frozen source changed '+path
            atomic_json(status,{'phase':name,'pid':os.getpid()})
            with (out/(name+'.log')).open('w',encoding='utf8') as log:subprocess.run([sys.executable,'-B',str(ROOT/'apple_research_v2'/script),*args],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
            atomic_json(out/(name+'_completed.json'),{'phase':'complete','script_sha256':sources[str(ROOT/'apple_research_v2'/script)]})
        atomic_json(status,{'phase':'experiments_complete_report_pending'})
    except BaseException as e:atomic_json(status,{'phase':'failed','error':repr(e),'traceback':traceback.format_exc()});raise
if __name__=='__main__':main()
