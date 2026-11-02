"""Run an explicit pre-mask experiment plan sequentially, one GPU child at a time."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback
from evaluate_fast_deployment import REPO,write,now,sha


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan',type=Path,required=True)
    a=p.parse_args();plan_path=a.plan.resolve()
    plan=json.loads(plan_path.read_text(encoding='utf-8'))
    report=(REPO/plan['report_directory']).resolve()
    if report.drive.upper()!='F:':raise ValueError('Report must be on F:')
    report.mkdir(parents=True,exist_ok=False)
    lock=report/'pipeline.lock'
    with lock.open('x') as stream:stream.write(str(os.getpid()))
    status={'schema':'premask_jobs_v1','phase':'running','pid':os.getpid(),
            'created_utc':now(),'plan':str(plan_path),'plan_sha256':sha(plan_path),
            'training':False,'original_weights_only':True,'automatic_promotion':False,'stages':[]}
    events=[];child=None
    write(report/'pipeline_status.json',status);write(report/'events.json',events)
    try:
        for item in plan['stages']:
            name=item['id']
            if Path(name).name!=name:raise ValueError('Invalid stage name')
            script=(REPO/item['script']).resolve()
            if script not in [(REPO/'scripts/evaluate_premask.py').resolve(),
                              (REPO/'scripts/summarize_premask_validation.py').resolve()]:
                raise ValueError('Only approved eval/CPU summary jobs are allowed')
            arguments=[sys.executable,'-B',str(script),*item['arguments']]
            row={'id':name,'phase':'running','started_utc':now(),'arguments':arguments,
                 'script_sha256':sha(script),'log':str(report/(name+'.log'))}
            status['stages'].append(row);status.update(current_stage=name,updated_utc=now())
            print('PREMASK_JOB_START '+name,flush=True)
            with Path(row['log']).open('x',encoding='utf-8') as log:
                child=subprocess.Popen(arguments,cwd=str(REPO),stdout=log,stderr=subprocess.STDOUT)
                row['pid']=child.pid;status['child_pid']=child.pid
                write(report/'pipeline_status.json',status)
                code=child.wait()
            child=None;row.update(exit_code=code,completed_utc=now(),phase='complete' if code==0 else 'failed')
            if code:raise RuntimeError('Stage failed: '+name+'; see '+row['log'])
            if sha(script)!=row['script_sha256'] or sha(plan_path)!=status['plan_sha256']:
                raise RuntimeError('Job source or plan changed during execution')
            events.append({'event':'stage_complete','id':name,'time':now()})
            status.update(child_pid=None,updated_utc=now())
            write(report/'pipeline_status.json',status);write(report/'events.json',events)
            print('PREMASK_JOB_COMPLETE '+name,flush=True)
        status.update(phase='complete',completed_utc=now(),child_pid=None)
        write(report/'pipeline_status.json',status)
    except BaseException as error:
        if child is not None and child.poll() is None:
            child.terminate()
            try:child.wait(timeout=20)
            except subprocess.TimeoutExpired:child.kill();child.wait(timeout=20)
        status.update(phase='failed',error=str(error),traceback=traceback.format_exc(),updated_utc=now(),child_pid=None)
        write(report/'pipeline_status.json',status)
        raise
    finally:
        lock.unlink(missing_ok=True)


if __name__=='__main__':main()
