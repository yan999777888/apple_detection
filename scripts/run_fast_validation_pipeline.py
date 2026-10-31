"""Sequential full-VAL deployment comparison; original weights stay immutable.

One child GPU evaluator at a time. Checkpoints are never trained or replaced.
Runs original-view batch-one and paired-backbone modes separately, since AMP
batching can alter numerical outputs. Local tracking accuracy is not inferred.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO/'scripts'))
from reproduce_strong_baseline import write, sha, source_inventory


def now():return datetime.now(timezone.utc).isoformat()


def claim_report(report):
    """Claim a fresh report without replacing completed or failed evidence."""
    report=Path(report)
    report.mkdir(parents=True,exist_ok=True)
    protected=('pipeline_status.json','events.json','validation_summary.json','完整验证结果.md')
    def reject_existing():
        if any((report/name).exists() for name in protected):
            raise FileExistsError('Existing pipeline evidence would be overwritten: '+str(report))
    reject_existing()
    lock=report/'pipeline.lock'
    with lock.open('x',encoding='utf-8') as stream:
        stream.write(json.dumps({'pid':os.getpid(),'created_utc':now()}))
    try:
        reject_existing()
    except BaseException:
        lock.unlink(missing_ok=True)
        raise
    return lock


def stop_owned_child(child):
    """Never let cleanup hide the original error or falsely assert exit."""
    result={'pid':child.pid,'exit_confirmed':False,'errors':[]}
    try:
        code=child.poll()
        if code is not None:
            result.update(exit_confirmed=True,returncode=code)
            return result
    except BaseException as error:
        result['errors'].append('poll: '+str(error))
    for action,timeout in (('terminate',20),('kill',10)):
        try:
            getattr(child,action)()
            result[action+'_requested']=True
        except BaseException as error:
            result['errors'].append(action+': '+str(error))
        try:
            code=child.wait(timeout=timeout)
            result.update(exit_confirmed=True,returncode=code)
            return result
        except BaseException as error:
            result['errors'].append('wait after '+action+': '+str(error))
    return result


def stage_arguments(stage,plan):
    return [sys.executable,'-B',str(REPO/'scripts/evaluate_fast_deployment.py'),
            '--config',str(REPO/plan['baseline_config']),'--stage',stage['stage'],
            '--output',str((REPO/stage['output']).resolve()),'--precision','fp16','--pre-nms-threshold','none',
            '--roi-chunk-size','64','--roi-precision','autocast','--no-skip-dead-amodal',
            '--short-side',str(stage['short_side']),'--max-size',str(stage['max_size']),
            '--views',stage['views'],'--progress-every','25',
            '--paired-backbone-batch2' if stage['paired'] else '--no-paired-backbone-batch2']


def baseline_evidence_files(plan):
    cfgpath=REPO/plan['baseline_config']
    cfg=json.loads(cfgpath.read_text(encoding='utf-8-sig'))
    base=(cfgpath.parent/cfg['path_base']).resolve()
    original=REPO/'runs/strong_baseline_20261030'
    return [original/'base/prediction_metadata.json',original/'small/prediction_metadata.json',
            original/'fusion_summary.json',(base/cfg['paths']['gate']).resolve()]


def run(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',type=Path,required=True)
    args=parser.parse_args(argv)
    plan_path=args.plan.resolve()
    plan=json.loads(plan_path.read_text(encoding='utf-8'))
    report=(REPO/plan['report_directory']).resolve()
    assert report.drive.upper()=='F:'
    lock=claim_report(report)
    status={'schema':'fast_validation_pipeline_v1','phase':'initializing','pid':os.getpid(),
            'started_utc':now(),'plan_path':str(plan_path),'plan_sha256':sha(plan_path),
            'training':False,'original_strong_weights':True,'stages':[],'events':[]}
    pinned={}
    child=None
    release_lock=True
    entry=None
    def save():
        status['updated_utc']=now()
        write(report/'pipeline_status.json',status)
        write(report/'events.json',status['events'])
    def event(kind,**fields):
        status['events'].append({'time_utc':now(),'kind':kind,**fields});save()
    try:
        pinned.update(source_inventory())
        pinned.update({name:sha(REPO/name) for name in plan['pin_sources']})
        baseline_pins={str(path.relative_to(REPO)):sha(path) for path in baseline_evidence_files(plan)}
        pinned.update(baseline_pins)
        status['baseline_evidence_sha256']=baseline_pins
        status['source_sha256']=pinned
        for name,expected in plan.get('expected_source_sha256',{}).items():
            assert sha(REPO/name)==expected,'Configured source hash differs: '+name
        save()
        for stage in plan['stages']:
            assert sha(plan_path)==status['plan_sha256']
            assert all(sha(REPO/name)==digest for name,digest in pinned.items()),'Pinned source changed'
            directory=(REPO/stage['output']).resolve()
            mode=stage['stage']
            metadata=directory/('fusion_summary.json' if mode=='fuse' else mode+'/prediction_metadata.json')
            assert not metadata.exists(),'Existing evidence would be overwritten: '+str(metadata)
            arguments=stage_arguments(stage,plan)
            entry={'id':stage['id'],'phase':'running','started_utc':now(),'arguments':arguments,
                   'metadata':str(metadata),'log':str(report/(stage['id']+'.log'))}
            status['stages'].append(entry)
            status.update(phase='running',current_stage=stage['id'],child_pid=None)
            with Path(entry['log']).open('x',encoding='utf-8') as log:
                child=subprocess.Popen(arguments,cwd=REPO,stdout=log,stderr=subprocess.STDOUT,
                                       creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
                status['child_pid']=child.pid
                event('stage_started',stage=stage['id'],child_pid=child.pid)
                while child.poll() is None:
                    if metadata.exists():
                        try:
                            m=json.loads(metadata.read_text(encoding='utf-8'))
                            entry['progress']={k:m.get(k) for k in ('phase','images','GT','updated_utc','error')}
                        except (OSError,json.JSONDecodeError):pass
                    save();time.sleep(10)
                code=child.returncode
                status['child_pid']=None;child=None
            entry['exit_code']=code
            if code:
                entry['phase']='failed'
                raise RuntimeError(f"Stage {stage['id']} failed (exit {code}); inspect {entry['log']}")
            m=json.loads(metadata.read_text(encoding='utf-8'))
            assert m['phase']=='complete' and (m['images'],m['GT'],m['empty'])==(814,3449,13)
            entry.update(phase='complete',completed_utc=now(),metadata_sha256=sha(metadata),
                         metrics=m.get('view_metrics',m.get('metrics')))
            event('stage_complete',stage=stage['id'],metrics=entry['metrics'])
        # All source-frame variants have now been independently re-inferred.
        status.update(phase='summarizing',current_stage='derived_cutoff_and_gates')
        save()
        summary=summarize(plan,report,stage_receipts=status['stages'],pinned=pinned)
        assert sha(plan_path)==status['plan_sha256'],'Validation plan changed'
        assert all(sha(REPO/name)==digest for name,digest in pinned.items())
        status.update(phase='complete',current_stage=None,completed_utc=now(),
                      summary=str(report/'validation_summary.json'),acceptance=summary['acceptance'])
        event('pipeline_complete',acceptance=summary['acceptance'])
    except BaseException as error:
        original_traceback=traceback.format_exc()
        if entry is not None and entry['phase']=='running':entry['phase']='failed'
        status.update(phase='failed',error=str(error),traceback=original_traceback,
                      child_pid=child.pid if child is not None else None,
                      cleanup={'phase':'stopping_owned_child' if child is not None else 'no_child'})
        # Persist the primary failure before cleanup; a failed wait must not
        # replace it or remove the lock while a GPU child may still be alive.
        release_lock=child is None
        try:
            event('pipeline_failed',error=str(error))
        except BaseException as persistence_error:
            status.setdefault('persistence_errors',[]).append(str(persistence_error))
        if child is not None:
            cleanup=stop_owned_child(child)
            release_lock=cleanup['exit_confirmed']
            status.update(cleanup=cleanup,child_pid=None if release_lock else child.pid,lock_retained=not release_lock)
            if not release_lock:
                try:
                    lock.write_text(json.dumps({'pipeline_pid':os.getpid(),'child_pid':child.pid,
                                               'reason':'Child exit unconfirmed; inspect failure status before removing lock',
                                               'updated_utc':now()}),encoding='utf-8')
                except BaseException as persistence_error:
                    status.setdefault('persistence_errors',[]).append(str(persistence_error))
        try:
            save()
        except BaseException as persistence_error:
            print('FAILED status persistence: '+str(persistence_error),file=sys.stderr,flush=True)
        raise
    finally:
        if release_lock:lock.unlink(missing_ok=True)


def read_bound_json(path,expected_sha256=None):
    path=Path(path).resolve()
    raw=path.read_bytes()
    digest=hashlib.sha256(raw).hexdigest()
    if expected_sha256 is not None and digest!=expected_sha256:
        raise ValueError('Evidence bytes changed: '+str(path))
    return json.loads(raw),{'path':str(path),'sha256':digest,'bytes':len(raw)}


def normalized_hashes(values):
    result={}
    for name,digest in values.items():
        key=name.replace('\\','/')
        if key in result and result[key]!=digest:
            raise ValueError('Conflicting source hash aliases: '+key)
        result[key]=digest
    return result


def require_sources(actual,pinned):
    current=normalized_hashes(pinned)
    recorded=normalized_hashes(actual)
    if not recorded or any(current.get(name)!=digest for name,digest in recorded.items()):
        raise ValueError('Evidence source inventory differs from pinned execution sources')


def require_full_scope(meta):
    if meta.get('phase')!='complete' or tuple(meta.get(k) for k in ('images','GT','empty'))!=(814,3449,13):
        raise ValueError('Complete original 814/3449/13 evidence is required')


def receipt_digest(path,receipts):
    key=str(Path(path).resolve())
    if key not in receipts:
        raise ValueError('No completed stage receipt for metadata: '+key)
    return receipts[key]


def add_binding(bindings,binding):
    path=binding['path']
    if path in bindings and bindings[path]!=binding:
        raise ValueError('Evidence changed between reads: '+path)
    bindings[path]=binding


def load_candidate_view(stage,view,plan,cfg,contract,evaluate,pinned,receipts,bindings):
    directory=REPO/stage['output']/stage['stage']
    meta_path=directory/'prediction_metadata.json'
    meta,meta_binding=read_bound_json(meta_path,receipt_digest(meta_path,receipts))
    require_full_scope(meta)
    require_sources(meta['source_sha256'],pinned)
    model=stage['stage']
    if (meta.get('model')!=model or meta.get('checkpoint_sha256')!=cfg[model+'_checkpoint_sha256']
            or meta.get('provenance_sha256')!=contract['provenance_sha256']
            or meta.get('annotation_sha256')!=contract['provenance']['binding']['current_annotation_sha256']
            or meta.get('config_sha256')!=sha(REPO/plan['baseline_config'])
            or meta.get('from_fresh_weight_inference') is not True or meta.get('original_strong_models_only') is not True
            or meta.get('training_performed') is not False or meta.get('source_before_after_identical') is not True
            or meta.get('config_before_after_identical') is not True):
        raise ValueError('Original strong checkpoint/data/fresh-inference lineage differs')
    restored=meta.get('strict_restore',{})
    if (restored.get('strict') is not True or restored.get('missing_keys')!=[] or restored.get('unexpected_keys')!=[]
            or restored.get('all_tensors_equal_to_checkpoint') is not True
            or meta.get('constructor_config',{}).get('backbone_name')!='convnext_'+model):
        raise ValueError('Original architecture/strict restoration evidence differs')
    expected_args=evaluate.build_parser().parse_args(stage_arguments(stage,plan)[3:])
    if meta['deployment']!=evaluate.deployment_spec(expected_args):
        raise ValueError('Candidate metadata differs from its locked stage specification')
    allowed=('original','hflip') if stage['views']=='both' else ('original',)
    if view not in allowed:
        raise ValueError('Requested view is absent from this stage specification')
    raw_sha=meta.get('predictions_sha256',{}).get(view)
    if not isinstance(raw_sha,str) or len(raw_sha)!=64:
        raise ValueError('Raw prediction hash is missing')
    rows,raw_binding=read_bound_json(directory/(view+'_predictions.json'),raw_sha)
    evaluate.audit_records(rows,contract)
    add_binding(bindings,meta_binding);add_binding(bindings,raw_binding)
    return meta,rows


def load_baseline(plan,cfg,contract,pinned,bindings):
    paths=baseline_evidence_files(plan)
    loaded=[]
    pinned_normal=normalized_hashes(pinned)
    for path in paths:
        key=str(path.relative_to(REPO)).replace('\\','/')
        if key not in pinned_normal:
            raise ValueError('Original baseline evidence was not pinned before inference: '+key)
        item,binding=read_bound_json(path,pinned_normal[key])
        add_binding(bindings,binding)
        loaded.append(item)
    base,small,fusion,gate=loaded
    for name,meta in (('base',base),('small',small)):
        require_full_scope(meta);require_sources(meta['source_sha256'],pinned)
        if (meta.get('model')!=name or meta.get('checkpoint_sha256')!=cfg[name+'_checkpoint_sha256']
                or meta.get('candidate_mode') is not False or meta.get('FP32') is not True
                or meta.get('source_before_after_identical') is not True
                or meta.get('from_fresh_weight_inference') is not True
                or meta.get('provenance_sha256')!=contract['provenance_sha256']):
            raise ValueError('Original FP32 single-model baseline lineage differs')
    require_full_scope(fusion)
    require_sources(fusion['source_sha256'],pinned);require_sources(gate['source_sha256'],pinned)
    if (gate.get('passed') is not True or tuple(gate.get(k) for k in ('images','GT','empty'))!=(814,3449,13)
            or fusion.get('candidate_mode') is not False or fusion.get('from_fresh_weight_inference') is not True
            or fusion.get('fusion')!=cfg['fusion'] or gate.get('fusion')!=fusion['at_confidence_0_75']
            or gate.get('at_recall90')!=fusion['at_recall90']
            or any(gate.get(name+'_checkpoint_sha256')!=cfg[name+'_checkpoint_sha256'] for name in ('base','small'))
            or tuple(fusion['at_confidence_0_75'].get(k) for k in ('tp','fp','fn'))!=(3105,183,344)
            or Path(gate['baseline_directory']).resolve()!=(REPO/'runs/strong_baseline_20261030').resolve()):
        raise ValueError('Published 94.43% baseline gate/fusion evidence differs')
    return {'base':base,'small':small,'fusion':fusion,'gate':gate}


def verify_input_bindings(bindings):
    if any(sha(path)!=binding['sha256'] for path,binding in bindings.items()):
        raise ValueError('Summary inputs changed during derivation')


def summarize(plan,report,*,stage_receipts,pinned):
    spec=importlib.util.spec_from_file_location('locked_fast_eval',REPO/'scripts/evaluate_fast_deployment.py')
    evaluate=importlib.util.module_from_spec(spec);spec.loader.exec_module(evaluate)
    import flip_tta
    from apple_seg.evaluate import load_papple_contract
    planned={stage['id']:stage for stage in plan['stages']}
    if len(planned)!=len(plan['stages']) or len(stage_receipts)!=len(planned):
        raise ValueError('Every unique planned stage requires a completed receipt')
    receipts={}
    for receipt in stage_receipts:
        stage=planned.get(receipt['id'])
        if (stage is None or receipt.get('phase')!='complete' or receipt.get('exit_code')!=0
                or receipt.get('arguments')!=stage_arguments(stage,plan)):
            raise ValueError('A completed stage receipt differs from the fixed plan')
        key=str(Path(receipt['metadata']).resolve())
        expected=(REPO/stage['output']/('fusion_summary.json' if stage['stage']=='fuse' else stage['stage']+'/prediction_metadata.json')).resolve()
        if key!=str(expected) or key in receipts:
            raise ValueError('Stage receipt metadata path is wrong or duplicated')
        receipts[key]=receipt['metadata_sha256']
    bindings={}
    cfgpath=REPO/plan['baseline_config']
    cfg_key=str(cfgpath.relative_to(REPO)).replace('\\','/')
    cfg_sha=normalized_hashes(pinned).get(cfg_key)
    if cfg_sha is None:raise ValueError('Original baseline config was not pinned')
    cfg,cfg_binding=read_bound_json(cfgpath,cfg_sha)
    add_binding(bindings,cfg_binding)
    pathbase=(cfgpath.parent/cfg['path_base']).resolve()
    contract=load_papple_contract((pathbase/cfg['paths']['provenance']).resolve())
    baseline=load_baseline(plan,cfg,contract,pinned,bindings)
    summary={'scope':{'images':814,'GT':3449,'empty':13},'from_fresh_original_weights':True,
             'single_batch1_separate_from_paired_batch2':True,
             'local_tracking_P_at_R90':None,
             'local_tracking_note':'512 centre-crop speed simulation has no tracking/GT-derived crop accuracy claim.',
             'models':{},'fusion':{},'acceptance':{},'derived_artifacts':{},
             'baseline_evidence_sha256':{str(path):bindings[str(path.resolve())]['sha256'] for path in baseline_evidence_files(plan)}}
    def store(name,value):
        path=report/name
        write(path,value)
        summary['derived_artifacts'][name]={'path':str(path.resolve()),'sha256':sha(path),'bytes':path.stat().st_size}
    def stage_for(output,model):
        matches=[stage for stage in plan['stages'] if stage['output']==output and stage['stage']==model]
        if len(matches)!=1:raise ValueError('Candidate output has no unique planned model stage')
        return matches[0]
    policy=evaluate.policy(.35)
    for tag,output,model in [('base_single',plan['single_output'],'base'),('small_single',plan['single_output'],'small'),
                             ('small512_fullframe',plan['small512_output'],'small')]:
        meta,rows=load_candidate_view(stage_for(output,model),'original',plan,cfg,contract,evaluate,pinned,receipts,bindings)
        filtered=filter_records(rows,.35)
        metric,curve=evaluate.summarize(filtered,flip_tta.metrics,policy)
        summary['models'][tag]={'unfiltered':meta['view_metrics']['original'],'cutoff035_equivalent':metric,
                              'candidate_source':str(REPO/output/model/'original_predictions.json'),
                              'candidate_counts':{'unfiltered':sum(len(row['pred']['scores']) for row in rows),
                                                  'cutoff035':sum(len(row['pred']['scores']) for row in filtered)},
                              'equivalence_scope':'Same final score tensors; descending stable greedy NMS high-score prefix invariant. Derived from fresh unfiltered AMP predictions, not a second weight inference.',
                              'deployment':meta['deployment']}
        store(tag+'_cutoff035_predictions.json',filtered)
        store(tag+'_cutoff035_curve.json',curve)
    pairroot=REPO/plan['paired_output']
    fusion_meta_path=pairroot/'fusion_summary.json'
    unfiltered,fusion_binding=read_bound_json(fusion_meta_path,receipt_digest(fusion_meta_path,receipts))
    require_full_scope(unfiltered);require_sources(unfiltered['source_sha256'],pinned)
    if (unfiltered.get('config_sha256')!=cfg_sha or unfiltered.get('provenance_sha256')!=contract['provenance_sha256']
            or unfiltered.get('from_fresh_weight_inference') is not True or unfiltered.get('training_performed') is not False
            or unfiltered.get('original_strong_models_only') is not True or unfiltered.get('model')!='fuse'
            or unfiltered.get('source_before_after_identical') is not True or unfiltered.get('config_before_after_identical') is not True
            or unfiltered.get('fusion')!=cfg['fusion'] or unfiltered.get('mask_fusion') is not False):
        raise ValueError('Fresh paired fusion lineage/configuration differs')
    add_binding(bindings,fusion_binding)
    raw_fused,fused_binding=read_bound_json(pairroot/'fused_predictions.json',unfiltered['predictions_sha256'])
    evaluate.audit_records(raw_fused,contract);add_binding(bindings,fused_binding)
    del raw_fused
    layers=[];pair_metas={}
    for model,view in [('base','original'),('base','hflip'),('small','original'),('small','hflip')]:
        meta,rows=load_candidate_view(stage_for(plan['paired_output'],model),view,plan,cfg,contract,evaluate,pinned,receipts,bindings)
        pair_metas[model]=meta
        record_path=str((pairroot/model/(view+'_predictions.json')).resolve())
        expected_input=unfiltered['inputs'][model+'_'+view]
        if str(Path(expected_input['path']).resolve())!=record_path or expected_input['sha256']!=bindings[record_path]['sha256']:
            raise ValueError('Fusion input record binding differs from the audited view')
        layers.append(filter_records(rows,.35))
    for model,meta in pair_metas.items():
        meta_path=str((pairroot/model/'prediction_metadata.json').resolve())
        expected_input=unfiltered['inputs'][model+'_metadata']
        if (str(Path(expected_input['path']).resolve())!=meta_path or expected_input['sha256']!=bindings[meta_path]['sha256']
                or meta['deployment']!=unfiltered['deployment']
                or evaluate.runtime_numerics(meta)!=unfiltered['model_runtime_numerics']):
            raise ValueError('Fusion metadata/runtime binding differs from its model stage')
    fused=flip_tta.fuse_records(layers,**cfg['fusion'])
    evaluate.audit_records(fused,contract)
    metrics,curve=evaluate.summarize(fused,flip_tta.metrics,policy)
    store('paired_cutoff035_fused_predictions.json',fused)
    store('paired_cutoff035_fusion_curve.json',curve)
    summary['fusion']={'unfiltered':unfiltered['metrics'],'cutoff035_equivalent':metrics,
                       'mask_fusion':False,'runtime_precision':'FP16 with protected FP32 coordinate/NMS operations',
                       'accuracy_derivation':'Four-view .35 evaluated by exact high-score-prefix filtering of fresh paired-backbone unfiltered outputs, followed by original fusion.'}
    original_fusion=baseline['fusion']
    for name,candidate in summary['fusion'].items():
        if not isinstance(candidate,dict) or 'at_recall90' not in candidate:continue
        at=candidate['at_recall90'];fixed=candidate['at_confidence_0_75']
        summary['acceptance'][name]={
            'maintains_original_P_at_R90':bool(at and at['precision']>=original_fusion['at_recall90']['precision'] and at['recall']>=.90),
            'maintains_original_fixed_0_75_P_and_R':bool(fixed['precision']>=original_fusion['at_confidence_0_75']['precision'] and fixed['recall']>=original_fusion['at_confidence_0_75']['recall']),
            'automatically_promoted':False}
    for tag,model in [('base_single','base'),('small_single','small')]:
        old=baseline[model]['view_metrics']['original']['at_recall90']
        new=summary['models'][tag]['cutoff035_equivalent']['at_recall90']
        summary['acceptance'][tag]={'maintains_original_single_view_P_at_R90':bool(new and new['precision']>=old['precision'] and new['recall']>=.90),
                                  'reference':old,'automatically_promoted':False}
    summary['original_baseline_preserved']=True
    verify_input_bindings(bindings)
    if any(sha(REPO/name)!=digest for name,digest in pinned.items()):
        raise ValueError('Pinned sources/baseline evidence changed during summary')
    summary['input_bindings']=bindings
    summary['input_bindings_before_after_identical']=True
    summary['finished_utc']=now()
    write(report/'validation_summary.json',summary)
    lines=['# 快速部署完整验证结果','',f"完成时间（UTC）：{summary['finished_utc']}",'',
           '所有配置均使用原 Base/Small 权重，从权重完整重推原 814 张验证集，GT3449、空图13，未训练。',
           'Batch=1 单视图与 Batch=2 原图/翻转分别重推；512 全图缩放与512局部裁剪速度模拟严格区分。','',
           '| 配置 | P@R≥90% | 对应 R | 阈值 | 固定.75 P | 固定.75 R |','|---|---:|---:|---:|---:|---:|']
    rows=[(tag,item['cutoff035_equivalent']) for tag,item in summary['models'].items()]
    rows.extend([('paired_four_view_no_cutoff',summary['fusion']['unfiltered']),('paired_four_view_cutoff035',metrics)])
    for name,item in rows:
        at=item['at_recall90'];f=item['at_confidence_0_75']
        lines.append('| '+name+' | '+(f"{at['precision']*100:.6f}% | {at['recall']*100:.6f}% | {at['confidence']:.8f}" if at else '未达到R90 | — | —')+f" | {f['precision']*100:.6f}% | {f['recall']*100:.6f}% |")
    lines.extend(['','0.35 表示最终分数候选截断；由新鲜未截断推理的高分前缀等价导出，低分PR段被截断。四视图重新融合，不能从单视图不变推定融合不变。',
                  '完整JSON含固定点TP/FP/FN、阈值曲线和保留精度判定。任何配置都没有自动替换原94.43%基线。',
                  '512局部跟踪没有独立P@R90：这里的small512_fullframe是全图缩放验证，不是D405实拍或卡尔曼跟踪测试。','',
                  f"[验证原始结果]({(report/'validation_summary.json').as_posix()})"])
    (report/'完整验证结果.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return summary


def filter_records(rows,threshold):
    import numpy as np
    cutoff=float(np.float32(threshold))
    result=[]
    for row in rows:
        indices=[i for i,score in enumerate(row['pred']['scores']) if score>=cutoff]
        result.append({'image_id':row['image_id'],'gt':row['gt'],
                       'pred':{'boxes':[row['pred']['boxes'][i] for i in indices],
                               'scores':[row['pred']['scores'][i] for i in indices]}})
    return result


if __name__=='__main__':run()
