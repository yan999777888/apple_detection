"""CPU-only evidence binding, original baseline and child-failure checks."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT=Path(__file__).parents[1]
def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module
pipeline=load('fast_pipeline_test',ROOT/'scripts/run_fast_validation_pipeline.py')
evaluate=load('fast_eval_pipeline_test',ROOT/'scripts/evaluate_fast_deployment.py')


@pytest.mark.parametrize('name',['pipeline_status.json','events.json'])
def test_report_never_replaces_previous_evidence(tmp_path,name):
    file=tmp_path/name;file.write_text('previous evidence')
    with pytest.raises(FileExistsError):pipeline.claim_report(tmp_path)
    assert file.read_text()=='previous evidence' and not (tmp_path/'pipeline.lock').exists()


def test_json_binding_covers_exact_bytes(tmp_path):
    file=tmp_path/'evidence.json';file.write_text('{"value": 1}')
    data,binding=pipeline.read_bound_json(file)
    assert data=={'value':1}
    file.write_text('{"value":1}')
    with pytest.raises(ValueError,match='Evidence bytes changed'):
        pipeline.read_bound_json(file,binding['sha256'])


@pytest.fixture
def candidate(tmp_path,monkeypatch):
    monkeypatch.setattr(pipeline,'REPO',tmp_path)
    cfgpath=tmp_path/'base.json';cfgpath.write_text('{}')
    plan={'baseline_config':'base.json'}
    stage={'id':'base_single','stage':'base','output':'run','short_side':1024,'max_size':1280,'views':'original','paired':False}
    directory=tmp_path/'run/base';directory.mkdir(parents=True)
    raw=directory/'original_predictions.json'
    rows=[{'image_id':'original','gt':{'boxes':[[0,0,2,2]]},'pred':{'boxes':[[0,0,2,2]],'scores':[.8]}}]
    pipeline.write(raw,rows)
    def audit(records,contract):
        if records!=contract['expected_rows']:raise ValueError('genuine original GT differs')
    frozen=SimpleNamespace(build_parser=evaluate.build_parser,deployment_spec=evaluate.deployment_spec,audit_records=audit)
    args=evaluate.build_parser().parse_args(pipeline.stage_arguments(stage,plan)[3:])
    meta={'phase':'complete','images':814,'GT':3449,'empty':13,'model':'base','checkpoint_sha256':'original_base',
          'source_sha256':{'code.py':'pinned_source'},'provenance_sha256':'provenance','annotation_sha256':'annotations',
          'config_sha256':pipeline.sha(cfgpath),'from_fresh_weight_inference':True,'original_strong_models_only':True,
          'training_performed':False,'source_before_after_identical':True,'config_before_after_identical':True,
          'strict_restore':{'strict':True,'missing_keys':[],'unexpected_keys':[],'all_tensors_equal_to_checkpoint':True},
          'constructor_config':{'backbone_name':'convnext_base'},'deployment':evaluate.deployment_spec(args),
          'predictions_sha256':{'original':pipeline.sha(raw)}}
    meta_path=directory/'prediction_metadata.json';pipeline.write(meta_path,meta)
    cfg={'base_checkpoint_sha256':'original_base'}
    contract={'provenance_sha256':'provenance','provenance':{'binding':{'current_annotation_sha256':'annotations'}},'expected_rows':deepcopy(rows)}
    receipts={str(meta_path.resolve()):pipeline.sha(meta_path)}
    def invoke():return pipeline.load_candidate_view(stage,'original',plan,cfg,contract,frozen,{'code.py':'pinned_source'},receipts,{})
    return invoke,raw,meta_path,rows,meta,receipts


def test_candidate_requires_metadata_and_raw_bindings(candidate):
    invoke,raw,meta_path,rows,meta,receipts=candidate
    invoke()
    rows[0]['pred']['scores']=[.9];pipeline.write(raw,rows)
    with pytest.raises(ValueError,match='Evidence bytes changed'):invoke()
    meta['predictions_sha256']['original']=pipeline.sha(raw);pipeline.write(meta_path,meta)
    with pytest.raises(ValueError,match='Evidence bytes changed'):invoke()


def test_original_gt_audit_still_applies_after_hash_binding(candidate):
    invoke,raw,meta_path,rows,meta,receipts=candidate
    rows[0]['gt']['boxes']=[];pipeline.write(raw,rows)
    meta['predictions_sha256']['original']=pipeline.sha(raw);pipeline.write(meta_path,meta)
    receipts[str(meta_path.resolve())]=pipeline.sha(meta_path)
    with pytest.raises(ValueError,match='genuine original GT'):invoke()


def test_real_original_baseline_paths_and_gate_match_without_weights():
    plan=json.loads((ROOT/'configs/fast_validation_20261031.json').read_text(encoding='utf-8'))
    cfgpath=ROOT/plan['baseline_config'];cfg=json.loads(cfgpath.read_text(encoding='utf-8'))
    pins=pipeline.source_inventory()
    for path in pipeline.baseline_evidence_files(plan):pins[str(path.relative_to(ROOT))]=pipeline.sha(path)
    meta=json.loads((ROOT/'runs/strong_baseline_20261030/base/prediction_metadata.json').read_text(encoding='utf-8'))
    bindings={}
    result=pipeline.load_baseline(plan,cfg,{'provenance_sha256':meta['provenance_sha256']},pins,bindings)
    assert result['gate']['passed'] and len(bindings)==4
    assert result['fusion']['at_confidence_0_75']['fp']==183


@pytest.mark.parametrize('confirmed',[False,True])
def test_child_cleanup_failure_always_saves_failed_and_retains_unconfirmed_lock(tmp_path,monkeypatch,confirmed):
    if tmp_path.drive.upper()!='F:':pytest.skip('Production pipeline explicitly requires F:')
    monkeypatch.setattr(pipeline,'REPO',tmp_path)
    monkeypatch.setattr(pipeline,'source_inventory',lambda:{})
    monkeypatch.setattr(pipeline,'baseline_evidence_files',lambda plan:[])
    class Child:
        pid=123456
        returncode=None
        def poll(self):raise RuntimeError('primary orchestration failure')
        def terminate(self):raise OSError('terminate failed')
        def kill(self):raise OSError('kill failed')
        def wait(self,timeout):
            if confirmed:return 9
            raise pipeline.subprocess.TimeoutExpired('owned child',timeout)
    monkeypatch.setattr(pipeline.subprocess,'Popen',lambda *a,**kw:Child())
    plan={'report_directory':'report','pin_sources':[],'baseline_config':'config.json',
          'stages':[{'id':'base','stage':'base','output':'run','short_side':1024,'max_size':1280,'views':'original','paired':False}]}
    path=tmp_path/'plan.json';pipeline.write(path,plan)
    with pytest.raises(RuntimeError,match='primary orchestration failure'):
        pipeline.run(['--plan',str(path)])
    status=json.loads((tmp_path/'report/pipeline_status.json').read_text(encoding='utf-8'))
    assert status['phase']=='failed' and status['cleanup']['exit_confirmed'] is confirmed
    assert (tmp_path/'report/pipeline.lock').exists() is not confirmed
    assert status['child_pid']==(None if confirmed else 123456)
    assert status['stages'][0]['phase']=='failed'


def test_source_aliases_do_not_weaken_identity():
    pipeline.require_sources({'src\\core.py':'frozen'},{'src/core.py':'frozen'})
    with pytest.raises(ValueError,match='source inventory differs'):
        pipeline.require_sources({'src/core.py':'changed'},{'src/core.py':'frozen'})
