"""Recompute a compact takeover evidence index from immutable run receipts."""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]

def read(path): return json.loads(path.read_text(encoding='utf8'))
def sha(path):
    with path.open('rb') as stream: return hashlib.file_digest(stream,'sha256').hexdigest()

def main():
    result={'scope':'development experiments; not independent whole-fruit grading', 'models':{}}
    for mode, folder in [('joint','joint_shared_e40_s20261106_v1'),('separate','separate_matched_e40_s20261112_v1')]:
        report_path=ROOT/'runs'/folder/'training_report.json'
        report=read(report_path)
        cp=Path(report['final_checkpoint']['file'])
        assert report['phase']=='complete' and report['epochs_completed']==40
        assert sha(cp)==report['final_checkpoint']['sha256']
        result['models'][mode]={'checkpoint':str(cp),'checkpoint_sha256':sha(cp),
            'epochs':40,'updates':report['optimizer_updates'],'component_updates':report['component_updates'],
            'manifest_sha256':report['recipe']['data_manifest_sha256'],
            'initialization_sha256':report['recipe']['canonical_initialization_sha256'],
            'report_sha256':sha(report_path),
            'validation':{v['task']:{'metrics':v['receipt']['metrics'],'per_class':v['receipt']['per_class'],
                                   'seen':v['receipt']['seen']}
                          for v in report['validation'] if v['epoch']==40 and v['variant']=='ema' and v['primary']}}
    for key in ('manifest_sha256','initialization_sha256'):
        assert result['models']['joint'][key]==result['models']['separate'][key]
    result['matched_image_benchmark']=read(ROOT/'benchmarks/joint_vs_separate_20261112/results.json')['models']
    for row in result['matched_image_benchmark'].values(): row.pop('observations')
    fast=read(ROOT/'benchmarks/separate_frozen_adapter_20261112_v1/results.json')
    parity=read(ROOT/'benchmarks/separate_frozen_adapter_20261112_v1/parity.json')
    assert fast['phase']=='complete' and fast['state_sha256_before']==fast['state_sha256_after']
    assert len(parity['rows'])==185 and len({r['path'] for r in parity['rows']})==185
    assert all(h['boxes_scores_classes_exact'] and h['mask_pixels_exact'] for r in parity['rows'] for h in r['heads'].values())
    result['frozen_adapter']={k:fast[k] for k in ('scope','precision','imgsz','rect','conf','max_det','timing',
        'parity_images','state_sha256_before','state_sha256_after','checkpoint_sha256','metric_warning')}
    bn=read(ROOT/'diagnostics/joint_bn_20261112_run_v1/results.json')
    assert bn['phase']=='complete'
    result['BN_diagnostic']={}
    for name,row in bn['variants'].items():
        assert row['calibration']['parameters_unchanged'] and row['calibration']['non_target_state_unchanged']
        result['BN_diagnostic'][name]={'validation':row['validation'],
                                      'calibration_examples':row['calibration']['examples'],
                                      'parameters_unchanged':True,'head_BN_unchanged':True}
    result['decision']='Retain separate final40 baseline; frozen predict adapter verified at square640/conf.25/FP32 only; no shared/BN variant promotion.'
    output=ROOT/'experiments/takeover_20261112/evidence_summary.json'
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8')
    print('PASS: checkpoint identities, complete final40, 185-image parity and BN-only guards; '+str(output))

if __name__=='__main__': main()
