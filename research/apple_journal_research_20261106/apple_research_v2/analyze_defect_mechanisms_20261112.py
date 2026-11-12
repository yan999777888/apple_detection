"""Read completed predictions: recoveries of the original37 low-score FN and FP categories."""
import sys,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'apple_research_v2/joint'));BASE=ROOT/'experiments/defect_algorithm_steps_20261112_v2'
from joint_state_v1 import atomic_json
def read(p):return json.loads(p.read_text(encoding='utf8'))
def analyze():
    from audit_defect_causes_20261112 import match
    gtrows=read(ROOT/'experiments/defect_cause_audit_20261112_v1/GT_records.json');low={f"{r['image_id']}#{r['GT_index']}" for r in gtrows if r['partition']=='dev' and r['category']=='low_score_correct_box'};assert len(low)==37
    data={'baseline':read(BASE/'mode_deployment_validation/o2o_nms_predictions.json')}
    for arm in read(BASE/'train/results.json')['arms']:data[arm]=[{**r,'pred':[p for p in r['pred'] if p[4]>=.25]} for r in read(BASE/'train'/arm/'predictions.json')]
    for mode in read(BASE/'all_mode_OOF_calibration/results.json')['modes']:data['OOF_'+mode]=read(BASE/'all_mode_OOF_calibration'/f'{mode}_predictions.json')
    results={}
    for name,rows in data.items():
        recovered=[];fp={'duplicate':0,'wrong_class':0,'localization_0.1_to_0.5':0,'no_near_annotated_GT':0}
        for r in rows:
            pairs,mat=match(r['pred'],r['GT']);recovered.extend(f"{r['image_id']}#{gi}" for gi in pairs.values() if f"{r['image_id']}#{gi}" in low)
            for pi,p in enumerate(r['pred']):
                if pi in pairs:continue
                same=max((mat[pi,gi] for gi,g in enumerate(r['GT']) if g['class']==int(p[5])),default=0);other=max((mat[pi,gi] for gi,g in enumerate(r['GT']) if g['class']!=int(p[5])),default=0)
                kind='duplicate' if same>=.5 else 'wrong_class' if other>=.5 else 'localization_0.1_to_0.5' if max(same,other)>=.1 else 'no_near_annotated_GT';fp[kind]+=1
        results[name]={'original_low37_recovered':len(recovered),'recovered_ids':sorted(recovered),'FP_categories':fp}
    assert results['baseline']['original_low37_recovered']==0
    doc={'scope':'fixed operating points; same original113dev/230GT; no_near_annotated_GT does NOT mean verified real defect or verified normal','original_low_score_FN_count':37,'modes':results};atomic_json(BASE/'mechanism_analysis.json',doc);return doc
if __name__=='__main__':print(json.dumps(analyze(),ensure_ascii=False))
