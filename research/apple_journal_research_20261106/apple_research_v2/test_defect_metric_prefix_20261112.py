"""Prove optimized OOF threshold counts agree with re-matching, including ties and class-specific cuts."""
import sys,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'apple_research_v2/joint'))
from joint_state_v1 import configure,atomic_json
configure(ROOT/'experiments/defect_algorithm_steps_20261112_v2/prefix_tests','cpu')
from audit_defect_causes_20261112 import match
from report_defect_algorithm_steps_20261112 import deployment_ap
cases=[{'image_id':'synthetic','GT':[{'class':0,'box':[10,10,20,20]},{'class':0,'box':[18,10,28,20]},{'class':1,'box':[30,30,50,50]}],'pred':[[10,10,20,20,.9,0],[11,10,21,20,.9,0],[30,30,50,50,.5,0],[30,30,50,50,.49,1],[18,10,28,20,.1,0]]}]
checks=0
for ts in ((.01,.01),(.1,.5),(.5,.1),(.9,.9),(.91,.5)):
    for r in cases:
        full,_=match(r['pred'],r['GT']);filtered=[p for p in r['pred'] if p[4]>=ts[int(p[5])]];actual,_=match(filtered,r['GT']);expected=sum(i in full for i,p in enumerate(r['pred']) if p[4]>=ts[int(p[5])]);assert len(actual)==expected;checks+=1
perfect=[{'image_id':'perfect','GT':[{'class':0,'box':[1,1,2,2]},{'class':1,'box':[3,3,4,4]}],'pred':[[1,1,2,2,.7,0],[3,3,4,4,.5,1]]}]
ap=deployment_ap(perfect);assert ap['mAP50']==ap['mAP50_95']==1.;checks+=1
empty=[{**r,'pred':[]} for r in perfect];ap=deployment_ap(empty);assert ap['mAP50']==ap['mAP50_95']==0.;checks+=1
atomic_json(ROOT/'experiments/defect_algorithm_steps_20261112_v2/prefix_tests/results.json',{'passed':checks,'scope':'score-prefix equivalence, ties, class cuts, perfect/empty AP endpoints'});print(json.dumps({'passed':checks}))
