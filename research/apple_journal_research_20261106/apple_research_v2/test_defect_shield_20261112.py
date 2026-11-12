"""CPU checks: exact disabled loss, shield negative gradients, protected GT, paired geometry."""
import sys,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'apple_research_v2/joint'))
from joint_state_v1 import configure,atomic_json
configure(ROOT/'experiments/defect_algorithm_steps_20261112_v2/cpu_tests','cpu')
import torch,numpy as np,copy
from joint_model_v1 import build_model
from defect_shield_train_20261112 import attach_shield,transform_boxes,make_batch
torch.set_num_threads(2);torch.manual_seed(20261112)
model=build_model('separate',device='cpu');checks=[]
for branch in ('one2many','one2one'):
    native=getattr(model.criteria['defect'],branch);shield=copy.deepcopy(native);attach_shield(shield)
    raw={'boxes':torch.full((1,4,8400),4.,requires_grad=True),'scores':torch.randn((1,2,8400),requires_grad=True),'feats':[torch.zeros((1,1,n,n)) for n in (80,40,20)]}
    batch={'batch_idx':torch.tensor([0]),'cls':torch.tensor([[0.]]),'bboxes':torch.tensor([[.5,.5,.15,.15]]),'ignore_boxes':[torch.empty(0,4)]}
    old=native.get_assigned_targets_and_loss(raw,batch);new=shield.get_assigned_targets_and_loss(raw,batch)
    assert torch.equal(old[1],new[1]);checks.append(branch+': empty shield exactly native')
    batch['ignore_boxes']=[torch.tensor([[0.,0.,640.,640.]])]
    new=shield.get_assigned_targets_and_loss(raw,batch);fg=old[0][0];assert fg.sum()>0
    go=torch.autograd.grad(old[1].sum(),raw['scores'],retain_graph=True)[0].permute(0,2,1)
    gn=torch.autograd.grad(new[1].sum(),raw['scores'],retain_graph=True)[0].permute(0,2,1)
    assert torch.equal(go[fg],gn[fg]);checks.append(branch+': all assigned positive gradients unchanged')
    points=old[0][3]*old[0][4];inside=((points>=272)&(points<=368)).all(-1)
    assert torch.equal(go[:,inside],gn[:,inside]);checks.append(branch+': all GT-interior gradients unchanged')
    assert gn[:,~inside][~fg[:,~inside]].abs().max()<1e-7;checks.append(branch+': background class gradients zero inside shield')
    assert torch.equal(old[1][[0,2]],new[1][[0,2]]);checks.append(branch+': regression losses unchanged')
b=[[100.,200.,300.,400.]];x,k=transform_boxes(b,1,0,0,False);f,kf=transform_boxes(b,1,0,0,True)
assert np.allclose(x,[[62.5,125.,187.5,250.]]) and np.allclose(f,[[452.5,125.,577.5,250.]]) and k.all() and kf.all();checks.append('original/flip coordinate test')
x,k=transform_boxes(b,.5,160,160,False);assert np.allclose(x,[[191.25,222.5,253.75,285.]]);checks.append('scale/translation coordinate test')
doc=json.loads((ROOT/'experiments/defect_algorithm_steps_20261112_v2/shield_prepare/defect_train_ignore.json').read_text(encoding='utf8'));samples=[r for r in json.loads((ROOT/'apple_research_v2/datasets/ssda_date_v2/manifest.json').read_text(encoding='utf8'))['samples'] if r['partition']=='train'];ignore={r['image_id']:r for r in doc['rows']}
batch=make_batch([0,1,0,2],samples,ignore,0,0,True,'cpu');model.validate_batch('defect',batch);checks.append('real augmented duplicate-draw minibatch valid; GT and ignore stay separate')
atomic_json(ROOT/'experiments/defect_algorithm_steps_20261112_v2/cpu_tests/results.json',{'passed':len(checks),'checks':checks});print(json.dumps({'passed':len(checks)}))
