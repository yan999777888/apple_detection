"""Derive bounded real-TRAIN proof for the additional geometry parameters."""
import ast
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
source = ROOT.parent/'od-shallow-detail/tmp/preflight_shallow_detail_real_train_cpu.py'
assert hashlib.sha256(source.read_bytes()).hexdigest() == 'da408c9fed6ea6a872c2c957195be9b08e0a5bfef38e01166708eb1be976718c'
value = source.read_text(encoding='utf-8')
changes = {
    'od_shallow_detail_real_train_cpu_preflight.json':'od_mask_geometry_real_train_cpu_preflight.json',
    'od_shallow_detail_papple_3080.json':'od_mask_geometry_papple_3080.json',
    "'visible_proposal_branch.rpn.', 'visible_proposal_branch.converter.', 'shallow_detail_fusion.')":
        "'visible_proposal_branch.rpn.', 'visible_proposal_branch.converter.', 'shallow_detail_fusion.', 'roi_heads.mask_geometry_head.', 'roi_heads.mask_geometry_refiner.')",
    'len(original_sources) == 15':'len(original_sources) == 17',
    'od_shallow_detail_real_train_cpu_smoke_v1':'od_mask_geometry_real_train_cpu_smoke_v1',
    '102ffa3c392650c1f523e7392ef6c8028dea88e3':'9ebc41a5d93fd48c6ca16b8a808ee2d01391d5c4',
    'source15_before':'source17_before','source15_after':'source17_after',
    "'shallow_detail'):":"'shallow_detail', 'mask_box_refinement'):",
    'len(model.state_dict()) == 388':'len(model.state_dict()) == 401',
    "f'shallow_detail_real_train_cpu_{time.time_ns()}'":"f'mask_geometry_real_train_cpu_{time.time_ns()}'",
    "report['sampler_after'] = sampler.state_dict()":"""geometry_proof = {}
    for prefix, count in (('roi_heads.mask_geometry_head',12),('roi_heads.mask_geometry_refiner',1)):
        grad = report['updates'][-1]['gradients_before_clip'][prefix]
        change = changes(model, initial)[prefix]
        assert grad['finite'] and len(grad['gradient_nonzero']) == count
        assert not grad['gradient_absent'] and not grad['gradient_all_zero']
        assert change['finite'] and len(change['changed']) == count and not change['unchanged']
        geometry_proof[prefix] = dict(second_update_all_finite_nonzero_gradients=True,
            final_all_states_changed=True, actual_parameter_tensors=count)
    report['geometry_learning_proof'] = geometry_proof
    report['sampler_after'] = sampler.state_dict()""",
    "report['actual_constructor'] = model.constructor_config":"""assert model.constructor_config['mask_box_refinement'] is True
    assert model.constructor_config['amodal_base'] is True
    geometry = model.roi_heads.mask_geometry_head
    source_A = model.roi_heads.od_mask_head
    for copied, original in ((geometry.mask_head,source_A.amodal_base_head),
                             (geometry.mask_predictor,source_A.amodal_base_predictor)):
        for name,tensor in copied.state_dict().items():
            assert torch.equal(tensor,original.state_dict()[name])
            assert tensor.data_ptr() != original.state_dict()[name].data_ptr()
    report['independent_A_geometry_initialization'] = geometry.initialization_provenance
    report['actual_constructor'] = model.constructor_config""",
}
for old,new in changes.items():
    assert old in value,old
    value = value.replace(old,new)
ast.parse(value)
path = ROOT/'tmp/preflight_mask_geometry_real_train_cpu.py'
assert not path.exists()
path.write_text(value,encoding='utf-8',newline='\n')
wrapper = (ROOT.parent/'od-shallow-detail/tmp/run_detail_cpu_preflight.py').read_text(encoding='utf-8')
wrapper = wrapper.replace('od_shallow_detail','od_mask_geometry').replace(
    'preflight_shallow_detail_real_train_cpu.py','preflight_mask_geometry_real_train_cpu.py')
ast.parse(wrapper)
(ROOT/'tmp/run_geometry_cpu_preflight.py').write_text(wrapper,encoding='utf-8',newline='\n')
print(hashlib.sha256(path.read_bytes()).hexdigest())
