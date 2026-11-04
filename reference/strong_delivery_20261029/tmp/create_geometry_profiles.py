"""A single geometry switch and output path define the paired experiment."""
from copy import deepcopy
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
base = json.loads((ROOT / 'configs/od_shallow_detail_papple_3080.json').read_text(encoding='utf-8'))
base['model']['amodal_base'] = True
base['loss_weights']['PApple'].update(loss_mask_geometry_box=1., loss_mask_geometry_mix=1.,
    loss_mask_geometry_bce=.5, loss_mask_geometry_dice=.5)
base['initialization_policy'] = ('Paired independent A-mask geometry refinement after the dual-proposal/context/detail '
    'cascade. Genuine in-memory parent A-base head and predictor are copied to an independent geometry head. '
    'Expanded-support predicted A yields differentiable first/last-occupancy boundaries; a learned tanh BoxCoder '
    'residual starts at zero, then final V/H/A and classifier are re-pooled in the final boxes. '
    'Only genuine known-A positives receive A-mask/raw-box/blended-box losses. Unknown A has no geometry edge. '
    'Both arms keep original R50 weights, PApple-only2P/8+4, own original fullVAL0 and the same recipe. '
    'Only mask_box_refinement and output differ; no accuracy gain is claimed.')
for enabled,name,output in ((True,'od_mask_geometry_papple_3080.json','runs/od_mask_geometry_papple_11'),
                            (False,'od_mask_geometry_control_3080.json','runs/od_mask_geometry_control_11')):
    value = deepcopy(base)
    if enabled:
        value['model']['mask_box_refinement'] = True
    value['paths']['output'] = output
    path = ROOT / 'configs' / name
    assert not path.exists()
    path.write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8',newline='\n')
print('Created paired geometry profiles with explicit LF bytes.')
